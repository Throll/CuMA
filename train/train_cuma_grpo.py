"""
CuMA GRPO (Group Relative Policy Optimization) Training Script.

Implements Conditional GRPO for cultural alignment as described in Section 3.4
of the CuMA paper. Uses GPT-4o as a demographic-aware reward judge to score
generated responses based on cultural alignment with the user's profile.

Usage:
    accelerate launch train/train_cuma_grpo.py \
        --model_name_or_path <base_model> \
        --embedding_model_name_or_path <embedding_model> \
        --adapter_path <sft_checkpoint> \
        --dataset_path <dataset> \
        --output_dir <output>
"""

import os
import re
import json
import asyncio
import torch
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import GRPOTrainer, GRPOConfig

try:
    import deepspeed.runtime.engine
    def safe_deepspeed_del(self):
        try:
            self.destroy()
        except Exception:
            pass
    deepspeed.runtime.engine.DeepSpeedEngine.__del__ = safe_deepspeed_del
except ImportError:
    pass

from utils.cultural_config import CuMAConfig
from models.cuma_model import apply_cuma, save_check_point
from models.demographic_encoder import DemographicEncoder


# ---------------------------------------------------------------------------
# Reward function: GPT-4o demographic-aware judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial and culturally aware judge. You will be given a user "
    "profile, a conversation context, and two AI responses. Your task is to "
    "determine which response is better suited for the specific user described "
    "in the profile. Consider the user's demographics, values, and preferences "
    "implied by their profile."
)

JUDGE_USER_TEMPLATE = (
    "Profile: {profile}\n"
    "Context: {context}\n"
    "Response A (reference): {response_ref}\n"
    "Response B (candidate): {response_candidate}\n\n"
    "Which response is better for this user? Output [[A]], [[B]], or [[Tie]]."
)


def _build_judge_messages(profile: str, context: str, response_ref: str, response_candidate: str) -> list:
    """Build chat messages for the GPT-4o judge."""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
            profile=profile,
            context=context,
            response_ref=response_ref,
            response_candidate=response_candidate,
        )},
    ]


def _parse_verdict(text: str) -> float:
    """Parse judge verdict into a scalar reward: win=1.0, tie=0.5, loss=0.0."""
    if "[[B]]" in text:
        return 1.0
    elif "[[Tie]]" in text:
        return 0.5
    else:  # [[A]] or parse failure → loss
        return 0.0


async def _judge_single(client, model: str, messages: list) -> float:
    """Call the OpenAI API for a single judgment."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=16,
            temperature=0.0,
        )
        return _parse_verdict(response.choices[0].message.content)
    except Exception:
        return 0.5  # Default to tie on API failure


def _clean_response(text: str) -> str:
    """Remove <think> and <tool_call> blocks from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    return text


class DemographicRewardFunction:
    """
    Callable reward function compatible with TRL's GRPOTrainer.
    
    Uses GPT-4o to judge whether a generated completion aligns with
    the user's demographic profile better than a reference response.
    
    The reward is:  win=1.0, tie=0.5, loss=0.0
    """

    def __init__(
        self,
        judge_model: str = "gpt-4o-2024-11-13",
        ref_responses: Optional[Dict[str, str]] = None,
    ):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
        self.judge_model = judge_model
        self.ref_responses = ref_responses or {}

    def __call__(
        self,
        prompts,
        completions,
        demographic: Optional[list] = None,
        context: Optional[list] = None,
        ref_response: Optional[list] = None,
        **kwargs,
    ) -> list[float]:
        """
        Args:
            prompts: List of prompt strings (or message lists).
            completions: List of completion strings (or message lists).
            demographic: List of demographic profile strings (from dataset column).
            context: List of conversation context strings (from dataset column).
            ref_response: List of reference response strings (from dataset column).
        Returns:
            List of float rewards.
        """
        # Extract text from completions (handle both str and message-list formats)
        completion_texts = []
        for c in completions:
            if isinstance(c, str):
                completion_texts.append(_clean_response(c))
            elif isinstance(c, list) and len(c) > 0:
                # Conversational format: last assistant message
                content = c[-1].get("content", "") if isinstance(c[-1], dict) else str(c[-1])
                completion_texts.append(_clean_response(content))
            else:
                completion_texts.append("")

        # Extract prompt text
        prompt_texts = []
        for p in prompts:
            if isinstance(p, str):
                prompt_texts.append(p)
            elif isinstance(p, list) and len(p) > 0:
                content = p[-1].get("content", "") if isinstance(p[-1], dict) else str(p[-1])
                prompt_texts.append(content)
            else:
                prompt_texts.append("")

        # Build judge calls
        tasks = []
        for i in range(len(completion_texts)):
            profile = demographic[i] if demographic else "Unknown"
            ctx = context[i] if context else prompt_texts[i]
            ref = ref_response[i] if ref_response else "I cannot answer this question."
            messages = _build_judge_messages(profile, ctx, ref, completion_texts[i])
            tasks.append(_judge_single(self.client, self.judge_model, messages))

        # Run all judge calls concurrently
        loop = asyncio.new_event_loop()
        try:
            rewards = loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            loop.close()

        return list(rewards)


# ---------------------------------------------------------------------------
# CuMA Model Wrapper (adapted from train_cuma_dpo.py)
# ---------------------------------------------------------------------------

class CulturalModelWrapper(nn.Module):
    """Wraps the CuMA model to handle demographic routing in forward pass."""

    def __init__(self, model, demographic_encoder=None):
        super().__init__()
        self.model = model
        self.demographic_encoder_container = [demographic_encoder]
        self.config = model.config
        self.generation_config = getattr(model, "generation_config", GenerationConfig())

    @property
    def demographic_encoder(self):
        return self.demographic_encoder_container[0]

    def _set_demographic(self, demographic):
        """Encode and set demographic embeddings on the router manager."""
        if demographic is None:
            return
        if isinstance(demographic, torch.Tensor):
            demographic_embeds = demographic.to(self.model.dtype)
        elif isinstance(demographic, list) and self.demographic_encoder is not None:
            demographic_embeds = self.demographic_encoder(demographic)
        else:
            return

        target = self.model
        if hasattr(target, "router_manager"):
            target.router_manager.set_demographic_embed(demographic_embeds)
        elif hasattr(target, "base_model") and hasattr(target.base_model, "router_manager"):
            target.base_model.router_manager.set_demographic_embed(demographic_embeds)

    def forward(self, input_ids, attention_mask=None, demographic=None, labels=None, use_cache=False, **kwargs):
        self._set_demographic(demographic)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )

    def generate(self, input_ids=None, attention_mask=None, demographic=None, **kwargs):
        self._set_demographic(demographic)
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    # Delegate required interfaces
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.set_output_embeddings(new_embeddings)

    def save_pretrained(self, output_dir):
        model_to_save = self.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module
        model_to_save.save_pretrained(output_dir)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# ---------------------------------------------------------------------------
# Custom GRPO Trainer with CuMA save logic
# ---------------------------------------------------------------------------

class CulturalGRPOTrainer(GRPOTrainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_world_process_zero():
            self.args.save_dir = output_dir

            model_to_save = self.model
            if hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module

            demographic_encoder = None
            if isinstance(model_to_save, CulturalModelWrapper):
                demographic_encoder = model_to_save.demographic_encoder
                model_to_save = model_to_save.model

            save_check_point(
                model_to_save,
                self.args,
                self.processing_class,
                demographic_encoder,
                cuma_config=getattr(self.args, "cuma_config", None),
            )


# ---------------------------------------------------------------------------
# Script arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments(GRPOConfig):
    model_name_or_path: str = field(default="../MyModels/Qwen3-8B")
    embedding_model_name_or_path: str = field(default="../MyModels/Qwen3-Embedding-0.6B")
    dataset_path: str = field(default="./data/facebook_sft_sampled_10000")
    adapter_path: Optional[str] = field(default=None, metadata={"help": "Path to SFT adapter checkpoint"})
    judge_model: str = field(default="gpt-4o-2024-11-13", metadata={"help": "OpenAI model for reward judging"})
    debug_mode: bool = field(default=False, metadata={"help": "Run on a small subset for debugging"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from transformers import HfArgumentParser

    parser = HfArgumentParser((ScriptArguments, CuMAConfig))
    args, cuma_config = parser.parse_args_into_dataclasses()
    args.cuma_config = cuma_config
    args.model = args.model_name_or_path

    # Force keep extra dataset columns (demographic, context, ref_response)
    args.remove_unused_columns = False

    print(f"Loading model from {args.model_name_or_path}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPOTrainer requires left padding

    # 2. Dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    if args.debug_mode:
        print("DEBUG MODE: Truncating dataset.")
        dataset["train"] = dataset["train"].select(range(min(20, len(dataset["train"]))))
        if "test" in dataset:
            dataset["test"] = dataset["test"].select(range(min(5, len(dataset["test"]))))

    # 3. Demographic Encoder
    use_precomputed = "demographic_embed" in dataset["train"].column_names
    demographic_encoder = None

    if use_precomputed:
        print("Using pre-computed demographic embeddings.")
        sample = dataset["train"][0]["demographic_embed"]
        cuma_config.demographic_embed_dim = len(sample) if isinstance(sample, list) else sample.shape[0]
    else:
        print("Initializing Demographic Encoder for on-the-fly encoding.")
        demographic_encoder = DemographicEncoder(model_name=args.embedding_model_name_or_path)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        demographic_encoder.to(device)
        cuma_config.demographic_embed_dim = demographic_encoder.embed_dim

    # 4. Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 5. Apply CuMA
    print("Applying CuMA...")
    cuma_config.torch_dtype = torch.bfloat16

    if args.adapter_path:
        print(f"Loading SFT adapter config from {args.adapter_path}")
        from safetensors.torch import load_file

        config_path = os.path.join(args.adapter_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved_config = json.load(f)
            for k, v in saved_config.items():
                if hasattr(cuma_config, k):
                    setattr(cuma_config, k, v)

    apply_cuma(model, cuma_config)

    if args.adapter_path:
        weights_path = os.path.join(args.adapter_path, "adapter_model.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded adapter weights.")

        encoder_path = os.path.join(args.adapter_path, "demographic_encoder.pt")
        if os.path.exists(encoder_path) and demographic_encoder is not None:
            device = next(demographic_encoder.parameters()).device
            demographic_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            print("Loaded demographic encoder weights.")

    # 6. Wrap model
    model_wrapper = CulturalModelWrapper(model, demographic_encoder)

    # 7. Reward function
    reward_fn = DemographicRewardFunction(judge_model=args.judge_model)

    # 8. Trainer
    print("Initializing CulturalGRPOTrainer...")

    if args.ddp_find_unused_parameters is None:
        args.ddp_find_unused_parameters = True

    trainer = CulturalGRPOTrainer(
        model=model_wrapper,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        processing_class=tokenizer,
    )

    print("Starting GRPO Training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(args.output_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
