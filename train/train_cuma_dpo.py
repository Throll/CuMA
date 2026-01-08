import os
import torch
import torch.distributed as dist
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig
from trl.trainer.utils import DPODataCollatorWithPadding, selective_log_softmax
from peft import LoraConfig, get_peft_model

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

# --- Custom Data Collator ---
@dataclass
class CulturalDPODataCollator(DPODataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor, str]]]) -> Dict[str, torch.Tensor]:
        # Extract demographics
        demographics = []
        for feature in features:
            if "demographic_embed" in feature:
                demographics.append(feature.pop("demographic_embed"))
            elif "demographic" in feature:
                demographics.append(feature.pop("demographic"))
        
        if len(demographics) == 0 and len(features) > 0:
             # Check if it's just missing from the dict but present in dataset
             pass 
             # print(f"WARNING: No demographics found in batch of size {len(features)}! Keys: {features[0].keys()}")

        # Ensure attention masks exist
        for feature in features:
            for key in ["prompt", "chosen", "rejected"]:
                input_key = f"{key}_input_ids"
                mask_key = f"{key}_attention_mask"
                if input_key in feature and mask_key not in feature:
                    feature[mask_key] = [1] * len(feature[input_key])
        
        # Call parent collator
        batch = super().__call__(features)
        
        # Add demographics back to batch
        if demographics:
            # Check if it's embedding (tensor or list of floats) or text (str)
            is_embedding = False
            if isinstance(demographics[0], torch.Tensor):
                is_embedding = True
            elif isinstance(demographics[0], list) and not isinstance(demographics[0][0], str):
                is_embedding = True
            
            if is_embedding:
                batch["demographic"] = torch.tensor(demographics) if isinstance(demographics[0], list) else torch.stack(demographics)
            else:
                batch["demographic"] = demographics
            
        return batch

# --- Custom Model Wrapper (Duplicated from train_cuma.py for independence) ---
class CulturalModelWrapper(nn.Module):
    def __init__(self, model, demographic_encoder=None):
        super().__init__()
        self.model = model
        # Store in a list to prevent submodule registration by DDP
        self.demographic_encoder_container = [demographic_encoder]
        self.config = model.config # Expose config for Trainer
        
    @property
    def demographic_encoder(self):
        return self.demographic_encoder_container[0]

    def forward(self, input_ids, attention_mask, demographic=None, labels=None, use_cache=False, **kwargs):
        # 1. Encode Demographics
        if demographic is not None:
            if isinstance(demographic, torch.Tensor):
                # demographic is already an embedding tensor [batch_size, embed_dim]
                demographic_embeds = demographic.to(self.model.dtype)
            elif isinstance(demographic, list) and self.demographic_encoder is not None:
                # demographic is a list of strings, encode it
                demographic_embeds = self.demographic_encoder(demographic)
            else:
                raise ValueError("Invalid demographic input or missing encoder")
            
            # 2. Set embeddings for Routers
            # Access the underlying model (might be wrapped by LoRA/PEFT)
            # self.model is the CuMA model
            if hasattr(self.model, 'router_manager'):
                self.model.router_manager.set_demographic_embed(demographic_embeds)
                # print("Set demographic embed on router_manager")
            elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'router_manager'):
                 # Handle PEFT wrapping
                 self.model.base_model.router_manager.set_demographic_embed(demographic_embeds)
                 # print("Set demographic embed on base_model.router_manager")
            else:
                print("Warning: router_manager not found in CulturalModelWrapper!")

        else:
            print("Warning: demographic is None in forward!")

        # 3. Forward pass
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs
        )
        return output
    
    def save_pretrained(self, output_dir):
        """Delegate save to the underlying model"""
        # Unwrap if needed (though self.model should be the base model)
        model_to_save = self.model
        if hasattr(model_to_save, 'module'):
            model_to_save = model_to_save.module
            
        model_to_save.save_pretrained(output_dir)
        # Also save demographic encoder if needed, but it's usually static or separate

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
        
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
        
    def get_output_embeddings(self):
        return self.model.get_output_embeddings()
        
    def set_output_embeddings(self, new_embeddings):
        self.model.set_output_embeddings(new_embeddings)
        
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

# --- Custom DPO Trainer ---
class CulturalDPOTrainer(DPOTrainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
            
        if self.is_world_process_zero():
            # Update save_dir in args for save_check_point
            self.args.save_dir = output_dir
            
            # Custom save logic
            model_to_save = self.model
            if hasattr(model_to_save, 'module'):
                model_to_save = model_to_save.module
            
            demographic_encoder = None
            if isinstance(model_to_save, CulturalModelWrapper):
                demographic_encoder = model_to_save.demographic_encoder
                model_to_save = model_to_save.model
                
            save_check_point(model_to_save, self.args, self.processing_class, demographic_encoder, cuma_config=getattr(self.args, 'cuma_config', None))

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], is_ref_model: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            padding_value=self.pad_token_id,
        )
        len_chosen = batch["chosen_input_ids"].shape[0]

        # Handle Demographics
        demographics = batch.get('demographic', None)
        concatenated_demographics = None
        if demographics is not None:
            if isinstance(demographics, torch.Tensor):
                concatenated_demographics = torch.cat((demographics, demographics), dim=0)
            else:
                concatenated_demographics = demographics + demographics

        model_kwargs = {}
        if self.aux_loss_enabled:
             model_kwargs["output_router_logits"] = True

        # Reconstruct inputs as in DPOTrainer
        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        # Concatenate
        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        
        # Loss mask
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
            dim=1,
        )
        
        model_kwargs["attention_mask"] = attention_mask
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            demographic=concatenated_demographics,
            use_cache=False,
            **model_kwargs
        )
        
        logits = outputs.logits
        
        # Calculate log probs
        # Offset logits by one to align with labels
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()
        
        # Compute log probs
        labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)
        
        all_logps = per_token_logps[:, 1:].sum(-1)
        
        # Split
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_logits = logits[:len_chosen]
        rejected_logits = logits[len_chosen:]
        
        # Calculate mean logits
        mean_chosen_logits = chosen_logits[loss_mask[:len_chosen]].mean()
        mean_rejected_logits = rejected_logits[loss_mask[len_chosen:]].mean()
        
        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "mean_chosen_logits": mean_chosen_logits,
            "mean_rejected_logits": mean_rejected_logits,
        }

# --- Main Script ---

@dataclass
class ScriptArguments(DPOConfig):
    model_name_or_path: str = field(default="../MyModels/Qwen3-0.6B")
    embedding_model_name_or_path: str = field(default="../MyModels/Qwen3-Embedding-0.6B")
    dataset_path: str = field(default="./data/facebook_dpo_encoded")
    demographic_encoder_path: str = field(default="./models/demographic_encoder") # Adjust path
    adapter_path: Optional[str] = field(default=None, metadata={"help": "Path to SFT adapter checkpoint"})
    debug_mode: bool = field(default=False, metadata={"help": "Run on a small subset for debugging"})

def main():    
    from transformers import HfArgumentParser
    parser = HfArgumentParser((ScriptArguments, CuMAConfig))
    args, cuma_config = parser.parse_args_into_dataclasses()
    args.cuma_config = cuma_config # Attach for Trainer access
    
    # Alias for save_check_point
    args.model = args.model_name_or_path
    
    # Force remove_unused_columns to False to ensure 'demographic' column is preserved
    args.remove_unused_columns = False
    
    # Setup Accelerator (handled by Trainer usually, but good for prints)
    
    print(f"Loading model from {args.model_name_or_path}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Force right padding for training
        
    # 2. Load Dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    
    if args.debug_mode:
        print("DEBUG MODE: Truncating dataset to 20 samples.")
        dataset['train'] = dataset['train'].select(range(20))
        dataset['test'] = dataset['test'].select(range(5))

    # 3. Initialize Demographic Encoder
    # Check if dataset has pre-computed embeddings
    use_precomputed_embeddings = "demographic_embed" in dataset['train'].column_names
    demographic_encoder = None
    
    if use_precomputed_embeddings:
        print("Found 'demographic_embed' column. Using pre-computed embeddings.")
        sample_embed = dataset['train'][0]['demographic_embed']
        if isinstance(sample_embed, list):
            cuma_config.demographic_embed_dim = len(sample_embed)
        elif hasattr(sample_embed, 'shape'):
            cuma_config.demographic_embed_dim = sample_embed.shape[0]
        print(f"Inferred demographic_embed_dim: {cuma_config.demographic_embed_dim}")
        
    else:
        print("'demographic_embed' column not found. Initializing Demographic Encoder for on-the-fly encoding.")
        print("Initializing Demographic Encoder...")
        demographic_encoder = DemographicEncoder(model_name=args.embedding_model_name_or_path) 
        # Move to GPU using LOCAL_RANK to avoid Accelerator conflict
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        demographic_encoder.to(device)
        cuma_config.demographic_embed_dim = demographic_encoder.embed_dim

    # 4. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # device_map="auto" # Removed for distributed training
    )
    
    # 5. Apply CuMa (CuMA)
    print("Applying CuMA...")
    
    if args.adapter_path:
        print(f"Loading SFT adapter from {args.adapter_path}")
        import json
        from safetensors.torch import load_file
        
        # Load config
        config_path = os.path.join(args.adapter_path, 'adapter_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            # Update cuma_config with saved config
            for k, v in saved_config.items():
                if hasattr(cuma_config, k):
                    setattr(cuma_config, k, v)
            print("Updated CuMa config from adapter checkpoint.")
    
    print(f"Setting demographic_embed_dim to {cuma_config.demographic_embed_dim}")
    
    cuma_config.torch_dtype = torch.bfloat16
    
    # This modifies the model in-place to add routers and adapters
    apply_cuma(model, cuma_config)
    
    if args.adapter_path:
        # Load weights
        weights_path = os.path.join(args.adapter_path, 'adapter_model.safetensors')
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded adapter weights.")
        else:
            print(f"Warning: adapter_model.safetensors not found in {args.adapter_path}")
            
        # Load demographic encoder
        encoder_path = os.path.join(args.adapter_path, 'demographic_encoder.pt')
        if os.path.exists(encoder_path) and demographic_encoder is not None:
             # Move to same device as encoder
             device = next(demographic_encoder.parameters()).device
             demographic_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
             print("Loaded demographic encoder weights.")
    
    # 6. Wrap Model
    # We wrap it to handle demographic inputs
    model_wrapper = CulturalModelWrapper(model, demographic_encoder)
    
    # 7. Reference Model
    # DPO needs a reference model. Usually it's the same as initial policy.
    # We can load it again or copy. For memory efficiency, we might want to use PEFT/LoRA 
    # and disable adapters for ref model, but CuMa is structural.
    # For simplicity, we'll load another instance or let DPOTrainer handle it (it copies if ref_model is None)
    # BUT, our model is custom wrapped. DPOTrainer might fail to copy correctly.
    # Let's load a fresh one for ref_model.
    
    print("Loading Reference Model...")
    ref_model_base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # device_map="auto" # Removed for distributed training
    )
    # Apply CuMa to ref model too (initialized same way)
    # Ideally, this should be the SFT model. 
    # If we are training from scratch (SFT-ed), we should load the SFT checkpoint.
    apply_cuma(ref_model_base, cuma_config)
    ref_model_wrapper = CulturalModelWrapper(ref_model_base, demographic_encoder)
    
    # 8. Trainer
    print("Initializing Trainer...")
    
    # Initialize custom collator
    data_collator = CulturalDPODataCollator(
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Ensure ddp_find_unused_parameters is True if not set
    if args.ddp_find_unused_parameters is None:
        args.ddp_find_unused_parameters = True
        print("Forcing ddp_find_unused_parameters=True for Cultural Model Wrapper")
    
    trainer = CulturalDPOTrainer(
        model=model_wrapper,
        ref_model=ref_model_wrapper,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting Training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(args.output_dir)
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
