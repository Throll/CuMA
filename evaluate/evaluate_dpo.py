import os
import json
import random
import torch
import argparse
import re
import time
import concurrent.futures
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, List
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from peft import PeftModel
from openai import OpenAI
from trl import DPOTrainer, DPOConfig
from models.cuma_model import load_checkpoint, apply_cuma
from train.train_cuma_dpo import CulturalDPOTrainer, CulturalDPODataCollator, CulturalModelWrapper

@dataclass
class ScriptArguments(DPOConfig):
    model_name_or_path: str = field(default="../MyModels/Qwen3-0.6B", metadata={"help": "Base model path"})
    sft_model_path: Optional[str] = field(default=None, metadata={"help": "SFT model path to use as reference model (optional)"})
    adapter_path: str = field(default="./checkpoints/lora_facebook_dpo_10k", metadata={"help": "LoRA adapter path"})
    embedding_model_name: str = field(default="../MyModels/Qwen3-Embedding-0.6B", metadata={"help": "Embedding model for CUMA"})
    dataset_path: str = field(default="./data/facebook_dpo_sampled_10k", metadata={"help": "Dataset path"})
    eval_output_path: str = field(default="./eval_results/dpo_evaluation.json", metadata={"help": "Output file for results"})
    num_samples: int = field(default=100, metadata={"help": "Number of samples to evaluate (set -1 for all)"})
    judge_model: str = field(default="gpt-4o", metadata={"help": "Model to use as judge"})
    api_key: str = field(default=None, metadata={"help": "OpenAI API Key (optional if set in env)"})
    base_url: str = field(default=None, metadata={"help": "OpenAI Base URL (optional)"})
    per_device_eval_batch_size: int = field(default=4, metadata={"help": "Batch size for evaluation"})
    base_responses_path: Optional[str] = field(default=None, metadata={"help": "Path to save/load base model responses"})
    
    # Override output_dir default to avoid error if not passed
    output_dir: str = field(default="./eval_results/dpo_trainer_logs", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})

def clean_response(text):
    """Remove <think> and <tool_call> blocks from the response."""
    if not text:
        return ""
    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove <tool_call>...</tool_call> blocks
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    # Also handle unclosed tags if any (sometimes models get cut off)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'<tool_call>.*', '', text, flags=re.DOTALL)
    # Remove any remaining tags like </think> or </tool_call>
    text = re.sub(r'</?think>', '', text)
    text = re.sub(r'</?tool_call>', '', text)
    return text.strip()

def get_few_shot_examples(dataset, num_shots=3):
    """Select random examples from the dataset to serve as few-shot demonstrations."""
    examples = []
    # Use train set for few-shots if available, else use a subset of test
    split = 'train' if 'train' in dataset else 'test'
    total = len(dataset[split])
    indices = random.sample(range(total), min(num_shots, total))
    
    for idx in indices:
        item = dataset[split][idx]
        examples.append({
            "profile": item.get('demographic', 'Unknown'),
            "prompt": item['prompt'],
            "chosen": item['chosen'],
            "rejected": item['rejected']
        })
    return examples

def format_judge_prompt(item, response_a, response_b, few_shots):
    """Construct the prompt for the LLM Judge."""
    
    system_prompt = (
        "You are an impartial and culturally aware judge. "
        "You will be given a user profile, a conversation context, and two AI responses. "
        "Your task is to determine which response is better suited for the specific user described in the profile. "
        "Consider the user's demographics, values, and preferences implied by their profile."
    )
    
    user_prompt = "Here are some examples of preferences for different users:\n\n"
    
    for i, ex in enumerate(few_shots):
        user_prompt += f"Example {i+1}:\n"
        user_prompt += f"Profile: {ex['profile']}\n"
        user_prompt += f"Context: {ex['prompt']}\n"
        user_prompt += f"Response A: {ex['chosen']}\n"
        user_prompt += f"Response B: {ex['rejected']}\n"
        user_prompt += "Verdict: [[A]]\n\n" # We assume Chosen is always better in few-shots
        
    user_prompt += "---\n\nNow, please evaluate the following case:\n"
    user_prompt += f"Profile: {item.get('demographic', 'Unknown')}\n"
    user_prompt += f"Context: {item['prompt']}\n"
    user_prompt += f"Response A: {response_a}\n"
    user_prompt += f"Response B: {response_b}\n"
    user_prompt += "\nWhich response is better? Output [[A]], [[B]], or [[Tie]]."
    
    return system_prompt, user_prompt

def generate_responses(model, tokenizer, dataset, args, demographic_encoder=None):
    """Generate responses for the test set."""
    model.eval()
    responses = []
    
    # Select subset
    test_data = dataset['test']
    if args.num_samples > 0 and args.num_samples < len(test_data):
        test_data = test_data.select(range(args.num_samples))
    
    print(f"Generating responses for {len(test_data)} samples...")
    
    # Prepare batches
    prompts = test_data['prompt']
    batch_size = args.per_device_eval_batch_size
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i : i + batch_size]
        
        # Handle CUMA demographics
        if demographic_encoder is not None:
            # Get demographics for this batch
            # Note: test_data is a Dataset, slicing returns dict of lists
            batch_demographics = test_data[i : i + batch_size]['demographic']
            # Handle None/Empty
            batch_demographics = [d if d else "Unknown" for d in batch_demographics]
            
            with torch.no_grad():
                # Encode and set
                demo_embeds = demographic_encoder(batch_demographics).to(model.device)
                model.router_manager.set_demographic_embed(demo_embeds)

        # Tokenize
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Clear demographics
        if demographic_encoder is not None:
            model.router_manager.clear()

        # Decode
        # We need to slice off the input prompt from the output
        input_lengths = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_lengths:]
        batch_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        responses.extend(batch_responses)
        
    return test_data, responses

def run_judge(test_data, model_responses, ref_responses, few_shots, args):
    """Run LLM-as-a-Judge evaluation in parallel."""
    
    client = OpenAI(
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL")
    )
    
    results = {
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "total": 0,
        "details": []
    }
    
    print(f"Running LLM Judge (Parallel, model: {args.judge_model})...")
    
    def judge_sample(i):
        item = test_data[i]
        model_resp = clean_response(model_responses[i])
        ref_resp = clean_response(ref_responses[i])
        
        # Randomize order to avoid position bias
        is_swapped = random.random() > 0.5
        if is_swapped:
            resp_a, resp_b = ref_resp, model_resp # A=Ref, B=Model
        else:
            resp_a, resp_b = model_resp, ref_resp # A=Model, B=Ref
            
        system_prompt, user_prompt = format_judge_prompt(item, resp_a, resp_b, few_shots)
        
        verdict_text = None
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=args.judge_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    timeout=30
                )
                verdict_text = completion.choices[0].message.content
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    return i, None, f"Error: {e}"
        
        # Parse Verdict
        winner = "Tie"
        if "[[A]]" in verdict_text:
            winner = "A"
        elif "[[B]]" in verdict_text:
            winner = "B"
        elif "[[Tie]]" in verdict_text:
            winner = "Tie"
        
        # Map back to Model vs Ref
        result_type = "tie"
        if winner == "Tie":
            result_type = "tie"
        elif (winner == "A" and not is_swapped) or (winner == "B" and is_swapped):
            result_type = "win"
        else:
            result_type = "loss"
            
        return i, {
            "prompt": item['prompt'],
            "model_response": model_resp,
            "ref_response": ref_resp,
            "verdict": verdict_text,
            "result": result_type
        }, None

    # Use ThreadPoolExecutor for parallel API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(judge_sample, i) for i in range(len(test_data))]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            idx, detail, error = future.result()
            if detail:
                results["details"].append(detail)
                if detail["result"] == "win":
                    results["wins"] += 1
                elif detail["result"] == "loss":
                    results["losses"] += 1
                else:
                    results["ties"] += 1
                results["total"] += 1
            else:
                print(f"Sample {idx} failed: {error}")
            
    return results

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    args.remove_unused_columns = False
    
    # 1. Load Dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    
    # 2. Prepare Few-Shot Examples
    few_shots = get_few_shot_examples(dataset)
    
    # 3. Load/Generate Base Model Responses
    test_subset = dataset['test']
    if args.num_samples > 0 and args.num_samples < len(test_subset):
        test_subset = test_subset.select(range(args.num_samples))
    
    base_responses = None
    if args.base_responses_path and os.path.exists(args.base_responses_path):
        print(f"Loading cached base responses from {args.base_responses_path}")
        with open(args.base_responses_path, 'r') as f:
            cached_data = json.load(f)
            # Ensure the cached data matches the current subset (simple check)
            if len(cached_data) >= len(test_subset):
                base_responses = cached_data[:len(test_subset)]
                print(f"Successfully loaded {len(base_responses)} base responses.")
    
    if base_responses is None:
        print(f"Loading base model from {args.model_name_or_path} to generate baseline...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        print("\n--- Generating Base Model Responses ---")
        _, base_responses = generate_responses(model, tokenizer, dataset, args, demographic_encoder=None)
        
        if args.base_responses_path:
            os.makedirs(os.path.dirname(args.base_responses_path), exist_ok=True)
            with open(args.base_responses_path, 'w') as f:
                json.dump(base_responses, f, indent=2)
            print(f"Base responses cached to {args.base_responses_path}")
        
        # Clear memory for adapter
        del model
        torch.cuda.empty_cache()

    # 4. Load Adapter and Generate Adapter Responses
    print(f"\n--- Loading Adapter and Generating Responses ---")
    # Reload base model for adapter (or use the one we have if we didn't delete it)
    # To save memory, we reload it here.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    demographic_encoder = None
    if args.adapter_path and os.path.exists(args.adapter_path):
        adapter_config_path = os.path.join(args.adapter_path, "adapter_config.json")
        is_cuma = False
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                if adapter_config.get("peft_type") == "CuMA":
                    is_cuma = True
        
        if is_cuma:
            print("Detected cuma adapter.")
            model, demographic_encoder, cuma_config = load_checkpoint(model, args.adapter_path, args.embedding_model_name, device=model.device)
            if demographic_encoder is not None:
                demographic_encoder.to(model.device)
        else:
            print("Loading standard LoRA adapter...")
            model = PeftModel.from_pretrained(model, args.adapter_path)
        
        _, adapter_responses = generate_responses(model, tokenizer, dataset, args, demographic_encoder)
    else:
        print("No adapter provided. Comparing base model against itself (for testing).")
        adapter_responses = base_responses

    # 5. Judge
    if args.api_key or os.getenv("OPENAI_API_KEY"):
        print("\n--- Running LLM Judge ---")
        results = run_judge(test_subset, adapter_responses, base_responses, few_shots, args)
        
        # 6. Save & Print
        win_rate = results["wins"] / results["total"] if results["total"] > 0 else 0
        print(f"\nEvaluation Complete!")
        print(f"Total Samples: {results['total']}")
        print(f"Wins: {results['wins']} (Model better than Base)")
        print(f"Losses: {results['losses']} (Base better than Model)")
        print(f"Ties: {results['ties']}")
        print(f"Win Rate (vs Base): {win_rate:.2%}")
        
        os.makedirs(os.path.dirname(args.eval_output_path), exist_ok=True)
        with open(args.eval_output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {args.eval_output_path}")
    else:
        print("\nSkipping Judge step (No API Key). Saving generations.")
        output_data = []
        for i, item in enumerate(test_subset):
            output_data.append({
                "prompt": item['prompt'],
                "demographic": item.get('demographic'),
                "model_response": adapter_responses[i],
                "base_response": base_responses[i]
            })
        
        os.makedirs(os.path.dirname(args.eval_output_path), exist_ok=True)
        with open(args.eval_output_path.replace('.json', '_generations.json'), 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Generations saved to {args.eval_output_path.replace('.json', '_generations.json')}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
