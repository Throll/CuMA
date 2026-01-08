import os
import json
import argparse
import random
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_from_disk
from sklearn.metrics import f1_score, accuracy_score

from utils.func import set_seed
from models.cuma_model import load_checkpoint
from models.mixlora_model import MixLoraModel
from models.hydralora_model import HydraLoraModel
from utils.dataset.facebook_dataset import FacebookDiscriminationDataset


@torch.no_grad()
def evaluate_facebook(model, demographic_encoder, eval_dataloader, tokenizer, accelerator, torch_dtype, adapter_type="base"):
    model.eval()
    if demographic_encoder:
        demographic_encoder.eval()
        
    results = []
    
    bar = tqdm(eval_dataloader, desc="Evaluating Facebook", dynamic_ncols=True, disable=not accelerator.is_local_main_process)
    
    for batch in bar:
        input_ids = batch['input_ids'].to(accelerator.device)
        attention_mask = batch['attention_mask'].to(accelerator.device)
        targets = batch['targets']
        demos = batch['demographics']
        
        # CuMA routing
        unwrapped_model = accelerator.unwrap_model(model)
        if adapter_type == "cuma":
            if demographic_encoder:
                demo_embeds = []
                for demo in demos:
                    embed = demographic_encoder(demo)
                    if isinstance(embed, torch.Tensor):
                        embed = embed.to(dtype=torch_dtype)
                    else:
                        embed = torch.tensor(embed, dtype=torch_dtype)
                    demo_embeds.append(embed)
                demo_embeds = torch.cat(demo_embeds, dim=0).to(device=accelerator.device)
            elif hasattr(unwrapped_model, 'router_manager') and unwrapped_model.router_manager.routers[0].demographic_dim == 0:
                demo_embeds = torch.zeros((input_ids.size(0), 0), dtype=torch_dtype, device=accelerator.device)
            else:
                raise ValueError("CuMA requires demographic encoder or dim=0")
            
            for router in unwrapped_model.router_manager.routers:
                router.set_demographic_embed(demo_embeds)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        
        # Get logits of the last token
        batch_size = input_ids.size(0)
        last_index = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(batch_size, device=accelerator.device)
        last_logits = outputs.logits[batch_indices, last_index]
        
        choice_tokens = ['A', 'B', 'C', 'D']
        choice_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in choice_tokens]
        choice_ids_tensor = torch.tensor(choice_ids, device=accelerator.device)
        
        for i in range(batch_size):
            sample_logits = last_logits[i, choice_ids_tensor]
            pred_idx = torch.argmax(sample_logits).item()
            pred_letter = choice_tokens[pred_idx]
            
            results.append({
                "target": targets[i],
                "pred": pred_letter,
                "correct": pred_letter == targets[i]
            })
            
        if adapter_type == "cuma":
            for router in unwrapped_model.router_manager.routers:
                router.clear()
                
    # Gather results
    all_results = gather_object(results)
    return all_results


def main(args):
    accelerator = Accelerator()
    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = 'left' # Use left padding for generation/logits
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map={"": accelerator.device}
    )
    
    adapter_type = args.adapter_type.lower()
    demographic_encoder = None
    
    if adapter_type == 'cuma':
        model, demographic_encoder, _ = load_checkpoint(
            base_model=base_model,
            adapter_path=args.peft_model_path,
            embedding_model_name=args.embedding_model,
            device=accelerator.device
        )
    elif adapter_type == 'mixlora':
        model = MixLoraModel.from_pretrained(base_model, args.peft_model_path)
    elif adapter_type == 'hydralora':
        model = HydraLoraModel.from_pretrained(base_model, args.peft_model_path)
    elif adapter_type in ['lora', 'prefix']:
        model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    else:
        model = base_model
        
    dataset = FacebookDiscriminationDataset(
        dataset_path=args.dataset,
        tokenizer=tokenizer,
        mode=args.mode,
        few_shot_k=args.few_shot,
        seed=args.seed,
        max_samples=args.max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=dataset.get_collate_fn()
    )
    
    model, dataloader = accelerator.prepare(model, dataloader)
    if demographic_encoder:
        demographic_encoder = accelerator.prepare(demographic_encoder)
        
    all_results = evaluate_facebook(
        model, demographic_encoder, dataloader, tokenizer, accelerator, torch_dtype, adapter_type
    )
    
    if accelerator.is_main_process:
        df = pd.DataFrame(all_results)
        acc = accuracy_score(df['target'], df['pred'])
        f1 = f1_score(df['target'], df['pred'], average='macro')
        
        print(f"\n{'='*50}")
        print(f"Mode: {args.mode} | Few-shot: {args.few_shot}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro-F1: {f1:.4f}")
        print(f"{'='*50}\n")
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            summary = {
                "accuracy": float(acc),
                "macro_f1": float(f1),
                "mode": args.mode,
                "few_shot": args.few_shot,
                "total": len(df)
            }
            with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)

    # Clean up
    del model
    if demographic_encoder:
        del demographic_encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../MyModels/Qwen3-8B")
    parser.add_argument("--embedding_model", type=str, default="../MyModels/Qwen3-Embedding-0.6B")
    parser.add_argument("--adapter_type", type=str, default="base")
    parser.add_argument("--peft_model_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="/root/autodl-tmp/data/facebook_sampled/test")
    parser.add_argument("--mode", type=str, default="persona", choices=["vanilla", "persona"])
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()
    main(args)
