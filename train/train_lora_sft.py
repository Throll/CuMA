import os
import argparse
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from utils.func import get_unique_dir, clear_directory
from utils.dataset import WVSDataset, ChatDataset, FacebookDataset, PRISMDataset


def train(model, tokenizer, optimizer, scheduler, train_dataloader, accelerator, args):
    """LoRA SFT training loop"""
    
    global_step = 0
    
    steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    total_epochs = math.ceil(args.max_steps / max(1, steps_per_epoch))
    
    accelerator.print(f"Training for {args.max_steps} steps (approx {total_epochs} epochs)")
    accelerator.print(f"Logging every {args.logging_steps} steps, Saving every {args.save_steps} steps")
    
    if accelerator.is_main_process:
        bar = tqdm(total=args.max_steps, desc="LoRA SFT", dynamic_ncols=True)

    epoch = 0
    running_loss = 0.0
    
    while global_step < args.max_steps:
        epoch += 1
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"], 
                    labels=batch["labels"],
                    use_cache=False
                )
                loss = outputs.loss
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                running_loss += loss.item()
                
                if accelerator.is_main_process:
                    bar.update(1)
                
                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / args.logging_steps
                    
                    if accelerator.is_main_process:
                        current_lr = scheduler.get_last_lr()[0]
                        accelerator.log({"train/loss": avg_loss, "train/lr": current_lr}, step=global_step)
                        bar.set_postfix(loss=f"{avg_loss:.4f}")
                    
                    running_loss = 0.0

                if global_step % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        ckpt_dir = os.path.join(args.save_dir, f"checkpoint-{global_step}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        print(f"\nCheckpoint saved to {ckpt_dir}")
                
                if global_step >= args.max_steps:
                    break
        
        if global_step >= args.max_steps:
            break
            
    if accelerator.is_main_process:
        bar.close()


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT training")
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--dataset_path", required=True, help="Dataset path")
    parser.add_argument("--save_dir", default="./checkpoints/lora_sft")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=0, help="Number of training epochs (overrides max_steps if > 0)")

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    
    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")

    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--schedule_name", choices=["linear", "cosine", "constant", "constant_with_warmup"],
                        default="cosine")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_bias", choices=["none", "all", "lora_only"], default="none")
    parser.add_argument("--target_modules", type=str, default="q_proj v_proj")
    parser.add_argument("--use_dora", action="store_true", help="Use Weight-Decomposed Low-Rank Adaptation (DoRA)")

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_demographic_system_prompt", action="store_true", help="Use demographic info as system prompt")
    parser.add_argument("--dataset_type", type=str, default="wvs", choices=["wvs", "chat", "facebook", "prism"], help="Type of dataset to use")
    args = parser.parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation_steps,
        mixed_precision="bf16" if args.bf16 else ("fp16" if args.fp16 else "no"),
        log_with="tensorboard",
        project_dir=args.save_dir
    )

    if accelerator.is_main_process:
        args.save_dir = get_unique_dir(args.save_dir)
        clear_directory(args.save_dir)
        accelerator.init_trackers("lora_sft", config=vars(args))

    set_seed(args.seed)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = 'right' # Ensure right padding for training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Handle target_modules string
    if isinstance(args.target_modules, str):
        import re
        args.target_modules = re.split(r'[,\s]+', args.target_modules.strip())
        args.target_modules = [m for m in args.target_modules if m]

    # LoRA Config
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=args.target_modules or None,
        use_dora=args.use_dora,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    dataset_class_map = {
        'wvs': WVSDataset,
        'chat': ChatDataset,
        'facebook': FacebookDataset,
        'prism': PRISMDataset,
    }
    
    if args.dataset_type not in dataset_class_map:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}. Must be one of {list(dataset_class_map.keys())}")
    
    DatasetClass = dataset_class_map[args.dataset_type]
    
    dataset_kwargs = {
        "dataset_path": args.dataset_path,
        "tokenizer": tokenizer,
        "model_name_or_path": args.model,
        "include_demographic": True,
        "max_length": args.max_length,
    }
    
    if args.dataset_type == "wvs":
        dataset_kwargs["use_demographic_system_prompt"] = args.use_demographic_system_prompt
    
    dataset = DatasetClass(**dataset_kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=args.num_workers
    )

    # Calculate total steps if epochs are provided
    steps_per_epoch = len(dataloader) // args.accumulation_steps
    if args.num_epochs > 0:
        args.max_steps = steps_per_epoch * args.num_epochs

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    scheduler = get_scheduler(
        args.schedule_name,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    train(model, tokenizer, optimizer, scheduler, dataloader, accelerator, args)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Final save...")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"Final model saved to {args.save_dir}")
        accelerator.end_training()


if __name__ == "__main__":
    main()