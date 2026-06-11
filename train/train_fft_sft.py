import os
import argparse
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from utils.func import get_unique_dir, clear_directory
from utils.dataset import WVSDataset, ChatDataset, FacebookDataset, PRISMDataset


def train(model, tokenizer, optimizer, scheduler, train_dataloader, accelerator, args):
    """Full Fine-Tuning SFT training loop"""
    
    global_step = 0
    
    # Calculate total steps if epochs are provided
    steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    if args.num_epochs > 0:
        args.max_steps = steps_per_epoch * args.num_epochs
    
    # Calculate total epochs needed to reach max_steps
    total_epochs = math.ceil(args.max_steps / max(1, steps_per_epoch))
    
    accelerator.print(f"Training for {args.max_steps} steps (approx {total_epochs} epochs)")
    accelerator.print(f"Logging every {args.logging_steps} steps, Saving every {args.save_steps} steps")
    
    if accelerator.is_main_process:
        bar = tqdm(total=args.max_steps, desc="FFT SFT", dynamic_ncols=True)

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
    parser = argparse.ArgumentParser(description="Full Fine-Tuning SFT Training")
    
    # Model parameters
    parser.add_argument('--model', type=str, required=True, help='Path to base model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/base_sft', help='Directory to save checkpoints')
    
    # Dataset parameters
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    
    # Training parameters
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum training steps')
    parser.add_argument("--num_epochs", type=int, default=0, help="Number of training epochs. If > 0, overrides max_steps")
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size per device')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    
    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    
    # Scheduler parameters
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps')
    parser.add_argument('--schedule_name', type=str, default='cosine',
                       choices=['linear', 'cosine', 'constant', 'constant_with_warmup'],
                       help='Learning rate scheduler')
    
    # Mixed precision
    parser.add_argument('--bf16', action='store_true', help='Use bfloat16 mixed precision')
    parser.add_argument('--fp16', action='store_true', help='Use float16 mixed precision')
    
    # Memory optimization
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    
    # Device parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument("--use_demographic_system_prompt", action="store_true", help="Use demographic info as system prompt")
    parser.add_argument("--dataset_type", type=str, default="wvs", choices=["wvs", "chat", "facebook"], help="Type of dataset to use")

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
        accelerator.init_trackers("fft_sft", config=vars(args))

    set_seed(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Force right padding for training
    
    # Load base model
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
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
    
    train_dataset = DatasetClass(**dataset_kwargs)
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.get_collate_fn(),
        num_workers=args.num_workers
    )
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Setup scheduler
    scheduler = get_scheduler(
        args.schedule_name,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )
    
    # Prepare with Accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    # Train
    train(model, tokenizer, optimizer, scheduler, dataloader, accelerator, args)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Final save...")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"Final model saved to {args.save_dir}")
        accelerator.end_training()


if __name__ == '__main__':
    main()