import os
import argparse
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import PrefixTuningConfig, get_peft_model, TaskType
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from utils.func import get_unique_dir, clear_directory
from utils.dataset import WVSDataset, ChatDataset, FacebookDataset, PRISMDataset


def train(model, tokenizer, optimizer, scheduler, train_dataloader, accelerator, args):
    """Prefix Tuning SFT training loop"""
    
    global_step = 0
    
    steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    total_epochs = math.ceil(args.max_steps / max(1, steps_per_epoch))
    
    accelerator.print(f"Training for {args.max_steps} steps (approx {total_epochs} epochs)")
    accelerator.print(f"Logging every {args.logging_steps} steps, Saving every {args.save_steps} steps")
    
    if accelerator.is_main_process:
        bar = tqdm(total=args.max_steps, desc="Prefix SFT", dynamic_ncols=True)

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
    parser = argparse.ArgumentParser(description="Prefix Tuning SFT training")
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--dataset_path", required=True, help="Dataset path")
    parser.add_argument("--save_dir", default="./checkpoints/prefix_sft")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=0, help="Number of training epochs (overrides max_steps if > 0)")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--schedule_name", default="cosine", choices=["linear", "cosine", "constant"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_demographic_system_prompt", action="store_true")
    parser.add_argument("--dataset_type", default="wvs", choices=["wvs", "chat", "facebook"])
    
    # Prefix Tuning specific
    parser.add_argument("--num_virtual_tokens", type=int, default=32, help="Number of virtual tokens for prefix tuning")
    parser.add_argument("--prefix_projection", action="store_true", help="Whether to use a projection MLP for prefix tuning")

    args = parser.parse_args()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation_steps,
        mixed_precision="bf16" if args.bf16 else ("fp16" if args.fp16 else "no"),
        log_with="tensorboard",
        project_dir=args.save_dir
    )

    if accelerator.is_main_process:
        args.save_dir = get_unique_dir(args.save_dir)
        clear_directory(args.save_dir)
        accelerator.init_trackers("prefix_sft", config=vars(args))

    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    # Prefix Tuning Config
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        prefix_projection=args.prefix_projection,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    dataset_class_map = {
        'wvs': WVSDataset,
        'chat': ChatDataset,
        'facebook': FacebookDataset,
    }
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
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.get_collate_fn(),
        num_workers=args.num_workers
    )
    
    if args.num_epochs > 0:
        args.max_steps = (len(train_dataloader) // args.accumulation_steps) * args.num_epochs

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        args.schedule_name,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )
    
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    train(model, tokenizer, optimizer, scheduler, train_dataloader, accelerator, args)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        accelerator.end_training()


if __name__ == "__main__":
    main()
