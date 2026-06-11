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

from utils.cultural_config import MixLoraConfig
from models.mixlora_model import apply_mixlora, save_mixlora_checkpoint
from utils.func import get_unique_dir, clear_directory
from utils.dataset import WVSDataset, ChatDataset, FacebookDataset, PRISMDataset

class MixLoraModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask, labels=None, use_cache=False, **kwargs):
        # Forward pass
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache
        )
        return output

def train(model_wrapper, tokenizer, optimizer, scheduler, train_dataloader, accelerator, args):
    """Training loop for MixLoRA with Accelerate"""
    
    # Loss function (standard LM loss)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)
    
    global_step = 0
    
    # Calculate total steps if epochs are provided
    steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    if args.num_epochs > 0:
        args.max_steps = steps_per_epoch * args.num_epochs
    
    total_epochs = math.ceil(args.max_steps / max(1, steps_per_epoch))
    
    accelerator.print(f"Training for {args.max_steps} steps (approx {total_epochs} epochs)")
    accelerator.print(f"Logging every {args.logging_steps} steps, Saving every {args.save_steps} steps")
    
    if accelerator.is_main_process:
        bar = tqdm(total=args.max_steps, desc="MixLoRA", dynamic_ncols=True)
    
    epoch = 0
    while global_step < args.max_steps:
        epoch += 1
        model_wrapper.train()
        
        for step, mini_batch in enumerate(train_dataloader):
            with accelerator.accumulate(model_wrapper):
                # Forward pass through wrapper
                output = model_wrapper(
                    input_ids=mini_batch['input_ids'],
                    attention_mask=mini_batch['attention_mask'],
                    use_cache=False
                )
                
                # Calculate LM Loss
                shift_logits = output.logits[..., :-1, :].contiguous()
                shift_labels = mini_batch['labels'][..., 1:].contiguous()
                
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Add auxiliary loss from routers
                unwrapped_model = accelerator.unwrap_model(model_wrapper).model
                if hasattr(unwrapped_model, 'router_manager'):
                    loss = unwrapped_model.router_manager.get_auxiliary_loss(
                        loss, 
                        mini_batch['attention_mask']
                    )
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_wrapper.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Clear router cache after each step
                if hasattr(unwrapped_model, 'router_manager'):
                    unwrapped_model.router_manager.clear_cache()
            
            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    bar.update(1)
                    bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
                
                # Logging
                if global_step % args.logging_steps == 0:
                    pass # Add logging if needed
                
                # Saving
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        save_dir = os.path.join(args.save_dir, f"checkpoint-{global_step}")
                        args_copy = argparse.Namespace(**vars(args))
                        args_copy.save_dir = save_dir
                        
                        # Unwrap model
                        unwrapped_model = accelerator.unwrap_model(model_wrapper).model
                        save_mixlora_checkpoint(unwrapped_model, args_copy, tokenizer)
            
            if global_step >= args.max_steps:
                break
                
    if accelerator.is_main_process:
        bar.close()
        # Final save
        save_mixlora_checkpoint(accelerator.unwrap_model(model_wrapper).model, args, tokenizer)

def main():
    parser = argparse.ArgumentParser()
    # Model and Data
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="wvs")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    
    # MixLoRA Hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--use_hydra_lora", action="store_true")
    parser.add_argument("--top_k_routing_strategy", action="store_true", default=True)
    parser.add_argument("--share_router_for_qkv", action="store_true", default=True)
    parser.add_argument("--share_router_for_w_i", action="store_true", default=True)
    parser.add_argument("--router_hidden_dim", type=int, default=512)
    parser.add_argument("--num_router_mlp_layers", type=int, default=1)
    parser.add_argument("--lambda_auxiliary", type=float, default=0.01)
    parser.add_argument("--use_load_balancing_loss", action="store_true", default=True)
    parser.add_argument("--target_modules", type=str, default=None, help="Comma separated list of target modules")
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation_steps)
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Apply MixLoRA
    if args.target_modules:
        args.target_modules = args.target_modules.split(',')
    
    config = MixLoraConfig(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dropout=args.dropout,
        num_experts=args.num_experts,
        top_k=args.top_k,
        use_hydra_lora=args.use_hydra_lora,
        top_k_routing_strategy=args.top_k_routing_strategy,
        share_router_for_qkv=args.share_router_for_qkv,
        share_router_for_w_i=args.share_router_for_w_i,
        router_hidden_dim=args.router_hidden_dim,
        num_router_mlp_layers=args.num_router_mlp_layers,
        use_load_balancing_loss=args.use_load_balancing_loss,
        lambda_auxiliary=args.lambda_auxiliary,
        target_modules=args.target_modules,
        torch_dtype=torch.bfloat16
    )
    
    model = apply_mixlora(model, config)
    model_wrapper = MixLoraModelWrapper(model)
    
    # Load Dataset
    if args.dataset == "wvs":
        train_dataset = WVSDataset(args.data_path, tokenizer, max_length=args.max_length)
    elif args.dataset == "chat":
        train_dataset = ChatDataset(args.data_path, tokenizer, max_length=args.max_length)
    elif args.dataset == "facebook":
        train_dataset = FacebookDataset(args.data_path, tokenizer, max_length=args.max_length)
    elif args.dataset == "prism":
        train_dataset = PRISMDataset(args.data_path, tokenizer, max_length=args.max_length)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
        
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=train_dataset.get_collate_fn()
    )
    
    # Optimizer and Scheduler
    optimizer = AdamW(model_wrapper.parameters(), lr=args.lr)
    
    num_training_steps = args.max_steps
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare for Accelerate
    model_wrapper, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model_wrapper, optimizer, train_dataloader, scheduler
    )
    
    # Train
    train(model_wrapper, tokenizer, optimizer, scheduler, train_dataloader, accelerator, args)

if __name__ == "__main__":
    main()
