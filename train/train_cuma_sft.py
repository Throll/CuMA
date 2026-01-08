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

from utils.cultural_config import CuMAConfig
from models.cuma_model import apply_cuma, save_check_point
from models.demographic_encoder import DemographicEncoder
from utils.func import get_unique_dir, clear_directory
from utils.dataset import WVSDataset, ChatDataset, FacebookDataset, PRISMDataset

class CulturalModelWrapper(nn.Module):
    def __init__(self, model, demographic_encoder=None):
        super().__init__()
        self.model = model
        self.demographic_encoder = demographic_encoder
        
    def forward(self, input_ids, attention_mask, demographic, labels=None, use_cache=False, **kwargs):
        # 1. Encode Demographics
        if isinstance(demographic, torch.Tensor):
            # Already encoded
            demographic_embeds = demographic.to(self.model.dtype)
        else:
            # demographic is a list of strings
            if self.demographic_encoder is None:
                raise ValueError("Received text demographics but demographic_encoder is None")
                
            demographic_embeds = []
            for profile in demographic:
                demo_embed = self.demographic_encoder(profile)
                demographic_embeds.append(demo_embed)
            
            demographic_embeds = torch.cat(demographic_embeds, dim=0)
        
        # Ensure demographic_embeds is on the correct device and dtype
        demographic_embeds = demographic_embeds.to(device=input_ids.device, dtype=self.model.dtype)

        # 2. Set embeddings for Routers
        # Access the underlying model (might be wrapped by LoRA/PEFT)
        # self.model is the CuMA model
        if hasattr(self.model, 'router_manager'):
            self.model.router_manager.set_demographic_embed(demographic_embeds)
            
        # 3. Forward pass
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache
        )
        return output

def train(model_wrapper, tokenizer, optimizer, scheduler, train_dataloader, accelerator, args):
    """Training loop for CuMA with Accelerate"""
    
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
        bar = tqdm(total=args.max_steps, desc="CUMA", dynamic_ncols=True)
    
    epoch = 0
    while global_step < args.max_steps:
        epoch += 1
        model_wrapper.train()
        
        for step, mini_batch in enumerate(train_dataloader):
            with accelerator.accumulate(model_wrapper):
                # Handle demographic input (embedding or text)
                demographic_input = mini_batch.get('demographic_embed', mini_batch.get('demographic'))
                
                # Forward pass through wrapper
                output = model_wrapper(
                    input_ids=mini_batch['input_ids'],
                    attention_mask=mini_batch['attention_mask'],
                    demographic=demographic_input,
                    use_cache=False
                )
                
                # Calculate LM Loss
                shift_logits = output.logits[..., :-1, :].contiguous()
                shift_labels = mini_batch['labels'][..., 1:].contiguous()
                
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Add auxiliary loss (Load Balancing)
                # We need to access the underlying model inside the wrapper
                # model_wrapper might be wrapped by DistributedDataParallel or DeepSpeed
                unwrapped_wrapper = accelerator.unwrap_model(model_wrapper)
                # unwrapped_wrapper is CulturalModelWrapper
                # unwrapped_wrapper.model is the CuMA model
                
                if hasattr(unwrapped_wrapper.model, 'router_manager') and args.use_load_balancing_loss:
                    loss = unwrapped_wrapper.model.router_manager.get_auxiliary_loss(loss, mini_batch['attention_mask'])
                
                # 4. Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_wrapper.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Clear router cache
                if hasattr(unwrapped_wrapper.model, 'router_manager'):
                    unwrapped_wrapper.model.router_manager.clear_cache()

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    bar.update(1)
                    if global_step % args.logging_steps == 0:
                        accelerator.log({"train_loss": loss.item(), "epoch": epoch}, step=global_step)
                
                if global_step % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_dir = os.path.join(args.save_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # Save model
                        unwrapped_wrapper = accelerator.unwrap_model(model_wrapper)
                        # unwrapped_wrapper.model is the LLM
                        save_check_point(unwrapped_wrapper.model, args, tokenizer)
                        accelerator.print(f"Saved checkpoint to {save_dir}")
            
            if global_step >= args.max_steps:
                break
        
    if accelerator.is_main_process:
        bar.close()

def main():
    parser = argparse.ArgumentParser()
    # Model args
    parser.add_argument("--model", type=str, required=True, help="Path to base model")
    parser.add_argument("--embedding_model", type=str, required=True, help="Path to embedding model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--dataset_type", type=str, default="wvs", choices=["wvs", "chat", "facebook", "prism"])
    parser.add_argument("--save_dir", type=str, required=True)
    
    # Training args
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_epochs", type=float, default=0.0, help="Number of training epochs. If > 0, overrides max_steps")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--demographic_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--schedule_name", type=str, default="linear")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    
    # CuMA args
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--use_hydra_lora", action="store_true")
    parser.add_argument("--top_k_routing_strategy", action="store_true")
    parser.add_argument("--target_modules", type=str, default="q_proj v_proj")
    parser.add_argument("--share_router_for_qkv", action="store_true")
    parser.add_argument("--share_router_for_w_i", action="store_true")
    parser.add_argument("--demographic_embed_dim", type=int, default=1024)
    parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument("--num_router_mlp_layers", type=int, default=2)
    parser.add_argument("--lambda_auxiliary", type=float, default=0.01)
    parser.add_argument("--use_load_balancing_loss", action="store_true")
    parser.add_argument("--num_encoder_proj_mlp_layers", type=int, default=0)
    parser.add_argument("--use_demographic_system_prompt", action="store_true", help="Use demographic info as system prompt")

    args = parser.parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation_steps,
        mixed_precision="bf16" if args.bf16 else ("fp16" if args.fp16 else "no"),
        log_with="tensorboard",
        project_dir=args.save_dir
    )
    
    if accelerator.is_main_process:
        clear_directory(args.save_dir)
        accelerator.init_trackers("cuma", config=vars(args))

    set_seed(args.seed)
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Force right padding for training
        
    # Load Dataset
    accelerator.print(f"Loading dataset from {args.dataset_path} (Type: {args.dataset_type})")
    
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

    # Determine torch_dtype early
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    # Check for pre-computed embeddings
    use_precomputed = False
    sample_embed = None
    try:
        # Try to inspect the first sample to see if embeddings are present
        # Note: This might be slow if dataset is large and not indexed, but usually fine for HF datasets
        sample = train_dataset[0]
        if 'demographic_embed' in sample:
             use_precomputed = True
             sample_embed = sample['demographic_embed']
        elif 'demographic' in sample:
             demo_val = sample['demographic']
             # Check if it looks like an embedding (list of floats or tensor)
             if isinstance(demo_val, (torch.Tensor, list)) and not isinstance(demo_val, str):
                 if isinstance(demo_val, list) and len(demo_val) > 0 and not isinstance(demo_val[0], str):
                      use_precomputed = True
                      sample_embed = demo_val
                 elif isinstance(demo_val, torch.Tensor):
                      use_precomputed = True
                      sample_embed = demo_val
    except Exception as e:
        accelerator.print(f"Warning: Could not check dataset for embeddings: {e}")

    if use_precomputed:
        accelerator.print("Detected pre-computed demographic embeddings. Skipping DemographicEncoder initialization.")
        demographic_encoder = None
        # Infer dim
        if isinstance(sample_embed, torch.Tensor):
            args.demographic_embed_dim = sample_embed.shape[-1]
        elif isinstance(sample_embed, list):
            args.demographic_embed_dim = len(sample_embed)
        accelerator.print(f"Inferred demographic_embed_dim: {args.demographic_embed_dim}")
    else:
        # Initialize DemographicEncoder
        accelerator.print("Initializing DemographicEncoder...")
        demographic_encoder = DemographicEncoder(
            model_name=args.embedding_model,
            embed_dim=args.demographic_embed_dim,
            torch_dtype=torch_dtype,
            num_proj_layer=args.num_encoder_proj_mlp_layers
        )

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    # Configure CuMA
    if isinstance(args.target_modules, str):
        args.target_modules = args.target_modules.split()
        
    args.target_modules_lora = args.target_modules

    config = CuMAConfig(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dropout=args.dropout,
        num_experts=args.num_experts,
        use_hydra_lora=args.use_hydra_lora,
        top_k=args.top_k,
        top_k_routing_strategy=args.top_k_routing_strategy,
        demographic_embed_dim=args.demographic_embed_dim,
        router_hidden_dim=args.router_hidden_dim,
        num_router_mlp_layers=args.num_router_mlp_layers,
        target_modules=args.target_modules,
        share_router_for_qkv=args.share_router_for_qkv,
        share_router_for_w_i=args.share_router_for_w_i,
        lambda_auxiliary=args.lambda_auxiliary,
        use_load_balancing_loss=args.use_load_balancing_loss,
        torch_dtype=torch_dtype,
        hidden_size=model.config.hidden_size,
        model_type=model.config.model_type
    )
    
    # Apply CuMA
    accelerator.print("Applying CuMA...")
    model = apply_cuma(model, config)
    
    # Wrap models
    model_wrapper = CulturalModelWrapper(model, demographic_encoder)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.get_collate_fn(),
        num_workers=args.num_workers
    )
    
    # Setup Optimizer
    # Now we optimize the wrapper's parameters
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_wrapper.named_parameters() if p.requires_grad],
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
    ]
    
    # Note: We lost the separate LR for demographic encoder here for simplicity in wrapper
    # If we want separate LR, we need to filter by name
    
    optimizer = AdamW(optimizer_grouped_parameters)
    
    # Setup Scheduler
    scheduler = get_scheduler(
        args.schedule_name,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )
    
    # Prepare with Accelerator
    # Now we only have one model to prepare
    model_wrapper, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model_wrapper, optimizer, train_dataloader, scheduler
    )
    
    # Train
    train(model_wrapper, tokenizer, optimizer, scheduler,
          train_dataloader, accelerator, args)
    
    # Final Save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print("Final save...")
        if not hasattr(args, 'target_modules_lora'):
            args.target_modules_lora = args.target_modules
            
        unwrapped_wrapper = accelerator.unwrap_model(model_wrapper)
        save_check_point(unwrapped_wrapper.model, args, tokenizer)
        accelerator.print(f"Training completed! Saved to {args.save_dir}")
        accelerator.end_training()

if __name__ == '__main__':
    main()
