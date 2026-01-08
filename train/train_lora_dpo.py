import os
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, HfArgumentParser
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType

# --- Main Script ---

@dataclass
class ScriptArguments(DPOConfig):
    model_name_or_path: str = field(default="../MyModels/Qwen3-0.6B")
    dataset_path: str = field(default="./data/facebook_dpo_encoded")
    debug_mode: bool = field(default=False, metadata={"help": "Run on a small subset for debugging"})
    
    # LoRA Config
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Adapter Path for Continuation
    adapter_path: Optional[str] = field(default=None, metadata={"help": "Path to existing LoRA adapter to continue training from"})

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Force remove_unused_columns to False to ensure 'demographic' column is preserved (though not used here)
    args.remove_unused_columns = False
    
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

    # 3. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # 4. Apply LoRA
    if args.adapter_path:
        print(f"Loading existing LoRA adapter from {args.adapter_path}...")
        from peft import PeftModel
        # Load the adapter onto the base model
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
        print("LoRA adapter loaded and set to trainable.")
        
        # When loading an existing adapter, we don't pass peft_config to DPOTrainer
        # because the model is already a PeftModel.
        peft_config = None
    else:
        print("Applying New LoRA...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    
    # 5. Reference Model (Optional, DPOTrainer can handle it if None, but explicit is safer for memory control)
    # For standard LoRA DPO, we usually don't need to load a separate ref model if we use peft_config in DPOTrainer.
    # DPOTrainer will use the peft model as policy and disable adapters for reference.
    # However, to be explicit and consistent with CuMa script structure:
    # We will let DPOTrainer handle the reference model creation (it creates a copy or uses peft magic).
    
    # 6. Trainer
    print("Initializing Trainer...")
    
    # Set dataset_num_proc to speed up processing
    # if args.dataset_num_proc is None:
    #     args.dataset_num_proc = 16
    #     print(f"Setting dataset_num_proc to {args.dataset_num_proc} for faster processing.")
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # DPOTrainer will handle this with peft_config
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    print("Starting Training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
