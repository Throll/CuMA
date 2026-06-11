import torch
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PrefixTuningConfig
import sys
import os

# Add project root to path
sys.path.append('/root/DemographLLM')
from models.cuma_model import CuMAModel
from models.mixlora_model import MixLoraModel
from models.hydralora_model import HydraLoraModel

model_id = "/root/autodl-tmp/Qwen3-8B"
config = AutoConfig.from_pretrained(model_id)

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

results = {}

# 1. P-Tuning v2
prefix_config = PrefixTuningConfig(
    num_virtual_tokens=32,
    prefix_projection=True,
    task_type="CAUSAL_LM"
)
with torch.device("meta"):
    base_model = AutoModelForCausalLM.from_config(config)
model = get_peft_model(base_model, prefix_config)
results["P-Tuning v2 (32 tokens)"] = count_trainable_params(model)

# 2. LoRA (r=8)
lora_config_r8 = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
with torch.device("meta"):
    base_model = AutoModelForCausalLM.from_config(config)
model = get_peft_model(base_model, lora_config_r8)
results["LoRA (r=8)"] = count_trainable_params(model)

# 3. LoRA (r=64)
lora_config_r64 = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
with torch.device("meta"):
    base_model = AutoModelForCausalLM.from_config(config)
model = get_peft_model(base_model, lora_config_r64)
results["LoRA (r=64)"] = count_trainable_params(model)

# 4. DoRA (r=64)
dora_config_r64 = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj"],
    use_dora=True,
    task_type="CAUSAL_LM"
)
with torch.device("meta"):
    base_model = AutoModelForCausalLM.from_config(config)
model = get_peft_model(base_model, dora_config_r64)
results["DoRA (r=64)"] = count_trainable_params(model)

# Custom Models (MixLoRA, HydraLoRA, CuMA)
# These might not support meta device if they use custom logic in __init__ that checks weights
# We'll use CPU and from_config (which doesn't load weights)

def get_custom_params(model_class, r, num_experts=8, demo_dim=0):
    base_model = AutoModelForCausalLM.from_config(config)
    model = model_class(
        base_model,
        lora_r=r,
        lora_alpha=r*2,
        num_experts=num_experts,
        target_modules=["q_proj", "v_proj"],
        router_hidden_dim=256,
        num_router_mlp_layers=2,
        demographic_dim=demo_dim
    )
    return count_trainable_params(model)

results["MixLoRA (r=64)"] = get_custom_params(MixLoraModel, 64, num_experts=8, demo_dim=0)
results["HydraLoRA (r=64)"] = get_custom_params(HydraLoraModel, 64, num_experts=8, demo_dim=0)
results["CuMA (r=8)"] = get_custom_params(CuMAModel, 8, num_experts=8, demo_dim=1024)
results["CuMA (r=64)"] = get_custom_params(CuMAModel, 64, num_experts=8, demo_dim=1024)

print("\n--- Trainable Parameters (Qwen3-8B) ---")
for k, v in results.items():
    print(f"{k}: {v:,}")
