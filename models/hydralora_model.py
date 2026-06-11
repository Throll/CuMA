import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os
import json

from utils.cultural_config import HydraLoraConfig
from safetensors.torch import load_file

class HydraLoraRouter(nn.Module):
    """Standard MoE Router that uses only hidden states (as in HydraLoRA paper)"""
    
    def __init__(self, config: HydraLoraConfig, input_dim: int, layer_id: int):
        super().__init__()
        self.num_experts = config.num_experts
        self.input_dim = input_dim
        self.layer_id = layer_id
        self.dropout = nn.Dropout(config.dropout)
        
        dtype = getattr(config, 'torch_dtype', torch.float32)
        if isinstance(dtype, str):
            if 'bfloat16' in dtype:
                dtype = torch.bfloat16
            elif 'float16' in dtype:
                dtype = torch.float16
            else:
                dtype = torch.float32
        self.torch_dtype = dtype
        
        # Routing strategy
        self.top_k_routing_strategy = config.top_k_routing_strategy
        self.top_k = config.top_k
        
        # Build MLP router
        if config.num_router_mlp_layers == 1:
            self.mlp = nn.Sequential(
                self.dropout,
                nn.Linear(input_dim, self.num_experts, dtype=self.torch_dtype)
            )
        else:
            layers = [
                self.dropout,
                nn.Linear(input_dim, config.router_hidden_dim, dtype=self.torch_dtype),
                nn.ReLU()
            ]
            
            for _ in range(config.num_router_mlp_layers - 2):
                layers.extend([
                    nn.Dropout(config.dropout),
                    nn.Linear(config.router_hidden_dim, config.router_hidden_dim, 
                             dtype=self.torch_dtype),
                    nn.ReLU()
                ])
            
            layers.extend([
                nn.Dropout(config.dropout),
                nn.Linear(config.router_hidden_dim, self.num_experts, dtype=self.torch_dtype)
            ])
            
            self.mlp = nn.Sequential(*layers)
        
        self.routing_weight: Optional[torch.Tensor] = None
        
    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
        Returns:
            routing_weight: [batch_size, seq_len, num_experts]
        """
        # Compute routing weights
        routing_logits = self.mlp(hidden_states)
        routing_weight = F.softmax(routing_logits, dim=-1)
        
        # Apply top-k routing if enabled
        if self.top_k_routing_strategy:
            top_k_values, top_k_indices = torch.topk(routing_weight, self.top_k, dim=-1)
            routing_weight = torch.full_like(routing_weight, 
                                            torch.finfo(routing_weight.dtype).min)
            routing_weight.scatter_(-1, top_k_indices, top_k_values)
            routing_weight = F.softmax(routing_weight, dim=-1)
        
        self.routing_weight = routing_weight
        return routing_weight
    
    def get_routing_weight(self):
        if self.routing_weight is None:
            raise ValueError("Router has not been called yet. Call forward() first.")
        return self.routing_weight
    
    def load_balancing_loss(self, attention_mask):
        """Compute load balancing loss to encourage expert diversity"""
        if self.routing_weight is None:
            return torch.tensor(0.0, device=attention_mask.device)
        
        mask = attention_mask.to(self.routing_weight.dtype)
        num_token = mask.sum()
        
        routing_weight = self.routing_weight * mask.unsqueeze(-1)
        
        # Count expert usage
        count = torch.sign(routing_weight * mask.unsqueeze(-1))
        freq = torch.sum(count.view(-1, self.num_experts), dim=0) / (num_token * self.top_k)
        
        # Proportion of routing weights
        prop = torch.sum(routing_weight.view(-1, self.num_experts), dim=0) / num_token
        
        loss = torch.sum(prop * freq) * self.num_experts
        return loss.unsqueeze(0)
    
    def clear(self):
        """Clear cached routing weights"""
        self.routing_weight = None


class HydraLoraRouterManager(nn.Module):
    """Manages all HydraLoRA routers and computes auxiliary losses"""
    
    def __init__(self, config: HydraLoraConfig, routers: nn.ModuleList):
        super().__init__()
        self.routers = routers
        self.use_load_balancing_loss = config.use_load_balancing_loss
        self.lambda_auxiliary = config.lambda_auxiliary
        self.lambda_lm = config.lambda_lm
        self.router_keys = {}
    
    def get_router(self, router_key: str) -> HydraLoraRouter:
        """Get router by key"""
        if router_key not in self.router_keys:
            raise ValueError(f"Router key '{router_key}' not found")
        return self.router_keys[router_key]
    
    def clear_cache(self):
        """Clear cached routing weights in all routers"""
        for router in self.routers:
            router.clear()
    
    def get_auxiliary_loss(self, loss, attention_mask, reduce='sum'):
        """Compute total loss including auxiliary losses"""
        if not self.use_load_balancing_loss:
            return loss
        
        auxiliary_losses = []
        for router in self.routers:
            auxiliary_losses.append(router.load_balancing_loss(attention_mask))
        
        if len(auxiliary_losses) == 0:
            return loss
        
        auxiliary_loss = torch.stack(auxiliary_losses, dim=0)
        
        if reduce == 'sum':
            auxiliary_loss = torch.sum(auxiliary_loss)
        elif reduce == 'mean':
            auxiliary_loss = torch.mean(auxiliary_loss)
        else:
            raise ValueError(f'reduce must be sum or mean, got {reduce}')
        
        total_loss = self.lambda_lm * loss + self.lambda_auxiliary * auxiliary_loss
        return total_loss
    
    def clear(self):
        """Clear all router caches"""
        for router in self.routers:
            router.clear()


class HydraLoraLinear(nn.Module):
    """HydraLoRA linear layer with shared A and multiple B experts"""
    
    _root_model_ref = None
    
    def __init__(self, in_features, out_features, bias, config, layer_name):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.layer_name = layer_name
        self.router_key = None
        self._router_manager = None
        
        dtype = getattr(config, 'torch_dtype', torch.float32)
        if isinstance(dtype, str):
            if 'bfloat16' in dtype:
                dtype = torch.bfloat16
            elif 'float16' in dtype:
                dtype = torch.float16
            else:
                dtype = torch.float32

        # Base linear layer
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.weight.requires_grad = False
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
            self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)
        
        # Shared A matrix for HydraLoRA
        self.lora_a = nn.Parameter(
            torch.randn(config.lora_r, in_features, dtype=dtype) * 0.01
        )
        
        # Multiple B matrices (experts)
        self.lora_b = nn.ParameterList([
            nn.Parameter(torch.zeros(out_features, config.lora_r, dtype=dtype))
            for _ in range(config.num_experts)
        ])
        
        self.scaling = config.lora_alpha / config.lora_r
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        output = F.linear(x, self.weight, self.bias)
        
        # Dummy usage for DDP
        if self.training:
            dummy = 0
            dummy += self.lora_a.sum() * 0
            for p in self.lora_b: dummy += p.sum() * 0
            output = output + dummy
        
        # Get router from router_manager
        if self._router_manager is not None:
            router_manager = self._router_manager
        elif self._root_model_ref is not None:
            router_manager = self._root_model_ref.router_manager
        else:
            raise RuntimeError("Router manager not set. Call apply_hydralora() correctly.")
        
        router = router_manager.get_router(self.router_key)
        routing_weights = router(x)
        
        # Precompute shared A part
        shared_a_output = self.dropout(x) @ self.lora_a.t()
        
        lora_output = torch.zeros_like(output)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.config.top_k, dim=-1)
        
        for k_idx in range(self.config.top_k):
            expert_indices = top_k_indices[..., k_idx]
            expert_weights = top_k_weights[..., k_idx:k_idx+1]
            
            for expert_id in range(self.config.num_experts):
                mask = (expert_indices == expert_id)
                if not mask.any():
                    continue
                
                # Use precomputed shared A output
                lora_result = shared_a_output[mask] @ self.lora_b[expert_id].t()
                
                weights_masked = expert_weights[mask]
                lora_result = lora_result * weights_masked
                
                lora_output[mask] += lora_result * self.scaling
        
        return output + lora_output


def apply_hydralora(model, config):
    """Apply HydraLoRA to model"""
    
    for param in model.parameters():
        param.requires_grad = False
        
    routers_dict = {}
    hydralora_layers = []
    
    target_modules = config.target_modules
    
    modules_to_replace = []
    for name, module in list(model.named_modules()):
        if not any(target in name for target in target_modules):
            continue
        
        if not isinstance(module, nn.Linear):
            continue
        
        modules_to_replace.append((name, module))
    
    replacement_count = 0
    for name, module in modules_to_replace:
        *parent_path, attr_name = name.split('.')
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        
        # Create HydraLoRA layer
        hydralora_layer = HydraLoraLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            config=config,
            layer_name=name
        )
        
        # Copy weights
        with torch.no_grad():
            hydralora_layer.weight.copy_(module.weight)
            if module.bias is not None:
                hydralora_layer.bias.copy_(module.bias)
        
        # Set router key
        layer_id = -1
        for part in name.split('.'):
            if part.isdigit():
                layer_id = int(part)
                break
        
        if config.share_router_for_qkv and any(t in name for t in ['q_proj', 'k_proj', 'v_proj']):
            router_key = f"layer_{layer_id}_qkv"
        elif config.share_router_for_w_i and any(t in name for t in ['gate_proj', 'up_proj']):
            router_key = f"layer_{layer_id}_wi"
        else:
            router_key = f"layer_{layer_id}_{name.split('.')[-1]}"
            
        hydralora_layer.router_key = router_key
        
        if router_key not in routers_dict:
            routers_dict[router_key] = HydraLoraRouter(config, module.in_features, layer_id)
            
        # Replace module
        setattr(parent, attr_name, hydralora_layer)
        hydralora_layers.append(hydralora_layer)
        replacement_count += 1
        
    print(f"Replaced {replacement_count} modules with HydraLoRA layers")
    
    # Create router manager
    routers = nn.ModuleList([routers_dict[k] for k in sorted(routers_dict.keys())])
    router_manager = HydraLoraRouterManager(config, routers)
    router_manager.router_keys = {k: routers_dict[k] for k in routers_dict.keys()}
    
    # Attach router manager to model
    model.router_manager = router_manager
    
    # Set root model reference in all HydraLoRA layers
    HydraLoraLinear._root_model_ref = model
    
    return model

class HydraLoraModel:
    @staticmethod
    def from_pretrained(model, name_or_path: Optional[str] = None):
        with open(os.path.join(name_or_path, "adapter_config.json")) as f:
            config_dict = json.load(f)
        
        config = HydraLoraConfig.from_config(config_dict)
        config.torch_dtype = model.dtype
        
        model = apply_hydralora(model, config)
        
        adapter_weights_path = os.path.join(name_or_path, 'adapter_model.safetensors')
        adapter_weights = load_file(adapter_weights_path)
        model.load_state_dict(adapter_weights, strict=False)
        
        return model


def save_hydralora_checkpoint(model, args, tokenizer, hydralora_config=None):
    config_source = hydralora_config if hydralora_config is not None else args
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    trainable_state_dict = {
        name: param for name, param in model.named_parameters() 
        if param.requires_grad
    }
    
    from safetensors.torch import save_file
    save_file(
        trainable_state_dict,
        os.path.join(args.save_dir, 'adapter_model.safetensors')
    )
    
    tokenizer.save_pretrained(args.save_dir)
    
    config_dict = {
        'lora_r': config_source.lora_r,
        'lora_alpha': config_source.lora_alpha,
        'dropout': config_source.dropout,
        'num_experts': config_source.num_experts,
        'top_k': config_source.top_k,
        'use_hydra_lora': True,
        'router_hidden_dim': config_source.router_hidden_dim,
        'num_router_mlp_layers': config_source.num_router_mlp_layers,
        'top_k_routing_strategy': config_source.top_k_routing_strategy,
        'use_load_balancing_loss': config_source.use_load_balancing_loss,
        'lambda_auxiliary': config_source.lambda_auxiliary,
        'target_modules': config_source.target_modules,
        'share_router_for_qkv': config_source.share_router_for_qkv,
        'share_router_for_w_i': config_source.share_router_for_w_i,
        'torch_dtype': str(model.dtype),
        'hidden_size': model.config.hidden_size,
        'base_model': args.model,
        'model_type': model.config.model_type,
        'peft_type': 'HydraLoRA',
    }
    
    with open(os.path.join(args.save_dir, 'adapter_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"HydraLoRA Checkpoint saved to {args.save_dir}")
