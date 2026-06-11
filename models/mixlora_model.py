import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os
import json

from utils.cultural_config import MixLoraConfig
from safetensors.torch import load_file

class MixLoraRouter(nn.Module):
    """Standard MoE Router that uses only hidden states"""
    
    def __init__(self, config: MixLoraConfig, input_dim: int, layer_id: int):
        super().__init__()
        self.num_experts = config.num_experts
        self.input_dim = input_dim
        self.layer_id = layer_id
        self.dropout = nn.Dropout(config.dropout)
        self.torch_dtype = config.torch_dtype
        
        # Routing strategy
        self.top_k_routing_strategy = config.top_k_routing_strategy
        self.top_k = config.top_k
        
        # Build MLP router
        if config.num_router_mlp_layers == 1:
            self.mlp = nn.Sequential(
                self.dropout,
                nn.Linear(input_dim, self.num_experts, dtype=config.torch_dtype)
            )
        else:
            layers = [
                self.dropout,
                nn.Linear(input_dim, config.router_hidden_dim, dtype=config.torch_dtype),
                nn.ReLU()
            ]
            
            for _ in range(config.num_router_mlp_layers - 2):
                layers.extend([
                    nn.Dropout(config.dropout),
                    nn.Linear(config.router_hidden_dim, config.router_hidden_dim, 
                             dtype=config.torch_dtype),
                    nn.ReLU()
                ])
            
            layers.extend([
                nn.Dropout(config.dropout),
                nn.Linear(config.router_hidden_dim, self.num_experts, dtype=config.torch_dtype)
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


class MixLoraRouterManager(nn.Module):
    """Manages all MixLoRA routers and computes auxiliary losses"""
    
    def __init__(self, config: MixLoraConfig, routers: nn.ModuleList):
        super().__init__()
        self.routers = routers
        self.use_load_balancing_loss = config.use_load_balancing_loss
        self.lambda_auxiliary = config.lambda_auxiliary
        self.lambda_lm = config.lambda_lm
        self.router_keys = {}
    
    def get_router(self, router_key: str) -> MixLoraRouter:
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


class MixLoraLinear(nn.Module):
    """MixLoRA linear layer"""
    
    _root_model_ref = None
    
    def __init__(self, in_features, out_features, bias, config, layer_name):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.layer_name = layer_name
        self.router_key = None
        self._router_manager = None
        
        # Base linear layer
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        if config.use_hydra_lora:
            # Shared A matrix for HydraLoRA
            self.lora_a = nn.Parameter(
                torch.randn(config.lora_r, in_features) * 0.01
            )
        else:
            # Separate A matrices per expert
            self.lora_a = nn.ParameterList([
                nn.Parameter(torch.randn(config.lora_r, in_features) * 0.01)
                for _ in range(config.num_experts)
            ])
        
        self.lora_b = nn.ParameterList([
            nn.Parameter(torch.zeros(out_features, config.lora_r))
            for _ in range(config.num_experts)
        ])
        
        self.scaling = config.lora_alpha / config.lora_r
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        output = F.linear(x, self.weight, self.bias)
        
        # Dummy usage for DDP
        if self.training:
            dummy = 0
            if isinstance(self.lora_a, nn.ParameterList):
                for p in self.lora_a: dummy += p.sum() * 0
            else:
                dummy += self.lora_a.sum() * 0
                
            if isinstance(self.lora_b, nn.ParameterList):
                for p in self.lora_b: dummy += p.sum() * 0
            else:
                dummy += self.lora_b.sum() * 0
            output = output + dummy
        
        # Get router from router_manager
        if self._router_manager is not None:
            router_manager = self._router_manager
        elif self._root_model_ref is not None:
            router_manager = self._root_model_ref.router_manager
        else:
            raise RuntimeError("Router manager not set. Call apply_mixlora() correctly.")
        
        router = router_manager.get_router(self.router_key)
        routing_weights = router(x)
        
        lora_output = torch.zeros_like(output)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.config.top_k, dim=-1)
        
        for k_idx in range(self.config.top_k):
            expert_indices = top_k_indices[..., k_idx]
            expert_weights = top_k_weights[..., k_idx:k_idx+1]
            
            for expert_id in range(self.config.num_experts):
                mask = (expert_indices == expert_id)
                if not mask.any():
                    continue
                
                x_masked = x[mask]
                
                if self.config.use_hydra_lora:
                    lora_result = self.dropout(x_masked) @ self.lora_a.t()
                else:
                    lora_result = self.dropout(x_masked) @ self.lora_a[expert_id].t()
                
                lora_result = lora_result @ self.lora_b[expert_id].t()
                
                weights_masked = expert_weights[mask]
                lora_result = lora_result * weights_masked
                
                lora_output[mask] += lora_result * self.scaling
        
        return output + lora_output


def apply_mixlora(model, config):
    """Apply MixLoRA to model"""
    
    for param in model.parameters():
        param.requires_grad = False
        
    routers_dict = {}
    mlora_layers = []
    
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
        
        # Create MixLoRA layer
        mlora_layer = MixLoraLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            config=config,
            layer_name=name
        )
        
        # Copy weights
        with torch.no_grad():
            mlora_layer.weight.copy_(module.weight)
            if module.bias is not None:
                mlora_layer.bias.copy_(module.bias)
        
        mlora_layer.weight.requires_grad = False
        if mlora_layer.bias is not None:
            mlora_layer.bias.requires_grad = False
            
        mlora_layer = mlora_layer.to(device=module.weight.device, dtype=module.weight.dtype)
        
        # Register router
        router_key = get_router_key(name, config)
        if router_key not in routers_dict:
            router = create_mixlora_router(config, name, module.in_features)
            router = router.to(device=module.weight.device, dtype=config.torch_dtype)
            routers_dict[router_key] = router
        
        mlora_layer.router_key = router_key
        mlora_layers.append(mlora_layer)
        
        setattr(parent, attr_name, mlora_layer)
        replacement_count += 1
        
    print(f"Applied MixLoRA to {replacement_count} layers")

    routers_list = nn.ModuleList(list(routers_dict.values()))
    router_manager = MixLoraRouterManager(config, routers_list)
    router_manager.router_keys = routers_dict
    
    model.add_module('router_manager', router_manager)
    
    for layer in mlora_layers:
        layer._router_manager = router_manager

    MixLoraLinear._root_model_ref = model
    
    return model


def get_router_key(layer_name, config):
    if config.share_router_for_qkv and any(x in layer_name for x in ['q_proj', 'k_proj', 'v_proj']):
        base_name = layer_name.rsplit('.', 1)[0]
        return f"{base_name}.qkv_shared"
    
    if config.share_router_for_w_i and any(x in layer_name for x in ['gate_proj', 'up_proj']):
        base_name = layer_name.rsplit('.', 1)[0]
        return f"{base_name}.w_i_shared"
    
    return layer_name


def create_mixlora_router(config, layer_name, input_dim):
    import re
    match = re.search(r'\.(\d+)\.', layer_name)
    layer_id = int(match.group(1)) if match else 0
    
    router = MixLoraRouter(
        config=config,
        input_dim=input_dim,
        layer_id=layer_id
    )
    
    return router


class MixLoraModel:
    @staticmethod
    def from_pretrained(model, name_or_path: Optional[str] = None):
        with open(os.path.join(name_or_path, "adapter_config.json")) as f:
            config_dict = json.load(f)
        
        config = MixLoraConfig.from_config(config_dict)
        config.torch_dtype = model.dtype
        
        model = apply_mixlora(model, config)
        
        adapter_weights_path = os.path.join(name_or_path, 'adapter_model.safetensors')
        adapter_weights = load_file(adapter_weights_path)
        model.load_state_dict(adapter_weights, strict=False)
        
        return model


def save_mixlora_checkpoint(model, args, tokenizer, mixlora_config=None):
    config_source = mixlora_config if mixlora_config is not None else args
    
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
        'use_hydra_lora': config_source.use_hydra_lora,
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
        'peft_type': 'MixLoRA',
    }
    
    with open(os.path.join(args.save_dir, 'adapter_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"MixLoRA Checkpoint saved to {args.save_dir}")
