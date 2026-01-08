import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os
import json

from utils.cultural_config import CuMAConfig
from safetensors.torch import load_file
from models.demographic_encoder import DemographicEncoder

from utils.cultural_config import CuMAConfig


class CulturalRouter(nn.Module):
    """Router that combines demographic embedding and question hidden states"""
    
    def __init__(self, config: CuMAConfig, input_dim: int, layer_id: int):
        super().__init__()
        self.num_experts = config.num_experts
        self.input_dim = input_dim
        self.demographic_dim = config.demographic_embed_dim
        self.layer_id = layer_id
        self.dropout = nn.Dropout(config.dropout)
        self.torch_dtype = config.torch_dtype
        
        # Routing strategy
        self.top_k_routing_strategy = config.top_k_routing_strategy
        self.top_k = config.top_k
        
        # Combine demographic embedding and hidden states
        combined_dim = input_dim + config.demographic_embed_dim
        
        # Build MLP router
        if config.num_router_mlp_layers == 1:
            self.mlp = nn.Sequential(
                self.dropout,
                nn.Linear(combined_dim, self.num_experts, dtype=config.torch_dtype)
            )
        else:
            layers = [
                self.dropout,
                nn.Linear(combined_dim, config.router_hidden_dim, dtype=config.torch_dtype),
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
        self.demographic_embed: Optional[torch.Tensor] = None
        
    def set_demographic_embed(self, demographic_embed: torch.Tensor):
        """Set demographic embedding for current batch"""
        self.demographic_embed = demographic_embed
    
    def forward(self, hidden_states: torch.Tensor, demographic_embed: torch.Tensor):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            demographic_embed: [batch_size, demographic_dim]
        Returns:
            routing_weight: [batch_size, seq_len, num_experts]
        """
        if demographic_embed is None:
            # Try to use cached demographic_embed if available
            if self.demographic_embed is not None:
                demographic_embed = self.demographic_embed
            else:
                raise ValueError(f"demographic_embed is None in CulturalRouter (layer {self.layer_id}). "
                                 "Make sure set_demographic_embed() is called before forward().")

        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Expand demographic embedding to match sequence length
        # demographic_embed: [batch_size, demographic_dim]
        # Check if batch_size matches
        if demographic_embed.shape[0] != batch_size:
            # This can happen in DPO concatenated forward where batch size doubles
            # If demographic_embed is [B, D] and hidden_states is [2*B, S, H]
            # We need to repeat demographic_embed
            if batch_size % demographic_embed.shape[0] == 0:
                repeat_factor = batch_size // demographic_embed.shape[0]
                demographic_embed = demographic_embed.repeat(repeat_factor, 1)
            else:
                raise RuntimeError(f"Batch size mismatch in CulturalRouter: hidden_states {hidden_states.shape}, demographic_embed {demographic_embed.shape}")

        demographic_expanded = demographic_embed.unsqueeze(1).expand(batch_size, seq_len, -1)
        demographic_expanded = demographic_expanded.to(hidden_states.device, dtype=hidden_states.dtype)
        combined = torch.cat([hidden_states, demographic_expanded], dim=-1)
                
        # Compute routing weights
        routing_logits = self.mlp(combined)
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
        self.demographic_embed = None


class CulturalRouterManager(nn.Module):
    """Manages all cultural routers and computes auxiliary losses"""
    
    def __init__(self, config: CuMAConfig, routers: nn.ModuleList):
        super().__init__()
        self.routers = routers
        self.use_load_balancing_loss = config.use_load_balancing_loss
        self.lambda_auxiliary = config.lambda_auxiliary
        self.lambda_lm = config.lambda_lm
        self.demographic_embed: Optional[torch.Tensor] = None
    
    def get_router(self, router_key: str) -> CulturalRouter:
        """Get router by key"""
        if router_key not in self.router_keys:
            raise ValueError(f"Router key '{router_key}' not found")
        return self.router_keys[router_key]
    
    def set_demographic_embed(self, demographic_embed: torch.Tensor):
        """Set demographic embedding in all routers"""
        for router in self.routers:
            router.set_demographic_embed(demographic_embed)

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


class MoRa(nn.Module):
    """Mixture of LoRA Experts"""
    
    def __init__(self, base_layer: nn.Linear, config: CuMAConfig):
        super().__init__()
        self.out_features, self.in_features = base_layer.weight.shape
        self.dtype_ = config.torch_dtype
        self.num_experts = config.num_experts
        self.rank = config.lora_r
        self.dropout = nn.Dropout(config.dropout)
        self.use_hydra_lora = config.use_hydra_lora
        
        # Initialize LoRA parameters
        if config.use_hydra_lora:
            # Shared A matrix across experts
            self.mora_a = nn.Parameter(
                torch.empty((self.rank, self.in_features), dtype=self.dtype_)
            )
        else:
            # Separate A matrix per expert
            self.mora_a = nn.Parameter(
                torch.empty((self.rank * self.num_experts, self.in_features), 
                           dtype=self.dtype_)
            )
        
        # Separate B matrix per expert
        self.mora_b = nn.Parameter(
            torch.empty((self.out_features, self.rank * self.num_experts), 
                       dtype=self.dtype_)
        )
        
        # Scaling factor
        self.scaling = config.lora_alpha / math.sqrt(config.lora_r)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mora_a, a=math.sqrt(5))
        nn.init.zeros_(self.mora_b)
    
    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor, 
                residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, in_features]
            gate: [batch_size, seq_len, num_experts]
            residual: [batch_size, seq_len, out_features]
        """
        hidden_states = self.dropout(hidden_states)
        hidden_states = F.linear(hidden_states, self.mora_a)
        
        target_shape = hidden_states.shape[:-1] + (self.num_experts, self.rank)
        
        if self.use_hydra_lora:
            # Expand shared A output to all experts
            hidden_states = hidden_states.unsqueeze(-2).expand(target_shape)
        else:
            # Reshape to separate experts
            hidden_states = hidden_states.view(target_shape)
        
        # Apply gating weights
        hidden_states = (hidden_states * gate.unsqueeze(-1)).view(
            hidden_states.shape[:-2] + (-1,)
        )
        
        # Apply B matrix and scaling
        hidden_states = F.linear(hidden_states, self.mora_b) * self.scaling
        
        return hidden_states.to(residual.dtype) + residual


class LoRA(nn.Module):
    """Standard LoRA for comparison"""
    
    def __init__(self, base_layer: nn.Linear, config: CuMAConfig):
        super().__init__()
        self.out_features, self.in_features = base_layer.weight.shape
        self.dtype_ = config.torch_dtype
        self.dropout = nn.Dropout(config.dropout)
        self.rank = config.lora_r
        
        # Scaling factor
        self.scaling = config.lora_alpha / math.sqrt(config.lora_r)
        
        # LoRA parameters
        self.lora_a = nn.Parameter(
            torch.empty((self.rank, self.in_features), dtype=self.dtype_)
        )
        self.lora_b = nn.Parameter(
            torch.empty((self.out_features, self.rank), dtype=self.dtype_)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
    
    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = F.linear(hidden_states, self.lora_a)
        hidden_states = F.linear(hidden_states, self.lora_b) * self.scaling
        return hidden_states + residual



class CuMALinear(nn.Module):
    """MLoRa linear layer for cultural adaptation"""
    
    _root_model_ref = None
    
    def __init__(self, in_features, out_features, bias, config, layer_name):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.layer_name = layer_name
        self.router_key = None
        self._router = None
        self._router_manager = None # Instance-specific router manager
        
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
    
    def set_router(self, router: CulturalRouter):
        """Set direct reference to router"""
        self._router = router
    
    def forward(self, x):
        output = F.linear(x, self.weight, self.bias)
        
        # Dummy usage for DDP to prevent unused parameter errors
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
            # Fallback to class attribute (deprecated but kept for safety)
            router_manager = self._root_model_ref.router_manager
        else:
            raise RuntimeError("Router manager not set. Call apply_cuma() correctly.")
        
        router = router_manager.get_router(self.router_key)
        
        demographic_embed = router.demographic_embed
        routing_weights = router(x, demographic_embed)
        
        lora_output = torch.zeros_like(output)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.config.top_k, dim=-1)
        
        for k_idx in range(self.config.top_k):
            expert_indices = top_k_indices[..., k_idx]
            expert_weights = top_k_weights[..., k_idx:k_idx+1]
            
            for expert_id in range(self.config.num_experts):
                mask = (expert_indices == expert_id)
                if not mask.any():
                    continue
                
                # Optimization: Only compute for masked tokens
                x_masked = x[mask] # [num_active, in_features]
                
                if self.config.use_hydra_lora:
                    lora_result = self.dropout(x_masked) @ self.lora_a.t()
                else:
                    lora_result = self.dropout(x_masked) @ self.lora_a[expert_id].t()
                
                lora_result = lora_result @ self.lora_b[expert_id].t()
                
                # Apply weights
                weights_masked = expert_weights[mask] # [num_active, 1]
                lora_result = lora_result * weights_masked
                
                # Add to output
                # We use index_add_ or scatter_add_ or just boolean indexing assignment
                # Since mask is boolean, we can assign back.
                # Note: lora_output[mask] += ... is in-place.
                # To avoid in-place gradient issues, we might need to be careful, 
                # but usually it's fine for intermediate tensors.
                # However, lora_output[mask] returns a view or copy? 
                # It returns a copy if advanced indexing is used.
                # So lora_output[mask] += ... modifies lora_output in place.
                
                lora_output[mask] += lora_result * self.scaling
        
        return output + lora_output
    
    def _get_root_model(self):
        """Find root model through parent chain"""
        visited = set()
        
        # Search through all parent modules via the module registry
        for name, module in self.named_modules():
            if id(module) in visited:
                continue
            visited.add(id(module))
            
            if hasattr(module, 'router_manager'):
                return module
    

def _get_module(model: nn.Module, target_name: str):
    """Get module by name from model"""
    for name, module in model.named_modules():
        if name == target_name:
            return module
    return None



def apply_cuma(model, config):
    """Apply CuMA to model"""
    
    for param in model.parameters():
        param.requires_grad = False
        
    routers_dict = {}
    mlora_layers = []
    
    target_modules = config.target_modules
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                         'gate_proj', 'up_proj', 'down_proj']
    
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
        
        # Create MLoRa layer
        mlora_layer = create_mlora_layer(module, config, layer_name=name)
        
        # Register router for this layer
        router_key = get_router_key(name, config)
        if router_key not in routers_dict:
            router = create_cultural_router(config, name, module.in_features)
            router = router.to(device=module.weight.device, dtype=config.torch_dtype)
            routers_dict[router_key] = router
        
        mlora_layer.router_key = router_key
        mlora_layers.append(mlora_layer)
        
        # Replace module
        setattr(parent, attr_name, mlora_layer)
        replacement_count += 1
        
    print(f"Applied CuMA to {replacement_count} layers")

    routers_list = nn.ModuleList(list(routers_dict.values()))
    router_manager = CulturalRouterManager(config, routers_list)
    router_manager.router_keys = routers_dict
    
    model.add_module('router_manager', router_manager)
    
    # Set router manager for all layers
    for layer in mlora_layers:
        layer._router_manager = router_manager

    CuMALinear._root_model_ref = model
    
    return model


def create_mlora_layer(linear_module, config, layer_name):
    """Create MLoRa layer without circular dependencies"""
    
    device = linear_module.weight.device
    dtype = linear_module.weight.dtype
        
    # Create new layer with copied weights
    mlora_layer = CuMALinear(
        in_features=linear_module.in_features,
        out_features=linear_module.out_features,
        bias=linear_module.bias is not None,
        config=config,
        layer_name=layer_name
    )
    
    mlora_layer = mlora_layer.to(device=device, dtype=dtype)
    
    # Copy weights from original layer
    with torch.no_grad():
        mlora_layer.weight.copy_(linear_module.weight)
        if linear_module.bias is not None:
            mlora_layer.bias.copy_(linear_module.bias)
    
    # Freeze base weights
    mlora_layer.weight.requires_grad = False
    if mlora_layer.bias is not None:
        mlora_layer.bias.requires_grad = False
    
    return mlora_layer


def get_router_key(layer_name, config):
    """Determine router key based on layer name and sharing strategy"""
    
    if config.share_router_for_qkv and any(x in layer_name for x in ['q_proj', 'k_proj', 'v_proj']):
        # Share router for QKV projections in same layer
        base_name = layer_name.rsplit('.', 1)[0]  # Remove the projection name
        return f"{base_name}.qkv_shared"
    
    if config.share_router_for_w_i and any(x in layer_name for x in ['gate_proj', 'up_proj']):
        # Share router for gate and up projections
        base_name = layer_name.rsplit('.', 1)[0]
        return f"{base_name}.w_i_shared"
    
    return layer_name


def create_cultural_router(config, layer_name, input_dim):
    """Create a cultural router instance"""
    
    # Extract layer_id from layer_name
    import re
    match = re.search(r'\.(\d+)\.', layer_name)
    layer_id = int(match.group(1)) if match else 0
    
    router = CulturalRouter(
        config=config,
        input_dim=input_dim,
        layer_id=layer_id
    )
    
    return router


class CuMAModel:
    """Wrapper class for loading CuMA models"""
    
    @staticmethod
    def from_pretrained(model, name_or_path: Optional[str] = None):
        with open(os.path.join(name_or_path, "config.json")) as f:
            config_dict = json.load(f)
        
        config = CuMAConfig.from_config(config_dict)
        config.torch_dtype = model.dtype
        
        model = apply_cuma(model, config)
        return model


def save_check_point(model, args, tokenizer, demographic_encoder=None, cuma_config=None):
    import os
    
    # Determine config source for CuMa parameters
    # If cuma_config is provided, use it. Otherwise fallback to args.
    config_source = cuma_config if cuma_config is not None else args
    
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
    
    # Save demographic encoder if provided
    if demographic_encoder is not None:
        torch.save(
            demographic_encoder.state_dict(),
            os.path.join(args.save_dir, 'demographic_encoder.pt')
        )
        print(f"Demographic encoder saved to {args.save_dir}/demographic_encoder.pt")
    
    # Save tokenizer
    tokenizer.save_pretrained(args.save_dir)
    
    # Save config for loading
    config_dict = {
        # LoRA parameters
        'lora_r': config_source.lora_r,
        'lora_alpha': config_source.lora_alpha,
        'dropout': config_source.dropout,
        
        # Expert parameters
        'num_experts': config_source.num_experts,
        'top_k': config_source.top_k,
        'use_hydra_lora': config_source.use_hydra_lora,
        
        # Router parameters
        'demographic_embed_dim': config_source.demographic_embed_dim,
        'router_hidden_dim': config_source.router_hidden_dim,
        'num_router_mlp_layers': config_source.num_router_mlp_layers,
        'top_k_routing_strategy': config_source.top_k_routing_strategy,
        
        # Loss parameters
        'use_load_balancing_loss': config_source.use_load_balancing_loss,
        'lambda_auxiliary': config_source.lambda_auxiliary,
        
        # Target modules
        'target_modules': config_source.target_modules,
        'target_modules_lora': config_source.target_modules_lora,
        'share_router_for_qkv': config_source.share_router_for_qkv,
        'share_router_for_w_i': config_source.share_router_for_w_i,
        
        # Demographic encoder config
        'num_encoder_proj_mlp_layers': config_source.num_encoder_proj_mlp_layers,
        
        # Model metadata
        'torch_dtype': str(model.dtype),
        'hidden_size': model.config.hidden_size,
        'base_model': args.model,
        'model_type': model.config.model_type,
        'peft_type': 'CuMA',
    }
    
    with open(os.path.join(args.save_dir, 'adapter_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Checkpoint saved to {args.save_dir}")
    
    
def load_checkpoint(base_model, adapter_path, embedding_model_name, device='cuda'):
    """Load CuMA from checkpoint"""
    
    # 1. Load adapter config
    config_path = os.path.join(adapter_path, 'adapter_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 2. Create config object
    config = CuMAConfig(**config_dict)
    print(config.torch_dtype)
    
    # 3. Apply CuMA to base model
    model = apply_cuma(base_model, config)
    
    # 4. Load adapter weights
    adapter_weights_path = os.path.join(adapter_path, 'adapter_model.safetensors')
    adapter_weights = load_file(adapter_weights_path)
    model.load_state_dict(adapter_weights, strict=False)
    
    # 5. Load demographic encoder
    if embedding_model_name is None:
        return model, None, config
        
    encoder_path = os.path.join(adapter_path, 'demographic_encoder.pt')
    
    demographic_encoder = DemographicEncoder(
        model_name=embedding_model_name,
        embed_dim=config.demographic_embed_dim,
        torch_dtype=config.torch_dtype,
        num_proj_layer=config.num_encoder_proj_mlp_layers,
        device=device
    )
    
    if not os.path.exists(encoder_path):
        print(f"Demographic_encoder.pt not found in {adapter_path}. Initializing fresh encoder")
        assert config.num_encoder_proj_mlp_layers==0, "Pretrained demographic encoder not found, but num_encoder_proj_mlp_layers > 0"
    else:
        encoder_weights = torch.load(encoder_path, map_location=device)
        demographic_encoder.load_state_dict(encoder_weights)
    
    return model, demographic_encoder, config

