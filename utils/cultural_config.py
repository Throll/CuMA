from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class CuMAConfig:
    """Configuration for CuMA"""
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    dropout: float = 0.1
    
    # Expert parameters
    num_experts: int = 8
    top_k: int = 2
    use_hydra_lora: bool = False
    
    # Router parameters
    demographic_embed_dim: int = 1024
    router_hidden_dim: int = 512
    num_router_mlp_layers: int = 2
    top_k_routing_strategy: bool = True
    
    # Loss parameters
    use_load_balancing_loss: bool = True
    lambda_auxiliary: float = 0.01
    lambda_lm: float = 1.0
    
    # Target modules
    target_modules: Optional[List[str]] = None
    target_modules_lora: Optional[List[str]] = None
    share_router_for_qkv: bool = True
    share_router_for_w_i: bool = True
    
    # Demographic encoder config
    num_encoder_proj_mlp_layers: int = 0
    
    # Model metadata
    torch_dtype: Optional[str] = None
    hidden_size: Optional[int] = None
    model_type: Optional[str] = None
    base_model: Optional[str] = None
    peft_type: str = 'CuMA'
    
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                   'gate_proj', 'up_proj', 'down_proj']
        if isinstance(self.torch_dtype, str):
            dtype_map = {
                'torch.float32': torch.float32,
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
            }
            if self.torch_dtype in dtype_map:
                self.torch_dtype = dtype_map[self.torch_dtype]
            else:
                print(f"Warnings: Unknown torch_dtype string '{self.torch_dtype}', setting to None")
                self.torch_dtype = None
    
    def export(self):
        """Export config as dictionary for saving"""
        return {
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'dropout': self.dropout,
            'num_experts': self.num_experts,
            'use_hydra_lora': self.use_hydra_lora,
            'top_k': self.top_k,
            'top_k_routing_strategy': self.top_k_routing_strategy,
            'demographic_embed_dim': self.demographic_embed_dim,
            'router_hidden_dim': self.router_hidden_dim,
            'num_router_mlp_layers': self.num_router_mlp_layers,
            'hidden_size': self.hidden_size,
            'model_type': self.model_type,
            'target_modules': self.target_modules,
            'target_modules_lora': self.target_modules_lora,
            'share_router_for_qkv': self.share_router_for_qkv,
            'share_router_for_w_i': self.share_router_for_w_i,
            'lambda_auxiliary': self.lambda_auxiliary,
            'lambda_lm': self.lambda_lm,
            'use_load_balancing_loss': self.use_load_balancing_loss,
        }
    
    @classmethod
    def from_config(cls, config_dict):
        """Load config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})
@dataclass
class MixLoraConfig:
    """Configuration for MixLoRA"""
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    dropout: float = 0.1
    
    # Expert parameters
    num_experts: int = 8
    top_k: int = 2
    use_hydra_lora: bool = False
    
    # Router parameters
    router_hidden_dim: int = 512
    num_router_mlp_layers: int = 1 # MixLoRA usually uses 1 layer
    top_k_routing_strategy: bool = True
    
    # Loss parameters
    use_load_balancing_loss: bool = True
    lambda_auxiliary: float = 0.01
    lambda_lm: float = 1.0
    
    # Target modules
    target_modules: Optional[List[str]] = None
    share_router_for_qkv: bool = True
    share_router_for_w_i: bool = True
    
    # Model metadata
    torch_dtype: Optional[str] = None
    hidden_size: Optional[int] = None
    model_type: Optional[str] = None
    base_model: Optional[str] = None
    peft_type: str = 'MixLoRA'
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                   'gate_proj', 'up_proj', 'down_proj']
        if isinstance(self.torch_dtype, str):
            dtype_map = {
                'torch.float32': torch.float32,
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
            }
            if self.torch_dtype in dtype_map:
                self.torch_dtype = dtype_map[self.torch_dtype]
            else:
                print(f"Warnings: Unknown torch_dtype string '{self.torch_dtype}', setting to None")
                self.torch_dtype = None
    
    def export(self):
        """Export config as dictionary for saving"""
        return {
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'dropout': self.dropout,
            'num_experts': self.num_experts,
            'use_hydra_lora': self.use_hydra_lora,
            'top_k': self.top_k,
            'top_k_routing_strategy': self.top_k_routing_strategy,
            'router_hidden_dim': self.router_hidden_dim,
            'num_router_mlp_layers': self.num_router_mlp_layers,
            'hidden_size': self.hidden_size,
            'model_type': self.model_type,
            'target_modules': self.target_modules,
            'share_router_for_qkv': self.share_router_for_qkv,
            'share_router_for_w_i': self.share_router_for_w_i,
            'lambda_auxiliary': self.lambda_auxiliary,
            'lambda_lm': self.lambda_lm,
            'use_load_balancing_loss': self.use_load_balancing_loss,
        }
    
    @classmethod
    def from_config(cls, config_dict):
        """Load config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})

@dataclass
class HydraLoraConfig:
    """Configuration for HydraLoRA"""
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    dropout: float = 0.1
    
    # Expert parameters
    num_experts: int = 8
    top_k: int = 2
    # HydraLoRA always uses shared A
    use_hydra_lora: bool = True
    
    # Router parameters
    router_hidden_dim: int = 512
    num_router_mlp_layers: int = 2
    top_k_routing_strategy: bool = True
    
    # Loss parameters
    use_load_balancing_loss: bool = True
    lambda_auxiliary: float = 0.01
    lambda_lm: float = 1.0
    
    # Target modules
    target_modules: Optional[List[str]] = None
    share_router_for_qkv: bool = True
    share_router_for_w_i: bool = True
    
    # Model metadata
    torch_dtype: Optional[str] = None
    hidden_size: Optional[int] = None
    model_type: Optional[str] = None
    base_model: Optional[str] = None
    peft_type: str = 'HydraLoRA'
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                   'gate_proj', 'up_proj', 'down_proj']
        if isinstance(self.torch_dtype, str):
            dtype_map = {
                'torch.float32': torch.float32,
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
            }
            if self.torch_dtype in dtype_map:
                self.torch_dtype = dtype_map[self.torch_dtype]
            else:
                print(f"Warnings: Unknown torch_dtype string '{self.torch_dtype}', setting to None")
                self.torch_dtype = None
    
    def export(self):
        """Export config as dictionary for saving"""
        return {
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'dropout': self.dropout,
            'num_experts': self.num_experts,
            'use_hydra_lora': self.use_hydra_lora,
            'top_k': self.top_k,
            'top_k_routing_strategy': self.top_k_routing_strategy,
            'router_hidden_dim': self.router_hidden_dim,
            'num_router_mlp_layers': self.num_router_mlp_layers,
            'hidden_size': self.hidden_size,
            'model_type': self.model_type,
            'target_modules': self.target_modules,
            'share_router_for_qkv': self.share_router_for_qkv,
            'share_router_for_w_i': self.share_router_for_w_i,
            'lambda_auxiliary': self.lambda_auxiliary,
            'lambda_lm': self.lambda_lm,
            'use_load_balancing_loss': self.use_load_balancing_loss,
        }
    
    @classmethod
    def from_config(cls, config_dict):
        """Load config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})
