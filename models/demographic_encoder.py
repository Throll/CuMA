# demographic_encoder.py

from http.client import UnimplementedFileMode
import os
from re import I
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class EmbeddingWrapper(nn.Module):
    def __init__(self, model_name: str, torch_dtype=None, device=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs = {}
        if torch_dtype:
            load_kwargs['dtype'] = torch_dtype
        if device:
            load_kwargs['device_map'] = device
        
        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.embed_dim = self.model.config.hidden_size

        # Freeze the embedding model to avoid overfitting in PEFT
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()  # Set to eval mode permanently

    @torch.no_grad()
    def encode(self, text) -> torch.Tensor:
        """Encode text to mean-pooled embedding."""
        if isinstance(text, str):
            pass
        elif isinstance(text, list) and isinstance(text[0], str):
            pass
        else:
            raise ValueError("Input text must be a string or list of strings.")
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb


class DemographicEncoder(nn.Module):
    """Convert structured demographic info into a unified embedding vector."""
    def __init__(self, model_name: str, embed_dim: int = 1024, torch_dtype=None, num_proj_layer: int = 0, device=None):
        super().__init__()
        self.embed_dim = embed_dim
        if embed_dim == 0:
            self.text_encoder = None
            self.proj = nn.Identity()
            return

        self.text_encoder = EmbeddingWrapper(model_name=model_name, torch_dtype=torch_dtype, device=device)
        self.cache = {}
        
        input_dim = self.text_encoder.embed_dim
        if num_proj_layer == 0:
            self.proj = nn.Identity()
        elif num_proj_layer == 1:
            self.proj = nn.Linear(input_dim, embed_dim, dtype=torch_dtype)
            nn.init.xavier_normal_(self.proj.weight, gain=0.01)
            nn.init.zeros_(self.proj.bias)
        elif num_proj_layer == 2:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, input_dim, dtype=torch_dtype),
                nn.GELU(),
                nn.Linear(input_dim, embed_dim, dtype=torch_dtype)
            )
            nn.init.xavier_normal_(self.proj[2].weight, gain=0.01)
            nn.init.zeros_(self.proj[2].bias)
            nn.init.kaiming_normal_(self.proj[0].weight)
            nn.init.zeros_(self.proj[0].bias)
        else:
            raise ValueError(f"Unsupported num_proj_layer: {num_proj_layer}. Only 0, 1, or 2 are supported.")

    def forward(self, text) -> torch.Tensor:
        """Forward pass: embed text → project."""
        if self.embed_dim == 0:
            # Return empty tensor with correct batch size
            batch_size = 1 if isinstance(text, str) else len(text)
            # Use same device and dtype as this module (Identity)
            # Although Identity has no params, we can use cpu/float as fallback
            # but it is better to return it in a way that torch.cat won't complain.
            return torch.zeros((batch_size, 0)).to(
                device=next(self.parameters()).device if any(self.parameters()) else "cpu"
            )

        device = self.proj.weight.device if hasattr(self.proj, 'weight') and hasattr(self.proj.weight, 'device') else next(self.text_encoder.parameters()).device
        
        # Handle single string
        if isinstance(text, str):
            if text in self.cache:
                return self.cache[text].to(device).unsqueeze(0) # Add batch dim
            
            text_emb = self.text_encoder.encode(text)
            z_demo = self.proj(text_emb)
            self.cache[text] = z_demo.detach().cpu().squeeze(0) # Store without batch dim
            return z_demo

        # Handle list of strings
        if isinstance(text, list):
            # Identify indices that need encoding
            indices_to_encode = []
            texts_to_encode = []
            cached_tensors = [None] * len(text)
            
            for i, t in enumerate(text):
                if t in self.cache:
                    cached_tensors[i] = self.cache[t].to(device)
                else:
                    indices_to_encode.append(i)
                    texts_to_encode.append(t)
            
            # Encode missing
            if texts_to_encode:
                # print(f"Encoding {len(texts_to_encode)} new demographics on {device}")
                text_emb = self.text_encoder.encode(texts_to_encode)
                z_demo_new = self.proj(text_emb)
                
                for i, idx in enumerate(indices_to_encode):
                    emb = z_demo_new[i]
                    cached_tensors[idx] = emb
                    self.cache[texts_to_encode[i]] = emb.detach().cpu()
            
            return torch.stack(cached_tensors)

        # Fallback for other types (e.g. tensor)
        text_emb = self.text_encoder.encode(text)
        z_demo = self.proj(text_emb)
        return z_demo 


# ---------------------------------------------------------
# Example usage (standalone test)
# ---------------------------------------------------------
if __name__ == "__main__":
    demo_profile = {
        "country": "Egypt",
        "age_group": "16-24 years old",
        "gender": "Male",
        "education": "Upper secondary education (ISCED 3)",
        "marital_status": "Single",
        "religion": "Muslim",
        "ethnicity": "EG: Coptic",
        "employment": "Unemployed"
    }

    # Test natural mode
    encoder_natural = DemographicEncoder(embed_dim=1024, mode="natural")
    z_demo_natural = encoder_natural(demo_profile)
    print("Natural mode z_demo shape:", z_demo_natural.shape)
    print("Natural mode sample:", z_demo_natural[0, :8])

    # Test structured mode
    # encoder_structured = DemographicEncoder(embed_dim=1024, mode="structured")
    # z_demo_structured = encoder_structured(demo_profile)
    # print("\nStructured mode z_demo shape:", z_demo_structured.shape)
    # print("Structured mode sample:", z_demo_structured[0, :8])