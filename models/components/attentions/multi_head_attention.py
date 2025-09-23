import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.embeddings.rope import RoPE
from models.components.embeddings.sinusoidal import SinusoidalPositionalEmbedding
from models.components.attentions.utils import get_default_causal_mask, scaled_dot_product_attention
from models.components.attentions.key_value_cache import load_kv_cache

class MultiHeadAttention(nn.Module):
    def __init__(self, num_kv_heads=8, embedding_dimension=1024, head_dimension = 512, **kwargs):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_kv_heads # Called num kv heads but really should be num qkv heads
        self.head_dimension = head_dimension

        # Project full embedding dimension to 3*embedding_dimension for QKV
        self.q_proj = nn.Linear(embedding_dimension, self.num_heads * self.head_dimension, bias=False)
        self.k_proj = nn.Linear(embedding_dimension, self.num_heads * self.head_dimension, bias=False)
        self.v_proj = nn.Linear(embedding_dimension, self.num_heads * self.head_dimension, bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.head_dimension, embedding_dimension, bias=False)

        self.positional_embedding = SinusoidalPositionalEmbedding(self.head_dimension)
        self.metadata_storage = {}

        self.kv_cache = load_kv_cache(kwargs.get("kv_cache", {}))

    def forward(self, x, attention_mask = None, **kwargs):
        B, T, E = x.shape
        assert E == self.embedding_dimension

        # Project QKV
        q = self.q_proj(x)

        # Try to pull from kv cache
        k, v = None, None
        if kwargs.get("use_kv_cache", False):
            k_prev, v_prev = self.kv_cache.get(
                0,
                B, 
                0, 
                T,
                0, self.num_heads
            )

            if k_prev is not None and v_prev is not None:
                k_cur = self.k_proj(x[:, -1:, :]).reshape(B, 1, self.num_heads, self.head_dimension).permute(0, 2, 1, 3)
                v_cur = self.v_proj(x[:, -1:, :]).reshape(B, 1, self.num_heads, self.head_dimension).permute(0, 2, 1, 3)

                # Apply RoPE **only to the new token**
                k_cur = self.positional_embedding(k_cur)
            
                k = torch.cat([k_prev, k_cur], dim=1)
                v = torch.cat([v_prev, v_cur], dim=1)
            
                self.kv_cache.cache(k_cur, v_cur, 0,B, T, T + 1, 0, self.num_heads)

            else: 
                k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dimension).permute(0, 2, 1, 3)
                v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dimension).permute(0, 2, 1, 3)

                k = self.positional_embedding(k)

                self.kv_cache.cache(k, v, 0,B, 0, T, 0, self.num_heads)

        else:
            k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dimension).permute(0, 2, 1, 3)
            v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dimension).permute(0, 2, 1, 3)
            
            k = self.positional_embedding(k)


        q = q.reshape(B, T, self.num_heads, self.head_dimension)

        # Rearrange for multi-head: [B, N, T, H]
        q = q.permute(0, 2, 1, 3)

        # Apply RoPE to queries and keys
        q = self.positional_embedding(q)

        # Scaled dot-product attention
        mask = get_default_causal_mask(T, x.device)
        attn_probs, attn_output = scaled_dot_product_attention(q, k, v, self.head_dimension, mask)

        # Merge heads
        out = attn_output.permute(0, 2, 1, 3).reshape(B, T, self.head_dimension * self.num_heads)
        out = self.o_proj(out)

        self.metadata_storage = {
            "attention_probabilities" : attn_probs
        }

        return out
    
    def metadata(self):
        return self.metadata_storage