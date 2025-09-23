import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.activations import swiglu
from models.components.embeddings.rope import RoPE
from models.components.embeddings.sinusoidal import SinusoidalPositionalEmbedding
from models.components.attentions.utils import get_default_causal_mask, scaled_dot_product_attention
from models.components.attentions.key_value_cache import load_kv_cache

import math

class MultiQueryAttention(nn.Module):
    def __init__(self, num_query_heads=8, embedding_dimension=1024, head_dimension=512, **kwargs):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.num_query_heads = num_query_heads
        assert embedding_dimension % num_query_heads == 0, "embedding dim must be divisible by num heads"

        self.q_proj = nn.Linear(self.embedding_dimension, self.num_query_heads * self.head_dimension, bias=False)
        self.k_proj = nn.Linear(self.embedding_dimension, self.head_dimension, bias=False)
        self.v_proj = nn.Linear(self.embedding_dimension, self.head_dimension, bias=False)

        self.o_proj = nn.Linear(self.head_dimension * self.num_query_heads, self.embedding_dimension, bias=False)

        self.positional_embedding = SinusoidalPositionalEmbedding(self.head_dimension)
        self.metadata_storage = {}

        self.kv_cache = load_kv_cache(kwargs.get("kv_cache", {}))

    def forward(self, x, attention_mask=None, **kwargs):
        B, T, E = x.shape
        assert E == self.embedding_dimension

        # Project queries
        q = self.q_proj(x).reshape(B, T, self.num_query_heads, self.head_dimension)

        k, v = None, None
        if kwargs.get("use_kv_cache", False):
            k_prev, v_prev = self.kv_cache.get(
                0,
                B,
                0,
                T,
                0,  # start idx
                1   # only 1 kv head in MQA
            )

            if k_prev is not None and v_prev is not None:
                # Only compute for the last token
                k_cur = self.k_proj(x[:, -1:, :]).reshape(B, 1, 1, self.head_dimension)
                v_cur = self.v_proj(x[:, -1:, :]).reshape(B, 1, 1, self.head_dimension)

                # Apply RoPE to new key
                k_cur = self.positional_embedding(k_cur)

                # Concatenate cached and new
                k = torch.cat([k_prev, k_cur], dim=1)
                v = torch.cat([v_prev, v_cur], dim=1)

                # Update cache
                self.kv_cache.cache(k_cur, v_cur, 0, B, T, T + 1, 0, 1)

            else:
                k = self.k_proj(x).reshape(B, T, 1, self.head_dimension)
                v = self.v_proj(x).reshape(B, T, 1, self.head_dimension)

                k = self.positional_embedding(k)

                self.kv_cache.cache(k, v, 0, B, 0, T, 0, 1)

        else:
            k = self.k_proj(x).reshape(B, T, 1, self.head_dimension)
            v = self.v_proj(x).reshape(B, T, 1, self.head_dimension)

            k = self.positional_embedding(k)

        # Rearrange for attention
        q = q.permute(0, 2, 1, 3)  # [B, Q, T, H]
        k = k.permute(0, 2, 1, 3)  # [B, 1, T, H]
        v = v.permute(0, 2, 1, 3)  # [B, 1, T, H]

        # Apply RoPE to queries
        q = self.positional_embedding(q)

        # Scaled dot-product attention
        mask = get_default_causal_mask(T, x.device)
        attn_probs, attn_output = scaled_dot_product_attention(q, k, v, self.head_dimension, mask)

        # Merge heads
        out = attn_output.permute(0, 2, 1, 3).reshape(B, T, self.num_query_heads * self.head_dimension)
        out = self.o_proj(out)

        self.metadata_storage = {
            "attention_probabilities": attn_probs
        }

        return out
    
    def metadata(self):
        return self.metadata_storage
    
    def reset(self):
        self.kv_cache.reset()
