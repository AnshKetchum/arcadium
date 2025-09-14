import torch
import torch.nn as nn
import torch.nn.functional as F


from models.components.activations import swiglu
from models.components.embeddings.rope import RoPE
from models.components.embeddings.sinusoidal import SinusoidalPositionalEmbedding
from models.components.attentions.utils import get_default_causal_mask, scaled_dot_product_attention

import math

class MultiQueryAttention(nn.Module):
    def __init__(self, num_query_heads=8, embedding_dimension=1024, head_dimension = 512, **kwargs):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.num_query_heads = num_query_heads
        assert embedding_dimension % num_query_heads == 0, "embedding dim must be divisible by num heads"

        # Project full embedding dimension to 3*embedding_dimension for QKV
        self.q_proj = nn.Linear(self.embedding_dimension, self.num_query_heads* self.head_dimension)
        self.kv_proj = nn.Linear(self.embedding_dimension, 2 * self.head_dimension)
        
        #nn.Linear(embedding_dimension, 3 * embedding_dimension, bias=False)
        self.o_proj = nn.Linear(self.head_dimension * self.num_query_heads, self.embedding_dimension, bias=False)

        self.positional_embedding = SinusoidalPositionalEmbedding(self.head_dimension)
        self.metadata_storage = {}

    def forward(self, x, attention_mask = None):
        B, T, E = x.shape
        assert E == self.embedding_dimension

        # Project QKV
        q = self.q_proj(x).reshape(B, T, self.num_query_heads, self.head_dimension) # [B, T, QD]
        kv = self.kv_proj(x).reshape(B, T, 2 * self.head_dimension).unsqueeze(2)
        
        k, v = kv.chunk(2, dim = -1) # [B, T, 1, H]

        # Rearrange for multi-head: [B, Q, T, H]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Apply RoPE to queries and keys
        q = self.positional_embedding(q)
        k = self.positional_embedding(k)

        # Scaled dot-product attention
        mask = get_default_causal_mask(T, x.device)  # [1, 1, T, T]
        attn_probs, attn_output = scaled_dot_product_attention(q, k, v, self.head_dimension, mask)

        # Merge heads
        out = attn_output.permute(0, 2, 1, 3).reshape(B, T, self.num_query_heads * self.head_dimension) 
        out = self.o_proj(out)

        self.metadata_storage = {
            "attention_probabilities" : attn_probs
        }

        return out
    
    def metadata(self):
        return self.metadata_storage