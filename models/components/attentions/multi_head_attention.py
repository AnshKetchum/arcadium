import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.activations import swiglu
from models.components.embeddings.rope import RoPE
from models.components.embeddings.sinusoidal import SinusoidalPositionalEmbedding
from models.components.attentions.utils import get_default_causal_mask

import math

def scaled_dot_product_attention(q, k, v, d_k):
    k_t = k.transpose(-2, -1)
    q_kT_product = (q @ k_t) / torch.sqrt(torch.tensor(d_k))
    attention = F.softmax(q_kT_product, dim = -1)
    values = attention @ v
    return attention, values

class MultiHeadAttention(nn.Module):
    def __init__(self, num_kv_heads=8, embedding_dimension=1024, head_dimension = 512):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_kv_heads # Called num kv heads but really should be num qkv heads
        self.head_dimension = head_dimension

        # Project full embedding dimension to 3*embedding_dimension for QKV
        self.qkv_proj = nn.Linear(embedding_dimension, 3 * self.num_heads * self.head_dimension, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dimension, embedding_dimension, bias=False)

        self.positional_embedding = SinusoidalPositionalEmbedding(self.head_dimension)
        self.metadata_storage = {}

    def forward(self, x, attention_mask = None):
        B, T, E = x.shape
        assert E == self.embedding_dimension

        # Project QKV
        qkv = self.qkv_proj(x)  # [B, T, 3*E]
        qkv = qkv.reshape(B, T, self.num_heads, 3 * self.head_dimension)
        q, k, v = qkv.chunk(3, dim=-1)

        # Rearrange for multi-head: [B, N, T, H]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Apply RoPE to queries and keys
        q = self.positional_embedding(q)
        k = self.positional_embedding(k)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dimension)

        # Causal mask for autoregressive attention
        mask = get_default_causal_mask(T, x.device)
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)  # [B, N, T, H]

        # Merge heads
        out = attn_output.permute(0, 2, 1, 3).reshape(B, T, self.head_dimension * self.num_heads)
        out = self.o_proj(out)

        self.metadata_storage = {
            "attention_probabilities" : attn_probs
        }

        return out
    
    def metadata(self):
        return self.metadata_storage