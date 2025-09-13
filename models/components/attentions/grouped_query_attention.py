import torch
import torch.nn as nn
import torch.nn.functional as F


from models.components.activations import swiglu
from models.components.embeddings.rope import RoPE
from models.components.embeddings.sinusoidal import SinusoidalPositionalEmbedding
from models.components.attentions.utils import get_default_causal_mask

import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, num_query_heads=8, num_kv_heads = 8, embedding_dimension=1024, head_dimension = 512, **kwargs):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.num_query_heads = num_query_heads
        self.num_groups = num_kv_heads

        assert num_query_heads % num_kv_heads == 0, "number of query heads must be divisible by number of groups"
        self.num_queries_per_group = int(self.num_query_heads / self.num_groups)

        # Project full embedding dimension to 3*embedding_dimension for QKV
        self.q_proj = nn.Linear(self.embedding_dimension, self.num_query_heads* self.head_dimension)
        self.kv_proj = nn.Linear(self.embedding_dimension, 2 * self.num_groups * self.head_dimension)
        
        #nn.Linear(embedding_dimension, 3 * embedding_dimension, bias=False)
        self.o_proj = nn.Linear(self.head_dimension * self.num_queries_per_group * self.num_groups, self.embedding_dimension, bias=False)

        self.positional_embedding = SinusoidalPositionalEmbedding(self.head_dimension)
        self.metadata_storage = {}

    def forward(self, x, attention_mask = None):
        B, T, E = x.shape
        assert E == self.embedding_dimension

        # Project QKV
        q = self.q_proj(x).reshape(B, T, self.num_groups, self.num_queries_per_group, self.head_dimension) # [B, T, G, Q, D]
        kv = self.kv_proj(x).reshape(B, T, self.num_groups, 1 , 2 * self.head_dimension) # [B, T, G, 1, D]
        k, v = kv.chunk(2, dim = -1) # [B, T, G, 1, D]

        # Rearrange for multi-head: [B, G, 1/Q, T, D]
        q = q.permute(0, 2, 3, 1, 4)
        k = k.permute(0, 2, 3, 1, 4)
        v = v.permute(0, 2, 3, 1, 4)

        # Apply RoPE to queries and keys
        q = self.positional_embedding(q)
        k = self.positional_embedding(k)

        # Scaled dot-product attention
        # [B, G, Q, T, D] x [B, G, 1, T, D] => [B, G, Q, T, T]
        attn_scores = torch.einsum("b g q t d, b g k s d -> b g q t s", q, k) / math.sqrt(self.head_dimension)
        # attn_scores = torch.einsum("b q t d, b v s d -> b q v t s", q, k) / math.sqrt(self.head_dimension)

        # Causal mask for autoregressive attention
        mask = get_default_causal_mask(T, x.device)  # [1, 1, T, T]
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)

        # [B, G, Q, T, T] x [B, G, 1, T, D] => [B, G, Q, T, D] => [B, T, G, Q, D]
        attn_output = torch.einsum("b g q t s, b g k s d -> b t g q d", attn_probs, v).reshape(B, T, self.num_groups * self.num_queries_per_group * self.head_dimension)
        # attn_output = torch.einsum("b q k t s, b k s d -> b t q d", attn_probs, v).reshape(B, T, self.num_query_heads * self.head_dimension)

        # Merge heads
        out = self.o_proj(attn_output)

        self.metadata_storage = {
            "attention_probabilities" : attn_probs
        }

        return out
    
    def metadata(self):
        return self.metadata_storage