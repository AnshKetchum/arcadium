import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.activations import swiglu
from models.components.rope import RoPE
import math

def scaled_dot_product_attention(q, k, v, d_k):
  k_t = k.transpose(-2, -1)
  q_kT_product = (q @ k_t) / torch.sqrt(torch.tensor(d_k))
  attention = F.softmax(q_kT_product, dim = -1)
  values = attention @ v
  return attention, values

class Attention(nn.Module):
    def __init__(self, num_heads=8, embedding_dimension=1024):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        assert embedding_dimension % num_heads == 0, "embedding dim must be divisible by num heads"
        self.head_dim = embedding_dimension // num_heads

        # Project full embedding dimension to 3*embedding_dimension for QKV
        self.qkv_proj = nn.Linear(embedding_dimension, 3 * embedding_dimension, bias=False)
        self.o_proj = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

        self.rope = RoPE(self.head_dim)
        self.metadata_storage = {}

    def forward(self, x, attention_mask = None):
        B, T, E = x.shape
        assert E == self.embedding_dimension

        # Project QKV
        qkv = self.qkv_proj(x)  # [B, T, 3*E]
        qkv = qkv.reshape(B, T, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Rearrange for multi-head: [B, N, T, H]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Apply RoPE to queries and keys
        q = self.rope(q)
        k = self.rope(k)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask for autoregressive attention
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)  # [B, N, T, H]

        # Merge heads
        out = attn_output.permute(0, 2, 1, 3).reshape(B, T, E)
        out = self.o_proj(out)

        self.metadata_storage = {
            "attention_probabilities" : attn_probs
        }

        return out
    
    def metadata(self):
        return self.metadata_storage

class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dimension = 512):
        super().__init__()
        # Project to 2*hidden_dim for SwiGLU
        self.fc1 = nn.Linear(input_dim, 2 * hidden_dimension)
        # Project back to model dimension
        self.fc2 = nn.Linear(hidden_dimension, output_dim)

    def forward(self, x):
        # x: [B, T, input_dim]
        x = self.fc1(x)            # [B, T, 2*hidden_dim]
        x = swiglu(x)              # [B, T, hidden_dim]
        x = self.fc2(x)            # [B, T, input_dim]
        return x
