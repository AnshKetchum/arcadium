import math
import torch 
import torch.nn.functional as F

def get_default_causal_mask(T: int, device) -> torch.Tensor:
    mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
    return mask

def scaled_dot_product_attention(q, k, v, d_k, mask):
    k_t = k.transpose(-2, -1)

    scores = q @ k_t
    scale = math.sqrt(d_k)
    scores = scores / scale

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_probs = F.softmax(scores, dim = -1)
    attn_output = torch.matmul(attn_probs, v)  # [B, N, T, H]
    return attn_probs, attn_output