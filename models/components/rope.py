import torch
import torch.nn as nn 

class RoPE(nn.Module):
    def __init__(self, dim, base=10000):
        """
        Rotary Position Embedding
        dim: head dimension (must be even)
        base: frequency base (default 10000, like in GPT/LLAMA)
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim
        self.base = base

        # Precompute inverse frequencies [dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """
        x: [batch, seq_len, n_heads, head_dim]
        seq_len: length of sequence (defaults to x.shape[1])
        returns: RoPE-applied tensor, same shape as x
        """
        if seq_len is None:
            seq_len = x.size(1)

        # [seq_len]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        # [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)

        # cos, sin terms for rotation
        cos = emb.cos()[None, :, None, :]  # [1, seq, 1, dim]
        sin = emb.sin()[None, :, None, :]

        # Apply rotation
        x1, x2 = x[..., ::2], x[..., 1::2]  # even/odd split
        x_rot = torch.cat([x1 * cos[..., ::2] - x2 * sin[..., ::2],
                           x1 * sin[..., ::2] + x2 * cos[..., ::2]], dim=-1)
        return x_rot
