import torch
import torch.nn as nn

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        """
        Standard sine/cosine positional embeddings (Vaswani et al.)
        dim: embedding dimension (E)
        base: frequency base (default 10000)
        """
        super().__init__()
        self.dim = dim
        self.base = base

        # Precompute the frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor):
        """
        x: [..., T, E]
        Returns: x + positional encoding, same shape
        """
        T = x.size(-2)  # sequence length
        device, dtype = x.device, x.dtype

        # [T]
        t = torch.arange(T, device=device, dtype=self.inv_freq.dtype)
        # [T, E/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # [T, E]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)

        # Shape to broadcast across all leading dims: [1,...,1,T,E]
        shape = [1] * (x.ndim - 2) + [T, self.dim]
        emb = emb.view(*shape).to(dtype=dtype)

        return x + emb
