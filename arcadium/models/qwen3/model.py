import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from arcadium.models.qwen3.configuration import Qwen3Config
from arcadium.models.output import LMOutput
from einops import rearrange


def _yarn_inv_freq(
    d_head: int,
    rope_base: float,
    scale: float,
    original_max_len: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """
    YaRN-modified inverse frequencies.

    Low-frequency dims (large wavelength) are interpolated by 1/scale.
    High-frequency dims (small wavelength) are kept as-is.
    A smooth ramp blends the two in between.
    """
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, d_head, 2).float() / d_head))
    if scale == 1.0:
        return inv_freq

    wavelengths = 2 * math.pi / inv_freq
    low_wavelen = original_max_len / beta_slow    # large: low-freq boundary
    high_wavelen = original_max_len / beta_fast   # small: high-freq boundary

    # gamma = 1 → keep (high freq), gamma = 0 → interpolate (low freq)
    gamma = ((low_wavelen - wavelengths) / (low_wavelen - high_wavelen)).clamp(0.0, 1.0)
    return gamma * inv_freq + (1.0 - gamma) * inv_freq / scale


def _apply_rope(x: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to x.
    x:        (..., T, d_head)
    inv_freq: (d_head // 2,)
    """
    T = x.shape[-2]
    t = torch.arange(T, device=x.device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)                    # (T, d_head//2)
    emb = torch.cat([freqs, freqs], dim=-1)             # (T, d_head)
    cos, sin = emb.cos(), emb.sin()

    # rotate_half: split first/second half
    half = x.shape[-1] // 2
    x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + x_rot * sin


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int,
        n_query_heads: int,
        n_kv_heads: int,
        rope_base: int = 10000,
        yarn_scale: float = 1.0,
        yarn_original_max_len: int = 4096,
    ):
        super().__init__()
        assert n_query_heads % n_kv_heads == 0
        self.n_kv_heads = n_kv_heads
        self.n_query_heads = n_query_heads
        self.d_head = d_head
        self.queries_per_group = n_query_heads // n_kv_heads

        self.q_norm = nn.RMSNorm(d_head)
        self.k_norm = nn.RMSNorm(d_head)

        self.q_proj = nn.Linear(d_model, d_head * n_query_heads, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_head * n_kv_heads, bias=False)
        self.out = nn.Linear(d_head * n_query_heads, d_model, bias=False)

        inv_freq = _yarn_inv_freq(d_head, rope_base, yarn_scale, yarn_original_max_len)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # YaRN attention scale: 1 / (mscale * sqrt(d_head))
        mscale = 0.1 * math.log(yarn_scale) + 1.0 if yarn_scale > 1.0 else 1.0
        self.attn_scale = 1.0 / (mscale * math.sqrt(d_head))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x)
        k, v = self.kv_proj(x).chunk(2, dim=-1)

        q = rearrange(q, "b t (qpg g d) -> b qpg g t d", qpg=self.queries_per_group, g=self.n_kv_heads, d=self.d_head)
        k = rearrange(k, "b t (g d) -> b 1 g t d", g=self.n_kv_heads, d=self.d_head)
        v = rearrange(v, "b t (g d) -> b 1 g t d", g=self.n_kv_heads, d=self.d_head)

        # QK-norm and RoPE in fp32 for numerical precision, then cast back
        orig_dtype = q.dtype
        q = _apply_rope(self.q_norm(q.float()), self.inv_freq).to(orig_dtype)
        k = _apply_rope(self.k_norm(k.float()), self.inv_freq).to(orig_dtype)

        # Expand k/v over qpg then reshape to [B, n_query_heads, T, d_head] for SDPA
        k = k.expand(B, self.queries_per_group, self.n_kv_heads, T, self.d_head)
        v = v.expand(B, self.queries_per_group, self.n_kv_heads, T, self.d_head)
        q = q.reshape(B, self.n_query_heads, T, self.d_head)
        k = k.reshape(B, self.n_query_heads, T, self.d_head)
        v = v.reshape(B, self.n_query_heads, T, self.d_head)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.out(out)


class MLP(nn.Module):
    """SwiGLU feed-forward: gate(x) * up(x) → down."""
    def __init__(self, d_model: int, mlp_expansion_factor: float = 4.0):
        super().__init__()
        hidden = int(d_model * mlp_expansion_factor)

        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_query_heads: int,
        n_kv_heads: int,
        mlp_expansion_factor: float = 4.0,
        rope_base: int = 10000,
        yarn_scale: float = 1.0,
        yarn_original_max_len: int = 4096,
    ):
        super().__init__()
        d_head = d_model // n_query_heads

        self.pre_attention_norm = nn.RMSNorm(d_model)
        self.pre_ffn_norm = nn.RMSNorm(d_model)

        self.attention = Attention(
            d_model, d_head, n_query_heads, n_kv_heads,
            rope_base, yarn_scale, yarn_original_max_len,
        )
        self.feed_forward_network = MLP(d_model, mlp_expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attention(self.pre_attention_norm(x)) + x
        ffn = self.feed_forward_network(self.pre_ffn_norm(attn)) + attn
        return ffn


class Qwen3(PreTrainedModel):
    config_class = Qwen3Config

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            Block(
                config.d_model,
                config.n_query_heads,
                config.n_kv_heads,
                config.mlp_expansion_factor,
                config.rope_base,
                config.yarn_scale,
                config.yarn_original_max_len,
            )
            for _ in range(config.n_blocks)
        ])
        self.final_norm = nn.RMSNorm(config.d_model)
        if config.tie_word_embeddings:
            self.lm_head = self.embedding.weight
            self._tied_weights_keys = ["lm_head"]
        else:
            self.lm_head = nn.Parameter(torch.empty(config.vocab_size, config.d_model))
            nn.init.normal_(self.lm_head, std=config.d_model ** -0.5)
        self.post_init()

    @property
    def vocab_size(self):
        return self.config.vocab_size

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> LMOutput:
        h = self.embedding(input_ids)
        hidden_states = []
        for block in self.blocks:
            h = block(h)
            hidden_states.append(h.detach().float().cpu())
        logits = F.linear(self.final_norm(h), self.lm_head)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
            )
        
        metadata = {
            "hidden_states" : hidden_states
        }

        return LMOutput(loss=loss, logits=logits, metadata=metadata)
