import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from arcadium.embeddings.rope import RoPE
from arcadium.models.olmo2.configuration import OLMo2Config
from arcadium.models.output import LMOutput


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope_base: int = 10000):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = nn.RMSNorm(self.d_head)
        self.k_norm = nn.RMSNorm(self.d_head)
        self.rope = RoPE(self.d_head, base=rope_base)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)

        # reshape to (B, T, n_heads, d_head) for RoPE, then transpose to (B, H, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.n_heads, self.d_head)

        q = self.q_norm(self.rope(q))
        k = self.k_norm(self.rope(k))

        q = q.transpose(1, 2)  # (B, H, T, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)

        scale = 1.0 / math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
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
    def __init__(self, d_model: int, n_heads: int, mlp_expansion_factor: float = 4.0, rope_base: int = 10000):
        super().__init__()
        self.attention = Attention(d_model, n_heads, rope_base)
        self.feed_forward = MLP(d_model, mlp_expansion_factor)
        self.post_attention_norm = nn.RMSNorm(d_model)
        self.post_ffn_norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # post-norm: normalize after residual addition (OLMo2 style)
        x = self.post_attention_norm(self.attention(x, mask) + x)
        x = self.post_ffn_norm(self.feed_forward(x) + x)
        return x


class OLMo2(PreTrainedModel):
    config_class = OLMo2Config

    def __init__(self, config: OLMo2Config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            Block(config.d_model, config.n_heads, config.mlp_expansion_factor, config.rope_base)
            for _ in range(config.n_blocks)
        ])
        self.final_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> LMOutput:
        h = self.embedding(input_ids)
        for block in self.blocks:
            h = block(h, mask)
        logits = self.lm_head(self.final_norm(h))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
            )

        return LMOutput(loss=loss, logits=logits)
