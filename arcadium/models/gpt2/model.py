import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from arcadium.embeddings.rope import RoPE
from arcadium.models.gpt2.configuration import GPT2Config
from arcadium.models.output import LMOutput


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)

        scale = 1.0 / math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_expansion_factor: float = 4.0):
        super().__init__()
        hidden = int(d_model * mlp_expansion_factor)

        self.attention = Attention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.post_attention_norm = nn.LayerNorm(d_model)
        self.post_ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.post_attention_norm(self.attention(x, mask) + x)
        x = self.post_ffn_norm(self.feed_forward(x) + x)
        return x


class GPT2(PreTrainedModel):
    config_class = GPT2Config

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_embedding = nn.Embedding(config.max_sequence_length, config.d_model)
        self.blocks = nn.ModuleList([
            Block(config.d_model, config.n_heads, config.mlp_expansion_factor)
            for _ in range(config.n_blocks)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
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
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> LMOutput:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)

        hidden_states = []
        h = self.embedding(input_ids) + self.positional_embedding(positions)
        for block in self.blocks:
            h = block(h, mask)
            if h is not None: 
                hidden_states.append(h.detach().float().cpu())
                
        logits = self.lm_head(self.final_norm(h))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
            )
        
        metadata = {
            "hidden_states": hidden_states,
        }

        return LMOutput(loss=loss, logits=logits, metadata=metadata)
