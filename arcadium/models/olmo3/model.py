import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from arcadium.embeddings.rope import RoPE
from arcadium.models.olmo3.configuration import OLMo3Config
from arcadium.models.output import LMOutput
from einops import einsum, rearrange


def rotary_embedding_tensor(input_tensor: torch.Tensor, rope_base: int = 10000) -> torch.Tensor:
    """ 
    
    1. create a (1, d) vector to with the inv freq angle: 

        theta = -2j / d 
        
        by interleaving two d/2 vectors
    
    2. repeat the vector to create a (1, T, d) tensor
    
    3. exp everything to get final (1, T, d) tensor
    
    
    """
    B, sequence_length, n, d_model = input_tensor.shape
    assert d_model % 2 == 0, "d_model should be evenly divisble"

    # create the single rope vector
    rope_frequencies = -2 * torch.arange(0, d_model // 2) / d_model

    # interleave the vector
    rope_dim_freqs = torch.empty((1, d_model))
    rope_dim_freqs[:, 0::2] = rope_frequencies
    rope_dim_freqs[:, 1::2] = rope_frequencies

    # repeat for each position and 
    seqlen_rope_dim_freqs = torch.repeat_interleave(rope_base ** rope_dim_freqs, sequence_length,  dim=0).unsqueeze(0)

    # (1 T) (1 T E) -> (1 T E)
    seq_tensor = torch.arange(sequence_length).unsqueeze(0)
    ret = torch.exp(einsum(seq_tensor, seqlen_rope_dim_freqs, "b t, b t e -> b t e") * 1j)

    # returns e^(m)
    return input_tensor * ret

class Attention(nn.Module):
    def __init__(self, d_model: int, n_query_heads, n_kv_heads: int, rope_base: int = 10000, sliding_window: int = -1):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.n_query_heads = n_query_heads
        self.d_head = d_model // n_kv_heads
        self.rope_base = rope_base
        self.sliding_window = sliding_window
        assert n_query_heads % n_kv_heads == 0
        self.queries_per_group = n_query_heads // n_kv_heads

        self.q_norm = nn.RMSNorm(self.d_head)
        self.k_norm = nn.RMSNorm(self.d_head)

        self.q_proj = nn.Linear(d_model, self.d_head * self.n_kv_heads * self.queries_per_group, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * self.d_head * self.n_kv_heads, bias=False)
        self.out = nn.Linear(self.d_head * n_query_heads * n_kv_heads, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)
        k, v = self.kv_proj(x).split(C, dim=-1)

        # (B, queries_per_group, groups, T, d_head)
        q = rearrange(q.view(B, T, self.queries_per_group, self.n_kv_heads, self.d_head), "b t q g d -> b q g t d")
        q = self.q_norm(rotary_embedding_tensor(q, self.rope_base)) # QK norm

        # (B, 1, groups, T, d_head)
        k = rearrange(k.view(B, T, 1, self.n_kv_heads, self.d_head), "b t q g d -> b q g t d")
        k = self.k_norm(rotary_embedding_tensor(k, self.rope_base))

        v = rearrange(v.view(B, T, 1, self.n_kv_heads, self.d_head), "b t q g d -> b q g t d")

        # causal mask default
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, 1, T, T)
        
        if self.sliding_window != -1:
            # update the mask so that row i in the attention matrix at best looks at positions [i - sliding_window, i]

            # create a matrix of distances
            rows = torch.arange(0, T, device=x.device).reshape(T, 1) 
            cols = torch.arange(0, T, device=x.device).reshape(1, T)

            sliding_window_mask = (rows - cols) <= self.sliding_window
            mask = sliding_window_mask.view(1, 1, 1, T, T).int() * mask

        scale = 1.0 / math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        # (B, queries_per_group, groups, T, d_head)
        out = rearrange(attn @ v, "b q g t d -> b t (q g d)")
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
    def __init__(self, d_model: int, n_query_heads: int, n_kv_heads: int, mlp_expansion_factor = 4.0, sliding_window: int = -1):
        super().__init__() 

        self.d_model = d_model 
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads

        self.pre_attention_norm = nn.RMSNorm(d_model)
        self.pre_ffn_norm = nn.RMSNorm(d_model)

        self.post_attention_norm = nn.RMSNorm(d_model)
        self.post_ffn_norm = nn.RMSNorm(d_model)

        self.attention = Attention(d_model, n_query_heads, n_kv_heads, sliding_window=sliding_window)

        self.feed_forward_network = MLP(d_model, mlp_expansion_factor)
    
    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        attn = self.post_attention_norm(self.attention(x, mask)) + x       
        ffn = self.post_ffn_norm(self.feed_forward_network(attn)) + attn
        return ffn

class OLMo3(PreTrainedModel):
    config_class = OLMo3Config

    def __init__(self, config: OLMo3Config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([Block(config.d_model, config.n_query_heads, config.n_kv_heads, sliding_window=(-1 if i and (i + 1) % (int(config.sliding_window_ratio) + 1) == 0 else int(config.sliding_window_ratio))) for i in range(config.n_blocks)])
        self.final_norm = nn.RMSNorm(config.d_model)
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
