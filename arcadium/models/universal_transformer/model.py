import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from arcadium.models.universal_transformer.configuration import UniversalTransformerConfig
from arcadium.models.output import LMOutput


# ---------------------------------------------------------------------------
# Universal Transformer
# ---------------------------------------------------------------------------

def sinusoidal_encoding(seq_len: int, d_model: int, device) -> torch.Tensor:
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def step_encoding(step: int, seq_len: int, d_model: int, device) -> torch.Tensor:
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(torch.tensor(step, dtype=torch.float, device=device) * div_term)
    pe[:, 1::2] = torch.cos(torch.tensor(step, dtype=torch.float, device=device) * div_term)
    return pe.unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class UniversalTransformer(PreTrainedModel):
    config_class = UniversalTransformerConfig

    def __init__(self, config: UniversalTransformerConfig):
        super().__init__(config)
        self.d_model   = config.d_model
        self.max_steps = config.max_steps
        self.eps       = config.eps
        self.tau       = config.tau

        self.embedding  = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks     = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_blocks)
        ])
        self.loop_gates = nn.ModuleList([nn.Linear(config.d_model, 1) for _ in range(config.n_blocks)])
        for gate in self.loop_gates:
            nn.init.constant_(gate.bias, 1.0)
        self.head = nn.Linear(config.d_model, config.vocab_size)
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
        collect_hidden_states: bool = False,
        analysis_mode: bool = False,
        n_steps: int = None,
        **kwargs,
    ) -> LMOutput:
        B, T = input_ids.shape
        device = input_ids.device

        o = self.embedding(input_ids)
        o = o + sinusoidal_encoding(T, self.d_model, device)

        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        act_loss = torch.tensor(0.0, device=device)
        hidden_states = [] if collect_hidden_states else None
        gate_probs       = [] if analysis_mode else None
        pct_halted_steps = [] if analysis_mode else None

        halting_prob  = input_ids.new_zeros(B, T, dtype=torch.float)
        remainders    = input_ids.new_zeros(B, T, dtype=torch.float)
        n_updates     = input_ids.new_zeros(B, T, dtype=torch.float)
        still_running = input_ids.new_ones(B, T, dtype=torch.bool)
        final_state   = input_ids.new_zeros(B, T, self.d_model, dtype=torch.float)

        if hidden_states is not None:
            hidden_states.append(o.detach().float().cpu())

        if self.max_steps > 1:
            steps = n_steps if n_steps is not None else self.max_steps

            for block, gate in zip(self.blocks, self.loop_gates):
                halting_prob  = input_ids.new_zeros(B, T, dtype=torch.float)
                remainders    = input_ids.new_zeros(B, T, dtype=torch.float)
                n_updates     = input_ids.new_zeros(B, T, dtype=torch.float)
                still_running = input_ids.new_ones(B, T, dtype=torch.bool)

                for step in range(steps):
                    o = o + step_encoding(step, T, self.d_model, device)
                    o = block(o, mask=causal_mask)

                    if hidden_states is not None:
                        hidden_states.append(o.detach().float().cpu())

                    p = torch.sigmoid(gate(o).squeeze(-1))

                    if gate_probs is not None:
                        gate_probs.append(p.detach().float().cpu())

                    new_halted = (
                        still_running
                        & ((halting_prob + p) >= 1.0 - self.eps)
                    )
                    still_running_next = (
                        still_running
                        & ((halting_prob + p) < 1.0 - self.eps)
                    )

                    remainders   += new_halted.float() * (1.0 - halting_prob).clamp(min=0)
                    halting_prob = (halting_prob + (
                        new_halted.float()           * (1.0 - halting_prob)
                        + still_running_next.float() * p
                    )).clamp(0, 1)

                    update_weights = (
                        new_halted.float()           * remainders
                        + still_running_next.float() * p
                    )

                    final_state += update_weights.unsqueeze(-1) * o

                    n_updates += still_running.float()

                    o = torch.where(
                        (~still_running_next).unsqueeze(-1),
                        o.detach(),
                        o,
                    )

                    still_running = still_running_next

                    if pct_halted_steps is not None:
                        pct_halted_steps.append((~still_running).float().mean().item())

                    if not analysis_mode and not still_running.any():
                        break

                ponder_time = n_updates + remainders
                act_loss += self.tau * ponder_time.mean()

        else:
            # standard stacked blocks mode (n_blocks > 1, no looping)
            blocks_to_run = self.blocks[:n_steps] if n_steps is not None else self.blocks
            for block in blocks_to_run:
                o = block(o, mask=causal_mask)
                if hidden_states is not None:
                    hidden_states.append(o.detach().float().cpu())
            final_state = o

        logits = self.head(final_state)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
            )

        metadata = {
            "n_updates": n_updates,
            "remainders": remainders,
            "metrics/model/avg_loops": n_updates.mean(),
            "metrics/model/avg_remainder": remainders.mean(),
            "hidden_states": hidden_states,
            "gate_probs": gate_probs,
            "pct_halted_steps": pct_halted_steps,
            "last_hidden": o,
        }

        return LMOutput(
            loss=loss,
            logits=logits,
            auxiliary_loss=act_loss,
            metadata=metadata,
        )
