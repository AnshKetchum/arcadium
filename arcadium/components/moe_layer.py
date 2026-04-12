import torch
import torch.nn as nn
import torch.nn.functional as F
from arcadium.components.activations import swiglu

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dimension, num_experts=4, k=2):
        """
        input_dim: model hidden dimension
        hidden_dim: intermediate FFN dimension per expert
        num_experts: total number of experts
        k: top-k experts to route each token to
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dimension
        self.num_experts = num_experts
        self.k = k

        # Define each expert as a SwiGLU FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 2 * hidden_dimension),
                nn.SiLU(),            # equivalent to swiglu but without gating, will implement proper swiglu in forward
                nn.Linear(hidden_dimension, output_dim)
            )
            for _ in range(num_experts)
        ])

        # Router: maps input tokens to expert logits
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        x: [B, T, input_dim]
        Returns (output, aux_loss) where aux_loss is the load-balancing loss.
        """
        B, T, D = x.shape
        # Flatten batch and sequence for routing: [B*T, D]
        x_flat = x.reshape(-1, D)

        # Compute routing logits
        router_logits = self.router(x_flat)  # [N, num_experts]
        # Full softmax for load-balancing loss (differentiable)
        router_probs = F.softmax(router_logits, dim=-1)  # [N, num_experts]

        topk_val, topk_idx = torch.topk(router_logits, self.k, dim=-1)  # [N, k]
        topk_scores = F.softmax(topk_val, dim=-1)                        # [N, k]

        # Load-balancing loss (Switch Transformer style):
        #   f_i = fraction of tokens dispatched to expert i (hard, non-differentiable)
        #   p_i = mean soft router probability for expert i (differentiable)
        #   L = num_experts * sum_i(f_i * p_i)
        one_hot = F.one_hot(topk_idx, num_classes=self.num_experts).float()  # [N, k, E]
        f = one_hot.sum(dim=1).mean(dim=0)  # [E]
        p = router_probs.mean(dim=0)        # [E]
        aux_loss = self.num_experts * (f.detach() * p).sum()

        # Prepare output tensor
        output_flat = torch.zeros_like(x_flat)

        # For each selected expert, compute contribution
        for i in range(self.k):
            expert_mask = F.one_hot(topk_idx[:, i], num_classes=self.num_experts).float()  # [N, E]

            for e_idx, expert in enumerate(self.experts):
                token_mask = expert_mask[:, e_idx].unsqueeze(-1)  # [N, 1]
                if token_mask.sum() == 0:
                    continue
                expert_out = swiglu(expert[0](x_flat))  # fc1 + SwiGLU
                expert_out = expert[2](expert_out)      # fc2
                output_flat += expert_out * token_mask * topk_scores[:, i].unsqueeze(-1)

        return output_flat.reshape(B, T, D), aux_loss
