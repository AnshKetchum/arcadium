import torch 
import torch.nn as nn 
import torch.nn.functional as F
from models.components.activations import swiglu

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
        """
        B, T, D = x.shape
        # Flatten batch and sequence for routing: [B*T, D]
        x_flat = x.reshape(-1, D)

        # Compute routing logits
        logits = self.router(x_flat)  # [B*T, num_experts]
        topk_val, topk_idx = torch.topk(logits, self.k, dim=-1)  # top-k experts per token
        topk_scores = F.softmax(topk_val, dim=-1)                # softmax over top-k

        # Prepare output tensor
        output_flat = torch.zeros_like(x_flat)

        # For each selected expert, compute contribution
        for i in range(self.k):
            # Mask for which token goes to which expert
            mask = F.one_hot(topk_idx[:, i], num_classes=self.num_experts).float()  # [B*T, num_experts]
            expert_mask = mask  # weight each token for that expert

            for e_idx, expert in enumerate(self.experts):
                # Select tokens routed to this expert
                token_mask = expert_mask[:, e_idx].unsqueeze(-1)  # [B*T, 1]
                if token_mask.sum() == 0:
                    continue
                expert_out = swiglu(expert[0](x_flat))  # fc1 + SwiGLU
                expert_out = expert[2](expert_out)      # fc2
                output_flat += expert_out * token_mask * topk_scores[:, i].unsqueeze(-1)

        return output_flat.reshape(B, T, D)