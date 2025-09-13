import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.activations import swiglu
from models.components.embeddings.rope import RoPE
import math

class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dimension = 512):
        super().__init__()
        # Project to 2*hidden_dim for SwiGLU
        self.fc1 = nn.Linear(input_dim, 2 * hidden_dimension)
        # Project back to model dimension
        self.fc2 = nn.Linear(hidden_dimension, output_dim)

    def forward(self, x):
        # x: [B, T, input_dim]
        x = self.fc1(x)            # [B, T, 2*hidden_dim]
        x = swiglu(x)              # [B, T, hidden_dim]
        x = self.fc2(x)            # [B, T, input_dim]
        return x
