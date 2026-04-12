from dataclasses import dataclass
from typing import Optional
import torch
from transformers.utils import ModelOutput


@dataclass
class LMOutput(ModelOutput):
    """Shared output type for all Arcadium language models."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    auxiliary_loss: Optional[torch.Tensor] = None
    metadata: Optional[dict] = None
