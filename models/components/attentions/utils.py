import torch 

def get_default_causal_mask(T: int, device) -> torch.Tensor:
    mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
    return mask
