import torch 
import torch.nn as nn 
import os, time 

def _activation_size(tensor: torch.Tensor) -> float:
    """Return memory footprint of a tensor in MB."""
    if not torch.is_tensor(tensor):
        return 0.0
    return tensor.numel() * tensor.element_size() / (1024 ** 2)


def register_activation_hooks(model: nn.Module, activation_stats: dict):
    """
    Registers forward hooks on each submodule to record activation memory.
    activation_stats is a dict[layer_name] = list of sizes across iterations.
    """
    hooks = []

    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            # skip containers (Sequential, ModuleList, etc.)
            continue

        def hook_fn(module, inp, out, name=name):
            size = 0.0
            if torch.is_tensor(out):
                size = _activation_size(out)
            elif isinstance(out, (list, tuple)):
                size = sum(_activation_size(o) for o in out if torch.is_tensor(o))
            activation_stats[name] = size

        hooks.append(module.register_forward_hook(hook_fn))

    return hooks
