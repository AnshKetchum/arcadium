from arcadium.optimizers.muon import SingleDeviceMuon, SingleDeviceMuonWithAuxAdam, DistributedMuonWithAuxAdamW
from torch.optim import AdamW

def load_optimizer(net, name="adamw", **kwargs):
    """
    Create optimizer for `net` according to `name`.

    Supported names:
      - "adam" / "adamw" : AdamW, betas=(0.9, 0.95), lr=3e-4
      - "muon"           : SingleDeviceMuon (single-GPU only), lr=0.02
      - "muon+adam" / "muon_with_aux_adam" / "muon_aux" / "muon_and_adam":
            DistributedMuonWithAuxAdamW — Muon for 2-D+ params, AdamW for 1-D params.
            All-reduces Muon grads explicitly so it is correct with or without DDP.

    Returns:
      An instantiated optimizer ready to use.
    """
    name = (name or "adamw").lower()

    if name in ("adam", "adamw"):
        lr = float(kwargs.get("lr", 3e-4))
        weight_decay = float(kwargs.get("weight_decay", 0.1))
        betas = tuple(kwargs.get("betas", (0.9, 0.95)))
        eps = float(kwargs.get("eps", 1e-8))
        return AdamW(net.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    if name == "muon":
        lr = float(kwargs.get("lr", 0.02))
        weight_decay = float(kwargs.get("weight_decay", 0))
        return SingleDeviceMuon(net.parameters(), lr=lr, weight_decay=weight_decay)

    if name in ("muon+adam", "muon_with_aux_adam", "muon_aux", "muon_and_adam"):
        muon_params = []
        adamw_params = []
        for p in net.parameters():
            if not p.requires_grad:
                continue
            # Muon for matrix-like params (weights, conv kernels); AdamW for 1-D (biases, norms).
            if p.dim() >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        lr = float(kwargs.get("lr", 3e-4))
        weight_decay = float(kwargs.get("weight_decay", 0.1))
        muon_lr = float(kwargs.get("muon_lr", lr))
        ns_steps = int(kwargs.get("ns_steps", 5))

        muon_group = {
            "params": muon_params,
            "use_muon": True,
            "lr": muon_lr,
            "momentum": 0.95,
            "weight_decay": 0,
            "ns_steps": ns_steps,
        }
        adamw_group = {
            "params": adamw_params,
            "use_muon": False,
            "lr": lr,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": weight_decay,
        }

        return DistributedMuonWithAuxAdamW([muon_group, adamw_group])

    raise ValueError(f"Unknown optimizer name '{name}'. Supported: 'adamw', 'muon', 'muon+adam'.")
