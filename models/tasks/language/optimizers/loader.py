from models.tasks.language.optimizers.muon import SingleDeviceMuon, SingleDeviceMuonWithAuxAdam
from torch.optim import Adam

def load_optimizer(net, name = "adam", **kwargs):
    """
    Create optimizer for `net` according to `name`.

    Supported names:
      - "adam" : torch.optim.Adam with lr=3e-4 (default)
      - "muon" : SingleDeviceMuon with lr=0.02 (default)
      - "muon+adam", "muon_with_aux_adam", "muon_aux" : SingleDeviceMuonWithAuxAdam
          uses Muon for parameters with dim >= 2 and Adam for 1-D parameters.

    Returns:
      An instantiated optimizer ready to use.
    """
    name = (name or "adam").lower()

    if name == "adam":
        lr = kwargs.get("lr", 3e-4)
        weight_decay = kwargs.get("weight_decay", 0)
        return Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    if name == "muon":
        lr = kwargs.get("lr", 3e-4)
        weight_decay = kwargs.get("weight_decay", 0)
        return SingleDeviceMuon(net.parameters(), lr=lr, weight_decay=weight_decay)

    if name in ("muon+adam", "muon_with_aux_adam", "muon_aux", "muon_and_adam"):
        muon_params = []
        adam_params = []
        for p in net.parameters():
            if not p.requires_grad:
                continue
            # Heuristic: apply Muon to matrix-like params (weights / conv kernels),
            # apply Adam to 1-D params (biases, layernorm/scale vectors).
            if p.dim() >= 2:
                muon_params.append(p)
            else:
                adam_params.append(p)

        lr = kwargs.get("lr", 3e-4)
        weight_decay = kwargs.get("weight_decay", 0)

        # Build groups exactly as SingleDeviceMuonWithAuxAdam expects.
        muon_group = {
            "params": muon_params,
            "use_muon": True,
            "lr": lr,
            "momentum": 0.95,
            "weight_decay": weight_decay,
        }
        adam_group = {
            "params": adam_params,
            "use_muon": False,
            "lr": lr,
            "betas": (0.9, 0.95),
            "eps": 1e-10,
            "weight_decay": weight_decay,
        }

        return SingleDeviceMuonWithAuxAdam([muon_group, adam_group])

    raise ValueError(f"Unknown optimizer name '{name}'. Supported: 'adam', 'muon', 'muon+adam'.")

    