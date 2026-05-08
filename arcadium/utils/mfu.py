"""
MFU (Model FLOPs Utilization) estimation.

Forward FLOPs are measured once at training start with fvcore on the actual
model + a representative input shape. We then divide observed FLOPs/second by
the device's advertised dense bf16 peak to get MFU.

Caveats:
  - fvcore counts linear / matmul / bmm / einsum / conv ops. It does NOT count
    `F.scaled_dot_product_attention` (the kernel is opaque to fvcore). A custom
    handler is registered below to attribute attention FLOPs as 4·B·H·T²·D.
  - bf16 peak is dense (no 2:1 sparsity). Numbers below are SXM where relevant.
"""
from __future__ import annotations

import torch

# Dense bf16 TFLOPs/s, no sparsity. SXM where applicable.
_BF16_PEAK_TFLOPS = {
    "H200": 989.0,
    "H100": 989.0,
    "A100": 312.0,
    "L40S": 362.0,
    "L40":  181.0,
    "RTX 6000 Ada": 364.0,
    "RTX 5090":     419.0,
    "RTX 4090":     165.0,
}


def device_peak_bf16_tflops(device: torch.device | int = 0) -> float:
    """Return advertised dense bf16 TFLOPs for `device`. 0.0 if unknown."""
    if not torch.cuda.is_available():
        return 0.0
    name = torch.cuda.get_device_name(device)
    for key, peak in _BF16_PEAK_TFLOPS.items():
        if key in name:
            return peak
    return 0.0


def _sdpa_flop_handler(inputs, outputs):
    """fvcore handler for aten::scaled_dot_product_attention.

    Attributes 4·H·T_q·T_k·D FLOPs per batch element (QK^T + AV). Causal masking
    halves the work in principle but most fused causal kernels still issue the
    full matmul + mask, so we keep the dense estimate as an upper bound on what
    the silicon must actually do.

    fvcore passes `torch._C.Value` objects (TorchScript IR), so we read shapes
    via `.type().sizes()` rather than `.shape`.
    """
    def _shape(v):
        try:
            return v.type().sizes()
        except Exception:
            return None

    q_shape = _shape(inputs[0])
    k_shape = _shape(inputs[1])
    if not q_shape or not k_shape:
        return 0
    # Q: (..., H_q, T_q, D); K: (..., H_kv, T_k, D)
    h_q, t_q, d_head = q_shape[-3], q_shape[-2], q_shape[-1]
    h_k, t_k = k_shape[-3], k_shape[-2]
    # Multiply over leading batch dims so we count the whole tensor's work.
    batch_prod = 1
    for s in q_shape[:-3]:
        batch_prod *= s
    h = max(h_q, h_k)  # GQA work happens at query head resolution
    return batch_prod * (2 * h * t_q * t_k * d_head + 2 * h * t_q * t_k * d_head)


def compute_forward_flops(net: torch.nn.Module, example_input: torch.Tensor) -> int:
    """Run fvcore once on `net(example_input)` and return total forward FLOPs.

    Falls back to 0 if fvcore is missing — caller should treat 0 as "MFU not
    available" and skip logging it.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return 0

    import gc
    was_training = net.training
    net.eval()
    flops = None
    try:
        with torch.no_grad():
            flops = FlopCountAnalysis(net, (example_input,))
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            flops.set_op_handle("aten::scaled_dot_product_attention", _sdpa_flop_handler)
            total = int(flops.total())
    except Exception as e:
        import traceback
        print(f"compute_forward_flops: trace failed ({type(e).__name__}: {e})")
        traceback.print_exc()
        total = 0
    finally:
        del flops
        if was_training:
            net.train()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return total


def forward_mfu(
    forward_flops_per_microbatch: int,
    grad_accum_steps: int,
    forward_seconds: float,
    world_size: int,
    peak_tflops_per_gpu: float,
) -> float:
    """Forward-only MFU as a fraction in [0, 1].

    forward_flops_per_microbatch is the per-rank flops from one forward pass.
    The denominator is per-rank peak × wall-clock forward time × world_size /
    world_size = peak × time, since each rank independently does its own
    forward. We return per-rank MFU.
    """
    if forward_flops_per_microbatch <= 0 or peak_tflops_per_gpu <= 0 or forward_seconds <= 0:
        return 0.0
    achieved = forward_flops_per_microbatch * grad_accum_steps  # per rank, per step
    peak = peak_tflops_per_gpu * 1e12 * forward_seconds         # per rank
    return achieved / peak
