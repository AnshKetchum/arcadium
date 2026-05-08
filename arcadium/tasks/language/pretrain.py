import argparse
import contextlib
from collections import defaultdict
import inspect
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from tqdm import tqdm
import wandb
import math
import glob
import re
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from safetensors.torch import load_file, save_file
from arcadium.tasks.language.loader import load_language_model, load_dataset
from arcadium.data.sequence_length import SequenceLengthSampler
from arcadium.optimizers.loader import load_optimizer
from arcadium.utils import load_config
from arcadium.utils.hooks import register_activation_hooks
from arcadium.utils.mfu import compute_forward_flops, device_peak_bf16_tflops
from arcadium.utils.visualize import plot_visualizations
from arcadium.tasks.language.generate import generate
from dotenv import load_dotenv
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from torch.distributed.optim import ZeroRedundancyOptimizer

load_dotenv()


def _unwrap(net):
    """Strip DDP, FSDP, and torch.compile wrappers to get the bare nn.Module.

    Order matters: DDP/FSDP wrap an OptimizedModule (when --compile is on),
    so unwrap DDP first, then peel off `_orig_mod` from the compiled wrapper.
    """
    if isinstance(net, (DDP, FSDP)):
        net = net.module
    if hasattr(net, "_orig_mod"):
        net = net._orig_mod
    return net


def _eval_module(net):
    """Module to use for inference (validation, generation, viz).

    For FSDP we keep the wrapper because parameters are sharded and only the
    FSDP forward knows how to all-gather them. Otherwise we peel DDP and the
    torch.compile wrapper so eval doesn't drag along DDP collectives or trigger
    unnecessary recompilation on shape/mode changes.
    """
    if isinstance(net, FSDP):
        return net
    return _unwrap(net)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_warmup_lr_lambda(step, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Cosine annealing with linear warmup. Returns the LR multiplier for step."""
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(run_dir: str):
    """
    Scan run_dir for checkpoint-{N} subdirectories and return the latest one.
    Returns (checkpoint_dir, iteration) or (None, 0) if none exist.
    """
    dirs = [
        d for d in glob.glob(os.path.join(run_dir, "checkpoint-*"))
        if os.path.isdir(d) and re.search(r"checkpoint-(\d+)$", d)
    ]
    if not dirs:
        return None, 0
    latest = max(dirs, key=lambda d: int(re.search(r"checkpoint-(\d+)$", d).group(1)))
    return latest, int(re.search(r"checkpoint-(\d+)$", latest).group(1))


def load_checkpoint(checkpoint_dir, net, optim, scheduler, parallel_mode="ddp"):
    """
    Restore model weights from safetensors and optimizer/scheduler from .pt files.
    Returns the trainer_state dict (used to recover cumulative_tokens etc.), or {}.
    """
    weights_path = os.path.join(checkpoint_dir, "model.safetensors")
    if isinstance(net, FSDP):
        state_dict = load_file(weights_path)
        with FSDP.state_dict_type(net, StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(rank0_only=False)):
            net.load_state_dict(state_dict)
    else:
        # _unwrap strips DDP and torch.compile so the saved checkpoint (which
        # has bare-module keys) loads cleanly regardless of wrappers.
        _unwrap(net).load_state_dict(load_file(weights_path))

    opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
    print(f"Attempting to load {opt_path}")
    if os.path.exists(opt_path):
        opt_state = torch.load(opt_path, map_location="cpu")
        if isinstance(net, FSDP):
            opt_state = FSDP.optim_state_dict_to_load(net, optim, opt_state)
        optim.load_state_dict(opt_state)

    sched_path = os.path.join(checkpoint_dir, "scheduler.pt")
    print(f"Attempting to load {sched_path}")
    if os.path.exists(sched_path):
        scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

    state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def training_step(net, batch, labels, fwd_events=None):
    """Forward pass + CE loss. Returns (logits, loss, metadata).

    If `fwd_events` is provided, it is a (start, end) pair of `torch.cuda.Event`s
    that are recorded around the forward pass so callers can measure GPU
    forward time later via `start.elapsed_time(end)`. Recording events is
    nearly free; reading their elapsed time forces a device sync, so do that
    only on log iterations.
    """
    if fwd_events is not None:
        fwd_events[0].record()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = net(batch)
    if fwd_events is not None:
        fwd_events[1].record()
    logits = output.logits if hasattr(output, "logits") else output
    metadata = output.metadata if hasattr(output, "metadata") else {}
    B, T, V = logits.shape
    # CE on bf16 logits; the kernel reduces in fp32 internally without
    # materializing a full fp32 [B,T,V] tensor (was ~10 GB at V=151936, T=4096, B=4).
    loss = F.cross_entropy(logits.view(B * T, V), labels.view(B * T))
    return logits, loss, metadata


def run_lm_eval(net, tokenizer, eval_conf, device):
    """
    Run lm-eval tasks defined in eval_conf against net and return aggregated results.

    eval_conf keys:
      tasks                - list of lm-eval task names (required)
      num_fewshot          - number of few-shot examples (default 0)
      limit                - cap examples per task, useful for quick checks (default None)
      batch_size           - passed to HFLM (default "auto")
      gpu_memory_fraction  - fraction of GPU VRAM to allow (0.0–1.0, default None = unlimited)
      num_runs             - number of times to repeat each eval; all runs are saved and
                             min/max are reported (default 1)

    Returns a dict keyed by task name. Each metric value is a dict:
      { "runs": [v0, v1, ...], "min": float, "max": float }
    When num_runs=1 the runs list has one element and min==max.

    Requires tokenizer to be a HuggingFace PreTrainedTokenizer.
    """
    mem_frac = eval_conf.get("gpu_memory_fraction")
    if mem_frac is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(mem_frac))

    num_runs = int(eval_conf.get("num_runs", 1))
    max_len = getattr(net.config, "max_position_embeddings", 1024)
    tasks = eval_conf["tasks"]
    num_fewshot = eval_conf.get("num_fewshot", 0)
    limit = eval_conf.get("limit", None)
    print(f"  lm-eval tasks={tasks} num_fewshot={num_fewshot} limit={limit} "
          f"batch_size={eval_conf.get('batch_size', 'auto')} max_length={max_len}")

    import logging
    logging.getLogger("lm_eval").setLevel(logging.INFO)

    all_runs = []
    for run_idx in range(num_runs):
        print(f"  lm-eval run {run_idx + 1}/{num_runs} starting...", flush=True)
        t_eval = time.time()
        lm = HFLM(
            pretrained=net,
            tokenizer=tokenizer,
            batch_size=eval_conf.get("batch_size", "auto"),
            max_length=max_len,
        )
        raw = simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            verbosity="INFO",
        )
        print(f"  lm-eval run {run_idx + 1}/{num_runs} done ({time.time() - t_eval:.1f}s)", flush=True)
        all_runs.append(raw.get("results", {}))

    aggregated = {}
    for task, metrics in all_runs[0].items():
        aggregated[task] = {}
        for metric, val in metrics.items():
            if isinstance(val, (int, float)):
                vals = [r[task][metric] for r in all_runs if isinstance(r[task].get(metric), (int, float))]
                aggregated[task][metric] = {"runs": vals, "min": min(vals), "max": max(vals)}
            else:
                aggregated[task][metric] = {"runs": [r[task].get(metric) for r in all_runs]}

    return aggregated


def run_validation(net, tokenizer, val_dataloader, device, run_dir, max_steps=10, step=0, local_rank=0):
    """
    Run validation for up to max_steps batches on every rank in parallel, then
    all-reduce the loss so all ranks share the same avg.  Only rank 0 generates
    sample continuations and writes val_iter_{step}.json.

    Returns (avg_val_loss, avg_val_perplexity, eval_loop_time_s, generation_time_s).
    `eval_loop_time_s` excludes the rank-0 generation (only the val batch loop +
    all-reduce). `generation_time_s` is rank-0 generate() wall clock; 0.0 on
    other ranks.
    """
    net.eval()
    val_loss_total = 0.0
    n_batches = 0
    val_iterator = iter(val_dataloader)

    _eval_t0 = time.time()
    with torch.no_grad():
        for _ in tqdm(range(max_steps), desc="Validation", disable=local_rank != 0):
            try:
                batch, labels, _ = next(val_iterator)
            except StopIteration:
                break
            batch, labels = batch.to(device), labels.to(device)
            logits, loss, metadata = training_step(net, batch=batch, labels=labels)
            val_loss_total += loss.item()
            n_batches += 1

    # All-reduce so every rank gets the same aggregate loss.
    if dist.is_initialized():
        t = torch.tensor([val_loss_total, float(n_batches)], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        val_loss_total, n_batches = t[0].item(), t[1].item()

    eval_loop_time_s = time.time() - _eval_t0
    avg_loss = val_loss_total / max(1, n_batches)

    generation_time_s = 0.0
    if local_rank == 0 and tokenizer is not None:
        generation_dir = os.path.join(run_dir, "generations")
        os.makedirs(generation_dir, exist_ok=True)
        generations = []
        _gen_t0 = time.time()
        for gen_idx in range(3):
            try:
                val_batch, _, _ = next(val_iterator)
            except StopIteration:
                break
            prompt_tokens = val_batch[0, : val_batch.shape[1] // 2].tolist()
            prompt_text = tokenizer.decode(prompt_tokens)
            output_text, _ = generate(
                prompt_text, tokenizer, net, device,
                max_output_length=50, generation_folder="",
                checkpoint_path="", tokenizer_path="",
            )
            generations.append({
                "iter": step, "generation_idx": gen_idx,
                "prompt": prompt_text, "output": output_text,
            })
        generation_time_s = time.time() - _gen_t0
        with open(os.path.join(generation_dir, f"val_iter_{step}.json"), "w") as f:
            json.dump(generations, f, indent=2)

    return avg_loss, math.exp(avg_loss), eval_loop_time_s, generation_time_s


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def pretrain(
    net,
    tokenizer,
    train_dataloader,
    val_dataloader,
    optim,
    scheduler,
    device,
    num_iters=10,
    num_epochs=1,
    num_val_iters=10,
    checkpoint_frequency=5,
    experiment_name=None,
    model_name=None,
    run_dir=None,
    profile_start=-1,
    profile_end=-1,
    capture_memory_snapshot=False,
    memory_snapshot_events=100_000,
    start_iter=0,
    cumulative_tokens_start=0,
    source_tokens_start=None,
    eval_config=None,
    num_visualize_generations=0,
    loss_viz_config=None,
    spectral_viz=False,
    local_rank=0,
    resumed=False,
    parallel_mode="ddp",
    grad_accum_steps=1,
    enable_viz=False,
    log_freq=50,
    forward_flops_per_microbatch=0,
    peak_bf16_tflops_per_gpu=0.0,
):
    """
    Main pretraining loop.

    Each checkpoint is saved as {run_dir}/checkpoint-{iter}/ and contains:
      model.safetensors + config.json  — model weights and architecture config
      <tokenizer files>                — saved via tokenizer.save_pretrained (HF tokenizers only)
      optimizer.pt / scheduler.pt      — optimizer and scheduler state for resumption
      activations.json                 — activation memory statistics
      eval_results.json                — lm-eval results (only if --eval_config is provided)
      trainer_state.json               — copy of the run-level training log
      loss_viz/                        — loss landscape plots (only if --loss-viz is set)

    trainer_state.json is also written at the run root after every checkpoint. It follows the
    same log_history schema as HuggingFace Trainer so tooling that reads HF checkpoints works.
    """
    net = net.to(device)
    net.train()
    data_iterator = iter(train_dataloader)
    current_epoch = 0

    _profiler_active = False
    profiler = None
    _memory_snapshot_active = False
    _memory_snapshot_path = None
    _profile_enabled = profile_start >= 0 and profile_end > profile_start
    _num_profile_steps = (profile_end - profile_start + 1) if _profile_enabled else 0
    _capture_memory_snapshot = (
        _profile_enabled and capture_memory_snapshot and torch.cuda.is_available()
    )
    cuda_profile_dir = memory_profile_dir = None
    if _profile_enabled:
        cuda_profile_dir = os.path.join(run_dir, "profiles", "cuda")
        os.makedirs(cuda_profile_dir, exist_ok=True)
        if _capture_memory_snapshot:
            memory_profile_dir = os.path.join(run_dir, "profiles", "memory")
            os.makedirs(memory_profile_dir, exist_ok=True)
        _mem_msg = (f" + memory snapshot (max_entries={memory_snapshot_events}) → {memory_profile_dir}"
                    if _capture_memory_snapshot else "")
        print(f"Profiler armed: will capture iters [{profile_start}, {profile_end}] "
              f"({_num_profile_steps} steps) → {cuda_profile_dir}{_mem_msg}")

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    activation_stats: dict = defaultdict(list)
    # Activation hooks add Python overhead per submodule per forward and inhibit
    # kernel fusion. Only register them when viz is explicitly requested.
    hooks = register_activation_hooks(net, activation_stats) if enable_viz else []
    cumulative_tokens = cumulative_tokens_start
    log_history = []

    _source_names: list[str] = getattr(train_dataloader.dataset, "_names", [])
    _per_source_tokens: dict[str, int] = {
        name: (source_tokens_start or {}).get(name, 0)
        for name in _source_names
    }

    print(f"Training started: iters {start_iter} → {num_iters}, device={device}, "
          f"grad_accum={grad_accum_steps}" + (" (resumed)" if resumed else ""), flush=True)
    # One-time computations: param memory is fixed for the run; optimizer state
    # memory is fixed once the first step has populated it. We log both lazily.
    _param_mem_mb = (
        sum(p.numel() * p.element_size() for p in net.parameters()) / (1024 ** 2)
        if torch.cuda.is_available() else 0.0
    )
    _opt_mem_mb_cached = 0.0

    for i in range(start_iter, num_iters):
        _is_log_iter = (local_rank == 0) and (i % log_freq == 0 or i == num_iters - 1)
        if torch.cuda.is_available() and _is_log_iter:
            torch.cuda.reset_peak_memory_stats(device)

        # ── Gradient accumulation loop ────────────────────────────────────
        # Each optimizer step consists of grad_accum_steps micro-batches.
        # DDP/FSDP gradient all-reduces are suppressed until the final step.
        # Loss is accumulated as a device tensor — no per-microstep .item() sync.
        _accum_loss_dev = torch.zeros((), device=device, dtype=torch.float32)
        _accum_metadata: dict = {}
        _batch_load_time = 0.0
        _step_source_tokens: dict[str, int] = {}
        _step_tokens = 0
        _last_seq_len = 0
        _data_exhausted = False
        _step_fwd_event_pairs: list[tuple] = []
        _step_bwd_event_pairs: list[tuple] = []

        t_compute = time.time()
        for _accum_step in range(grad_accum_steps):
            t_batch = time.time()
            try:
                batch, labels, source_idx = next(data_iterator)
            except StopIteration:
                current_epoch += 1
                if current_epoch >= num_epochs:
                    if local_rank == 0:
                        print(f"Data exhausted after {current_epoch} epoch(s) at step {i}.")
                    _data_exhausted = True
                    break
                if local_rank == 0:
                    print(f"Starting epoch {current_epoch + 1}/{num_epochs} at step {i}.")
                data_iterator = iter(train_dataloader)
                batch, labels, source_idx = next(data_iterator)
            _batch_load_time += time.time() - t_batch

            _last_seq_len = labels.shape[1]
            _step_tokens += labels.numel()

            # source_idx is a CPU tensor from the dataloader; bincount avoids a
            # python loop with .item() calls per source.
            if _source_names:
                counts = torch.bincount(source_idx.view(-1), minlength=len(_source_names)).tolist()
                for j, name in enumerate(_source_names):
                    tokens = counts[j] * _last_seq_len
                    _per_source_tokens[name] = _per_source_tokens.get(name, 0) + tokens
                    _step_source_tokens[name] = _step_source_tokens.get(name, 0) + tokens

            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Defer gradient sync to the last micro-step (DDP / FSDP).
            _is_last_accum = (_accum_step == grad_accum_steps - 1)
            _sync_ctx = (contextlib.nullcontext()
                         if _is_last_accum
                         else net.no_sync())

            # Per-microstep CUDA events around forward and backward — read
            # elapsed time on log iters so we don't sync every step. The
            # backward window includes DDP/FSDP gradient all-reduces on the
            # final accum step (no_sync suppresses comm on earlier ones).
            if torch.cuda.is_available() and _is_log_iter:
                _fwd_events = (torch.cuda.Event(enable_timing=True),
                               torch.cuda.Event(enable_timing=True))
                _bwd_events = (torch.cuda.Event(enable_timing=True),
                               torch.cuda.Event(enable_timing=True))
            else:
                _fwd_events = None
                _bwd_events = None

            with _sync_ctx:
                _, loss_step, metadata_step = training_step(
                    net, batch=batch, labels=labels, fwd_events=_fwd_events,
                )
                if _bwd_events is not None:
                    _bwd_events[0].record()
                (loss_step / grad_accum_steps).backward()
                if _bwd_events is not None:
                    _bwd_events[1].record()

            if _fwd_events is not None:
                _step_fwd_event_pairs.append(_fwd_events)
            if _bwd_events is not None:
                _step_bwd_event_pairs.append(_bwd_events)

            _accum_loss_dev = _accum_loss_dev + loss_step.detach().float()
            if not _accum_metadata:
                _accum_metadata = metadata_step

        if _data_exhausted:
            break

        compute_time = time.time() - t_compute

        t2 = time.time()
        if isinstance(net, FSDP):
            net.clip_grad_norm_(max_norm=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()
        # Free gradient tensors immediately — they've been consumed by the optimizer.
        # This reclaims ~2× param memory before the checkpoint/validation block.
        optim.zero_grad(set_to_none=True)
        optimizer_time = time.time() - t2

        iter_time = compute_time + optimizer_time
        cumulative_tokens += _step_tokens
        current_lr = scheduler.get_last_lr()[0]

        # Single sync per step (not per microstep) for loss readout.
        loss_val = (_accum_loss_dev / grad_accum_steps).item()
        train_ppl = math.exp(loss_val)

        if _is_log_iter:
            if torch.cuda.is_available():
                # Optimizer state dict walk is O(num_params) Python work; cache it.
                if _opt_mem_mb_cached == 0.0:
                    _opt_mem_mb_cached = sum(
                        v.numel() * v.element_size()
                        for state in optim.state.values()
                        if state is not None
                        for v in state.values()
                        if torch.is_tensor(v)
                    ) / (1024 ** 2)
                allocated_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
                peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                allocated_mem = peak_mem = 0.0

            total_activation_mem = sum(activation_stats.values()) if enable_viz else 0.0

            # Sum GPU forward + backward time across the microbatches in this
            # step. The first elapsed_time() call forces a device sync; that's
            # fine because we only do it on log iters.
            forward_time_s = 0.0
            backward_time_s = 0.0
            if _step_fwd_event_pairs:
                # Sync once on the earliest event — all subsequent reads are
                # then guaranteed to have completed.
                _step_fwd_event_pairs[0][0].synchronize()
                forward_time_s = sum(
                    s.elapsed_time(e) for s, e in _step_fwd_event_pairs
                ) / 1000.0
            if _step_bwd_event_pairs:
                backward_time_s = sum(
                    s.elapsed_time(e) for s, e in _step_bwd_event_pairs
                ) / 1000.0

            # Residual time not captured by forward/backward/optimizer/data.
            # Includes Python overhead, grad clip, lr scheduler, accum_loss
            # bookkeeping, and any unattributed CUDA work.
            other_time_s = max(
                0.0,
                iter_time - forward_time_s - backward_time_s
                - optimizer_time - _batch_load_time,
            )

            forward_tflops_per_sec = 0.0
            forward_mfu_frac = 0.0
            if forward_flops_per_microbatch > 0 and forward_time_s > 0:
                # Per-rank achieved FLOPs across all microbatches in this step.
                step_fwd_flops = forward_flops_per_microbatch * grad_accum_steps
                forward_tflops_per_sec = step_fwd_flops / forward_time_s / 1e12
                if peak_bf16_tflops_per_gpu > 0:
                    forward_mfu_frac = forward_tflops_per_sec / peak_bf16_tflops_per_gpu

            wandb_log = {
                # Top-level: training quality + scheduler.
                "lm_loss": loss_val,
                "train_perplexity": train_ppl,
                "learning_rate": current_lr,

                # data/* — what the model has seen, GLOBALLY (across all
                # ranks). The internal `cumulative_tokens` variable is
                # per-rank; we multiply by world_size on the way out so the
                # number in wandb is the total tokens trained on, which is
                # what people actually want to read.
                "data/cumulative_tokens": cumulative_tokens * world_size,
                "data/cumulative_tokens_per_rank": cumulative_tokens,
                "data/tokens_this_step": _step_tokens * world_size,
                "data/sequence_length": _last_seq_len,

                # timing/* — wall-clock and throughput. The first five sum to
                # iter_time by construction (other_time is the residual).
                "timing/iter_time": iter_time,
                "timing/forward_time": forward_time_s,
                "timing/backward_time": backward_time_s,
                "timing/optimizer_time": optimizer_time,
                "timing/batch_load_time": _batch_load_time,
                "timing/other_time": other_time_s,
                "timing/tokens_per_sec": _step_tokens / max(iter_time, 1e-9),

                # compute/* — FLOPs and MFU.
                "compute/forward_flops_per_step": (
                    forward_flops_per_microbatch * grad_accum_steps
                ),
                "compute/forward_tflops_per_sec": forward_tflops_per_sec,
                "compute/forward_mfu": forward_mfu_frac,
                "compute/peak_bf16_tflops_per_gpu": peak_bf16_tflops_per_gpu,

                # vram/* — memory footprint.
                "vram/params_MB": _param_mem_mb,
                "vram/optimizer_state_MB": _opt_mem_mb_cached,
                "vram/activations_MB": total_activation_mem,
                "vram/allocated_MB": allocated_mem,
                "vram/peak_allocated_MB": peak_mem,
            }

            for k, v in _accum_metadata.items():
                if k.startswith("metrics/model/"):
                    wandb_log["model/" + k[len("metrics/model/"):]] = v
            for name in _source_names:
                # Per-source values are also globalized via world_size. They
                # have small variance vs a true all-reduce because each rank
                # samples sources independently, but the categorical
                # distribution averages out over time.
                wandb_log[f"data/{name}/tokens"] = _step_source_tokens.get(name, 0) * world_size
                wandb_log[f"data/{name}/cumulative_tokens"] = _per_source_tokens.get(name, 0) * world_size
            wandb.log(wandb_log, step=i)

        if _profile_enabled and i == profile_start:
            _trace_name = (
                f"epoch{current_epoch}_step{i}_numsteps{_num_profile_steps}_rank{local_rank}"
            )
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    cuda_profile_dir, worker_name=_trace_name
                ),
                record_shapes=True, profile_memory=True, with_stack=True,
            )
            profiler.start()
            _profiler_active = True
            print(f"[iter {i}] Profiler started "
                  f"(epoch={current_epoch}, step={i}, num_steps={_num_profile_steps})")
            if _capture_memory_snapshot:
                torch.cuda.memory._record_memory_history(max_entries=memory_snapshot_events)
                _memory_snapshot_active = True
                _memory_snapshot_path = os.path.join(memory_profile_dir, f"{_trace_name}.pickle")
                print(f"[iter {i}] CUDA memory recording started "
                      f"(max_entries={memory_snapshot_events})")
        if _profiler_active:
            profiler.step()
        if _profiler_active and i == profile_end:
            profiler.stop()
            _profiler_active = False
            profiler = None
            print(f"[iter {i}] Profiler stopped → {cuda_profile_dir}")
            if _memory_snapshot_active:
                torch.cuda.memory._dump_snapshot(_memory_snapshot_path)
                torch.cuda.memory._record_memory_history(enabled=None)
                _memory_snapshot_active = False
                print(f"[iter {i}] CUDA memory snapshot dumped → {_memory_snapshot_path}")

        do_checkpoint = (i % checkpoint_frequency == 0 or i == num_iters - 1)
        skip_viz = resumed and i == start_iter
        checkpoint_dir = os.path.join(run_dir, f"checkpoint-{i}")

        # All-rank operations that must precede rank-0 I/O.
        _fsdp_model_state = _fsdp_optim_state = None
        if do_checkpoint and isinstance(net, FSDP):
            with FSDP.state_dict_type(net, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                _fsdp_model_state = net.state_dict()
            _fsdp_optim_state = FSDP.optim_state_dict(net, optim)
        if do_checkpoint and isinstance(optim, ZeroRedundancyOptimizer):
            optim.consolidate_state_dict(to=0)

        # Per-phase wall clocks, populated below. Emitted to wandb under
        # timing/checkpoint/* once all phases complete.
        _phase_times: dict[str, float] = {}

        # Rank-0-only I/O: save weights, tokenizer, scheduler, activations.
        if local_rank == 0 and do_checkpoint:
            _ckpt_t0 = time.time()
            print(f"[iter {i}] checkpoint start", flush=True)
            os.makedirs(checkpoint_dir, exist_ok=True)

            _t = time.time()
            print(f"[iter {i}]   saving weights...", flush=True)
            if isinstance(net, FSDP):
                save_file(_fsdp_model_state, os.path.join(checkpoint_dir, "model.safetensors"))
                _unwrap(net).config.save_pretrained(checkpoint_dir)
                torch.save(_fsdp_optim_state, os.path.join(checkpoint_dir, "optimizer.pt"))
            else:
                # Save the bare module (no DDP / no compile prefix) so checkpoints
                # remain portable across compile on/off and across parallel modes.
                _unwrap(net).save_pretrained(checkpoint_dir, safe_serialization=True)
                torch.save(optim.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

            if tokenizer is not None:
                tokenizer.save_pretrained(checkpoint_dir)

            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

            with open(os.path.join(checkpoint_dir, "activations.json"), "w") as f:
                json.dump(dict(activation_stats), f)
            _phase_times["save"] = time.time() - _t
            print(f"[iter {i}]   saving weights done ({_phase_times['save']:.1f}s)", flush=True)

        # Sync all ranks after rank-0 I/O.
        if do_checkpoint and dist.is_initialized():
            dist.barrier()

        # All-rank: validation (each rank runs its own batches; loss is all-reduced).
        # All-rank: visualizations that can be parallelised across ranks.
        if do_checkpoint:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            eval_net = _eval_module(net)

            _t = time.time()
            if local_rank == 0:
                print(f"[iter {i}]   validation start...", flush=True)
            avg_val_loss, avg_val_ppl, _val_loop_s, _val_gen_s = run_validation(
                eval_net, tokenizer, val_dataloader, device,
                run_dir=run_dir, max_steps=num_val_iters, step=i,
                local_rank=local_rank,
            )
            net.train()
            _phase_times["validation_total"] = time.time() - _t
            _phase_times["validation_eval_loop"] = _val_loop_s
            _phase_times["validation_generation"] = _val_gen_s
            if local_rank == 0:
                print(f"[iter {i}]   validation done ({_phase_times['validation_total']:.1f}s "
                      f"= eval {_val_loop_s:.1f}s + gen {_val_gen_s:.1f}s)", flush=True)
                wandb.log({"val_loss_avg": avg_val_loss, "val_perplexity_avg": avg_val_ppl}, step=i)

            if not skip_viz:
                if local_rank == 0:
                    _t = time.time()
                    print(f"[iter {i}]   plot_visualizations start...", flush=True)
                    plot_visualizations(eval_net, os.path.join(run_dir, "visualizations"), i)
                    _phase_times["plot_visualizations"] = time.time() - _t
                    print(f"[iter {i}]   plot_visualizations done "
                          f"({_phase_times['plot_visualizations']:.1f}s)", flush=True)

                # Sync after any rank-0-only work above before entering the
                # distributed spectral-viz collective (all_gather_object).
                if spectral_viz and dist.is_initialized():
                    dist.barrier()

                if spectral_viz:
                    _t = time.time()
                    if local_rank == 0:
                        print(f"[iter {i}]   spectral viz start...", flush=True)
                    from hyperviz.spectral_visualizer import SpectralVisualizer
                    spectral_viz_dir = os.path.join(checkpoint_dir, "viz")
                    sv = SpectralVisualizer(save_directory=spectral_viz_dir)
                    eval_net.eval()
                    with torch.no_grad():
                        sv.visualize(eval_net, rank=local_rank, world_size=world_size)
                    net.train()
                    _phase_times["spectral_viz"] = time.time() - _t
                    if local_rank == 0:
                        print(f"[iter {i}]   spectral viz done ({_phase_times['spectral_viz']:.1f}s) "
                              f"→ {spectral_viz_dir}/spectral_values/", flush=True)

            if local_rank == 0:
                log_history.append({
                    "step": i,
                    "epoch": round(i / num_iters, 4),
                    "loss": round(loss_val, 4),
                    "perplexity": round(train_ppl, 4),
                    "val_loss": round(avg_val_loss, 4),
                    "val_perplexity": round(avg_val_ppl, 4),
                    "learning_rate": current_lr,
                    "cumulative_tokens": cumulative_tokens,
                    "source_tokens": dict(_per_source_tokens),
                    "iter_time": round(iter_time, 4),
                })
                trainer_state = {
                    "global_step": i,
                    "epoch": round(i / num_iters, 4),
                    "max_steps": num_iters,
                    "model_name": model_name,
                    "log_history": log_history,
                }
                for dest in [run_dir, checkpoint_dir]:
                    with open(os.path.join(dest, "trainer_state.json"), "w") as f:
                        json.dump(trainer_state, f, indent=2)

                print(
                    f"[iter {i}] checkpoint done ({time.time()-_ckpt_t0:.1f}s) "
                    f"loss={loss_val:.4f} ppl={train_ppl:.2f} "
                    f"val_loss={avg_val_loss:.4f} val_ppl={avg_val_ppl:.2f} "
                    f"lr={current_lr:.2e} tokens={cumulative_tokens * world_size:,} "
                    f"(per-rank {cumulative_tokens:,}) "
                    f"→ {checkpoint_dir}",
                    flush=True,
                )

        # ---------------------------------------------------------------------------
        # Distributed hyperviz — each rank generates from its own val prompt,
        # all hidden-state trajectories are gathered to rank 0 for visualization.
        # ---------------------------------------------------------------------------
        if do_checkpoint and not skip_viz and num_visualize_generations > 0 and tokenizer is not None:
            _t = time.time()
            if local_rank == 0:
                print(f"[iter {i}]   hyperviz start ({num_visualize_generations} tokens, "
                      f"{dist.get_world_size() if dist.is_initialized() else 1} rank(s))...", flush=True)
            _eval_net_viz = _eval_module(net)
            viz_batch, _, _ = next(iter(val_dataloader))
            viz_prompt_tokens = viz_batch[0, :viz_batch.shape[1] // 2].tolist()
            viz_prompt = tokenizer.decode(viz_prompt_tokens)
            _eval_net_viz.eval()
            _, hidden_states_per_step = generate(
                viz_prompt, tokenizer, _eval_net_viz, device,
                max_output_length=num_visualize_generations,
                generation_folder="", checkpoint_path="", tokenizer_path="",
                collect_hidden_states=True,
            )
            net.train()

            if dist.is_initialized() and dist.get_world_size() > 1:
                _all_steps = [None] * dist.get_world_size()
                dist.all_gather_object(_all_steps, hidden_states_per_step or [])
                combined_steps = [hs for rank_steps in _all_steps for hs in rank_steps]
            else:
                combined_steps = hidden_states_per_step or []

            if local_rank == 0 and combined_steps:
                from hyperviz import Visualizer
                from hyperviz.trajectory import Trajectory
                viz_dir = os.path.join(checkpoint_dir, "viz")
                visualizer = Visualizer(viz_dir)
                for step_hs in combined_steps:
                    if step_hs is not None:
                        visualizer.add(Trajectory(hidden_states=step_hs))
                visualizer.visualize()
                visualizer.clear()
                _phase_times["hyperviz"] = time.time() - _t
                print(f"[iter {i}]   hyperviz done ({_phase_times['hyperviz']:.1f}s) → {viz_dir}", flush=True)

        # ---------------------------------------------------------------------------
        # Distributed loss landscape — grid cells are partitioned across ranks;
        # rank 0 plots and saves.
        # ---------------------------------------------------------------------------
        if do_checkpoint and not skip_viz and loss_viz_config is not None:
            _t = time.time()
            if local_rank == 0:
                print(f"[iter {i}]   loss landscape start...", flush=True)
            from hyperviz.loss_visualizer import LossVisualizer

            class _LMCriterion(nn.Module):
                """Unwraps LMOutput and reshapes (B,T,V) logits for cross_entropy."""
                def forward(self, output, targets):
                    logits = output.logits if hasattr(output, "logits") else output
                    if logits.dim() == 3:
                        B, T, V = logits.shape
                        logits = logits.view(B * T, V)
                        targets = targets.view(B * T)
                    return F.cross_entropy(logits, targets)

            loss_viz_dir = os.path.join(checkpoint_dir, "viz")
            loss_visualizer = LossVisualizer(
                save_directory=loss_viz_dir,
                criterion=_LMCriterion(),
                grid_points=loss_viz_config.get("grid_points", 20),
                grid_range=loss_viz_config.get("grid_range", 1.0),
                eval_batches=loss_viz_config.get("eval_batches", 50),
                save_interactive_visualization=loss_viz_config.get("interactive", False),
            )
            activation_stats.clear()
            _eval_net_loss = _eval_module(net)
            _eval_net_loss.eval()
            loss_visualizer.visualize(_eval_net_loss, val_dataloader, device)
            del loss_visualizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            net.train()
            _phase_times["loss_viz"] = time.time() - _t
            if local_rank == 0:
                print(f"[iter {i}]   loss landscape done ({_phase_times['loss_viz']:.1f}s) → {loss_viz_dir}", flush=True)

        # lm-eval runs on ALL ranks so the work is distributed across GPUs.
        # Only rank 0 saves results and logs to wandb.
        if do_checkpoint and eval_config is not None and tokenizer is not None:
            _t = time.time()
            _eval_net = _eval_module(net)
            if local_rank == 0:
                _num_runs = int(eval_config.get("num_runs", 1))
                print(f"[iter {i}] Starting lm-eval on all ranks ({_num_runs} run(s)): {eval_config['tasks']}", flush=True)
            _eval_net.eval()
            eval_results = run_lm_eval(_eval_net, tokenizer, eval_config, device)
            net.train()
            _phase_times["lm_eval"] = time.time() - _t
            if local_rank == 0:
                _ckpt_dir = os.path.join(run_dir, f"checkpoint-{i}")
                _num_runs = int(eval_config.get("num_runs", 1))
                eval_out = {
                    "iter": i,
                    "tasks": eval_config["tasks"],
                    "num_fewshot": eval_config.get("num_fewshot", 0),
                    "num_runs": _num_runs,
                    "results": eval_results,
                }
                with open(os.path.join(_ckpt_dir, "eval_results.json"), "w") as f:
                    json.dump(eval_out, f, indent=2)
                wandb_metrics = {}
                for task, res in eval_results.items():
                    for metric, agg in res.items():
                        if not isinstance(agg, dict) or "min" not in agg:
                            continue
                        wandb_metrics[f"eval/{task}/{metric}/min"] = agg["min"]
                        wandb_metrics[f"eval/{task}/{metric}/max"] = agg["max"]
                        for run_idx, v in enumerate(agg["runs"]):
                            if isinstance(v, (int, float)):
                                wandb_metrics[f"eval/{task}/{metric}/run_{run_idx}"] = v
                wandb.log(wandb_metrics, step=i)
                print(f"[iter {i}] lm-eval saved → {_ckpt_dir}/eval_results.json")

        # Emit per-phase wall-clocks to wandb in a single log call. These are
        # rank-0-only because most timing is captured rank-0-only above; the
        # phases that run on all ranks (lm_eval, hyperviz, spectral_viz) take
        # a global time so any rank's value is representative.
        if do_checkpoint and local_rank == 0 and _phase_times:
            # validation_eval_loop + validation_generation are sub-buckets of
            # validation_total — exclude them from the total to avoid double
            # counting.
            _subkeys = {"validation_eval_loop", "validation_generation"}
            _total = sum(v for k, v in _phase_times.items() if k not in _subkeys)
            wandb.log(
                {f"timing/checkpoint/{k}": v for k, v in _phase_times.items()}
                | {"timing/checkpoint/total": _total},
                step=i,
            )

        # Wait for rank 0 to finish lm-eval logging before the next backward.
        if do_checkpoint and dist.is_initialized():
            dist.barrier()

        activation_stats.clear()

    for h in hooks:
        h.remove()
    if _profiler_active and profiler is not None:
        profiler.stop()

def setup():
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)

def cleanup():
    dist.destroy_process_group()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pretrain an arcadium language model.")

    parser.add_argument("--model_config", type=str, default="configs/models/tiny-moe-64-emb.yaml",
                        help="Path to model architecture YAML")
    parser.add_argument("--training_config", type=str, default="configs/training/basic.yaml",
                        help="Path to training hyperparameter YAML")
    parser.add_argument("--eval_config", type=str, default=None,
                        help="(Optional) Path to lm-eval config YAML. When provided, lm-eval tasks "
                             "run at every checkpoint and results are saved to eval_results.json. "
                             "Expected keys: tasks (list), num_fewshot (int), limit (int|null), "
                             "batch_size (int|'auto').")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="(Deprecated) Legacy path — tokenizer is now specified in model config")
    parser.add_argument("--profile-start", type=int, default=-1,
                        help="Iteration to start torch.profiler CUDA profiling (-1 = disabled)")
    parser.add_argument("--profile-end", type=int, default=-1,
                        help="Iteration to stop torch.profiler CUDA profiling (inclusive, must be > --profile-start)")
    parser.add_argument("--profile-relative", action="store_true",
                        help="Treat --profile-start/--profile-end as offsets from the resume iteration "
                             "(e.g. --profile-start 10 --profile-end 50 --profile-relative profiles "
                             "iters resume+10 through resume+50)")
    parser.add_argument("--capture-memory-snapshot", action="store_true",
                        help="Also dump a torch.cuda memory snapshot pickle for the same iteration "
                             "window as --profile-start/--profile-end. Saved to "
                             "{run_dir}/profiles/memory/.")
    parser.add_argument("--memory-snapshot-events", type=int, default=100_000,
                        help="Maximum number of allocator events captured per CUDA memory snapshot. "
                             "Default: 100000.")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to an existing run directory to resume training from. "
                             "The latest checkpoint-{N} inside it will be loaded automatically.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint subdirectory to load (e.g. checkpoint-800). "
                             "Must be used with --load. Overrides auto-detection of the latest checkpoint.")
    parser.add_argument("--base-run-dir", type=str, default="checkpoints",
                        help="Base directory under which new run folders are created. "
                             "Default: checkpoints/")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose dataset output")
    parser.add_argument("--num-visualize-generations", type=int, default=0,
                        help="Number of tokens to generate per checkpoint for hyperviz analysis. "
                             "0 = disabled. When set, hidden states are collected during generation "
                             "and saved to {checkpoint}/viz/.")
    parser.add_argument("--loss-viz", action="store_true",
                        help="Enable loss landscape visualization at every checkpoint. "
                             "Saves 3D surface, 2D contour, and 1D slice plots to "
                             "{checkpoint}/loss_viz/ using filter-normalized random directions.")
    parser.add_argument("--loss-viz-grid-points", type=int, default=20,
                        help="Grid resolution for loss landscape sweep (N×N). Default: 20.")
    parser.add_argument("--loss-viz-grid-range", type=float, default=1.0,
                        help="α and β are swept over [-range, +range]. Default: 1.0.")
    parser.add_argument("--loss-viz-eval-batches", type=int, default=50,
                        help="Validation batches used to estimate loss at each grid point. Default: 50.")
    parser.add_argument("--loss-viz-interactive", action="store_true",
                        help="Also save an interactive 3D HTML file (requires plotly).")
    parser.add_argument("--val-batch-size", type=int, default=None,
                        help="Batch size for validation. Defaults to the training batch size.")
    parser.add_argument("--spectral-viz", action="store_true",
                        help="Compute and save singular-value distributions for all 2-D weight "
                             "matrices at every checkpoint. Saves plots to "
                             "{checkpoint}/viz/spectral_values/.")
    parser.add_argument("--enable-viz", action="store_true",
                        help="Master switch for always-on visualization instrumentation: "
                             "(1) per-submodule activation-memory forward hooks and "
                             "(2) wandb.watch gradient logging. Both add measurable overhead "
                             "and are off by default. Per-checkpoint viz flags "
                             "(--num-visualize-generations, --loss-viz, --spectral-viz) are "
                             "independent of this and remain opt-in via their own flags.")
    parser.add_argument("--log-freq", type=int, default=50,
                        help="How often (in steps) to write per-iteration metrics to wandb "
                             "and walk CUDA memory stats. Smaller = more wandb rows + more "
                             "Python overhead per iter. Default: 50.")

    compile_group = parser.add_argument_group("compile")
    compile_group.add_argument("--compile", dest="compile", action="store_true", default=True,
                               help="torch.compile() the model before DDP/FSDP wrap. "
                                    "On by default — significant speedup at >100M params on H100. "
                                    "Use --no-compile to disable.")
    compile_group.add_argument("--no-compile", dest="compile", action="store_false",
                               help="Disable torch.compile.")
    compile_group.add_argument("--compile-mode", type=str, default="default",
                               choices=["default", "reduce-overhead",
                                        "max-autotune", "max-autotune-no-cudagraphs"],
                               help="torch.compile mode. 'default' is safest with DDP. "
                                    "'max-autotune-no-cudagraphs' can be faster but recompiles slower. "
                                    "Avoid 'reduce-overhead'/'max-autotune' with DDP — cudagraphs "
                                    "and DDP gradient buckets fight.")

    parallel_group = parser.add_argument_group("parallelism")
    parallel_group.add_argument("--num-dp-ranks", type=int, default=0,
                                help="Number of ranks for data parallelism. "
                                     "0 = disabled (single-process). When >0, launch with "
                                     "`torchrun --nproc-per-node=<N>`.")
    parallel_group.add_argument("--parallel-mode", type=str, default="ddp",
                                choices=["ddp", "fsdp", "zero1"],
                                help="Parallelism strategy when --num-dp-ranks > 0. "
                                     "ddp: DistributedDataParallel (default). "
                                     "fsdp: FullyShardedDataParallel (shards params+grads+optimizer). "
                                     "zero1: DDP + ZeroRedundancyOptimizer (shards optimizer state only).")

    args = parser.parse_args()

    conf = load_config(args.training_config, "parameters")

    for path in [args.model_config, args.training_config,
                 conf["training_data_config"], conf["validation_data_config"]]:
        assert os.path.exists(path), f"Config not found: {path}"

    eval_config = None
    if args.eval_config:
        assert os.path.exists(args.eval_config), f"Eval config not found: {args.eval_config}"
        eval_config = load_config(args.eval_config, "parameters")

    loss_viz_config = None
    if args.loss_viz:
        loss_viz_config = {
            "grid_points": args.loss_viz_grid_points,
            "grid_range":  args.loss_viz_grid_range,
            "eval_batches": args.loss_viz_eval_batches,
            "interactive":  args.loss_viz_interactive,
        }

    if args.load:
        assert os.path.exists(args.load), f"Run directory not found: {args.load}"
        run_dir = args.load
    else:
        run_name = f"{conf['run_name']}-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        run_dir = os.path.join(args.base_run_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        for path in [args.model_config, args.training_config,
                     conf["training_data_config"], conf["validation_data_config"]]:
            shutil.copy(path, run_dir)
        if args.tokenizer_path and os.path.exists(args.tokenizer_path):
            shutil.copy(args.tokenizer_path, run_dir)
        if args.eval_config:
            shutil.copy(args.eval_config, run_dir)

    use_distributed = args.num_dp_ranks > 0
    parallel_mode = args.parallel_mode if use_distributed else "none"
    local_rank = 0
    if use_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        setup()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name, _, net, tokenizer = load_language_model(args.model_config, device)
    assert net.vocab_size >= len(tokenizer), (
        f"Model vocab size {net.vocab_size} < tokenizer size {len(tokenizer)}"
    )

    x = torch.zeros((1, conf["sequence_length"]), dtype=torch.long, device=device)
    if local_rank == 0:
        summary(net, input_data=x)

    # Trace forward FLOPs once on the bare module before DDP/FSDP wrapping.
    # We use the actual (micro_batch, sequence_length) shape so the figure
    # matches a real microstep. fvcore is robust to fp32 weights + bf16 autocast
    # (it counts ops, not bytes).
    forward_flops_per_microbatch = 0
    if local_rank == 0:
        fwd_example = torch.zeros(
            (conf["micro_batch_size"], conf["sequence_length"]),
            dtype=torch.long, device=device,
        )
        forward_flops_per_microbatch = compute_forward_flops(net, fwd_example)
        del fwd_example
        torch.cuda.empty_cache()
    if use_distributed:
        flops_t = torch.tensor([forward_flops_per_microbatch], dtype=torch.long, device=device)
        torch.distributed.broadcast(flops_t, src=0)
        forward_flops_per_microbatch = int(flops_t.item())
    peak_bf16_tflops_per_gpu = device_peak_bf16_tflops(device)
    if local_rank == 0:
        if forward_flops_per_microbatch > 0:
            print(
                f"Forward FLOPs / microbatch (B={conf['micro_batch_size']}, "
                f"T={conf['sequence_length']}): {forward_flops_per_microbatch:,}"
            )
        else:
            print("fvcore unavailable or trace failed — compute/forward_mfu will log 0.")
        if peak_bf16_tflops_per_gpu > 0:
            print(f"Device bf16 peak: {peak_bf16_tflops_per_gpu:.1f} TFLOPs/s "
                  f"({torch.cuda.get_device_name(device)})")
        else:
            print(f"Unknown GPU peak for {torch.cuda.get_device_name(device)} — "
                  "compute/forward_mfu will log 0.")

    # Compile BEFORE DDP/FSDP wrap. The DDP forward then dispatches into the
    # compiled graph for the per-rank microbatch, while DDP's bucketed
    # gradient all-reduces stay in eager (which is what we want — DDP and
    # cudagraphs don't compose well). Stick to mode="default" with DDP;
    # 'reduce-overhead' / 'max-autotune' use cudagraphs and can deadlock.
    if args.compile:
        if local_rank == 0:
            print(f"torch.compile(mode={args.compile_mode!r}) — first forward will be slow "
                  "(graph capture + autotune); subsequent steps should be much faster.")
        net = torch.compile(net, mode=args.compile_mode)

    if use_distributed:
        if parallel_mode == "fsdp":
            net = FSDP(net, device_id=local_rank)
        else:
            net = DDP(net, device_ids=[local_rank])

    micro_batch_size = conf["micro_batch_size"]
    global_batch_size = conf["global_batch_size"]
    if use_distributed:
        assert global_batch_size % (micro_batch_size * args.num_dp_ranks) == 0, (
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"micro_batch_size ({micro_batch_size}) * num_dp_ranks ({args.num_dp_ranks}) "
            f"= {micro_batch_size * args.num_dp_ranks}"
        )
        grad_accum_steps = global_batch_size // (micro_batch_size * args.num_dp_ranks)
    else:
        assert global_batch_size % micro_batch_size == 0, (
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"micro_batch_size ({micro_batch_size})"
        )
        grad_accum_steps = global_batch_size // micro_batch_size

    train_seq = conf["training_sequence_length"]
    train_dataset = load_dataset(conf["training_data_config"], train_seq["start"],
                                 debug=args.verbose, val=False, tokenizer=tokenizer)
    if use_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = SequenceLengthSampler(
            len(train_dataset), micro_batch_size,
            train_seq["start"], train_seq["end"], train_seq["steps"],
            name="train_scheduler", shuffle=False,
        )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        num_workers=conf["num_workers"],
        sampler=train_sampler,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=conf["num_workers"] > 0,
    )

    val_dataset = load_dataset(conf["validation_data_config"], conf["validation_sequence_length"],
                               debug=args.verbose, val=True, tokenizer=tokenizer)
    val_batch_size = args.val_batch_size if args.val_batch_size is not None else micro_batch_size
    # num_workers=0 avoids spawning 7×N worker processes simultaneously when all
    # ranks start validation at the same time, which can trigger the OOM killer.
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size,
                                shuffle=True, num_workers=0)

    optim = load_optimizer(net, **conf["optimizer"])
    if parallel_mode == "zero1":
        _param_groups = optim.param_groups
        _optim_class = type(optim)
        _sig_params = inspect.signature(_optim_class.__init__).parameters
        _optim_defaults = {k: v for k, v in optim.defaults.items() if k in _sig_params}
        del optim
        optim = ZeroRedundancyOptimizer(_param_groups, optimizer_class=_optim_class, **_optim_defaults)
    warmup_steps = conf.get("warmup_steps", min(2000, conf["training_steps"] // 20))
    min_lr_ratio = conf.get("min_lr_ratio", 0.01)
    scheduler = LambdaLR(
        optim,
        lambda step: cosine_warmup_lr_lambda(step, warmup_steps, conf["training_steps"], min_lr_ratio),
    )

    start_iter = 0
    cumulative_tokens_start = 0
    source_tokens_start: dict = {}
    resume_info = ""
    if args.load:
        if args.checkpoint:
            ckpt_dir = os.path.join(run_dir, args.checkpoint)
            assert os.path.isdir(ckpt_dir), f"Checkpoint not found: {ckpt_dir}"
            latest_iter = int(re.search(r"checkpoint-(\d+)$", ckpt_dir).group(1))
        else:
            ckpt_dir, latest_iter = find_latest_checkpoint(run_dir)
        if ckpt_dir:
            state = load_checkpoint(ckpt_dir, net, optim, scheduler, parallel_mode=parallel_mode)
            start_iter = latest_iter + 1
            history = state.get("log_history", [])
            cumulative_tokens_start = history[-1].get("cumulative_tokens", 0) if history else 0
            source_tokens_start = history[-1].get("source_tokens", {}) if history else {}
            resume_info = f" (resumed from step {latest_iter})"
            print(f"Resumed from {ckpt_dir}")
        else:
            print(f"No checkpoint found in {run_dir}, starting from scratch.")

    profile_start = args.profile_start
    profile_end = args.profile_end
    if args.profile_relative and profile_start >= 0:
        profile_start += start_iter
        profile_end += start_iter

    if local_rank == 0:
        assert os.getenv("WANDB_API_KEY", None), "Wandb API key is none"
        wandb.init(
            project=conf["experiment_name"],
            name=f"{conf['run_name']}{resume_info}",
            config={
                "model": name,
                "micro_batch_size": micro_batch_size,
                "global_batch_size": global_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "sequence_length": conf["sequence_length"],
                "lr": conf["lr"],
                "num_iters": conf["training_steps"],
                "epochs": conf.get("epochs", 1),
                "warmup_steps": warmup_steps,
                "min_lr_ratio": min_lr_ratio,
                "profile_start": profile_start,
                "profile_end": profile_end,
                "resumed": args.load is not None,
                "start_iter": start_iter,
                "num_dp_ranks": args.num_dp_ranks,
                "parallel_mode": parallel_mode,
                "enable_viz": args.enable_viz,
                "log_freq": args.log_freq,
                "compile": args.compile,
                "compile_mode": args.compile_mode if args.compile else None,
            },
        )
        # wandb.watch is expensive on multi-billion-param models (it logs grad
        # and/or param histograms for every tensor). Only enable when viz is on,
        # and even then: gradients only, every 1000 steps.
        if args.enable_viz:
            wandb.watch(net, log="gradients", log_freq=1000)

    pretrain(
        net=net,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optim=optim,
        scheduler=scheduler,
        device=device,
        num_iters=conf["training_steps"],
        num_epochs=conf.get("epochs", 1),
        num_val_iters=conf["val_steps"],
        checkpoint_frequency=conf["checkpoint_frequency"],
        experiment_name=conf["experiment_name"],
        model_name=name,
        run_dir=run_dir,
        profile_start=profile_start,
        profile_end=profile_end,
        capture_memory_snapshot=args.capture_memory_snapshot,
        memory_snapshot_events=args.memory_snapshot_events,
        start_iter=start_iter,
        cumulative_tokens_start=cumulative_tokens_start,
        source_tokens_start=source_tokens_start,
        eval_config=eval_config,
        num_visualize_generations=args.num_visualize_generations,
        loss_viz_config=loss_viz_config,
        spectral_viz=args.spectral_viz,
        local_rank=local_rank,
        resumed=args.load is not None and start_iter > 0,
        parallel_mode=parallel_mode,
        grad_accum_steps=grad_accum_steps,
        enable_viz=args.enable_viz,
        log_freq=args.log_freq,
        forward_flops_per_microbatch=forward_flops_per_microbatch,
        peak_bf16_tflops_per_gpu=peak_bf16_tflops_per_gpu,
    )

    if use_distributed:
        cleanup()


if __name__ == "__main__":
    main()
