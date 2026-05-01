import argparse
from collections import defaultdict
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
        raw_net = net.module if isinstance(net, DDP) else net
        raw_net.load_state_dict(load_file(weights_path))

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

def training_step(net, batch, labels):
    """Forward pass. Returns (logits, loss, metadata)."""
    output = net(batch)
    logits = output.logits if hasattr(output, "logits") else output
    metadata = output.metadata if hasattr(output, "metadata") else {}
    B, T, V = logits.shape
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

    all_runs = []
    for run_idx in range(num_runs):
        if num_runs > 1:
            print(f"  lm-eval run {run_idx + 1}/{num_runs}")
        lm = HFLM(
            pretrained=net,
            tokenizer=tokenizer,
            batch_size=eval_conf.get("batch_size", "auto"),
            max_length=max_len,
        )
        raw = simple_evaluate(
            model=lm,
            tasks=eval_conf["tasks"],
            num_fewshot=eval_conf.get("num_fewshot", 0),
            limit=eval_conf.get("limit", None),
        )
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


def run_validation(net, tokenizer, val_dataloader, device, run_dir, max_steps=10, step=0):
    """
    Run validation for up to max_steps batches, then generate a few samples.

    Saves generations to {run_dir}/generations/val_iter_{step}.json.
    Returns (avg_val_loss, avg_val_perplexity).
    """
    net.eval()
    val_loss_total = 0.0
    n_batches = 0
    val_iterator = iter(val_dataloader)

    with torch.no_grad():
        for _ in tqdm(range(max_steps), desc="Validation"):
            try:
                batch, labels, _ = next(val_iterator)
            except StopIteration:
                break
            batch, labels = batch.to(device), labels.to(device)
            logits, loss, metadata = training_step(net, batch=batch, labels=labels)
            val_loss_total += loss.item()
            n_batches += 1

    avg_loss = val_loss_total / max(1, n_batches)

    generation_dir = os.path.join(run_dir, "generations")
    os.makedirs(generation_dir, exist_ok=True)
    generations = []
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

    with open(os.path.join(generation_dir, f"val_iter_{step}.json"), "w") as f:
        json.dump(generations, f, indent=2)

    return avg_loss, math.exp(avg_loss)


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
    if profile_start >= 0 and profile_end > profile_start:
        profile_dir = os.path.join(run_dir, "profile")
        os.makedirs(profile_dir, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
            record_shapes=True, profile_memory=True, with_stack=True,
        )
        print(f"Profiler armed: will capture iters [{profile_start}, {profile_end}] → {profile_dir}")

    activation_stats = defaultdict(list)
    hooks = register_activation_hooks(net, activation_stats)
    cumulative_tokens = cumulative_tokens_start
    log_history = []

    _source_names: list[str] = getattr(train_dataloader.dataset, "_names", [])
    _per_source_tokens: dict[str, int] = {
        name: (source_tokens_start or {}).get(name, 0)
        for name in _source_names
    }

    print(f"Training started: iters {start_iter} → {num_iters}, device={device}" + (" (resumed)" if resumed else ""))
    for i in range(start_iter, num_iters):
        optim.zero_grad()

        t_batch = time.time()
        try:
            batch, labels, source_idx = next(data_iterator)
        except StopIteration:
            current_epoch += 1
            if current_epoch >= num_epochs:
                print(f"Data exhausted after {current_epoch} epoch(s) at step {i}.")
                break
            print(f"Starting epoch {current_epoch + 1}/{num_epochs} at step {i}.")
            data_iterator = iter(train_dataloader)
            batch, labels, source_idx = next(data_iterator)
        batch_load_time = time.time() - t_batch

        # Accumulate per-source token counts (source_idx is a [B] int tensor)
        _step_source_tokens: dict[str, int] = {}
        seq_len = labels.shape[1]
        for j, name in enumerate(_source_names):
            n = int((source_idx == j).sum().item())
            tokens = n * seq_len
            _per_source_tokens[name] = _per_source_tokens.get(name, 0) + tokens
            _step_source_tokens[name] = tokens

        batch, labels = batch.to(device), labels.to(device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.time()
        _, loss, metadata = training_step(net, batch=batch, labels=labels)
        forward_time = time.time() - t0

        t1 = time.time()
        loss.backward()
        backward_time = time.time() - t1

        t2 = time.time()
        if isinstance(net, FSDP):
            net.clip_grad_norm_(max_norm=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()
        optimizer_time = time.time() - t2

        iter_time = forward_time + backward_time + optimizer_time
        cumulative_tokens += labels.numel()
        current_lr = scheduler.get_last_lr()[0]
        train_ppl = math.exp(loss.item())

        if torch.cuda.is_available():
            param_mem = sum(p.numel() * p.element_size() for p in net.parameters()) / (1024**2)
            opt_mem = sum(
                v.numel() * v.element_size()
                for state in optim.state.values()
                for v in state.values()
                if torch.is_tensor(v)
            ) / (1024**2)
            allocated_mem = torch.cuda.memory_allocated(device) / (1024**2)
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            param_mem = opt_mem = allocated_mem = peak_mem = 0.0

        total_activation_mem = sum(activation_stats.values())

        wandb_log = {
            "lm_loss": loss.item(),
            "train_perplexity": train_ppl,
            "learning_rate": current_lr,
            "iter_time": iter_time,
            "batch_load_time": batch_load_time,
            "cumulative_tokens": cumulative_tokens,
            "train_sequence_length": batch.shape[1],
            "params_VRAM_MB": param_mem,
            "optimizer_state_VRAM_MB": opt_mem,
            "activations_VRAM_MB": total_activation_mem,
            "allocated_VRAM_MB": allocated_mem,
            "peak_allocated_VRAM_MB": peak_mem,
        }

        for k, v in metadata.items():
            if k.startswith("metrics/model/"):
                wandb_log["model/" + k[len("metrics/model/"):]] = v
        for name in _source_names:
            wandb_log[f"data/{name}/tokens"] = _step_source_tokens.get(name, 0)
            wandb_log[f"data/{name}/cumulative_tokens"] = _per_source_tokens.get(name, 0)
        if local_rank == 0:
            wandb.log(wandb_log, step=i)

        if profiler is not None and i == profile_start:
            profiler.start()
            _profiler_active = True
            print(f"[iter {i}] Profiler started")
        if _profiler_active:
            profiler.step()
        if _profiler_active and i == profile_end:
            profiler.stop()
            _profiler_active = False
            profiler = None
            print(f"[iter {i}] Profiler stopped → {profile_dir}")

        do_checkpoint = (i % checkpoint_frequency == 0 or i == num_iters - 1)

        # All-rank operations that must precede rank-0 I/O.
        _fsdp_model_state = _fsdp_optim_state = None
        if do_checkpoint and isinstance(net, FSDP):
            with FSDP.state_dict_type(net, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                _fsdp_model_state = net.state_dict()
            _fsdp_optim_state = FSDP.optim_state_dict(net, optim)
        if do_checkpoint and isinstance(optim, ZeroRedundancyOptimizer):
            optim.consolidate_state_dict(to=0)

        if local_rank == 0 and do_checkpoint:
            checkpoint_dir = os.path.join(run_dir, f"checkpoint-{i}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            if isinstance(net, FSDP):
                save_file(_fsdp_model_state, os.path.join(checkpoint_dir, "model.safetensors"))
                net.module.config.save_pretrained(checkpoint_dir)
                torch.save(_fsdp_optim_state, os.path.join(checkpoint_dir, "optimizer.pt"))
            else:
                raw_net = net.module if isinstance(net, DDP) else net
                raw_net.save_pretrained(checkpoint_dir, safe_serialization=True)
                torch.save(optim.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

            if tokenizer is not None:
                tokenizer.save_pretrained(checkpoint_dir)

            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

            with open(os.path.join(checkpoint_dir, "activations.json"), "w") as f:
                json.dump(dict(activation_stats), f)

            eval_net = net if isinstance(net, FSDP) else (net.module if isinstance(net, DDP) else net)
            avg_val_loss, avg_val_ppl = run_validation(
                eval_net, tokenizer, val_dataloader, device,
                run_dir=run_dir, max_steps=num_val_iters, step=i,
            )
            net.train()
            wandb.log({"val_loss_avg": avg_val_loss, "val_perplexity_avg": avg_val_ppl}, step=i)

            if eval_config is not None and tokenizer is not None:
                num_runs = int(eval_config.get("num_runs", 1))
                print(f"[iter {i}] Running lm-eval ({num_runs} run(s)): {eval_config['tasks']}")
                eval_net.eval()
                eval_results = run_lm_eval(eval_net, tokenizer, eval_config, device)
                net.train()
                eval_out = {
                    "iter": i,
                    "tasks": eval_config["tasks"],
                    "num_fewshot": eval_config.get("num_fewshot", 0),
                    "num_runs": num_runs,
                    "results": eval_results,
                }
                with open(os.path.join(checkpoint_dir, "eval_results.json"), "w") as f:
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
                print(f"[iter {i}] lm-eval saved → {checkpoint_dir}/eval_results.json")

            skip_viz = resumed and i == start_iter
            if not skip_viz:
                plot_visualizations(eval_net, os.path.join(run_dir, "visualizations"), i)

            if not skip_viz and num_visualize_generations > 0 and tokenizer is not None:
                viz_batch, _, _ = next(iter(val_dataloader))
                viz_prompt_tokens = viz_batch[0, :viz_batch.shape[1] // 2].tolist()
                viz_prompt = tokenizer.decode(viz_prompt_tokens)
                eval_net.eval()
                _, hidden_states_per_step = generate(
                    viz_prompt, tokenizer, eval_net, device,
                    max_output_length=num_visualize_generations,
                    generation_folder="", checkpoint_path="", tokenizer_path="",
                    collect_hidden_states=True,
                )
                net.train()

                if hidden_states_per_step:
                    from hyperviz import Visualizer
                    from hyperviz.trajectory import Trajectory
                    viz_dir = os.path.join(checkpoint_dir, "viz")
                    visualizer = Visualizer(viz_dir)
                    for step_hs in hidden_states_per_step:
                        if step_hs is not None:
                            visualizer.add(Trajectory(hidden_states=step_hs))
                    visualizer.visualize()
                    visualizer.clear()
                    print(f"[iter {i}] hyperviz → {viz_dir}")

            if not skip_viz and loss_viz_config is not None:
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
                eval_net.eval()
                loss_visualizer.visualize(eval_net, val_dataloader, device)
                del loss_visualizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                net.train()
                print(f"[iter {i}] loss landscape → {loss_viz_dir}")

            if not skip_viz and spectral_viz:
                from hyperviz.spectral_visualizer import SpectralVisualizer
                spectral_viz_dir = os.path.join(checkpoint_dir, "viz")
                sv = SpectralVisualizer(save_directory=spectral_viz_dir)
                eval_net.eval()
                with torch.no_grad():
                    sv.visualize(eval_net)
                net.train()
                print(f"[iter {i}] spectral viz → {spectral_viz_dir}/spectral_values/")

            log_history.append({
                "step": i,
                "epoch": round(i / num_iters, 4),
                "loss": round(loss.item(), 4),
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
                f"[iter {i}] loss={loss.item():.4f} ppl={train_ppl:.2f} "
                f"val_loss={avg_val_loss:.4f} val_ppl={avg_val_ppl:.2f} "
                f"lr={current_lr:.2e} tokens={cumulative_tokens} "
                f"→ {checkpoint_dir}"
            )

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

    if use_distributed:
        if parallel_mode == "fsdp":
            net = FSDP(net, device_id=local_rank)
        else:
            net = DDP(net, device_ids=[local_rank])

    train_seq = conf["training_sequence_length"]
    train_dataset = load_dataset(conf["training_data_config"], train_seq["start"],
                                 debug=args.verbose, val=False, tokenizer=tokenizer)
    if use_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = SequenceLengthSampler(
            len(train_dataset), conf["batch_size"],
            train_seq["start"], train_seq["end"], train_seq["steps"],
            name="train_scheduler", shuffle=False,
        )
    train_dataloader = DataLoader(train_dataset, batch_size=conf["batch_size"],
                                  num_workers=conf["num_workers"], sampler=train_sampler)

    val_dataset = load_dataset(conf["validation_data_config"], conf["validation_sequence_length"],
                               debug=args.verbose, val=True, tokenizer=tokenizer)
    val_batch_size = args.val_batch_size if args.val_batch_size is not None else conf["batch_size"]
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size,
                                shuffle=True, num_workers=conf["num_workers"])

    optim = load_optimizer(net, **conf["optimizer"])
    if parallel_mode == "zero1":
        _param_groups = optim.param_groups
        _optim_class = type(optim)
        _optim_defaults = dict(optim.defaults)
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
                "batch_size": conf["batch_size"],
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
            },
        )
        wandb.watch(net, log="all", log_freq=100)

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
    )

    if use_distributed:
        cleanup()


if __name__ == "__main__":
    main()
