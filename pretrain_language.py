import argparse
from collections import defaultdict
import json
import shutil
import torch
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
from safetensors.torch import load_file
from models.loader import load_language_model, load_dataset
from models.tasks.language.datasets.sequence_length import SequenceLengthSampler
from models.tasks.language.optimizers.loader import load_optimizer
from utils import load_config
from dotenv import load_dotenv
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from hooks import register_activation_hooks
from visualize_transformer import plot_visualizations
from generate_language import generate

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


def load_checkpoint(checkpoint_dir, net, optim, scheduler):
    """
    Restore model weights from safetensors and optimizer/scheduler from .pt files.
    Returns the trainer_state dict (used to recover cumulative_tokens etc.), or {}.
    """
    weights_path = os.path.join(checkpoint_dir, "model.safetensors")
    net.load_state_dict(load_file(weights_path))

    for fname, obj in [("optimizer.pt", optim), ("scheduler.pt", scheduler)]:
        path = os.path.join(checkpoint_dir, fname)
        if os.path.exists(path):
            obj.load_state_dict(torch.load(path, map_location="cpu"))

    state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def training_step(net, batch, labels):
    """Forward pass. Returns (logits, loss)."""
    output = net(batch)
    logits = output.logits if hasattr(output, "logits") else output
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), labels.view(B * T))
    return logits, loss


def run_lm_eval(net, tokenizer, eval_conf, device):
    """
    Run lm-eval tasks defined in eval_conf against net and return the results dict.

    eval_conf keys:
      tasks       - list of lm-eval task names (required)
      num_fewshot - number of few-shot examples (default 0)
      limit       - cap examples per task, useful for quick checks (default None)
      batch_size  - passed to HFLM (default "auto")

    Requires tokenizer to be a HuggingFace PreTrainedTokenizer.
    """
    max_len = getattr(net.config, "max_position_embeddings", 1024)
    lm = HFLM(
        pretrained=net,
        tokenizer=tokenizer,
        batch_size=eval_conf.get("batch_size", "auto"),
        max_length=max_len,
    )
    results = simple_evaluate(
        model=lm,
        tasks=eval_conf["tasks"],
        num_fewshot=eval_conf.get("num_fewshot", 0),
        limit=eval_conf.get("limit", None),
    )
    return results.get("results", {})


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
                batch, labels = next(val_iterator)
            except StopIteration:
                break
            batch, labels = batch.to(device), labels.to(device)
            _, loss = training_step(net, batch=batch, labels=labels)
            val_loss_total += loss.item()
            n_batches += 1

    avg_loss = val_loss_total / max(1, n_batches)

    # Generations
    generation_dir = os.path.join(run_dir, "generations")
    os.makedirs(generation_dir, exist_ok=True)
    generations = []
    for gen_idx in range(3):
        try:
            val_batch, _ = next(val_iterator)
        except StopIteration:
            break
        prompt_tokens = val_batch[0, : val_batch.shape[1] // 2].tolist()
        prompt_text = tokenizer.decode(prompt_tokens)
        output_text = generate(
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
    num_val_iters=10,
    checkpoint_frequency=5,
    experiment_name=None,
    model_name=None,
    run_dir=None,
    profile_steps=0,
    start_iter=0,
    cumulative_tokens_start=0,
    eval_config=None,
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

    trainer_state.json is also written at the run root after every checkpoint. It follows the
    same log_history schema as HuggingFace Trainer so tooling that reads HF checkpoints works.

    Args:
        net: LanguageModel (PreTrainedModel).
        tokenizer: HuggingFace PreTrainedTokenizer.
        train_dataloader / val_dataloader: DataLoaders for train and validation splits.
        optim / scheduler: Optimizer and LR scheduler.
        device: torch.device.
        num_iters: Total training steps.
        num_val_iters: Validation batches to evaluate per checkpoint.
        checkpoint_frequency: Save a checkpoint every N steps (and always at the final step).
        experiment_name: WandB project name.
        model_name: Model identifier used in logging.
        run_dir: Root directory for this run (checkpoints, logs, and configs live here).
        profile_steps: Steps to profile with torch.profiler; 0 disables profiling.
        start_iter: First iteration index (non-zero when resuming).
        cumulative_tokens_start: Token count carried over from a resumed checkpoint.
        eval_config: Dict of lm-eval settings (tasks, num_fewshot, limit, batch_size).
                     If None, lm-eval is skipped. Loaded from --eval_config YAML in main().
    """
    net = net.to(device)
    net.train()
    data_iterator = iter(train_dataloader)

    # Profiler
    profiler = None
    if profile_steps > 0:
        profile_dir = os.path.join(run_dir, "profile")
        os.makedirs(profile_dir, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=profile_steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
            record_shapes=True, profile_memory=True, with_stack=True,
        )
        profiler.start()

    activation_stats = defaultdict(list)
    hooks = register_activation_hooks(net, activation_stats)
    cumulative_tokens = cumulative_tokens_start
    log_history = []

    for i in range(start_iter, num_iters):
        optim.zero_grad()

        t_batch = time.time()
        try:
            batch, labels = next(data_iterator)
        except StopIteration:
            print(f"Data exhausted at step {i}.")
            break
        batch_load_time = time.time() - t_batch

        batch, labels = batch.to(device), labels.to(device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.time()
        _, loss = training_step(net, batch=batch, labels=labels)
        forward_time = time.time() - t0

        t1 = time.time()
        loss.backward()
        backward_time = time.time() - t1

        t2 = time.time()
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

        wandb.log({
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
        }, step=i)

        if profiler is not None:
            profiler.step()

        # --- Checkpoint ---
        if i % checkpoint_frequency == 0 or i == num_iters - 1:
            checkpoint_dir = os.path.join(run_dir, f"checkpoint-{i}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Model weights (safetensors) + config.json
            net.save_pretrained(checkpoint_dir, safe_serialization=True)

            # Tokenizer (HF-backed models only)
            if tokenizer is not None:
                tokenizer.save_pretrained(checkpoint_dir)

            # Optimizer & scheduler (needed to resume training)
            torch.save(optim.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

            # Activation memory statistics
            with open(os.path.join(checkpoint_dir, "activations.json"), "w") as f:
                json.dump(dict(activation_stats), f)

            # Validation
            avg_val_loss, avg_val_ppl = run_validation(
                net, tokenizer, val_dataloader, device,
                run_dir=run_dir, max_steps=num_val_iters, step=i,
            )
            net.train()
            wandb.log({"val_loss_avg": avg_val_loss, "val_perplexity_avg": avg_val_ppl}, step=i)

            # lm-eval (optional)
            if eval_config is not None and tokenizer is not None:
                print(f"[iter {i}] Running lm-eval: {eval_config['tasks']}")
                net.eval()
                eval_results = run_lm_eval(net, tokenizer, eval_config, device)
                net.train()
                eval_out = {
                    "iter": i,
                    "tasks": eval_config["tasks"],
                    "num_fewshot": eval_config.get("num_fewshot", 0),
                    "results": eval_results,
                }
                with open(os.path.join(checkpoint_dir, "eval_results.json"), "w") as f:
                    json.dump(eval_out, f, indent=2)
                for task, res in eval_results.items():
                    for metric, val in res.items():
                        if isinstance(val, (int, float)):
                            wandb.log({f"eval/{task}/{metric}": val}, step=i)
                print(f"[iter {i}] lm-eval saved → {checkpoint_dir}/eval_results.json")

            # Visualizations
            plot_visualizations(net, os.path.join(run_dir, "visualizations"), i)

            # trainer_state.json — written to both run root and checkpoint dir
            log_history.append({
                "step": i,
                "epoch": round(i / num_iters, 4),
                "loss": round(loss.item(), 4),
                "perplexity": round(train_ppl, 4),
                "val_loss": round(avg_val_loss, 4),
                "val_perplexity": round(avg_val_ppl, 4),
                "learning_rate": current_lr,
                "cumulative_tokens": cumulative_tokens,
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
    if profiler is not None:
        profiler.stop()


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
    parser.add_argument("--profile", type=int, default=0,
                        help="Number of steps to profile with torch.profiler (0 = disabled)")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to an existing run directory to resume training from. "
                             "The latest checkpoint-{N} inside it will be loaded automatically.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose dataset output")

    args = parser.parse_args()

    conf = load_config(args.training_config, "parameters")

    for path in [args.model_config, args.training_config,
                 conf["training_data_config"], conf["validation_data_config"]]:
        assert os.path.exists(path), f"Config not found: {path}"

    eval_config = None
    if args.eval_config:
        assert os.path.exists(args.eval_config), f"Eval config not found: {args.eval_config}"
        eval_config = load_config(args.eval_config, "parameters")

    # Run directory — reuse existing if resuming, create new one otherwise
    if args.load:
        assert os.path.exists(args.load), f"Run directory not found: {args.load}"
        run_dir = args.load
    else:
        run_name = f"{conf['experiment_name']}-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        run_dir = os.path.join("checkpoints", run_name)
        os.makedirs(run_dir, exist_ok=True)
        # Copy all configs into the run directory for reproducibility
        for path in [args.model_config, args.training_config,
                     conf["training_data_config"], conf["validation_data_config"]]:
            shutil.copy(path, run_dir)
        if args.tokenizer_path and os.path.exists(args.tokenizer_path):
            shutil.copy(args.tokenizer_path, run_dir)
        if args.eval_config:
            shutil.copy(args.eval_config, run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name, _, net, tokenizer = load_language_model(args.model_config, device)
    assert net.vocab_size >= len(tokenizer), (
        f"Model vocab size {net.vocab_size} < tokenizer size {len(tokenizer)}"
    )

    x = torch.zeros((1, conf["sequence_length"]), dtype=torch.long, device=device)
    summary(net, input_data=x)

    # Datasets & dataloaders
    train_seq = conf["training_sequence_length"]
    train_dataset = load_dataset(conf["training_data_config"], train_seq["start"],
                                 debug=args.verbose, val=False)
    seqlen_sampler = SequenceLengthSampler(
        len(train_dataset), conf["batch_size"],
        train_seq["start"], train_seq["end"], train_seq["steps"],
        name="train_scheduler", shuffle=False,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=conf["batch_size"],
                                  num_workers=conf["num_workers"], sampler=seqlen_sampler)

    val_dataset = load_dataset(conf["validation_data_config"], conf["validation_sequence_length"],
                               debug=args.verbose, val=True)
    val_dataloader = DataLoader(val_dataset, batch_size=conf["batch_size"],
                                shuffle=True, num_workers=conf["num_workers"])

    # Optimizer & LR schedule
    optim = load_optimizer(net, **conf["optimizer"])
    warmup_steps = conf.get("warmup_steps", min(2000, conf["epochs"] // 20))
    min_lr_ratio = conf.get("min_lr_ratio", 0.01)
    scheduler = LambdaLR(
        optim,
        lambda step: cosine_warmup_lr_lambda(step, warmup_steps, conf["epochs"], min_lr_ratio),
    )

    # Resume from checkpoint if --load was given
    start_iter = 0
    cumulative_tokens_start = 0
    resume_info = ""
    if args.load:
        ckpt_dir, latest_iter = find_latest_checkpoint(run_dir)
        if ckpt_dir:
            state = load_checkpoint(ckpt_dir, net, optim, scheduler)
            start_iter = latest_iter + 1
            history = state.get("log_history", [])
            cumulative_tokens_start = history[-1].get("cumulative_tokens", 0) if history else 0
            resume_info = f" (resumed from step {latest_iter})"
            print(f"Resumed from {ckpt_dir}")
        else:
            print(f"No checkpoint found in {run_dir}, starting from scratch.")

    wandb.init(
        project=conf["experiment_name"],
        name=f"{conf['experiment_name']}{resume_info}",
        config={
            "model": name,
            "batch_size": conf["batch_size"],
            "sequence_length": conf["sequence_length"],
            "lr": conf["lr"],
            "num_iters": conf["epochs"],
            "warmup_steps": warmup_steps,
            "min_lr_ratio": min_lr_ratio,
            "profile_steps": args.profile,
            "resumed": args.load is not None,
            "start_iter": start_iter,
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
        num_iters=conf["epochs"],
        num_val_iters=conf["val_steps"],
        checkpoint_frequency=conf["checkpoint_frequency"],
        experiment_name=conf["experiment_name"],
        model_name=name,
        run_dir=run_dir,
        profile_steps=args.profile,
        start_iter=start_iter,
        cumulative_tokens_start=cumulative_tokens_start,
        eval_config=eval_config,
    )


if __name__ == "__main__":
    main()
