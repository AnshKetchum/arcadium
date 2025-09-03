import argparse
from collections import defaultdict
import json
import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import wandb
from models.loader import load_language_model, load_dataset, load_tokenizer
from utils import load_config
from dotenv import load_dotenv 
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.optim import Adam
from hooks import register_activation_hooks
from visualize_transformer import plot_visualizations

load_dotenv()

def loss_function(logits, labels):
    # logits: [B, T, V], labels: [B, T]
    B, T, V = logits.shape
    logits = logits.view(B * T, V)   # [B*T, V]
    labels = labels.view(B * T)      # [B*T]
    return F.cross_entropy(logits, labels)

def training_step(net: nn.Module, batch: torch.Tensor, labels: torch.Tensor):
    logits = net(batch)
    loss = loss_function(logits, labels)
    return logits, loss

def run_validation(
    net: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_steps: int = 10,
    step: int = 0,
):
    net.eval()
    val_loss_total = 0.0
    n_batches = 0
    val_iterator = iter(val_dataloader)

    # Setup activation hooks for validation
    activation_stats = defaultdict(list)
    hooks = register_activation_hooks(net, activation_stats)

    with torch.no_grad():
        for j in range(max_steps):
            try:
                batch, labels = next(val_iterator)
            except StopIteration:
                break

            batch_mem = batch.numel() * batch.element_size() / (1024 ** 2)
            labels_mem = labels.numel() * labels.element_size() / (1024 ** 2)
            batch, labels = batch.to(device), labels.to(device)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            logits, loss = training_step(net, batch=batch, labels=labels)  # same step but no backward
            if torch.cuda.is_available():
                forward_allocated_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
                forward_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                forward_allocated_mem = forward_peak_mem = 0.0

            val_loss_total += loss.item()
            n_batches += 1

            # Compute activation memory
            total_activation_mem = sum(activation_stats.values())  # MB

            if torch.cuda.is_available():
                param_mem = sum(p.numel() * p.element_size() for p in net.parameters()) / (1024 ** 2)
                allocated_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
                peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                rem_vram_alloc = allocated_mem - (param_mem + batch_mem + labels_mem + total_activation_mem)
                rem_vram_peak = peak_mem - (param_mem + batch_mem + labels_mem + total_activation_mem)
                rem_vram_alloc = max(rem_vram_alloc, 0.0)
                rem_vram_peak = max(rem_vram_peak, 0.0)
            else:
                param_mem = batch_mem = labels_mem = total_activation_mem = allocated_mem = peak_mem = 0.0
                rem_vram_alloc = rem_vram_peak = 0.0

            # Log to wandb (per val step)
            wandb.log({
                "val_loss": loss.item(),
                "val_params_VRAM_MB": param_mem,
                "val_batch_VRAM_MB": batch_mem,
                "val_labels_VRAM_MB": labels_mem,
                "val_activations_VRAM_MB": total_activation_mem,
                "val_allocated_VRAM_MB": allocated_mem,
                "val_peak_allocated_VRAM_MB": peak_mem,
                "val_forward_allocated_VRAM_MB": forward_allocated_mem,
                "val_forward_peak_VRAM_MB": forward_peak_mem,
                "val_rem_vram_alloc_MB": rem_vram_alloc,
                "val_rem_vram_peak_MB": rem_vram_peak,
            }, step=step)

            activation_stats.clear()

    # Cleanup hooks
    for h in hooks:
        h.remove()

    avg_val_loss = val_loss_total / max(1, n_batches)
    return avg_val_loss


def pretrain(
    net: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    num_iters=10,
    num_val_iters = 10,
    checkpoint_frequency: int = 5,
    experiment_name=None,
    model_name: str = None
):
    # Checkpoint folder
    checkpoint_folder_name = f"{experiment_name}-{model_name}-{time.strftime('%Y-%m-%d-%H:%M:%S')}".strip().replace(" ", "-")
    ckpt = os.path.join("checkpoints", checkpoint_folder_name)
    os.makedirs(ckpt, exist_ok=True)

    net = net.to(device)
    net.train()
    data_iterator = iter(train_dataloader)
    
    # --- Activation hook setup ---
    activation_stats = defaultdict(list)
    hooks = register_activation_hooks(net, activation_stats)
    cumulative_tokens = 0

    for i in range(num_iters):
        start_time = time.time()
        optim.zero_grad()

        try:
            batch, labels = next(data_iterator)
        except StopIteration:
            print(f"Stopped at {i + 1} training iterations.")
            break

        batch_mem = batch.numel() * batch.element_size() / (1024 ** 2)
        labels_mem = labels.numel() * labels.element_size() / (1024 ** 2)
        batch, labels = batch.to(device), labels.to(device)

        # --- Forward pass ---
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        logits, loss = training_step(net, batch=batch, labels=labels)
        if torch.cuda.is_available():
            forward_allocated_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
            forward_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # --- Backward pass ---
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        loss.backward()
        if torch.cuda.is_available():
            backward_allocated_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
            backward_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # --- Optimizer step ---
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        optim.step()
        if torch.cuda.is_available():
            optimizer_allocated_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
            optimizer_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        elapsed = time.time() - start_time
        iters_per_sec = 1 / elapsed if elapsed > 0 else 0.0

        # --- Compute activation memory from hooks ---
        total_activation_mem = sum(activation_stats.values())  # MB

        # ---- VRAM breakdown ----
        if torch.cuda.is_available():
            param_mem = sum(p.numel() * p.element_size() for p in net.parameters()) / (1024 ** 2)
            opt_mem = sum(
                v.numel() * v.element_size()
                for state in optim.state.values()
                for v in state.values()
                if torch.is_tensor(v)
            ) / (1024 ** 2)
            allocated_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            # remaining VRAM not accounted for
            rem_vram_alloc = allocated_mem - (param_mem + opt_mem + batch_mem + labels_mem + total_activation_mem)
            rem_vram_peak = peak_mem - (param_mem + opt_mem + batch_mem + labels_mem + total_activation_mem)
            rem_vram_alloc = max(rem_vram_alloc, 0.0)
            rem_vram_peak = max(rem_vram_peak, 0.0)
        else:
            param_mem = opt_mem = batch_mem = labels_mem = total_activation_mem = allocated_mem = peak_mem = 0.0
            forward_allocated_mem = forward_peak_mem = backward_allocated_mem = backward_peak_mem = 0.0
            optimizer_allocated_mem = optimizer_peak_mem = 0.0
            rem_vram_alloc = rem_vram_peak = 0.0

        # Update token counter (assuming labels length == tokens trained)
        cumulative_tokens += labels.numel()

        # --- Log metrics to wandb ---
        wandb.log({
            "lm_loss": loss.item(),
            "iters_per_sec": iters_per_sec,
            "cumulative_tokens": cumulative_tokens,
            "params_VRAM_MB": param_mem,
            "optimizer_state_VRAM_MB": opt_mem,
            "batch_VRAM_MB": batch_mem,
            "labels_VRAM_MB": labels_mem,
            "activations_VRAM_MB": total_activation_mem,
            "allocated_VRAM_MB": allocated_mem,
            "peak_allocated_VRAM_MB": peak_mem,
            "forward_allocated_VRAM_MB": forward_allocated_mem,
            "forward_peak_VRAM_MB": forward_peak_mem,
            "backward_allocated_VRAM_MB": backward_allocated_mem,
            "backward_peak_VRAM_MB": backward_peak_mem,
            "optimizer_allocated_VRAM_MB": optimizer_allocated_mem,
            "optimizer_peak_VRAM_MB": optimizer_peak_mem,
            "rem_vram_alloc_MB": rem_vram_alloc,
            "rem_vram_peak_MB": rem_vram_peak,
        }, step=i)

        # --- Checkpoint + JSON dump ---
        if i % checkpoint_frequency == 0 or i == num_iters - 1:
            print(
                f"[iter {i}] Loss {loss.item():.4f}, iters/sec {iters_per_sec:.2f}, "
                f"Tokens {cumulative_tokens}, "
                f"Params {param_mem:.1f}MB, Opt {opt_mem:.1f}MB, Batch {batch_mem:.1f}MB, "
                f"Acts {total_activation_mem:.1f}MB, VRAM {allocated_mem:.1f}MB (peak {peak_mem:.1f}MB), "
                f"Fwd {forward_allocated_mem:.1f}/{forward_peak_mem:.1f}MB, "
                f"Bwd {backward_allocated_mem:.1f}/{backward_peak_mem:.1f}MB, "
                f"Opt step {optimizer_allocated_mem:.1f}/{optimizer_peak_mem:.1f}MB"
            )
            checkpoint_path = os.path.join(ckpt, f"{model_name}-iter{i}.pt")
            torch.save({
                "iter": i,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": loss.item(),
            }, checkpoint_path)
            print(f"[iter {i}] Saved checkpoint to {checkpoint_path}")

            # Dump total activation VRAM to JSON
            json_path = os.path.join(ckpt, f"{model_name}-activations-iter{i}.json")
            with open(json_path, "w") as f:
                json.dump(activation_stats, f)
            print(f"[iter {i}] Saved activation VRAM JSON to {json_path}")

            plot_visualizations(
                net, 
                ckpt,
                i
            )
            
            # ---- Validation run ----
            avg_val_loss = run_validation(
                net, val_dataloader, device, max_steps=num_val_iters, step=i
            )
            print(f"[iter {i}] Validation avg loss: {avg_val_loss:.4f}")



        activation_stats.clear()  # reset after each forward to prevent blowup
            
    # Cleanup hooks after training
    for h in hooks:
        h.remove()


def main():
    parser = argparse.ArgumentParser(description="Pretrain a language model.")

    # Config files
    parser.add_argument("--model_config", type=str, default="configs/models/tiny-moe-64-emb.yaml",
                        help="Path to model config YAML")
    parser.add_argument("--training_config", type=str, default="configs/training/basic.yaml",
                        help="Path to training config YAML")

    args = parser.parse_args()

    # --- Load configs ---
    model_config = args.model_config
    training_config = args.training_config

    # Training configuration
    conf = load_config(training_config, "parameters")
    print("Training Config:", conf)

    training_data_config = conf["training_data_config"]
    validation_data_config = conf["validation_data_config"]

    assert os.path.exists(training_data_config), f"Data config not found: {training_data_config}"
    assert os.path.exists(validation_data_config), f"Data config not found: {validation_data_config}"
    assert os.path.exists(model_config), f"Model config not found: {model_config}"
    assert os.path.exists(training_config), f"Training config not found: {training_config}"

    # Load tokenizer
    tokenizer = load_tokenizer(training_data_config)

    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    name, model_type, net = load_language_model(model_config, device)


    # Model summary
    x = torch.zeros((1, conf["sequence_length"]), dtype=torch.long).to(device)
    summary(net, input_data=x)

    # Train Dataset + dataloader
    train_dataset = load_dataset(training_data_config, tokenizer, conf["sequence_length"], net.get_output_dimension())
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
    )

    # Train Dataset + dataloader
    val_dataset = load_dataset(validation_data_config, tokenizer, conf["sequence_length"], net.get_output_dimension())
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
    )


    # Optimizer
    optim = Adam(net.parameters(), lr=conf["lr"])

    # WandB init
    wandb.init(
        project=conf["experiment_name"],
        config={
            "batch_size": conf["batch_size"],
            "sequence_length": conf["sequence_length"],
            "lr": conf["lr"],
            "num_epochs": conf["epochs"],
            "model": name,
        },
    )
    wandb.watch(net, log="all", log_freq=100)

    # Save tokenizer
    tokenizer.save()


    # Pretrain
    pretrain(
        net,
        train_dataloader,
        val_dataloader,
        optim,
        device,
        conf["epochs"],
        conf["val_steps"],
        conf["checkpoint_frequency"],
        conf["experiment_name"],
        name,
    )


if __name__ == "__main__":
    main()
