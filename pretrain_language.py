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

load_dotenv()

def loss_function(inputs, labels):
    return F.cross_entropy(inputs, labels)

def training_step(net: nn.Module, batch: torch.Tensor, labels: torch.Tensor):
    logits = net(batch)
    loss = loss_function(logits, labels)
    return logits, loss

def pretrain(
    net: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    num_iters=10,
    checkpoint_frequency: int = 5,
    experiment_name=None,
    model_name: str = None
):
    # Checkpoint folder
    ckpt = os.path.join("checkpoints", f"{experiment_name}-{time.strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs(ckpt, exist_ok=True)

    net = net.to(device)
    net.train()
    data_iterator = iter(data_loader)
    
    # --- Activation hook setup ---
    activation_stats = defaultdict(list)
    hooks = register_activation_hooks(net, activation_stats)

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
        activation_stats.clear()  # reset after each forward to prevent blowup

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

        # --- Log metrics to wandb ---
        wandb.log({
            "lm_loss": loss.item(),
            "iters_per_sec": iters_per_sec,
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
        if i % checkpoint_frequency == 0:
            print(
                f"[iter {i}] Loss {loss.item():.4f}, iters/sec {iters_per_sec:.2f}, "
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
                json.dump({"total_activation_VRAM_MB": total_activation_mem}, f)
            print(f"[iter {i}] Saved activation VRAM JSON to {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Pretrain a language model.")

    # Config files
    parser.add_argument("--data_config", type=str, default="configs/data/simple-corpus.yaml",
                        help="Path to dataset config YAML")
    parser.add_argument("--model_config", type=str, default="configs/models/tiny-moe.yaml",
                        help="Path to model config YAML")
    parser.add_argument("--training_config", type=str, default="configs/training/basic.yaml",
                        help="Path to training config YAML")

    args = parser.parse_args()

    # --- Load configs ---
    data_config = args.data_config
    model_config = args.model_config
    training_config = args.training_config

    assert os.path.exists(data_config), f"Data config not found: {data_config}"
    assert os.path.exists(model_config), f"Model config not found: {model_config}"
    assert os.path.exists(training_config), f"Training config not found: {training_config}"

    # Load tokenizer
    tokenizer = load_tokenizer(data_config)

    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    name, model_type, net = load_language_model(model_config, device)

    # Training configuration
    conf = load_config(training_config, "parameters")
    print("Training Config:", conf)

    # Model summary
    x = torch.zeros((1, conf["sequence_length"]), dtype=torch.long).to(device)
    summary(net, input_data=x)

    # Dataset + dataloader
    dataset = load_dataset(data_config, tokenizer, conf["sequence_length"], net.get_output_dimension())
    dataloader = DataLoader(
        dataset,
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
        dataloader,
        optim,
        device,
        conf["epochs"],
        conf["checkpoint_frequency"],
        conf["experiment_name"],
        name,
    )


if __name__ == "__main__":
    main()
