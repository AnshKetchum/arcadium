
import os
import torch
import matplotlib.pyplot as plt

def plot_visualizations(net: torch.nn.Module, ckpt_dir: str, iter_num: int):
    """
    Save attention maps from each Attention layer in `net` into the checkpoint folder.

    Args:
        net (torch.nn.Module): The model containing Attention layers with metadata().
        ckpt_dir (str): The checkpoint directory where visualizations will be saved.
        iter_num (int): Current training iteration (used for folder naming).
    """
    # Create a folder for this iteration’s visualizations
    vis_dir = os.path.join(ckpt_dir, f"attn_maps_iter{iter_num}")
    os.makedirs(vis_dir, exist_ok=True)

    for i, metadata in enumerate(net.metadata()):
        if not metadata or "attention_probabilities" not in metadata:
            continue

        attn_probs = metadata["attention_probabilities"]  # [B, N, T, T]
        B, N, T, _ = attn_probs.shape

        # For visualization: pick first batch element
        probs = attn_probs[0].detach().cpu()  # [N, T, T]

        # Save one plot per head
        for head in range(N):
            plt.figure(figsize=(6, 5))
            plt.imshow(probs[head], cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.title(f"Layer {i} - Head {head}")
            plt.xlabel("Key positions")
            plt.ylabel("Query positions")

            out_path = os.path.join(vis_dir, f"layer{i}_head{head}.png")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()

    print(f"[iter {iter_num}] Saved attention visualizations to {vis_dir}")
