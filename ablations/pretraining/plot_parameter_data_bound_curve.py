#!/usr/bin/env python3
import os
import json
import argparse
import matplotlib.pyplot as plt
import glob
import re

def plot_roofline_like(checkpoints_dir, save_path=None):
    plt.figure(figsize=(7, 5))

    # assign a unique color per run
    subfolders = [f for f in sorted(os.listdir(checkpoints_dir)) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    cmap = plt.get_cmap("tab10")  # 10 distinct colors, cycles after that

    for idx, folder in enumerate(subfolders):
        subdir = os.path.join(checkpoints_dir, folder)

        # Find all metrics files
        pattern = os.path.join(subdir, "moe-basic-1m-64-emb-metrics-iter*.json")
        files = sorted(glob.glob(pattern))

        x_vals, y_vals, labels = [], [], []
        for fpath in files:
            match = re.search(r"iter(\d+)\.json", fpath)
            if not match:
                continue
            iter_num = int(match.group(1))

            with open(fpath, "r") as f:
                data = json.load(f)

            trained_tokens = data["trained_tokens"]
            param_count = data["param_count"]
            loss = data["loss"]

            ratio = trained_tokens / param_count
            loss_norm = loss / iter_num if iter_num > 0 else loss

            x_vals.append(ratio)
            y_vals.append(loss_norm)
            labels.append(f"iter{iter_num}")

        if x_vals:
            color = cmap(idx % 10)
            plt.scatter(x_vals, y_vals, c=[color], s=60, label=folder)
            for x, y, label in zip(x_vals, y_vals, labels):
                plt.text(x, y, label, fontsize=7, ha="right", color=color)

    plt.xlabel("Tokens / Parameters")
    plt.ylabel("Loss / Iter")
    plt.title("Roofline-like Plot Across All Iters")
    plt.grid(True)
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot roofline-like graph from MoE checkpoints")
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Path to checkpoints directory")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the plot instead of showing it")
    args = parser.parse_args()

    plot_roofline_like(args.checkpoints_dir, args.save)

if __name__ == "__main__":
    main()
