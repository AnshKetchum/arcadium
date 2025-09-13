#!/usr/bin/env python3
import os
import json
import argparse
import matplotlib.pyplot as plt
import glob
import re
from collections import defaultdict

def plot_loss_vs_params(checkpoints_dir, save_path=None):
    plt.figure(figsize=(7, 5))

    subfolders = [f for f in sorted(os.listdir(checkpoints_dir)) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    cmap = plt.get_cmap("tab10")  # distinct colors, cycles every 10

    # Collect data grouped by iteration number across runs
    iter_data = defaultdict(list)  # {iter_num: [(param_count, loss, folder)]}

    for idx, folder in enumerate(subfolders):
        subdir = os.path.join(checkpoints_dir, folder)

        pattern = os.path.join(subdir, "moe-basic-1m-64-emb-1*-metrics-iter*.json")
        files = sorted(glob.glob(pattern))

        for fpath in files:
            match = re.search(r"iter(\d+)\.json", fpath)
            if not match:
                continue
            iter_num = int(match.group(1))

            with open(fpath, "r") as f:
                data = json.load(f)

            param_count = data["param_count"]
            loss = data["loss"]

            iter_data[iter_num].append((param_count, loss, folder))

    # Plot per iteration group
    for idx, (iter_num, points) in enumerate(sorted(iter_data.items())):
        color = cmap(idx % 10)

        # sort points by param_count for cleaner lines
        points_sorted = sorted(points, key=lambda x: x[0])
        x_vals = [p[0] for p in points_sorted]
        y_vals = [p[1] for p in points_sorted]

        # connect points of the same iter across runs
        plt.plot(x_vals, y_vals, marker="o", linestyle="-", color=color, label=f"iter {iter_num}")


    plt.xlabel("Parameters")
    plt.ylabel("Loss")
    plt.title("Loss vs Parameters Across Iterations")
    plt.grid(True)
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Loss vs Parameters from MoE checkpoints")
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Path to checkpoints directory")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the plot instead of showing it")
    args = parser.parse_args()

    plot_loss_vs_params(args.checkpoints_dir, args.save)


if __name__ == "__main__":
    main()
