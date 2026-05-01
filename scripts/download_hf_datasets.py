#!/usr/bin/env python3
"""
Download a HuggingFace dataset to a local directory for arcadium training.

Usage
-----
# Download an entire repo (all parquet shards)
python scripts/download_hf_datasets.py HuggingFaceFW/fineweb-edu data/fineweb-edu

# Download only parquet files matching a glob pattern
python scripts/download_hf_datasets.py HuggingFaceFW/fineweb-edu data/fineweb-edu \
    --patterns "data/CC-MAIN-2024-10/*.parquet"

# Download a JSONL dataset
python scripts/download_hf_datasets.py bigcode/the-stack data/the-stack \
    --patterns "data/python/*.jsonl"

# Authenticate (required for gated repos)
huggingface-cli login
"""

import argparse
import os
import sys


def download(repo_id: str, local_dir: str, patterns: list[str] | None = None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is not installed. Run: pip install huggingface-hub")

    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading {repo_id} → {local_dir}")
    if patterns:
        print(f"  patterns: {patterns}")

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type="dataset",
        allow_patterns=patterns or None,
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )
    print(f"Done. Files saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace dataset for arcadium training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("repo_id", help="HuggingFace repo id (e.g. HuggingFaceFW/fineweb-edu)")
    parser.add_argument("local_dir", help="Local directory to save files into")
    parser.add_argument(
        "--patterns",
        nargs="*",
        metavar="GLOB",
        help="Glob patterns to restrict which files are downloaded "
             "(e.g. 'data/*.parquet'). Omit to download everything.",
    )
    args = parser.parse_args()
    download(args.repo_id, args.local_dir, args.patterns)


if __name__ == "__main__":
    main()
