# inspect_force_u16.py
import os
import numpy as np
from transformers import GPT2TokenizerFast

def inspect_one(path, num_tokens=1024):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    dt = np.dtype("<u2")  # little-endian uint16
    arr = np.memmap(path, dtype=dt, mode="r")
    print(f"File: {path}")
    print(f"  File size: {os.path.getsize(path):,} bytes")
    print(f"  Interpreting as '<u2' -> elements: {len(arr):,}")
    print(f"  Min ID: {int(arr.min())}, Max ID: {int(arr.max())}")

    # keep only values inside vocab to produce clean decode sample
    valid = arr[arr < tokenizer.vocab_size]
    if len(valid) == 0:
        print("  No tokens < vocab_size found")
        return
    sample = valid[:num_tokens].astype(np.int64).tolist()
    text = tokenizer.decode(sample)
    print("\n  Sample decode (first valid block):\n")
    print("  " + text[:1000].replace("\n", " ") + "...\n")

if __name__ == "__main__":
    folder = "data/fineweb10B"     # adjust to your path
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".bin")])
    for f in files[:3]:
        inspect_one(f)
