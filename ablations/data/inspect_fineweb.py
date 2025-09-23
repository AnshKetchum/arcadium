# inspect_fineweb.py
import os
import numpy as np
from transformers import GPT2TokenizerFast

CANDIDATES = [
    ("<u2", "uint16 LE"),
    (">u2", "uint16 BE"),
    ("<u4", "uint32 LE"),
    (">u4", "uint32 BE"),
    ("<i4", "int32 LE"),
    (">i4", "int32 BE"),
]

def analyze_file(path: str, tokenizer, sample_tokens: int = 2048):
    print(f"Inspecting {path}")
    filesize = os.path.getsize(path)
    print(f"  File size: {filesize:,} bytes")

    best = None
    diagnostics = []

    for dtype_str, name in CANDIDATES:
        dt = np.dtype(dtype_str)
        itemsize = dt.itemsize
        try:
            # compute number of elements if interpreted with this dtype
            num_elems = filesize // itemsize
            if num_elems == 0:
                continue

            # map a small sample (don't map whole file)
            take = min(num_elems, sample_tokens)
            arr = np.memmap(path, dtype=dt, mode="r")[:take].astype(np.int64, copy=False)

            # fraction inside vocabulary
            inside = (arr >= 0) & (arr < tokenizer.vocab_size)
            frac_inside = float(np.sum(inside)) / len(arr)

            # some small heuristics: mean, max (to spot huge values)
            mean_val = int(np.mean(arr))
            max_val = int(arr.max())
            min_val = int(arr.min())

            diagnostics.append((name, dtype_str, num_elems, frac_inside, min_val, max_val, itemsize))
        except Exception as e:
            diagnostics.append((name, dtype_str, None, 0.0, None, None, None))

    # print diagnostics sorted by frac_inside desc
    diagnostics.sort(key=lambda x: x[3], reverse=True)
    print("  Candidate diagnostics (sorted by fraction inside GPT-2 vocab):")
    for d in diagnostics:
        name, dtype_str, num_elems, frac_inside, lo, hi, itemsize = d
        print(f"    {name:12} dtype={dtype_str:4} elems={num_elems} itemsize={itemsize} frac_in_vocab={frac_inside:.4f} min={lo} max={hi}")

    # pick best candidate: first with frac_inside >= 0.98 or highest
    picked = diagnostics[0]
    name, dtype_str, num_elems, frac_inside, lo, hi, itemsize = picked
    print()
    print(f"  -> Suggested dtype: {name} (dtype='{dtype_str}'), fraction in vocab={frac_inside:.4f}")
    if frac_inside < 0.50:
        print("     WARNING: low fraction in vocab — file might be corrupt or use an exotic encoding.")
    # show a decoded sample if candidate has decent fraction
    if frac_inside > 0.02:
        dt = np.dtype(dtype_str)
        arr_full = np.memmap(path, dtype=dt, mode="r")
        # pick a sample of valid tokens for decoding
        valid_mask = (arr_full >= 0) & (arr_full < tokenizer.vocab_size)
        # find first valid chunk of length sample_tokens
        idxs = np.where(valid_mask)[0]
        if len(idxs) == 0:
            print("  No valid tokens found to decode.")
            return
        start = idxs[0]
        end = min(start + sample_tokens, len(arr_full))
        sample = arr_full[start:end].astype(np.int64).tolist()
        text = tokenizer.decode(sample)
        print("\n  Sample decoded text (first valid block):\n")
        print("  " + text[:1000].replace("\n", " ") + "...\n")
    print("-" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data/fineweb10B", help="folder with .bin shards")
    parser.add_argument("--file", default=None, help="inspect a single file")
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    if args.file:
        analyze_file(args.file, tokenizer)
    else:
        folder = args.folder
        files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".bin")])
        if not files:
            raise SystemExit(f"No .bin files in {folder}")
        # Inspect first few files (or all if you prefer)
        for f in files[:6]:
            analyze_file(f, tokenizer)
