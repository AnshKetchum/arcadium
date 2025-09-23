# Seeing an awful number of 0 tokens. what???
# Results of this script - ~13% are BOS tokens
# I guess that's the non-ideality we'll have to deal with.
import numpy as np
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
path = "data/fineweb10B/fineweb_train_000002.bin"
arr = np.memmap(path, dtype="<u2", mode="r")

total = len(arr)
print("total tokens:", total)

# bincount over full uint16 range (65536) is reasonable in memory
counts = np.bincount(arr, minlength=65536)

zero_count = int(counts[0])
print("id=0 count:", zero_count, f"({zero_count/total:.4%})")

# show top-20 most common token ids and their small decoded repr
topk = 100
top_ids = counts.argsort()[::-1][:topk]
for tid in top_ids:
    c = counts[tid]
    text = tokenizer.decode([int(tid)])
    # shorten long decodes for readability
    short = text.replace("\n", " ")
    if len(short) > 40: short = short[:37] + "..."
    print(f"id={tid:5d} count={c:10d} frac={c/total:.4%} token_repr={repr(short)}")
