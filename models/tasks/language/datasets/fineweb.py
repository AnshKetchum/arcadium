import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FineWebBinaryDataset(Dataset):
    """
    Random-sampling FineWeb dataset reading .bin shards (uint16 LE).
    - idx can be an int or a tuple (ignored_index, seq_len) — seq_len overrides default.
    - Lazily memory-maps shards (no full-file loads).
    - Resamples until a sequence with all token ids < vocab_size is found.
    - __len__ is effectively infinite (2**31).
    """

    def __init__(
        self,
        shard_folder: str,
        sequence_length: int,
        val: bool = False,
        vocab_size: int = 50257,
        debug: bool = False,
        max_attempts: int | None = None,  # None means keep trying forever
    ):
        super().__init__()
        self.shard_folder = shard_folder
        self.sequence_length = sequence_length
        self.val = val
        self.vocab_size = vocab_size
        self.debug = debug
        self.max_attempts = max_attempts

        # find shards
        all_files = sorted(os.listdir(shard_folder))
        if val:
            self.shard_paths = [os.path.join(shard_folder, f) for f in all_files if f.startswith("fineweb_val")]
        else:
            self.shard_paths = [os.path.join(shard_folder, f) for f in all_files if f.startswith("fineweb_train")]

        if len(self.shard_paths) == 0:
            raise ValueError(f"No shards found in {shard_folder} for val={val}")

        # compute token counts for each shard (uint16 -> 2 bytes each)
        self.shard_lengths = [os.path.getsize(p) // 2 for p in self.shard_paths]

        # lazy memmap cache
        self._memmaps = [None] * len(self.shard_paths)
        self._dtype = np.dtype("<u2")  # explicit little-endian uint16

        if self.debug:
            total_tokens = sum(self.shard_lengths)
            print(f"[FineWebBinaryRandomDataset] found {len(self.shard_paths)} shards, total tokens={total_tokens:,}")

    def __len__(self):
        # make it appear infinite for streaming training
        return 2**31

    def _get_memmap(self, shard_idx: int):
        mm = self._memmaps[shard_idx]
        if mm is None:
            path = self.shard_paths[shard_idx]
            # mode='r' ensures OS-level mmap; dtype is little-endian uint16
            mm = np.memmap(path, dtype=self._dtype, mode="r")
            self._memmaps[shard_idx] = mm
        return mm

    def __getitem__(self, idx):
        """
        idx may be:
          - int: ignored, sample with default sequence_length
          - (any, seq_len): use seq_len instead of default
        The method samples randomly and resamples until an in-vocab sequence is found.
        """
        # get seq_len override from tuple-index if present
        if isinstance(idx, tuple):
            # accept forms like (i, seq_len) or (something, seq_len)
            if len(idx) >= 2:
                _, seq_len = idx[0], idx[1]
            else:
                seq_len = self.sequence_length
        else:
            seq_len = self.sequence_length

        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        attempts = 0
        while True:
            attempts += 1
            # optional stop condition
            if (self.max_attempts is not None) and (attempts > self.max_attempts):
                # fallback: return a sequence but clamp/replace OOV tokens to eos (vocab_size-1)
                if self.debug:
                    print(f"[WARN] max_attempts reached ({self.max_attempts}); returning clamped sequence")
                return self._sample_and_clamp(seq_len)

            # pick a random shard
            shard_idx = np.random.randint(0, len(self.shard_paths))
            n_tokens = self.shard_lengths[shard_idx]
            mm = self._get_memmap(shard_idx)

            # choose start
            if n_tokens <= seq_len:
                start = 0
            else:
                # inclusive low, exclusive high
                start = np.random.randint(0, n_tokens - seq_len - 1)

            end = start + seq_len + 1  # include next-token label
            seq_np = mm[start:end]

            # if short (near file end) pad with zeros
            if seq_np.shape[0] < seq_len + 1:
                pad_amt = (seq_len + 1) - seq_np.shape[0]
                seq_np = np.pad(seq_np, (0, pad_amt), constant_values=0)

            # check validity
            if int(seq_np.max()) < self.vocab_size:
                # convert to tensors and return
                x = torch.from_numpy(seq_np[:-1].astype(np.int64, copy=False)).long()
                y = torch.from_numpy(seq_np[1:].astype(np.int64, copy=False)).long()
                if self.debug:
                    print(f"[OK] shard={shard_idx} start={start} seq_len={seq_len} attempts={attempts}")
                return x, y

            # else skip and try again
            if self.debug:
                print(f"[SKIP] shard={shard_idx} start={start} seq_len={seq_len} max_token={int(seq_np.max())} attempts={attempts}")

    def _sample_and_clamp(self, seq_len: int):
        """
        Fallback sampling: pick a random shard + range and clamp any OOV tokens to vocab_size-1.
        Returns (x, y) tensors.
        """
        shard_idx = np.random.randint(0, len(self.shard_paths))
        n_tokens = self.shard_lengths[shard_idx]
        mm = self._get_memmap(shard_idx)

        if n_tokens <= seq_len:
            start = 0
        else:
            start = np.random.randint(0, n_tokens - seq_len - 1)
        seq_np = mm[start:start + seq_len + 1]
        if seq_np.shape[0] < seq_len + 1:
            seq_np = np.pad(seq_np, (0, (seq_len + 1) - seq_np.shape[0]), constant_values=0)

        # clamp OOV tokens to vocab_size-1
        seq_np = seq_np.astype(np.int64, copy=True)
        seq_np[seq_np >= self.vocab_size] = self.vocab_size - 1

        x = torch.from_numpy(seq_np[:-1]).long()
        y = torch.from_numpy(seq_np[1:]).long()
        if self.debug:
            print(f"[CLAMP] shard={shard_idx} start={start} seq_len={seq_len}")
        return x, y
