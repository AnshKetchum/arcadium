import glob
import os
from collections import deque

import torch
from torch.utils.data import Dataset, get_worker_info


class TextDataset(Dataset):
    """
    Token-packing dataset over local parquet / JSONL files.

    Documents are tokenized on the fly, concatenated with EOS, and packed into
    non-overlapping windows of `sequence_length` tokens. The token buffer refills
    lazily from a streaming HuggingFace IterableDataset, so memory usage is O(buffer)
    regardless of corpus size.

    Lazy init: the dataset iterator is created on the first __getitem__ call, after
    DataLoader forks worker processes. Each worker gets an independent iterator with
    a per-worker shuffle seed so workers sample different document orderings.

    Args:
        path            : directory containing parquet / jsonl files, or a single file.
        text_key        : field name holding the document text.
        tokenizer       : HuggingFace PreTrainedTokenizer.
        sequence_length : number of tokens per training sample.
        format          : "parquet" | "jsonl" | "json"  (default: "parquet").
        val             : if True, use a fixed seed so val order is deterministic.
        debug           : print progress messages.
    """

    def __init__(
        self,
        path: str,
        text_key: str,
        tokenizer,
        sequence_length: int,
        format: str = "parquet",
        val: bool = False,
        debug: bool = False,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.format = format
        self.val = val
        self.debug = debug
        self._eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        ext_map = {"parquet": "*.parquet", "jsonl": "*.jsonl", "json": "*.json"}
        if os.path.isdir(path):
            pattern = os.path.join(path, "**", ext_map.get(format, f"*.{format}"))
            self._files = sorted(glob.glob(pattern, recursive=True))
        else:
            self._files = [path]

        if not self._files:
            raise ValueError(f"[TextDataset] No {format} files found in {path!r}")

        # HuggingFace load_dataset uses "json" for both .json and .jsonl
        self._hf_format = "json" if format in ("jsonl", "json") else format

        # Deferred until after DataLoader worker fork
        self._doc_iter = None
        self._token_buffer: deque = deque()

        if debug:
            print(f"[TextDataset] {len(self._files)} {format} file(s) at {path!r}, text_key={text_key!r}")

    # ------------------------------------------------------------------
    # Streaming iterator (lazy, per-worker)
    # ------------------------------------------------------------------

    def _build_iter(self):
        from datasets import load_dataset as hf_load_dataset

        worker = get_worker_info()
        seed = (worker.id if worker is not None else 0) + (42 if self.val else 1337)

        ds = hf_load_dataset(
            self._hf_format,
            data_files=self._files,
            split="train",
            streaming=True,
        ).shuffle(seed=seed, buffer_size=10_000)

        return iter(ds)

    def _get_iter(self):
        if self._doc_iter is None:
            self._doc_iter = self._build_iter()
        return self._doc_iter

    # ------------------------------------------------------------------
    # Token buffer management
    # ------------------------------------------------------------------

    def _fill_buffer(self, min_tokens: int):
        while len(self._token_buffer) < min_tokens:
            try:
                doc = next(self._get_iter())
            except StopIteration:
                self._doc_iter = self._build_iter()
                try:
                    doc = next(self._doc_iter)
                except StopIteration:
                    break
            text = (doc.get(self.text_key) or "").strip()
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            self._token_buffer.extend(tokens)
            self._token_buffer.append(self._eos_id)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return 2**31

    def __getitem__(self, idx):
        needed = self.sequence_length + 1
        self._fill_buffer(needed)

        available = min(needed, len(self._token_buffer))
        chunk = [self._token_buffer.popleft() for _ in range(available)]
        while len(chunk) < needed:
            chunk.append(self._eos_id)

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
