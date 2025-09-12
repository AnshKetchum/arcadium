from typing import Iterator, Callable, Optional
import torch
from torch.utils.data import Sampler
import random

# -------- SCHEDULE FUNCTIONS --------
def linear_schedule(start: int, end: int, cur: int, num_iters: int) -> int:
    """Linear interpolation between start and end."""
    progress = min(cur / num_iters, 1.0)
    return int(start + (end - start) * progress)

# -------- SCHEDULER --------
class SequenceLengthScheduler:
    def __init__(
        self,
        start: int,
        end: int,
        num_iters: int,
        schedule_fn: Optional[Callable[[int, int, int, int], int]] = None,
        debug: bool = False,
        name: str = "scheduler",
    ):
        assert end > start, f"End value {end} must be greater than start {start}"

        self.start = start
        self.end = end
        self.num_iters = num_iters
        self.schedule_fn = schedule_fn or linear_schedule

        self.timer = 0
        self.sequence_length = start
        self.debug = debug
        self.name = name

    def next_seqlen(self) -> int:
        if self.timer >= self.num_iters:
            return self.end

        self.timer += 1
        prev_seqlen = self.sequence_length
        self.sequence_length = min(
            self.schedule_fn(self.start, self.end, self.timer, self.num_iters),
            self.end,
        )

        if self.debug:
            print(
                f"{self.name} SEQUENCE LENGTH {prev_seqlen} -> {self.sequence_length} "
                f"(iter={self.timer}/{self.num_iters})"
            )
        return self.sequence_length

    def get_seqlen(self) -> int:
        if self.debug:
            print(
                f"{self.name} returning SEQLEN {self.sequence_length} "
                f"(iter={self.timer})"
            )
        return int(self.sequence_length)


# -------- SAMPLER --------
class SequenceLengthSampler(Sampler):
    """Sampler yielding (index, seq_len) pairs, with lazy index generation."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        start: int,
        end: int,
        num_iters: int,
        name: str = "scheduler",
        debug: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.scheduler = SequenceLengthScheduler(
            start, end, num_iters, debug=debug, name=name
        )
        self.shuffle = shuffle
        self.seed = seed or torch.initial_seed()

    def __iter__(self) -> Iterator:
        g = torch.Generator()
        g.manual_seed(self.seed)

        # Instead of materializing all indices, sample them batch by batch
        if self.shuffle:
            # Use torch.randint to generate random indices in chunks
            for batch_start in range(0, self.dataset_size, self.batch_size):
                batch_size = min(self.batch_size, self.dataset_size - batch_start)
                batch_indices = torch.randint(
                    low=0, high=self.dataset_size, size=(batch_size,), generator=g
                ).tolist()
                seq_len = self.scheduler.next_seqlen()
                for idx in batch_indices:
                    yield (idx, seq_len)
        else:
            # Sequential indices (lazy range)
            for batch_start in range(0, self.dataset_size, self.batch_size):
                batch_indices = range(batch_start, min(batch_start + self.batch_size, self.dataset_size))
                seq_len = self.scheduler.next_seqlen()
                for idx in batch_indices:
                    yield (idx, seq_len)

    def __len__(self):
        return self.dataset_size
