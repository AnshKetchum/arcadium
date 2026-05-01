import numpy as np
from torch.utils.data import Dataset


class WeightedMixDataset(Dataset):
    """
    Samples from multiple datasets according to normalized weights.

    At each __getitem__ call a dataset is selected by categorical sampling
    proportional to its weight, then one item is drawn from that dataset.
    Weights need not sum to 1; they are normalized internally.

    Returns (x, y, source_idx) where source_idx is the integer index of the
    dataset that produced this sample — used by the training loop to track
    per-source token counts.

    Args:
        datasets_and_weights : list of (dataset, weight) tuples.
                               Weight values may be any positive float.
        names                : optional list of human-readable source names,
                               one per dataset. Defaults to "0", "1", ...
    """

    def __init__(self, datasets_and_weights: list, names: list[str] | None = None):
        assert datasets_and_weights, "At least one (dataset, weight) pair required"
        self._datasets, weights = zip(*datasets_and_weights)
        weights = list(weights)
        total = sum(weights)
        assert total > 0, "Weights must sum to a positive value"

        # Cumulative probabilities for O(k) categorical sampling
        self._cum_probs: list[float] = []
        cumulative = 0.0
        for w in weights:
            cumulative += w / total
            self._cum_probs.append(cumulative)
        # Clamp last bucket to exactly 1.0 to absorb float rounding
        self._cum_probs[-1] = 1.0

        self._names: list[str] = (
            list(names) if names is not None
            else [str(i) for i in range(len(self._datasets))]
        )

    def __len__(self):
        return 2**31

    def __getitem__(self, idx):
        r = np.random.random()
        for i, cp in enumerate(self._cum_probs):
            if r < cp:
                x, y = self._datasets[i][idx]
                return x, y, i
        x, y = self._datasets[-1][idx]
        return x, y, len(self._datasets) - 1
