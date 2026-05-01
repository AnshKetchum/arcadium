import numpy as np
from torch.utils.data import Dataset


class WeightedMixDataset(Dataset):
    """
    Samples from multiple datasets according to normalized weights.

    At each __getitem__ call a dataset is selected by categorical sampling
    proportional to its weight, then one item is drawn from that dataset.
    Weights need not sum to 1; they are normalized internally.

    Args:
        datasets_and_weights : list of (dataset, weight) tuples.
                               Weight values may be any positive float.
    """

    def __init__(self, datasets_and_weights: list):
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

        if len(self._datasets) == 1:
            names = [getattr(d, '__class__', type(d)).__name__ for d in self._datasets]
        else:
            names = [getattr(d, '__class__', type(d)).__name__ for d in self._datasets]
        self._names = names

    def __len__(self):
        return 2**31

    def __getitem__(self, idx):
        r = np.random.random()
        for i, cp in enumerate(self._cum_probs):
            if r < cp:
                return self._datasets[i][idx]
        return self._datasets[-1][idx]
