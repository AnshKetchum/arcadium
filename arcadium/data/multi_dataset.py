import torch
from torch.utils.data import Dataset
from .config import INF

class AggregatedRoundRobinDataset(Dataset):
    """
    Aggregates multiple PyTorch datasets and samples from them in round-robin fashion.
    """
    def __init__(self, datasets):
        """
        datasets: list of Dataset objects
        """
        super().__init__()
        assert len(datasets) > 0, "At least one dataset must be provided"
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.dataset_idx = 0  # Round-robin pointer

    def __len__(self):
        # Return an arbitrarily large number for continuous sampling
        return INF

    def __getitem__(self, index):
        # Round-robin sampling
        current_dataset = self.datasets[self.dataset_idx]
        sample = current_dataset[index]  # Index is ignored for random sampling datasets

        # Move to next dataset
        self.dataset_idx = (self.dataset_idx + 1) % self.num_datasets
        return sample
