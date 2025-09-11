import torch
import os
import pickle
from torch.utils.data import Dataset


class DocumentLanguageModelDatasetFromShardsRandomSampling(Dataset):
    """
    Dataset that loads pre-tokenized shards (pickled) and samples sequences randomly
    in a round-robin manner across shards.
    """
    def __init__(self, shard_folder: str, sequence_length: int):
        super().__init__()
        self.shard_folder = shard_folder
        self.sequence_length = sequence_length

        # Verify folder exists
        assert os.path.isdir(shard_folder), f"{shard_folder} is not a valid directory"
        
        # List all shard files
        self.shards = [
            os.path.join(shard_folder, f) 
            for f in os.listdir(shard_folder) 
            if os.path.isfile(os.path.join(shard_folder, f)) and f.endswith(".pkl")
        ]
        assert len(self.shards) > 0, "No shard files found in folder"

        # Load all shards into memory
        self.token_lists = []
        for shard_path in self.shards:
            with open(shard_path, "rb") as f:
                shard_data = pickle.load(f)
            for doc in shard_data["documents"]:
                tokens = doc["tokens"]
                if len(tokens) > sequence_length:
                    self.token_lists.append(tokens)

        assert len(self.token_lists) > 0, "No document is long enough for the given sequence length"

        # Keep track of max start index per document
        self.max_starts = [len(t) - sequence_length - 1 for t in self.token_lists]

        # Round-robin pointer
        self.doc_idx = 0

        print(f"Loaded {len(self.token_lists)} documents from {len(self.shards)} shards "
              f"for random sampling with sequence length {sequence_length}")

    def __len__(self):
        # Effectively infinite dataset
        return 2**31  

    def __getitem__(self, index: int):
        # Round-robin document selection
        tokens = self.token_lists[self.doc_idx]
        max_start = self.max_starts[self.doc_idx]

        start_idx = torch.randint(0, max_start, (1,)).item()
        seq = tokens[start_idx : start_idx + self.sequence_length + 1]  # include next token

        # Input is all tokens except last
        input_seq = seq[:-1]        # [seq_len]
        target_seq = seq[1:]        # [seq_len], next-token labels

        # Update doc index for next sample (round-robin)
        self.doc_idx = (self.doc_idx + 1) % len(self.token_lists)

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
