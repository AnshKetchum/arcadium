import torch
import os
import pickle
from torch.utils.data import Dataset
from models.tasks.language.datasets.sequence_length import SequenceLengthScheduler

class DocumentLanguageModelDatasetFromShardsRandomSampling(Dataset):
    """
    Dataset that loads pre-tokenized shards (pickled) and samples sequences randomly
    in a round-robin manner across shards, with dynamic sequence lengths.
    """
    def __init__(self, shard_folder: str, sequence_length: int, debug = False):
        super().__init__()
        self.shard_folder = shard_folder
        self.sequence_length = sequence_length
        self.debug = debug

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
                self.token_lists.append(tokens)

        assert len(self.token_lists) > 0, "No documents found in shards"

        # Round-robin pointer
        self.doc_idx = 0

        print(f"Loaded {len(self.token_lists)} documents from {len(self.shards)} shards "
              f"for dynamic sequence length sampling")

    def __len__(self):
        # Effectively infinite dataset
        return 2**31  

    def __getitem__(self, item):
        # Round-robin document selection
        if isinstance(item, tuple):
            index, seq_len = item
        else:
            seq_len = self.sequence_length

        tokens = self.token_lists[self.doc_idx]

        if self.debug:
            print("Dataset receives", seq_len)
            
        max_start = max(len(tokens) - seq_len - 1, 0)
        if max_start == 0:
            start_idx = 0
        else:
            start_idx = torch.randint(0, max_start, (1,)).item()

        seq = tokens[start_idx : start_idx + seq_len + 1]  # include next token

        input_seq = seq[:-1]   # [seq_len]
        target_seq = seq[1:]   # [seq_len], next-token labels

        # Update doc index for next sample (round-robin)
        self.doc_idx = (self.doc_idx + 1) % len(self.token_lists)

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
