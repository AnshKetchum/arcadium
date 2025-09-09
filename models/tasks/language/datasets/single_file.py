import torch 
import os 
from torch.utils.data import Dataset
from models.tasks.language.tokenizer import BaseTokenizer

INF = int(1e4)

class DocumentLanguageModelDatasetFromFileRandomSampling(Dataset):
    
    def __init__(self, filepath: str, tokenizer, sequence_length: int):
        super().__init__()
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        # Verify that the filepath exists
        assert os.path.exists(filepath)

        with open(self.filepath, "r") as f:
            self.tokens = (
                [self.tokenizer.get_beginning_of_sequence_token()] +
                self.tokenizer.encode(f.read()) +
                [self.tokenizer.get_end_of_sequence_token()]
            )

        print("Total tokens", len(self.tokens))

        # We can draw start indices up to len(tokens) - sequence_length - 1
        self.max_start = len(self.tokens) - sequence_length - 1
        assert self.max_start > 0, "Text is too short for given sequence length"

    def __len__(self):
        # Infinite-style dataset (random sampling)
        return 2**31  # instead of INF, keep it practical

    def __getitem__(self, index: int):
        # Ignore index, just sample randomly
        start_idx = torch.randint(0, self.max_start, (1,)).item()
        
        # Grab sequence including next-token
        seq = self.tokens[start_idx : start_idx + self.sequence_length + 1]  # length = seq_len + 1

        input_seq = seq[:-1]    # [seq_len]
        target_seq = seq[1:]    # [seq_len], next-token labels

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

