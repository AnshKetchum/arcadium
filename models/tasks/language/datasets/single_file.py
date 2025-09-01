import torch 
import os 
from torch.utils.data import Dataset
from models.tasks.language.tokenizer import BaseTokenizer

INF = int(1e4)

class DocumentLanguageModelDatasetFromFileRandomSampling(Dataset):
    
    def __init__(self, filepath: str, tokenizer, sequence_length: int, model_vocab_size: int):
        super().__init__()
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.model_vocab_size = model_vocab_size

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
        # Make dataset length proportional to possible windows
        return INF

    def __getitem__(self, index: int):
        # Ignore index, just sample randomly
        start_idx = torch.randint(0, self.max_start, (1,)).item()
        seq = self.tokens[start_idx : start_idx + self.sequence_length]
        target_id = self.tokens[start_idx + self.sequence_length]

        # Build one-hot target
        one_hot = torch.zeros(self.model_vocab_size, dtype=torch.long)
        one_hot[target_id] = 1

        return torch.tensor(seq, dtype=torch.long), one_hot
