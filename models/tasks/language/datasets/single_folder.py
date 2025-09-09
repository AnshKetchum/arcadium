import torch
import os
from torch.utils.data import Dataset
from models.tasks.language.tokenizer import BaseTokenizer

INF = int(1e4)

class DocumentLanguageModelDatasetFromFolderRandomSampling(Dataset):
    """
    Dataset for a folder of text files. Samples sequences randomly in a round-robin manner across files.
    """
    def __init__(self, folderpath: str, tokenizer: BaseTokenizer, sequence_length: int):
        super().__init__()
        self.folderpath = folderpath
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        # Verify folder exists
        assert os.path.isdir(folderpath), f"{folderpath} is not a valid directory"
        
        # List all files in folder
        self.files = [
            os.path.join(folderpath, f) 
            for f in os.listdir(folderpath) 
            if os.path.isfile(os.path.join(folderpath, f))
        ]
        assert len(self.files) > 0, "No files found in folder"

        # Load and tokenize each file
        self.token_lists = []
        for filepath in self.files:
            with open(filepath, "r") as f:
                tokens = (
                    [tokenizer.get_beginning_of_sequence_token()] +
                    tokenizer.encode(f.read()) +
                    [tokenizer.get_end_of_sequence_token()]
                )
                if len(tokens) > sequence_length:  # Only keep files long enough
                    self.token_lists.append(tokens)

        assert len(self.token_lists) > 0, "No file is long enough for the given sequence length"

        # Keep track of max start index per file
        self.max_starts = [len(t) - sequence_length - 1 for t in self.token_lists]

        # Round-robin pointer
        self.file_idx = 0

        print(f"Loaded {len(self.token_lists)} files for random sampling with sequence length {sequence_length}")

    def __len__(self):
        # Effectively infinite dataset
        return 2**31  

    def __getitem__(self, index: int):
        # Round-robin file selection
        tokens = self.token_lists[self.file_idx]
        max_start = self.max_starts[self.file_idx]

        start_idx = torch.randint(0, max_start, (1,)).item()
        seq = tokens[start_idx : start_idx + self.sequence_length + 1]  # include next token

        # Input is all tokens except last
        input_seq = seq[:-1]        # [seq_len]
        target_seq = seq[1:]        # [seq_len], next-token labels

        # Update file index for next sample (round-robin)
        self.file_idx = (self.file_idx + 1) % len(self.token_lists)

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

