import os
import pickle
import re
from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str):
        pass 
    
    @abstractmethod
    def decode_single(self, token_id: int):
        pass

    @abstractmethod
    def get_beginning_of_sequence_token(self) -> int:
        pass

    @abstractmethod
    def get_end_of_sequence_token(self) -> int:
        pass
    
    @classmethod 
    @abstractmethod
    def get_tokens(cls, corpus: str):
        pass

    @abstractmethod
    def ingest(self, additional_corpus):
        pass 
    
    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, file_path: str):
        pass


class BasicTokenizer(BaseTokenizer):
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, corpus: list[str] = None):
        self.mapping = {}
        self.reverse_mapping = []

        # Always ingest BOS/EOS first so they have reserved ids
        self.manual_ingest(self.BOS_TOKEN)
        self.manual_ingest(self.EOS_TOKEN)
        
        if corpus:
            for c in corpus:
                self.manual_ingest(c)
            print("Initialized tokenizer with", len(self.mapping), "tokens")
        else: 
            print("Default initialization")
    
    @classmethod
    def get_tokens(cls, corpus: str):
        """
        Split on whitespace and punctuation, keeping punctuation as separate tokens.
        """
        # This regex will separate words and punctuation into distinct tokens
        tokens = re.findall(r"\w+|[^\w\s]", corpus, re.UNICODE)
        return tokens
    
    def ingest(self, additional_corpus: list[str]):
        for c in additional_corpus:
            self.manual_ingest(c)
    
    def manual_ingest(self, tok: str):
        if tok not in self.mapping:
            self.mapping[tok] = len(self.reverse_mapping)
            self.reverse_mapping.append(tok)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs, inserting BOS and EOS around sentences.
        """
        raw_tokens = BasicTokenizer.get_tokens(text)

        encoded_tokens = []
        encoded_tokens.append(self.mapping[self.BOS_TOKEN])  # Start with BOS

        for tok in raw_tokens:
            if tok not in self.mapping:
                self.manual_ingest(tok)
            encoded_tokens.append(self.mapping[tok])

            if tok == ".":  # End of sentence
                encoded_tokens.append(self.mapping[self.EOS_TOKEN])
                encoded_tokens.append(self.mapping[self.BOS_TOKEN])

        # If text didn’t end in period, still append EOS
        if encoded_tokens[-1] != self.mapping[self.EOS_TOKEN]:
            encoded_tokens.append(self.mapping[self.EOS_TOKEN])

        return encoded_tokens
    
    def decode_single(self, token_id: int) -> str:
        return self.reverse_mapping[token_id]

    def get_beginning_of_sequence_token(self) -> int:
        return self.mapping[self.BOS_TOKEN]
        
    def get_end_of_sequence_token(self) -> int:
        return self.mapping[self.EOS_TOKEN]
    
    def size(self):
        assert len(self.mapping) == len(self.reverse_mapping), (
            f"Mapping has a size of {len(self.mapping)} "
            f"while reverse mapping has a size of {len(self.reverse_mapping)}"
        )
        return len(self.mapping)

    def save(self, path: str):
        """
        Save tokenizer mappings to file.
        """
        assert path is not None, "Path must be provided for saving"
        data = {
            "mapping": self.mapping,
            "reverse_mapping": self.reverse_mapping
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print("Successfully saved", len(self.mapping), "tokens to", path)

    def load(self, file_path: str):
        """
        Load tokenizer mappings from file.
        """
        assert file_path is not None and os.path.isfile(file_path), "Invalid file path"
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        self.mapping = data["mapping"]
        self.reverse_mapping = data["reverse_mapping"]
        print("Successfully loaded", len(self.mapping), "tokens from", file_path)

    def summary(self):
        vocab_size = len(self.mapping)
        print(f"Tokenizer summary:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Token IDs range from 0 to {vocab_size - 1}")

