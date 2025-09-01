import os
import pickle
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
    def get_tokens(cls, corpus: str):
        pass

    def ingest(self, additional_corpus):
        pass 
    
    def save(self, ):
        pass

    def load(self, file_path: str):
        pass

class BasicTokenizer(BaseTokenizer):
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, corpus: list[str] = None):

        self.mapping = {}
        self.reverse_mapping= []

        self.manual_ingest(self.BOS_TOKEN)
        self.manual_ingest(self.EOS_TOKEN)
        
        if corpus:
            self.mapping = { c:i for i, c in enumerate(corpus) }
            self.reverse_mapping = corpus
            print("Initialized tokenizer with", len(corpus), "tokens")
        else: 
            print("Default initialization")
    
    @classmethod
    def get_tokens(cls, corpus: str):
        return corpus.split()
    
    def ingest(self, additional_corpus: list[str]):
        for c in additional_corpus:
            self.manual_ingest(c)
    
    def manual_ingest(self, tok: str):
        if tok not in self.mapping:
            self.mapping[tok] = len(self.reverse_mapping)
            self.reverse_mapping.append(tok)
    
    def encode(self, text: str) -> list[int]:
        tokens = [self.mapping[w] for w in BasicTokenizer.get_tokens(text)]
        return tokens
    
    def decode_single(self, token_id: int) -> str:
        return self.reverse_mapping[token_id]

    def get_beginning_of_sequence_token(self)-> int:
        return self.encode(self.BOS_TOKEN)[0]
        
    def get_end_of_sequence_token(self) -> int:
        return self.encode(self.EOS_TOKEN)[0]

    def save(self, path: str = None):

        save_location = ""
        if path:
            assert os.path.exists(path)
            assert os.path.isdir(path)
            save_location = path
    
        data = {
            "mapping" : self.mapping,
            "reverse_mapping" : self.reverse_mapping
        }

        with open(os.path.join(save_location, "tokenizer.pkl"), "wb") as f:
            pickle.dump(data, f)
         
        print("Successfully saved", len(self.mapping), "tokens.")

    def load(self, path: str = None):

        load_location = ""
        if path:
            assert os.path.exists(path)
            assert os.path.isfile(path)
    
        with open(path, "rb") as f:
            data = pickle.load(f)

            self.mapping = data["mapping"]
            self.reverse_mapping = data["reverse_mapping"]
         
        print("Successfully loaded", len(self.mapping), "tokens.")