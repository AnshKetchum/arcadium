import os
from transformers import AutoTokenizer
from models.tasks.language.tokenizers.base import BaseTokenizer

class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, model_name: str = "gpt2"):
        # Load a pretrained tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            # Ensure pad token exists for some models
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Initialized HuggingFaceTokenizer with vocab size {self.size()}")

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs (with BOS/EOS).
        """
        bos = self.get_beginning_of_sequence_token()
        eos = self.get_end_of_sequence_token()
        
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return [bos] + encoded + [eos]

    def decode_single(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], clean_up_tokenization_spaces=True)

    def get_beginning_of_sequence_token(self) -> int:
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        # fallback: use CLS if available, else pad
        return self.tokenizer.cls_token_id or self.tokenizer.pad_token_id

    def get_end_of_sequence_token(self) -> int:
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        # fallback: use SEP if available, else pad
        return self.tokenizer.sep_token_id or self.tokenizer.pad_token_id

    @classmethod
    def get_tokens(cls, corpus: str):
        # Temporary tokenizer for splitting only
        tmp_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return tmp_tokenizer.tokenize(corpus)

    def ingest(self, additional_corpus):
        """
        HuggingFace tokenizers are pretrained, so this is a no-op.
        """
        print("Ingest skipped: HuggingFaceTokenizer uses a fixed vocabulary.")

    def save(self, directory: str):
        """
        Save tokenizer to a directory.
        """
        assert directory is not None
        os.makedirs(directory, exist_ok=True)

        path = os.path.join(directory, "gpt-2")
        
        self.tokenizer.save_pretrained(path)
        print(f"Saved HuggingFaceTokenizer to {path}")

    def load(self, directory: str):
        """
        Load tokenizer from a directory.
        """
        path = os.path.join(directory, "gpt-2")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"Loaded HuggingFaceTokenizer from {path}")

    def size(self) -> int:
        return self.tokenizer.vocab_size

    def summary(self):
        print("Tokenizer summary:")
        print(f"  Vocabulary size: {self.size()}")
        print(f"  BOS token id: {self.get_beginning_of_sequence_token()}")
        print(f"  EOS token id: {self.get_end_of_sequence_token()}")
