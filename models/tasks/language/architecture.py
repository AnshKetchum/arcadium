import torch.nn as nn 


class LanguageModel(nn.Module):
    def __init__(self, transformer_cls, config, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, config.decoder.input_dimension)
        self.transformer = transformer_cls(config)
        self.lm_head = nn.Linear(config.decoder.output_dimension, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, x, **kwargs):
        # x: [batch, seq_len] integer tokens

        # Sanity checks to verify that the input going into the model is what we expect
        assert x.max().item() < self.vocab_size, f"Token ID {x.max().item()} exceeds vocab size {self.vocab_size}"
        assert x.min().item() >= 0, f"Token ID {x.min().item()} is negative"

        x = self.embed(x)  # [batch, seq_len, hidden_dim]
        h = self.transformer(x, **kwargs)  # [batch, seq_len, hidden_dim_out]
        logits = self.lm_head(h)  # [batch, seq_len, vocab_size]
        return logits

    def get_output_dimension(self):
        return self.vocab_size
    
    def metadata(self):
        return self.transformer.metadata()