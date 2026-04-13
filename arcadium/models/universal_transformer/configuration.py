from transformers import PretrainedConfig


class UniversalTransformerConfig(PretrainedConfig):
    model_type = "arcadium_universal_transformer"

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        max_steps: int = 8,
        n_blocks: int = 1,
        eps: float = 0.01,
        tau: float = 0.01,
        dropout: float = 0.1,
        max_sequence_length: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_steps = max_steps
        self.n_blocks = n_blocks
        self.eps = eps
        self.tau = tau
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
