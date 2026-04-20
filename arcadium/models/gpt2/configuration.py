from transformers import PretrainedConfig


class GPT2Config(PretrainedConfig):
    model_type = "arcadium_gpt2"

    def __init__(
        self,
        n_blocks: int = 12,
        d_model: int = 768,
        n_heads: int = 12,
        vocab_size: int = 50257,
        max_sequence_length: int = 1024,
        mlp_expansion_factor: float = 4.0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.mlp_expansion_factor = mlp_expansion_factor


# standard size presets
def gpt2_small() -> GPT2Config:
    return GPT2Config(n_blocks=12, d_model=768, n_heads=12)

def gpt2_medium() -> GPT2Config:
    return GPT2Config(n_blocks=24, d_model=1024, n_heads=16)

def gpt2_large() -> GPT2Config:
    return GPT2Config(n_blocks=36, d_model=1280, n_heads=20)

def gpt2_xl() -> GPT2Config:
    return GPT2Config(n_blocks=48, d_model=1600, n_heads=25)
