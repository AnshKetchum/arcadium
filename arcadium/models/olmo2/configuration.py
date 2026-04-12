from transformers import PretrainedConfig


class OLMo2Config(PretrainedConfig):
    model_type = "arcadium_olmo2"

    def __init__(
        self,
        n_blocks: int = 16,
        d_model: int = 2048,
        n_heads: int = 16,
        vocab_size: int = 50280,
        max_sequence_length: int = 4096,
        mlp_expansion_factor: float = 4.0,
        rope_base: int = 10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.mlp_expansion_factor = mlp_expansion_factor
        self.rope_base = rope_base


# standard size presets (approximate OLMo2 paper configs)
def olmo2_1b() -> OLMo2Config:
    return OLMo2Config(n_blocks=16, d_model=2048, n_heads=16)

def olmo2_7b() -> OLMo2Config:
    return OLMo2Config(n_blocks=32, d_model=4096, n_heads=32)

def olmo2_13b() -> OLMo2Config:
    return OLMo2Config(n_blocks=40, d_model=5120, n_heads=40)
