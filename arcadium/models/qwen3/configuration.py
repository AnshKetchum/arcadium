from transformers import PretrainedConfig


class Qwen3Config(PretrainedConfig):
    model_type = "arcadium_qwen3"

    def __init__(
        self,
        n_blocks: int = 16,
        d_model: int = 2048,
        n_query_heads: int = 16,
        n_kv_heads: int = 16,
        vocab_size: int = 50280,
        max_sequence_length: int = 4096,
        mlp_expansion_factor: float = 4.0,
        sliding_window_ratio: float = 3.0,
        rope_base: int = 10000,
        yarn_scale: float = 1.0,
        yarn_original_max_len: int = 4096,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=False, **kwargs)
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.mlp_expansion_factor = mlp_expansion_factor
        self.sliding_window_ratio = sliding_window_ratio
        self.rope_base = rope_base
        self.yarn_scale = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len


# standard size presets (approximate OLMo2 paper configs)
def Qwen3_1b() -> Qwen3Config:
    return Qwen3Config(n_blocks=16, d_model=2048, n_heads=16)

def Qwen3_7b() -> Qwen3Config:
    return Qwen3Config(n_blocks=32, d_model=4096, n_heads=32)

def Qwen3_13b() -> Qwen3Config:
    return Qwen3Config(n_blocks=40, d_model=5120, n_heads=40)
