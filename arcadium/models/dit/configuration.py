from transformers import PretrainedConfig


class DiTConfig(PretrainedConfig):
    model_type = "arcadium_dit"

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma


# Standard size presets — each returns a DiTConfig, patch_size can be overridden via kwargs
def DiT_XL_2(**kwargs) -> DiTConfig:
    return DiTConfig(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs) -> DiTConfig:
    return DiTConfig(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs) -> DiTConfig:
    return DiTConfig(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs) -> DiTConfig:
    return DiTConfig(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs) -> DiTConfig:
    return DiTConfig(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs) -> DiTConfig:
    return DiTConfig(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs) -> DiTConfig:
    return DiTConfig(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs) -> DiTConfig:
    return DiTConfig(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs) -> DiTConfig:
    return DiTConfig(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs) -> DiTConfig:
    return DiTConfig(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs) -> DiTConfig:
    return DiTConfig(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs) -> DiTConfig:
    return DiTConfig(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)