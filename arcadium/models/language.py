import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from arcadium.models.output import LMOutput


class LanguageModelConfig(PretrainedConfig):
    model_type = "arcadium_lm"

    def __init__(
        self,
        vocab_size: int = 50257,
        architecture_type: str = "dense",
        # Decoder params
        decoder_layers: int = 8,
        input_dimension: int = 64,
        output_dimension: int = 64,
        hidden_dimension: int = 128,
        norm_eps: float = 1e-5,
        # Attention params
        attention_type: str = "multi-head",
        num_query_heads: int = 8,
        num_kv_heads: int = 8,
        embedding_dimension: int = 64,
        head_dimension: int = 64,
        # MoE-specific
        experts: int = 8,
        top_k: int = 2,
        # Context window — used by lm-eval and HF ecosystem to bound sequence length
        max_position_embeddings: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.architecture_type = architecture_type
        self.decoder_layers = decoder_layers
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_dimension = hidden_dimension
        self.norm_eps = norm_eps
        self.attention_type = attention_type
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.experts = experts
        self.top_k = top_k
        self.max_position_embeddings = max_position_embeddings


def _build_transformer(config: LanguageModelConfig):
    from arcadium.models.dense import (
        DenseTransformer, DenseTransformerParams, DenseModelParams, DenseDecoderParams,
    )
    from arcadium.models.moe import (
        MoETransformer, MoETransformerParams, MoEModelParams, MoEDecoderParams,
    )
    from arcadium.components.attentions import AttentionParameters

    attn_params = AttentionParameters(
        type=config.attention_type,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        embedding_dimension=config.embedding_dimension,
        head_dimension=config.head_dimension,
    )

    if config.architecture_type == "dense":
        params = DenseTransformerParams(
            model=DenseModelParams(decoder_layers=config.decoder_layers),
            decoder=DenseDecoderParams(
                input_dimension=config.input_dimension,
                output_dimension=config.output_dimension,
                hidden_dimension=config.hidden_dimension,
                norm_eps=config.norm_eps,
                attention=attn_params,
            ),
        )
        return DenseTransformer(params)

    if config.architecture_type == "moe":
        params = MoETransformerParams(
            model=MoEModelParams(decoder_layers=config.decoder_layers),
            decoder=MoEDecoderParams(
                input_dimension=config.input_dimension,
                output_dimension=config.output_dimension,
                hidden_dimension=config.hidden_dimension,
                experts=config.experts,
                top_k=config.top_k,
                norm_eps=config.norm_eps,
                attention=attn_params,
            ),
        )
        return MoETransformer(params)

    raise ValueError(f"Unknown architecture_type: {config.architecture_type!r}")


class LanguageModel(PreTrainedModel):
    config_class = LanguageModelConfig

    def __init__(self, config: LanguageModelConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.input_dimension)
        self.transformer = _build_transformer(config)
        self.lm_head = nn.Linear(config.output_dimension, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    @property
    def vocab_size(self):
        return self.config.vocab_size

    def forward(self, input_ids, labels=None, **kwargs):
        assert input_ids.max().item() < self.config.vocab_size, (
            f"Token ID {input_ids.max().item()} exceeds vocab size {self.config.vocab_size}"
        )
        assert input_ids.min().item() >= 0, (
            f"Token ID {input_ids.min().item()} is negative"
        )

        x = self.embed(input_ids)
        h, aux_loss = self.transformer(x, **kwargs)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
            )

        return LMOutput(loss=loss, logits=logits, auxiliary_loss=aux_loss)

    def get_output_dimension(self):
        return self.vocab_size

    def metadata(self):
        return self.transformer.metadata()
