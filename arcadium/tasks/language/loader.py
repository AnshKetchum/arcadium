import torch
from typing import Tuple
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from arcadium.models.language import LanguageModel, LanguageModelConfig

# Datasets
from arcadium.data.text_dataset import TextDataset
from arcadium.data.weighted_mix import WeightedMixDataset

from arcadium.utils import load_config


def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# Per-type defaults for format and text_key.
_SOURCE_DEFAULTS = {
    "fineweb": {"format": "parquet", "text_key": "text"},
    "local":   {"format": "jsonl",   "text_key": "text"},
    "hf":      {"format": "parquet", "text_key": "text"},
    "text":    {"format": "parquet", "text_key": "text"},
}


def _build_single_source(src: dict, sequence_length: int, tokenizer, debug: bool, **kwargs):
    if tokenizer is None:
        raise ValueError("tokenizer is required — all datasets use on-the-fly tokenization.")
    src_type = src.get("type", "text")
    defaults = _SOURCE_DEFAULTS.get(src_type, {"format": "parquet", "text_key": "text"})
    return TextDataset(
        path=src["path"],
        text_key=src.get("text_key", defaults["text_key"]),
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        format=src.get("format", defaults["format"]),
        val=kwargs.get("val", False),
        debug=debug,
    )


def load_dataset(data_config: str, sequence_length: int, debug=False, tokenizer=None, **kwargs):
    """
    Build a dataset from a YAML config.

    All configs use the unified mix format:

        parameters:
          type: "mix"
          sources:
            - name: "fineweb"
              type: "fineweb"      # "fineweb" | "local" | "hf"
              path: "data/fineweb10B"
              weight: 1.0

            - name: "code"
              type: "hf"           # HuggingFace parquet / JSONL, downloaded locally
              path: "data/the-stack"
              format: "parquet"    # "parquet" | "jsonl"
              text_key: "content"
              weight: 0.3

    A single-source config is just a mix with one entry — weights are normalized,
    so weight: 1.0 on a lone source is a no-op.

    tokenizer is required (all sources tokenize on the fly).
    """
    conf = load_config(data_config, "parameters")
    assert conf.get("type") == "mix", (
        f"Data config {data_config!r} must have type: 'mix'. "
        "All configs now use the unified sources format."
    )

    sources = conf["sources"]
    datasets_and_weights = []
    for src in sources:
        ds = _build_single_source(src, sequence_length, tokenizer, debug, **kwargs)
        weight = float(src.get("weight", 1.0))
        datasets_and_weights.append((ds, weight))
        if debug:
            print(f"[load_dataset] source {src.get('name', src['type'])!r} weight={weight}")
    return WeightedMixDataset(datasets_and_weights)


def _build_model(arch: dict, vocab_size: int) -> PreTrainedModel:
    architecture = arch["architecture"]
    cfg = {k: v for k, v in arch.items() if k != "architecture"}
    cfg["vocab_size"] = vocab_size

    if architecture == "gpt2":
        from arcadium.models.gpt2 import GPT2, GPT2Config
        return GPT2(GPT2Config(**cfg))

    if architecture == "olmo2":
        from arcadium.models.olmo2 import OLMo2, OLMo2Config
        return OLMo2(OLMo2Config(**cfg))

    if architecture == "universal_transformer":
        from arcadium.models.universal_transformer import UniversalTransformer, UniversalTransformerConfig
        return UniversalTransformer(UniversalTransformerConfig(**cfg))

    if architecture == "qwen3":
        from arcadium.models.qwen3 import Qwen3, Qwen3Config
        return Qwen3(Qwen3Config(**cfg))

    if architecture in ("dense", "moe"):
        from arcadium.components.attentions import AttentionParameters
        config = LanguageModelConfig(
            vocab_size=vocab_size,
            architecture_type=architecture,
            decoder_layers=cfg["decoder_layers"],
            input_dimension=cfg["input_dimension"],
            output_dimension=cfg["output_dimension"],
            hidden_dimension=cfg["hidden_dimension"],
            norm_eps=cfg.get("norm_eps", 1e-5),
            attention_type=cfg["attention_type"],
            num_query_heads=cfg["num_query_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            embedding_dimension=cfg["embedding_dimension"],
            head_dimension=cfg["head_dimension"],
            experts=cfg.get("experts", 8),
            top_k=cfg.get("top_k", 2),
            max_position_embeddings=cfg.get("max_position_embeddings", 1024),
        )
        return LanguageModel(config)

    raise ValueError(f"Unknown architecture: {architecture!r}")


def load_language_model(
    configuration_path: str,
    device: torch.device,
) -> Tuple[str, str, PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a model and its tokenizer from a YAML config.
    Returns (name, architecture, model, tokenizer).
    """
    conf = load_config(configuration_path, "parameters")

    tokenizer_config = load_config(conf["tokenizer"]["config"], "parameters")
    tokenizer = load_tokenizer(**tokenizer_config)

    arch = load_config(conf["architecture_config"])
    print("Configuring ...", conf["name"])

    net = _build_model(arch, vocab_size=len(tokenizer)).to(device)
    return conf["name"], arch["architecture"], net, tokenizer


def load_language_model_from_pretrained(
    pretrained_dir: str,
    device: torch.device,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a model and tokenizer from a directory saved with save_pretrained().
    """
    net = LanguageModel.from_pretrained(pretrained_dir).to(device)
    tokenizer = load_tokenizer(pretrained_dir)
    return net, tokenizer
