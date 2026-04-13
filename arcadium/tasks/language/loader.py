import torch
from typing import Tuple
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from arcadium.models.language import LanguageModel, LanguageModelConfig

# Datasets
from arcadium.data.sequence_length import SequenceLengthScheduler
from arcadium.data.single_folder import DocumentLanguageModelDatasetFromShardsRandomSampling
from arcadium.data.fineweb import FineWebBinaryDataset
from arcadium.data.multi_dataset import AggregatedRoundRobinDataset

from arcadium.utils import load_config


def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_dataset(data_config: str, sequence_length: int, debug=False, **kwargs):
    conf = load_config(data_config, "parameters")
    LOCAL = "local"
    FINEWEB = "fineweb"

    datasets = []

    if conf["type"] == LOCAL:
        for fldr in conf.get("folders", []):
            datasets.append(DocumentLanguageModelDatasetFromShardsRandomSampling(
                fldr, sequence_length, debug=debug, **kwargs
            ))
    elif conf["type"] == FINEWEB:
        for fldr in conf.get("folders", []):
            datasets.append(FineWebBinaryDataset(
                fldr, sequence_length, debug=debug, **kwargs
            ))

    return AggregatedRoundRobinDataset(datasets)


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
