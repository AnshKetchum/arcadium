import torch
import os
from typing import Tuple
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from models.tasks.language.architecture import LanguageModel, LanguageModelConfig

# Datasets
from models.tasks.language.datasets.sequence_length import SequenceLengthScheduler
from models.tasks.language.datasets.single_folder import DocumentLanguageModelDatasetFromShardsRandomSampling
from models.tasks.language.datasets.fineweb import FineWebBinaryDataset
from models.tasks.language.datasets.multi_dataset import AggregatedRoundRobinDataset

from utils import load_config


def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    """
    Load a HuggingFace tokenizer from a pretrained model name (e.g. "gpt2") or a
    local directory saved with tokenizer.save_pretrained().
    """
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


def _config_from_yaml(conf_dict: dict, vocab_size: int) -> LanguageModelConfig:
    """Build a LanguageModelConfig from the model YAML configuration dict."""
    architecture_type = conf_dict["type"]
    cfg = conf_dict["configuration"]
    decoder = cfg["decoder"]
    attn = decoder["attention"]

    kwargs = dict(
        vocab_size=vocab_size,
        architecture_type=architecture_type,
        decoder_layers=cfg["model"]["decoder_layers"],
        input_dimension=decoder["input_dimension"],
        output_dimension=decoder["output_dimension"],
        hidden_dimension=decoder["hidden_dimension"],
        norm_eps=decoder.get("norm_eps", 1e-5),
        attention_type=attn["type"],
        num_query_heads=attn["num_query_heads"],
        num_kv_heads=attn["num_kv_heads"],
        embedding_dimension=attn["embedding_dimension"],
        head_dimension=attn["head_dimension"],
    )

    if architecture_type == "moe":
        kwargs["experts"] = decoder.get("experts", 8)
        kwargs["top_k"] = decoder.get("top_k", 2)

    return LanguageModelConfig(**kwargs)


def load_language_model(
    configuration_path: str,
    device: torch.device,
) -> Tuple[str, str, LanguageModel, PreTrainedTokenizerBase]:
    """
    Load a model and its tokenizer from a YAML config.
    Returns (name, architecture_type, model, tokenizer).
    """
    conf_dict = load_config(configuration_path, "parameters")

    tokenizer_config = load_config(conf_dict["tokenizer"]["config"], "parameters")
    tokenizer = load_tokenizer(**tokenizer_config)

    conf_name = conf_dict["name"]
    conf_type = conf_dict["type"]
    print("Configuring ...", conf_name)

    hf_config = _config_from_yaml(conf_dict, vocab_size=len(tokenizer))
    net = LanguageModel(hf_config).to(device)

    return conf_name, conf_type, net, tokenizer


def load_language_model_from_pretrained(
    pretrained_dir: str,
    device: torch.device,
) -> Tuple[LanguageModel, PreTrainedTokenizerBase]:
    """
    Load a model and tokenizer from a directory saved with save_pretrained().
    Tokenizer files are expected to live directly in pretrained_dir.
    """
    net = LanguageModel.from_pretrained(pretrained_dir).to(device)
    tokenizer = load_tokenizer(pretrained_dir)
    return net, tokenizer
