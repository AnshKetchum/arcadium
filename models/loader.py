from omegaconf import OmegaConf
import torch 
import torch.nn as nn
import os 
from typing import Tuple

# mixture of experts
from models.moe import MoETransformerParams, MoETransformer
from models.moe import MoETransformer
from models.tasks.language.architecture import LanguageModel
from models.tasks.language.tokenizers.base import BaseTokenizer, BasicTokenizer
from models.tasks.language.tokenizers.huggingface import HuggingFaceTokenizer

# Datasets
from models.tasks.language.datasets.sequence_length import SequenceLengthScheduler
from models.tasks.language.datasets.single_folder import DocumentLanguageModelDatasetFromShardsRandomSampling
from models.tasks.language.datasets.multi_dataset import AggregatedRoundRobinDataset

from utils import load_config

def ingest_file(fl: str, tok: BasicTokenizer):
    assert os.path.exists(fl), f"Path {fl} no longer exists"

    with open(fl, "r") as f:
        data = f.read()
        tokens = BasicTokenizer.get_tokens(data)

    tok.ingest(tokens)

def load_tokenizer(tokenizer_path: str, tokenizer_type: str = "basic", **kwargs):
    BASIC_TOKENIZER = "basic"
    HUGGINGFACE_TOKENIZER = "huggingface"

    if tokenizer_type == BASIC_TOKENIZER:
        tok = BasicTokenizer()
    elif tokenizer_type == HUGGINGFACE_TOKENIZER:
        tok = HuggingFaceTokenizer(kwargs["model_name"])
    
    if tokenizer_path and os.path.exists(tokenizer_path):
        tok.load(tokenizer_path)
    
    return tok

def load_dataset(data_config: str, sequence_length: int, debug = False): 

    conf = load_config(data_config, "parameters")

    datasets = []

    for fldr in conf.get("folders", []):
        datasets.append(DocumentLanguageModelDatasetFromShardsRandomSampling(
            fldr, 
            sequence_length,
            debug=debug
        ))

    return AggregatedRoundRobinDataset(datasets)

def load_language_model(configuration_path: str, device: torch.device) -> Tuple[str, str, LanguageModel, BaseTokenizer]:
    # Load the entire model configuration
    conf_dict = load_config(configuration_path, "parameters")

    ## Loading the tokenizer 
    tokenizer_config = load_config(conf_dict['tokenizer']['config'], "parameters")
    tokenizer = load_tokenizer(**tokenizer_config)
    
    ## Loading the model

    # Extract name
    conf_name = conf_dict["name"]
    print("Configuring ... ", conf_name)

    # Extract type
    conf_type = conf_dict["type"]

    # Extract Configuration 
    conf = MoETransformerParams(**conf_dict["configuration"])

    if conf_type == "moe":
        net = LanguageModel(MoETransformer, conf, tokenizer.size())
        net = net.to(device)
    if conf_type == "dense":
        raise NotImplementedError()

    return conf_name, conf_type, net, tokenizer