from omegaconf import OmegaConf
import torch 
import torch.nn as nn
import os 
from typing import Tuple

# mixture of experts
from models.moe import MoETransformerParams, MoETransformer
from models.moe import MoETransformer
from models.tasks.language.model import LanguageModel
from models.tasks.language.tokenizer import BasicTokenizer

# Datasets
from models.tasks.language.datasets.single_file import DocumentLanguageModelDatasetFromFileRandomSampling
from models.tasks.language.datasets.single_folder import DocumentLanguageModelDatasetFromFolderRandomSampling
from models.tasks.language.datasets.multi_dataset import AggregatedRoundRobinDataset

from utils import load_config

def ingest_file(fl: str, tok: BasicTokenizer):
    assert os.path.exists(fl), f"Path {fl} no longer exists"

    with open(fl, "r") as f:
        data = f.read()
        tokens = BasicTokenizer.get_tokens(data)

    tok.ingest(tokens)

def load_tokenizer(data_config: str):

    conf = load_config(data_config, "parameters")

    tok = BasicTokenizer()

    for fl in conf.get("files", []):
        ingest_file(fl, tok)
    
    for fldr in conf.get("folders", []):
        for fl in os.listdir(fldr):
            ingest_file(os.path.join(fldr, fl), tok)
        
    return tok

def load_prexisting_tokenizer(tokenizer_path: str):

    assert os.path.exists(tokenizer_path) and os.path.isfile(tokenizer_path)
    tok = BasicTokenizer()
    tok.load(tokenizer_path)
    return tok


def load_dataset(data_config: str, tokenizer: BasicTokenizer, sequence_length: int, model_vocab_size: int): 

    conf = load_config(data_config, "parameters")

    datasets = []

    for fl in conf.get("files",  []):
        datasets.append(DocumentLanguageModelDatasetFromFileRandomSampling(
            fl, 
            tokenizer, 
            sequence_length,
        ))

    for fldr in conf.get("folders", []):
        datasets.append(DocumentLanguageModelDatasetFromFolderRandomSampling(
            fldr, 
            tokenizer, 
            sequence_length,
        ))

    return AggregatedRoundRobinDataset(datasets)

def load_language_model(configuration_path: str, device: torch.device) -> Tuple[str, str, LanguageModel]:
    # Load the entire model configuration
    yaml_conf = OmegaConf.load(configuration_path)

    # Grab the configuration as a dict
    conf_params = OmegaConf.to_container(yaml_conf, resolve=True)
    conf_dict = conf_params["parameters"]

    # Extract name
    conf_name = conf_dict["name"]
    print("Configuring ... ", conf_name)

    # Extract type
    conf_type = conf_dict["type"]

    # Extract Configuration 
    conf = MoETransformerParams(**conf_dict["configuration"])

    if conf_type == "moe":
        vocab_size = conf_params["language_model"]["vocab_size"]
        net = LanguageModel(MoETransformer, conf, vocab_size)
        net = net.to(device)
    if conf_type == "dense":
        raise NotImplementedError()

    return conf_name, conf_type, net