import torch 
from torch.utils.data import Dataset
from models.tasks.language.tokenizers.base import BaseTokenizer
from .config import INF
import os 

NUM_TOKENS = 128
class RandomLanguageModelDataset(Dataset):

    def __len__(self):
        return 100000
    
    def __getitem__(self, index):
        return torch.zeros(NUM_TOKENS, dtype=torch.int), torch.zeros(NUM_TOKENS, dtype=torch.long)

