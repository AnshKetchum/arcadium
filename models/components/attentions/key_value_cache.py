import torch 
import torch.nn as nn 

from pydantic import BaseModel

class KVCacheParams(BaseModel):
    max_batch_size:int = 48
    max_sequence_length:int = 1024
    num_key_heads:int = 8
    num_value_heads:int = 8
    key_dimension:int = 512
    value_dimension:int = 512
    enabled:bool = False

class KVCache:
    """
        A hopefully simple interface 
    """
    def __init__(self, config: KVCacheParams):
        self.max_batch_size = config.max_batch_size
        self.max_sequence_length = config.max_sequence_length
        self.key_dimension = config.key_dimension
        self.num_key_heads = config.num_key_heads
        self.num_value_heads = config.num_value_heads
        self.value_dimension = config.value_dimension
        self.enabled = config.enabled
        self.device = "cuda"
        self.empty = True

        if self.enabled:
            self.key_cache = torch.zeros(self.max_batch_size, self.num_key_heads, self.max_sequence_length, self.key_dimension, device=self.device)
            self.value_cache = torch.zeros(self.max_batch_size, self.num_value_heads, self.max_sequence_length, self.value_dimension, device=self.device)
    
    def cache(self, keys, values, batch_idx_start, batch_idx_end, sequence_length_idx_start, sequence_length_idx_end, num_heads_start, num_heads_end):
        if self.enabled:
            self.key_cache[batch_idx_start:batch_idx_end, num_heads_start: num_heads_end, sequence_length_idx_start : sequence_length_idx_end] = keys
            self.value_cache[batch_idx_start:batch_idx_end, num_heads_start: num_heads_end, sequence_length_idx_start : sequence_length_idx_end] = values
            self.empty = False
        return True

    def get(self, batch_idx_start, batch_idx_end, sequence_length_idx_start, sequence_length_idx_end, num_heads_start, num_heads_end):
        if self.enabled:
            if self.empty:
                return None, None

            k = self.key_cache[batch_idx_start:batch_idx_end, num_heads_start: num_heads_end, sequence_length_idx_start : sequence_length_idx_end]
            v = self.value_cache[batch_idx_start:batch_idx_end, num_heads_start: num_heads_end, sequence_length_idx_start : sequence_length_idx_end]
            return k, v
        else:
            return None, None
    
    def reset(self):
        if self.enabled:
            self.key_cache = torch.zeros(self.max_batch_size, self.num_key_heads, self.max_sequence_length, self.key_dimension, device=self.device)
            self.value_cache = torch.zeros(self.max_batch_size, self.num_value_heads, self.max_sequence_length, self.value_dimension, device=self.device)
            self.empty = True
    
def load_kv_cache(kv_cache_params: dict):
    params = KVCacheParams(**kv_cache_params)
    kv_cache = KVCache(params)
    return kv_cache