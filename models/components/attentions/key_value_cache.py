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
    def __init__(self, config: KVCacheParams, **kwargs):
        self.max_batch_size = config.max_batch_size
        self.max_sequence_length = config.max_sequence_length
        self.key_dimension = config.key_dimension
        self.num_key_heads = config.num_key_heads
        self.num_value_heads = config.num_value_heads
        self.value_dimension = config.value_dimension
        self.enabled = config.enabled
        self.device = "cuda"
        self.empty = True

        assert "recompute_function" in kwargs, "Args need to specify recompute function for attention"
        self.recompute_function = kwargs.get("recompute_function")

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
        
    def get_or_recompute(self, x: torch.Tensor):
        B, T, _ = x.shape

        k, v = None, None
        k_prev, v_prev = self.get(
            0,
            B, 
            0, 
            T,
            0, self.num_key_heads
        )

        if k_prev is not None and v_prev is not None:
            k_cur, v_cur = self.recompute_function(x[:, -1, :])
            k = torch.cat([k_prev, k_cur], dim=1)
            v = torch.cat([v_prev, v_cur], dim=1)
            self.cache(k_cur, v_cur, 0,B, T, T + 1, 0, self.num_key_heads)
            return k, v
        else: 
            k, v = self.recompute_function(x)
            self.cache(k, v, 0,B, 0, T, 0, self.num_key_heads)
            return k, v
        
    
    def reset(self):
        if self.enabled:
            self.key_cache = torch.zeros(self.max_batch_size, self.num_key_heads, self.max_sequence_length, self.key_dimension, device=self.device)
            self.value_cache = torch.zeros(self.max_batch_size, self.num_value_heads, self.max_sequence_length, self.value_dimension, device=self.device)
            self.empty = True
    
def load_kv_cache(kv_cache_params: dict, **kwargs):
    params = KVCacheParams(**kv_cache_params)
    kv_cache = KVCache(params, **kwargs)
    return kv_cache