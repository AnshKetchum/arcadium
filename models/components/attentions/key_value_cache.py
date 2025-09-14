import torch 
import torch.nn as nn 

class KVCache:
    """
        A hopefully simple interface 
    """
    def __init__(self, max_batch_size, max_sequence_length, num_key_heads, num_value_heads, key_dimension, value_dimension, enabled = True):
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.key_dimension = key_dimension
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.value_dimension = value_dimension
        self.enabled = enabled

        if self.enabled:
            self.key_cache = torch.zeros(self.max_batch_size, self.num_key_heads, self.max_sequence_length, self.key_dimension)
            self.value_cache = torch.zeros(self.max_batch_size, self.num_value_heads, self.max_sequence_length, self.value_dimension)
    
    def cache(self, keys, values, batch_idx_start, batch_idx_end, sequence_length_idx_start, sequence_length_idx_end, num_heads_start, num_heads_end):
        if self.enabled:
            self.key_cache[batch_idx_start:batch_idx_end, num_heads_start: num_heads_end, sequence_length_idx_start : sequence_length_idx_end] = keys
            self.value_cache[batch_idx_start:batch_idx_end, num_heads_start: num_heads_end, sequence_length_idx_start : sequence_length_idx_end] = values
        return True

    def get(self, batch_idx_start, batch_idx_end, sequence_length_idx_start, sequence_length_idx_end, num_heads_start, num_heads_end):
        if self.enabled:
            k = self.key_cache[batch_idx_start:batch_idx_end, num_heads_start: num_heads_end, sequence_length_idx_start : sequence_length_idx_end]
            v = self.value_cache[batch_idx_start:batch_idx_end, num_heads_start: num_heads_end, sequence_length_idx_start : sequence_length_idx_end]
            return k, v
        else:
            return None, None
    
    def reset(self):
        if self.enabled:
            self.key_cache = torch.zeros(self.max_batch_size, self.num_key_heads, self.max_sequence_length, self.key_dimension)
            self.value_cache = torch.zeros(self.max_batch_size, self.num_value_heads, self.max_sequence_length, self.value_dimension)
    