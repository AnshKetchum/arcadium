import torch
import triton
import triton.language as tl
import time
import math
import matplotlib.pyplot as plt
import numpy as np

@triton.jit 
def tiled_flash_attention_kernel(q_ptr, k_ptr, v_ptr, output_ptr, d_k, mask_ptr, T, H, 
                                TILE_SEQUENCE_LENGTH_KEYS: tl.constexpr,
                                TILE_SEQUENCE_LENGTH_QUERIES: tl.constexpr,
                                MAX_HEAD_DIMENSION: tl.constexpr, 
                                MAX_SEQLEN_DIMENSION: tl.constexpr):
    """
    Memory-efficient Flash Attention kernel with online softmax
    Directly computes output without storing attention probabilities
    """
    pid_i = tl.program_id(axis=0)
    
    # Load Q block indices and data
    q_row_indices = pid_i * TILE_SEQUENCE_LENGTH_QUERIES + tl.arange(0, TILE_SEQUENCE_LENGTH_QUERIES)
    q_mask = q_row_indices < T
    head_indices = tl.arange(0, MAX_HEAD_DIMENSION)
    q_slice = tl.load(q_ptr + q_row_indices[:, None] * H + head_indices[None, :], 
                     q_mask[:, None] & (head_indices < H)[None, :])
    
    # Online softmax registers
    l = tl.zeros([TILE_SEQUENCE_LENGTH_QUERIES], dtype=tl.float32)
    m = tl.full([TILE_SEQUENCE_LENGTH_QUERIES], -float('inf'), dtype=tl.float32)
    
    # Output accumulator - initialize to zero
    o = tl.zeros([TILE_SEQUENCE_LENGTH_QUERIES, MAX_HEAD_DIMENSION], dtype=tl.float32)
    
    # Process K,V chunks
    for seq_start in range(0, T, TILE_SEQUENCE_LENGTH_KEYS):
        k_col_indices = seq_start + tl.arange(0, TILE_SEQUENCE_LENGTH_KEYS)
        k_mask = k_col_indices < T
        
        # Load K transpose slice
        kt_slice = tl.load(k_ptr + k_col_indices[None, :] * H + head_indices[:, None], 
                          k_mask[None, :] & (head_indices < H)[:, None],
                          eviction_policy="evict_last")
        
        # Load V slice 
        v_slice = tl.load(v_ptr + k_col_indices[:, None] * H + head_indices[None, :], 
                         k_mask[:, None] & (head_indices < H)[None, :],
                         eviction_policy="evict_last")
        
        # Compute Q @ K^T with scaling and masking
        qkt_chunk = tl.dot(q_slice, kt_slice) / tl.sqrt(1.0 * d_k)
        mask_chunk = tl.load(mask_ptr + q_row_indices[:, None] * T + k_col_indices[None, :], 
                           q_mask[:, None] & k_mask[None, :],
                           eviction_policy="evict_last")
        qkt_chunk += mask_chunk
        
        # Update online softmax statistics
        chunk_max = tl.max(qkt_chunk, axis=1)
        m_prev = m
        m = tl.maximum(m, chunk_max)
        
        # Compute new normalization factor
        l_prev = l
        alpha = tl.exp(m_prev - m)
        l = alpha * l + tl.sum(tl.exp(qkt_chunk - m[:, None]), axis=1)
        
        # Update output according to FlashAttention algorithm:
        # o_i = o_{i-1} * (l_{i-1} / l_i) * exp(m_{i-1} - m_i) + (exp(x_i - m_i) / l_i) * V[i, :]
        
        # First term: rescale previous output
        o = o * alpha[:, None] * (l_prev / l)[:, None]
        
        # Second term: add current chunk contribution
        attn_weights = tl.exp(qkt_chunk - m[:, None]) / l[:, None]
        chunk_output = tl.dot(attn_weights, v_slice)
        o = o + chunk_output
    
    # Store final output
    tl.store(output_ptr + q_row_indices[:, None] * H + head_indices[None, :],
            o, q_mask[:, None] & (head_indices < H)[None, :])


def flash_attention_launcher(q, k, v, d_k, mask):
    """
    Flash Attention launcher that computes attention output directly
    without storing intermediate attention probabilities
    """
    T, d = q.shape
    
    output = torch.zeros((T, d), dtype=q.dtype, device=q.device)
    TILE_DIMENSION_KEYS = 16
    TILE_DIMENSION_QUERIES = 16
    
    C_t = math.ceil(T / TILE_DIMENSION_QUERIES)
    grid = lambda meta: (C_t,)
    
    tiled_flash_attention_kernel[grid](
        q, 
        k,
        v,
        output,
        d_k,
        mask,
        T,
        d,
        TILE_SEQUENCE_LENGTH_KEYS=TILE_DIMENSION_KEYS,
        TILE_SEQUENCE_LENGTH_QUERIES=TILE_DIMENSION_QUERIES,
        MAX_HEAD_DIMENSION=triton.next_power_of_2(d), 
        MAX_SEQLEN_DIMENSION=triton.next_power_of_2(T)
    )

    return output

def fast_attention_launcher(q, k, v, d_k, mask):
    """
    Fused approach - single kernel
    """
    return flash_attention_launcher(q, k, v, d_k, mask)