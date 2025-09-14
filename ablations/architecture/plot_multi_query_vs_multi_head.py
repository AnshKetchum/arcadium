import matplotlib.pyplot as plt 
import torch
import time
from models.components.attentions.grouped_query_attention import GroupedQueryAttention
from models.components.attentions.multi_head_attention import MultiHeadAttention
from models.components.attentions.multi_query_attention import MultiQueryAttention

# Benchmark function
def benchmark_attention(attn_cls, config_list, B=8, T=128, E=1024, head_dim=64, warmup=3, trials=100):
    times = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for config in config_list:
        # Create attention layer based on config
        if isinstance(config, tuple):  # GQA case: (num_query_heads, num_kv_heads)
            num_query_heads, num_kv_heads = config
            attn = attn_cls(num_query_heads=num_query_heads, num_kv_heads=num_kv_heads, 
                          embedding_dimension=E, head_dimension=head_dim).to(device)
        else:  # MHA/MQA case: just num_heads
            if attn_cls == MultiQueryAttention:
                attn = attn_cls(num_query_heads=config, embedding_dimension=E, head_dimension=head_dim).to(device)
            else:  # MHA
                attn = attn_cls(num_kv_heads=config, embedding_dimension=E, head_dimension=head_dim).to(device)
        
        x = torch.randn(B, T, E, device=device)
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = attn(x)
        
        torch.cuda.synchronize()
        
        # Time measurement
        start_time = time.time()
        for _ in range(trials):
            with torch.no_grad():
                out = attn(x)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / trials
        times.append(avg_time)
        
        # Clean up
        del attn, x, out
        torch.cuda.empty_cache()
    
    return times

def main():
    # Configuration matching the target graph
    group_counts = [1, 4, 8, 16, 32, 64]  # GQA groups (num_kv_heads)
    H = 64  # Total number of query heads (constant)
    
    # Benchmark parameters
    B, T, E = 8, 128, 1024
    head_dim = 64
    
    print("Running benchmarks...")
    
    # MHA: Always uses H heads for Q, K, V
    print("Benchmarking MHA...")
    mha_configs = [H] * len(group_counts)  # Always 64 heads
    mha_times = benchmark_attention(MultiHeadAttention, mha_configs, B, T, E, head_dim)
    
    # MQA: Always uses H query heads, 1 key/value head
    print("Benchmarking MQA...")
    mqa_configs = [H] * len(group_counts)  # Always 64 query heads
    mqa_times = benchmark_attention(MultiQueryAttention, mqa_configs, B, T, E, head_dim)
    
    # GQA: H query heads, variable number of kv heads (groups)
    print("Benchmarking GQA...")
    gqa_configs = [(H, kv_heads) for kv_heads in group_counts]
    gqa_times = benchmark_attention(GroupedQueryAttention, gqa_configs, B, T, E, head_dim)
    
    print("Creating plot...")
    
    # Create the plot matching the target image
    plt.figure(figsize=(10, 6))
    
    # Plot with matching style
    plt.plot(group_counts, mha_times, 'rs-', linewidth=2, markersize=8, 
             markerfacecolor='lightcoral', label='MHA', alpha=0.8)
    plt.plot(group_counts, gqa_times, 'bo-', linewidth=2, markersize=8, 
             markerfacecolor='steelblue', label='GQA', alpha=0.8)
    plt.plot(group_counts, mqa_times, 'o-', color='orange', linewidth=2, markersize=8, 
             markerfacecolor='orange', label='MQA', alpha=0.8)
    
    # Formatting to match target image
    plt.xlabel('GQA groups', fontsize=12)
    plt.ylabel('Time per sample (s)', fontsize=12)
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    # Set x-axis ticks to match target
    plt.xticks(group_counts, group_counts)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    plt.legend(loc='upper left', fontsize=10)
    
    # Set y-axis limits to better show the relationship
    plt.ylim(bottom=min(min(mqa_times), min(gqa_times)) * 0.8)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("gqa_benchmark_replication.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print results
    print("\nResults:")
    print("Groups\tMHA Time\tGQA Time\tMQA Time")
    for i, groups in enumerate(group_counts):
        print(f"{groups}\t{mha_times[i]:.6f}\t{gqa_times[i]:.6f}\t{mqa_times[i]:.6f}")

if __name__ == "__main__":
    main()