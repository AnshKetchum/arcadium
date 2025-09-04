import torch
import fast_reduction
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

def manual_reduction(array: torch.Tensor):
    s = 0.0
    for elem in array:
        s += elem
    return s

def inefficient_reduction(array: torch.Tensor):
    return fast_reduction.inefficient_reduction_kernel(array)

def single_block_thread_based_reduction(array: torch.Tensor):
    return fast_reduction.single_block_thread_based_reduction(array)

def multi_block_reduction(array: torch.Tensor):
    return fast_reduction.multi_block_thread_based_reduction(array)

def time_function(func, array, num_runs=50):
    # Warm-up
    for _ in range(5):
        _ = func(array)
    if array.is_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        result = func(array)
        if array.is_cuda:
            torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / num_runs  # average latency (seconds)

if __name__ == "__main__":
    sizes = [
        100, 500,
        1_000, 2_000, 5_000,
        10_000, 20_000, 50_000,
        100_000, 200_000 
    ]


    manual_latencies = []
    fast_latencies = []
    torch_latencies = []
    single_block_latencies = []
    multi_block_latencies = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for size in tqdm(sizes, desc="Benchmarking sizes"):  # <-- progress bar
        data = torch.randn(size).to(device)

        # Fewer runs for large arrays
        runs = 100 if size <= 10000 else 10

        manual_time = time_function(manual_reduction, data, num_runs=runs)
        fast_time = time_function(inefficient_reduction, data, num_runs=runs)
        single_block_time = time_function(single_block_thread_based_reduction, data, num_runs=runs)
        multi_block_time = time_function(multi_block_reduction, data, num_runs=runs)
        torch_time = time_function(torch.sum, data, num_runs=runs)

        manual_latencies.append(manual_time * 1e6)  # μs
        fast_latencies.append(fast_time * 1e6)
        single_block_latencies.append(single_block_time * 1e6)
        multi_block_latencies.append(multi_block_time * 1e6)
        torch_latencies.append(torch_time * 1e6)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, manual_latencies, marker="o", label="Manual Reduction")
    plt.plot(sizes, fast_latencies, marker="s", label="Fast Reduction")
    plt.plot(sizes, single_block_latencies, marker="s", label="single_block GPU Reduction Latencies")
    plt.plot(sizes, multi_block_latencies, marker="s", label="Multi Block GPU Reduction Latencies")
    plt.plot(sizes, torch_latencies, marker="^", label="PyTorch Sum")
    plt.xlabel("Number of Elements")
    plt.ylabel("Latency (μs)")
    plt.title("Reduction Latency vs Array Size (CPU)")
    plt.legend()
    plt.grid(True, which="both")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()
