import torch
import fast_reduction
import time
import numpy as np

# Assuming a 1D tensor
def manual_reduction(array: torch.Tensor):
    sum = 0.0
    for elem in array:
        sum += elem
    return sum

def fast_inefficient_reduction(array: torch.Tensor):
    return fast_reduction.inefficient_reduction_kernel(
        array
    )

def time_function(func, array, name, num_runs=1000):
    """Time a function over multiple runs"""
    # Warm up (especially important for CUDA)
    for _ in range(10):
        result = func(array)
    
    # Synchronize GPU if using CUDA tensors
    if array.is_cuda:
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result = func(array)
        if array.is_cuda:
            torch.cuda.synchronize()  # Ensure GPU work is complete
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / num_runs
    return result, avg_time

if __name__ == "__main__":
    # Test with small array first
    print("=== Small Array Test ===")
    small_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    manual_result = manual_reduction(small_data)
    fast_result = fast_inefficient_reduction(small_data)
    
    print(f"Manual reduction result: {manual_result}")
    print(f"Fast reduction result: {fast_result}")
    print(f"Results match: {torch.isclose(torch.tensor(manual_result), fast_result)}")
    print()
    
    # Test with larger arrays for meaningful timing
    sizes = [1000, 10000, 100000, 1000000]
    
    for size in sizes:
        print(f"=== Array Size: {size:,} ===")
        data = torch.randn(size)
        
        # Time manual reduction
        manual_result, manual_time = time_function(manual_reduction, data, "Manual", num_runs=100 if size <= 10000 else 10)
        
        # Time fast reduction
        fast_result, fast_time = time_function(fast_inefficient_reduction, data, "Fast", num_runs=100 if size <= 10000 else 10)
        
        # Compare with PyTorch's built-in sum for reference
        pytorch_result, pytorch_time = time_function(torch.sum, data, "PyTorch", num_runs=1000)
        
        print(f"Manual reduction:     {manual_time*1e6:8.2f} μs")
        print(f"Fast reduction:       {fast_time*1e6:8.2f} μs") 
        print(f"PyTorch sum:          {pytorch_time*1e6:8.2f} μs")
        
        # Calculate speedups
        if manual_time > 0:
            speedup_vs_manual = manual_time / fast_time
            print(f"Speedup vs manual:    {speedup_vs_manual:8.2f}x")
        
        if pytorch_time > 0:
            speedup_vs_pytorch = pytorch_time / fast_time
            print(f"Speedup vs PyTorch:   {speedup_vs_pytorch:8.2f}x")
        
        # Verify correctness
        manual_close = torch.isclose(torch.tensor(manual_result), fast_result, rtol=1e-5)
        pytorch_close = torch.isclose(pytorch_result, fast_result, rtol=1e-5)
        print(f"Manual vs Fast match:   {manual_close}")
        print(f"PyTorch vs Fast match:  {pytorch_close}")
        print()
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        print("=== CUDA Performance Test ===")
        cuda_data = torch.randn(1000000, device='cuda')
        
        print("CUDA tensor (1M elements):")
        fast_result_cuda, fast_time_cuda = time_function(fast_inefficient_reduction, cuda_data, "Fast CUDA", num_runs=100)
        pytorch_result_cuda, pytorch_time_cuda = time_function(torch.sum, cuda_data, "PyTorch CUDA", num_runs=1000)
        
        print(f"Fast reduction (CUDA):  {fast_time_cuda*1e6:8.2f} μs")
        print(f"PyTorch sum (CUDA):     {pytorch_time_cuda*1e6:8.2f} μs")
        
        if pytorch_time_cuda > 0:
            cuda_speedup = pytorch_time_cuda / fast_time_cuda
            print(f"CUDA Speedup vs PyTorch: {cuda_speedup:8.2f}x")
        
        cuda_match = torch.isclose(pytorch_result_cuda, fast_result_cuda, rtol=1e-5)
        print(f"CUDA results match:      {cuda_match}")
    else:
        print("CUDA not available for testing")