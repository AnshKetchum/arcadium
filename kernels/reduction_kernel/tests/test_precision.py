import torch
import fast_reduction
import numpy as np

def test_reduction_kernels():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizes_to_test = [10, 100, 1000, 5000, 10000]

    for size in sizes_to_test:
        # Random input tensor in FP32
        data = torch.randn(size, dtype=torch.float32, device=device)

        # Reference result (manual CPU sum)
        ref = data.sum().item()

        # Single-block kernel
        single_block_out = fast_reduction.single_block_thread_based_reduction(data)
        single_block_out_val = single_block_out.item() if isinstance(single_block_out, torch.Tensor) else float(single_block_out)
        assert np.isclose(single_block_out_val, ref, rtol=1e-5, atol=1e-6), \
            f"Single-block kernel failed for size={size}: got {single_block_out_val}, expected {ref}"

        # Multi-block kernel
        multi_block_out = fast_reduction.multi_block_thread_based_reduction(data)
        multi_block_out_val = multi_block_out.item() if isinstance(multi_block_out, torch.Tensor) else float(multi_block_out)
        assert np.isclose(multi_block_out_val, ref, rtol=1e-5, atol=1e-6), \
            f"Multi-block kernel failed for size={size}: got {multi_block_out_val}, expected {ref}"

        # Inefficient kernel (optional reference comparison)
        inefficient_out = fast_reduction.inefficient_reduction_kernel(data)
        inefficient_out_val = inefficient_out.item() 
        assert np.isclose(inefficient_out_val, ref, rtol=1e-5, atol=1e-5), \
            f"Inefficient kernel failed for size={size}: got {inefficient_out_val}, expected {ref}"

        print(f"Passed all kernels for size={size}, sum={ref:.6f}")
