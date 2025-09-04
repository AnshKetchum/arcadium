#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <vector>

#include "reduction_kernel.h"
#include "thread_reduction.cuh"

#define DEBUG 0

namespace reduction_kernels {

    // This kernel is designed to handle large 1D arrays >>> hardware grid-block-thread products
    // It will first work to bring the results within the a fixed buffer array, which can then be tree-reduced
    template <typename scalar_t> 
    __global__ void large_array_thread_based_reduction_kernel(
        const scalar_t* __restrict__ tensor_to_reduce,
        scalar_t* temp_buffer, // A buffer of zeroes
        int64_t tensor_length
    ) {
        int64_t tid = threadIdx.x;
        int64_t num_threads = blockDim.x;
        
        // Use Kahan summation for better numerical precision
        double sum = 0.0;
        double c = 0.0; // Running compensation for lost low-order bits
        
        // Each thread processes multiple elements with stride to improve memory access
        for (int64_t i = tid; i < tensor_length; i += num_threads) {
            double y = static_cast<double>(tensor_to_reduce[i]) - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        
        temp_buffer[tid] = static_cast<scalar_t>(sum);
    }

    template <typename scalar_t>
    torch::Tensor single_block_thread_reduction_launcher(
        const torch::Tensor tensor_to_reduce
    ) {
        // Sanity checks on input
        TORCH_CHECK(tensor_to_reduce.dim() == 1, "Reduction tensor should have a dimension of 1");
        TORCH_CHECK(tensor_to_reduce.device().is_cuda(), "Reduction tensor must be on CUDA/HIP device");
        TORCH_CHECK(tensor_to_reduce.is_contiguous(), "Tensor must be contiguous");

        auto reduction_result = torch::zeros({}, tensor_to_reduce.options());

        scalar_t* reduction_ptr = reduction_result.data_ptr<scalar_t>();
        const scalar_t* tensor_to_reduce_ptr = tensor_to_reduce.data_ptr<scalar_t>();

        int64_t tensor_size = tensor_to_reduce.size(0);
        
        // Handle empty tensor
        if (tensor_size == 0) {
            return reduction_result;
        }

        // Configure threads - use power of 2 for better reduction efficiency
        int64_t num_threads = std::min<int64_t>(512, tensor_size);
        // Round down to nearest power of 2 for optimal tree reduction
        num_threads = 1 << (int)(log2((double)num_threads));
        
        dim3 threads(num_threads);
        dim3 blocks(1);

        size_t shmem_size = num_threads * sizeof(double);
        
        // Check shared memory limit
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if (shmem_size > prop.sharedMemPerBlock) {
            num_threads = prop.sharedMemPerBlock / sizeof(double);
            num_threads = 1 << (int)(log2((double)num_threads)); // Round down to power of 2
            threads = dim3(num_threads);
            shmem_size = num_threads * sizeof(double);
        }

        if (tensor_size > num_threads) {
            // Two-stage reduction for large arrays
            auto temporary_buffer = torch::zeros({num_threads}, tensor_to_reduce.options());
            scalar_t* temporary_buffer_ptr = temporary_buffer.data_ptr<scalar_t>();

            // Stage 1: Reduce large array to temporary buffer
            large_array_thread_based_reduction_kernel<scalar_t><<<blocks, threads>>>(
                tensor_to_reduce_ptr,
                temporary_buffer_ptr,
                tensor_size
            );
            
            // Check for kernel errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                TORCH_CHECK(false, "CUDA kernel error in large array reduction: ", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();

            // Stage 2: Reduce temporary buffer to final result
            thread_reduction_kernel<scalar_t><<<1, threads, shmem_size>>>(
                temporary_buffer_ptr,
                reduction_ptr,
                num_threads
            );

        } else {
            // Single-stage reduction for small arrays
            thread_reduction_kernel<scalar_t><<<1, threads, shmem_size>>>(
                tensor_to_reduce_ptr,
                reduction_ptr,
                tensor_size
            );
        }

        // Check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel error in thread reduction: ", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        
        return reduction_result;
    }

    // Dispatcher
    torch::Tensor single_block_thread_based_reduction(const torch::Tensor tensor_to_reduce) {
        switch (tensor_to_reduce.scalar_type()) {
            case at::kFloat: return single_block_thread_reduction_launcher<float>(tensor_to_reduce);
            case at::kDouble: return single_block_thread_reduction_launcher<double>(tensor_to_reduce);
            case at::kInt: return single_block_thread_reduction_launcher<int32_t>(tensor_to_reduce);
            case at::kLong: return single_block_thread_reduction_launcher<int64_t>(tensor_to_reduce);
            case at::kBFloat16: return single_block_thread_reduction_launcher<at::BFloat16>(tensor_to_reduce);
            case at::kHalf: return single_block_thread_reduction_launcher<at::Half>(tensor_to_reduce);
            default:
                TORCH_CHECK(false, "Unsupported tensor dtype for reduction");
        }
    }
}