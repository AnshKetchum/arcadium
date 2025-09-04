#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <vector>

#include "thread_reduction.cuh"
#include "reduction_kernel.h"

#define DEBUG 0

namespace reduction_kernels {

    // This kernel handles large arrays by having each thread process multiple elements with stride
    template <typename scalar_t> 
    __global__ void large_array_stride_reduction_kernel(
        const scalar_t* __restrict__ tensor_to_reduce,
        scalar_t* temp_buffer,
        int64_t tensor_length
    ) {
        int tid = threadIdx.x;
        int num_threads = blockDim.x;
        
        // Use Kahan summation for better numerical precision
        double sum = 0.0;
        double c = 0.0; // Running compensation for lost low-order bits
        
        // Each thread processes elements with stride = num_threads
        for (int64_t i = tid; i < tensor_length; i += num_threads) {
            double y = static_cast<double>(tensor_to_reduce[i]) - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        
        temp_buffer[tid] = static_cast<scalar_t>(sum);
    }

    template <typename scalar_t>
    torch::Tensor multi_block_thread_reduction_launcher(
        const torch::Tensor tensor_to_reduce
    ) {
        // Sanity checks on input
        TORCH_CHECK(tensor_to_reduce.dim() == 1, "Reduction tensor should have a dimension of 1");
        TORCH_CHECK(tensor_to_reduce.device().is_cuda(), "Reduction tensor must be on CUDA/HIP device");
        TORCH_CHECK(tensor_to_reduce.is_contiguous(), "Tensor must be contiguous");

        auto reduction_result = torch::zeros({}, tensor_to_reduce.options());
        
        int64_t tensor_size = tensor_to_reduce.size(0);
        
        // Handle empty tensor
        if (tensor_size == 0) {
            return reduction_result;
        }

        scalar_t* reduction_ptr = reduction_result.data_ptr<scalar_t>();
        const scalar_t* tensor_to_reduce_ptr = tensor_to_reduce.data_ptr<scalar_t>();

        // Configure threads - use power of 2 for optimal reduction, up to 512
        int64_t num_threads = std::min<int64_t>(512, tensor_size);
        if (num_threads > 1) {
            // Round down to nearest power of 2
            int log_threads = (int)(log2((double)num_threads));
            num_threads = 1 << log_threads;
        }
        num_threads = std::max(1L, num_threads);
        
        // Shared memory size for double precision accumulation
        size_t shmem_size = num_threads * sizeof(double);
        
        // Get device properties for shared memory check
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        // Check shared memory limit and adjust if necessary
        if (shmem_size > prop.sharedMemPerBlock) {
            num_threads = prop.sharedMemPerBlock / sizeof(double);
            if (num_threads > 1) {
                int log_threads = (int)(log2((double)num_threads));
                num_threads = 1 << log_threads;
            }
            num_threads = std::max(1L, num_threads);
            shmem_size = num_threads * sizeof(double);
        }

        dim3 threads(num_threads);
        dim3 blocks(1);

        if (tensor_size > num_threads) {
            // Single-block stride reduction approach (like the working single-block kernel)
            auto temporary_buffer = torch::zeros({num_threads}, tensor_to_reduce.options());
            scalar_t* temporary_buffer_ptr = temporary_buffer.data_ptr<scalar_t>();

            // Stage 1: Each thread reduces its assigned elements with stride
            large_array_stride_reduction_kernel<scalar_t><<<blocks, threads>>>(
                tensor_to_reduce_ptr,
                temporary_buffer_ptr,
                tensor_size
            );
            
            cudaDeviceSynchronize();

            // Stage 2: Tree reduction of the partial sums
            thread_reduction_kernel<scalar_t><<<1, threads, shmem_size>>>(
                temporary_buffer_ptr,
                reduction_ptr,
                num_threads
            );
            
        } else {
            // Small array that fits in one block - direct reduction
            thread_reduction_kernel<scalar_t><<<1, threads, shmem_size>>>(
                tensor_to_reduce_ptr,
                reduction_ptr,
                tensor_size
            );
        }

        // Check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel error in multi-block thread reduction: ", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        
        return reduction_result;
    }

    // Dispatcher
    torch::Tensor multi_block_thread_based_reduction(const torch::Tensor tensor_to_reduce) {
        switch (tensor_to_reduce.scalar_type()) {
            case at::kFloat: return multi_block_thread_reduction_launcher<float>(tensor_to_reduce);
            case at::kDouble: return multi_block_thread_reduction_launcher<double>(tensor_to_reduce);
            case at::kInt: return multi_block_thread_reduction_launcher<int32_t>(tensor_to_reduce);
            case at::kLong: return multi_block_thread_reduction_launcher<int64_t>(tensor_to_reduce);
            case at::kBFloat16: return multi_block_thread_reduction_launcher<at::BFloat16>(tensor_to_reduce);
            case at::kHalf: return multi_block_thread_reduction_launcher<at::Half>(tensor_to_reduce);
            default:
                TORCH_CHECK(false, "Unsupported tensor dtype for reduction");
        }
    }
}