#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <vector>
#include "reduction_kernel.h"

#define DEBUG 0

namespace reduction_kernels {

    // Small array reduction kernel
    // This will reduce from a small-enough 1D tensor to an actual scalar tensor
    template <typename scalar_t> 
    __global__ void thread_reduction_kernel(
        const scalar_t* __restrict__ tensor_to_reduce,
        scalar_t* reduction_result,
        int64_t tensor_length
    ) {
        // This sdata is shared memory accessible across THREADS, not BLOCKS
        // It's on a per-block basis. So, each thread should get MEM PER BLOCK / NUM THREADS bytes

        // Cast the dynamically allocated shared memory
        extern __shared__ double sdata [];

        int tid = threadIdx.x;
        int blockSize = blockDim.x;

        // Load data into shared memory with bounds checking
        if (tid < tensor_length) {
            sdata[tid] = static_cast<double>(tensor_to_reduce[tid]);
        } else {
            sdata[tid] = 0.0;
        }
        __syncthreads();

        // Perform reduction in shared memory using tree reduction
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < blockSize) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Write result for this block to global memory
        if (tid == 0) {
            *reduction_result = static_cast<scalar_t>(sdata[0]);
        }
    }
    
}