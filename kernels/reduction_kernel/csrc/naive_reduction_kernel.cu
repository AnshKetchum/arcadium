#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <vector>
#include "reduction_kernel.h"

#define DEBUG 0

namespace reduction_kernels {

    // A really inefficient reduction kernel that essentially "mocks" what a CPU does
    template <typename scalar_t> 
    __global__ void somewhat_slow_reduction_kernel(
        const scalar_t* __restrict__ tensor_to_reduce,
        scalar_t* reduction_result,
        int64_t tensor_length
    ) {
        if(threadIdx.x == 0 && blockIdx.x == 0) {
            scalar_t sum = 0;
            for (int i = 0; i < tensor_length; i++) {
                sum += tensor_to_reduce[i];
            }
            *reduction_result = sum;
        }
    }


    template <typename scalar_t>
    torch::Tensor slow_reduction_launcher(
        const torch::Tensor tensor_to_reduce
    ) {
        // Sanity checks on input
        TORCH_CHECK(tensor_to_reduce.dim() == 1, "Reduction tensor should have a dimension of 1");
        TORCH_CHECK(tensor_to_reduce.device().is_cuda(), "Reduction tensor must be on CUDA/HIP device");

        auto reduction_result = torch::full({}, 0, tensor_to_reduce.options());

        scalar_t* reduction_ptr  = reduction_result.data_ptr<scalar_t>();
        scalar_t* tensor_to_reduce_ptr = tensor_to_reduce.data_ptr<scalar_t>();

        somewhat_slow_reduction_kernel<scalar_t><<<1, 1>>>(
            tensor_to_reduce_ptr,
            reduction_ptr,
            tensor_to_reduce.size(0)
        );

        cudaDeviceSynchronize();

        return reduction_result;
    }

    // Dispatcher
    torch::Tensor inefficient_reduction_kernel(const torch::Tensor tensor_to_reduce) {
        switch (tensor_to_reduce.scalar_type()) {
            case at::kFloat: return slow_reduction_launcher<float>(tensor_to_reduce);
            case at::kDouble: return slow_reduction_launcher<double>(tensor_to_reduce);
            case at::kInt: return slow_reduction_launcher<int32_t>(tensor_to_reduce);
            case at::kLong: return slow_reduction_launcher<int64_t>(tensor_to_reduce);
            case at::kBFloat16: return slow_reduction_launcher<at::BFloat16>(tensor_to_reduce);
            case at::kHalf: return slow_reduction_launcher<at::Half>(tensor_to_reduce);
            default:
                TORCH_CHECK(false, "Unsupported tensor dtype for reduction");
        }
    }


}
