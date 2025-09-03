#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <vector>
#include "reduction_kernel.h"

#define DEBUG 0

namespace reduction_kernels {

    // Templated based reduction
    // TODO: convert this bad boi into a kernel
    template <typename scalar_t>
    torch::Tensor slow_reduction(
        const torch::Tensor tensor_to_reduce
    ) {
        // Sanity checks on input
        TORCH_CHECK(tensor_to_reduce.dim() == 1, "Reduction tensor should have a dimension of 1");

        scalar_t sum = 0;

        // Extract the data from the pointer
        auto data = tensor_to_reduce.data_ptr<float>();

        // Extract the dimension of the tensor
        int64_t n = tensor_to_reduce.sizes()[0];

        for(int64_t i = 0; i < n; i++) {
            // std::cout << data[i] << " ";
            sum += data[i];
        }
        // std::cout << "\n";

        return torch::full({}, sum, tensor_to_reduce.options());
    }

    // Dispatcher
    torch::Tensor inefficient_reduction_kernel(const torch::Tensor tensor_to_reduce) {
        switch (tensor_to_reduce.scalar_type()) {
            case at::kFloat: return slow_reduction<float>(tensor_to_reduce);
            case at::kDouble: return slow_reduction<double>(tensor_to_reduce);
            case at::kInt: return slow_reduction<int32_t>(tensor_to_reduce);
            case at::kLong: return slow_reduction<int64_t>(tensor_to_reduce);
            case at::kBFloat16: return slow_reduction<at::BFloat16>(tensor_to_reduce);
            case at::kHalf: return slow_reduction<at::Half>(tensor_to_reduce);
            default:
                TORCH_CHECK(false, "Unsupported tensor dtype for reduction");
        }
    }


}
