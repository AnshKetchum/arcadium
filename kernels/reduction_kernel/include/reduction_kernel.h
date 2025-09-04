#include <iostream>
#include <ATen/ATen.h>
#include <torch/extension.h>


namespace reduction_kernels {

    torch::Tensor inefficient_reduction_kernel(
        const torch::Tensor tensor_to_reduce
    );

    torch::Tensor single_block_thread_based_reduction(
        const torch::Tensor tensor_to_reduce
    );

    torch::Tensor multi_block_thread_based_reduction(
        const torch::Tensor tensor_to_reduce
    );
}