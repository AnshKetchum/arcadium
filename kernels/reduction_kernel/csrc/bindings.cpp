#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "reduction_kernel.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("inefficient_reduction_kernel",
          &reduction_kernels::inefficient_reduction_kernel,
          "Inefficient reduction kernel",
          py::arg("tensor_to_reduce"));

    m.def("single_block_thread_based_reduction",
          &reduction_kernels::single_block_thread_based_reduction,
          "Uses one block and purely threads to perform a reduction",
          py::arg("tensor_to_reduce"));

    m.def("multi_block_thread_based_reduction",
          &reduction_kernels::multi_block_thread_based_reduction,
          "Uses threads + blocks to perform a reduction",
          py::arg("tensor_to_reduce"));
}