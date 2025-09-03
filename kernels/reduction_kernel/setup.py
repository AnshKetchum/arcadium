import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths, library_paths

PROJECT_NAME = "fast_reduction"
this_dir = os.path.dirname(os.path.abspath(__file__))

# -allow-unsupported-compiler is needed for my setup with RTX 4060

nvcc_flags = ["-std=c++17"]
cxx_flags = ["-std=c++17"]

# Automatically find all .cpp and .cu files in csrc/
sources = glob.glob(os.path.join(this_dir, "csrc", "*.cpp")) + \
          glob.glob(os.path.join(this_dir, "csrc", "*.cu"))

setup(
    name=PROJECT_NAME,
    ext_modules=[
        CUDAExtension(
            name=PROJECT_NAME,
            sources=sources,
            include_dirs=[
                os.path.join(this_dir, "include"),  # optional include folder
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,  
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)