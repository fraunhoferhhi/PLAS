import os
import torch
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "plas"

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules



setup(
    name=library_name,
    version="0.2",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=[
        "numpy",
        "torch",
        "kornia",
        "tqdm",
        # example/eval deps, maybe split?
        # sort_3d_gaussians
        "pandas",
        "plyfile",
        "trimesh",
        # sort_rgb_img
        "click",
        "pillow",
        # eval/compare_plas_flas
        "opencv-python",
        # eval/flas
        # 'lap',
        "matplotlib",
        "scipy",
        "easydict",
        "imagecodecs"
    ],
    # entry_points={
    #     "console_scripts": [],
    # },
    package_data={
        "plas": ["../img/*.jpg", "primes.txt"],
    },
    include_package_data=True,
    cmdclass={"build_ext": BuildExtension},
)
