import os
from pathlib import Path
from torch.utils.cpp_extension import load
from .utils.file import get_all_files
from .utils.package import find_eigen, get_cuda_arch

PROJECT_DIR = Path(__file__).absolute().parent

cpp_standard, min_compute_capability = get_cuda_arch()


verbose = False

if verbose:
    print(f"Targeting C++ standard {cpp_standard}")

base_nvcc_flags = [
    f"-std=c++{cpp_standard}",
    "--extended-lambda",
    "--expt-relaxed-constexpr",
    # The following definitions must be undefined
    # to support half-precision operation.
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

if os.name == "posix":
    base_cflags = [f"-std=c++{cpp_standard}"]
    base_nvcc_flags += ["-Xcompiler=-Wno-float-conversion", "-Xcompiler=-fno-strict-aliasing", "-O3"]
elif os.name == "nt":
    base_cflags = [f"/std:c++{cpp_standard}"]


source_dir = PROJECT_DIR

proj_cu_files = get_all_files(source_dir / "cpp", "*.cu")
proj_cpp_files = get_all_files(source_dir / "cpp", "*.cpp")

proj_src_files = proj_cu_files + proj_cpp_files
proj_src_files = [str(f.absolute()) for f in proj_src_files]

include_dirs = [f"{PROJECT_DIR}/cpp", find_eigen()]


base_definitions = []


cpp_backend = load(
    name="camera3d_cpp",
    extra_cflags=base_cflags,
    extra_cuda_cflags=base_nvcc_flags,
    sources=proj_src_files,
    extra_include_paths=include_dirs,
    verbose=verbose,
    with_cuda=True,
)
if verbose:
    print("Load CPP Extension")

__all__ = ["cpp_backend"]
