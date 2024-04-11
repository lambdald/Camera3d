import os
from pkg_resources import parse_version
from typing import Tuple, List
import torch
import subprocess
import re


def find_eigen(min_ver=(3, 3, 7), verbose=False):
    """Helper to find or download the Eigen C++ library"""
    try_paths = [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        os.path.expanduser("~/.local/include/eigen3"),
        "C:/Program Files/eigen3",
        "C:/Program Files (x86)/eigen3",
    ]
    WORLD_VER_STR = "#define EIGEN_WORLD_VERSION"
    MAJOR_VER_STR = "#define EIGEN_MAJOR_VERSION"
    MINOR_VER_STR = "#define EIGEN_MINOR_VERSION"
    EIGEN_WEB_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2"
    TMP_EIGEN_FILE = "tmp_eigen.tar.bz2"
    TMP_EIGEN_DIR = "/tmp/eigen-3.4.0"
    min_ver_str = ".".join(map(str, min_ver))

    eigen_path = None
    for path in try_paths:
        macros_path = os.path.join(path, "Eigen/src/Core/util/Macros.h")
        if os.path.exists(macros_path):
            macros = open(macros_path, "r").read().split("\n")
            world_ver, major_ver, minor_ver = None, None, None
            for line in macros:
                if line.startswith(WORLD_VER_STR):
                    world_ver = int(line[len(WORLD_VER_STR) :])
                elif line.startswith(MAJOR_VER_STR):
                    major_ver = int(line[len(MAJOR_VER_STR) :])
                elif line.startswith(MINOR_VER_STR):
                    minor_ver = int(line[len(MAJOR_VER_STR) :])

            if world_ver is None or major_ver is None or minor_ver is None:
                print("Failed to parse macros file", macros_path)
            else:
                ver = (world_ver, major_ver, minor_ver)
                ver_str = ".".join(map(str, ver))
                if ver < min_ver:
                    print("Found unsuitable Eigen version", ver_str, "at", path, "(need >= " + min_ver_str + ")")
                else:
                    if verbose:
                        print("Found Eigen version", ver_str, "at", path)
                    eigen_path = path
                    break

    if eigen_path is None:
        try:
            import urllib.request

            print("Couldn't find Eigen locally, downloading...")
            req = urllib.request.Request(
                EIGEN_WEB_URL,
                data=None,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36"
                },
            )

            with urllib.request.urlopen(req) as resp, open(TMP_EIGEN_FILE, "wb") as file:
                data = resp.read()
                file.write(data)
            import tarfile

            tar = tarfile.open(TMP_EIGEN_FILE)
            tar.extractall(os.path.dirname(TMP_EIGEN_DIR))
            tar.close()
            eigen_path = TMP_EIGEN_DIR
            os.remove(TMP_EIGEN_FILE)
        except:
            print("Download failed, failed to find Eigen")

    if eigen_path is not None:
        if verbose:
            print("Found eigen at", eigen_path)
    else:
        raise FileNotFoundError("Failed to find Eigen")

    return eigen_path


def min_supported_compute_capability(cuda_version):
    if cuda_version >= parse_version("12.0"):
        return 50
    else:
        return 20


def max_supported_compute_capability(cuda_version):
    if cuda_version < parse_version("11.0"):
        return 75
    elif cuda_version < parse_version("11.1"):
        return 80
    elif cuda_version < parse_version("11.8"):
        return 86
    else:
        return 90


def get_cuda_arch(verbose=False) -> Tuple[int, int]:
    """
    return: cpp_std, compute_capability
    """

    assert torch.cuda.is_available()
    cpp_standard = 14

    major, minor = torch.cuda.get_device_capability()
    compute_capabilities = [major * 10 + minor]
    # print(f"Obtained compute capability {compute_capabilities[0]} from PyTorch")

    # Get CUDA version and make sure the targeted compute capability is compatible
    nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode()
    cuda_version = re.search(r"release (\S+),", nvcc_out)

    if cuda_version:
        cuda_version = parse_version(cuda_version.group(1))
        if verbose:
            print(f"Detected CUDA version {cuda_version}")
        if cuda_version >= parse_version("11.0"):
            cpp_standard = 17

        supported_compute_capabilities = [
            cc
            for cc in compute_capabilities
            if cc >= min_supported_compute_capability(cuda_version)
            and cc <= max_supported_compute_capability(cuda_version)
        ]

        if not supported_compute_capabilities:
            supported_compute_capabilities = [max_supported_compute_capability(cuda_version)]

        if supported_compute_capabilities != compute_capabilities:
            print(
                f"WARNING: Compute capabilities {compute_capabilities} are not all supported by the installed CUDA version {cuda_version}. Targeting {supported_compute_capabilities} instead."
            )
            compute_capabilities = supported_compute_capabilities

    min_compute_capability = min(compute_capabilities)
    return cpp_standard, min_compute_capability
