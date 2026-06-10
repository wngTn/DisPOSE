# Copyright 2021 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------------------------------------------------------------------------------------------------------
# Modified from
# https://github.com/chengdazhi/
# Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# and Deformable DETR
# --------------------------------------------------------------------------------------------------------------------

import os
import glob

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]

        # Auto-detect GPU architecture, with override via TORCH_CUDA_ARCH_LIST
        gencode_flags = []
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        supported_archs = []
        for arch in torch.cuda.get_arch_list():
            if not arch.startswith("sm_"):
                continue
            arch_num = arch.split("_", 1)[1]
            if arch_num.endswith("a"):
                arch_num = arch_num[:-1]
            if len(arch_num) < 2:
                continue
            supported_archs.append((int(arch_num[:-1]), int(arch_num[-1])))
        supported_archs = sorted(set(supported_archs))

        if arch_list:
            # User-specified architectures, e.g. "8.9" or "8.9;12.1+PTX"
            for arch in arch_list.replace(",", ";").split(";"):
                arch = arch.strip()
                if not arch:
                    continue
                ptx = "+PTX" in arch
                arch = arch.replace("+PTX", "").strip()
                major, minor = arch.split(".")
                code_num = f"{major}{minor}"
                gencode_flags.append(
                    f"-gencode=arch=compute_{code_num},code=sm_{code_num}"
                )
                if ptx:
                    gencode_flags.append(
                        f"-gencode=arch=compute_{code_num},code=compute_{code_num}"
                    )
        else:
            # Auto-detect from available GPUs
            seen = set()
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                if supported_archs and (major, minor) not in supported_archs:
                    fallback_major, fallback_minor = supported_archs[-1]
                    print(
                        "Detected GPU capability "
                        f"{major}.{minor}, but the installed PyTorch/CUDA stack "
                        f"supports up to {fallback_major}.{fallback_minor}. "
                        "Falling back to the highest supported PTX target."
                    )
                    major, minor = fallback_major, fallback_minor
                if (major, minor) not in seen:
                    seen.add((major, minor))
                    code_num = f"{major}{minor}"
                    # SASS for this GPU + PTX for forward compatibility
                    gencode_flags.append(
                        f"-gencode=arch=compute_{code_num},code=sm_{code_num}"
                    )
                    gencode_flags.append(
                        f"-gencode=arch=compute_{code_num},code=compute_{code_num}"
                    )

        if not gencode_flags:
            raise RuntimeError(
                "Could not detect GPU architecture. "
                "Set TORCH_CUDA_ARCH_LIST (e.g. '8.9') and retry."
            )

        print(f"Building with gencode flags: {gencode_flags}")

        extra_compile_args["nvcc"] = [
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ] + gencode_flags
    else:
        raise NotImplementedError("CUDA is not available")


    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "Deformable",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="Deformable",
    version="1.0",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
