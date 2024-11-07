# run: python setup.py install
# requires gcc version>9 (module load gcc/11.2.0)

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
setup(
    name='call_cspmm',
    ext_modules=[
        CUDAExtension(
            'call_cspmm',
            ['cu_spmm.cpp', 'cuSPMM.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)

setup(
    name='call_dummy',
    ext_modules=[
        CUDAExtension(
            'call_dummy',
            ['cu_dummy.cpp', 'cuSPMM.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)