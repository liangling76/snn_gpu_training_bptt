from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='snnlib',
    version='1',
    ext_modules=[
        CUDAExtension(
            'snnlib', ['cuda/snn.cpp', 'cuda/snn_kernel_fp.cu', 'cuda/snn_kernel_bp.cu'],
            extra_compile_args={'cxx':[], 'nvcc': ['-arch=sm_70']})
    ],
    cmdclass={'build_ext': BuildExtension}, install_requires=['torch']
)

