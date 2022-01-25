from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name='bilateralfilter',
    ext_modules=[
        CUDAExtension('bilateralfilter_gpu_lib', [
            'csrc/bilateralfilter_gpu.cu',
            'csrc/bf_layer_gpu_forward.cu',
            'csrc/bf_layer_gpu_backward.cu',
        ]),
        CppExtension('bilateralfilter_cpu_lib', [
            'csrc/bilateralfilter_cpu.cpp',
            'csrc/bf_layer_cpu_forward.cpp',
            'csrc/bf_layer_cpu_backward.cpp'],
            extra_compile_args=['-fopenmp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
