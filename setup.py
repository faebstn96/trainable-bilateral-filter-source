from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='bilateralfilter_torch',
    version='1.1.0',
    author='Fabian Wagner',
    author_email='fabian.wagner@fau.de',
    description='Trainable Bilateral Filter Layer (PyTorch)',
    setup_requires=['torch'],
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/faebstn96/trainable-bilateral-filter-source',
    py_modules=['bilateral_filter_layer', 'example_filter', 'example_optimization', 'gradcheck'],
    ext_modules=[
        CUDAExtension('bilateralfilter_gpu_lib', [
            'csrc/bilateralfilter_gpu.cu',
            'csrc/bf_layer_gpu_forward.cu',
            'csrc/bf_layer_gpu_backward.cu',
        ],
                      include_dirs=['utils', 'csrc'],),
        CppExtension('bilateralfilter_cpu_lib', [
            'csrc/bilateralfilter_cpu.cpp',
            'csrc/bf_layer_cpu_forward.cpp',
            'csrc/bf_layer_cpu_backward.cpp',
        ],
                     include_dirs=['utils', 'csrc'],
                     extra_compile_args=['-fopenmp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
