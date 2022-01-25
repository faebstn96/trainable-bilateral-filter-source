set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.5/bin/nvcc)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(bilateral_filter CXX CUDA)

find_package(Torch REQUIRED)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

add_executable(main main.cu bf_layer_gpu_forward.cu bf_layer_gpu_backward.cu bf_layer_cpu_forward.cpp bf_layer_cpu_backward.cpp)

set_target_properties(
        main
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 14)
include_directories(/usr/include/python3.8)