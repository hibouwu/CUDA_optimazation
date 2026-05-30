cmake_minimum_required(VERSION 3.20)
project(cuda_optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(reduce_bench REDUCE/src/main.cu)
target_compile_options(reduce_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

add_executable(gemm_bench GEMM/src/main.cu)
target_compile_options(gemm_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(gemm_bench PRIVATE CUDA::cublas)
