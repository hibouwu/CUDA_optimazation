cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 110 CACHE STRING "CUDA architecture")
endif()
project(tma_bench LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
add_executable(tma_bench tma_bench.cu)
target_compile_features(tma_bench PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(tma_bench PRIVATE CUDA::cuda_driver)
target_compile_options(tma_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

