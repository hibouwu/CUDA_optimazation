cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architecture")
endif()
project(st_shared_1d_bench LANGUAGES CXX CUDA)
add_executable(st_shared_1d_bench st_shared_1d_bench.cu)
target_compile_features(st_shared_1d_bench PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(st_shared_1d_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

