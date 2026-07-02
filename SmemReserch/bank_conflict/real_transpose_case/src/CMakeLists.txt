cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architecture")
endif()

project(real_transpose_bench LANGUAGES CXX CUDA)

add_executable(real_transpose_bench real_transpose_bench.cu)
target_compile_features(real_transpose_bench PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(real_transpose_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
