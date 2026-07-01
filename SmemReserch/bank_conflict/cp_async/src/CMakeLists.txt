cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architecture")
endif()
project(cp_async_bench LANGUAGES CXX CUDA)
add_executable(cp_async_bench cp_async_bench.cu)
target_compile_features(cp_async_bench PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(cp_async_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

