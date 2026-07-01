cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
      "CUDA architectures, for example 80, 90, 100, or 110")
endif()

project(smem_bank_bench LANGUAGES CXX CUDA)

add_executable(smem_bank_bench smem_bank_bench.cu)
target_compile_features(smem_bank_bench PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(smem_bank_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

target_compile_options(smem_bank_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
