cmake_minimum_required(VERSION 3.20)
project(cuda_optimization LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

add_executable(reduce_bench REDUCE/src/main.cu)
target_compile_options(reduce_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_include_directories(reduce_bench PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

add_executable(gemm_bench GEMM/src/main.cu)
target_compile_options(gemm_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_link_libraries(gemm_bench PRIVATE CUDA::cublas)
