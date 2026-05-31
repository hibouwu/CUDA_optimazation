cmake_minimum_required(VERSION 3.20)
project(cuda_optimization LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE DETECTED_COMPUTE_CAP
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REGEX REPLACE "\n.*" "" DETECTED_COMPUTE_CAP "${DETECTED_COMPUTE_CAP}")
  string(REPLACE "." "" DETECTED_CUDA_ARCH "${DETECTED_COMPUTE_CAP}")
  if(DETECTED_CUDA_ARCH MATCHES "^[0-9]+$")
    set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_CUDA_ARCH} CACHE STRING "CUDA architectures" FORCE)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
  endif()
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
target_include_directories(reduce_bench PRIVATE
  REDUCE/include
  ${CUDAToolkit_INCLUDE_DIRS}
)

add_executable(gemm_bench GEMM/src/main.cu)
target_compile_options(gemm_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_include_directories(gemm_bench PRIVATE
  GEMM/include
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(gemm_bench PRIVATE CUDA::cublas)

add_executable(gemm_tune_blocksize GEMM/src/tune_blocksize.cu)
target_compile_options(gemm_tune_blocksize PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_include_directories(gemm_tune_blocksize PRIVATE
  GEMM/include
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(gemm_tune_blocksize PRIVATE CUDA::cublas)
