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

option(CUDA_PTXAS_VERBOSE "Print ptxas resource usage for CUDA targets" OFF)

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
string(FIND "${CMAKE_CUDA_ARCHITECTURES}" "120a" HAS_SM120A_ARCH)
if(NOT HAS_SM120A_ARCH EQUAL -1)
  target_compile_definitions(gemm_bench PRIVATE
    TC3_COMPILED_SM120A_NARROW_MMA=1
  )
endif()
if(CUDA_PTXAS_VERBOSE)
  target_compile_options(gemm_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
  )
endif()
target_include_directories(gemm_bench PRIVATE
  GEMM/include
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(gemm_bench PRIVATE CUDA::cublas CUDA::cuda_driver)

add_executable(v7_resource_probe GEMM/src/v7_resource_probe.cu)
target_compile_options(v7_resource_probe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
if(CUDA_PTXAS_VERBOSE)
  target_compile_options(v7_resource_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
  )
endif()
target_include_directories(v7_resource_probe PRIVATE
  GEMM/include
  GEMM/generated
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(v7_resource_probe PRIVATE CUDA::cudart)

add_executable(gemm_tune_blocksize GEMM/src/tune_blocksize.cu)
target_compile_options(gemm_tune_blocksize PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_include_directories(gemm_tune_blocksize PRIVATE
  GEMM/include
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(gemm_tune_blocksize PRIVATE CUDA::cublas)

add_executable(transformer_bench
  TRANSFORMER/src/main.cu
  TRANSFORMER/include/transformer_common.cuh
  TRANSFORMER/include/transformer_kernels.cuh
  TRANSFORMER/include/transformer_benchmark.cuh
)
target_compile_options(transformer_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_include_directories(transformer_bench PRIVATE
  TRANSFORMER/include
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(transformer_bench PRIVATE CUDA::cudart)

function(add_gmem_to_smem_demo target_name source_path)
  add_executable(${target_name} ${source_path})
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  )
  target_include_directories(${target_name} PRIVATE
    GEMMComponents/GEMMGmemToSmem/common
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(${target_name} PRIVATE CUDA::cudart CUDA::cuda_driver)
endfunction()

add_gmem_to_smem_demo(
  gmem_to_smem_00_naive
  GEMMComponents/GEMMGmemToSmem/00_naive_gmem_to_smem/demo.cu
)
add_gmem_to_smem_demo(
  gmem_to_smem_01_coalesced
  GEMMComponents/GEMMGmemToSmem/01_coalesced_gmem_to_smem/demo.cu
)
add_gmem_to_smem_demo(
  gmem_to_smem_02_vectorized_float4
  GEMMComponents/GEMMGmemToSmem/02_vectorized_float4_load/demo.cu
)
add_gmem_to_smem_demo(
  gmem_to_smem_03_row_col_major
  GEMMComponents/GEMMGmemToSmem/03_row_col_major_addressing/demo.cu
)
add_gmem_to_smem_demo(
  gmem_to_smem_03_transposed_smem_store
  GEMMComponents/GEMMGmemToSmem/03_transposed_smem_store/demo.cu
)
add_gmem_to_smem_demo(
  gmem_to_smem_04_coalesced_transposed_store
  GEMMComponents/GEMMGmemToSmem/04_coalesced_load_transposed_store/demo.cu
)
add_gmem_to_smem_demo(
  gmem_to_smem_04_smem_padding
  GEMMComponents/GEMMGmemToSmem/04_smem_padding_bank_conflict/demo.cu
)
add_gmem_to_smem_demo(
  gmem_to_smem_05_smem_swizzle
  GEMMComponents/GEMMGmemToSmem/05_smem_swizzle_store/demo.cu
)
add_gmem_to_smem_demo(
  gmem_to_smem_05_transpose_padding
  GEMMComponents/GEMMGmemToSmem/05_transpose_padding_bank_conflict/demo.cu
)
add_gmem_to_smem_demo(
  gmem_to_smem_06_predicated_tile_load
  GEMMComponents/GEMMGmemToSmem/06_predicated_tile_load/demo.cu
)
add_gmem_to_smem_demo(
  gmem_to_smem_07_double_buffer
  GEMMComponents/GEMMGmemToSmem/07_double_buffer_gmem_to_smem/demo.cu
)
add_gmem_to_smem_demo(
  gmem_to_smem_08_cp_async
  GEMMComponents/GEMMGmemToSmem/08_cp_async_gmem_to_smem/demo.cu
)
add_gmem_to_smem_demo(
  gmem_to_smem_09_tma
  GEMMComponents/GEMMGmemToSmem/09_tma_gmem_to_smem/demo.cu
)
