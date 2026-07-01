cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES OFF)
project(tcgen05_smem_operand_bench LANGUAGES CXX CUDA)
add_executable(tcgen05_smem_operand_bench tcgen05_smem_operand_bench.cu)
target_compile_features(tcgen05_smem_operand_bench PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(tcgen05_smem_operand_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_110a,code=sm_110a>)
