cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 110 CACHE STRING "CUDA architecture")
endif()

project(register_research LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(register_bench register_bench.cu)
target_compile_features(register_bench PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(register_bench PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(register_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)

add_executable(sass_register_bench sass_driver.cpp)
target_compile_features(sass_register_bench PRIVATE cxx_std_17)
target_link_libraries(sass_register_bench PRIVATE CUDA::cuda_driver)

set(SASS_TEMPLATE_CUBIN
    "${CMAKE_CURRENT_BINARY_DIR}/sass_template.sm_${CMAKE_CUDA_ARCHITECTURES}.cubin")
add_custom_command(
  OUTPUT "${SASS_TEMPLATE_CUBIN}"
  COMMAND "${CMAKE_CUDA_COMPILER}" --cubin -O3 -lineinfo
          "-arch=sm_${CMAKE_CUDA_ARCHITECTURES}"
          "${CMAKE_CURRENT_SOURCE_DIR}/sass_template.cu"
          -o "${SASS_TEMPLATE_CUBIN}"
  DEPENDS sass_template.cu
  VERBATIM)
add_custom_target(sass_template ALL DEPENDS "${SASS_TEMPLATE_CUBIN}")

set(SASS_LOP3_TEMPLATE_CUBIN
    "${CMAKE_CURRENT_BINARY_DIR}/sass_lop3_template.sm_${CMAKE_CUDA_ARCHITECTURES}.cubin")
add_custom_command(
  OUTPUT "${SASS_LOP3_TEMPLATE_CUBIN}"
  COMMAND "${CMAKE_CUDA_COMPILER}" --cubin -O3 -lineinfo
          "-arch=sm_${CMAKE_CUDA_ARCHITECTURES}"
          "${CMAKE_CURRENT_SOURCE_DIR}/sass_lop3_template.cu"
          -o "${SASS_LOP3_TEMPLATE_CUBIN}"
  DEPENDS sass_lop3_template.cu
  VERBATIM)
add_custom_target(sass_lop3_template ALL DEPENDS "${SASS_LOP3_TEMPLATE_CUBIN}")

set(SASS_IMAD_TEMPLATE_CUBIN
    "${CMAKE_CURRENT_BINARY_DIR}/sass_imad_template.sm_${CMAKE_CUDA_ARCHITECTURES}.cubin")
add_custom_command(
  OUTPUT "${SASS_IMAD_TEMPLATE_CUBIN}"
  COMMAND "${CMAKE_CUDA_COMPILER}" --cubin -O3 -lineinfo
          "-arch=sm_${CMAKE_CUDA_ARCHITECTURES}"
          "${CMAKE_CURRENT_SOURCE_DIR}/sass_imad_template.cu"
          -o "${SASS_IMAD_TEMPLATE_CUBIN}"
  DEPENDS sass_imad_template.cu
  VERBATIM)
add_custom_target(sass_imad_template ALL DEPENDS "${SASS_IMAD_TEMPLATE_CUBIN}")

set(SASS_FMA_TEMPLATE_CUBIN
    "${CMAKE_CURRENT_BINARY_DIR}/sass_fma_template.sm_${CMAKE_CUDA_ARCHITECTURES}.cubin")
add_custom_command(
  OUTPUT "${SASS_FMA_TEMPLATE_CUBIN}"
  COMMAND "${CMAKE_CUDA_COMPILER}" --cubin -O3 -lineinfo
          "-arch=sm_${CMAKE_CUDA_ARCHITECTURES}"
          "${CMAKE_CURRENT_SOURCE_DIR}/sass_fma_template.cu"
          -o "${SASS_FMA_TEMPLATE_CUBIN}"
  DEPENDS sass_fma_template.cu
  VERBATIM)
add_custom_target(sass_fma_template ALL DEPENDS "${SASS_FMA_TEMPLATE_CUBIN}")
