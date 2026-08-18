cmake_minimum_required(VERSION 3.24)
project(cutlass_gemm_cpp_guide LANGUAGES CXX)

option(GUIDE_ENABLE_CUDA_CASES "Build the SM110a CUDA cases" ON)
option(GUIDE_ENABLE_GPU_TESTS "Register runtime tests that require Thor" OFF)
option(GUIDE_ENABLE_EXPERIMENTAL_CASES "Build block-scaled/sparse candidate cases" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/versions.lock.json GUIDE_TOOLCHAIN_LOCK_SHA)
set(GUIDE_CUTLASS_SHA e05f953a5b3d38adc240df2ff928e0421c2abba3)

add_library(guide_host INTERFACE)
target_include_directories(guide_host INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_host_references tests/host/test_references.cpp)
target_link_libraries(test_host_references PRIVATE guide_host)
add_test(NAME host.references COMMAND test_host_references)
set_tests_properties(host.references PROPERTIES LABELS "host")

add_test(
  NAME host.case_contracts
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_cases.py
          --root ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(host.case_contracts PROPERTIES LABELS "host;schema")

add_test(
  NAME host.coverage_summary
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/summarize_coverage.py
          --root ${CMAKE_CURRENT_SOURCE_DIR} --check)
set_tests_properties(host.coverage_summary PROPERTIES LABELS "host;docs")

add_test(
  NAME host.cutlass_source_contracts
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/mechanism/test_cutlass_source_contracts.py)
set_tests_properties(host.cutlass_source_contracts PROPERTIES LABELS "host;mechanism;track-b")

if(GUIDE_ENABLE_CUDA_CASES)
  enable_language(CUDA)
  find_package(CUDAToolkit 13.0 REQUIRED)

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  set(CUTLASS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass)
  if(NOT EXISTS ${CUTLASS_ROOT}/include/cutlass/cutlass.h)
    message(FATAL_ERROR "CUTLASS submodule is missing; run git submodule update --init --recursive")
  endif()

  add_library(guide_cuda INTERFACE)
  target_include_directories(guide_cuda INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUTLASS_ROOT}/include
    ${CUTLASS_ROOT}/tools/util/include)
  target_link_libraries(guide_cuda INTERFACE CUDA::cudart CUDA::cuda_driver)
  target_compile_options(guide_cuda INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_110a,code=sm_110a>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

  function(add_guide_case case_id)
    file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/cases/${case_id}/case.cu case_source_sha)
    add_executable(${case_id} cases/${case_id}/case.cu)
    set_property(TARGET ${case_id} PROPERTY CUDA_ARCHITECTURES OFF)
    target_link_libraries(${case_id} PRIVATE guide_cuda)
    target_compile_definitions(${case_id} PRIVATE
      GUIDE_CASE_ID="${case_id}"
      GUIDE_CUTLASS_SHA="${GUIDE_CUTLASS_SHA}"
      GUIDE_TOOLCHAIN_LOCK_SHA="${GUIDE_TOOLCHAIN_LOCK_SHA}"
      GUIDE_SOURCE_SHA="${case_source_sha}")
    if(GUIDE_ENABLE_GPU_TESTS)
      add_test(NAME runtime.${case_id}
        COMMAND ${case_id} --verify --seed 20260817 --json-out
                ${CMAKE_CURRENT_BINARY_DIR}/results/${case_id}.json)
      set_tests_properties(runtime.${case_id} PROPERTIES LABELS "runtime-sm110a;core")
    endif()
  endfunction()

  add_guide_case(dense_f16_1sm_p128)
  add_guide_case(dense_bf16_1sm_p128)
  add_guide_case(dense_fp8_1sm_p128)
  add_guide_case(dense_f16_2sm_p256x128x128)
  add_guide_case(epilogue_bias_relu_f16_p128)
  add_guide_case(tail_dense_f16_p130x129x127)

  if(GUIDE_ENABLE_EXPERIMENTAL_CASES)
    add_guide_case(bs_mxfp8_1sm_p128)
    add_guide_case(bs_mxfp4_1sm_p128x128x256)
    add_guide_case(bs_nvfp4_1sm_p128x128x256)
    add_guide_case(sparse_bs_nvfp4_1sm_p128x128x256)
  endif()
endif()
