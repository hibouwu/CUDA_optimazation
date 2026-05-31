#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                << cudaGetErrorString(err__) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                        \
  } while (0)

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t status__ = (call);                                         \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                  \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__         \
                << " - status " << status__ << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                        \
  } while (0)

constexpr int kWarmup = 5;
constexpr int kRepeat = 100;
constexpr int kWarpSize = 32;

inline float abs_float(float value) { return value < 0.0f ? -value : value; }

__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}
