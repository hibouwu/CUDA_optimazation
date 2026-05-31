#pragma once

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

constexpr int kBlockSize = 256;
constexpr int kWarmup = 5;
constexpr int kRepeat = 20;

inline float abs_float(float value) { return value < 0.0f ? -value : value; }

inline double abs_double(double value) { return value < 0.0 ? -value : value; }
