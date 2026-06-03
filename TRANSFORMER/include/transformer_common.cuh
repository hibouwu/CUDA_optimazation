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

constexpr int kTransformerWarmup = 5;
constexpr int kTransformerRepeat = 100;

struct TransformerShape {
  int batch;
  int seq_len;
  int hidden;
  int num_heads;
  int head_dim;
};

inline void print_shape(const TransformerShape& shape) {
  std::cout << "B=" << shape.batch << ", S=" << shape.seq_len
            << ", H=" << shape.hidden << ", heads=" << shape.num_heads
            << ", head_dim=" << shape.head_dim << '\n';
}

inline float transformer_gbytes(size_t bytes, float ms) {
  return static_cast<float>(bytes) / (ms * 1.0e6f);
}
