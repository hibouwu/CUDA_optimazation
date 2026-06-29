#pragma once

#include <cuda_runtime.h>

namespace gemm_sm110::requant {

struct IdentityScale {
  __host__ __device__ __forceinline__ float normalize(float value,
                                                       int) const {
    return value;
  }
};

struct PerTensorScale {
  float inverse_scale;

  __host__ __device__ explicit PerTensorScale(float inverse_scale_)
      : inverse_scale(inverse_scale_) {}

  __host__ __device__ __forceinline__ float normalize(float value,
                                                       int) const {
    return value * inverse_scale;
  }
};

template <int BlockSize>
struct PerBlockScale {
  static_assert(BlockSize > 0, "BlockSize must be positive");

  const float* inverse_scales;

  __host__ __device__ explicit PerBlockScale(const float* inverse_scales_)
      : inverse_scales(inverse_scales_) {}

  __host__ __device__ __forceinline__ float normalize(
      float value, int linear_index) const {
    return value * inverse_scales[linear_index / BlockSize];
  }
};

using PerNvfp4BlockScale = PerBlockScale<16>;

}  // namespace gemm_sm110::requant
