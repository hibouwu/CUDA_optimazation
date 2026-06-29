#pragma once

#include "e2m1_encode.cuh"
#include "pack_fp4.cuh"
#include "scale_policy.cuh"
#include "sm110_tcgen05_epilogue.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace gemm_sm110::requant {

template <typename ScalePolicy>
struct ReferenceE2M1Backend {
  ScalePolicy scale;

  __host__ __device__ explicit ReferenceE2M1Backend(ScalePolicy scale_)
      : scale(scale_) {}

  __host__ __device__ __forceinline__ std::uint8_t quantize(
      float accumulator, int linear_index) const {
    return encode_e2m1_rn(scale.normalize(accumulator, linear_index));
  }

  __host__ __device__ __forceinline__ std::uint8_t quantize_and_pack_x2(
      float accumulator0, float accumulator1, int linear_index0) const {
    return pack_e2m1x2(quantize(accumulator0, linear_index0),
                       quantize(accumulator1, linear_index0 + 1));
  }
};

}  // namespace gemm_sm110::requant
