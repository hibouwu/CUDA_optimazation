#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace gemm_sm110::requant {

// Encodes one normalized FP32 value as an E2M1 nibble:
//   bit 3: sign, bits 2:1: exponent, bit 0: mantissa.
//
// Finite magnitudes are {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Values outside the
// representable range saturate to 6. NaN is mapped to zero. Midpoints use
// round-to-nearest-even.
__host__ __device__ __forceinline__ std::uint8_t encode_e2m1_rn(
    float value) {
  if (value != value) {
    return 0u;
  }

  const bool negative = value < 0.0f;
  const float magnitude = negative ? -value : value;
  std::uint8_t bits = 0u;

  if (magnitude <= 0.25f) {
    bits = 0x0u;  // 0
  } else if (magnitude < 0.75f) {
    bits = 0x1u;  // 0.5
  } else if (magnitude <= 1.25f) {
    bits = 0x2u;  // 1
  } else if (magnitude < 1.75f) {
    bits = 0x3u;  // 1.5
  } else if (magnitude <= 2.5f) {
    bits = 0x4u;  // 2
  } else if (magnitude < 3.5f) {
    bits = 0x5u;  // 3
  } else if (magnitude <= 5.0f) {
    bits = 0x6u;  // 4
  } else {
    bits = 0x7u;  // 6, including infinity
  }

  return bits == 0u
             ? 0u
             : static_cast<std::uint8_t>(bits | (negative ? 0x8u : 0u));
}

}  // namespace gemm_sm110::requant
