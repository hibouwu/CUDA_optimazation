#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace gemm_sm110::requant {

// Encodes one normalized FP32 value as an E2M1 nibble:
//   bit 3: sign, bits 2:1: exponent, bit 0: mantissa.
//
// Finite magnitudes are {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Values outside the
// representable range saturate to 6. NaN is mapped to positive zero.
// Midpoints use round-to-nearest-even.  The sign bit is retained when a
// finite negative value rounds to zero, matching PTX/CUTLASS E2M1 signed-zero
// semantics.
__host__ __device__ __forceinline__ std::uint8_t encode_e2m1_rn(
    float value) {
  std::uint32_t raw = 0u;
#if defined(__CUDA_ARCH__)
  raw = __float_as_uint(value);
#else
  std::memcpy(&raw, &value, sizeof(raw));
#endif
  const std::uint32_t magnitude = raw & 0x7fffffffu;
  if (magnitude > 0x7f800000u) {
    return 0u;
  }

  // These bit thresholds encode the exact RNE midpoint policy.  A threshold
  // ending in ...001 excludes an exact midpoint whose lower E2M1 encoding is
  // even; a threshold at the exact midpoint selects the even upper encoding.
  const std::uint8_t encoded_magnitude = static_cast<std::uint8_t>(
      (magnitude >= 0x3e800001u) +  // just above 0.25
      (magnitude >= 0x3f400000u) +  // 0.75
      (magnitude >= 0x3fa00001u) +  // just above 1.25
      (magnitude >= 0x3fe00000u) +  // 1.75
      (magnitude >= 0x40200001u) +  // just above 2.5
      (magnitude >= 0x40600000u) +  // 3.5
      (magnitude >= 0x40a00001u));  // just above 5.0
  const std::uint8_t sign = static_cast<std::uint8_t>((raw >> 28) & 0x8u);
  return static_cast<std::uint8_t>(sign | encoded_magnitude);
}

}  // namespace gemm_sm110::requant
