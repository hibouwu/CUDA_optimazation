#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace gemm_sm110::requant {

// Match PTX packed floating-point ordering: the first logical value occupies
// the high nibble and the second logical value occupies the low nibble.
__host__ __device__ __forceinline__ std::uint8_t pack_e2m1x2(
    std::uint8_t value0, std::uint8_t value1) {
  return static_cast<std::uint8_t>(((value0 & 0x0fu) << 4) |
                                   (value1 & 0x0fu));
}

__host__ __device__ __forceinline__ std::uint16_t pack_e2m1x4(
    std::uint8_t value0, std::uint8_t value1, std::uint8_t value2,
    std::uint8_t value3) {
  return (static_cast<std::uint16_t>(pack_e2m1x2(value0, value1)) << 8) |
         static_cast<std::uint16_t>(pack_e2m1x2(value2, value3));
}

__host__ __device__ __forceinline__ std::uint32_t pack_e2m1x8(
    std::uint8_t value0, std::uint8_t value1, std::uint8_t value2,
    std::uint8_t value3, std::uint8_t value4, std::uint8_t value5,
    std::uint8_t value6, std::uint8_t value7) {
  return (static_cast<std::uint32_t>(
              pack_e2m1x4(value0, value1, value2, value3))
          << 16) |
         static_cast<std::uint32_t>(
             pack_e2m1x4(value4, value5, value6, value7));
}

}  // namespace gemm_sm110::requant
