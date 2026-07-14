#pragma once

#include "e2m1_encode.cuh"
#include "pack_fp4.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace gemm_sm110::requant {

constexpr int kNvfp4BlockSize = 16;
constexpr float kNvfp4E2M1Max = 6.0f;
constexpr float kNvfp4E4M3Max = 448.0f;

struct Nvfp4ReferenceResult {
  std::vector<std::uint8_t> quantized;
  std::vector<std::uint8_t> block_scales;
  float tensor_scale = 1.0f;
};

inline std::size_t nvfp4_quantized_bytes(std::size_t elements) {
  return (elements + 1) / 2;
}

inline std::size_t nvfp4_block_count(std::size_t elements) {
  return (elements + kNvfp4BlockSize - 1) / kNvfp4BlockSize;
}

inline std::size_t nvfp4_padded_elements(std::size_t elements) {
  return nvfp4_block_count(elements) * kNvfp4BlockSize;
}

inline float decode_positive_e4m3(std::uint8_t bits) {
  const int exponent = (bits >> 3) & 0x0f;
  const int mantissa = bits & 0x07;
  if (exponent == 0) {
    return std::ldexp(static_cast<float>(mantissa), -9);
  }
  if (exponent == 0x0f && mantissa == 0x07) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                    exponent - 7);
}

inline float decode_e2m1(std::uint8_t bits) {
  static constexpr float magnitudes[8] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = magnitudes[bits & 0x07u];
  return (bits & 0x08u) != 0u ? -magnitude : magnitude;
}

inline std::uint8_t encode_positive_e4m3_round_up(float value) {
  if (!(value > 0.0f)) return 0u;
  for (int bits = 1; bits <= 0x7e; ++bits) {
    if (decode_positive_e4m3(static_cast<std::uint8_t>(bits)) >= value) {
      return static_cast<std::uint8_t>(bits);
    }
  }
  return 0x7eu;
}

inline float make_nvfp4_tensor_scale(const std::vector<float>& input) {
  float global_amax = 0.0f;
  for (float value : input) {
    global_amax = std::max(global_amax, std::fabs(value));
  }
  return global_amax > 0.0f
             ? global_amax / (kNvfp4E4M3Max * kNvfp4E2M1Max)
             : 1.0f;
}

inline Nvfp4ReferenceResult make_nvfp4_reference(
    const std::vector<float>& input) {
  Nvfp4ReferenceResult result;
  const std::size_t logical_elements = input.size();
  const std::size_t padded_elements = nvfp4_padded_elements(logical_elements);
  result.quantized.assign(nvfp4_quantized_bytes(padded_elements), 0u);
  result.block_scales.assign(nvfp4_block_count(padded_elements), 0u);
  result.tensor_scale = make_nvfp4_tensor_scale(input);
  const float inverse_tensor_scale = 1.0f / result.tensor_scale;

  for (std::size_t block = 0; block < result.block_scales.size(); ++block) {
    const std::size_t begin = block * kNvfp4BlockSize;
    float normalized_amax = 0.0f;
    float values[kNvfp4BlockSize]{};
    for (int i = 0; i < kNvfp4BlockSize; ++i) {
      const std::size_t index = begin + i;
      values[i] = index < logical_elements ? input[index] : 0.0f;
      normalized_amax =
          std::max(normalized_amax,
                   std::fabs(values[i] * inverse_tensor_scale));
    }

    const std::uint8_t scale_bits =
        encode_positive_e4m3_round_up(normalized_amax / kNvfp4E2M1Max);
    result.block_scales[block] = scale_bits;
    const float block_scale = decode_positive_e4m3(scale_bits);
    const float multiplier =
        block_scale > 0.0f ? inverse_tensor_scale / block_scale : 0.0f;

    for (int i = 0; i < kNvfp4BlockSize; i += 2) {
      const std::uint8_t value0 = encode_e2m1_rn(values[i] * multiplier);
      const std::uint8_t value1 = encode_e2m1_rn(values[i + 1] * multiplier);
      result.quantized[(begin + i) / 2] = pack_e2m1x2(value0, value1);
    }
  }
  return result;
}

}  // namespace gemm_sm110::requant
