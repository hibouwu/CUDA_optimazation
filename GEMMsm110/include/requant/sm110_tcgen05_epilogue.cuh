#pragma once

#include "scale_policy.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace gemm_sm110::requant {

#if defined(__CUDA_ARCH_FEAT_SM110_ALL)
#define GEMM_SM110_HAS_NATIVE_REQUANT 1
#else
#define GEMM_SM110_HAS_NATIVE_REQUANT 0
#endif

struct Sm110TmemAccumulatorPair {
  float value0;
  float value1;
};

struct Sm110Nvfp4BlockScale {
  // One E4M3 value. The conversion instruction produces e4m3x2 with the same
  // value duplicated; only one byte needs to be stored for the 16-value block.
  std::uint8_t e4m3_bits;
  float decoded;
};

// Collective across one complete warp. Every lane must provide the same TMEM
// address and execute this function without divergence.
__device__ __forceinline__ Sm110TmemAccumulatorPair
sm110_tcgen05_load_32x32b_x2(unsigned int tmem_address) {
  Sm110TmemAccumulatorPair accumulator{0.0f, 0.0f};

#if GEMM_SM110_HAS_NATIVE_REQUANT
  unsigned int value0_bits = 0u;
  unsigned int value1_bits = 0u;
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];"
      : "=r"(value0_bits), "=r"(value1_bits)
      : "r"(tmem_address)
      : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

  accumulator.value0 = __uint_as_float(value0_bits);
  accumulator.value1 = __uint_as_float(value1_bits);
#endif

  return accumulator;
}

__device__ __forceinline__ float sm110_decode_positive_e4m3_scale(
    std::uint8_t e4m3_bits) {
#if GEMM_SM110_HAS_NATIVE_REQUANT
  const unsigned short packed_e4m3x2 =
      static_cast<unsigned short>(e4m3_bits) |
      (static_cast<unsigned short>(e4m3_bits) << 8);
  unsigned int packed_f16x2 = 0u;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
               : "=r"(packed_f16x2)
               : "h"(packed_e4m3x2));
  return __half2float(
      __ushort_as_half(static_cast<unsigned short>(packed_f16x2)));
#else
  (void)e4m3_bits;
  return 0.0f;
#endif
}

// Quantize block_amax(x / s_global) / 6 upward to E4M3, then decode the
// stored scale back to FP32 for the following E2M1 conversion. Rounding the
// positive scale upward prevents the largest value in the block from
// exceeding the E2M1 finite maximum after normalization.
__device__ __forceinline__ Sm110Nvfp4BlockScale
sm110_make_nvfp4_block_scale(float normalized_block_amax) {
  Sm110Nvfp4BlockScale result{0u, 0.0f};

#if GEMM_SM110_HAS_NATIVE_REQUANT
  if (!(normalized_block_amax > 0.0f)) {
    return result;
  }

  constexpr float kE2M1Max = 6.0f;
  const float unquantized_scale = normalized_block_amax / kE2M1Max;
  unsigned short packed_e4m3x2 = 0u;
  asm volatile(
      "{\n"
      "  .reg .b16 fp8_scale;\n"
      "  cvt.rn.satfinite.e4m3x2.f32 fp8_scale, %1, %1;\n"
      "  mov.b16 %0, fp8_scale;\n"
      "}"
      : "=h"(packed_e4m3x2)
      : "f"(unquantized_scale));

  result.e4m3_bits = static_cast<std::uint8_t>(packed_e4m3x2);
  result.decoded = sm110_decode_positive_e4m3_scale(result.e4m3_bits);

  // Positive finite E4M3 encodings are monotonic. 0x7e is +448, the largest
  // finite E4M3 value; 0x7f is NaN and must not be produced.
  if (result.decoded < unquantized_scale && result.e4m3_bits < 0x7eu) {
    ++result.e4m3_bits;
    result.decoded = sm110_decode_positive_e4m3_scale(result.e4m3_bits);
  }
#else
  (void)normalized_block_amax;
#endif

  return result;
}

// Native Blackwell conversion. The two E2M1 values are returned packed in the
// low byte of the destination register.
__device__ __forceinline__ std::uint8_t sm110_cvt_e2m1x2_rn_satfinite(
    float value0, float value1) {
#if GEMM_SM110_HAS_NATIVE_REQUANT
  unsigned short packed = 0u;
  asm volatile(
      "{\n"
      "  .reg .b8 fp4;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 fp4, %1, %2;\n"
      "  cvt.u16.u8 %0, fp4;\n"
      "}"
      : "=h"(packed)
      : "f"(value0), "f"(value1));
  return static_cast<std::uint8_t>(packed);
#else
  (void)value0;
  (void)value1;
  return 0u;
#endif
}

// Quantize two accumulator values using the NVFP4 hierarchical scale:
//   x_normalized = x / s_global
//   x_e2m1       = E2M1(x_normalized / s_block)
//
// block_scale.decoded must be the FP32 value decoded from the stored E4M3
// scale, not the unrounded FP32 scale.
__device__ __forceinline__ std::uint8_t sm110_requant_nvfp4_e2m1x2(
    float value0, float value1, float inverse_tensor_scale,
    Sm110Nvfp4BlockScale block_scale) {
#if GEMM_SM110_HAS_NATIVE_REQUANT
  if (!(block_scale.decoded > 0.0f)) {
    return 0u;
  }

  const float inverse_block_scale = 1.0f / block_scale.decoded;
  const float quant_multiplier =
      inverse_tensor_scale * inverse_block_scale;
  return sm110_cvt_e2m1x2_rn_satfinite(value0 * quant_multiplier,
                                       value1 * quant_multiplier);
#else
  (void)value0;
  (void)value1;
  (void)inverse_tensor_scale;
  (void)block_scale;
  return 0u;
#endif
}

template <typename ScalePolicy>
struct Sm110Tcgen05E2M1Epilogue {
  ScalePolicy scale;

  __host__ __device__ explicit Sm110Tcgen05E2M1Epilogue(ScalePolicy scale_)
      : scale(scale_) {}

  // This primitive performs the SM110-specific TMEM -> registers -> E2M1
  // portion of the epilogue. Matrix-coordinate mapping and the final
  // cooperative store remain the responsibility of the enclosing GEMM
  // epilogue.
  __device__ __forceinline__ std::uint8_t load_requant_x2(
      unsigned int tmem_address, int linear_index0,
      int linear_index1) const {
    const Sm110TmemAccumulatorPair accumulator =
        sm110_tcgen05_load_32x32b_x2(tmem_address);
    const float normalized0 =
        scale.normalize(accumulator.value0, linear_index0);
    const float normalized1 =
        scale.normalize(accumulator.value1, linear_index1);
    return sm110_cvt_e2m1x2_rn_satfinite(normalized0, normalized1);
  }
};

struct Sm110Tcgen05Nvfp4Epilogue {
  // Collective across one complete warp. The caller computes one block scale
  // per 16 logical output values and supplies the scale corresponding to this
  // lane/register pair.
  __device__ __forceinline__ std::uint8_t load_requant_x2(
      unsigned int tmem_address, float inverse_tensor_scale,
      Sm110Nvfp4BlockScale block_scale) const {
    const Sm110TmemAccumulatorPair accumulator =
        sm110_tcgen05_load_32x32b_x2(tmem_address);
    return sm110_requant_nvfp4_e2m1x2(
        accumulator.value0, accumulator.value1, inverse_tensor_scale,
        block_scale);
  }
};

}  // namespace gemm_sm110::requant
