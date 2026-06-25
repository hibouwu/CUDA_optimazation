#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// tc3 = SM120 narrow-precision MMA bring-up.
//
// This is intentionally not a GEMM performance kernel yet.  The goal is to
// prove the correct GeForce Blackwell instruction family first:
//
//   mma.sync.aligned.kind::f8f6f4
//   mma.sync.aligned.kind::mxf8f6f4.block_scale
//   mma.sync.aligned.kind::mxf4.block_scale
//
// These are the SM120 narrow/block-scaled MMA paths documented by CUTLASS.
// Do not use SM100/SM110 tcgen05/TMEM instructions for RTX 50-series sm_120.
//
// ptxas rejects kind::f8f6f4 for plain -arch=sm_120 in CUDA 13.0.  Build this
// probe with a family-specific target, for example:
//
//   CUDA_ARCH=120a ./scripts/run_gemm_backend.sh tc3

#ifndef TC3_COMPILED_SM120A_NARROW_MMA
#define TC3_COMPILED_SM120A_NARROW_MMA 0
#endif

#if TC3_COMPILED_SM120A_NARROW_MMA && defined(__CUDA_ARCH_FEAT_SM120_ALL)
#define TC3_HAS_SM120A_NARROW_MMA 1
#else
#define TC3_HAS_SM120A_NARROW_MMA 0
#endif

__host__ __device__ constexpr bool tc3_sm120a_narrow_mma_available() {
  return TC3_COMPILED_SM120A_NARROW_MMA != 0;
}

__global__ void hgemm_tc3_sm120a_f8f6f4_mma_probe(float* c) {
#if TC3_HAS_SM120A_NARROW_MMA
  if (threadIdx.x >= 32) return;

  uint32_t a0 = 0;
  uint32_t a1 = 0;
  uint32_t a2 = 0;
  uint32_t a3 = 0;
  uint32_t b0 = 0;
  uint32_t b1 = 0;
  float d0 = 0.0f;
  float d1 = 0.0f;
  float d2 = 0.0f;
  float d3 = 0.0f;

  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4."
      "f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0),
        "f"(d1), "f"(d2), "f"(d3));

  if (threadIdx.x == 0) {
    c[blockIdx.x] = 1.0f + d0 + d1 + d2 + d3;
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    c[0] = 0.0f;
  }
#endif
}
