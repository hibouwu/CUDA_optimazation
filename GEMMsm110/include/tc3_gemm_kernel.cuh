#pragma once

#include "gemm_common.cuh"

#include <cuda_runtime.h>
#include <cstdint>

// tc3(sm110) = TCGen05 / TMEM bring-up.
//
// This probe is intentionally narrow: it proves that the sm110a build can
// issue tcgen05 TMEM allocate/store/load/deallocate instructions. It is not a
// numerical GEMM kernel yet, and it does not include the full TMA->SMEM->MMA
// descriptor path.

#if defined(__CUDA_ARCH_FEAT_SM110_ALL)
#define TC3_SM110_HAS_TCGEN05 1
#else
#define TC3_SM110_HAS_TCGEN05 0
#endif

struct Tc3Sm110Shape {
  static constexpr int kThreads = 128;
  static constexpr unsigned int kTmemColumns = 128;
};

__host__ __device__ constexpr bool tc3_sm110_tcgen05_available() {
  return TC3_SM110_HAS_TCGEN05 != 0;
}

__device__ __forceinline__ unsigned int tc3_sm110_smem_u32_ptr(const void* p) {
  return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

template <typename Shape = Tc3Sm110Shape>
__global__ void hgemm_tc3_sm110_tcgen05_tmem_probe(float* c, int tiles) {
  __shared__ unsigned int tmem_base;
  (void)tmem_base;

#if TC3_SM110_HAS_TCGEN05
  if (threadIdx.x == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(tc3_sm110_smem_u32_ptr(&tmem_base)), "r"(Shape::kTmemColumns)
        : "memory");

    const unsigned int one_bits = 0x3f800000u;
    asm volatile("tcgen05.st.sync.aligned.16x64b.x1.b32 [%0], {%1};"
                 :
                 : "r"(tmem_base), "r"(one_bits)
                 : "memory");

    unsigned int loaded = 0u;
    asm volatile("tcgen05.ld.sync.aligned.16x64b.x1.b32 {%0}, [%1];"
                 : "=r"(loaded)
                 : "r"(tmem_base)
                 : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

    float out = 0.0f;
    *reinterpret_cast<unsigned int*>(&out) = loaded;
    if (blockIdx.x < tiles) {
      c[blockIdx.x] = out;
    }

    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :
                 : "r"(tmem_base), "r"(Shape::kTmemColumns)
                 : "memory");
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    c[0] = 0.0f;
  }
#endif
}
