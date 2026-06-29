#pragma once

#include "tc4_gemm_kernel.cuh"

// tc5(sm110/Thor) = CLC/persistent scheduling layer on top of TCGen05/TMEM.
//
// Boundary:
//   tc3 proves a minimal TCGen05/TMEM instruction path.
//   tc4 documents the full work-tile pipeline scaffold.
//   tc5 starts the Thor-specific scheduling path:
//     - persistent CTA workers
//     - static and dynamic CLC fallback tile assignment
//     - one TMEM allocation per worker, reused across work tiles
//
// This is still a probe, not a numerical GEMM.  It deliberately avoids copying
// the SM120 FP8 mma.sync path, because Thor/sm110 should target TCGen05/TMEM.

struct Tc5Sm110Shape {
  static constexpr int kBlockM = Tc4Sm110Shape::kBlockM;
  static constexpr int kBlockN = Tc4Sm110Shape::kBlockN;
  static constexpr int kThreads = 128;
  static constexpr unsigned int kTmemColumns = Tc4Sm110Shape::kTmemColumns;
};

struct Tc5Sm110WorkTile {
  int tile_id;
  int valid;
};

__host__ __device__ constexpr bool tc5_sm110_launch_available() {
  return tc3_sm110_tcgen05_available();
}

template <typename Shape>
__device__ __forceinline__ Tc5Sm110WorkTile tc5_sm110_static_fetch(
    int worker_id, int iter, int total_tiles) {
  const int tile_id = worker_id + iter * gridDim.x;
  return {tile_id, tile_id < total_tiles ? 1 : 0};
}

template <typename Shape>
__device__ __forceinline__ Tc5Sm110WorkTile tc5_sm110_dynamic_fetch(
    int* work_counter, int total_tiles) {
  const int tile_id = atomicAdd(work_counter, 1);
  return {tile_id, tile_id < total_tiles ? 1 : 0};
}

template <typename Shape>
__device__ __forceinline__ void tc5_sm110_tmem_alloc(unsigned int* tmem_base) {
#if TC3_SM110_HAS_TCGEN05
  if (threadIdx.x == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(tc3_sm110_smem_u32_ptr(tmem_base)), "r"(Shape::kTmemColumns)
        : "memory");
  }
#endif
}

template <typename Shape>
__device__ __forceinline__ void tc5_sm110_tmem_dealloc(unsigned int tmem_base) {
#if TC3_SM110_HAS_TCGEN05
  if (threadIdx.x == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :
                 : "r"(tmem_base), "r"(Shape::kTmemColumns)
                 : "memory");
  }
#endif
}

template <typename Shape>
__device__ __forceinline__ void tc5_sm110_tmem_probe_store_load(
    unsigned int tmem_base, float* c, Tc5Sm110WorkTile work) {
#if TC3_SM110_HAS_TCGEN05
  // tcgen05.st/ld are warp-collective operations. Restrict the probe to one
  // complete warp; executing them from lane 0 alone is undefined.
  if (threadIdx.x < kWarpSize && work.valid) {
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
    if ((threadIdx.x & (kWarpSize - 1)) == 0) {
      c[work.tile_id] = out;
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    c[0] = 0.0f;
  }
#endif
}

template <typename Shape = Tc5Sm110Shape>
__global__ void hgemm_tc5a_sm110_clc_static_tcgen05_tmem_persistent_probe(
    float* c, int total_tiles) {
  __shared__ unsigned int tmem_base;
  __shared__ Tc5Sm110WorkTile work;
  (void)tmem_base;
  (void)work;

#if TC3_SM110_HAS_TCGEN05
  tc5_sm110_tmem_alloc<Shape>(&tmem_base);
  __syncthreads();

  for (int iter = 0;; ++iter) {
    if (threadIdx.x == 0) {
      work = tc5_sm110_static_fetch<Shape>(static_cast<int>(blockIdx.x), iter,
                                           total_tiles);
    }
    __syncthreads();
    if (!work.valid) break;

    tc5_sm110_tmem_probe_store_load<Shape>(tmem_base, c, work);
    __syncthreads();
  }

  tc5_sm110_tmem_dealloc<Shape>(tmem_base);
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    c[0] = 0.0f;
  }
#endif
}

template <typename Shape = Tc5Sm110Shape>
__global__ void hgemm_tc5b_sm110_clc_dynamic_tcgen05_tmem_persistent_probe(
    float* c, int total_tiles, int* work_counter) {
  __shared__ unsigned int tmem_base;
  __shared__ Tc5Sm110WorkTile work;
  (void)tmem_base;
  (void)work;

#if TC3_SM110_HAS_TCGEN05
  tc5_sm110_tmem_alloc<Shape>(&tmem_base);
  __syncthreads();

  while (true) {
    if (threadIdx.x == 0) {
      work = tc5_sm110_dynamic_fetch<Shape>(work_counter, total_tiles);
    }
    __syncthreads();
    if (!work.valid) break;

    tc5_sm110_tmem_probe_store_load<Shape>(tmem_base, c, work);
    __syncthreads();
  }

  tc5_sm110_tmem_dealloc<Shape>(tmem_base);
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    c[0] = 0.0f;
  }
#endif
}
