#pragma once

#include "gemm_common.cuh"

#include <cuda/barrier>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <new>

namespace tc3_cde = cuda::device::experimental;

// tc3 = SM120a FP8 MMA GEMM bring-up.
//
// This is a correctness/perf bring-up kernel, not an optimized GEMM.  It proves
// the GeForce Blackwell narrow-MMA data path with a real GEMM workload:
//
//   mma.sync.aligned.kind::f8f6f4
//   mma.sync.aligned.kind::mxf8f6f4.block_scale
//   mma.sync.aligned.kind::mxf4.block_scale
//
// These are the SM120 narrow/block-scaled MMA paths documented by CUTLASS.
// Do not use SM100/SM110 tcgen05/TMEM instructions for RTX 50-series sm_120.
//
// ptxas rejects kind::f8f6f4 for plain -arch=sm_120 in CUDA 13.0.  Build this
// kernel with a family-specific target, for example:
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

struct Tc3Fp8Shape {
  static constexpr int kBlockM = 128;
  static constexpr int kBlockN = 64;
  static constexpr int kBlockK = 32;
  static constexpr int kStages = 2;
  static constexpr int kWarps = 8;
};

using tc3_block_barrier = cuda::barrier<cuda::thread_scope_block>;
struct alignas(tc3_block_barrier) tc3_block_barrier_storage {
  unsigned char bytes[sizeof(tc3_block_barrier)];
};

inline void tc3_encode_rowmajor_tensor_map_fp8(CUtensorMap& tensor_map,
                                               __nv_fp8_e4m3* global_address,
                                               int rows, int cols, int box_rows,
                                               int box_cols) {
  const cuuint64_t global_dim[2] = {static_cast<cuuint64_t>(cols),
                                    static_cast<cuuint64_t>(rows)};
  const cuuint64_t global_strides[1] = {static_cast<cuuint64_t>(cols)};
  const cuuint32_t box_dim[2] = {static_cast<cuuint32_t>(box_cols),
                                 static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t element_strides[2] = {1u, 1u};
  CHECK_CU(cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, global_address,
      global_dim, global_strides, box_dim, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

__device__ __forceinline__ uint32_t tc3_pack_fp8x4(const uint8_t* bytes,
                                                   int stride, int row0,
                                                   int col0) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |= static_cast<uint32_t>(bytes[row0 * stride + col0 + i]) << (8 * i);
  }
  return out;
}

template <int Rows, int Cols>
__device__ __forceinline__ void tc3_copy_tile_u8_fallback(
    int tid, int ld, const uint8_t* src, uint8_t* dst) {
  for (int elem = tid; elem < Rows * Cols; elem += blockDim.x) {
    const int row = elem / Cols;
    const int col = elem % Cols;
    dst[row * Cols + col] = src[row * ld + col];
  }
}

template <int BM, int BK>
__device__ __forceinline__ tc3_block_barrier::arrival_token
tc3_launch_a_tile_tma(int tid, int block_row, int tile_k,
                      const __nv_fp8_e4m3* a, int lda,
                      const CUtensorMap* a_map, uint8_t* a_smem,
                      tc3_block_barrier& bar) {
#if __CUDA_ARCH__ >= 900
  tc3_block_barrier::arrival_token token{};
  if (tid == 0) {
    cuda::device::barrier_expect_tx(bar, BM * BK);
    tc3_cde::cp_async_bulk_tensor_2d_global_to_shared(a_smem, a_map, tile_k,
                                                      block_row, bar);
    tc3_cde::cp_async_bulk_commit_group();
    token = bar.arrive();
  }
  return token;
#else
  tc3_copy_tile_u8_fallback<BM, BK>(
      tid, lda,
      reinterpret_cast<const uint8_t*>(a) + block_row * lda + tile_k, a_smem);
  return {};
#endif
}

template <int BK, int BN>
__device__ __forceinline__ tc3_block_barrier::arrival_token
tc3_launch_b_tile_tma(int tid, int block_col, int tile_k,
                      const __nv_fp8_e4m3* b, int ldb,
                      const CUtensorMap* b_map, uint8_t* b_smem,
                      tc3_block_barrier& bar) {
#if __CUDA_ARCH__ >= 900
  tc3_block_barrier::arrival_token token{};
  if (tid == 0) {
    cuda::device::barrier_expect_tx(bar, BK * BN);
    tc3_cde::cp_async_bulk_tensor_2d_global_to_shared(b_smem, b_map, block_col,
                                                      tile_k, bar);
    tc3_cde::cp_async_bulk_commit_group();
    token = bar.arrive();
  }
  return token;
#else
  tc3_copy_tile_u8_fallback<BK, BN>(
      tid, ldb,
      reinterpret_cast<const uint8_t*>(b) + tile_k * ldb + block_col, b_smem);
  return {};
#endif
}

__device__ __forceinline__ void tc3_wait_tma_stage(
    int tid, tc3_block_barrier& a_bar,
    tc3_block_barrier::arrival_token&& a_token, tc3_block_barrier& b_bar,
    tc3_block_barrier::arrival_token&& b_token) {
#if __CUDA_ARCH__ >= 900
  if (tid == 0) {
    a_bar.wait(static_cast<tc3_block_barrier::arrival_token&&>(a_token));
    b_bar.wait(static_cast<tc3_block_barrier::arrival_token&&>(b_token));
  }
  __syncthreads();
  tc3_cde::fence_proxy_async_shared_cta();
  __syncthreads();
#else
  __syncthreads();
#endif
}

constexpr size_t tc3_tma_fp8_smem_bytes() {
  return static_cast<size_t>(Tc3Fp8Shape::kStages) * Tc3Fp8Shape::kBlockK *
         (Tc3Fp8Shape::kBlockM + Tc3Fp8Shape::kBlockN);
}

__global__ void hgemm_tc3_sm120a_fp8_tma_mma_128x64x32(
    int m, int n, int k, float alpha, const __nv_fp8_e4m3* a,
    const __grid_constant__ CUtensorMap* const a_map, const __nv_fp8_e4m3* b,
    const __grid_constant__ CUtensorMap* const b_map, float beta, float* c) {
#if TC3_HAS_SM120A_NARROW_MMA
  using Shape = Tc3Fp8Shape;
  extern __shared__ __align__(128) unsigned char tc3_smem[];
  uint8_t* a_smem = reinterpret_cast<uint8_t*>(tc3_smem);
  uint8_t* b_smem = a_smem + Shape::kStages * Shape::kBlockM * Shape::kBlockK;
  __shared__ tc3_block_barrier_storage a_bar_storage[Shape::kStages];
  __shared__ tc3_block_barrier_storage b_bar_storage[Shape::kStages];

  auto* a_bar = reinterpret_cast<tc3_block_barrier*>(a_bar_storage);
  auto* b_bar = reinterpret_cast<tc3_block_barrier*>(b_bar_storage);

  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int stage = 0; stage < Shape::kStages; ++stage) {
      init(&a_bar[stage], 1);
      init(&b_bar[stage], 1);
    }
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int tile_m = blockIdx.y * Shape::kBlockM;
  const int tile_n = blockIdx.x * Shape::kBlockN;
  const int warp_m = tile_m + warp_id * 16;
  const int num_tiles = k / Shape::kBlockK;

  float d[8][4];
#pragma unroll
  for (int col_tile = 0; col_tile < 8; ++col_tile) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      d[col_tile][i] = 0.0f;
    }
  }

  tc3_block_barrier::arrival_token a_tokens[Shape::kStages];
  tc3_block_barrier::arrival_token b_tokens[Shape::kStages];
  const int preload_tiles = num_tiles < Shape::kStages ? num_tiles
                                                       : Shape::kStages;
  for (int preload = 0; preload < preload_tiles; ++preload) {
    const int stage = preload;
    a_tokens[stage] = tc3_launch_a_tile_tma<Shape::kBlockM, Shape::kBlockK>(
        tid, tile_m, preload * Shape::kBlockK, a, k, a_map,
        a_smem + stage * Shape::kBlockM * Shape::kBlockK, a_bar[stage]);
    b_tokens[stage] = tc3_launch_b_tile_tma<Shape::kBlockK, Shape::kBlockN>(
        tid, tile_n, preload * Shape::kBlockK, b, n, b_map,
        b_smem + stage * Shape::kBlockK * Shape::kBlockN, b_bar[stage]);
  }

  for (int tile = 0; tile < num_tiles; ++tile) {
    const int stage = tile % Shape::kStages;
    tc3_wait_tma_stage(
        tid, a_bar[stage],
        static_cast<tc3_block_barrier::arrival_token&&>(a_tokens[stage]),
        b_bar[stage],
        static_cast<tc3_block_barrier::arrival_token&&>(b_tokens[stage]));

    const uint8_t* a_stage =
        a_smem + stage * Shape::kBlockM * Shape::kBlockK;
    const uint8_t* b_stage =
        b_smem + stage * Shape::kBlockK * Shape::kBlockN;

    const int smem_row = warp_id * 16 + group;
    const uint32_t a0 =
        tc3_pack_fp8x4(a_stage, Shape::kBlockK, smem_row, thread_in_group * 4);
    const uint32_t a1 =
        tc3_pack_fp8x4(a_stage, Shape::kBlockK, smem_row + 8,
                       thread_in_group * 4);
    const uint32_t a2 =
        tc3_pack_fp8x4(a_stage, Shape::kBlockK, smem_row,
                       thread_in_group * 4 + 16);
    const uint32_t a3 =
        tc3_pack_fp8x4(a_stage, Shape::kBlockK, smem_row + 8,
                       thread_in_group * 4 + 16);

#pragma unroll
    for (int col_tile = 0; col_tile < 8; ++col_tile) {
      uint32_t b0 = 0;
      uint32_t b1 = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        b0 |= static_cast<uint32_t>(
                  b_stage[(thread_in_group * 4 + i) * Shape::kBlockN +
                          col_tile * 8 + group])
              << (8 * i);
        b1 |= static_cast<uint32_t>(
                  b_stage[(thread_in_group * 4 + i + 16) * Shape::kBlockN +
                          col_tile * 8 + group])
              << (8 * i);
      }

      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4."
          "f32.e4m3.e4m3.f32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5, %6, %7}, "
          "{%8, %9}, "
          "{%0, %1, %2, %3};\n"
          : "+f"(d[col_tile][0]), "+f"(d[col_tile][1]),
            "+f"(d[col_tile][2]), "+f"(d[col_tile][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __syncthreads();

    const int next_tile = tile + Shape::kStages;
    if (next_tile < num_tiles) {
      a_tokens[stage] = tc3_launch_a_tile_tma<Shape::kBlockM, Shape::kBlockK>(
          tid, tile_m, next_tile * Shape::kBlockK, a, k, a_map,
          a_smem + stage * Shape::kBlockM * Shape::kBlockK, a_bar[stage]);
      b_tokens[stage] = tc3_launch_b_tile_tma<Shape::kBlockK, Shape::kBlockN>(
          tid, tile_n, next_tile * Shape::kBlockK, b, n, b_map,
          b_smem + stage * Shape::kBlockK * Shape::kBlockN, b_bar[stage]);
    }
  }

  const int row0 = warp_m + group;
  const int row1 = warp_m + group + 8;
#pragma unroll
  for (int col_tile = 0; col_tile < 8; ++col_tile) {
    const int col0 = tile_n + col_tile * 8 + thread_in_group * 2;
    if (row0 < m && col0 + 1 < n) {
      c[row0 * n + col0] =
          alpha * d[col_tile][0] + beta * c[row0 * n + col0];
      c[row0 * n + col0 + 1] =
          alpha * d[col_tile][1] + beta * c[row0 * n + col0 + 1];
    }
    if (row1 < m && col0 + 1 < n) {
      c[row1 * n + col0] =
          alpha * d[col_tile][2] + beta * c[row1 * n + col0];
      c[row1 * n + col0 + 1] =
          alpha * d[col_tile][3] + beta * c[row1 * n + col0 + 1];
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    c[0] = 0.0f;
  }
#endif
}
