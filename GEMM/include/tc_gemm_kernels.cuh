#pragma once

#include "gemm_common.cuh"

#include <cuda/barrier>
#include <cuda_fp16.h>
#include <mma.h>
#include <new>

using namespace nvcuda;
namespace cde = cuda::device::experimental;

template <typename T>
using wmma_a_rowmajor_16 = wmma::fragment<wmma::matrix_a, 16, 16, 16, T,
                                          wmma::row_major>;
template <typename T>
using wmma_b_rowmajor_16 = wmma::fragment<wmma::matrix_b, 16, 16, 16, T,
                                          wmma::row_major>;
using wmma_accum_f32_16 = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ void tc_apply_epilogue(
    float alpha, float beta, wmma_accum_f32_16& c_frag,
    const wmma_accum_f32_16& old_c_frag) {
  if (alpha == 1.0f && beta == 0.0f) return;
#pragma unroll
  for (int i = 0; i < c_frag.num_elements; ++i) {
    c_frag.x[i] =
        alpha * c_frag.x[i] + (beta != 0.0f ? beta * old_c_frag.x[i] : 0.0f);
  }
}

template <typename Shape>
__device__ __forceinline__ void tc_warp_coords(int warp_id, int& warp_row,
                                               int& warp_col) {
  warp_row = warp_id / Shape::kWarpTilesN;
  warp_col = warp_id % Shape::kWarpTilesN;
}

struct Tc1Shape {
  static constexpr int kTileM = 16;
  static constexpr int kTileN = 16;
  static constexpr int kTileK = 16;
};

struct Tc2PipelineShape {
  static constexpr int kBlockM = 128;
  static constexpr int kBlockN = 64;
  static constexpr int kBlockK = 32;
  static constexpr int kWarpTilesN = 2;
};

template <int Stages>
constexpr size_t tc_tma_wmma_smem_bytes() {
  return static_cast<size_t>(Stages) * Tc2PipelineShape::kBlockK *
         (Tc2PipelineShape::kBlockM + Tc2PipelineShape::kBlockN) *
         sizeof(half);
}

using tc_block_barrier = cuda::barrier<cuda::thread_scope_block>;
struct alignas(tc_block_barrier) tc_block_barrier_storage {
  unsigned char bytes[sizeof(tc_block_barrier)];
};

inline void tc_encode_rowmajor_tensor_map_2d(CUtensorMap& tensor_map,
                                             half* global_address, int rows,
                                             int cols, int box_rows,
                                             int box_cols,
                                             CUtensorMapSwizzle swizzle =
                                                 CU_TENSOR_MAP_SWIZZLE_NONE) {
  const cuuint64_t global_dim[2] = {static_cast<cuuint64_t>(cols),
                                    static_cast<cuuint64_t>(rows)};
  const cuuint64_t global_strides[1] = {
      static_cast<cuuint64_t>(cols * sizeof(half))};
  const cuuint32_t box_dim[2] = {static_cast<cuuint32_t>(box_cols),
                                 static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t element_strides[2] = {1u, 1u};
  CHECK_CU(cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, global_address,
      global_dim, global_strides, box_dim, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

template <int Rows, int Cols>
__device__ __forceinline__ void tc2_copy_tile_fallback(int tid, int ld,
                                                       const half* src,
                                                       half* dst) {
  for (int elem = tid; elem < Rows * Cols; elem += blockDim.x) {
    const int row = elem / Cols;
    const int col = elem % Cols;
    dst[row * Cols + col] = src[row * ld + col];
  }
}

template <int BM, int BK>
__device__ __forceinline__ tc_block_barrier::arrival_token tc2_launch_a_tile_tma(
    int tid, int block_row, int tile_k, const half* a, int lda,
    const CUtensorMap* a_map, half* a_smem, tc_block_barrier& bar) {
#if __CUDA_ARCH__ >= 900
  tc_block_barrier::arrival_token token{};
  if (tid == 0) {
    cuda::device::barrier_expect_tx(bar, BM * BK * sizeof(half));
    cde::cp_async_bulk_tensor_2d_global_to_shared(a_smem, a_map, tile_k,
                                                  block_row, bar);
    cde::cp_async_bulk_commit_group();
    token = bar.arrive();
  }
  return token;
#else
  tc2_copy_tile_fallback<BM, BK>(tid, lda, a + block_row * lda + tile_k, a_smem);
  return {};
#endif
}

template <int BK, int BN>
__device__ __forceinline__ tc_block_barrier::arrival_token tc2_launch_b_tile_tma(
    int tid, int block_col, int tile_k, const half* b, int ldb,
    const CUtensorMap* b_map, half* b_smem, tc_block_barrier& bar) {
#if __CUDA_ARCH__ >= 900
  tc_block_barrier::arrival_token token{};
  if (tid == 0) {
    cuda::device::barrier_expect_tx(bar, BK * BN * sizeof(half));
    cde::cp_async_bulk_tensor_2d_global_to_shared(b_smem, b_map, block_col,
                                                  tile_k, bar);
    cde::cp_async_bulk_commit_group();
    token = bar.arrive();
  }
  return token;
#else
  tc2_copy_tile_fallback<BK, BN>(tid, ldb, b + tile_k * ldb + block_col, b_smem);
  return {};
#endif
}

__device__ __forceinline__ void tc2_wait_tma_stage(
    int tid, tc_block_barrier& a_bar, tc_block_barrier::arrival_token&& a_token,
    tc_block_barrier& b_bar, tc_block_barrier::arrival_token&& b_token) {
#if __CUDA_ARCH__ >= 900
  if (tid == 0) {
    a_bar.wait(static_cast<tc_block_barrier::arrival_token&&>(a_token));
    b_bar.wait(static_cast<tc_block_barrier::arrival_token&&>(b_token));
  }
  __syncthreads();
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
#else
  __syncthreads();
#endif
}

__device__ __forceinline__ void tc2_apply_epilogue_2x2(
    float alpha, float beta, wmma_accum_f32_16 c_frag[2][2],
    const wmma_accum_f32_16 old_c_frag[2][2]) {
  if (alpha == 1.0f && beta == 0.0f) return;
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
      for (int i = 0; i < c_frag[mi][ni].num_elements; ++i) {
        c_frag[mi][ni].x[i] =
            alpha * c_frag[mi][ni].x[i] +
            (beta != 0.0f ? beta * old_c_frag[mi][ni].x[i] : 0.0f);
      }
    }
  }
}

// tc1: WMMA FP16 Tensor Core baseline。
// 一个 warp 计算一个 16x16 的 C tile，A/B 为 row-major FP16，累加到 FP32。
// 这是 Tensor Core 路径的最小正确版本，先要求 m/n/k 都是 16 的倍数。
__global__ void hgemm_tc1_wmma_16x16(int m, int n, int k, float alpha,
                                     const half* a, const half* b, float beta,
                                     float* c) {
  const int row = blockIdx.y * Tc1Shape::kTileM;
  const int col = blockIdx.x * Tc1Shape::kTileN;

  wmma_a_rowmajor_16<half> a_frag;
  wmma_b_rowmajor_16<half> b_frag;
  wmma_accum_f32_16 c_frag;
  wmma_accum_f32_16 old_c_frag;

  wmma::fill_fragment(c_frag, 0.0f);
  if (beta != 0.0f) {
    wmma::load_matrix_sync(old_c_frag, c + row * n + col, n,
                           wmma::mem_row_major);
  }

  for (int p = 0; p < k; p += Tc1Shape::kTileK) {
    wmma::load_matrix_sync(a_frag, a + row * k + p, k);
    wmma::load_matrix_sync(b_frag, b + p * n + col, n);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  tc_apply_epilogue(alpha, beta, c_frag, old_c_frag);

  wmma::store_matrix_sync(c + row * n + col, c_frag, n, wmma::mem_row_major);
}

// TMA + WMMA staged mainloop.
// Persistent=false 用普通 2D grid；Persistent=true 用 1D grid-stride 静态调度 C
// tiles，方便比较 persistent CTA 对调度和 locality 的影响。
template <int Stages, bool Persistent = false>
__global__ void hgemm_tc_tma_wmma_128x64x32(
    int m, int n, int k, float alpha, const half* a,
    const __grid_constant__ CUtensorMap* const a_map, const half* b,
    const __grid_constant__ CUtensorMap* const b_map, float beta, float* c) {
  static_assert(Stages >= 2, "TMA pipeline needs at least two stages");
  using Shape = Tc2PipelineShape;

  extern __shared__ __align__(128) unsigned char tc_smem[];
  half* as = reinterpret_cast<half*>(tc_smem);
  half* bs = as + Stages * Shape::kBlockM * Shape::kBlockK;
  __shared__ tc_block_barrier_storage a_bar_storage[Stages];
  __shared__ tc_block_barrier_storage b_bar_storage[Stages];

  auto* a_bar = reinterpret_cast<tc_block_barrier*>(a_bar_storage);
  auto* b_bar = reinterpret_cast<tc_block_barrier*>(b_bar_storage);

  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int stage = 0; stage < Stages; ++stage) {
      init(&a_bar[stage], 1);
      init(&b_bar[stage], 1);
    }
  }
  __syncthreads();

  const int warp_id = tid / kWarpSize;
  int warp_row = 0;
  int warp_col = 0;
  tc_warp_coords<Shape>(warp_id, warp_row, warp_col);

  const int total_tiles_n = n / Shape::kBlockN;
  const int total_tiles_m = m / Shape::kBlockM;
  const int total_output_tiles = total_tiles_m * total_tiles_n;
  const int first_output_tile =
      Persistent ? static_cast<int>(blockIdx.x)
                 : static_cast<int>(blockIdx.y) * total_tiles_n +
                       static_cast<int>(blockIdx.x);
  const int output_tile_stride = Persistent ? static_cast<int>(gridDim.x)
                                            : total_output_tiles;
  const int num_tiles = k / Shape::kBlockK;

  for (int output_tile = first_output_tile; output_tile < total_output_tiles;
       output_tile += output_tile_stride) {
    const int tile_m = output_tile / total_tiles_n;
    const int tile_n = output_tile - tile_m * total_tiles_n;
    const int block_row = tile_m * Shape::kBlockM;
    const int block_col = tile_n * Shape::kBlockN;
    const int c_row = block_row + warp_row * 32;
    const int c_col = block_col + warp_col * 32;

    wmma_a_rowmajor_16<half> a_frag[2];
    wmma_b_rowmajor_16<half> b_frag[2];
    wmma_accum_f32_16 c_frag[2][2];
    wmma_accum_f32_16 old_c_frag[2][2];

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
      for (int ni = 0; ni < 2; ++ni) {
        wmma::fill_fragment(c_frag[mi][ni], 0.0f);
      }
    }
    if (beta != 0.0f) {
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
          wmma::load_matrix_sync(old_c_frag[mi][ni],
                                 c + (c_row + mi * 16) * n + c_col + ni * 16,
                                 n, wmma::mem_row_major);
        }
      }
    }

    tc_block_barrier::arrival_token a_tokens[Stages];
    tc_block_barrier::arrival_token b_tokens[Stages];
    const int preload_tiles = num_tiles < Stages ? num_tiles : Stages;

    for (int preload = 0; preload < preload_tiles; ++preload) {
      half* a_stage = as + preload * Shape::kBlockM * Shape::kBlockK;
      half* b_stage = bs + preload * Shape::kBlockK * Shape::kBlockN;
      a_tokens[preload] =
          tc2_launch_a_tile_tma<Shape::kBlockM, Shape::kBlockK>(
              tid, block_row, preload * Shape::kBlockK, a, k, a_map, a_stage,
              a_bar[preload]);
      b_tokens[preload] =
          tc2_launch_b_tile_tma<Shape::kBlockK, Shape::kBlockN>(
              tid, block_col, preload * Shape::kBlockK, b, n, b_map, b_stage,
              b_bar[preload]);
    }

    if (num_tiles > 0) {
      tc2_wait_tma_stage(
          tid, a_bar[0],
          static_cast<tc_block_barrier::arrival_token&&>(a_tokens[0]), b_bar[0],
          static_cast<tc_block_barrier::arrival_token&&>(b_tokens[0]));
    }

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      const int stage = tile_idx % Stages;
      half* a_stage = as + stage * Shape::kBlockM * Shape::kBlockK;
      half* b_stage = bs + stage * Shape::kBlockK * Shape::kBlockN;

#pragma unroll
      for (int kk = 0; kk < Shape::kBlockK; kk += 16) {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          wmma::load_matrix_sync(a_frag[mi],
                                 a_stage + (warp_row * 32 + mi * 16) *
                                               Shape::kBlockK +
                                     kk,
                                 Shape::kBlockK);
        }
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
          wmma::load_matrix_sync(b_frag[ni],
                                 b_stage + kk * Shape::kBlockN +
                                     warp_col * 32 + ni * 16,
                                 Shape::kBlockN);
        }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
          for (int ni = 0; ni < 2; ++ni) {
            wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni],
                           c_frag[mi][ni]);
          }
        }
      }

      const int refill_tile_idx = tile_idx + Stages;
      if (refill_tile_idx < num_tiles) {
        a_tokens[stage] =
            tc2_launch_a_tile_tma<Shape::kBlockM, Shape::kBlockK>(
                tid, block_row, refill_tile_idx * Shape::kBlockK, a, k, a_map,
                a_stage, a_bar[stage]);
        b_tokens[stage] =
            tc2_launch_b_tile_tma<Shape::kBlockK, Shape::kBlockN>(
                tid, block_col, refill_tile_idx * Shape::kBlockK, b, n, b_map,
                b_stage, b_bar[stage]);
      }

      const int next_tile_idx = tile_idx + 1;
      if (next_tile_idx < num_tiles) {
        const int next_stage = next_tile_idx % Stages;
        tc2_wait_tma_stage(
            tid, a_bar[next_stage],
            static_cast<tc_block_barrier::arrival_token&&>(a_tokens[next_stage]),
            b_bar[next_stage],
            static_cast<tc_block_barrier::arrival_token&&>(b_tokens[next_stage]));
      }
    }

    tc2_apply_epilogue_2x2(alpha, beta, c_frag, old_c_frag);

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
      for (int ni = 0; ni < 2; ++ni) {
        wmma::store_matrix_sync(c + (c_row + mi * 16) * n + c_col + ni * 16,
                                c_frag[mi][ni], n, wmma::mem_row_major);
      }
    }
    __syncthreads();
  }
}
