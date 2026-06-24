#pragma once

#include "gemm_common.cuh"

#include <cuda/barrier>
#include <cuda_fp16.h>
#include <mma.h>

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
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 32;
  static constexpr int kBlockK = 16;
  static constexpr int kWarpTilesN = 2;
};

using tc_block_barrier = cuda::barrier<cuda::thread_scope_block>;

inline void tc_encode_rowmajor_tensor_map_2d(CUtensorMap& tensor_map,
                                             half* global_address, int rows,
                                             int cols, int box_rows,
                                             int box_cols) {
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
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
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
__device__ __forceinline__ void tc2_load_a_tile_tma(int tid, int block_row,
                                                    int tile_k,
                                                    const half* a,
                                                    int lda,
                                                    const CUtensorMap* a_map,
                                                    half* a_smem,
                                                    tc_block_barrier& bar) {
#if __CUDA_ARCH__ >= 900
  if (tid == 0) {
    cuda::device::barrier_expect_tx(bar, BM * BK * sizeof(half));
    cde::cp_async_bulk_tensor_2d_global_to_shared(a_smem, a_map, tile_k,
                                                  block_row, bar);
    cde::cp_async_bulk_commit_group();
    bar.arrive_and_wait();
  }
  __syncthreads();
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
#else
  tc2_copy_tile_fallback<BM, BK>(tid, lda, a + block_row * lda + tile_k, a_smem);
  __syncthreads();
#endif
}

template <int BK, int BN>
__device__ __forceinline__ void tc2_load_b_tile_tma(int tid, int block_col,
                                                    int tile_k,
                                                    const half* b,
                                                    int ldb,
                                                    const CUtensorMap* b_map,
                                                    half* b_smem,
                                                    tc_block_barrier& bar) {
#if __CUDA_ARCH__ >= 900
  if (tid == 0) {
    cuda::device::barrier_expect_tx(bar, BK * BN * sizeof(half));
    cde::cp_async_bulk_tensor_2d_global_to_shared(b_smem, b_map, block_col,
                                                  tile_k, bar);
    cde::cp_async_bulk_commit_group();
    bar.arrive_and_wait();
  }
  __syncthreads();
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
#else
  tc2_copy_tile_fallback<BK, BN>(tid, ldb, b + tile_k * ldb + block_col, b_smem);
  __syncthreads();
#endif
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

// tc2: TMA + WMMA。
// A/B tile 通过 Tensor Memory Accelerator 以 2D tensor map 形式搬到
// shared memory，计算部分仍然保持 warp-level WMMA。
// 当前版本先聚焦最小正确的 TMA + WMMA 路径，不叠加 WGMMA / swizzle。
__global__ void hgemm_tc2_tma_wmma_64x32x16(
    int m, int n, int k, float alpha, const half* a,
    const __grid_constant__ CUtensorMap* a_map, const half* b,
    const __grid_constant__ CUtensorMap* b_map, float beta, float* c) {
  using Shape = Tc2PipelineShape;

  __shared__ half as[Shape::kBlockM][Shape::kBlockK];
  __shared__ half bs[Shape::kBlockK][Shape::kBlockN];
  __shared__ tc_block_barrier a_bar;
  __shared__ tc_block_barrier b_bar;

  const int tid = threadIdx.x;
  if (tid == 0) {
    init(&a_bar, 1);
    init(&b_bar, 1);
  }
  __syncthreads();

  const int warp_id = tid / kWarpSize;
  int warp_row = 0;
  int warp_col = 0;
  tc_warp_coords<Shape>(warp_id, warp_row, warp_col);

  const int block_row = blockIdx.y * Shape::kBlockM;
  const int block_col = blockIdx.x * Shape::kBlockN;
  const int c_row = block_row + warp_row * 16;
  const int c_col = block_col + warp_col * 16;

  wmma_a_rowmajor_16<half> a_frag;
  wmma_b_rowmajor_16<half> b_frag;
  wmma_accum_f32_16 c_frag;
  wmma_accum_f32_16 old_c_frag;

  wmma::fill_fragment(c_frag, 0.0f);
  if (beta != 0.0f) {
    wmma::load_matrix_sync(old_c_frag, c + c_row * n + c_col, n,
                           wmma::mem_row_major);
  }

  const int num_tiles = k / Shape::kBlockK;
  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    tc2_load_a_tile_tma<Shape::kBlockM, Shape::kBlockK>(
        tid, block_row, tile_idx * Shape::kBlockK, a, k, a_map, &as[0][0],
        a_bar);
    tc2_load_b_tile_tma<Shape::kBlockK, Shape::kBlockN>(
        tid, block_col, tile_idx * Shape::kBlockK, b, n, b_map, &bs[0][0],
        b_bar);

    wmma::load_matrix_sync(a_frag, &as[warp_row * 16][0], Shape::kBlockK);
    wmma::load_matrix_sync(b_frag, &bs[0][warp_col * 16], Shape::kBlockN);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  tc_apply_epilogue(alpha, beta, c_frag, old_c_frag);

  wmma::store_matrix_sync(c + c_row * n + c_col, c_frag, n,
                          wmma::mem_row_major);
}
