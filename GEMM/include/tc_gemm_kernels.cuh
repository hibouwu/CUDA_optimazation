#pragma once

#include "gemm_common.cuh"

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

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
  static constexpr int kStages = 2;
  static constexpr int kAStride = kBlockK + 8;
  static constexpr int kBStride = kBlockN + 8;
};

__device__ __forceinline__ void tc_cp_async_ca_16(void* smem_ptr,
                                                  const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
  const unsigned int smem_addr =
      static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(gmem_ptr));
#else
  *reinterpret_cast<float4*>(smem_ptr) =
      *reinterpret_cast<const float4*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void tc_cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int PendingGroups>
__device__ __forceinline__ void tc_cp_async_wait_group() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(PendingGroups));
#endif
}

template <int BM, int BK, int AStride>
__device__ __forceinline__ void tc2_copy_a_tile_async(int tid, int k,
                                                      const half* src,
                                                      half* dst) {
  constexpr int kChunkElems = 8;
  constexpr int kChunks = (BM * BK) / kChunkElems;
  static_assert((BM * BK) % kChunkElems == 0);
  for (int chunk = tid; chunk < kChunks; chunk += blockDim.x) {
    const int elem = chunk * kChunkElems;
    const int row = elem / BK;
    const int col = elem % BK;
    tc_cp_async_ca_16(&dst[row * AStride + col], &src[row * k + col]);
  }
}

template <int BK, int BN, int BStride>
__device__ __forceinline__ void tc2_copy_b_tile_async(int tid, int n,
                                                      const half* src,
                                                      half* dst) {
  constexpr int kChunkElems = 8;
  constexpr int kChunks = (BK * BN) / kChunkElems;
  static_assert((BK * BN) % kChunkElems == 0);
  for (int chunk = tid; chunk < kChunks; chunk += blockDim.x) {
    const int elem = chunk * kChunkElems;
    const int row = elem / BN;
    const int col = elem % BN;
    tc_cp_async_ca_16(&dst[row * BStride + col], &src[row * n + col]);
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

// tc2: cp.async + double-buffered WMMA mainloop。
// 这是当前 Tensor Core 主线版本：A/B tile 通过 cp.async 异步搬到
// shared memory，用 2-stage double buffer 在计算当前 tile 时预取下一 tile。
// 一个 256-thread CTA 含 8 个 warp，协同计算 64x32x16 的 block tile，
// 每个 warp 负责一个 16x16 输出子块。
__global__ void hgemm_tc2_cp_async_dbuf_wmma_64x32x16(
    int m, int n, int k, float alpha, const half* a, const half* b, float beta,
    float* c) {
  using Shape = Tc2PipelineShape;

  __shared__ half as[Shape::kStages][Shape::kBlockM][Shape::kAStride];
  __shared__ half bs[Shape::kStages][Shape::kBlockK][Shape::kBStride];

  const int tid = threadIdx.x;
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
  tc2_copy_a_tile_async<Shape::kBlockM, Shape::kBlockK, Shape::kAStride>(
      tid, k, a + block_row * k, &as[0][0][0]);
  tc2_copy_b_tile_async<Shape::kBlockK, Shape::kBlockN, Shape::kBStride>(
      tid, n, b + block_col, &bs[0][0][0]);
  tc_cp_async_commit_group();
  tc_cp_async_wait_group<0>();
  __syncthreads();

  int stage = 0;
  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    const int next_tile_idx = tile_idx + 1;
    const int next_stage = stage ^ 1;
    if (next_tile_idx < num_tiles) {
      tc2_copy_a_tile_async<Shape::kBlockM, Shape::kBlockK, Shape::kAStride>(
          tid, k, a + block_row * k + next_tile_idx * Shape::kBlockK,
          &as[next_stage][0][0]);
      tc2_copy_b_tile_async<Shape::kBlockK, Shape::kBlockN, Shape::kBStride>(
          tid, n, b + next_tile_idx * Shape::kBlockK * n + block_col,
          &bs[next_stage][0][0]);
      tc_cp_async_commit_group();
    }

    wmma::load_matrix_sync(a_frag, &as[stage][warp_row * 16][0],
                           Shape::kAStride);
    wmma::load_matrix_sync(b_frag, &bs[stage][0][warp_col * 16],
                           Shape::kBStride);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    if (next_tile_idx < num_tiles) {
      tc_cp_async_wait_group<0>();
      __syncthreads();
      stage = next_stage;
    }
  }

  tc_apply_epilogue(alpha, beta, c_frag, old_c_frag);

  wmma::store_matrix_sync(c + c_row * n + c_col, c_frag, n,
                          wmma::mem_row_major);
}
