#pragma once

// Stage 1/2 paired experiment:
//   Rank3=false, Sw128=false -> tc1a (2D TMA, INTER SMEM descriptor)
//   Rank3=true,  Sw128=false -> tc1b (3D TMA, INTER SMEM descriptor)
//   Rank3=false, Sw128=true  -> tc2a (2D TMA, SW128 SMEM descriptor)
//   Rank3=true,  Sw128=true  -> tc2b (3D TMA, SW128 SMEM descriptor)
//
// All four variants are raw CUDA + inline PTX.  They intentionally keep a
// single-stage mainloop so that rank and shared-memory descriptor effects stay
// isolated from the deeper tc3+ pipeline changes.

#include "../sm110_ptx_helpers.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace gemm_sm110::backends {

template <bool Rank3, bool Sw128>
__global__ __launch_bounds__(128)
void tc12_raw_tma_tcgen05_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk, float* output,
    int m, int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  constexpr int kTileK = 64;
  constexpr int kMmaK = 16;
  constexpr int kThreads = 128;
  constexpr int kAStageBytes = kTileM * kTileK * sizeof(half);
  constexpr int kBStageBytes = kTileN * kTileK * sizeof(half);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / ptx::kWarpSize;
  const int tile_n_count = n / kTileN;
  const int tile_m = static_cast<int>(blockIdx.x) / tile_n_count;
  const int tile_n = static_cast<int>(blockIdx.x) % tile_n_count;
  const int offset_m = tile_m * kTileM;
  const int offset_n = tile_n * kTileN;

  extern __shared__ __align__(1024) char dynamic_smem[];
  const uint32_t smem = ptx::smem_address(dynamic_smem);
  const uint32_t a_smem = smem;
  const uint32_t b_smem = smem + kAStageBytes;

  __shared__ alignas(16) uint64_t tma_barrier;
  __shared__ alignas(16) uint64_t mma_barrier;
  __shared__ alignas(16) uint32_t tmem_base;
  const uint32_t tma_barrier_address = ptx::smem_address(&tma_barrier);
  const uint32_t mma_barrier_address = ptx::smem_address(&mma_barrier);

  if (warp == 0 && ptx::elect_one()) {
    ptx::mbarrier_init(tma_barrier_address, 1);
    ptx::mbarrier_init(mma_barrier_address, 1);
    ptx::fence_mbarrier_init_release_cluster();
  }
  if (warp == 0) {
    ptx::tmem_alloc(ptx::smem_address(&tmem_base), kTileN);
  }
  __syncthreads();

  constexpr uint32_t instruction_descriptor =
      (1U << 4U) |
      (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
      (static_cast<uint32_t>(kTileM) >> 4U << 24U);

  int tma_phase = 0;
  int mma_phase = 0;
  const int k_tiles = k / kTileK;

  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    if (warp == 0 && ptx::elect_one()) {
      if constexpr (Rank3) {
        ptx::tma_load_3d(a_smem, &tensor_map_a, 0, offset_m, k_tile,
                         tma_barrier_address);
        ptx::tma_load_3d(b_smem, &tensor_map_b_nk, 0, offset_n, k_tile,
                         tma_barrier_address);
      } else {
        const int offset_k = k_tile * kTileK;
        ptx::tma_load_2d(a_smem, &tensor_map_a, offset_k, offset_m,
                         tma_barrier_address);
        ptx::tma_load_2d(b_smem, &tensor_map_b_nk, offset_k, offset_n,
                         tma_barrier_address);
      }
      ptx::mbarrier_arrive_expect_tx(
          tma_barrier_address, kAStageBytes + kBStageBytes);
    }

    ptx::mbarrier_wait(tma_barrier_address, tma_phase);
    tma_phase ^= 1;
    ptx::tcgen05_fence_after_thread_sync();

    if (warp == 0 && ptx::elect_one()) {
#pragma unroll
      for (int k_block = 0; k_block < kTileK / kMmaK; ++k_block) {
        const uint32_t a_block = a_smem + k_block * kMmaK * sizeof(half);
        const uint32_t b_block = b_smem + k_block * kMmaK * sizeof(half);
        const uint64_t descriptor_a =
            Sw128 ? ptx::sw128_k_major_descriptor(a_block)
                  : ptx::inter_k_major_descriptor(a_block);
        const uint64_t descriptor_b =
            Sw128 ? ptx::sw128_k_major_descriptor(b_block)
                  : ptx::inter_k_major_descriptor(b_block);
        ptx::mma_f16(tmem_base, descriptor_a, descriptor_b,
                     instruction_descriptor,
                     k_tile != 0 || k_block != 0);
      }
      ptx::mma_commit(mma_barrier_address);
    }

    ptx::mbarrier_wait(mma_barrier_address, mma_phase);
    mma_phase ^= 1;
  }

  ptx::tcgen05_fence_after_thread_sync();
  static_assert(kThreads == kTileM);
  for (int n_block = 0; n_block < kTileN / 8; ++n_block) {
    float values[8];
    const uint32_t address =
        tmem_base + ((warp * 32) << 16) + n_block * 8;
    ptx::tmem_load_32x32b_x8(address, values);
    float* dst = output +
                 static_cast<size_t>(offset_m + tid) * n +
                 offset_n + n_block * 8;
    ptx::store_global_l1_no_allocate_v8_f32(dst, values);
  }

  __syncthreads();
  if (warp == 0) {
    ptx::tmem_dealloc(tmem_base, kTileN);
  }
#else
  (void)tensor_map_a;
  (void)tensor_map_b_nk;
  (void)output;
  (void)m;
  (void)n;
  (void)k;
#endif
}

template <bool Rank3, bool Sw128>
class Tc12Runner {
 public:
  Tc12Runner(const half* a, const half* b_nk, float* d,
             int m, int n, int k)
      : output_(d), m_(m), n_(n), k_(k) {
    if (m % kTileM != 0 || n % kTileN != 0 || k % kTileK != 0) {
      std::fprintf(stderr,
                   "tc1/tc2 raw kernels require M,N multiples of 128 "
                   "and K a multiple of 64\n");
      std::abort();
    }

    if constexpr (Rank3) {
      if constexpr (Sw128) {
        ptx::encode_tiled_3d_sw128(
            &tensor_map_a_, a, m, k, kTileM, kTileK);
        ptx::encode_tiled_3d_sw128(
            &tensor_map_b_, b_nk, n, k, kTileN, kTileK);
      } else {
        ptx::encode_tiled_3d_inter(
            &tensor_map_a_, a, m, k, kTileM, kTileK);
        ptx::encode_tiled_3d_inter(
            &tensor_map_b_, b_nk, n, k, kTileN, kTileK);
      }
    } else {
      if constexpr (Sw128) {
        ptx::encode_tiled_2d_sw128(
            &tensor_map_a_, a, m, k, kTileM);
        ptx::encode_tiled_2d_sw128(
            &tensor_map_b_, b_nk, n, k, kTileN);
      } else {
        ptx::encode_tiled_2d_inter(
            &tensor_map_a_, a, m, k, kTileM);
        ptx::encode_tiled_2d_inter(
            &tensor_map_b_, b_nk, n, k, kTileN);
      }
    }

    auto* kernel = &tc12_raw_tma_tcgen05_kernel<Rank3, Sw128>;
    check_cuda(
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kStageBytes),
        "cudaFuncSetAttribute(tc1/tc2 raw)");
  }

  void launch() {
    const int grid = (m_ / kTileM) * (n_ / kTileN);
    tc12_raw_tma_tcgen05_kernel<Rank3, Sw128>
        <<<grid, 128, kStageBytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, k_);
    check_cuda(cudaGetLastError(),
               "tc12_raw_tma_tcgen05_kernel launch");
  }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 128;
  static constexpr int kTileK = 64;
  static constexpr int kStageBytes =
      (kTileM + kTileN) * kTileK * sizeof(half);

  static void check_cuda(cudaError_t status, const char* where) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                 cudaGetErrorString(status));
    std::abort();
  }

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b_{};
  float* output_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
};

using Tc1aRunner = Tc12Runner<false, false>;
using Tc1bRunner = Tc12Runner<true, false>;
using Tc2aRunner = Tc12Runner<false, true>;
using Tc2bRunner = Tc12Runner<true, true>;

}  // namespace gemm_sm110::backends
