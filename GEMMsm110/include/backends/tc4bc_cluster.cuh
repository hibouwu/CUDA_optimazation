#pragma once

// Stage 4b/4c paired 2-SM experiment.
//
// tc4b: 2-SM cluster MMA, TMA and MMA issued by warp 0.
// tc4c: same raw data path, with warp 1 as the dedicated TMA producer.
//
// The kernel uses CUDA tensor maps plus the local inline-PTX wrappers.  It
// does not instantiate CuTe tensors, CUTLASS MMA atoms, or CUTLASS cluster
// launch helpers.

#include "../sm110_ptx_helpers.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>

namespace gemm_sm110::backends {

template <bool WarpSpecialized, int TileK = 128, int Stages = 2>
__global__ __launch_bounds__(128)
void tc4bc_raw_2sm_cluster_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk, float* output,
    int m, int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kClusterTileM = 256;
  constexpr int kLocalTileM = 128;
  constexpr int kTileN = 256;
  constexpr int kLocalTileN = 128;
  constexpr int kTmemAllocColumns = 512;
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
  static_assert(TileK % kTmaK == 0);
  static_assert(TileK % kMmaK == 0);

  constexpr int kKChunks = TileK / kTmaK;
  constexpr int kAChunkBytes = kLocalTileM * kTmaK * sizeof(half);
  constexpr int kBChunkBytes = kLocalTileN * kTmaK * sizeof(half);
  constexpr int kAStageBytes = kKChunks * kAChunkBytes;
  constexpr int kBStageBytes = kKChunks * kBChunkBytes;
  constexpr int kStageBytes = kAStageBytes + kBStageBytes;
  constexpr uint16_t kClusterMask2Sm = 0x3;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / ptx::kWarpSize;
  const int cluster_rank = static_cast<int>(ptx::block_rank_in_cluster());
  const bool leader_cta = cluster_rank == 0;
  const bool consumer_warp = warp == 0;
  const bool producer_warp = WarpSpecialized ? warp == 1 : warp == 0;

  const int tile_m = static_cast<int>(blockIdx.x) / 2;
  const int tile_n = static_cast<int>(blockIdx.y);
  const int offset_m = tile_m * kClusterTileM;
  const int local_offset_m = offset_m + cluster_rank * kLocalTileM;
  const int offset_n = tile_n * kTileN;

  extern __shared__ __align__(1024) char dynamic_smem[];
  const uint32_t smem = ptx::smem_address(dynamic_smem);

  __shared__ alignas(16) uint64_t tma_barrier[Stages];
  __shared__ alignas(16) uint64_t mma_barrier[Stages];
  __shared__ alignas(16) uint32_t tmem_base;
  const uint32_t tma_barrier_base = ptx::smem_address(tma_barrier);
  const uint32_t mma_barrier_base = ptx::smem_address(mma_barrier);

  if (consumer_warp && ptx::elect_one()) {
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
      ptx::mbarrier_init(tma_barrier_base + stage * sizeof(uint64_t), 1);
      ptx::mbarrier_init(mma_barrier_base + stage * sizeof(uint64_t), 1);
    }
    ptx::fence_mbarrier_init_release_cluster();
  }
  if (consumer_warp) {
    ptx::tmem_alloc<2>(ptx::smem_address(&tmem_base), kTmemAllocColumns);
  }
  __syncthreads();

  constexpr uint32_t instruction_descriptor =
      (1U << 4U) |
      (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
      (static_cast<uint32_t>(kClusterTileM) >> 4U << 24U);

  int tma_phase[Stages] = {};
  int mma_phase[Stages] = {};
  const int k_tiles = k / TileK;

  auto issue_load = [&](int k_tile) {
    if (!producer_warp || !ptx::elect_one()) return;

    const int stage = k_tile % Stages;
    const uint32_t barrier = tma_barrier_base + stage * sizeof(uint64_t);
    const uint32_t stage_smem = smem + stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b_smem = stage_smem + kAStageBytes;
    const int offset_k = k_tile * TileK;

#pragma unroll
    for (int chunk = 0; chunk < kKChunks; ++chunk) {
      const int chunk_k = offset_k + chunk * kTmaK;
      ptx::tma_load_2d(a_smem + chunk * kAChunkBytes, &tensor_map_a,
                       chunk_k, local_offset_m, barrier);
      ptx::tma_load_2d(b_smem + chunk * kBChunkBytes, &tensor_map_b_nk,
                       chunk_k, offset_n + cluster_rank * kLocalTileN,
                       barrier);
    }
    ptx::mbarrier_arrive_expect_tx(barrier,
                                   kAStageBytes + kBStageBytes);
  };

  auto issue_mma = [&](int k_tile) {
    const int stage = k_tile % Stages;
    const uint32_t tma_barrier_address =
        tma_barrier_base + stage * sizeof(uint64_t);
    ptx::mbarrier_wait(tma_barrier_address, tma_phase[stage]);
    tma_phase[stage] ^= 1;
    ptx::tcgen05_fence_after_thread_sync();

    if (!leader_cta || !consumer_warp || !ptx::elect_one()) return;

    const uint32_t stage_smem = smem + stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b_smem = stage_smem + kAStageBytes;

#pragma unroll
    for (int k_block = 0; k_block < TileK / kMmaK; ++k_block) {
      const int chunk = k_block / (kTmaK / kMmaK);
      const int block_in_chunk = k_block % (kTmaK / kMmaK);
      const uint32_t a_block =
          a_smem + chunk * kAChunkBytes + block_in_chunk * 32;
      const uint32_t b_block =
          b_smem + chunk * kBChunkBytes + block_in_chunk * 32;
      const uint64_t descriptor_a = ptx::sw128_k_major_descriptor(a_block);
      const uint64_t descriptor_b = ptx::sw128_k_major_descriptor(b_block);
      ptx::mma_f16_cta_group2(tmem_base, descriptor_a, descriptor_b,
                              instruction_descriptor,
                              k_tile != 0 || k_block != 0);
    }
    ptx::mma_commit_multicast<2>(
        mma_barrier_base + stage * sizeof(uint64_t), kClusterMask2Sm);
  };

  const int prologue = k_tiles < Stages ? k_tiles : Stages;
  for (int k_tile = 0; k_tile < prologue; ++k_tile) {
    issue_load(k_tile);
  }

  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    const int stage = k_tile % Stages;
    issue_mma(k_tile);

    const int reuse_tile = k_tile + Stages;
    if (reuse_tile < k_tiles) {
      const uint32_t mma_barrier_address =
          mma_barrier_base + stage * sizeof(uint64_t);
      ptx::mbarrier_wait(mma_barrier_address, mma_phase[stage]);
      mma_phase[stage] ^= 1;
      issue_load(reuse_tile);
    }
  }

  const int final_stages = k_tiles < Stages ? k_tiles : Stages;
  for (int stage = 0; stage < final_stages; ++stage) {
    ptx::mbarrier_wait(mma_barrier_base + stage * sizeof(uint64_t),
                       mma_phase[stage]);
    mma_phase[stage] ^= 1;
  }

  ptx::tcgen05_fence_after_thread_sync();
  for (int n_block = 0; n_block < kTileN / 8; ++n_block) {
    float values[8];
    const uint32_t address =
        tmem_base + ((warp * 32) << 16) + n_block * 8;
    ptx::tmem_load_32x32b_x8(address, values);
    float* dst = output +
                 static_cast<size_t>(local_offset_m + tid) * n +
                 offset_n + n_block * 8;
    ptx::store_global_l1_no_allocate_v8_f32(dst, values);
  }

  __syncthreads();
  if (consumer_warp) {
    ptx::tmem_relinquish_alloc_permit<2>();
    ptx::tmem_dealloc<2>(tmem_base, kTmemAllocColumns);
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

template <bool WarpSpecialized>
class Tc4bcRunner {
 public:
  Tc4bcRunner(const half* a, const half* b_nk, float* d,
              int m, int n, int k)
      : output_(d), m_(m), n_(n), k_(k) {
    if (m % kClusterTileM != 0 || n % kTileN != 0 || k % kTileK != 0) {
      std::fprintf(stderr,
                   "tc4b/tc4c raw kernel requires M a multiple of 256, "
                   "N a multiple of 256, and K a multiple of 128\n");
      std::abort();
    }

    ptx::encode_tiled_2d_sw128(&tensor_map_a_, a, m, k, kLocalTileM);
    ptx::encode_tiled_2d_sw128(&tensor_map_b_, b_nk, n, k, kLocalTileN);

    auto* kernel =
        &tc4bc_raw_2sm_cluster_kernel<WarpSpecialized, kTileK, kStages>;
    constexpr int smem_bytes =
        kStages * (kLocalTileM + kLocalTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc4b/tc4c raw)");

  }

  void launch() {
    auto* kernel =
        &tc4bc_raw_2sm_cluster_kernel<WarpSpecialized, kTileK, kStages>;
    constexpr int smem_bytes =
        kStages * (kLocalTileM + kLocalTileN) * kTileK * sizeof(half);
    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3((m_ / kClusterTileM) * 2, n_ / kTileN, 1);
    config.blockDim = dim3(128, 1, 1);
    config.dynamicSmemBytes = smem_bytes;
    config.attrs = attrs;
    config.numAttrs = 1;

    check_cuda(cudaLaunchKernelEx(&config, kernel, tensor_map_a_,
                                  tensor_map_b_, output_, m_, n_, k_),
               "tc4bc_raw_2sm_cluster_kernel launch");
  }

 private:
  static constexpr int kClusterTileM = 256;
  static constexpr int kLocalTileM = 128;
  static constexpr int kTileN = 256;
  static constexpr int kLocalTileN = 128;
  static constexpr int kTileK = 128;
  static constexpr int kStages = 2;

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

using Tc4bRunner = Tc4bcRunner<false>;
using Tc4cRunner = Tc4bcRunner<true>;

}  // namespace gemm_sm110::backends
