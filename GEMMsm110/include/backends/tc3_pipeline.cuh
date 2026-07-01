#pragma once

// Stage 3 handwritten TMA + TCGen05 pipeline.
//
// This kernel deliberately does not use CuTe Tensor, TiledMMA, TMA atoms,
// collectives, or a CUTLASS kernel schedule.  CUDA tensor maps and the thin
// inline-PTX wrappers in sm110_ptx_helpers.cuh are the only abstractions.

#include "../sm110_ptx_helpers.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace gemm_sm110::backends {

template <int TileN = 128, int TileK = 64, int Stages = 2>
__global__ __launch_bounds__(128)
void tc3_raw_pipeline_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk, float* output,
    int m, int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kTileM = 128;
  constexpr int kMmaK = 16;
  constexpr int kThreads = 128;
  constexpr int kAStageBytes = kTileM * TileK * sizeof(half);
  constexpr int kBStageBytes = TileN * TileK * sizeof(half);
  constexpr int kStageBytes = kAStageBytes + kBStageBytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / ptx::kWarpSize;
  const int tile_n_count = n / TileN;
  const int tile_m = static_cast<int>(blockIdx.x) / tile_n_count;
  const int tile_n = static_cast<int>(blockIdx.x) % tile_n_count;
  const int offset_m = tile_m * kTileM;
  const int offset_n = tile_n * TileN;

  extern __shared__ __align__(1024) char dynamic_smem[];
  const uint32_t smem = ptx::smem_address(dynamic_smem);

  __shared__ alignas(16) uint64_t tma_barrier[Stages];
  __shared__ alignas(16) uint64_t mma_barrier;
  __shared__ alignas(16) uint32_t tmem_base;
  const uint32_t tma_barrier_base = ptx::smem_address(tma_barrier);
  const uint32_t mma_barrier_address = ptx::smem_address(&mma_barrier);

  if (warp == 0 && ptx::elect_one()) {
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
      ptx::mbarrier_init(tma_barrier_base + stage * sizeof(uint64_t), 1);
    }
    ptx::mbarrier_init(mma_barrier_address, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  } else if (warp == 1) {
    ptx::tmem_alloc(ptx::smem_address(&tmem_base), TileN);
  }
  __syncthreads();

  // FP16 A/B, FP32 accumulator, K-major operands.
  constexpr uint32_t instruction_descriptor =
      (1U << 4U) |
      (static_cast<uint32_t>(TileN) >> 3U << 17U) |
      (static_cast<uint32_t>(kTileM) >> 4U << 24U);

  int tma_phase = 0;
  int mma_phase = 0;
  const int k_tiles = k / TileK;

  auto issue_load = [&](int k_tile) {
    if (warp != 0 || !ptx::elect_one()) return;
    const int stage = k_tile % Stages;
    const uint32_t barrier =
        tma_barrier_base + stage * sizeof(uint64_t);
    const uint32_t stage_smem = smem + stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b_smem = stage_smem + kAStageBytes;
    const int offset_k = k_tile * TileK;

    ptx::tma_load_2d(a_smem, &tensor_map_a, offset_k, offset_m,
                     barrier);
    ptx::tma_load_2d(b_smem, &tensor_map_b_nk, offset_k, offset_n,
                     barrier);
    ptx::mbarrier_arrive_expect_tx(
        barrier, kAStageBytes + kBStageBytes);
  };

  auto issue_mma = [&](int k_tile) {
    const int stage = k_tile % Stages;
    const uint32_t barrier =
        tma_barrier_base + stage * sizeof(uint64_t);
    ptx::mbarrier_wait(barrier, tma_phase);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    if (stage == Stages - 1) tma_phase ^= 1;
    if (warp != 0 || !ptx::elect_one()) return;

    const uint32_t stage_smem = smem + stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b_smem = stage_smem + kAStageBytes;

    // A and the pre-transposed B[N,K] are both K-major.  A single TileK=64
    // slab contains four K=16 TCGen05 operations.
#pragma unroll
    for (int k_block = 0; k_block < TileK / kMmaK; ++k_block) {
      const uint64_t descriptor_a =
          ptx::sw128_k_major_descriptor(a_smem + k_block * 32);
      const uint64_t descriptor_b =
          ptx::sw128_k_major_descriptor(b_smem + k_block * 32);
      ptx::mma_f16(tmem_base, descriptor_a, descriptor_b,
                   instruction_descriptor,
                   k_tile != 0 || k_block != 0);
    }
    ptx::mma_commit(mma_barrier_address);
  };

  const int prologue = min(k_tiles, Stages - 1);
  for (int k_tile = 0; k_tile < prologue; ++k_tile) {
    issue_load(k_tile);
  }
  for (int k_tile = 0; k_tile < k_tiles - Stages + 1; ++k_tile) {
    issue_load(k_tile + Stages - 1);
    issue_mma(k_tile);
    ptx::mbarrier_wait(mma_barrier_address, mma_phase);
    mma_phase ^= 1;
  }
  for (int k_tile = max(0, k_tiles - Stages + 1);
       k_tile < k_tiles; ++k_tile) {
    issue_mma(k_tile);
    ptx::mbarrier_wait(mma_barrier_address, mma_phase);
    mma_phase ^= 1;
  }

  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  static_assert(kThreads == kTileM);
  for (int n_block = 0; n_block < TileN / 8; ++n_block) {
    float values[8];
    const uint32_t address =
        tmem_base + ((warp * 32) << 16) + n_block * 8;
    ptx::tmem_load_32x32b_x8(address, values);
    float* dst = output +
                 static_cast<size_t>(offset_m + tid) * n +
                 offset_n + n_block * 8;
    reinterpret_cast<float4*>(dst)[0] =
        make_float4(values[0], values[1], values[2], values[3]);
    reinterpret_cast<float4*>(dst)[1] =
        make_float4(values[4], values[5], values[6], values[7]);
  }

  __syncthreads();
  if (warp == 0) {
    ptx::tmem_dealloc(tmem_base, TileN);
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

class Tc3Runner {
 public:
  Tc3Runner(const half* a, const half* b_nk, float* d,
            int m, int n, int k)
      : output_(d), m_(m), n_(n), k_(k) {
    if (m % kTileM != 0 || n % kTileN != 0 || k % kTileK != 0) {
      std::fprintf(stderr,
                   "tc3 raw kernel requires M,N multiples of 128 and "
                   "K a multiple of 64\n");
      std::abort();
    }

    ptx::encode_tiled_2d_sw128(
        &tensor_map_a_, a, m, k, kTileM);
    ptx::encode_tiled_2d_sw128(
        &tensor_map_b_, b_nk, n, k, kTileN);

    auto* kernel = &tc3_raw_pipeline_kernel<kTileN, kTileK, kStages>;
    const int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes),
        "cudaFuncSetAttribute(tc3 raw)");
  }

  void launch() {
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    const int grid = (m_ / kTileM) * (n_ / kTileN);
    tc3_raw_pipeline_kernel<kTileN, kTileK, kStages>
        <<<grid, 128, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, k_);
    check_cuda(cudaGetLastError(), "tc3_raw_pipeline_kernel launch");
  }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 128;
  static constexpr int kTileK = 64;
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

}  // namespace gemm_sm110::backends
