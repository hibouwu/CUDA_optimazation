#pragma once

// Stage 6: validated TCGen05 mainloop plus fused NVFP4 epilogue.
//
// Fast path:
//   - fixed 128x256x128 1-SM persistent TCGen05 mainloop
//   - warp0 issues MMA, warp1 issues TMA, all 128 threads perform readback
//   - TMEM accumulator readback converts directly to packed E2M1 values
//     and one E4M3 block scale per 16 row-major outputs
//
// Fallback:
//   - one CUDA correctness kernel computes 16 row-major outputs per thread
//     group and writes the same NVFP4 layout.  This path is intentionally
//     slow and exists to keep arbitrary shape/boundary cases testable.

#include "../requant/sm110_tcgen05_epilogue.cuh"
#include "../sm110_ptx_helpers.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace gemm_sm110::backends {

constexpr int kTc6Nvfp4BlockSize = 16;

__global__ void tc6_nvfp4_cleanup_kernel(
    const half* a, const half* b_nk, std::uint8_t* quantized,
    std::uint8_t* block_scales, int m, int n, int k,
    float inverse_tensor_scale, int total_groups) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  const int group =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (group >= total_groups) return;

  constexpr int kBlock = kTc6Nvfp4BlockSize;
  float values[kBlock]{};
  float normalized_amax = 0.0f;
  const int element_base = group * kBlock;
  const int elements = m * n;

#pragma unroll
  for (int item = 0; item < kBlock; ++item) {
    const int linear = element_base + item;
    float acc = 0.0f;
    if (linear < elements) {
      const int row = linear / n;
      const int col = linear - row * n;
      for (int kk = 0; kk < k; ++kk) {
        acc += __half2float(a[static_cast<size_t>(row) * k + kk]) *
               __half2float(b_nk[static_cast<size_t>(col) * k + kk]);
      }
    }
    values[item] = acc;
    normalized_amax =
        fmaxf(normalized_amax, fabsf(acc * inverse_tensor_scale));
  }

  const auto scale =
      gemm_sm110::requant::sm110_make_nvfp4_block_scale(normalized_amax);
  block_scales[group] = scale.e4m3_bits;

#pragma unroll
  for (int item = 0; item < kBlock; item += 2) {
    quantized[(element_base + item) / 2] =
        gemm_sm110::requant::sm110_requant_nvfp4_e2m1x2(
            values[item], values[item + 1], inverse_tensor_scale, scale);
  }
#else
  (void)a;
  (void)b_nk;
  (void)quantized;
  (void)block_scales;
  (void)m;
  (void)n;
  (void)k;
  (void)inverse_tensor_scale;
  (void)total_groups;
#endif
}

// Historical 2-SM persistent tc6 sketch kept out of the build: finite-input
// validation showed that the current cta_group::2 TMEM accumulator mapping is
// not modeled correctly yet.  The registered tc6 backend below uses the
// validated 1-SM persistent path.
#if 0
template <int TileK = 128, int Stages = 2>
__global__ __launch_bounds__(256)
void tc6_raw_nvfp4_persistent_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk,
    std::uint8_t* quantized, std::uint8_t* block_scales, int m, int n,
    int k, int tiles_m, int tiles_n, float inverse_tensor_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kTileM = 256;
  constexpr int kTileN = 64;
  constexpr int kLocalRows = 128;
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
  constexpr int kAccumulatorSlots = 2;
  constexpr int kNvfp4Block = kTc6Nvfp4BlockSize;
  static_assert(TileK % kTmaK == 0);
  static_assert(TileK % kMmaK == 0);
  static_assert(kTileN % kNvfp4Block == 0);

  constexpr int kKChunks = TileK / kTmaK;
  constexpr int kAChunkBytes = kTileM * kTmaK * sizeof(half);
  constexpr int kBChunkBytes = kTileN * kTmaK * sizeof(half);
  constexpr int kAStageBytes = kKChunks * kAChunkBytes;
  constexpr int kBStageBytes = kKChunks * kBChunkBytes;
  constexpr int kStageBytes = kAStageBytes + kBStageBytes;
  constexpr uint16_t kClusterMask2Sm = 0x3;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / ptx::kWarpSize;
  const int cluster_rank = static_cast<int>(ptx::block_rank_in_cluster());
  const bool leader_cta = cluster_rank == 0;
  const bool consumer_warp = warp == 0;
  const bool producer_warp = warp == 1;
  const bool epilogue_warp = warp >= 4;

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
    ptx::tmem_alloc<2>(ptx::smem_address(&tmem_base),
                       kTileN * kAccumulatorSlots);
  }
  __syncthreads();
  ptx::cluster_sync();

  constexpr uint32_t instruction_descriptor =
      (1U << 4U) |
      (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
      (static_cast<uint32_t>(kTileM) >> 4U << 24U);

  int tma_phase[Stages] = {};
  int mma_phase[Stages] = {};
  const int k_tiles = k / TileK;

  const int worker_cluster = static_cast<int>(blockIdx.x) / 2;
  const int worker_clusters = static_cast<int>(gridDim.x) / 2;
  int static_work_id = worker_cluster;
  int accumulator_slot = 0;

  auto issue_load = [&](int k_tile, int tile_m, int tile_n) {
    if (!leader_cta || !producer_warp || !ptx::elect_one()) return;

    const int stage = k_tile % Stages;
    const uint32_t barrier = tma_barrier_base + stage * sizeof(uint64_t);
    const uint32_t stage_smem = smem + stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b_smem = stage_smem + kAStageBytes;
    const int offset_k = k_tile * TileK;
    const int offset_m = tile_m * kTileM;
    const int offset_n = tile_n * kTileN;

#pragma unroll
    for (int chunk = 0; chunk < kKChunks; ++chunk) {
      const int chunk_k = offset_k + chunk * kTmaK;
      ptx::tma_load_2d(a_smem + chunk * kAChunkBytes, &tensor_map_a,
                       chunk_k, offset_m, barrier);
      ptx::tma_load_2d(b_smem + chunk * kBChunkBytes, &tensor_map_b_nk,
                       chunk_k, offset_n, barrier);
    }
    ptx::mbarrier_arrive_expect_tx(barrier,
                                   kAStageBytes + kBStageBytes);
  };

  auto issue_mma = [&](int k_tile, uint32_t accumulator) {
    const int stage = k_tile % Stages;
    if (leader_cta) {
      const uint32_t tma_barrier_address =
          tma_barrier_base + stage * sizeof(uint64_t);
      ptx::mbarrier_wait(tma_barrier_address, tma_phase[stage]);
      tma_phase[stage] ^= 1;
      ptx::tcgen05_fence_after_thread_sync();
    }

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
      ptx::mma_f16<2>(accumulator, descriptor_a, descriptor_b,
                      instruction_descriptor,
                      k_tile != 0 || k_block != 0);
    }
    ptx::mma_commit_multicast<2>(
        mma_barrier_base + stage * sizeof(uint64_t), kClusterMask2Sm);
  };

  auto store_nvfp4_tile = [&](int tile_m, int tile_n,
                              uint32_t accumulator) {
    if (!epilogue_warp) return;

    const int epilogue_thread = tid - 128;
    const int epilogue_warp_id = epilogue_thread / ptx::kWarpSize;
    const int row = tile_m * kTileM + cluster_rank * kLocalRows +
                    epilogue_thread;
    const int offset_n = tile_n * kTileN;

    for (int n_group = 0; n_group < kTileN / kNvfp4Block; ++n_group) {
      float lo[8];
      float hi[8];
      const int col = offset_n + n_group * kNvfp4Block;
      const uint32_t address_lo =
          accumulator + ((epilogue_warp_id * 32) << 16) +
          n_group * kNvfp4Block;
      const uint32_t address_hi = address_lo + 8;
      ptx::tmem_load_32x32b_x8(address_lo, lo);
      ptx::tmem_load_32x32b_x8(address_hi, hi);

      float normalized_amax = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        normalized_amax =
            fmaxf(normalized_amax, fabsf(lo[i] * inverse_tensor_scale));
        normalized_amax =
            fmaxf(normalized_amax, fabsf(hi[i] * inverse_tensor_scale));
      }

      const auto scale =
          gemm_sm110::requant::sm110_make_nvfp4_block_scale(
              normalized_amax);
      const size_t linear = static_cast<size_t>(row) * n + col;
      block_scales[linear / kNvfp4Block] = scale.e4m3_bits;

#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        quantized[(linear + i) / 2] =
            gemm_sm110::requant::sm110_requant_nvfp4_e2m1x2(
                lo[i], lo[i + 1], inverse_tensor_scale, scale);
      }
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        quantized[(linear + 8 + i) / 2] =
            gemm_sm110::requant::sm110_requant_nvfp4_e2m1x2(
                hi[i], hi[i + 1], inverse_tensor_scale, scale);
      }
    }
  };

  while (static_work_id < tiles_m * tiles_n) {
    const int tile_m = static_work_id / tiles_n;
    const int tile_n = static_work_id % tiles_n;
    const uint32_t accumulator =
        tmem_base + accumulator_slot * kTileN;

    const int prologue = k_tiles < Stages ? k_tiles : Stages;
    for (int k_tile = 0; k_tile < prologue; ++k_tile) {
      issue_load(k_tile, tile_m, tile_n);
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
      const int stage = k_tile % Stages;
      issue_mma(k_tile, accumulator);

      const int reuse_tile = k_tile + Stages;
      if (reuse_tile < k_tiles) {
        const uint32_t mma_barrier_address =
            mma_barrier_base + stage * sizeof(uint64_t);
        ptx::mbarrier_wait(mma_barrier_address, mma_phase[stage]);
        mma_phase[stage] ^= 1;
        issue_load(reuse_tile, tile_m, tile_n);
      }
    }

    const int final_stages = k_tiles < Stages ? k_tiles : Stages;
    for (int stage = 0; stage < final_stages; ++stage) {
      ptx::mbarrier_wait(mma_barrier_base + stage * sizeof(uint64_t),
                         mma_phase[stage]);
      mma_phase[stage] ^= 1;
    }

    ptx::tcgen05_fence_after_thread_sync();
    __syncthreads();
    store_nvfp4_tile(tile_m, tile_n, accumulator);
    __syncthreads();

    accumulator_slot ^= 1;
    static_work_id += worker_clusters;
  }

  ptx::cluster_sync();
  if (consumer_warp) {
    ptx::tmem_dealloc<2>(tmem_base, kTileN * kAccumulatorSlots);
  }
#else
  (void)tensor_map_a;
  (void)tensor_map_b_nk;
  (void)quantized;
  (void)block_scales;
  (void)m;
  (void)n;
  (void)k;
  (void)tiles_m;
  (void)tiles_n;
  (void)inverse_tensor_scale;
#endif
}
#endif

template <int TileN = 256, int TileK = 128, int Stages = 2>
__global__ __launch_bounds__(128)
void tc6_raw_nvfp4_persistent_1sm_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk,
    std::uint8_t* quantized, std::uint8_t* block_scales, int m, int n,
    int k, int tiles_m, int tiles_n, float inverse_tensor_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kTileM = 128;
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
  constexpr int kThreads = 128;
  constexpr int kNvfp4Block = kTc6Nvfp4BlockSize;
  static_assert(TileK % kTmaK == 0);
  static_assert(TileK % kMmaK == 0);
  static_assert(TileN % kNvfp4Block == 0);

  constexpr int kKChunks = TileK / kTmaK;
  constexpr int kAChunkBytes = kTileM * kTmaK * sizeof(half);
  constexpr int kBChunkBytes = TileN * kTmaK * sizeof(half);
  constexpr int kAStageBytes = kKChunks * kAChunkBytes;
  constexpr int kBStageBytes = kKChunks * kBChunkBytes;
  constexpr int kStageBytes = kAStageBytes + kBStageBytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / ptx::kWarpSize;

  extern __shared__ __align__(1024) char dynamic_smem[];
  const uint32_t smem = ptx::smem_address(dynamic_smem);

  __shared__ alignas(16) uint64_t tma_barrier[Stages];
  __shared__ alignas(16) uint64_t mma_barrier[Stages];
  __shared__ alignas(16) uint32_t tmem_base;
  __shared__ alignas(16) int shared_work_id;
  const uint32_t tma_barrier_base = ptx::smem_address(tma_barrier);
  const uint32_t mma_barrier_base = ptx::smem_address(mma_barrier);

  if (warp == 0 && ptx::elect_one()) {
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
      ptx::mbarrier_init(tma_barrier_base + stage * sizeof(uint64_t), 1);
      ptx::mbarrier_init(mma_barrier_base + stage * sizeof(uint64_t), 1);
    }
    ptx::fence_mbarrier_init_release_cluster();
  }
  if (warp == 0) {
    ptx::tmem_alloc(ptx::smem_address(&tmem_base), TileN);
  }
  __syncthreads();

  constexpr uint32_t instruction_descriptor =
      (1U << 4U) |
      (static_cast<uint32_t>(TileN) >> 3U << 17U) |
      (static_cast<uint32_t>(kTileM) >> 4U << 24U);

  int tma_phase[Stages] = {};
  int mma_phase[Stages] = {};
  const int k_tiles = k / TileK;
  const int total_tiles = tiles_m * tiles_n;
  int static_work_id = static_cast<int>(blockIdx.x);

  auto fetch_work = [&]() {
    if (warp == 0 && ptx::elect_one()) {
      shared_work_id = static_work_id;
      static_work_id += static_cast<int>(gridDim.x);
    }
    __syncthreads();
    return shared_work_id;
  };

  auto issue_load = [&](int k_tile, int tile_m, int tile_n) {
    if (warp != 1 || !ptx::elect_one()) return;

    const int stage = k_tile % Stages;
    const uint32_t barrier =
        tma_barrier_base + stage * sizeof(uint64_t);
    const uint32_t stage_smem = smem + stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b_smem = stage_smem + kAStageBytes;
    const int offset_k = k_tile * TileK;
    const int offset_m = tile_m * kTileM;
    const int offset_n = tile_n * TileN;

#pragma unroll
    for (int chunk = 0; chunk < kKChunks; ++chunk) {
      const int chunk_k = offset_k + chunk * kTmaK;
      ptx::tma_load_2d(a_smem + chunk * kAChunkBytes, &tensor_map_a,
                       chunk_k, offset_m, barrier);
      ptx::tma_load_2d(b_smem + chunk * kBChunkBytes, &tensor_map_b_nk,
                       chunk_k, offset_n, barrier);
    }
    ptx::mbarrier_arrive_expect_tx(
        barrier, kAStageBytes + kBStageBytes);
  };

  auto issue_mma = [&](int k_tile) {
    const int stage = k_tile % Stages;
    const uint32_t tma_barrier_address =
        tma_barrier_base + stage * sizeof(uint64_t);
    ptx::mbarrier_wait(tma_barrier_address, tma_phase[stage]);
    tma_phase[stage] ^= 1;
    ptx::tcgen05_fence_after_thread_sync();

    if (warp != 0 || !ptx::elect_one()) return;

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
      ptx::mma_f16(tmem_base, descriptor_a, descriptor_b,
                   instruction_descriptor,
                   k_tile != 0 || k_block != 0);
    }
    ptx::mma_commit(mma_barrier_base + stage * sizeof(uint64_t));
  };

  auto store_tile = [&](int tile_m, int tile_n) {
    const int offset_m = tile_m * kTileM;
    const int offset_n = tile_n * TileN;

    static_assert(kThreads == kTileM);
    for (int n_group = 0; n_group < TileN / kNvfp4Block; ++n_group) {
      float lo[8];
      float hi[8];
      const int col = offset_n + n_group * kNvfp4Block;
      const uint32_t address_lo =
          tmem_base + ((warp * 32) << 16) + n_group * kNvfp4Block;
      const uint32_t address_hi = address_lo + 8;
      ptx::tmem_load_32x32b_x8(address_lo, lo);
      ptx::tmem_load_32x32b_x8(address_hi, hi);

      float normalized_amax = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        normalized_amax =
            fmaxf(normalized_amax, fabsf(lo[i] * inverse_tensor_scale));
        normalized_amax =
            fmaxf(normalized_amax, fabsf(hi[i] * inverse_tensor_scale));
      }

      const auto scale =
          gemm_sm110::requant::sm110_make_nvfp4_block_scale(
              normalized_amax);
      const size_t linear = static_cast<size_t>(offset_m + tid) * n + col;
      block_scales[linear / kNvfp4Block] = scale.e4m3_bits;

#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        quantized[(linear + i) / 2] =
            gemm_sm110::requant::sm110_requant_nvfp4_e2m1x2(
                lo[i], lo[i + 1], inverse_tensor_scale, scale);
      }
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        quantized[(linear + 8 + i) / 2] =
            gemm_sm110::requant::sm110_requant_nvfp4_e2m1x2(
                hi[i], hi[i + 1], inverse_tensor_scale, scale);
      }
    }
  };

  while (true) {
    const int work_id = fetch_work();
    if (work_id >= total_tiles) break;

    const int tile_m = work_id / tiles_n;
    const int tile_n = work_id - tile_m * tiles_n;

    const int prologue = k_tiles < Stages ? k_tiles : Stages;
    for (int k_tile = 0; k_tile < prologue; ++k_tile) {
      issue_load(k_tile, tile_m, tile_n);
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
        issue_load(reuse_tile, tile_m, tile_n);
      }
    }

    const int final_stages = k_tiles < Stages ? k_tiles : Stages;
    for (int stage = 0; stage < final_stages; ++stage) {
      ptx::mbarrier_wait(mma_barrier_base + stage * sizeof(uint64_t),
                         mma_phase[stage]);
      mma_phase[stage] ^= 1;
    }

    ptx::tcgen05_fence_after_thread_sync();
    store_tile(tile_m, tile_n);
    __syncthreads();
  }

  __syncthreads();
  if (warp == 0) {
    ptx::tmem_dealloc(tmem_base, TileN);
  }
#else
  (void)tensor_map_a;
  (void)tensor_map_b_nk;
  (void)quantized;
  (void)block_scales;
  (void)m;
  (void)n;
  (void)k;
  (void)tiles_m;
  (void)tiles_n;
  (void)inverse_tensor_scale;
#endif
}

class Tc6Runner {
 public:
  Tc6Runner(const half* a, const half* b_nk, std::uint8_t* quantized,
            std::uint8_t* block_scales, int m, int n, int k,
            float inverse_tensor_scale)
      : a_(a),
        b_nk_(b_nk),
        quantized_(quantized),
        block_scales_(block_scales),
        m_(m),
        n_(n),
        k_(k),
        inverse_tensor_scale_(inverse_tensor_scale) {
    has_fast_path_ =
        m_ % kTileM == 0 && n_ % kTileN == 0 && k_ % kTileK == 0 &&
        n_ % kTc6Nvfp4BlockSize == 0;

    if (has_fast_path_) {
      ptx::encode_tiled_2d_sw128(&tensor_map_a_, a, m_, k_, kTileM);
      ptx::encode_tiled_2d_sw128(&tensor_map_b_, b_nk, n_, k_, kTileN);
    }

    auto* kernel =
        &tc6_raw_nvfp4_persistent_1sm_kernel<kTileN, kTileK, kStages>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc6 raw)");
  }

  void launch() {
    if (!has_fast_path_) {
      launch_cleanup();
      return;
    }

    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc6 raw)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc6 raw)");
    const int worker_ctas =
        std::min(total_tiles, std::max(1, properties.multiProcessorCount));

    tc6_raw_nvfp4_persistent_1sm_kernel<kTileN, kTileK, kStages>
        <<<worker_ctas, 128, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, quantized_, block_scales_,
            m_, n_, k_, tiles_m, tiles_n, inverse_tensor_scale_);
    check_cuda(cudaGetLastError(),
               "tc6_raw_nvfp4_persistent_kernel launch");
  }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 256;
  static constexpr int kTileK = 128;
  static constexpr int kStages = 2;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                 cudaGetErrorString(status));
    std::abort();
  }

  void launch_cleanup() {
    const int elements = m_ * n_;
    const int groups =
        (elements + kTc6Nvfp4BlockSize - 1) / kTc6Nvfp4BlockSize;
    dim3 block(128, 1, 1);
    dim3 grid((groups + static_cast<int>(block.x) - 1) /
                  static_cast<int>(block.x),
              1, 1);
    tc6_nvfp4_cleanup_kernel<<<grid, block>>>(
        a_, b_nk_, quantized_, block_scales_, m_, n_, k_,
        inverse_tensor_scale_, groups);
    check_cuda(cudaGetLastError(), "tc6_nvfp4_cleanup_kernel launch");
  }

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b_{};
  const half* a_ = nullptr;
  const half* b_nk_ = nullptr;
  std::uint8_t* quantized_ = nullptr;
  std::uint8_t* block_scales_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  float inverse_tensor_scale_ = 1.0f;
  bool has_fast_path_ = false;
};

}  // namespace gemm_sm110::backends
