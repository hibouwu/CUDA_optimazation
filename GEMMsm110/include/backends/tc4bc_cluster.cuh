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

#include <algorithm>
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
  ptx::cluster_sync();

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
  ptx::cluster_sync();
  if (consumer_warp) {
    ptx::tmem_relinquish_alloc_permit<2>();
  }
  ptx::cluster_sync();
  if (consumer_warp) {
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

template <int TileK = 128, int Stages = 2, int TileN = 256,
          bool StoreTransposed = false, int EpilogueWarps = 4,
          bool StoreOutput = true, int RowMajorSmemStoreCols = 0,
          int ClusterTileM = 256>
__global__ __launch_bounds__((EpilogueWarps + 2) * 32)
void tc4c_overlap_2tile_2sm_cluster_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk, float* output,
    int m, int n, int k, int tiles_m, int tiles_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kClusterTileM = ClusterTileM;
  constexpr int kLocalTileM = ClusterTileM / 2;
  constexpr int kTileN = TileN;
  constexpr int kLocalTileN = TileN / 2;
  constexpr int kTmemColumnsPerBuffer = TileN < 256 ? 256 : TileN;
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
  constexpr int kEpilogueWarps = EpilogueWarps;
  constexpr int kRowMajorSmemStoreCols = RowMajorSmemStoreCols;
  constexpr int kTmaWarp = kEpilogueWarps;
  constexpr int kMmaWarp = kEpilogueWarps + 1;
  constexpr uint16_t kClusterMask2Sm = 0x3;
  static_assert(TileN % 16 == 0);
  static_assert(TileN % 2 == 0);
  static_assert(TileK % kTmaK == 0);
  static_assert(TileK % kMmaK == 0);
  static_assert(kClusterTileM == 128 || kClusterTileM == 256);
  static_assert(kEpilogueWarps == 2 || kEpilogueWarps == 4 ||
                kEpilogueWarps == 8);
  static_assert((kClusterTileM == 128 && kEpilogueWarps == 2) ||
                (kClusterTileM == 256 &&
                 (kEpilogueWarps == 4 || kEpilogueWarps == 8)));
  static_assert(kRowMajorSmemStoreCols == 0 ||
                (kClusterTileM == 256 && kEpilogueWarps == 4 &&
                 !StoreTransposed &&
                 TileN %
                         (kRowMajorSmemStoreCols == 0
                              ? 1
                              : kRowMajorSmemStoreCols) ==
                     0 &&
                 kRowMajorSmemStoreCols % 8 == 0));

  constexpr int kKChunks = TileK / kTmaK;
  constexpr int kAChunkBytes = kLocalTileM * kTmaK * sizeof(half);
  constexpr int kBChunkBytes = kLocalTileN * kTmaK * sizeof(half);
  constexpr int kAStageBytes = kKChunks * kAChunkBytes;
  constexpr int kBStageBytes = kKChunks * kBChunkBytes;
  constexpr int kStageBytes = kAStageBytes + kBStageBytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / ptx::kWarpSize;
  const int lane = tid % ptx::kWarpSize;
  const int cluster_rank = static_cast<int>(ptx::block_rank_in_cluster());
  const bool leader_cta = cluster_rank == 0;

  extern __shared__ __align__(1024) char dynamic_smem[];
  const uint32_t smem = ptx::smem_address(dynamic_smem);

  __shared__ alignas(16) uint64_t tma_barrier[Stages];
  __shared__ alignas(16) uint64_t mma_barrier[Stages];
  __shared__ alignas(16) uint64_t mainloop_barrier[2];
  __shared__ alignas(16) uint32_t tmem_base;

  const uint32_t tma_barrier_base = ptx::smem_address(tma_barrier);
  const uint32_t mma_barrier_base = ptx::smem_address(mma_barrier);
  const uint32_t mainloop_barrier_base =
      ptx::smem_address(mainloop_barrier);

  if (warp == kMmaWarp && ptx::elect_one()) {
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
      ptx::mbarrier_init(tma_barrier_base + stage * sizeof(uint64_t), 1);
      ptx::mbarrier_init(mma_barrier_base + stage * sizeof(uint64_t), 1);
    }
#pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
      ptx::mbarrier_init(mainloop_barrier_base + stage * sizeof(uint64_t),
                         1);
    }
    ptx::fence_mbarrier_init_release_cluster();
  }
  if (warp == kMmaWarp) {
    ptx::tmem_alloc<2>(ptx::smem_address(&tmem_base),
                       kTmemColumnsPerBuffer * 2);
  }
  __syncthreads();
  ptx::cluster_sync();

  constexpr uint32_t instruction_descriptor =
      (1U << 4U) |
      (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
      (static_cast<uint32_t>(kClusterTileM) >> 4U << 24U);

  const int k_tiles = k / TileK;
  const int total_tiles = tiles_m * tiles_n;
  const int worker_cluster = static_cast<int>(blockIdx.x) / 2;
  const int worker_clusters = static_cast<int>(gridDim.x) / 2;

  auto tile_coordinates = [&](int work_id, int& tile_m, int& tile_n) {
    if (tiles_n == 4) {
      tile_m = work_id >> 2;
      tile_n = work_id & 3;
    } else if (tiles_n == 3) {
      tile_m = work_id / 3;
      tile_n = work_id - tile_m * 3;
    } else {
      tile_m = work_id / tiles_n;
      tile_n = work_id - tile_m * tiles_n;
    }
  };

  auto issue_load = [&](int k_tile, int tile_m, int tile_n,
                        int tma_stage) {
    const uint32_t barrier =
        tma_barrier_base + tma_stage * sizeof(uint64_t);
    const uint32_t stage_smem = smem + tma_stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b_smem = stage_smem + kAStageBytes;
    const int offset_k = k_tile * TileK;
    const int offset_m =
        tile_m * kClusterTileM + cluster_rank * kLocalTileM;
    const int offset_n = tile_n * kTileN + cluster_rank * kLocalTileN;

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

  auto issue_mma = [&](int k_tile, int tma_stage, int tmem_stage) {
    const uint32_t stage_smem = smem + tma_stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b_smem = stage_smem + kAStageBytes;
    const uint32_t accumulator =
        tmem_base + tmem_stage * kTmemColumnsPerBuffer;

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
      ptx::mma_f16_cta_group2(accumulator, descriptor_a, descriptor_b,
                              instruction_descriptor,
                              k_tile != 0 || k_block != 0);
    }
  };

  auto epilogue_sync = []() {
    asm volatile("bar.sync %0, %1;"
                 :
                 : "r"(1), "r"(kEpilogueWarps * ptx::kWarpSize)
                 : "memory");
  };

  auto store_tile = [&](int tile_m, int tile_n, int tmem_stage) {
    if constexpr (!StoreOutput) {
      return;
    }
    if (warp >= kEpilogueWarps) return;
    const int column_base =
        kEpilogueWarps == 8 ? (warp >> 2) * (kTileN / 2) : 0;
    constexpr int kColumnBlocks =
        kEpilogueWarps == 8 ? (kTileN / 2) / 8 : kTileN / 8;
    const int column_blocks = kColumnBlocks;
    const int row = (warp & 3) * ptx::kWarpSize + lane;
    const int offset_m =
        tile_m * kClusterTileM + cluster_rank * kLocalTileM;
    if (offset_m + row >= m) return;
    const int offset_n = tile_n * kTileN;
    const uint32_t base_address =
        tmem_base + tmem_stage * kTmemColumnsPerBuffer + (row << 16) +
        column_base;
    float* row_dst = output +
                     static_cast<size_t>(offset_m + row) * n +
                     offset_n + column_base;

    if constexpr (kRowMajorSmemStoreCols > 0) {
      constexpr int kSmemStride = kRowMajorSmemStoreCols + 1;
      float* store_smem =
          reinterpret_cast<float*>(dynamic_smem + Stages * kStageBytes) +
          warp * ptx::kWarpSize * kSmemStride;
      const int warp_row_base = warp * ptx::kWarpSize;
      const int row_in_warp = lane;
      const int global_row = offset_m + warp_row_base + row_in_warp;
      const bool valid_load_row = global_row < m;

#pragma unroll
      for (int col_chunk = 0; col_chunk < kTileN;
           col_chunk += kRowMajorSmemStoreCols) {
        if (valid_load_row) {
#pragma unroll
          for (int n_block = 0; n_block < kRowMajorSmemStoreCols / 8;
               ++n_block) {
            float values[8];
            ptx::tmem_load_32x32b_x8(
                tmem_base + tmem_stage * kTmemColumnsPerBuffer +
                    ((warp_row_base + row_in_warp) << 16) +
                    col_chunk + n_block * 8,
                values);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              store_smem[row_in_warp * kSmemStride + n_block * 8 + i] =
                  values[i];
            }
          }
        }
        __syncwarp();

        const int local_col = lane * 8;
        if (local_col < kRowMajorSmemStoreCols) {
          for (int local_row = 0; local_row < ptx::kWarpSize;
               ++local_row) {
            const int row_out = offset_m + warp_row_base + local_row;
            if (row_out < m) {
              float values[8];
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                values[i] = store_smem[local_row * kSmemStride +
                                       local_col + i];
              }
              ptx::store_global_l1_no_allocate_v8_f32(
                  output + static_cast<size_t>(row_out) * n + offset_n +
                      col_chunk + local_col,
                  values);
            }
          }
        }
        __syncwarp();
      }
      return;
    }

    float values_even[8];
    float values_odd[8];
    ptx::tmem_load_32x32b_x8_no_wait(base_address, values_even);
#pragma unroll
    for (int n_block = 0; n_block < column_blocks; ++n_block) {
      ptx::tmem_load_wait();
      const bool use_even = (n_block & 1) == 0;
      float* dst = row_dst + n_block * 8;
      if (n_block + 1 < column_blocks) {
        if (use_even) {
          ptx::tmem_load_32x32b_x8_no_wait(
              base_address + (n_block + 1) * 8, values_odd);
        } else {
          ptx::tmem_load_32x32b_x8_no_wait(
              base_address + (n_block + 1) * 8, values_even);
        }
      }
      if (use_even) {
        if constexpr (StoreTransposed) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            if (offset_n + column_base + n_block * 8 + i < n) {
              ptx::store_global_l1_no_allocate_f32(
                  output +
                      static_cast<size_t>(offset_n + column_base +
                                          n_block * 8 + i) *
                          m +
                      offset_m + row,
                  values_even[i]);
            }
          }
        } else {
          ptx::store_global_l1_no_allocate_v8_f32(dst, values_even);
        }
      } else {
        if constexpr (StoreTransposed) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            if (offset_n + column_base + n_block * 8 + i < n) {
              ptx::store_global_l1_no_allocate_f32(
                  output +
                      static_cast<size_t>(offset_n + column_base +
                                          n_block * 8 + i) *
                          m +
                      offset_m + row,
                  values_odd[i]);
            }
          }
        } else {
          ptx::store_global_l1_no_allocate_v8_f32(dst, values_odd);
        }
      }
    }
  };

  if (warp == kTmaWarp && ptx::elect_one()) {
    int tma_stage = 0;
    int mma_phase = 1;
    for (int work_id = worker_cluster; work_id < total_tiles;
         work_id += worker_clusters) {
      int tile_m = 0;
      int tile_n = 0;
      tile_coordinates(work_id, tile_m, tile_n);
#pragma unroll
      for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        ptx::mbarrier_wait(mma_barrier_base + tma_stage * sizeof(uint64_t),
                           mma_phase);
        issue_load(k_tile, tile_m, tile_n, tma_stage);
        tma_stage = (tma_stage + 1) % Stages;
        if (tma_stage == 0) mma_phase ^= 1;
      }
    }
  } else if (warp == kMmaWarp) {
    int tma_stage = 0;
    int tma_phase = 0;
    int tmem_stage = 0;
    for (int work_id = worker_cluster; work_id < total_tiles;
         work_id += worker_clusters) {
#pragma unroll
      for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        ptx::mbarrier_wait(tma_barrier_base +
                               tma_stage * sizeof(uint64_t),
                           tma_phase);
        ptx::tcgen05_fence_after_thread_sync();
        if (leader_cta && ptx::elect_one()) {
          issue_mma(k_tile, tma_stage, tmem_stage);
          ptx::mma_commit_multicast<2>(
              mma_barrier_base + tma_stage * sizeof(uint64_t),
              kClusterMask2Sm);
        }
        tma_stage = (tma_stage + 1) % Stages;
        if (tma_stage == 0) tma_phase ^= 1;
      }
      if (leader_cta && ptx::elect_one()) {
        ptx::mma_commit_multicast<2>(
            mainloop_barrier_base + tmem_stage * sizeof(uint64_t),
            kClusterMask2Sm);
      }
      tmem_stage ^= 1;
    }
  } else if (warp < kEpilogueWarps) {
    int tmem_stage = 0;
    int mainloop_phase = 0;
    for (int work_id = worker_cluster; work_id < total_tiles;
         work_id += worker_clusters) {
      if (warp == 0 && ptx::elect_one()) {
        ptx::mbarrier_wait(mainloop_barrier_base +
                               tmem_stage * sizeof(uint64_t),
                           mainloop_phase);
      }
      epilogue_sync();
      ptx::tcgen05_fence_after_thread_sync();

      int tile_m = 0;
      int tile_n = 0;
      tile_coordinates(work_id, tile_m, tile_n);
      store_tile(tile_m, tile_n, tmem_stage);

      tmem_stage ^= 1;
      if (tmem_stage == 0) mainloop_phase ^= 1;
    }
  }

  __syncthreads();
  ptx::cluster_sync();
  if (warp == kMmaWarp) {
    ptx::tmem_relinquish_alloc_permit<2>();
  }
  ptx::cluster_sync();
  if (warp == kMmaWarp) {
    ptx::tmem_dealloc<2>(tmem_base, kTmemColumnsPerBuffer * 2);
  }
#else
  (void)tensor_map_a;
  (void)tensor_map_b_nk;
  (void)output;
  (void)m;
  (void)n;
  (void)k;
  (void)tiles_m;
  (void)tiles_n;
#endif
}

template <int TileK = 64, int Stages = 3, int EpilogueWarps = 4,
          bool StoreOutput = true>
__global__ __launch_bounds__((EpilogueWarps + 2) * 32)
void tc4c_overlap_split_n192_2sm_cluster_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b128_nk,
    const __grid_constant__ CUtensorMap tensor_map_b64_nk, float* output,
    int m, int n, int k, int tiles_m, int tiles_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kClusterTileM = 256;
  constexpr int kLocalTileM = 128;
  constexpr int kTileN = 192;
  constexpr int kTileN0 = 128;
  constexpr int kTileN1 = 64;
  constexpr int kLocalTileN0 = kTileN0 / 2;
  constexpr int kLocalTileN1 = kTileN1 / 2;
  constexpr int kTmemColumnsPerBuffer = 256;
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
  constexpr int kEpilogueWarps = EpilogueWarps;
  constexpr int kTmaWarp = kEpilogueWarps;
  constexpr int kMmaWarp = kEpilogueWarps + 1;
  constexpr uint16_t kClusterMask2Sm = 0x3;
  static_assert(TileK % kTmaK == 0);
  static_assert(TileK % kMmaK == 0);
  static_assert(kEpilogueWarps == 4);

  constexpr int kKChunks = TileK / kTmaK;
  constexpr int kAChunkBytes = kLocalTileM * kTmaK * sizeof(half);
  constexpr int kB0ChunkBytes = kLocalTileN0 * kTmaK * sizeof(half);
  constexpr int kB1ChunkBytes = kLocalTileN1 * kTmaK * sizeof(half);
  constexpr int kAStageBytes = kKChunks * kAChunkBytes;
  constexpr int kB0StageBytes = kKChunks * kB0ChunkBytes;
  constexpr int kB1StageBytes = kKChunks * kB1ChunkBytes;
  constexpr int kStageBytes =
      kAStageBytes + kB0StageBytes + kB1StageBytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / ptx::kWarpSize;
  const int lane = tid % ptx::kWarpSize;
  const int cluster_rank = static_cast<int>(ptx::block_rank_in_cluster());
  const bool leader_cta = cluster_rank == 0;

  extern __shared__ __align__(1024) char dynamic_smem[];
  const uint32_t smem = ptx::smem_address(dynamic_smem);

  __shared__ alignas(16) uint64_t tma_barrier[Stages];
  __shared__ alignas(16) uint64_t mma_barrier[Stages];
  __shared__ alignas(16) uint64_t mainloop_barrier[2];
  __shared__ alignas(16) uint32_t tmem_base;

  const uint32_t tma_barrier_base = ptx::smem_address(tma_barrier);
  const uint32_t mma_barrier_base = ptx::smem_address(mma_barrier);
  const uint32_t mainloop_barrier_base =
      ptx::smem_address(mainloop_barrier);

  if (warp == kMmaWarp && ptx::elect_one()) {
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
      ptx::mbarrier_init(tma_barrier_base + stage * sizeof(uint64_t), 1);
      ptx::mbarrier_init(mma_barrier_base + stage * sizeof(uint64_t), 1);
    }
#pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
      ptx::mbarrier_init(mainloop_barrier_base + stage * sizeof(uint64_t),
                         1);
    }
    ptx::fence_mbarrier_init_release_cluster();
  }
  if (warp == kMmaWarp) {
    ptx::tmem_alloc<2>(ptx::smem_address(&tmem_base),
                       kTmemColumnsPerBuffer * 2);
  }
  __syncthreads();
  ptx::cluster_sync();

  constexpr uint32_t instruction_descriptor_n128 =
      (1U << 4U) |
      (static_cast<uint32_t>(kTileN0) >> 3U << 17U) |
      (static_cast<uint32_t>(kClusterTileM) >> 4U << 24U);
  constexpr uint32_t instruction_descriptor_n64 =
      (1U << 4U) |
      (static_cast<uint32_t>(kTileN1) >> 3U << 17U) |
      (static_cast<uint32_t>(kClusterTileM) >> 4U << 24U);

  const int k_tiles = k / TileK;
  const int total_tiles = tiles_m * tiles_n;
  const int worker_cluster = static_cast<int>(blockIdx.x) / 2;
  const int worker_clusters = static_cast<int>(gridDim.x) / 2;

  auto tile_coordinates = [&](int work_id, int& tile_m, int& tile_n) {
    if (tiles_n == 4) {
      tile_m = work_id >> 2;
      tile_n = work_id & 3;
    } else {
      tile_m = work_id / tiles_n;
      tile_n = work_id - tile_m * tiles_n;
    }
  };

  auto issue_load = [&](int k_tile, int tile_m, int tile_n,
                        int tma_stage) {
    const uint32_t barrier =
        tma_barrier_base + tma_stage * sizeof(uint64_t);
    const uint32_t stage_smem = smem + tma_stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b0_smem = stage_smem + kAStageBytes;
    const uint32_t b1_smem = b0_smem + kB0StageBytes;
    const int offset_k = k_tile * TileK;
    const int offset_m =
        tile_m * kClusterTileM + cluster_rank * kLocalTileM;
    const int offset_n = tile_n * kTileN;

#pragma unroll
    for (int chunk = 0; chunk < kKChunks; ++chunk) {
      const int chunk_k = offset_k + chunk * kTmaK;
      ptx::tma_load_2d(a_smem + chunk * kAChunkBytes, &tensor_map_a,
                       chunk_k, offset_m, barrier);
      ptx::tma_load_2d(b0_smem + chunk * kB0ChunkBytes,
                       &tensor_map_b128_nk, chunk_k,
                       offset_n + cluster_rank * kLocalTileN0, barrier);
      ptx::tma_load_2d(b1_smem + chunk * kB1ChunkBytes,
                       &tensor_map_b64_nk, chunk_k,
                       offset_n + kTileN0 +
                           cluster_rank * kLocalTileN1,
                       barrier);
    }
    ptx::mbarrier_arrive_expect_tx(
        barrier, kAStageBytes + kB0StageBytes + kB1StageBytes);
  };

  auto issue_mma = [&](int k_tile, int tma_stage, int tmem_stage) {
    const uint32_t stage_smem = smem + tma_stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b0_smem = stage_smem + kAStageBytes;
    const uint32_t b1_smem = b0_smem + kB0StageBytes;
    const uint32_t accumulator =
        tmem_base + tmem_stage * kTmemColumnsPerBuffer;
    const uint32_t accumulator_tail = accumulator + kTileN0;

#pragma unroll
    for (int k_block = 0; k_block < TileK / kMmaK; ++k_block) {
      const int chunk = k_block / (kTmaK / kMmaK);
      const int block_in_chunk = k_block % (kTmaK / kMmaK);
      const uint32_t a_block =
          a_smem + chunk * kAChunkBytes + block_in_chunk * 32;
      const uint32_t b0_block =
          b0_smem + chunk * kB0ChunkBytes + block_in_chunk * 32;
      const uint32_t b1_block =
          b1_smem + chunk * kB1ChunkBytes + block_in_chunk * 32;
      const uint64_t descriptor_a = ptx::sw128_k_major_descriptor(a_block);
      const uint64_t descriptor_b0 =
          ptx::sw128_k_major_descriptor(b0_block);
      const uint64_t descriptor_b1 =
          ptx::sw128_k_major_descriptor(b1_block);
      const bool accumulate = k_tile != 0 || k_block != 0;
      ptx::mma_f16_cta_group2(accumulator, descriptor_a, descriptor_b0,
                              instruction_descriptor_n128, accumulate);
      ptx::mma_f16_cta_group2(accumulator_tail, descriptor_a,
                              descriptor_b1, instruction_descriptor_n64,
                              accumulate);
    }
  };

  auto epilogue_sync = []() {
    asm volatile("bar.sync %0, %1;"
                 :
                 : "r"(1), "r"(kEpilogueWarps * ptx::kWarpSize)
                 : "memory");
  };

  auto store_columns = [&](int offset_m, int offset_n, uint32_t base,
                           int column_count) {
    const int row = warp * ptx::kWarpSize + lane;
    if (offset_m + row >= m) return;
    const uint32_t base_address = base + (row << 16);
    float* row_dst =
        output + static_cast<size_t>(offset_m + row) * n + offset_n;

    float values_even[8];
    float values_odd[8];
    ptx::tmem_load_32x32b_x8_no_wait(base_address, values_even);
#pragma unroll
    for (int n_block = 0; n_block < column_count / 8; ++n_block) {
      ptx::tmem_load_wait();
      const bool use_even = (n_block & 1) == 0;
      float* dst = row_dst + n_block * 8;
      if (n_block + 1 < column_count / 8) {
        if (use_even) {
          ptx::tmem_load_32x32b_x8_no_wait(
              base_address + (n_block + 1) * 8, values_odd);
        } else {
          ptx::tmem_load_32x32b_x8_no_wait(
              base_address + (n_block + 1) * 8, values_even);
        }
      }
      if (use_even) {
        ptx::store_global_l1_no_allocate_v8_f32(dst, values_even);
      } else {
        ptx::store_global_l1_no_allocate_v8_f32(dst, values_odd);
      }
    }
  };

  auto store_tile = [&](int tile_m, int tile_n, int tmem_stage) {
    if constexpr (!StoreOutput) {
      return;
    }
    if (warp >= kEpilogueWarps) return;
    const int offset_m =
        tile_m * kClusterTileM + cluster_rank * kLocalTileM;
    const int offset_n = tile_n * kTileN;
    const uint32_t accumulator =
        tmem_base + tmem_stage * kTmemColumnsPerBuffer;
    store_columns(offset_m, offset_n, accumulator, kTileN0);
    store_columns(offset_m, offset_n + kTileN0, accumulator + kTileN0,
                  kTileN1);
  };

  if (warp == kTmaWarp && ptx::elect_one()) {
    int tma_stage = 0;
    int mma_phase = 1;
    for (int work_id = worker_cluster; work_id < total_tiles;
         work_id += worker_clusters) {
      int tile_m = 0;
      int tile_n = 0;
      tile_coordinates(work_id, tile_m, tile_n);
#pragma unroll
      for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        ptx::mbarrier_wait(mma_barrier_base + tma_stage * sizeof(uint64_t),
                           mma_phase);
        issue_load(k_tile, tile_m, tile_n, tma_stage);
        tma_stage = (tma_stage + 1) % Stages;
        if (tma_stage == 0) mma_phase ^= 1;
      }
    }
  } else if (warp == kMmaWarp) {
    int tma_stage = 0;
    int tma_phase = 0;
    int tmem_stage = 0;
    for (int work_id = worker_cluster; work_id < total_tiles;
         work_id += worker_clusters) {
#pragma unroll
      for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        ptx::mbarrier_wait(tma_barrier_base +
                               tma_stage * sizeof(uint64_t),
                           tma_phase);
        ptx::tcgen05_fence_after_thread_sync();
        if (leader_cta && ptx::elect_one()) {
          issue_mma(k_tile, tma_stage, tmem_stage);
          ptx::mma_commit_multicast<2>(
              mma_barrier_base + tma_stage * sizeof(uint64_t),
              kClusterMask2Sm);
        }
        tma_stage = (tma_stage + 1) % Stages;
        if (tma_stage == 0) tma_phase ^= 1;
      }
      if (leader_cta && ptx::elect_one()) {
        ptx::mma_commit_multicast<2>(
            mainloop_barrier_base + tmem_stage * sizeof(uint64_t),
            kClusterMask2Sm);
      }
      tmem_stage ^= 1;
    }
  } else if (warp < kEpilogueWarps) {
    int tmem_stage = 0;
    int mainloop_phase = 0;
    for (int work_id = worker_cluster; work_id < total_tiles;
         work_id += worker_clusters) {
      if (warp == 0 && ptx::elect_one()) {
        ptx::mbarrier_wait(mainloop_barrier_base +
                               tmem_stage * sizeof(uint64_t),
                           mainloop_phase);
      }
      epilogue_sync();
      ptx::tcgen05_fence_after_thread_sync();

      int tile_m = 0;
      int tile_n = 0;
      tile_coordinates(work_id, tile_m, tile_n);
      store_tile(tile_m, tile_n, tmem_stage);

      tmem_stage ^= 1;
      if (tmem_stage == 0) mainloop_phase ^= 1;
    }
  }

  __syncthreads();
  ptx::cluster_sync();
  if (warp == kMmaWarp) {
    ptx::tmem_relinquish_alloc_permit<2>();
  }
  ptx::cluster_sync();
  if (warp == kMmaWarp) {
    ptx::tmem_dealloc<2>(tmem_base, kTmemColumnsPerBuffer * 2);
  }
#else
  (void)tensor_map_a;
  (void)tensor_map_b128_nk;
  (void)tensor_map_b64_nk;
  (void)output;
  (void)m;
  (void)n;
  (void)k;
  (void)tiles_m;
  (void)tiles_n;
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
    if constexpr (WarpSpecialized) {
      auto* overlap_kernel =
          &tc4c_overlap_2tile_2sm_cluster_kernel<kTileK, kStages>;
      check_cuda(cudaFuncSetAttribute(
                     overlap_kernel,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     smem_bytes),
                 "cudaFuncSetAttribute(tc4c overlap)");
    }

  }

  void launch() {
    if constexpr (WarpSpecialized) {
      if (m_ % kClusterTileM == 0 && n_ % kTileN == 0 &&
          k_ % kTileK == 0) {
        launch_overlap();
        return;
      }
    }

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

  void launch_overlap() {
    auto* kernel = &tc4c_overlap_2tile_2sm_cluster_kernel<kTileK, kStages>;
    constexpr int smem_bytes =
        kStages * (kLocalTileM + kLocalTileN) * kTileK * sizeof(half);
    const int tiles_m = m_ / kClusterTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;
    const int worker_clusters = (total_tiles + 1) / 2;

    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(worker_clusters * 2, 1, 1);
    config.blockDim = dim3(192, 1, 1);
    config.dynamicSmemBytes = smem_bytes;
    config.attrs = attrs;
    config.numAttrs = 1;

    check_cuda(cudaLaunchKernelEx(&config, kernel, tensor_map_a_,
                                  tensor_map_b_, output_, m_, n_, k_,
                                  tiles_m, tiles_n),
               "tc4c_overlap_2tile_2sm_cluster_kernel launch");
  }

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

template <int RunnerTileN = 192, int RunnerTileK = 64,
          int RunnerStages = 4, int RunnerEpilogueWarps = 4,
          bool RunnerStoreOutput = true, int RunnerRowMajorSmemStoreCols = 0,
          int RunnerClusterTileM = 256>
class Tc4cOverlapPaddedRowsRunner {
 public:
  Tc4cOverlapPaddedRowsRunner(const half* a_padded, const half* b_nk,
                              float* d, int output_m, int padded_m,
                              int n, int k)
      : output_(d),
        output_m_(output_m),
        padded_m_(padded_m),
        n_(n),
        k_(k) {
    if (output_m_ <= 0 || output_m_ > padded_m_ ||
        padded_m_ % kClusterTileM != 0 || n_ % kTileN != 0 ||
        k_ % kTileK != 0) {
      std::fprintf(stderr,
                   "Tc4cOverlapPaddedRowsRunner requires padded M%%256=0, "
                   "N%%TileN=0, K%%TileK=0, and output_m <= padded_m\n");
      std::abort();
    }

    ptx::encode_tiled_2d_sw128(&tensor_map_a_, a_padded, padded_m_, k_,
                               kLocalTileM);
    ptx::encode_tiled_2d_sw128(&tensor_map_b_, b_nk, n_, k_, kLocalTileN);

    auto* kernel =
        &tc4c_overlap_2tile_2sm_cluster_kernel<kTileK, kStages, kTileN,
                                               false, kEpilogueWarps,
                                               kStoreOutput,
                                               kRowMajorSmemStoreCols,
                                               kClusterTileM>;
    constexpr int smem_bytes =
        kStages * (kLocalTileM + kLocalTileN) * kTileK * sizeof(half);
    constexpr int store_smem_bytes =
        kRowMajorSmemStoreCols > 0
            ? kEpilogueWarps * ptx::kWarpSize *
                  (kRowMajorSmemStoreCols + 1) * sizeof(float)
            : 0;
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes + store_smem_bytes),
               "cudaFuncSetAttribute(tc4c padded-row overlap)");
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                   cudaSharedmemCarveoutMaxShared),
               "cudaFuncSetAttribute(tc4c padded-row carveout)");
  }

  void launch(cudaStream_t stream = 0) {
    auto* kernel =
        &tc4c_overlap_2tile_2sm_cluster_kernel<kTileK, kStages, kTileN,
                                               false, kEpilogueWarps,
                                               kStoreOutput,
                                               kRowMajorSmemStoreCols,
                                               kClusterTileM>;
    constexpr int smem_bytes =
        kStages * (kLocalTileM + kLocalTileN) * kTileK * sizeof(half);
    constexpr int store_smem_bytes =
        kRowMajorSmemStoreCols > 0
            ? kEpilogueWarps * ptx::kWarpSize *
                  (kRowMajorSmemStoreCols + 1) * sizeof(float)
            : 0;
    const int tiles_m = padded_m_ / kClusterTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;
    int worker_clusters = (total_tiles + 1) / 2;
    if (const char* override = std::getenv("TC4C_WORKER_CLUSTERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_clusters = std::min(total_tiles, requested);
      }
    }

    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(worker_clusters * 2, 1, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = smem_bytes + store_smem_bytes;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 1;

    check_cuda(cudaLaunchKernelEx(&config, kernel, tensor_map_a_,
                                  tensor_map_b_, output_, output_m_, n_, k_,
                                  tiles_m, tiles_n),
               "tc4c padded-row overlap launch");
  }

 private:
  static constexpr int kClusterTileM = RunnerClusterTileM;
  static constexpr int kLocalTileM = RunnerClusterTileM / 2;
  static constexpr int kTileN = RunnerTileN;
  static constexpr int kLocalTileN = RunnerTileN / 2;
  static constexpr int kTileK = RunnerTileK;
  static constexpr int kStages = RunnerStages;
  static constexpr int kEpilogueWarps = RunnerEpilogueWarps;
  static constexpr bool kStoreOutput = RunnerStoreOutput;
  static constexpr int kRowMajorSmemStoreCols =
      RunnerRowMajorSmemStoreCols;
  static constexpr int kThreads = (kEpilogueWarps + 2) * 32;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                 cudaGetErrorString(status));
    std::abort();
  }

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b_{};
  float* output_ = nullptr;
  int output_m_ = 0;
  int padded_m_ = 0;
  int n_ = 0;
  int k_ = 0;
};

template <int RunnerTileK = 64, int RunnerStages = 3,
          int RunnerEpilogueWarps = 4, bool RunnerStoreOutput = true>
class Tc4cOverlapSplitN192PaddedRowsRunner {
 public:
  Tc4cOverlapSplitN192PaddedRowsRunner(const half* a_padded,
                                       const half* b_nk, float* d,
                                       int output_m, int padded_m, int n,
                                       int k)
      : output_(d),
        output_m_(output_m),
        padded_m_(padded_m),
        n_(n),
        k_(k) {
    if (output_m_ <= 0 || output_m_ > padded_m_ ||
        padded_m_ % kClusterTileM != 0 || n_ % kTileN != 0 ||
        k_ % kTileK != 0) {
      std::fprintf(stderr,
                   "Tc4cOverlapSplitN192PaddedRowsRunner requires "
                   "padded M%%256=0, N%%192=0, K%%TileK=0, and "
                   "output_m <= padded_m\n");
      std::abort();
    }

    ptx::encode_tiled_2d_sw128(&tensor_map_a_, a_padded, padded_m_, k_,
                               kLocalTileM);
    ptx::encode_tiled_2d_sw128(&tensor_map_b128_, b_nk, n_, k_,
                               kLocalTileN0);
    ptx::encode_tiled_2d_sw128(&tensor_map_b64_, b_nk, n_, k_,
                               kLocalTileN1);

    auto* kernel =
        &tc4c_overlap_split_n192_2sm_cluster_kernel<
            kTileK, kStages, kEpilogueWarps, kStoreOutput>;
    constexpr int smem_bytes =
        kStages * (kLocalTileM + kLocalTileN0 + kLocalTileN1) *
        kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc4c split-n192 overlap)");
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                   cudaSharedmemCarveoutMaxShared),
               "cudaFuncSetAttribute(tc4c split-n192 carveout)");
  }

  void launch() {
    auto* kernel =
        &tc4c_overlap_split_n192_2sm_cluster_kernel<
            kTileK, kStages, kEpilogueWarps, kStoreOutput>;
    constexpr int smem_bytes =
        kStages * (kLocalTileM + kLocalTileN0 + kLocalTileN1) *
        kTileK * sizeof(half);
    const int tiles_m = padded_m_ / kClusterTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;
    int worker_clusters = (total_tiles + 1) / 2;
    if (const char* override = std::getenv("TC4C_WORKER_CLUSTERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_clusters = std::min(total_tiles, requested);
      }
    }

    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(worker_clusters * 2, 1, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = smem_bytes;
    config.attrs = attrs;
    config.numAttrs = 1;

    check_cuda(cudaLaunchKernelEx(
                   &config, kernel, tensor_map_a_, tensor_map_b128_,
                   tensor_map_b64_, output_, output_m_, n_, k_, tiles_m,
                   tiles_n),
               "tc4c split-n192 overlap launch");
  }

 private:
  static constexpr int kClusterTileM = 256;
  static constexpr int kLocalTileM = 128;
  static constexpr int kTileN = 192;
  static constexpr int kTileN0 = 128;
  static constexpr int kTileN1 = 64;
  static constexpr int kLocalTileN0 = kTileN0 / 2;
  static constexpr int kLocalTileN1 = kTileN1 / 2;
  static constexpr int kTileK = RunnerTileK;
  static constexpr int kStages = RunnerStages;
  static constexpr int kEpilogueWarps = RunnerEpilogueWarps;
  static constexpr bool kStoreOutput = RunnerStoreOutput;
  static constexpr int kThreads = (kEpilogueWarps + 2) * 32;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                 cudaGetErrorString(status));
    std::abort();
  }

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b128_{};
  CUtensorMap tensor_map_b64_{};
  float* output_ = nullptr;
  int output_m_ = 0;
  int padded_m_ = 0;
  int n_ = 0;
  int k_ = 0;
};

template <int RunnerTileN = 64, int RunnerTileK = 64,
          int RunnerStages = 4>
class Tc4cOverlapTransposedStoreRunner {
 public:
  Tc4cOverlapTransposedStoreRunner(const half* a, const half* b_nk,
                                   float* d, int m, int n, int k)
      : Tc4cOverlapTransposedStoreRunner(a, b_nk, d, m, n, n, k) {}

  Tc4cOverlapTransposedStoreRunner(const half* a, const half* b_nk,
                                   float* d, int m, int output_n,
                                   int padded_n, int k)
      : output_(d), m_(m), output_n_(output_n), padded_n_(padded_n), k_(k) {
    if (m_ % kClusterTileM != 0 || output_n_ <= 0 ||
        output_n_ > padded_n_ || padded_n_ % kTileN != 0 ||
        k_ % kTileK != 0) {
      std::fprintf(stderr,
                   "Tc4cOverlapTransposedStoreRunner requires M%%256=0, "
                   "padded N%%TileN=0, K%%TileK=0, and output_n <= "
                   "padded_n\n");
      std::abort();
    }

    ptx::encode_tiled_2d_sw128(&tensor_map_a_, a, m_, k_, kLocalTileM);
    ptx::encode_tiled_2d_sw128(&tensor_map_b_, b_nk, padded_n_, k_,
                               kLocalTileN);

    auto* kernel =
        &tc4c_overlap_2tile_2sm_cluster_kernel<kTileK, kStages, kTileN,
                                               true>;
    constexpr int smem_bytes =
        kStages * (kLocalTileM + kLocalTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc4c transposed overlap)");
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                   cudaSharedmemCarveoutMaxShared),
               "cudaFuncSetAttribute(tc4c transposed carveout)");
  }

  void launch() {
    auto* kernel =
        &tc4c_overlap_2tile_2sm_cluster_kernel<kTileK, kStages, kTileN,
                                               true>;
    constexpr int smem_bytes =
        kStages * (kLocalTileM + kLocalTileN) * kTileK * sizeof(half);
    const int tiles_m = m_ / kClusterTileM;
    const int tiles_n = padded_n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;
    int worker_clusters = (total_tiles + 1) / 2;
    if (const char* override = std::getenv("TC4C_WORKER_CLUSTERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_clusters = std::min(total_tiles, requested);
      }
    }

    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(worker_clusters * 2, 1, 1);
    config.blockDim = dim3(192, 1, 1);
    config.dynamicSmemBytes = smem_bytes;
    config.attrs = attrs;
    config.numAttrs = 1;

    check_cuda(cudaLaunchKernelEx(&config, kernel, tensor_map_a_,
                                  tensor_map_b_, output_, m_, output_n_, k_,
                                  tiles_m, tiles_n),
               "tc4c transposed overlap launch");
  }

 private:
  static constexpr int kClusterTileM = 256;
  static constexpr int kLocalTileM = 128;
  static constexpr int kTileN = RunnerTileN;
  static constexpr int kLocalTileN = RunnerTileN / 2;
  static constexpr int kTileK = RunnerTileK;
  static constexpr int kStages = RunnerStages;

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
  int output_n_ = 0;
  int padded_n_ = 0;
  int k_ = 0;
};

using Tc4bRunner = Tc4bcRunner<false>;
using Tc4cRunner = Tc4bcRunner<true>;

}  // namespace gemm_sm110::backends
