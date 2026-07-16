#pragma once

// Stage 5 persistent schedulers built on the validated tc4a 1-SM path.
//
// Shared computation path:
//   - fixed 128x256x128 TCGen05 mainloop
//   - two SW128 TMA stages
//   - warp 0 issues TCGen05 MMA
//   - warp 1 issues TMA loads
//   - all 128 threads read TMEM and write FP32 D
//
// Scheduler variable:
//   UseDynamic=false -> tc5a resident persistent workers + static grid stride
//   UseDynamic=true  -> tc5b resident persistent workers + software work queue
//
// Stage 5 intentionally keeps the 1-SM tc4a mainloop as the scheduling
// substrate.  The verified 2-SM path lives in tc4bc_cluster.cuh as a separate
// stage-4 comparison point.

#include "../sm110_ptx_helpers.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace gemm_sm110::backends {

constexpr int kSw128TmaLeadingDimensionAlignment = 64;

__global__ void tc5_boundary_cleanup_kernel(
    const half* a, const half* b_nk, float* output, int m, int n, int k,
    int fast_m, int fast_n, int fast_k) {
  const int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int row = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;

  const bool has_fast_value =
      row < fast_m && col < fast_n && fast_k > 0;
  if (has_fast_value && fast_k == k) return;

  float acc = has_fast_value
                  ? output[static_cast<size_t>(row) * n + col]
                  : 0.0f;
  const int begin_k = has_fast_value ? fast_k : 0;
  for (int kk = begin_k; kk < k; ++kk) {
    acc += __half2float(a[static_cast<size_t>(row) * k + kk]) *
           __half2float(b_nk[static_cast<size_t>(col) * k + kk]);
  }
  output[static_cast<size_t>(row) * n + col] = acc;
}

template <bool UseDynamic, int TileN = 256, int TileK = 128,
          int Stages = 2>
__global__ __launch_bounds__(128)
void tc5_raw_persistent_1sm_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk, float* output,
    int m, int n, int k, int tiles_m, int tiles_n, int* work_counter) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kTileM = 128;
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
  constexpr int kThreads = 128;
  static_assert(TileK % kTmaK == 0);
  static_assert(TileK % kMmaK == 0);

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
      if constexpr (UseDynamic) {
        shared_work_id = atomicAdd(work_counter, 1);
      } else {
        shared_work_id = static_work_id;
        static_work_id += static_cast<int>(gridDim.x);
      }
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
    for (int n_block = 0; n_block < TileN / 8; ++n_block) {
      float values[8];
      const uint32_t address =
          tmem_base + ((warp * 32) << 16) + n_block * 8;
      ptx::tmem_load_32x32b_x8(address, values);
      float* dst = output +
                   static_cast<size_t>(offset_m + tid) * n +
                   offset_n + n_block * 8;
      ptx::store_global_l1_no_allocate_v8_f32(dst, values);
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
  (void)output;
  (void)m;
  (void)n;
  (void)k;
  (void)tiles_m;
  (void)tiles_n;
  (void)work_counter;
#endif
}

template <int TileM = 128, int TileN = 256, int TileK = 64,
          int Stages = 4, int EpilogueWarps = (TileM == 64 ? 4 : TileM / 32)>
__global__ __launch_bounds__(
    (EpilogueWarps + 2) * 32)
void tc5h_overlap_epilogue_1sm_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk, float* output,
    int m, int n, int k, int tiles_m, int tiles_n,
    int m64_panel_stride) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
  constexpr int kEpilogueWarps = EpilogueWarps;
  constexpr int kTmaWarp = kEpilogueWarps;
  constexpr int kMmaWarp = kEpilogueWarps + 1;
  static_assert(TileK % kTmaK == 0);
  static_assert(TileK % kMmaK == 0);
  static_assert(TileN % 16 == 0);
  static_assert(TileM == 64 || TileM == 128);
  static_assert(TileM != 64 || kEpilogueWarps == 4);
  static_assert(TileM != 128 || kEpilogueWarps == 4 ||
                kEpilogueWarps == 8);

  constexpr int kKChunks = TileK / kTmaK;
  constexpr int kAChunkBytes = TileM * kTmaK * sizeof(half);
  constexpr int kBChunkBytes = TileN * kTmaK * sizeof(half);
  constexpr int kAStageBytes = kKChunks * kAChunkBytes;
  constexpr int kBStageBytes = kKChunks * kBChunkBytes;
  constexpr int kStageBytes = kAStageBytes + kBStageBytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / ptx::kWarpSize;
  const int lane = tid % ptx::kWarpSize;

  extern __shared__ __align__(1024) char dynamic_smem[];
  const uint32_t smem = ptx::smem_address(dynamic_smem);

  __shared__ alignas(16) uint64_t tma_barrier[Stages];
  __shared__ alignas(16) uint64_t mma_barrier[Stages];
  __shared__ alignas(16) uint64_t mainloop_barrier[2];
  __shared__ alignas(16) uint64_t epilogue_barrier[2];
  __shared__ alignas(16) uint32_t tmem_base;

  const uint32_t tma_barrier_base = ptx::smem_address(tma_barrier);
  const uint32_t mma_barrier_base = ptx::smem_address(mma_barrier);
  const uint32_t mainloop_barrier_base =
      ptx::smem_address(mainloop_barrier);
  const uint32_t epilogue_barrier_base =
      ptx::smem_address(epilogue_barrier);

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
      ptx::mbarrier_init(epilogue_barrier_base + stage * sizeof(uint64_t),
                         kEpilogueWarps);
    }
    ptx::fence_mbarrier_init_release_cluster();
  }
  if (warp == kMmaWarp) {
    ptx::tmem_alloc(ptx::smem_address(&tmem_base), TileN * 2);
  }
  __syncthreads();

  constexpr uint32_t instruction_descriptor =
      (1U << 4U) |
      (static_cast<uint32_t>(TileN) >> 3U << 17U) |
      (static_cast<uint32_t>(TileM) >> 4U << 24U);

  const int k_tiles = k / TileK;
  const int total_tiles = tiles_m * tiles_n;
  const int tiles_n_mask = tiles_n - 1;
  const int tiles_n_log2 = __ffs(tiles_n) - 1;
  const bool tiles_n_power2 = (tiles_n & tiles_n_mask) == 0;

  auto tile_coordinates = [&](int work_id, int& tile_m, int& tile_n) {
    if (tiles_n_power2) {
      tile_m = work_id >> tiles_n_log2;
      tile_n = work_id & tiles_n_mask;
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
    const int offset_m = tile_m * TileM;
    const int offset_n = tile_n * TileN;

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
    const uint32_t accumulator = tmem_base + tmem_stage * TileN;

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
      ptx::mma_f16(accumulator, descriptor_a, descriptor_b,
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
    if (warp >= kEpilogueWarps) return;
    const int offset_m = tile_m * TileM;
    const int offset_n = tile_n * TileN;
    if constexpr (TileM == 64) {
      if (m64_panel_stride < 0) {
        const int row = warp * ptx::kWarpSize + lane;
        if (row >= TileM) return;
        const int debug_panel_stride = -m64_panel_stride;
        const uint32_t tmem_row =
            (row & 15) + ((row >> 4) * debug_panel_stride);
        const uint32_t base_address =
            tmem_base + tmem_stage * TileN + (tmem_row << 16);
        for (int n_block = 0; n_block < TileN / 8; ++n_block) {
          float values[8];
          ptx::tmem_load_32x32b_x8(base_address + n_block * 8, values);
          float* dst = output +
                       static_cast<size_t>(offset_m + row) * n +
                       offset_n + n_block * 8;
          ptx::store_global_l1_no_allocate_v8_f32(dst, values);
        }
        return;
      }
      const int lane_group = lane >> 2;
      const int lane_pair = lane & 3;
      const int row_base = warp * 16;
      const uint32_t tmem_stage_base = tmem_base + tmem_stage * TileN;
      const uint32_t tmem_row = warp * m64_panel_stride;

      for (int col_group = 0; col_group < TileN / 64; ++col_group) {
        float values[32];
        const uint32_t address =
            tmem_stage_base + (tmem_row << 16) + col_group * 64;
        ptx::tmem_load_16x256b_x8(address, values);

#pragma unroll
        for (int d = 0; d < 2; ++d) {
          const int row = row_base + lane_group + d * 8;
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int col =
                offset_n + col_group * 64 + e * 8 + lane_pair * 2;
            const int value_index = e * 4 + d * 2;
            float* dst =
                output + static_cast<size_t>(offset_m + row) * n + col;
            ptx::store_global_l1_no_allocate_v2_f32(
                dst, values[value_index], values[value_index + 1]);
          }
        }
      }
    } else {
      const int column_base =
          kEpilogueWarps == 8 ? (warp >> 2) * (TileN / 2) : 0;
      const int column_blocks =
          kEpilogueWarps == 8 ? (TileN / 2) / 8 : TileN / 8;

      const int row = (warp & 3) * ptx::kWarpSize + lane;
      const uint32_t base_address =
          tmem_base + tmem_stage * TileN + (row << 16) + column_base;
      float* row_dst = output +
                       static_cast<size_t>(offset_m + row) * n +
                       offset_n + column_base;

      float values_even[8];
      float values_odd[8];
      ptx::tmem_load_32x32b_x8_no_wait(base_address, values_even);
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
          ptx::store_global_l1_no_allocate_v8_f32(dst, values_even);
        } else {
          ptx::store_global_l1_no_allocate_v8_f32(dst, values_odd);
        }
      }
    }
  };

  if (warp == kTmaWarp && ptx::elect_one()) {
    int tma_stage = 0;
    int mma_phase = 1;
    for (int work_id = static_cast<int>(blockIdx.x);
         work_id < total_tiles; work_id += static_cast<int>(gridDim.x)) {
      int tile_m = 0;
      int tile_n = 0;
      tile_coordinates(work_id, tile_m, tile_n);
      for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        ptx::mbarrier_wait(mma_barrier_base + tma_stage * sizeof(uint64_t),
                           mma_phase);
        issue_load(k_tile, tile_m, tile_n, tma_stage);
        tma_stage = (tma_stage + 1) % Stages;
        if (tma_stage == 0) mma_phase ^= 1;
      }
    }
  } else if (warp == kMmaWarp && ptx::elect_one()) {
    int tma_stage = 0;
    int tma_phase = 0;
    int tmem_stage = 0;
    int epilogue_phase = 1;
    for (int work_id = static_cast<int>(blockIdx.x);
         work_id < total_tiles; work_id += static_cast<int>(gridDim.x)) {
      ptx::mbarrier_wait(epilogue_barrier_base +
                             tmem_stage * sizeof(uint64_t),
                         epilogue_phase);
      for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        ptx::mbarrier_wait(tma_barrier_base +
                               tma_stage * sizeof(uint64_t),
                           tma_phase);
        ptx::tcgen05_fence_after_thread_sync();
        issue_mma(k_tile, tma_stage, tmem_stage);
        ptx::mma_commit(mma_barrier_base + tma_stage * sizeof(uint64_t));
        tma_stage = (tma_stage + 1) % Stages;
        if (tma_stage == 0) tma_phase ^= 1;
      }
      ptx::mma_commit(mainloop_barrier_base +
                      tmem_stage * sizeof(uint64_t));
      tmem_stage ^= 1;
      if (tmem_stage == 0) epilogue_phase ^= 1;
    }
  } else if (warp < kEpilogueWarps) {
    int tmem_stage = 0;
    int mainloop_phase = 0;
    for (int work_id = static_cast<int>(blockIdx.x);
         work_id < total_tiles; work_id += static_cast<int>(gridDim.x)) {
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

      if (ptx::elect_one()) {
        ptx::mbarrier_arrive(epilogue_barrier_base +
                             tmem_stage * sizeof(uint64_t));
      }
      tmem_stage ^= 1;
      if (tmem_stage == 0) mainloop_phase ^= 1;
    }
  }

  __syncthreads();
  if (warp == kMmaWarp) {
    ptx::tmem_dealloc(tmem_base, TileN * 2);
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
  (void)m64_panel_stride;
#endif
}

template <int TileN = 256, int TileK = 64, int Stages = 3>
__global__ __launch_bounds__(192)
void tc5l_b_reuse_m256n256_1sm_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk, float* output,
    int m, int n, int k, int tiles_m, int tiles_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kLocalTileM = 128;
  constexpr int kGroupTileM = 256;
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
  constexpr int kEpilogueWarps = 4;
  constexpr int kTmaWarp = 4;
  constexpr int kMmaWarp = 5;
  static_assert(TileK % kTmaK == 0);
  static_assert(TileK % kMmaK == 0);
  static_assert(TileN == 256);

  constexpr int kKChunks = TileK / kTmaK;
  constexpr int kAChunkBytes = kLocalTileM * kTmaK * sizeof(half);
  constexpr int kBChunkBytes = TileN * kTmaK * sizeof(half);
  constexpr int kAStageBytes = kKChunks * kAChunkBytes;
  constexpr int kBStageBytes = kKChunks * kBChunkBytes;
  constexpr int kStageBytes = 2 * kAStageBytes + kBStageBytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / ptx::kWarpSize;
  const int lane = tid % ptx::kWarpSize;

  extern __shared__ __align__(1024) char dynamic_smem[];
  const uint32_t smem = ptx::smem_address(dynamic_smem);

  __shared__ alignas(16) uint64_t tma_barrier[Stages];
  __shared__ alignas(16) uint64_t mma_barrier[Stages];
  __shared__ alignas(16) uint32_t tmem_base;

  const uint32_t tma_barrier_base = ptx::smem_address(tma_barrier);
  const uint32_t mma_barrier_base = ptx::smem_address(mma_barrier);

  if (warp == kMmaWarp && ptx::elect_one()) {
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
      ptx::mbarrier_init(tma_barrier_base + stage * sizeof(uint64_t), 1);
      ptx::mbarrier_init(mma_barrier_base + stage * sizeof(uint64_t), 1);
    }
    ptx::fence_mbarrier_init_release_cluster();
  }
  if (warp == kMmaWarp) {
    ptx::tmem_alloc(ptx::smem_address(&tmem_base), TileN * 2);
  }
  __syncthreads();

  constexpr uint32_t instruction_descriptor =
      (1U << 4U) |
      (static_cast<uint32_t>(TileN) >> 3U << 17U) |
      (static_cast<uint32_t>(kLocalTileM) >> 4U << 24U);

  int tma_phase[Stages] = {};
  int mma_phase[Stages] = {};
  const int k_tiles = k / TileK;
  const int total_tiles = tiles_m * tiles_n;

  auto tile_coordinates = [&](int work_id, int& tile_m, int& tile_n) {
    tile_m = work_id / tiles_n;
    tile_n = work_id - tile_m * tiles_n;
  };

  auto issue_load = [&](int k_tile, int tile_m, int tile_n) {
    if (warp != kTmaWarp || !ptx::elect_one()) return;

    const int stage = k_tile % Stages;
    const uint32_t barrier =
        tma_barrier_base + stage * sizeof(uint64_t);
    const uint32_t stage_smem = smem + stage * kStageBytes;
    const uint32_t a0_smem = stage_smem;
    const uint32_t a1_smem = a0_smem + kAStageBytes;
    const uint32_t b_smem = a1_smem + kAStageBytes;
    const int offset_k = k_tile * TileK;
    const int offset_m = tile_m * kGroupTileM;
    const int offset_n = tile_n * TileN;

#pragma unroll
    for (int chunk = 0; chunk < kKChunks; ++chunk) {
      const int chunk_k = offset_k + chunk * kTmaK;
      ptx::tma_load_2d(a0_smem + chunk * kAChunkBytes, &tensor_map_a,
                       chunk_k, offset_m, barrier);
      ptx::tma_load_2d(a1_smem + chunk * kAChunkBytes, &tensor_map_a,
                       chunk_k, offset_m + kLocalTileM, barrier);
      ptx::tma_load_2d(b_smem + chunk * kBChunkBytes, &tensor_map_b_nk,
                       chunk_k, offset_n, barrier);
    }
    ptx::mbarrier_arrive_expect_tx(
        barrier, 2 * kAStageBytes + kBStageBytes);
  };

  auto issue_mma = [&](int k_tile) {
    const int stage = k_tile % Stages;
    const uint32_t tma_barrier_address =
        tma_barrier_base + stage * sizeof(uint64_t);
    ptx::mbarrier_wait(tma_barrier_address, tma_phase[stage]);
    tma_phase[stage] ^= 1;
    ptx::tcgen05_fence_after_thread_sync();

    if (warp != kMmaWarp || !ptx::elect_one()) return;

    const uint32_t stage_smem = smem + stage * kStageBytes;
    const uint32_t a0_smem = stage_smem;
    const uint32_t a1_smem = a0_smem + kAStageBytes;
    const uint32_t b_smem = a1_smem + kAStageBytes;
    const uint32_t accumulator0 = tmem_base;
    const uint32_t accumulator1 = tmem_base + TileN;

#pragma unroll
    for (int k_block = 0; k_block < TileK / kMmaK; ++k_block) {
      const int chunk = k_block / (kTmaK / kMmaK);
      const int block_in_chunk = k_block % (kTmaK / kMmaK);
      const uint32_t a0_block =
          a0_smem + chunk * kAChunkBytes + block_in_chunk * 32;
      const uint32_t a1_block =
          a1_smem + chunk * kAChunkBytes + block_in_chunk * 32;
      const uint32_t b_block =
          b_smem + chunk * kBChunkBytes + block_in_chunk * 32;
      const uint64_t descriptor_a0 =
          ptx::sw128_k_major_descriptor(a0_block);
      const uint64_t descriptor_a1 =
          ptx::sw128_k_major_descriptor(a1_block);
      const uint64_t descriptor_b = ptx::sw128_k_major_descriptor(b_block);
      const bool accumulate = k_tile != 0 || k_block != 0;
      ptx::mma_f16(accumulator0, descriptor_a0, descriptor_b,
                   instruction_descriptor, accumulate);
      ptx::mma_f16(accumulator1, descriptor_a1, descriptor_b,
                   instruction_descriptor, accumulate);
    }
    ptx::mma_commit(mma_barrier_base + stage * sizeof(uint64_t));
  };

  auto store_accumulator = [&](int tile_m, int tile_n, int m_group,
                               uint32_t accumulator) {
    if (warp >= kEpilogueWarps) return;
    const int row = warp * ptx::kWarpSize + lane;
    const int offset_m = tile_m * kGroupTileM + m_group * kLocalTileM;
    const int offset_n = tile_n * TileN;
    const uint32_t base_address = accumulator + (row << 16);

    float values_even[8];
    float values_odd[8];
    ptx::tmem_load_32x32b_x8_no_wait(base_address, values_even);
    for (int n_block = 0; n_block < TileN / 8; ++n_block) {
      ptx::tmem_load_wait();
      const bool use_even = (n_block & 1) == 0;
      float* dst = output +
                   static_cast<size_t>(offset_m + row) * n +
                   offset_n + n_block * 8;
      if (n_block + 1 < TileN / 8) {
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

  for (int work_id = static_cast<int>(blockIdx.x);
       work_id < total_tiles; work_id += static_cast<int>(gridDim.x)) {
    int tile_m = 0;
    int tile_n = 0;
    tile_coordinates(work_id, tile_m, tile_n);

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
    store_accumulator(tile_m, tile_n, 0, tmem_base);
    store_accumulator(tile_m, tile_n, 1, tmem_base + TileN);
    __syncthreads();
  }

  __syncthreads();
  if (warp == kMmaWarp) {
    ptx::tmem_dealloc(tmem_base, TileN * 2);
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

template <int TileN = 128, int TileK = 64, int Stages = 4>
__global__ __launch_bounds__(192)
void tc5m_overlap_b_reuse_m256n128_1sm_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk, float* output,
    int m, int n, int k, int tiles_m, int tiles_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kLocalTileM = 128;
  constexpr int kGroupTileM = 256;
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
  constexpr int kEpilogueWarps = 4;
  constexpr int kTmaWarp = 4;
  constexpr int kMmaWarp = 5;
  static_assert(TileK % kTmaK == 0);
  static_assert(TileK % kMmaK == 0);
  static_assert(TileN == 128);

  constexpr int kKChunks = TileK / kTmaK;
  constexpr int kAChunkBytes = kLocalTileM * kTmaK * sizeof(half);
  constexpr int kBChunkBytes = TileN * kTmaK * sizeof(half);
  constexpr int kAStageBytes = kKChunks * kAChunkBytes;
  constexpr int kBStageBytes = kKChunks * kBChunkBytes;
  constexpr int kStageBytes = 2 * kAStageBytes + kBStageBytes;
  constexpr int kTmemColumnsPerBuffer = 2 * TileN;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / ptx::kWarpSize;
  const int lane = tid % ptx::kWarpSize;

  extern __shared__ __align__(1024) char dynamic_smem[];
  const uint32_t smem = ptx::smem_address(dynamic_smem);

  __shared__ alignas(16) uint64_t tma_barrier[Stages];
  __shared__ alignas(16) uint64_t mma_barrier[Stages];
  __shared__ alignas(16) uint64_t mainloop_barrier[2];
  __shared__ alignas(16) uint64_t epilogue_barrier[2];
  __shared__ alignas(16) uint32_t tmem_base;

  const uint32_t tma_barrier_base = ptx::smem_address(tma_barrier);
  const uint32_t mma_barrier_base = ptx::smem_address(mma_barrier);
  const uint32_t mainloop_barrier_base =
      ptx::smem_address(mainloop_barrier);
  const uint32_t epilogue_barrier_base =
      ptx::smem_address(epilogue_barrier);

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
      ptx::mbarrier_init(epilogue_barrier_base + stage * sizeof(uint64_t),
                         kEpilogueWarps);
    }
    ptx::fence_mbarrier_init_release_cluster();
  }
  if (warp == kMmaWarp) {
    ptx::tmem_alloc(ptx::smem_address(&tmem_base),
                    kTmemColumnsPerBuffer * 2);
  }
  __syncthreads();

  constexpr uint32_t instruction_descriptor =
      (1U << 4U) |
      (static_cast<uint32_t>(TileN) >> 3U << 17U) |
      (static_cast<uint32_t>(kLocalTileM) >> 4U << 24U);

  const int k_tiles = k / TileK;
  const int total_tiles = tiles_m * tiles_n;

  auto tile_coordinates = [&](int work_id, int& tile_m, int& tile_n) {
    tile_m = work_id / tiles_n;
    tile_n = work_id - tile_m * tiles_n;
  };

  auto issue_load = [&](int k_tile, int tile_m, int tile_n,
                        int tma_stage) {
    const uint32_t barrier =
        tma_barrier_base + tma_stage * sizeof(uint64_t);
    const uint32_t stage_smem = smem + tma_stage * kStageBytes;
    const uint32_t a0_smem = stage_smem;
    const uint32_t a1_smem = a0_smem + kAStageBytes;
    const uint32_t b_smem = a1_smem + kAStageBytes;
    const int offset_k = k_tile * TileK;
    const int offset_m = tile_m * kGroupTileM;
    const int offset_n = tile_n * TileN;

#pragma unroll
    for (int chunk = 0; chunk < kKChunks; ++chunk) {
      const int chunk_k = offset_k + chunk * kTmaK;
      ptx::tma_load_2d(a0_smem + chunk * kAChunkBytes, &tensor_map_a,
                       chunk_k, offset_m, barrier);
      ptx::tma_load_2d(a1_smem + chunk * kAChunkBytes, &tensor_map_a,
                       chunk_k, offset_m + kLocalTileM, barrier);
      ptx::tma_load_2d(b_smem + chunk * kBChunkBytes, &tensor_map_b_nk,
                       chunk_k, offset_n, barrier);
    }
    ptx::mbarrier_arrive_expect_tx(
        barrier, 2 * kAStageBytes + kBStageBytes);
  };

  auto issue_mma = [&](int k_tile, int tma_stage, int tmem_stage) {
    const uint32_t stage_smem = smem + tma_stage * kStageBytes;
    const uint32_t a0_smem = stage_smem;
    const uint32_t a1_smem = a0_smem + kAStageBytes;
    const uint32_t b_smem = a1_smem + kAStageBytes;
    const uint32_t accumulator_base =
        tmem_base + tmem_stage * kTmemColumnsPerBuffer;
    const uint32_t accumulator0 = accumulator_base;
    const uint32_t accumulator1 = accumulator_base + TileN;

#pragma unroll
    for (int k_block = 0; k_block < TileK / kMmaK; ++k_block) {
      const int chunk = k_block / (kTmaK / kMmaK);
      const int block_in_chunk = k_block % (kTmaK / kMmaK);
      const uint32_t a0_block =
          a0_smem + chunk * kAChunkBytes + block_in_chunk * 32;
      const uint32_t a1_block =
          a1_smem + chunk * kAChunkBytes + block_in_chunk * 32;
      const uint32_t b_block =
          b_smem + chunk * kBChunkBytes + block_in_chunk * 32;
      const uint64_t descriptor_a0 =
          ptx::sw128_k_major_descriptor(a0_block);
      const uint64_t descriptor_a1 =
          ptx::sw128_k_major_descriptor(a1_block);
      const uint64_t descriptor_b = ptx::sw128_k_major_descriptor(b_block);
      const bool accumulate = k_tile != 0 || k_block != 0;
      ptx::mma_f16(accumulator0, descriptor_a0, descriptor_b,
                   instruction_descriptor, accumulate);
      ptx::mma_f16(accumulator1, descriptor_a1, descriptor_b,
                   instruction_descriptor, accumulate);
    }
  };

  auto epilogue_sync = []() {
    asm volatile("bar.sync %0, %1;"
                 :
                 : "r"(1), "r"(kEpilogueWarps * ptx::kWarpSize)
                 : "memory");
  };

  auto store_accumulator = [&](int tile_m, int tile_n, int m_group,
                               uint32_t accumulator) {
    if (warp >= kEpilogueWarps) return;
    const int row = warp * ptx::kWarpSize + lane;
    const int offset_m = tile_m * kGroupTileM + m_group * kLocalTileM;
    const int offset_n = tile_n * TileN;
    const uint32_t base_address = accumulator + (row << 16);

    float values_even[8];
    float values_odd[8];
    ptx::tmem_load_32x32b_x8_no_wait(base_address, values_even);
    for (int n_block = 0; n_block < TileN / 8; ++n_block) {
      ptx::tmem_load_wait();
      const bool use_even = (n_block & 1) == 0;
      float* dst = output +
                   static_cast<size_t>(offset_m + row) * n +
                   offset_n + n_block * 8;
      if (n_block + 1 < TileN / 8) {
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
    const uint32_t accumulator_base =
        tmem_base + tmem_stage * kTmemColumnsPerBuffer;
    store_accumulator(tile_m, tile_n, 0, accumulator_base);
    store_accumulator(tile_m, tile_n, 1, accumulator_base + TileN);
  };

  if (warp == kTmaWarp && ptx::elect_one()) {
    int tma_stage = 0;
    int mma_phase = 1;
    for (int work_id = static_cast<int>(blockIdx.x);
         work_id < total_tiles; work_id += static_cast<int>(gridDim.x)) {
      int tile_m = 0;
      int tile_n = 0;
      tile_coordinates(work_id, tile_m, tile_n);
      for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        ptx::mbarrier_wait(mma_barrier_base + tma_stage * sizeof(uint64_t),
                           mma_phase);
        issue_load(k_tile, tile_m, tile_n, tma_stage);
        tma_stage = (tma_stage + 1) % Stages;
        if (tma_stage == 0) mma_phase ^= 1;
      }
    }
  } else if (warp == kMmaWarp && ptx::elect_one()) {
    int tma_stage = 0;
    int tma_phase = 0;
    int tmem_stage = 0;
    int epilogue_phase = 1;
    for (int work_id = static_cast<int>(blockIdx.x);
         work_id < total_tiles; work_id += static_cast<int>(gridDim.x)) {
      ptx::mbarrier_wait(epilogue_barrier_base +
                             tmem_stage * sizeof(uint64_t),
                         epilogue_phase);
      for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        ptx::mbarrier_wait(tma_barrier_base +
                               tma_stage * sizeof(uint64_t),
                           tma_phase);
        ptx::tcgen05_fence_after_thread_sync();
        issue_mma(k_tile, tma_stage, tmem_stage);
        ptx::mma_commit(mma_barrier_base + tma_stage * sizeof(uint64_t));
        tma_stage = (tma_stage + 1) % Stages;
        if (tma_stage == 0) tma_phase ^= 1;
      }
      ptx::mma_commit(mainloop_barrier_base +
                      tmem_stage * sizeof(uint64_t));
      tmem_stage ^= 1;
      if (tmem_stage == 0) epilogue_phase ^= 1;
    }
  } else if (warp < kEpilogueWarps) {
    int tmem_stage = 0;
    int mainloop_phase = 0;
    for (int work_id = static_cast<int>(blockIdx.x);
         work_id < total_tiles; work_id += static_cast<int>(gridDim.x)) {
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

      if (ptx::elect_one()) {
        ptx::mbarrier_arrive(epilogue_barrier_base +
                             tmem_stage * sizeof(uint64_t));
      }
      tmem_stage ^= 1;
      if (tmem_stage == 0) mainloop_phase ^= 1;
    }
  }

  __syncthreads();
  if (warp == kMmaWarp) {
    ptx::tmem_dealloc(tmem_base, kTmemColumnsPerBuffer * 2);
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

template <bool UseDynamic, int RunnerTileN = 256, int RunnerTileK = 128,
          int RunnerStages = 2>
class Tc5Runner {
 public:
  Tc5Runner(const half* a, const half* b_nk, float* d,
            int m, int n, int k)
      : a_(a), b_nk_(b_nk), output_(d), m_(m), n_(n), k_(k) {
    fast_m_ = (m_ / kTileM) * kTileM;
    fast_n_ = (n_ / kTileN) * kTileN;
    fast_k_ = (k_ / kTileK) * kTileK;
    has_fast_path_ =
        fast_m_ > 0 && fast_n_ > 0 && fast_k_ > 0 && n_ % 4 == 0 &&
        k_ % kSw128TmaLeadingDimensionAlignment == 0;

    if (has_fast_path_) {
      ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, fast_m_,
                                         fast_k_, k_, kTileM);
      ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, fast_n_,
                                         fast_k_, k_, kTileN);
    }

    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<UseDynamic, kTileN, kTileK,
                                       kStages>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 raw 1sm)");

    if constexpr (UseDynamic) {
      check_cuda(cudaMalloc(&work_counter_, sizeof(int)),
                 "cudaMalloc(tc5 work counter)");
    }
  }

  ~Tc5Runner() {
    if (work_counter_ != nullptr) {
      cudaFree(work_counter_);
    }
  }

  void launch() {
    if (!has_fast_path_) {
      launch_cleanup();
      return;
    }

    if constexpr (UseDynamic) {
      check_cuda(cudaMemset(work_counter_, 0, sizeof(int)),
                 "cudaMemset(tc5 work counter)");
    }

    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = fast_m_ / kTileM;
    const int tiles_n = fast_n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5 raw 1sm)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 raw 1sm)");
    int active_blocks_per_sm = 1;
    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<UseDynamic, kTileN, kTileK,
                                       kStages>;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks_per_sm, kernel, 128, smem_bytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(tc5 raw 1sm)");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    const int worker_ctas = std::min(
        total_tiles,
        std::max(1, properties.multiProcessorCount * active_blocks_per_sm));

    tc5_raw_persistent_1sm_kernel<UseDynamic, kTileN, kTileK, kStages>
        <<<worker_ctas, 128, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, fast_k_,
            tiles_m, tiles_n, work_counter_);
    check_cuda(cudaGetLastError(),
               "tc5_raw_persistent_1sm_kernel launch");
    launch_cleanup();
  }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = RunnerTileN;
  static constexpr int kTileK = RunnerTileK;
  static constexpr int kStages = RunnerStages;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                 cudaGetErrorString(status));
    std::abort();
  }

  void launch_cleanup() {
    if (has_fast_path_ && fast_m_ == m_ && fast_n_ == n_ &&
        fast_k_ == k_) {
      return;
    }

    dim3 block(16, 16, 1);
    dim3 grid((n_ + static_cast<int>(block.x) - 1) /
                  static_cast<int>(block.x),
              (m_ + static_cast<int>(block.y) - 1) /
                  static_cast<int>(block.y),
              1);
    tc5_boundary_cleanup_kernel<<<grid, block>>>(
        a_, b_nk_, output_, m_, n_, k_,
        has_fast_path_ ? fast_m_ : 0, has_fast_path_ ? fast_n_ : 0,
        has_fast_path_ ? fast_k_ : 0);
    check_cuda(cudaGetLastError(), "tc5_boundary_cleanup_kernel launch");
  }

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b_{};
  const half* a_ = nullptr;
  const half* b_nk_ = nullptr;
  float* output_ = nullptr;
  int* work_counter_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  int fast_m_ = 0;
  int fast_n_ = 0;
  int fast_k_ = 0;
  bool has_fast_path_ = false;
};

template <int RunnerTileM = 128, int RunnerTileN = 256,
          int RunnerTileK = 64, int RunnerStages = 4,
          int RunnerEpilogueWarps =
              (RunnerTileM == 64 ? 4 : RunnerTileM / 32)>
class Tc5OverlapRunner {
 public:
  Tc5OverlapRunner(const half* a, const half* b_nk, float* d,
                   int m, int n, int k)
      : a_(a), b_nk_(b_nk), output_(d), m_(m), n_(n), k_(k) {
    fast_m_ = (m_ / kTileM) * kTileM;
    fast_n_ = (n_ / kTileN) * kTileN;
    fast_k_ = (k_ / kTileK) * kTileK;
    has_fast_path_ =
        fast_m_ > 0 && fast_n_ > 0 && fast_k_ > 0 && n_ % 4 == 0 &&
        k_ % kSw128TmaLeadingDimensionAlignment == 0;

    if (has_fast_path_) {
      ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, fast_m_,
                                         fast_k_, k_, kTileM);
      ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, fast_n_,
                                         fast_k_, k_, kTileN);
    }

    auto* kernel =
        &tc5h_overlap_epilogue_1sm_kernel<kTileM, kTileN, kTileK,
                                          kStages, kEpilogueWarps>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5h overlap epilogue)");
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                   cudaSharedmemCarveoutMaxShared),
               "cudaFuncSetAttribute(tc5h shared memory carveout)");
    if (has_fast_path_) {
      const int tiles_m = fast_m_ / kTileM;
      const int tiles_n = fast_n_ / kTileN;
      const int total_tiles = tiles_m * tiles_n;

      int device = 0;
      cudaDeviceProp properties{};
      check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5h)");
      check_cuda(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties(tc5h)");
      const int sm_count = std::max(1, properties.multiProcessorCount);
      worker_ctas_ = std::min(total_tiles, sm_count);
      if (total_tiles > sm_count && total_tiles <= 2 * sm_count) {
        worker_ctas_ = (total_tiles + 1) / 2;
      }
      if (const char* override = std::getenv("TC5H_WORKERS")) {
        const int requested = std::atoi(override);
        if (requested > 0) {
          worker_ctas_ = std::min(total_tiles, requested);
        }
      }
    }

    if constexpr (kTileM == 64) {
      if (const char* override = std::getenv("TC5K_M64_PANEL_STRIDE")) {
        const int requested = std::atoi(override);
        if (requested != 0) {
          m64_panel_stride_ = requested;
        }
      }
    }

  }

  void launch() {
    if (!has_fast_path_) {
      launch_cleanup();
      return;
    }

    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = fast_m_ / kTileM;
    const int tiles_n = fast_n_ / kTileN;
    tc5h_overlap_epilogue_1sm_kernel<kTileM, kTileN, kTileK, kStages,
                                     kEpilogueWarps>
        <<<worker_ctas_, kThreads, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, fast_k_,
            tiles_m, tiles_n, m64_panel_stride_);
    check_cuda(cudaGetLastError(),
               "tc5h_overlap_epilogue_1sm_kernel launch");
    if (fast_m_ != m_ || fast_n_ != n_ || fast_k_ != k_) {
      launch_cleanup();
    }
  }

 private:
  static constexpr int kTileM = RunnerTileM;
  static constexpr int kTileN = RunnerTileN;
  static constexpr int kTileK = RunnerTileK;
  static constexpr int kStages = RunnerStages;
  static constexpr int kEpilogueWarps = RunnerEpilogueWarps;
  static constexpr int kThreads = (kEpilogueWarps + 2) * 32;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                 cudaGetErrorString(status));
    std::abort();
  }

  void launch_cleanup() {
    if (has_fast_path_ && fast_m_ == m_ && fast_n_ == n_ &&
        fast_k_ == k_) {
      return;
    }

    dim3 block(16, 16, 1);
    dim3 grid((n_ + static_cast<int>(block.x) - 1) /
                  static_cast<int>(block.x),
              (m_ + static_cast<int>(block.y) - 1) /
                  static_cast<int>(block.y),
              1);
    tc5_boundary_cleanup_kernel<<<grid, block>>>(
        a_, b_nk_, output_, m_, n_, k_,
        has_fast_path_ ? fast_m_ : 0, has_fast_path_ ? fast_n_ : 0,
        has_fast_path_ ? fast_k_ : 0);
    check_cuda(cudaGetLastError(), "tc5h cleanup kernel launch");
  }

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b_{};
  const half* a_ = nullptr;
  const half* b_nk_ = nullptr;
  float* output_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  int fast_m_ = 0;
  int fast_n_ = 0;
  int fast_k_ = 0;
  int worker_ctas_ = 1;
  int m64_panel_stride_ = 32;
  bool has_fast_path_ = false;
};

template <int RunnerTileN = 256, int RunnerTileK = 64,
          int RunnerStages = 3>
class Tc5BReuseRunner {
 public:
  Tc5BReuseRunner(const half* a, const half* b_nk, float* d,
                  int m, int n, int k)
      : a_(a), b_nk_(b_nk), output_(d), m_(m), n_(n), k_(k) {
    fast_m_ = (m_ / kGroupTileM) * kGroupTileM;
    fast_n_ = (n_ / kTileN) * kTileN;
    fast_k_ = (k_ / kTileK) * kTileK;
    has_fast_path_ =
        fast_m_ > 0 && fast_n_ > 0 && fast_k_ > 0 && n_ % 4 == 0 &&
        k_ % kSw128TmaLeadingDimensionAlignment == 0;

    if (has_fast_path_) {
      ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, fast_m_,
                                         fast_k_, k_, kLocalTileM);
      ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, fast_n_,
                                         fast_k_, k_, kTileN);
    }

    auto* kernel =
        &tc5l_b_reuse_m256n256_1sm_kernel<kTileN, kTileK, kStages>;
    constexpr int smem_bytes =
        kStages * (2 * kLocalTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5l b-reuse)");
  }

  void launch() {
    if (!has_fast_path_) {
      launch_cleanup();
      return;
    }

    constexpr int smem_bytes =
        kStages * (2 * kLocalTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = fast_m_ / kGroupTileM;
    const int tiles_n = fast_n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5l)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5l)");
    const int sm_count = std::max(1, properties.multiProcessorCount);
    int worker_ctas = std::min(total_tiles, sm_count);
    if (const char* override = std::getenv("TC5L_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas = std::min(total_tiles, requested);
      }
    }

    tc5l_b_reuse_m256n256_1sm_kernel<kTileN, kTileK, kStages>
        <<<worker_ctas, kThreads, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, fast_k_,
            tiles_m, tiles_n);
    check_cuda(cudaGetLastError(),
               "tc5l_b_reuse_m256n256_1sm_kernel launch");
    launch_cleanup();
  }

 private:
  static constexpr int kLocalTileM = 128;
  static constexpr int kGroupTileM = 256;
  static constexpr int kTileN = RunnerTileN;
  static constexpr int kTileK = RunnerTileK;
  static constexpr int kStages = RunnerStages;
  static constexpr int kThreads = 192;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                 cudaGetErrorString(status));
    std::abort();
  }

  void launch_cleanup() {
    if (has_fast_path_ && fast_m_ == m_ && fast_n_ == n_ &&
        fast_k_ == k_) {
      return;
    }

    dim3 block(16, 16, 1);
    dim3 grid((n_ + static_cast<int>(block.x) - 1) /
                  static_cast<int>(block.x),
              (m_ + static_cast<int>(block.y) - 1) /
                  static_cast<int>(block.y),
              1);
    tc5_boundary_cleanup_kernel<<<grid, block>>>(
        a_, b_nk_, output_, m_, n_, k_,
        has_fast_path_ ? fast_m_ : 0, has_fast_path_ ? fast_n_ : 0,
        has_fast_path_ ? fast_k_ : 0);
    check_cuda(cudaGetLastError(), "tc5l cleanup kernel launch");
  }

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b_{};
  const half* a_ = nullptr;
  const half* b_nk_ = nullptr;
  float* output_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  int fast_m_ = 0;
  int fast_n_ = 0;
  int fast_k_ = 0;
  bool has_fast_path_ = false;
};

template <int RunnerTileN = 128, int RunnerTileK = 64,
          int RunnerStages = 4>
class Tc5OverlapBReuseRunner {
 public:
  Tc5OverlapBReuseRunner(const half* a, const half* b_nk, float* d,
                         int m, int n, int k)
      : a_(a), b_nk_(b_nk), output_(d), m_(m), n_(n), k_(k) {
    fast_m_ = (m_ / kGroupTileM) * kGroupTileM;
    fast_n_ = (n_ / kTileN) * kTileN;
    fast_k_ = (k_ / kTileK) * kTileK;
    has_fast_path_ =
        fast_m_ > 0 && fast_n_ > 0 && fast_k_ > 0 && n_ % 4 == 0 &&
        k_ % kSw128TmaLeadingDimensionAlignment == 0;

    if (has_fast_path_) {
      ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, fast_m_,
                                         fast_k_, k_, kLocalTileM);
      ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, fast_n_,
                                         fast_k_, k_, kTileN);
    }

    auto* kernel =
        &tc5m_overlap_b_reuse_m256n128_1sm_kernel<kTileN, kTileK,
                                                  kStages>;
    constexpr int smem_bytes =
        kStages * (2 * kLocalTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5m overlap b-reuse)");
  }

  void launch() {
    if (!has_fast_path_) {
      launch_cleanup();
      return;
    }

    constexpr int smem_bytes =
        kStages * (2 * kLocalTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = fast_m_ / kGroupTileM;
    const int tiles_n = fast_n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5m)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5m)");
    const int sm_count = std::max(1, properties.multiProcessorCount);
    int worker_ctas = std::min(total_tiles, sm_count);
    if (total_tiles > sm_count && total_tiles <= 2 * sm_count) {
      worker_ctas = (total_tiles + 1) / 2;
    }
    if (const char* override = std::getenv("TC5M_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas = std::min(total_tiles, requested);
      }
    }

    tc5m_overlap_b_reuse_m256n128_1sm_kernel<kTileN, kTileK, kStages>
        <<<worker_ctas, kThreads, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, fast_k_,
            tiles_m, tiles_n);
    check_cuda(cudaGetLastError(),
               "tc5m_overlap_b_reuse_m256n128_1sm_kernel launch");
    launch_cleanup();
  }

 private:
  static constexpr int kLocalTileM = 128;
  static constexpr int kGroupTileM = 256;
  static constexpr int kTileN = RunnerTileN;
  static constexpr int kTileK = RunnerTileK;
  static constexpr int kStages = RunnerStages;
  static constexpr int kThreads = 192;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                 cudaGetErrorString(status));
    std::abort();
  }

  void launch_cleanup() {
    if (has_fast_path_ && fast_m_ == m_ && fast_n_ == n_ &&
        fast_k_ == k_) {
      return;
    }

    dim3 block(16, 16, 1);
    dim3 grid((n_ + static_cast<int>(block.x) - 1) /
                  static_cast<int>(block.x),
              (m_ + static_cast<int>(block.y) - 1) /
                  static_cast<int>(block.y),
              1);
    tc5_boundary_cleanup_kernel<<<grid, block>>>(
        a_, b_nk_, output_, m_, n_, k_,
        has_fast_path_ ? fast_m_ : 0, has_fast_path_ ? fast_n_ : 0,
        has_fast_path_ ? fast_k_ : 0);
    check_cuda(cudaGetLastError(), "tc5m cleanup kernel launch");
  }

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b_{};
  const half* a_ = nullptr;
  const half* b_nk_ = nullptr;
  float* output_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  int fast_m_ = 0;
  int fast_n_ = 0;
  int fast_k_ = 0;
  bool has_fast_path_ = false;
};

using Tc5aRunner = Tc5Runner<false>;
using Tc5bRunner = Tc5Runner<true>;
using Tc5cRunner = Tc5Runner<false, 128, 128, 2>;
using Tc5dRunner = Tc5Runner<false, 256, 64, 2>;
using Tc5eRunner = Tc5Runner<false, 128, 64, 2>;
using Tc5fRunner = Tc5Runner<false, 256, 128, 1>;
using Tc5gRunner = Tc5Runner<false, 256, 64, 1>;
using Tc5hRunner = Tc5OverlapRunner<128, 256, 64, 4>;
using Tc5iRunner = Tc5OverlapRunner<128, 128, 64, 6>;
using Tc5jRunner = Tc5OverlapRunner<128, 256, 128, 2>;
using Tc5kRunner = Tc5OverlapRunner<64, 256, 64, 5>;
using Tc5lRunner = Tc5BReuseRunner<256, 64, 2>;
using Tc5mRunner = Tc5OverlapBReuseRunner<128, 64, 4>;

}  // namespace gemm_sm110::backends
