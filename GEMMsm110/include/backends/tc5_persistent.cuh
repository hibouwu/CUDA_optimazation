#pragma once

// Stage 5 persistent schedulers built on the validated tc4a 1-SM path.
//
// Unless a kernel documents a different shape or warp assignment, the shared
// computation path is:
//   - a 128x256 output tile and a TCGen05 FP16 x FP16 -> FP32 mainloop;
//   - SW128 K-major TMA tiles, with each TMA transaction covering K=64;
//   - warp 0 issuing TCGen05 MMA and warp 1 issuing TMA loads;
//   - all 128 threads reading the accumulator from TMEM and storing FP32 D.
//
// Input layout contract:
//   A is logically MxK and B is logically KxN, but the TMA descriptor for B
//   addresses the pre-transposed physical NxK buffer named b_nk.  Output is
//   row-major unless a runner explicitly selects a transposed-store variant.
//
// Synchronization contract:
//   tma_barrier[s] protects "TMA has filled SMEM stage s"; mma_barrier[s]
//   protects "TCGen05 has stopped reading SMEM stage s".  Barrier phase bits
//   toggle whenever a circular buffer wraps.  The overlapped kernels add a
//   two-entry mainloop/epilogue handshake for double-buffered TMEM.  Changing
//   a wait, arrive, commit, fence, or phase update can therefore create a
//   cross-iteration race even when the code still compiles.
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

// SW128 tensor maps require the physical K-leading dimension to be aligned to
// 64 elements.  Runners use this constant to reject shapes before encoding a
// descriptor that the kernels cannot consume safely.
constexpr int kSw128TmaLeadingDimensionAlignment = 64;

// Completes the rows, columns, or K suffix excluded from the aligned TCGen05
// fast path.  If an element already contains the fast-path prefix dot product,
// accumulation resumes at fast_k; otherwise the complete scalar dot product
// is evaluated.  This kernel is a correctness fallback, not a tuned epilogue.
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

// Split-K writes one full MxN partial matrix per split.  This second kernel
// performs the deterministic elementwise reduction into the user output.
template <int SplitK>
__global__ void tc5_splitk_reduce_kernel(const float* partials,
                                         float* output,
                                         size_t elements) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (idx >= elements) return;
  float value = 0.0f;
#pragma unroll
  for (int split = 0; split < SplitK; ++split) {
    value += partials[static_cast<size_t>(split) * elements + idx];
  }
  output[idx] = value;
}

// Baseline 1-SM persistent kernel.  Each resident CTA uses a static grid-stride
// sequence of (split, tile_m, tile_n) work items, avoiding a global work
// counter.  The same implementation supplies row-major, transposed-store,
// M=64, strided-output, and Split-K runners through template parameters.
//
// Required fast-path bounds are supplied by the runner: M/N/K must already be
// rounded to complete tiles, and each Split-K slice must contain an integral
// number of TileK chunks.  output_stride applies only to row-major output;
// Split-K places consecutive partial matrices at split * M * output_stride.
template <int TileN = 256, int TileK = 128, int Stages = 2,
          bool StoreTransposed = false, int TileM = 128,
          bool StoreTransposedViaSmem = false>
__global__ __launch_bounds__(128)
void tc5_raw_persistent_1sm_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk, float* output,
    int m, int n, int k, int tiles_m, int tiles_n,
    int split_k_count = 1, int output_stride = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kTileM = TileM;
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
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
  const int lane = tid & (ptx::kWarpSize - 1);

  extern __shared__ __align__(1024) char dynamic_smem[];
  const uint32_t smem = ptx::smem_address(dynamic_smem);

  // One producer/consumer handshake per circular SMEM stage.  TMEM is CTA-wide
  // storage; the elected lane performs control instructions, while all lanes
  // later participate in readback.
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

  // Phase bits are local replicas of each mbarrier generation.  They toggle
  // after a wait because the next use of the same stage belongs to the next
  // circular-buffer generation.
  int tma_phase[Stages] = {};
  int mma_phase[Stages] = {};
  const int slice_k = k / split_k_count;
  const int k_tiles = slice_k / TileK;
  const int tiles_per_split = tiles_m * tiles_n;
  const int total_tiles = tiles_per_split * split_k_count;
  int static_work_id = static_cast<int>(blockIdx.x);

  // Broadcast one CTA-wide work id.  Every CTA owns blockIdx.x + q*gridDim.x;
  // this is static persistent scheduling, not CLC and not an atomic queue.
  auto fetch_work = [&]() {
    if (warp == 0 && ptx::elect_one()) {
      shared_work_id = static_work_id;
      static_work_id += static_cast<int>(gridDim.x);
    }
    __syncthreads();
    return shared_work_id;
  };

  // Warp 1's elected lane produces one SMEM stage.  The expected byte count
  // must exactly match all TMA requests associated with this barrier.
  auto issue_load = [&](int k_tile, int tile_m, int tile_n,
                        int split_k_start) {
    if (warp != 1 || !ptx::elect_one()) return;

    const int stage = k_tile % Stages;
    const uint32_t barrier =
        tma_barrier_base + stage * sizeof(uint64_t);
    const uint32_t stage_smem = smem + stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b_smem = stage_smem + kAStageBytes;
    const int offset_k = split_k_start + k_tile * TileK;
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

  // All threads first observe TMA completion; then warp 0's elected lane emits
  // the TCGen05 sequence.  mma_commit releases the SMEM stage for reuse only
  // after TCGen05 has finished consuming its descriptors.
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

  // Read one completed accumulator tile from TMEM.  Direct row-major stores are
  // vectorized across N.  A transposed output either uses scalar stores or an
  // explicit SMEM transpose so that the final global writes remain vectorized.
  auto store_tile = [&](int tile_m, int tile_n, int split) {
    const int offset_m = tile_m * kTileM;
    const int offset_n = tile_n * TileN;
    const int output_ld = output_stride > 0 ? output_stride : n;
    float* output_base =
        output + static_cast<size_t>(split) * m * output_ld;

    if constexpr (StoreTransposed && StoreTransposedViaSmem &&
                  kTileM == 128) {
      float* transpose_smem =
          reinterpret_cast<float*>(dynamic_smem + Stages * kStageBytes);
      for (int n_block = 0; n_block < TileN / 8; ++n_block) {
        float values[8];
        const uint32_t address =
            tmem_base + ((warp * 32) << 16) + n_block * 8;
        ptx::tmem_load_32x32b_x8(address, values);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          transpose_smem[(n_block * 8 + i) * kTileM + tid] = values[i];
        }
      }
      __syncthreads();
      for (int idx = tid; idx < TileN * kTileM; idx += 128) {
        const int row = idx / kTileM;
        const int col = idx - row * kTileM;
        output_base[static_cast<size_t>(offset_n + row) * m +
                    offset_m + col] = transpose_smem[idx];
      }
      return;
    }

    if constexpr (kTileM == 64) {
      const int row_group = warp;
      const int row = row_group * 16 + lane;
      if (row_group >= 4) return;
      for (int n_block = 0; n_block < TileN / 8; ++n_block) {
        float values[8];
        const uint32_t address =
            tmem_base + ((row_group * 32) << 16) + n_block * 8;
        ptx::tmem_load_32x32b_x8(address, values);
        if (lane >= 16) continue;
        if constexpr (StoreTransposed) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            ptx::store_global_l1_no_allocate_f32(
                output_base +
                    static_cast<size_t>(offset_n + n_block * 8 + i) * m +
                    offset_m + row,
                values[i]);
          }
        } else {
          float* dst = output_base +
                       static_cast<size_t>(offset_m + row) * output_ld +
                       offset_n + n_block * 8;
          ptx::store_global_l1_no_allocate_v8_f32(dst, values);
        }
      }
      return;
    }

    if (tid >= kTileM) return;
    for (int n_block = 0; n_block < TileN / 8; ++n_block) {
      float values[8];
      const uint32_t address =
          tmem_base + ((warp * 32) << 16) + n_block * 8;
      ptx::tmem_load_32x32b_x8(address, values);
      if constexpr (StoreTransposed) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          ptx::store_global_l1_no_allocate_f32(
              output_base +
                  static_cast<size_t>(offset_n + n_block * 8 + i) * m +
                  offset_m + tid,
              values[i]);
        }
      } else {
        float* dst = output_base +
                     static_cast<size_t>(offset_m + tid) * output_ld +
                     offset_n + n_block * 8;
        ptx::store_global_l1_no_allocate_v8_f32(dst, values);
      }
    }
  };

  // For every output tile: prefill the circular pipeline, alternate MMA with
  // refills, drain all outstanding MMA consumers, then reuse the sole TMEM
  // accumulator for the next work item.
  while (true) {
    const int work_id = fetch_work();
    if (work_id >= total_tiles) break;

    const int split = work_id / tiles_per_split;
    const int tile_work_id = work_id - split * tiles_per_split;
    const int tile_m = tile_work_id / tiles_n;
    const int tile_n = tile_work_id - tile_m * tiles_n;
    const int split_k_start = split * slice_k;

    const int prologue = k_tiles < Stages ? k_tiles : Stages;
    for (int k_tile = 0; k_tile < prologue; ++k_tile) {
      issue_load(k_tile, tile_m, tile_n, split_k_start);
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
        issue_load(reuse_tile, tile_m, tile_n, split_k_start);
      }
    }

    const int final_stages = k_tiles < Stages ? k_tiles : Stages;
    for (int stage = 0; stage < final_stages; ++stage) {
      ptx::mbarrier_wait(mma_barrier_base + stage * sizeof(uint64_t),
                         mma_phase[stage]);
      mma_phase[stage] ^= 1;
    }

    ptx::tcgen05_fence_after_thread_sync();
    store_tile(tile_m, tile_n, split);
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
  (void)split_k_count;
  (void)output_stride;
#endif
}

// N-pair specialization for tail-oriented shapes.  Two adjacent M128xN256
// tiles share the same A load and occupy consecutive TMEM column ranges.  The
// final pair may contain one tile; pair_count keeps its TMA byte contract,
// TCGen05 issue count, and store loop consistent.
template <int TileK = 64, int Stages = 2>
__global__ __launch_bounds__(128)
void tc5_tail_mn_pair_n_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk, float* output,
    int m, int n, int k, int tiles_m, int tile_pairs_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kTileM = 128;
  constexpr int kTileN = 256;
  constexpr int kPairColumns = 2 * kTileN;
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
  static_assert(TileK % kTmaK == 0);
  static_assert(TileK % kMmaK == 0);

  constexpr int kKChunks = TileK / kTmaK;
  constexpr int kAChunkBytes = kTileM * kTmaK * sizeof(half);
  constexpr int kBChunkBytes = kTileN * kTmaK * sizeof(half);
  constexpr int kAStageBytes = kKChunks * kAChunkBytes;
  constexpr int kBStageBytes = kKChunks * kBChunkBytes;
  constexpr int kStageBytes = kAStageBytes + 2 * kBStageBytes;

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
    ptx::tmem_alloc(ptx::smem_address(&tmem_base), kPairColumns);
  }
  __syncthreads();

  constexpr uint32_t instruction_descriptor =
      (1U << 4U) |
      (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
      (static_cast<uint32_t>(kTileM) >> 4U << 24U);

  int tma_phase[Stages] = {};
  int mma_phase[Stages] = {};
  const int k_tiles = k / TileK;
  const int total_tiles = tiles_m * tile_pairs_n;
  int static_work_id = static_cast<int>(blockIdx.x);

  // Static persistent work assignment; shared_work_id makes all four warps
  // agree on the pair before any warp starts touching barriers or TMEM.
  auto fetch_work = [&]() {
    if (warp == 0 && ptx::elect_one()) {
      shared_work_id = static_work_id;
      static_work_id += static_cast<int>(gridDim.x);
    }
    __syncthreads();
    return shared_work_id;
  };

  // Load A once and one or two B tiles.  The expected transaction bytes vary
  // with pair_count, so an absent tail tile never leaves the barrier waiting.
  auto issue_load = [&](int k_tile, int tile_m, int pair_n,
                        int pair_count) {
    if (warp != 1 || !ptx::elect_one()) return;

    const int stage = k_tile % Stages;
    const uint32_t barrier =
        tma_barrier_base + stage * sizeof(uint64_t);
    const uint32_t stage_smem = smem + stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b0_smem = stage_smem + kAStageBytes;
    const uint32_t b1_smem = b0_smem + kBStageBytes;
    const int offset_k = k_tile * TileK;
    const int offset_m = tile_m * kTileM;
    const int offset_n = pair_n * kPairColumns;

#pragma unroll
    for (int chunk = 0; chunk < kKChunks; ++chunk) {
      const int chunk_k = offset_k + chunk * kTmaK;
      ptx::tma_load_2d(a_smem + chunk * kAChunkBytes, &tensor_map_a,
                       chunk_k, offset_m, barrier);
      ptx::tma_load_2d(b0_smem + chunk * kBChunkBytes, &tensor_map_b_nk,
                       chunk_k, offset_n, barrier);
      if (pair_count == 2) {
        ptx::tma_load_2d(b1_smem + chunk * kBChunkBytes, &tensor_map_b_nk,
                         chunk_k, offset_n + kTileN, barrier);
      }
    }
    ptx::mbarrier_arrive_expect_tx(
        barrier, kAStageBytes + pair_count * kBStageBytes);
  };

  auto issue_mma = [&](int k_tile, int pair_count) {
    const int stage = k_tile % Stages;
    const uint32_t tma_barrier_address =
        tma_barrier_base + stage * sizeof(uint64_t);
    ptx::mbarrier_wait(tma_barrier_address, tma_phase[stage]);
    tma_phase[stage] ^= 1;
    ptx::tcgen05_fence_after_thread_sync();

    if (warp != 0 || !ptx::elect_one()) return;

    const uint32_t stage_smem = smem + stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b0_smem = stage_smem + kAStageBytes;
    const uint32_t b1_smem = b0_smem + kBStageBytes;

#pragma unroll
    for (int k_block = 0; k_block < TileK / kMmaK; ++k_block) {
      const int chunk = k_block / (kTmaK / kMmaK);
      const int block_in_chunk = k_block % (kTmaK / kMmaK);
      const uint32_t a_block =
          a_smem + chunk * kAChunkBytes + block_in_chunk * 32;
      const uint32_t b0_block =
          b0_smem + chunk * kBChunkBytes + block_in_chunk * 32;
      const uint64_t descriptor_a = ptx::sw128_k_major_descriptor(a_block);
      const uint64_t descriptor_b0 =
          ptx::sw128_k_major_descriptor(b0_block);
      const bool accumulate = k_tile != 0 || k_block != 0;
      ptx::mma_f16(tmem_base, descriptor_a, descriptor_b0,
                   instruction_descriptor, accumulate);
      if (pair_count == 2) {
        const uint32_t b1_block =
            b1_smem + chunk * kBChunkBytes + block_in_chunk * 32;
        const uint64_t descriptor_b1 =
            ptx::sw128_k_major_descriptor(b1_block);
        ptx::mma_f16(tmem_base + kTileN, descriptor_a, descriptor_b1,
                     instruction_descriptor, accumulate);
      }
    }
    ptx::mma_commit(mma_barrier_base + stage * sizeof(uint64_t));
  };

  auto store_tile = [&](int tile_m, int pair_n, int pair_count) {
    const int offset_m = tile_m * kTileM;
    const int offset_n = pair_n * kPairColumns;
    if (tid >= kTileM) return;

    for (int pair = 0; pair < pair_count; ++pair) {
      const uint32_t pair_tmem = tmem_base + pair * kTileN;
      const int pair_offset_n = offset_n + pair * kTileN;
      for (int n_block = 0; n_block < kTileN / 8; ++n_block) {
        float values[8];
        const uint32_t address =
            pair_tmem + ((warp * 32) << 16) + n_block * 8;
        ptx::tmem_load_32x32b_x8(address, values);
        float* dst = output +
                     static_cast<size_t>(offset_m + tid) * n +
                     pair_offset_n + n_block * 8;
        ptx::store_global_l1_no_allocate_v8_f32(dst, values);
      }
    }
  };

  while (true) {
    const int work_id = fetch_work();
    if (work_id >= total_tiles) break;

    const int tile_m = work_id / tile_pairs_n;
    const int pair_n = work_id - tile_m * tile_pairs_n;
    const int pair_count =
        (pair_n + 1) * kPairColumns <= n ? 2 : 1;

    const int prologue = k_tiles < Stages ? k_tiles : Stages;
    for (int k_tile = 0; k_tile < prologue; ++k_tile) {
      issue_load(k_tile, tile_m, pair_n, pair_count);
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
      const int stage = k_tile % Stages;
      issue_mma(k_tile, pair_count);

      const int reuse_tile = k_tile + Stages;
      if (reuse_tile < k_tiles) {
        const uint32_t mma_barrier_address =
            mma_barrier_base + stage * sizeof(uint64_t);
        ptx::mbarrier_wait(mma_barrier_address, mma_phase[stage]);
        mma_phase[stage] ^= 1;
        issue_load(reuse_tile, tile_m, pair_n, pair_count);
      }
    }

    const int final_stages = k_tiles < Stages ? k_tiles : Stages;
    for (int stage = 0; stage < final_stages; ++stage) {
      ptx::mbarrier_wait(mma_barrier_base + stage * sizeof(uint64_t),
                         mma_phase[stage]);
      mma_phase[stage] ^= 1;
    }

    ptx::tcgen05_fence_after_thread_sync();
    store_tile(tile_m, pair_n, pair_count);
    __syncthreads();
  }

  __syncthreads();
  if (warp == 0) {
    ptx::tmem_dealloc(tmem_base, kPairColumns);
  }
#else
  (void)tensor_map_a;
  (void)tensor_map_b_nk;
  (void)output;
  (void)m;
  (void)n;
  (void)k;
  (void)tiles_m;
  (void)tile_pairs_n;
#endif
}

// M-pair counterpart of tc5_tail_mn_pair_n_kernel.  Two vertically adjacent
// M128 tiles share B, while A0/A1 and their accumulators remain independent.
// StoreTransposed selects D[N,M] addressing without changing the GEMM math.
template <int TileK = 64, int Stages = 2, int PairTileN = 256,
          bool StoreTransposed = false>
__global__ __launch_bounds__(128)
void tc5_tail_mn_pair_m_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk, float* output,
    int m, int n, int k, int tile_pairs_m, int tiles_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kTileM = 128;
  constexpr int kTileN = PairTileN;
  constexpr int kPairRows = 2 * kTileM;
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
  static_assert(kTileN % 16 == 0);
  static_assert(TileK % kTmaK == 0);
  static_assert(TileK % kMmaK == 0);

  constexpr int kKChunks = TileK / kTmaK;
  constexpr int kAChunkBytes = kTileM * kTmaK * sizeof(half);
  constexpr int kBChunkBytes = kTileN * kTmaK * sizeof(half);
  constexpr int kAStageBytes = kKChunks * kAChunkBytes;
  constexpr int kBStageBytes = kKChunks * kBChunkBytes;
  constexpr int kStageBytes = 2 * kAStageBytes + kBStageBytes;

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
    ptx::tmem_alloc(ptx::smem_address(&tmem_base), 2 * kTileN);
  }
  __syncthreads();

  constexpr uint32_t instruction_descriptor =
      (1U << 4U) |
      (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
      (static_cast<uint32_t>(kTileM) >> 4U << 24U);

  int tma_phase[Stages] = {};
  int mma_phase[Stages] = {};
  const int k_tiles = k / TileK;
  const int total_tiles = tile_pairs_m * tiles_n;
  int static_work_id = static_cast<int>(blockIdx.x);

  auto fetch_work = [&]() {
    if (warp == 0 && ptx::elect_one()) {
      shared_work_id = static_work_id;
      static_work_id += static_cast<int>(gridDim.x);
    }
    __syncthreads();
    return shared_work_id;
  };

  auto issue_load = [&](int k_tile, int pair_m, int tile_n,
                        int pair_count) {
    if (warp != 1 || !ptx::elect_one()) return;

    const int stage = k_tile % Stages;
    const uint32_t barrier =
        tma_barrier_base + stage * sizeof(uint64_t);
    const uint32_t stage_smem = smem + stage * kStageBytes;
    const uint32_t a0_smem = stage_smem;
    const uint32_t a1_smem = a0_smem + kAStageBytes;
    const uint32_t b_smem = a1_smem + kAStageBytes;
    const int offset_k = k_tile * TileK;
    const int offset_m = pair_m * kPairRows;
    const int offset_n = tile_n * kTileN;

#pragma unroll
    for (int chunk = 0; chunk < kKChunks; ++chunk) {
      const int chunk_k = offset_k + chunk * kTmaK;
      ptx::tma_load_2d(a0_smem + chunk * kAChunkBytes, &tensor_map_a,
                       chunk_k, offset_m, barrier);
      if (pair_count == 2) {
        ptx::tma_load_2d(a1_smem + chunk * kAChunkBytes, &tensor_map_a,
                         chunk_k, offset_m + kTileM, barrier);
      }
      ptx::tma_load_2d(b_smem + chunk * kBChunkBytes, &tensor_map_b_nk,
                       chunk_k, offset_n, barrier);
    }
    ptx::mbarrier_arrive_expect_tx(
        barrier, pair_count * kAStageBytes + kBStageBytes);
  };

  auto issue_mma = [&](int k_tile, int pair_count) {
    const int stage = k_tile % Stages;
    const uint32_t tma_barrier_address =
        tma_barrier_base + stage * sizeof(uint64_t);
    ptx::mbarrier_wait(tma_barrier_address, tma_phase[stage]);
    tma_phase[stage] ^= 1;
    ptx::tcgen05_fence_after_thread_sync();

    if (warp != 0 || !ptx::elect_one()) return;

    const uint32_t stage_smem = smem + stage * kStageBytes;
    const uint32_t a0_smem = stage_smem;
    const uint32_t a1_smem = a0_smem + kAStageBytes;
    const uint32_t b_smem = a1_smem + kAStageBytes;

#pragma unroll
    for (int k_block = 0; k_block < TileK / kMmaK; ++k_block) {
      const int chunk = k_block / (kTmaK / kMmaK);
      const int block_in_chunk = k_block % (kTmaK / kMmaK);
      const uint32_t a0_block =
          a0_smem + chunk * kAChunkBytes + block_in_chunk * 32;
      const uint32_t b_block =
          b_smem + chunk * kBChunkBytes + block_in_chunk * 32;
      const uint64_t descriptor_a0 =
          ptx::sw128_k_major_descriptor(a0_block);
      const uint64_t descriptor_b =
          ptx::sw128_k_major_descriptor(b_block);
      const bool accumulate = k_tile != 0 || k_block != 0;
      ptx::mma_f16(tmem_base, descriptor_a0, descriptor_b,
                   instruction_descriptor, accumulate);
      if (pair_count == 2) {
        const uint32_t a1_block =
            a1_smem + chunk * kAChunkBytes + block_in_chunk * 32;
        const uint64_t descriptor_a1 =
            ptx::sw128_k_major_descriptor(a1_block);
        ptx::mma_f16(tmem_base + kTileN, descriptor_a1, descriptor_b,
                     instruction_descriptor, accumulate);
      }
    }
    ptx::mma_commit(mma_barrier_base + stage * sizeof(uint64_t));
  };

  auto store_tile = [&](int pair_m, int tile_n, int pair_count) {
    const int offset_m = pair_m * kPairRows;
    const int offset_n = tile_n * kTileN;
    if (tid >= kTileM) return;

    for (int pair = 0; pair < pair_count; ++pair) {
      const uint32_t pair_tmem = tmem_base + pair * kTileN;
      const int pair_offset_m = offset_m + pair * kTileM;
      for (int n_block = 0; n_block < kTileN / 8; ++n_block) {
        float values[8];
        const uint32_t address =
            pair_tmem + ((warp * 32) << 16) + n_block * 8;
        ptx::tmem_load_32x32b_x8(address, values);
        if constexpr (StoreTransposed) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            ptx::store_global_l1_no_allocate_f32(
                output +
                    static_cast<size_t>(offset_n + n_block * 8 + i) * m +
                    pair_offset_m + tid,
                values[i]);
          }
        } else {
          float* dst = output +
                       static_cast<size_t>(pair_offset_m + tid) * n +
                       offset_n + n_block * 8;
          ptx::store_global_l1_no_allocate_v8_f32(dst, values);
        }
      }
    }
  };

  while (true) {
    const int work_id = fetch_work();
    if (work_id >= total_tiles) break;

    const int pair_m = work_id / tiles_n;
    const int tile_n = work_id - pair_m * tiles_n;
    const int pair_count =
        (pair_m + 1) * kPairRows <= m ? 2 : 1;

    const int prologue = k_tiles < Stages ? k_tiles : Stages;
    for (int k_tile = 0; k_tile < prologue; ++k_tile) {
      issue_load(k_tile, pair_m, tile_n, pair_count);
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
      const int stage = k_tile % Stages;
      issue_mma(k_tile, pair_count);

      const int reuse_tile = k_tile + Stages;
      if (reuse_tile < k_tiles) {
        const uint32_t mma_barrier_address =
            mma_barrier_base + stage * sizeof(uint64_t);
        ptx::mbarrier_wait(mma_barrier_address, mma_phase[stage]);
        mma_phase[stage] ^= 1;
        issue_load(reuse_tile, pair_m, tile_n, pair_count);
      }
    }

    const int final_stages = k_tiles < Stages ? k_tiles : Stages;
    for (int stage = 0; stage < final_stages; ++stage) {
      ptx::mbarrier_wait(mma_barrier_base + stage * sizeof(uint64_t),
                         mma_phase[stage]);
      mma_phase[stage] ^= 1;
    }

    ptx::tcgen05_fence_after_thread_sync();
    store_tile(pair_m, tile_n, pair_count);
    __syncthreads();
  }

  __syncthreads();
  if (warp == 0) {
    ptx::tmem_dealloc(tmem_base, 2 * kTileN);
  }
#else
  (void)tensor_map_a;
  (void)tensor_map_b_nk;
  (void)output;
  (void)m;
  (void)n;
  (void)k;
  (void)tile_pairs_m;
  (void)tiles_n;
#endif
}

// Warp-specialized persistent kernel used by tc5a and its tuning variants.
// EpilogueWarps store tile q while the dedicated TMA and MMA warps build tile
// q+1.  Two TMEM accumulator buffers make that overlap legal; Stages controls
// the independent circular SMEM pipeline feeding the MMA warp.
//
// Warp assignment:
//   [0, EpilogueWarps) : TMEM readback and global stores
//   EpilogueWarps      : elected TMA producer lane
//   EpilogueWarps + 1  : elected TCGen05 consumer lane and TMEM allocator
//
// FixedTilesN/FixedKTiles/FixedTotalTiles are compile-time experiment knobs for
// fixed benchmark shapes.  Zero means derive the value from runtime dimensions.
template <int TileM = 128, int TileN = 256, int TileK = 64,
          int Stages = 4, int EpilogueWarps = TileM / 32,
          bool StoreTransposed = false, int FixedTilesN = 0,
          int FixedKTiles = 0, int FixedTotalTiles = 0,
          bool StoreTransposedViaSmem = false>
__global__ __launch_bounds__(
    (EpilogueWarps + 2) * 32)
void tc5a_overlap_epilogue_1sm_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk, float* output,
    int m, int n, int k, int tiles_m, int tiles_n) {
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
  static_assert(kEpilogueWarps == 4 || kEpilogueWarps == 8);

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

  // tma/mma barriers own circular SMEM stages.  mainloop/epilogue barriers own
  // the two TMEM buffers: mainloop announces a completed accumulator, while
  // all epilogue warps arrive before MMA is allowed to overwrite that buffer.
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

  const int k_tiles = FixedKTiles > 0 ? FixedKTiles : k / TileK;
  const int total_tiles =
      FixedTotalTiles > 0 ? FixedTotalTiles : tiles_m * tiles_n;
  const int effective_tiles_n = FixedTilesN > 0 ? FixedTilesN : tiles_n;
  const int tiles_n_mask = effective_tiles_n - 1;
  const int tiles_n_log2 = __ffs(effective_tiles_n) - 1;
  const bool tiles_n_power2 =
      (effective_tiles_n & tiles_n_mask) == 0;

  // Division-free mappings are kept for common/fixed N-tile counts because
  // every persistent warp repeats this conversion for every output tile.
  auto tile_coordinates = [&](int work_id, int& tile_m, int& tile_n) {
    if constexpr (FixedTilesN == 3) {
      tile_m = work_id / 3;
      tile_n = work_id - tile_m * 3;
    } else {
      if (tiles_n_power2) {
        tile_m = work_id >> tiles_n_log2;
        tile_n = work_id & tiles_n_mask;
      } else {
        tile_m = work_id / effective_tiles_n;
        tile_n = work_id - tile_m * effective_tiles_n;
      }
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

  // Named barrier 1 synchronizes only the epilogue warps.  A CTA-wide barrier
  // would deadlock because the producer warps execute independent loops.
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
    const int column_base =
        kEpilogueWarps == 8 ? (warp >> 2) * (TileN / 2) : 0;
    const int column_blocks =
        kEpilogueWarps == 8 ? (TileN / 2) / 8 : TileN / 8;

    const int row =
        TileM == 64 ? (warp & 3) * 16 + lane
                    : (warp & 3) * ptx::kWarpSize + lane;
    const uint32_t tmem_row = TileM == 64 ? (warp & 3) * 32 : row;
    const uint32_t base_address =
        tmem_base + tmem_stage * TileN + (tmem_row << 16) + column_base;
    float* row_dst = output +
                     static_cast<size_t>(offset_m + row) * n +
                     offset_n + column_base;

    if constexpr (StoreTransposed && StoreTransposedViaSmem &&
                  TileM == 128 && kEpilogueWarps == 4) {
      float* transpose_smem =
          reinterpret_cast<float*>(dynamic_smem + Stages * kStageBytes);
      for (int n_block = 0; n_block < column_blocks; ++n_block) {
        float values[8];
        ptx::tmem_load_32x32b_x8(base_address + n_block * 8, values);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          transpose_smem[(column_base + n_block * 8 + i) * TileM + row] =
              values[i];
        }
      }
      epilogue_sync();

      const int epilogue_tid = warp * ptx::kWarpSize + lane;
      constexpr int kRowVectors = TileM / 8;
      constexpr int kTotalVectors = TileN * kRowVectors;
      for (int vec = epilogue_tid; vec < kTotalVectors;
           vec += kEpilogueWarps * ptx::kWarpSize) {
        const int local_n = vec / kRowVectors;
        const int row_vec = (vec - local_n * kRowVectors) * 8;
        float values[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          values[i] = transpose_smem[local_n * TileM + row_vec + i];
        }
        ptx::store_global_l1_no_allocate_v8_f32(
            output + static_cast<size_t>(offset_n + local_n) * m +
                offset_m + row_vec,
            values);
      }
      epilogue_sync();
      return;
    }

    // Double-buffer TMEM loads in registers: while one x8 vector is being
    // written to GMEM, the next no-wait read can make progress.
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
        if constexpr (StoreTransposed) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            if constexpr (TileM == 64) {
              if (lane < 16) {
                ptx::store_global_l1_no_allocate_f32(
                    output +
                        static_cast<size_t>(offset_n + column_base +
                                            n_block * 8 + i) *
                            m +
                        offset_m + row,
                    values_even[i]);
              }
            } else {
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
          if constexpr (TileM == 64) {
            if (lane < 16) {
              ptx::store_global_l1_no_allocate_v8_f32(dst, values_even);
            }
          } else {
            ptx::store_global_l1_no_allocate_v8_f32(dst, values_even);
          }
        }
      } else {
        if constexpr (StoreTransposed) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            if constexpr (TileM == 64) {
              if (lane < 16) {
                ptx::store_global_l1_no_allocate_f32(
                    output +
                        static_cast<size_t>(offset_n + column_base +
                                            n_block * 8 + i) *
                            m +
                        offset_m + row,
                    values_odd[i]);
              }
            } else {
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
          if constexpr (TileM == 64) {
            if (lane < 16) {
              ptx::store_global_l1_no_allocate_v8_f32(dst, values_odd);
            }
          } else {
            ptx::store_global_l1_no_allocate_v8_f32(dst, values_odd);
          }
        }
      }
    }
  };

  // The three branches below are long-lived warp roles.  Their independent
  // loops communicate only through the barrier generations described above.
  if (warp == kTmaWarp && ptx::elect_one()) {
    // Producer: wait until MMA releases a SMEM stage, then refill it by TMA.
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
    // Consumer: wait for TMA, accumulate one output tile, then publish the
    // completed TMEM buffer.  The epilogue wait prevents premature overwrite.
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
    // Epilogue: wait once per TMEM buffer, cooperatively store the tile, and
    // return one arrival per warp so the MMA role can recycle the buffer.
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
#endif
}

// M-pair warp-specialized kernel for transposed output.  It computes two
// M128xTileN tiles that share B, places them side by side in each TMEM buffer,
// and overlaps their combined epilogue with the next pair's mainloop.  Unlike
// the non-overlapped pair-M kernel, this variant requires complete M pairs.
template <int TileN = 64, int TileK = 64, int Stages = 2,
          int EpilogueWarps = 4>
__global__ __launch_bounds__(
    (EpilogueWarps + 2) * 32)
void tc5_pair_m_overlap_transposed_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b_nk, float* output,
    int m, int n, int k, int tile_pairs_m, int tiles_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kTileM = 128;
  constexpr int kPairRows = 2 * kTileM;
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
  constexpr int kEpilogueWarps = EpilogueWarps;
  constexpr int kTmaWarp = kEpilogueWarps;
  constexpr int kMmaWarp = kEpilogueWarps + 1;
  constexpr int kTmemColumnsPerBuffer = 2 * TileN;
  static_assert(TileK % kTmaK == 0);
  static_assert(TileK % kMmaK == 0);
  static_assert(TileN % 16 == 0);
  static_assert(kEpilogueWarps == 4);

  constexpr int kKChunks = TileK / kTmaK;
  constexpr int kAChunkBytes = kTileM * kTmaK * sizeof(half);
  constexpr int kBChunkBytes = TileN * kTmaK * sizeof(half);
  constexpr int kAStageBytes = kKChunks * kAChunkBytes;
  constexpr int kBStageBytes = kKChunks * kBChunkBytes;
  constexpr int kStageBytes = 2 * kAStageBytes + kBStageBytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / ptx::kWarpSize;
  const int lane = tid % ptx::kWarpSize;

  extern __shared__ __align__(1024) char dynamic_smem[];
  const uint32_t smem = ptx::smem_address(dynamic_smem);

  // The barrier topology mirrors tc5a_overlap_epilogue_1sm_kernel; only the
  // per-stage payload (A0 + A1 + B) and TMEM ownership (two M tiles) differ.
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
      (static_cast<uint32_t>(kTileM) >> 4U << 24U);

  const int k_tiles = k / TileK;
  const int total_tiles = tile_pairs_m * tiles_n;

  auto tile_coordinates = [&](int work_id, int& pair_m, int& tile_n) {
    pair_m = work_id / tiles_n;
    tile_n = work_id - pair_m * tiles_n;
  };

  auto issue_load = [&](int k_tile, int pair_m, int tile_n,
                        int tma_stage) {
    const uint32_t barrier =
        tma_barrier_base + tma_stage * sizeof(uint64_t);
    const uint32_t stage_smem = smem + tma_stage * kStageBytes;
    const uint32_t a0_smem = stage_smem;
    const uint32_t a1_smem = a0_smem + kAStageBytes;
    const uint32_t b_smem = a1_smem + kAStageBytes;
    const int offset_k = k_tile * TileK;
    const int offset_m = pair_m * kPairRows;
    const int offset_n = tile_n * TileN;

#pragma unroll
    for (int chunk = 0; chunk < kKChunks; ++chunk) {
      const int chunk_k = offset_k + chunk * kTmaK;
      ptx::tma_load_2d(a0_smem + chunk * kAChunkBytes, &tensor_map_a,
                       chunk_k, offset_m, barrier);
      ptx::tma_load_2d(a1_smem + chunk * kAChunkBytes, &tensor_map_a,
                       chunk_k, offset_m + kTileM, barrier);
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
    const uint32_t accumulator0 =
        tmem_base + tmem_stage * kTmemColumnsPerBuffer;
    const uint32_t accumulator1 = accumulator0 + TileN;

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

  auto store_tile = [&](int pair_m, int tile_n, int tmem_stage) {
    if (warp >= kEpilogueWarps) return;
    const int row = warp * ptx::kWarpSize + lane;
    const int offset_m = pair_m * kPairRows;
    const int offset_n = tile_n * TileN;

#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      const uint32_t base_address =
          tmem_base + tmem_stage * kTmemColumnsPerBuffer + pair * TileN +
          (row << 16);
      const int row_out = offset_m + pair * kTileM + row;
      float values_even[8];
      float values_odd[8];
      ptx::tmem_load_32x32b_x8_no_wait(base_address, values_even);
#pragma unroll
      for (int n_block = 0; n_block < TileN / 8; ++n_block) {
        ptx::tmem_load_wait();
        const bool use_even = (n_block & 1) == 0;
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
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            ptx::store_global_l1_no_allocate_f32(
                output +
                    static_cast<size_t>(offset_n + n_block * 8 + i) * m +
                    row_out,
                values_even[i]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            ptx::store_global_l1_no_allocate_f32(
                output +
                    static_cast<size_t>(offset_n + n_block * 8 + i) * m +
                    row_out,
                values_odd[i]);
          }
        }
      }
    }
  };

  if (warp == kTmaWarp && ptx::elect_one()) {
    int tma_stage = 0;
    int mma_phase = 1;
    for (int work_id = static_cast<int>(blockIdx.x);
         work_id < total_tiles; work_id += static_cast<int>(gridDim.x)) {
      int pair_m = 0;
      int tile_n = 0;
      tile_coordinates(work_id, pair_m, tile_n);
      for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        ptx::mbarrier_wait(mma_barrier_base + tma_stage * sizeof(uint64_t),
                           mma_phase);
        issue_load(k_tile, pair_m, tile_n, tma_stage);
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

      int pair_m = 0;
      int tile_n = 0;
      tile_coordinates(work_id, pair_m, tile_n);
      store_tile(pair_m, tile_n, tmem_stage);

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
  (void)tile_pairs_m;
  (void)tiles_n;
#endif
}

// ---------------------------------------------------------------------------
// Host launch wrappers
// ---------------------------------------------------------------------------
// Constructors validate the shape contract, encode persistent CUtensorMap
// descriptors, and opt the selected specialization into its dynamic-SMEM size.
// launch() only enqueues work and checks launch errors; it does not synchronize
// the device.  Most aligned-only tuning runners honor TC5H_WORKERS to cap the
// number of resident CTAs for controlled scheduling experiments.

// General row-major wrapper.  It rounds arbitrary M/N/K down to a legal tiled
// interior, launches the raw persistent kernel there, and uses the scalar
// cleanup kernel for all excluded rows, columns, and K suffix values.
template <int RunnerTileN = 256, int RunnerTileK = 128,
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
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 raw 1sm)");

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
    const int total_tiles = tiles_m * tiles_n;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5 raw 1sm)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 raw 1sm)");
    int active_blocks_per_sm = 1;
    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages>;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks_per_sm, kernel, 128, smem_bytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(tc5 raw 1sm)");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    const int worker_ctas = std::min(
        total_tiles,
        std::max(1, properties.multiProcessorCount * active_blocks_per_sm));

    tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages>
        <<<worker_ctas, 128, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, fast_k_,
            tiles_m, tiles_n);
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
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  int fast_m_ = 0;
  int fast_n_ = 0;
  int fast_k_ = 0;
  bool has_fast_path_ = false;
};

// Aligned row-major wrapper with a caller-provided leading dimension for D.
// Padding between rows is preserved; unlike Tc5Runner, no boundary cleanup is
// available, so all three problem dimensions must satisfy the tile contract.
template <int RunnerTileN = 256, int RunnerTileK = 128,
          int RunnerStages = 2>
class Tc5StridedRunner {
 public:
  Tc5StridedRunner(const half* a, const half* b_nk, float* d,
                   int m, int n, int k, int output_stride)
      : output_(d),
        m_(m),
        n_(n),
        k_(k),
        output_stride_(output_stride) {
    if (m_ % kTileM != 0 || n_ % kTileN != 0 ||
        k_ % kTileK != 0 || n_ % 4 != 0 ||
        k_ % kSw128TmaLeadingDimensionAlignment != 0 ||
        output_stride_ < n_) {
      std::fprintf(stderr,
                   "Tc5StridedRunner requires aligned M/N/K, N%%4=0, "
                   "and output_stride >= N\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, n_, k_, k_,
                                       kTileN);

    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 strided)");
  }

  void launch() {
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5 strided)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 strided)");
    int active_blocks_per_sm = 1;
    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages>;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks_per_sm, kernel, 128, smem_bytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(tc5 "
               "strided)");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    int worker_ctas = std::min(
        total_tiles,
        std::max(1, properties.multiProcessorCount * active_blocks_per_sm));
    if (const char* override = std::getenv("TC5H_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas = std::min(total_tiles, requested);
      }
    }

    tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages>
        <<<worker_ctas, 128, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, k_,
            tiles_m, tiles_n, 1, output_stride_);
    check_cuda(cudaGetLastError(), "tc5_raw_persistent_1sm_strided launch");
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

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b_{};
  float* output_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  int output_stride_ = 0;
};

// Launches the N-pair kernel.  Pairing reuses A across two adjacent N256 tiles;
// an odd final N tile is handled by pair_count inside the kernel.
template <int RunnerTileK = 64, int RunnerStages = 2>
class Tc5TailMnPairNRunner {
 public:
  Tc5TailMnPairNRunner(const half* a, const half* b_nk, float* d,
                       int m, int n, int k)
      : output_(d), m_(m), n_(n), k_(k) {
    if (m_ % kTileM != 0 || n_ % kTileN != 0 ||
        k_ % kTileK != 0 || n_ % 4 != 0 ||
        k_ % kSw128TmaLeadingDimensionAlignment != 0) {
      std::fprintf(stderr,
                   "Tc5TailMnPairNRunner requires aligned M/N/K and "
                   "N%%4=0\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, n_, k_, k_,
                                       kTileN);

    auto* kernel = &tc5_tail_mn_pair_n_kernel<kTileK, kStages>;
    constexpr int smem_bytes =
        kStages * (kTileM + 2 * kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 tail-mn pair-n)");
  }

  void launch() {
    constexpr int smem_bytes =
        kStages * (kTileM + 2 * kTileN) * kTileK * sizeof(half);
    const int tiles_m = m_ / kTileM;
    const int tile_pairs_n = (n_ + kPairColumns - 1) / kPairColumns;
    const int total_tiles = tiles_m * tile_pairs_n;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5 tail-mn pair-n)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 tail-mn pair-n)");
    int active_blocks_per_sm = 1;
    auto* kernel = &tc5_tail_mn_pair_n_kernel<kTileK, kStages>;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks_per_sm, kernel, 128, smem_bytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(tc5 "
               "tail-mn pair-n)");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    int worker_ctas = std::min(
        total_tiles,
        std::max(1, properties.multiProcessorCount * active_blocks_per_sm));
    if (const char* override = std::getenv("TC5H_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas = std::min(total_tiles, requested);
      }
    }

    tc5_tail_mn_pair_n_kernel<kTileK, kStages>
        <<<worker_ctas, 128, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, k_,
            tiles_m, tile_pairs_n);
    check_cuda(cudaGetLastError(), "tc5_tail_mn_pair_n_kernel launch");
  }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 256;
  static constexpr int kPairColumns = 2 * kTileN;
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
  int n_ = 0;
  int k_ = 0;
};

// Launches the M-pair kernel.  Pairing reuses B across two adjacent M128 tiles;
// an odd final M tile is handled by pair_count inside the kernel.
template <int RunnerTileK = 64, int RunnerStages = 2>
class Tc5TailMnPairMRunner {
 public:
  Tc5TailMnPairMRunner(const half* a, const half* b_nk, float* d,
                       int m, int n, int k)
      : output_(d), m_(m), n_(n), k_(k) {
    if (m_ % kTileM != 0 || n_ % kTileN != 0 ||
        k_ % kTileK != 0 || n_ % 4 != 0 ||
        k_ % kSw128TmaLeadingDimensionAlignment != 0) {
      std::fprintf(stderr,
                   "Tc5TailMnPairMRunner requires aligned M/N/K and "
                   "N%%4=0\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, n_, k_, k_,
                                       kTileN);

    auto* kernel = &tc5_tail_mn_pair_m_kernel<kTileK, kStages>;
    constexpr int smem_bytes =
        kStages * (2 * kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 tail-mn pair-m)");
  }

  void launch() {
    constexpr int smem_bytes =
        kStages * (2 * kTileM + kTileN) * kTileK * sizeof(half);
    const int tile_pairs_m = (m_ + kPairRows - 1) / kPairRows;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tile_pairs_m * tiles_n;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5 tail-mn pair-m)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 tail-mn pair-m)");
    int active_blocks_per_sm = 1;
    auto* kernel = &tc5_tail_mn_pair_m_kernel<kTileK, kStages>;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks_per_sm, kernel, 128, smem_bytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(tc5 "
               "tail-mn pair-m)");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    int worker_ctas = std::min(
        total_tiles,
        std::max(1, properties.multiProcessorCount * active_blocks_per_sm));
    if (const char* override = std::getenv("TC5H_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas = std::min(total_tiles, requested);
      }
    }

    tc5_tail_mn_pair_m_kernel<kTileK, kStages>
        <<<worker_ctas, 128, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, k_,
            tile_pairs_m, tiles_n);
    check_cuda(cudaGetLastError(), "tc5_tail_mn_pair_m_kernel launch");
  }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 256;
  static constexpr int kPairRows = 2 * kTileM;
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
  int n_ = 0;
  int k_ = 0;
};

// Non-overlapped M-pair variant that writes transposed D[N,M].  This separates
// pairwise operand reuse from the cost/benefit of epilogue overlap.
template <int RunnerTileN = 64, int RunnerTileK = 64,
          int RunnerStages = 2>
class Tc5PairMTransposedStoreRunner {
 public:
  Tc5PairMTransposedStoreRunner(const half* a, const half* b_nk, float* d,
                                int m, int n, int k)
      : output_(d), m_(m), n_(n), k_(k) {
    if (m_ % kTileM != 0 || n_ % kTileN != 0 ||
        k_ % kTileK != 0 || n_ % 4 != 0 ||
        k_ % kSw128TmaLeadingDimensionAlignment != 0) {
      std::fprintf(stderr,
                   "Tc5PairMTransposedStoreRunner requires aligned M/N/K "
                   "and N%%4=0\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, n_, k_, k_,
                                       kTileN);

    auto* kernel =
        &tc5_tail_mn_pair_m_kernel<kTileK, kStages, kTileN, true>;
    constexpr int smem_bytes =
        kStages * (2 * kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 pair-m transposed store)");
  }

  void launch() {
    constexpr int smem_bytes =
        kStages * (2 * kTileM + kTileN) * kTileK * sizeof(half);
    const int tile_pairs_m = (m_ + kPairRows - 1) / kPairRows;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tile_pairs_m * tiles_n;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device),
               "cudaGetDevice(tc5 pair-m transposed)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 pair-m transposed)");
    int active_blocks_per_sm = 1;
    auto* kernel =
        &tc5_tail_mn_pair_m_kernel<kTileK, kStages, kTileN, true>;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks_per_sm, kernel, 128, smem_bytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(tc5 pair-m "
               "transposed)");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    int worker_ctas = std::min(
        total_tiles,
        std::max(1, properties.multiProcessorCount * active_blocks_per_sm));
    if (const char* override = std::getenv("TC5H_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas = std::min(total_tiles, requested);
      }
    }

    tc5_tail_mn_pair_m_kernel<kTileK, kStages, kTileN, true>
        <<<worker_ctas, 128, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, k_,
            tile_pairs_m, tiles_n);
    check_cuda(cudaGetLastError(),
               "tc5_pair_m_transposed_store_kernel launch");
  }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = RunnerTileN;
  static constexpr int kPairRows = 2 * kTileM;
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
  int n_ = 0;
  int k_ = 0;
};

// Warp-specialized M-pair transposed-store variant.  Complete M256 groups are
// mandatory because the kernel always overlaps a two-tile TMEM epilogue.
template <int RunnerTileN = 64, int RunnerTileK = 64,
          int RunnerStages = 2, int RunnerEpilogueWarps = 4>
class Tc5PairMOverlapTransposedStoreRunner {
 public:
  Tc5PairMOverlapTransposedStoreRunner(const half* a, const half* b_nk,
                                       float* d, int m, int n, int k)
      : output_(d), m_(m), n_(n), k_(k) {
    if (m_ % kPairRows != 0 || n_ % kTileN != 0 ||
        k_ % kTileK != 0 || n_ % 4 != 0 ||
        k_ % kSw128TmaLeadingDimensionAlignment != 0) {
      std::fprintf(stderr,
                   "Tc5PairMOverlapTransposedStoreRunner requires "
                   "M%%256=0, aligned N/K, and N%%4=0\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, n_, k_, k_,
                                       kTileN);

    auto* kernel =
        &tc5_pair_m_overlap_transposed_kernel<kTileN, kTileK, kStages,
                                              kEpilogueWarps>;
    constexpr int smem_bytes =
        kStages * (2 * kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 pair-m overlap transposed)");
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                   cudaSharedmemCarveoutMaxShared),
               "cudaFuncSetAttribute(tc5 pair-m overlap transposed "
               "carveout)");
  }

  void launch() {
    constexpr int smem_bytes =
        kStages * (2 * kTileM + kTileN) * kTileK * sizeof(half);
    const int tile_pairs_m = m_ / kPairRows;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tile_pairs_m * tiles_n;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device),
               "cudaGetDevice(tc5 pair-m overlap transposed)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 pair-m overlap transposed)");
    int active_blocks_per_sm = 1;
    auto* kernel =
        &tc5_pair_m_overlap_transposed_kernel<kTileN, kTileK, kStages,
                                              kEpilogueWarps>;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks_per_sm, kernel, kThreads, smem_bytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(tc5 pair-m "
               "overlap transposed)");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    int worker_ctas = std::min(
        total_tiles,
        std::max(1, properties.multiProcessorCount * active_blocks_per_sm));
    if (const char* override = std::getenv("TC5H_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas = std::min(total_tiles, requested);
      }
    }

    tc5_pair_m_overlap_transposed_kernel<kTileN, kTileK, kStages,
                                         kEpilogueWarps>
        <<<worker_ctas, kThreads, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, k_,
            tile_pairs_m, tiles_n);
    check_cuda(cudaGetLastError(),
               "tc5_pair_m_overlap_transposed_store_kernel launch");
  }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = RunnerTileN;
  static constexpr int kPairRows = 2 * kTileM;
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

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b_{};
  float* output_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
};

// N=192 tail kernel with a 128+64 decomposition.  TCGen05 descriptors encode
// N=128 and N=64 separately, but both operations accumulate into adjacent
// ranges of a 256-column TMEM buffer and appear as one M128xN192 output tile.
//
// ClusterMOrder changes only the static work-id traversal order; this remains
// a 1-SM MMA kernel even when a host runner launches CTAs in clusters of two.
// SkipEpilogueWait is an experimental no-wait switch: it removes explicit TMEM
// reuse back-pressure, so its numerical correctness must be validated for the
// selected schedule instead of being inferred from the default path.
template <int TileK = 64, int Stages = 4, bool ClusterMOrder = false,
          bool SkipEpilogueWait = false>
__global__ __launch_bounds__(192)
void tc5_tail_mn_n192_overlap_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b128_nk,
    const __grid_constant__ CUtensorMap tensor_map_b64_nk, float* output,
    int m, int n, int k, int tiles_m, int tiles_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr int kTileM = 128;
  constexpr int kTileN = 192;
  constexpr int kTileN0 = 128;
  constexpr int kTileN1 = 64;
  constexpr int kMmaK = 16;
  constexpr int kTmaK = 64;
  constexpr int kEpilogueWarps = 4;
  constexpr int kTmaWarp = 4;
  constexpr int kMmaWarp = 5;
  constexpr int kTmemColumnsPerBuffer = 256;
  static_assert(TileK % kTmaK == 0);
  static_assert(TileK % kMmaK == 0);

  constexpr int kKChunks = TileK / kTmaK;
  constexpr int kAChunkBytes = kTileM * kTmaK * sizeof(half);
  constexpr int kB0ChunkBytes = kTileN0 * kTmaK * sizeof(half);
  constexpr int kB1ChunkBytes = kTileN1 * kTmaK * sizeof(half);
  constexpr int kAStageBytes = kKChunks * kAChunkBytes;
  constexpr int kB0StageBytes = kKChunks * kB0ChunkBytes;
  constexpr int kB1StageBytes = kKChunks * kB1ChunkBytes;
  constexpr int kStageBytes =
      kAStageBytes + kB0StageBytes + kB1StageBytes;

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
      if constexpr (!SkipEpilogueWait) {
        ptx::mbarrier_init(epilogue_barrier_base +
                               stage * sizeof(uint64_t),
                           kEpilogueWarps);
      }
    }
    ptx::fence_mbarrier_init_release_cluster();
  }
  if (warp == kMmaWarp) {
    ptx::tmem_alloc(ptx::smem_address(&tmem_base),
                    kTmemColumnsPerBuffer * 2);
  }
  __syncthreads();

  constexpr uint32_t instruction_descriptor_n128 =
      (1U << 4U) |
      (static_cast<uint32_t>(kTileN0) >> 3U << 17U) |
      (static_cast<uint32_t>(kTileM) >> 4U << 24U);
  constexpr uint32_t instruction_descriptor_n64 =
      (1U << 4U) |
      (static_cast<uint32_t>(kTileN1) >> 3U << 17U) |
      (static_cast<uint32_t>(kTileM) >> 4U << 24U);

  const int k_tiles = k / TileK;
  const int total_tiles = tiles_m * tiles_n;
  const int worker_ctas = static_cast<int>(gridDim.x);

  // The default mapping specializes the observed four-N-tile shape.  Cluster
  // M order groups adjacent M tiles when CTAs are launched as x=2 clusters.
  auto tile_coordinates = [&](int work_id, int& tile_m, int& tile_n) {
    if constexpr (ClusterMOrder) {
      tile_n = work_id / tiles_m;
      tile_m = work_id - tile_n * tiles_m;
    } else {
      tile_m = work_id >> 2;
      tile_n = work_id & 3;
    }
  };

  // A is shared by the N=128 body and N=64 tail; B uses two tensor maps so each
  // TMA transfer retains a native, aligned box shape.
  auto issue_load = [&](int k_tile, int tile_m, int tile_n,
                        int tma_stage) {
    const uint32_t barrier =
        tma_barrier_base + tma_stage * sizeof(uint64_t);
    const uint32_t stage_smem = smem + tma_stage * kStageBytes;
    const uint32_t a_smem = stage_smem;
    const uint32_t b0_smem = stage_smem + kAStageBytes;
    const uint32_t b1_smem = b0_smem + kB0StageBytes;
    const int offset_k = k_tile * TileK;
    const int offset_m = tile_m * kTileM;
    const int offset_n = tile_n * kTileN;

#pragma unroll
    for (int chunk = 0; chunk < kKChunks; ++chunk) {
      const int chunk_k = offset_k + chunk * kTmaK;
      ptx::tma_load_2d(a_smem + chunk * kAChunkBytes, &tensor_map_a,
                       chunk_k, offset_m, barrier);
      ptx::tma_load_2d(b0_smem + chunk * kB0ChunkBytes,
                       &tensor_map_b128_nk, chunk_k, offset_n, barrier);
      ptx::tma_load_2d(b1_smem + chunk * kB1ChunkBytes,
                       &tensor_map_b64_nk, chunk_k, offset_n + kTileN0,
                       barrier);
    }
    ptx::mbarrier_arrive_expect_tx(
        barrier, kAStageBytes + kB0StageBytes + kB1StageBytes);
  };

  // Emit two MMAs per K block.  The second accumulator starts at TMEM column
  // 128, immediately following the body accumulator.
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
      ptx::mma_f16(accumulator, descriptor_a, descriptor_b0,
                   instruction_descriptor_n128, accumulate);
      ptx::mma_f16(accumulator_tail, descriptor_a, descriptor_b1,
                   instruction_descriptor_n64, accumulate);
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
    if (warp >= kEpilogueWarps) return;
    const int offset_m = tile_m * kTileM;
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
    for (int work_id = static_cast<int>(blockIdx.x);
         work_id < total_tiles; work_id += worker_ctas) {
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
  } else if (warp == kMmaWarp) {
    int tma_stage = 0;
    int tma_phase = 0;
    int tmem_stage = 0;
    int epilogue_phase = 1;
    for (int work_id = static_cast<int>(blockIdx.x);
         work_id < total_tiles; work_id += worker_ctas) {
      if constexpr (!SkipEpilogueWait) {
      if (ptx::elect_one()) {
        ptx::mbarrier_wait(epilogue_barrier_base +
                               tmem_stage * sizeof(uint64_t),
                           epilogue_phase);
      }
      }
      for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        ptx::mbarrier_wait(tma_barrier_base +
                               tma_stage * sizeof(uint64_t),
                           tma_phase);
        ptx::tcgen05_fence_after_thread_sync();
        if (ptx::elect_one()) {
          issue_mma(k_tile, tma_stage, tmem_stage);
          ptx::mma_commit(mma_barrier_base +
                          tma_stage * sizeof(uint64_t));
        }
        tma_stage = (tma_stage + 1) % Stages;
        if (tma_stage == 0) tma_phase ^= 1;
      }
      if (ptx::elect_one()) {
        ptx::mma_commit(mainloop_barrier_base +
                        tmem_stage * sizeof(uint64_t));
      }
      tmem_stage ^= 1;
      if (tmem_stage == 0) epilogue_phase ^= 1;
    }
  } else if (warp < kEpilogueWarps) {
    int tmem_stage = 0;
    int mainloop_phase = 0;
    for (int work_id = static_cast<int>(blockIdx.x);
         work_id < total_tiles; work_id += worker_ctas) {
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

      if constexpr (!SkipEpilogueWait) {
      if (ptx::elect_one()) {
        ptx::mbarrier_arrive(epilogue_barrier_base +
                             tmem_stage * sizeof(uint64_t));
      }
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

// N=192 aligned wrapper.  The worker heuristic intentionally uses at most one
// initial CTA per SM and may reduce the count for two- or three-wave workloads,
// trading fewer resident workers for a longer persistent sequence per worker.
template <int RunnerTileK = 64, int RunnerStages = 4,
          bool SkipEpilogueWait = false>
class Tc5TailMnN192Runner {
 public:
  Tc5TailMnN192Runner(const half* a, const half* b_nk, float* d,
                      int m, int n, int k)
      : output_(d), m_(m), n_(n), k_(k) {
    if (m_ % kTileM != 0 || n_ % kTileN != 0 ||
        k_ % kTileK != 0 || n_ % 4 != 0 ||
        k_ % kSw128TmaLeadingDimensionAlignment != 0) {
      std::fprintf(stderr,
                   "Tc5TailMnN192Runner requires M%%128=0, N%%192=0, "
                   "K%%TileK=0, N%%4=0\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b128_, b_nk, n_, k_,
                                       k_, kTileN0);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b64_, b_nk, n_, k_,
                                       k_, kTileN1);

    auto* kernel =
        &tc5_tail_mn_n192_overlap_kernel<kTileK, kStages, false,
                                         kSkipEpilogueWait>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 tail-mn N192 overlap)");
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                   cudaSharedmemCarveoutMaxShared),
               "cudaFuncSetAttribute(tc5 tail-mn N192 carveout)");

    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5 tail-mn N192)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 tail-mn N192)");
    const int sm_count = std::max(1, properties.multiProcessorCount);
    worker_ctas_ = std::min(total_tiles, sm_count);
    if (total_tiles > sm_count && total_tiles <= 3 * sm_count) {
      worker_ctas_ = (total_tiles + 1) / 2;
    }
    if (const char* override = std::getenv("TC5H_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas_ = std::min(total_tiles, requested);
      }
    }
  }

  void launch() {
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    tc5_tail_mn_n192_overlap_kernel<kTileK, kStages, false,
                                    kSkipEpilogueWait>
        <<<worker_ctas_, kThreads, smem_bytes>>>(
            tensor_map_a_, tensor_map_b128_, tensor_map_b64_, output_,
            m_, n_, k_, tiles_m, tiles_n);
    check_cuda(cudaGetLastError(), "tc5_tail_mn_n192_overlap launch");
  }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 192;
  static constexpr int kTileN0 = 128;
  static constexpr int kTileN1 = 64;
  static constexpr int kTileK = RunnerTileK;
  static constexpr int kStages = RunnerStages;
  static constexpr bool kSkipEpilogueWait = SkipEpilogueWait;
  static constexpr int kThreads = 192;

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
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  int worker_ctas_ = 1;
};

// Same 1-SM N=192 computation, launched in x=2 CTA clusters to study launch and
// traversal effects.  The kernel does not issue cta_group::2 MMA.  Grid size is
// rounded to a positive even number because every hardware cluster needs two
// CTAs, even when the last CTA receives no work item.
template <int RunnerTileK = 64, int RunnerStages = 4,
          bool ClusterMOrder = false, bool SkipEpilogueWait = false>
class Tc5TailMnN192ClusterLaunchRunner {
 public:
  Tc5TailMnN192ClusterLaunchRunner(const half* a, const half* b_nk,
                                   float* d, int m, int n, int k)
      : output_(d), m_(m), n_(n), k_(k) {
    if (m_ % kTileM != 0 || n_ % kTileN != 0 ||
        k_ % kTileK != 0 || n_ % 4 != 0 ||
        k_ % kSw128TmaLeadingDimensionAlignment != 0) {
      std::fprintf(stderr,
                   "Tc5TailMnN192ClusterLaunchRunner requires M%%128=0, "
                   "N%%192=0, K%%TileK=0, N%%4=0\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b128_, b_nk, n_, k_,
                                       k_, kTileN0);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b64_, b_nk, n_, k_,
                                       k_, kTileN1);

    auto* kernel =
        &tc5_tail_mn_n192_overlap_kernel<kTileK, kStages, kClusterMOrder,
                                         kSkipEpilogueWait>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 tail-mn N192 cluster)");
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                   cudaSharedmemCarveoutMaxShared),
               "cudaFuncSetAttribute(tc5 tail-mn N192 cluster carveout)");

    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device),
               "cudaGetDevice(tc5 tail-mn N192 cluster)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 tail-mn N192 cluster)");
    const int sm_count = std::max(1, properties.multiProcessorCount);
    worker_ctas_ = std::min(total_tiles, sm_count);
    if (total_tiles > sm_count && total_tiles <= 3 * sm_count) {
      worker_ctas_ = (total_tiles + 1) / 2;
    }
    if (const char* override = std::getenv("TC5H_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas_ = std::min(total_tiles, requested);
      }
    }
    worker_ctas_ = std::max(2, worker_ctas_);
    if ((worker_ctas_ & 1) != 0) {
      ++worker_ctas_;
    }
  }

  void launch() {
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    auto* kernel =
        &tc5_tail_mn_n192_overlap_kernel<kTileK, kStages, kClusterMOrder,
                                         kSkipEpilogueWait>;

    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(worker_ctas_, 1, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = smem_bytes;
    config.attrs = attrs;
    config.numAttrs = 1;

    check_cuda(cudaLaunchKernelEx(&config, kernel, tensor_map_a_,
                                  tensor_map_b128_, tensor_map_b64_,
                                  output_, m_, n_, k_, tiles_m, tiles_n),
               "tc5_tail_mn_n192_cluster_overlap launch");
  }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 192;
  static constexpr int kTileN0 = 128;
  static constexpr int kTileN1 = 64;
  static constexpr int kTileK = RunnerTileK;
  static constexpr int kStages = RunnerStages;
  static constexpr bool kClusterMOrder = ClusterMOrder;
  static constexpr bool kSkipEpilogueWait = SkipEpilogueWait;
  static constexpr int kThreads = 192;

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
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  int worker_ctas_ = 2;
};

// Direct transposed-store wrapper around the baseline persistent kernel.
// RunnerTileM selects the M128 or M64 accumulator readback mapping.
template <int RunnerTileN = 64, int RunnerTileK = 128,
          int RunnerStages = 2, int RunnerTileM = 128>
class Tc5TransposedStoreRunner {
 public:
  Tc5TransposedStoreRunner(const half* a, const half* b_nk, float* d,
                           int m, int n, int k)
      : a_(a), b_nk_(b_nk), output_(d), m_(m), n_(n), k_(k) {
    if (m_ % kTileM != 0 || n_ % kTileN != 0 || k_ % kTileK != 0 ||
        n_ % 4 != 0 || k_ % kSw128TmaLeadingDimensionAlignment != 0) {
      std::fprintf(stderr,
                   "Tc5TransposedStoreRunner requires aligned "
                   "M/N/K and N%%4=0\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, n_, k_, k_,
                                       kTileN);

    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, true,
                                       kTileM>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 transposed store)");
  }

  void launch() {
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5 transposed store)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 transposed store)");
    int active_blocks_per_sm = 1;
    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, true,
                                       kTileM>;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks_per_sm, kernel, 128, smem_bytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(tc5 "
               "transposed store)");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    int worker_ctas = std::min(
        total_tiles,
        std::max(1, properties.multiProcessorCount * active_blocks_per_sm));
    if (const char* override = std::getenv("TC5H_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas = std::min(total_tiles, requested);
      }
    }

    tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, true, kTileM>
        <<<worker_ctas, 128, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, k_,
            tiles_m, tiles_n);
    check_cuda(cudaGetLastError(),
               "tc5_raw_persistent_1sm_transposed_store launch");
  }

 private:
  static constexpr int kTileM = RunnerTileM;
  static constexpr int kTileN = RunnerTileN;
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
  const half* a_ = nullptr;
  const half* b_nk_ = nullptr;
  float* output_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
};

// Transposed output via an explicit TileN x TileM FP32 SMEM scratch tile.  This
// consumes additional dynamic SMEM but converts scattered scalar writes into
// coalesced x8 stores, allowing the two epilogue strategies to be compared.
template <int RunnerTileN = 64, int RunnerTileK = 128,
          int RunnerStages = 2>
class Tc5TransposedSmemStoreRunner {
 public:
  Tc5TransposedSmemStoreRunner(const half* a, const half* b_nk, float* d,
                               int m, int n, int k)
      : output_(d), m_(m), n_(n), k_(k) {
    if (m_ % kTileM != 0 || n_ % kTileN != 0 || k_ % kTileK != 0 ||
        n_ % 4 != 0 || k_ % kSw128TmaLeadingDimensionAlignment != 0) {
      std::fprintf(stderr,
                   "Tc5TransposedSmemStoreRunner requires aligned M/N/K "
                   "and N%%4=0\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, n_, k_, k_,
                                       kTileN);

    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, true,
                                       kTileM, true>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half) +
        kTileM * kTileN * sizeof(float);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 transposed smem store)");
  }

  void launch() {
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half) +
        kTileM * kTileN * sizeof(float);
    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5 smem store)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 smem store)");
    int active_blocks_per_sm = 1;
    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, true,
                                       kTileM, true>;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks_per_sm, kernel, 128, smem_bytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(tc5 smem "
               "store)");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    int worker_ctas = std::min(
        total_tiles,
        std::max(1, properties.multiProcessorCount * active_blocks_per_sm));
    if (const char* override = std::getenv("TC5H_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas = std::min(total_tiles, requested);
      }
    }

    tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, true,
                                  kTileM, true>
        <<<worker_ctas, 128, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, k_,
            tiles_m, tiles_n);
    check_cuda(cudaGetLastError(),
               "tc5_raw_persistent_1sm_transposed_smem_store launch");
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

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b_{};
  float* output_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
};

// Transposed Split-K wrapper.  Each K slice writes a private MxN partial
// buffer; tc5_splitk_reduce_kernel then combines the slices in D[N,M] order.
// The owned device buffer makes this runner non-copyable.
template <int SplitK = 4, int RunnerTileN = 64, int RunnerTileK = 128,
          int RunnerStages = 2>
class Tc5TransposedStoreSplitKRunner {
 public:
  Tc5TransposedStoreSplitKRunner(const half* a, const half* b_nk, float* d,
                                 int m, int n, int k)
      : a_(a), b_nk_(b_nk), output_(d), m_(m), n_(n), k_(k) {
    if (m_ % kTileM != 0 || n_ % kTileN != 0 || k_ % SplitK != 0 ||
        (k_ / SplitK) % kTileK != 0 || n_ % 4 != 0 ||
        k_ % kSw128TmaLeadingDimensionAlignment != 0) {
      std::fprintf(stderr,
                   "Tc5TransposedStoreSplitKRunner requires aligned "
                   "M/N/K, split slices aligned to TileK, and N%%4=0\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, n_, k_, k_,
                                       kTileN);

    partial_elements_ = static_cast<size_t>(m_) * n_;
    check_cuda(cudaMalloc(&partials_, partial_elements_ * SplitK *
                                         sizeof(float)),
               "cudaMalloc(tc5 split-k partials)");

    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, true>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 transposed split-k)");
  }

  ~Tc5TransposedStoreSplitKRunner() {
    if (partials_ != nullptr) {
      cudaFree(partials_);
    }
  }

  Tc5TransposedStoreSplitKRunner(const Tc5TransposedStoreSplitKRunner&) =
      delete;
  Tc5TransposedStoreSplitKRunner& operator=(
      const Tc5TransposedStoreSplitKRunner&) = delete;

  void launch() {
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n * SplitK;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5 split-k)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 split-k)");
    int active_blocks_per_sm = 1;
    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, true>;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks_per_sm, kernel, 128, smem_bytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(tc5 "
               "split-k)");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    const int worker_ctas = std::min(
        total_tiles,
        std::max(1, properties.multiProcessorCount * active_blocks_per_sm));

    tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, true>
        <<<worker_ctas, 128, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, partials_, m_, n_, k_,
            tiles_m, tiles_n, SplitK);
    check_cuda(cudaGetLastError(),
               "tc5_raw_persistent_1sm_transposed_splitk launch");

    constexpr int kReduceThreads = 256;
    const int reduce_blocks = static_cast<int>(
        (partial_elements_ + kReduceThreads - 1) / kReduceThreads);
    tc5_splitk_reduce_kernel<SplitK><<<reduce_blocks, kReduceThreads>>>(
        partials_, output_, partial_elements_);
    check_cuda(cudaGetLastError(), "tc5_splitk_reduce_kernel launch");
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

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b_{};
  const half* a_ = nullptr;
  const half* b_nk_ = nullptr;
  float* output_ = nullptr;
  float* partials_ = nullptr;
  size_t partial_elements_ = 0;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
};

// Warp-specialized transposed-store wrapper for the tc5a overlap kernel.
// StoreTransposedViaSmem chooses scalar direct stores or the coalescing scratch
// path without changing the mainloop or worker-count experiment.
template <int RunnerTileN = 64, int RunnerTileK = 64,
          int RunnerStages = 4, int RunnerEpilogueWarps = 4,
          bool StoreTransposedViaSmem = false>
class Tc5OverlapTransposedStoreRunner {
 public:
  Tc5OverlapTransposedStoreRunner(const half* a, const half* b_nk, float* d,
                                  int m, int n, int k)
      : a_(a), b_nk_(b_nk), output_(d), m_(m), n_(n), k_(k) {
    if (m_ % kTileM != 0 || n_ % kTileN != 0 || k_ % kTileK != 0 ||
        n_ % 4 != 0 || k_ % kSw128TmaLeadingDimensionAlignment != 0) {
      std::fprintf(stderr,
                   "Tc5OverlapTransposedStoreRunner requires aligned "
                   "M/N/K and N%%4=0\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, n_, k_, k_,
                                       kTileN);

    auto* kernel =
        &tc5a_overlap_epilogue_1sm_kernel<kTileM, kTileN, kTileK,
                                          kStages, kEpilogueWarps, true,
                                          0, 0, 0, kStoreViaSmem>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half) +
        (kStoreViaSmem ? kTileM * kTileN * sizeof(float) : 0);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 overlap transposed store)");
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                   cudaSharedmemCarveoutMaxShared),
               "cudaFuncSetAttribute(tc5 overlap transposed carveout)");

    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device),
               "cudaGetDevice(tc5 overlap transposed)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 overlap transposed)");
    const int sm_count = std::max(1, properties.multiProcessorCount);
    worker_ctas_ = std::min(total_tiles, sm_count);
    if (total_tiles > sm_count && total_tiles <= 2 * sm_count) {
      worker_ctas_ = (total_tiles + 1) / 2;
    } else if (total_tiles > 2 * sm_count && total_tiles <= 3 * sm_count) {
      worker_ctas_ = (total_tiles + 1) / 2;
    }
    if (const char* override = std::getenv("TC5H_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas_ = std::min(total_tiles, requested);
      }
    }
  }

  void launch() {
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half) +
        (kStoreViaSmem ? kTileM * kTileN * sizeof(float) : 0);
    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    tc5a_overlap_epilogue_1sm_kernel<kTileM, kTileN, kTileK, kStages,
                                     kEpilogueWarps, true, 0, 0, 0,
                                     kStoreViaSmem>
        <<<worker_ctas_, kThreads, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, k_,
            tiles_m, tiles_n);
    check_cuda(cudaGetLastError(),
               "tc5a_overlap_epilogue_1sm_transposed_store launch");
  }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = RunnerTileN;
  static constexpr int kTileK = RunnerTileK;
  static constexpr int kStages = RunnerStages;
  static constexpr int kEpilogueWarps = RunnerEpilogueWarps;
  static constexpr bool kStoreViaSmem = StoreTransposedViaSmem;
  static constexpr int kThreads = (kEpilogueWarps + 2) * 32;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                 cudaGetErrorString(status));
    std::abort();
  }

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b_{};
  const half* a_ = nullptr;
  const half* b_nk_ = nullptr;
  float* output_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  int worker_ctas_ = 1;
};

// Cluster-launch version of Tc5OverlapTransposedStoreRunner.  Clustering
// changes CTA placement only; each CTA still owns an independent 1-SM
// tile and an independent TMEM allocation.
template <int RunnerTileN = 64, int RunnerTileK = 64,
          int RunnerStages = 4, int RunnerEpilogueWarps = 4>
class Tc5OverlapTransposedStoreClusterLaunchRunner {
 public:
  Tc5OverlapTransposedStoreClusterLaunchRunner(const half* a,
                                               const half* b_nk, float* d,
                                               int m, int n, int k)
      : a_(a), b_nk_(b_nk), output_(d), m_(m), n_(n), k_(k) {
    if (m_ % kTileM != 0 || n_ % kTileN != 0 || k_ % kTileK != 0 ||
        n_ % 4 != 0 || k_ % kSw128TmaLeadingDimensionAlignment != 0) {
      std::fprintf(stderr,
                   "Tc5OverlapTransposedStoreClusterLaunchRunner requires "
                   "aligned M/N/K and N%%4=0\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, n_, k_, k_,
                                       kTileN);

    auto* kernel =
        &tc5a_overlap_epilogue_1sm_kernel<kTileM, kTileN, kTileK,
                                          kStages, kEpilogueWarps, true>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 overlap transposed cluster)");
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                   cudaSharedmemCarveoutMaxShared),
               "cudaFuncSetAttribute(tc5 overlap transposed cluster "
               "carveout)");

    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device),
               "cudaGetDevice(tc5 overlap transposed cluster)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 overlap transposed cluster)");
    const int sm_count = std::max(1, properties.multiProcessorCount);
    worker_ctas_ = std::min(total_tiles, sm_count);
    if (total_tiles > sm_count && total_tiles <= 2 * sm_count) {
      worker_ctas_ = (total_tiles + 1) / 2;
    } else if (total_tiles > 2 * sm_count && total_tiles <= 3 * sm_count) {
      worker_ctas_ = (total_tiles + 1) / 2;
    }
    if (const char* override = std::getenv("TC5H_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas_ = std::min(total_tiles, requested);
      }
    }
    worker_ctas_ = std::max(2, worker_ctas_);
    if ((worker_ctas_ & 1) != 0) {
      ++worker_ctas_;
    }
  }

  void launch() {
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    auto* kernel =
        &tc5a_overlap_epilogue_1sm_kernel<kTileM, kTileN, kTileK,
                                          kStages, kEpilogueWarps, true>;

    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(worker_ctas_, 1, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = smem_bytes;
    config.attrs = attrs;
    config.numAttrs = 1;

    check_cuda(cudaLaunchKernelEx(&config, kernel, tensor_map_a_,
                                  tensor_map_b_, output_, m_, n_, k_,
                                  tiles_m, tiles_n),
               "tc5a_overlap_epilogue_1sm_transposed_cluster launch");
  }

 private:
  static constexpr int kTileM = 128;
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

  CUtensorMap tensor_map_a_{};
  CUtensorMap tensor_map_b_{};
  const half* a_ = nullptr;
  const half* b_nk_ = nullptr;
  float* output_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  int worker_ctas_ = 2;
};

// M64 specialization of the baseline row-major persistent kernel.  Only the
// first 64 logical rows participate in TMEM readback for each output tile.
template <int RunnerTileN = 64, int RunnerTileK = 128,
          int RunnerStages = 2>
class Tc5M64Runner {
 public:
  Tc5M64Runner(const half* a, const half* b_nk, float* d,
               int m, int n, int k)
      : a_(a), b_nk_(b_nk), output_(d), m_(m), n_(n), k_(k) {
    if (m_ % kTileM != 0 || n_ % kTileN != 0 || k_ % kTileK != 0 ||
        n_ % 4 != 0 || k_ % kSw128TmaLeadingDimensionAlignment != 0) {
      std::fprintf(stderr,
                   "Tc5M64Runner requires aligned M/N/K and N%%4=0\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, n_, k_, k_,
                                       kTileN);

    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, false,
                                       kTileM>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 m64)");
  }

  void launch() {
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5 m64)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 m64)");
    int active_blocks_per_sm = 1;
    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, false,
                                       kTileM>;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks_per_sm, kernel, 128, smem_bytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(tc5 m64)");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    const int worker_ctas = std::min(
        total_tiles,
        std::max(1, properties.multiProcessorCount * active_blocks_per_sm));

    tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, false, kTileM>
        <<<worker_ctas, 128, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, k_,
            tiles_m, tiles_n);
    check_cuda(cudaGetLastError(), "tc5_raw_persistent_1sm_m64 launch");
  }

 private:
  static constexpr int kTileM = 64;
  static constexpr int kTileN = RunnerTileN;
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
  const half* a_ = nullptr;
  const half* b_nk_ = nullptr;
  float* output_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
};

// Row-major Split-K counterpart of Tc5TransposedStoreSplitKRunner.  The main
// kernel writes SplitK contiguous MxN partial matrices, followed by a separate
// elementwise reduction kernel.  No atomic accumulation is used.
template <int SplitK = 4, int RunnerTileN = 256, int RunnerTileK = 128,
          int RunnerStages = 1, int RunnerTileM = 64>
class Tc5RowMajorSplitKRunner {
 public:
  Tc5RowMajorSplitKRunner(const half* a, const half* b_nk, float* d,
                          int m, int n, int k)
      : output_(d), m_(m), n_(n), k_(k) {
    if (m_ % kTileM != 0 || n_ % kTileN != 0 || k_ % SplitK != 0 ||
        (k_ / SplitK) % kTileK != 0 || n_ % 4 != 0 ||
        k_ % kSw128TmaLeadingDimensionAlignment != 0) {
      std::fprintf(stderr,
                   "Tc5RowMajorSplitKRunner requires aligned M/N/K, split "
                   "slices aligned to TileK, and N%%4=0\n");
      std::abort();
    }
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_a_, a, m_, k_, k_,
                                       kTileM);
    ptx::encode_tiled_2d_sw128_strided(&tensor_map_b_, b_nk, n_, k_, k_,
                                       kTileN);

    partial_elements_ = static_cast<size_t>(m_) * n_;
    check_cuda(cudaMalloc(&partials_, partial_elements_ * SplitK *
                                         sizeof(float)),
               "cudaMalloc(tc5 row-major split-k partials)");

    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, false,
                                       kTileM>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5 row-major split-k)");
  }

  ~Tc5RowMajorSplitKRunner() {
    if (partials_ != nullptr) {
      cudaFree(partials_);
    }
  }

  Tc5RowMajorSplitKRunner(const Tc5RowMajorSplitKRunner&) = delete;
  Tc5RowMajorSplitKRunner& operator=(const Tc5RowMajorSplitKRunner&) =
      delete;

  void launch() {
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = m_ / kTileM;
    const int tiles_n = n_ / kTileN;
    const int total_tiles = tiles_m * tiles_n * SplitK;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5 row split-k)");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(tc5 row split-k)");
    int active_blocks_per_sm = 1;
    auto* kernel =
        &tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, false,
                                       kTileM>;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks_per_sm, kernel, 128, smem_bytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(tc5 row "
               "split-k)");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    int worker_ctas = std::min(
        total_tiles,
        std::max(1, properties.multiProcessorCount * active_blocks_per_sm));
    if (const char* override = std::getenv("TC5H_WORKERS")) {
      const int requested = std::atoi(override);
      if (requested > 0) {
        worker_ctas = std::min(total_tiles, requested);
      }
    }

    tc5_raw_persistent_1sm_kernel<kTileN, kTileK, kStages, false, kTileM>
        <<<worker_ctas, 128, smem_bytes>>>(
            tensor_map_a_, tensor_map_b_, partials_, m_, n_, k_,
            tiles_m, tiles_n, SplitK);
    check_cuda(cudaGetLastError(),
               "tc5_raw_persistent_1sm_row_major_splitk launch");

    constexpr int kReduceThreads = 256;
    const int reduce_blocks = static_cast<int>(
        (partial_elements_ + kReduceThreads - 1) / kReduceThreads);
    tc5_splitk_reduce_kernel<SplitK><<<reduce_blocks, kReduceThreads>>>(
        partials_, output_, partial_elements_);
    check_cuda(cudaGetLastError(), "tc5_row_major_splitk_reduce launch");
  }

 private:
  static constexpr int kTileM = RunnerTileM;
  static constexpr int kTileN = RunnerTileN;
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
  float* partials_ = nullptr;
  size_t partial_elements_ = 0;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
};

// General row-major wrapper for the warp-specialized overlap kernel.  Like
// Tc5Runner, it accepts arbitrary shapes by combining a rounded tiled interior
// with scalar cleanup.  Fixed* parameters exist only for shape-specialized
// experiments and must agree with the runtime problem supplied by the caller.
template <int RunnerTileM = 128, int RunnerTileN = 256,
          int RunnerTileK = 64, int RunnerStages = 4,
          int RunnerEpilogueWarps = RunnerTileM / 32,
          int RunnerFixedTilesN = 0, int RunnerFixedKTiles = 0,
          int RunnerFixedTotalTiles = 0>
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
        &tc5a_overlap_epilogue_1sm_kernel<kTileM, kTileN, kTileK,
                                          kStages, kEpilogueWarps, false,
                                          kFixedTilesN, kFixedKTiles,
                                          kFixedTotalTiles>;
    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   smem_bytes),
               "cudaFuncSetAttribute(tc5a overlap epilogue)");
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                   cudaSharedmemCarveoutMaxShared),
               "cudaFuncSetAttribute(tc5a shared memory carveout)");
    if (has_fast_path_) {
      const int tiles_m = fast_m_ / kTileM;
      const int tiles_n = fast_n_ / kTileN;
      const int total_tiles =
          kFixedTotalTiles > 0 ? kFixedTotalTiles : tiles_m * tiles_n;

      int device = 0;
      cudaDeviceProp properties{};
      check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5a)");
      check_cuda(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties(tc5a)");
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

  }

  void launch(cudaStream_t stream = 0) {
    if (!has_fast_path_) {
      launch_cleanup();
      return;
    }

    constexpr int smem_bytes =
        kStages * (kTileM + kTileN) * kTileK * sizeof(half);
    const int tiles_m = fast_m_ / kTileM;
    const int tiles_n = fast_n_ / kTileN;
    tc5a_overlap_epilogue_1sm_kernel<kTileM, kTileN, kTileK, kStages,
                                     kEpilogueWarps, false, kFixedTilesN,
                                     kFixedKTiles, kFixedTotalTiles>
        <<<worker_ctas_, kThreads, smem_bytes, stream>>>(
            tensor_map_a_, tensor_map_b_, output_, m_, n_, fast_k_,
            tiles_m, tiles_n);
    check_cuda(cudaGetLastError(),
               "tc5a_overlap_epilogue_1sm_kernel launch");
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
  static constexpr int kFixedTilesN = RunnerFixedTilesN;
  static constexpr int kFixedKTiles = RunnerFixedKTiles;
  static constexpr int kFixedTotalTiles = RunnerFixedTotalTiles;
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
    check_cuda(cudaGetLastError(), "tc5a cleanup kernel launch");
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
  bool has_fast_path_ = false;
};
// Public stage-5 experiment names.  Keep these aliases explicit: the lettered
// backends are benchmark identities, while their template arguments record the
// actual MxNxK tile, SMEM stage count, and scheduling family being compared.
using Tc5aRunner = Tc5OverlapRunner<128, 256, 64, 4>;   // overlap, M128N256K64
using Tc5cRunner = Tc5Runner<>;                         // raw, M128N256K128 S2
using Tc5dRunner = Tc5Runner<128, 128, 2>;              // raw, M128N128K128 S2
using Tc5eRunner = Tc5Runner<256, 64, 2>;               // raw, M128N256K64 S2
using Tc5fRunner = Tc5Runner<128, 64, 2>;               // raw, M128N128K64 S2
using Tc5gRunner = Tc5Runner<256, 128, 1>;              // raw, M128N256K128 S1
using Tc5hRunner = Tc5Runner<256, 64, 1>;               // raw, M128N256K64 S1
using Tc5iRunner = Tc5OverlapRunner<128, 128, 64, 6>;   // overlap, N128 K64 S6
using Tc5jRunner = Tc5OverlapRunner<128, 256, 128, 2>;  // overlap, N256 K128 S2

}  // namespace gemm_sm110::backends
