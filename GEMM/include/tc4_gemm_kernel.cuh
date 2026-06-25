#pragma once

#include "gemm_common.cuh"
#include "tc3_gemm_kernel.cuh"

#include <cuda/barrier>
#include <cuda_fp16.h>

// tc4 = Blackwell mainloop rewrite.
//
// Goal:
//   Implement the full pipeline shown in docs/per.md and the pasted pipeline
//   diagram:
//
//     Scheduler / CLC
//     Mainloop Load producer warp
//     MMA consumer warp issuing SM120 narrow/block-scaled mma.sync
//     Epilogue Load warp
//     Epilogue warps consuming register accumulators and writing D
//     TMA store path
//
// This is intentionally separate from tc3. tc3 proves the minimum SM120a
// narrow-MMA instruction path. tc4 is where performance-oriented SM120
// organization belongs.

namespace tc4_cde = cuda::device::experimental;

enum class Tc4WarpRole : int {
  kMma = 0,
  kScheduler = 1,
  kMainloopLoad = 2,
  kEpilogueLoad = 3,
  kEpilogue0 = 4,
  kEpilogue1 = 5,
  kEpilogue2 = 6,
  kEpilogue3 = 7,
};

struct Tc4BlackwellShape {
  static constexpr int kBlockM = 128;
  static constexpr int kBlockN = 64;
  static constexpr int kBlockK = 32;
  static constexpr int kStages = 3;
  static constexpr int kWarps = 8;

  static constexpr int kMainloopWarp = 2;
  static constexpr int kMmaWarp = 0;
  static constexpr int kSchedulerWarp = 1;
  static constexpr int kEpilogueLoadWarp = 3;
  static constexpr int kEpilogueWarpBegin = 4;
  static constexpr int kEpilogueWarpEnd = 8;

  // SM120 mma.sync accumulators live in registers.
  static constexpr int kAccumulatorElementsPerMmaWarp = 128;
};

template <typename Shape = Tc4BlackwellShape>
constexpr size_t tc4_pipeline_smem_bytes() {
  const size_t mainloop_smem =
      static_cast<size_t>(Shape::kStages) * Shape::kBlockK *
      (Shape::kBlockM + Shape::kBlockN) * sizeof(half);
  const size_t epilogue_smem =
      static_cast<size_t>(Shape::kBlockM) * Shape::kBlockN * sizeof(float);
  return mainloop_smem + 2 * epilogue_smem;
}

using tc4_block_barrier = cuda::barrier<cuda::thread_scope_block>;
struct alignas(tc4_block_barrier) tc4_barrier_storage {
  unsigned char bytes[sizeof(tc4_block_barrier)];
};

struct Tc4WorkTile {
  int block_m;
  int block_n;
};

template <typename Shape>
struct Tc4PipelineState {
  tc4_barrier_storage mainloop_full[Shape::kStages];
  tc4_barrier_storage mainloop_empty[Shape::kStages];
  tc4_barrier_storage load_order_bar;
  tc4_barrier_storage accumulator_bar;
  tc4_barrier_storage epilogue_load_bar;
  tc4_barrier_storage epilogue_store_bar;

  int next_work_tile;
};

__device__ __forceinline__ Tc4WarpRole tc4_warp_role() {
  const int warp_id = threadIdx.x / kWarpSize;
  if (warp_id == 0) return Tc4WarpRole::kMma;
  if (warp_id == 1) return Tc4WarpRole::kScheduler;
  if (warp_id == 2) return Tc4WarpRole::kMainloopLoad;
  if (warp_id == 3) return Tc4WarpRole::kEpilogueLoad;
  return static_cast<Tc4WarpRole>(warp_id);
}

template <typename Shape>
__device__ __forceinline__ Tc4WorkTile tc4_scheduler_fetch_next_work(
    Tc4PipelineState<Shape>* state, int total_tiles_n, int total_tiles) {
  // Static persistent worker baseline:
  //   work = blockIdx.x + iter * gridDim.x
  //
  // Future CLC version:
  //   scheduler warp issues CLC query / try_cancel / fetch_next_work and
  //   multicasts the tile coordinate to all role warps.
  const int tile_id = state->next_work_tile;
  state->next_work_tile += static_cast<int>(gridDim.x);

  Tc4WorkTile tile{};
  if (tile_id >= total_tiles) {
    tile.block_m = -1;
    tile.block_n = -1;
    return tile;
  }
  tile.block_m = (tile_id / total_tiles_n) * Shape::kBlockM;
  tile.block_n = (tile_id % total_tiles_n) * Shape::kBlockN;
  return tile;
}

template <typename Shape>
__device__ __forceinline__ void tc4_mainloop_load_producer(
    Tc4PipelineState<Shape>* state, Tc4WorkTile tile, int k_tiles,
    const half* a, const CUtensorMap* a_map, const half* b,
    const CUtensorMap* b_map, half* a_smem, half* b_smem) {
  // Warp role: Mainloop Load.
  //
  // Pipeline:
  //   acquire empty stage
  //   TMA load A/B/SFA/SFB
  //   arrive mainloop_full[stage]
  //   after prologue, arrive LoadOrderPipeline so Epilogue Load can start
  (void)state;
  (void)tile;
  (void)k_tiles;
  (void)a;
  (void)a_map;
  (void)b;
  (void)b_map;
  (void)a_smem;
  (void)b_smem;
}

template <typename Shape>
__device__ __forceinline__ void tc4_mma_consumer_sm120_mma(
    Tc4PipelineState<Shape>* state, Tc4WorkTile tile, int k_tiles,
    const half* a_smem, const half* b_smem) {
  // Warp role: MMA.
  //
  // Pipeline:
  //   wait mainloop_full[stage]
  //   issue SM120 mma.sync A/B(SMEM/register fragments) -> D(registers)
  //   release mainloop_empty[stage]
  //   arrive accumulator_bar for epilogue
  (void)state;
  (void)tile;
  (void)k_tiles;
  (void)a_smem;
  (void)b_smem;
}

template <typename Shape>
__device__ __forceinline__ void tc4_epilogue_load_producer(
    Tc4PipelineState<Shape>* state, Tc4WorkTile tile, const float* c,
    const CUtensorMap* c_map, float* c_smem) {
  // Warp role: Epilogue Load.
  //
  // Pipeline:
  //   wait LoadOrderPipeline
  //   TMA load C / beta*C from GMEM -> SMEM
  //   arrive epilogue_load_bar
  (void)state;
  (void)tile;
  (void)c;
  (void)c_map;
  (void)c_smem;
}

template <typename Shape>
__device__ __forceinline__ void tc4_epilogue_consumer_accumulator(
    Tc4PipelineState<Shape>* state, Tc4WorkTile tile, float alpha, float beta,
    const float* c_smem, float* d_smem) {
  // Warp role: Epilogue 4-7.
  //
  // Pipeline:
  //   wait accumulator_bar
  //   wait epilogue_load_bar
  //   consume D register accumulators from the MMA warp/group handoff
  //   apply alpha/beta/postprocess
  //   store final D registers -> SMEM
  //   arrive epilogue_store_bar
  (void)state;
  (void)tile;
  (void)alpha;
  (void)beta;
  (void)c_smem;
  (void)d_smem;
}

template <typename Shape>
__device__ __forceinline__ void tc4_epilogue_tma_store(
    Tc4PipelineState<Shape>* state, Tc4WorkTile tile, const float* d_smem,
    float* d, const CUtensorMap* d_map) {
  // Producer: Epilogue warps.
  // Consumer: TMA store engine.
  //
  // Pipeline:
  //   wait epilogue_store_bar
  //   TMA store D(SMEM) -> GMEM
  //   wait store completion before SMEM reuse
  (void)state;
  (void)tile;
  (void)d_smem;
  (void)d;
  (void)d_map;
}

#if 0
// Design skeleton only. Enable after tc3 has proven the SM120a narrow-MMA
// instruction path and after the SM120 TMA + MMA fragment layout is selected.
template <typename Shape = Tc4BlackwellShape>
__global__ void hgemm_tc4_blackwell_ws_pipeline(
    int m, int n, int k, float alpha, const half* a,
    const CUtensorMap* a_map, const half* b, const CUtensorMap* b_map,
    float beta, const float* c_in, const CUtensorMap* c_map, float* d,
    const CUtensorMap* d_map) {
  extern __shared__ __align__(128) unsigned char smem[];
  half* a_smem = reinterpret_cast<half*>(smem);
  half* b_smem =
      a_smem + Shape::kStages * Shape::kBlockM * Shape::kBlockK;
  float* c_smem = reinterpret_cast<float*>(
      b_smem + Shape::kStages * Shape::kBlockK * Shape::kBlockN);
  float* d_smem = c_smem + Shape::kBlockM * Shape::kBlockN;

  __shared__ Tc4PipelineState<Shape> state;
  const Tc4WarpRole role = tc4_warp_role();

  // Scheduler initializes barriers and first static work tile.
  if (role == Tc4WarpRole::kScheduler) {
    state.next_work_tile = static_cast<int>(blockIdx.x);
  }
  __syncthreads();

  const int total_tiles_n = n / Shape::kBlockN;
  const int total_tiles = (m / Shape::kBlockM) * total_tiles_n;

  while (true) {
    Tc4WorkTile tile{};
    if (role == Tc4WarpRole::kScheduler) {
      tile = tc4_scheduler_fetch_next_work<Shape>(&state, total_tiles_n,
                                                  total_tiles);
    }

    // TODO: multicast tile to all role warps.
    if (tile.block_m < 0) break;

    const int k_tiles = k / Shape::kBlockK;
    if (role == Tc4WarpRole::kMainloopLoad) {
      tc4_mainloop_load_producer<Shape>(&state, tile, k_tiles, a, a_map, b,
                                        b_map, a_smem, b_smem);
    } else if (role == Tc4WarpRole::kMma) {
      tc4_mma_consumer_sm120_mma<Shape>(&state, tile, k_tiles, a_smem,
                                        b_smem);
    } else if (role == Tc4WarpRole::kEpilogueLoad) {
      tc4_epilogue_load_producer<Shape>(&state, tile, c_in, c_map, c_smem);
    } else {
      tc4_epilogue_consumer_accumulator<Shape>(&state, tile, alpha, beta,
                                               c_smem, d_smem);
      tc4_epilogue_tma_store<Shape>(&state, tile, d_smem, d, d_map);
    }
  }
}
#endif
