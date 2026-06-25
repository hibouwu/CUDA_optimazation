#pragma once

#include "gemm_common.cuh"
#include "tc3_gemm_kernel.cuh"

#include <cuda/barrier>
#include <cuda_fp16.h>
#include <cstdint>

// tc4(sm110) = Blackwell work-tile pipeline scaffold.
//
// This file mirrors the pipeline shown in docs/per.md and
// docs/pasted-pipeline-style figure:
//
//   Scheduler/CLC -> Mainloop Load -> SMEM stages -> TCGen05 MMA/TMEM
//   -> Epilogue Load -> Epilogue -> TMA store
//
// It is deliberately more concrete than the old placeholder: every lane in the
// figure has a warp role, a named pipeline, and a function boundary. The real
// launch is still disabled until the TCGen05 MMA descriptor, TMEM accumulator
// layout, and TMA store path are implemented and validated.

enum class Tc4Sm110WarpRole : int {
  kMma = 0,
  kScheduler = 1,
  kMainloopLoad = 2,
  kEpilogueLoad = 3,
  kEpilogue0 = 4,
  kEpilogue1 = 5,
  kEpilogue2 = 6,
  kEpilogue3 = 7,
};

struct Tc4Sm110Shape {
  static constexpr int kBlockM = 128;
  static constexpr int kBlockN = 64;
  static constexpr int kBlockK = 32;

  // The reference diagram uses four SMEM stages. Keeping this as the tc4 target
  // makes the intended pipeline different from tc3's minimal bring-up.
  static constexpr int kStages = 4;
  static constexpr int kWarps = 8;
  static constexpr int kThreads = kWarps * kWarpSize;

  static constexpr int kMmaWarp = 0;
  static constexpr int kSchedulerWarp = 1;
  static constexpr int kMainloopLoadWarp = 2;
  static constexpr int kEpilogueLoadWarp = 3;
  static constexpr int kEpilogueWarpBegin = 4;
  static constexpr int kEpilogueWarpEnd = 8;

  static constexpr unsigned int kTmemColumns = 128;
};

template <typename Shape = Tc4Sm110Shape>
constexpr size_t tc4_sm110_mainloop_smem_bytes() {
  return static_cast<size_t>(Shape::kStages) * Shape::kBlockK *
         (Shape::kBlockM + Shape::kBlockN) * sizeof(half);
}

template <typename Shape = Tc4Sm110Shape>
constexpr size_t tc4_sm110_epilogue_smem_bytes() {
  return static_cast<size_t>(Shape::kBlockM) * Shape::kBlockN * sizeof(float);
}

template <typename Shape = Tc4Sm110Shape>
constexpr size_t tc4_sm110_pipeline_smem_bytes() {
  return tc4_sm110_mainloop_smem_bytes<Shape>() +
         2 * tc4_sm110_epilogue_smem_bytes<Shape>();
}

using tc4_sm110_barrier = cuda::barrier<cuda::thread_scope_block>;
struct alignas(tc4_sm110_barrier) tc4_sm110_barrier_storage {
  unsigned char bytes[sizeof(tc4_sm110_barrier)];
};

struct Tc4Sm110WorkTile {
  int tile_id;
  int block_m;
  int block_n;
  int valid;
};

template <typename Shape>
struct Tc4Sm110PipelineState {
  // MainloopPipeline: TMA producer -> TCGen05 MMA consumer.
  tc4_sm110_barrier_storage mainloop_full[Shape::kStages];
  tc4_sm110_barrier_storage mainloop_empty[Shape::kStages];

  // CLCPipeline and CLCThrottlePipeline from the reference diagram.
  tc4_sm110_barrier_storage clc_response_bar;
  tc4_sm110_barrier_storage clc_throttle_bar;

  // LoadOrderPipeline: mainload prologue must get TMA priority before epiload.
  tc4_sm110_barrier_storage load_order_bar;

  // AccumulatorPipeline: TCGen05/TMEM accumulator handoff to epilogue.
  tc4_sm110_barrier_storage accumulator_bar;

  // Epilogue load/store side pipelines.
  tc4_sm110_barrier_storage epilogue_load_bar;
  tc4_sm110_barrier_storage epilogue_store_bar;

  Tc4Sm110WorkTile current_tile;
  int next_work_tile;
  int mainloop_epoch;
  int accumulator_epoch;
};

__device__ __forceinline__ Tc4Sm110WarpRole tc4_sm110_warp_role() {
  const int warp_id = threadIdx.x / kWarpSize;
  if (warp_id == 0) return Tc4Sm110WarpRole::kMma;
  if (warp_id == 1) return Tc4Sm110WarpRole::kScheduler;
  if (warp_id == 2) return Tc4Sm110WarpRole::kMainloopLoad;
  if (warp_id == 3) return Tc4Sm110WarpRole::kEpilogueLoad;
  return static_cast<Tc4Sm110WarpRole>(warp_id);
}

template <typename Shape>
__device__ __forceinline__ bool tc4_sm110_is_epilogue_warp(
    Tc4Sm110WarpRole role) {
  const int role_id = static_cast<int>(role);
  return role_id >= Shape::kEpilogueWarpBegin &&
         role_id < Shape::kEpilogueWarpEnd;
}

template <typename Shape>
__device__ __forceinline__ void tc4_sm110_initialize_shared_state(
    Tc4Sm110PipelineState<Shape>* state, int first_tile) {
  if (threadIdx.x == 0) {
    state->current_tile = {-1, -1, -1, 0};
    state->next_work_tile = first_tile;
    state->mainloop_epoch = 0;
    state->accumulator_epoch = 0;
  }
}

template <typename Shape>
__device__ __forceinline__ Tc4Sm110WorkTile
tc4_sm110_scheduler_static_fetch(Tc4Sm110PipelineState<Shape>* state,
                                 int total_tiles_n, int total_tiles) {
  // Static persistent-worker fallback for the future CLC path:
  // each CTA owns tile blockIdx.x, blockIdx.x + gridDim.x, ...
  const int tile_id = state->next_work_tile;
  state->next_work_tile += static_cast<int>(gridDim.x);

  Tc4Sm110WorkTile tile{};
  tile.tile_id = tile_id;
  if (tile_id >= total_tiles) {
    tile.block_m = -1;
    tile.block_n = -1;
    tile.valid = 0;
    return tile;
  }

  tile.block_m = (tile_id / total_tiles_n) * Shape::kBlockM;
  tile.block_n = (tile_id % total_tiles_n) * Shape::kBlockN;
  tile.valid = 1;
  return tile;
}

template <typename Shape>
__device__ __forceinline__ void tc4_sm110_scheduler_lane(
    Tc4Sm110PipelineState<Shape>* state, int total_tiles_n, int total_tiles) {
  // Reference-figure lane: Scheduler / CLC.
  //
  // Target behavior:
  //   1. wait CLCThrottlePipeline before over-producing work requests
  //   2. query CLC or fallback software persistent queue
  //   3. multicast work tile to all role warps in the CTA
  //   4. arrive CLCPipeline so role warps can fetch the tile
  if ((threadIdx.x & (kWarpSize - 1)) == 0) {
    state->current_tile =
        tc4_sm110_scheduler_static_fetch<Shape>(state, total_tiles_n,
                                                total_tiles);
  }
}

template <typename Shape>
__device__ __forceinline__ void tc4_sm110_mainloop_load_lane(
    Tc4Sm110PipelineState<Shape>* state, Tc4Sm110WorkTile tile, int k_tiles,
    const half* a, const CUtensorMap* a_map, const half* b,
    const CUtensorMap* b_map, half* a_smem, half* b_smem) {
  // Reference-figure lane: Main Load.
  //
  // Target behavior per K tile:
  //   acquire mainloop_empty[stage]
  //   TMA load A/B/SFA/SFB -> SMEM stage
  //   arrive mainloop_full[stage]
  //   after prologue, arrive LoadOrderPipeline so Epilogue Load may start
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
__device__ __forceinline__ void tc4_sm110_mma_lane(
    Tc4Sm110PipelineState<Shape>* state, Tc4Sm110WorkTile tile, int k_tiles,
    const half* a_smem, const half* b_smem, unsigned int* tmem_base) {
  // Reference-figure lane: MMA.
  //
  // Target behavior:
  //   wait mainloop_full[stage]
  //   copy/format operands from SMEM for TCGen05
  //   issue tcgen05.mma into TMEM accumulator
  //   release mainloop_empty[stage]
  //   arrive AccumulatorPipeline when the work tile accumulator is ready
  (void)state;
  (void)tile;
  (void)k_tiles;
  (void)a_smem;
  (void)b_smem;
  (void)tmem_base;
}

template <typename Shape>
__device__ __forceinline__ void tc4_sm110_epilogue_load_lane(
    Tc4Sm110PipelineState<Shape>* state, Tc4Sm110WorkTile tile, const float* c,
    const CUtensorMap* c_map, float* c_smem) {
  // Reference-figure lane: epi load.
  //
  // Target behavior:
  //   wait LoadOrderPipeline
  //   TMA load C or beta*C side input -> SMEM
  //   arrive EpilogueLoadPipeline
  (void)state;
  (void)tile;
  (void)c;
  (void)c_map;
  (void)c_smem;
}

template <typename Shape>
__device__ __forceinline__ void tc4_sm110_epilogue_lane(
    Tc4Sm110PipelineState<Shape>* state, Tc4Sm110WorkTile tile, float alpha,
    float beta, const float* c_smem, float* d_smem,
    const unsigned int* tmem_base) {
  // Reference-figure lane: epilogue.
  //
  // Target behavior:
  //   wait AccumulatorPipeline
  //   load D accumulator from TMEM -> registers
  //   wait EpilogueLoadPipeline if beta/C is required
  //   apply alpha/beta/postprocess
  //   write D tile into SMEM staging buffer
  //   arrive EpiStorePipeline
  (void)state;
  (void)tile;
  (void)alpha;
  (void)beta;
  (void)c_smem;
  (void)d_smem;
  (void)tmem_base;
}

template <typename Shape>
__device__ __forceinline__ void tc4_sm110_tma_store_lane(
    Tc4Sm110PipelineState<Shape>* state, Tc4Sm110WorkTile tile,
    const float* d_smem, float* d, const CUtensorMap* d_map) {
  // Reference-figure lane: TMA store from epilogue output.
  //
  // Target behavior:
  //   wait EpiStorePipeline
  //   TMA store D(SMEM) -> GMEM
  //   wait store completion before this worker reuses epilogue SMEM
  (void)state;
  (void)tile;
  (void)d_smem;
  (void)d;
  (void)d_map;
}

template <typename Shape = Tc4Sm110Shape>
__global__ void hgemm_tc4_sm110_ws_pipeline_design(
    int m, int n, int k, float alpha, const half* a, const CUtensorMap* a_map,
    const half* b, const CUtensorMap* b_map, float beta, const float* c_in,
    const CUtensorMap* c_map, float* d, const CUtensorMap* d_map) {
  extern __shared__ __align__(128) unsigned char smem[];
  half* a_smem = reinterpret_cast<half*>(smem);
  half* b_smem =
      a_smem + Shape::kStages * Shape::kBlockM * Shape::kBlockK;
  float* c_smem = reinterpret_cast<float*>(
      b_smem + Shape::kStages * Shape::kBlockK * Shape::kBlockN);
  float* d_smem = c_smem + Shape::kBlockM * Shape::kBlockN;

  __shared__ Tc4Sm110PipelineState<Shape> state;
  __shared__ unsigned int tmem_base;

  tc4_sm110_initialize_shared_state<Shape>(&state, static_cast<int>(blockIdx.x));
  __syncthreads();

  const Tc4Sm110WarpRole role = tc4_sm110_warp_role();
  const int total_tiles_n = ceil_div(n, Shape::kBlockN);
  const int total_tiles = ceil_div(m, Shape::kBlockM) * total_tiles_n;
  const int k_tiles = ceil_div(k, Shape::kBlockK);

  while (true) {
    if (role == Tc4Sm110WarpRole::kScheduler) {
      tc4_sm110_scheduler_lane<Shape>(&state, total_tiles_n, total_tiles);
    }
    __syncthreads();

    const Tc4Sm110WorkTile tile = state.current_tile;
    if (!tile.valid) break;

    if (role == Tc4Sm110WarpRole::kMainloopLoad) {
      tc4_sm110_mainloop_load_lane<Shape>(&state, tile, k_tiles, a, a_map, b,
                                          b_map, a_smem, b_smem);
    } else if (role == Tc4Sm110WarpRole::kMma) {
      tc4_sm110_mma_lane<Shape>(&state, tile, k_tiles, a_smem, b_smem,
                                &tmem_base);
    } else if (role == Tc4Sm110WarpRole::kEpilogueLoad) {
      tc4_sm110_epilogue_load_lane<Shape>(&state, tile, c_in, c_map, c_smem);
    } else if (tc4_sm110_is_epilogue_warp<Shape>(role)) {
      tc4_sm110_epilogue_lane<Shape>(&state, tile, alpha, beta, c_smem, d_smem,
                                     &tmem_base);
      tc4_sm110_tma_store_lane<Shape>(&state, tile, d_smem, d, d_map);
    }
    __syncthreads();
  }
}

__host__ __device__ constexpr bool tc4_sm110_pipeline_scaffold_available() {
  return true;
}

__host__ __device__ constexpr bool tc4_sm110_launch_available() {
  return false;
}
