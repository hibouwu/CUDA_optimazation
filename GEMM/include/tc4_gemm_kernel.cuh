#pragma once

#include "gemm_common.cuh"
#include "tc3_gemm_kernel.cuh"

#include <cuda/barrier>
#include <cuda_fp16.h>
#include <cstdint>

// tc4 = SM120 Blackwell mainloop rewrite experiments.
//
// This file mirrors the work-tile pipeline diagram at the code-organization
// level:
//
//   Scheduler/CLC -> Mainloop Load -> SMEM stages -> SM120 MMA
//   -> Epilogue Load -> Epilogue -> TMA store
//
// Important boundary:
//   This is the RTX 50-series SM120 route. It should use SM120
//   mma.sync.aligned.kind::f8f6f4 / block-scaled MMA instructions. It should
//   not use SM100/SM110 tcgen05/TMEM instructions. Accumulators are modeled as
//   register-resident data handed from MMA organization to epilogue.
//
// tc4a/tc4b deliberately stay non-warp-specialized so they can isolate
// mainloop layout and synchronization choices. Full producer/consumer warp
// specialization belongs to tc5.

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

  // tc4 keeps one more stage than tc3 to study latency hiding without the
  // shared-memory and barrier pressure of a 4-stage pipeline.
  static constexpr int kStages = 3;
  static constexpr int kWarps = 8;
  static constexpr int kThreads = kWarps * kWarpSize;

  static constexpr int kMmaWarp = 0;
  static constexpr int kSchedulerWarp = 1;
  static constexpr int kMainloopLoadWarp = 2;
  static constexpr int kEpilogueLoadWarp = 3;
  static constexpr int kEpilogueWarpBegin = 4;
  static constexpr int kEpilogueWarpEnd = 8;

  // One 128x64 CTA tile is decomposed into m16n8k32 MMA atoms.
  static constexpr int kMmaAtomsM = kBlockM / 16;
  static constexpr int kMmaAtomsN = kBlockN / 8;

  // SM120 FP8 MMA consumes each B operand as two packed u32 registers per
  // m16n8k32 atom. Prepacking B once per TMA stage removes repeated strided
  // byte gathers from the MMA issue loop.
  static constexpr int kMmaGroups = 8;
  static constexpr int kThreadsPerMmaGroup = 4;
  static constexpr int kColumnTiles = kBlockN / 8;
  static constexpr int kPackedBWordsPerStage =
      kColumnTiles * kMmaGroups * kThreadsPerMmaGroup * 2;
};

template <typename Shape = Tc4BlackwellShape>
constexpr size_t tc4_mainloop_smem_bytes() {
  // FP8 tc3 uses byte-sized operands, but this scaffold keeps the staging size
  // conservative while the exact SM120 swizzled layout is still undecided.
  return static_cast<size_t>(Shape::kStages) * Shape::kBlockK *
         (Shape::kBlockM + Shape::kBlockN);
}

template <typename Shape = Tc4BlackwellShape>
constexpr size_t tc4_epilogue_smem_bytes() {
  return static_cast<size_t>(Shape::kBlockM) * Shape::kBlockN * sizeof(float);
}

template <typename Shape = Tc4BlackwellShape>
constexpr size_t tc4_pipeline_smem_bytes() {
  return tc4_mainloop_smem_bytes<Shape>() +
         2 * tc4_epilogue_smem_bytes<Shape>();
}

using tc4_block_barrier = cuda::barrier<cuda::thread_scope_block>;
struct alignas(tc4_block_barrier) tc4_barrier_storage {
  unsigned char bytes[sizeof(tc4_block_barrier)];
};

struct Tc4WorkTile {
  int tile_id;
  int block_m;
  int block_n;
  int valid;
};

template <typename Shape>
struct Tc4PipelineState {
  // MainloopPipeline: TMA producer -> SM120 MMA consumer.
  tc4_barrier_storage mainloop_full[Shape::kStages];
  tc4_barrier_storage mainloop_empty[Shape::kStages];

  // CLCPipeline and CLCThrottlePipeline from the reference diagram.
  tc4_barrier_storage clc_response_bar;
  tc4_barrier_storage clc_throttle_bar;

  // LoadOrderPipeline: mainload prologue gets TMA bandwidth before epiload.
  tc4_barrier_storage load_order_bar;

  // AccumulatorPipeline: register accumulator handoff to epilogue organization.
  tc4_barrier_storage accumulator_bar;

  // Epilogue load/store side pipelines.
  tc4_barrier_storage epilogue_load_bar;
  tc4_barrier_storage epilogue_store_bar;

  Tc4WorkTile current_tile;
  int next_work_tile;
  int mainloop_epoch;
  int accumulator_epoch;
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
__device__ __forceinline__ bool tc4_is_epilogue_warp(Tc4WarpRole role) {
  const int role_id = static_cast<int>(role);
  return role_id >= Shape::kEpilogueWarpBegin &&
         role_id < Shape::kEpilogueWarpEnd;
}

template <typename Shape>
__device__ __forceinline__ void tc4_initialize_shared_state(
    Tc4PipelineState<Shape>* state, int first_tile) {
  if (threadIdx.x == 0) {
    state->current_tile = {-1, -1, -1, 0};
    state->next_work_tile = first_tile;
    state->mainloop_epoch = 0;
    state->accumulator_epoch = 0;
  }
}

template <typename Shape>
__device__ __forceinline__ Tc4WorkTile tc4_scheduler_static_fetch(
    Tc4PipelineState<Shape>* state, int total_tiles_n, int total_tiles) {
  // Static persistent-worker fallback for the future CLC path:
  // each CTA owns tile blockIdx.x, blockIdx.x + gridDim.x, ...
  const int tile_id = state->next_work_tile;
  state->next_work_tile += static_cast<int>(gridDim.x);

  Tc4WorkTile tile{};
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
__device__ __forceinline__ void tc4_scheduler_lane(
    Tc4PipelineState<Shape>* state, int total_tiles_n, int total_tiles) {
  // Reference-figure lane: Scheduler / CLC.
  //
  // Target behavior:
  //   1. wait CLCThrottlePipeline before over-producing work requests
  //   2. query CLC or fallback software persistent queue
  //   3. multicast work tile to all role warps in the CTA
  //   4. arrive CLCPipeline so role warps can fetch the tile
  if ((threadIdx.x & (kWarpSize - 1)) == 0) {
    state->current_tile =
        tc4_scheduler_static_fetch<Shape>(state, total_tiles_n, total_tiles);
  }
}

template <typename Shape>
__device__ __forceinline__ void tc4_mainloop_load_lane(
    Tc4PipelineState<Shape>* state, Tc4WorkTile tile, int k_tiles,
    const __nv_fp8_e4m3* a, const CUtensorMap* a_map,
    const __nv_fp8_e4m3* b, const CUtensorMap* b_map, uint8_t* a_smem,
    uint8_t* b_smem) {
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
__device__ __forceinline__ void tc4_mma_lane(
    Tc4PipelineState<Shape>* state, Tc4WorkTile tile, int k_tiles,
    const uint8_t* a_smem, const uint8_t* b_smem, float* accumulator_regs) {
  // Reference-figure lane: MMA.
  //
  // Target behavior:
  //   wait mainloop_full[stage]
  //   read/pack FP8 fragments from SMEM
  //   issue SM120 mma.sync.aligned.kind::f8f6f4 or block-scaled MMA
  //   release mainloop_empty[stage]
  //   arrive AccumulatorPipeline when the work tile accumulator is ready
  (void)state;
  (void)tile;
  (void)k_tiles;
  (void)a_smem;
  (void)b_smem;
  (void)accumulator_regs;
}

template <typename Shape>
__device__ __forceinline__ void tc4_epilogue_load_lane(
    Tc4PipelineState<Shape>* state, Tc4WorkTile tile, const float* c,
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
__device__ __forceinline__ void tc4_epilogue_lane(
    Tc4PipelineState<Shape>* state, Tc4WorkTile tile, float alpha, float beta,
    const float* c_smem, float* d_smem, const float* accumulator_regs) {
  // Reference-figure lane: epilogue.
  //
  // Target behavior:
  //   wait AccumulatorPipeline
  //   consume register accumulator handoff from the MMA lane
  //   wait EpilogueLoadPipeline if beta/C is required
  //   apply alpha/beta/postprocess or quantization
  //   write D tile into SMEM staging buffer
  //   arrive EpiStorePipeline
  (void)state;
  (void)tile;
  (void)alpha;
  (void)beta;
  (void)c_smem;
  (void)d_smem;
  (void)accumulator_regs;
}

template <typename Shape>
__device__ __forceinline__ void tc4_tma_store_lane(
    Tc4PipelineState<Shape>* state, Tc4WorkTile tile, const float* d_smem,
    float* d, const CUtensorMap* d_map) {
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

template <typename Shape = Tc4BlackwellShape>
__global__ void hgemm_tc4_sm120_ws_pipeline_design(
    int m, int n, int k, float alpha, const __nv_fp8_e4m3* a,
    const CUtensorMap* a_map, const __nv_fp8_e4m3* b,
    const CUtensorMap* b_map, float beta, const float* c_in,
    const CUtensorMap* c_map, float* d, const CUtensorMap* d_map) {
  extern __shared__ __align__(128) unsigned char smem[];
  uint8_t* a_smem = reinterpret_cast<uint8_t*>(smem);
  uint8_t* b_smem = a_smem + Shape::kStages * Shape::kBlockM * Shape::kBlockK;
  float* c_smem = reinterpret_cast<float*>(
      b_smem + Shape::kStages * Shape::kBlockK * Shape::kBlockN);
  float* d_smem = c_smem + Shape::kBlockM * Shape::kBlockN;

  __shared__ Tc4PipelineState<Shape> state;
  __shared__ float accumulator_regs[Shape::kMmaAtomsM * Shape::kMmaAtomsN];

  tc4_initialize_shared_state<Shape>(&state, static_cast<int>(blockIdx.x));
  __syncthreads();

  const Tc4WarpRole role = tc4_warp_role();
  const int total_tiles_n = ceil_div(n, Shape::kBlockN);
  const int total_tiles = ceil_div(m, Shape::kBlockM) * total_tiles_n;
  const int k_tiles = ceil_div(k, Shape::kBlockK);

  while (true) {
    if (role == Tc4WarpRole::kScheduler) {
      tc4_scheduler_lane<Shape>(&state, total_tiles_n, total_tiles);
    }
    __syncthreads();

    const Tc4WorkTile tile = state.current_tile;
    if (!tile.valid) break;

    if (role == Tc4WarpRole::kMainloopLoad) {
      tc4_mainloop_load_lane<Shape>(&state, tile, k_tiles, a, a_map, b, b_map,
                                    a_smem, b_smem);
    } else if (role == Tc4WarpRole::kMma) {
      tc4_mma_lane<Shape>(&state, tile, k_tiles, a_smem, b_smem,
                          accumulator_regs);
    } else if (role == Tc4WarpRole::kEpilogueLoad) {
      tc4_epilogue_load_lane<Shape>(&state, tile, c_in, c_map, c_smem);
    } else if (tc4_is_epilogue_warp<Shape>(role)) {
      tc4_epilogue_lane<Shape>(&state, tile, alpha, beta, c_smem, d_smem,
                               accumulator_regs);
      tc4_tma_store_lane<Shape>(&state, tile, d_smem, d, d_map);
    }
    __syncthreads();
  }
}

__host__ __device__ constexpr bool tc4_pipeline_scaffold_available() {
  return true;
}

__host__ __device__ constexpr bool tc4_launch_available() {
  return tc3_sm120a_narrow_mma_available();
}

constexpr size_t tc4_tma_fp8_smem_bytes() {
  const size_t tma_bytes = static_cast<size_t>(Tc4BlackwellShape::kStages) *
                           Tc4BlackwellShape::kBlockK *
                           (Tc4BlackwellShape::kBlockM +
                            Tc4BlackwellShape::kBlockN);
  const size_t packed_b_bytes =
      static_cast<size_t>(Tc4BlackwellShape::kStages) *
      Tc4BlackwellShape::kPackedBWordsPerStage * sizeof(uint32_t);
  return tma_bytes + packed_b_bytes;
}

constexpr size_t tc4a_tma_fp8_smem_bytes() {
  return tc4_tma_fp8_smem_bytes();
}

constexpr size_t tc4b_tma_fp8_smem_bytes() {
  const size_t tma_bytes = static_cast<size_t>(Tc4BlackwellShape::kStages) *
                           Tc4BlackwellShape::kBlockK *
                           (Tc4BlackwellShape::kBlockM +
                            Tc4BlackwellShape::kBlockN);
  const size_t packed_b_bytes =
      static_cast<size_t>(Tc4BlackwellShape::kStages) *
      Tc4BlackwellShape::kPackedBWordsPerStage * sizeof(uint32_t);
  return tma_bytes + packed_b_bytes;
}

template <typename Shape>
__device__ __forceinline__ void tc4_init_tma_barriers(
    int tid, tc3_block_barrier* a_bar, tc3_block_barrier* b_bar) {
  if (tid == 0) {
    for (int stage = 0; stage < Shape::kStages; ++stage) {
      init(&a_bar[stage], 1);
      init(&b_bar[stage], 1);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void tc4_zero_accumulators(float d[8][4]) {
#pragma unroll
  for (int col_tile = 0; col_tile < 8; ++col_tile) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      d[col_tile][i] = 0.0f;
    }
  }
}

template <typename Shape>
__device__ __forceinline__ void tc4_preload_mainloop(
    int tid, int tile_m, int tile_n, int k, int n, const __nv_fp8_e4m3* a,
    const CUtensorMap* a_map, const __nv_fp8_e4m3* b,
    const CUtensorMap* b_map, uint8_t* a_smem, uint8_t* b_smem,
    tc3_block_barrier* a_bar, tc3_block_barrier* b_bar,
    tc3_block_barrier::arrival_token* a_tokens,
    tc3_block_barrier::arrival_token* b_tokens, int num_tiles) {
  const int preload_tiles =
      num_tiles < Shape::kStages ? num_tiles : Shape::kStages;
  for (int preload = 0; preload < preload_tiles; ++preload) {
    const int stage = preload;
    a_tokens[stage] = tc3_launch_a_tile_tma<Shape::kBlockM, Shape::kBlockK>(
        tid, tile_m, preload * Shape::kBlockK, a, k, a_map,
        a_smem + stage * Shape::kBlockM * Shape::kBlockK, a_bar[stage]);
    b_tokens[stage] = tc3_launch_b_tile_tma<Shape::kBlockK, Shape::kBlockN>(
        tid, tile_n, preload * Shape::kBlockK, b, n, b_map,
        b_smem + stage * Shape::kBlockK * Shape::kBlockN, b_bar[stage]);
  }
}

template <typename Shape>
__device__ __forceinline__ void tc4_wait_mainloop_stage(
    int tid, int stage, tc3_block_barrier* a_bar, tc3_block_barrier* b_bar,
    tc3_block_barrier::arrival_token* a_tokens,
    tc3_block_barrier::arrival_token* b_tokens) {
  tc3_wait_tma_stage(
      tid, a_bar[stage],
      static_cast<tc3_block_barrier::arrival_token&&>(a_tokens[stage]),
      b_bar[stage],
      static_cast<tc3_block_barrier::arrival_token&&>(b_tokens[stage]));
}

__device__ __forceinline__ uint32_t tc4_load_u32_smem_aligned(
    const uint8_t* ptr) {
  return *reinterpret_cast<const uint32_t*>(ptr);
}

template <typename Shape>
__device__ __forceinline__ int tc4_b_pack_pair_index(
    int col_tile, int group, int thread_in_group) {
  return ((col_tile * Shape::kMmaGroups + group) *
              Shape::kThreadsPerMmaGroup +
          thread_in_group) *
         2;
}

template <typename Shape>
__device__ __forceinline__ void tc4_pack_b_stage(
    int tid, const uint8_t* b_stage, uint32_t* b_pack_stage) {
  constexpr int kPackPairs = Shape::kColumnTiles * Shape::kMmaGroups *
                             Shape::kThreadsPerMmaGroup;
  for (int pair = tid; pair < kPackPairs; pair += blockDim.x) {
    const int thread_in_group = pair % Shape::kThreadsPerMmaGroup;
    const int group =
        (pair / Shape::kThreadsPerMmaGroup) % Shape::kMmaGroups;
    const int col_tile =
        pair / (Shape::kThreadsPerMmaGroup * Shape::kMmaGroups);

    uint32_t b0 = 0;
    uint32_t b1 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      b0 |= static_cast<uint32_t>(
                b_stage[(thread_in_group * 4 + i) * Shape::kBlockN +
                        col_tile * 8 + group])
            << (8 * i);
      b1 |= static_cast<uint32_t>(
                b_stage[(thread_in_group * 4 + i + 16) * Shape::kBlockN +
                        col_tile * 8 + group])
            << (8 * i);
    }

    const int out = pair * 2;
    b_pack_stage[out] = b0;
    b_pack_stage[out + 1] = b1;
  }
}

__device__ __forceinline__ void tc4_stage_pack_barrier() {
  __syncthreads();
}

__device__ __forceinline__ void tc4_stage_refill_barrier() {
  __syncthreads();
}

template <typename Shape>
__device__ __forceinline__ void tc4_consume_mainloop_stage(
    int warp_id, int group, int thread_in_group, const uint8_t* a_stage,
    const uint32_t* b_pack_stage, float d[8][4]) {
  const int smem_row = warp_id * 16 + group;
  const uint32_t a0 =
      tc4_load_u32_smem_aligned(a_stage + smem_row * Shape::kBlockK +
                                thread_in_group * 4);
  const uint32_t a1 =
      tc4_load_u32_smem_aligned(a_stage + (smem_row + 8) * Shape::kBlockK +
                                thread_in_group * 4);
  const uint32_t a2 =
      tc4_load_u32_smem_aligned(a_stage + smem_row * Shape::kBlockK +
                                thread_in_group * 4 + 16);
  const uint32_t a3 =
      tc4_load_u32_smem_aligned(a_stage + (smem_row + 8) * Shape::kBlockK +
                                thread_in_group * 4 + 16);

#pragma unroll
  for (int col_tile = 0; col_tile < Shape::kColumnTiles; ++col_tile) {
    const int b_pair =
        tc4_b_pack_pair_index<Shape>(col_tile, group, thread_in_group);
    const uint32_t b0 = b_pack_stage[b_pair];
    const uint32_t b1 = b_pack_stage[b_pair + 1];

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4."
        "f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[col_tile][0]), "+f"(d[col_tile][1]),
          "+f"(d[col_tile][2]), "+f"(d[col_tile][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
}

template <typename Shape>
__device__ __forceinline__ void tc4_refill_mainloop_stage(
    int tid, int stage, int next_tile, int tile_m, int tile_n, int k, int n,
    const __nv_fp8_e4m3* a, const CUtensorMap* a_map,
    const __nv_fp8_e4m3* b, const CUtensorMap* b_map, uint8_t* a_smem,
    uint8_t* b_smem, tc3_block_barrier* a_bar, tc3_block_barrier* b_bar,
    tc3_block_barrier::arrival_token* a_tokens,
    tc3_block_barrier::arrival_token* b_tokens, int num_tiles) {
  if (next_tile < num_tiles) {
    a_tokens[stage] = tc3_launch_a_tile_tma<Shape::kBlockM, Shape::kBlockK>(
        tid, tile_m, next_tile * Shape::kBlockK, a, k, a_map,
        a_smem + stage * Shape::kBlockM * Shape::kBlockK, a_bar[stage]);
    b_tokens[stage] = tc3_launch_b_tile_tma<Shape::kBlockK, Shape::kBlockN>(
        tid, tile_n, next_tile * Shape::kBlockK, b, n, b_map,
        b_smem + stage * Shape::kBlockK * Shape::kBlockN, b_bar[stage]);
  }
}

template <typename Shape>
__device__ __forceinline__ void tc4_store_accumulators(
    int m, int n, int tile_n, int warp_m, int group, int thread_in_group,
    float alpha, float beta, float* c, float d[8][4]) {
  const int row0 = warp_m + group;
  const int row1 = warp_m + group + 8;
#pragma unroll
  for (int col_tile = 0; col_tile < 8; ++col_tile) {
    const int col0 = tile_n + col_tile * 8 + thread_in_group * 2;
    if (row0 < m && col0 + 1 < n) {
      float2 out = {alpha * d[col_tile][0], alpha * d[col_tile][1]};
      float2* dst = reinterpret_cast<float2*>(&c[row0 * n + col0]);
      if (beta != 0.0f) {
        const float2 old = *dst;
        out.x += beta * old.x;
        out.y += beta * old.y;
      }
      *dst = out;
    } else {
      if (row0 < m && col0 < n) {
        c[row0 * n + col0] =
            alpha * d[col_tile][0] + beta * c[row0 * n + col0];
      }
      if (row0 < m && col0 + 1 < n) {
        c[row0 * n + col0 + 1] =
            alpha * d[col_tile][1] + beta * c[row0 * n + col0 + 1];
      }
    }
    if (row1 < m && col0 + 1 < n) {
      float2 out = {alpha * d[col_tile][2], alpha * d[col_tile][3]};
      float2* dst = reinterpret_cast<float2*>(&c[row1 * n + col0]);
      if (beta != 0.0f) {
        const float2 old = *dst;
        out.x += beta * old.x;
        out.y += beta * old.y;
      }
      *dst = out;
    } else {
      if (row1 < m && col0 < n) {
        c[row1 * n + col0] =
            alpha * d[col_tile][2] + beta * c[row1 * n + col0];
      }
      if (row1 < m && col0 + 1 < n) {
        c[row1 * n + col0 + 1] =
            alpha * d[col_tile][3] + beta * c[row1 * n + col0 + 1];
      }
    }
  }
}

__global__ void hgemm_tc4a_sm120a_fp8_tma_3stage_prepack_mma_128x64x32(
    int m, int n, int k, float alpha, const __nv_fp8_e4m3* a,
    const __grid_constant__ CUtensorMap* const a_map, const __nv_fp8_e4m3* b,
    const __grid_constant__ CUtensorMap* const b_map, float beta, float* c) {
#if TC3_HAS_SM120A_NARROW_MMA
  using Shape = Tc4BlackwellShape;
  extern __shared__ __align__(128) unsigned char tc4_smem[];
  uint8_t* a_smem = reinterpret_cast<uint8_t*>(tc4_smem);
  uint8_t* b_smem = a_smem + Shape::kStages * Shape::kBlockM * Shape::kBlockK;
  uint32_t* b_pack_smem = reinterpret_cast<uint32_t*>(
      b_smem + Shape::kStages * Shape::kBlockK * Shape::kBlockN);
  __shared__ tc3_block_barrier_storage a_bar_storage[Shape::kStages];
  __shared__ tc3_block_barrier_storage b_bar_storage[Shape::kStages];

  auto* a_bar = reinterpret_cast<tc3_block_barrier*>(a_bar_storage);
  auto* b_bar = reinterpret_cast<tc3_block_barrier*>(b_bar_storage);

  const int tid = threadIdx.x;
  tc4_init_tma_barriers<Shape>(tid, a_bar, b_bar);

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int tile_m = blockIdx.y * Shape::kBlockM;
  const int tile_n = blockIdx.x * Shape::kBlockN;
  const int warp_m = tile_m + warp_id * 16;
  const int num_tiles = k / Shape::kBlockK;

  float d[8][4];
  tc4_zero_accumulators(d);

  tc3_block_barrier::arrival_token a_tokens[Shape::kStages];
  tc3_block_barrier::arrival_token b_tokens[Shape::kStages];
  tc4_preload_mainloop<Shape>(tid, tile_m, tile_n, k, n, a, a_map, b, b_map,
                              a_smem, b_smem, a_bar, b_bar, a_tokens,
                              b_tokens, num_tiles);

  for (int tile = 0; tile < num_tiles; ++tile) {
    const int stage = tile % Shape::kStages;
    tc4_wait_mainloop_stage<Shape>(tid, stage, a_bar, b_bar, a_tokens,
                                   b_tokens);

    const uint8_t* a_stage =
        a_smem + stage * Shape::kBlockM * Shape::kBlockK;
    const uint8_t* b_stage =
        b_smem + stage * Shape::kBlockK * Shape::kBlockN;
    uint32_t* b_pack_stage =
        b_pack_smem + stage * Shape::kPackedBWordsPerStage;

    tc4_pack_b_stage<Shape>(tid, b_stage, b_pack_stage);
    tc4_stage_pack_barrier();
    tc4_consume_mainloop_stage<Shape>(warp_id, group, thread_in_group, a_stage,
                                      b_pack_stage, d);
    tc4_stage_refill_barrier();

    const int next_tile = tile + Shape::kStages;
    tc4_refill_mainloop_stage<Shape>(tid, stage, next_tile, tile_m, tile_n, k,
                                     n, a, a_map, b, b_map, a_smem, b_smem,
                                     a_bar, b_bar, a_tokens, b_tokens,
                                     num_tiles);
  }

  tc4_store_accumulators<Shape>(m, n, tile_n, warp_m, group, thread_in_group,
                                alpha, beta, c, d);
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    c[0] = 0.0f;
  }
#endif
}

__global__ void hgemm_tc4b_sm120a_fp8_tma_3stage_swizzle_mma_128x64x32(
    int m, int n, int k, float alpha, const __nv_fp8_e4m3* a,
    const __grid_constant__ CUtensorMap* const a_map, const __nv_fp8_e4m3* b,
    const __grid_constant__ CUtensorMap* const b_map, float beta, float* c) {
#if TC3_HAS_SM120A_NARROW_MMA
  using Shape = Tc4BlackwellShape;
  extern __shared__ __align__(128) unsigned char tc4_smem[];
  uint8_t* a_smem = reinterpret_cast<uint8_t*>(tc4_smem);
  uint8_t* b_smem = a_smem + Shape::kStages * Shape::kBlockM * Shape::kBlockK;
  uint32_t* b_pack_smem = reinterpret_cast<uint32_t*>(
      b_smem + Shape::kStages * Shape::kBlockK * Shape::kBlockN);
  __shared__ tc3_block_barrier_storage a_bar_storage[Shape::kStages];
  __shared__ tc3_block_barrier_storage b_bar_storage[Shape::kStages];

  auto* a_bar = reinterpret_cast<tc3_block_barrier*>(a_bar_storage);
  auto* b_bar = reinterpret_cast<tc3_block_barrier*>(b_bar_storage);

  const int tid = threadIdx.x;
  tc4_init_tma_barriers<Shape>(tid, a_bar, b_bar);

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int tile_m = blockIdx.y * Shape::kBlockM;
  const int tile_n = blockIdx.x * Shape::kBlockN;
  const int warp_m = tile_m + warp_id * 16;
  const int num_tiles = k / Shape::kBlockK;

  float d[8][4];
  tc4_zero_accumulators(d);

  tc3_block_barrier::arrival_token a_tokens[Shape::kStages];
  tc3_block_barrier::arrival_token b_tokens[Shape::kStages];
  tc4_preload_mainloop<Shape>(tid, tile_m, tile_n, k, n, a, a_map, b, b_map,
                              a_smem, b_smem, a_bar, b_bar, a_tokens,
                              b_tokens, num_tiles);

  for (int tile = 0; tile < num_tiles; ++tile) {
    const int stage = tile % Shape::kStages;
    tc4_wait_mainloop_stage<Shape>(tid, stage, a_bar, b_bar, a_tokens,
                                   b_tokens);

    const uint8_t* a_stage =
        a_smem + stage * Shape::kBlockM * Shape::kBlockK;
    const uint8_t* b_stage =
        b_smem + stage * Shape::kBlockK * Shape::kBlockN;
    uint32_t* b_pack_stage =
        b_pack_smem + stage * Shape::kPackedBWordsPerStage;

    tc4_pack_b_stage<Shape>(tid, b_stage, b_pack_stage);
    tc4_stage_pack_barrier();
    tc4_consume_mainloop_stage<Shape>(warp_id, group, thread_in_group, a_stage,
                                      b_pack_stage, d);
    tc4_stage_refill_barrier();

    const int next_tile = tile + Shape::kStages;
    tc4_refill_mainloop_stage<Shape>(tid, stage, next_tile, tile_m, tile_n, k,
                                     n, a, a_map, b, b_map, a_smem, b_smem,
                                     a_bar, b_bar, a_tokens, b_tokens,
                                     num_tiles);
  }

  tc4_store_accumulators<Shape>(m, n, tile_n, warp_m, group, thread_in_group,
                                alpha, beta, c, d);
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    c[0] = 0.0f;
  }
#endif
}
