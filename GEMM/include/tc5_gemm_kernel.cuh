#pragma once

#include "gemm_common.cuh"
#include "tc4_gemm_kernel.cuh"

#ifndef TC5_DEBUG_PROGRESS
#define TC5_DEBUG_PROGRESS 0
#endif

#if TC5_DEBUG_PROGRESS
#define TC5_DEBUG_PRINT(...)                                                   \
  do {                                                                         \
    if (blockIdx.x == 0 && lane == 0) printf(__VA_ARGS__);                     \
  } while (0)
#else
#define TC5_DEBUG_PRINT(...)                                                   \
  do {                                                                         \
  } while (0)
#endif

// tc5 = SM120 CLC / Cluster Launch Control scheduling bring-up.
//
// Boundary:
//   tc4a/tc4b are non-warp-specialized experiments for TMA stage count,
//   operand layout, swizzle, and synchronization cost.
//
//   tc5 is reserved for the scheduling layer:
//     CLC / cluster launch control work-tile acquisition
//     persistent cluster/CTA worker loop
//     software fallback when hardware CLC is not exposed by the toolchain
//     CLC throttle policy to avoid over-producing tile requests
//
//   Warp roles follow the SM120/CUTLASS-style fast path:
//     warp 0     : scheduler / CLC work-tile fetch
//     warp 1     : mainload / TMA A/B producer
//     warps 2..9 : MMA consumers with register-fragment epilogue
//
//   Keeping the accumulator in registers is the key difference from the
//   earlier full-role scaffold.  A separate epilogue warp needs a shared-memory
//   accumulator handoff on SM120, which costs occupancy and serialization.

static constexpr int kTc5SchedulerWarp = 0;
static constexpr int kTc5MainloadWarp = 1;
static constexpr int kTc5MmaWarpBegin = 2;
static constexpr int kTc5MmaWarpEnd = kTc5MmaWarpBegin + Tc4BlackwellShape::kWarps;
static constexpr int kTc5ClcParticipants =
    1 + 1 + Tc4BlackwellShape::kWarps;  // scheduler + mainload + MMA
static constexpr int kTc5MainloopParticipants = 1 + Tc4BlackwellShape::kWarps;

struct Tc5ClcWorkTile {
  int tile_id;
  int block_m;
  int block_n;
  int valid;
};

struct Tc5ClcConfig {
  int total_tiles_m;
  int total_tiles_n;
  int total_tiles;
  int worker_count;
};

template <typename Shape>
__host__ __device__ constexpr Tc5ClcConfig tc5_make_clc_config(int m, int n) {
  Tc5ClcConfig config{};
  config.total_tiles_m = ceil_div(m, Shape::kBlockM);
  config.total_tiles_n = ceil_div(n, Shape::kBlockN);
  config.total_tiles = config.total_tiles_m * config.total_tiles_n;
  config.worker_count = 0;
  return config;
}

template <typename Shape>
__device__ __forceinline__ Tc5ClcWorkTile tc5_tile_from_id(
    int tile_id, const Tc5ClcConfig& config) {
  Tc5ClcWorkTile tile{};
  tile.tile_id = tile_id;
  if (tile_id >= config.total_tiles) {
    tile.block_m = -1;
    tile.block_n = -1;
    tile.valid = 0;
    return tile;
  }
  tile.block_m = (tile_id / config.total_tiles_n) * Shape::kBlockM;
  tile.block_n = (tile_id % config.total_tiles_n) * Shape::kBlockN;
  tile.valid = 1;
  return tile;
}

template <typename Shape>
__device__ __forceinline__ Tc5ClcWorkTile tc5_static_clc_fetch(
    int worker_id, int iteration, const Tc5ClcConfig& config) {
  const int tile_id = worker_id + iteration * gridDim.x;
  return tc5_tile_from_id<Shape>(tile_id, config);
}

template <typename Shape>
__device__ __forceinline__ Tc5ClcWorkTile tc5_dynamic_clc_fetch(
    int* work_counter, const Tc5ClcConfig& config) {
  const int tile_id = atomicAdd(work_counter, 1);
  return tc5_tile_from_id<Shape>(tile_id, config);
}

__host__ __device__ constexpr bool tc5_launch_available() {
  return tc4_launch_available();
}

constexpr size_t tc5a_tma_fp8_smem_bytes() {
  return tc4b_tma_fp8_smem_bytes();
}

struct Tc5StagePipeline {
  tc3_block_barrier_storage clc_response;
  tc3_block_barrier_storage tma_ready[Tc4BlackwellShape::kStages];
  tc3_block_barrier_storage stage_empty[Tc4BlackwellShape::kStages];
};

enum Tc5ClcMode {
  kTc5ClcStatic = 0,
  kTc5ClcDynamic = 1,
};

template <typename Shape>
__device__ __forceinline__ void tc5_init_stage_pipeline(
    int tid, Tc5StagePipeline* pipeline) {
  auto* tma_ready = reinterpret_cast<tc3_block_barrier*>(pipeline->tma_ready);
  auto* stage_empty =
      reinterpret_cast<tc3_block_barrier*>(pipeline->stage_empty);
  if (tid == 0) {
    init(reinterpret_cast<tc3_block_barrier*>(&pipeline->clc_response),
         kTc5ClcParticipants);
    for (int stage = 0; stage < Shape::kStages; ++stage) {
      init(&tma_ready[stage], kTc5MainloopParticipants);
      init(&stage_empty[stage], kTc5MainloopParticipants);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void tc5_warp_lane0_arrive_and_wait(
    int lane, tc3_block_barrier& bar) {
  __syncwarp();
  if (lane == 0) {
    bar.arrive_and_wait();
  }
  __syncwarp();
}

__device__ __forceinline__ void tc5_warp_lane0_arrive(
    int lane, tc3_block_barrier& bar) {
  __syncwarp();
  if (lane == 0) {
    auto token = bar.arrive();
    (void)token;
  }
  __syncwarp();
}

template <typename Shape>
__device__ __forceinline__ void tc5_producer_wait_tma_stage(
    int tid, int stage, tc3_block_barrier* a_bar, tc3_block_barrier* b_bar,
    tc3_block_barrier::arrival_token* a_tokens,
    tc3_block_barrier::arrival_token* b_tokens) {
#if __CUDA_ARCH__ >= 900
  if (tid == 0) {
    a_bar[stage].wait(
        static_cast<tc3_block_barrier::arrival_token&&>(a_tokens[stage]));
    b_bar[stage].wait(
        static_cast<tc3_block_barrier::arrival_token&&>(b_tokens[stage]));
    tc3_cde::fence_proxy_async_shared_cta();
  }
#else
  __syncthreads();
#endif
}

template <typename Shape>
__device__ __forceinline__ void tc5_pack_b_stage_by_warp(
    int warp_id, int lane, const uint8_t* b_stage, uint32_t* b_pack_stage) {
  constexpr int kPackPairs = Shape::kColumnTiles * Shape::kMmaGroups *
                             Shape::kThreadsPerMmaGroup;
  for (int pair = warp_id * 32 + lane; pair < kPackPairs;
       pair += Shape::kWarps * 32) {
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

template <typename Shape>
__device__ __forceinline__ void tc5_produce_work_tile_pipeline(
    int role_tid, int lane, int m,
    int n, int k, float alpha, const __nv_fp8_e4m3* a,
    const CUtensorMap* a_map, const __nv_fp8_e4m3* b,
    const CUtensorMap* b_map, float beta, float* c, Tc5ClcWorkTile work,
    uint8_t* a_smem, uint8_t* b_smem, uint32_t* b_pack_smem,
    tc3_block_barrier* a_bar, tc3_block_barrier* b_bar,
    Tc5StagePipeline* pipeline) {
  using PipeShape = Shape;
  auto* tma_ready = reinterpret_cast<tc3_block_barrier*>(pipeline->tma_ready);
  auto* stage_empty =
      reinterpret_cast<tc3_block_barrier*>(pipeline->stage_empty);

  const int tile_m = work.block_m;
  const int tile_n = work.block_n;
  const int num_tiles_k = k / PipeShape::kBlockK;

  tc3_block_barrier::arrival_token a_tokens[PipeShape::kStages];
  tc3_block_barrier::arrival_token b_tokens[PipeShape::kStages];
  tc4_preload_mainloop<PipeShape>(
      role_tid, tile_m, tile_n, k, n, a, a_map, b, b_map, a_smem, b_smem, a_bar,
      b_bar, a_tokens, b_tokens, num_tiles_k);
  TC5_DEBUG_PRINT("mainload preload done tile=%d num_k=%d\n", work.tile_id,
                  num_tiles_k);

  for (int tile = 0; tile < num_tiles_k; ++tile) {
    const int stage = tile % PipeShape::kStages;
    TC5_DEBUG_PRINT("mainload wait tma tile=%d ktile=%d stage=%d\n",
                    work.tile_id, tile, stage);
    tc5_producer_wait_tma_stage<PipeShape>(role_tid, stage, a_bar, b_bar,
                                           a_tokens, b_tokens);
    TC5_DEBUG_PRINT("mainload tma ready tile=%d ktile=%d stage=%d\n",
                    work.tile_id, tile, stage);
    tc5_warp_lane0_arrive(lane, tma_ready[stage]);
    TC5_DEBUG_PRINT("mainload wait empty tile=%d ktile=%d stage=%d\n",
                    work.tile_id, tile, stage);
    tc5_warp_lane0_arrive_and_wait(lane, stage_empty[stage]);
    TC5_DEBUG_PRINT("mainload refill tile=%d ktile=%d stage=%d\n",
                    work.tile_id, tile, stage);

    const int next_tile = tile + PipeShape::kStages;
    tc4_refill_mainloop_stage<PipeShape>(
        role_tid, stage, next_tile, tile_m, tile_n, k, n, a, a_map, b, b_map,
        a_smem, b_smem, a_bar, b_bar, a_tokens, b_tokens, num_tiles_k);
  }
}

template <typename Shape>
__device__ __forceinline__ void tc5_store_accumulators_to_global(
    int consumer_warp_id, int group, int thread_in_group, float d[8][4],
    int m, int n, float alpha, float beta, float* c, Tc5ClcWorkTile work) {
  const int row0 = consumer_warp_id * 16 + group;
  const int row1 = row0 + 8;
#pragma unroll
  for (int col_tile = 0; col_tile < Shape::kColumnTiles; ++col_tile) {
    const int col0 = col_tile * 8 + thread_in_group * 2;
    const int global_row0 = work.block_m + row0;
    const int global_row1 = work.block_m + row1;
    const int global_col0 = work.block_n + col0;
    const int global_col1 = global_col0 + 1;
    if (global_row0 < m && global_col1 < n) {
      float2 out = {alpha * d[col_tile][0], alpha * d[col_tile][1]};
      float2* dst = reinterpret_cast<float2*>(&c[global_row0 * n + global_col0]);
      if (beta != 0.0f) {
        const float2 old = *dst;
        out.x += beta * old.x;
        out.y += beta * old.y;
      }
      *dst = out;
    } else {
      if (global_row0 < m && global_col0 < n) {
        c[global_row0 * n + global_col0] =
            alpha * d[col_tile][0] + beta * c[global_row0 * n + global_col0];
      }
      if (global_row0 < m && global_col1 < n) {
        c[global_row0 * n + global_col1] =
            alpha * d[col_tile][1] + beta * c[global_row0 * n + global_col1];
      }
    }
    if (global_row1 < m && global_col1 < n) {
      float2 out = {alpha * d[col_tile][2], alpha * d[col_tile][3]};
      float2* dst = reinterpret_cast<float2*>(&c[global_row1 * n + global_col0]);
      if (beta != 0.0f) {
        const float2 old = *dst;
        out.x += beta * old.x;
        out.y += beta * old.y;
      }
      *dst = out;
    } else {
      if (global_row1 < m && global_col0 < n) {
        c[global_row1 * n + global_col0] =
            alpha * d[col_tile][2] + beta * c[global_row1 * n + global_col0];
      }
      if (global_row1 < m && global_col1 < n) {
        c[global_row1 * n + global_col1] =
            alpha * d[col_tile][3] + beta * c[global_row1 * n + global_col1];
      }
    }
  }
}

template <typename Shape>
__device__ __forceinline__ void tc5_consume_work_tile_pipeline(
    int lane, int consumer_warp_id, int group, int thread_in_group, int k,
    Tc5ClcWorkTile work,
    uint8_t* a_smem, uint8_t* b_smem, uint32_t* b_pack_smem,
    int m, int n, float alpha, float beta, float* c,
    Tc5StagePipeline* pipeline) {
  using PipeShape = Shape;
  auto* tma_ready = reinterpret_cast<tc3_block_barrier*>(pipeline->tma_ready);
  auto* stage_empty =
      reinterpret_cast<tc3_block_barrier*>(pipeline->stage_empty);

  const int num_tiles_k = k / PipeShape::kBlockK;

  float d[8][4];
  tc4_zero_accumulators(d);

  for (int tile = 0; tile < num_tiles_k; ++tile) {
    const int stage = tile % PipeShape::kStages;
    if (consumer_warp_id == 0) {
      TC5_DEBUG_PRINT("mma wait tma tile=%d ktile=%d stage=%d\n",
                      work.tile_id, tile, stage);
    }
    tc5_warp_lane0_arrive_and_wait(lane, tma_ready[stage]);
    if (consumer_warp_id == 0) {
      TC5_DEBUG_PRINT("mma got tma tile=%d ktile=%d stage=%d\n", work.tile_id,
                      tile, stage);
    }

    const uint8_t* a_stage =
        a_smem + stage * PipeShape::kBlockM * PipeShape::kBlockK;
    const uint8_t* b_stage =
        b_smem + stage * PipeShape::kBlockK * PipeShape::kBlockN;
    uint32_t* b_pack_stage =
        b_pack_smem + stage * PipeShape::kPackedBWordsPerStage;

    tc5_pack_b_stage_by_warp<PipeShape>(consumer_warp_id, lane, b_stage,
                                        b_pack_stage);
    __syncwarp();
    tc4_consume_mainloop_stage<PipeShape>(
        consumer_warp_id, group, thread_in_group, a_stage, b_pack_stage, d);
    if (consumer_warp_id == 0) {
      TC5_DEBUG_PRINT("mma arrive empty tile=%d ktile=%d stage=%d\n",
                      work.tile_id, tile, stage);
    }
    tc5_warp_lane0_arrive(lane, stage_empty[stage]);
  }

  tc5_store_accumulators_to_global<PipeShape>(
      consumer_warp_id, group, thread_in_group, d, m, n, alpha, beta, c, work);
}

__global__ void hgemm_tc5a_sm120a_fp8_clc_static_tma_swizzle_mma_128x64x32(
    int m, int n, int k, float alpha, const __nv_fp8_e4m3* a,
    const __grid_constant__ CUtensorMap* const a_map, const __nv_fp8_e4m3* b,
    const __grid_constant__ CUtensorMap* const b_map, float beta, float* c) {
#if TC3_HAS_SM120A_NARROW_MMA
  using Shape = Tc4BlackwellShape;
  extern __shared__ __align__(128) unsigned char tc5_smem[];
  uint8_t* a_smem = reinterpret_cast<uint8_t*>(tc5_smem);
  uint8_t* b_smem = a_smem + Shape::kStages * Shape::kBlockM * Shape::kBlockK;
  uint32_t* b_pack_smem = reinterpret_cast<uint32_t*>(
      b_smem + Shape::kStages * Shape::kBlockK * Shape::kBlockN);
  __shared__ tc3_block_barrier_storage a_bar_storage[Shape::kStages];
  __shared__ tc3_block_barrier_storage b_bar_storage[Shape::kStages];
  __shared__ Tc5StagePipeline pipeline;

  auto* a_bar = reinterpret_cast<tc3_block_barrier*>(a_bar_storage);
  auto* b_bar = reinterpret_cast<tc3_block_barrier*>(b_bar_storage);

  const int tid = threadIdx.x;
  tc4_init_tma_barriers<Shape>(tid, a_bar, b_bar);
  tc5_init_stage_pipeline<Shape>(tid, &pipeline);

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const bool is_scheduler = warp_id == kTc5SchedulerWarp;
  const bool is_mainload = warp_id == kTc5MainloadWarp;
  const bool is_mma =
      warp_id >= kTc5MmaWarpBegin && warp_id < kTc5MmaWarpEnd;
  const int consumer_warp_id = warp_id - kTc5MmaWarpBegin;
  __shared__ Tc5ClcWorkTile work_slots[2];
  auto* clc_response =
      reinterpret_cast<tc3_block_barrier*>(&pipeline.clc_response);

  const Tc5ClcConfig config = tc5_make_clc_config<Shape>(m, n);

  if (is_scheduler) {
    for (int iter = 0, slot = 0;; ++iter, slot ^= 1) {
      if (lane == 0) {
        work_slots[slot] = tc5_static_clc_fetch<Shape>(
            static_cast<int>(blockIdx.x), iter, config);
      }
      tc5_warp_lane0_arrive_and_wait(lane, *clc_response);
      const Tc5ClcWorkTile work = work_slots[slot];
      if (!work.valid) break;
    }
  } else if (is_mainload) {
    for (int slot = 0;; slot ^= 1) {
      tc5_warp_lane0_arrive_and_wait(lane, *clc_response);
      const Tc5ClcWorkTile work = work_slots[slot];
      if (!work.valid) break;

      tc5_produce_work_tile_pipeline<Shape>(
          lane, lane, m, n, k, alpha, a, a_map, b, b_map, beta, c, work,
          a_smem, b_smem, b_pack_smem, a_bar, b_bar, &pipeline);
    }
  } else if (is_mma) {
    for (int slot = 0;; slot ^= 1) {
      tc5_warp_lane0_arrive_and_wait(lane, *clc_response);
      const Tc5ClcWorkTile work = work_slots[slot];
      if (!work.valid) break;
      tc5_consume_work_tile_pipeline<Shape>(
          lane, consumer_warp_id, group, thread_in_group, k, work, a_smem,
          b_smem, b_pack_smem, m, n, alpha, beta, c, &pipeline);
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    c[0] = 0.0f;
  }
#endif
}

__global__ void hgemm_tc5b_sm120a_fp8_clc_dynamic_tma_swizzle_mma_128x64x32(
    int m, int n, int k, float alpha, const __nv_fp8_e4m3* a,
    const __grid_constant__ CUtensorMap* const a_map, const __nv_fp8_e4m3* b,
    const __grid_constant__ CUtensorMap* const b_map, int* work_counter,
    float beta, float* c) {
#if TC3_HAS_SM120A_NARROW_MMA
  using Shape = Tc4BlackwellShape;
  extern __shared__ __align__(128) unsigned char tc5_smem[];
  uint8_t* a_smem = reinterpret_cast<uint8_t*>(tc5_smem);
  uint8_t* b_smem = a_smem + Shape::kStages * Shape::kBlockM * Shape::kBlockK;
  uint32_t* b_pack_smem = reinterpret_cast<uint32_t*>(
      b_smem + Shape::kStages * Shape::kBlockK * Shape::kBlockN);
  __shared__ tc3_block_barrier_storage a_bar_storage[Shape::kStages];
  __shared__ tc3_block_barrier_storage b_bar_storage[Shape::kStages];
  __shared__ Tc5StagePipeline pipeline;

  auto* a_bar = reinterpret_cast<tc3_block_barrier*>(a_bar_storage);
  auto* b_bar = reinterpret_cast<tc3_block_barrier*>(b_bar_storage);

  const int tid = threadIdx.x;
  tc4_init_tma_barriers<Shape>(tid, a_bar, b_bar);
  tc5_init_stage_pipeline<Shape>(tid, &pipeline);

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const bool is_scheduler = warp_id == kTc5SchedulerWarp;
  const bool is_mainload = warp_id == kTc5MainloadWarp;
  const bool is_mma =
      warp_id >= kTc5MmaWarpBegin && warp_id < kTc5MmaWarpEnd;
  const int consumer_warp_id = warp_id - kTc5MmaWarpBegin;
  const Tc5ClcConfig config = tc5_make_clc_config<Shape>(m, n);
  __shared__ Tc5ClcWorkTile work_slots[2];
  auto* clc_response =
      reinterpret_cast<tc3_block_barrier*>(&pipeline.clc_response);

  if (is_scheduler) {
    for (int slot = 0;; slot ^= 1) {
      if (lane == 0) {
        work_slots[slot] = tc5_dynamic_clc_fetch<Shape>(work_counter, config);
        TC5_DEBUG_PRINT("sched slot=%d tile=%d valid=%d\n", slot,
                        work_slots[slot].tile_id, work_slots[slot].valid);
      }
      tc5_warp_lane0_arrive_and_wait(lane, *clc_response);
      const Tc5ClcWorkTile work = work_slots[slot];
      if (!work.valid) break;
    }
  } else if (is_mainload) {
    for (int slot = 0;; slot ^= 1) {
      TC5_DEBUG_PRINT("mainload wait clc slot=%d\n", slot);
      tc5_warp_lane0_arrive_and_wait(lane, *clc_response);
      const Tc5ClcWorkTile work = work_slots[slot];
      TC5_DEBUG_PRINT("mainload got slot=%d tile=%d valid=%d\n", slot,
                      work.tile_id, work.valid);
      if (!work.valid) break;

      tc5_produce_work_tile_pipeline<Shape>(
          lane, lane, m, n, k, alpha, a, a_map, b, b_map, beta, c, work,
          a_smem, b_smem, b_pack_smem, a_bar, b_bar, &pipeline);
    }
  } else if (is_mma) {
    for (int slot = 0;; slot ^= 1) {
      if (consumer_warp_id == 0) {
        TC5_DEBUG_PRINT("mma wait clc slot=%d\n", slot);
      }
      tc5_warp_lane0_arrive_and_wait(lane, *clc_response);
      const Tc5ClcWorkTile work = work_slots[slot];
      if (consumer_warp_id == 0) {
        TC5_DEBUG_PRINT("mma got slot=%d tile=%d valid=%d empty=%d\n", slot,
                        work.tile_id, work.valid, 1);
      }
      if (!work.valid) break;
      tc5_consume_work_tile_pipeline<Shape>(
          lane, consumer_warp_id, group, thread_in_group, k, work, a_smem,
          b_smem, b_pack_smem, m, n, alpha, beta, c, &pipeline);
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    c[0] = 0.0f;
  }
#endif
}
