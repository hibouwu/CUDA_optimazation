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
// The older cta_group::2/CLC sketch is intentionally not used here.  Correct
// finite-input validation showed that its TMEM accumulator ownership model was
// incomplete.  This file keeps the scheduling stage correct before adding a
// verified 2-SM specialization.

#include "../sm110_ptx_helpers.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace gemm_sm110::backends {

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
      reinterpret_cast<float4*>(dst)[0] =
          make_float4(values[0], values[1], values[2], values[3]);
      reinterpret_cast<float4*>(dst)[1] =
          make_float4(values[4], values[5], values[6], values[7]);
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

template <bool UseDynamic>
class Tc5Runner {
 public:
  Tc5Runner(const half* a, const half* b_nk, float* d,
            int m, int n, int k)
      : a_(a), b_nk_(b_nk), output_(d), m_(m), n_(n), k_(k) {
    fast_m_ = (m_ / kTileM) * kTileM;
    fast_n_ = (n_ / kTileN) * kTileN;
    fast_k_ = (k_ / kTileK) * kTileK;
    has_fast_path_ =
        fast_m_ > 0 && fast_n_ > 0 && fast_k_ > 0 && n_ % 4 == 0;

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
    const int worker_ctas =
        std::min(total_tiles, std::max(1, properties.multiProcessorCount));

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
    if (fast_m_ == m_ && fast_n_ == n_ && fast_k_ == k_) return;

    dim3 block(16, 16, 1);
    dim3 grid((n_ + static_cast<int>(block.x) - 1) /
                  static_cast<int>(block.x),
              (m_ + static_cast<int>(block.y) - 1) /
                  static_cast<int>(block.y),
              1);
    tc5_boundary_cleanup_kernel<<<grid, block>>>(
        a_, b_nk_, output_, m_, n_, k_, fast_m_, fast_n_, fast_k_);
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

using Tc5aRunner = Tc5Runner<false>;
using Tc5bRunner = Tc5Runner<true>;

}  // namespace gemm_sm110::backends
