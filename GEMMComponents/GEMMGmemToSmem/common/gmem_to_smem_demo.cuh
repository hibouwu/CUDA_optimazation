#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace gmem_to_smem {

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                << cudaGetErrorString(err__) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define CHECK_CU(call)                                                       \
  do {                                                                       \
    CUresult status__ = (call);                                              \
    if (status__ != CUDA_SUCCESS) {                                          \
      const char* name__ = nullptr;                                          \
      const char* desc__ = nullptr;                                          \
      cuGetErrorName(status__, &name__);                                     \
      cuGetErrorString(status__, &desc__);                                   \
      std::cerr << "CUDA driver error at " << __FILE__ << ":" << __LINE__   \
                << " - " << (name__ != nullptr ? name__ : "UNKNOWN")        \
                << ": " << (desc__ != nullptr ? desc__ : "unknown")         \
                << std::endl;                                                \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;
constexpr int kNaiveBlock = 16;

enum class DemoKind {
  kNaive,
  kCoalesced,
  kVectorizedFloat4,
  kRowColMajorAddressing,
  kTransposedSmemStore,
  kCoalescedLoadTransposedStore,
  kSmemPaddingBankConflict,
  kSmemSwizzleStore,
  kTransposePaddingBankConflict,
  kPredicatedTileLoad,
  kDoubleBuffer,
  kCpAsync,
  kTmaScaffold,
};

struct DemoConfig {
  int rows = 4096;
  int cols = 4096;
  int iters = 200;
  int warmup = 20;
};

struct DemoMeta {
  const char* name;
  const char* summary;
  bool transpose_output;
  bool requires_cols_multiple_of_4;
  bool is_tma_scaffold;
};

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

inline DemoMeta demo_meta(DemoKind kind) {
  switch (kind) {
    case DemoKind::kNaive:
      return {"naive_gmem_to_smem",
              "threadIdx.x walks rows first, so global loads are intentionally uncoalesced",
              false,
              false,
              false};
    case DemoKind::kCoalesced:
      return {"coalesced_gmem_to_smem",
              "threadIdx.x walks contiguous columns to improve global transaction efficiency",
              false,
              false,
              false};
    case DemoKind::kVectorizedFloat4:
      return {"vectorized_float4_load",
              "each thread issues a 16-byte float4 transaction into shared memory",
              false,
              true,
              false};
    case DemoKind::kRowColMajorAddressing:
      return {"row_col_major_addressing",
              "row-major global memory is restaged into a column-major shared-memory tile",
              true,
              false,
              false};
    case DemoKind::kTransposedSmemStore:
      return {"transposed_smem_store",
              "global rows are stored transposed inside shared memory before writing out a transpose",
              true,
              false,
              false};
    case DemoKind::kCoalescedLoadTransposedStore:
      return {"coalesced_load_transposed_store",
              "classic transpose pattern: coalesced global loads, transposed shared-memory staging",
              true,
              false,
              false};
    case DemoKind::kSmemPaddingBankConflict:
      return {"smem_padding_bank_conflict",
              "padding shared-memory stride by +1 removes transpose-time bank conflicts",
              true,
              false,
              false};
    case DemoKind::kSmemSwizzleStore:
      return {"smem_swizzle_store",
              "XOR swizzle changes the shared-memory bank mapping without adding padding",
              true,
              false,
              false};
    case DemoKind::kTransposePaddingBankConflict:
      return {"transpose_padding_bank_conflict",
              "transpose path with padding and explicit tail predication for practical GEMM tiles",
              true,
              false,
              false};
    case DemoKind::kPredicatedTileLoad:
      return {"predicated_tile_load",
              "tile edges are guarded so non-multiple matrix shapes are safe to benchmark",
              false,
              false,
              false};
    case DemoKind::kDoubleBuffer:
      return {"double_buffer_gmem_to_smem",
              "two shared-memory stages ping-pong while a row tile streams across K/column tiles",
              false,
              false,
              false};
    case DemoKind::kCpAsync:
      return {"cp_async_gmem_to_smem",
              "Ampere+ asynchronous global-to-shared copy using cp.async per 16-byte lane",
              false,
              true,
              false};
    case DemoKind::kTmaScaffold:
      return {"tma_gmem_to_smem",
              "driver-side tensor map setup plus a device-side scaffold for future cp.async.bulk.tensor",
              false,
              false,
              true};
  }
  return {"unknown", "unknown", false, false, false};
}

inline void print_usage(const char* binary) {
  std::cout << "usage: " << binary << " [rows] [cols] [iters] [warmup]\n";
}

inline DemoConfig parse_args(int argc, char** argv, DemoKind kind) {
  DemoConfig cfg;
  if (kind == DemoKind::kPredicatedTileLoad ||
      kind == DemoKind::kTransposePaddingBankConflict) {
    cfg.rows = 4093;
    cfg.cols = 4091;
  }
  if (argc > 1) {
    cfg.rows = std::atoi(argv[1]);
  }
  if (argc > 2) {
    cfg.cols = std::atoi(argv[2]);
  }
  if (argc > 3) {
    cfg.iters = std::atoi(argv[3]);
  }
  if (argc > 4) {
    cfg.warmup = std::atoi(argv[4]);
  }
  if (argc > 5 || cfg.rows <= 0 || cfg.cols <= 0 || cfg.iters <= 0 ||
      cfg.warmup < 0) {
    print_usage(argv[0]);
    std::exit(EXIT_FAILURE);
  }
  return cfg;
}

inline void init_matrix(std::vector<float>& host, int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      host[row * cols + col] =
          static_cast<float>(((row * 131) + (col * 17)) % 251) * 0.125f;
    }
  }
}

inline std::vector<float> build_reference(const std::vector<float>& input,
                                          int rows,
                                          int cols,
                                          bool transpose_output) {
  std::vector<float> ref(input.size(), 0.0f);
  if (!transpose_output) {
    ref = input;
    return ref;
  }
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      ref[col * rows + row] = input[row * cols + col];
    }
  }
  return ref;
}

inline float max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b) {
  float max_diff = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
  }
  return max_diff;
}

inline void print_header(const DemoMeta& meta,
                         const DemoConfig& cfg,
                         const cudaDeviceProp& prop) {
  std::cout << "demo: " << meta.name << "\n";
  std::cout << "summary: " << meta.summary << "\n";
  std::cout << "device: " << prop.name << " (sm_" << prop.major << prop.minor
            << ")\n";
  std::cout << "shape: " << cfg.rows << " x " << cfg.cols << ", iters="
            << cfg.iters << ", warmup=" << cfg.warmup << "\n";
}

template <typename LaunchFn>
int run_benchmark(int argc, char** argv, DemoKind kind, LaunchFn launch_fn) {
  const DemoMeta meta = demo_meta(kind);
  const DemoConfig cfg = parse_args(argc, argv, kind);
  if (meta.requires_cols_multiple_of_4 && (cfg.cols % 4 != 0)) {
    std::cerr << "error: this demo requires cols % 4 == 0 for 16-byte lanes\n";
    return EXIT_FAILURE;
  }

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  print_header(meta, cfg, prop);

  const size_t elem_count = static_cast<size_t>(cfg.rows) * cfg.cols;
  const size_t bytes = elem_count * sizeof(float);
  std::vector<float> host_in(elem_count);
  std::vector<float> host_out(elem_count, 0.0f);
  init_matrix(host_in, cfg.rows, cfg.cols);
  const std::vector<float> host_ref =
      build_reference(host_in, cfg.rows, cfg.cols, meta.transpose_output);

  float* d_in = nullptr;
  float* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, bytes));
  CHECK_CUDA(cudaMalloc(&d_out, bytes));
  CHECK_CUDA(
      cudaMemcpy(d_in, host_in.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0, bytes));

  for (int i = 0; i < cfg.warmup; ++i) {
    launch_fn(d_in, d_out, cfg);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < cfg.iters; ++i) {
    launch_fn(d_in, d_out, cfg);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CHECK_CUDA(cudaMemcpy(
      host_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  const float diff = max_abs_diff(host_out, host_ref);
  const double avg_ms = static_cast<double>(elapsed_ms) / cfg.iters;
  const double gbps =
      (static_cast<double>(bytes) * 2.0) / (avg_ms * 1.0e6);
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "avg_ms: " << avg_ms << "\n";
  std::cout << "effective_gbps: " << gbps << "\n";
  std::cout << "max_abs_diff: " << diff << "\n";
  std::cout << "matched: " << (diff <= 1.0e-5f ? "true" : "false") << "\n";

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  return diff <= 1.0e-5f ? EXIT_SUCCESS : EXIT_FAILURE;
}

__global__ void kernel_naive_uncoalesced(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int rows,
                                         int cols) {
  __shared__ float tile[kNaiveBlock][kNaiveBlock];
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    tile[threadIdx.x][threadIdx.y] = in[row * cols + col];
  }
  __syncthreads();
  if (row < rows && col < cols) {
    out[row * cols + col] = tile[threadIdx.x][threadIdx.y];
  }
}

__global__ void kernel_coalesced(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int rows,
                                 int cols) {
  __shared__ float tile[kNaiveBlock][kNaiveBlock];
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    tile[threadIdx.y][threadIdx.x] = in[row * cols + col];
  }
  __syncthreads();
  if (row < rows && col < cols) {
    out[row * cols + col] = tile[threadIdx.y][threadIdx.x];
  }
}

__global__ void kernel_vectorized_float4(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int rows,
                                         int cols) {
  __shared__ float4 tile[kBlockRows][kTileDim / 4];
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (row < rows && col + 3 < cols) {
    const float4 value =
        reinterpret_cast<const float4*>(in + row * cols + col)[0];
    tile[threadIdx.y][threadIdx.x] = value;
  }
  __syncthreads();
  if (row < rows && col + 3 < cols) {
    reinterpret_cast<float4*>(out + row * cols + col)[0] =
        tile[threadIdx.y][threadIdx.x];
  }
}

__global__ void kernel_row_col_major_addressing(const float* __restrict__ in,
                                                float* __restrict__ out,
                                                int rows,
                                                int cols) {
  __shared__ float tile[kNaiveBlock][kNaiveBlock];
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    tile[threadIdx.x][threadIdx.y] = in[row * cols + col];
  }
  __syncthreads();
  if (row < rows && col < cols) {
    out[col * rows + row] = tile[threadIdx.x][threadIdx.y];
  }
}

__global__ void kernel_transposed_smem_store(const float* __restrict__ in,
                                             float* __restrict__ out,
                                             int rows,
                                             int cols) {
  __shared__ float tile[kTileDim][kTileDim];
  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y0 = blockIdx.y * kTileDim + threadIdx.y;
  #pragma unroll
  for (int i = 0; i < kTileDim; i += kBlockRows) {
    const int y = y0 + i;
    if (x < cols && y < rows) {
      tile[threadIdx.x][threadIdx.y + i] = in[y * cols + x];
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < kTileDim; i += kBlockRows) {
    const int y = y0 + i;
    if (x < cols && y < rows) {
      out[x * rows + y] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}

__global__ void kernel_coalesced_transpose(const float* __restrict__ in,
                                           float* __restrict__ out,
                                           int rows,
                                           int cols) {
  __shared__ float tile[kTileDim][kTileDim];
  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y0 = blockIdx.y * kTileDim + threadIdx.y;
  #pragma unroll
  for (int i = 0; i < kTileDim; i += kBlockRows) {
    const int y = y0 + i;
    if (x < cols && y < rows) {
      tile[threadIdx.y + i][threadIdx.x] = in[y * cols + x];
    }
  }
  __syncthreads();
  const int tx = blockIdx.y * kTileDim + threadIdx.x;
  const int ty0 = blockIdx.x * kTileDim + threadIdx.y;
  #pragma unroll
  for (int i = 0; i < kTileDim; i += kBlockRows) {
    const int ty = ty0 + i;
    if (tx < rows && ty < cols) {
      out[ty * rows + tx] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}

__global__ void kernel_padded_transpose(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        int rows,
                                        int cols) {
  __shared__ float tile[kTileDim][kTileDim + 1];
  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y0 = blockIdx.y * kTileDim + threadIdx.y;
  #pragma unroll
  for (int i = 0; i < kTileDim; i += kBlockRows) {
    const int y = y0 + i;
    if (x < cols && y < rows) {
      tile[threadIdx.y + i][threadIdx.x] = in[y * cols + x];
    }
  }
  __syncthreads();
  const int tx = blockIdx.y * kTileDim + threadIdx.x;
  const int ty0 = blockIdx.x * kTileDim + threadIdx.y;
  #pragma unroll
  for (int i = 0; i < kTileDim; i += kBlockRows) {
    const int ty = ty0 + i;
    if (tx < rows && ty < cols) {
      out[ty * rows + tx] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}

__global__ void kernel_swizzled_transpose(const float* __restrict__ in,
                                          float* __restrict__ out,
                                          int rows,
                                          int cols) {
  __shared__ float tile[kTileDim][kTileDim];
  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y0 = blockIdx.y * kTileDim + threadIdx.y;
  #pragma unroll
  for (int i = 0; i < kTileDim; i += kBlockRows) {
    const int local_row = threadIdx.y + i;
    const int y = y0 + i;
    const int swizzled_col = threadIdx.x ^ (local_row & 0x7);
    if (x < cols && y < rows) {
      tile[local_row][swizzled_col] = in[y * cols + x];
    }
  }
  __syncthreads();
  const int tx = blockIdx.y * kTileDim + threadIdx.x;
  const int ty0 = blockIdx.x * kTileDim + threadIdx.y;
  #pragma unroll
  for (int i = 0; i < kTileDim; i += kBlockRows) {
    const int local_col = threadIdx.y + i;
    const int ty = ty0 + i;
    const int swizzled_row = threadIdx.x ^ (local_col & 0x7);
    if (tx < rows && ty < cols) {
      out[ty * rows + tx] = tile[local_col][swizzled_row];
    }
  }
}

__global__ void kernel_predicated_copy(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int rows,
                                       int cols) {
  __shared__ float tile[kTileDim][kTileDim];
  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y0 = blockIdx.y * kTileDim + threadIdx.y;
  #pragma unroll
  for (int i = 0; i < kTileDim; i += kBlockRows) {
    const int y = y0 + i;
    tile[threadIdx.y + i][threadIdx.x] =
        (x < cols && y < rows) ? in[y * cols + x] : 0.0f;
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < kTileDim; i += kBlockRows) {
    const int y = y0 + i;
    if (x < cols && y < rows) {
      out[y * cols + x] = tile[threadIdx.y + i][threadIdx.x];
    }
  }
}

__global__ void kernel_double_buffer(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     int rows,
                                     int cols) {
  __shared__ float stage[2][kBlockRows][kTileDim];
  const int row = blockIdx.x * kBlockRows + threadIdx.y;
  const int lane_col = threadIdx.x;
  if (row >= rows) {
    return;
  }
  int stage_idx = 0;
  if (lane_col < cols) {
    stage[stage_idx][threadIdx.y][lane_col] = in[row * cols + lane_col];
  } else {
    stage[stage_idx][threadIdx.y][lane_col] = 0.0f;
  }
  __syncthreads();
  for (int tile_col = 0; tile_col < cols; tile_col += kTileDim) {
    const int next_tile_col = tile_col + kTileDim;
    if (next_tile_col < cols) {
      const int next_stage = stage_idx ^ 1;
      const int next_col = next_tile_col + lane_col;
      stage[next_stage][threadIdx.y][lane_col] =
          (next_col < cols) ? in[row * cols + next_col] : 0.0f;
    }
    const int global_col = tile_col + lane_col;
    if (global_col < cols) {
      out[row * cols + global_col] = stage[stage_idx][threadIdx.y][lane_col];
    }
    __syncthreads();
    stage_idx ^= 1;
  }
}

__device__ __forceinline__ unsigned smem_addr_u32(const void* ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16(void* smem_ptr,
                                            const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr_u32(smem_ptr)),
               "l"(gmem_ptr));
#else
  (void)smem_ptr;
  (void)gmem_ptr;
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" : :);
#endif
}

__device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" : :);
#endif
}

__global__ void kernel_cp_async_copy(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     int rows,
                                     int cols) {
  __shared__ float4 tile[kBlockRows][kTileDim / 4];
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  float4 zeros = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (row < rows && col + 3 < cols) {
    cp_async_16(&tile[threadIdx.y][threadIdx.x], in + row * cols + col);
  } else {
    tile[threadIdx.y][threadIdx.x] = zeros;
  }
  cp_async_commit();
  cp_async_wait();
  __syncthreads();
  if (row < rows && col + 3 < cols) {
    reinterpret_cast<float4*>(out + row * cols + col)[0] =
        tile[threadIdx.y][threadIdx.x];
  }
}

__global__ void kernel_tma_scaffold(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int rows,
                                    int cols) {
  __shared__ float tile[kNaiveBlock][kNaiveBlock];
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    tile[threadIdx.y][threadIdx.x] = in[row * cols + col];
  }
  __syncthreads();
  if (row < rows && col < cols) {
    out[row * cols + col] = tile[threadIdx.y][threadIdx.x];
  }
}

inline void launch_naive(const float* d_in,
                         float* d_out,
                         const DemoConfig& cfg) {
  const dim3 block(kNaiveBlock, kNaiveBlock);
  const dim3 grid(ceil_div(cfg.rows, block.x), ceil_div(cfg.cols, block.y));
  kernel_naive_uncoalesced<<<grid, block>>>(d_in, d_out, cfg.rows, cfg.cols);
  CHECK_CUDA(cudaGetLastError());
}

inline void launch_coalesced(const float* d_in,
                             float* d_out,
                             const DemoConfig& cfg) {
  const dim3 block(kNaiveBlock, kNaiveBlock);
  const dim3 grid(ceil_div(cfg.cols, block.x), ceil_div(cfg.rows, block.y));
  kernel_coalesced<<<grid, block>>>(d_in, d_out, cfg.rows, cfg.cols);
  CHECK_CUDA(cudaGetLastError());
}

inline void launch_vectorized_float4(const float* d_in,
                                     float* d_out,
                                     const DemoConfig& cfg) {
  const dim3 block(kTileDim / 4, kBlockRows);
  const dim3 grid(ceil_div(cfg.cols / 4, block.x), ceil_div(cfg.rows, block.y));
  kernel_vectorized_float4<<<grid, block>>>(d_in, d_out, cfg.rows, cfg.cols);
  CHECK_CUDA(cudaGetLastError());
}

inline void launch_row_col_major_addressing(const float* d_in,
                                            float* d_out,
                                            const DemoConfig& cfg) {
  const dim3 block(kNaiveBlock, kNaiveBlock);
  const dim3 grid(ceil_div(cfg.cols, block.x), ceil_div(cfg.rows, block.y));
  kernel_row_col_major_addressing<<<grid, block>>>(d_in, d_out, cfg.rows,
                                                   cfg.cols);
  CHECK_CUDA(cudaGetLastError());
}

inline void launch_transposed_smem_store(const float* d_in,
                                         float* d_out,
                                         const DemoConfig& cfg) {
  const dim3 block(kTileDim, kBlockRows);
  const dim3 grid(ceil_div(cfg.cols, kTileDim), ceil_div(cfg.rows, kTileDim));
  kernel_transposed_smem_store<<<grid, block>>>(d_in, d_out, cfg.rows,
                                                cfg.cols);
  CHECK_CUDA(cudaGetLastError());
}

inline void launch_coalesced_load_transposed_store(const float* d_in,
                                                   float* d_out,
                                                   const DemoConfig& cfg) {
  const dim3 block(kTileDim, kBlockRows);
  const dim3 grid(ceil_div(cfg.cols, kTileDim), ceil_div(cfg.rows, kTileDim));
  kernel_coalesced_transpose<<<grid, block>>>(d_in, d_out, cfg.rows, cfg.cols);
  CHECK_CUDA(cudaGetLastError());
}

inline void launch_smem_padding_bank_conflict(const float* d_in,
                                              float* d_out,
                                              const DemoConfig& cfg) {
  const dim3 block(kTileDim, kBlockRows);
  const dim3 grid(ceil_div(cfg.cols, kTileDim), ceil_div(cfg.rows, kTileDim));
  kernel_padded_transpose<<<grid, block>>>(d_in, d_out, cfg.rows, cfg.cols);
  CHECK_CUDA(cudaGetLastError());
}

inline void launch_smem_swizzle_store(const float* d_in,
                                      float* d_out,
                                      const DemoConfig& cfg) {
  const dim3 block(kTileDim, kBlockRows);
  const dim3 grid(ceil_div(cfg.cols, kTileDim), ceil_div(cfg.rows, kTileDim));
  kernel_swizzled_transpose<<<grid, block>>>(d_in, d_out, cfg.rows, cfg.cols);
  CHECK_CUDA(cudaGetLastError());
}

inline void launch_transpose_padding_bank_conflict(const float* d_in,
                                                   float* d_out,
                                                   const DemoConfig& cfg) {
  const dim3 block(kTileDim, kBlockRows);
  const dim3 grid(ceil_div(cfg.cols, kTileDim), ceil_div(cfg.rows, kTileDim));
  kernel_padded_transpose<<<grid, block>>>(d_in, d_out, cfg.rows, cfg.cols);
  CHECK_CUDA(cudaGetLastError());
}

inline void launch_predicated_tile_load(const float* d_in,
                                        float* d_out,
                                        const DemoConfig& cfg) {
  const dim3 block(kTileDim, kBlockRows);
  const dim3 grid(ceil_div(cfg.cols, kTileDim), ceil_div(cfg.rows, kTileDim));
  kernel_predicated_copy<<<grid, block>>>(d_in, d_out, cfg.rows, cfg.cols);
  CHECK_CUDA(cudaGetLastError());
}

inline void launch_double_buffer(const float* d_in,
                                 float* d_out,
                                 const DemoConfig& cfg) {
  const dim3 block(kTileDim, kBlockRows);
  const dim3 grid(ceil_div(cfg.rows, kBlockRows));
  kernel_double_buffer<<<grid, block>>>(d_in, d_out, cfg.rows, cfg.cols);
  CHECK_CUDA(cudaGetLastError());
}

inline void launch_cp_async(const float* d_in,
                            float* d_out,
                            const DemoConfig& cfg) {
  const dim3 block(kTileDim / 4, kBlockRows);
  const dim3 grid(ceil_div(cfg.cols / 4, block.x), ceil_div(cfg.rows, block.y));
  kernel_cp_async_copy<<<grid, block>>>(d_in, d_out, cfg.rows, cfg.cols);
  CHECK_CUDA(cudaGetLastError());
}

inline void launch_tma_scaffold(const float* d_in,
                                float* d_out,
                                const DemoConfig& cfg) {
  const dim3 block(kNaiveBlock, kNaiveBlock);
  const dim3 grid(ceil_div(cfg.cols, block.x), ceil_div(cfg.rows, block.y));
  kernel_tma_scaffold<<<grid, block>>>(d_in, d_out, cfg.rows, cfg.cols);
  CHECK_CUDA(cudaGetLastError());
}

inline int run_demo_entry(int argc, char** argv, DemoKind kind) {
  switch (kind) {
    case DemoKind::kNaive:
      return run_benchmark(argc, argv, kind, launch_naive);
    case DemoKind::kCoalesced:
      return run_benchmark(argc, argv, kind, launch_coalesced);
    case DemoKind::kVectorizedFloat4:
      return run_benchmark(argc, argv, kind, launch_vectorized_float4);
    case DemoKind::kRowColMajorAddressing:
      return run_benchmark(argc, argv, kind, launch_row_col_major_addressing);
    case DemoKind::kTransposedSmemStore:
      return run_benchmark(argc, argv, kind, launch_transposed_smem_store);
    case DemoKind::kCoalescedLoadTransposedStore:
      return run_benchmark(argc, argv, kind,
                           launch_coalesced_load_transposed_store);
    case DemoKind::kSmemPaddingBankConflict:
      return run_benchmark(argc, argv, kind,
                           launch_smem_padding_bank_conflict);
    case DemoKind::kSmemSwizzleStore:
      return run_benchmark(argc, argv, kind, launch_smem_swizzle_store);
    case DemoKind::kTransposePaddingBankConflict:
      return run_benchmark(argc, argv, kind,
                           launch_transpose_padding_bank_conflict);
    case DemoKind::kPredicatedTileLoad:
      return run_benchmark(argc, argv, kind, launch_predicated_tile_load);
    case DemoKind::kDoubleBuffer:
      return run_benchmark(argc, argv, kind, launch_double_buffer);
    case DemoKind::kCpAsync:
      return run_benchmark(argc, argv, kind, launch_cp_async);
    case DemoKind::kTmaScaffold:
      break;
  }

  const DemoMeta meta = demo_meta(kind);
  const DemoConfig cfg = parse_args(argc, argv, kind);
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  print_header(meta, cfg, prop);

  const size_t elem_count = static_cast<size_t>(cfg.rows) * cfg.cols;
  const size_t bytes = elem_count * sizeof(float);
  std::vector<float> host_in(elem_count);
  std::vector<float> host_out(elem_count, 0.0f);
  init_matrix(host_in, cfg.rows, cfg.cols);
  const std::vector<float> host_ref = build_reference(host_in, cfg.rows, cfg.cols, false);

  float* d_in = nullptr;
  float* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, bytes));
  CHECK_CUDA(cudaMalloc(&d_out, bytes));
  CHECK_CUDA(cudaMemcpy(d_in, host_in.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0, bytes));

  CHECK_CU(cuInit(0));
  CUtensorMap tensor_map{};
  cuuint64_t global_dim[2] = {static_cast<cuuint64_t>(cfg.rows),
                              static_cast<cuuint64_t>(cfg.cols)};
  cuuint64_t global_strides[1] = {
      static_cast<cuuint64_t>(cfg.cols * sizeof(float))};
  cuuint32_t box_dim[2] = {kTileDim, kTileDim};
  cuuint32_t element_strides[2] = {1, 1};
  CHECK_CU(cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, d_in, global_dim,
      global_strides, box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  std::cout << "tensor_map: encoded successfully\n";
  std::cout << "note: device code is a TMA scaffold. replace kernel_tma_scaffold"
               " with cp.async.bulk.tensor on Hopper/Blackwell when ready.\n";

  for (int i = 0; i < cfg.warmup; ++i) {
    launch_tma_scaffold(d_in, d_out, cfg);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < cfg.iters; ++i) {
    launch_tma_scaffold(d_in, d_out, cfg);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CHECK_CUDA(cudaMemcpy(
      host_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  const float diff = max_abs_diff(host_out, host_ref);
  const double avg_ms = static_cast<double>(elapsed_ms) / cfg.iters;
  const double gbps =
      (static_cast<double>(bytes) * 2.0) / (avg_ms * 1.0e6);
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "avg_ms: " << avg_ms << "\n";
  std::cout << "effective_gbps: " << gbps << "\n";
  std::cout << "max_abs_diff: " << diff << "\n";
  std::cout << "matched: " << (diff <= 1.0e-5f ? "true" : "false") << "\n";

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  return diff <= 1.0e-5f ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace gmem_to_smem
