#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err__ = (call);                                            \
    if (err__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err__));                             \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

static constexpr int kUnroll = 16;
static constexpr int kBytesPerOp = 4;

enum class Mode { kRead, kWrite };

__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t ld_shared_u32(uint32_t* ptr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(smem_addr(ptr)) : "memory");
  return v;
}

__device__ __forceinline__ void st_shared_u32(uint32_t* ptr, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" :: "r"(smem_addr(ptr)), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t pattern(uint32_t x, uint32_t i) {
  return (x * 1664525u + 1013904223u) ^ (i * 0x9e3779b9u);
}

template <Mode kMode>
__global__ __launch_bounds__(256, 1)
void smem_bank_kernel(int iters,
                      int warmup_iters,
                      int stride_words,
                      size_t element_mask,
                      uint32_t* sink,
                      unsigned long long* cycles_out) {
  extern __shared__ __align__(16) uint32_t smem[];
  const uint32_t tid = threadIdx.x;
  for (size_t i = tid; i <= element_mask; i += blockDim.x) {
    smem[i] = pattern(uint32_t(i + blockIdx.x * 65537u), 0u);
  }
  __syncthreads();

  uint32_t acc = pattern(blockIdx.x * blockDim.x + tid, 1u);
  const size_t lane_base = size_t(tid) * size_t(stride_words);
  const size_t round_stride = size_t(blockDim.x) * size_t(stride_words) * kUnroll + 1u;
  size_t idx[kUnroll];
  #pragma unroll
  for (int u = 0; u < kUnroll; ++u) {
    idx[u] = (lane_base + size_t(u)) & element_mask;
  }

  __shared__ unsigned long long block_start;
  __syncthreads();
  if (tid == 0) block_start = clock64();
  __syncthreads();

  const int total_iters = warmup_iters + iters;
  for (int i = 0; i < total_iters; ++i) {
    if (i == warmup_iters) {
      __syncthreads();
      if (tid == 0) block_start = clock64();
      __syncthreads();
    }
    if constexpr (kMode == Mode::kRead) {
      #pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        acc ^= ld_shared_u32(smem + idx[u]) + uint32_t(i + u);
      }
    } else {
      #pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        st_shared_u32(smem + idx[u], pattern(acc + uint32_t(idx[u]), uint32_t(i + u)));
        acc += uint32_t(i + u) ^ uint32_t(idx[u]);
      }
    }
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      idx[u] = (idx[u] + round_stride) & element_mask;
    }
  }
  __syncthreads();
  if (tid == 0) cycles_out[blockIdx.x] = clock64() - block_start;
  sink[blockIdx.x * blockDim.x + tid] = acc;
}

const char* mode_name(Mode mode) {
  return mode == Mode::kRead ? "read" : "write";
}

Mode parse_mode(const char* text) {
  if (std::strcmp(text, "read") == 0) return Mode::kRead;
  if (std::strcmp(text, "write") == 0) return Mode::kWrite;
  std::fprintf(stderr, "unknown mode: %s\n", text);
  std::exit(2);
}

struct Options {
  Mode mode = Mode::kRead;
  int iters = 8192;
  int warmup_iters = 128;
  int threads = 256;
  int blocks_per_sm = 1;
  int stride_words = 1;
  size_t shared_words = 8192;
  bool csv = false;
  bool csv_header = false;
};

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      o.mode = parse_mode(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      o.iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup-iters") == 0 && i + 1 < argc) {
      o.warmup_iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      o.threads = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--blocks-per-sm") == 0 && i + 1 < argc) {
      o.blocks_per_sm = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--stride-words") == 0 && i + 1 < argc) {
      o.stride_words = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--shared-words") == 0 && i + 1 < argc) {
      o.shared_words = std::strtoull(argv[++i], nullptr, 0);
    } else if (std::strcmp(argv[i], "--csv") == 0) {
      o.csv = true;
    } else if (std::strcmp(argv[i], "--csv-header") == 0) {
      o.csv_header = true;
    } else {
      std::fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
      std::exit(2);
    }
  }
  return o;
}

using KernelFn = void (*)(int, int, int, size_t, uint32_t*, unsigned long long*);

KernelFn kernel_for_mode(Mode mode) {
  return mode == Mode::kRead ? smem_bank_kernel<Mode::kRead> : smem_bank_kernel<Mode::kWrite>;
}

int main(int argc, char** argv) {
  Options o = parse_args(argc, argv);
  if (o.csv_header) {
    std::puts("mode,stride_words,requested_bytes,elapsed_cycles,bytes_per_cycle,per_sm_bytes_per_cycle,sm_count,blocks,blocks_per_sm,threads,iters,warmup_iters,unroll,shared_words,occupancy_blocks_per_sm");
    return 0;
  }
  if (o.iters <= 0 || o.warmup_iters < 0 || o.threads <= 0 || o.threads > 256 ||
      o.blocks_per_sm <= 0 || o.stride_words <= 0) {
    std::fprintf(stderr, "invalid launch parameters\n");
    return 2;
  }
  size_t shared_words = 1;
  while (shared_words < o.shared_words) shared_words <<= 1;
  const size_t required_words =
      size_t(o.threads - 1) * size_t(o.stride_words) + size_t(kUnroll);
  if (shared_words < required_words) {
    while (shared_words < required_words) {
      shared_words <<= 1;
    }
  }
  const size_t shared_bytes = shared_words * sizeof(uint32_t);

  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  KernelFn kernel = kernel_for_mode(o.mode);
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  int(shared_bytes)));
  int occupancy = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, kernel, o.threads, shared_bytes));
  if (o.blocks_per_sm > occupancy) {
    std::fprintf(stderr, "blocks-per-sm %d exceeds occupancy %d\n", o.blocks_per_sm, occupancy);
    return 2;
  }
  const int blocks = sm_count * o.blocks_per_sm;
  uint32_t* d_sink = nullptr;
  unsigned long long* d_cycles = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, size_t(blocks) * o.threads * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_cycles, size_t(blocks) * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_sink, 0, size_t(blocks) * o.threads * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_cycles, 0, size_t(blocks) * sizeof(unsigned long long)));

  kernel<<<blocks, o.threads, shared_bytes>>>(o.iters, o.warmup_iters, o.stride_words,
                                              shared_words - 1u, d_sink, d_cycles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unsigned long long> cycles;
  cycles.resize(static_cast<size_t>(blocks));
  CUDA_CHECK(cudaMemcpy(cycles.data(), d_cycles, cycles.size() * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  unsigned long long elapsed = 0;
  for (auto c : cycles) elapsed = std::max(elapsed, c);
  const unsigned long long ops =
      static_cast<unsigned long long>(blocks) * o.threads * o.iters * kUnroll;
  const unsigned long long requested = ops * kBytesPerOp;
  const double bpc = elapsed ? double(requested) / double(elapsed) : 0.0;

  if (o.csv) {
    std::printf("%s,%d,%llu,%llu,%.6f,%.6f,%d,%d,%d,%d,%d,%d,%d,%zu,%d\n",
                mode_name(o.mode), o.stride_words, requested, elapsed, bpc,
                bpc / sm_count, sm_count, blocks, o.blocks_per_sm, o.threads,
                o.iters, o.warmup_iters, kUnroll, shared_words, occupancy);
  } else {
    std::printf("mode=%s stride_words=%d requested_bytes=%llu elapsed_cycles=%llu bytes_per_cycle=%.6f\n",
                mode_name(o.mode), o.stride_words, requested, elapsed, bpc);
  }

  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_cycles));
  return 0;
}
