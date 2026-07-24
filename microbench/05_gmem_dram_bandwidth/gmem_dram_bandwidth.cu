#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do {                                             \
  cudaError_t err__ = (call);                                             \
  if (err__ != cudaSuccess) {                                             \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                 cudaGetErrorString(err__));                              \
    std::exit(1);                                                         \
  }                                                                       \
} while (0)

static constexpr int kUnroll = 8;

enum class Mode {
  kRead,
  kWrite,
  kCopy,
};

__device__ __forceinline__ uint4 ld_cg(uint4 const* ptr) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr)
               : "memory");
  return v;
}

__device__ __forceinline__ void st_cg(uint4* ptr, uint4 v) {
  asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

__device__ __forceinline__ uint4 make_pattern(uint32_t x, uint32_t y) {
  return make_uint4(0x9e3779b9u ^ x,
                    0x7f4a7c15u + x * 3u + y,
                    0x94d049bbu ^ (x << 1) ^ (y * 17u),
                    0x2545f491u + x * 17u + y * 13u);
}

__global__ void init_kernel(uint4* data, size_t elements) {
  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(blockDim.x) * gridDim.x;
  for (; idx < elements; idx += stride) {
    data[idx] = make_pattern(uint32_t(idx), 0u);
  }
}

template <Mode kMode>
__global__ __launch_bounds__(256, 4)
void gmem_kernel(uint4* src,
                 uint4* dst,
                 size_t element_mask,
                 int iters,
                 uint32_t* sink,
                 unsigned long long* cycles_out) {
  const unsigned int tid = threadIdx.x;
  const unsigned int global_tid = blockIdx.x * blockDim.x + tid;
  const unsigned int total_threads = gridDim.x * blockDim.x;
  const size_t step = size_t(total_threads);
  const size_t round_stride = step * size_t(kUnroll) + 1u;
  size_t idx0 = (size_t(global_tid) + step * 0u) & element_mask;
  size_t idx1 = (size_t(global_tid) + step * 1u) & element_mask;
  size_t idx2 = (size_t(global_tid) + step * 2u) & element_mask;
  size_t idx3 = (size_t(global_tid) + step * 3u) & element_mask;
  size_t idx4 = (size_t(global_tid) + step * 4u) & element_mask;
  size_t idx5 = (size_t(global_tid) + step * 5u) & element_mask;
  size_t idx6 = (size_t(global_tid) + step * 6u) & element_mask;
  size_t idx7 = (size_t(global_tid) + step * 7u) & element_mask;
  uint4 acc = make_pattern(global_tid, 1u);
  __shared__ unsigned long long block_start;

  __syncthreads();
  if (tid == 0) block_start = clock64();
  __syncthreads();

  for (int i = 0; i < iters; ++i) {
    if constexpr (kMode == Mode::kRead) {
      uint4 v0 = ld_cg(src + idx0);
      uint4 v1 = ld_cg(src + idx1);
      uint4 v2 = ld_cg(src + idx2);
      uint4 v3 = ld_cg(src + idx3);
      uint4 v4 = ld_cg(src + idx4);
      uint4 v5 = ld_cg(src + idx5);
      uint4 v6 = ld_cg(src + idx6);
      uint4 v7 = ld_cg(src + idx7);
      acc.x ^= v0.x + v0.y + v0.z + v0.w + v1.x + v1.y + v1.z + v1.w;
      acc.y += v2.x ^ v2.y ^ v2.z ^ v2.w ^ v3.x ^ v3.y ^ v3.z ^ v3.w ^ uint32_t(i);
      acc.z ^= v4.x + v4.y + v4.z + v4.w + v5.x + v5.y + v5.z + v5.w + acc.x;
      acc.w += v6.x ^ v6.y ^ v6.z ^ v6.w ^ v7.x ^ v7.y ^ v7.z ^ v7.w ^ acc.y;
    } else if constexpr (kMode == Mode::kWrite) {
      st_cg(dst + idx0, make_pattern(acc.x + uint32_t(idx0), uint32_t(i)));
      st_cg(dst + idx1, make_pattern(acc.y + uint32_t(idx1), uint32_t(i) + 1u));
      st_cg(dst + idx2, make_pattern(acc.z + uint32_t(idx2), uint32_t(i) + 2u));
      st_cg(dst + idx3, make_pattern(acc.w + uint32_t(idx3), uint32_t(i) + 3u));
      st_cg(dst + idx4, make_pattern(acc.x + uint32_t(idx4), uint32_t(i) + 4u));
      st_cg(dst + idx5, make_pattern(acc.y + uint32_t(idx5), uint32_t(i) + 5u));
      st_cg(dst + idx6, make_pattern(acc.z + uint32_t(idx6), uint32_t(i) + 6u));
      st_cg(dst + idx7, make_pattern(acc.w + uint32_t(idx7), uint32_t(i) + 7u));
      acc.x += uint32_t(i) + 1u;
      acc.y ^= acc.x;
      acc.z += acc.y;
      acc.w ^= acc.z;
    } else {
      uint4 v0 = ld_cg(src + idx0);
      uint4 v1 = ld_cg(src + idx1);
      uint4 v2 = ld_cg(src + idx2);
      uint4 v3 = ld_cg(src + idx3);
      uint4 v4 = ld_cg(src + idx4);
      uint4 v5 = ld_cg(src + idx5);
      uint4 v6 = ld_cg(src + idx6);
      uint4 v7 = ld_cg(src + idx7);
      st_cg(dst + idx0, v0);
      st_cg(dst + idx1, v1);
      st_cg(dst + idx2, v2);
      st_cg(dst + idx3, v3);
      st_cg(dst + idx4, v4);
      st_cg(dst + idx5, v5);
      st_cg(dst + idx6, v6);
      st_cg(dst + idx7, v7);
      acc.x ^= v0.x + v4.x;
      acc.y += v1.y ^ v5.y;
      acc.z ^= v2.z + v6.z;
      acc.w += v3.w ^ v7.w;
    }
    idx0 = (idx0 + round_stride) & element_mask;
    idx1 = (idx1 + round_stride) & element_mask;
    idx2 = (idx2 + round_stride) & element_mask;
    idx3 = (idx3 + round_stride) & element_mask;
    idx4 = (idx4 + round_stride) & element_mask;
    idx5 = (idx5 + round_stride) & element_mask;
    idx6 = (idx6 + round_stride) & element_mask;
    idx7 = (idx7 + round_stride) & element_mask;
  }
  if constexpr (kMode == Mode::kWrite || kMode == Mode::kCopy) {
    __threadfence();
  }

  __syncthreads();
  if (tid == 0) {
    unsigned long long stop = clock64();
    cycles_out[blockIdx.x] = stop - block_start;
  }
  sink[global_tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

static size_t floor_power_of_two(size_t x) {
  if (x == 0) return 0;
  size_t p = 1;
  while (p <= x / 2) p <<= 1;
  return p;
}

static const char* mode_name(Mode mode) {
  switch (mode) {
    case Mode::kRead: return "read-stream";
    case Mode::kWrite: return "write-stream";
    case Mode::kCopy: return "copy-stream";
  }
  return "unknown";
}

static Mode parse_mode(const char* text) {
  if (std::strcmp(text, "read-stream") == 0) return Mode::kRead;
  if (std::strcmp(text, "write-stream") == 0) return Mode::kWrite;
  if (std::strcmp(text, "copy-stream") == 0) return Mode::kCopy;
  std::fprintf(stderr, "unknown mode: %s\n", text);
  std::exit(2);
}

struct Options {
  Mode mode = Mode::kRead;
  int iters = 4096;
  int warmup_iters = 32;
  int blocks_per_sm = 4;
  int threads = 256;
  size_t bytes = 268435456ull;
  bool csv = false;
  bool csv_header = false;
};

static void usage(const char* argv0) {
  std::printf("Usage: %s --mode read-stream|write-stream|copy-stream [options]\n", argv0);
}

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      opt.mode = parse_mode(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      opt.iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup-iters") == 0 && i + 1 < argc) {
      opt.warmup_iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--blocks-per-sm") == 0 && i + 1 < argc) {
      opt.blocks_per_sm = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      opt.threads = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
      opt.bytes = std::strtoull(argv[++i], nullptr, 0);
    } else if (std::strcmp(argv[i], "--csv") == 0) {
      opt.csv = true;
    } else if (std::strcmp(argv[i], "--csv-header") == 0) {
      opt.csv_header = true;
    } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown option: %s\n", argv[i]);
      return opt;
    }
  }
  return opt;
}

using KernelFn = void (*)(uint4*, uint4*, size_t, int, uint32_t*, unsigned long long*);

static KernelFn kernel_for_mode(Mode mode) {
  switch (mode) {
    case Mode::kRead: return gmem_kernel<Mode::kRead>;
    case Mode::kWrite: return gmem_kernel<Mode::kWrite>;
    case Mode::kCopy: return gmem_kernel<Mode::kCopy>;
  }
  return nullptr;
}

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  if (opt.csv_header) {
    std::puts("mode,requested_bytes,elapsed_cycles,bytes_per_cycle,per_sm_bytes_per_cycle,ops,bytes_per_op,sm_count,blocks,blocks_per_sm,threads,iters,unroll,working_set_bytes,working_set_mib,requested_to_working_set_ratio,index_stride_elements,stream_period_iters,occupancy_blocks_per_sm");
    return 0;
  }
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  size_t bytes = floor_power_of_two(opt.bytes);
  bytes = std::max(bytes, size_t(64) * 1024 * 1024);
  size_t elements = bytes / sizeof(uint4);
  if ((elements & (elements - 1)) != 0) {
    std::fprintf(stderr, "working-set elements must be power of two\n");
    return 2;
  }
  int blocks = prop.multiProcessorCount * opt.blocks_per_sm;
  KernelFn kernel = kernel_for_mode(opt.mode);
  int occupancy = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, opt.threads, 0));
  if (opt.blocks_per_sm > occupancy) {
    std::fprintf(stderr, "blocks_per_sm=%d exceeds occupancy=%d\n", opt.blocks_per_sm, occupancy);
    return 2;
  }

  uint4* d_src = nullptr;
  uint4* d_dst = nullptr;
  uint32_t* d_sink = nullptr;
  unsigned long long* d_cycles = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, elements * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_dst, elements * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink, size_t(blocks) * size_t(opt.threads) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_cycles, size_t(blocks) * sizeof(unsigned long long)));
  init_kernel<<<std::min(4096, blocks * 16), 256>>>(d_src, elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemset(d_dst, 0, elements * sizeof(uint4)));
  CUDA_CHECK(cudaDeviceSynchronize());

  if (opt.warmup_iters > 0) {
    kernel<<<blocks, opt.threads>>>(d_src, d_dst, elements - 1u, opt.warmup_iters, d_sink, d_cycles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  kernel<<<blocks, opt.threads>>>(d_src, d_dst, elements - 1u, opt.iters, d_sink, d_cycles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unsigned long long> cycles(static_cast<size_t>(blocks));
  CUDA_CHECK(cudaMemcpy(cycles.data(), d_cycles, cycles.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  unsigned long long elapsed = 0;
  for (auto c : cycles) elapsed = std::max(elapsed, c);
  unsigned long long ops = static_cast<unsigned long long>(blocks) * opt.threads * opt.iters * kUnroll;
  int bytes_per_op = (opt.mode == Mode::kCopy) ? 32 : 16;
  unsigned long long requested = ops * bytes_per_op;
  double bpc = double(requested) / double(elapsed);
  size_t stride = size_t(blocks) * size_t(opt.threads) * size_t(kUnroll) + 1u;
  size_t period = elements;
  size_t m = stride & (elements - 1u);
  if (m != 0) {
    size_t a = elements, b = m;
    while (b) {
      size_t t = a % b;
      a = b;
      b = t;
    }
    period = elements / a;
  }

  if (opt.csv) {
    std::printf("%s,%llu,%llu,%.6f,%.6f,%llu,%d,%d,%d,%d,%d,%d,%d,%zu,%.3f,%.6f,%zu,%zu,%d\n",
                mode_name(opt.mode), requested, elapsed, bpc, bpc / prop.multiProcessorCount,
                ops, bytes_per_op, prop.multiProcessorCount, blocks, opt.blocks_per_sm,
                opt.threads, opt.iters, kUnroll, bytes, double(bytes) / 1048576.0,
                double(requested) / double(bytes), stride, period, occupancy);
  } else {
    std::printf("mode=%s requested_bytes=%llu elapsed_cycles=%llu bytes_per_cycle=%.6f working_set_mib=%.3f requested_to_working_set_ratio=%.3f\n",
                mode_name(opt.mode), requested, elapsed, bpc, double(bytes) / 1048576.0,
                double(requested) / double(bytes));
  }

  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_cycles));
  return 0;
}
