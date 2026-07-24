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
static constexpr int kBytesPerOp = 16;

enum class Mode {
  kReadCa,
  kReadCg,
  kWriteWb,
  kWriteCg,
};

__device__ __forceinline__ uint4 ld_global_ca_u128(uint4 const* ptr) {
  uint4 v;
  asm volatile(
      "ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr)
      : "memory");
  return v;
}

__device__ __forceinline__ uint4 ld_global_cg_u128(uint4 const* ptr) {
  uint4 v;
  asm volatile(
      "ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr)
      : "memory");
  return v;
}

__device__ __forceinline__ void st_global_wb_u128(uint4* ptr, uint4 v) {
  asm volatile(
      "st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};"
      :
      : "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
      : "memory");
}

__device__ __forceinline__ void st_global_cg_u128(uint4* ptr, uint4 v) {
  asm volatile(
      "st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
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
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (; idx < elements; idx += stride) {
    data[idx] = make_pattern(uint32_t(idx), 0u);
  }
}

template <Mode kMode>
__global__ __launch_bounds__(256, 1)
void l1_bandwidth_kernel(uint4* buffer,
                         size_t elements_per_cta,
                         size_t element_mask,
                         int iters,
                         int warmup_rounds,
                         uint32_t* sink,
                         unsigned long long* cycles_out) {
  const unsigned int tid = threadIdx.x;
  const unsigned int block_linear = blockIdx.x;
  uint4* base = buffer + size_t(block_linear) * elements_per_cta;
  uint4 acc = make_pattern(block_linear * blockDim.x + tid, 1u);

  const size_t step = size_t(blockDim.x);
  const size_t round_stride = step * size_t(kUnroll) + 1u;

  if constexpr (kMode == Mode::kReadCa) {
    for (int r = 0; r < warmup_rounds; ++r) {
      for (size_t idx = tid; idx < elements_per_cta; idx += blockDim.x) {
        uint4 v = ld_global_ca_u128(base + idx);
        acc.x ^= v.x;
      }
    }
  } else if constexpr (kMode == Mode::kReadCg) {
    for (int r = 0; r < warmup_rounds; ++r) {
      for (size_t idx = tid; idx < elements_per_cta; idx += blockDim.x) {
        uint4 v = ld_global_cg_u128(base + idx);
        acc.x ^= v.x;
      }
    }
  }

  size_t idx0 = (size_t(tid) + step * 0u) & element_mask;
  size_t idx1 = (size_t(tid) + step * 1u) & element_mask;
  size_t idx2 = (size_t(tid) + step * 2u) & element_mask;
  size_t idx3 = (size_t(tid) + step * 3u) & element_mask;
  size_t idx4 = (size_t(tid) + step * 4u) & element_mask;
  size_t idx5 = (size_t(tid) + step * 5u) & element_mask;
  size_t idx6 = (size_t(tid) + step * 6u) & element_mask;
  size_t idx7 = (size_t(tid) + step * 7u) & element_mask;

  __shared__ unsigned long long block_start;
  __syncthreads();
  if (tid == 0) {
    block_start = clock64();
  }
  __syncthreads();

  if constexpr (kMode == Mode::kReadCa || kMode == Mode::kReadCg) {
    for (int i = 0; i < iters; ++i) {
      uint4 v0, v1, v2, v3, v4, v5, v6, v7;
      if constexpr (kMode == Mode::kReadCa) {
        v0 = ld_global_ca_u128(base + idx0);
        v1 = ld_global_ca_u128(base + idx1);
        v2 = ld_global_ca_u128(base + idx2);
        v3 = ld_global_ca_u128(base + idx3);
        v4 = ld_global_ca_u128(base + idx4);
        v5 = ld_global_ca_u128(base + idx5);
        v6 = ld_global_ca_u128(base + idx6);
        v7 = ld_global_ca_u128(base + idx7);
      } else {
        v0 = ld_global_cg_u128(base + idx0);
        v1 = ld_global_cg_u128(base + idx1);
        v2 = ld_global_cg_u128(base + idx2);
        v3 = ld_global_cg_u128(base + idx3);
        v4 = ld_global_cg_u128(base + idx4);
        v5 = ld_global_cg_u128(base + idx5);
        v6 = ld_global_cg_u128(base + idx6);
        v7 = ld_global_cg_u128(base + idx7);
      }
      acc.x ^= v0.x + v1.y + v2.z + v3.w + v4.x + v5.y + v6.z + v7.w;
      acc.y += v0.y ^ v1.z ^ v2.w ^ v3.x ^ v4.y ^ v5.z ^ v6.w ^ v7.x ^ uint32_t(i);
      acc.z ^= v0.z + v1.w + v2.x + v3.y + v4.z + v5.w + v6.x + v7.y + acc.x;
      acc.w += v0.w ^ v1.x ^ v2.y ^ v3.z ^ v4.w ^ v5.x ^ v6.y ^ v7.z ^ acc.y;
      idx0 = (idx0 + round_stride) & element_mask;
      idx1 = (idx1 + round_stride) & element_mask;
      idx2 = (idx2 + round_stride) & element_mask;
      idx3 = (idx3 + round_stride) & element_mask;
      idx4 = (idx4 + round_stride) & element_mask;
      idx5 = (idx5 + round_stride) & element_mask;
      idx6 = (idx6 + round_stride) & element_mask;
      idx7 = (idx7 + round_stride) & element_mask;
    }
  } else {
    for (int i = 0; i < iters; ++i) {
      uint4 v0 = make_pattern(acc.x + uint32_t(idx0), uint32_t(i));
      uint4 v1 = make_pattern(acc.y + uint32_t(idx1), uint32_t(i) + 1u);
      uint4 v2 = make_pattern(acc.z + uint32_t(idx2), uint32_t(i) + 2u);
      uint4 v3 = make_pattern(acc.w + uint32_t(idx3), uint32_t(i) + 3u);
      uint4 v4 = make_pattern(acc.x + uint32_t(idx4), uint32_t(i) + 4u);
      uint4 v5 = make_pattern(acc.y + uint32_t(idx5), uint32_t(i) + 5u);
      uint4 v6 = make_pattern(acc.z + uint32_t(idx6), uint32_t(i) + 6u);
      uint4 v7 = make_pattern(acc.w + uint32_t(idx7), uint32_t(i) + 7u);
      if constexpr (kMode == Mode::kWriteWb) {
        st_global_wb_u128(base + idx0, v0);
        st_global_wb_u128(base + idx1, v1);
        st_global_wb_u128(base + idx2, v2);
        st_global_wb_u128(base + idx3, v3);
        st_global_wb_u128(base + idx4, v4);
        st_global_wb_u128(base + idx5, v5);
        st_global_wb_u128(base + idx6, v6);
        st_global_wb_u128(base + idx7, v7);
      } else {
        st_global_cg_u128(base + idx0, v0);
        st_global_cg_u128(base + idx1, v1);
        st_global_cg_u128(base + idx2, v2);
        st_global_cg_u128(base + idx3, v3);
        st_global_cg_u128(base + idx4, v4);
        st_global_cg_u128(base + idx5, v5);
        st_global_cg_u128(base + idx6, v6);
        st_global_cg_u128(base + idx7, v7);
      }
      acc.x += 0x9e3779b9u + uint32_t(i);
      acc.y ^= acc.x + uint32_t(tid);
      acc.z += acc.y ^ uint32_t(idx0 + idx7);
      acc.w ^= acc.z + uint32_t(i);
      idx0 = (idx0 + round_stride) & element_mask;
      idx1 = (idx1 + round_stride) & element_mask;
      idx2 = (idx2 + round_stride) & element_mask;
      idx3 = (idx3 + round_stride) & element_mask;
      idx4 = (idx4 + round_stride) & element_mask;
      idx5 = (idx5 + round_stride) & element_mask;
      idx6 = (idx6 + round_stride) & element_mask;
      idx7 = (idx7 + round_stride) & element_mask;
    }
    __threadfence();
  }

  __syncthreads();
  if (tid == 0) {
    unsigned long long stop = clock64();
    cycles_out[blockIdx.x] = stop - block_start;
  }
  sink[block_linear * blockDim.x + tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

static size_t floor_power_of_two(size_t x) {
  if (x == 0) return 0;
  size_t p = 1;
  while (p <= x / 2) p <<= 1;
  return p;
}

static const char* mode_name(Mode mode) {
  switch (mode) {
    case Mode::kReadCa: return "read-ca";
    case Mode::kReadCg: return "read-cg";
    case Mode::kWriteWb: return "write-wb";
    case Mode::kWriteCg: return "write-cg";
  }
  return "unknown";
}

static Mode parse_mode(const char* text) {
  if (std::strcmp(text, "read-ca") == 0) return Mode::kReadCa;
  if (std::strcmp(text, "read-cg") == 0) return Mode::kReadCg;
  if (std::strcmp(text, "write-wb") == 0) return Mode::kWriteWb;
  if (std::strcmp(text, "write-cg") == 0) return Mode::kWriteCg;
  std::fprintf(stderr, "unknown mode: %s\n", text);
  std::exit(2);
}

struct Options {
  Mode mode = Mode::kReadCa;
  int iters = 4096;
  int warmup_rounds = 2;
  int threads = 256;
  int blocks = 0;
  size_t bytes_per_cta = 16384;
  bool csv = false;
  bool csv_header = false;
};

static void usage(const char* argv0) {
  std::printf(
      "Usage: %s --mode read-ca|read-cg|write-wb|write-cg [options]\n"
      "  --iters N\n"
      "  --warmup-rounds N\n"
      "  --threads N\n"
      "  --blocks N          0 = SM count\n"
      "  --bytes-per-cta N   rounded down to power of two\n"
      "  --csv\n"
      "  --csv-header\n",
      argv0);
}

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      opt.mode = parse_mode(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      opt.iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup-rounds") == 0 && i + 1 < argc) {
      opt.warmup_rounds = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      opt.threads = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
      opt.blocks = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--bytes-per-cta") == 0 && i + 1 < argc) {
      opt.bytes_per_cta = std::strtoull(argv[++i], nullptr, 0);
    } else if (std::strcmp(argv[i], "--csv") == 0) {
      opt.csv = true;
    } else if (std::strcmp(argv[i], "--csv-header") == 0) {
      opt.csv_header = true;
    } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(2);
    }
  }
  return opt;
}

using KernelFn = void (*)(uint4*, size_t, size_t, int, int, uint32_t*, unsigned long long*);

static KernelFn kernel_for_mode(Mode mode) {
  switch (mode) {
    case Mode::kReadCa: return l1_bandwidth_kernel<Mode::kReadCa>;
    case Mode::kReadCg: return l1_bandwidth_kernel<Mode::kReadCg>;
    case Mode::kWriteWb: return l1_bandwidth_kernel<Mode::kWriteWb>;
    case Mode::kWriteCg: return l1_bandwidth_kernel<Mode::kWriteCg>;
  }
  return nullptr;
}

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  if (opt.csv_header) {
    std::puts("mode,requested_bytes,elapsed_cycles,bytes_per_cycle,per_sm_bytes_per_cycle,ops,bytes_per_op,sm_count,blocks,threads,iters,unroll,warmup_rounds,bytes_per_cta,elements_per_cta,index_stride_elements,stream_period_iters");
    return 0;
  }
  if (opt.iters <= 0 || opt.threads <= 0 || opt.threads > 256 ||
      opt.warmup_rounds < 0) {
    std::fprintf(stderr, "invalid options\n");
    return 2;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int blocks = opt.blocks > 0 ? opt.blocks : prop.multiProcessorCount;

  size_t bytes_per_cta = floor_power_of_two(opt.bytes_per_cta);
  bytes_per_cta = std::max(bytes_per_cta, size_t(opt.threads * kUnroll * kBytesPerOp));
  bytes_per_cta = floor_power_of_two(bytes_per_cta);
  const size_t elements_per_cta = bytes_per_cta / sizeof(uint4);
  if (elements_per_cta == 0 || (elements_per_cta & (elements_per_cta - 1)) != 0) {
    std::fprintf(stderr, "bytes-per-cta must map to power-of-two uint4 elements\n");
    return 2;
  }

  uint4* d_buffer = nullptr;
  uint32_t* d_sink = nullptr;
  unsigned long long* d_cycles = nullptr;
  const size_t total_elements = size_t(blocks) * elements_per_cta;
  CUDA_CHECK(cudaMalloc(&d_buffer, total_elements * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink, size_t(blocks) * size_t(opt.threads) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_cycles, size_t(blocks) * sizeof(unsigned long long)));

  init_kernel<<<std::min(1024, blocks * 4), 256>>>(d_buffer, total_elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  KernelFn kernel = kernel_for_mode(opt.mode);
  kernel<<<blocks, opt.threads>>>(d_buffer, elements_per_cta, elements_per_cta - 1u,
                                  opt.iters, opt.warmup_rounds, d_sink, d_cycles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unsigned long long> cycles(static_cast<size_t>(blocks));
  CUDA_CHECK(cudaMemcpy(cycles.data(), d_cycles, cycles.size() * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  unsigned long long elapsed_cycles = 0;
  for (auto c : cycles) elapsed_cycles = std::max(elapsed_cycles, c);

  const unsigned long long ops =
      static_cast<unsigned long long>(blocks) *
      static_cast<unsigned long long>(opt.threads) *
      static_cast<unsigned long long>(opt.iters) *
      static_cast<unsigned long long>(kUnroll);
  const unsigned long long requested_bytes = ops * kBytesPerOp;
  const double bpc = double(requested_bytes) / double(elapsed_cycles);
  const double per_sm = bpc / double(blocks);
  const size_t index_stride = size_t(opt.threads) * size_t(kUnroll) + 1u;

  size_t period = elements_per_cta;
  size_t stride_mod = index_stride & (elements_per_cta - 1u);
  if (stride_mod != 0) {
    size_t a = elements_per_cta, b = stride_mod;
    while (b != 0) {
      size_t t = a % b;
      a = b;
      b = t;
    }
    period = elements_per_cta / a;
  }

  if (opt.csv) {
    std::printf("%s,%llu,%llu,%.6f,%.6f,%llu,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%zu,%zu\n",
                mode_name(opt.mode), requested_bytes, elapsed_cycles, bpc, per_sm,
                ops, kBytesPerOp, prop.multiProcessorCount, blocks, opt.threads,
                opt.iters, kUnroll, opt.warmup_rounds, bytes_per_cta,
                elements_per_cta, index_stride, period);
  } else {
    std::printf("mode=%s requested_bytes=%llu elapsed_cycles=%llu bytes_per_cycle=%.6f per_sm_bytes_per_cycle=%.6f sm_count=%d blocks=%d threads=%d iters=%d bytes_per_cta=%zu\n",
                mode_name(opt.mode), requested_bytes, elapsed_cycles, bpc, per_sm,
                prop.multiProcessorCount, blocks, opt.threads, opt.iters,
                bytes_per_cta);
  }

  CUDA_CHECK(cudaFree(d_buffer));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_cycles));
  return 0;
}
