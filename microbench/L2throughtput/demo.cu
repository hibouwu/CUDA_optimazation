#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
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
  kReadSame,
  kReadUnique,
  kWriteUnique,
};

__device__ __forceinline__ uint4 ld_global_cg_u128(uint4 const* ptr) {
  return *ptr;
}

__device__ __forceinline__ uint4 ld_global_cg_u128_volatile(uint4 const* ptr) {
  uint4 v;
  asm volatile(
      "ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr)
      : "memory");
  return v;
}

__device__ __forceinline__ void st_global_cg_u128(uint4* ptr, uint4 v) {
  asm volatile(
      "st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
      :
      : "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
      : "memory");
}

__global__ void init_buffer_kernel(uint4* data, size_t elements) {
  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (; idx < elements; idx += stride) {
    uint32_t x = uint32_t(idx);
    data[idx] = make_uint4(0x9e3779b9u ^ x,
                           0x7f4a7c15u + x * 3u,
                           0x94d049bbu ^ (x << 1),
                           0x2545f491u + x * 17u);
  }
}

template <Mode kMode>
__global__ __launch_bounds__(256, 4)
void l2_throughput_kernel(uint4* buffer,
                          size_t element_mask,
                          int iters,
                          uint32_t* sink,
                          unsigned long long* cycles_out) {
  unsigned int tid = threadIdx.x;
  unsigned int global_tid = blockIdx.x * blockDim.x + tid;
  unsigned int total_threads = gridDim.x * blockDim.x;
  uint4 acc = make_uint4(global_tid + 1u,
                         global_tid * 3u + 5u,
                         global_tid * 7u + 11u,
                         global_tid * 13u + 17u);
  __shared__ unsigned long long block_start;

  __syncthreads();
  if (tid == 0) {
    block_start = clock64();
  }
  __syncthreads();

  if constexpr (kMode == Mode::kReadSame) {
    uint4 const* ptr = buffer;
    for (int i = 0; i < iters; ++i) {
      uint4 v0 = ld_global_cg_u128_volatile(ptr);
      uint4 v1 = ld_global_cg_u128_volatile(ptr);
      uint4 v2 = ld_global_cg_u128_volatile(ptr);
      uint4 v3 = ld_global_cg_u128_volatile(ptr);
      uint4 v4 = ld_global_cg_u128_volatile(ptr);
      uint4 v5 = ld_global_cg_u128_volatile(ptr);
      uint4 v6 = ld_global_cg_u128_volatile(ptr);
      uint4 v7 = ld_global_cg_u128_volatile(ptr);
      acc.x ^= v0.x + v1.x + v2.x + v3.x;
      acc.y += v4.y ^ v5.y ^ v6.y ^ v7.y ^ uint32_t(i);
      acc.z ^= v0.z + v2.z + v4.z + v6.z + acc.x;
      acc.w += v1.w ^ v3.w ^ v5.w ^ v7.w ^ acc.y;
    }
  } else if constexpr (kMode == Mode::kReadUnique) {
    size_t step = size_t(total_threads);
    size_t round_stride = step * size_t(kUnroll) + 1u;
    size_t idx0 = (size_t(global_tid) + step * 0u) & element_mask;
    size_t idx1 = (size_t(global_tid) + step * 1u) & element_mask;
    size_t idx2 = (size_t(global_tid) + step * 2u) & element_mask;
    size_t idx3 = (size_t(global_tid) + step * 3u) & element_mask;
    size_t idx4 = (size_t(global_tid) + step * 4u) & element_mask;
    size_t idx5 = (size_t(global_tid) + step * 5u) & element_mask;
    size_t idx6 = (size_t(global_tid) + step * 6u) & element_mask;
    size_t idx7 = (size_t(global_tid) + step * 7u) & element_mask;
    for (int i = 0; i < iters; ++i) {
      uint4 v0 = ld_global_cg_u128_volatile(buffer + idx0);
      uint4 v1 = ld_global_cg_u128_volatile(buffer + idx1);
      uint4 v2 = ld_global_cg_u128_volatile(buffer + idx2);
      uint4 v3 = ld_global_cg_u128_volatile(buffer + idx3);
      uint4 v4 = ld_global_cg_u128_volatile(buffer + idx4);
      uint4 v5 = ld_global_cg_u128_volatile(buffer + idx5);
      uint4 v6 = ld_global_cg_u128_volatile(buffer + idx6);
      uint4 v7 = ld_global_cg_u128_volatile(buffer + idx7);
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
    size_t step = size_t(total_threads);
    size_t round_stride = step * size_t(kUnroll) + 1u;
    size_t idx0 = (size_t(global_tid) + step * 0u) & element_mask;
    size_t idx1 = (size_t(global_tid) + step * 1u) & element_mask;
    size_t idx2 = (size_t(global_tid) + step * 2u) & element_mask;
    size_t idx3 = (size_t(global_tid) + step * 3u) & element_mask;
    size_t idx4 = (size_t(global_tid) + step * 4u) & element_mask;
    size_t idx5 = (size_t(global_tid) + step * 5u) & element_mask;
    size_t idx6 = (size_t(global_tid) + step * 6u) & element_mask;
    size_t idx7 = (size_t(global_tid) + step * 7u) & element_mask;
    for (int i = 0; i < iters; ++i) {
      uint4 v0 = make_uint4(acc.x + uint32_t(i), acc.y, acc.z ^ uint32_t(idx0), acc.w);
      uint4 v1 = make_uint4(acc.x + 1u, acc.y + uint32_t(i), acc.z ^ uint32_t(idx1), acc.w + 1u);
      uint4 v2 = make_uint4(acc.x + 2u, acc.y + 3u, acc.z ^ uint32_t(idx2), acc.w + uint32_t(i));
      uint4 v3 = make_uint4(acc.x + 5u, acc.y + 7u, acc.z ^ uint32_t(idx3), acc.w + 11u);
      uint4 v4 = make_uint4(acc.x + 13u, acc.y + 17u, acc.z ^ uint32_t(idx4), acc.w + 19u);
      uint4 v5 = make_uint4(acc.x + 23u, acc.y + 29u, acc.z ^ uint32_t(idx5), acc.w + 31u);
      uint4 v6 = make_uint4(acc.x + 37u, acc.y + 41u, acc.z ^ uint32_t(idx6), acc.w + 43u);
      uint4 v7 = make_uint4(acc.x + 47u, acc.y + 53u, acc.z ^ uint32_t(idx7), acc.w + 59u);
      st_global_cg_u128(buffer + idx0, v0);
      st_global_cg_u128(buffer + idx1, v1);
      st_global_cg_u128(buffer + idx2, v2);
      st_global_cg_u128(buffer + idx3, v3);
      st_global_cg_u128(buffer + idx4, v4);
      st_global_cg_u128(buffer + idx5, v5);
      st_global_cg_u128(buffer + idx6, v6);
      st_global_cg_u128(buffer + idx7, v7);
      acc.x += 0x9e3779b9u + uint32_t(i);
      acc.y ^= acc.x + uint32_t(global_tid);
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
  if (threadIdx.x == 0) {
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
    case Mode::kReadSame: return "read-same";
    case Mode::kReadUnique: return "read-unique";
    case Mode::kWriteUnique: return "write-unique";
  }
  return "unknown";
}

static Mode parse_mode(const char* text) {
  if (std::strcmp(text, "read-same") == 0) return Mode::kReadSame;
  if (std::strcmp(text, "read-unique") == 0) return Mode::kReadUnique;
  if (std::strcmp(text, "write-unique") == 0) return Mode::kWriteUnique;
  std::fprintf(stderr, "unknown mode: %s\n", text);
  std::exit(2);
}

struct Options {
  Mode mode = Mode::kReadUnique;
  int iters = 4096;
  int warmup_iters = 64;
  int blocks_per_sm = 4;
  int threads_per_block = 256;
  size_t working_set_bytes = 0;
  bool csv = false;
  bool csv_header = false;
};

static void usage(const char* argv0) {
  std::printf(
      "Usage:\n"
      "  %s [--mode read-same|read-unique|write-unique] [options]\n"
      "\n"
      "Options:\n"
      "  --iters N             Timed outer-loop iterations (default: 4096)\n"
      "  --warmup-iters N      Untimed warmup iterations (default: 64)\n"
      "  --blocks-per-sm N     CUDA blocks launched per SM (default: 4)\n"
      "  --threads N           Threads per block, max 256 (default: 256)\n"
      "  --bytes N             L2 working-set bytes, rounded down to power of two\n"
      "  --csv                 Print one CSV data row\n"
      "  --csv-header          Print the CSV header and exit\n",
      argv0);
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
      opt.threads_per_block = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
      opt.working_set_bytes = std::strtoull(argv[++i], nullptr, 0);
    } else if (std::strcmp(argv[i], "--csv") == 0) {
      opt.csv = true;
    } else if (std::strcmp(argv[i], "--csv-header") == 0) {
      opt.csv_header = true;
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      usage(argv[0]);
      std::exit(2);
    }
  }

  if (opt.iters <= 0 || opt.warmup_iters < 0 || opt.blocks_per_sm <= 0 ||
      opt.threads_per_block <= 0 || opt.threads_per_block > 256) {
    std::fprintf(stderr, "invalid launch or iteration option\n");
    std::exit(2);
  }
  return opt;
}

template <Mode kMode>
static void launch_l2_kernel(uint4* buffer,
                             size_t element_mask,
                             int iters,
                             int blocks,
                             int threads,
                             uint32_t* sink,
                             unsigned long long* cycles) {
  l2_throughput_kernel<kMode><<<blocks, threads>>>(
      buffer, element_mask, iters, sink, cycles);
  CUDA_CHECK(cudaGetLastError());
}

static void run_l2_kernel(Mode mode,
                          uint4* buffer,
                          size_t element_mask,
                          int iters,
                          int blocks,
                          int threads,
                          uint32_t* sink,
                          unsigned long long* cycles) {
  switch (mode) {
    case Mode::kReadSame:
      launch_l2_kernel<Mode::kReadSame>(
          buffer, element_mask, iters, blocks, threads, sink, cycles);
      break;
    case Mode::kReadUnique:
      launch_l2_kernel<Mode::kReadUnique>(
          buffer, element_mask, iters, blocks, threads, sink, cycles);
      break;
    case Mode::kWriteUnique:
      launch_l2_kernel<Mode::kWriteUnique>(
          buffer, element_mask, iters, blocks, threads, sink, cycles);
      break;
  }
}

static int occupancy_blocks_per_sm(Mode mode, int threads) {
  int blocks = 0;
  switch (mode) {
    case Mode::kReadSame:
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, l2_throughput_kernel<Mode::kReadSame>, threads, 0));
      break;
    case Mode::kReadUnique:
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, l2_throughput_kernel<Mode::kReadUnique>, threads, 0));
      break;
    case Mode::kWriteUnique:
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, l2_throughput_kernel<Mode::kWriteUnique>, threads, 0));
      break;
  }
  return blocks;
}

static void print_csv_header() {
  std::puts("mode,requested_bytes,elapsed_cycles,bytes_per_cycle,ops,bytes_per_op,"
            "sm_count,blocks,blocks_per_sm,threads_per_block,iters,unroll,"
            "occupancy_blocks_per_sm,l2_cache_bytes,working_set_bytes,"
            "touched_footprint_bytes,index_stride_elements,stream_period_iters,"
            "requested_to_working_set_ratio");
}

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  if (opt.csv_header) {
    print_csv_header();
    return 0;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  size_t default_bytes = prop.l2CacheSize > 0
      ? floor_power_of_two(size_t(prop.l2CacheSize) / 2)
      : size_t(8) * 1024 * 1024;
  default_bytes = std::max(default_bytes, size_t(1) * 1024 * 1024);
  size_t working_set_bytes = opt.working_set_bytes ? opt.working_set_bytes
                                                   : default_bytes;
  working_set_bytes = floor_power_of_two(working_set_bytes);
  working_set_bytes = std::max(working_set_bytes, size_t(kBytesPerOp));

  size_t elements = working_set_bytes / sizeof(uint4);
  elements = floor_power_of_two(elements);
  if (elements == 0) elements = 1;
  working_set_bytes = elements * sizeof(uint4);
  size_t element_mask = elements - 1;

  int threads = opt.threads_per_block;
  int occupancy_limit = occupancy_blocks_per_sm(opt.mode, threads);
  if (opt.blocks_per_sm > occupancy_limit) {
    std::fprintf(stderr,
                 "blocks_per_sm=%d exceeds occupancy limit %d for mode=%s "
                 "and threads=%d; lower --blocks-per-sm or --threads\n",
                 opt.blocks_per_sm, occupancy_limit, mode_name(opt.mode),
                 threads);
    return 2;
  }

  int blocks = prop.multiProcessorCount * opt.blocks_per_sm;
  size_t active_threads = size_t(blocks) * size_t(threads);

  uint4* d_buffer = nullptr;
  uint32_t* d_sink = nullptr;
  unsigned long long* d_cycles = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buffer, working_set_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, active_threads * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_cycles, size_t(blocks) * sizeof(unsigned long long)));

  int init_blocks = std::min(4096, std::max(1, blocks * 4));
  init_buffer_kernel<<<init_blocks, 256>>>(d_buffer, elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  if (opt.warmup_iters > 0) {
    run_l2_kernel(opt.mode, d_buffer, element_mask, opt.warmup_iters, blocks,
                  threads, d_sink, d_cycles);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaMemset(d_cycles, 0, size_t(blocks) * sizeof(unsigned long long)));
  run_l2_kernel(opt.mode, d_buffer, element_mask, opt.iters, blocks, threads,
                d_sink, d_cycles);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unsigned long long> h_cycles(blocks);
  CUDA_CHECK(cudaMemcpy(h_cycles.data(), d_cycles,
                        h_cycles.size() * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  unsigned long long elapsed_cycles =
      *std::max_element(h_cycles.begin(), h_cycles.end());

  long double ops = static_cast<long double>(active_threads) *
                    static_cast<long double>(opt.iters) *
                    static_cast<long double>(kUnroll);
  long double requested_bytes = ops * static_cast<long double>(kBytesPerOp);
  long double bytes_per_cycle =
      elapsed_cycles ? requested_bytes / static_cast<long double>(elapsed_cycles)
                     : 0.0L;
  size_t touched_footprint = opt.mode == Mode::kReadSame
      ? size_t(kBytesPerOp)
      : working_set_bytes;
  size_t index_stride_elements = opt.mode == Mode::kReadSame
      ? size_t(0)
      : active_threads * size_t(kUnroll) + 1u;
  size_t stream_period_iters = index_stride_elements == 0
      ? size_t(0)
      : elements / std::gcd(elements, index_stride_elements);
  long double requested_to_working_set_ratio =
      requested_bytes / static_cast<long double>(working_set_bytes);

  if (opt.csv) {
    std::printf("%s,%.0Lf,%llu,%.6Lf,%.0Lf,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%zu,%zu,%.6Lf\n",
                mode_name(opt.mode),
                requested_bytes,
                elapsed_cycles,
                bytes_per_cycle,
                ops,
                kBytesPerOp,
                prop.multiProcessorCount,
                blocks,
                opt.blocks_per_sm,
                threads,
                opt.iters,
                kUnroll,
                occupancy_limit,
                prop.l2CacheSize,
                working_set_bytes,
                touched_footprint,
                index_stride_elements,
                stream_period_iters,
                requested_to_working_set_ratio);
  } else {
    std::printf("device=%s\n", prop.name);
    std::printf("sm_count=%d\n", prop.multiProcessorCount);
    std::printf("l2_cache_bytes=%d\n", prop.l2CacheSize);
    std::printf("mode=%s\n", mode_name(opt.mode));
    std::printf("blocks=%d\n", blocks);
    std::printf("blocks_per_sm=%d\n", opt.blocks_per_sm);
    std::printf("threads_per_block=%d\n", threads);
    std::printf("occupancy_blocks_per_sm=%d\n", occupancy_limit);
    std::printf("iters=%d\n", opt.iters);
    std::printf("unroll=%d\n", kUnroll);
    std::printf("bytes_per_op=%d\n", kBytesPerOp);
    std::printf("working_set_bytes=%zu\n", working_set_bytes);
    std::printf("touched_footprint_bytes=%zu\n", touched_footprint);
    std::printf("index_stride_elements=%zu\n", index_stride_elements);
    std::printf("stream_period_iters=%zu\n", stream_period_iters);
    std::printf("requested_to_working_set_ratio=%.6Lf\n",
                requested_to_working_set_ratio);
    std::printf("requested_bytes=%.0Lf\n", requested_bytes);
    std::printf("elapsed_cycles=%llu\n", elapsed_cycles);
    std::printf("bytes_per_cycle=%.6Lf\n", bytes_per_cycle);
    std::printf("per_sm_bytes_per_cycle=%.6Lf\n",
                bytes_per_cycle / static_cast<long double>(prop.multiProcessorCount));
  }

  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_buffer));
  return 0;
}
