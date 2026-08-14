#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t error__ = (call);                                            \
    if (error__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(error__));                             \
      std::exit(3);                                                          \
    }                                                                        \
  } while (0)

namespace {

constexpr int kUnroll = 8;
constexpr int kBytesPerOperation = 16;

enum class Direction { kRead, kWrite };
enum class Residency { kL2, kHbm };

__device__ __forceinline__ unsigned long long global_nanoseconds() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ uint4 load_cg(const uint4* pointer) {
  return __ldcg(pointer);
}

__device__ __forceinline__ void store_cg(uint4* pointer, uint4 value) {
  asm volatile(
      "st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
      :
      : "l"(pointer), "r"(value.x), "r"(value.y), "r"(value.z),
        "r"(value.w)
      : "memory");
}

__device__ __forceinline__ uint4 pattern(unsigned index, unsigned iteration) {
  return make_uint4(0x9e3779b9u ^ index,
                    0x7f4a7c15u + index * 3u + iteration,
                    0x94d049bbu ^ (index << 1) ^ iteration * 17u,
                    0x2545f491u + index * 17u + iteration * 13u);
}

__global__ void initialize(uint4* data, size_t elements) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = size_t(blockDim.x) * gridDim.x;
  for (; index < elements; index += stride) {
    data[index] = pattern(static_cast<unsigned>(index), 0);
  }
}

template <Direction kDirection>
__global__ __launch_bounds__(256, 4) void bandwidth_kernel(
    uint4* data, size_t element_mask, int iterations, unsigned* sink,
    unsigned long long* start_ns, unsigned long long* stop_ns,
    unsigned* smids) {
  const unsigned tid = threadIdx.x;
  const unsigned global_tid = blockIdx.x * blockDim.x + tid;
  const unsigned total_threads = gridDim.x * blockDim.x;
  const size_t step = size_t(total_threads);
  // One iteration covers a contiguous grid-wide chunk.  Advancing by exactly
  // that chunk keeps each warp's 16-B requests coalesced and 128-B aligned;
  // the aggregate set of thread/item lanes still sweeps every power-of-two
  // working-set segment even though one individual lane need not do so.
  const size_t round_stride = step * size_t(kUnroll);
  size_t indices[kUnroll];
  #pragma unroll
  for (int item = 0; item < kUnroll; ++item) {
    indices[item] =
        (size_t(global_tid) + step * static_cast<size_t>(item)) & element_mask;
  }
  uint4 accumulator = pattern(global_tid, 1);

  __syncthreads();
  if (tid == 0) start_ns[blockIdx.x] = global_nanoseconds();
  __syncthreads();

  #pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if constexpr (kDirection == Direction::kRead) {
      const uint4 v0 = load_cg(data + indices[0]);
      const uint4 v1 = load_cg(data + indices[1]);
      const uint4 v2 = load_cg(data + indices[2]);
      const uint4 v3 = load_cg(data + indices[3]);
      const uint4 v4 = load_cg(data + indices[4]);
      const uint4 v5 = load_cg(data + indices[5]);
      const uint4 v6 = load_cg(data + indices[6]);
      const uint4 v7 = load_cg(data + indices[7]);
      // Every lane of every uint4 must remain live.  Counting 16 B per
      // operation while consuming only a subset lets ptxas scalarize the load
      // and overstates requested bandwidth.
      accumulator.x ^= v0.x + v1.x + v2.x + v3.x +
                       v4.x + v5.x + v6.x + v7.x;
      accumulator.y += v0.y ^ v1.y ^ v2.y ^ v3.y ^
                       v4.y ^ v5.y ^ v6.y ^ v7.y;
      accumulator.z ^= v0.z + v1.z + v2.z + v3.z +
                       v4.z + v5.z + v6.z + v7.z + accumulator.x;
      accumulator.w += v0.w ^ v1.w ^ v2.w ^ v3.w ^
                       v4.w ^ v5.w ^ v6.w ^ v7.w ^ accumulator.y;
    } else {
      #pragma unroll
      for (int item = 0; item < kUnroll; ++item) {
        store_cg(data + indices[item],
                 pattern(global_tid + static_cast<unsigned>(item),
                         static_cast<unsigned>(iteration)));
      }
      accumulator.x += static_cast<unsigned>(iteration) + 1u;
      accumulator.y ^= accumulator.x + global_tid;
      accumulator.z += accumulator.y;
      accumulator.w ^= accumulator.z;
    }
    #pragma unroll
    for (int item = 0; item < kUnroll; ++item) {
      indices[item] = (indices[item] + round_stride) & element_mask;
    }
  }
  if constexpr (kDirection == Direction::kWrite) __threadfence();

  __syncthreads();
  if (tid == 0) {
    stop_ns[blockIdx.x] = global_nanoseconds();
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smids[blockIdx.x]));
  }
  sink[global_tid] =
      accumulator.x ^ accumulator.y ^ accumulator.z ^ accumulator.w;
}

struct Options {
  Direction direction = Direction::kRead;
  Residency residency = Residency::kL2;
  int iterations = 4096;
  int warmup_iterations = 64;
  int blocks_per_sm = 4;
  int threads = 256;
  size_t bytes = 16ull << 20;
};

const char* direction_name(Direction direction) {
  return direction == Direction::kRead ? "read" : "write";
}

const char* residency_name(Residency residency) {
  return residency == Residency::kL2 ? "l2" : "hbm";
}

size_t floor_power_of_two(size_t value) {
  if (!value) return 0;
  size_t result = 1;
  while (result <= value / 2) result <<= 1;
  return result;
}

void usage(const char* program) {
  std::fprintf(
      stderr,
      "Usage: %s --direction read|write --residency l2|hbm "
      "[--bytes N] [--iters N] [--warmup-iters N] "
      "[--blocks-per-sm 1..4] [--threads 32..256]\n",
      program);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--direction") == 0 && i + 1 < argc) {
      const char* value = argv[++i];
      if (std::strcmp(value, "read") == 0) options.direction = Direction::kRead;
      else if (std::strcmp(value, "write") == 0) options.direction = Direction::kWrite;
      else { usage(argv[0]); std::exit(2); }
    } else if (std::strcmp(argv[i], "--residency") == 0 && i + 1 < argc) {
      const char* value = argv[++i];
      if (std::strcmp(value, "l2") == 0) options.residency = Residency::kL2;
      else if (std::strcmp(value, "hbm") == 0) options.residency = Residency::kHbm;
      else { usage(argv[0]); std::exit(2); }
    } else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
      options.bytes = std::strtoull(argv[++i], nullptr, 0);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      options.iterations = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup-iters") == 0 && i + 1 < argc) {
      options.warmup_iterations = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--blocks-per-sm") == 0 && i + 1 < argc) {
      options.blocks_per_sm = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      options.threads = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      usage(argv[0]);
      std::exit(2);
    }
  }
  options.bytes = floor_power_of_two(options.bytes);
  if (options.iterations <= 0 || options.warmup_iterations < 0 ||
      options.blocks_per_sm <= 0 || options.blocks_per_sm > 4 ||
      options.threads < 32 || options.threads > 256 ||
      options.threads % 32 != 0 || options.bytes < (1ull << 20)) {
    usage(argv[0]);
    std::exit(2);
  }
  return options;
}

using Kernel = void (*)(uint4*, size_t, int, unsigned*, unsigned long long*,
                        unsigned long long*, unsigned*);

Kernel kernel_for(Direction direction) {
  return direction == Direction::kRead
      ? bandwidth_kernel<Direction::kRead>
      : bandwidth_kernel<Direction::kWrite>;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  const int blocks = properties.multiProcessorCount * options.blocks_per_sm;
  const size_t elements = options.bytes / sizeof(uint4);
  if (!elements || (elements & (elements - 1))) {
    std::fprintf(stderr, "working set must contain a power-of-two uint4 count\n");
    return 2;
  }
  Kernel kernel = kernel_for(options.direction);
  int occupancy = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, kernel, options.threads, 0));
  if (options.blocks_per_sm > occupancy) {
    std::fprintf(stderr, "blocks_per_sm=%d exceeds occupancy=%d\n",
                 options.blocks_per_sm, occupancy);
    return 2;
  }

  uint4* d_data = nullptr;
  unsigned* d_sink = nullptr;
  unsigned long long* d_start_ns = nullptr;
  unsigned long long* d_stop_ns = nullptr;
  unsigned* d_smids = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, options.bytes));
  CUDA_CHECK(cudaMalloc(
      &d_sink, size_t(blocks) * options.threads * sizeof(*d_sink)));
  CUDA_CHECK(cudaMalloc(&d_start_ns, blocks * sizeof(*d_start_ns)));
  CUDA_CHECK(cudaMalloc(&d_stop_ns, blocks * sizeof(*d_stop_ns)));
  CUDA_CHECK(cudaMalloc(&d_smids, blocks * sizeof(*d_smids)));

  initialize<<<std::min(4096, std::max(1, blocks * 8)), 256>>>(
      d_data, elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  if (options.warmup_iterations) {
    kernel<<<blocks, options.threads>>>(
        d_data, elements - 1, options.warmup_iterations, d_sink, d_start_ns,
        d_stop_ns, d_smids);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  kernel<<<blocks, options.threads>>>(
      d_data, elements - 1, options.iterations, d_sink, d_start_ns, d_stop_ns,
      d_smids);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unsigned long long> starts(blocks);
  std::vector<unsigned long long> stops(blocks);
  std::vector<unsigned> smids(blocks);
  std::vector<unsigned> sink(size_t(blocks) * options.threads);
  CUDA_CHECK(cudaMemcpy(starts.data(), d_start_ns,
                        blocks * sizeof(*d_start_ns), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(stops.data(), d_stop_ns,
                        blocks * sizeof(*d_stop_ns), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(smids.data(), d_smids,
                        blocks * sizeof(*d_smids), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sink.data(), d_sink, sink.size() * sizeof(*d_sink),
                        cudaMemcpyDeviceToHost));

  const auto start_min = *std::min_element(starts.begin(), starts.end());
  const auto stop_max = *std::max_element(stops.begin(), stops.end());
  const auto elapsed_ns = stop_max - start_min;
  int unique_smid_count = 0;
  for (int i = 0; i < blocks; ++i) {
    bool first = true;
    for (int j = 0; j < i; ++j) {
      if (smids[j] == smids[i]) first = false;
    }
    if (first) ++unique_smid_count;
  }
  const unsigned sink_xor = std::accumulate(
      sink.begin(), sink.end(), 0u, [](unsigned left, unsigned right) {
        return left ^ right;
      });
  const unsigned long long operation_count =
      static_cast<unsigned long long>(blocks) * options.threads *
      options.iterations * kUnroll;
  const unsigned long long requested_bytes =
      operation_count * kBytesPerOperation;
  const double bytes_per_second = elapsed_ns
      ? static_cast<double>(requested_bytes) * 1.0e9 / elapsed_ns
      : 0.0;

  std::printf(
      "case_id=%s_%s_aggregate\nmode=%s-%s\nresidency=%s\ndirection=%s\n"
      "sm_count=%d\nblocks=%d\nblocks_per_sm=%d\n"
      "unique_smid_count=%d\nthreads=%d\noccupancy_blocks_per_sm=%d\n"
      "iterations=%d\nwarmup_iterations=%d\nunroll=%d\n"
      "bytes_per_operation=%d\nworking_set_bytes=%zu\n"
      "operation_count=%llu\nrequested_bytes=%llu\n"
      "globaltimer_start_min_ns=%llu\nglobaltimer_stop_max_ns=%llu\n"
      "globaltimer_elapsed_ns=%llu\nbytes_per_second=%.9e\nsink=%u\n",
      residency_name(options.residency), direction_name(options.direction),
      residency_name(options.residency), direction_name(options.direction),
      residency_name(options.residency), direction_name(options.direction),
      properties.multiProcessorCount, blocks, options.blocks_per_sm,
      unique_smid_count, options.threads, occupancy, options.iterations,
      options.warmup_iterations, kUnroll, kBytesPerOperation, options.bytes,
      operation_count, requested_bytes, start_min, stop_max, elapsed_ns,
      bytes_per_second, sink_xor);

  CUDA_CHECK(cudaFree(d_smids));
  CUDA_CHECK(cudaFree(d_stop_ns));
  CUDA_CHECK(cudaFree(d_start_ns));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_data));
  return 0;
}
