#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t error__ = (call);                                            \
    if (error__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(error__));                             \
      std::exit(3);                                                          \
    }                                                                        \
  } while (0)

namespace {

constexpr int kTmemColumns = 512;

__device__ __forceinline__ unsigned smem_u32(const void* pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ unsigned long long global_nanoseconds() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ void tmem_store_x8(
    unsigned address, unsigned v0, unsigned v1, unsigned v2, unsigned v3,
    unsigned v4, unsigned v5, unsigned v6, unsigned v7) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], "
      "{%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(address), "r"(v0), "r"(v1), "r"(v2), "r"(v3), "r"(v4),
        "r"(v5), "r"(v6), "r"(v7)
      : "memory");
}

__device__ __forceinline__ void tmem_load_x8(
    unsigned address, unsigned (&values)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]),
        "=r"(values[3]), "=r"(values[4]), "=r"(values[5]),
        "=r"(values[6]), "=r"(values[7])
      : "r"(address)
      : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_x16(
    unsigned address, unsigned (&values)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
      : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]),
        "=r"(values[3]), "=r"(values[4]), "=r"(values[5]),
        "=r"(values[6]), "=r"(values[7]), "=r"(values[8]),
        "=r"(values[9]), "=r"(values[10]), "=r"(values[11]),
        "=r"(values[12]), "=r"(values[13]), "=r"(values[14]),
        "=r"(values[15])
      : "r"(address)
      : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

template <int RegistersPerLane, int ActiveWarps>
__global__ __launch_bounds__(128, 1) void readback_kernel(
    int iterations, unsigned long long* start_ns,
    unsigned long long* stop_ns, unsigned* smids, unsigned* sink) {
  __shared__ unsigned tmem_base;

  if (threadIdx.x < 32) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(smem_u32(&tmem_base)), "r"(kTmemColumns)
        : "memory");
  }
  __syncthreads();

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const unsigned address = tmem_base + static_cast<unsigned>(warp * 32);
  if (warp < ActiveWarps) {
    const unsigned lane = static_cast<unsigned>(threadIdx.x) & 31u;
    const unsigned seed = 0x3f000000u + lane + static_cast<unsigned>(warp * 256);
    tmem_store_x8(address, seed + 0u, seed + 1u, seed + 2u, seed + 3u,
                  seed + 4u, seed + 5u, seed + 6u, seed + 7u);
    tmem_store_x8(address + 8u, seed + 8u, seed + 9u, seed + 10u,
                  seed + 11u, seed + 12u, seed + 13u, seed + 14u,
                  seed + 15u);
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  }
  __syncthreads();

  if (threadIdx.x == 0) start_ns[blockIdx.x] = global_nanoseconds();
  __syncthreads();

  unsigned values[RegistersPerLane];
  #pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (warp < ActiveWarps) {
      if constexpr (RegistersPerLane == 8) {
        tmem_load_x8(address, values);
      } else {
        tmem_load_x16(address, values);
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    stop_ns[blockIdx.x] = global_nanoseconds();
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smids[blockIdx.x]));
  }

  unsigned checksum = 0;
  if (warp < ActiveWarps) {
    #pragma unroll
    for (int i = 0; i < RegistersPerLane; ++i) checksum ^= values[i];
    atomicXor(sink, checksum + static_cast<unsigned>(threadIdx.x));
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :
        : "r"(tmem_base), "r"(kTmemColumns)
        : "memory");
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
        ::: "memory");
  }
}

struct Options {
  int registers_per_lane = 16;
  int active_warps = 4;
  int iterations = 10000;
  int blocks_per_sm = 1;
};

void usage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s [--registers 8|16] [--warps 1|4] "
               "[--iters N] [--blocks-per-sm 1]\n",
               program);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--registers") == 0 && i + 1 < argc) {
      options.registers_per_lane = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warps") == 0 && i + 1 < argc) {
      options.active_warps = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      options.iterations = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--blocks-per-sm") == 0 && i + 1 < argc) {
      options.blocks_per_sm = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      usage(argv[0]);
      std::exit(2);
    }
  }
  if ((options.registers_per_lane != 8 && options.registers_per_lane != 16) ||
      (options.active_warps != 1 && options.active_warps != 4) ||
      options.iterations <= 0 || options.blocks_per_sm != 1) {
    usage(argv[0]);
    std::exit(2);
  }
  return options;
}

template <int RegistersPerLane, int ActiveWarps>
void launch(int blocks, const Options& options, unsigned long long* start_ns,
            unsigned long long* stop_ns, unsigned* smids, unsigned* sink) {
  readback_kernel<RegistersPerLane, ActiveWarps><<<blocks, 128>>>(
      options.iterations, start_ns, stop_ns, smids, sink);
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major != 11) {
    std::fprintf(stderr, "requires an SM110-family device, found sm_%d%d\n",
                 properties.major, properties.minor);
    return 4;
  }
  const int blocks = properties.multiProcessorCount * options.blocks_per_sm;

  unsigned long long* d_start_ns = nullptr;
  unsigned long long* d_stop_ns = nullptr;
  unsigned* d_smids = nullptr;
  unsigned* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_start_ns, blocks * sizeof(*d_start_ns)));
  CUDA_CHECK(cudaMalloc(&d_stop_ns, blocks * sizeof(*d_stop_ns)));
  CUDA_CHECK(cudaMalloc(&d_smids, blocks * sizeof(*d_smids)));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(*d_sink)));
  CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(*d_sink)));

  if (options.registers_per_lane == 8 && options.active_warps == 1) {
    launch<8, 1>(blocks, options, d_start_ns, d_stop_ns, d_smids, d_sink);
  } else if (options.registers_per_lane == 8) {
    launch<8, 4>(blocks, options, d_start_ns, d_stop_ns, d_smids, d_sink);
  } else if (options.active_warps == 1) {
    launch<16, 1>(blocks, options, d_start_ns, d_stop_ns, d_smids, d_sink);
  } else {
    launch<16, 4>(blocks, options, d_start_ns, d_stop_ns, d_smids, d_sink);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unsigned long long> starts(blocks);
  std::vector<unsigned long long> stops(blocks);
  std::vector<unsigned> smids(blocks);
  unsigned sink = 0;
  CUDA_CHECK(cudaMemcpy(starts.data(), d_start_ns,
                        blocks * sizeof(*d_start_ns), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(stops.data(), d_stop_ns,
                        blocks * sizeof(*d_stop_ns), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(smids.data(), d_smids,
                        blocks * sizeof(*d_smids), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&sink, d_sink, sizeof(sink), cudaMemcpyDeviceToHost));

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

  const unsigned long long bytes_per_instruction =
      32ull * static_cast<unsigned long long>(options.registers_per_lane) * 4ull;
  const unsigned long long issued_bytes =
      static_cast<unsigned long long>(blocks) * options.active_warps *
      options.iterations * bytes_per_instruction;
  const double bytes_per_second = elapsed_ns
      ? static_cast<double>(issued_bytes) * 1.0e9 / elapsed_ns
      : 0.0;

  std::printf(
      "case_id=tmem_ld_32x32b_x%d_warps%d\n"
      "sm_count=%d\nblocks=%d\nunique_smid_count=%d\n"
      "active_warps=%d\nregisters_per_lane=%d\niterations=%d\n"
      "bytes_per_instruction=%llu\nissued_bytes=%llu\n"
      "globaltimer_start_min_ns=%llu\nglobaltimer_stop_max_ns=%llu\n"
      "globaltimer_elapsed_ns=%llu\nbytes_per_second=%.9e\nsink=%u\n",
      options.registers_per_lane, options.active_warps,
      properties.multiProcessorCount, blocks, unique_smid_count,
      options.active_warps, options.registers_per_lane, options.iterations,
      bytes_per_instruction, issued_bytes, start_min, stop_max, elapsed_ns,
      bytes_per_second, sink);

  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_smids));
  CUDA_CHECK(cudaFree(d_stop_ns));
  CUDA_CHECK(cudaFree(d_start_ns));
  return 0;
}
