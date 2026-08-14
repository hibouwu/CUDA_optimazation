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
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(error__));                             \
      std::exit(3);                                                          \
    }                                                                        \
  } while (0)

namespace {

constexpr int kThreads = 128;
constexpr int kTmemColumns = 512;
constexpr int kScaleRows = 32;
constexpr int kWordsPerRow = 4;
constexpr int kSourceBytesPerCopy =
    kScaleRows * kWordsPerRow * static_cast<int>(sizeof(uint32_t));
constexpr int kMulticastPartitions = 4;
constexpr unsigned kDestinationBaseColumn = 384;
constexpr unsigned kDestinationColumnsPerCopy = 4;
constexpr unsigned kDestinationSlots = 32;

__device__ __forceinline__ unsigned smem_u32(const void* pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ unsigned long long global_nanoseconds() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ uint64_t make_smem_desc(
    const void* pointer, unsigned leading_u128, unsigned stride_u128) {
  const unsigned address = smem_u32(pointer);
  uint64_t descriptor = 0;
  descriptor |= uint64_t((address >> 4) & 0x3fffu);
  descriptor |= uint64_t(leading_u128 & 0x3fffu) << 16;
  descriptor |= uint64_t(stride_u128 & 0x3fffu) << 32;
  descriptor |= uint64_t(1) << 46;
  return descriptor;
}

__device__ __forceinline__ void barrier_init(
    unsigned long long* barrier, unsigned arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(smem_u32(barrier)), "r"(arrivals)
               : "memory");
}

__device__ __forceinline__ void barrier_wait(
    unsigned long long* barrier, unsigned phase) {
  const unsigned address = smem_u32(barrier);
  const unsigned suspend_ticks = 0x989680u;
  asm volatile(
      "{ .reg .pred complete; scale_wait: "
      "mbarrier.try_wait.parity.shared::cta.b64 complete, [%0], %1, %2; "
      "@complete bra scale_done; bra scale_wait; scale_done: }"
      :
      : "r"(address), "r"(phase), "r"(suspend_ticks)
      : "memory");
}

__device__ __forceinline__ void scale_copy(
    unsigned destination, uint64_t source_descriptor) {
  asm volatile(
      "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
      :
      : "r"(destination), "l"(source_descriptor)
      : "memory");
}

__device__ __forceinline__ void commit_and_wait(
    unsigned long long* barrier, unsigned phase) {
  const unsigned address = smem_u32(barrier);
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
      "[%0];"
      :
      : "r"(address)
      : "memory");
  barrier_wait(barrier, phase);
}

__device__ __forceinline__ void tmem_load_x4(
    unsigned address, unsigned (&values)[4]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x4.b32 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]),
        "=r"(values[3])
      : "r"(address)
      : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__global__ __launch_bounds__(kThreads, 1) void scale_ingress_kernel(
    int iterations, int copies_per_commit, unsigned long long* start_ns,
    unsigned long long* stop_ns, unsigned* smids, unsigned* mismatches,
    unsigned* sink) {
  __shared__ alignas(128) uint32_t scale_atom[kScaleRows][kWordsPerRow];
  __shared__ alignas(8) unsigned long long done_barrier;
  __shared__ unsigned tmem_base;

  for (int index = threadIdx.x; index < kScaleRows * kWordsPerRow;
       index += blockDim.x) {
    const unsigned row = static_cast<unsigned>(index / kWordsPerRow);
    const unsigned word = static_cast<unsigned>(index % kWordsPerRow);
    scale_atom[row][word] =
        0x51000000u ^ (row * 0x00010101u) ^ (word * 0x11111111u);
  }
  if (threadIdx.x == 0) barrier_init(&done_barrier, 1);
  __syncthreads();

  // TMEM management operations are warp-collective even though cp/commit are
  // single-thread issue operations.
  if (threadIdx.x < 32) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(smem_u32(&tmem_base)), "r"(kTmemColumns)
        : "memory");
  }
  __syncthreads();

  // No-swizzle descriptor: adjacent 128-bit source rows are 16 B apart;
  // the stride from row 0 to row 8 is 128 B. Descriptor fields are in 16-B
  // units, hence leading=1 and stride=8.
  const uint64_t descriptor = make_smem_desc(scale_atom, 1, 8);
  const unsigned destination_base = tmem_base + kDestinationBaseColumn;
  unsigned phase = 0;

  __syncthreads();
  if (threadIdx.x == 0) start_ns[blockIdx.x] = global_nanoseconds();
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int begin = 0; begin < iterations; begin += copies_per_commit) {
      const int end = min(iterations, begin + copies_per_commit);
      for (int copy = begin; copy < end; ++copy) {
        // One 32x128b copy occupies four TMEM columns.  A commit batch uses
        // 32 distinct slots rather than issuing overlapping asynchronous
        // writes to the same destination.
        const unsigned slot = static_cast<unsigned>(copy) % kDestinationSlots;
        scale_copy(destination_base + slot * kDestinationColumnsPerCopy,
                   descriptor);
      }
      commit_and_wait(&done_barrier, phase);
      phase ^= 1u;
    }
    stop_ns[blockIdx.x] = global_nanoseconds();
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smids[blockIdx.x]));
  }

  // Publish cp completion from the single issuer before a warp collectively
  // reads one of the four multicast partitions for a value check.
  if (threadIdx.x == 0) {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    unsigned values[4];
    const unsigned final_slot =
        static_cast<unsigned>(iterations - 1) % kDestinationSlots;
    const unsigned final_destination =
        destination_base + final_slot * kDestinationColumnsPerCopy;
    tmem_load_x4(final_destination, values);
    const unsigned lane = static_cast<unsigned>(threadIdx.x);
    unsigned local_mismatches = 0;
    unsigned checksum = 0;
    #pragma unroll
    for (int word = 0; word < 4; ++word) {
      const unsigned expected = scale_atom[lane][word];
      local_mismatches += values[word] != expected;
      checksum ^= values[word];
    }
    if (local_mismatches) atomicAdd(mismatches, local_mismatches);
    atomicXor(sink, checksum + lane);
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
  int iterations = 16384;
  int copies_per_commit = 32;
  int blocks_per_sm = 1;
};

void usage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s [--iters N] [--copies-per-commit N] "
               "[--blocks-per-sm 1]\n",
               program);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      options.iterations = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--copies-per-commit") == 0 &&
               i + 1 < argc) {
      options.copies_per_commit = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--blocks-per-sm") == 0 &&
               i + 1 < argc) {
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
  if (options.iterations <= 0 || options.copies_per_commit <= 0 ||
      options.copies_per_commit > options.iterations ||
      options.copies_per_commit > static_cast<int>(kDestinationSlots) ||
      options.blocks_per_sm != 1) {
    usage(argv[0]);
    std::exit(2);
  }
  return options;
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
  unsigned* d_mismatches = nullptr;
  unsigned* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_start_ns, blocks * sizeof(*d_start_ns)));
  CUDA_CHECK(cudaMalloc(&d_stop_ns, blocks * sizeof(*d_stop_ns)));
  CUDA_CHECK(cudaMalloc(&d_smids, blocks * sizeof(*d_smids)));
  CUDA_CHECK(cudaMalloc(&d_mismatches, sizeof(*d_mismatches)));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(*d_sink)));
  CUDA_CHECK(cudaMemset(d_mismatches, 0, sizeof(*d_mismatches)));
  CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(*d_sink)));

  scale_ingress_kernel<<<blocks, kThreads>>>(
      options.iterations, options.copies_per_commit, d_start_ns, d_stop_ns,
      d_smids, d_mismatches, d_sink);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unsigned long long> starts(blocks);
  std::vector<unsigned long long> stops(blocks);
  std::vector<unsigned> smids(blocks);
  unsigned mismatches = 0;
  unsigned sink = 0;
  CUDA_CHECK(cudaMemcpy(starts.data(), d_start_ns,
                        blocks * sizeof(*d_start_ns), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(stops.data(), d_stop_ns,
                        blocks * sizeof(*d_stop_ns), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(smids.data(), d_smids,
                        blocks * sizeof(*d_smids), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&mismatches, d_mismatches, sizeof(mismatches),
                        cudaMemcpyDeviceToHost));
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

  const unsigned long long instruction_count =
      static_cast<unsigned long long>(blocks) * options.iterations;
  const unsigned long long issued_source_bytes =
      instruction_count * kSourceBytesPerCopy;
  const unsigned long long multicast_destination_bytes =
      issued_source_bytes * kMulticastPartitions;
  const double bytes_per_second = elapsed_ns
      ? static_cast<double>(issued_source_bytes) * 1.0e9 / elapsed_ns
      : 0.0;

  std::printf(
      "case_id=tmem_scale_ingress_32x128b_warpx4\n"
      "sm_count=%d\nblocks=%d\nunique_smid_count=%d\n"
      "iterations=%d\ncopies_per_commit=%d\n"
      "destination_slots=%u\ndestination_columns_per_copy=%u\n"
      "source_bytes_per_instruction=%d\nmulticast_partitions=%d\n"
      "instruction_count=%llu\nissued_source_bytes=%llu\n"
      "multicast_destination_bytes=%llu\n"
      "globaltimer_start_min_ns=%llu\nglobaltimer_stop_max_ns=%llu\n"
      "globaltimer_elapsed_ns=%llu\nbytes_per_second=%.9e\n"
      "value_mismatches=%u\nsink=%u\n",
      properties.multiProcessorCount, blocks, unique_smid_count,
      options.iterations, options.copies_per_commit,
      kDestinationSlots, kDestinationColumnsPerCopy,
      kSourceBytesPerCopy, kMulticastPartitions, instruction_count,
      issued_source_bytes, multicast_destination_bytes, start_min, stop_max,
      elapsed_ns, bytes_per_second, mismatches, sink);

  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_mismatches));
  CUDA_CHECK(cudaFree(d_smids));
  CUDA_CHECK(cudaFree(d_stop_ns));
  CUDA_CHECK(cudaFree(d_start_ns));
  return mismatches == 0 ? 0 : 5;
}
