#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error_ = (call);                                           \
    if (error_ != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string(#call) + ": " +                      \
                               cudaGetErrorString(error_));                      \
    }                                                                           \
  } while (false)

namespace {

constexpr int kThreads = 256;
constexpr int kConsumerLanes = 32;
constexpr int kTileBytes = 4096;
constexpr int kBoxXBytes = 32;
constexpr int kBoxY = kTileBytes / kBoxXBytes;
constexpr int kAtomBytes = 16;

struct Options {
  std::string case_selector = "all";
  int iters = 1000;
  int warmups = 3;
  int repeats = 10;
  bool list_cases = false;
};

enum class CaseKind {
  kLoadOnly,
  kStoreOnly,
  kLoadConsumer,
  kProducerStore,
  kRoundTrip,
};

struct CaseSpec {
  const char* experiment;
  const char* name;
  CaseKind kind;
  CUtensorMapSwizzle swizzle;
  int swizzle_atoms;
  const char* direction;
  const char* consumer;
  const char* producer;
  int tma_operations;
};

constexpr CaseSpec kCases[] = {
    {"T0", "T0a_gmem_to_smem_no_swizzle_copy", CaseKind::kLoadOnly,
     CU_TENSOR_MAP_SWIZZLE_NONE, 0, "gmem_to_smem", "none", "none", 1},
    {"T0", "T0b_smem_to_gmem_no_swizzle_copy", CaseKind::kStoreOnly,
     CU_TENSOR_MAP_SWIZZLE_NONE, 0, "smem_to_gmem", "none", "none", 1},

    {"T1", "T1a_load_no_swizzle_column_consumer", CaseKind::kLoadConsumer,
     CU_TENSOR_MAP_SWIZZLE_NONE, 0, "gmem_to_smem", "column", "none", 1},
    {"T1", "T1b_load_32b_swizzle_matched_consumer", CaseKind::kLoadConsumer,
     CU_TENSOR_MAP_SWIZZLE_32B, 2, "gmem_to_smem", "matched_column", "none", 1},
    {"T1", "T1c_load_64b_swizzle_matched_consumer", CaseKind::kLoadConsumer,
     CU_TENSOR_MAP_SWIZZLE_64B, 4, "gmem_to_smem", "matched_column", "none", 1},
    {"T1", "T1d_load_128b_swizzle_matched_consumer", CaseKind::kLoadConsumer,
     CU_TENSOR_MAP_SWIZZLE_128B, 8, "gmem_to_smem", "matched_column", "none",
     1},

    {"T2", "T2a_column_producer_store_no_swizzle", CaseKind::kProducerStore,
     CU_TENSOR_MAP_SWIZZLE_NONE, 0, "smem_to_gmem", "none", "column", 1},
    {"T2", "T2b_matched_producer_store_32b_swizzle",
     CaseKind::kProducerStore, CU_TENSOR_MAP_SWIZZLE_32B, 2, "smem_to_gmem",
     "none", "matched_column", 1},
    {"T2", "T2c_matched_producer_store_64b_swizzle",
     CaseKind::kProducerStore, CU_TENSOR_MAP_SWIZZLE_64B, 4, "smem_to_gmem",
     "none", "matched_column", 1},
    {"T2", "T2d_matched_producer_store_128b_swizzle",
     CaseKind::kProducerStore, CU_TENSOR_MAP_SWIZZLE_128B, 8, "smem_to_gmem",
     "none", "matched_column", 1},

    {"T3", "T3a_load_store_no_swizzle", CaseKind::kRoundTrip,
     CU_TENSOR_MAP_SWIZZLE_NONE, 0, "round_trip", "none", "none", 2},
    {"T3", "T3b_load_store_32b_swizzle", CaseKind::kRoundTrip,
     CU_TENSOR_MAP_SWIZZLE_32B, 2, "round_trip", "none", "none", 2},
    {"T3", "T3c_load_store_64b_swizzle", CaseKind::kRoundTrip,
     CU_TENSOR_MAP_SWIZZLE_64B, 4, "round_trip", "none", "none", 2},
    {"T3", "T3d_load_store_128b_swizzle", CaseKind::kRoundTrip,
     CU_TENSOR_MAP_SWIZZLE_128B, 8, "round_trip", "none", "none", 2},
};

__device__ __forceinline__ unsigned smem_addr(const void* pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void mbarrier_init(unsigned barrier) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
               :
               : "r"(barrier)
               : "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx(unsigned barrier,
                                                   unsigned bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
      "_, [%0], %1;"
      :
      : "r"(barrier), "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void mbarrier_wait(unsigned barrier,
                                              unsigned phase) {
  asm volatile(
      "{\n"
      ".reg .pred ready;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
      "ready, [%0], %1;\n"
      "@!ready bra.uni WAIT_%=;\n"
      "}"
      :
      : "r"(barrier), "r"(phase)
      : "memory");
}

__device__ __forceinline__ void async_proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d(
    unsigned destination, const CUtensorMap* tensor_map, unsigned barrier) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile."
      "mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(destination), "l"(tensor_map), "r"(0), "r"(0), "r"(barrier)
      : "memory");
}

__device__ __forceinline__ void tma_store_2d(
    const CUtensorMap* tensor_map, unsigned source) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
      "[%0, {%1, %2}], [%3];"
      :
      : "l"(tensor_map), "r"(0), "r"(0), "r"(source)
      : "memory");
}

__device__ __forceinline__ void bulk_commit_wait() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ uint4 load_shared_u32x4(const std::uint8_t* ptr) {
  uint4 value;
  asm volatile("ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "r"(smem_addr(ptr))
               : "memory");
  return value;
}

__device__ __forceinline__ void store_shared_u32x4(std::uint8_t* ptr,
                                                   const uint4& value) {
  asm volatile("st.volatile.shared.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "r"(smem_addr(ptr)), "r"(value.x), "r"(value.y),
                 "r"(value.z), "r"(value.w)
               : "memory");
}

template <int SwizzleAtoms>
__host__ __device__ __forceinline__ int swizzled_byte_offset(
    int logical_byte_offset) {
  if constexpr (SwizzleAtoms == 0) {
    return logical_byte_offset;
  } else {
    constexpr int swizzle_bytes = SwizzleAtoms * kAtomBytes;
    const int row = logical_byte_offset / kBoxXBytes;
    const int byte_in_row = logical_byte_offset % kBoxXBytes;
    const int atom_in_row = byte_in_row / kAtomBytes;
    const int byte_in_atom = byte_in_row % kAtomBytes;
    const int physical_atom = (row % SwizzleAtoms) ^ atom_in_row;
    return row * swizzle_bytes + physical_atom * kAtomBytes + byte_in_atom;
  }
}

__device__ __forceinline__ std::uint8_t initial_tile_byte(int offset) {
  return static_cast<std::uint8_t>((offset * 17 + 3) & 0xff);
}

template <int SwizzleAtoms>
__device__ __forceinline__ int column_atom_offset(int lane) {
  return swizzled_byte_offset<SwizzleAtoms>(lane * kBoxXBytes);
}

template <CaseKind Kind, int SwizzleAtoms>
__global__ void tma_case_kernel(
    const __grid_constant__ CUtensorMap load_map,
    const __grid_constant__ CUtensorMap store_map, std::uint64_t* checksums,
    std::uint64_t* consumer_sink, int iters) {
  constexpr int shared_row_bytes =
      SwizzleAtoms == 0 ? kBoxXBytes : SwizzleAtoms * kAtomBytes;
  constexpr int shared_bytes = shared_row_bytes * kBoxY;
  __shared__ alignas(1024) std::uint8_t tile[shared_bytes];
  __shared__ alignas(8) std::uint64_t barrier_storage;

  const int tid = threadIdx.x;
  for (int logical_offset = tid; logical_offset < kTileBytes;
       logical_offset += blockDim.x) {
    tile[swizzled_byte_offset<SwizzleAtoms>(logical_offset)] =
        initial_tile_byte(logical_offset);
  }
  if (tid == 0) {
    const unsigned barrier = smem_addr(&barrier_storage);
    mbarrier_init(barrier);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  async_proxy_fence();
  __syncthreads();

  const unsigned barrier = smem_addr(&barrier_storage);
  const unsigned tile_address = smem_addr(tile);
  unsigned phase = 0;
  std::uint64_t accumulator = 0;

  for (int iteration = 0; iteration < iters; ++iteration) {
    if constexpr (Kind == CaseKind::kLoadOnly ||
                  Kind == CaseKind::kLoadConsumer ||
                  Kind == CaseKind::kRoundTrip) {
      if (tid == 0) {
        mbarrier_expect_tx(barrier, kTileBytes);
        tma_load_2d(tile_address, &load_map, barrier);
      }
      mbarrier_wait(barrier, phase);
      phase ^= 1;
      __syncthreads();
    }

    if constexpr (Kind == CaseKind::kLoadConsumer) {
      if (tid < kConsumerLanes) {
        const int offset = column_atom_offset<SwizzleAtoms>(tid);
        const uint4 value = load_shared_u32x4(tile + offset);
        accumulator += static_cast<std::uint64_t>(value.x) + value.y + value.z +
                       value.w;
      }
      __syncthreads();
    }

    if constexpr (Kind == CaseKind::kProducerStore) {
      if (tid < kConsumerLanes) {
        const std::uint32_t byte =
            static_cast<std::uint32_t>((tid + iteration + 1) & 0xff);
        const std::uint32_t word = byte * 0x01010101U;
        const uint4 value = {word, word, word, word};
        const int offset = column_atom_offset<SwizzleAtoms>(tid);
        store_shared_u32x4(tile + offset, value);
      }
      async_proxy_fence();
      __syncthreads();
    }

    if constexpr (Kind == CaseKind::kStoreOnly ||
                  Kind == CaseKind::kProducerStore ||
                  Kind == CaseKind::kRoundTrip) {
      if (tid == 0) {
        tma_store_2d(&store_map, tile_address);
        bulk_commit_wait();
      }
      __syncthreads();
    }
  }

  std::uint64_t checksum = 0;
  for (int logical_offset = tid; logical_offset < kTileBytes;
       logical_offset += blockDim.x) {
    checksum += tile[swizzled_byte_offset<SwizzleAtoms>(logical_offset)];
  }
  checksums[tid] = checksum;
  consumer_sink[tid] = accumulator;
}

void check_driver(CUresult status, const char* where) {
  if (status == CUDA_SUCCESS) return;
  const char* text = "unknown driver error";
  cuGetErrorString(status, &text);
  throw std::runtime_error(std::string(where) + ": " + text);
}

CUtensorMap make_tensor_map(CUtensorMapSwizzle swizzle, void* global_address) {
  CUtensorMap tensor_map{};
  const cuuint64_t dimensions[2] = {kBoxXBytes, kBoxY};
  const cuuint64_t strides[1] = {kBoxXBytes};
  const cuuint32_t box[2] = {kBoxXBytes, kBoxY};
  const cuuint32_t element_strides[2] = {1, 1};
  check_driver(
      cuTensorMapEncodeTiled(
          &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, global_address,
          dimensions, strides, box, element_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
          CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
  return tensor_map;
}

template <CaseKind Kind, int SwizzleAtoms>
void launch_typed(const CUtensorMap& load_map, const CUtensorMap& store_map,
                  std::uint64_t* checksums, std::uint64_t* consumer_sink,
                  int iters) {
  tma_case_kernel<Kind, SwizzleAtoms><<<1, kThreads>>>(
      load_map, store_map, checksums, consumer_sink, iters);
  CUDA_CHECK(cudaGetLastError());
}

template <CaseKind Kind>
void launch_swizzle(const CaseSpec& spec, const CUtensorMap& load_map,
                    const CUtensorMap& store_map, std::uint64_t* checksums,
                    std::uint64_t* consumer_sink, int iters) {
  switch (spec.swizzle_atoms) {
    case 0:
      launch_typed<Kind, 0>(load_map, store_map, checksums, consumer_sink,
                            iters);
      break;
    case 2:
      launch_typed<Kind, 2>(load_map, store_map, checksums, consumer_sink,
                            iters);
      break;
    case 4:
      launch_typed<Kind, 4>(load_map, store_map, checksums, consumer_sink,
                            iters);
      break;
    case 8:
      launch_typed<Kind, 8>(load_map, store_map, checksums, consumer_sink,
                            iters);
      break;
    default:
      throw std::runtime_error("unsupported swizzle atom count");
  }
}

void launch_case(const CaseSpec& spec, const CUtensorMap& load_map,
                 const CUtensorMap& store_map, std::uint64_t* checksums,
                 std::uint64_t* consumer_sink, int iters) {
  switch (spec.kind) {
    case CaseKind::kLoadOnly:
      launch_swizzle<CaseKind::kLoadOnly>(
          spec, load_map, store_map, checksums, consumer_sink, iters);
      break;
    case CaseKind::kStoreOnly:
      launch_swizzle<CaseKind::kStoreOnly>(
          spec, load_map, store_map, checksums, consumer_sink, iters);
      break;
    case CaseKind::kLoadConsumer:
      launch_swizzle<CaseKind::kLoadConsumer>(
          spec, load_map, store_map, checksums, consumer_sink, iters);
      break;
    case CaseKind::kProducerStore:
      launch_swizzle<CaseKind::kProducerStore>(
          spec, load_map, store_map, checksums, consumer_sink, iters);
      break;
    case CaseKind::kRoundTrip:
      launch_swizzle<CaseKind::kRoundTrip>(
          spec, load_map, store_map, checksums, consumer_sink, iters);
      break;
  }
}

std::vector<std::uint8_t> make_input() {
  std::vector<std::uint8_t> input(kTileBytes);
  for (int index = 0; index < kTileBytes; ++index) {
    input[index] = static_cast<std::uint8_t>((index * 13 + 7) & 0xff);
  }
  return input;
}

std::vector<std::uint8_t> make_initial_logical_tile() {
  std::vector<std::uint8_t> tile(kTileBytes);
  for (int index = 0; index < kTileBytes; ++index) {
    tile[index] = static_cast<std::uint8_t>((index * 17 + 3) & 0xff);
  }
  return tile;
}

std::vector<std::uint8_t> expected_store_output(const CaseSpec& spec,
                                                int iters) {
  std::vector<std::uint8_t> logical = make_initial_logical_tile();
  if (spec.kind == CaseKind::kProducerStore) {
    for (int lane = 0; lane < kConsumerLanes; ++lane) {
      const std::uint8_t value =
          static_cast<std::uint8_t>((lane + iters) & 0xff);
      const int logical_offset = lane * kBoxXBytes;
      std::fill_n(logical.begin() + logical_offset, kAtomBytes, value);
    }
  }
  return logical;
}

bool check_case(const CaseSpec& spec, int iters,
                const std::vector<std::uint8_t>& input,
                const std::vector<std::uint8_t>& output,
                const std::vector<std::uint64_t>& checksums) {
  if (spec.kind == CaseKind::kLoadOnly ||
      spec.kind == CaseKind::kLoadConsumer) {
    const std::uint64_t expected =
        std::accumulate(input.begin(), input.end(), std::uint64_t{0});
    const std::uint64_t actual =
        std::accumulate(checksums.begin(), checksums.end(), std::uint64_t{0});
    return actual == expected;
  }
  const std::vector<std::uint8_t> expected =
      spec.kind == CaseKind::kRoundTrip
          ? input
          : expected_store_output(spec, iters);
  return output == expected;
}

const char* swizzle_name(CUtensorMapSwizzle swizzle) {
  switch (swizzle) {
    case CU_TENSOR_MAP_SWIZZLE_NONE:
      return "none";
    case CU_TENSOR_MAP_SWIZZLE_32B:
      return "32B";
    case CU_TENSOR_MAP_SWIZZLE_64B:
      return "64B";
    case CU_TENSOR_MAP_SWIZZLE_128B:
      return "128B";
    default:
      return "unknown";
  }
}

int shared_footprint_bytes(const CaseSpec& spec) {
  const int row_bytes =
      spec.swizzle_atoms == 0 ? kBoxXBytes : spec.swizzle_atoms * kAtomBytes;
  return row_bytes * kBoxY;
}

struct Timing {
  double average_ms;
  double minimum_ms;
};

Timing measure_case(const CaseSpec& spec, const CUtensorMap& load_map,
                    const CUtensorMap& store_map, std::uint64_t* checksums,
                    std::uint64_t* consumer_sink, const Options& options) {
  for (int warmup = 0; warmup < options.warmups; ++warmup) {
    launch_case(spec, load_map, store_map, checksums, consumer_sink,
                options.iters);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<double> samples;
  samples.reserve(options.repeats);
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    CUDA_CHECK(cudaEventRecord(start));
    launch_case(spec, load_map, store_map, checksums, consumer_sink,
                options.iters);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    samples.push_back(milliseconds);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  return {sum / samples.size(),
          *std::min_element(samples.begin(), samples.end())};
}

int parse_integer(const char* text, const char* option, bool allow_zero) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  const long minimum = allow_zero ? 0 : 1;
  if (!*text || *end || value < minimum ||
      value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("invalid value for ") + option);
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      std::cout
          << "Usage: tma_bench [--case all|T0|T1|T2|T3|case-name]"
          << " [--iters N] [--warmups N] [--repeats N] [--list-cases]\n";
      std::exit(0);
    }
    if (argument == "--list-cases") {
      options.list_cases = true;
      continue;
    }
    if (++index >= argc) {
      throw std::invalid_argument("missing value after " + argument);
    }
    if (argument == "--case") {
      options.case_selector = argv[index];
    } else if (argument == "--iters") {
      options.iters = parse_integer(argv[index], "--iters", false);
    } else if (argument == "--warmups") {
      options.warmups = parse_integer(argv[index], "--warmups", true);
    } else if (argument == "--repeats") {
      options.repeats = parse_integer(argv[index], "--repeats", false);
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  return options;
}

std::vector<const CaseSpec*> select_cases(const std::string& selector) {
  std::vector<const CaseSpec*> selected;
  for (const auto& spec : kCases) {
    if (selector == "all" || selector == spec.experiment ||
        selector == spec.name) {
      selected.push_back(&spec);
    }
  }
  if (selected.empty()) {
    throw std::invalid_argument("unknown case selector: " + selector);
  }
  return selected;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto selected = select_cases(options.case_selector);
    if (options.list_cases) {
      for (const CaseSpec* spec : selected) {
        std::cout << spec->name << '\n';
      }
      return 0;
    }

    check_driver(cuInit(0), "cuInit");
    const std::vector<std::uint8_t> host_input = make_input();
    std::vector<std::uint8_t> host_output(kTileBytes);
    std::vector<std::uint64_t> host_checksums(kThreads);

    std::uint8_t* input = nullptr;
    std::uint8_t* output = nullptr;
    std::uint64_t* checksums = nullptr;
    std::uint64_t* consumer_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&input, kTileBytes));
    CUDA_CHECK(cudaMalloc(&output, kTileBytes));
    CUDA_CHECK(cudaMalloc(&checksums, kThreads * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&consumer_sink, kThreads * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMemcpy(input, host_input.data(), kTileBytes,
                          cudaMemcpyHostToDevice));

    std::cout << "experiment,case,direction,swizzle,box_x_bytes,box_y,"
                 "shared_bytes,"
                 "consumer,producer,tma_operations,iters,avg_ms,min_ms,"
                 "tma_bytes,effective_GBps,correctness\n";
    bool all_passed = true;
    for (const CaseSpec* spec : selected) {
      CUDA_CHECK(cudaMemset(output, 0, kTileBytes));
      CUDA_CHECK(cudaMemset(checksums, 0,
                            kThreads * sizeof(std::uint64_t)));
      CUDA_CHECK(cudaMemset(consumer_sink, 0,
                            kThreads * sizeof(std::uint64_t)));
      const CUtensorMap load_map = make_tensor_map(spec->swizzle, input);
      const CUtensorMap store_map = make_tensor_map(spec->swizzle, output);
      const Timing timing =
          measure_case(*spec, load_map, store_map, checksums, consumer_sink,
                       options);

      CUDA_CHECK(cudaMemcpy(host_checksums.data(), checksums,
                            kThreads * sizeof(std::uint64_t),
                            cudaMemcpyDeviceToHost));
      if (spec->kind == CaseKind::kStoreOnly ||
          spec->kind == CaseKind::kProducerStore ||
          spec->kind == CaseKind::kRoundTrip) {
        CUDA_CHECK(cudaMemcpy(host_output.data(), output, kTileBytes,
                              cudaMemcpyDeviceToHost));
      }
      const bool correct =
          check_case(*spec, options.iters, host_input, host_output,
                     host_checksums);
      all_passed = all_passed && correct;

      const long long tma_bytes =
          1LL * options.iters * kTileBytes * spec->tma_operations;
      const double effective_gbps =
          static_cast<double>(tma_bytes) / (timing.average_ms * 1.0e6);
      std::cout << spec->experiment << ',' << spec->name << ','
                << spec->direction << ',' << swizzle_name(spec->swizzle) << ','
                << kBoxXBytes << ',' << kBoxY << ','
                << shared_footprint_bytes(*spec) << ',' << spec->consumer
                << ',' << spec->producer << ',' << spec->tma_operations << ','
                << options.iters << ',' << std::fixed << std::setprecision(6)
                << timing.average_ms << ',' << timing.minimum_ms << ','
                << tma_bytes << ',' << std::setprecision(3) << effective_gbps
                << ',' << (correct ? "PASS" : "FAIL") << '\n';
    }

    CUDA_CHECK(cudaFree(consumer_sink));
    CUDA_CHECK(cudaFree(checksums));
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(input));
    return all_passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
