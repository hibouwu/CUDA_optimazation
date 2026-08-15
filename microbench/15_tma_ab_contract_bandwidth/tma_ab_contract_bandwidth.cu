#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t error__ = (call);                                           \
    if (error__ != cudaSuccess) {                                           \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(error__));                            \
      std::exit(3);                                                         \
    }                                                                       \
  } while (0)

#define CU_CHECK(call)                                                       \
  do {                                                                       \
    CUresult error__ = (call);                                               \
    if (error__ != CUDA_SUCCESS) {                                           \
      const char* message__ = "unknown";                                    \
      cuGetErrorString(error__, &message__);                                 \
      std::fprintf(stderr, "CUDA driver error %s:%d: %s\n", __FILE__,       \
                   __LINE__, message__);                                     \
      std::exit(3);                                                          \
    }                                                                        \
  } while (0)

namespace {

constexpr int kMaxStages = 8;

enum class Mode { kL2Hit, kDramStream };

struct Options {
  Mode mode = Mode::kL2Hit;
  std::string case_id = "unspecified";
  std::size_t target_working_set_bytes = 16ull << 20;
  int bm = 128;
  int bn = 256;
  int bk = 64;
  int value_bits = 16;
  int scale_block = 0;
  int stages = 4;
  int threads = 192;
  int controller_thread = 128;
  int blocks = 0;
  int blocks_per_sm = 1;
  int row_stride_elements = 2048;
  int iters = 4096;
  int warmup_iters = 1024;
  int expected_sm_count = 20;
  bool contract_only = false;
  bool csv = false;
  bool csv_header = false;
};

struct ValueLayout {
  CUtensorMapDataType data_type;
  CUtensorMapSwizzle swizzle;
  int container_bytes;
  int values_per_container;
  int row_containers;
  int row_stride_containers;
  int request_row_bytes;
  int request_x_containers;
  int chunks;
};

struct BufferLayout {
  std::size_t a_value_bytes = 0;
  std::size_t b_value_bytes = 0;
  std::size_t a_scale_bytes = 0;
  std::size_t b_scale_bytes = 0;
  std::size_t a_allocation_bytes = 0;
  std::size_t b_allocation_bytes = 0;
  std::size_t a_scale_allocation_bytes = 0;
  std::size_t b_scale_allocation_bytes = 0;
  std::size_t stage_bytes = 0;
  std::size_t allocation_bytes = 0;
  int a_scale_map_rows = 0;
  int b_scale_map_rows = 0;
  int scale_groups_padded = 0;
};

__device__ __forceinline__ std::uint32_t smem_u32(const void* pointer) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ unsigned long long global_nanoseconds() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ void mbarrier_init(std::uint32_t address,
                                               std::uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(address), "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait(std::uint32_t address,
                                               std::uint32_t phase) {
  constexpr std::uint32_t kTicks = 0x989680;
  asm volatile(
      "{ .reg .pred p; wait_loop_%=: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; "
      "@!p bra wait_loop_%=; }"
      :
      : "r"(address), "r"(phase), "r"(kTicks)
      : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    std::uint32_t barrier, std::uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
      :
      : "r"(barrier), "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void tma_load_2d(
    std::uint32_t destination, const CUtensorMap* map, int x, int y,
    std::uint32_t barrier) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global."
      "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(destination), "l"(map), "r"(x), "r"(y), "r"(barrier)
      : "memory");
}

__global__ void tma_ab_contract_kernel(
    const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b,
    const __grid_constant__ CUtensorMap map_scale_a,
    const __grid_constant__ CUtensorMap map_scale_b, int bm, int bn,
    int request_x_containers, int value_chunks, int value_request_bytes_a,
    int value_request_bytes_b, int a_value_bytes, int b_value_bytes,
    int a_scale_bytes, int b_scale_bytes, int a_scale_map_rows,
    int b_scale_map_rows, int stages, int controller_thread, int total_tiles,
    int iters, int warmup_iters, unsigned long long* cycles,
    unsigned long long* start_ns, unsigned long long* stop_ns,
    std::uint32_t* smids, std::uint32_t* sink) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ alignas(16) std::uint64_t barriers[kMaxStages];
  __shared__ unsigned long long start_clock;
  __shared__ unsigned long long stop_clock;

  const int tid = static_cast<int>(threadIdx.x);
  const int stage_bytes =
      a_value_bytes + b_value_bytes + a_scale_bytes + b_scale_bytes;
  if (tid == controller_thread) {
    for (int stage = 0; stage < stages; ++stage) {
      mbarrier_init(smem_u32(&barriers[stage]), 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    start_clock = 0;
    stop_clock = 0;
  }
  __syncthreads();

  std::uint32_t phases[kMaxStages] = {};
  std::uint32_t checksum = 0;
  const int total_operations = warmup_iters + iters;
  const int tiles_per_block =
      (total_tiles + static_cast<int>(gridDim.x) - 1) /
      static_cast<int>(gridDim.x);

  auto wait_for = [&](int operation, bool consume) {
    const int stage = operation & (stages - 1);
    const std::uint32_t barrier = smem_u32(&barriers[stage]);
    mbarrier_wait(barrier, phases[stage]);
    phases[stage] ^= 1;
    if (consume) {
      volatile std::uint32_t* first = reinterpret_cast<std::uint32_t*>(
          smem + static_cast<std::size_t>(stage) * stage_bytes);
      volatile std::uint32_t* last = reinterpret_cast<std::uint32_t*>(
          smem + static_cast<std::size_t>(stage + 1) * stage_bytes -
          sizeof(std::uint32_t));
      checksum ^= *first;
      checksum ^= *last;
    }
  };

  auto issue = [&](int operation) {
    const int stage = operation & (stages - 1);
    const int local_tile = operation % (tiles_per_block > 0 ? tiles_per_block : 1);
    const int tile =
        (static_cast<int>(blockIdx.x) * tiles_per_block + local_tile) %
        total_tiles;
    unsigned char* stage_smem =
        smem + static_cast<std::size_t>(stage) * stage_bytes;
    const std::uint32_t barrier = smem_u32(&barriers[stage]);
    int destination_offset = 0;
    for (int chunk = 0; chunk < value_chunks; ++chunk) {
      tma_load_2d(
          smem_u32(stage_smem + destination_offset), &map_a,
          chunk * request_x_containers, tile * bm, barrier);
      destination_offset += value_request_bytes_a;
    }
    for (int chunk = 0; chunk < value_chunks; ++chunk) {
      tma_load_2d(
          smem_u32(stage_smem + destination_offset), &map_b,
          chunk * request_x_containers, tile * bn, barrier);
      destination_offset += value_request_bytes_b;
    }
    if (a_scale_bytes != 0) {
      tma_load_2d(smem_u32(stage_smem + destination_offset), &map_scale_a,
                  0, tile * a_scale_map_rows, barrier);
      destination_offset += a_scale_bytes;
      tma_load_2d(smem_u32(stage_smem + destination_offset), &map_scale_b,
                  0, tile * b_scale_map_rows, barrier);
      destination_offset += b_scale_bytes;
    }
    mbarrier_arrive_expect_tx(barrier,
                              static_cast<std::uint32_t>(stage_bytes));
  };

  auto run_window = [&](int begin, int count, bool consume) {
    for (int local = 0; local < count; ++local) {
      if (local >= stages) {
        wait_for(begin + local - stages, consume);
      }
      issue(begin + local);
    }
    const int drain_begin = count > stages ? count - stages : 0;
    for (int local = drain_begin; local < count; ++local) {
      wait_for(begin + local, consume);
    }
  };

  if (tid == controller_thread) {
    run_window(0, warmup_iters, false);
  }
  __syncthreads();
  if (tid == controller_thread) {
    start_clock = clock64();
    start_ns[blockIdx.x] = global_nanoseconds();
    run_window(warmup_iters, iters, true);
    stop_clock = clock64();
    stop_ns[blockIdx.x] = global_nanoseconds();
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smids[blockIdx.x]));
    cycles[blockIdx.x] = stop_clock - start_clock;
    atomicXor(sink, checksum + static_cast<std::uint32_t>(total_operations));
  }
}

[[noreturn]] void fail(const char* message) {
  std::fprintf(stderr, "%s\n", message);
  std::exit(2);
}

int parse_positive(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 0);
  if (end == text || *end != '\0' || value <= 0 ||
      value > std::numeric_limits<int>::max()) {
    std::fprintf(stderr, "%s must be a positive int\n", name);
    std::exit(2);
  }
  return static_cast<int>(value);
}

std::size_t parse_size(const char* text, const char* name) {
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 0);
  if (end == text || *end != '\0' || value == 0) {
    std::fprintf(stderr, "%s must be a positive size\n", name);
    std::exit(2);
  }
  return static_cast<std::size_t>(value);
}

const char* mode_name(Mode mode) {
  return mode == Mode::kL2Hit ? "l2-hit" : "dram-stream";
}

Mode parse_mode(const char* text) {
  if (std::strcmp(text, "l2-hit") == 0) return Mode::kL2Hit;
  if (std::strcmp(text, "dram-stream") == 0) return Mode::kDramStream;
  fail("--mode must be l2-hit or dram-stream");
}

void usage(const char* program) {
  std::fprintf(
      stderr,
      "Usage: %s --case-id ID --mode l2-hit|dram-stream "
      "--bm M --bn N --bk K --value-bits 4|8|16|32 "
      "--scale-block 0|16|32 --stages 1|2|4|8 "
      "--row-stride-elements N --threads N --controller-thread N "
      "--bytes N --iters N --warmup-iters N "
      "[--blocks N] [--blocks-per-sm N] [--expected-sm-count N] "
      "[--contract-only] [--csv] [--csv-header]\n",
      program);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const char* argument = argv[index];
    auto value = [&]() -> const char* {
      if (index + 1 >= argc) {
        usage(argv[0]);
        std::exit(2);
      }
      return argv[++index];
    };
    if (std::strcmp(argument, "--case-id") == 0) {
      options.case_id = value();
    } else if (std::strcmp(argument, "--mode") == 0) {
      options.mode = parse_mode(value());
    } else if (std::strcmp(argument, "--bytes") == 0) {
      options.target_working_set_bytes = parse_size(value(), "--bytes");
    } else if (std::strcmp(argument, "--bm") == 0) {
      options.bm = parse_positive(value(), "--bm");
    } else if (std::strcmp(argument, "--bn") == 0) {
      options.bn = parse_positive(value(), "--bn");
    } else if (std::strcmp(argument, "--bk") == 0) {
      options.bk = parse_positive(value(), "--bk");
    } else if (std::strcmp(argument, "--value-bits") == 0) {
      options.value_bits = parse_positive(value(), "--value-bits");
    } else if (std::strcmp(argument, "--scale-block") == 0) {
      options.scale_block = std::atoi(value());
    } else if (std::strcmp(argument, "--stages") == 0) {
      options.stages = parse_positive(value(), "--stages");
    } else if (std::strcmp(argument, "--threads") == 0) {
      options.threads = parse_positive(value(), "--threads");
    } else if (std::strcmp(argument, "--controller-thread") == 0) {
      options.controller_thread = std::atoi(value());
    } else if (std::strcmp(argument, "--blocks") == 0) {
      options.blocks = std::atoi(value());
    } else if (std::strcmp(argument, "--blocks-per-sm") == 0) {
      options.blocks_per_sm = parse_positive(value(), "--blocks-per-sm");
    } else if (std::strcmp(argument, "--row-stride-elements") == 0) {
      options.row_stride_elements =
          parse_positive(value(), "--row-stride-elements");
    } else if (std::strcmp(argument, "--iters") == 0) {
      options.iters = parse_positive(value(), "--iters");
    } else if (std::strcmp(argument, "--warmup-iters") == 0) {
      options.warmup_iters = parse_positive(value(), "--warmup-iters");
    } else if (std::strcmp(argument, "--expected-sm-count") == 0) {
      options.expected_sm_count =
          parse_positive(value(), "--expected-sm-count");
    } else if (std::strcmp(argument, "--contract-only") == 0) {
      options.contract_only = true;
    } else if (std::strcmp(argument, "--csv") == 0) {
      options.csv = true;
    } else if (std::strcmp(argument, "--csv-header") == 0) {
      options.csv_header = true;
    } else if (std::strcmp(argument, "--help") == 0 ||
               std::strcmp(argument, "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      usage(argv[0]);
      std::exit(2);
    }
  }
  return options;
}

int ceil_div(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

int round_up(int value, int alignment) {
  return ceil_div(value, alignment) * alignment;
}

ValueLayout make_value_layout(const Options& options) {
  ValueLayout layout{};
  if (options.value_bits == 4) {
    layout.data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    layout.container_bytes = 1;
    layout.values_per_container = 2;
  } else if (options.value_bits == 8) {
    layout.data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    layout.container_bytes = 1;
    layout.values_per_container = 1;
  } else if (options.value_bits == 16) {
    layout.data_type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
    layout.container_bytes = 2;
    layout.values_per_container = 1;
  } else if (options.value_bits == 32) {
    layout.data_type = CU_TENSOR_MAP_DATA_TYPE_UINT32;
    layout.container_bytes = 4;
    layout.values_per_container = 1;
  } else {
    fail("--value-bits must be one of 4, 8, 16, 32");
  }
  if (options.bk % layout.values_per_container != 0 ||
      options.row_stride_elements % layout.values_per_container != 0) {
    fail("BK and row stride must be divisible by packed container width");
  }
  layout.row_containers = options.bk / layout.values_per_container;
  layout.row_stride_containers =
      options.row_stride_elements / layout.values_per_container;
  const int row_bytes = layout.row_containers * layout.container_bytes;
  layout.request_row_bytes = std::min(row_bytes, 128);
  if (row_bytes % layout.request_row_bytes != 0) {
    fail("value row bytes must split into equal TMA requests");
  }
  layout.chunks = row_bytes / layout.request_row_bytes;
  layout.request_x_containers =
      layout.request_row_bytes / layout.container_bytes;
  if (layout.request_row_bytes == 32) {
    layout.swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
  } else if (layout.request_row_bytes == 64) {
    layout.swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
  } else if (layout.request_row_bytes == 128) {
    layout.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  } else {
    fail("each value TMA request row must be 32, 64, or 128 bytes");
  }
  return layout;
}

BufferLayout make_buffer_layout(const Options& options,
                                const ValueLayout& values,
                                int total_tiles) {
  BufferLayout layout;
  const std::size_t row_bytes =
      static_cast<std::size_t>(values.row_containers) *
      values.container_bytes;
  layout.a_value_bytes = row_bytes * options.bm;
  layout.b_value_bytes = row_bytes * options.bn;
  const std::size_t value_stride_bytes =
      static_cast<std::size_t>(values.row_stride_containers) *
      values.container_bytes;
  layout.a_allocation_bytes =
      value_stride_bytes * options.bm * total_tiles;
  layout.b_allocation_bytes =
      value_stride_bytes * options.bn * total_tiles;
  if (options.scale_block != 0) {
    const int groups = ceil_div(options.bk, options.scale_block);
    layout.scale_groups_padded = round_up(groups, 4);
    const int a_scale_vectors = round_up(options.bm, 128);
    const int b_scale_vectors = round_up(options.bn, 128);
    layout.a_scale_bytes =
        static_cast<std::size_t>(layout.scale_groups_padded) *
        a_scale_vectors;
    layout.b_scale_bytes =
        static_cast<std::size_t>(layout.scale_groups_padded) *
        b_scale_vectors;
    constexpr int kScalePhysicalRowBytes = 32;
    if (layout.a_scale_bytes % kScalePhysicalRowBytes != 0 ||
        layout.b_scale_bytes % kScalePhysicalRowBytes != 0) {
      fail("scale transport atom must contain complete 32-byte rows");
    }
    layout.a_scale_map_rows =
        static_cast<int>(layout.a_scale_bytes / kScalePhysicalRowBytes);
    layout.b_scale_map_rows =
        static_cast<int>(layout.b_scale_bytes / kScalePhysicalRowBytes);
    layout.a_scale_allocation_bytes =
        layout.a_scale_bytes * total_tiles;
    layout.b_scale_allocation_bytes =
        layout.b_scale_bytes * total_tiles;
  }
  layout.stage_bytes = layout.a_value_bytes + layout.b_value_bytes +
                       layout.a_scale_bytes + layout.b_scale_bytes;
  layout.allocation_bytes =
      layout.a_allocation_bytes + layout.b_allocation_bytes +
      layout.a_scale_allocation_bytes + layout.b_scale_allocation_bytes;
  return layout;
}

void encode_value_map(CUtensorMap* map, void* base,
                      const ValueLayout& layout, int height,
                      int total_tiles) {
  constexpr std::uint32_t kRank = 2;
  std::uint64_t global_dim[kRank] = {
      static_cast<std::uint64_t>(layout.row_stride_containers),
      static_cast<std::uint64_t>(height) * total_tiles,
  };
  std::uint64_t global_stride[kRank - 1] = {
      static_cast<std::uint64_t>(layout.row_stride_containers) *
      layout.container_bytes,
  };
  std::uint32_t box_dim[kRank] = {
      static_cast<std::uint32_t>(layout.request_x_containers),
      static_cast<std::uint32_t>(height),
  };
  std::uint32_t element_stride[kRank] = {1u, 1u};
  CU_CHECK(cuTensorMapEncodeTiled(
      map, layout.data_type, kRank, base, global_dim, global_stride,
      box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      layout.swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

void encode_scale_map(CUtensorMap* map, void* base, int map_rows,
                      int total_tiles) {
  constexpr std::uint32_t kRank = 2;
  constexpr std::uint64_t kRowStrideBytes = 32;
  std::uint64_t global_dim[kRank] = {
      kRowStrideBytes,
      static_cast<std::uint64_t>(map_rows) * total_tiles,
  };
  std::uint64_t global_stride[kRank - 1] = {kRowStrideBytes};
  std::uint32_t box_dim[kRank] = {
      static_cast<std::uint32_t>(kRowStrideBytes),
      static_cast<std::uint32_t>(map_rows),
  };
  std::uint32_t element_stride[kRank] = {1u, 1u};
  CU_CHECK(cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank, base, global_dim,
      global_stride, box_dim, element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

const char* swizzle_name(CUtensorMapSwizzle swizzle) {
  if (swizzle == CU_TENSOR_MAP_SWIZZLE_32B) return "32B";
  if (swizzle == CU_TENSOR_MAP_SWIZZLE_64B) return "64B";
  if (swizzle == CU_TENSOR_MAP_SWIZZLE_128B) return "128B";
  return "unknown";
}

void validate_options(const Options& options) {
  if (options.case_id.empty() || options.case_id == "unspecified") {
    fail("--case-id is required");
  }
  for (char character : options.case_id) {
    if (!(std::isalnum(static_cast<unsigned char>(character)) ||
          character == '.' || character == '_' || character == '-')) {
      fail("--case-id contains an invalid character");
    }
  }
  if (options.bm <= 0 || options.bn <= 0 || options.bk <= 0) {
    fail("BM, BN, and BK must be positive");
  }
  if (options.stages <= 0 || options.stages > kMaxStages ||
      (options.stages & (options.stages - 1)) != 0) {
    fail("--stages must be a power of two in [1, 8]");
  }
  if (options.threads <= 0 || options.threads > 1024 ||
      options.controller_thread < 0 ||
      options.controller_thread >= options.threads) {
    fail("thread/controller contract is invalid");
  }
  if (options.blocks < 0 || options.blocks_per_sm <= 0) {
    fail("block contract is invalid");
  }
  if (options.row_stride_elements < options.bk) {
    fail("row stride cannot be smaller than BK");
  }
  if (!(options.scale_block == 0 || options.scale_block == 16 ||
        options.scale_block == 32)) {
    fail("--scale-block must be 0, 16, or 32");
  }
  if (options.scale_block != 0 && options.value_bits != 4) {
    fail("block scale is currently defined only for packed 4-bit values");
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options options = parse_options(argc, argv);
  if (options.csv_header) {
    std::puts(
        "case_id,mode,bm,bn,bk,value_bits,scale_block,stages,"
        "requests_per_stage,value_chunks,value_swizzle,row_stride_elements,"
        "threads,controller_thread,blocks,sm_count,unique_smid_count,"
        "initialization,"
        "a_value_bytes,b_value_bytes,a_scale_bytes,b_scale_bytes,stage_bytes,"
        "dynamic_smem_bytes,total_tiles,"
        "working_set_bytes,allocation_bytes,iters,warmup_iters,requested_bytes,"
        "globaltimer_start_min_ns,globaltimer_stop_max_ns,"
        "globaltimer_elapsed_ns,bytes_per_second,occupancy_blocks_per_sm,sink");
    if (argc == 2) return 0;
  }
  validate_options(options);
  const ValueLayout values = make_value_layout(options);

  int sm_count = options.expected_sm_count;
  if (!options.contract_only) {
    CUDA_CHECK(cudaDeviceGetAttribute(
        &sm_count, cudaDevAttrMultiProcessorCount, 0));
  }
  const int blocks = options.blocks > 0
                         ? options.blocks
                         : sm_count * options.blocks_per_sm;
  if (options.mode == Mode::kL2Hit && blocks != 1) {
    fail("l2-hit contract requires exactly one CTA");
  }
  if (options.mode == Mode::kDramStream && blocks != sm_count) {
    fail("dram-stream contract requires exactly one CTA per SM");
  }

  BufferLayout one_tile = make_buffer_layout(options, values, 1);
  int total_tiles = static_cast<int>(std::max<std::size_t>(
      1, options.target_working_set_bytes / one_tile.stage_bytes));
  if (options.mode == Mode::kDramStream) {
    total_tiles = std::max(total_tiles, blocks);
  }
  const BufferLayout buffers =
      make_buffer_layout(options, values, total_tiles);
  const std::size_t working_set_bytes =
      static_cast<std::size_t>(total_tiles) * buffers.stage_bytes;
  if (buffers.stage_bytes >
      std::numeric_limits<std::uint32_t>::max()) {
    fail("stage payload exceeds mbarrier transaction-count range");
  }
  const int requests_per_stage =
      2 * values.chunks + (options.scale_block == 0 ? 0 : 2);
  const std::size_t dynamic_smem_bytes =
      static_cast<std::size_t>(options.stages) * buffers.stage_bytes;
  if (options.contract_only) {
    std::printf(
        "case_id=%s contract_only=1 mode=%s bm=%d bn=%d bk=%d "
        "value_bits=%d scale_block=%d stages=%d requests_per_stage=%d "
        "value_chunks=%d value_swizzle=%s row_stride_elements=%d "
        "threads=%d controller_thread=%d blocks=%d sm_count=%d "
        "initialization=cuda_memset_zero "
        "a_value_bytes=%zu b_value_bytes=%zu a_scale_bytes=%zu "
        "b_scale_bytes=%zu stage_bytes=%zu dynamic_smem_bytes=%zu "
        "working_set_bytes=%zu allocation_bytes=%zu total_tiles=%d "
        "iters=%d warmup_iters=%d\n",
        options.case_id.c_str(), mode_name(options.mode), options.bm,
        options.bn, options.bk, options.value_bits, options.scale_block,
        options.stages, requests_per_stage, values.chunks,
        swizzle_name(values.swizzle), options.row_stride_elements,
        options.threads, options.controller_thread, blocks, sm_count,
        buffers.a_value_bytes, buffers.b_value_bytes,
        buffers.a_scale_bytes, buffers.b_scale_bytes, buffers.stage_bytes,
        dynamic_smem_bytes, working_set_bytes, buffers.allocation_bytes,
        total_tiles, options.iters, options.warmup_iters);
    return 0;
  }

  unsigned char* data_a = nullptr;
  unsigned char* data_b = nullptr;
  unsigned char* scale_a = nullptr;
  unsigned char* scale_b = nullptr;
  CUDA_CHECK(cudaMalloc(&data_a, buffers.a_allocation_bytes));
  CUDA_CHECK(cudaMalloc(&data_b, buffers.b_allocation_bytes));
  if (options.scale_block != 0) {
    CUDA_CHECK(cudaMalloc(&scale_a, buffers.a_scale_allocation_bytes));
    CUDA_CHECK(cudaMalloc(&scale_b, buffers.b_scale_allocation_bytes));
  } else {
    scale_a = data_a;
    scale_b = data_b;
  }
  // Fault in and initialize every backing page before the kernel's own
  // warmup/timed regions.  This keeps first-touch page establishment out of
  // the reported TMA interval even when GEMM row strides make the allocation
  // much larger than the requested payload working set.
  CUDA_CHECK(cudaMemset(data_a, 0, buffers.a_allocation_bytes));
  CUDA_CHECK(cudaMemset(data_b, 0, buffers.b_allocation_bytes));
  if (options.scale_block != 0) {
    CUDA_CHECK(cudaMemset(scale_a, 0, buffers.a_scale_allocation_bytes));
    CUDA_CHECK(cudaMemset(scale_b, 0, buffers.b_scale_allocation_bytes));
  }

  CU_CHECK(cuInit(0));
  CUtensorMap map_a{};
  CUtensorMap map_b{};
  CUtensorMap map_scale_a{};
  CUtensorMap map_scale_b{};
  encode_value_map(&map_a, data_a, values, options.bm, total_tiles);
  encode_value_map(&map_b, data_b, values, options.bn, total_tiles);
  if (options.scale_block != 0) {
    encode_scale_map(&map_scale_a, scale_a, buffers.a_scale_map_rows,
                     total_tiles);
    encode_scale_map(&map_scale_b, scale_b, buffers.b_scale_map_rows,
                     total_tiles);
  } else {
    // The kernel does not dereference scale maps when scale bytes are zero.
    // Reuse two already valid descriptors instead of manufacturing a tiny
    // dummy tensor map whose legal minimum box can vary by architecture.
    map_scale_a = map_a;
    map_scale_b = map_b;
  }

  unsigned long long* device_cycles = nullptr;
  unsigned long long* device_start_ns = nullptr;
  unsigned long long* device_stop_ns = nullptr;
  std::uint32_t* device_smids = nullptr;
  std::uint32_t* device_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&device_cycles,
                        sizeof(unsigned long long) * blocks));
  CUDA_CHECK(cudaMalloc(&device_start_ns,
                        sizeof(unsigned long long) * blocks));
  CUDA_CHECK(cudaMalloc(&device_stop_ns,
                        sizeof(unsigned long long) * blocks));
  CUDA_CHECK(cudaMalloc(&device_smids, sizeof(std::uint32_t) * blocks));
  CUDA_CHECK(cudaMalloc(&device_sink, sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMemset(device_sink, 0, sizeof(std::uint32_t)));

  int max_optin_smem = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  if (dynamic_smem_bytes > static_cast<std::size_t>(max_optin_smem)) {
    fail("dynamic shared-memory contract exceeds device opt-in limit");
  }
  CUDA_CHECK(cudaFuncSetAttribute(
      tma_ab_contract_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(dynamic_smem_bytes)));
  CUDA_CHECK(cudaFuncSetAttribute(
      tma_ab_contract_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
  int occupancy_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy_blocks, tma_ab_contract_kernel, options.threads,
      dynamic_smem_bytes));
  if (occupancy_blocks <= 0) {
    fail("occupancy is zero for requested contract");
  }

  const int value_request_bytes_a =
      options.bm * values.request_row_bytes;
  const int value_request_bytes_b =
      options.bn * values.request_row_bytes;
  tma_ab_contract_kernel<<<blocks, options.threads, dynamic_smem_bytes>>>(
      map_a, map_b, map_scale_a, map_scale_b, options.bm, options.bn,
      values.request_x_containers, values.chunks, value_request_bytes_a,
      value_request_bytes_b, static_cast<int>(buffers.a_value_bytes),
      static_cast<int>(buffers.b_value_bytes),
      static_cast<int>(buffers.a_scale_bytes),
      static_cast<int>(buffers.b_scale_bytes), buffers.a_scale_map_rows,
      buffers.b_scale_map_rows, options.stages, options.controller_thread,
      total_tiles, options.iters, options.warmup_iters, device_cycles,
      device_start_ns, device_stop_ns, device_smids, device_sink);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unsigned long long> cycles(blocks);
  std::vector<unsigned long long> starts(blocks);
  std::vector<unsigned long long> stops(blocks);
  std::vector<std::uint32_t> smids(blocks);
  std::uint32_t sink = 0;
  CUDA_CHECK(cudaMemcpy(cycles.data(), device_cycles,
                        sizeof(unsigned long long) * blocks,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(starts.data(), device_start_ns,
                        sizeof(unsigned long long) * blocks,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(stops.data(), device_stop_ns,
                        sizeof(unsigned long long) * blocks,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(smids.data(), device_smids,
                        sizeof(std::uint32_t) * blocks,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&sink, device_sink, sizeof(sink),
                        cudaMemcpyDeviceToHost));

  const unsigned long long start_min =
      *std::min_element(starts.begin(), starts.end());
  const unsigned long long stop_max =
      *std::max_element(stops.begin(), stops.end());
  if (stop_max <= start_min) fail("invalid globaltimer interval");
  const unsigned long long elapsed_ns = stop_max - start_min;
  const unsigned long long requested_bytes =
      static_cast<unsigned long long>(blocks) * options.iters *
      buffers.stage_bytes;
  const double bytes_per_second =
      static_cast<double>(requested_bytes) * 1.0e9 /
      static_cast<double>(elapsed_ns);
  std::sort(smids.begin(), smids.end());
  const int unique_smid_count =
      static_cast<int>(std::unique(smids.begin(), smids.end()) -
                       smids.begin());
  if (options.csv) {
    std::printf(
        "%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%s,"
        "%zu,%zu,%zu,%zu,%zu,%zu,%d,%zu,%zu,%d,%d,%llu,%llu,%llu,%llu,%.9f,%d,%u\n",
        options.case_id.c_str(), mode_name(options.mode), options.bm,
        options.bn, options.bk, options.value_bits, options.scale_block,
        options.stages, requests_per_stage, values.chunks,
        swizzle_name(values.swizzle), options.row_stride_elements,
        options.threads, options.controller_thread, blocks, sm_count,
        unique_smid_count, "cuda_memset_zero", buffers.a_value_bytes,
        buffers.b_value_bytes,
        buffers.a_scale_bytes, buffers.b_scale_bytes, buffers.stage_bytes,
        dynamic_smem_bytes, total_tiles, working_set_bytes,
        buffers.allocation_bytes, options.iters,
        options.warmup_iters, requested_bytes, start_min, stop_max,
        elapsed_ns, bytes_per_second, occupancy_blocks, sink);
  } else {
    std::printf(
        "case_id=%s mode=%s bm=%d bn=%d bk=%d value_bits=%d "
        "scale_block=%d stages=%d requests_per_stage=%d value_chunks=%d "
        "value_swizzle=%s row_stride_elements=%d threads=%d "
        "controller_thread=%d blocks=%d sm_count=%d unique_smid_count=%d "
        "initialization=cuda_memset_zero "
        "a_value_bytes=%zu b_value_bytes=%zu a_scale_bytes=%zu "
        "b_scale_bytes=%zu stage_bytes=%zu dynamic_smem_bytes=%zu "
        "total_tiles=%d working_set_bytes=%zu allocation_bytes=%zu "
        "iters=%d warmup_iters=%d "
        "requested_bytes=%llu globaltimer_start_min_ns=%llu "
        "globaltimer_stop_max_ns=%llu globaltimer_elapsed_ns=%llu "
        "bytes_per_second=%.9f occupancy_blocks_per_sm=%d sink=%u\n",
        options.case_id.c_str(), mode_name(options.mode), options.bm,
        options.bn, options.bk, options.value_bits, options.scale_block,
        options.stages, requests_per_stage, values.chunks,
        swizzle_name(values.swizzle), options.row_stride_elements,
        options.threads, options.controller_thread, blocks, sm_count,
        unique_smid_count, buffers.a_value_bytes, buffers.b_value_bytes,
        buffers.a_scale_bytes, buffers.b_scale_bytes, buffers.stage_bytes,
        dynamic_smem_bytes, total_tiles, working_set_bytes,
        buffers.allocation_bytes, options.iters,
        options.warmup_iters, requested_bytes, start_min, stop_max,
        elapsed_ns, bytes_per_second, occupancy_blocks, sink);
  }

  CUDA_CHECK(cudaFree(device_sink));
  CUDA_CHECK(cudaFree(device_smids));
  CUDA_CHECK(cudaFree(device_stop_ns));
  CUDA_CHECK(cudaFree(device_start_ns));
  CUDA_CHECK(cudaFree(device_cycles));
  if (options.scale_block != 0) {
    CUDA_CHECK(cudaFree(scale_b));
    CUDA_CHECK(cudaFree(scale_a));
  }
  CUDA_CHECK(cudaFree(data_b));
  CUDA_CHECK(cudaFree(data_a));
  return 0;
}
