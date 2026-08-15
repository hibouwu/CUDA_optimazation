#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "sm110_ptx_helpers.cuh"

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    const cudaError_t error__ = (call);                                     \
    if (error__ != cudaSuccess) {                                           \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(error__));                            \
      std::exit(3);                                                         \
    }                                                                       \
  } while (0)

namespace {

namespace ptx = gemm_sm110::ptx;

constexpr int kTileM = 128;
constexpr int kTileN = 256;
constexpr int kTileK = 64;
constexpr int kMmaK = 16;
constexpr int kAStageBytes = kTileM * kTileK * sizeof(half);
constexpr int kBStageBytes = kTileN * kTileK * sizeof(half);
constexpr int kStageBytes = kAStageBytes + kBStageBytes;
constexpr int kOutputBytesPerTask = kTileM * kTileN * sizeof(float);
constexpr int kEpilogueWarps = 4;
constexpr int kTmaWarp = 4;
constexpr int kMmaWarp = 5;
constexpr int kThreads = 192;
constexpr int kAccumulatorBuffers = 2;
constexpr int kMaxStages = 4;
constexpr int kMaxKTiles = 64;
constexpr int kMaxOutputTasks = 32;

enum class Mode : int {
  kTmaOnly = 0,
  kMmaOnly = 1,
  kOverlap = 2,
  kFull = 3,
};

struct Options {
  std::string case_id;
  Mode mode = Mode::kFull;
  int stages = 4;
  int k_tiles = 32;
  int output_tasks = 1;
  int warmup_launches = 3;
  int expected_sm_count = 20;
  bool contract_only = false;
  bool allow_non_sm110 = false;
  bool csv = false;
  bool csv_header = false;
};

struct Trace {
  std::uint64_t start_ns;
  std::uint64_t first_tma_done_ns;
  std::uint64_t last_tma_done_ns;
  std::uint64_t first_mma_done_ns;
  std::uint64_t last_mma_done_ns;
  std::uint64_t first_epilogue_start_ns;
  std::uint64_t last_store_done_ns;
  std::uint64_t kernel_exit_ns;
  std::uint32_t smid;
  std::uint32_t mode;
  std::uint32_t stages;
  std::uint32_t k_tiles;
  std::uint32_t output_tasks;
};

__device__ __forceinline__ std::uint64_t global_nanoseconds() {
  std::uint64_t value = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ std::uint32_t current_smid() {
  std::uint32_t value = 0;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  return value;
}

template <int Stages>
__global__ __launch_bounds__(kThreads)
void tc5a_pipeline_dag_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b, float* output,
    int k_tiles, int output_tasks, Mode mode, Trace* trace) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  static_assert(Stages == 1 || Stages == 2 || Stages == 4);
  constexpr std::uint32_t kInstructionDescriptor =
      (1U << 4U) |
      (static_cast<std::uint32_t>(kTileN) >> 3U << 17U) |
      (static_cast<std::uint32_t>(kTileM) >> 4U << 24U);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / ptx::kWarpSize;
  const int lane = tid % ptx::kWarpSize;
  const bool needs_tma = mode != Mode::kMmaOnly;
  const bool needs_mma = mode != Mode::kTmaOnly;
  const bool needs_epilogue = mode == Mode::kFull;
  const int total_k_operations = k_tiles * output_tasks;

  extern __shared__ __align__(1024) unsigned char dynamic_smem[];
  const std::uint32_t smem = ptx::smem_address(dynamic_smem);

  __shared__ alignas(16) std::uint64_t tma_barrier[Stages];
  __shared__ alignas(16) std::uint64_t mma_barrier[Stages];
  __shared__ alignas(16) std::uint64_t mainloop_barrier[kAccumulatorBuffers];
  __shared__ alignas(16) std::uint64_t epilogue_barrier[kAccumulatorBuffers];
  __shared__ alignas(16) std::uint32_t tmem_base;

  const std::uint32_t tma_barrier_base = ptx::smem_address(tma_barrier);
  const std::uint32_t mma_barrier_base = ptx::smem_address(mma_barrier);
  const std::uint32_t mainloop_barrier_base =
      ptx::smem_address(mainloop_barrier);
  const std::uint32_t epilogue_barrier_base =
      ptx::smem_address(epilogue_barrier);

  if (warp == kMmaWarp && ptx::elect_one()) {
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
      ptx::mbarrier_init(tma_barrier_base + stage * sizeof(std::uint64_t), 1);
      ptx::mbarrier_init(mma_barrier_base + stage * sizeof(std::uint64_t), 1);
    }
#pragma unroll
    for (int buffer = 0; buffer < kAccumulatorBuffers; ++buffer) {
      ptx::mbarrier_init(
          mainloop_barrier_base + buffer * sizeof(std::uint64_t), 1);
      ptx::mbarrier_init(
          epilogue_barrier_base + buffer * sizeof(std::uint64_t),
          kEpilogueWarps);
    }
    ptx::fence_mbarrier_init_release_cluster();
  }
  if (needs_mma && warp == kMmaWarp) {
    ptx::tmem_alloc(ptx::smem_address(&tmem_base),
                    kTileN * kAccumulatorBuffers);
  }
  if (!needs_tma) {
    for (int byte = tid * static_cast<int>(sizeof(std::uint32_t));
         byte < Stages * kStageBytes;
         byte += static_cast<int>(blockDim.x) *
                 static_cast<int>(sizeof(std::uint32_t))) {
      *reinterpret_cast<std::uint32_t*>(dynamic_smem + byte) = 0U;
    }
  }
  __syncthreads();

  if (tid == 0) {
    trace->start_ns = global_nanoseconds();
    trace->smid = current_smid();
    trace->mode = static_cast<std::uint32_t>(mode);
    trace->stages = Stages;
    trace->k_tiles = static_cast<std::uint32_t>(k_tiles);
    trace->output_tasks = static_cast<std::uint32_t>(output_tasks);
  }
  __syncthreads();

  auto issue_load = [&](int output_task, int k_tile, int stage) {
    (void)output_task;
    const std::uint32_t barrier =
        tma_barrier_base + stage * sizeof(std::uint64_t);
    const std::uint32_t stage_smem = smem + stage * kStageBytes;
    const std::uint32_t a_smem = stage_smem;
    const std::uint32_t b_smem = stage_smem + kAStageBytes;
    const int offset_k = k_tile * kTileK;
    // Reuse the same complete A/B K range across persistent output tasks.
    // Warm-up launches make that bounded range hot in L2.  This campaign
    // therefore isolates causal pipeline timing; unique-address and
    // cold-DRAM service are measured by the separate resource campaign.
    const int offset_m = 0;
    ptx::tma_load_2d(a_smem, &tensor_map_a, offset_k, offset_m, barrier);
    ptx::tma_load_2d(b_smem, &tensor_map_b, offset_k, 0, barrier);
    ptx::mbarrier_arrive_expect_tx(barrier, kStageBytes);
  };

  auto issue_mma = [&](int k_tile, int stage, int accumulator_buffer) {
    const std::uint32_t stage_smem = smem + stage * kStageBytes;
    const std::uint32_t a_smem = stage_smem;
    const std::uint32_t b_smem = stage_smem + kAStageBytes;
    const std::uint32_t accumulator =
        tmem_base + accumulator_buffer * kTileN;
#pragma unroll
    for (int k_block = 0; k_block < kTileK / kMmaK; ++k_block) {
      const std::uint32_t a_block = a_smem + k_block * 32;
      const std::uint32_t b_block = b_smem + k_block * 32;
      const std::uint64_t descriptor_a =
          ptx::sw128_k_major_descriptor(a_block);
      const std::uint64_t descriptor_b =
          ptx::sw128_k_major_descriptor(b_block);
      ptx::mma_f16(accumulator, descriptor_a, descriptor_b,
                   kInstructionDescriptor,
                   k_tile != 0 || k_block != 0);
    }
  };

  auto epilogue_sync = []() {
    asm volatile("bar.sync %0, %1;"
                 :
                 : "r"(1), "r"(kEpilogueWarps * ptx::kWarpSize)
                 : "memory");
  };

  auto store_output_task = [&](int output_task, int accumulator_buffer) {
    const int row = warp * ptx::kWarpSize + lane;
    const std::uint32_t base =
        tmem_base + accumulator_buffer * kTileN + (row << 16);
    float* row_output =
        output + static_cast<std::size_t>(output_task * kTileM + row) *
                     kTileN;
    float even[8];
    float odd[8];
    ptx::tmem_load_32x32b_x8_no_wait(base, even);
    for (int block = 0; block < kTileN / 8; ++block) {
      ptx::tmem_load_wait();
      const bool use_even = (block & 1) == 0;
      if (block + 1 < kTileN / 8) {
        if (use_even) {
          ptx::tmem_load_32x32b_x8_no_wait(base + (block + 1) * 8, odd);
        } else {
          ptx::tmem_load_32x32b_x8_no_wait(base + (block + 1) * 8, even);
        }
      }
      if (use_even) {
        ptx::store_global_l1_no_allocate_v8_f32(row_output + block * 8,
                                                 even);
      } else {
        ptx::store_global_l1_no_allocate_v8_f32(row_output + block * 8,
                                                 odd);
      }
    }
  };

  if (mode == Mode::kTmaOnly) {
    if (warp == kTmaWarp && ptx::elect_one()) {
      int stage = 0;
      int release_phase = 1;
      for (int operation = 0; operation < total_k_operations; ++operation) {
        ptx::mbarrier_wait(
            mma_barrier_base + stage * sizeof(std::uint64_t),
            release_phase);
        issue_load(operation / k_tiles, operation % k_tiles, stage);
        stage = (stage + 1) % Stages;
        if (stage == 0) release_phase ^= 1;
      }
    } else if (warp == kMmaWarp && ptx::elect_one()) {
      int stage = 0;
      int completion_phase = 0;
      for (int operation = 0; operation < total_k_operations; ++operation) {
        ptx::mbarrier_wait(
            tma_barrier_base + stage * sizeof(std::uint64_t),
            completion_phase);
        const std::uint64_t now = global_nanoseconds();
        if (operation == 0) trace->first_tma_done_ns = now;
        if (operation + 1 == total_k_operations) trace->last_tma_done_ns = now;
        ptx::mbarrier_arrive(
            mma_barrier_base + stage * sizeof(std::uint64_t));
        stage = (stage + 1) % Stages;
        if (stage == 0) completion_phase ^= 1;
      }
    }
  } else if (mode == Mode::kMmaOnly) {
    if (warp == kMmaWarp && ptx::elect_one()) {
      int stage = 0;
      int release_phase = 1;
      for (int operation = 0; operation < total_k_operations; ++operation) {
        ptx::mbarrier_wait(
            tma_barrier_base + stage * sizeof(std::uint64_t),
            release_phase);
        issue_mma(operation % k_tiles, stage, 0);
        ptx::mma_commit(
            mma_barrier_base + stage * sizeof(std::uint64_t));
        stage = (stage + 1) % Stages;
        if (stage == 0) release_phase ^= 1;
      }
    } else if (warp == 0 && ptx::elect_one()) {
      int stage = 0;
      int completion_phase = 0;
      for (int operation = 0; operation < total_k_operations; ++operation) {
        ptx::mbarrier_wait(
            mma_barrier_base + stage * sizeof(std::uint64_t),
            completion_phase);
        const std::uint64_t now = global_nanoseconds();
        if (operation == 0) trace->first_mma_done_ns = now;
        if (operation + 1 == total_k_operations) trace->last_mma_done_ns = now;
        ptx::mbarrier_arrive(
            tma_barrier_base + stage * sizeof(std::uint64_t));
        stage = (stage + 1) % Stages;
        if (stage == 0) completion_phase ^= 1;
      }
    }
  } else {
    if (warp == kTmaWarp && ptx::elect_one()) {
      int stage = 0;
      int mma_phase = 1;
      for (int output_task = 0; output_task < output_tasks; ++output_task) {
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
          ptx::mbarrier_wait(
              mma_barrier_base + stage * sizeof(std::uint64_t), mma_phase);
          issue_load(output_task, k_tile, stage);
          stage = (stage + 1) % Stages;
          if (stage == 0) mma_phase ^= 1;
        }
      }
    } else if (warp == kMmaWarp && ptx::elect_one()) {
      int stage = 0;
      int tma_phase = 0;
      int accumulator_buffer = 0;
      int epilogue_phase = 1;
      int operation = 0;
      for (int output_task = 0; output_task < output_tasks; ++output_task) {
        if (needs_epilogue) {
          ptx::mbarrier_wait(
              epilogue_barrier_base +
                  accumulator_buffer * sizeof(std::uint64_t),
              epilogue_phase);
        }
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile, ++operation) {
          ptx::mbarrier_wait(
              tma_barrier_base + stage * sizeof(std::uint64_t), tma_phase);
          const std::uint64_t now = global_nanoseconds();
          if (operation == 0) trace->first_tma_done_ns = now;
          if (operation + 1 == total_k_operations) {
            trace->last_tma_done_ns = now;
          }
          ptx::tcgen05_fence_after_thread_sync();
          issue_mma(k_tile, stage, accumulator_buffer);
          ptx::mma_commit(
              mma_barrier_base + stage * sizeof(std::uint64_t));
          stage = (stage + 1) % Stages;
          if (stage == 0) tma_phase ^= 1;
        }
        ptx::mma_commit(
            mainloop_barrier_base +
            accumulator_buffer * sizeof(std::uint64_t));
        accumulator_buffer ^= 1;
        if (accumulator_buffer == 0) epilogue_phase ^= 1;
      }
    } else if (warp == 0 && ptx::elect_one()) {
      // This waiter is live before stage 0 can be reused, so the first MMA
      // completion timestamp cannot be confused with a later parity epoch.
      ptx::mbarrier_wait(mma_barrier_base, 0);
      trace->first_mma_done_ns = global_nanoseconds();
    }

    if (needs_epilogue && warp < kEpilogueWarps) {
      int accumulator_buffer = 0;
      int mainloop_phase = 0;
      for (int output_task = 0; output_task < output_tasks; ++output_task) {
        if (warp == 0 && ptx::elect_one()) {
          ptx::mbarrier_wait(
              mainloop_barrier_base +
                  accumulator_buffer * sizeof(std::uint64_t),
              mainloop_phase);
          const std::uint64_t now = global_nanoseconds();
          if (output_task == 0) {
            trace->first_epilogue_start_ns = now;
          }
          if (output_task + 1 == output_tasks) {
            trace->last_mma_done_ns = now;
          }
        }
        epilogue_sync();
        ptx::tcgen05_fence_after_thread_sync();
        store_output_task(output_task, accumulator_buffer);
        epilogue_sync();
        if (warp == 0 && ptx::elect_one() &&
            output_task + 1 == output_tasks) {
          trace->last_store_done_ns = global_nanoseconds();
        }
        if (ptx::elect_one()) {
          ptx::mbarrier_arrive(
              epilogue_barrier_base +
              accumulator_buffer * sizeof(std::uint64_t));
        }
        accumulator_buffer ^= 1;
        if (accumulator_buffer == 0) mainloop_phase ^= 1;
      }
    } else if (!needs_epilogue && warp == 0 && ptx::elect_one()) {
      ptx::mbarrier_wait(mainloop_barrier_base, 0);
      trace->last_mma_done_ns = global_nanoseconds();
    }
  }

  __syncthreads();
  if (tid == 0) {
    if (mode == Mode::kTmaOnly) {
      trace->kernel_exit_ns = trace->last_tma_done_ns;
    } else if (mode == Mode::kFull) {
      trace->kernel_exit_ns = trace->last_store_done_ns;
    } else {
      trace->kernel_exit_ns = trace->last_mma_done_ns;
    }
  }
  __syncthreads();
  if (needs_mma && warp == kMmaWarp) {
    ptx::tmem_dealloc(tmem_base, kTileN * kAccumulatorBuffers);
  }
#else
  (void)tensor_map_a;
  (void)tensor_map_b;
  (void)output;
  (void)k_tiles;
  (void)output_tasks;
  (void)mode;
  (void)trace;
#endif
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
    std::fprintf(stderr, "%s must be a positive integer\n", name);
    std::exit(2);
  }
  return static_cast<int>(value);
}

const char* mode_name(Mode mode) {
  switch (mode) {
    case Mode::kTmaOnly:
      return "tma-only";
    case Mode::kMmaOnly:
      return "mma-only";
    case Mode::kOverlap:
      return "overlap";
    case Mode::kFull:
      return "full";
  }
  return "unknown";
}

Mode parse_mode(const char* text) {
  if (std::strcmp(text, "tma-only") == 0) return Mode::kTmaOnly;
  if (std::strcmp(text, "mma-only") == 0) return Mode::kMmaOnly;
  if (std::strcmp(text, "overlap") == 0) return Mode::kOverlap;
  if (std::strcmp(text, "full") == 0) return Mode::kFull;
  fail("--mode must be tma-only, mma-only, overlap, or full");
}

void usage(const char* program) {
  std::fprintf(
      stderr,
      "Usage: %s --case-id ID --mode MODE --stages 1|2|4 "
      "--k-tiles 1|2|4|8|16|32|64 --output-tasks 1|2|4|8|16|32 "
      "[--warmup-launches N] [--expected-sm-count N] "
      "[--contract-only] [--allow-non-sm110] [--csv] [--csv-header]\n",
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
    } else if (std::strcmp(argument, "--stages") == 0) {
      options.stages = parse_positive(value(), "--stages");
    } else if (std::strcmp(argument, "--k-tiles") == 0) {
      options.k_tiles = parse_positive(value(), "--k-tiles");
    } else if (std::strcmp(argument, "--output-tasks") == 0) {
      options.output_tasks = parse_positive(value(), "--output-tasks");
    } else if (std::strcmp(argument, "--warmup-launches") == 0) {
      options.warmup_launches = parse_positive(value(), "--warmup-launches");
    } else if (std::strcmp(argument, "--expected-sm-count") == 0) {
      options.expected_sm_count =
          parse_positive(value(), "--expected-sm-count");
    } else if (std::strcmp(argument, "--contract-only") == 0) {
      options.contract_only = true;
    } else if (std::strcmp(argument, "--allow-non-sm110") == 0) {
      options.allow_non_sm110 = true;
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

bool in_power_two_set(int value, int maximum) {
  return value > 0 && value <= maximum && (value & (value - 1)) == 0;
}

void validate_options(const Options& options) {
  if (options.case_id.empty()) fail("--case-id is required");
  for (char character : options.case_id) {
    if (!(std::isalnum(static_cast<unsigned char>(character)) ||
          character == '.' || character == '_' || character == '-')) {
      fail("--case-id contains an invalid character");
    }
  }
  if (!in_power_two_set(options.stages, kMaxStages)) {
    fail("--stages must be 1, 2, or 4");
  }
  if (!in_power_two_set(options.k_tiles, kMaxKTiles)) {
    fail("--k-tiles must be 1, 2, 4, 8, 16, 32, or 64");
  }
  if (!in_power_two_set(options.output_tasks, kMaxOutputTasks)) {
    fail("--output-tasks must be 1, 2, 4, 8, 16, or 32");
  }
  if (options.mode != Mode::kFull && options.output_tasks != 1) {
    fail("only full mode may sweep multiple persistent output tasks");
  }
  if (options.mode == Mode::kMmaOnly && options.stages != 4) {
    fail("mma-only uses the four-entry completion ring contract");
  }
}

template <int Stages>
void configure_and_launch(const Options& options, const CUtensorMap& map_a,
                          const CUtensorMap& map_b, float* output,
                          Trace* trace, cudaStream_t stream) {
  auto* kernel = &tc5a_pipeline_dag_kernel<Stages>;
  constexpr int kDynamicSmemBytes = Stages * kStageBytes;
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kDynamicSmemBytes));
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
  kernel<<<1, kThreads, kDynamicSmemBytes, stream>>>(
      map_a, map_b, output, options.k_tiles, options.output_tasks,
      options.mode, trace);
  CUDA_CHECK(cudaGetLastError());
}

void launch(const Options& options, const CUtensorMap& map_a,
            const CUtensorMap& map_b, float* output, Trace* trace,
            cudaStream_t stream = nullptr) {
  if (options.stages == 1) {
    configure_and_launch<1>(options, map_a, map_b, output, trace, stream);
  } else if (options.stages == 2) {
    configure_and_launch<2>(options, map_a, map_b, output, trace, stream);
  } else {
    configure_and_launch<4>(options, map_a, map_b, output, trace, stream);
  }
}

std::uint64_t elapsed(std::uint64_t start, std::uint64_t stop) {
  if (start == 0 || stop < start) return 0;
  return stop - start;
}

void print_header() {
  std::puts(
      "case_id,precision_id,mode,stages,k_tiles,output_tasks,"
      "total_k_operations,threads,"
      "tma_requests_per_k_tile,a_bytes_per_k_tile,b_bytes_per_k_tile,"
      "payload_bytes_per_k_tile,mma_instructions_per_k_tile,"
      "accumulator_buffers,output_bytes_per_task,dynamic_smem_bytes,"
      "residency,initialization,warmup_launches,sm_count,smid,"
      "start_ns,first_tma_done_ns,last_tma_done_ns,first_mma_done_ns,"
      "last_mma_done_ns,first_epilogue_start_ns,last_store_done_ns,"
      "kernel_exit_ns,first_tma_latency_ns,tma_completion_span_ns,"
      "tma_interval_ns,first_mma_latency_ns,mma_completion_span_ns,"
      "mma_interval_ns,epilogue_to_store_ns,last_mma_to_store_ns,"
      "total_measured_ns");
}

void print_row(const Options& options, int sm_count, const Trace& trace) {
  const int total_operations = options.k_tiles * options.output_tasks;
  const std::uint64_t first_tma = elapsed(trace.start_ns,
                                          trace.first_tma_done_ns);
  const std::uint64_t tma_span = elapsed(trace.first_tma_done_ns,
                                        trace.last_tma_done_ns);
  const double tma_interval =
      total_operations > 1 && tma_span != 0
          ? static_cast<double>(tma_span) / (total_operations - 1)
          : 0.0;
  const std::uint64_t first_mma = elapsed(trace.start_ns,
                                          trace.first_mma_done_ns);
  const std::uint64_t mma_span = elapsed(trace.first_mma_done_ns,
                                        trace.last_mma_done_ns);
  const double mma_interval =
      total_operations > 1 && mma_span != 0
          ? static_cast<double>(mma_span) / (total_operations - 1)
          : 0.0;
  const std::uint64_t epilogue = elapsed(trace.first_epilogue_start_ns,
                                        trace.last_store_done_ns);
  const std::uint64_t last_mma_to_store = elapsed(
      trace.last_mma_done_ns, trace.last_store_done_ns);
  const std::uint64_t total = elapsed(trace.start_ns, trace.kernel_exit_ns);
  std::printf(
      "%s,fp16_f32,%s,%d,%d,%d,%d,%d,2,%d,%d,%d,4,%d,%d,%d,hot_l2,"
      "cuda_memset_zero,%d,%d,%u,"
      "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
      "%llu,%llu,%.9f,%llu,%llu,%.9f,%llu,%llu,%llu\n",
      options.case_id.c_str(), mode_name(options.mode), options.stages,
      options.k_tiles, options.output_tasks, total_operations, kThreads,
      kAStageBytes, kBStageBytes, kStageBytes, kAccumulatorBuffers,
      kOutputBytesPerTask, options.stages * kStageBytes,
      options.warmup_launches, sm_count, trace.smid,
      static_cast<unsigned long long>(trace.start_ns),
      static_cast<unsigned long long>(trace.first_tma_done_ns),
      static_cast<unsigned long long>(trace.last_tma_done_ns),
      static_cast<unsigned long long>(trace.first_mma_done_ns),
      static_cast<unsigned long long>(trace.last_mma_done_ns),
      static_cast<unsigned long long>(trace.first_epilogue_start_ns),
      static_cast<unsigned long long>(trace.last_store_done_ns),
      static_cast<unsigned long long>(trace.kernel_exit_ns),
      static_cast<unsigned long long>(first_tma),
      static_cast<unsigned long long>(tma_span), tma_interval,
      static_cast<unsigned long long>(first_mma),
      static_cast<unsigned long long>(mma_span), mma_interval,
      static_cast<unsigned long long>(epilogue),
      static_cast<unsigned long long>(last_mma_to_store),
      static_cast<unsigned long long>(total));
}

void validate_trace(const Options& options, const Trace& trace) {
  if (trace.start_ns == 0 || trace.kernel_exit_ns < trace.start_ns ||
      trace.mode != static_cast<std::uint32_t>(options.mode) ||
      trace.stages != static_cast<std::uint32_t>(options.stages) ||
      trace.k_tiles != static_cast<std::uint32_t>(options.k_tiles) ||
      trace.output_tasks != static_cast<std::uint32_t>(options.output_tasks)) {
    fail("runtime trace header or total interval is invalid");
  }
  if (options.mode != Mode::kMmaOnly &&
      !(trace.start_ns <= trace.first_tma_done_ns &&
        trace.first_tma_done_ns <= trace.last_tma_done_ns)) {
    fail("TMA completion timestamps are not monotonic");
  }
  if (options.mode != Mode::kTmaOnly &&
      !(trace.start_ns <= trace.first_mma_done_ns &&
        trace.first_mma_done_ns <= trace.last_mma_done_ns)) {
    fail("MMA completion timestamps are not monotonic");
  }
  if (options.mode == Mode::kFull &&
      !(trace.first_mma_done_ns <= trace.first_epilogue_start_ns &&
        trace.first_epilogue_start_ns <= trace.last_store_done_ns &&
        trace.last_store_done_ns == trace.kernel_exit_ns)) {
    fail("full-pipeline epilogue timestamps are not monotonic");
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options options = parse_options(argc, argv);
  if (options.csv_header) {
    print_header();
    if (argc == 2) return 0;
  }
  validate_options(options);

  if (options.contract_only) {
    Trace trace{};
    trace.mode = static_cast<std::uint32_t>(options.mode);
    trace.stages = static_cast<std::uint32_t>(options.stages);
    trace.k_tiles = static_cast<std::uint32_t>(options.k_tiles);
    trace.output_tasks = static_cast<std::uint32_t>(options.output_tasks);
    if (options.csv) print_row(options, options.expected_sm_count, trace);
    return 0;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  if (properties.multiProcessorCount != options.expected_sm_count) {
    fail("runtime SM count does not match --expected-sm-count");
  }
  if (!options.allow_non_sm110 &&
      !(properties.major == 11 && properties.minor == 0)) {
    fail("hardware evidence requires an SM110 device");
  }

  const int k_elements = options.k_tiles * kTileK;
  const std::size_t a_elements =
      static_cast<std::size_t>(kTileM) * k_elements;
  const std::size_t b_elements =
      static_cast<std::size_t>(kTileN) * k_elements;
  const std::size_t output_elements =
      static_cast<std::size_t>(options.output_tasks) * kTileM * kTileN;

  half* a = nullptr;
  half* b = nullptr;
  float* output = nullptr;
  Trace* device_trace = nullptr;
  CUDA_CHECK(cudaMalloc(&a, a_elements * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&b, b_elements * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&output, output_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_trace, sizeof(Trace)));
  CUDA_CHECK(cudaMemset(a, 0, a_elements * sizeof(half)));
  CUDA_CHECK(cudaMemset(b, 0, b_elements * sizeof(half)));
  CUDA_CHECK(cudaMemset(output, 0, output_elements * sizeof(float)));

  CUtensorMap map_a{};
  CUtensorMap map_b{};
  ptx::encode_tiled_2d_sw128_strided(
      &map_a, a, kTileM, k_elements, k_elements,
      kTileM);
  ptx::encode_tiled_2d_sw128_strided(
      &map_b, b, kTileN, k_elements, k_elements, kTileN);

  for (int warmup = 0; warmup < options.warmup_launches; ++warmup) {
    CUDA_CHECK(cudaMemset(device_trace, 0, sizeof(Trace)));
    launch(options, map_a, map_b, output, device_trace);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemset(device_trace, 0, sizeof(Trace)));
  launch(options, map_a, map_b, output, device_trace);
  CUDA_CHECK(cudaDeviceSynchronize());

  Trace trace{};
  CUDA_CHECK(cudaMemcpy(&trace, device_trace, sizeof(Trace),
                        cudaMemcpyDeviceToHost));
  validate_trace(options, trace);
  if (options.csv) print_row(options, properties.multiProcessorCount, trace);

  CUDA_CHECK(cudaFree(device_trace));
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(a));
  return 0;
}
