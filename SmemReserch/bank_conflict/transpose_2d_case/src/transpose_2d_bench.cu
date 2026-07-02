#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error_ = (call);                                          \
    if (error_ != cudaSuccess) {                                                \
      throw std::runtime_error(std::string(#call) + ": " +                      \
                               cudaGetErrorString(error_));                     \
    }                                                                           \
  } while (0)

constexpr int kRows = 32;
constexpr int kLanes = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kLanes * kWarps;
constexpr int kMaxPitch = 64;

struct Options {
  std::string case_name = "all";
  int iters = 100000;
  int warmups = 5;
  int repeats = 20;
  bool list_cases = false;
};

enum class AccessPattern {
  kTransposeScalar,
  kBroadcastSameAddr,
  kMulticast2Addr,
  kMulticast4Addr,
  kSameBankDifferentAddr,
  kTransposeVector,
  kXorSwizzle
};

struct CaseSpec {
  const char* experiment_cli;
  const char* experiment_name;
  const char* case_name;
  AccessPattern pattern;
  int pitch;
  int vector_width;
};

const std::vector<CaseSpec> kCases = {
    {"E0", "E0_basic_pitch_effect", "E0_load_pitch32",
     AccessPattern::kTransposeScalar, 32, 1},

    {"E1", "E1_pitch_sweep", "E1_load_pitch1",
     AccessPattern::kTransposeScalar, 1, 1},
    {"E1", "E1_pitch_sweep", "E1_load_pitch2",
     AccessPattern::kTransposeScalar, 2, 1},
    {"E1", "E1_pitch_sweep", "E1_load_pitch4",
     AccessPattern::kTransposeScalar, 4, 1},
    {"E1", "E1_pitch_sweep", "E1_load_pitch8",
     AccessPattern::kTransposeScalar, 8, 1},
    {"E1", "E1_pitch_sweep", "E1_load_pitch16",
     AccessPattern::kTransposeScalar, 16, 1},
    {"E1", "E1_pitch_sweep", "E1_load_pitch31",
     AccessPattern::kTransposeScalar, 31, 1},
    {"E1", "E1_pitch_sweep", "E1_load_pitch32",
     AccessPattern::kTransposeScalar, 32, 1},
    {"E1", "E1_pitch_sweep", "E1_load_pitch33",
     AccessPattern::kTransposeScalar, 33, 1},

    {"E2", "E2_broadcast_multicast", "E2_load_broadcast_same_addr",
     AccessPattern::kBroadcastSameAddr, 32, 1},
    {"E2", "E2_broadcast_multicast", "E2_load_multicast_2addr",
     AccessPattern::kMulticast2Addr, 32, 1},
    {"E2", "E2_broadcast_multicast", "E2_load_multicast_4addr",
     AccessPattern::kMulticast4Addr, 32, 1},
    {"E2", "E2_broadcast_multicast", "E2_load_conflict_same_bank_diff_addr",
     AccessPattern::kSameBankDifferentAddr, 32, 1},

    {"E3", "E3_vector_width", "E3_load_f32_pitch32",
     AccessPattern::kTransposeScalar, 32, 1},
    {"E3", "E3_vector_width", "E3_load_f32_pitch33",
     AccessPattern::kTransposeScalar, 33, 1},
    {"E3", "E3_vector_width", "E3_load_f32x2_pitch32",
     AccessPattern::kTransposeVector, 32, 2},
    {"E3", "E3_vector_width", "E3_load_f32x2_pitch33",
     AccessPattern::kTransposeVector, 33, 2},
    {"E3", "E3_vector_width", "E3_load_f32x4_pitch32",
     AccessPattern::kTransposeVector, 32, 4},
    {"E3", "E3_vector_width", "E3_load_f32x4_pitch33",
     AccessPattern::kTransposeVector, 33, 4},

    {"E4", "E4_software_swizzle", "E4_load_xor_swizzle_pitch32",
     AccessPattern::kXorSwizzle, 32, 1},
};

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ float load_shared_f32(const float* ptr) {
  float value;
  asm volatile("ld.volatile.shared.f32 %0, [%1];"
               : "=f"(value)
               : "r"(smem_addr(ptr))
               : "memory");
  return value;
}

__device__ __forceinline__ float2 load_shared_v2(const float* ptr) {
  float2 value;
  asm volatile("ld.volatile.shared.v2.f32 {%0, %1}, [%2];"
               : "=f"(value.x), "=f"(value.y)
               : "r"(smem_addr(ptr))
               : "memory");
  return value;
}

__device__ __forceinline__ float4 load_shared_v4(const float* ptr) {
  float4 value;
  asm volatile("ld.volatile.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
               : "r"(smem_addr(ptr))
               : "memory");
  return value;
}

__device__ __forceinline__ void initialize_tile(float* tile, int words) {
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = tid; i < words; i += kThreads) {
    tile[i] = static_cast<float>((i % 251) + 1);
  }
  __syncthreads();
}

template <int VectorWidth>
__device__ __forceinline__ int vector_base_col(int row, int warp, int pitch) {
  if constexpr (VectorWidth == 2) {
    const int base = warp * 2;
    return base + ((2 - ((row * pitch + base) & 1)) & 1);
  } else {
    const int group = pitch == 33 ? (warp % 7) : warp;
    const int base = group * 4;
    return base + ((4 - ((row * pitch + base) & 3)) & 3);
  }
}

template <AccessPattern Pattern, int VectorWidth>
__device__ __forceinline__ int base_index_for_lane(int lane, int warp,
                                                   int pitch) {
  if constexpr (Pattern == AccessPattern::kTransposeScalar) {
    return lane * pitch + warp;
  } else if constexpr (Pattern == AccessPattern::kBroadcastSameAddr) {
    return 0;
  } else if constexpr (Pattern == AccessPattern::kMulticast2Addr) {
    return lane < 16 ? 0 : 1;
  } else if constexpr (Pattern == AccessPattern::kMulticast4Addr) {
    return lane / 8;
  } else if constexpr (Pattern == AccessPattern::kSameBankDifferentAddr) {
    return lane * 32;
  } else if constexpr (Pattern == AccessPattern::kTransposeVector) {
    return lane * pitch + vector_base_col<VectorWidth>(lane, warp, pitch);
  } else {
    const int physical_col = warp ^ (lane & 31);
    return lane * pitch + physical_col;
  }
}

template <AccessPattern Pattern, int VectorWidth>
__global__ void load_kernel(float* result, int iters, int pitch) {
  __shared__ __align__(16) float tile[kRows * kMaxPitch];
  initialize_tile(tile, kRows * kMaxPitch);

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * kLanes + lane;
  const int base_index = base_index_for_lane<Pattern, VectorWidth>(lane, warp, pitch);
  const float* ptr = &tile[base_index];

  float accumulator = 0.0f;
  for (int i = 0; i < iters; ++i) {
    if constexpr (VectorWidth == 1) {
      accumulator += load_shared_f32(ptr);
    } else if constexpr (VectorWidth == 2) {
      const float2 value = load_shared_v2(ptr);
      accumulator += value.x + value.y;
    } else {
      const float4 value = load_shared_v4(ptr);
      accumulator += value.x + value.y + value.z + value.w;
    }
  }
  result[tid] = accumulator;
}

struct Theory {
  int unique_banks = 0;
  int conflict_degree = 0;
};

int host_vector_base_col(int row, int warp, int pitch, int vector_width) {
  if (vector_width == 2) {
    const int base = warp * 2;
    return base + ((2 - ((row * pitch + base) & 1)) & 1);
  }
  const int group = pitch == 33 ? (warp % 7) : warp;
  const int base = group * 4;
  return base + ((4 - ((row * pitch + base) & 3)) & 3);
}

std::vector<int> host_word_indices(const CaseSpec& spec, int lane, int warp) {
  std::vector<int> indices;
  indices.reserve(spec.vector_width);

  int base_index = 0;
  switch (spec.pattern) {
    case AccessPattern::kTransposeScalar:
      base_index = lane * spec.pitch + warp;
      break;
    case AccessPattern::kBroadcastSameAddr:
      base_index = 0;
      break;
    case AccessPattern::kMulticast2Addr:
      base_index = lane < 16 ? 0 : 1;
      break;
    case AccessPattern::kMulticast4Addr:
      base_index = lane / 8;
      break;
    case AccessPattern::kSameBankDifferentAddr:
      base_index = lane * 32;
      break;
    case AccessPattern::kTransposeVector:
      base_index =
          lane * spec.pitch +
          host_vector_base_col(lane, warp, spec.pitch, spec.vector_width);
      break;
    case AccessPattern::kXorSwizzle:
      base_index = lane * spec.pitch + (warp ^ (lane & 31));
      break;
  }

  for (int i = 0; i < spec.vector_width; ++i) {
    indices.push_back(base_index + i);
  }
  return indices;
}

Theory compute_theory(const CaseSpec& spec) {
  std::array<std::vector<int>, 32> bank_words;
  for (int lane = 0; lane < kLanes; ++lane) {
    for (int word : host_word_indices(spec, lane, 0)) {
      const int bank = word % 32;
      auto& words = bank_words[bank];
      if (std::find(words.begin(), words.end(), word) == words.end()) {
        words.push_back(word);
      }
    }
  }

  Theory theory;
  for (const auto& words : bank_words) {
    if (!words.empty()) {
      ++theory.unique_banks;
      theory.conflict_degree =
          std::max(theory.conflict_degree, static_cast<int>(words.size()));
    }
  }
  return theory;
}

const char* operation_name(int vector_width) {
  if (vector_width == 1) return "ld.shared.f32";
  if (vector_width == 2) return "ld.shared.v2.f32";
  return "ld.shared.v4.f32";
}

bool matches_selector(const std::string& selector, const CaseSpec& spec) {
  return selector == "all" || selector == spec.experiment_cli ||
         selector == spec.case_name;
}

template <AccessPattern Pattern>
void launch_pattern(const CaseSpec& spec, const Options& options,
                    float* device_result) {
  const dim3 block(kLanes, kWarps, 1);
  if (spec.vector_width == 1) {
    load_kernel<Pattern, 1><<<1, block>>>(device_result, options.iters, spec.pitch);
  } else if (spec.vector_width == 2) {
    load_kernel<Pattern, 2><<<1, block>>>(device_result, options.iters, spec.pitch);
  } else {
    load_kernel<Pattern, 4><<<1, block>>>(device_result, options.iters, spec.pitch);
  }
  CUDA_CHECK(cudaGetLastError());
}

void launch_case(const CaseSpec& spec, const Options& options,
                 float* device_result) {
  switch (spec.pattern) {
    case AccessPattern::kTransposeScalar:
      launch_pattern<AccessPattern::kTransposeScalar>(spec, options, device_result);
      break;
    case AccessPattern::kBroadcastSameAddr:
      launch_pattern<AccessPattern::kBroadcastSameAddr>(spec, options, device_result);
      break;
    case AccessPattern::kMulticast2Addr:
      launch_pattern<AccessPattern::kMulticast2Addr>(spec, options, device_result);
      break;
    case AccessPattern::kMulticast4Addr:
      launch_pattern<AccessPattern::kMulticast4Addr>(spec, options, device_result);
      break;
    case AccessPattern::kSameBankDifferentAddr:
      launch_pattern<AccessPattern::kSameBankDifferentAddr>(spec, options,
                                                            device_result);
      break;
    case AccessPattern::kTransposeVector:
      launch_pattern<AccessPattern::kTransposeVector>(spec, options, device_result);
      break;
    case AccessPattern::kXorSwizzle:
      launch_pattern<AccessPattern::kXorSwizzle>(spec, options, device_result);
      break;
  }
}

void print_measurement(const CaseSpec& spec, const Options& options,
                       float* device_result) {
  for (int i = 0; i < options.warmups; ++i) {
    launch_case(spec, options, device_result);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<float> elapsed_ms;
  elapsed_ms.reserve(options.repeats);
  for (int i = 0; i < options.repeats; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch_case(spec, options, device_result);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    elapsed_ms.push_back(milliseconds);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  const double avg_ms =
      std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0) /
      elapsed_ms.size();
  const double min_ms =
      *std::min_element(elapsed_ms.begin(), elapsed_ms.end());
  const double total_bytes = static_cast<double>(options.iters) * kThreads *
                             spec.vector_width * sizeof(float);
  const double effective_gbps = total_bytes / (avg_ms * 1.0e6);
  const Theory theory = compute_theory(spec);

  std::cout << spec.experiment_name << ',' << spec.case_name << ','
            << operation_name(spec.vector_width) << ',' << spec.pitch << ','
            << spec.vector_width << ',' << theory.unique_banks << ','
            << theory.conflict_degree << ',' << options.iters << ','
            << std::fixed << std::setprecision(6) << avg_ms << ',' << min_ms
            << ',' << std::setprecision(3) << effective_gbps << '\n';
}

int parse_integer(const char* value, const char* flag) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!*value || *end || parsed < 0 || parsed > 2147483647L) {
    throw std::invalid_argument(std::string("invalid value for ") + flag + ": " +
                                value);
  }
  return static_cast<int>(parsed);
}

void print_usage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " [--case all|E0|E1|E2|E3|E4|case_name] [--iters N]"
         " [--warmups N] [--repeats N] [--list-cases]\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--list-cases") {
      options.list_cases = true;
      continue;
    }
    if (++i >= argc) {
      throw std::invalid_argument("missing value after " + arg);
    }
    if (arg == "--case") {
      options.case_name = argv[i];
    } else if (arg == "--iters") {
      options.iters = parse_integer(argv[i], "--iters");
    } else if (arg == "--warmups") {
      options.warmups = parse_integer(argv[i], "--warmups");
    } else if (arg == "--repeats") {
      options.repeats = parse_integer(argv[i], "--repeats");
    } else {
      throw std::invalid_argument("unknown option: " + arg);
    }
  }
  if (options.iters < 1) {
    throw std::invalid_argument("--iters must be positive");
  }
  if (options.repeats < 1) {
    throw std::invalid_argument("--repeats must be positive");
  }
  return options;
}

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);

    std::vector<const CaseSpec*> selected;
    for (const auto& spec : kCases) {
      if (matches_selector(options.case_name, spec)) {
        selected.push_back(&spec);
      }
    }
    if (selected.empty()) {
      throw std::invalid_argument("unknown case selector: " + options.case_name);
    }

    if (options.list_cases) {
      for (const CaseSpec* spec : selected) {
        std::cout << spec->case_name << '\n';
      }
      return 0;
    }

    float* device_result = nullptr;
    CUDA_CHECK(cudaMalloc(&device_result, kThreads * sizeof(float)));

    std::cout << "experiment,case,operation,pitch,vector_width,"
                 "theoretical_unique_banks,theoretical_conflict_degree,"
                 "iters,avg_ms,min_ms,effective_GBps\n";
    for (const CaseSpec* spec : selected) {
      print_measurement(*spec, options, device_result);
    }

    CUDA_CHECK(cudaFree(device_result));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
