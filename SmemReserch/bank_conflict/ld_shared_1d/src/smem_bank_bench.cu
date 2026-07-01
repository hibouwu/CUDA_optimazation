#include "smem_bank_bench.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error_ = (call);                                                 \
    if (error_ != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string(#call) + ": " +                       \
                               cudaGetErrorString(error_));                       \
    }                                                                            \
  } while (0)

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
  unsigned addr;
  asm volatile("{ .reg .u64 uaddr;\n"
               "  cvta.to.shared.u64 uaddr, %1;\n"
               "  cvt.u32.u64 %0, uaddr;\n"
               "}"
               : "=r"(addr)
               : "l"(ptr));
  return addr;
}

__device__ __forceinline__ float load_shared_f32(const float* ptr) {
  float value;
  const unsigned addr = smem_addr(ptr);
  asm volatile("ld.volatile.shared.f32 %0, [%1];"
               : "=f"(value)
               : "r"(addr)
               : "memory");
  return value;
}

__device__ __forceinline__ float2 load_shared_v2(const float* ptr) {
  float2 value;
  const unsigned addr = smem_addr(ptr);
  asm volatile("ld.volatile.shared.v2.f32 {%0, %1}, [%2];"
               : "=f"(value.x), "=f"(value.y)
               : "r"(addr)
               : "memory");
  return value;
}

__device__ __forceinline__ float4 load_shared_v4(const float* ptr) {
  float4 value;
  const unsigned addr = smem_addr(ptr);
  asm volatile("ld.volatile.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
               : "r"(addr)
               : "memory");
  return value;
}

template <int N>
__device__ __forceinline__ void initialize_shared(float (&s)[N]) {
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = tid; i < N; i += blockDim.x * blockDim.y) {
    s[i] = static_cast<float>((i % 251) + 1);
  }
  __syncthreads();
}

__global__ void baseline_unique_banks(float* result, int num_iters) {
  __shared__ float s[kWarps * kLanes];
  initialize_shared(s);
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float* ptr = &s[warp * kLanes + lane];
  float accumulator = 0.0f;
  for (int i = 0; i < num_iters; ++i) {
    accumulator += load_shared_f32(ptr);
  }
  result[warp * kLanes + lane] = accumulator;
}

__global__ void stride_conflict_sweep(float* result, int num_iters, int stride,
                                      int offset) {
  __shared__ float s[1024];
  initialize_shared(s);
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float* ptr = &s[lane * stride + offset];
  float accumulator = 0.0f;
  for (int i = 0; i < num_iters; ++i) {
    accumulator += load_shared_f32(ptr);
  }
  result[warp * kLanes + lane] = accumulator;
}

__global__ void same_bank_32way_2d(float* result, int num_iters) {
  // Row-major [32][32]: &s[lane][0] has linear index lane * 32.
  __shared__ float s[kLanes][kLanes];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  float* linear = &s[0][0];
  for (int i = tid; i < kLanes * kLanes; i += kThreads) {
    linear[i] = static_cast<float>((i % 251) + 1);
  }
  __syncthreads();
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float* ptr = &s[lane][0];
  float accumulator = 0.0f;
  for (int i = 0; i < num_iters; ++i) {
    accumulator += load_shared_f32(ptr);
  }
  result[warp * kLanes + lane] = accumulator;
}

__global__ void broadcast_same_address(float* result, int num_iters) {
  __shared__ float s[kWarps * kLanes];
  initialize_shared(s);
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float* ptr = &s[warp * kLanes];
  float accumulator = 0.0f;
  for (int i = 0; i < num_iters; ++i) {
    accumulator += load_shared_f32(ptr);
  }
  result[warp * kLanes + lane] = accumulator;
}

__global__ void multicast_hash(float* result, int num_iters) {
  __shared__ float s[kWarps * kLanes];
  initialize_shared(s);
  const unsigned lane = threadIdx.x;
  const int warp = threadIdx.y;
  const unsigned hash = (lane * 2654435761u) >> 16;
  const float* ptr = &s[warp * kLanes + hash % kLanes];
  float accumulator = 0.0f;
  for (int i = 0; i < num_iters; ++i) {
    accumulator += load_shared_f32(ptr);
  }
  result[warp * kLanes + lane] = accumulator;
}

__global__ void vectorized_v4_contiguous(float* result, int num_iters) {
  __shared__ __align__(16) float s[kWarps * 128];
  initialize_shared(s);
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float* ptr = &s[warp * 128 + lane * 4];
  float accumulator = 0.0f;
  for (int i = 0; i < num_iters; ++i) {
    const float4 value = load_shared_v4(ptr);
    accumulator += value.x + value.y + value.z + value.w;
  }
  result[warp * kLanes + lane] = accumulator;
}

__global__ void vectorized_v2_multicast_pairs(float* result, int num_iters) {
  __shared__ __align__(8) float s[kWarps * kLanes];
  initialize_shared(s);
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float* ptr = &s[warp * kLanes + (lane / 2) * 2];
  float accumulator = 0.0f;
  for (int i = 0; i < num_iters; ++i) {
    const float2 value = load_shared_v2(ptr);
    accumulator += value.x + value.y;
  }
  result[warp * kLanes + lane] = accumulator;
}

__global__ void vectorized_v4_multicast_quads(float* result, int num_iters) {
  __shared__ __align__(16) float s[kWarps * kLanes];
  initialize_shared(s);
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float* ptr = &s[warp * kLanes + (lane / 4) * 4];
  float accumulator = 0.0f;
  for (int i = 0; i < num_iters; ++i) {
    const float4 value = load_shared_v4(ptr);
    accumulator += value.x + value.y + value.z + value.w;
  }
  result[warp * kLanes + lane] = accumulator;
}

struct CaseSpec {
  const char* cli_name;
  const char* output_name;
  int bytes_per_load;
};

const std::vector<CaseSpec> kCases = {
    {"baseline", "baseline_unique_banks", 4},
    {"stride", "stride_conflict_sweep", 4},
    {"same_bank_32way_2d", "same_bank_32way_2d", 4},
    {"broadcast", "broadcast_same_address", 4},
    {"multicast_hash", "multicast_hash", 4},
    {"v4_contiguous", "vectorized_v4_contiguous", 16},
    {"v2_multicast_pairs", "vectorized_v2_multicast_pairs", 8},
    {"v4_multicast_quads", "vectorized_v4_multicast_quads", 16},
};

void launch_case(const CaseSpec& spec, const BenchOptions& options,
                 float* device_result) {
  const dim3 block(kLanes, kWarps, 1);
  if (std::string(spec.cli_name) == "baseline") {
    baseline_unique_banks<<<1, block>>>(device_result, options.num_iters);
  } else if (std::string(spec.cli_name) == "stride") {
    stride_conflict_sweep<<<1, block>>>(device_result, options.num_iters,
                                        options.stride, options.offset);
  } else if (std::string(spec.cli_name) == "same_bank_32way_2d") {
    same_bank_32way_2d<<<1, block>>>(device_result, options.num_iters);
  } else if (std::string(spec.cli_name) == "broadcast") {
    broadcast_same_address<<<1, block>>>(device_result, options.num_iters);
  } else if (std::string(spec.cli_name) == "multicast_hash") {
    multicast_hash<<<1, block>>>(device_result, options.num_iters);
  } else if (std::string(spec.cli_name) == "v4_contiguous") {
    vectorized_v4_contiguous<<<1, block>>>(device_result, options.num_iters);
  } else if (std::string(spec.cli_name) == "v2_multicast_pairs") {
    vectorized_v2_multicast_pairs<<<1, block>>>(device_result,
                                                options.num_iters);
  } else if (std::string(spec.cli_name) == "v4_multicast_quads") {
    vectorized_v4_multicast_quads<<<1, block>>>(device_result,
                                                options.num_iters);
  }
  CUDA_CHECK(cudaGetLastError());
}

void print_measurement(const CaseSpec& spec, const BenchOptions& options,
                       float* device_result) {
  for (int i = 0; i < options.num_warmups; ++i) {
    launch_case(spec, options, device_result);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> elapsed_ms;
  elapsed_ms.reserve(options.num_repeats);
  for (int i = 0; i < options.num_repeats; ++i) {
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
  const long long loads_per_thread = options.num_iters;
  const long long bytes_per_thread =
      loads_per_thread * static_cast<long long>(spec.bytes_per_load);
  const long long total_bytes = bytes_per_thread * kThreads;
  const double effective_gbps = total_bytes / (avg_ms * 1.0e6);
  const int reported_stride =
      std::string(spec.cli_name) == "stride" ? options.stride : 0;
  const int reported_offset =
      std::string(spec.cli_name) == "stride" ? options.offset : 0;

  std::cout << spec.output_name << ',' << reported_stride << ','
            << reported_offset << ',' << options.num_iters << ',' << std::fixed
            << std::setprecision(6) << avg_ms << ',' << min_ms << ','
            << loads_per_thread << ',' << bytes_per_thread << ',' << total_bytes
            << ',' << std::setprecision(3) << effective_gbps << '\n';
}

int parse_int(const char* flag, const char* value) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!value[0] || *end != '\0' || parsed < 0 || parsed > 2147483647L) {
    throw std::invalid_argument(std::string("invalid value for ") + flag +
                                ": " + value);
  }
  return static_cast<int>(parsed);
}

void print_usage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " --case CASE [--iters N] [--warmups N] [--repeats N]"
      << " [--stride 1|2|4|8|16|32] [--offset 0..31]\n";
}

BenchOptions parse_options(int argc, char** argv) {
  BenchOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument("missing value after " + arg);
    }
    const char* value = argv[++i];
    if (arg == "--case") {
      options.case_name = value;
    } else if (arg == "--iters") {
      options.num_iters = parse_int("--iters", value);
    } else if (arg == "--warmups") {
      options.num_warmups = parse_int("--warmups", value);
    } else if (arg == "--repeats") {
      options.num_repeats = parse_int("--repeats", value);
    } else if (arg == "--stride") {
      options.stride = parse_int("--stride", value);
    } else if (arg == "--offset") {
      options.offset = parse_int("--offset", value);
    } else {
      throw std::invalid_argument("unknown option: " + arg);
    }
  }
  if (options.num_iters <= 0) {
    throw std::invalid_argument("--iters must be positive");
  }
  if (options.num_repeats <= 0) {
    throw std::invalid_argument("--repeats must be positive");
  }
  const std::vector<int> valid_strides = {1, 2, 4, 8, 16, 32};
  if (std::find(valid_strides.begin(), valid_strides.end(), options.stride) ==
      valid_strides.end()) {
    throw std::invalid_argument("--stride must be one of 1,2,4,8,16,32");
  }
  if (options.offset < 0 || options.offset > 31) {
    throw std::invalid_argument("--offset must be in [0,31]");
  }
  return options;
}

int main(int argc, char** argv) {
  try {
    const BenchOptions options = parse_options(argc, argv);
    std::vector<const CaseSpec*> selected;
    for (const auto& spec : kCases) {
      if (options.case_name == "all" || options.case_name == spec.cli_name) {
        selected.push_back(&spec);
      }
    }
    if (selected.empty()) {
      print_usage(argv[0]);
      throw std::invalid_argument("unknown case: " + options.case_name);
    }

    float* device_result = nullptr;
    CUDA_CHECK(cudaMalloc(&device_result, kThreads * sizeof(float)));
    std::cout << "case,stride,offset,iters,avg_ms,min_ms,loads_per_thread,"
                 "bytes_per_thread,total_bytes,effective_GBps\n";
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
