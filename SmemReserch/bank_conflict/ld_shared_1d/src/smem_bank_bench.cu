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

__device__ __forceinline__ float run_scalar_loads(const float* ptr,
                                                  int num_iters) {
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
#pragma unroll 1
  for (int i = 0; i < num_iters; ++i) {
    const float value0 = load_shared_f32(ptr);
    const float value1 = load_shared_f32(ptr);
    const float value2 = load_shared_f32(ptr);
    const float value3 = load_shared_f32(ptr);
    acc0 += value0;
    acc1 += value1;
    acc2 += value2;
    acc3 += value3;
  }
  return (acc0 + acc1) + (acc2 + acc3);
}

__device__ __forceinline__ float run_v2_loads(const float* ptr, int num_iters) {
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
#pragma unroll 1
  for (int i = 0; i < num_iters; ++i) {
    const float2 value0 = load_shared_v2(ptr);
    const float2 value1 = load_shared_v2(ptr);
    const float2 value2 = load_shared_v2(ptr);
    const float2 value3 = load_shared_v2(ptr);
    acc0 += value0.x + value0.y;
    acc1 += value1.x + value1.y;
    acc2 += value2.x + value2.y;
    acc3 += value3.x + value3.y;
  }
  return (acc0 + acc1) + (acc2 + acc3);
}

__device__ __forceinline__ float run_v4_loads(const float* ptr, int num_iters) {
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
#pragma unroll 1
  for (int i = 0; i < num_iters; ++i) {
    const float4 value0 = load_shared_v4(ptr);
    const float4 value1 = load_shared_v4(ptr);
    const float4 value2 = load_shared_v4(ptr);
    const float4 value3 = load_shared_v4(ptr);
    acc0 += value0.x + value0.y + value0.z + value0.w;
    acc1 += value1.x + value1.y + value1.z + value1.w;
    acc2 += value2.x + value2.y + value2.z + value2.w;
    acc3 += value3.x + value3.y + value3.z + value3.w;
  }
  return (acc0 + acc1) + (acc2 + acc3);
}

__global__ void v0_unique_banks(float* result, int num_iters) {
  __shared__ float s[kWarps * kLanes];
  initialize_shared(s);
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float* ptr = &s[warp * kLanes + lane];
  result[warp * kLanes + lane] = run_scalar_loads(ptr, num_iters);
}

template <int Stride>
__global__ void v1_stride_conflict(float* result, int num_iters) {
  __shared__ float s[1024];
  initialize_shared(s);
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float* ptr = &s[lane * Stride];
  result[warp * kLanes + lane] = run_scalar_loads(ptr, num_iters);
}

__global__ void v2_broadcast(float* result, int num_iters) {
  __shared__ float s[kWarps * kLanes];
  initialize_shared(s);
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float* ptr = &s[warp * kLanes];
  result[warp * kLanes + lane] = run_scalar_loads(ptr, num_iters);
}

__global__ void v3_v4_contiguous(float* result, int num_iters) {
  __shared__ __align__(16) float s[kWarps * 128];
  initialize_shared(s);
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float* ptr = &s[warp * 128 + lane * 4];
  result[warp * kLanes + lane] = run_v4_loads(ptr, num_iters);
}

__global__ void v4a_v2_multicast_pairs(float* result, int num_iters) {
  __shared__ __align__(8) float s[kWarps * kLanes];
  initialize_shared(s);
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float* ptr = &s[warp * kLanes + (lane / 2) * 2];
  result[warp * kLanes + lane] = run_v2_loads(ptr, num_iters);
}

__global__ void v4b_v4_multicast_quads(float* result, int num_iters) {
  __shared__ __align__(16) float s[kWarps * kLanes];
  initialize_shared(s);
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float* ptr = &s[warp * kLanes + (lane / 4) * 4];
  result[warp * kLanes + lane] = run_v4_loads(ptr, num_iters);
}

enum class CaseId { v0, v1a, v1b, v1c, v1d, v1e, v2, v3, v4a, v4b };

struct CaseSpec {
  const char* cli_name;
  CaseId id;
  int stride;
  int bytes_per_load;
};

const std::vector<CaseSpec> kCases = {
    {"v0", CaseId::v0, 1, 4},   {"v1a", CaseId::v1a, 2, 4},
    {"v1b", CaseId::v1b, 4, 4}, {"v1c", CaseId::v1c, 8, 4},
    {"v1d", CaseId::v1d, 16, 4},
    {"v1e", CaseId::v1e, 32, 4},
    {"v2", CaseId::v2, 0, 4},   {"v3", CaseId::v3, 0, 16},
    {"v4a", CaseId::v4a, 0, 8}, {"v4b", CaseId::v4b, 0, 16},
};

void launch_case(const CaseSpec& spec, const BenchOptions& options,
                 float* device_result) {
  const dim3 block(kLanes, kWarps, 1);
  switch (spec.id) {
    case CaseId::v0:
      v0_unique_banks<<<1, block>>>(device_result, options.num_iters);
      break;
    case CaseId::v1a:
      v1_stride_conflict<2><<<1, block>>>(device_result, options.num_iters);
      break;
    case CaseId::v1b:
      v1_stride_conflict<4><<<1, block>>>(device_result, options.num_iters);
      break;
    case CaseId::v1c:
      v1_stride_conflict<8><<<1, block>>>(device_result, options.num_iters);
      break;
    case CaseId::v1d:
      v1_stride_conflict<16><<<1, block>>>(device_result, options.num_iters);
      break;
    case CaseId::v1e:
      v1_stride_conflict<32><<<1, block>>>(device_result, options.num_iters);
      break;
    case CaseId::v2:
      v2_broadcast<<<1, block>>>(device_result, options.num_iters);
      break;
    case CaseId::v3:
      v3_v4_contiguous<<<1, block>>>(device_result, options.num_iters);
      break;
    case CaseId::v4a:
      v4a_v2_multicast_pairs<<<1, block>>>(device_result, options.num_iters);
      break;
    case CaseId::v4b:
      v4b_v4_multicast_quads<<<1, block>>>(device_result, options.num_iters);
      break;
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
  const long long loads_per_thread =
      static_cast<long long>(options.num_iters) * kIndependentLoads;
  const long long bytes_per_thread =
      loads_per_thread * static_cast<long long>(spec.bytes_per_load);
  const long long total_bytes = bytes_per_thread * kThreads;
  const double effective_gbps = total_bytes / (avg_ms * 1.0e6);
  std::cout << spec.cli_name << ',' << spec.stride << ',' << options.num_iters
            << ',' << std::fixed
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
      << " --case all|v0|v1a|v1b|v1c|v1d|v1e|v2|v3|v4a|v4b"
      << " [--iters N] [--warmups N] [--repeats N]\n";
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
    std::cout << "case,stride,iters,avg_ms,min_ms,loads_per_thread,"
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
