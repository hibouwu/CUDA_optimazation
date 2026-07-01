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
    const cudaError_t error_ = (call);                                           \
    if (error_ != cudaSuccess)                                                   \
      throw std::runtime_error(cudaGetErrorString(error_));                      \
  } while (0)

constexpr int kLanes = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kLanes * kWarps;

struct Options {
  std::string case_name = "all";
  int stride = 1;
  int offset = 0;
  int iters = 100000;
  int warmups = 5;
  int repeats = 20;
};

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

__device__ __forceinline__ void store_f32(float* ptr, float value) {
  asm volatile("st.volatile.shared.f32 [%0], %1;"
               :
               : "r"(smem_addr(ptr)), "f"(value)
               : "memory");
}

__device__ __forceinline__ void store_v2(float* ptr, float2 value) {
  asm volatile("st.volatile.shared.v2.f32 [%0], {%1, %2};"
               :
               : "r"(smem_addr(ptr)), "f"(value.x), "f"(value.y)
               : "memory");
}

__device__ __forceinline__ void store_v4(float* ptr, float4 value) {
  asm volatile("st.volatile.shared.v4.f32 [%0], {%1, %2, %3, %4};"
               :
               : "r"(smem_addr(ptr)), "f"(value.x), "f"(value.y),
                 "f"(value.z), "f"(value.w)
               : "memory");
}

__global__ void scalar_store_kernel(float* result, int iters, int mode,
                                    int stride, int offset) {
  __shared__ float s[kWarps * 1024];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  int index = warp * 1024 + lane;
  float value = static_cast<float>(warp * kLanes + lane + 1);
  if (mode == 1) {
    index = warp * 1024 + lane * stride + offset;
  } else if (mode == 2) {
    index = warp * 1024 + lane * 32;
  } else if (mode == 3) {
    index = warp * 1024;
    value = static_cast<float>(warp + 1);
  }
  for (int i = 0; i < iters; ++i) store_f32(&s[index], value);
  __syncthreads();
  result[warp * kLanes + lane] = s[index];
}

__global__ void vector_store_kernel(float* result, int iters, int width,
                                    bool multicast) {
  __shared__ __align__(16) float s[kWarps * 128];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int group = multicast ? lane / width : lane;
  const int index = warp * 128 + group * width;
  const float base = static_cast<float>(warp * 128 + group * width + 1);
  if (width == 2) {
    const float2 value = make_float2(base, base + 1.0f);
    for (int i = 0; i < iters; ++i) store_v2(&s[index], value);
  } else {
    const float4 value = make_float4(base, base + 1.0f, base + 2.0f,
                                     base + 3.0f);
    for (int i = 0; i < iters; ++i) store_v4(&s[index], value);
  }
  __syncthreads();
  result[warp * kLanes + lane] = s[index];
}

struct CaseSpec {
  const char* cli;
  const char* label;
  int bytes;
};

const std::vector<CaseSpec> kCases = {
    {"baseline", "baseline_unique_banks", 4},
    {"stride", "stride_conflict_sweep", 4},
    {"same_bank_32way_2d", "same_bank_32way_2d", 4},
    {"same_address", "same_address_same_value", 4},
    {"v2_contiguous", "vectorized_v2_contiguous", 8},
    {"v4_contiguous", "vectorized_v4_contiguous", 16},
    {"v2_multicast_pairs", "vectorized_v2_same_destination_pairs", 8},
    {"v4_multicast_quads", "vectorized_v4_same_destination_quads", 16},
};

void launch(const CaseSpec& spec, const Options& o, float* result) {
  const dim3 block(kLanes, kWarps);
  const std::string name = spec.cli;
  if (name == "baseline")
    scalar_store_kernel<<<1, block>>>(result, o.iters, 0, 1, 0);
  else if (name == "stride")
    scalar_store_kernel<<<1, block>>>(result, o.iters, 1, o.stride, o.offset);
  else if (name == "same_bank_32way_2d")
    scalar_store_kernel<<<1, block>>>(result, o.iters, 2, 32, 0);
  else if (name == "same_address")
    scalar_store_kernel<<<1, block>>>(result, o.iters, 3, 1, 0);
  else if (name == "v2_contiguous")
    vector_store_kernel<<<1, block>>>(result, o.iters, 2, false);
  else if (name == "v4_contiguous")
    vector_store_kernel<<<1, block>>>(result, o.iters, 4, false);
  else if (name == "v2_multicast_pairs")
    vector_store_kernel<<<1, block>>>(result, o.iters, 2, true);
  else
    vector_store_kernel<<<1, block>>>(result, o.iters, 4, true);
  CUDA_CHECK(cudaGetLastError());
}

void measure(const CaseSpec& spec, const Options& o, float* result) {
  for (int i = 0; i < o.warmups; ++i) launch(spec, o, result);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  std::vector<float> samples;
  for (int i = 0; i < o.repeats; ++i) {
    CUDA_CHECK(cudaEventRecord(begin));
    launch(spec, o, result);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
    samples.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  const double avg =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  const double min = *std::min_element(samples.begin(), samples.end());
  const long long bytes_per_thread = 1LL * o.iters * spec.bytes;
  const long long total_bytes = bytes_per_thread * kThreads;
  std::cout << spec.label << ','
            << (std::string(spec.cli) == "stride" ? o.stride : 0) << ','
            << (std::string(spec.cli) == "stride" ? o.offset : 0) << ','
            << o.iters << ',' << std::fixed << std::setprecision(6) << avg
            << ',' << min << ',' << o.iters << ',' << bytes_per_thread << ','
            << total_bytes << ',' << std::setprecision(3)
            << total_bytes / (avg * 1.0e6) << '\n';
}

int integer(const char* flag, const char* text) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!*text || *end || value < 0 || value > 2147483647L)
    throw std::invalid_argument(std::string("invalid ") + flag);
  return static_cast<int>(value);
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "--case CASE --stride N --offset N --iters N "
                   "--warmups N --repeats N\n";
      std::exit(0);
    }
    if (++i >= argc) throw std::invalid_argument("missing option value");
    if (arg == "--case") o.case_name = argv[i];
    else if (arg == "--stride") o.stride = integer("--stride", argv[i]);
    else if (arg == "--offset") o.offset = integer("--offset", argv[i]);
    else if (arg == "--iters") o.iters = integer("--iters", argv[i]);
    else if (arg == "--warmups") o.warmups = integer("--warmups", argv[i]);
    else if (arg == "--repeats") o.repeats = integer("--repeats", argv[i]);
    else throw std::invalid_argument("unknown option " + arg);
  }
  const std::vector<int> strides = {1, 2, 4, 8, 16, 32};
  if (std::find(strides.begin(), strides.end(), o.stride) == strides.end())
    throw std::invalid_argument("stride must be 1,2,4,8,16,32");
  if (o.offset < 0 || o.offset > 31 || o.iters < 1 || o.repeats < 1)
    throw std::invalid_argument("invalid numeric range");
  return o;
}

int main(int argc, char** argv) {
  try {
    const Options o = parse(argc, argv);
    float* result = nullptr;
    CUDA_CHECK(cudaMalloc(&result, kThreads * sizeof(float)));
    std::cout << "case,stride,offset,iters,avg_ms,min_ms,stores_per_thread,"
                 "bytes_per_thread,total_bytes,effective_GBps\n";
    bool found = false;
    for (const auto& spec : kCases) {
      if (o.case_name == "all" || o.case_name == spec.cli) {
        found = true;
        measure(spec, o, result);
      }
    }
    CUDA_CHECK(cudaFree(result));
    if (!found) throw std::invalid_argument("unknown case " + o.case_name);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}

