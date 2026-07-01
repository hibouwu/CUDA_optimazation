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
constexpr int kThreads = 256;

struct Options {
  std::string case_name = "all";
  int iters = 100000;
  int warmups = 5;
  int repeats = 20;
};

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}
__device__ __forceinline__ float ld_shared(const float* ptr) {
  float value;
  asm volatile("ld.volatile.shared.f32 %0, [%1];"
               : "=f"(value)
               : "r"(smem_addr(ptr))
               : "memory");
  return value;
}
__device__ __forceinline__ void st_shared(float* ptr, float value) {
  asm volatile("st.volatile.shared.f32 [%0], %1;"
               :
               : "r"(smem_addr(ptr)), "f"(value)
               : "memory");
}

template <int Pitch, bool Store>
__global__ void transpose_kernel(float* result, int iters) {
  __shared__ float tile[32 * Pitch];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * kLanes + lane;
  for (int index = tid; index < 32 * Pitch; index += kThreads)
    tile[index] = static_cast<float>((index % 251) + 1);
  __syncthreads();

  float accumulator = 0;
  float* ptr = &tile[lane * Pitch + warp];
  if constexpr (Store) {
    const float value = static_cast<float>(warp + 1);
    for (int i = 0; i < iters; ++i) st_shared(ptr, value);
    __syncthreads();
    accumulator = *ptr;
  } else {
    for (int i = 0; i < iters; ++i) accumulator += ld_shared(ptr);
  }
  result[tid] = accumulator;
}

struct CaseSpec {
  const char* cli;
  const char* label;
  int pitch;
  bool store;
};
const std::vector<CaseSpec> kCases = {
    {"load_pitch32", "transpose_load_pitch32", 32, false},
    {"load_pitch33", "transpose_load_pitch33", 33, false},
    {"store_pitch32", "transpose_store_pitch32", 32, true},
    {"store_pitch33", "transpose_store_pitch33", 33, true},
};

void launch(const CaseSpec& spec, const Options& o, float* result) {
  const dim3 block(kLanes, kWarps);
  if (spec.pitch == 32 && !spec.store)
    transpose_kernel<32, false><<<1, block>>>(result, o.iters);
  else if (spec.pitch == 33 && !spec.store)
    transpose_kernel<33, false><<<1, block>>>(result, o.iters);
  else if (spec.pitch == 32)
    transpose_kernel<32, true><<<1, block>>>(result, o.iters);
  else
    transpose_kernel<33, true><<<1, block>>>(result, o.iters);
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
  const long long total = 1LL * o.iters * kThreads * sizeof(float);
  std::cout << spec.label << ',' << (spec.store ? "store" : "load") << ','
            << spec.pitch << ',' << o.iters << ',' << std::fixed
            << std::setprecision(6) << avg << ',' << min << ',' << total << ','
            << std::setprecision(3) << total / (avg * 1.0e6) << '\n';
}

int integer(const char* text) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!*text || *end || value < 0 || value > 2147483647L)
    throw std::invalid_argument("invalid integer");
  return static_cast<int>(value);
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (++i >= argc) throw std::invalid_argument("missing option value");
    if (arg == "--case") o.case_name = argv[i];
    else if (arg == "--iters") o.iters = integer(argv[i]);
    else if (arg == "--warmups") o.warmups = integer(argv[i]);
    else if (arg == "--repeats") o.repeats = integer(argv[i]);
    else throw std::invalid_argument("unknown option " + arg);
  }
  if (o.iters < 1 || o.repeats < 1) throw std::invalid_argument("invalid range");
  return o;
}

int main(int argc, char** argv) {
  try {
    const Options o = parse(argc, argv);
    float* result = nullptr;
    CUDA_CHECK(cudaMalloc(&result, kThreads * sizeof(float)));
    std::cout << "case,operation,pitch,iters,avg_ms,min_ms,total_bytes,"
                 "effective_GBps\n";
    bool found = false;
    for (const auto& spec : kCases) {
      if (o.case_name == "all" || o.case_name == spec.cli) {
        found = true;
        measure(spec, o, result);
      }
    }
    CUDA_CHECK(cudaFree(result));
    if (!found) throw std::invalid_argument("unknown case");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}

