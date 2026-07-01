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
constexpr int kWordsPerWarp = 1024;

struct Options {
  std::string case_name = "all";
  int stride_words = 4;
  int iters = 10000;
  int warmups = 5;
  int repeats = 20;
};

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16;"
               :
               : "r"(smem_addr(dst)), "l"(src)
               : "memory");
}

__device__ __forceinline__ void cp_commit_wait() {
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__global__ void cp_async_kernel(const float4* input, float* result, int iters,
                                int stride_words, bool same_source) {
  __shared__ __align__(16) float tile[kWarps * kWordsPerWarp];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * kLanes + lane;
  const int word = warp * kWordsPerWarp + lane * stride_words;
  const float4* src = same_source ? &input[warp] : &input[tid];
  for (int i = 0; i < iters; ++i) {
    cp_async_16(&tile[word], src);
    cp_commit_wait();
  }
  __syncthreads();
  result[tid] = tile[word];
}

struct CaseSpec {
  const char* cli;
  const char* label;
  int stride_words;
  bool same_source;
};

const std::vector<CaseSpec> kCases = {
    {"contiguous", "contiguous_16B", 4, false},
    {"stride", "destination_stride_words", 0, false},
    {"source_broadcast", "same_source_distinct_destinations", 4, true},
};

void launch(const CaseSpec& spec, const Options& o, const float4* input,
            float* result) {
  const int stride = std::string(spec.cli) == "stride" ? o.stride_words
                                                        : spec.stride_words;
  cp_async_kernel<<<1, dim3(kLanes, kWarps)>>>(input, result, o.iters, stride,
                                               spec.same_source);
  CUDA_CHECK(cudaGetLastError());
}

void measure(const CaseSpec& spec, const Options& o, const float4* input,
             float* result) {
  for (int i = 0; i < o.warmups; ++i) launch(spec, o, input, result);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  std::vector<float> samples;
  for (int i = 0; i < o.repeats; ++i) {
    CUDA_CHECK(cudaEventRecord(begin));
    launch(spec, o, input, result);
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
  const long long total = 1LL * o.iters * kThreads * 16;
  const int stride = std::string(spec.cli) == "stride" ? o.stride_words
                                                        : spec.stride_words;
  std::cout << spec.label << ',' << stride << ',' << o.iters << ','
            << std::fixed << std::setprecision(6) << avg << ',' << min << ','
            << total << ',' << std::setprecision(3)
            << total / (avg * 1.0e6) << '\n';
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
    else if (arg == "--stride-words") o.stride_words = integer(argv[i]);
    else if (arg == "--iters") o.iters = integer(argv[i]);
    else if (arg == "--warmups") o.warmups = integer(argv[i]);
    else if (arg == "--repeats") o.repeats = integer(argv[i]);
    else throw std::invalid_argument("unknown option " + arg);
  }
  const std::vector<int> strides = {4, 8, 16, 32};
  if (std::find(strides.begin(), strides.end(), o.stride_words) ==
          strides.end() ||
      o.iters < 1 || o.repeats < 1)
    throw std::invalid_argument("invalid option range");
  return o;
}

int main(int argc, char** argv) {
  try {
    const Options o = parse(argc, argv);
    std::vector<float4> host(kThreads);
    for (int i = 0; i < kThreads; ++i)
      host[i] = make_float4(i + 1, i + 2, i + 3, i + 4);
    float4* input = nullptr;
    float* result = nullptr;
    CUDA_CHECK(cudaMalloc(&input, host.size() * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&result, kThreads * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input, host.data(), host.size() * sizeof(float4),
                          cudaMemcpyHostToDevice));
    std::cout << "case,stride_words,iters,avg_ms,min_ms,total_bytes,"
                 "effective_GBps\n";
    bool found = false;
    for (const auto& spec : kCases) {
      if (o.case_name == "all" || o.case_name == spec.cli) {
        found = true;
        measure(spec, o, input, result);
      }
    }
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(result));
    if (!found) throw std::invalid_argument("unknown case");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
