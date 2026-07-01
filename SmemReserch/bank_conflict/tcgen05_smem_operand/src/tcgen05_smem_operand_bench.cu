#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
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

#if defined(__CUDA_ARCH_FEAT_SM110_ALL)
#define HAS_TCGEN05_SM110 1
#else
#define HAS_TCGEN05_SM110 0
#endif

constexpr int kThreads = 128;
constexpr int kTileM = 128;
constexpr int kTileN = 64;
constexpr int kMmaK = 16;
constexpr int kTmemColumns = 64;
constexpr int kABytes = 16 * 1024;
constexpr int kBBytes = 8 * 1024;
constexpr int kMmasPerCommit = 16;

struct Options {
  std::string case_name = "all";
  int iters = 10000;
  int warmups = 5;
  int repeats = 20;
};

struct CaseSpec {
  const char* cli;
  unsigned swizzle_span;
  unsigned swizzle_code;
};

const std::vector<CaseSpec> kCases = {
    {"swizzle_32b", 32, 6},
    {"swizzle_64b", 64, 4},
    {"swizzle_128b", 128, 2},
};

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__host__ __device__ constexpr std::uint64_t encode_smem(std::uint64_t value) {
  return (value & 0x3ffffULL) >> 4ULL;
}

__device__ __forceinline__ std::uint64_t make_k_major_descriptor(
    unsigned address, unsigned swizzle_span, unsigned swizzle_code) {
  const unsigned stride_bytes = 8 * swizzle_span;
  return encode_smem(address) |
         (encode_smem(stride_bytes) << 32ULL) |
         (1ULL << 46ULL) |
         (static_cast<std::uint64_t>(swizzle_code) << 61ULL);
}

__device__ __forceinline__ void mbarrier_init(unsigned barrier) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
               :
               : "r"(barrier)
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

__global__ void tcgen05_operand_kernel(float* result, int iters,
                                       unsigned swizzle_span,
                                       unsigned swizzle_code) {
  __shared__ alignas(1024) __half operand_a[kABytes / sizeof(__half)];
  __shared__ alignas(1024) __half operand_b[kBBytes / sizeof(__half)];
  __shared__ unsigned tmem_base;
  __shared__ alignas(8) std::uint64_t mma_barrier_storage;
  const int tid = threadIdx.x;
  for (int i = tid; i < kABytes / 2; i += kThreads)
    operand_a[i] = __float2half(1.0f);
  for (int i = tid; i < kBBytes / 2; i += kThreads)
    operand_b[i] = __float2half(1.0f);

#if HAS_TCGEN05_SM110
  const unsigned barrier = smem_addr(&mma_barrier_storage);
  if (tid == 0) {
    mbarrier_init(barrier);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid < 32) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(smem_addr(&tmem_base)), "r"(kTmemColumns)
        : "memory");
  }
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  constexpr std::uint32_t instruction_descriptor =
      (1U << 4U) |
      (static_cast<std::uint32_t>(kTileN) >> 3U << 17U) |
      (static_cast<std::uint32_t>(kTileM) >> 4U << 24U);
  const std::uint64_t descriptor_a = make_k_major_descriptor(
      smem_addr(operand_a), swizzle_span, swizzle_code);
  const std::uint64_t descriptor_b = make_k_major_descriptor(
      smem_addr(operand_b), swizzle_span, swizzle_code);

  unsigned phase = 0;
  for (int first = 0; first < iters; first += kMmasPerCommit) {
    if (tid == 0) {
      const int remaining = iters - first;
      const int count =
          remaining < kMmasPerCommit ? remaining : kMmasPerCommit;
      for (int i = 0; i < count; ++i) {
        asm volatile(
            "{\n"
            ".reg .pred use_d;\n"
            "setp.ne.b32 use_d, 0, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, use_d;\n"
            "}"
            :
            : "r"(tmem_base), "l"(descriptor_a), "l"(descriptor_b),
              "r"(instruction_descriptor)
            : "memory");
      }
      asm volatile(
          "tcgen05.commit.cta_group::1."
          "mbarrier::arrive::one.shared::cluster.b64 [%0];"
          :
          : "r"(barrier)
          : "memory");
    }
    mbarrier_wait(barrier, phase);
    phase ^= 1;
  }
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  __syncthreads();

  float value = 0.0f;
  if (tid < 32) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 : "=f"(value)
                 : "r"(tmem_base)
                 : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    result[tid] = value;
  }
  __syncthreads();
  if (tid < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :
                 : "r"(tmem_base), "r"(kTmemColumns)
                 : "memory");
  }
#else
  if (tid == 0) result[0] = 0.0f;
#endif
}

void launch(const CaseSpec& spec, const Options& o, float* result) {
  tcgen05_operand_kernel<<<1, kThreads>>>(result, o.iters, spec.swizzle_span,
                                          spec.swizzle_code);
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
  std::cout << spec.cli << ',' << spec.swizzle_span << ',' << o.iters << ','
            << std::fixed << std::setprecision(6) << avg << ',' << min << ','
            << std::setprecision(3) << o.iters / (avg * 1.0e3) << '\n';
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
    CUDA_CHECK(cudaMalloc(&result, 32 * sizeof(float)));
    std::cout << "case,swizzle_span_bytes,iters,avg_ms,min_ms,"
                 "million_mma_per_second\n";
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
