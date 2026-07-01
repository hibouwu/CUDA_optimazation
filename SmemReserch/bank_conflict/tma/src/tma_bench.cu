#include <cuda.h>
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

constexpr int kThreads = 256;
constexpr int kTileBytes = 4096;
constexpr int kGlobalSide = 128;

struct Options {
  std::string case_name = "all";
  int iters = 10000;
  int warmups = 5;
  int repeats = 20;
};

struct CaseSpec {
  const char* cli;
  CUtensorMapSwizzle swizzle;
  int box_x;
  int box_y;
};

const std::vector<CaseSpec> kCases = {
    {"swizzle_none", CU_TENSOR_MAP_SWIZZLE_NONE, 128, 32},
    {"swizzle_32b", CU_TENSOR_MAP_SWIZZLE_32B, 32, 128},
    {"swizzle_64b", CU_TENSOR_MAP_SWIZZLE_64B, 64, 64},
    {"swizzle_128b", CU_TENSOR_MAP_SWIZZLE_128B, 128, 32},
};

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
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

__device__ __forceinline__ void tma_load_2d(
    unsigned dst, const CUtensorMap* tensor_map, int x, int y,
    unsigned barrier) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile."
      "mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(dst), "l"(tensor_map), "r"(x), "r"(y), "r"(barrier)
      : "memory");
}

__global__ void tma_kernel(const __grid_constant__ CUtensorMap tensor_map,
                           std::uint32_t* result, int iters) {
  __shared__ alignas(1024) std::uint8_t tile[kTileBytes];
  __shared__ alignas(8) std::uint64_t barrier_storage;
  const int tid = threadIdx.x;
  const unsigned barrier = smem_addr(&barrier_storage);
  if (tid == 0) {
    mbarrier_init(barrier);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  unsigned phase = 0;
  for (int i = 0; i < iters; ++i) {
    if (tid == 0) {
      tma_load_2d(smem_addr(tile), &tensor_map, 0, 0, barrier);
      mbarrier_expect_tx(barrier, kTileBytes);
    }
    mbarrier_wait(barrier, phase);
    phase ^= 1;
  }
  __syncthreads();
  std::uint32_t checksum = 0;
  for (int i = tid; i < kTileBytes; i += blockDim.x) checksum += tile[i];
  result[tid] = checksum;
}

void check_driver(CUresult status, const char* where) {
  if (status == CUDA_SUCCESS) return;
  const char* text = "unknown driver error";
  cuGetErrorString(status, &text);
  throw std::runtime_error(std::string(where) + ": " + text);
}

CUtensorMap make_tensor_map(const CaseSpec& spec, void* input) {
  CUtensorMap map{};
  const cuuint64_t dims[2] = {kGlobalSide, kGlobalSide};
  const cuuint64_t strides[1] = {kGlobalSide};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(spec.box_x),
                             static_cast<cuuint32_t>(spec.box_y)};
  const cuuint32_t element_strides[2] = {1, 1};
  check_driver(
      cuTensorMapEncodeTiled(
          &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, input, dims, strides, box,
          element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, spec.swizzle,
          CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
  return map;
}

void launch(const CUtensorMap& map, const Options& o, std::uint32_t* result) {
  tma_kernel<<<1, kThreads>>>(map, result, o.iters);
  CUDA_CHECK(cudaGetLastError());
}

void measure(const CaseSpec& spec, const Options& o, void* input,
             std::uint32_t* result) {
  const CUtensorMap map = make_tensor_map(spec, input);
  for (int i = 0; i < o.warmups; ++i) launch(map, o, result);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  std::vector<float> samples;
  for (int i = 0; i < o.repeats; ++i) {
    CUDA_CHECK(cudaEventRecord(begin));
    launch(map, o, result);
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
  const long long total = 1LL * o.iters * kTileBytes;
  std::cout << spec.cli << ',' << spec.box_x << ',' << spec.box_y << ','
            << o.iters << ',' << std::fixed << std::setprecision(6) << avg
            << ',' << min << ',' << total << ',' << std::setprecision(3)
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
    check_driver(cuInit(0), "cuInit");
    std::vector<std::uint8_t> host(kGlobalSide * kGlobalSide);
    for (std::size_t i = 0; i < host.size(); ++i) host[i] = i & 0xff;
    std::uint8_t* input = nullptr;
    std::uint32_t* result = nullptr;
    CUDA_CHECK(cudaMalloc(&input, host.size()));
    CUDA_CHECK(cudaMalloc(&result, kThreads * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemcpy(input, host.data(), host.size(),
                          cudaMemcpyHostToDevice));
    std::cout << "case,box_x,box_y,iters,avg_ms,min_ms,total_bytes,"
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
