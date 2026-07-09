#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

#define CHECK_CUDA_RET(call)                                                 \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      print_csv(columns, "cuda_error", cudaGetErrorName(err__), 0, 0, 0, 0); \
      return 3;                                                              \
    }                                                                        \
  } while (0)

#if defined(__CUDA_ARCH_FEAT_SM110_ALL)
#define SM110_HAS_TCGEN05 1
#else
#define SM110_HAS_TCGEN05 0
#endif

struct ProbeResult {
  uint32_t marker;
  uint32_t tmem_base;
  uint32_t tail_column;
  uint32_t front_bits;
  uint32_t tail_bits;
};

__device__ __forceinline__ uint32_t smem_u32(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__global__ __launch_bounds__(128, 1)
void tmem_size_probe_kernel(int columns, ProbeResult* result) {
  __shared__ uint32_t tmem_base;

#if SM110_HAS_TCGEN05
  if (threadIdx.x < 32) {
    uint32_t dst = smem_u32(&tmem_base);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(dst), "r"(columns)
        : "memory");
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    constexpr uint32_t kFrontBits = 0x3f800000u;  // 1.0f
    constexpr uint32_t kTailBits = 0x40000000u;   // 2.0f

    // 32x32b.x1 touches exactly one 32-bit TMEM column across 32 lanes, so
    // probing columns - 1 checks the requested allocation's last column.
    uint32_t tail_column = columns > 0 ? static_cast<uint32_t>(columns - 1) : 0u;
    uint32_t tail_addr = tmem_base + tail_column;

    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
                 :: "r"(tmem_base), "r"(kFrontBits)
                 : "memory");
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
                 :: "r"(tail_addr), "r"(kTailBits)
                 : "memory");

    uint32_t front_loaded = 0;
    uint32_t tail_loaded = 0;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 : "=r"(front_loaded)
                 : "r"(tmem_base)
                 : "memory");
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 : "=r"(tail_loaded)
                 : "r"(tail_addr)
                 : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

    if (threadIdx.x == 0) {
      result->marker = 0x544d454du;  // "TMEM"
      result->tmem_base = tmem_base;
      result->tail_column = tail_column;
      result->front_bits = front_loaded;
      result->tail_bits = tail_loaded;
    }
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_base), "r"(columns)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                 ::: "memory");
  }
#else
  if (threadIdx.x == 0) {
    result->marker = 0;
    result->tmem_base = 0;
    result->tail_column = 0;
    result->front_bits = 0;
    result->tail_bits = 0;
  }
#endif
}

void print_usage(char const* argv0) {
  std::cerr << "usage: " << argv0 << " --columns <n> [--csv]\n";
}

void print_device() {
  cudaDeviceProp prop{};
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(err) << "\n";
    return;
  }
  std::cout << "GPU: " << prop.name << "\n";
  std::cout << "compute_capability: " << prop.major << "." << prop.minor << "\n";
  std::cout << "sm_count: " << prop.multiProcessorCount << "\n";
}

void print_csv(int columns,
               char const* status,
               char const* cuda_error,
               uint32_t base,
               uint32_t tail_column,
               int front_ok,
               int tail_ok) {
  long long bytes = static_cast<long long>(columns) * 128LL * 4LL;
  double kib = static_cast<double>(bytes) / 1024.0;
  std::cout << columns << "," << bytes << "," << std::fixed << std::setprecision(1)
            << kib << "," << base << "," << tail_column << "," << front_ok << ","
            << tail_ok << "," << status << "," << cuda_error << "\n";
}

int run_probe(int columns, bool csv) {
  if (columns <= 0) {
    std::cerr << "columns must be positive\n";
    return 2;
  }

  cudaError_t err = cudaFree(nullptr);
  if (err != cudaSuccess) {
    print_csv(columns, "cuda_error", cudaGetErrorName(err), 0, 0, 0, 0);
    return 3;
  }

  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    print_csv(columns, "cuda_error", cudaGetErrorName(err), 0, 0, 0, 0);
    return 3;
  }

  if (!csv) {
    print_device();
    std::cout << "requested_columns: " << columns << "\n";
    std::cout << "requested_bytes: " << static_cast<long long>(columns) * 512LL << "\n";
  }

  if (prop.major != 11) {
    print_csv(columns, "skipped_non_sm110", "none", 0, 0, 0, 0);
    return 0;
  }

  ProbeResult zero{};
  ProbeResult host{};
  ProbeResult* device = nullptr;
  CHECK_CUDA_RET(cudaMalloc(&device, sizeof(ProbeResult)));
  CHECK_CUDA_RET(cudaMemcpy(device, &zero, sizeof(ProbeResult), cudaMemcpyHostToDevice));

  tmem_size_probe_kernel<<<1, 128>>>(columns, device);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    print_csv(columns, "cuda_error", cudaGetErrorName(err), 0, 0, 0, 0);
    cudaFree(device);
    return 3;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    print_csv(columns, "cuda_error", cudaGetErrorName(err), 0, 0, 0, 0);
    cudaFree(device);
    return 3;
  }

  CHECK_CUDA_RET(cudaMemcpy(&host, device, sizeof(ProbeResult), cudaMemcpyDeviceToHost));
  CHECK_CUDA_RET(cudaFree(device));

  bool marker_ok = host.marker == 0x544d454du;
  bool front_ok = host.front_bits == 0x3f800000u;
  bool tail_ok = host.tail_bits == 0x40000000u;
  char const* status = marker_ok && front_ok && tail_ok ? "ok" : "data_mismatch";
  print_csv(columns, status, "none", host.tmem_base, host.tail_column,
            front_ok ? 1 : 0, tail_ok ? 1 : 0);
  return marker_ok && front_ok && tail_ok ? 0 : 4;
}

}  // namespace

int main(int argc, char** argv) {
  int columns = 0;
  bool csv = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--columns") == 0 && i + 1 < argc) {
      columns = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--csv") == 0) {
      csv = true;
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      print_usage(argv[0]);
      return 2;
    }
  }

  if (columns <= 0) {
    print_usage(argv[0]);
    return 2;
  }
  return run_probe(columns, csv);
}
