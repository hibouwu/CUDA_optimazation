#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call) do {                                             \
  cudaError_t err__ = (call);                                             \
  if (err__ != cudaSuccess) {                                             \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                 cudaGetErrorString(err__));                              \
    std::exit(1);                                                         \
  }                                                                       \
} while (0)

__device__ __forceinline__ uint32_t smem_u32(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__global__ __launch_bounds__(128, 1)
void tmem_alloc_kernel(int columns, uint32_t* base_out) {
  __shared__ uint32_t tmem_base;

  if (threadIdx.x < 32) {
    uint32_t dst = smem_u32(&tmem_base);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(dst), "r"(columns));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    base_out[blockIdx.x] = tmem_base;
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_base), "r"(columns));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::);
  }
}

static void print_attr(const char* name, cudaDeviceAttr attr, bool bytes = false) {
  int value = 0;
  cudaError_t err = cudaDeviceGetAttribute(&value, attr, 0);
  if (err == cudaSuccess) {
    if (bytes) {
      std::printf("%s=%d bytes (%.1f KiB)\n", name, value, value / 1024.0);
    } else {
      std::printf("%s=%d\n", name, value);
    }
  } else {
    std::printf("%s=unavailable (%s)\n", name, cudaGetErrorString(err));
    cudaGetLastError();
  }
}

static void print_device_props() {
  int runtime_version = 0;
  int driver_version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  std::printf("cuda_runtime_version=%d\n", runtime_version);
  std::printf("cuda_driver_version=%d\n", driver_version);
  std::printf("device_name=%s\n", prop.name);
  std::printf("compute_capability=%d.%d\n", prop.major, prop.minor);
  std::printf("sm_count=%d\n", prop.multiProcessorCount);
  std::printf("warp_size=%d\n", prop.warpSize);
  std::printf("max_threads_per_block=%d\n", prop.maxThreadsPerBlock);
  std::printf("max_threads_per_sm=%d\n", prop.maxThreadsPerMultiProcessor);
  std::printf("max_blocks_per_sm=%d\n", prop.maxBlocksPerMultiProcessor);
  std::printf("regs_per_block=%d\n", prop.regsPerBlock);
  std::printf("regs_per_sm=%d\n", prop.regsPerMultiprocessor);
  std::printf("shared_mem_per_block=%zu bytes (%.1f KiB)\n",
              prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0);
  std::printf("shared_mem_per_block_optin=%zu bytes (%.1f KiB)\n",
              prop.sharedMemPerBlockOptin, prop.sharedMemPerBlockOptin / 1024.0);
  std::printf("shared_mem_per_sm=%zu bytes (%.1f KiB)\n",
              prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor / 1024.0);
  std::printf("reserved_shared_mem_per_block=%zu bytes (%.1f KiB)\n",
              prop.reservedSharedMemPerBlock, prop.reservedSharedMemPerBlock / 1024.0);
  std::printf("l2_cache_size=%d bytes (%.1f KiB)\n", prop.l2CacheSize, prop.l2CacheSize / 1024.0);

  print_attr("attr_max_shared_memory_per_block", cudaDevAttrMaxSharedMemoryPerBlock, true);
  print_attr("attr_max_shared_memory_per_block_optin", cudaDevAttrMaxSharedMemoryPerBlockOptin, true);
  print_attr("attr_max_shared_memory_per_multiprocessor",
             cudaDevAttrMaxSharedMemoryPerMultiprocessor, true);
  print_attr("attr_reserved_shared_memory_per_block",
             cudaDevAttrReservedSharedMemoryPerBlock, true);
  print_attr("attr_max_blocks_per_multiprocessor", cudaDevAttrMaxBlocksPerMultiprocessor);
  print_attr("attr_max_threads_per_multiprocessor", cudaDevAttrMaxThreadsPerMultiProcessor);
  print_attr("attr_max_registers_per_multiprocessor", cudaDevAttrMaxRegistersPerMultiprocessor);
}

static void run_tmem_probe(int columns) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  uint32_t* d_base = nullptr;
  uint32_t h_base = 0;
  CUDA_CHECK(cudaMalloc(&d_base, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_base, 0, sizeof(uint32_t)));
  tmem_alloc_kernel<<<1, 128>>>(columns, d_base);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(&h_base, d_base, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_base));

  std::printf("tmem_probe_columns=%d\n", columns);
  std::printf("tmem_probe_blocks=1\n");
  std::printf("tmem_probe_threads_per_block=128\n");
  std::printf("tmem_alloc_base=%u\n", h_base);
  std::printf("tmem_probe_status=ok\n");
}

int main(int argc, char** argv) {
  if (argc == 3 && std::strcmp(argv[1], "--tmem") == 0) {
    run_tmem_probe(std::atoi(argv[2]));
    return 0;
  }
  print_device_props();
  return 0;
}
