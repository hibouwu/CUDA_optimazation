
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                           \
  cudaError_t err__ = (call);                                           \
  if (err__ != cudaSuccess) {                                           \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                 cudaGetErrorString(err__));                            \
    std::exit(1);                                                       \
  }                                                                     \
} while (0)

static constexpr long long kMacPerInst = 1048576LL;
static constexpr char kPrecision[] = "FP8";
static constexpr char kMode[] = "dense";
static constexpr char kShape[] = "M128N256K32";
static constexpr char kLaunch[] = "FullSM4WarpBlock";
static constexpr int kBlockThreads = 128;
static constexpr int kWarpsPerBlock = 4;

__device__ __forceinline__ uint32_t smem_u32(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ bool warp_leader() {
  return (threadIdx.x & 31) == 0;
}

__device__ __forceinline__ uint64_t make_smem_desc(void const* ptr, uint32_t leading_u128, uint32_t stride_u128) {
  uint32_t addr = smem_u32(ptr);
  uint64_t desc = 0;
  desc |= uint64_t((addr >> 4) & 0x3fff);
  desc |= uint64_t(leading_u128 & 0x3fff) << 16;
  desc |= uint64_t(stride_u128 & 0x3fff) << 32;
  desc |= uint64_t(1) << 46;
  return desc;
}

__device__ __forceinline__ void barrier_init(uint64_t* barrier, uint32_t arrive_count) {
  uint32_t addr = smem_u32(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(arrive_count));
}

__device__ __forceinline__ void barrier_wait(uint64_t* barrier, uint32_t phase) {
  uint32_t addr = smem_u32(barrier);
  uint32_t ticks = 0x989680;
  asm volatile(
      "{ .reg .pred p; wait_loop: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; "
      "@p bra wait_done; bra wait_loop; wait_done: }"
      :: "r"(addr), "r"(phase), "r"(ticks));
}

__device__ __forceinline__ uint64_t make_idesc_bf16() {
  uint32_t d = 0;
  d |= 1u << 4;        // C format F32
  d |= 1u << 7;        // A BF16
  d |= 1u << 10;       // B BF16
  d |= 32u << 17;      // N = 256
  d |= 8u << 24;       // M = 128
  return uint64_t(d) << 32;
}

__device__ __forceinline__ uint64_t make_idesc_fp8() {
  uint32_t d = 0;
  d |= 1u << 4;        // C format F32, A/B E4M3
  d |= 32u << 17;      // N = 256
  d |= 8u << 24;       // M = 128
  return uint64_t(d) << 32;
}

__device__ __forceinline__ uint64_t make_idesc_fp4() {
  uint32_t d = 0;
  d |= 5u << 7;        // A E2M1
  d |= 5u << 10;       // B E2M1
  d |= 32u << 17;      // N = 256
  d |= 0u << 23;       // UE4M3 scale format
  d |= 8u << 24;       // M = 128
  return uint64_t(d) << 32;
}

__global__ __launch_bounds__(128, 1)
void tcgen05_kernel(int iters, unsigned long long* cycles_out) {
  __shared__ alignas(16) uint8_t smem_a[32768];
  __shared__ alignas(16) uint8_t smem_b[32768];
  __shared__ alignas(8) uint64_t done_barrier;
  __shared__ uint32_t tmem_base;

  for (int i = threadIdx.x; i < int(sizeof(smem_a)); i += blockDim.x) {
    smem_a[i] = uint8_t(i * 13 + 1);
    smem_b[i] = uint8_t(i * 17 + 3);
  }

  if (threadIdx.x == 0) {
    barrier_init(&done_barrier, kWarpsPerBlock);
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    uint32_t dst = smem_u32(&tmem_base);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(dst), "r"(512));
  }
  __syncthreads();

  uint64_t desc_a = make_smem_desc(smem_a, 8, 4);
  uint64_t desc_b = make_smem_desc(smem_b, 8, 4);
  uint64_t idesc = make_idesc_fp8();

  __syncthreads();
  unsigned long long start = clock64();

  for (int i = 0; i < iters; ++i) {
    uint32_t scale = (i == 0) ? 0u : 1u;
    uint32_t tmem_c = tmem_base;
    
    if (warp_leader()) {
      asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %4, 0;"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5,%6,%7,%8}, p; }"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idesc >> 32)), "r"(scale),
           "r"(0), "r"(0), "r"(0), "r"(0));
    }
  }

  uint32_t bar_addr = smem_u32(&done_barrier);
  if (warp_leader()) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar_addr));
  }
  if (threadIdx.x == 0) {
    barrier_wait(&done_barrier, 0);
  }
  __syncthreads();

  unsigned long long stop = clock64();
  if (threadIdx.x == 0) {
    cycles_out[blockIdx.x] = stop - start;
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_base), "r"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::);
  }
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? std::atoi(argv[1]) : 10000;
  double freq_hz = argc > 2 ? std::atof(argv[2]) : 1575000000.0;

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount;
  int active_blocks = prop.multiProcessorCount;

  unsigned long long* d_cycles = nullptr;
  unsigned long long* h_cycles = new unsigned long long[blocks];
  CUDA_CHECK(cudaMalloc(&d_cycles, blocks * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_cycles, 0, blocks * sizeof(unsigned long long)));

  tcgen05_kernel<<<blocks, kBlockThreads>>>(iters, d_cycles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_cycles, d_cycles, blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

  unsigned long long max_cycles = 0;
  for (int i = 0; i < blocks; ++i) {
    max_cycles = max_cycles > h_cycles[i] ? max_cycles : h_cycles[i];
  }

  double inst_per_active_block = double(kWarpsPerBlock) * double(iters);
  double macs_per_active_block = inst_per_active_block * double(kMacPerInst);
  double macs_per_cycle_per_active_block = macs_per_active_block / double(max_cycles);
  double thor_macs_per_cycle = macs_per_cycle_per_active_block * double(active_blocks);
  double thor_macs_per_second = thor_macs_per_cycle * freq_hz;
  double thor_tflops = 2.0 * thor_macs_per_second / 1.0e12;

  std::printf("mode=%s\n", kMode);
  std::printf("precision=%s\n", kPrecision);
  std::printf("shape=%s\n", kShape);
  std::printf("launch=%s\n", kLaunch);
  std::printf("sm_count=%d\n", prop.multiProcessorCount);
  std::printf("active_blocks=%d\n", active_blocks);
  std::printf("block_threads=%d\n", kBlockThreads);
  std::printf("warps_per_block=%d\n", kWarpsPerBlock);
  std::printf("iters=%d\n", iters);
  std::printf("cycles=%llu\n", max_cycles);
  std::printf("macs_per_cycle_per_active_block=%.6f\n", macs_per_cycle_per_active_block);
  std::printf("thor_macs_per_cycle=%.6f\n", thor_macs_per_cycle);
  std::printf("thor_tflops=%.6f\n", thor_tflops);

  CUDA_CHECK(cudaFree(d_cycles));
  delete[] h_cycles;
  return 0;
}
