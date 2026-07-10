
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

static constexpr long long kMacPerInst = 2097152LL;
static constexpr int kBlockThreads = 128;
static constexpr int kWarpsPerBlock = 4;
static constexpr int kTmemAllocCols = 512;
static constexpr int kMainloopKBlocks = 16;
static constexpr int kNoiseCpPerMma = 0;
static constexpr int kRequiredCpPerMma = 0;
static constexpr int kCommitArriveCount = 1;
static constexpr int kInitialBarrierArriveCount = 1;
static constexpr int kMmaAPanelBytes = 4096;
static constexpr int kMmaBPanelBytes = 8192;
static constexpr int kDescLeading = 4;
static constexpr int kDescStride = 2;
static constexpr int kEffectiveBytesPerCp = 2048;
static constexpr char kCaseId[] = "ss_mainloop_k16_base";
static constexpr char kCaseLabel[] = "SS mainloop K16 baseline";
static constexpr char kMmaPath[] = "SS";
static constexpr char kIssueMode[] = "same";
static constexpr char kPrecision[] = "FP4";
static constexpr char kShape[] = "M128N256K64";
static constexpr char kCpSuffix[] = "128x128b.b8x16.b4x16_p64";

__device__ __forceinline__ uint32_t smem_u32(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ bool same_issuer_thread() {
  return threadIdx.x == 0;
}

__device__ __forceinline__ bool split_cp_issuer_thread() {
  return threadIdx.x == 0;
}

__device__ __forceinline__ bool split_mma_issuer_thread() {
  return threadIdx.x == 32;
}

__device__ __forceinline__ bool split_issuer_thread() {
  return split_cp_issuer_thread() || split_mma_issuer_thread();
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

__device__ __forceinline__ void commit_and_wait_from(uint64_t* barrier, uint32_t phase, bool should_commit) {
  uint32_t bar_addr = smem_u32(barrier);
  if (should_commit) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar_addr));
  }
  if (threadIdx.x == 0) {
    barrier_wait(barrier, phase);
  }
  __syncthreads();
}

__device__ __forceinline__ uint64_t make_idesc_bf16() {
  uint32_t d = 0;
  d |= 1u << 4;
  d |= 1u << 7;
  d |= 1u << 10;
  d |= 32u << 17;
  d |= 8u << 24;
  return uint64_t(d) << 32;
}

__device__ __forceinline__ uint64_t make_idesc_fp8() {
  uint32_t d = 0;
  d |= 1u << 4;
  d |= 32u << 17;
  d |= 8u << 24;
  return uint64_t(d) << 32;
}

__device__ __forceinline__ uint64_t make_idesc_fp4() {
  uint32_t d = 0;
  d |= 5u << 7;
  d |= 5u << 10;
  d |= 32u << 17;
  d |= 0u << 23;
  d |= 8u << 24;
  return uint64_t(d) << 32;
}

__device__ __forceinline__ void issue_ss_mma(
    uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint64_t idesc,
    uint32_t tsfa, uint32_t tsfb, uint32_t scale) {
  asm volatile(
    "{ .reg .pred p; setp.ne.b32 p, %4, 0;"
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; }"
    :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(uint32_t(idesc >> 32)), "r"(scale),
       "r"(tsfa), "r"(tsfb));
}

__device__ __forceinline__ void issue_ts_mma(
    uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint64_t idesc,
    uint32_t tsfa, uint32_t tsfb, uint32_t scale) {
  asm volatile(
    "{ .reg .pred p; setp.ne.b32 p, %4, 0;"
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], [%1], %2, %3, [%5], [%6], p; }"
    :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(uint32_t(idesc >> 32)), "r"(scale),
       "r"(tsfa), "r"(tsfb));
}

__device__ __forceinline__ void issue_cp(uint32_t taddr, uint64_t s_desc) {
  asm volatile("tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [%0], %1;" :: "r"(taddr), "l"(s_desc) : "memory");
}

__global__ __launch_bounds__(128, 1)
void tcgen05_kernel(int iters, unsigned long long* cycles_out) {
  __shared__ alignas(16) uint8_t smem_a[65536];
  __shared__ alignas(16) uint8_t smem_b[131072];
  __shared__ alignas(8) uint64_t done_barrier;
  __shared__ uint32_t tmem_base;

  for (int i = threadIdx.x; i < int(sizeof(smem_a)); i += blockDim.x) {
    smem_a[i] = uint8_t(i * 13 + 1);
  }
  for (int i = threadIdx.x; i < int(sizeof(smem_b)); i += blockDim.x) {
    smem_b[i] = uint8_t(i * 17 + 3);
  }

  if (threadIdx.x == 0) {
    barrier_init(&done_barrier, kInitialBarrierArriveCount);
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    uint32_t dst = smem_u32(&tmem_base);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(dst), "r"(kTmemAllocCols));
  }
  __syncthreads();

  uint64_t a_desc = make_smem_desc(smem_a, kDescLeading, kDescStride);
  uint64_t b_desc = make_smem_desc(smem_b, kDescLeading, kDescStride);
  uint64_t idesc = make_idesc_fp4();
  uint32_t d_tmem = tmem_base;
  uint32_t a_tmem0 = tmem_base + 256;
  uint32_t a_tmem1 = tmem_base + 320;
  uint32_t tsfa = tmem_base + 384; uint32_t tsfb = tmem_base + 448;

  uint32_t phase = 0;


  __syncthreads();
  unsigned long long start = clock64();


  for (int i = 0; i < iters; ++i) {
    for (int k_block = 0; k_block < kMainloopKBlocks; ++k_block) {
      uint32_t scale = ((i == 0) && (k_block == 0)) ? 0u : 1u;
      uint64_t a_desc_k = make_smem_desc(
          smem_a + k_block * uint32_t(kMmaAPanelBytes), kDescLeading, kDescStride);
      uint64_t b_desc_k = make_smem_desc(
          smem_b + k_block * uint32_t(kMmaBPanelBytes), kDescLeading, kDescStride);
      if (same_issuer_thread()) {
        for (int noise = 0; noise < kNoiseCpPerMma; ++noise) {
          uint32_t dst = (noise & 1) ? a_tmem1 : a_tmem0;
          issue_cp(dst, a_desc_k);
        }
        issue_ss_mma(d_tmem, a_desc_k, b_desc_k, idesc, tsfa, tsfb, scale);
      }
    }
    commit_and_wait_from(&done_barrier, phase, same_issuer_thread());
    phase ^= 1;
  }


  unsigned long long stop = clock64();
  if (threadIdx.x == 0) {
    cycles_out[blockIdx.x] = stop - start;
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_base), "r"(kTmemAllocCols));
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

  long long mma_instruction_count =
      static_cast<long long>(active_blocks) * static_cast<long long>(kMainloopKBlocks) * static_cast<long long>(iters);
  long long required_cp_instruction_count =
      static_cast<long long>(active_blocks) * static_cast<long long>(kMainloopKBlocks) *
      static_cast<long long>(kRequiredCpPerMma) * static_cast<long long>(iters);
  long long noise_cp_instruction_count =
      static_cast<long long>(active_blocks) * static_cast<long long>(kMainloopKBlocks) *
      static_cast<long long>(kNoiseCpPerMma) * static_cast<long long>(iters);
  long long cp_instruction_count = required_cp_instruction_count + noise_cp_instruction_count;
  long long k_groups = static_cast<long long>(active_blocks) * static_cast<long long>(iters);
  double elapsed_seconds = double(max_cycles) / freq_hz;
  double tflops = 2.0 * double(kMacPerInst) * double(mma_instruction_count) / elapsed_seconds / 1.0e12;
  double bytes_per_cycle = cp_instruction_count > 0
      ? double(cp_instruction_count) * double(kEffectiveBytesPerCp) / double(max_cycles)
      : 0.0;
  double cycles_per_cp = cp_instruction_count > 0
      ? double(max_cycles) / double(cp_instruction_count)
      : 0.0;
  double cycles_per_cta_iter = double(max_cycles) / double(iters);
  double cycles_per_mma = double(max_cycles) / double(static_cast<long long>(iters) * kMainloopKBlocks);

  std::printf("case_id=%s\n", kCaseId);
  std::printf("case_label=%s\n", kCaseLabel);
  std::printf("mma_path=%s\n", kMmaPath);
  std::printf("issue_mode=%s\n", kIssueMode);
  std::printf("precision=%s\n", kPrecision);
  std::printf("shape=%s\n", kShape);
  std::printf("cp_suffix=%s\n", kCpSuffix);
  std::printf("sm_count=%d\n", prop.multiProcessorCount);
  std::printf("active_blocks=%d\n", active_blocks);
  std::printf("block_threads=%d\n", kBlockThreads);
  std::printf("warps_per_block=%d\n", kWarpsPerBlock);
  std::printf("commit_arrive_count=%d\n", kCommitArriveCount);
  std::printf("iters=%d\n", iters);
  std::printf("k_blocks=%d\n", kMainloopKBlocks);
  std::printf("noise_cp_per_mma=%d\n", kNoiseCpPerMma);
  std::printf("required_cp_per_mma=%d\n", kRequiredCpPerMma);
  std::printf("cycles=%llu\n", max_cycles);
  std::printf("mma_instruction_count=%lld\n", mma_instruction_count);
  std::printf("required_cp_instruction_count=%lld\n", required_cp_instruction_count);
  std::printf("noise_cp_instruction_count=%lld\n", noise_cp_instruction_count);
  std::printf("cp_instruction_count=%lld\n", cp_instruction_count);
  std::printf("k_groups=%lld\n", k_groups);
  std::printf("effective_bytes_per_cp=%d\n", kEffectiveBytesPerCp);
  std::printf("thor_tflops=%.6f\n", tflops);
  std::printf("bytes_per_cycle=%.6f\n", bytes_per_cycle);
  std::printf("cycles_per_cp=%.6f\n", cycles_per_cp);
  std::printf("cycles_per_cta_iter=%.6f\n", cycles_per_cta_iter);
  std::printf("cycles_per_mma=%.6f\n", cycles_per_mma);

  CUDA_CHECK(cudaFree(d_cycles));
  delete[] h_cycles;
  return 0;
}
