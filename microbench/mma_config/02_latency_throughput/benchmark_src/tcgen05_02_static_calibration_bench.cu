#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err__));                                 \
      std::exit(3);                                                            \
    }                                                                          \
  } while (0)

#define CU_CHECK(call)                                                         \
  do {                                                                         \
    CUresult err__ = (call);                                                   \
    if (err__ != CUDA_SUCCESS) {                                               \
      const char* msg__ = "unknown";                                           \
      cuGetErrorString(err__, &msg__);                                         \
      std::fprintf(stderr, "CUDA driver error %s:%d: %s\n", __FILE__,          \
                   __LINE__, msg__);                                           \
      std::exit(3);                                                            \
    }                                                                          \
  } while (0)

namespace {

enum DType : int { kFp16 = 0, kBf16 = 1 };
enum Mode : int { kEmpty = 0, kCommitWait = 1, kForcedWait = 2, kBatch = 3, kCtaSync = 4 };
enum DMode : int { kSameD = 0, kRingD = 1 };
enum AddrMode : int { kAddrSame = 0, kAddrPingpong = 1, kAddrRotating = 2 };
enum Collector : int {
  kDiscard = 0,
  kFillLastuse = 1,
  kFillUseLastuse = 2,
  kFillUseDiscard = 3
};
enum Interference : int {
  kNoInterference = 0,
  kRegisterAlu = 1,
  kPredOffShared = 2,
  kL1HitGlobal = 3,
  kLdShared = 4,
  kInterferenceOnly = 5
};

struct Result {
  unsigned long long cycles;
  unsigned long long polls;
  int guard_ok;
  float sink;
};

struct HostOptions {
  int dtype = kBf16;
  int n = 128;
  int q = 4;
  int iterations = 512;
  int mode = kBatch;
  int d_mode = kSameD;
  int addr_mode = kAddrSame;
  int collector = kDiscard;
  int input_d = 0;
  int wait_hint = 0;
  int active_blocks = 0;
  int interference = kNoInterference;
  int interference_ops = 0;
};

__device__ __forceinline__ uint32_t smem_u32(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__host__ __device__ constexpr uint64_t encode_smem(uint64_t value) {
  return (value & 0x3ffffULL) >> 4ULL;
}

__device__ __forceinline__ uint64_t make_k_major_sw128_desc(uint32_t smem) {
  constexpr uint32_t span = 128;
  constexpr uint32_t stride_bytes = 8 * span;
  uint64_t desc = encode_smem(smem) |
                  (encode_smem(stride_bytes) << 32ULL) |
                  (1ULL << 46ULL);
  desc |= uint64_t(2) << 61ULL;
  return desc;
}

__device__ __forceinline__ void mbarrier_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(addr), "r"(count)
               : "memory");
}

template <int Hint>
__device__ __forceinline__ void mbarrier_wait(uint32_t addr, uint32_t phase) {
  asm volatile(
      "{ .reg .pred p; wait_loop_%=: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; "
      "@!p bra wait_loop_%=; }"
      :
      : "r"(addr), "r"(phase), "r"(uint32_t(Hint))
      : "memory");
}

template <int Hint>
__device__ __forceinline__ unsigned long long mbarrier_wait_count(uint32_t addr,
                                                                  uint32_t phase) {
  unsigned long long polls = 0;
  uint32_t ready = 0;
  do {
    asm volatile(
        "{ .reg .pred p; "
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3; "
        "selp.u32 %0, 1, 0, p; }"
        : "=r"(ready)
        : "r"(addr), "r"(phase), "r"(uint32_t(Hint))
        : "memory");
    ++polls;
  } while (!ready);
  return polls;
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t addr,
                                                          uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
      :
      : "r"(addr), "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void tma_load_3d(uint32_t dst,
                                            const CUtensorMap* map,
                                            int slot,
                                            uint32_t barrier) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global."
      "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
      :
      : "r"(dst), "l"(map), "r"(0), "r"(0), "r"(slot), "r"(barrier)
      : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc(uint32_t dst_smem, uint32_t cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :
               : "r"(dst_smem), "r"(cols)
               : "memory");
}

__device__ __forceinline__ void tmem_dealloc(uint32_t base, uint32_t cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
               :
               : "r"(base), "r"(cols)
               : "memory");
}

__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_x8(uint32_t addr,
                                              uint32_t v0, uint32_t v1,
                                              uint32_t v2, uint32_t v3,
                                              uint32_t v4, uint32_t v5,
                                              uint32_t v6, uint32_t v7) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], "
      "{%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3), "r"(v4),
        "r"(v5), "r"(v6), "r"(v7)
      : "memory");
}

__device__ __forceinline__ void tmem_store_x1(uint32_t addr, uint32_t value) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
               :
               : "r"(addr), "r"(value)
               : "memory");
}

__device__ __forceinline__ void tmem_load_x1(uint32_t addr, uint32_t& value) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
               : "=r"(value)
               : "r"(addr)
               : "memory");
}

__device__ __forceinline__ void tmem_load_x8(uint32_t addr, float (&values)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]),
        "=f"(values[3]), "=f"(values[4]), "=f"(values[5]),
        "=f"(values[6]), "=f"(values[7])
      : "r"(addr)
      : "memory");
}

template <int DTypeId, int N>
__device__ __forceinline__ uint32_t make_idesc() {
  uint32_t d = 0;
  d |= 1u << 4;
  if constexpr (DTypeId == kBf16) {
    d |= 1u << 7;
    d |= 1u << 10;
  }
  d |= (uint32_t(N) >> 3U) << 17U;
  d |= 8u << 24U;
  return d;
}

__device__ __forceinline__ void issue_mma_discard(uint32_t d_tmem,
                                                  uint64_t a_desc,
                                                  uint64_t b_desc,
                                                  uint32_t idesc,
                                                  uint32_t use_d) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0; "
      "tcgen05.mma.cta_group::1.kind::f16.collector::a::discard "
      "[%0], %1, %2, %3, {%5,%6,%7,%8}, p; }"
      :
      : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(use_d),
        "r"(0), "r"(0), "r"(0), "r"(0)
      : "memory");
}

__device__ __forceinline__ void issue_mma_fill(uint32_t d_tmem,
                                               uint64_t a_desc,
                                               uint64_t b_desc,
                                               uint32_t idesc,
                                               uint32_t use_d) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0; "
      "tcgen05.mma.cta_group::1.kind::f16.collector::a::fill "
      "[%0], %1, %2, %3, {%5,%6,%7,%8}, p; }"
      :
      : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(use_d),
        "r"(0), "r"(0), "r"(0), "r"(0)
      : "memory");
}

__device__ __forceinline__ void issue_mma_use(uint32_t d_tmem,
                                              uint64_t a_desc,
                                              uint64_t b_desc,
                                              uint32_t idesc,
                                              uint32_t use_d) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0; "
      "tcgen05.mma.cta_group::1.kind::f16.collector::a::use "
      "[%0], %1, %2, %3, {%5,%6,%7,%8}, p; }"
      :
      : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(use_d),
        "r"(0), "r"(0), "r"(0), "r"(0)
      : "memory");
}

__device__ __forceinline__ void issue_mma_lastuse(uint32_t d_tmem,
                                                  uint64_t a_desc,
                                                  uint64_t b_desc,
                                                  uint32_t idesc,
                                                  uint32_t use_d) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0; "
      "tcgen05.mma.cta_group::1.kind::f16.collector::a::lastuse "
      "[%0], %1, %2, %3, {%5,%6,%7,%8}, p; }"
      :
      : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(use_d),
        "r"(0), "r"(0), "r"(0), "r"(0)
      : "memory");
}

template <int CollectorMode>
__device__ __forceinline__ void issue_selected_mma(int q,
                                                   int q_last,
                                                   uint32_t d_tmem,
                                                   uint64_t a_desc,
                                                   uint64_t b_desc,
                                                   uint32_t idesc,
                                                   uint32_t use_d) {
  if constexpr (CollectorMode == kDiscard) {
    issue_mma_discard(d_tmem, a_desc, b_desc, idesc, use_d);
  } else if constexpr (CollectorMode == kFillLastuse) {
    if ((q & 1) == 0) {
      issue_mma_fill(d_tmem, a_desc, b_desc, idesc, use_d);
    } else {
      issue_mma_lastuse(d_tmem, a_desc, b_desc, idesc, use_d);
    }
  } else {
    if (q == 0) {
      issue_mma_fill(d_tmem, a_desc, b_desc, idesc, use_d);
    } else if (q == q_last) {
      if constexpr (CollectorMode == kFillUseDiscard) {
        issue_mma_discard(d_tmem, a_desc, b_desc, idesc, use_d);
      } else {
        issue_mma_lastuse(d_tmem, a_desc, b_desc, idesc, use_d);
      }
    } else {
      issue_mma_use(d_tmem, a_desc, b_desc, idesc, use_d);
    }
  }
}

__device__ __forceinline__ void mma_commit(uint32_t barrier_addr) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
      :
      : "r"(barrier_addr)
      : "memory");
}

template <int Hint>
__device__ __forceinline__ unsigned long long commit_wait(uint32_t barrier_addr,
                                                          uint32_t phase) {
  mma_commit(barrier_addr);
  return mbarrier_wait_count<Hint>(barrier_addr, phase);
}

template <int Addr>
__device__ __forceinline__ int slot_for_q(int q) {
  if constexpr (Addr == kAddrPingpong) return q & 1;
  if constexpr (Addr == kAddrRotating) return q & 3;
  return 0;
}

template <int N, int DModeId>
__device__ __forceinline__ int d_base_for_q(int q) {
  if constexpr (DModeId == kSameD) return 0;
  constexpr int kMax = 512 / N;
  return (q % kMax) * N;
}

template <int ModeId>
__device__ __forceinline__ void run_interference(unsigned int* shared_words,
                                                 const float* l1_data,
                                                 int ops,
                                                 float* sink) {
  const int tid = int(threadIdx.x);
  const int warp = tid >> 5;
  if constexpr (ModeId == kNoInterference) {
    return;
  } else {
    if (warp == 0 || warp > 3) return;
    float acc = float(tid & 7);
    for (int i = 0; i < ops; ++i) {
      if constexpr (ModeId == kRegisterAlu) {
        acc = acc * 1.00024414f + 0.125f;
      } else if constexpr (ModeId == kPredOffShared) {
        unsigned int v = 0;
        const unsigned int addr = smem_u32(shared_words + ((tid + i) & 1023));
        asm volatile(
            "{ .reg .pred p; setp.ne.u32 p, %2, %2; @p ld.shared.u32 %0, [%1]; }"
            : "=r"(v)
            : "r"(addr), "r"(i)
            : "memory");
        acc += float(v & 1);
      } else if constexpr (ModeId == kL1HitGlobal) {
        acc += l1_data[(tid + i * 17) & 1023];
      } else {
        unsigned int v = shared_words[(tid * 17 + i * 31) & 1023];
        acc += float(v & 255) * 0.001f;
      }
    }
    if ((tid & 31) == 0) atomicAdd(sink, acc);
  }
}

template <int DTypeId, int N, int Q, int ModeId, int DModeId,
          int InputD, int Addr, int CollectorMode, int Hint, int InterferenceMode>
__global__ __launch_bounds__(128, 1)
void static_calibration_kernel(const __grid_constant__ CUtensorMap map_a,
                               const __grid_constant__ CUtensorMap map_b,
                               int iterations,
                               int interference_ops,
                               float* output,
                               Result* results,
                               const float* l1_data) {
  extern __shared__ __align__(1024) unsigned char smem[];
  constexpr int kSlots = Addr == kAddrRotating ? 4 : (Addr == kAddrPingpong ? 2 : 1);
  constexpr int kAStride = 16 * 1024;
  constexpr int kBStride = 32 * 1024;
  constexpr int kTmemColumns = 512;
  unsigned char* smem_a = smem;
  unsigned char* smem_b = smem_a + kSlots * kAStride;
  unsigned int* shared_words = reinterpret_cast<unsigned int*>(smem_b + kSlots * kBStride);

  __shared__ alignas(16) uint64_t tma_barrier;
  __shared__ alignas(16) uint64_t done_barrier;
  __shared__ alignas(16) uint32_t tmem_base;
  __shared__ int guard_ok;
  __shared__ float sink;

  const int tid = int(threadIdx.x);
  if (tid == 0) {
    guard_ok = 1;
    sink = 0.0f;
    mbarrier_init(smem_u32(&tma_barrier), 1);
    mbarrier_init(smem_u32(&done_barrier), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < 1024; i += blockDim.x) {
    shared_words[i] = unsigned(i * 1103515245u + 12345u);
  }
  __syncthreads();

  uint32_t tma_phase = 0;
  for (int slot = 0; slot < kSlots; ++slot) {
    if (tid == 0) {
      const uint32_t bar = smem_u32(&tma_barrier);
      tma_load_3d(smem_u32(smem_a + slot * kAStride), &map_a, slot, bar);
      tma_load_3d(smem_u32(smem_b + slot * kBStride), &map_b, slot, bar);
      mbarrier_arrive_expect_tx(bar, uint32_t(128 * 16 * 2 + N * 16 * 2));
    }
    mbarrier_wait<0x989680>(smem_u32(&tma_barrier), tma_phase);
    tma_phase ^= 1;
  }
  tcgen05_fence_after_thread_sync();
  __syncthreads();

  if (tid < 32) {
    tmem_alloc(smem_u32(&tmem_base), kTmemColumns);
  }
  __syncthreads();

  for (int col = 0; col < kTmemColumns; col += 8) {
    if (tid < 128) {
      const int warp = tid >> 5;
      const uint32_t addr = tmem_base + ((warp * 32) << 16) + col;
      tmem_store_x8(addr, 0, 0, 0, 0, 0, 0, 0, 0);
    }
  }
  tmem_wait_st();
  __syncthreads();

  constexpr int kGuardCol = DModeId == kSameD ? N : ((512 / N) * N < 512 ? (512 / N) * N : -1);
  if constexpr (kGuardCol >= 0) {
  if (tid < 128) {
    const int warp = tid >> 5;
    tmem_store_x1(tmem_base + ((warp * 32) << 16) + kGuardCol, 0x7f4a1234u);
  }
  }
  tmem_wait_st();
  __syncthreads();

  uint64_t a_desc[kSlots];
  uint64_t b_desc[kSlots];
#pragma unroll
  for (int s = 0; s < kSlots; ++s) {
    a_desc[s] = make_k_major_sw128_desc(smem_u32(smem_a + s * kAStride));
    b_desc[s] = make_k_major_sw128_desc(smem_u32(smem_b + s * kBStride));
  }
  const uint32_t idesc = make_idesc<DTypeId, N>();
  const uint32_t bar = smem_u32(&done_barrier);
  uint32_t phase = 0;
  unsigned long long local_polls = 0;
  uint32_t control_sink = 0;

  __syncthreads();
  const unsigned long long start = clock64();

  for (int iter = 0; iter < iterations; ++iter) {
    if constexpr (ModeId == kEmpty) {
#pragma unroll
      for (int q = 0; q < Q; ++q) {
        control_sink += uint32_t(q + iter + 1);
        asm volatile("add.u32 %0, %0, 1;" : "+r"(control_sink));
      }
    } else if constexpr (ModeId == kCommitWait) {
      if (tid == 0) local_polls += commit_wait<Hint>(bar, phase);
      phase ^= 1;
    } else if constexpr (ModeId == kCtaSync) {
      __syncthreads();
    } else if constexpr (ModeId == kForcedWait) {
#pragma unroll
      for (int q = 0; q < Q; ++q) {
        const int slot = slot_for_q<Addr>(q);
        const uint32_t d_tmem = tmem_base + d_base_for_q<N, DModeId>(q);
        if (tid == 0 && InterferenceMode != kInterferenceOnly) {
          issue_selected_mma<CollectorMode>(q, Q - 1, d_tmem,
                                            a_desc[slot], b_desc[slot],
                                            idesc, InputD ? 1u : 0u);
          local_polls += commit_wait<Hint>(bar, phase);
        }
        phase ^= 1;
        run_interference<InterferenceMode>(shared_words, l1_data,
                                           interference_ops, &sink);
      }
    } else {
#pragma unroll
      for (int q = 0; q < Q; ++q) {
        const int slot = slot_for_q<Addr>(q);
        const uint32_t d_tmem = tmem_base + d_base_for_q<N, DModeId>(q);
        if (tid == 0 && InterferenceMode != kInterferenceOnly) {
          issue_selected_mma<CollectorMode>(q, Q - 1, d_tmem,
                                            a_desc[slot], b_desc[slot],
                                            idesc, InputD ? 1u : 0u);
        }
        run_interference<InterferenceMode>(shared_words, l1_data,
                                           interference_ops, &sink);
      }
      if (tid == 0 && InterferenceMode != kInterferenceOnly) {
        local_polls += commit_wait<Hint>(bar, phase);
      }
      phase ^= 1;
    }
  }

  const unsigned long long stop = clock64();
  if (tid == 0) {
    results[blockIdx.x].cycles = stop - start;
    results[blockIdx.x].polls = local_polls;
    results[blockIdx.x].sink = sink + float(control_sink & 0xffffu);
  }
  __syncthreads();

  if constexpr (kGuardCol >= 0) {
  if (tid < 128) {
    const int warp = tid >> 5;
    uint32_t guard = 0;
    tmem_load_x1(tmem_base + ((warp * 32) << 16) + kGuardCol, guard);
    tmem_wait_ld();
    if (guard != 0x7f4a1234u) atomicExch(&guard_ok, 0);
  }
  }
  __syncthreads();

  for (int col = 0; col < kTmemColumns; col += 8) {
    if (tid < 128) {
      const int warp = tid >> 5;
      const int lane = tid & 31;
      float values[8];
      tmem_load_x8(tmem_base + ((warp * 32) << 16) + col, values);
      tmem_wait_ld();
      const int row = warp * 32 + lane;
      for (int j = 0; j < 8; ++j) {
        output[((blockIdx.x * 128 + row) * kTmemColumns) + col + j] = values[j];
      }
    }
  }
  __syncthreads();

  if (tid == 0) results[blockIdx.x].guard_ok = guard_ok;
  __syncthreads();

  if (tid < 32) {
    tmem_dealloc(tmem_base, kTmemColumns);
    tmem_relinquish();
  }
}

int parse_int(int argc, char** argv, const char* key, int def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], key) == 0) return std::atoi(argv[i + 1]);
  }
  return def;
}

std::string parse_str(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  }
  return def;
}

int parse_dtype(const std::string& s) {
  if (s == "fp16") return kFp16;
  if (s == "bf16") return kBf16;
  std::cerr << "bad dtype\n";
  std::exit(2);
}

int parse_mode(const std::string& s) {
  if (s == "empty") return kEmpty;
  if (s == "commit_wait") return kCommitWait;
  if (s == "forced_wait") return kForcedWait;
  if (s == "batch") return kBatch;
  if (s == "cta_sync") return kCtaSync;
  std::cerr << "bad mode\n";
  std::exit(2);
}

int parse_d_mode(const std::string& s) {
  if (s == "same") return kSameD;
  if (s == "ring") return kRingD;
  std::cerr << "bad d mode\n";
  std::exit(2);
}

int parse_addr_mode(const std::string& s) {
  if (s == "same") return kAddrSame;
  if (s == "pingpong") return kAddrPingpong;
  if (s == "rotating") return kAddrRotating;
  std::cerr << "bad addr mode\n";
  std::exit(2);
}

int parse_collector(const std::string& s) {
  if (s == "discard") return kDiscard;
  if (s == "fill_lastuse") return kFillLastuse;
  if (s == "fill_use_lastuse") return kFillUseLastuse;
  if (s == "fill_use_discard") return kFillUseDiscard;
  std::cerr << "bad collector\n";
  std::exit(2);
}

int parse_interference(const std::string& s) {
  if (s == "none") return kNoInterference;
  if (s == "register_alu") return kRegisterAlu;
  if (s == "predicated_off_load") return kPredOffShared;
  if (s == "l1_hit_global") return kL1HitGlobal;
  if (s == "ld_shared") return kLdShared;
  if (s == "interference_only") return kInterferenceOnly;
  std::cerr << "bad interference\n";
  std::exit(2);
}

CUtensorMapSwizzle swizzle128() { return CU_TENSOR_MAP_SWIZZLE_128B; }

void encode_tma_3d(CUtensorMap* map, void* base, int dtype, int rows, int slots) {
  const uint32_t rank = 3;
  uint64_t global_dim[rank] = {16u, uint64_t(rows), uint64_t(slots)};
  uint64_t global_stride[rank - 1] = {32u, uint64_t(rows * 16 * 2)};
  uint32_t box_dim[rank] = {16u, uint32_t(rows), 1u};
  uint32_t element_stride[rank] = {1u, 1u, 1u};
  CUtensorMapDataType tma_dtype =
      dtype == kBf16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  CU_CHECK(cuTensorMapEncodeTiled(map, tma_dtype, rank, base, global_dim,
                                  global_stride, box_dim, element_stride,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle128(),
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

float make_a_value(int slot, int row, int k) {
  return 0.03125f * float(((slot * 11 + row * 5 + k * 3) % 17) - 8);
}

float make_b_value(int slot, int col, int k) {
  return 0.03125f * float(((slot * 7 + col * 13 + k * 9) % 19) - 9);
}

template <typename T>
float to_float(T v);

template <>
float to_float<half>(half v) { return __half2float(v); }

template <>
float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
void fill_operand(int slots, int rows, bool is_a,
                  std::vector<T>& storage,
                  std::vector<float>& quantized) {
  storage.resize(size_t(slots) * rows * 16);
  quantized.resize(storage.size());
  for (int s = 0; s < slots; ++s) {
    for (int r = 0; r < rows; ++r) {
      for (int k = 0; k < 16; ++k) {
        const float v = is_a ? make_a_value(s, r, k) : make_b_value(s, r, k);
        T q;
        if constexpr (std::is_same<T, half>::value) q = __float2half(v);
        else q = __float2bfloat16(v);
        const size_t idx = (size_t(s) * rows + r) * 16 + k;
        storage[idx] = q;
        quantized[idx] = to_float<T>(q);
      }
    }
  }
}

int slot_host(int addr, int q) {
  if (addr == kAddrPingpong) return q & 1;
  if (addr == kAddrRotating) return q & 3;
  return 0;
}

int d_base_host(int n, int dmode, int q) {
  if (dmode == kSameD) return 0;
  return (q % (512 / n)) * n;
}

int a_slot_for_collector(int collector, int addr, int q, int q_last) {
  if (collector == kDiscard) return slot_host(addr, q);
  if (collector == kFillLastuse) return slot_host(addr, q & ~1);
  if (collector == kFillUseDiscard && q == q_last) return slot_host(addr, q);
  return slot_host(addr, 0);
}

void product_for_slots(const std::vector<float>& a,
                       const std::vector<float>& b,
                       int a_slot,
                       int b_slot,
                       int n,
                       std::vector<float>& product) {
  product.assign(size_t(128) * n, 0.0f);
  for (int row = 0; row < 128; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int k = 0; k < 16; ++k) {
        const float av = a[(size_t(a_slot) * 128 + row) * 16 + k];
        const float bv = b[(size_t(b_slot) * n + col) * 16 + k];
        sum += av * bv;
      }
      product[size_t(row) * n + col] = sum;
    }
  }
}

double max_abs_error(const std::vector<float>& got,
                     const std::vector<float>& a,
                     const std::vector<float>& b,
                     const HostOptions& o,
                     int blocks) {
  std::vector<float> ref(size_t(128) * 512, 0.0f);
  std::vector<unsigned char> touched(512, 0);
  for (int iter = 0; iter < o.iterations; ++iter) {
      if (o.mode == kEmpty || o.mode == kCommitWait || o.interference == kInterferenceOnly) break;
    for (int q = 0; q < o.q; ++q) {
      const int a_slot = a_slot_for_collector(o.collector, o.addr_mode, q, o.q - 1);
      const int b_slot = slot_host(o.addr_mode, q);
      const int base = d_base_host(o.n, o.d_mode, q);
      std::vector<float> product;
      product_for_slots(a, b, a_slot, b_slot, o.n, product);
      for (int col = 0; col < o.n; ++col) touched[base + col] = 1;
      for (int row = 0; row < 128; ++row) {
        for (int col = 0; col < o.n; ++col) {
          const size_t dst = size_t(row) * 512 + base + col;
          const float p = product[size_t(row) * o.n + col];
          if (o.input_d) ref[dst] += p;
          else ref[dst] = p;
        }
      }
    }
  }
  double err = 0.0;
  for (int block = 0; block < blocks; ++block) {
    const size_t block_base = size_t(block) * 128 * 512;
    for (int row = 0; row < 128; ++row) {
      for (int col = 0; col < 512; ++col) {
        if (!touched[col]) continue;
        const size_t idx = size_t(row) * 512 + col;
        err = std::max(err, std::abs(double(got[block_base + idx]) - double(ref[idx])));
      }
    }
  }
  return err;
}

template <int DTypeId, int N, int Q, int ModeId, int DModeId,
          int InputD, int Addr, int CollectorMode, int Hint, int InterferenceMode>
void launch_typed(const CUtensorMap& map_a, const CUtensorMap& map_b,
                  const HostOptions& o, int blocks, float* d_output,
                  Result* d_results, const float* d_l1, cudaEvent_t start_ev,
                  cudaEvent_t stop_ev) {
  constexpr int kSlots = Addr == kAddrRotating ? 4 : (Addr == kAddrPingpong ? 2 : 1);
  constexpr int kSmemBytes = kSlots * (16 * 1024 + 32 * 1024) + 4096;
  auto kernel = static_calibration_kernel<DTypeId, N, Q, ModeId, DModeId,
                                          InputD, Addr, CollectorMode, Hint,
                                          InterferenceMode>;
  if (kSmemBytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kSmemBytes));
  }
  CUDA_CHECK(cudaEventRecord(start_ev));
  kernel<<<blocks, 128, kSmemBytes>>>(map_a, map_b, o.iterations,
                                      o.interference_ops, d_output,
                                      d_results, d_l1);
  CUDA_CHECK(cudaEventRecord(stop_ev));
}

#define DISPATCH_HINT(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, COLLVAL, IMODEVAL) \
  if (o.wait_hint == 0) { \
    launch_typed<DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, COLLVAL, 0, IMODEVAL>(map_a, map_b, o, blocks, d_output, d_results, d_l1, start_ev, stop_ev); \
  } else if (o.wait_hint == 32) { \
    launch_typed<DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, COLLVAL, 32, IMODEVAL>(map_a, map_b, o, blocks, d_output, d_results, d_l1, start_ev, stop_ev); \
  } else { \
    launch_typed<DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, COLLVAL, 0x989680, IMODEVAL>(map_a, map_b, o, blocks, d_output, d_results, d_l1, start_ev, stop_ev); \
  }

#define DISPATCH_ALL(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, COLLVAL) \
  if (o.interference == kNoInterference) { DISPATCH_HINT(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, COLLVAL, kNoInterference) } \
  else if (o.interference == kRegisterAlu) { DISPATCH_HINT(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, COLLVAL, kRegisterAlu) } \
  else if (o.interference == kPredOffShared) { DISPATCH_HINT(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, COLLVAL, kPredOffShared) } \
  else if (o.interference == kL1HitGlobal) { DISPATCH_HINT(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, COLLVAL, kL1HitGlobal) } \
  else if (o.interference == kLdShared) { DISPATCH_HINT(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, COLLVAL, kLdShared) } \
  else { DISPATCH_HINT(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, COLLVAL, kInterferenceOnly) }

#define DISPATCH_COLL(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL) \
  if (o.collector == kDiscard) { DISPATCH_ALL(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, kDiscard) } \
  else if (o.collector == kFillLastuse) { DISPATCH_ALL(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, kFillLastuse) } \
  else if (o.collector == kFillUseLastuse) { DISPATCH_ALL(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, kFillUseLastuse) } \
  else { DISPATCH_ALL(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, ADDRVAL, kFillUseDiscard) }

#define DISPATCH_ADDR(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL) \
  if (o.addr_mode == kAddrSame) { DISPATCH_COLL(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, kAddrSame) } \
  else if (o.addr_mode == kAddrPingpong) { DISPATCH_COLL(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, kAddrPingpong) } \
  else { DISPATCH_COLL(DT, NVAL, QVAL, MODEVAL, DMODEVAL, INVAL, kAddrRotating) }

#define DISPATCH_INPUT(DT, NVAL, QVAL, MODEVAL, DMODEVAL) \
  if (o.input_d) { DISPATCH_ADDR(DT, NVAL, QVAL, MODEVAL, DMODEVAL, 1) } \
  else { DISPATCH_ADDR(DT, NVAL, QVAL, MODEVAL, DMODEVAL, 0) }

#define DISPATCH_DMODE(DT, NVAL, QVAL, MODEVAL) \
  if (o.d_mode == kSameD) { DISPATCH_INPUT(DT, NVAL, QVAL, MODEVAL, kSameD) } \
  else { DISPATCH_INPUT(DT, NVAL, QVAL, MODEVAL, kRingD) }

#define DISPATCH_MODE(DT, NVAL, QVAL) \
  if (o.mode == kEmpty) { DISPATCH_DMODE(DT, NVAL, QVAL, kEmpty) } \
  else if (o.mode == kCommitWait) { DISPATCH_DMODE(DT, NVAL, QVAL, kCommitWait) } \
  else if (o.mode == kForcedWait) { DISPATCH_DMODE(DT, NVAL, QVAL, kForcedWait) } \
  else { DISPATCH_DMODE(DT, NVAL, QVAL, kBatch) }

#define DISPATCH_Q(DT, NVAL) \
  if (o.q == 1) { DISPATCH_MODE(DT, NVAL, 1) } \
  else if (o.q == 2) { DISPATCH_MODE(DT, NVAL, 2) } \
  else if (o.q == 4) { DISPATCH_MODE(DT, NVAL, 4) } \
  else if (o.q == 8) { DISPATCH_MODE(DT, NVAL, 8) } \
  else if (o.q == 16) { DISPATCH_MODE(DT, NVAL, 16) } \
  else if (o.q == 32) { DISPATCH_MODE(DT, NVAL, 32) } \
  else if (o.q == 64) { DISPATCH_MODE(DT, NVAL, 64) } \
  else { std::cerr << "unsupported q\n"; return 2; }

int run_case(int argc, char** argv) {
  HostOptions o;
  const std::string dtype_s = parse_str(argc, argv, "--dtype", "bf16");
  o.dtype = parse_dtype(dtype_s);
  o.n = parse_int(argc, argv, "--n", 128);
  o.q = parse_int(argc, argv, "--q", 4);
  o.iterations = parse_int(argc, argv, "--iterations", 512);
  o.mode = parse_mode(parse_str(argc, argv, "--mode", "batch"));
  o.d_mode = parse_d_mode(parse_str(argc, argv, "--d-mode", "same"));
  o.addr_mode = parse_addr_mode(parse_str(argc, argv, "--addr-mode", "same"));
  o.collector = parse_collector(parse_str(argc, argv, "--collector", "discard"));
  o.input_d = parse_int(argc, argv, "--input-d", 0);
  o.wait_hint = parse_int(argc, argv, "--wait-hint", 0);
  o.active_blocks = parse_int(argc, argv, "--active-blocks", 0);
  o.interference = parse_interference(parse_str(argc, argv, "--interference", "none"));
  o.interference_ops = parse_int(argc, argv, "--interference-ops", 0);

  CUDA_CHECK(cudaFree(nullptr));
  CU_CHECK(cuInit(0));
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  if (prop.major != 11) {
    std::cout << "status=skipped_non_sm110\n";
    return 0;
  }
  int blocks = o.active_blocks > 0 ? std::min(o.active_blocks, prop.multiProcessorCount)
                                   : prop.multiProcessorCount;

  const int slots = o.addr_mode == kAddrRotating ? 4 : (o.addr_mode == kAddrPingpong ? 2 : 1);
  std::vector<float> a_quant, b_quant;
  void* d_a = nullptr;
  void* d_b = nullptr;
  if (o.dtype == kFp16) {
    std::vector<half> a_host, b_host;
    fill_operand(slots, 128, true, a_host, a_quant);
    fill_operand(slots, o.n, false, b_host, b_quant);
    CUDA_CHECK(cudaMalloc(&d_a, a_host.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, b_host.size() * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_a, a_host.data(), a_host.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b_host.data(), b_host.size() * sizeof(half), cudaMemcpyHostToDevice));
  } else {
    std::vector<__nv_bfloat16> a_host, b_host;
    fill_operand(slots, 128, true, a_host, a_quant);
    fill_operand(slots, o.n, false, b_host, b_quant);
    CUDA_CHECK(cudaMalloc(&d_a, a_host.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_host.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_a, a_host.data(), a_host.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b_host.data(), b_host.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  }

  CUtensorMap map_a{}, map_b{};
  encode_tma_3d(&map_a, d_a, o.dtype, 128, slots);
  encode_tma_3d(&map_b, d_b, o.dtype, o.n, slots);

  std::vector<float> l1_host(1024);
  for (int i = 0; i < 1024; ++i) l1_host[i] = float((i * 17) & 255) * 0.001f;
  float* d_l1 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_l1, l1_host.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_l1, l1_host.data(), l1_host.size() * sizeof(float), cudaMemcpyHostToDevice));

  float* d_output = nullptr;
  Result* d_results = nullptr;
  const size_t output_count = size_t(blocks) * 128 * 512;
  CUDA_CHECK(cudaMalloc(&d_output, output_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_results, blocks * sizeof(Result)));
  CUDA_CHECK(cudaMemset(d_output, 0, output_count * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_results, 0, blocks * sizeof(Result)));
  cudaEvent_t start_ev{}, stop_ev{};
  CUDA_CHECK(cudaEventCreate(&start_ev));
  CUDA_CHECK(cudaEventCreate(&stop_ev));

#ifdef STATIC_SINGLE_CASE
  launch_typed<CFG_DTYPE, CFG_N, CFG_Q, CFG_MODE, CFG_DMODE,
               CFG_INPUT_D, CFG_ADDR, CFG_COLLECTOR, CFG_WAIT_HINT,
               CFG_INTERFERENCE>(map_a, map_b, o, blocks, d_output,
                                  d_results, d_l1, start_ev, stop_ev);
#else
  if (o.dtype == kBf16 && o.n == 128) { DISPATCH_Q(kBf16, 128) }
  else if (o.dtype == kBf16 && o.n == 256) { DISPATCH_Q(kBf16, 256) }
  else if (o.dtype == kFp16 && o.n == 128) { DISPATCH_Q(kFp16, 128) }
  else if (o.dtype == kFp16 && o.n == 256) { DISPATCH_Q(kFp16, 256) }
  else {
    std::cerr << "unsupported dtype/n\n";
    return 2;
  }
#endif

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventSynchronize(stop_ev));
  CUDA_CHECK(cudaDeviceSynchronize());
  float event_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&event_ms, start_ev, stop_ev));

  std::vector<Result> h_results(blocks);
  std::vector<float> h_output(output_count);
  CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, blocks * sizeof(Result), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost));

  unsigned long long max_cycles = 0;
  unsigned long long polls = 0;
  int guard_ok = 1;
  double sink = 0.0;
  for (const auto& r : h_results) {
    max_cycles = std::max(max_cycles, r.cycles);
    polls += r.polls;
    guard_ok &= r.guard_ok;
    sink += r.sink;
  }
  const double err = max_abs_error(h_output, a_quant, b_quant, o, blocks);

  std::cout << "status=ok\n";
  std::cout << "gpu=" << prop.name << "\n";
  std::cout << "compute_capability=" << prop.major << "." << prop.minor << "\n";
  std::cout << "sm_count=" << prop.multiProcessorCount << "\n";
  std::cout << "launch_blocks=" << blocks << "\n";
  std::cout << "dtype=" << dtype_s << "\n";
  std::cout << "n=" << o.n << "\n";
  std::cout << "q=" << o.q << "\n";
  std::cout << "iterations=" << o.iterations << "\n";
  std::cout << "mode=" << parse_str(argc, argv, "--mode", "batch") << "\n";
  std::cout << "d_mode=" << parse_str(argc, argv, "--d-mode", "same") << "\n";
  std::cout << "addr_mode=" << parse_str(argc, argv, "--addr-mode", "same") << "\n";
  std::cout << "collector=" << parse_str(argc, argv, "--collector", "discard") << "\n";
  std::cout << "input_d=" << o.input_d << "\n";
  std::cout << "wait_hint=" << o.wait_hint << "\n";
  std::cout << "interference=" << parse_str(argc, argv, "--interference", "none") << "\n";
  std::cout << "interference_ops=" << o.interference_ops << "\n";
  std::cout << "elapsed_cycles=" << max_cycles << "\n";
  std::cout << "event_ms=" << std::fixed << std::setprecision(6) << event_ms << "\n";
  std::cout << "poll_count=" << polls << "\n";
  std::cout << "max_abs_error=" << std::setprecision(9) << err << "\n";
  std::cout << "guard_ok=" << guard_ok << "\n";
  std::cout << "interference_sink=" << sink << "\n";

  CUDA_CHECK(cudaEventDestroy(start_ev));
  CUDA_CHECK(cudaEventDestroy(stop_ev));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_l1));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_results));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  return run_case(argc, argv);
}
