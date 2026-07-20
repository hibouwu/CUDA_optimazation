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
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#define MMA_CONFIG_CUDA_CHECK(call)                                      \
  do {                                                                  \
    cudaError_t err__ = (call);                                         \
    if (err__ != cudaSuccess) {                                         \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,         \
                   __LINE__, cudaGetErrorString(err__));                \
      std::exit(3);                                                     \
    }                                                                   \
  } while (0)

#define MMA_CONFIG_CU_CHECK(call)                                        \
  do {                                                                  \
    CUresult err__ = (call);                                            \
    if (err__ != CUDA_SUCCESS) {                                        \
      const char* msg__ = "unknown";                                    \
      cuGetErrorString(err__, &msg__);                                  \
      std::fprintf(stderr, "CUDA driver error %s:%d: %s\n", __FILE__,  \
                   __LINE__, msg__);                                    \
      std::exit(3);                                                     \
    }                                                                   \
  } while (0)

namespace mma_config {

enum DTypeId : int { kFp16 = 0, kBf16 = 1 };
enum LayoutId : int { kLayoutNone = 0, kLayoutSw32 = 1, kLayoutSw64 = 2, kLayoutSw128 = 3 };
enum CollectorProtocol : int {
  kCollectorNone = 0,
  kCollectorDiscard = 1,
  kCollectorFillLastuse = 2,
  kCollectorFillUseLastuse = 3,
  kCollectorFillUseDiscard = 4
};
enum OperandAddressMode : int { kOperandSame = 0, kOperandPingpong = 1, kOperandRotating = 2 };
enum InterferenceMode : int {
  kInterferenceNone = 0,
  kInterferenceAlu = 1,
  kInterferencePredOffLoad = 2,
  kInterferenceL1HitGlobal = 3,
  kInterferenceLdShared = 4,
  kInterferenceOnly = 5
};
enum AliasClass : int { kAliasNone = 0, kAliasPartial = 1, kAliasFull = 2 };
enum WaitMode : int { kWaitNoCount = 0, kWaitCount = 1 };

struct Options {
  int dtype;
  int n;
  int layout;
  int q;
  int iterations;
  int collector_protocol;
  int collector_reuse;
  int ws_mode;
  int ws_buffer_count;
  int operand_address_mode;
  int input_d;
  int tmem_columns;
  int d_base_column;
  int d_tile_base_delta;
  int independent_d_count;
  int d_reuse_distance;
  int commit_interval;
  int pending_mbarriers;
  int wait_mode;
  int smem_base_offset;
  int interference_mode;
  int interference_ops_per_iter;
  int interference_warps;
  int active_blocks;
  int operand_slots;
  int a_bytes;
  int b_bytes;
  int a_slot_stride;
  int b_slot_stride;
  int dynamic_smem_bytes;
};

struct KernelResult {
  unsigned long long cycles;
  unsigned long long poll_count;
  int guard_ok;
  int cuda_arch;
  float interference_sink;
};

__device__ __forceinline__ uint32_t smem_u32(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__host__ __device__ constexpr uint64_t encode_smem(uint64_t value) {
  return (value & 0x3ffffULL) >> 4ULL;
}

__device__ __forceinline__ uint64_t make_k_major_desc(uint32_t smem, int layout) {
  uint32_t span = 16;
  uint32_t swizzle_code = 0;
  if (layout == kLayoutSw32) {
    span = 32;
    swizzle_code = 6;
  } else if (layout == kLayoutSw64) {
    span = 64;
    swizzle_code = 4;
  } else if (layout == kLayoutSw128) {
    span = 128;
    swizzle_code = 2;
  }
  const uint32_t stride_bytes = 8 * span;
  uint64_t desc = encode_smem(smem) |
                  (encode_smem(stride_bytes) << 32ULL) |
                  (1ULL << 46ULL);
  if (swizzle_code != 0) desc |= static_cast<uint64_t>(swizzle_code) << 61ULL;
  return desc;
}

__device__ __forceinline__ void mbarrier_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(addr), "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait_nocount(uint32_t addr, uint32_t phase) {
  constexpr uint32_t kTicks = 0x989680;
  asm volatile(
      "{ .reg .pred p; wait_loop_%=: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; "
      "@!p bra wait_loop_%=; }"
      :
      : "r"(addr), "r"(phase), "r"(kTicks)
      : "memory");
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_once(uint32_t addr,
                                                           uint32_t phase) {
  constexpr uint32_t kTicks = 0x989680;
  uint32_t ready = 0;
  asm volatile(
      "{ .reg .pred p; "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3; "
      "selp.u32 %0, 1, 0, p; }"
      : "=r"(ready)
      : "r"(addr), "r"(phase), "r"(kTicks)
      : "memory");
  return ready;
}

__device__ __forceinline__ unsigned long long mbarrier_wait_count(uint32_t addr,
                                                                  uint32_t phase) {
  unsigned long long polls = 0;
  while (!mbarrier_try_wait_once(addr, phase)) {
    ++polls;
  }
  return polls;
}

__device__ __forceinline__ void mma_commit(uint32_t barrier_addr) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
      :
      : "r"(barrier_addr)
      : "memory");
}

__device__ __forceinline__ void tma_load_3d(uint32_t dst,
                                            const CUtensorMap* map,
                                            int x,
                                            int y,
                                            int z,
                                            uint32_t barrier) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global."
      "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
      :
      : "r"(dst), "l"(map), "r"(x), "r"(y), "r"(z), "r"(barrier)
      : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t barrier,
                                                          uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
      :
      : "r"(barrier), "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc(uint32_t dst_smem, uint32_t columns) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :
               : "r"(dst_smem), "r"(columns)
               : "memory");
}

__device__ __forceinline__ void tmem_dealloc(uint32_t base, uint32_t columns) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
               :
               : "r"(base), "r"(columns)
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

__device__ __forceinline__ void tmem_store_x1(uint32_t addr, uint32_t value) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
               :
               : "r"(addr), "r"(value)
               : "memory");
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
  tmem_wait_ld();
}

__device__ __forceinline__ uint32_t make_idesc(int dtype, int n) {
  uint32_t d = 0;
  d |= 1u << 4;
  if (dtype == kBf16) {
    d |= 1u << 7;
    d |= 1u << 10;
  }
  d |= (static_cast<uint32_t>(n) >> 3U) << 17U;
  d |= 8u << 24U;
  return d;
}

#define MMA_CONFIG_ISSUE_AS(NAME, SUFFIX)                                      \
__device__ __forceinline__ void NAME(uint32_t d_tmem, uint64_t a_desc,         \
                                     uint64_t b_desc, uint32_t idesc,          \
                                     uint32_t use_d) {                         \
  asm volatile(                                                                \
      "{ .reg .pred p; setp.ne.b32 p, %4, 0; "                                 \
      "tcgen05.mma.cta_group::1.kind::f16" SUFFIX                              \
      " [%0], %1, %2, %3, {%5,%6,%7,%8}, p; }"                                 \
      :                                                                        \
      : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(use_d),         \
        "r"(0), "r"(0), "r"(0), "r"(0)                                        \
      : "memory");                                                            \
}

MMA_CONFIG_ISSUE_AS(issue_mma_as_none, "")
MMA_CONFIG_ISSUE_AS(issue_mma_as_discard, ".collector::a::discard")
MMA_CONFIG_ISSUE_AS(issue_mma_as_fill, ".collector::a::fill")
MMA_CONFIG_ISSUE_AS(issue_mma_as_use, ".collector::a::use")
MMA_CONFIG_ISSUE_AS(issue_mma_as_lastuse, ".collector::a::lastuse")

#define MMA_CONFIG_ISSUE_WS(NAME, SUFFIX)                                      \
__device__ __forceinline__ void NAME(uint32_t d_tmem, uint64_t a_desc,         \
                                     uint64_t b_desc, uint32_t idesc,          \
                                     uint32_t use_d) {                         \
  asm volatile(                                                                \
      "{ .reg .pred p; setp.ne.b32 p, %4, 0; "                                 \
      "tcgen05.mma.ws.cta_group::1.kind::f16" SUFFIX                           \
      " [%0], %1, %2, %3, p; }"                                                \
      :                                                                        \
      : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(use_d)          \
      : "memory");                                                            \
}

MMA_CONFIG_ISSUE_WS(issue_mma_ws_b0_discard, ".collector::b0::discard")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b0_fill, ".collector::b0::fill")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b0_use, ".collector::b0::use")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b0_lastuse, ".collector::b0::lastuse")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b1_discard, ".collector::b1::discard")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b1_fill, ".collector::b1::fill")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b1_use, ".collector::b1::use")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b1_lastuse, ".collector::b1::lastuse")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b2_discard, ".collector::b2::discard")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b2_fill, ".collector::b2::fill")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b2_use, ".collector::b2::use")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b2_lastuse, ".collector::b2::lastuse")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b3_discard, ".collector::b3::discard")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b3_fill, ".collector::b3::fill")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b3_use, ".collector::b3::use")
MMA_CONFIG_ISSUE_WS(issue_mma_ws_b3_lastuse, ".collector::b3::lastuse")

__device__ __forceinline__ int collector_op_for_mma(const Options& o, int q) {
  if (o.collector_protocol == kCollectorNone) return 0;
  if (o.collector_protocol == kCollectorDiscard) return 1;
  const int r = max(0, o.collector_reuse);
  const int period = r + 2;
  const int pos = q % period;
  if (pos == 0) return 2;
  if (pos == period - 1) {
    return o.collector_protocol == kCollectorFillUseDiscard ? 1 : 4;
  }
  return 3;
}

__device__ __forceinline__ void issue_mma_selected(const Options& o,
                                                   int q,
                                                   uint32_t d_tmem,
                                                   uint64_t a_desc,
                                                   uint64_t b_desc,
                                                   uint32_t idesc,
                                                   uint32_t use_d) {
  const int op = collector_op_for_mma(o, q);
  if (!o.ws_mode) {
    if (op == 1) issue_mma_as_discard(d_tmem, a_desc, b_desc, idesc, use_d);
    else if (op == 2) issue_mma_as_fill(d_tmem, a_desc, b_desc, idesc, use_d);
    else if (op == 3) issue_mma_as_use(d_tmem, a_desc, b_desc, idesc, use_d);
    else if (op == 4) issue_mma_as_lastuse(d_tmem, a_desc, b_desc, idesc, use_d);
    else issue_mma_as_none(d_tmem, a_desc, b_desc, idesc, use_d);
    return;
  }
  const int buffers = max(1, min(4, o.ws_buffer_count));
  const int b = (q / max(1, o.collector_reuse + 2)) % buffers;
  if (b == 0) {
    if (op == 2) issue_mma_ws_b0_fill(d_tmem, a_desc, b_desc, idesc, use_d);
    else if (op == 3) issue_mma_ws_b0_use(d_tmem, a_desc, b_desc, idesc, use_d);
    else if (op == 4) issue_mma_ws_b0_lastuse(d_tmem, a_desc, b_desc, idesc, use_d);
    else issue_mma_ws_b0_discard(d_tmem, a_desc, b_desc, idesc, use_d);
  } else if (b == 1) {
    if (op == 2) issue_mma_ws_b1_fill(d_tmem, a_desc, b_desc, idesc, use_d);
    else if (op == 3) issue_mma_ws_b1_use(d_tmem, a_desc, b_desc, idesc, use_d);
    else if (op == 4) issue_mma_ws_b1_lastuse(d_tmem, a_desc, b_desc, idesc, use_d);
    else issue_mma_ws_b1_discard(d_tmem, a_desc, b_desc, idesc, use_d);
  } else if (b == 2) {
    if (op == 2) issue_mma_ws_b2_fill(d_tmem, a_desc, b_desc, idesc, use_d);
    else if (op == 3) issue_mma_ws_b2_use(d_tmem, a_desc, b_desc, idesc, use_d);
    else if (op == 4) issue_mma_ws_b2_lastuse(d_tmem, a_desc, b_desc, idesc, use_d);
    else issue_mma_ws_b2_discard(d_tmem, a_desc, b_desc, idesc, use_d);
  } else {
    if (op == 2) issue_mma_ws_b3_fill(d_tmem, a_desc, b_desc, idesc, use_d);
    else if (op == 3) issue_mma_ws_b3_use(d_tmem, a_desc, b_desc, idesc, use_d);
    else if (op == 4) issue_mma_ws_b3_lastuse(d_tmem, a_desc, b_desc, idesc, use_d);
    else issue_mma_ws_b3_discard(d_tmem, a_desc, b_desc, idesc, use_d);
  }
}

__device__ __forceinline__ int operand_slot_for_q(const Options& o, int q) {
  if (o.operand_address_mode == kOperandPingpong) return q & 1;
  if (o.operand_address_mode == kOperandRotating) return q & 3;
  return 0;
}

__device__ __forceinline__ int d_base_for_q(const Options& o, int q) {
  if (o.d_tile_base_delta == 0 || o.d_reuse_distance <= 1) {
    return o.d_base_column;
  }
  const int capacity = max(1, (o.tmem_columns - o.d_base_column - o.n) /
                                  max(1, o.d_tile_base_delta) + 1);
  const int active = max(1, min(o.d_reuse_distance, capacity));
  const int idx = q % active;
  return o.d_base_column + idx * o.d_tile_base_delta;
}

__device__ __forceinline__ uint32_t float_bits(float x) {
  return __float_as_uint(x);
}

__host__ __device__ __forceinline__ float initial_d_value(int row, int col) {
  return 0.03125f * float(((row * 7 + col * 3) & 15) - 7);
}

__device__ __forceinline__ void initialize_tmem_columns(uint32_t tmem_base,
                                                        const Options& o) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (tid >= 128) return;
  for (int col = 0; col < o.tmem_columns; col += 8) {
    const uint32_t addr = tmem_base + ((warp * 32) << 16) + col;
    uint32_t bits[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int c = col + j;
      const float v = c < o.tmem_columns ? initial_d_value(warp * 32 + lane, c) : 0.0f;
      bits[j] = float_bits(v);
    }
    tmem_store_x8(addr, bits[0], bits[1], bits[2], bits[3],
                  bits[4], bits[5], bits[6], bits[7]);
  }
  tmem_wait_st();
}

__device__ __forceinline__ void initialize_guard_columns(uint32_t tmem_base,
                                                         const Options& o,
                                                         int* guard_col_a,
                                                         int* guard_col_b) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= 128) return;
  const int warp = tid >> 5;
  const uint32_t sentinel = 0x7f4a1234u;
  int g0 = -1;
  int g1 = -1;
  if (o.d_base_column > 0) g0 = o.d_base_column - 1;
  const int active = (o.d_tile_base_delta == 0 || o.d_reuse_distance <= 1)
                         ? 1
                         : max(1, min(o.d_reuse_distance,
                                      (o.tmem_columns - o.d_base_column - o.n) /
                                          max(1, o.d_tile_base_delta) + 1));
  const int max_base = o.d_base_column + (active - 1) * o.d_tile_base_delta;
  const int after = max_base + o.n;
  if (after < o.tmem_columns) g1 = after;
  *guard_col_a = g0;
  *guard_col_b = g1;
  if (g0 >= 0) {
    tmem_store_x1(tmem_base + ((warp * 32) << 16) + g0, sentinel);
  }
  if (g1 >= 0 && g1 != g0) {
    tmem_store_x1(tmem_base + ((warp * 32) << 16) + g1, sentinel);
  }
  tmem_wait_st();
}

__device__ __forceinline__ void check_guard_columns(uint32_t tmem_base,
                                                    int g0,
                                                    int g1,
                                                    int* guard_ok) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= 128) return;
  const int warp = tid >> 5;
  const uint32_t sentinel = 0x7f4a1234u;
  uint32_t value = 0;
  if (g0 >= 0) {
    tmem_load_x1(tmem_base + ((warp * 32) << 16) + g0, value);
    tmem_wait_ld();
    if (value != sentinel) atomicExch(guard_ok, 0);
  }
  if (g1 >= 0 && g1 != g0) {
    tmem_load_x1(tmem_base + ((warp * 32) << 16) + g1, value);
    tmem_wait_ld();
    if (value != sentinel) atomicExch(guard_ok, 0);
  }
}

__device__ __forceinline__ void run_interference(const Options& o,
                                                 const float* l1_data,
                                                 unsigned int* shared_words,
                                                 float* sink) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  if (o.interference_mode == kInterferenceNone) return;
  if (warp >= o.interference_warps) return;
  if (o.interference_mode != kInterferenceOnly && tid == 0) return;
  float acc = float(tid & 7);
  const int ops = max(0, o.interference_ops_per_iter);
  for (int i = 0; i < ops; ++i) {
    if (o.interference_mode == kInterferenceAlu) {
      acc = acc * 1.00024414f + 0.125f;
    } else if (o.interference_mode == kInterferencePredOffLoad) {
      unsigned int v = 0;
      const unsigned int addr = smem_u32(shared_words + ((tid + i) & 1023));
      asm volatile(
          "{ .reg .pred p; setp.ne.u32 p, %2, %2; @p ld.shared.u32 %0, [%1]; }"
          : "=r"(v)
          : "r"(addr), "r"(i)
          : "memory");
      acc += float(v & 1);
    } else if (o.interference_mode == kInterferenceL1HitGlobal) {
      acc += l1_data[(tid + i * 17) & 1023];
    } else if (o.interference_mode == kInterferenceLdShared ||
               o.interference_mode == kInterferenceOnly) {
      unsigned int v = shared_words[(tid * 17 + i * 31) & 1023];
      acc += float(v & 255) * 0.001f;
    }
  }
  if ((tid & 31) == 0) atomicAdd(sink, acc);
}

__global__ __launch_bounds__(128, 1)
void tcgen05_config_kernel(const __grid_constant__ CUtensorMap map_a,
                           const __grid_constant__ CUtensorMap map_b,
                           Options o,
                           float* output,
                           KernelResult* result,
                           const float* l1_data) {
  extern __shared__ __align__(1024) unsigned char dynamic_smem[];
  __shared__ alignas(16) uint64_t tma_barrier;
  __shared__ alignas(16) uint64_t mma_barriers[8];
  __shared__ alignas(16) uint32_t tmem_base;
  __shared__ int guard_ok;
  __shared__ int guard_col_a;
  __shared__ int guard_col_b;
  __shared__ float interference_sink;
  __shared__ unsigned int shared_noise[1024];

  const int tid = static_cast<int>(threadIdx.x);
  if (tid == 0) {
    guard_ok = 1;
    guard_col_a = -1;
    guard_col_b = -1;
    interference_sink = 0.0f;
    mbarrier_init(smem_u32(&tma_barrier), 1);
    for (int i = 0; i < max(1, min(8, o.pending_mbarriers)); ++i) {
      mbarrier_init(smem_u32(&mma_barriers[i]), 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < 1024; i += blockDim.x) {
    shared_noise[i] = static_cast<unsigned int>(i * 1103515245u + 12345u);
  }
  __syncthreads();

  const int slots = max(1, min(4, o.operand_slots));
  unsigned char* smem_a_base = dynamic_smem;
  unsigned char* smem_b_base = dynamic_smem + slots * o.a_slot_stride;

  uint32_t tma_phase = 0;
  for (int slot = 0; slot < slots; ++slot) {
    if (tid == 0) {
      const uint32_t a_dst = smem_u32(smem_a_base + slot * o.a_slot_stride + o.smem_base_offset);
      const uint32_t b_dst = smem_u32(smem_b_base + slot * o.b_slot_stride + o.smem_base_offset);
      const uint32_t bar = smem_u32(&tma_barrier);
      tma_load_3d(a_dst, &map_a, 0, 0, slot, bar);
      tma_load_3d(b_dst, &map_b, 0, 0, slot, bar);
      mbarrier_arrive_expect_tx(bar, static_cast<uint32_t>(o.a_bytes + o.b_bytes));
    }
    mbarrier_wait_nocount(smem_u32(&tma_barrier), tma_phase);
    tma_phase ^= 1;
  }
  tcgen05_fence_after_thread_sync();
  __syncthreads();

  if (tid < 32) {
    tmem_alloc(smem_u32(&tmem_base), static_cast<uint32_t>(o.tmem_columns));
  }
  __syncthreads();

  initialize_tmem_columns(tmem_base, o);
  __syncthreads();
  initialize_guard_columns(tmem_base, o, &guard_col_a, &guard_col_b);
  __syncthreads();

  const uint32_t idesc = make_idesc(o.dtype, o.n);
  const int commit_interval = max(1, o.commit_interval);
  const int pending = max(1, min(8, o.pending_mbarriers));
  uint32_t phases[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  unsigned long long local_polls = 0;

  __syncthreads();
  const unsigned long long start = clock64();

  for (int iter = 0; iter < o.iterations; ++iter) {
    int commits = 0;
    int waits = 0;
    const bool run_mma = o.interference_mode != kInterferenceOnly;
    for (int q = 0; q < o.q; ++q) {
      const int slot = operand_slot_for_q(o, q);
      const uint32_t a_smem = smem_u32(smem_a_base + slot * o.a_slot_stride + o.smem_base_offset);
      const uint32_t b_smem = smem_u32(smem_b_base + slot * o.b_slot_stride + o.smem_base_offset);
      const uint64_t a_desc = make_k_major_desc(a_smem, o.layout);
      const uint64_t b_desc = make_k_major_desc(b_smem, o.layout);
      const uint32_t d_tmem = tmem_base + d_base_for_q(o, q);
      if (tid == 0 && run_mma) {
        issue_mma_selected(o, q, d_tmem, a_desc, b_desc, idesc,
                           o.input_d ? 1u : 0u);
      }
      run_interference(o, l1_data, shared_noise, &interference_sink);
      const bool need_commit = ((q + 1) % commit_interval == 0) || (q + 1 == o.q);
      if (need_commit && run_mma) {
        const int idx = commits % pending;
        if (tid == 0) mma_commit(smem_u32(&mma_barriers[idx]));
        ++commits;
        if (commits - waits >= pending || q + 1 == o.q) {
          const int widx = waits % pending;
          if (tid == 0) {
            if (o.wait_mode == kWaitCount) {
              local_polls += mbarrier_wait_count(smem_u32(&mma_barriers[widx]), phases[widx]);
            } else {
              mbarrier_wait_nocount(smem_u32(&mma_barriers[widx]), phases[widx]);
            }
          }
          __syncthreads();
          phases[widx] ^= 1;
          ++waits;
        }
      }
    }
    while (waits < commits) {
      const int widx = waits % pending;
      if (tid == 0) {
        if (o.wait_mode == kWaitCount) {
          local_polls += mbarrier_wait_count(smem_u32(&mma_barriers[widx]), phases[widx]);
        } else {
          mbarrier_wait_nocount(smem_u32(&mma_barriers[widx]), phases[widx]);
        }
      }
      __syncthreads();
      phases[widx] ^= 1;
      ++waits;
    }
  }

  const unsigned long long stop = clock64();
  if (tid == 0) {
    result[blockIdx.x].cycles = stop - start;
    result[blockIdx.x].poll_count = local_polls;
#if defined(__CUDA_ARCH__)
    result[blockIdx.x].cuda_arch = __CUDA_ARCH__;
#else
    result[blockIdx.x].cuda_arch = 0;
#endif
    result[blockIdx.x].interference_sink = interference_sink;
  }
  __syncthreads();

  check_guard_columns(tmem_base, guard_col_a, guard_col_b, &guard_ok);
  __syncthreads();

  for (int col = 0; col < o.tmem_columns; col += 8) {
    if (tid < 128) {
      const int warp = tid >> 5;
      const int lane = tid & 31;
      float values[8];
      const uint32_t addr = tmem_base + ((warp * 32) << 16) + col;
      tmem_load_x8(addr, values);
      const int row = warp * 32 + lane;
      for (int j = 0; j < 8; ++j) {
        if (col + j < o.tmem_columns) {
          output[((blockIdx.x * 128 + row) * o.tmem_columns) + col + j] = values[j];
        }
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    result[blockIdx.x].guard_ok = guard_ok;
  }
  __syncthreads();

  if (tid < 32) {
    tmem_dealloc(tmem_base, static_cast<uint32_t>(o.tmem_columns));
    tmem_relinquish();
  }
}

inline int align_up(int value, int align) {
  return ((value + align - 1) / align) * align;
}

inline int parse_int_arg(int argc, char** argv, const char* name, int def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
  }
  return def;
}

inline std::string parse_str_arg(int argc, char** argv, const char* name,
                                 const char* def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
  }
  return def;
}

inline bool has_arg(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return true;
  }
  return false;
}

inline int dtype_id(const std::string& dtype) {
  if (dtype == "fp16") return kFp16;
  if (dtype == "bf16") return kBf16;
  std::cerr << "unsupported dtype: " << dtype << "\n";
  std::exit(2);
}

inline int shape_n(const std::string& shape) {
  if (shape == "m128n64k16") return 64;
  if (shape == "m128n128k16") return 128;
  if (shape == "m128n256k16") return 256;
  std::cerr << "unsupported shape: " << shape << "\n";
  std::exit(2);
}

inline int layout_id(const std::string& layout) {
  if (layout == "none") return kLayoutNone;
  if (layout == "sw32") return kLayoutSw32;
  if (layout == "sw64") return kLayoutSw64;
  if (layout == "sw128") return kLayoutSw128;
  std::cerr << "unsupported layout: " << layout << "\n";
  std::exit(2);
}

inline CUtensorMapSwizzle map_swizzle(int layout) {
  if (layout == kLayoutSw32) return CU_TENSOR_MAP_SWIZZLE_32B;
  if (layout == kLayoutSw64) return CU_TENSOR_MAP_SWIZZLE_64B;
  if (layout == kLayoutSw128) return CU_TENSOR_MAP_SWIZZLE_128B;
  return CU_TENSOR_MAP_SWIZZLE_NONE;
}

inline int layout_span_bytes(int layout, int logical_row_bytes) {
  if (layout == kLayoutSw32) return 32;
  if (layout == kLayoutSw64) return 64;
  if (layout == kLayoutSw128) return 128;
  return logical_row_bytes;
}

inline int collector_protocol_id(const std::string& s) {
  if (s == "none") return kCollectorNone;
  if (s == "discard") return kCollectorDiscard;
  if (s == "fill_lastuse") return kCollectorFillLastuse;
  if (s == "fill_use_lastuse") return kCollectorFillUseLastuse;
  if (s == "fill_use_discard") return kCollectorFillUseDiscard;
  std::cerr << "unsupported collector protocol: " << s << "\n";
  std::exit(2);
}

inline int operand_mode_id(const std::string& s) {
  if (s == "same") return kOperandSame;
  if (s == "pingpong") return kOperandPingpong;
  if (s == "rotating") return kOperandRotating;
  std::cerr << "unsupported operand address mode: " << s << "\n";
  std::exit(2);
}

inline int interference_mode_id(const std::string& s) {
  if (s == "none") return kInterferenceNone;
  if (s == "register_alu") return kInterferenceAlu;
  if (s == "predicated_off_load") return kInterferencePredOffLoad;
  if (s == "l1_hit_global") return kInterferenceL1HitGlobal;
  if (s == "ld_shared") return kInterferenceLdShared;
  if (s == "interference_only") return kInterferenceOnly;
  std::cerr << "unsupported interference mode: " << s << "\n";
  std::exit(2);
}

inline int wait_mode_id(const std::string& s) {
  if (s == "nocount") return kWaitNoCount;
  if (s == "count") return kWaitCount;
  std::cerr << "unsupported wait mode: " << s << "\n";
  std::exit(2);
}

inline void encode_tma_3d(CUtensorMap* map,
                          void* base,
                          int dtype,
                          int rows,
                          int k,
                          int slots,
                          int layout) {
  const int element_bytes = 2;
  const uint32_t rank = 3;
  uint64_t global_dim[rank] = {
      static_cast<uint64_t>(k),
      static_cast<uint64_t>(rows),
      static_cast<uint64_t>(slots)};
  uint64_t global_stride[rank - 1] = {
      static_cast<uint64_t>(k * element_bytes),
      static_cast<uint64_t>(rows * k * element_bytes)};
  uint32_t box_dim[rank] = {
      static_cast<uint32_t>(k),
      static_cast<uint32_t>(rows),
      1u};
  uint32_t element_stride[rank] = {1u, 1u, 1u};
  CUtensorMapDataType tma_dtype =
      dtype == kBf16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  MMA_CONFIG_CU_CHECK(cuTensorMapEncodeTiled(
      map, tma_dtype, rank, base, global_dim, global_stride, box_dim,
      element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, map_swizzle(layout),
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

inline float make_a_value(int slot, int row, int k) {
  return 0.0625f * float(((slot * 11 + row * 5 + k * 3) % 17) - 8);
}

inline float make_b_value(int slot, int col, int k) {
  return 0.03125f * float(((slot * 7 + col * 13 + k * 9) % 19) - 9);
}

inline void fill_l1_data(std::vector<float>& values) {
  values.resize(1024);
  for (int i = 0; i < 1024; ++i) values[i] = float((i * 17) & 255) * 0.001f;
}

template <typename T>
inline float to_float_host(T v);

template <>
inline float to_float_host<half>(half v) {
  return __half2float(v);
}

template <>
inline float to_float_host<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
inline void fill_operand_typed(int slots, int rows, int k, bool is_a,
                               std::vector<T>& storage,
                               std::vector<float>& quantized) {
  storage.resize(static_cast<size_t>(slots) * rows * k);
  quantized.resize(storage.size());
  for (int s = 0; s < slots; ++s) {
    for (int r = 0; r < rows; ++r) {
      for (int kk = 0; kk < k; ++kk) {
        const float v = is_a ? make_a_value(s, r, kk) : make_b_value(s, r, kk);
        T q;
        if constexpr (std::is_same<T, half>::value) {
          q = __float2half(v);
        } else {
          q = __float2bfloat16(v);
        }
        const size_t idx = (static_cast<size_t>(s) * rows + r) * k + kk;
        storage[idx] = q;
        quantized[idx] = to_float_host<T>(q);
      }
    }
  }
}

inline void product_for_slot(const std::vector<float>& a,
                             const std::vector<float>& b,
                             int slot,
                             int n,
                             std::vector<float>& product) {
  product.assign(static_cast<size_t>(128) * n, 0.0f);
  for (int row = 0; row < 128; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int kk = 0; kk < 16; ++kk) {
        const float av = a[(static_cast<size_t>(slot) * 128 + row) * 16 + kk];
        const float bv = b[(static_cast<size_t>(slot) * n + col) * 16 + kk];
        sum += av * bv;
      }
      product[static_cast<size_t>(row) * n + col] = sum;
    }
  }
}

inline int operand_slot_host(const Options& o, int q) {
  if (o.operand_address_mode == kOperandPingpong) return q & 1;
  if (o.operand_address_mode == kOperandRotating) return q & 3;
  return 0;
}

inline int d_base_host(const Options& o, int q) {
  if (o.d_tile_base_delta == 0 || o.d_reuse_distance <= 1) {
    return o.d_base_column;
  }
  const int capacity = std::max(1, (o.tmem_columns - o.d_base_column - o.n) /
                                      std::max(1, o.d_tile_base_delta) + 1);
  const int active = std::max(1, std::min(o.d_reuse_distance, capacity));
  return o.d_base_column + (q % active) * o.d_tile_base_delta;
}

inline void build_reference(const Options& o,
                            const std::vector<float>& a,
                            const std::vector<float>& b,
                            std::vector<float>& ref) {
  ref.resize(static_cast<size_t>(128) * o.tmem_columns);
  for (int row = 0; row < 128; ++row) {
    for (int col = 0; col < o.tmem_columns; ++col) {
      ref[static_cast<size_t>(row) * o.tmem_columns + col] =
          initial_d_value(row, col);
    }
  }
  std::vector<std::vector<float>> products(o.operand_slots);
  for (int s = 0; s < o.operand_slots; ++s) {
    product_for_slot(a, b, s, o.n, products[s]);
  }
  for (int iter = 0; iter < o.iterations; ++iter) {
    for (int q = 0; q < o.q; ++q) {
      if (o.interference_mode == kInterferenceOnly) continue;
      const int slot = operand_slot_host(o, q);
      const int base = d_base_host(o, q);
      if (base < 0 || base + o.n > o.tmem_columns) continue;
      for (int row = 0; row < 128; ++row) {
        for (int col = 0; col < o.n; ++col) {
          const size_t dst = static_cast<size_t>(row) * o.tmem_columns + base + col;
          const float p = products[slot][static_cast<size_t>(row) * o.n + col];
          if (o.input_d) ref[dst] += p;
          else ref[dst] = p;
        }
      }
    }
  }
}

inline double max_abs_diff_touched(const std::vector<float>& got,
                                   const std::vector<float>& ref,
                                   const Options& o,
                                   int blocks) {
  std::vector<unsigned char> touched(o.tmem_columns, 0);
  for (int q = 0; q < o.q; ++q) {
    const int base = d_base_host(o, q);
    if (base < 0 || base + o.n > o.tmem_columns) continue;
    for (int col = 0; col < o.n; ++col) touched[base + col] = 1;
  }
  double max_abs = 0.0;
  for (int b = 0; b < blocks; ++b) {
    const size_t base = static_cast<size_t>(b) * 128 * o.tmem_columns;
    for (int row = 0; row < 128; ++row) {
      for (int col = 0; col < o.tmem_columns; ++col) {
        if (!touched[col]) continue;
        const size_t idx = static_cast<size_t>(row) * o.tmem_columns + col;
        const double diff = std::abs(double(got[base + idx]) - double(ref[idx]));
        max_abs = std::max(max_abs, diff);
      }
    }
  }
  return max_abs;
}

inline void print_kv(const char* key, const std::string& value) {
  std::cout << key << '=' << value << '\n';
}

inline void print_kv(const char* key, long long value) {
  std::cout << key << '=' << value << '\n';
}

inline void print_kv(const char* key, int value) {
  std::cout << key << '=' << value << '\n';
}

inline void print_kv(const char* key, double value) {
  std::cout << key << '=' << std::fixed << std::setprecision(9) << value << '\n';
}

inline int run_main(int argc, char** argv) {
  const std::string dtype_s = parse_str_arg(argc, argv, "--dtype", "bf16");
  const std::string shape_s = parse_str_arg(argc, argv, "--shape", "m128n128k16");
  const std::string layout_s = parse_str_arg(argc, argv, "--layout", "sw128");
  const std::string collector_s = parse_str_arg(argc, argv, "--collector-protocol", "discard");
  const std::string operand_s = parse_str_arg(argc, argv, "--operand-address-mode", "same");
  const std::string interference_s = parse_str_arg(argc, argv, "--interference-mode", "none");
  const std::string wait_s = parse_str_arg(argc, argv, "--wait-mode", "nocount");
  Options o{};
  o.dtype = dtype_id(dtype_s);
  o.n = shape_n(shape_s);
  o.layout = layout_id(layout_s);
  o.q = parse_int_arg(argc, argv, "--q", 1);
  o.iterations = parse_int_arg(argc, argv, "--iterations", 1);
  o.collector_protocol = collector_protocol_id(collector_s);
  o.collector_reuse = parse_int_arg(argc, argv, "--collector-reuse", 0);
  o.ws_mode = parse_int_arg(argc, argv, "--ws", 0);
  o.ws_buffer_count = parse_int_arg(argc, argv, "--ws-buffer-count", 1);
  o.operand_address_mode = operand_mode_id(operand_s);
  o.input_d = parse_int_arg(argc, argv, "--input-d", 0);
  o.tmem_columns = parse_int_arg(argc, argv, "--tmem-columns", 512);
  o.d_base_column = parse_int_arg(argc, argv, "--d-base-column", 0);
  o.d_tile_base_delta = parse_int_arg(argc, argv, "--d-tile-base-delta", o.n);
  o.independent_d_count = parse_int_arg(argc, argv, "--independent-d-count",
                                        std::max(1, o.tmem_columns / std::max(1, o.n)));
  o.d_reuse_distance = parse_int_arg(argc, argv, "--d-reuse-distance", o.independent_d_count);
  o.commit_interval = parse_int_arg(argc, argv, "--commit-interval", std::max(1, o.q));
  o.pending_mbarriers = parse_int_arg(argc, argv, "--pending-mbarriers", 1);
  o.wait_mode = wait_mode_id(wait_s);
  o.smem_base_offset = parse_int_arg(argc, argv, "--smem-base-offset", 0);
  o.interference_mode = interference_mode_id(interference_s);
  o.interference_ops_per_iter = parse_int_arg(argc, argv, "--interference-ops-per-iter", 0);
  o.interference_warps = parse_int_arg(argc, argv, "--interference-warps", 4);
  o.active_blocks = parse_int_arg(argc, argv, "--active-blocks", 0);
  o.operand_slots = o.operand_address_mode == kOperandRotating ? 4 :
                    (o.operand_address_mode == kOperandPingpong ? 2 : 1);
  o.operand_slots = std::max(1, std::min(4, parse_int_arg(argc, argv, "--operand-slots", o.operand_slots)));
  o.a_bytes = 128 * 16 * 2;
  o.b_bytes = o.n * 16 * 2;
  const int a_footprint = 128 * layout_span_bytes(o.layout, 16 * 2);
  const int b_footprint = o.n * layout_span_bytes(o.layout, 16 * 2);
  o.a_slot_stride = align_up(a_footprint + o.smem_base_offset + 256, 1024);
  o.b_slot_stride = align_up(b_footprint + o.smem_base_offset + 256, 1024);
  o.dynamic_smem_bytes = o.operand_slots * (o.a_slot_stride + o.b_slot_stride);

  if (o.q <= 0 || o.iterations <= 0 || o.tmem_columns <= 0) {
    std::cerr << "q, iterations and tmem columns must be positive\n";
    return 2;
  }
  if (o.d_base_column < 0 || o.d_base_column + o.n > o.tmem_columns) {
    print_kv("status", "invalid_static");
    print_kv("invalid_reason", "d_tile_exceeds_tmem_columns");
    return 0;
  }
  if (o.d_tile_base_delta < 0) {
    print_kv("status", "invalid_static");
    print_kv("invalid_reason", "negative_d_tile_base_delta");
    return 0;
  }
  if (o.d_tile_base_delta == 0) {
    o.independent_d_count = 1;
    o.d_reuse_distance = 1;
  } else {
    const int address_capacity =
        std::max(1, (o.tmem_columns - o.d_base_column - o.n) /
                        std::max(1, o.d_tile_base_delta) + 1);
    o.d_reuse_distance = std::max(1, std::min(o.d_reuse_distance, address_capacity));
    const int independent_capacity =
        o.d_tile_base_delta >= o.n
            ? std::max(1, (o.tmem_columns - o.d_base_column - o.n) / o.n + 1)
            : 1;
    o.independent_d_count =
        std::max(1, std::min(o.independent_d_count, independent_capacity));
  }

  MMA_CONFIG_CUDA_CHECK(cudaFree(nullptr));
  int dev = 0;
  MMA_CONFIG_CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  MMA_CONFIG_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int blocks = o.active_blocks > 0 ? std::min(o.active_blocks, prop.multiProcessorCount)
                                   : prop.multiProcessorCount;
  if (blocks <= 0) blocks = 1;

  if (prop.major != 11) {
    print_kv("status", "skipped_non_sm110");
    print_kv("gpu", prop.name);
    print_kv("compute_capability", std::to_string(prop.major) + "." + std::to_string(prop.minor));
    return 0;
  }

  std::vector<float> a_quant;
  std::vector<float> b_quant;
  void* d_a = nullptr;
  void* d_b = nullptr;
  CUtensorMap map_a{};
  CUtensorMap map_b{};
  if (o.dtype == kFp16) {
    std::vector<half> a_host, b_host;
    fill_operand_typed(o.operand_slots, 128, 16, true, a_host, a_quant);
    fill_operand_typed(o.operand_slots, o.n, 16, false, b_host, b_quant);
    MMA_CONFIG_CUDA_CHECK(cudaMalloc(&d_a, a_host.size() * sizeof(half)));
    MMA_CONFIG_CUDA_CHECK(cudaMalloc(&d_b, b_host.size() * sizeof(half)));
    MMA_CONFIG_CUDA_CHECK(cudaMemcpy(d_a, a_host.data(), a_host.size() * sizeof(half), cudaMemcpyHostToDevice));
    MMA_CONFIG_CUDA_CHECK(cudaMemcpy(d_b, b_host.data(), b_host.size() * sizeof(half), cudaMemcpyHostToDevice));
  } else {
    std::vector<__nv_bfloat16> a_host, b_host;
    fill_operand_typed(o.operand_slots, 128, 16, true, a_host, a_quant);
    fill_operand_typed(o.operand_slots, o.n, 16, false, b_host, b_quant);
    MMA_CONFIG_CUDA_CHECK(cudaMalloc(&d_a, a_host.size() * sizeof(__nv_bfloat16)));
    MMA_CONFIG_CUDA_CHECK(cudaMalloc(&d_b, b_host.size() * sizeof(__nv_bfloat16)));
    MMA_CONFIG_CUDA_CHECK(cudaMemcpy(d_a, a_host.data(), a_host.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    MMA_CONFIG_CUDA_CHECK(cudaMemcpy(d_b, b_host.data(), b_host.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  }
  MMA_CONFIG_CU_CHECK(cuInit(0));
  encode_tma_3d(&map_a, d_a, o.dtype, 128, 16, o.operand_slots, o.layout);
  encode_tma_3d(&map_b, d_b, o.dtype, o.n, 16, o.operand_slots, o.layout);

  std::vector<float> l1_host;
  fill_l1_data(l1_host);
  float* d_l1 = nullptr;
  MMA_CONFIG_CUDA_CHECK(cudaMalloc(&d_l1, l1_host.size() * sizeof(float)));
  MMA_CONFIG_CUDA_CHECK(cudaMemcpy(d_l1, l1_host.data(), l1_host.size() * sizeof(float), cudaMemcpyHostToDevice));

  float* d_output = nullptr;
  KernelResult* d_result = nullptr;
  const size_t output_count = static_cast<size_t>(blocks) * 128 * o.tmem_columns;
  MMA_CONFIG_CUDA_CHECK(cudaMalloc(&d_output, output_count * sizeof(float)));
  MMA_CONFIG_CUDA_CHECK(cudaMalloc(&d_result, blocks * sizeof(KernelResult)));
  MMA_CONFIG_CUDA_CHECK(cudaMemset(d_output, 0, output_count * sizeof(float)));
  MMA_CONFIG_CUDA_CHECK(cudaMemset(d_result, 0, blocks * sizeof(KernelResult)));

  auto kernel = &tcgen05_config_kernel;
  if (o.dynamic_smem_bytes > 48 * 1024) {
    MMA_CONFIG_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        o.dynamic_smem_bytes));
  }
  tcgen05_config_kernel<<<blocks, 128, o.dynamic_smem_bytes>>>(map_a, map_b, o, d_output, d_result, d_l1);
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    print_kv("status", "cuda_error");
    print_kv("invalid_reason", cudaGetErrorString(launch_err));
    return 0;
  }
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    print_kv("status", "cuda_error");
    print_kv("invalid_reason", cudaGetErrorString(sync_err));
    return 0;
  }

  std::vector<KernelResult> results(blocks);
  std::vector<float> output(output_count);
  MMA_CONFIG_CUDA_CHECK(cudaMemcpy(results.data(), d_result, blocks * sizeof(KernelResult), cudaMemcpyDeviceToHost));
  MMA_CONFIG_CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> ref;
  build_reference(o, a_quant, b_quant, ref);
  const double max_abs = max_abs_diff_touched(output, ref, o, blocks);
  unsigned long long max_cycles = 0;
  unsigned long long polls = 0;
  int guard_ok = 1;
  double sink = 0.0;
  for (const auto& r : results) {
    max_cycles = std::max(max_cycles, r.cycles);
    polls += r.poll_count;
    guard_ok &= r.guard_ok;
    sink += r.interference_sink;
  }

  print_kv("status", "ok");
  print_kv("experiment", MMA_CONFIG_EXPERIMENT);
  print_kv("gpu", prop.name);
  print_kv("compute_capability", std::to_string(prop.major) + "." + std::to_string(prop.minor));
  print_kv("sm_count", prop.multiProcessorCount);
  print_kv("active_blocks", blocks);
  print_kv("dtype", dtype_s);
  print_kv("shape", shape_s);
  print_kv("m", 128);
  print_kv("n", o.n);
  print_kv("k", 16);
  print_kv("layout", layout_s);
  print_kv("q", o.q);
  print_kv("iterations", o.iterations);
  print_kv("collector_protocol", collector_s);
  print_kv("collector_reuse", o.collector_reuse);
  print_kv("ws_mode", o.ws_mode);
  print_kv("ws_buffer_count", o.ws_buffer_count);
  print_kv("operand_address_mode", operand_s);
  print_kv("input_d", o.input_d);
  print_kv("tmem_columns", o.tmem_columns);
  print_kv("d_base_column", o.d_base_column);
  print_kv("d_tile_base_delta", o.d_tile_base_delta);
  print_kv("independent_d_count", o.independent_d_count);
  print_kv("d_reuse_distance", o.d_reuse_distance);
  print_kv("commit_interval", o.commit_interval);
  print_kv("pending_mbarriers", o.pending_mbarriers);
  print_kv("wait_polling_mode", wait_s);
  print_kv("smem_base_offset", o.smem_base_offset);
  print_kv("interference_mode", interference_s);
  print_kv("interference_ops_per_iter", o.interference_ops_per_iter);
  print_kv("interference_warps", o.interference_warps);
  print_kv("elapsed_cycles", static_cast<long long>(max_cycles));
  print_kv("poll_count", static_cast<long long>(polls));
  print_kv("max_abs_error", max_abs);
  print_kv("guard_ok", guard_ok);
  print_kv("interference_sink", sink);

  MMA_CONFIG_CUDA_CHECK(cudaFree(d_a));
  MMA_CONFIG_CUDA_CHECK(cudaFree(d_b));
  MMA_CONFIG_CUDA_CHECK(cudaFree(d_l1));
  MMA_CONFIG_CUDA_CHECK(cudaFree(d_output));
  MMA_CONFIG_CUDA_CHECK(cudaFree(d_result));
  return 0;
}

}  // namespace mma_config

int main(int argc, char** argv) {
  return mma_config::run_main(argc, argv);
}
