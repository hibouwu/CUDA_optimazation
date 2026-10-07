#pragma once
#include <cstdint>
#ifndef R14_PAIR
#error "R14_PAIR: 0 main, 1 output, 2 supply, 3 critical"
#endif
// Per CTA: 4 control words, then 3 roles x 8 tile records x 24 words.
// Critical uses only record 0: first producer work and final output source release.
constexpr int R14Tiles = 8, R14Words = 24, R14CtaWords = 4 + 3 * R14Tiles * R14Words;
__constant__ uint64_t* r14_trace_ptr;
enum R14Event {
  R14_WORK, R14_FIRST_MMA, R14_MAIN_END, R14_EPI_PERMIT,
  R14_STORE_RETURN, R14_SOURCE_RELEASE, R14_NEXT_START, R14_LOAD_RETURN
};
constexpr unsigned R14EventMask = R14_PAIR == 0 ? 0x06 : R14_PAIR == 1 ? 0x2c
                                             : R14_PAIR == 2 ? 0x03 : 0x21;
__device__ __forceinline__ uint64_t r14_cycle_stamp() {
  uint64_t cycle;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycle) :: "memory");
  return cycle;
}
__device__ __forceinline__ bool r14_selected_cta() {
#if R14_PAIR == 3
  return true;
#else
  return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
#endif
}
__device__ __forceinline__ uint64_t* r14_base() {
  unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  return r14_trace_ptr + block * R14CtaWords;
}
__device__ __forceinline__ void r14_write(int role, int tile, int event,
                                        uint64_t cycles, uint64_t ns) {
  if (tile < 0 || tile >= R14Tiles) { r14_base()[3] = 1; return; }
  uint64_t* dst = r14_base() + 4 + (role * R14Tiles + tile) * R14Words;
  dst[4 + event * 2] = cycles;
#if R14_PAIR == 3
  dst[5 + event * 2] = ns;
#else
  // Fine-stage ns slots stay zero from the matched pre-call memset: unobserved.
  (void)ns;
#endif
}
__device__ __forceinline__ void r14_critical_finish() {
#if R14_PAIR == 3
  uint64_t cycles = r14_cycle_stamp(), ns;
  unsigned sm;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns) :: "memory");
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  r14_write(0, 0, R14_SOURCE_RELEASE, cycles, ns);
  r14_base()[1] = 1;
  r14_base()[2] = uint64_t(sm) + 1;
#endif
}
__device__ __forceinline__ void r14_stamp(int event, int tile, bool last_cta_tile = false) {
#if R14_PAIR == 3
  // Pingpong's store_tail is per output tile. Only the last CTA tile qualifies;
  // the ordered epilogue barrier makes it later than the other consumer's source.
#if R14_CFG == 1
  if (event == R14_SOURCE_RELEASE && last_cta_tile &&
      threadIdx.x >= 128 && threadIdx.x % 128 == 0) r14_critical_finish();
#endif
#else
  if (!((R14EventMask >> event) & 1)) return;
  if (!r14_selected_cta() || threadIdx.x % 128 != 0) return;
  int role = threadIdx.x / 128;
  uint64_t cycles = r14_cycle_stamp();
  r14_write(role, tile, event, cycles, 0);
#endif
}
template<class Work>
__device__ __forceinline__ void r14_begin(Work const& work, int tile) {
#if R14_PAIR == 3
  if (threadIdx.x != 0 || tile != 0) return;
#else
  if (!r14_selected_cta() || threadIdx.x % 128 != 0) return;
#endif
  int role = threadIdx.x / 128;
#if R14_PAIR >= 2
  uint64_t cycles = r14_cycle_stamp(), ns = 0;
#if R14_PAIR == 3
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns) :: "memory");
#endif
#endif
  if (tile >= R14Tiles) { r14_base()[3] = 1; return; }
  uint64_t* dst = r14_base() + 4 + (role * R14Tiles + tile) * R14Words;
  unsigned sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  dst[0] = work.M_idx; dst[1] = work.N_idx; dst[2] = work.L_idx; dst[3] = sm;
#if R14_PAIR >= 2
  r14_write(role, tile, R14_WORK, cycles, ns);
#endif
  r14_base()[role] = tile + 1;
}
// Cooperative's epilogue thread_idx is physical threadIdx.x % 256. Its issuing
// warp is physical 256..287 (role 2). No additional waits or barrier changes.
__device__ __forceinline__ void r14_source_acquire(int count, int pending, int chunks) {
#if R14_CFG != 1 && R14_PAIR == 1
  if (!r14_selected_cta() || threadIdx.x != 256) return;
  int completed = count - pending;
  if (completed > 0 && completed % chunks == 0)
    r14_stamp(R14_SOURCE_RELEASE, completed / chunks - 1);
#endif
}
__device__ __forceinline__ void r14_source_tail() {
#if R14_CFG != 1 && R14_PAIR == 3
  // Cooperative calls this original tail once after its complete tile loop.
  if (threadIdx.x == 256) r14_critical_finish();
#elif R14_CFG != 1 && R14_PAIR == 1
  if (r14_selected_cta() && threadIdx.x == 256) {
    int tile = int(r14_base()[2]) - 1;
    if (tile >= 0) {
      uint64_t* record = r14_base() + 4 + (2 * R14Tiles + tile) * R14Words;
      if (record[4 + 2 * R14_SOURCE_RELEASE] == 0) r14_stamp(R14_SOURCE_RELEASE, tile);
    }
  }
#endif
}
