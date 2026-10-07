#pragma once
#include <cstdint>
#ifndef V05_PAIR
#error "V05_PAIR: 0 main, 1 output, 2 supply, 3 critical"
#endif
// Per CTA: 4 control words, then 3 roles x 32 tile records x 24 words.
// Critical uses only record 0: first producer work and final output source release.
constexpr int V05Tiles = 32, V05Words = 24, V05CtaWords = 4 + 3 * V05Tiles * V05Words;
__constant__ uint64_t* v05_trace_ptr;
enum V05Event {
  V05_WORK, V05_FIRST_MMA, V05_MAIN_END, V05_EPI_PERMIT,
  V05_STORE_RETURN, V05_SOURCE_RELEASE, V05_NEXT_START, V05_LOAD_RETURN
};
constexpr unsigned V05EventMask = V05_PAIR == 0 ? 0x06 : V05_PAIR == 1 ? 0x2c
                                             : V05_PAIR == 2 ? 0x03 : 0x21;
__device__ __forceinline__ uint64_t v05_cycle_stamp() {
  uint64_t cycle;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycle) :: "memory");
  return cycle;
}
__device__ __forceinline__ bool v05_selected_cta() {
#if V05_PAIR == 3
  return true;
#else
  return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
#endif
}
__device__ __forceinline__ uint64_t* v05_base() {
  unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  return v05_trace_ptr + block * V05CtaWords;
}
__device__ __forceinline__ void v05_write(int role, int tile, int event,
                                        uint64_t cycles, uint64_t ns) {
  if (tile < 0 || tile >= V05Tiles) { v05_base()[3] = 1; return; }
  uint64_t* dst = v05_base() + 4 + (role * V05Tiles + tile) * V05Words;
  dst[4 + event * 2] = cycles;
#if V05_PAIR == 3
  dst[5 + event * 2] = ns;
#else
  // Fine-stage ns slots stay zero from the matched pre-call memset: unobserved.
  (void)ns;
#endif
}
__device__ __forceinline__ void v05_critical_finish() {
#if V05_PAIR == 3
  uint64_t cycles = v05_cycle_stamp(), ns;
  unsigned sm;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns) :: "memory");
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  v05_write(0, 0, V05_SOURCE_RELEASE, cycles, ns);
  v05_base()[1] = 1;
  v05_base()[2] = uint64_t(sm) + 1;
#endif
}
__device__ __forceinline__ void v05_stamp(int event, int tile, bool last_cta_tile = false) {
#if V05_PAIR == 3
  // Pingpong's store_tail is per output tile. Only the last CTA tile qualifies;
  // the ordered epilogue barrier makes it later than the other consumer's source.
#if V05_CFG == 1
  if (event == V05_SOURCE_RELEASE && last_cta_tile &&
      threadIdx.x >= 128 && threadIdx.x % 128 == 0) v05_critical_finish();
#endif
#else
  if (!((V05EventMask >> event) & 1)) return;
  if (!v05_selected_cta() || threadIdx.x % 128 != 0) return;
  int role = threadIdx.x / 128;
  uint64_t cycles = v05_cycle_stamp();
  v05_write(role, tile, event, cycles, 0);
#endif
}
template<class Work>
__device__ __forceinline__ void v05_begin(Work const& work, int tile) {
#if V05_PAIR == 3
  if (threadIdx.x != 0 || tile != 0) return;
#else
  if (!v05_selected_cta() || threadIdx.x % 128 != 0) return;
#endif
  int role = threadIdx.x / 128;
#if V05_PAIR >= 2
  uint64_t cycles = v05_cycle_stamp(), ns = 0;
#if V05_PAIR == 3
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns) :: "memory");
#endif
#endif
  if (tile >= V05Tiles) { v05_base()[3] = 1; return; }
  uint64_t* dst = v05_base() + 4 + (role * V05Tiles + tile) * V05Words;
  unsigned sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  dst[0] = work.M_idx; dst[1] = work.N_idx; dst[2] = work.L_idx; dst[3] = sm;
#if V05_PAIR >= 2
  v05_write(role, tile, V05_WORK, cycles, ns);
#endif
  v05_base()[role] = tile + 1;
}
// Cooperative's epilogue thread_idx is physical threadIdx.x % 256. Its issuing
// warp is physical 256..287 (role 2). No additional waits or barrier changes.
__device__ __forceinline__ void v05_source_acquire(int count, int pending, int chunks) {
#if V05_CFG != 1 && V05_PAIR == 1
  if (!v05_selected_cta() || threadIdx.x != 256) return;
  int completed = count - pending;
  if (completed > 0 && completed % chunks == 0)
    v05_stamp(V05_SOURCE_RELEASE, completed / chunks - 1);
#endif
}
__device__ __forceinline__ void v05_source_tail() {
#if V05_CFG != 1 && V05_PAIR == 3
  // Cooperative calls this original tail once after its complete tile loop.
  if (threadIdx.x == 256) v05_critical_finish();
#elif V05_CFG != 1 && V05_PAIR == 1
  if (v05_selected_cta() && threadIdx.x == 256) {
    int tile = int(v05_base()[2]) - 1;
    if (tile >= 0) {
      uint64_t* record = v05_base() + 4 + (2 * V05Tiles + tile) * V05Words;
      if (record[4 + 2 * V05_SOURCE_RELEASE] == 0) v05_stamp(V05_SOURCE_RELEASE, tile);
    }
  }
#endif
}
