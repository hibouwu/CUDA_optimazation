#pragma once

// The E trace uses the frozen V06 hook sites and four additional call boundaries.
// Only the first thread of each consumer WG writes its own record. These are
// leader observations, not the last of the 128 ordered-barrier arrivals.
#ifndef CFG_B_PINGPONG_TRACE
#define CFG_B_PINGPONG_TRACE 0
#endif

#include <cstdint>

struct EClockPair { uint64_t cycle, ns; };
enum {
  V06_FIRST_MMA = 0, V06_MAIN_END = 1, V06_EPI_PERMIT = 2, V06_EPI_DONE = 3,
  E_MMA_RETURN = 4, E_MATH_ARRIVE_AFTER = 5, E_STORE_RETURN = 6,
  E_EPILOGUE_ARRIVE_AFTER = 7, EEventCount = 8
};
struct EOutputTrace {
  uint64_t valid, m_idx, n_idx, l_idx, local_seq;
  EClockPair events[EEventCount];
};
struct EFinalTrace { EClockPair time; uint64_t tiles, smid; };
struct EPingpongTrace {
  EClockPair entry, producer_first_work;
  uint64_t entry_smid, overflow;
  EOutputTrace outputs[2][2];
  EFinalTrace finals[2];
};

#if CFG_B_PINGPONG_TRACE
__device__ __constant__ EPingpongTrace* e_trace_pointer;
__device__ __forceinline__ EClockPair e_clocks() {
  EClockPair result;
  asm volatile("mov.u64 %0, %%clock64; mov.u64 %1, %%globaltimer;"
      : "=l"(result.cycle), "=l"(result.ns) :: "memory");
  return result;
}
__device__ __forceinline__ unsigned e_smid() {
  unsigned result;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(result));
  return result;
}
__device__ __forceinline__ void v06_entry() {
  if (threadIdx.x == 0) {
    e_trace_pointer->entry = e_clocks();
    e_trace_pointer->entry_smid = e_smid();
  }
}
template<class Work>
__device__ __forceinline__ void v06_begin(Work const& work, int tile) {
  if (threadIdx.x == 0 && tile == 0)
    e_trace_pointer->producer_first_work = e_clocks();
  if (threadIdx.x < 128 || threadIdx.x % 128 != 0) return;
  if (tile < 0 || tile >= 2) { e_trace_pointer->overflow = 1; return; }
  auto& record = e_trace_pointer->outputs[threadIdx.x / 128 - 1][tile];
  record.valid = 1;
  record.m_idx = work.M_idx; record.n_idx = work.N_idx; record.l_idx = work.L_idx;
  record.local_seq = tile;
}
__device__ __forceinline__ void e_stamp(int event, int tile) {
  if (threadIdx.x < 128 || threadIdx.x % 128 != 0) return;
  if (tile < 0 || tile >= 2) { e_trace_pointer->overflow = 1; return; }
  e_trace_pointer->outputs[threadIdx.x / 128 - 1][tile].events[event] = e_clocks();
}
__device__ __forceinline__ void v06_stamp(int event, int tile) { e_stamp(event, tile); }
__device__ __forceinline__ void v06_final(int tiles) {
  if (threadIdx.x < 128 || threadIdx.x % 128 != 0) return;
  auto& record = e_trace_pointer->finals[threadIdx.x / 128 - 1];
  record.time = e_clocks(); record.tiles = tiles; record.smid = e_smid();
}
#else
__device__ __forceinline__ void v06_entry() {}
template<class Work> __device__ __forceinline__ void v06_begin(Work const&, int) {}
__device__ __forceinline__ void v06_stamp(int, int) {}
__device__ __forceinline__ void e_stamp(int, int) {}
__device__ __forceinline__ void v06_final(int) {}
#endif
