#pragma once
// V06 light per-CTA stamps (clock64; globaltimer only at CTA entry and final release).
// R18/R19: six words per tile; fields 4,5 record M_idx+1,N_idx+1.
// Per CTA (uint64 words):
//   [0] entry cycles      [1] entry ns       [2] smid+1     [3] producer first-work cycles
//   [4] role1 final cyc   [5] role1 final ns [6] role1 tiles
//   [7] role2 final cyc   [8] role2 final ns [9] role2 tiles [10] overflow flag
//   [16 + ((role-1)*V06Tiles + tile)*4 + e], e = FIRST_MMA, MAIN_END, EPI_PERMIT, EPI_DONE
// role = threadIdx.x / 128 (1, 2 = consumer warpgroups). Only thread %128 == 0 writes.
// EPI_DONE: cooperative = epilogue store() returned; pingpong = store_tail() returned.
// Final: cooperative = after the post-loop store_tail of the TMA-issuing thread 256;
//        pingpong = each consumer warpgroup leaving its work loop (after its last store_tail).
#include <cstdint>
constexpr int V06Tiles = 64, V06Head = 16, V06CtaWords = V06Head + 2 * V06Tiles * 6;
__constant__ uint64_t* v06_trace_ptr;
enum V06Event { V06_FIRST_MMA, V06_MAIN_END, V06_EPI_PERMIT, V06_EPI_DONE };

__device__ __forceinline__ uint64_t v06_clock() {
  uint64_t c;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(c) :: "memory");
  return c;
}
__device__ __forceinline__ uint64_t v06_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  return t;
}
__device__ __forceinline__ uint64_t* v06_base() {
  unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  return v06_trace_ptr + size_t(block) * V06CtaWords;
}
__device__ __forceinline__ void v06_entry() {
  if (threadIdx.x != 0) return;
  uint64_t c = v06_clock(), t = v06_ns();
  unsigned sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  uint64_t* b = v06_base();
  b[0] = c; b[1] = t; b[2] = uint64_t(sm) + 1;
}
#ifdef V08_ENDS
// V08 "ends": only CTA entry, producer first work and final release; no per-tile writes.
template <class Work>
__device__ __forceinline__ void v06_begin(Work const&, int tile) {
  if (threadIdx.x == 0 && tile == 0) v06_base()[3] = v06_clock();
}
__device__ __forceinline__ void v06_stamp(int, int) {}
#else
// Called at the top of each work-loop iteration by producer and consumer roles.
template <class Work>
__device__ __forceinline__ void v06_begin(Work const& work, int tile) {
  if (threadIdx.x == 0 && tile == 0) v06_base()[3] = v06_clock();
  #ifndef R18_LIGHT
  if(threadIdx.x>=128 && threadIdx.x%128==0){
    if(tile<0||tile>=V06Tiles){v06_base()[10]=1;return;}
    auto out=v06_base()+V06Head+((threadIdx.x/128-1)*V06Tiles+tile)*6;
    out[4]=uint64_t(work.M_idx)+1;out[5]=uint64_t(work.N_idx)+1;
  }
  #endif
}
__device__ __forceinline__ void v06_stamp(int event, int tile) {
  if (threadIdx.x % 128 != 0 || threadIdx.x < 128) return;
  int role = threadIdx.x / 128;
  uint64_t c = v06_clock();
  if (tile < 0 || tile >= V06Tiles) { v06_base()[10] = 1; return; }
  v06_base()[V06Head + ((role - 1) * V06Tiles + tile) * 6 + event] = c;
}
#endif
__device__ __forceinline__ void v06_final(int tiles) {
  if (threadIdx.x % 128 != 0 || threadIdx.x < 128) return;
  int role = threadIdx.x / 128;
  uint64_t c = v06_clock(), t = v06_ns();
  uint64_t* b = v06_base() + 4 + (role - 1) * 3;
  b[0] = c; b[1] = t; b[2] = uint64_t(tiles);
  unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
  v06_base()[11+role-1]=uint64_t(sm)+1;
}
