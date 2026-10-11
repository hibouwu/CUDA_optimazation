#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define R13_WAIT_INLINE __host__ __device__ __forceinline__
#else
#define R13_WAIT_INLINE inline
#endif

// Only the fifteen full-wait intervals inside CTA0's second-output C1 -> C16.
struct R13WaitSum {
  uint64_t cycles=0, count=0, mask=0, first_begin=0, last_end=0, invalid=0;

  R13_WAIT_INLINE void record(int group, uint64_t begin, uint64_t end) {
    if (group < 2 || group > 16) { invalid |= 1; return; }
    uint64_t bit = uint64_t(1) << group;
    if (mask & bit) invalid |= 2;
    if (end < begin || (count && begin < last_end)) invalid |= 4;
    if (count == 0) first_begin = begin;
    if (end >= begin) {
      uint64_t duration = end - begin;
      if (UINT64_MAX - cycles < duration) invalid |= 8;
      else cycles += duration;
    }
    mask |= bit;
    ++count;
    last_end = end;
  }
};

#undef R13_WAIT_INLINE
