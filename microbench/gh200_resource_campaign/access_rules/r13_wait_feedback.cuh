#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define R13_FEEDBACK_INLINE __host__ __device__ __forceinline__
#else
#define R13_FEEDBACK_INLINE inline
#endif

R13_FEEDBACK_INLINE unsigned r13_wait_feedback_mask(unsigned mode) {
  return mode == 1 ? 0x1fffc : mode == 2 ? 0x4 : mode == 3 ? 0x10000 : 0;
}

R13_FEEDBACK_INLINE bool r13_wait_feedback_selected(unsigned mask, int group) {
  return group >= 0 && group < 64 && ((uint64_t(mask) >> group) & 1);
}

struct R13WaitSum {
  uint64_t cycles=0, count=0, mask=0, first_begin=0, last_end=0, invalid=0;
  uint64_t group2_cycles=0, group16_cycles=0;

  R13_FEEDBACK_INLINE void record(int group, uint64_t begin, uint64_t end) {
    if (group < 2 || group > 16) { invalid |= 1; return; }
    uint64_t bit = uint64_t(1) << group;
    if (mask & bit) invalid |= 2;
    if (end < begin || (count && begin < last_end)) invalid |= 4;
    if (count == 0) first_begin = begin;
    if (end >= begin) {
      uint64_t duration = end - begin;
      if (UINT64_MAX - cycles < duration) invalid |= 8;
      else cycles += duration;
      if (group == 2) group2_cycles = duration;
      if (group == 16) group16_cycles = duration;
    }
    mask |= bit;
    ++count;
    last_end = end;
  }
};

#undef R13_FEEDBACK_INLINE
