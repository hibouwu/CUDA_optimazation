#pragma once
#include <cstdint>

// Conditional reference-domain bracket, not a physical-nanosecond certificate.
// Zero on either prerequisite read makes the complete record invalid. No retry.
struct CfgCGtBracket {
  uint64_t before, cycles, after;
};

__device__ __forceinline__ CfgCGtBracket cfg_c_gt_bracket() {
  CfgCGtBracket stamp;
  asm volatile(
    "{ .reg .pred valid; .reg .u64 before,cycles,after;\n"
    "  mov.u64 before,%%globaltimer;\n"
    "  setp.eq.u64 valid,before,0; @valid bra CFG_C_GT_INVALID;\n"
    "  mov.u64 cycles,%%clock64;\n"
    "  setp.eq.u64 valid,cycles,0; @valid bra CFG_C_GT_INVALID;\n"
    "  mov.u64 after,%%globaltimer; bra CFG_C_GT_DONE;\n"
    "CFG_C_GT_INVALID: mov.u64 before,0; mov.u64 cycles,0; mov.u64 after,0;\n"
    "CFG_C_GT_DONE: mov.u64 %0,before; mov.u64 %1,cycles; mov.u64 %2,after; }\n"
    : "=l"(stamp.before), "=l"(stamp.cycles), "=l"(stamp.after)
    : : "memory");
  return stamp;
}
