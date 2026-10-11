#pragma once
#include <cstdint>

// Candidate read pairing only. Runtime validity, measurement enclosure and
// perturbation remain unqualified until integrated and measured by the owner.
struct R09ClockStamp {
  uint64_t clock_before;
  uint64_t globaltimer_ns;
  uint64_t clock_after;
};

__device__ __forceinline__ R09ClockStamp r09_guarded_clock_stamp() {
  R09ClockStamp stamp;
  // At most two reads of a zero counter. The backedges and the exit branches
  // must survive compilation; otherwise this is not a dependency witness.
  // All-zero output on exhaustion remains invalid, with no silent fallback.
  asm volatile(
    "{ .reg .pred nonzero,retry; .reg .u32 attempts; .reg .u64 before,ns,after;\n"
    "  mov.u32 attempts,0;\n"
    "R09_READ_BEFORE:\n"
    "  mov.u64 before,%%clock64; add.u32 attempts,attempts,1;\n"
    "  setp.ne.u64 nonzero,before,0; @nonzero bra R09_READ_GT_START;\n"
    "  setp.lt.u32 retry,attempts,2; @retry bra R09_READ_BEFORE;\n"
    "  bra R09_INVALID;\n"
    "R09_READ_GT_START: mov.u32 attempts,0;\n"
    "R09_READ_GT:\n"
    "  mov.u64 ns,%%globaltimer; add.u32 attempts,attempts,1;\n"
    "  setp.ne.u64 nonzero,ns,0; @nonzero bra R09_READ_AFTER;\n"
    "  setp.lt.u32 retry,attempts,2; @retry bra R09_READ_GT;\n"
    "  bra R09_INVALID;\n"
    "R09_READ_AFTER: mov.u64 after,%%clock64; bra R09_DONE;\n"
    "R09_INVALID: mov.u64 before,0; mov.u64 ns,0; mov.u64 after,0;\n"
    "R09_DONE: mov.u64 %0,before; mov.u64 %1,ns; mov.u64 %2,after; }\n"
    : "=l"(stamp.clock_before),"=l"(stamp.globaltimer_ns),"=l"(stamp.clock_after)
    : : "memory");
  return stamp;
}
