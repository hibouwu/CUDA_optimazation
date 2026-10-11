#include "r09_clock_stamp.cuh"
static_assert(sizeof(uint64_t)==8,"64-bit timestamps required");
static_assert(sizeof(R09ClockStamp)==24,"three-word stamp required");
static_assert(alignof(R09ClockStamp)==8,"eight-byte alignment required");
extern "C" __global__ void r09_clock_stamp_ptx_fixture(uint64_t* output) {
  R09ClockStamp stamp_0=r09_guarded_clock_stamp();
  output[0]=stamp_0.clock_before;
  output[1]=stamp_0.globaltimer_ns;
  output[2]=stamp_0.clock_after;
  R09ClockStamp stamp_1=r09_guarded_clock_stamp();
  output[3]=stamp_1.clock_before;
  output[4]=stamp_1.globaltimer_ns;
  output[5]=stamp_1.clock_after;
}
