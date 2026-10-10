#pragma once
// Default: clock64 events, globaltimer at CTA entry/final. R18_DUAL_CLOCK adds
// globaltimer to the same four per-tile sites and to producer first work.
// R18_MATCH_DUAL_LAYOUT reserves the same space without adding those reads.
// Fields 4,5 record M_idx+1,N_idx+1; wide fields 6..9 hold the four globaltimer events.
// Per CTA (uint64 words):
//   [0] entry cycles      [1] entry ns       [2] smid+1     [3] producer first-work cycles
//   [4] role1 final cyc   [5] role1 final ns [6] role1 tiles
//   [7] role2 final cyc   [8] role2 final ns [9] role2 tiles [10] overflow flag
//   R15_OUTPUT_NS only: [13]/[14] first-tile issuer EPI_PERMIT/EPI_DONE globaltimer.
//   The R15 single-tile cases use thread 256; [8] remains the store_tail endpoint.
//   R18_DUAL_CLOCK: [13] producer first-work ns.
//   [16 + ((role-1)*V06Tiles + tile)*V06TileWords + e], e = FIRST_MMA, MAIN_END, EPI_PERMIT, EPI_DONE
// role = threadIdx.x / 128 (1, 2 = consumer warpgroups). Only thread %128 == 0 writes.
// EPI_DONE: cooperative = epilogue store() returned; pingpong = store_tail() returned.
// Final: cooperative = after the post-loop store_tail of the TMA-issuing thread 256;
//        pingpong = each consumer warpgroup leaving its work loop (after its last store_tail).
#include <cstdint>
#if defined(R18_DUAL_CLOCK) || defined(R18_MATCH_DUAL_LAYOUT)
constexpr int V06TileWords = 10;
#else
constexpr int V06TileWords = 6;
#endif
constexpr int V06Tiles = 64, V06Head = 16, V06CtaWords = V06Head + 2 * V06Tiles * V06TileWords;
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

#ifdef R13_FIRST_SUPPLY
__device__ __forceinline__ bool r13_first_target(int output_seq) {
  return blockIdx.x==55 && blockIdx.y==0 && blockIdx.z==0 && output_seq==0 && r13_first_supply_events;
}
__device__ __forceinline__ void r13_first_issue_stamp() {
  uint64_t c=v06_clock(),n=v06_ns();
  r10_producer_delay_trace[4]=c; r10_producer_delay_trace[5]=n;
}
#endif

#ifdef R13_READY_RELEASE
__device__ __forceinline__ void r13_first_release_stamp() {
  uint64_t c=v06_clock(),n=v06_ns();
  r10_producer_delay_trace[16]=c; r10_producer_delay_trace[17]=n;
}
#endif

#ifdef R13_READY_OBSERVER
// Read-only phase-0 observation in the otherwise idle static-scheduler warp.
// Test timestamps are bounds: false uses the pre-test read, true the exit read.
__device__ __forceinline__ void r13_observe_ready(unsigned bar) {
  if (threadIdx.x!=32 || blockIdx.x!=55 || blockIdx.y || blockIdx.z || !r13_ready_observer) return;
  uint64_t sc,sn,lc,ln,ec,en;
  unsigned success,tests,sm;
  asm volatile(
    "{ .reg .pred ready,again; .reg .u32 addr,n,ok;\n"
    "  .reg .u64 sc,sn,lc,ln,ec,en,pc,pn;\n"
    "  mov.u32 addr,%8; mov.u32 n,0; mov.u32 ok,0;\n"
    "  mov.u64 lc,0; mov.u64 ln,0;\n"
    "  mov.u64 sc,%%clock64; mov.u64 sn,%%globaltimer;\n"
    "R13_READY_LOOP:\n"
    "  mov.u64 pc,%%clock64; mov.u64 pn,%%globaltimer;\n"
    "  mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 ready,[addr],0;\n"
    "  add.u32 n,n,1; @ready bra R13_READY_SUCCESS;\n"
    "  mov.u64 lc,pc; mov.u64 ln,pn;\n"
    "  setp.lt.u32 again,n,4096; @again bra R13_READY_LOOP;\n"
    "  bra R13_READY_DONE;\n"
    "R13_READY_SUCCESS: mov.u32 ok,1;\n"
    "R13_READY_DONE:\n"
    "  mov.u64 ec,%%clock64; mov.u64 en,%%globaltimer;\n"
    "  mov.u64 %0,sc; mov.u64 %1,sn; mov.u64 %2,lc; mov.u64 %3,ln;\n"
    "  mov.u64 %4,ec; mov.u64 %5,en; mov.u32 %6,ok; mov.u32 %7,n; }\n"
    : "=l"(sc),"=l"(sn),"=l"(lc),"=l"(ln),"=l"(ec),"=l"(en),"=r"(success),"=r"(tests)
    : "r"(bar) : "memory");
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  auto out=r10_producer_delay_trace+8;
  out[0]=sc;out[1]=sn;out[2]=lc;out[3]=ln;out[4]=ec;out[5]=en;
  out[6]=tests;out[7]=uint64_t(success)|(uint64_t(sm+1)<<1);
}
#endif

#if defined(R13_STAGE_SPARSE) || defined(R13_STAGE_PROGRESS)
// Same 784-word allocation as ordinary plain. Only CTA0 writes selected fields.
// Word 10 remains the existing overflow sentinel. Event values are same-SM clocks.
struct R13SparseProducer {
  uint64_t acquire_begin=0, acquire_end=0, issue_end=0;
  uint64_t refill_begin=0, refill_end=0, refill_issue_end=0;
  // Actual sampled slot/phase pairs: [2:0]/[3] and [6:4]/[7].
  // Keep one metadata register live beside six clocks in the 40-register producer.
  uint32_t state=0;
};
#ifdef R13_STAGE_PROGRESS
struct R13SparseConsumer {
  uint64_t wait_begin=0, next_wait_begin=0, retire_end=0;
  uint64_t commit1=0, commit16=0, commit17=0, commit63=0;
  uint64_t first_mma=0, mma_return=0, main_end=0;
#ifdef R13_STAGE_HANDOFF
  uint64_t sender_mma_return=0, math_wait_begin=0, math_wait_end=0, first_wait_begin=0;
#endif
};
#else
struct R13SparseConsumer {
  uint64_t wait_begin=0, wait_end=0, commit_end=0, retire_end=0;
  uint64_t release_begin=0, release_end=0, first_mma=0, main_end=0;
  uint64_t mma_return=0, tail_begin=0;
  int read_slot=0, release_slot=0, release_phase=0;
};
#endif
__device__ __forceinline__ bool r13_sparse_cta() {
  return blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0;
}
__device__ __forceinline__ void r13_sparse_producer_flush(R13SparseProducer const& s) {
  auto b=v06_base();
  b[4]=uint64_t(s.state&7)+1; b[5]=uint64_t((s.state>>3)&1)+1;
  b[6]=s.acquire_begin; b[7]=s.acquire_end; b[8]=s.issue_end;
  b[9]=s.refill_begin; b[11]=s.refill_end; b[12]=s.refill_issue_end;
  b[27]=uint64_t((s.state>>4)&7)+1; b[28]=uint64_t((s.state>>7)&1)+1;
  b[29]=1; b[30]=16; b[31]=0x52313301;
}
#ifdef R13_STAGE_PROGRESS
__device__ __forceinline__ void r13_sparse_consumer_flush(R13SparseConsumer const& s) {
  auto b=v06_base();
  b[15]=s.wait_begin; b[16]=s.next_wait_begin; b[17]=s.commit16; b[18]=s.retire_end;
  b[19]=s.commit17; b[20]=s.commit1; b[21]=s.first_mma; b[22]=s.main_end;
  b[24]=s.commit63; b[32]=s.mma_return; b[31]=0x52313302;
#ifdef R13_STAGE_HANDOFF
  b[35]=s.math_wait_begin; b[36]=s.math_wait_end; b[37]=s.first_wait_begin;
  b[31]=0x52313303;
#endif
}
#else
__device__ __forceinline__ void r13_sparse_consumer_flush(R13SparseConsumer const& s) {
  auto b=v06_base();
  b[15]=s.wait_begin; b[16]=s.wait_end; b[17]=s.commit_end; b[18]=s.retire_end;
  b[19]=s.release_begin; b[20]=s.release_end; b[21]=s.first_mma; b[22]=s.main_end;
  b[24]=uint64_t(s.read_slot)+1; b[25]=uint64_t(s.release_slot)+1;
  b[26]=uint64_t(s.release_phase)+1; b[32]=s.mma_return; b[33]=s.tail_begin;
}
#endif
__device__ __forceinline__ void v06_entry() {
#ifdef R13_STAGE_PROGRESS
  if (!r13_sparse_cta() || threadIdx.x!=256) return;
  unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
  v06_base()[1]=uint64_t(sm)+1;
#else
  if (!r13_sparse_cta() || threadIdx.x!=0) return;
  unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
  auto b=v06_base();b[0]=v06_clock();b[1]=uint64_t(sm)+1;
#endif
}
template<class Work>
__device__ __forceinline__ void v06_begin(Work const& work,int tile) {
  if (!r13_sparse_cta()) return;
#ifndef R13_STAGE_PROGRESS
  if (threadIdx.x==0 && tile==1) {v06_base()[2]=uint64_t(work.M_idx)+1;v06_base()[3]=uint64_t(work.N_idx)+1;}
#endif
  // Consumer1's first local work item is the CTA's second global output.
  if (threadIdx.x==256 && tile==0) {v06_base()[13]=uint64_t(work.M_idx)+1;v06_base()[14]=uint64_t(work.N_idx)+1;}
#ifdef R13_STAGE_HANDOFF
  if (threadIdx.x==128 && tile==0) {v06_base()[39]=uint64_t(work.M_idx)+1;v06_base()[40]=uint64_t(work.N_idx)+1;}
#endif
}
__device__ __forceinline__ void v06_stamp(int,int) {}
__device__ __forceinline__ void v06_final(int) {
  if (!r13_sparse_cta() || threadIdx.x!=256) return;
  unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
  v06_base()[23]=uint64_t(sm)+1;
}
#else

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
  if (threadIdx.x == 0 && tile == 0) {
    v06_base()[3] = v06_clock();
#ifdef R18_DUAL_CLOCK
    v06_base()[13] = v06_ns();
#endif
  }
  #ifndef R18_LIGHT
  if(threadIdx.x>=128 && threadIdx.x%128==0){
    if(tile<0||tile>=V06Tiles){v06_base()[10]=1;return;}
    auto out=v06_base()+V06Head+((threadIdx.x/128-1)*V06Tiles+tile)*V06TileWords;
    out[4]=uint64_t(work.M_idx)+1;out[5]=uint64_t(work.N_idx)+1;
  }
  #endif
}
__device__ __forceinline__ void v06_stamp(int event, int tile) {
  if (threadIdx.x % 128 != 0 || threadIdx.x < 128) return;
  int role = threadIdx.x / 128;
  uint64_t c = v06_clock();
#ifdef R18_DUAL_CLOCK
  uint64_t ns = v06_ns();
#endif
#ifdef R15_OUTPUT_NS
  if (threadIdx.x == 256 && tile == 0 &&
      (event == V06_EPI_PERMIT || event == V06_EPI_DONE))
    v06_base()[event == V06_EPI_PERMIT ? 13 : 14] = v06_ns();
#endif
  if (tile < 0 || tile >= V06Tiles) { v06_base()[10] = 1; return; }
  auto out = v06_base() + V06Head + ((role - 1) * V06Tiles + tile) * V06TileWords;
  out[event] = c;
#ifdef R18_DUAL_CLOCK
  out[6 + event] = ns;
#endif
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

#endif // R13_STAGE_SPARSE
