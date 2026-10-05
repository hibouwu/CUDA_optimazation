#include <cuda_runtime.h>
#include <cstdint>

// Compile-only SM90 candidate. Required host domain: one full32-thread CTA,
// iterations>0, mask==UINT32_MAX. No GPU launch/qualification is supplied here.
// Both half-warps execute two matching full-mask warp barriers per pair. The
// first protects cross-half reads; the second prevents next-pair write/read races.
__device__ __forceinline__ std::uint64_t divergent_pair_cycle_v2() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
  return value;
}

__device__ __forceinline__ unsigned divergent_warp_pair_v2(unsigned tid,
    unsigned address,unsigned peer_address,unsigned initial,unsigned phase_word,
    unsigned mask) {
  unsigned consumed;
  // Keep two distinct producer/barrier/consumer/barrier paths in one opaque
  // PTX block. Do not assume the compiler preserves WARPSYNC: actual SASS is
  // still the acceptance criterion. sm_6x's same-PC convergence restriction is
  // not applied to this sm_90 candidate; all32 named lanes take part each pair.
  asm volatile(
      "{ .reg .pred lower; .reg .b32 token;\n"
      "setp.lt.u32 lower, %1, 16;\n"
      "@lower bra LOW;\n"
      "add.u32 token, %4, 324508639;\n"
      "xor.b32 token, token, %5;\n"
      "st.volatile.shared.u32 [%2], token;\n"
      "bar.warp.sync %6;\n"
      "ld.volatile.shared.u32 %0, [%3];\n"
      "bar.warp.sync %6;\n"
      "bra DONE;\n"
      "LOW:\n"
      "xor.b32 token, %4, %5;\n"
      "st.volatile.shared.u32 [%2], token;\n"
      "bar.warp.sync %6;\n"
      "ld.volatile.shared.u32 %0, [%3];\n"
      "bar.warp.sync %6;\n"
      "DONE: }"
      : "=&r"(consumed)
      : "r"(tid),"r"(address),"r"(peer_address),"r"(initial),
        "r"(phase_word),"r"(mask)
      : "memory");
  return consumed;
}

template<bool Warp>
__device__ __forceinline__ void divergent_pair_body_v2(int iterations,
    unsigned mask,unsigned seed,unsigned* output,std::uint64_t* clocks) {
  const unsigned tid=threadIdx.x;
  const unsigned peer=tid^16u;
  __shared__ unsigned tokens[32];
  __shared__ std::uint64_t start;
  const unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(&tokens[tid]));
  const unsigned peer_address=static_cast<unsigned>(__cvta_generic_to_shared(&tokens[peer]));
  unsigned last=0,checksum=0,completed=0;
  const unsigned initial=seed+17u*tid;
  if(tid==0)start=divergent_pair_cycle_v2();
  __syncthreads();
  #pragma unroll 1
  for(int iteration=0;iteration<iterations;++iteration) {
    #pragma unroll 8
    for(int position=0;position<8;++position) {
      const unsigned phase=unsigned(iteration)*8u+unsigned(position);
      const unsigned phase_word=(phase+1u)*2246822519u;
      if constexpr(Warp) {
        last=divergent_warp_pair_v2(tid,address,peer_address,initial,phase_word,mask);
      } else {
        // CTA aligned barrier stays at the common merge, unlike the warp
        // candidate's cross-path corresponding barriers. This is a valid
        // identical-data-task comparison, not an instruction-for-instruction
        // subtraction baseline or a claim of a bare warp barrier latency.
        const unsigned token=(initial+(tid>=16?324508639u:0u))^phase_word;
        asm volatile("st.volatile.shared.u32 [%0], %1;" :: "r"(address),"r"(token) : "memory");
        asm volatile("bar.sync 1;" ::: "memory");
        asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(last) : "r"(peer_address) : "memory");
        asm volatile("bar.sync 1;" ::: "memory");
      }
      checksum+=last;
      ++completed;
    }
  }
  output[tid*4]=initial;
  output[tid*4+1]=last;
  output[tid*4+2]=checksum;
  output[tid*4+3]=completed;
  __syncthreads();
  if(tid==0){clocks[0]=start;clocks[1]=divergent_pair_cycle_v2();}
}
extern "C" __global__ void warp_divergent_pair_v2(int iterations,unsigned mask,
    unsigned seed,unsigned* output,std::uint64_t* clocks) {
  divergent_pair_body_v2<true>(iterations,mask,seed,output,clocks);
}
extern "C" __global__ void cta_divergent_pair_v2(int iterations,unsigned mask,
    unsigned seed,unsigned* output,std::uint64_t* clocks) {
  divergent_pair_body_v2<false>(iterations,mask,seed,output,clocks);
}
