#include <cuda_runtime.h>
#include <cstdint>

// Compile-only candidates. Host must launch one full32-thread CTA and pass
// mask==UINT32_MAX. No host runner, qualification or latency claim is supplied.
__device__ __forceinline__ std::uint64_t warp_candidate_cycle() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
  return value;
}

extern "C" __global__ void warp_mask_bare_v1(int iterations, unsigned mask,
    unsigned seed, unsigned* output, std::uint64_t* clocks) {
  const unsigned tid=threadIdx.x;
  __shared__ std::uint64_t start;
  if(tid==0)start=warp_candidate_cycle();
  __syncthreads();
  #pragma unroll 1
  for(int iteration=0;iteration<iterations;++iteration) {
    #pragma unroll 8
    for(int phase=0;phase<8;++phase)
      asm volatile("bar.warp.sync %0;" :: "r"(mask) : "memory");
  }
  output[tid]=seed+17u*tid;
  __syncthreads();
  if(tid==0){clocks[0]=start;clocks[1]=warp_candidate_cycle();}
}

template<bool Warp>
__device__ __forceinline__ void warp_shared_pair_body_v1(int iterations,
    unsigned mask, unsigned seed, unsigned* output, std::uint64_t* clocks) {
  const unsigned tid=threadIdx.x;
  __shared__ volatile unsigned tokens[32];
  __shared__ std::uint64_t start;
  unsigned last=0,checksum=0,completed=0;
  if(tid==0)start=warp_candidate_cycle();
  __syncthreads();
  #pragma unroll 1
  for(int iteration=0;iteration<iterations;++iteration) {
    #pragma unroll 8
    for(int position=0;position<8;++position) {
      const unsigned phase=unsigned(iteration)*8u+unsigned(position);
      tokens[tid]=(seed+17u*tid)^((phase+1u)*2246822519u);
      if constexpr(Warp)asm volatile("bar.warp.sync %0;" :: "r"(mask) : "memory");
      else asm volatile("bar.sync 1;" ::: "memory");
      last=tokens[(tid+1u)%32u];
      checksum+=last;
      // Protect readers from next-phase writers; this is a two-barrier pair.
      if constexpr(Warp)asm volatile("bar.warp.sync %0;" :: "r"(mask) : "memory");
      else asm volatile("bar.sync 1;" ::: "memory");
      ++completed;
    }
  }
  output[tid*4]=seed+17u*tid;
  output[tid*4+1]=last;
  output[tid*4+2]=checksum;
  output[tid*4+3]=completed;
  __syncthreads();
  if(tid==0){clocks[0]=start;clocks[1]=warp_candidate_cycle();}
}
extern "C" __global__ void warp_shared_pair_v1(int iterations,unsigned mask,
    unsigned seed,unsigned* output,std::uint64_t* clocks) {
  warp_shared_pair_body_v1<true>(iterations,mask,seed,output,clocks);
}
extern "C" __global__ void cta_shared_pair_v1(int iterations,unsigned mask,
    unsigned seed,unsigned* output,std::uint64_t* clocks) {
  warp_shared_pair_body_v1<false>(iterations,mask,seed,output,clocks);
}
