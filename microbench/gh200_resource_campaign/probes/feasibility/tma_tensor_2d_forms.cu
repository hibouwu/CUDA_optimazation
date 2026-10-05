// S15 compile-only rank2 UINT16 tensor forms. main launches no CUDA work.
// The caller would need a successfully host-encoded immutable tensor map,
// one CTA with128threads and Q+1056 dynamic shared bytes. No runtime claim.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
static_assert(sizeof(CUtensorMap)==128,"tensor map ABI");

__device__ __forceinline__ unsigned addr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;
}
__device__ __forceinline__ unsigned index2d(unsigned x,unsigned y,unsigned w,bool sw128) {
  return sw128?y*64+((x/8)^(y%8))*8+x%8:y*w+x;
}

extern "C" __global__ void tma_tensor_2d_g2s_form(
    const __grid_constant__ CUtensorMap map,uint16_t* observed,
    unsigned w,unsigned h,bool sw128) {
  extern __shared__ __align__(16) unsigned char storage[];
  const unsigned delta=(1024-(addr(storage)&1023))&1023;
  auto* tile=reinterpret_cast<uint16_t*>(storage+delta);
  const unsigned q=2*w*h,bar=addr(storage+delta+q);
  if(threadIdx.x==0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(bar):"memory");
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
  }
  __syncthreads();
  if(threadIdx.x==0) {
    // Representative first slot only; finite-matrix runtime is not this kernel.
    const int x=0,y=0;
    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"::"r"(bar),"r"(q):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 ::"r"(addr(tile)),"l"(&map),"r"(x),"r"(y),"r"(bar):"memory");
    unsigned long long token;
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];":"=l"(token):"r"(bar):"memory");
    const auto start=now_ns();
    for(;;) {
      unsigned done;
      asm volatile("{.reg .pred p; mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2, 64; selp.b32 %0, 1, 0, p;}"
                   :"=r"(done):"r"(bar),"l"(token):"memory");
      if(done)break;
      if(now_ns()-start>=1000000000ull)asm volatile("trap;":::"memory");
    }
  }
  __syncthreads();
  for(unsigned n=threadIdx.x;n<w*h;n+=blockDim.x)
    observed[n]=tile[index2d(n%w,n/w,w,sw128)];
  __syncthreads();
  if(threadIdx.x==0)asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(bar):"memory");
}

extern "C" __global__ void tma_tensor_2d_s2g_form(
    const __grid_constant__ CUtensorMap map,unsigned w,unsigned h,bool sw128,unsigned seed) {
  extern __shared__ __align__(16) unsigned char storage[];
  const unsigned delta=(1024-(addr(storage)&1023))&1023;
  auto* tile=reinterpret_cast<uint16_t*>(storage+delta);
  for(unsigned n=threadIdx.x;n<w*h;n+=blockDim.x) {
    const unsigned x=n%w,y=n/w;
    tile[index2d(x,y,w,sw128)]=uint16_t(17ull*x+31ull*y+uint64_t(seed));
  }
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  __syncthreads();
  if(threadIdx.x==0) {
    const int x=0,y=0;
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
                 "[%0, {%1, %2}], [%3];"
                 ::"l"(&map),"r"(x),"r"(y),"r"(addr(tile)):"memory");
    asm volatile("cp.async.bulk.commit_group;":::"memory");
    asm volatile("cp.async.bulk.wait_group 0;":::"memory");
  }
  __syncthreads();
}
int main(){return 0;}
