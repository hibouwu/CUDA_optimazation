#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); \
  std::exit(1); } } while(0)

static constexpr unsigned kIdesc = 136314896u;
static constexpr long long kWorkPerInstruction = 524288LL;
static constexpr int kWarpsPerBlock = 1;
static constexpr int kThreads = 32;
static constexpr char kCaseId[] = "fp16_f32_m128n128k16_single_warp_block";
static constexpr char kPrecision[] = "fp16_f32";
static constexpr char kWorkUnit[] = "flop";
static constexpr int kLogicalBits = 16;
static constexpr int kDescriptorBits = 16;

__device__ __forceinline__ unsigned smem_u32(void const* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ unsigned long long smem_desc(
    void const* p, unsigned leading, unsigned stride) {
  unsigned long long d=0; unsigned a=smem_u32(p);
  d|=static_cast<unsigned long long>((a>>4)&0x3fff);
  d|=static_cast<unsigned long long>(leading&0x3fff)<<16;
  d|=static_cast<unsigned long long>(stride&0x3fff)<<32;
  d|=1ull<<46; return d;
}

__device__ __forceinline__ unsigned long long global_nanoseconds() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ void wait_barrier(unsigned long long* b) {
  unsigned a=smem_u32(b); unsigned ticks=0x989680;
  asm volatile("{ .reg .pred p; L_wait: mbarrier.try_wait.parity.shared::cta.b64 "
               "p,[%0],0,%1; @p bra L_done; bra L_wait; L_done: }"::"r"(a),"r"(ticks));
}

__global__ __launch_bounds__(kThreads,1) void benchmark(
    int iters, unsigned long long* cycles_out, unsigned long long* start_ns_out,
    unsigned long long* stop_ns_out, unsigned* smid_out) {
  __shared__ alignas(16) unsigned char a[32768];
  __shared__ alignas(16) unsigned char b[32768];
  __shared__ alignas(8) unsigned long long done;
  __shared__ unsigned tmem;
  for(int i=threadIdx.x;i<32768;i+=blockDim.x){a[i]=(i*13+1)&255;b[i]=(i*17+3)&255;}
  if(threadIdx.x==0) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::
      "r"(smem_u32(&done)),"r"(kWarpsPerBlock));
  __syncthreads();
  if(threadIdx.x<32) asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"::
      "r"(smem_u32(&tmem)),"r"(512));
  __syncthreads();
  unsigned long long ad=smem_desc(a,16,8);
  unsigned long long bd=smem_desc(b,16,8);
  unsigned long long start_ns=global_nanoseconds();
  unsigned long long start=clock64();
  for(int i=0;i<iters;++i){
    unsigned enable=i!=0; unsigned dst=tmem;
    unsigned tsa=tmem+256, tsb=tmem+384;
    if((threadIdx.x&31)==0){
      asm volatile("{ .reg .pred p; setp.ne.b32 p,%4,0; tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,{%5,%6,%7,%8},p; }"::"r"(dst),"l"(ad),"l"(bd),"r"(kIdesc),"r"(enable),"r"(0),"r"(0),"r"(0),"r"(0));
    }
  }
  if((threadIdx.x&31)==0) asm volatile(
    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::
    "r"(smem_u32(&done)));
  if(threadIdx.x==0) wait_barrier(&done);
  __syncthreads(); unsigned long long stop=clock64();
  unsigned long long stop_ns=global_nanoseconds();
  if(threadIdx.x==0){cycles_out[blockIdx.x]=stop-start;
                    start_ns_out[blockIdx.x]=start_ns;
                    stop_ns_out[blockIdx.x]=stop_ns;
                    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_out[blockIdx.x]));}
  __syncthreads();
  if(threadIdx.x<32){
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(tmem),"r"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"::);
  }
}

int main(int argc,char** argv){
  int iters=argc>1?std::atoi(argv[1]):10000;
  double freq=argc>2?std::atof(argv[2]):1575000000.0;
  cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,0));
  int blocks=1;
  unsigned long long *dc=nullptr,*db=nullptr,*de=nullptr; unsigned *ds=nullptr;
  CUDA_CHECK(cudaMalloc(&dc,blocks*sizeof(*dc)));
  CUDA_CHECK(cudaMalloc(&db,blocks*sizeof(*db)));
  CUDA_CHECK(cudaMalloc(&de,blocks*sizeof(*de)));
  CUDA_CHECK(cudaMalloc(&ds,blocks*sizeof(*ds)));
  cudaEvent_t begin{},end{}; CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
  benchmark<<<blocks,kThreads>>>(1,dc,db,de,ds); CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(begin));
  benchmark<<<blocks,kThreads>>>(iters,dc,db,de,ds); CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
  float elapsed_ms=0.0f; CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms,begin,end));
  auto* hc=new unsigned long long[blocks]; auto* hb=new unsigned long long[blocks];
  auto* he=new unsigned long long[blocks];
  auto* hs=new unsigned[blocks];
  CUDA_CHECK(cudaMemcpy(hc,dc,blocks*sizeof(*dc),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb,db,blocks*sizeof(*db),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(he,de,blocks*sizeof(*de),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hs,ds,blocks*sizeof(*ds),cudaMemcpyDeviceToHost));
  unsigned long long cycles=0,start_min=~0ull,stop_max=0;
  for(int i=0;i<blocks;++i){if(hc[i]>cycles) cycles=hc[i];
                            if(hb[i]<start_min) start_min=hb[i];
                            if(he[i]>stop_max) stop_max=he[i];}
  unsigned long long nanoseconds=stop_max-start_min;
  int unique_smids=0;
  for(int i=0;i<blocks;++i){bool first=true; for(int j=0;j<i;++j) if(hs[j]==hs[i]) first=false;
                            if(first) ++unique_smids;}
  double issued=double(blocks)*kWarpsPerBlock*iters*kWorkPerInstruction;
  double elapsed_seconds=double(nanoseconds)*1.0e-9;
  double rate=issued/elapsed_seconds;
  std::printf("case_id=%s\nprecision_id=%s\nwork_unit=%s\nsm_count=%d\n"
              "logical_input_bits=%d\ndescriptor_storage_bits=%d\nclock_hz=%.0f\n"
              "blocks=%d\nunique_smid_count=%d\nwarps_per_block=%d\niters=%d\n"
              "cycles=%llu\nglobaltimer_start_min=%llu\nglobaltimer_stop_max=%llu\n"
              "globaltimer_nanoseconds=%llu\nelapsed_seconds=%.9e\n"
              "host_kernel_elapsed_seconds=%.9e\nissued_work=%.0f\nrate_per_second=%.9e\n",
              kCaseId,kPrecision,kWorkUnit,p.multiProcessorCount,kLogicalBits,
              kDescriptorBits,freq,blocks,unique_smids,
              kWarpsPerBlock,iters,cycles,start_min,stop_max,nanoseconds,elapsed_seconds,
              double(elapsed_ms)*1.0e-3,issued,rate);
  CUDA_CHECK(cudaEventDestroy(begin)); CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaFree(dc)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(de));
  CUDA_CHECK(cudaFree(ds)); delete[] hc; delete[] hb; delete[] he; delete[] hs; return 0;
}
