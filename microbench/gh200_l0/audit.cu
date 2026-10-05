
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);} }while(0)
struct Result { unsigned long long cycles, start_ns, stop_ns, start_cycle, stop_cycle; unsigned smid; };
__device__ unsigned long long descriptor(void* p) {
  unsigned a=static_cast<unsigned>(__cvta_generic_to_shared(p));
  // K-major interleaved ((8,8),(8,2)):((8,64),(1,512)), in 16-bit elements.
  // Adjacent K core matrix: 1024 bytes. Adjacent eight rows: 128 bytes.
  return ((a & 0x3ffffu)>>4) | (64ull<<16) | (8ull<<32);
}



#include <set>
#include <chrono>
#include <random>
#include <functional>
__device__ unsigned long long timer_ns(){unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}

#include <map>
#include <string>
#include <cstring>

__global__ void f32_c8_q16_t256_p0(int iterations, Result* r, double* out) {
  float d[8][1]={};
  
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(d[c][0]) : "f"((float)0.5), "f"((float)0.25)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<8;++c) {
    #pragma unroll
    for(int j=0;j<1;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  auto stop_ns=timer_ns();
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_cycle=start; r[blockIdx.x].stop_cycle=stop; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*1+lane]=double(d[c][j]);
}

__global__ void f64_c8_q16_t256_p0(int iterations, Result* r, double* out) {
  double d[8][1]={};
  
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(d[c][0]) : "d"((double)0.5), "d"((double)0.25)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<8;++c) {
    #pragma unroll
    for(int j=0;j<1;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  auto stop_ns=timer_ns();
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_cycle=start; r[blockIdx.x].stop_cycle=stop; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*1+lane]=double(d[c][j]);
}

__global__ void f16x2_c8_q16_t256_p0(int iterations, Result* r, double* out) {
  unsigned d[8][1]={};
  
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("fma.rn.f16x2 %0,%0,%1,%2;" : "+r"(d[c][0]) : "r"((unsigned)939538432u), "r"((unsigned)872428544u)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<8;++c) {
    #pragma unroll
    for(int j=0;j<1;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  auto stop_ns=timer_ns();
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_cycle=start; r[blockIdx.x].stop_cycle=stop; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<2;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*2+lane]=double(__half2float(__ushort_as_half(static_cast<unsigned short>(d[c][j] >> (16*lane)))));
}

__global__ void mma_f16_c8_q16_t256_p0(int iterations, Result* r, double* out) {
  float d[8][4]={};
  
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]) : "r"(0x2c002c00u)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<8;++c) {
    #pragma unroll
    for(int j=0;j<4;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  auto stop_ns=timer_ns();
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_cycle=start; r[blockIdx.x].stop_cycle=stop; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*4+j)*1+lane]=double(d[c][j]);
}

__global__ void wgmma_f16_c2_q16_t128_p0(int iterations, Result* r, double* out) {
  float d[2][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x2c00;b[i]=0x2c00;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  double sum=0;
  #pragma unroll
  for(int c=0;c<2;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  auto stop_ns=timer_ns();
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_cycle=start; r[blockIdx.x].stop_cycle=stop; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*2+c)*32+j)*1+lane]=double(d[c][j]);
}

__global__ void wgmma_f16_c2_q16_t128_p3(int iterations, Result* r, double* out) {
  float d[2][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x2c00;b[i]=0x2c00;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 3;" ::: "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  double sum=0;
  #pragma unroll
  for(int c=0;c<2;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  auto stop_ns=timer_ns();
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_cycle=start; r[blockIdx.x].stop_cycle=stop; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*2+c)*32+j)*1+lane]=double(d[c][j]);
}

__global__ void wgmma_f16_c2_q16_t128_p7(int iterations, Result* r, double* out) {
  float d[2][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x2c00;b[i]=0x2c00;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 7;" ::: "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  double sum=0;
  #pragma unroll
  for(int c=0;c<2;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  auto stop_ns=timer_ns();
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_cycle=start; r[blockIdx.x].stop_cycle=stop; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*2+c)*32+j)*1+lane]=double(d[c][j]);
}


long long wall_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
struct Probe {
  std::string name,kind; int threads,outputs,chains,batch;long long work;double increment;
  void(*launch)(int,int,Result*,double*);cudaFuncAttributes attr;int occupancy;
};
template<class K> Probe make_probe(K kernel,std::string name,std::string kind,int threads,int outputs,int chains,int batch,long long work,double increment,void(*launch)(int,int,Result*,double*)){
  Probe p{name,kind,threads,outputs,chains,batch,work,increment,launch,{},0};
  CK(cudaFuncGetAttributes(&p.attr,kernel));if(p.attr.localSizeBytes){fprintf(stderr,"spill %s\n",name.c_str());exit(5);}
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&p.occupancy,kernel,threads,0));return p;
}
struct Buffers {
  Result* dr; double* dout;std::vector<Result> rs;std::vector<double> out;cudaEvent_t begin,end;
  Buffers(int blocks,size_t count):rs(blocks),out(count){CK(cudaMalloc(&dr,blocks*sizeof(Result)));CK(cudaMalloc(&dout,count*sizeof(double)));CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));}
  ~Buffers(){cudaFree(dr);cudaFree(dout);cudaEventDestroy(begin);cudaEventDestroy(end);}
};
// Excludes copies and host checking. Empty iterations measures the same fixed timer/drain envelope.
double execute(Probe const& p,Buffers& b,int scope,int sms,int iterations,int round,int order,const char* phase,bool emit=true){
  int blocks=scope?sms*std::min(4,p.occupancy):1;
  long long before=wall_ns();
  CK(cudaEventRecord(b.begin));p.launch(blocks,iterations,b.dr,b.dout);
  CK(cudaGetLastError());CK(cudaEventRecord(b.end));CK(cudaEventSynchronize(b.end));
  long long after=wall_ns();float ms=0;CK(cudaEventElapsedTime(&ms,b.begin,b.end));
  CK(cudaMemcpy(b.rs.data(),b.dr,blocks*sizeof(Result),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(b.out.data(),b.dout,size_t(blocks)*p.outputs*sizeof(double),cudaMemcpyDeviceToHost));
  double expected=iterations?(p.increment?iterations*p.increment:0.5):0;
  double error=0;for(size_t i=0;i<size_t(blocks)*p.outputs;++i){double x=b.out[i];if(!std::isfinite(x))exit(3);error=std::max(error,std::abs(x-expected));}
  if(error>1e-6){fprintf(stderr,"mismatch %s %d %.17g\n",p.name.c_str(),iterations,error);exit(4);}
  unsigned long long first=~0ull,last=0,max_cycles=0;std::map<unsigned,std::pair<unsigned long long,unsigned long long>> spans;
  for(int i=0;i<blocks;++i){auto r=b.rs[i];first=std::min(first,r.start_ns);last=std::max(last,r.stop_ns);max_cycles=std::max(max_cycles,r.cycles);
    if(!spans.count(r.smid))spans[r.smid]={r.start_cycle,r.stop_cycle};
    else{auto &s=spans[r.smid];s.first=std::min(s.first,r.start_cycle);s.second=std::max(s.second,r.stop_cycle);}}
  if(scope && iterations && int(spans.size())!=sms){fprintf(stderr,"SM coverage failure\n");exit(7);}
  unsigned long long sm_cycles=0;for(auto &s:spans)sm_cycles+=s.second.second-s.second.first;
  long long work=(long long)blocks*iterations*p.batch*p.chains*p.work;
  if(emit){
    printf("{\"case\":\"%s\",\"kind\":\"%s\",\"scope\":\"%s\",\"phase\":\"%s\",\"round\":%d,\"order\":%d,\"iterations\":%d,\"blocks\":%d,\"threads\":%d,\"unique_sms\":%zu,\"max_block_cycles\":%llu,\"summed_sm_span_cycles\":%llu,\"start_ns\":%llu,\"stop_ns\":%llu,\"event_ms\":%.9g,\"work_flop\":%lld,\"flop_per_sm_cycle\":%.12g,\"max_abs_error\":%.9g,\"registers_per_thread\":%d,\"occupancy_limit_ctas_per_sm\":%d,\"host_start_unix_ns\":%lld,\"host_stop_unix_ns\":%lld,\"blocks_detail\":[",p.name.c_str(),p.kind.c_str(),scope?"full_gpu":"single_cta",phase,round,order,iterations,blocks,p.threads,spans.size(),max_cycles,sm_cycles,first,last,ms,work,sm_cycles?double(work)/sm_cycles:0,error,p.attr.numRegs,p.occupancy,before,after);
    for(int i=0;i<blocks;++i){auto r=b.rs[i];printf("%s{\"smid\":%u,\"cycles\":%llu,\"start_cycle\":%llu,\"stop_cycle\":%llu,\"start_ns\":%llu,\"stop_ns\":%llu}",i?",":"",r.smid,r.cycles,r.start_cycle,r.stop_cycle,r.start_ns,r.stop_ns);}
    printf("]}\n");fflush(stdout);
  }
  return ms;
}
int main(int argc,char** argv){
  cudaDeviceProp dev{};CK(cudaGetDeviceProperties(&dev,0));if(dev.major!=9)return 1;
  printf("{\"device\":\"%s\",\"sms\":%d,\"campaign\":\"audit_retest\",\"seed\":20260930,\"rounds\":12}\n",dev.name,dev.multiProcessorCount);
  std::vector<Probe> probes;

probes.push_back(make_probe(f32_c8_q16_t256_p0,"f32_c8_q16_t256_p0","f32",256,2048,8,16,512,0,
        [](int blocks,int it,Result* r,double* out){f32_c8_q16_t256_p0<<<blocks,256>>>(it,r,out);}));
probes.push_back(make_probe(f64_c8_q16_t256_p0,"f64_c8_q16_t256_p0","f64",256,2048,8,16,512,0,
        [](int blocks,int it,Result* r,double* out){f64_c8_q16_t256_p0<<<blocks,256>>>(it,r,out);}));
probes.push_back(make_probe(f16x2_c8_q16_t256_p0,"f16x2_c8_q16_t256_p0","f16x2",256,4096,8,16,1024,0,
        [](int blocks,int it,Result* r,double* out){f16x2_c8_q16_t256_p0<<<blocks,256>>>(it,r,out);}));
probes.push_back(make_probe(mma_f16_c8_q16_t256_p0,"mma_f16_c8_q16_t256_p0","mma_f16",256,8192,8,16,32768,1,
        [](int blocks,int it,Result* r,double* out){mma_f16_c8_q16_t256_p0<<<blocks,256>>>(it,r,out);}));
probes.push_back(make_probe(wgmma_f16_c2_q16_t128_p0,"wgmma_f16_c2_q16_t128_p0","wgmma_f16",128,8192,2,16,131072,1,
        [](int blocks,int it,Result* r,double* out){wgmma_f16_c2_q16_t128_p0<<<blocks,128>>>(it,r,out);}));
probes.push_back(make_probe(wgmma_f16_c2_q16_t128_p3,"wgmma_f16_c2_q16_t128_p3","wgmma_f16",128,8192,2,16,131072,1,
        [](int blocks,int it,Result* r,double* out){wgmma_f16_c2_q16_t128_p3<<<blocks,128>>>(it,r,out);}));
probes.push_back(make_probe(wgmma_f16_c2_q16_t128_p7,"wgmma_f16_c2_q16_t128_p7","wgmma_f16",128,8192,2,16,131072,1,
        [](int blocks,int it,Result* r,double* out){wgmma_f16_c2_q16_t128_p7<<<blocks,128>>>(it,r,out);}));

  Buffers b(dev.multiProcessorCount*4,size_t(dev.multiProcessorCount)*4*65536);
  if(argc>1){int idx=atoi(argv[1]);if(idx<0||idx>=int(probes.size()))return 1;
    execute(probes[idx],b,0,dev.multiProcessorCount,64,0,0,"profile");return 0;}
  int order=0;
  // Warmup samples are retained. Stop when last five full-GPU durations have CV <= 1%, after >=8 launches; cap 30.
  std::vector<bool> ready(probes.size(),false);
  for(size_t j=0;j<probes.size();++j){std::vector<double> samples;
    for(int w=0;w<30;++w){samples.push_back(execute(probes[j],b,1,dev.multiProcessorCount,65536,w,order++,"warmup"));
      if(w>=7){double mean=0,var=0;for(int k=w-4;k<=w;++k)mean+=samples[k]/5;
        for(int k=w-4;k<=w;++k)var+=(samples[k]-mean)*(samples[k]-mean)/4;
        if(std::sqrt(var)/mean<=0.01){ready[j]=true;break;}}}
    fprintf(stderr,"warmup case=%s stable=%d launches=%zu\n",probes[j].name.c_str(),ready[j]?1:0,samples.size());}
  struct Job{int probe,scope,iterations;};
  std::vector<Job> jobs;
  for(int j=0;j<int(probes.size());++j){jobs.push_back({j,0,0});for(int scope:{0,1})for(int it:{8192,32768,65536})jobs.push_back({j,scope,it});}
  std::mt19937 rng(20260930);
  for(int round=0;round<12;++round){std::shuffle(jobs.begin(),jobs.end(),rng);
    for(auto job:jobs){
      if(job.iterations)execute(probes[job.probe],b,job.scope,dev.multiProcessorCount,4096,round,order++,"transition_warmup");
      execute(probes[job.probe],b,job.scope,dev.multiProcessorCount,job.iterations,round,order++,job.iterations?"measure":"empty_control");}}
  return 0;
}
