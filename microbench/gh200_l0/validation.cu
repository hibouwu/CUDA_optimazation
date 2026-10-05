
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



__host__ __device__ float av(int r,int k){return (1+(r%3)+(k%2))/64.0f;}
__host__ __device__ float bv(int k,int n){return (1+(n%5)+(k%3))/64.0f;}
__device__ unsigned pack(float a,float b){return __half_as_ushort(__float2half_rn(a))|(unsigned(__half_as_ushort(__float2half_rn(b)))<<16);}

__global__ void validate_f32_c8_q16_t256_p0(int iterations, Result* r, double* out) {
  float d[8][1]={};
  for(int c=0;c<8;++c)d[c][0]=float((threadIdx.x%17)+c)/128.0f;
  
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(d[c][0]) : "f"((float)1.0), "f"((float)0.0078125)); }
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

__global__ void validate_mma_f16_c8_q16_t256_p0(int iterations, Result* r, double* out) {
  float d[8][4]={};
  int g=(threadIdx.x%32)/4,t=threadIdx.x%4;
  unsigned ar[4],br[2];
  for(int j=0;j<4;++j){int r=g+(j%2)*8,k=t*2+(j/2)*8;ar[j]=pack(av(r,k),av(r,k+1));}
  for(int j=0;j<2;++j){int k=t*2+j*8; br[j]=pack(bv(k,g),bv(k+1,g));}
  
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]) : "r"(ar[0]),"r"(ar[1]),"r"(ar[2]),"r"(ar[3]),"r"(br[0]),"r"(br[1])); }
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

__global__ void validate_wgmma_f16_c2_q16_t128_p7(int iterations, Result* r, double* out) {
  float d[2][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){
    int r=i/16,k=i%16;
    int idx=(r%8)*8+(r/8)*64+(k%8)+(k/8)*512;
    a[idx]=__half_as_ushort(__float2half_rn(av(r,k)));
    b[idx]=__half_as_ushort(__float2half_rn(bv(k,r)));
  }
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


template<class K> void check(K kernel,const char* name,int type,int threads,int chains,int regs,Result* dr,double* dout){
  for(int it:{1,3,17}){
    kernel<<<1,threads>>>(it,dr,dout);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    std::vector<double> out(threads*chains*regs);CK(cudaMemcpy(out.data(),dout,out.size()*sizeof(double),cudaMemcpyDeviceToHost));
    double err=0;
    for(int t=0;t<threads;++t)for(int c=0;c<chains;++c)for(int j=0;j<regs;++j){
      double expected=0;
      if(type==0)expected=((t%17)+c+it*16)/128.0;
      else {int row=(t%32)/4+((j%4)/2)*8+(type==2?(t/32)*16:0);
        int col=(t%4)*2+(j%2)+(type==2?(j/4)*8:0);
        for(int k=0;k<16;++k)expected+=double(av(row,k))*double(bv(k,col));
        expected*=it*16;}
      double got=out[(t*chains+c)*regs+j];if(!std::isfinite(got))exit(3);
      err=std::max(err,std::abs(got-expected));
    }
    printf("{\"case\":\"%s\",\"iterations\":%d,\"checked_outputs\":%zu,\"max_abs_error\":%.17g}\n",name,it,out.size(),err);
    if(err!=0)exit(4);
  }
}
int main(){Result* dr;double* dout;CK(cudaMalloc(&dr,sizeof(Result)));CK(cudaMalloc(&dout,65536*sizeof(double)));
  check(validate_f32_c8_q16_t256_p0,"f32_exact_counter",0,256,8,1,dr,dout);
  check(validate_mma_f16_c8_q16_t256_p0,"mma_nonuniform",1,256,8,4,dr,dout);
  check(validate_wgmma_f16_c2_q16_t128_p7,"wgmma_nonuniform",2,128,2,32,dr,dout);
  CK(cudaFree(dr));CK(cudaFree(dout));return 0;}
