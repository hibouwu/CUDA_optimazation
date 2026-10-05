
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);} }while(0)
struct Result { unsigned long long cycles, start_ns, stop_ns; unsigned smid; };
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

__global__ void f32_c8_q16_t32_p0(int iterations, Result* r, double* out) {
  float d[8][1]={};
  
  __shared__ volatile double drain[32];
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*1+lane]=double(d[c][j]);
}

__global__ void f32_c8_q16_t128_p0(int iterations, Result* r, double* out) {
  float d[8][1]={};
  
  __shared__ volatile double drain[128];
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*1+lane]=double(d[c][j]);
}

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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*1+lane]=double(d[c][j]);
}

__global__ void f64_c8_q16_t32_p0(int iterations, Result* r, double* out) {
  double d[8][1]={};
  
  __shared__ volatile double drain[32];
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*1+lane]=double(d[c][j]);
}

__global__ void f64_c8_q16_t128_p0(int iterations, Result* r, double* out) {
  double d[8][1]={};
  
  __shared__ volatile double drain[128];
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*1+lane]=double(d[c][j]);
}

__global__ void f16_c8_q16_t32_p0(int iterations, Result* r, double* out) {
  unsigned short d[8][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("fma.rn.f16 %0,%0,%1,%2;" : "+h"(d[c][0]) : "h"((unsigned short)14336u), "h"((unsigned short)13312u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*1+lane]=double(__half2float(__ushort_as_half(d[c][j])));
}

__global__ void f16_c8_q16_t128_p0(int iterations, Result* r, double* out) {
  unsigned short d[8][1]={};
  
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("fma.rn.f16 %0,%0,%1,%2;" : "+h"(d[c][0]) : "h"((unsigned short)14336u), "h"((unsigned short)13312u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*1+lane]=double(__half2float(__ushort_as_half(d[c][j])));
}

__global__ void f16_c8_q16_t256_p0(int iterations, Result* r, double* out) {
  unsigned short d[8][1]={};
  
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("fma.rn.f16 %0,%0,%1,%2;" : "+h"(d[c][0]) : "h"((unsigned short)14336u), "h"((unsigned short)13312u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*1+lane]=double(__half2float(__ushort_as_half(d[c][j])));
}

__global__ void f16x2_c8_q16_t32_p0(int iterations, Result* r, double* out) {
  unsigned d[8][1]={};
  
  __shared__ volatile double drain[32];
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<2;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*2+lane]=double(__half2float(__ushort_as_half(static_cast<unsigned short>(d[c][j] >> (16*lane)))));
}

__global__ void f16x2_c8_q16_t128_p0(int iterations, Result* r, double* out) {
  unsigned d[8][1]={};
  
  __shared__ volatile double drain[128];
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<2;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*2+lane]=double(__half2float(__ushort_as_half(static_cast<unsigned short>(d[c][j] >> (16*lane)))));
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<2;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*2+lane]=double(__half2float(__ushort_as_half(static_cast<unsigned short>(d[c][j] >> (16*lane)))));
}

__global__ void bf16_c8_q16_t32_p0(int iterations, Result* r, double* out) {
  unsigned short d[8][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("fma.rn.bf16 %0,%0,%1,%2;" : "+h"(d[c][0]) : "h"((unsigned short)16128u), "h"((unsigned short)16000u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*1+lane]=double(__bfloat162float(__ushort_as_bfloat16(d[c][j])));
}

__global__ void bf16_c8_q16_t128_p0(int iterations, Result* r, double* out) {
  unsigned short d[8][1]={};
  
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("fma.rn.bf16 %0,%0,%1,%2;" : "+h"(d[c][0]) : "h"((unsigned short)16128u), "h"((unsigned short)16000u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*1+lane]=double(__bfloat162float(__ushort_as_bfloat16(d[c][j])));
}

__global__ void bf16_c8_q16_t256_p0(int iterations, Result* r, double* out) {
  unsigned short d[8][1]={};
  
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("fma.rn.bf16 %0,%0,%1,%2;" : "+h"(d[c][0]) : "h"((unsigned short)16128u), "h"((unsigned short)16000u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*1+lane]=double(__bfloat162float(__ushort_as_bfloat16(d[c][j])));
}

__global__ void bf16x2_c8_q16_t32_p0(int iterations, Result* r, double* out) {
  unsigned d[8][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("fma.rn.bf16x2 %0,%0,%1,%2;" : "+r"(d[c][0]) : "r"((unsigned)1056980736u), "r"((unsigned)1048592000u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<2;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*2+lane]=double(__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(d[c][j] >> (16*lane)))));
}

__global__ void bf16x2_c8_q16_t128_p0(int iterations, Result* r, double* out) {
  unsigned d[8][1]={};
  
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("fma.rn.bf16x2 %0,%0,%1,%2;" : "+r"(d[c][0]) : "r"((unsigned)1056980736u), "r"((unsigned)1048592000u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<2;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*2+lane]=double(__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(d[c][j] >> (16*lane)))));
}

__global__ void bf16x2_c8_q16_t256_p0(int iterations, Result* r, double* out) {
  unsigned d[8][1]={};
  
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("fma.rn.bf16x2 %0,%0,%1,%2;" : "+r"(d[c][0]) : "r"((unsigned)1056980736u), "r"((unsigned)1048592000u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<2;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*1+j)*2+lane]=double(__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(d[c][j] >> (16*lane)))));
}

__global__ void mma_f16_c8_q16_t32_p0(int iterations, Result* r, double* out) {
  float d[8][4]={};
  
  __shared__ volatile double drain[32];
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*4+j)*1+lane]=double(d[c][j]);
}

__global__ void mma_f16_c8_q16_t128_p0(int iterations, Result* r, double* out) {
  float d[8][4]={};
  
  __shared__ volatile double drain[128];
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*4+j)*1+lane]=double(d[c][j]);
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*4+j)*1+lane]=double(d[c][j]);
}

__global__ void mma_bf16_c8_q16_t32_p0(int iterations, Result* r, double* out) {
  float d[8][4]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]) : "r"(0x3d803d80u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*4+j)*1+lane]=double(d[c][j]);
}

__global__ void mma_bf16_c8_q16_t128_p0(int iterations, Result* r, double* out) {
  float d[8][4]={};
  
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]) : "r"(0x3d803d80u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*4+j)*1+lane]=double(d[c][j]);
}

__global__ void mma_bf16_c8_q16_t256_p0(int iterations, Result* r, double* out) {
  float d[8][4]={};
  
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]) : "r"(0x3d803d80u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*4+j)*1+lane]=double(d[c][j]);
}

__global__ void mma_tf32_c8_q16_t32_p0(int iterations, Result* r, double* out) {
  float d[8][4]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]) : "r"(0x3d800000u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*4+j)*1+lane]=double(d[c][j]);
}

__global__ void mma_tf32_c8_q16_t128_p0(int iterations, Result* r, double* out) {
  float d[8][4]={};
  
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]) : "r"(0x3d800000u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*4+j)*1+lane]=double(d[c][j]);
}

__global__ void mma_tf32_c8_q16_t256_p0(int iterations, Result* r, double* out) {
  float d[8][4]={};
  
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]) : "r"(0x3d800000u)); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*4+j)*1+lane]=double(d[c][j]);
}

__global__ void mma_f64_c8_q16_t32_p0(int iterations, Result* r, double* out) {
  double d[8][2]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+d"(d[c][0]), "+d"(d[c][1]) : "d"(0.0625), "d"(0.0625)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<8;++c) {
    #pragma unroll
    for(int j=0;j<2;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  auto stop_ns=timer_ns();
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<2;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*2+j)*1+lane]=double(d[c][j]);
}

__global__ void mma_f64_c8_q16_t128_p0(int iterations, Result* r, double* out) {
  double d[8][2]={};
  
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+d"(d[c][0]), "+d"(d[c][1]) : "d"(0.0625), "d"(0.0625)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<8;++c) {
    #pragma unroll
    for(int j=0;j<2;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  auto stop_ns=timer_ns();
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<2;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*2+j)*1+lane]=double(d[c][j]);
}

__global__ void mma_f64_c8_q16_t256_p0(int iterations, Result* r, double* out) {
  double d[8][2]={};
  
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<8;++c) { asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+d"(d[c][0]), "+d"(d[c][1]) : "d"(0.0625), "d"(0.0625)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<8;++c) {
    #pragma unroll
    for(int j=0;j<2;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  auto stop_ns=timer_ns();
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<8;++c) for(int j=0;j<2;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*8+c)*2+j)*1+lane]=double(d[c][j]);
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*2+c)*32+j)*1+lane]=double(d[c][j]);
}

__global__ void wgmma_f16_c2_q16_t256_p0(int iterations, Result* r, double* out) {
  float d[2][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x2c00;b[i]=0x2c00;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[256];
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*2+c)*32+j)*1+lane]=double(d[c][j]);
}

__global__ void wgmma_f16_c2_q16_t256_p3(int iterations, Result* r, double* out) {
  float d[2][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x2c00;b[i]=0x2c00;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[256];
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*2+c)*32+j)*1+lane]=double(d[c][j]);
}

__global__ void wgmma_f16_c2_q16_t256_p7(int iterations, Result* r, double* out) {
  float d[2][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x2c00;b[i]=0x2c00;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[256];
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*2+c)*32+j)*1+lane]=double(d[c][j]);
}

__global__ void wgmma_bf16_c2_q16_t128_p0(int iterations, Result* r, double* out) {
  float d[2][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x3d80;b[i]=0x3d80;}
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
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*2+c)*32+j)*1+lane]=double(d[c][j]);
}

__global__ void wgmma_bf16_c2_q16_t128_p3(int iterations, Result* r, double* out) {
  float d[2][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x3d80;b[i]=0x3d80;}
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
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*2+c)*32+j)*1+lane]=double(d[c][j]);
}

__global__ void wgmma_bf16_c2_q16_t128_p7(int iterations, Result* r, double* out) {
  float d[2][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x3d80;b[i]=0x3d80;}
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
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*2+c)*32+j)*1+lane]=double(d[c][j]);
}

__global__ void wgmma_bf16_c2_q16_t256_p0(int iterations, Result* r, double* out) {
  float d[2][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x3d80;b[i]=0x3d80;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*2+c)*32+j)*1+lane]=double(d[c][j]);
}

__global__ void wgmma_bf16_c2_q16_t256_p3(int iterations, Result* r, double* out) {
  float d[2][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x3d80;b[i]=0x3d80;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*2+c)*32+j)*1+lane]=double(d[c][j]);
}

__global__ void wgmma_bf16_c2_q16_t256_p7(int iterations, Result* r, double* out) {
  float d[2][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x3d80;b[i]=0x3d80;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[256];
  __syncthreads();
  auto start_ns=timer_ns();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
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
  if(threadIdx.x==0) {r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns; asm volatile("mov.u32 %0, %%smid;":"=r"(r[blockIdx.x].smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[(((blockIdx.x*blockDim.x+threadIdx.x)*2+c)*32+j)*1+lane]=double(d[c][j]);
}


long long wall_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
template<class K> void run_stress(K kernel,const char* name,const char* kind,int threads,int per_block_outputs,int chains,int batch,long long work,double increment,int scope,int sms) {
  cudaFuncAttributes attr{};CK(cudaFuncGetAttributes(&attr,kernel));
  if(attr.localSizeBytes){fprintf(stderr,"unexpected local memory: %s\n",name);exit(5);}
  int occupancy=0;CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,threads,0));
  if(occupancy<1)exit(6);
  int blocks=scope?sms*std::min(4,occupancy):1;
  size_t output_count=size_t(per_block_outputs)*blocks;
  Result* dr;double* dout;CK(cudaMalloc(&dr,blocks*sizeof(Result)));CK(cudaMalloc(&dout,output_count*sizeof(double)));
  std::vector<Result> results(blocks);std::vector<double> outputs(output_count);
  cudaEvent_t start,stop;CK(cudaEventCreate(&start));CK(cudaEventCreate(&stop));
  auto launch=[&](int iterations,int rep){
    long long before=wall_ns();
    CK(cudaEventRecord(start));kernel<<<blocks,threads>>>(iterations,dr,dout);
    CK(cudaGetLastError());CK(cudaEventRecord(stop));CK(cudaEventSynchronize(stop));
    long long after=wall_ns();
    float ms=0;CK(cudaEventElapsedTime(&ms,start,stop));
    CK(cudaMemcpy(results.data(),dr,blocks*sizeof(Result),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(outputs.data(),dout,output_count*sizeof(double),cudaMemcpyDeviceToHost));
    double expected=increment?iterations*increment:0.5;
    double error=0;for(double x:outputs){if(!std::isfinite(x)){fprintf(stderr,"nonfinite %s\n",name);exit(3);}error=std::max(error,std::abs(x-expected));}
    if(error>1e-6){fprintf(stderr,"mismatch %s iterations=%d expected=%g error=%g\n",name,iterations,expected,error);exit(4);}
    unsigned long long first=~0ull,last=0,max_cycles=0;std::set<unsigned> smids;
    for(auto r:results){first=std::min(first,r.start_ns);last=std::max(last,r.stop_ns);max_cycles=std::max(max_cycles,r.cycles);smids.insert(r.smid);}
    if(scope && int(smids.size())!=sms){fprintf(stderr,"incomplete SM coverage %s got=%zu expected=%d\n",name,smids.size(),sms);exit(7);}
    long long flop=(long long)blocks*iterations*batch*chains*work;
    if(rep>=0){
      printf("{\"case\":\"%s\",\"kind\":\"%s\",\"scope\":\"%s\",\"repeat\":%d,\"iterations\":%d,\"blocks\":%d,\"threads\":%d,\"unique_sms\":%zu,\"max_block_cycles\":%llu,\"start_ns\":%llu,\"stop_ns\":%llu,\"event_ms\":%.9g,\"work_flop\":%lld,\"tflops\":%.9g,\"max_abs_error\":%.9g,\"registers_per_thread\":%d,\"static_smem_bytes\":%zu,\"occupancy_limit_ctas_per_sm\":%d,\"host_start_unix_ns\":%lld,\"host_stop_unix_ns\":%lld,\"blocks_detail\":[",name,kind,scope?"full_gpu":"single_cta",rep,iterations,blocks,threads,smids.size(),max_cycles,first,last,ms,flop,double(flop)/(last-first)/1000.0,error,attr.numRegs,attr.sharedSizeBytes,occupancy,before,after);
      for(int i=0;i<blocks;++i){auto r=results[i];printf("%s{\"smid\":%u,\"cycles\":%llu,\"start_ns\":%llu,\"stop_ns\":%llu}",i?",":"",r.smid,r.cycles,r.start_ns,r.stop_ns);}
      printf("]}\n");fflush(stdout);
    }
    return ms;
  };
  launch(4096,-1);
  double pilot_ms=launch(8192,-1);
  // About 100 ms per measured kernel; bound even elementwise K accumulation to 2^24 additions.
  int max_iterations=increment?65536:1048576;
  int iterations=std::clamp(int(8192*100.0/std::max(pilot_ms,0.01)),8192,max_iterations);
  launch(iterations,-1);
  for(int rep=0;rep<5;++rep)launch(iterations,rep);
  CK(cudaEventDestroy(start));CK(cudaEventDestroy(stop));CK(cudaFree(dr));CK(cudaFree(dout));
}
int main(){
  cudaDeviceProp p{};CK(cudaGetDeviceProperties(&p,0));if(p.major!=9)return 1;
  printf("{\"device\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"global_memory_bytes\":%zu,\"campaign\":\"sustained_issue\"}\n",p.name,p.major,p.minor,p.multiProcessorCount,p.totalGlobalMem);
  std::vector<std::function<void()>> jobs;

jobs.push_back([=](){run_stress(f32_c8_q16_t32_p0,"f32_c8_q16_t32_p0","f32",32,256,8,16,64,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f32_c8_q16_t32_p0,"f32_c8_q16_t32_p0","f32",32,256,8,16,64,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f32_c8_q16_t128_p0,"f32_c8_q16_t128_p0","f32",128,1024,8,16,256,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f32_c8_q16_t128_p0,"f32_c8_q16_t128_p0","f32",128,1024,8,16,256,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f32_c8_q16_t256_p0,"f32_c8_q16_t256_p0","f32",256,2048,8,16,512,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f32_c8_q16_t256_p0,"f32_c8_q16_t256_p0","f32",256,2048,8,16,512,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f64_c8_q16_t32_p0,"f64_c8_q16_t32_p0","f64",32,256,8,16,64,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f64_c8_q16_t32_p0,"f64_c8_q16_t32_p0","f64",32,256,8,16,64,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f64_c8_q16_t128_p0,"f64_c8_q16_t128_p0","f64",128,1024,8,16,256,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f64_c8_q16_t128_p0,"f64_c8_q16_t128_p0","f64",128,1024,8,16,256,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f64_c8_q16_t256_p0,"f64_c8_q16_t256_p0","f64",256,2048,8,16,512,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f64_c8_q16_t256_p0,"f64_c8_q16_t256_p0","f64",256,2048,8,16,512,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f16_c8_q16_t32_p0,"f16_c8_q16_t32_p0","f16",32,256,8,16,64,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f16_c8_q16_t32_p0,"f16_c8_q16_t32_p0","f16",32,256,8,16,64,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f16_c8_q16_t128_p0,"f16_c8_q16_t128_p0","f16",128,1024,8,16,256,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f16_c8_q16_t128_p0,"f16_c8_q16_t128_p0","f16",128,1024,8,16,256,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f16_c8_q16_t256_p0,"f16_c8_q16_t256_p0","f16",256,2048,8,16,512,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f16_c8_q16_t256_p0,"f16_c8_q16_t256_p0","f16",256,2048,8,16,512,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f16x2_c8_q16_t32_p0,"f16x2_c8_q16_t32_p0","f16x2",32,512,8,16,128,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f16x2_c8_q16_t32_p0,"f16x2_c8_q16_t32_p0","f16x2",32,512,8,16,128,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f16x2_c8_q16_t128_p0,"f16x2_c8_q16_t128_p0","f16x2",128,2048,8,16,512,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f16x2_c8_q16_t128_p0,"f16x2_c8_q16_t128_p0","f16x2",128,2048,8,16,512,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f16x2_c8_q16_t256_p0,"f16x2_c8_q16_t256_p0","f16x2",256,4096,8,16,1024,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(f16x2_c8_q16_t256_p0,"f16x2_c8_q16_t256_p0","f16x2",256,4096,8,16,1024,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(bf16_c8_q16_t32_p0,"bf16_c8_q16_t32_p0","bf16",32,256,8,16,64,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(bf16_c8_q16_t32_p0,"bf16_c8_q16_t32_p0","bf16",32,256,8,16,64,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(bf16_c8_q16_t128_p0,"bf16_c8_q16_t128_p0","bf16",128,1024,8,16,256,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(bf16_c8_q16_t128_p0,"bf16_c8_q16_t128_p0","bf16",128,1024,8,16,256,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(bf16_c8_q16_t256_p0,"bf16_c8_q16_t256_p0","bf16",256,2048,8,16,512,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(bf16_c8_q16_t256_p0,"bf16_c8_q16_t256_p0","bf16",256,2048,8,16,512,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(bf16x2_c8_q16_t32_p0,"bf16x2_c8_q16_t32_p0","bf16x2",32,512,8,16,128,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(bf16x2_c8_q16_t32_p0,"bf16x2_c8_q16_t32_p0","bf16x2",32,512,8,16,128,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(bf16x2_c8_q16_t128_p0,"bf16x2_c8_q16_t128_p0","bf16x2",128,2048,8,16,512,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(bf16x2_c8_q16_t128_p0,"bf16x2_c8_q16_t128_p0","bf16x2",128,2048,8,16,512,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(bf16x2_c8_q16_t256_p0,"bf16x2_c8_q16_t256_p0","bf16x2",256,4096,8,16,1024,0,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(bf16x2_c8_q16_t256_p0,"bf16x2_c8_q16_t256_p0","bf16x2",256,4096,8,16,1024,0,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_f16_c8_q16_t32_p0,"mma_f16_c8_q16_t32_p0","mma_f16",32,1024,8,16,4096,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_f16_c8_q16_t32_p0,"mma_f16_c8_q16_t32_p0","mma_f16",32,1024,8,16,4096,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_f16_c8_q16_t128_p0,"mma_f16_c8_q16_t128_p0","mma_f16",128,4096,8,16,16384,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_f16_c8_q16_t128_p0,"mma_f16_c8_q16_t128_p0","mma_f16",128,4096,8,16,16384,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_f16_c8_q16_t256_p0,"mma_f16_c8_q16_t256_p0","mma_f16",256,8192,8,16,32768,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_f16_c8_q16_t256_p0,"mma_f16_c8_q16_t256_p0","mma_f16",256,8192,8,16,32768,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_bf16_c8_q16_t32_p0,"mma_bf16_c8_q16_t32_p0","mma_bf16",32,1024,8,16,4096,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_bf16_c8_q16_t32_p0,"mma_bf16_c8_q16_t32_p0","mma_bf16",32,1024,8,16,4096,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_bf16_c8_q16_t128_p0,"mma_bf16_c8_q16_t128_p0","mma_bf16",128,4096,8,16,16384,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_bf16_c8_q16_t128_p0,"mma_bf16_c8_q16_t128_p0","mma_bf16",128,4096,8,16,16384,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_bf16_c8_q16_t256_p0,"mma_bf16_c8_q16_t256_p0","mma_bf16",256,8192,8,16,32768,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_bf16_c8_q16_t256_p0,"mma_bf16_c8_q16_t256_p0","mma_bf16",256,8192,8,16,32768,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_tf32_c8_q16_t32_p0,"mma_tf32_c8_q16_t32_p0","mma_tf32",32,1024,8,16,2048,0.5,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_tf32_c8_q16_t32_p0,"mma_tf32_c8_q16_t32_p0","mma_tf32",32,1024,8,16,2048,0.5,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_tf32_c8_q16_t128_p0,"mma_tf32_c8_q16_t128_p0","mma_tf32",128,4096,8,16,8192,0.5,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_tf32_c8_q16_t128_p0,"mma_tf32_c8_q16_t128_p0","mma_tf32",128,4096,8,16,8192,0.5,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_tf32_c8_q16_t256_p0,"mma_tf32_c8_q16_t256_p0","mma_tf32",256,8192,8,16,16384,0.5,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_tf32_c8_q16_t256_p0,"mma_tf32_c8_q16_t256_p0","mma_tf32",256,8192,8,16,16384,0.5,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_f64_c8_q16_t32_p0,"mma_f64_c8_q16_t32_p0","mma_f64",32,512,8,16,512,0.25,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_f64_c8_q16_t32_p0,"mma_f64_c8_q16_t32_p0","mma_f64",32,512,8,16,512,0.25,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_f64_c8_q16_t128_p0,"mma_f64_c8_q16_t128_p0","mma_f64",128,2048,8,16,2048,0.25,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_f64_c8_q16_t128_p0,"mma_f64_c8_q16_t128_p0","mma_f64",128,2048,8,16,2048,0.25,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_f64_c8_q16_t256_p0,"mma_f64_c8_q16_t256_p0","mma_f64",256,4096,8,16,4096,0.25,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(mma_f64_c8_q16_t256_p0,"mma_f64_c8_q16_t256_p0","mma_f64",256,4096,8,16,4096,0.25,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_f16_c2_q16_t128_p0,"wgmma_f16_c2_q16_t128_p0","wgmma_f16",128,8192,2,16,131072,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_f16_c2_q16_t128_p0,"wgmma_f16_c2_q16_t128_p0","wgmma_f16",128,8192,2,16,131072,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_f16_c2_q16_t128_p3,"wgmma_f16_c2_q16_t128_p3","wgmma_f16",128,8192,2,16,131072,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_f16_c2_q16_t128_p3,"wgmma_f16_c2_q16_t128_p3","wgmma_f16",128,8192,2,16,131072,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_f16_c2_q16_t128_p7,"wgmma_f16_c2_q16_t128_p7","wgmma_f16",128,8192,2,16,131072,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_f16_c2_q16_t128_p7,"wgmma_f16_c2_q16_t128_p7","wgmma_f16",128,8192,2,16,131072,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_f16_c2_q16_t256_p0,"wgmma_f16_c2_q16_t256_p0","wgmma_f16",256,16384,2,16,262144,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_f16_c2_q16_t256_p0,"wgmma_f16_c2_q16_t256_p0","wgmma_f16",256,16384,2,16,262144,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_f16_c2_q16_t256_p3,"wgmma_f16_c2_q16_t256_p3","wgmma_f16",256,16384,2,16,262144,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_f16_c2_q16_t256_p3,"wgmma_f16_c2_q16_t256_p3","wgmma_f16",256,16384,2,16,262144,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_f16_c2_q16_t256_p7,"wgmma_f16_c2_q16_t256_p7","wgmma_f16",256,16384,2,16,262144,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_f16_c2_q16_t256_p7,"wgmma_f16_c2_q16_t256_p7","wgmma_f16",256,16384,2,16,262144,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_bf16_c2_q16_t128_p0,"wgmma_bf16_c2_q16_t128_p0","wgmma_bf16",128,8192,2,16,131072,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_bf16_c2_q16_t128_p0,"wgmma_bf16_c2_q16_t128_p0","wgmma_bf16",128,8192,2,16,131072,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_bf16_c2_q16_t128_p3,"wgmma_bf16_c2_q16_t128_p3","wgmma_bf16",128,8192,2,16,131072,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_bf16_c2_q16_t128_p3,"wgmma_bf16_c2_q16_t128_p3","wgmma_bf16",128,8192,2,16,131072,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_bf16_c2_q16_t128_p7,"wgmma_bf16_c2_q16_t128_p7","wgmma_bf16",128,8192,2,16,131072,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_bf16_c2_q16_t128_p7,"wgmma_bf16_c2_q16_t128_p7","wgmma_bf16",128,8192,2,16,131072,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_bf16_c2_q16_t256_p0,"wgmma_bf16_c2_q16_t256_p0","wgmma_bf16",256,16384,2,16,262144,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_bf16_c2_q16_t256_p0,"wgmma_bf16_c2_q16_t256_p0","wgmma_bf16",256,16384,2,16,262144,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_bf16_c2_q16_t256_p3,"wgmma_bf16_c2_q16_t256_p3","wgmma_bf16",256,16384,2,16,262144,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_bf16_c2_q16_t256_p3,"wgmma_bf16_c2_q16_t256_p3","wgmma_bf16",256,16384,2,16,262144,1,1,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_bf16_c2_q16_t256_p7,"wgmma_bf16_c2_q16_t256_p7","wgmma_bf16",256,16384,2,16,262144,1,0,p.multiProcessorCount);});
jobs.push_back([=](){run_stress(wgmma_bf16_c2_q16_t256_p7,"wgmma_bf16_c2_q16_t256_p7","wgmma_bf16",256,16384,2,16,262144,1,1,p.multiProcessorCount);});
std::mt19937 rng(20260930);std::shuffle(jobs.begin(),jobs.end(),rng);for(auto &job:jobs)job();return 0;}