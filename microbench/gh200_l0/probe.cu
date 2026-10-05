
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);} }while(0)
struct Result { unsigned long long cycles; unsigned smid; };
__device__ unsigned long long descriptor(void* p) {
  unsigned a=static_cast<unsigned>(__cvta_generic_to_shared(p));
  // K-major interleaved ((8,8),(8,2)):((8,64),(1,512)), in 16-bit elements.
  // Adjacent K core matrix: 1024 bytes. Adjacent eight rows: 128 bytes.
  return ((a & 0x3ffffu)>>4) | (64ull<<16) | (8ull<<32);
}


__global__ void f32_c1_q16(int iterations, Result* r, double* out) {
  float d[1][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(d[c][0]) : "f"((float)0.5), "f"((float)0.25)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<1;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*1+j)*1+lane]=double(d[c][j]);
}


__global__ void f32_c8_q16(int iterations, Result* r, double* out) {
  float d[8][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
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
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*8+c)*1+j)*1+lane]=double(d[c][j]);
}


__global__ void f64_c1_q16(int iterations, Result* r, double* out) {
  double d[1][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(d[c][0]) : "d"((double)0.5), "d"((double)0.25)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<1;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*1+j)*1+lane]=double(d[c][j]);
}


__global__ void f64_c8_q16(int iterations, Result* r, double* out) {
  double d[8][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
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
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*8+c)*1+j)*1+lane]=double(d[c][j]);
}


__global__ void f16_c1_q16(int iterations, Result* r, double* out) {
  unsigned short d[1][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("fma.rn.f16 %0,%0,%1,%2;" : "+h"(d[c][0]) : "h"((unsigned short)14336u), "h"((unsigned short)13312u)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<1;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*1+j)*1+lane]=double(__half2float(__ushort_as_half(d[c][j])));
}


__global__ void f16_c8_q16(int iterations, Result* r, double* out) {
  unsigned short d[8][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
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
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*8+c)*1+j)*1+lane]=double(__half2float(__ushort_as_half(d[c][j])));
}


__global__ void f16x2_c1_q16(int iterations, Result* r, double* out) {
  unsigned d[1][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("fma.rn.f16x2 %0,%0,%1,%2;" : "+r"(d[c][0]) : "r"((unsigned)939538432u), "r"((unsigned)872428544u)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<1;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<1;++j) for(int lane=0;lane<2;++lane)
    out[((threadIdx.x*1+c)*1+j)*2+lane]=double(__half2float(__ushort_as_half(static_cast<unsigned short>(d[c][j] >> (16*lane)))));
}


__global__ void f16x2_c8_q16(int iterations, Result* r, double* out) {
  unsigned d[8][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
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
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<2;++lane)
    out[((threadIdx.x*8+c)*1+j)*2+lane]=double(__half2float(__ushort_as_half(static_cast<unsigned short>(d[c][j] >> (16*lane)))));
}


__global__ void bf16_c1_q16(int iterations, Result* r, double* out) {
  unsigned short d[1][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("fma.rn.bf16 %0,%0,%1,%2;" : "+h"(d[c][0]) : "h"((unsigned short)16128u), "h"((unsigned short)16000u)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<1;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*1+j)*1+lane]=double(__bfloat162float(__ushort_as_bfloat16(d[c][j])));
}


__global__ void bf16_c8_q16(int iterations, Result* r, double* out) {
  unsigned short d[8][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
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
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*8+c)*1+j)*1+lane]=double(__bfloat162float(__ushort_as_bfloat16(d[c][j])));
}


__global__ void bf16x2_c1_q16(int iterations, Result* r, double* out) {
  unsigned d[1][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("fma.rn.bf16x2 %0,%0,%1,%2;" : "+r"(d[c][0]) : "r"((unsigned)1056980736u), "r"((unsigned)1048592000u)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<1;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<1;++j) for(int lane=0;lane<2;++lane)
    out[((threadIdx.x*1+c)*1+j)*2+lane]=double(__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(d[c][j] >> (16*lane)))));
}


__global__ void bf16x2_c8_q16(int iterations, Result* r, double* out) {
  unsigned d[8][1]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
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
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<8;++c) for(int j=0;j<1;++j) for(int lane=0;lane<2;++lane)
    out[((threadIdx.x*8+c)*1+j)*2+lane]=double(__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(d[c][j] >> (16*lane)))));
}


__global__ void mma_f16_c1_q16(int iterations, Result* r, double* out) {
  float d[1][4]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]) : "r"(0x2c002c00u)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<4;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*4+j)*1+lane]=double(d[c][j]);
}


__global__ void mma_f16_c8_q16(int iterations, Result* r, double* out) {
  float d[8][4]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
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
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<8;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*8+c)*4+j)*1+lane]=double(d[c][j]);
}


__global__ void mma_bf16_c1_q16(int iterations, Result* r, double* out) {
  float d[1][4]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]) : "r"(0x3d803d80u)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<4;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*4+j)*1+lane]=double(d[c][j]);
}


__global__ void mma_bf16_c8_q16(int iterations, Result* r, double* out) {
  float d[8][4]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
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
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<8;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*8+c)*4+j)*1+lane]=double(d[c][j]);
}


__global__ void mma_tf32_c1_q16(int iterations, Result* r, double* out) {
  float d[1][4]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]) : "r"(0x3d800000u)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<4;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*4+j)*1+lane]=double(d[c][j]);
}


__global__ void mma_tf32_c8_q16(int iterations, Result* r, double* out) {
  float d[8][4]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
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
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<8;++c) for(int j=0;j<4;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*8+c)*4+j)*1+lane]=double(d[c][j]);
}


__global__ void mma_f64_c1_q16(int iterations, Result* r, double* out) {
  double d[1][2]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+d"(d[c][0]), "+d"(d[c][1]) : "d"(0.0625), "d"(0.0625)); }
    }
    
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<2;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<2;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*2+j)*1+lane]=double(d[c][j]);
}


__global__ void mma_f64_c8_q16(int iterations, Result* r, double* out) {
  double d[8][2]={};
  
  __shared__ volatile double drain[32];
  __syncthreads();
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
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<8;++c) for(int j=0;j<2;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*8+c)*2+j)*1+lane]=double(d[c][j]);
}


__global__ void wgmma_f16_c1_q1(int iterations, Result* r, double* out) {
  float d[1][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x2c00;b[i]=0x2c00;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<1;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*32+j)*1+lane]=double(d[c][j]);
}


__global__ void wgmma_f16_c1_q4(int iterations, Result* r, double* out) {
  float d[1][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x2c00;b[i]=0x2c00;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<4;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*32+j)*1+lane]=double(d[c][j]);
}


__global__ void wgmma_f16_c1_q16(int iterations, Result* r, double* out) {
  float d[1][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x2c00;b[i]=0x2c00;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*32+j)*1+lane]=double(d[c][j]);
}


__global__ void wgmma_f16_c2_q1(int iterations, Result* r, double* out) {
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
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<1;++q) {
      #pragma unroll
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<2;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*2+c)*32+j)*1+lane]=double(d[c][j]);
}


__global__ void wgmma_f16_c2_q4(int iterations, Result* r, double* out) {
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
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<4;++q) {
      #pragma unroll
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<2;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*2+c)*32+j)*1+lane]=double(d[c][j]);
}


__global__ void wgmma_f16_c2_q16(int iterations, Result* r, double* out) {
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
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<2;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*2+c)*32+j)*1+lane]=double(d[c][j]);
}


__global__ void wgmma_bf16_c1_q1(int iterations, Result* r, double* out) {
  float d[1][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x3d80;b[i]=0x3d80;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<1;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*32+j)*1+lane]=double(d[c][j]);
}


__global__ void wgmma_bf16_c1_q4(int iterations, Result* r, double* out) {
  float d[1][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x3d80;b[i]=0x3d80;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<4;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*32+j)*1+lane]=double(d[c][j]);
}


__global__ void wgmma_bf16_c1_q16(int iterations, Result* r, double* out) {
  float d[1][32]={};
  __shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x3d80;b[i]=0x3d80;}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile double drain[128];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<1;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<1;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<1;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*1+c)*32+j)*1+lane]=double(d[c][j]);
}


__global__ void wgmma_bf16_c2_q1(int iterations, Result* r, double* out) {
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
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<1;++q) {
      #pragma unroll
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<2;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*2+c)*32+j)*1+lane]=double(d[c][j]);
}


__global__ void wgmma_bf16_c2_q4(int iterations, Result* r, double* out) {
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
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<4;++q) {
      #pragma unroll
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<2;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*2+c)*32+j)*1+lane]=double(d[c][j]);
}


__global__ void wgmma_bf16_c2_q16(int iterations, Result* r, double* out) {
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
  auto start=clock64();
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<16;++q) {
      #pragma unroll
      for(int c=0;c<2;++c) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]), "+f"(d[c][4]), "+f"(d[c][5]), "+f"(d[c][6]), "+f"(d[c][7]), "+f"(d[c][8]), "+f"(d[c][9]), "+f"(d[c][10]), "+f"(d[c][11]), "+f"(d[c][12]), "+f"(d[c][13]), "+f"(d[c][14]), "+f"(d[c][15]), "+f"(d[c][16]), "+f"(d[c][17]), "+f"(d[c][18]), "+f"(d[c][19]), "+f"(d[c][20]), "+f"(d[c][21]), "+f"(d[c][22]), "+f"(d[c][23]), "+f"(d[c][24]), "+f"(d[c][25]), "+f"(d[c][26]), "+f"(d[c][27]), "+f"(d[c][28]), "+f"(d[c][29]), "+f"(d[c][30]), "+f"(d[c][31]) : "l"(ad), "l"(bd) : "memory"); }
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  double sum=0;
  #pragma unroll
  for(int c=0;c<2;++c) {
    #pragma unroll
    for(int j=0;j<32;++j) sum+=double(d[c][j]);
  }
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}
  for(int c=0;c<2;++c) for(int j=0;j<32;++j) for(int lane=0;lane<1;++lane)
    out[((threadIdx.x*2+c)*32+j)*1+lane]=double(d[c][j]);
}


template<class K> void run(K kernel,const char* name,int threads,int outputs,int chains,int batch,long long work,int iterations,double expected,int repeats,Result* dr,double* dout) {
  cudaFuncAttributes attr{}; CK(cudaFuncGetAttributes(&attr,kernel));
  if(attr.localSizeBytes) {fprintf(stderr,"local memory used by %s: %zu\n",name,attr.localSizeBytes);exit(5);}
  int occupancy=0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,threads,0));
  std::vector<double> out(outputs);
  cudaEvent_t start,stop; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
  for(int rep=-2;rep<repeats;++rep) {
    CK(cudaEventRecord(start));
    kernel<<<1,threads>>>(iterations,dr,dout);
    CK(cudaGetLastError()); CK(cudaEventRecord(stop)); CK(cudaEventSynchronize(stop));
    float ms=0; CK(cudaEventElapsedTime(&ms,start,stop));
    Result r{}; CK(cudaMemcpy(&r,dr,sizeof(r),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(out.data(),dout,outputs*sizeof(double),cudaMemcpyDeviceToHost));
    double err=0; for(double x:out) {if(!std::isfinite(x)){fprintf(stderr,"nonfinite %s\n",name);exit(3);} err=std::max(err,std::abs(x-expected));}
    if(err>1e-6) {fprintf(stderr,"mismatch %s expected=%g max_err=%g\n",name,expected,err);exit(4);}
    if(rep>=0) printf("{\"case\":\"%s\",\"iterations\":%d,\"repeat\":%d,\"cycles\":%llu,\"smid\":%u,\"event_ms\":%.9g,\"max_abs_error\":%.9g,\"registers_per_thread\":%d,\"static_smem_bytes\":%zu,\"occupancy_limit_ctas_per_sm\":%d,\"ptx_collectives\":%lld,\"work_flop\":%lld}\n",name,iterations,rep,r.cycles,r.smid,ms,err,attr.numRegs,attr.sharedSizeBytes,occupancy,(long long)iterations*batch*chains,(long long)iterations*batch*chains*work);
    fflush(stdout);
  }
  CK(cudaEventDestroy(start));CK(cudaEventDestroy(stop));
}
int main(int argc,char** argv) {
  int repeats=argc>1?atoi(argv[1]):7; if(repeats<1||repeats>100)return 1;
  cudaDeviceProp p{};CK(cudaGetDeviceProperties(&p,0));
  if(p.major!=9){fprintf(stderr,"requires Hopper SM90\n");return 1;}
  printf("{\"device\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"global_memory_bytes\":%zu,\"scope\":\"single_cta\"}\n",p.name,p.major,p.minor,p.multiProcessorCount,p.totalGlobalMem);
  Result* dr; double* dout;CK(cudaMalloc(&dr,sizeof(Result)));CK(cudaMalloc(&dout,65536*sizeof(double)));
  for(int iterations:{128,512,2048}) {

run(f32_c1_q16,"f32_c1_q16",32,32,1,16,64,iterations,0.5,repeats,dr,dout);
run(f32_c8_q16,"f32_c8_q16",32,256,8,16,64,iterations,0.5,repeats,dr,dout);
run(f64_c1_q16,"f64_c1_q16",32,32,1,16,64,iterations,0.5,repeats,dr,dout);
run(f64_c8_q16,"f64_c8_q16",32,256,8,16,64,iterations,0.5,repeats,dr,dout);
run(f16_c1_q16,"f16_c1_q16",32,32,1,16,64,iterations,0.5,repeats,dr,dout);
run(f16_c8_q16,"f16_c8_q16",32,256,8,16,64,iterations,0.5,repeats,dr,dout);
run(f16x2_c1_q16,"f16x2_c1_q16",32,64,1,16,128,iterations,0.5,repeats,dr,dout);
run(f16x2_c8_q16,"f16x2_c8_q16",32,512,8,16,128,iterations,0.5,repeats,dr,dout);
run(bf16_c1_q16,"bf16_c1_q16",32,32,1,16,64,iterations,0.5,repeats,dr,dout);
run(bf16_c8_q16,"bf16_c8_q16",32,256,8,16,64,iterations,0.5,repeats,dr,dout);
run(bf16x2_c1_q16,"bf16x2_c1_q16",32,64,1,16,128,iterations,0.5,repeats,dr,dout);
run(bf16x2_c8_q16,"bf16x2_c8_q16",32,512,8,16,128,iterations,0.5,repeats,dr,dout);
run(mma_f16_c1_q16,"mma_f16_c1_q16",32,128,1,16,4096,iterations,double(iterations)*16*16/256.0,repeats,dr,dout);
run(mma_f16_c8_q16,"mma_f16_c8_q16",32,1024,8,16,4096,iterations,double(iterations)*16*16/256.0,repeats,dr,dout);
run(mma_bf16_c1_q16,"mma_bf16_c1_q16",32,128,1,16,4096,iterations,double(iterations)*16*16/256.0,repeats,dr,dout);
run(mma_bf16_c8_q16,"mma_bf16_c8_q16",32,1024,8,16,4096,iterations,double(iterations)*16*16/256.0,repeats,dr,dout);
run(mma_tf32_c1_q16,"mma_tf32_c1_q16",32,128,1,16,2048,iterations,double(iterations)*16*8/256.0,repeats,dr,dout);
run(mma_tf32_c8_q16,"mma_tf32_c8_q16",32,1024,8,16,2048,iterations,double(iterations)*16*8/256.0,repeats,dr,dout);
run(mma_f64_c1_q16,"mma_f64_c1_q16",32,64,1,16,512,iterations,double(iterations)*16/64.0,repeats,dr,dout);
run(mma_f64_c8_q16,"mma_f64_c8_q16",32,512,8,16,512,iterations,double(iterations)*16/64.0,repeats,dr,dout);
run(wgmma_f16_c1_q1,"wgmma_f16_c1_q1",128,4096,1,1,131072,iterations,double(iterations)*1/16.0,repeats,dr,dout);
run(wgmma_f16_c1_q4,"wgmma_f16_c1_q4",128,4096,1,4,131072,iterations,double(iterations)*4/16.0,repeats,dr,dout);
run(wgmma_f16_c1_q16,"wgmma_f16_c1_q16",128,4096,1,16,131072,iterations,double(iterations)*16/16.0,repeats,dr,dout);
run(wgmma_f16_c2_q1,"wgmma_f16_c2_q1",128,8192,2,1,131072,iterations,double(iterations)*1/16.0,repeats,dr,dout);
run(wgmma_f16_c2_q4,"wgmma_f16_c2_q4",128,8192,2,4,131072,iterations,double(iterations)*4/16.0,repeats,dr,dout);
run(wgmma_f16_c2_q16,"wgmma_f16_c2_q16",128,8192,2,16,131072,iterations,double(iterations)*16/16.0,repeats,dr,dout);
run(wgmma_bf16_c1_q1,"wgmma_bf16_c1_q1",128,4096,1,1,131072,iterations,double(iterations)*1/16.0,repeats,dr,dout);
run(wgmma_bf16_c1_q4,"wgmma_bf16_c1_q4",128,4096,1,4,131072,iterations,double(iterations)*4/16.0,repeats,dr,dout);
run(wgmma_bf16_c1_q16,"wgmma_bf16_c1_q16",128,4096,1,16,131072,iterations,double(iterations)*16/16.0,repeats,dr,dout);
run(wgmma_bf16_c2_q1,"wgmma_bf16_c2_q1",128,8192,2,1,131072,iterations,double(iterations)*1/16.0,repeats,dr,dout);
run(wgmma_bf16_c2_q4,"wgmma_bf16_c2_q4",128,8192,2,4,131072,iterations,double(iterations)*4/16.0,repeats,dr,dout);
run(wgmma_bf16_c2_q16,"wgmma_bf16_c2_q16",128,8192,2,16,131072,iterations,double(iterations)*16/16.0,repeats,dr,dout);
} CK(cudaFree(dr));CK(cudaFree(dout));return 0;}
