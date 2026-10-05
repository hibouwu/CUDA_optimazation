#!/usr/bin/env python3
"""Nonuniform, short correctness probes; these are not throughput samples."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent


def generate():
    source=(ROOT/'audit.cu').read_text()
    code=[source.split('__global__ void ',1)[0],r'''
__host__ __device__ float av(int r,int k){return (1+(r%3)+(k%2))/64.0f;}
__host__ __device__ float bv(int k,int n){return (1+(n%5)+(k%3))/64.0f;}
__device__ unsigned pack(float a,float b){return __half_as_ushort(__float2half_rn(a))|(unsigned(__half_as_ushort(__float2half_rn(b)))<<16);}
''']
    names=['f32_c8_q16_t256_p0','mma_f16_c8_q16_t256_p0','wgmma_f16_c2_q16_t128_p7']
    for name in names:
        body=re.search(r'__global__ void '+name+r'\(.*?\n}\n',source,re.S).group()
        body=body.replace(name,'validate_'+name)
        if name.startswith('f32'):
            body=body.replace('float d[8][1]={};','float d[8][1]={};\n  for(int c=0;c<8;++c)d[c][0]=float((threadIdx.x%17)+c)/128.0f;')
            body=body.replace('"f"((float)0.5), "f"((float)0.25)', '"f"((float)1.0), "f"((float)0.0078125)')
        elif name.startswith('mma'):
            body=body.replace('float d[8][4]={};',r'''float d[8][4]={};
  int g=(threadIdx.x%32)/4,t=threadIdx.x%4;
  unsigned ar[4],br[2];
  for(int j=0;j<4;++j){int r=g+(j%2)*8,k=t*2+(j/2)*8;ar[j]=pack(av(r,k),av(r,k+1));}
  for(int j=0;j<2;++j){int k=t*2+j*8; br[j]=pack(bv(k,g),bv(k+1,g));}''')
            body=body.replace('{%4,%4,%4,%4}, {%4,%4}', '{%4,%5,%6,%7}, {%8,%9}')
            body=body.replace('"r"(0x2c002c00u)', '"r"(ar[0]),"r"(ar[1]),"r"(ar[2]),"r"(ar[3]),"r"(br[0]),"r"(br[1])')
        else:
            body=body.replace('for(int i=threadIdx.x;i<1024;i+=blockDim.x){a[i]=0x2c00;b[i]=0x2c00;}',r'''
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){
    int r=i/16,k=i%16;
    int idx=(r%8)*8+(r/8)*64+(k%8)+(k/8)*512;
    a[idx]=__half_as_ushort(__float2half_rn(av(r,k)));
    b[idx]=__half_as_ushort(__float2half_rn(bv(k,r)));
  }''')
        code.append(body)
    code.append(r'''
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
''')
    (ROOT/'validation.cu').write_text('\n'.join(code))


if __name__=='__main__':generate()
