#!/usr/bin/env python3
"""Generate dependency-explicit, single-CTA SM90a compute probes."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent


def generate():
    code = [r'''
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
''']
    cases=[]
    for kind in ['f32','f64','f16','f16x2','bf16','bf16x2','mma_f16','mma_bf16','mma_tf32','mma_f64','wgmma_f16','wgmma_bf16']:
        wg=kind.startswith('wgmma')
        mma=kind.startswith('mma_')
        for chains in ([1,2] if wg else [1,8]):
            for batch in ([1,4,16] if wg else [16]):
                name=f'{kind}_c{chains}_q{batch}'
                threads=128 if wg else 32
                registers=32 if wg else (2 if kind=='mma_f64' else 4 if mma else 1)
                dtype='double' if kind in ['f64','mma_f64'] else 'float' if (mma or wg or kind=='f32') else 'unsigned short' if kind in ['f16','bf16'] else 'unsigned'
                init=f'{dtype} d[{chains}][{registers}]={{}};'
                setup=''
                if wg:
                    bits='0x2c00' if kind=='wgmma_f16' else '0x3d80'  # 1/16
                    setup=f'''__shared__ __align__(128) unsigned short a[1024], b[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x){{a[i]={bits};b[i]={bits};}}
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto ad=descriptor(a), bd=descriptor(b);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");'''
                    operands=', '.join(f'%{i}' for i in range(32))
                    outputs=', '.join(f'"+f"(d[c][{i}])' for i in range(32))
                    itype='f16' if kind=='wgmma_f16' else 'bf16'
                    op=f'''asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.{itype}.{itype} {{{operands}}}, %32, %33, 1, 1, 1, 0, 0;" : {outputs} : "l"(ad), "l"(bd) : "memory");'''
                    ptx=f'wgmma.mma_async.sync.aligned.m64n64k16.f32.{itype}.{itype}'
                    work=2*64*64*16
                    expected=f'double(iterations)*{batch}/16.0'
                elif mma:
                    if kind=='mma_f64':
                        op='asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+d"(d[c][0]), "+d"(d[c][1]) : "d"(0.0625), "d"(0.0625));'
                        ptx='mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64'
                        work=2*8*8*4
                        expected='double(iterations)*16/64.0'
                    else:
                        itype={'mma_f16':'f16','mma_bf16':'bf16','mma_tf32':'tf32'}[kind]
                        k=8 if itype=='tf32' else 16
                        bits={'f16':'0x2c002c00u','bf16':'0x3d803d80u','tf32':'0x3d800000u'}[itype]
                        ptx=f'mma.sync.aligned.m16n8k{k}.row.col.f32.{itype}.{itype}.f32'
                        op=f'''asm volatile("{ptx} {{%0,%1,%2,%3}}, {{%4,%4,%4,%4}}, {{%4,%4}}, {{%0,%1,%2,%3}};" : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3]) : "r"({bits}));'''
                        work=2*16*8*k
                        expected=f'double(iterations)*16*{k}/256.0'
                else:
                    ptx=f'fma.rn.{kind}'
                    if kind in ['f32','f64']:
                        con='f' if kind=='f32' else 'd'
                        op=f'asm volatile("{ptx} %0,%0,%1,%2;" : "+{con}"(d[c][0]) : "{con}"(({dtype})0.5), "{con}"(({dtype})0.25));'
                    else:
                        bf=kind.startswith('bf')
                        a=0x3f00 if bf else 0x3800
                        b=0x3e80 if bf else 0x3400
                        if kind.endswith('x2'):
                            a=a|(a<<16); b=b|(b<<16)
                        con='r' if kind.endswith('x2') else 'h'
                        op=f'asm volatile("{ptx} %0,%0,%1,%2;" : "+{con}"(d[c][0]) : "{con}"(({dtype}){a}u), "{con}"(({dtype}){b}u));'
                    work=threads*2*(2 if kind.endswith('x2') else 1)
                    expected='0.5'
                decode='double(d[c][j])'
                if not (mma or wg or kind in ['f32','f64']):
                    bits='static_cast<unsigned short>(d[c][j] >> (16*lane))' if kind.endswith('x2') else 'd[c][j]'
                    decode=f'double(__bfloat162float(__ushort_as_bfloat16({bits})))' if kind.startswith('bf') else f'double(__half2float(__ushort_as_half({bits})))'
                lanes=2 if kind.endswith('x2') else 1
                wait='asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");' if wg else ''
                # SMEM capture + CTA barrier forces completion of register results before stop.
                code.append(f'''
__global__ void {name}(int iterations, Result* r, double* out) {{
  {init}
  {setup}
  __shared__ volatile double drain[{threads}];
  __syncthreads();
  auto start=clock64();
  for(int it=0;it<iterations;++it) {{
    #pragma unroll
    for(int q=0;q<{batch};++q) {{
      #pragma unroll
      for(int c=0;c<{chains};++c) {{ {op} }}
    }}
    {wait}
  }}
  double sum=0;
  #pragma unroll
  for(int c=0;c<{chains};++c) {{
    #pragma unroll
    for(int j=0;j<{registers};++j) sum+=double(d[c][j]);
  }}
  drain[threadIdx.x]=sum;
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==0) {{r->cycles=stop-start; asm volatile("mov.u32 %0, %%smid;":"=r"(r->smid));}}
  for(int c=0;c<{chains};++c) for(int j=0;j<{registers};++j) for(int lane=0;lane<{lanes};++lane)
    out[((threadIdx.x*{chains}+c)*{registers}+j)*{lanes}+lane]={decode};
}}
''')
                cases.append(dict(name=name,kind=kind,chains=chains,batch=batch,threads=threads,outputs=threads*chains*registers*lanes,work_per_collective=work,ptx=ptx,expected=expected))
    code.append(r'''
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
''')
    for c in cases:
        code.append(f'run({c["name"]},"{c["name"]}",{c["threads"]},{c["outputs"]},{c["chains"]},{c["batch"]},{c["work_per_collective"]},iterations,{c["expected"]},repeats,dr,dout);')
    code.append('} CK(cudaFree(dr));CK(cudaFree(dout));return 0;}\n')
    (ROOT/'probe.cu').write_text('\n'.join(code))
    (ROOT/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(f'Generated {len(cases)} kernels, {len(cases)*3} timing configurations')


if __name__=='__main__':
    generate()
