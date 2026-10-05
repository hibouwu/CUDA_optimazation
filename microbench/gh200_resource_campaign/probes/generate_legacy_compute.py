#!/usr/bin/env python3
"""Generate v2 arithmetic probes from the independently gated finite contracts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FAMILIES = ('legacy_fma', 'legacy_mma', 'legacy_wgmma')


def specialization(case):
    p = case['parameters']
    return (p['ptx'], case['threads'], p['chains'], p['batch'], p['wait'],
            p['drain']['post_loop_wgmma_wait'], p['operand_source_form'])


def symbol(case):
    identity=json.dumps(specialization(case),separators=(',', ':'))
    return 'lc_' + hashlib.sha256(identity.encode()).hexdigest()[:20]


def encode16(kind, value):
    return ('__bfloat16_as_ushort(__float2bfloat16_rn(' + value + '))' if 'bf16' in kind
            else '__half_as_ushort(__float2half_rn(' + value + '))')


def decode16(kind, value):
    return ('__bfloat162float(__ushort_as_bfloat16(' + value + '))' if 'bf16' in kind
            else '__half2float(__ushort_as_half(' + value + '))')


def fma_body(case):
    p=case['parameters'];kind=p['input_type'];chains=p['chains'];lanes=p['packed_lanes']
    dtype='double' if kind=='f64' else 'float' if kind=='f32' else 'unsigned' if lanes==2 else 'unsigned short'
    constraint='d' if kind=='f64' else 'f' if kind=='f32' else 'r' if lanes==2 else 'h'
    initial='(float)((static_cast<unsigned long long>(threadIdx.x)+c+lane+seed)%7)/16.0f'
    addend='(float)(1+(static_cast<unsigned long long>(threadIdx.x)+3*c+lane+seed)%7)/32.0f'
    if kind in ('f32','f64'):
        setup=f'''{dtype} d[{chains}],b[{chains}];const {dtype} a=0.5;
  #pragma unroll
  for(int c=0;c<{chains};++c) {{const int lane=0;d[c]=nonuniform?{initial}:0;b[c]=nonuniform?{addend}:0.25;}}'''
        output='double(d[c])'
    else:
        a=encode16(kind,'0.5f');setup=f'''{dtype} d[{chains}]={{}},b[{chains}]={{}};
  const {dtype} a={a}'''+(f'|(unsigned({a})<<16)' if lanes==2 else '')+';\n'
        setup+=f'''  #pragma unroll
  for(int c=0;c<{chains};++c) {{
    #pragma unroll
    for(int lane=0;lane<{lanes};++lane) {{
      unsigned short dv={encode16(kind,'(nonuniform?'+initial+':0.0f)')};
      unsigned short bv={encode16(kind,'(nonuniform?'+addend+':0.25f)')};
      d[c]|=static_cast<{dtype}>(dv)<<(16*lane);b[c]|=static_cast<{dtype}>(bv)<<(16*lane);
    }}
  }}'''
        output='double('+decode16(kind,'static_cast<unsigned short>(d[c]>>(16*lane))')+')'
    instruction=f'asm volatile("{p["ptx"]} %0,%0,%1,%2;" : "+{constraint}"(d[c]) : "{constraint}"(a), "{constraint}"(b[c]));'
    loop=f'''  #pragma unroll 1
  for(int it=0;it<iterations;++it) {{
    #pragma unroll
    for(int q=0;q<{p['batch']};++q) {{
      #pragma unroll
      for(int c=0;c<{chains};++c) {{{instruction}}}
    }}
  }}'''
    drain=f'  double sum=0;\n  #pragma unroll\n  for(int c=0;c<{chains};++c)sum+=double(d[c]);'
    write=f'''  #pragma unroll
  for(int c=0;c<{chains};++c) {{
    #pragma unroll
    for(int lane=0;lane<{lanes};++lane)
      output[((blockIdx.x*blockDim.x+threadIdx.x)*{chains}+c)*{lanes}+lane]={output};
  }}'''
    return setup,loop,drain,write


def mma_body(case):
    """Pack each operand from logical coordinates; the host oracle stays separate."""
    p = case['parameters']; kind = p['input_type']; chains = p['chains']
    m, n, k = p['shape']; outputs = m * n // 32
    dtype = 'double' if kind == 'f64' else 'float'
    operand = 'double' if kind == 'f64' else 'unsigned'
    na, nb = ((1, 1) if kind == 'f64' else (4, 2))
    setup = f'''const unsigned lane=threadIdx.x%32,group=threadIdx.x/32;
  {operand} a[{na}]={{}},b[{nb}]={{}};{dtype} d[{chains}][{outputs}];
  #pragma unroll
  for(int c=0;c<{chains};++c) {{
    #pragma unroll
    for(int j=0;j<{outputs};++j)d[c][j]=nonuniform?{dtype}(1+group+c)/64:0;
  }}
'''
    if kind in ('f16', 'bf16'):
        setup += f'''  #pragma unroll
  for(int e=0;e<8;++e) {{
    unsigned row=lane/4+((e/2)%2)*8,col=(lane%4)*2+e%2+(e/4)*8;
    float value=nonuniform?float(1+row+2*col+seed%3)/256.0f:0.0625f;
    a[e/2]|=unsigned({encode16(kind, 'value')})<<(16*(e%2));
  }}
  #pragma unroll
  for(int e=0;e<4;++e) {{
    unsigned row=(lane%4)*2+e%2+(e/2)*8,col=lane/4;
    float value=nonuniform?float(1+col+3*row+seed%5)/256.0f:0.0625f;
    b[e/2]|=unsigned({encode16(kind, 'value')})<<(16*(e%2));
  }}
'''
    elif kind == 'tf32':
        setup += '''  #pragma unroll
  for(int e=0;e<4;++e) {
    unsigned row=lane/4+(e%2)*8,col=lane%4+(e/2)*4;
    float value=nonuniform?float(1+row+2*col+seed%3)/256.0f:0.0625f;
    a[e]=__float_as_uint(value);
  }
  #pragma unroll
  for(int e=0;e<2;++e) {
    unsigned row=lane%4+e*4,col=lane/4;
    float value=nonuniform?float(1+col+3*row+seed%5)/256.0f:0.0625f;
    b[e]=__float_as_uint(value);
  }
'''
    elif kind == 'f64':
        setup += '''  a[0]=nonuniform?double(1+lane/4+2*(lane%4)+seed%3)/256.0:0.0625;
  b[0]=nonuniform?double(1+lane/4+3*(lane%4)+seed%5)/256.0:0.0625;
'''
    else:
        raise ValueError('unsupported legacy MMA input ' + kind)
    regs = lambda first, count: '{' + ','.join('%'+str(i) for i in range(first, first+count)) + '}'
    outputs_asm = ','.join(f'"+{ "d" if kind=="f64" else "f" }"(d[c][{j}])' for j in range(outputs))
    inputs_asm = ','.join(f'"{ "d" if kind=="f64" else "r" }"({name}[{j}])'
                          for name, count in [('a', na), ('b', nb)] for j in range(count))
    instruction = (f'asm volatile("{p["ptx"]} {regs(0, outputs)}, {regs(outputs, na)}, '
                   f'{regs(outputs+na, nb)}, {regs(0, outputs)};" : {outputs_asm} : {inputs_asm});')
    loop = f'''  #pragma unroll 1
  for(int it=0;it<iterations;++it) {{
    #pragma unroll
    for(int q=0;q<{p['batch']};++q) {{
      #pragma unroll
      for(int c=0;c<{chains};++c) {{{instruction}}}
    }}
  }}'''
    drain = f'''  double sum=0;
  #pragma unroll
  for(int c=0;c<{chains};++c) {{
    #pragma unroll
    for(int j=0;j<{outputs};++j)sum+=double(d[c][j]);
  }}'''
    write = f'''  #pragma unroll
  for(int c=0;c<{chains};++c) {{
    #pragma unroll
    for(int j=0;j<{outputs};++j)
      output[((blockIdx.x*blockDim.x+threadIdx.x)*{chains}+c)*{outputs}+j]=double(d[c][j]);
  }}'''
    return setup, loop, drain, write


def wgmma_body(case):
    p = case['parameters']; kind = p['input_type']; chains = p['chains']
    rs = p['operand_source_form'] == 'RS'
    if p['shape'] != [64, 64, 16] or kind not in ('f16', 'bf16'):
        raise ValueError('unimplemented WGMMA representative')
    setup = f'''const unsigned local_thread=threadIdx.x%128,group=threadIdx.x/128;
  __shared__ __align__(128) unsigned short bs[1024];
'''
    if not rs:
        setup += f'  __shared__ __align__(128) unsigned short as[1024];\n'
    setup += f'''  for(unsigned i=threadIdx.x;i<1024;i+=blockDim.x) {{
    unsigned outer=i/16,inner=i%16;
    unsigned offset=(outer%8)*8+(outer/8)*64+inner%8+(inner/8)*512;
    float b_value=nonuniform?float(1+outer+3*inner+seed%5)/256.0f:0.0625f;
    bs[offset]={encode16(kind, 'b_value')};
'''
    if not rs:
        setup += f'''    float a_value=nonuniform?float(1+outer+2*inner+seed%3)/256.0f:0.0625f;
    as[offset]={encode16(kind, 'a_value')};
'''
    setup += '  }\n  __syncthreads();\n  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n'
    setup += '  unsigned long long bd=lc_descriptor(bs);\n'
    if rs:
        setup += f'''  unsigned ar[4]={{}};
  #pragma unroll
  for(int e=0;e<8;++e) {{
    unsigned row=(local_thread/32)*16+(local_thread%32)/4+((e/2)%2)*8;
    unsigned col=(local_thread%4)*2+e%2+(e/4)*8;
    float value=nonuniform?float(1+row+2*col+seed%3)/256.0f:0.0625f;
    ar[e/2]|=unsigned({encode16(kind, 'value')})<<(16*(e%2));
  }}
  #pragma unroll
  for(int e=0;e<4;++e)asm volatile("" : "+r"(ar[e]) :: "memory");
'''
    else:
        setup += '  unsigned long long ad=lc_descriptor(as);\n'
    setup += f'''  float d[{chains}][32];
  #pragma unroll
  for(int c=0;c<{chains};++c) {{
    #pragma unroll
    for(int j=0;j<32;++j) {{
      d[c][j]=nonuniform?float(1+group+c)/64.0f:0.0f;
      // Compiler operand fence: initialization precedes the WGMMA register fence.
      asm volatile("" : "+f"(d[c][j]) :: "memory");
    }}
  }}
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
'''
    outputs = ','.join(f'"+f"(d[c][{j}])' for j in range(32))
    inputs = ','.join(f'"r"(ar[{j}])' for j in range(4)) + ',"l"(bd)' if rs else '"l"(ad),"l"(bd)'
    instruction = f'asm volatile("{p["instruction_template"]}" : {outputs} : {inputs} : "memory");'
    loop = f'''  #pragma unroll 1
  for(int it=0;it<iterations;++it) {{
    #pragma unroll
    for(int q=0;q<{p['batch']};++q) {{
      #pragma unroll
      for(int c=0;c<{chains};++c) {{{instruction}}}
    }}
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned {p['wait']};" ::: "memory");
  }}'''
    if p['drain']['post_loop_wgmma_wait'] == 0:
        loop += '\n  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");'
    drain = f'''  double sum=0;
  #pragma unroll
  for(int c=0;c<{chains};++c) {{
    #pragma unroll
    for(int j=0;j<32;++j) {{
      // Keep consumers after the completion wait at the compiler boundary.
      asm volatile("" : "+f"(d[c][j]) :: "memory");
      sum+=double(d[c][j]);
    }}
  }}'''
    write = f'''  #pragma unroll
  for(int c=0;c<{chains};++c) {{
    #pragma unroll
    for(int j=0;j<32;++j)
      output[((blockIdx.x*blockDim.x+threadIdx.x)*{chains}+c)*32+j]=double(d[c][j]);
  }}'''
    return setup, loop, drain, write


def kernel(case):
    width = case['parameters']['collective_width_threads']
    if width == 1:
        setup,loop,drain,write=fma_body(case)
    elif width == 32:
        setup,loop,drain,write=mma_body(case)
    elif width == 128:
        setup,loop,drain,write=wgmma_body(case)
    else:
        raise ValueError('unknown collective width')
    return f'''__global__ void {symbol(case)}(int iterations,unsigned seed,bool nonuniform,gh::Stamp* stamps,double* output) {{
  {setup}
  __shared__ volatile double drain[{case['threads']}];
  __shared__ unsigned long long t0,c0;
  __syncthreads();
  if(threadIdx.x==0){{t0=lc_ns();c0=clock64();}}
  __syncthreads();
{loop}
{drain}
  drain[threadIdx.x]=sum;
  __syncthreads();
  if(threadIdx.x==0){{auto c1=static_cast<unsigned long long>(clock64());auto t1=lc_ns();stamps[blockIdx.x]={{t0,t1,c0,c1,lc_smid()}};}}
{write}
}}
'''


def generate(family, root=ROOT):
    contract=json.loads((root/'microbench/gh200_resource_campaign/contracts'/f'{family}.json').read_text())
    code=['''// Generated by generate_legacy_compute.py; edit the generator, not this file.
#include "../common/legacy_compute_runtime.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
__device__ __forceinline__ unsigned long long lc_ns(){unsigned long long x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x));return x;}
__device__ __forceinline__ unsigned lc_smid(){unsigned x;asm volatile("mov.u32 %0, %%smid;":"=r"(x));return x;}
__device__ __forceinline__ unsigned long long lc_descriptor(void* p){
  unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(p));
  return ((address&0x3ffffu)>>4)|(64ull<<16)|(8ull<<32);
}
''']
    seen=set()
    for case in contract['cases']:
        sym=symbol(case)
        if sym not in seen:code.append(kernel(case));seen.add(sym)
    code.append('int main(int argc,char**argv) try {\n  const std::vector<legacy_compute::Case> cases={')
    for case in contract['cases']:
        p=case['parameters'];dims=p['shape'] or [0,0,0];ip=case['iteration_policy'];dynamic=ip['kind']=='calibrated'
        fields=[json.dumps(case['id']),json.dumps(p['input_type']),symbol(case),str(case['threads']),str(p['chains']),str(p['batch']),str(p['packed_lanes']),*(str(x) for x in dims),str(p['collective_width_threads']),str(case['scope']=='all_gpu').lower(),str(dynamic).lower(),str(case['iterations']),str(ip.get('max_iterations',case['iterations']))]
        code.append('    {'+','.join(fields)+'},')
    code.append('  };\n  return legacy_compute::run(argc,argv,cases);\n} catch(const std::exception& e){std::cerr<<e.what()<<"\\n";return 2;}\n')
    return '\n'.join(code)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=FAMILIES,required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();text=generate(args.family)
    output=Path(__file__).resolve().parent/(args.family+'.cu')
    if args.check:
        if output.read_text()!=text:raise ValueError('generated source differs')
    else:output.write_text(text)
    print(json.dumps({'family':args.family,'sha256':hashlib.sha256(text.encode()).hexdigest(),'check':args.check}))


if __name__=='__main__':main()
