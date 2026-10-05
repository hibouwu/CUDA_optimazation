#!/usr/bin/env python3
"""Extend the archived arithmetic probes with sustained, bounded issuance."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent


def generate():
    original = (ROOT/'probe.cu').read_text()
    base = json.loads((ROOT/'cases.json').read_text())
    header = original.split('__global__ void ',1)[0]
    header = header.replace('unsigned long long cycles; unsigned smid;',
                            'unsigned long long cycles, start_ns, stop_ns; unsigned smid;')
    header += '\n#include <set>\n#include <chrono>\n#include <random>\n#include <functional>\n'
    header += '__device__ unsigned long long timer_ns(){unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}\n'
    code=[header]
    cases=[]
    for c in base:
        wg=c['kind'].startswith('wgmma')
        if c['chains'] != (2 if wg else 8) or c['batch'] != 16:
            continue
        match=re.search(r'__global__ void '+re.escape(c['name'])+r'\(.*?\n}\n',original,re.S)
        assert match
        for threads in ([128,256] if wg else [32,128,256]):
            for pending in ([0,3,7] if wg else [0]):
                name=c['name']+f'_t{threads}_p{pending}'
                body=match.group().replace(c['name'],name)
                body=body.replace(f'drain[{c["threads"]}]',f'drain[{threads}]')
                if wg:
                    body=body.replace('wgmma.wait_group.sync.aligned 0;',f'wgmma.wait_group.sync.aligned {pending};')
                    body=body.replace('  double sum=0;', '  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n  double sum=0;')
                body=body.replace('  auto start=clock64();','  auto start_ns=timer_ns();\n  auto start=clock64();')
                body=body.replace('  auto stop=clock64();','  auto stop=clock64();\n  auto stop_ns=timer_ns();')
                body=body.replace('r->cycles=stop-start;', 'r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_ns=start_ns; r[blockIdx.x].stop_ns=stop_ns;')
                body=body.replace('"=r"(r->smid)','"=r"(r[blockIdx.x].smid)')
                body=body.replace('out[((threadIdx.x*', 'out[(((blockIdx.x*blockDim.x+threadIdx.x)*')
                factor=threads//c['threads']
                entry=dict(c,name=name,threads=threads,outputs=c['outputs']*factor,
                           work_per_collective=c['work_per_collective']*factor,
                           pending_after_wait=pending,base_name=c['name'])
                cases.append(entry)
                code.append(body)
    code.append(r'''
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
''')
    for c in cases:
        increment=1 if c['kind'].startswith('wgmma') or c['kind'] in ['mma_f16','mma_bf16'] else .5 if c['kind']=='mma_tf32' else .25 if c['kind']=='mma_f64' else 0
        for scope in [0,1]:
            args=f'{c["name"]},"{c["name"]}","{c["kind"]}",{c["threads"]},{c["outputs"]},{c["chains"]},{c["batch"]},{c["work_per_collective"]},{increment},{scope},p.multiProcessorCount'
            code.append(f'jobs.push_back([=](){{run_stress({args});}});')
    code.append('std::mt19937 rng(20260930);std::shuffle(jobs.begin(),jobs.end(),rng);for(auto &job:jobs)job();return 0;}')
    (ROOT/'stress.cu').write_text('\n'.join(code))
    (ROOT/'stress_cases.json').write_text(json.dumps(dict(base_source_sha256=hashlib.sha256(original.encode()).hexdigest(),cases=cases),indent=2)+'\n')
    print(f'{len(cases)} kernels, {len(cases)*2} scope/configuration pairs, {len(cases)*10} measured launches')


if __name__=='__main__':generate()
