#!/usr/bin/env python3
"""Prespecified audit replay: rounds, fixed lengths, per-SM counters and controls."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parent


def generate():
    source=ROOT/'results/20260930-sustained/stress.cu'
    text=source.read_text()
    original=json.loads((source.parent/'stress_cases.json').read_text())['cases']
    names=['f32_c8_q16_t256_p0','f64_c8_q16_t256_p0','f16x2_c8_q16_t256_p0',
           'mma_f16_c8_q16_t256_p0']+[f'wgmma_f16_c2_q16_t128_p{p}' for p in [0,3,7]]
    cases=[c for name in names for c in original if c['name']==name]
    assert len(cases)==7
    header=text.split('__global__ void ',1)[0]
    header=header.replace('cycles, start_ns, stop_ns;', 'cycles, start_ns, stop_ns, start_cycle, stop_cycle;')
    header+='#include <map>\n#include <string>\n#include <cstring>\n'
    code=[header]
    for c in cases:
        body=re.search(r'__global__ void '+re.escape(c['name'])+r'\(.*?\n}\n',text,re.S).group()
        body=body.replace('r[blockIdx.x].cycles=stop-start;', 'r[blockIdx.x].cycles=stop-start; r[blockIdx.x].start_cycle=start; r[blockIdx.x].stop_cycle=stop;')
        code.append(body)
    code.append(r'''
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
''')
    for c in cases:
        name=c['name'];increment=1 if c['kind'].startswith('wgmma') or c['kind']=='mma_f16' else 0
        code.append(f'''probes.push_back(make_probe({name},"{name}","{c['kind']}",{c['threads']},{c['outputs']},{c['chains']},{c['batch']},{c['work_per_collective']},{increment},
        [](int blocks,int it,Result* r,double* out){{{name}<<<blocks,{c['threads']}>>>(it,r,out);}}));''')
    code.append(r'''
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
''')
    (ROOT/'audit.cu').write_text('\n'.join(code))
    (ROOT/'audit_cases.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=cases,rounds=12,lengths=[8192,32768,65536],warmup_cv_threshold=.01),indent=2)+'\n')
    print('Generated 7 audit kernels, 42 measured configurations x 12 rounds + empty/warmup controls')


if __name__=='__main__':generate()
