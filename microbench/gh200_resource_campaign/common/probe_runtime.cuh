#pragma once
// Shared host-side protocol for v2 probes. No family-specific work formula lives here.
#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define GH_CUDA(call) do { cudaError_t gh_status_=(call); if(gh_status_!=cudaSuccess) \
  throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(gh_status_)); } while(0)

namespace gh {
using u64 = unsigned long long;
#ifndef GH_TIMER_RESOLUTION_V3
#define GH_TIMER_RESOLUTION_V3 0
#endif
enum class TimerPolicy { StrictPositiveNs, ZeroWorkOneCta, ZeroWorkGrid };
inline TimerPolicy control_timer_policy(bool zero_work,bool one_cta) {
  if(!GH_TIMER_RESOLUTION_V3 || !zero_work)return TimerPolicy::StrictPositiveNs;
  return one_cta?TimerPolicy::ZeroWorkOneCta:TimerPolicy::ZeroWorkGrid;
}
struct Stamp { u64 begin_ns,end_ns,begin_cycle,end_cycle; unsigned smid; };
inline std::string quote(const std::string& s) {
  std::ostringstream out;out<<'"';
  for(unsigned char c:s) {
    if(c=='"' || c=='\\') out<<'\\'<<c;
    else if(c=='\n') out<<"\\n";
    else if(c<32) out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
    else out<<c;
  }
  return out.str()+'"';
}
inline u64 integer(const char* value,u64 minimum,u64 maximum) {
  if(!value || !*value || *value=='-') throw std::runtime_error("invalid unsigned argument");
  errno=0;char* end=nullptr;auto n=std::strtoull(value,&end,10);
  if(errno || *end || n<minimum || n>maximum) throw std::runtime_error("argument out of range");
  return n;
}
struct Device {
  cudaDeviceProp prop{};std::string uuid;int driver=0,runtime=0;
};
inline Device device() {
  int count=0;GH_CUDA(cudaGetDeviceCount(&count));
  if(count!=1) throw std::runtime_error("exactly one visible GPU required");
  Device d;GH_CUDA(cudaGetDeviceProperties(&d.prop,0));
  if(d.prop.major!=9 || d.prop.minor!=0 || std::string(d.prop.name).find("GH200")==std::string::npos)
    throw std::runtime_error("GH200 CC9.0 required");
  GH_CUDA(cudaDriverGetVersion(&d.driver));GH_CUDA(cudaRuntimeGetVersion(&d.runtime));
  std::ostringstream id;id<<"GPU-";
  for(int i=0;i<16;++i) {
    if(i==4||i==6||i==8||i==10)id<<'-';
    id<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(static_cast<unsigned char>(d.prop.uuid.bytes[i]));
  }
  d.uuid=id.str();return d;
}
inline void emit_device(const Device& d) {
  const auto& p=d.prop;
  std::cout<<"{\"schema_version\":2,\"type\":\"device\",\"uuid\":"<<quote(d.uuid)
    <<",\"name\":"<<quote(p.name)<<",\"cc\":\"9.0\",\"sms\":"<<p.multiProcessorCount
    <<",\"driver_version\":"<<d.driver<<",\"runtime_version\":"<<d.runtime
    <<",\"global_memory_bytes\":"<<p.totalGlobalMem<<",\"l2_cache_bytes\":"<<p.l2CacheSize
    <<",\"registers_per_sm\":"<<p.regsPerMultiprocessor
    <<",\"smem_per_sm_bytes\":"<<p.sharedMemPerMultiprocessor
    <<",\"smem_per_cta_optin_bytes\":"<<p.sharedMemPerBlockOptin<<"}\n";
}
struct Observation {
  std::vector<Stamp> stamps;
  double event_ms=0;
  u64 errors=0,checked_elements=0;
  std::string method,input_conditions;
  TimerPolicy timer_policy=TimerPolicy::StrictPositiveNs;
};
inline std::pair<u64,u64> envelope(const Observation& obs) {
  if(obs.stamps.empty())throw std::runtime_error("empty CTA record set");
  const bool zero=obs.timer_policy!=TimerPolicy::StrictPositiveNs;
  if(zero&&!GH_TIMER_RESOLUTION_V3)throw std::runtime_error("timer revision not compiled in");
  if(obs.timer_policy==TimerPolicy::ZeroWorkOneCta&&obs.stamps.size()!=1)
    throw std::runtime_error("local-cycle control requires exactly one CTA");
  u64 first=std::numeric_limits<u64>::max(),last=0;
  for(auto s:obs.stamps) {
    if(s.end_ns<s.begin_ns || (!zero&&s.end_ns==s.begin_ns) || s.end_cycle<=s.begin_cycle) {
      std::ostringstream diagnostic;
      diagnostic<<"invalid CTA timer: begin_ns="<<s.begin_ns<<" end_ns="<<s.end_ns
                <<" begin_cycle="<<s.begin_cycle<<" end_cycle="<<s.end_cycle<<" smid="<<s.smid;
      throw std::runtime_error(diagnostic.str());
    }
    first=std::min(first,s.begin_ns);last=std::max(last,s.end_ns);
  }
  if(last==first&&obs.timer_policy!=TimerPolicy::ZeroWorkOneCta)
    throw std::runtime_error("grid globaltimer envelope must be positive");
  if(!std::isfinite(obs.event_ms) || obs.event_ms<=0 || double(last-first)>obs.event_ms*1.05e6)
    throw std::runtime_error("CUDA event/globaltimer cross-check failed");
  if(obs.errors)throw std::runtime_error("numerical validation failed");
  return {first,last};
}
inline double tail_cv(const std::vector<u64>& samples) {
  if(samples.size()<5)throw std::runtime_error("five warmup samples required");
  double mean=0,var=0;auto start=samples.size()-5;
  for(size_t i=start;i<samples.size();++i)mean+=double(samples[i])/5;
  if(mean<=0)throw std::runtime_error("invalid warmup mean");
  for(size_t i=start;i<samples.size();++i)var+=(double(samples[i])-mean)*(double(samples[i])-mean)/4;
  return std::sqrt(var)/mean;
}
struct Warmup {
  std::vector<u64> samples,samples_cycles;
  std::string basis="elapsed_ns";
  TimerPolicy timer_policy=TimerPolicy::StrictPositiveNs;
  bool converged=false;
};
// execute() retains its allocation, launches exactly the requested measured loop,
// and returns validated timestamps. Warmup correctness is checked as well.
template<class Execute> Warmup warmup(Execute execute) {
  Warmup w;
  for(int i=0;i<30;++i) {
    auto o=execute();auto bounds=envelope(o);w.samples.push_back(bounds.second-bounds.first);
    if(i==0) {
      w.timer_policy=o.timer_policy;
      w.basis=o.timer_policy==TimerPolicy::ZeroWorkOneCta?"cta_clock64_cycles":"elapsed_ns";
    } else if(w.timer_policy!=o.timer_policy)throw std::runtime_error("warmup timer policy changed");
    if(o.stamps.size()==1)w.samples_cycles.push_back(o.stamps[0].end_cycle-o.stamps[0].begin_cycle);
    const auto& selected=w.basis=="cta_clock64_cycles"?w.samples_cycles:w.samples;
    if(w.samples.size()>=8 && tail_cv(selected)<=0.02) {w.converged=true;break;}
  }
  return w;
}
inline void emit_trial(const std::string& id,int iterations,unsigned seed,int threads,
                       const std::string& scope,const std::string& unit,u64 work,u64 read,u64 write,
                       const Observation& o,const Warmup& w,const std::string& extension="") {
  auto bounds=envelope(o);
  if(o.timer_policy!=TimerPolicy::StrictPositiveNs&&(work||read||write))
    throw std::runtime_error("zero-work timer cannot carry measured workload");
  if(o.timer_policy!=w.timer_policy)throw std::runtime_error("trial/warmup timer policy mismatch");
  if(o.timer_policy==TimerPolicy::ZeroWorkOneCta&&scope!="one_cta")
    throw std::runtime_error("cross-CTA cycle control forbidden");
  if(o.timer_policy==TimerPolicy::ZeroWorkGrid&&scope!="all_gpu")
    throw std::runtime_error("grid timer control requires all_gpu scope");
  if(GH_TIMER_RESOLUTION_V3&&scope=="one_cta"&&w.samples_cycles.size()!=w.samples.size())
    throw std::runtime_error("local warmup cycle/ns array lengths differ");
  std::cout<<std::setprecision(17)<<"{\"schema_version\":2,\"type\":\"trial\",\"case_id\":"<<quote(id)
    <<",\"iterations\":"<<iterations<<",\"seed\":"<<seed<<",\"threads\":"<<threads
    <<",\"blocks\":"<<o.stamps.size()<<",\"scope\":"<<quote(scope)<<",\"errors\":"<<o.errors
    <<",\"correctness\":{\"method\":"<<quote(o.method)<<",\"checked_elements\":"<<o.checked_elements
    <<",\"input_conditions\":"<<quote(o.input_conditions)<<"}"
    <<",\"work_unit\":"<<quote(unit)<<",\"work_count\":"<<work
    <<",\"read_payload_bytes\":"<<read<<",\"write_payload_bytes\":"<<write
    <<",\"start_ns\":"<<bounds.first<<",\"stop_ns\":"<<bounds.second<<",\"event_ms\":"<<o.event_ms
    <<",\"warmup_converged\":"<<(w.converged?"true":"false")<<",\"warmup_samples_ns\":[";
  for(size_t i=0;i<w.samples.size();++i)std::cout<<(i?",":"")<<w.samples[i];
  std::cout<<']';
  if(GH_TIMER_RESOLUTION_V3) {
    size_t equal=0;for(auto stamp:o.stamps)equal+=stamp.end_ns==stamp.begin_ns;
    std::cout<<",\"timer_resolution_revision\":\"zero-work-controls-v3\",\"warmup_basis\":"<<quote(w.basis)
      <<",\"globaltimer_distinguishable\":"<<(bounds.second>bounds.first?"true":"false")
      <<",\"zero_ns_ctas\":"<<equal<<",\"warmup_samples_cycles\":[";
    if(scope=="one_cta")for(size_t i=0;i<w.samples_cycles.size();++i)std::cout<<(i?",":"")<<w.samples_cycles[i];
    std::cout<<']';
  }
  std::cout<<",\"blocks_detail\":[";
  for(size_t i=0;i<o.stamps.size();++i) {
    auto s=o.stamps[i];std::cout<<(i?",":"")<<"{\"block_id\":"<<i<<",\"smid\":"<<s.smid
      <<",\"start_ns\":"<<s.begin_ns<<",\"stop_ns\":"<<s.end_ns
      <<",\"start_cycle\":"<<s.begin_cycle<<",\"stop_cycle\":"<<s.end_cycle<<"}";
  }
  std::cout<<"],\"cache_residency_proven\":false,\"physical_hbm_bytes_proven\":false";
  if(!extension.empty())std::cout<<','<<extension;
  std::cout<<"}\n"<<std::flush;
}
}
