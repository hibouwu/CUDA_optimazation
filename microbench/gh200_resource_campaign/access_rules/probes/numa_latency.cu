// Memory-affinity development probe: discard on dedicated scratch lines, then CG cold/hot loads.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#define CHECK(call) do { auto e=(call); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
constexpr int SharedBytes=132096;
struct Record { uint64_t cold,hot; uint32_t value; };
struct Actor { unsigned smid; int selected; uint64_t begin_cycle,end_cycle,begin_ns,end_ns; };
__device__ __forceinline__ uint64_t ns() { uint64_t x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)::"memory");return x; }
__device__ __forceinline__ uint64_t cycles() { uint64_t x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x; }
__device__ __forceinline__ uint64_t timed_load(uint32_t const* address,unsigned shared_address,uint32_t& value) {
  uint64_t duration;
  asm volatile("{.reg .u64 a,b; .reg .u32 v;\n"
    "mov.u64 a, %%clock64;\n"
    "ld.global.cg.u32 v,[%2];\n"
    "st.volatile.shared.u32 [%3],v;\n"
    "mov.u64 b, %%clock64;\n"
    "sub.u64 %0,b,a; mov.u32 %1,v;\n}"
    :"=l"(duration),"=r"(value):"l"(address),"r"(shared_address):"memory");
  return duration;
}
__global__ void probe(uint32_t* allocation,uint32_t const* pairs,int samples,int repeats,
                     int const* selected,int count,Record* records,Actor* actors) {
  extern __shared__ unsigned char reserved[];
  unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
  int owner=-1;
  for(int i=0;i<count;++i)if(sm==unsigned(selected[i]))owner=i;
  if(threadIdx.x==0){actors[blockIdx.x].smid=sm;actors[blockIdx.x].selected=owner;reserved[0]=0;}
  cooperative_groups::this_grid().sync();
  if(owner<0 || threadIdx.x!=0)return;
  unsigned shared_address=unsigned(__cvta_generic_to_shared(reserved));
  Actor& actor=actors[blockIdx.x];actor.begin_cycle=cycles();actor.begin_ns=ns();
  for(int rep=0;rep<repeats;++rep)for(int sample=0;sample<samples;++sample)for(int half=0;half<2;++half){
    size_t offset=size_t(pairs[sample])*8192+half*4096+owner*128;
    uint32_t* address=reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(allocation)+offset);
    asm volatile("discard.global.L2 [%0],128; membar.gl;"::"l"(address):"memory");
    uint32_t value;
    uint64_t cold=timed_load(address,shared_address,value);
    uint32_t const* hot_address=address+(value&1u);
    uint64_t hot=timed_load(hot_address,shared_address,value);
    records[((size_t(owner)*repeats+rep)*samples+sample)*2+half]={cold,hot,value};
  }
  actor.end_ns=ns();actor.end_cycle=cycles();
}
int main(int argc,char**argv)try{
  int samples=512,repeats=3,seed=17;
  for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("missing argument");std::string name=argv[i];int value=std::stoi(argv[i+1]);
    if(name=="--samples")samples=value;else if(name=="--repeats")repeats=value;else if(name=="--seed")seed=value;else throw std::runtime_error("unknown argument");}
  if(samples<1 || samples>8192 || repeats<1)throw std::runtime_error("invalid sample count");
  cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));if(prop.major!=9 || !prop.cooperativeLaunch)throw std::runtime_error("requires cooperative SM90");
  std::vector<int> selected={0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30};
  std::vector<uint32_t> pairs(8192);std::iota(pairs.begin(),pairs.end(),0);std::mt19937 generator(seed);std::shuffle(pairs.begin(),pairs.end(),generator);pairs.resize(samples);
  unsigned char* raw;CHECK(cudaMalloc(&raw,66ull*1024*1024));auto base=(uintptr_t(raw)+2097151)&~uintptr_t(2097151);auto*allocation=reinterpret_cast<uint32_t*>(base);CHECK(cudaMemset(allocation,0,64ull*1024*1024));
  uint32_t*device_pairs;int*device_selected;Record*device_records;Actor*device_actors;
  CHECK(cudaMalloc(&device_pairs,pairs.size()*4));CHECK(cudaMemcpy(device_pairs,pairs.data(),pairs.size()*4,cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&device_selected,selected.size()*sizeof(int)));CHECK(cudaMemcpy(device_selected,selected.data(),selected.size()*sizeof(int),cudaMemcpyHostToDevice));
  size_t nrecords=selected.size()*repeats*samples*2;CHECK(cudaMalloc(&device_records,nrecords*sizeof(Record)));CHECK(cudaMalloc(&device_actors,prop.multiProcessorCount*sizeof(Actor)));
  CHECK(cudaFuncSetAttribute(probe,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));int occupancy;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,probe,128,SharedBytes));if(occupancy!=1)throw std::runtime_error("requires one CTA per SM");
  int count=selected.size();void*args[]={&allocation,&device_pairs,&samples,&repeats,&device_selected,&count,&device_records,&device_actors};CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(probe),prop.multiProcessorCount,128,args,SharedBytes));CHECK(cudaDeviceSynchronize());
  std::vector<Record> data(nrecords);std::vector<Actor> actors(prop.multiProcessorCount);CHECK(cudaMemcpy(data.data(),device_records,nrecords*sizeof(Record),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(actors.data(),device_actors,actors.size()*sizeof(Actor),cudaMemcpyDeviceToHost));
  std::vector<int> counts(selected.size());for(auto const&a:actors)if(a.selected>=0)++counts[a.selected];for(int n:counts)if(n!=1)throw std::runtime_error("selected SM coverage mismatch");
  cudaFuncAttributes attr;CHECK(cudaFuncGetAttributes(&attr,probe));auto u=reinterpret_cast<unsigned char*>(prop.uuid.bytes);char uuid[80];std::snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  std::cout<<"{\"schema\":\"gh200-memory-affinity/1\",\"gpu_uuid\":\""<<uuid<<"\",\"allocation_address\":"<<base<<",\"allocation_bytes\":67108864,\"samples\":"<<samples<<",\"repeats\":"<<repeats<<",\"seed\":"<<seed<<",\"registers\":"<<attr.numRegs<<",\"local_bytes\":"<<attr.localSizeBytes<<",\"smem\":"<<SharedBytes<<",\"occupancy\":"<<occupancy<<",\"pairs\":[";
  for(int i=0;i<samples;++i){if(i)std::cout<<',';std::cout<<pairs[i];}std::cout<<"],\"actors\":[";bool first=true;
  for(auto const&a:actors)if(a.selected>=0){if(!first)std::cout<<',';first=false;std::cout<<"{\"index\":"<<a.selected<<",\"smid\":"<<a.smid<<",\"cycles\":"<<a.end_cycle-a.begin_cycle<<",\"ns\":"<<a.end_ns-a.begin_ns<<"}";}
  std::cout<<"],\"records\":[";for(size_t i=0;i<nrecords;++i){if(i)std::cout<<',';std::cout<<'['<<data[i].cold<<','<<data[i].hot<<','<<data[i].value<<']';}std::cout<<"]}\n";
  CHECK(cudaFree(raw));CHECK(cudaFree(device_pairs));CHECK(cudaFree(device_selected));CHECK(cudaFree(device_records));CHECK(cudaFree(device_actors));
}catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}
