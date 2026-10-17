// Small same/cross-GPC reference contrasts for unresolved SM pairs.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
namespace cg=cooperative_groups;
#define CUDA_CHECK(call) do {auto error=(call);if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));} while(0)
constexpr int shared_bytes=132096,threads=1024,words=16384,rounds=256;
struct Record {uint64_t begin_ns,end_ns,begin_c,end_c;unsigned smid;};
__device__ uint64_t ns(){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
__global__ void read_probe(unsigned const* data,int const* actors,Record* records,unsigned* sums,uint64_t* gate) {
  extern __shared__ unsigned scratch[];
  scratch[threadIdx.x]=threadIdx.x;
  unsigned smid;asm volatile("mov.u32 %0, %%smid;":"=r"(smid));
  int actor=actors[smid];
  auto grid=cg::this_grid();grid.sync();
  if(blockIdx.x==0&&threadIdx.x==0)*gate=ns()+100000;
  grid.sync();
  uint64_t start=*gate;
  if(actor<0)return;
  while(ns()<start){}
  __syncthreads();
  uint64_t begin_ns=ns(),begin_c=clock64();
  unsigned sum=0,index[4];
  #pragma unroll
  for(int j=0;j<4;++j)index[j]=threadIdx.x*4+j*threads*4;
  for(int repeat=0;repeat<rounds;++repeat) {
    unsigned values[16];
    #pragma unroll
    for(int j=0;j<4;++j) {
      auto pointer=data+actor*words+index[j];
      asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(values[j*4]),"=r"(values[j*4+1]),"=r"(values[j*4+2]),"=r"(values[j*4+3]):"l"(pointer):"memory");
    }
    #pragma unroll
    for(int j=0;j<16;++j)sum+=values[j];
    #pragma unroll
    for(int j=0;j<4;++j)index[j]=values[j*4];
  }
  __syncthreads();
  uint64_t end_c=clock64(),end_ns=ns();
  sums[actor*threads+threadIdx.x]=sum+scratch[threadIdx.x];
  if(threadIdx.x==0)records[actor]={begin_ns,end_ns,begin_c,end_c,smid};
}
int main(int argc,char** argv) try {
  int process_repeats=argc>1?std::stoi(argv[1]):7;
  cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
  if(prop.multiProcessorCount!=132)throw std::runtime_error("reference 132-SM device required");
  CUDA_CHECK(cudaFuncSetAttribute(read_probe,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
  int occupancy;CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,read_probe,threads,shared_bytes));
  if(occupancy!=1)throw std::runtime_error("one CTA per SM required");
  char uuid[80];auto u=reinterpret_cast<unsigned char*>(prop.uuid.bytes);
  std::snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
    u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  unsigned *data,*sums;int* actors;Record* records;uint64_t* gate;
  CUDA_CHECK(cudaMalloc(&data,8*words*4));CUDA_CHECK(cudaMalloc(&sums,8*threads*4));
  CUDA_CHECK(cudaMalloc(&actors,132*4));CUDA_CHECK(cudaMalloc(&records,8*sizeof(Record)));CUDA_CHECK(cudaMalloc(&gate,8));
  std::vector<unsigned> input(8*words);
  for(unsigned i=0;i<input.size();++i)input[i]=i%4==0?4*((i/4+128)%(words/4)):17*i+3;
  CUDA_CHECK(cudaMemcpy(data,input.data(),input.size()*4,cudaMemcpyHostToDevice));
  std::vector<std::vector<int>> conditions={{0,1,16,17,32,33},
    {0,1,16,17,32,33,48,49},{0,1,16,17,32,33,2,3},
    {0,1,16,17,32,33,124,125},{0,1,16,17,32,33,126,127},
    {0,1,16,17,32,33,128,129},{0,1,16,17,32,33,130,131}};
  std::mt19937 random(20261015);
  for(int repeat=0;repeat<process_repeats;++repeat) {
    std::vector<int> order={0,1,2,3,4,5,6};std::shuffle(order.begin(),order.end(),random);
    for(int condition:order) {
      auto const& selected=conditions[condition];std::vector<int> mapping(132,-1);
      for(unsigned a=0;a<selected.size();++a)mapping[selected[a]]=a;
      CUDA_CHECK(cudaMemcpy(actors,mapping.data(),132*4,cudaMemcpyHostToDevice));
      void* arguments[]={&data,&actors,&records,&sums,&gate};
      for(int launch=0;launch<4;++launch) {
        CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(read_probe),dim3(132),dim3(threads),arguments,shared_bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      std::vector<Record> timing(selected.size());std::vector<unsigned> output(selected.size()*threads);
      CUDA_CHECK(cudaMemcpy(timing.data(),records,timing.size()*sizeof(Record),cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(output.data(),sums,output.size()*4,cudaMemcpyDeviceToHost));
      for(unsigned a=0;a<selected.size();++a)for(unsigned t=0;t<threads;++t) {
        unsigned expected=0;
        for(unsigned j=0;j<4;++j) {
          unsigned index=t*4+j*threads*4;
          for(unsigned step=0;step<32;++step) {
            for(unsigned v=0;v<4;++v)expected+=input[a*words+index+v];
            index=input[a*words+index];
          }
        }
        expected=expected*(rounds/32)+t;
        if(output[a*threads+t]!=expected||timing[a].smid!=unsigned(selected[a]))throw std::runtime_error("load checksum or placement failed");
      }
      std::cout<<"{\"gpu_uuid\":\""<<uuid<<"\",\"condition\":"<<condition<<",\"repeat\":"<<repeat
        <<",\"bytes_per_actor\":"<<uint64_t(words)*4*rounds<<",\"errors\":0,\"actors\":[";
      for(unsigned a=0;a<timing.size();++a) {if(a)std::cout<<',';auto r=timing[a];
        std::cout<<'['<<r.smid<<','<<r.begin_ns<<','<<r.end_ns<<','<<r.begin_c<<','<<r.end_c<<']';}
      std::cout<<"]}\n";
    }
  }
  CUDA_CHECK(cudaFree(data));CUDA_CHECK(cudaFree(sums));CUDA_CHECK(cudaFree(actors));CUDA_CHECK(cudaFree(records));CUDA_CHECK(cudaFree(gate));
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
