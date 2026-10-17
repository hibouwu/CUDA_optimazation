// Cluster co-residency identifies same-GPC edges; no latency price is fitted.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace cg = cooperative_groups;
#define CUDA_CHECK(call) do {auto error=(call); if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));} while(0)
struct Placement {unsigned block, cluster, rank, smid, check;};
constexpr int shared_bytes=132096;

__global__ void placement_probe(Placement* output, unsigned cluster_size) {
  extern __shared__ unsigned data[];
  data[threadIdx.x]=blockIdx.x+threadIdx.x;
  auto cluster=cg::this_cluster();
  cluster.sync();
  unsigned smid;
  asm volatile("mov.u32 %0, %%smid;":"=r"(smid));
  uint64_t begin=clock64();
  while(clock64()-begin<100000) {}
  if(threadIdx.x==0) output[blockIdx.x]={unsigned(blockIdx.x),unsigned(blockIdx.x)/cluster_size,
    unsigned(cluster.block_rank()),smid,data[127]};
  cluster.sync();
}

int main() try {
  cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
  CUDA_CHECK(cudaFuncSetAttribute(placement_probe,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
  int occupancy;CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,placement_probe,128,shared_bytes));
  if(occupancy!=1) throw std::runtime_error("expected one CTA per SM");
  char uuid[80];auto u=reinterpret_cast<unsigned char*>(prop.uuid.bytes);
  std::snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
    u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  Placement* device;CUDA_CHECK(cudaMalloc(&device,512*sizeof(Placement)));
  for(unsigned size=2;size<=8;++size) {
    cudaLaunchAttribute attribute{};attribute.id=cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x=size;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;
    cudaLaunchConfig_t config{};config.gridDim=dim3(size*64);config.blockDim=dim3(128);
    config.dynamicSmemBytes=shared_bytes;config.attrs=&attribute;config.numAttrs=1;
    for(unsigned repeat=0;repeat<3;++repeat) {
      CUDA_CHECK(cudaMemset(device,0xff,size*64*sizeof(Placement)));
      CUDA_CHECK(cudaLaunchKernelEx(&config,placement_probe,device,size));
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<Placement> host(size*64);
      CUDA_CHECK(cudaMemcpy(host.data(),device,host.size()*sizeof(Placement),cudaMemcpyDeviceToHost));
      std::cout<<"{\"gpu_uuid\":\""<<uuid<<"\",\"sm_count\":"<<prop.multiProcessorCount
        <<",\"cluster_size\":"<<size<<",\"repeat\":"<<repeat<<",\"occupancy\":"<<occupancy<<",\"placements\":[";
      for(unsigned i=0;i<host.size();++i) {
        auto p=host[i];
        if(p.block!=i || p.cluster!=i/size || p.rank!=i%size || p.check!=i+127 || p.smid>=unsigned(prop.multiProcessorCount))
          throw std::runtime_error("placement record mismatch");
        if(i)std::cout<<',';
        std::cout<<"["<<p.block<<','<<p.cluster<<','<<p.rank<<','<<p.smid<<"]";
      }
      std::cout<<"]}\n";
    }
  }
  CUDA_CHECK(cudaFree(device));
} catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
