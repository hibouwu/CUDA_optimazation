// B01 single-CTA tensor transport with a 128 MiB rotating address set.
#include "r01_support.hpp"
#include <cuda.h>
constexpr int Q=16384,W=64,H=128;
__device__ unsigned address(const void* p){return unsigned(__cvta_generic_to_shared(p));}
template<bool Input>
__global__ void tensor_large(const __grid_constant__ CUtensorMap map,int slots,unsigned generation,uint16_t* last,RStamp* stamp){
  __shared__ __align__(1024) uint16_t tile[Q/2];__shared__ uint64_t barrier;
  for(int q=threadIdx.x;q<Q/2;q+=blockDim.x)tile[q]=Input?0xffff:uint16_t(q*17+19+generation);
  if(threadIdx.x==0&&Input)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(address(&barrier)):"memory");
  asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();
  RStamp s{};if(threadIdx.x==0)stamp_begin(s);__syncthreads();
#pragma unroll 1
  for(int slot=0;slot<slots;++slot){
    if(threadIdx.x==0){int x=0,y=slot*H;
      if constexpr(Input){
        asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0],%1;"::"r"(address(&barrier)),"r"(Q):"memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"::"r"(address(tile)),"l"(&map),"r"(x),"r"(y),"r"(address(&barrier)):"memory");
        uint64_t token;asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0,[%1];":"=l"(token):"r"(address(&barrier)):"memory");
        unsigned done=0;while(!done)asm volatile("{.reg .pred p;mbarrier.try_wait.acquire.cta.shared::cta.b64 p,[%1],%2,64;selp.b32 %0,1,0,p;}":"=r"(done):"r"(address(&barrier)),"l"(token):"memory");
      }else{
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%1,%2}],[%3];"::"l"(&map),"r"(x),"r"(y),"r"(address(tile)):"memory");
        asm volatile("cp.async.bulk.commit_group;cp.async.bulk.wait_group 0;":::"memory");
      }
    }__syncthreads();
  }
  if(threadIdx.x==0)stamp_end(s,stamp);
  for(int q=threadIdx.x;q<Q/2;q+=blockDim.x)last[q]=tile[q];
  __syncthreads();if(threadIdx.x==0&&Input)asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(address(&barrier)):"memory");
}
int main(int argc,char**argv){try{
  int input=1,slots=8192;
  for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("missing value");std::string k=argv[i];if(k=="--input")input=std::stoi(argv[i+1]);else if(k=="--slots")slots=std::stoi(argv[i+1]);else throw std::runtime_error("option");}
  if((input!=0&&input!=1)||(slots!=3&&slots!=8192))throw std::runtime_error("coordinate");
  cudaDeviceProp device{};CUDA_CHECK(cudaGetDeviceProperties(&device,0));if(slots==8192&&uint64_t(slots)*Q<=uint64_t(device.l2CacheSize))throw std::runtime_error("working set not above L2");
  std::vector<uint16_t> host(size_t(slots)*Q/2);
  for(size_t q=0;q<host.size();++q)host[q]=input?uint16_t((q%(Q/2))*17+(q/(Q/2))*73+19):0xffff;
  DeviceBuffer<uint16_t> data(host.size()),last(Q/2);DeviceBuffer<RStamp> stamps(1);RStamp s{};
  CUDA_CHECK(cudaMemcpy(data.pointer,host.data(),host.size()*2,cudaMemcpyHostToDevice));
  alignas(64) CUtensorMap map{};cuuint64_t dims[]={W,cuuint64_t(slots)*H},strides[]={W*2};cuuint32_t box[]={W,H},step[]={1,1};
  if(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT16,2,data.pointer,dims,strides,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)throw std::runtime_error("tensor map");
  cudaFuncAttributes attr{};CUDA_CHECK(cudaFuncGetAttributes(&attr,input?tensor_large<true>:tensor_large<false>));
  unsigned generation=0;
  auto launch=[&](){++generation;if(input)tensor_large<true><<<1,128>>>(map,slots,generation,last.pointer,stamps.pointer);else tensor_large<false><<<1,128>>>(map,slots,generation,last.pointer,stamps.pointer);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(&s,stamps.pointer,sizeof(s),cudaMemcpyDeviceToHost));return s.end_cycle-s.begin_cycle;};
  auto warm=warm_up(launch);auto cycles=launch();std::vector<uint16_t> output(input?Q/2:host.size());
  CUDA_CHECK(cudaMemcpy(output.data(),input?last.pointer:data.pointer,output.size()*2,cudaMemcpyDeviceToHost));save_binary("transport.u16",output);
  std::cout<<"{\"kind\":\"tma\",\"input\":"<<input<<",\"generation\":"<<generation<<",\"slots\":"<<slots<<",\"payload_bytes\":"<<Q<<",\"l2_bytes\":"<<device.l2CacheSize<<",\"cycles\":"<<cycles<<",\"registers\":"<<attr.numRegs<<",\"local_bytes\":"<<attr.localSizeBytes<<",\"first_sm\":"<<s.first_sm<<",\"last_sm\":"<<s.last_sm<<",\"warmup_converged\":"<<(warm.converged?"true":"false")<<",\"warmup\":";vector_json(warm.warmup);std::cout<<"}\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
