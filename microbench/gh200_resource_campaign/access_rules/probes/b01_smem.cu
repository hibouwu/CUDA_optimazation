// B01 independent LDS.128 + STS.128, disjoint address regions and values.
#include "r01_support.hpp"
__global__ void joint_smem(int iterations,unsigned* saved,unsigned* sums,RStamp* stamp){
  extern __shared__ unsigned memory[];
  int t=threadIdx.x,lane=t%32,warp=t/32,words=blockDim.x*32;
  for(int i=t;i<words;i+=blockDim.x){memory[i]=17u*i+19;memory[words+i]=0xdeadbeef;}
  __syncthreads();RStamp s{};if(t==0)stamp_begin(s);__syncthreads();
  unsigned sum=0;
#pragma unroll 1
  for(int i=0;i<iterations;++i){
#pragma unroll
    for(int slot=0;slot<8;++slot){
      unsigned index=(warp*8+slot)*128+lane*4;
      unsigned rd=unsigned(__cvta_generic_to_shared(memory+index));
      unsigned wr=unsigned(__cvta_generic_to_shared(memory+words+index));
      unsigned a,b,c,d;
      asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(a),"=r"(b),"=r"(c),"=r"(d):"r"(rd):"memory");
      // Store values have no dependence on the read; the read has its own sink.
      unsigned x=unsigned(i)*31+index+7;
      asm volatile("st.volatile.shared.v4.u32 [%0],{%1,%2,%3,%4};"::"r"(wr),"r"(x),"r"(x+1),"r"(x+2),"r"(x+3):"memory");
      sum+=a+b+c+d;
    }
  }
  // Retain each read consumer before CTA join and the final clock.
  __syncthreads();reinterpret_cast<volatile unsigned*>(memory)[t]=sum;__syncthreads();if(t==0)stamp_end(s,stamp);
  sums[t]=sum;
  for(int i=t;i<words;i+=blockDim.x)saved[i]=memory[words+i];
}
int main(int argc,char**argv){try{
  int warps=1,iterations=256;
  for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("missing value");std::string k=argv[i];if(k=="--warps")warps=std::stoi(argv[i+1]);else if(k=="--iterations")iterations=std::stoi(argv[i+1]);else throw std::runtime_error("option");}
  if((warps!=1&&warps!=8)||iterations<1)throw std::runtime_error("coordinate");
  int threads=warps*32,words=threads*32;DeviceBuffer<unsigned> saved(words),sums(threads);DeviceBuffer<RStamp> stamps(1);RStamp s{};
  CUDA_CHECK(cudaFuncSetAttribute(joint_smem,cudaFuncAttributeMaxDynamicSharedMemorySize,words*8));
  cudaFuncAttributes attr{};CUDA_CHECK(cudaFuncGetAttributes(&attr,joint_smem));
  auto launch=[&](){joint_smem<<<1,threads,words*8>>>(iterations,saved.pointer,sums.pointer,stamps.pointer);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(&s,stamps.pointer,sizeof(s),cudaMemcpyDeviceToHost));return s.end_cycle-s.begin_cycle;};
  auto warm=warm_up(launch);auto cycles=launch();std::vector<unsigned> out(words),sink(threads);
  CUDA_CHECK(cudaMemcpy(out.data(),saved.pointer,words*4,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(sink.data(),sums.pointer,threads*4,cudaMemcpyDeviceToHost));
  save_binary("stored.u32",out);save_binary("sums.u32",sink);
  std::cout<<"{\"kind\":\"smem\",\"warps\":"<<warps<<",\"iterations\":"<<iterations<<",\"cycles\":"<<cycles<<",\"registers\":"<<attr.numRegs<<",\"local_bytes\":"<<attr.localSizeBytes<<",\"first_sm\":"<<s.first_sm<<",\"last_sm\":"<<s.last_sm<<",\"warmup_converged\":"<<(warm.converged?"true":"false")<<",\"warmup\":";vector_json(warm.warmup);std::cout<<"}\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
