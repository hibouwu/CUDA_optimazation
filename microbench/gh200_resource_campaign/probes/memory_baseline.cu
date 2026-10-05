#include "../common/probe_runtime.cuh"
// Derived from v1. An entry timing barrier prevents a leading warp from
// issuing counted work before the recorded thread-0 start timestamp.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>
#define CK(call) do { cudaError_t probe_cuda_status_=(call); if(probe_cuda_status_!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(probe_cuda_status_));exit(2);} } while(0)
using Stamp = gh::Stamp;
__device__ unsigned long long ns() { unsigned long long x; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(x)); return x; }
__device__ unsigned smid() { unsigned x; asm volatile("mov.u32 %0, %%smid;" : "=r"(x)); return x; }
__device__ unsigned value(size_t i, unsigned seed) { return unsigned(i)*17u+seed; }
__global__ void initialize(unsigned *a, size_t n, unsigned seed) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) a[i]=value(i,seed);
}
__global__ void validate_store(const unsigned *a,size_t n,unsigned seed,unsigned long long *errors) {
  unsigned long long local=0;
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)
    local += a[i] != unsigned(i)*17u+seed;
  if(local) atomicAdd(errors,local);
}
__device__ uint4 load4(const uint4 *p,bool ca) {
  uint4 v;
  if(ca) asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p) : "memory");
  else asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ void store4(uint4 *p,uint4 v) {
  asm volatile("st.global.wb.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p),"r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w) : "memory");
}
__global__ void global_path(const uint4 *a,uint4 *b,size_t per_block,int iters,int mode,bool ca,unsigned seed,Stamp *stamp,unsigned *checksum) {
  unsigned sum=0;
  __syncthreads();
  auto c0=static_cast<unsigned long long>(clock64()); auto t0=ns();
  __syncthreads();
  for(int it=0;it<iters;++it) {
    for(size_t j=threadIdx.x;j<per_block;j+=blockDim.x) {
      size_t i=size_t(blockIdx.x)*per_block+j;
      uint4 v;
      if(mode!=1) {
        v=load4(a+i,ca);
        sum += v.x+v.y+v.z+v.w;
      } else v=make_uint4(value(i*4,seed),value(i*4+1,seed),value(i*4+2,seed),value(i*4+3,seed));
      if(mode!=0) store4(b+i,v);
    }
  }
  if(mode!=0) __threadfence();
  __syncthreads();
  auto t1=ns(); auto c1=static_cast<unsigned long long>(clock64());
  if(threadIdx.x==0) stamp[blockIdx.x]={t0,t1,c0,c1,smid()};
  checksum[blockIdx.x*blockDim.x+threadIdx.x]=sum;
}
template<bool Write>
__global__ void shared_path(int iters,int stride,Stamp *stamp,unsigned *checksum) {
  __shared__ unsigned data[8192];
  for(int i=threadIdx.x;i<8192;i+=blockDim.x) data[i]=Write ? 0xdeadbeefu : unsigned(i)*17u+3u;
  __syncthreads();
  unsigned sum=0;
  auto c0=static_cast<unsigned long long>(clock64());auto t0=ns();
  __syncthreads();
  #pragma unroll 1
  for(int it=0;it<iters;++it) {
    #pragma unroll
    for(int q=0;q<8;++q) {
      unsigned i=((threadIdx.x+q*blockDim.x)*stride)&8191u;
      unsigned addr=unsigned(__cvta_generic_to_shared(data+i)),v;
      if constexpr(Write) {
        v=i*17u+3u;
        asm volatile("st.volatile.shared.u32 [%0], %1;" :: "r"(addr),"r"(v) : "memory");
      } else {
        asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
        sum+=v;
      }
    }
  }
  __syncthreads();
  auto t1=ns();auto c1=static_cast<unsigned long long>(clock64());
  if(threadIdx.x==0) stamp[blockIdx.x]={t0,t1,c0,c1,smid()};
  if constexpr(Write) {
    for(int q=0;q<8;++q) {unsigned i=threadIdx.x+q*blockDim.x;sum+=data[i];}
  }
  checksum[blockIdx.x*blockDim.x+threadIdx.x]=sum;
}

__global__ void empty_window(Stamp *stamp) {
  __syncthreads();
  auto c0=static_cast<unsigned long long>(clock64());auto t0=ns();
  __syncthreads();
  auto t1=ns();auto c1=static_cast<unsigned long long>(clock64());
  if(threadIdx.x==0)stamp[blockIdx.x]={t0,t1,c0,c1,smid()};
}
struct Case {
  std::string id,mode,scope;size_t bytes;int iterations,stride;
};
std::vector<Case> cases() {
  std::vector<Case> c;
  for(int s:{1,2,4,8,16,32})c.push_back({"smem_read_stride"+std::to_string(s),"smem_read","one_cta",32768,8192,s});
  c.push_back({"smem_write_stride1","smem_write","one_cta",32768,8192,1});
  c.push_back({"global_read_ca_8m","global_read_ca","all_gpu",8*1024*1024,64,1});
  c.push_back({"global_read_cg_8m","global_read_cg","all_gpu",8*1024*1024,64,1});
  c.push_back({"global_read_cg_256m","global_read_cg","all_gpu",256*1024*1024,16,1});
  c.push_back({"global_write_256m","global_write","all_gpu",256*1024*1024,16,1});
  c.push_back({"global_duplex_128m","global_duplex","all_gpu",128*1024*1024,16,1});
  c.push_back({"empty_one_cta","empty","one_cta",0,1,1});
  c.push_back({"empty_all_gpu","empty","all_gpu",0,1,1});
  return c;
}
int main(int argc,char** argv) try {
  auto device=gh::device();gh::emit_device(device);
  if(argc==2 && std::string(argv[1])=="device")return 0;
  if(argc!=4)throw std::runtime_error("CASE_ID ITERATIONS SEED");
  auto table=cases();auto found=std::find_if(table.begin(),table.end(),[&](const Case& c){return c.id==argv[1];});
  if(found==table.end())throw std::runtime_error("unknown case");
  Case c=*found;int iters=gh::integer(argv[2],1,65536);unsigned seed=gh::integer(argv[3],1,1000000);
  if(iters!=c.iterations)throw std::runtime_error("frozen iterations mismatch");
  const auto& p=device.prop;
  bool shared=c.mode=="smem_read" || c.mode=="smem_write",empty=c.mode=="empty";
  bool ca=c.mode=="global_read_ca";int operation=c.mode=="global_write"?1:c.mode=="global_duplex"?2:0;
  const int threads=256,blocks=c.scope=="one_cta"?1:p.multiProcessorCount*4;
  size_t per_block=(c.bytes+size_t(blocks)*threads*16-1)/(size_t(blocks)*threads*16)*threads;
  size_t words=per_block*blocks*4,actual=words*4;
  uint4 *a=nullptr,*b=nullptr;Stamp *ds;unsigned *dc;unsigned long long *de;
  GH_CUDA(cudaMalloc(&ds,blocks*sizeof(Stamp)));
  GH_CUDA(cudaMalloc(&dc,size_t(blocks)*threads*sizeof(unsigned)));
  GH_CUDA(cudaMalloc(&de,sizeof(unsigned long long)));
  if(!shared&&!empty) {
    GH_CUDA(cudaMalloc(&a,actual));GH_CUDA(cudaMalloc(&b,actual));
    initialize<<<p.multiProcessorCount*4,256>>>((unsigned*)a,words,seed);
    GH_CUDA(cudaGetLastError());GH_CUDA(cudaMemset(b,0,actual));GH_CUDA(cudaDeviceSynchronize());
  }
  cudaEvent_t begin,end;GH_CUDA(cudaEventCreate(&begin));GH_CUDA(cudaEventCreate(&end));
  auto execute=[&]() {
    GH_CUDA(cudaEventRecord(begin));
    if(empty)empty_window<<<blocks,threads>>>(ds);
    else if(shared) {
      if(c.mode=="smem_write")shared_path<true><<<blocks,threads>>>(iters,c.stride,ds,dc);
      else shared_path<false><<<blocks,threads>>>(iters,c.stride,ds,dc);
    } else global_path<<<blocks,threads>>>(a,b,per_block,iters,operation,ca,seed,ds,dc);
    GH_CUDA(cudaGetLastError());GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));
    gh::Observation obs;float ms=0;GH_CUDA(cudaEventElapsedTime(&ms,begin,end));obs.event_ms=ms;
    obs.timer_policy=gh::control_timer_policy(empty,c.scope=="one_cta");
    obs.stamps.resize(blocks);GH_CUDA(cudaMemcpy(obs.stamps.data(),ds,blocks*sizeof(Stamp),cudaMemcpyDeviceToHost));
    if(!empty) {
      std::vector<unsigned> sums(size_t(blocks)*threads);
      GH_CUDA(cudaMemcpy(sums.data(),dc,sums.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
      for(int block=0;block<blocks;++block)for(int t=0;t<threads;++t) {
        uint64_t expected=0;
        if(shared) {
          for(int q=0;q<8;++q)expected+=uint32_t((((t+q*threads)*c.stride)&8191u)*17u+3u);
          if(c.mode=="smem_read")expected*=iters;
        } else if(operation!=1) {
          uint64_t n=per_block/threads,first=uint64_t(block)*per_block+t;
          expected=(272u*(n*first+uint64_t(threads)*n*(n-1)/2)+(102u+uint64_t(seed)*4)*n)*iters;
        }
        obs.errors+=sums[block*threads+t]!=uint32_t(expected);++obs.checked_elements;
      }
      if(!shared&&operation!=0) {
        GH_CUDA(cudaMemset(de,0,sizeof(unsigned long long)));
        validate_store<<<p.multiProcessorCount*4,256>>>((unsigned*)b,words,seed,de);GH_CUDA(cudaGetLastError());
        unsigned long long errors=0;GH_CUDA(cudaMemcpy(&errors,de,sizeof(errors),cudaMemcpyDeviceToHost));
        obs.errors+=errors;obs.checked_elements+=words;
      }
    }
    obs.method=empty?"empty_window_timestamps":shared?"host_per_thread_address_checksum":"host_checksum_and_optional_full_array_store_validation";
    obs.input_conditions=shared?"shared[i]=17*i+3; write initialization poison":empty?"no memory workload":"global[i]=17*i+seed modulo 2^32";
    if(c.scope=="all_gpu"&&!empty) {
      std::set<unsigned> ids;for(auto s:obs.stamps)ids.insert(s.smid);
      if(int(ids.size())!=p.multiProcessorCount)throw std::runtime_error("full GPU SM coverage incomplete");
    }
    gh::envelope(obs);return obs;
  };
  auto warm=gh::warmup(execute);
  auto obs=execute();gh::u64 rb=0,wb=0;
  if(!empty) {
    if(shared) {gh::u64 amount=gh::u64(blocks)*threads*8*4*iters;if(c.mode=="smem_read")rb=amount;else wb=amount;}
    else {if(operation!=1)rb=gh::u64(actual)*iters;if(operation!=0)wb=gh::u64(actual)*iters;}
  }
  std::ostringstream extra;extra<<"\"requested_working_set_bytes\":"<<c.bytes<<",\"allocation_per_array_bytes\":"<<(shared?32768:empty?0:actual)<<",\"stride\":"<<c.stride;
  gh::emit_trial(c.id,iters,seed,threads,c.scope,empty?"operation":"byte",rb+wb,rb,wb,obs,warm,extra.str());
  GH_CUDA(cudaEventDestroy(begin));GH_CUDA(cudaEventDestroy(end));
  GH_CUDA(cudaFree(ds));GH_CUDA(cudaFree(dc));GH_CUDA(cudaFree(de));
  if(a)GH_CUDA(cudaFree(a));if(b)GH_CUDA(cudaFree(b));
  return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 2;}
