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
struct Stamp { unsigned long long begin_ns,end_ns,begin_cycle,end_cycle; unsigned smid; };
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
int attribute(cudaDeviceAttr a) {int v=0;CK(cudaDeviceGetAttribute(&v,a,0));return v;}
void print_device(const cudaDeviceProp &p) {
  char uuid[33];
  for(int i=0;i<16;++i) snprintf(uuid+2*i,3,"%02x",static_cast<unsigned char>(p.uuid.bytes[i]));
  printf("{\"type\":\"device\",\"name\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"global_memory_bytes\":%zu,\"l2_cache_bytes\":%d,\"registers_per_sm\":%d,\"smem_per_sm_bytes\":%zu,\"smem_per_cta_optin_bytes\":%zu,\"max_threads_per_sm\":%d,\"max_blocks_per_sm\":%d,\"cluster_launch\":%d,\"peak_clock_rate_khz\":%d,\"peak_memory_clock_rate_khz\":%d,\"memory_bus_width_bits\":%d,\"warp_size\":%d,\"uuid_hex\":\"%s\"}\n",
    p.name,p.major,p.minor,p.multiProcessorCount,p.totalGlobalMem,p.l2CacheSize,p.regsPerMultiprocessor,p.sharedMemPerMultiprocessor,p.sharedMemPerBlockOptin,p.maxThreadsPerMultiProcessor,
    attribute(cudaDevAttrMaxBlocksPerMultiprocessor),attribute(cudaDevAttrClusterLaunch),attribute(cudaDevAttrClockRate),attribute(cudaDevAttrMemoryClockRate),p.memoryBusWidth,p.warpSize,uuid);
}
int main(int argc,char **argv) {
  int count=0;CK(cudaGetDeviceCount(&count));if(count!=1){fprintf(stderr,"Expected exactly one allocated GPU\n");return 1;}
  cudaDeviceProp p{};CK(cudaGetDeviceProperties(&p,0));
  if(p.major!=9 || p.minor!=0 || !strstr(p.name,"GH200")) {fprintf(stderr,"Expected GH200 SM90\n");return 1;}
  print_device(p); if(argc==1 || std::string(argv[1])=="device")return 0;
  if(argc!=7){fprintf(stderr,"mode bytes iterations stride trial seed\n");return 1;}
  std::string mode=argv[1];size_t requested=strtoull(argv[2],nullptr,10);int iters=atoi(argv[3]),stride=atoi(argv[4]),trial=atoi(argv[5]);unsigned seed=strtoul(argv[6],nullptr,10);
  bool shared=mode=="smem_read" || mode=="smem_write";
  bool ca=mode=="global_read_ca";int operation=(mode=="global_write"?1:mode=="global_duplex"?2:0);
  if(!shared && mode!="global_read_ca" && mode!="global_read_cg" && mode!="global_write" && mode!="global_duplex")return 1;
  if(iters<1 || iters>65536 || stride<1 || stride>32 || trial<0 || requested>size_t(512)*1024*1024)return 1;
  if(shared && mode=="smem_write" && stride!=1)return 1;
  const int threads=256,blocks=shared?1:p.multiProcessorCount*4;
  size_t per_block=(requested+size_t(blocks)*threads*16-1)/(size_t(blocks)*threads*16)*threads;
  size_t words=per_block*blocks*4, actual=words*4;
  if(!shared && per_block==0)return 1;
  uint4 *a=nullptr,*b=nullptr;Stamp *ds;unsigned *dc;unsigned long long *de;
  CK(cudaMalloc(&ds,blocks*sizeof(Stamp)));CK(cudaMalloc(&dc,size_t(blocks)*threads*sizeof(unsigned)));CK(cudaMalloc(&de,sizeof(unsigned long long)));
  if(!shared) {
    CK(cudaMalloc(&a,actual));CK(cudaMalloc(&b,actual));
    initialize<<<p.multiProcessorCount*4,256>>>((unsigned*)a,words,seed);CK(cudaGetLastError());
    CK(cudaMemset(b,0,actual));CK(cudaDeviceSynchronize());
  }
  auto launch=[&](int iterations) {
    if(shared) {
      if(mode=="smem_write")shared_path<true><<<blocks,threads>>>(iterations,stride,ds,dc);
      else shared_path<false><<<blocks,threads>>>(iterations,stride,ds,dc);
    } else global_path<<<blocks,threads>>>(a,b,per_block,iterations,operation,ca,seed,ds,dc);
    CK(cudaGetLastError());
  };
  // Warmup traverses the same allocation; cache residency is not asserted.
  launch(2);launch(2);CK(cudaDeviceSynchronize());
  cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
  CK(cudaEventRecord(begin));launch(iters);CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));
  float event_ms;CK(cudaEventElapsedTime(&event_ms,begin,end));
  std::vector<Stamp> stamps(blocks);std::vector<unsigned> sums(size_t(blocks)*threads);
  CK(cudaMemcpy(stamps.data(),ds,blocks*sizeof(Stamp),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sums.data(),dc,sums.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
  unsigned long long errors=0;
  for(int block=0;block<blocks;++block)for(int t=0;t<threads;++t) {
    uint64_t expected=0;
    if(shared) {
      for(int q=0;q<8;++q) expected+=uint32_t((((t+q*threads)*stride)&8191u)*17u+3u);
      if(mode=="smem_read")expected*=iters;
    } else if(operation!=1) {
      uint64_t n=per_block/threads;
      uint64_t first=uint64_t(block)*per_block+t;
      expected=(272u*(n*first+uint64_t(threads)*n*(n-1)/2)+(102u+uint64_t(seed)*4)*n)*iters;
    }
    errors+=sums[block*threads+t] != uint32_t(expected);
  }
  if(!shared && operation!=0) {
    CK(cudaMemset(de,0,sizeof(unsigned long long)));
    validate_store<<<p.multiProcessorCount*4,256>>>((unsigned*)b,words,seed,de);CK(cudaGetLastError());
    unsigned long long e;CK(cudaMemcpy(&e,de,sizeof(e),cudaMemcpyDeviceToHost));errors+=e;
  }
  uint64_t first=~0ull,last=0,maxcycles=0;std::set<unsigned> ids;
  for(auto s:stamps){if(s.end_ns<=s.begin_ns || s.end_cycle<=s.begin_cycle)return 4;first=std::min(first,uint64_t(s.begin_ns));last=std::max(last,uint64_t(s.end_ns));maxcycles=std::max(maxcycles,uint64_t(s.end_cycle-s.begin_cycle));ids.insert(s.smid);}
  uint64_t readbytes=0,writebytes=0;
  if(shared) {uint64_t bytes=uint64_t(blocks)*threads*8*4*iters;if(mode=="smem_read")readbytes=bytes;else writebytes=bytes;}
  else {if(operation!=1)readbytes=uint64_t(actual)*iters;if(operation!=0)writebytes=uint64_t(actual)*iters;}
  printf("{\"type\":\"trial\",\"mode\":\"%s\",\"trial\":%d,\"seed\":%u,\"threads\":%d,\"blocks\":%d,\"expected_sms\":%d,\"observed_sms\":%zu,\"requested_working_set_bytes\":%zu,\"allocation_per_array_bytes\":%zu,\"iterations\":%d,\"stride\":%d,\"read_payload_bytes\":%llu,\"write_payload_bytes\":%llu,\"start_ns\":%llu,\"stop_ns\":%llu,\"max_cta_cycles\":%llu,\"event_ms\":%.9g,\"payload_gbytes_per_second\":%.12g,\"errors\":%llu,\"cache_residency_proven\":false,\"blocks_detail\":[",
    mode.c_str(),trial,seed,threads,blocks,shared?1:p.multiProcessorCount,ids.size(),requested,shared?size_t(32768):actual,iters,stride,
    (unsigned long long)readbytes,(unsigned long long)writebytes,(unsigned long long)first,(unsigned long long)last,(unsigned long long)maxcycles,event_ms,double(readbytes+writebytes)/double(last-first),errors);
  for(int i=0;i<blocks;++i){auto s=stamps[i];printf("%s{\"smid\":%u,\"start_ns\":%llu,\"stop_ns\":%llu,\"start_cycle\":%llu,\"stop_cycle\":%llu}",i?",":"",s.smid,s.begin_ns,s.end_ns,s.begin_cycle,s.end_cycle);}
  printf("]}\n");
  CK(cudaFree(ds));CK(cudaFree(dc));CK(cudaFree(de));if(a)CK(cudaFree(a));if(b)CK(cudaFree(b));
  CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));
  return errors?3:0;
}
