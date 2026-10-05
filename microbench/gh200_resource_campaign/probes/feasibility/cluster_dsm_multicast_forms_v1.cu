// S17 A-period feasibility only. No launcher, query CLI or benchmark admission.
// PTX 8.8 cp.async.bulk multicast and mbarrier.arrive.expect_tx forms:
// https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
namespace cg = cooperative_groups;
using u64 = unsigned long long;
constexpr unsigned threads = 128;
constexpr unsigned dsm_words = 1024;
constexpr unsigned bulk_words = 4096;
struct FormStamp {u64 begin_ns,end_ns,begin_cycle,end_cycle;unsigned smid;};
__device__ __forceinline__ unsigned shared_address(const void* pointer) {
  return unsigned(__cvta_generic_to_shared(pointer));
}
__device__ __forceinline__ u64 global_time() {
  u64 value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory");return value;
}
__device__ __forceinline__ u64 cycles() {
  u64 value;asm volatile("mov.u64 %0, %%clock64;":"=l"(value)::"memory");return value;
}
__device__ __forceinline__ FormStamp start_stamp() {
  FormStamp stamp{};
  if(threadIdx.x==0) {
    asm volatile("mov.u32 %0, %%smid;":"=r"(stamp.smid));
    stamp.begin_ns=global_time();stamp.begin_cycle=cycles();
  }
  return stamp;
}
__device__ __forceinline__ void end_stamp(FormStamp stamp,FormStamp* output) {
  if(threadIdx.x==0) {
    stamp.end_cycle=cycles();stamp.end_ns=global_time();output[blockIdx.x]=stamp;
  }
}
__device__ __forceinline__ void export_guards(const unsigned* storage,unsigned words,unsigned* guards) {
  if(threadIdx.x<4) {
    guards[blockIdx.x*8+threadIdx.x]=storage[threadIdx.x];
    guards[blockIdx.x*8+4+threadIdx.x]=storage[words+4+threadIdx.x];
  }
}
// Mode0/1: local/remote read. Mode2/3: local/remote write.
template<unsigned C,unsigned Mode>
__device__ __forceinline__ void dsm_form(unsigned iterations,unsigned seed,unsigned* trace,
    unsigned* sums,unsigned* guards,FormStamp* stamps,u64* completion) {
  __shared__ __align__(16) unsigned storage[dsm_words+8];
  auto cluster=cg::this_cluster();const unsigned rank=cluster.block_rank();
  unsigned* tile=storage+4;
  for(unsigned word=threadIdx.x;word<dsm_words;word+=threads)
    tile[word]=17u*((blockIdx.x/C*C+rank)*dsm_words+word)+seed;
  if(threadIdx.x<4) {storage[threadIdx.x]=0xd15ea5e0u+threadIdx.x;storage[dsm_words+4+threadIdx.x]=0xd15ea5e4u+threadIdx.x;}
  __syncthreads();cluster.sync(); // all remote target CTAs exist before mapping/access
  const unsigned target=(Mode==1||Mode==3)?(rank+1)%C:rank;
  volatile unsigned* destination=cluster.map_shared_rank(tile,target);
  unsigned sum=0;
  FormStamp stamp=start_stamp();__syncthreads();
  #pragma unroll 1
  for(unsigned item=0;item<iterations;++item) {
    if constexpr(Mode<2) {
      #pragma unroll
      for(unsigned j=0;j<8;++j) {
        const unsigned word=threadIdx.x+threads*j;
        const unsigned value=destination[word];sum+=value;
        trace[(u64(blockIdx.x)*iterations+item)*dsm_words+word]=value;
      }
      __syncthreads();cluster.sync(); // same control sequence for local and remote
    } else {
      #pragma unroll
      for(unsigned j=0;j<8;++j) {
        const unsigned word=threadIdx.x+threads*j;
        destination[word]=29u*((blockIdx.x/C*C+rank)*dsm_words+word)+seed+31u*item;
      }
      __syncthreads();cluster.sync(); // producer completion before target consumes
      #pragma unroll
      for(unsigned j=0;j<8;++j) {
        const unsigned word=threadIdx.x+threads*j;
        const unsigned value=reinterpret_cast<volatile unsigned*>(tile)[word];sum+=value;
        trace[(u64(blockIdx.x)*iterations+item)*dsm_words+word]=value;
      }
      __syncthreads();cluster.sync(); // consumers done before next remote overwrite
    }
  }
  end_stamp(stamp,stamps);
  cluster.sync(); // independent exit gate; no remote access after this point
  sums[u64(blockIdx.x)*threads+threadIdx.x]=sum;export_guards(storage,dsm_words,guards);
  if(threadIdx.x==0) {completion[blockIdx.x*3]=iterations;completion[blockIdx.x*3+1]=rank;completion[blockIdx.x*3+2]=C;}
}
template<unsigned C>
__device__ __forceinline__ void cluster_sync_form(unsigned iterations,FormStamp* stamps,u64* completion) {
  auto cluster=cg::this_cluster();cluster.sync();
  FormStamp stamp=start_stamp();__syncthreads();
  #pragma unroll 1
  for(unsigned item=0;item<iterations;++item)cluster.sync();
  end_stamp(stamp,stamps);cluster.sync();
  if(threadIdx.x==0) {completion[blockIdx.x*3]=iterations;completion[blockIdx.x*3+1]=cluster.block_rank();completion[blockIdx.x*3+2]=C;}
}
__device__ __forceinline__ void wait_receiver(unsigned barrier,u64 token,u64* failure) {
  const u64 begin=global_time();
  for(;;) {
    unsigned done;
    asm volatile("{ .reg .pred p; mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2, 64; selp.b32 %0, 1, 0, p; }"
      :"=r"(done):"r"(barrier),"l"(token):"memory");
    if(done)return;
    if(global_time()-begin>=1000000000ull) {
      *reinterpret_cast<volatile u64*>(failure)=1;__threadfence();asm volatile("trap;":::"memory");
    }
  }
}
template<unsigned C,bool All>
__device__ __forceinline__ void multicast_form(const unsigned* global,unsigned iterations,
    unsigned seed,unsigned* trace,unsigned* guards,FormStamp* stamps,u64* tokens,u64* completion) {
  __shared__ __align__(16) unsigned storage[bulk_words+8];
  __shared__ __align__(8) u64 barrier_object;
  auto cluster=cg::this_cluster();const unsigned rank=cluster.block_rank();
  constexpr unsigned short_mask=1u<<(C-1),full_mask=(1u<<C)-1;
  constexpr std::uint16_t mask=All?full_mask:short_mask;
  const bool receiver=(mask&(1u<<rank))!=0;
  unsigned* tile=storage+4;const unsigned barrier=shared_address(&barrier_object);
  for(unsigned word=threadIdx.x;word<bulk_words;word+=threads)
    tile[word]=~(17u*((blockIdx.x/C*32)*bulk_words+word)+seed);
  if(threadIdx.x<4) {storage[threadIdx.x]=0xd15ea5e0u+threadIdx.x;storage[bulk_words+4+threadIdx.x]=0xd15ea5e4u+threadIdx.x;}
  if(receiver&&threadIdx.x==0)asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(barrier):"memory");
  // Each generic producer publishes its payload; thread0 publishes local init.
  asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();cluster.sync();
  FormStamp stamp=start_stamp();__syncthreads();u64 token=0,received=0;
  #pragma unroll 1
  for(unsigned item=0;item<iterations;++item) {
    if(receiver) {
      for(unsigned word=threadIdx.x;word<bulk_words;word+=threads)
        tile[word]=~(17u*((blockIdx.x/C*32+item%32)*bulk_words+word)+seed);
      if(threadIdx.x==0) {
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
          :"=l"(token):"r"(barrier),"r"(16384):"memory");
        tokens[u64(blockIdx.x)*iterations+item]=token; // opaque, not exact-numeric evidence
      }
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
    }
    __syncthreads();cluster.sync(); // armed gate: every selected receiver ready
    if(rank==0&&threadIdx.x==0) {
      const unsigned* source=global+(u64(blockIdx.x/C)*32+item%32)*bulk_words;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
        ::"r"(shared_address(tile)),"l"(source),"r"(16384),"r"(barrier),"h"(mask):"memory");
    }
    if(receiver&&threadIdx.x==0)wait_receiver(barrier,token,completion+u64(blockIdx.x)*6+5);
    __syncthreads(); // receiver acquire -> local consumers
    for(unsigned word=threadIdx.x;word<bulk_words;word+=threads)
      trace[(u64(blockIdx.x)*iterations+item)*bulk_words+word]=reinterpret_cast<volatile unsigned*>(tile)[word];
    __syncthreads();cluster.sync(); // consumer-done -> slot release, includes nonreceivers
    if(receiver&&threadIdx.x==0)++received;
  }
  end_stamp(stamp,stamps);
  cluster.sync(); // all async requests and consumers complete before invalidate/exit
  if(receiver&&threadIdx.x==0)asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(barrier):"memory");
  __syncthreads();export_guards(storage,bulk_words,guards);
  if(threadIdx.x==0) {
    const u64 at=u64(blockIdx.x)*6;completion[at]=received;completion[at+1]=rank;
    completion[at+2]=mask;completion[at+3]=C;completion[at+4]=receiver?1:0;completion[at+5]=0;
  }
}
#define CLUSTER_FORMS(C) \
extern "C" __global__ void s17_local_read_c##C(unsigned i,unsigned s,unsigned* t,unsigned* sums,unsigned* guards,FormStamp* stamps,u64* done){dsm_form<C,0>(i,s,t,sums,guards,stamps,done);} \
extern "C" __global__ void s17_dsm_read_c##C(unsigned i,unsigned s,unsigned* t,unsigned* sums,unsigned* guards,FormStamp* stamps,u64* done){dsm_form<C,1>(i,s,t,sums,guards,stamps,done);} \
extern "C" __global__ void s17_local_write_c##C(unsigned i,unsigned s,unsigned* t,unsigned* sums,unsigned* guards,FormStamp* stamps,u64* done){dsm_form<C,2>(i,s,t,sums,guards,stamps,done);} \
extern "C" __global__ void s17_dsm_write_c##C(unsigned i,unsigned s,unsigned* t,unsigned* sums,unsigned* guards,FormStamp* stamps,u64* done){dsm_form<C,3>(i,s,t,sums,guards,stamps,done);} \
extern "C" __global__ void s17_cluster_sync_c##C(unsigned i,FormStamp* stamps,u64* done){cluster_sync_form<C>(i,stamps,done);} \
extern "C" __global__ void s17_bulk_single_c##C(const unsigned* g,unsigned i,unsigned s,unsigned* t,unsigned* guards,FormStamp* stamps,u64* tokens,u64* done){multicast_form<C,false>(g,i,s,t,guards,stamps,tokens,done);} \
extern "C" __global__ void s17_bulk_all_c##C(const unsigned* g,unsigned i,unsigned s,unsigned* t,unsigned* guards,FormStamp* stamps,u64* tokens,u64* done){multicast_form<C,true>(g,i,s,t,guards,stamps,tokens,done);}
CLUSTER_FORMS(2) CLUSTER_FORMS(4) CLUSTER_FORMS(8)
// Syntax-only host helper, never called by main/worker. Caller owns config/attribute.
struct ClusterCapabilities {int launch_support,potential_size,active_clusters;cudaFuncAttributes resource;};
extern "C" cudaError_t s17_compile_query_host(const void* kernel,unsigned C,
    ClusterCapabilities* out,cudaLaunchConfig_t* config,cudaLaunchAttribute* attribute) {
  if(C!=2&&C!=4&&C!=8)return cudaErrorInvalidValue;
  int device=0;cudaError_t error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;
  error=cudaDeviceGetAttribute(&out->launch_support,cudaDevAttrClusterLaunch,device);if(error!=cudaSuccess)return error;
  *config={};config->gridDim=dim3(C);config->blockDim=dim3(128);config->dynamicSmemBytes=0;
  attribute->id=cudaLaunchAttributeClusterDimension;attribute->val.clusterDim.x=C;
  attribute->val.clusterDim.y=1;attribute->val.clusterDim.z=1;config->attrs=attribute;config->numAttrs=1;
  error=cudaFuncGetAttributes(&out->resource,kernel);if(error!=cudaSuccess)return error;
  error=cudaOccupancyMaxPotentialClusterSize(&out->potential_size,const_cast<void*>(kernel),config);if(error!=cudaSuccess)return error;
  return cudaOccupancyMaxActiveClusters(&out->active_clusters,const_cast<void*>(kernel),config);
}
int main(){std::fputs("S17 feasibility compile-only; queries and kernel execution disabled\n",stderr);return 2;}
