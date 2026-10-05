// S17 B1 finite short probe. Original feasibility source remains frozen.
// PTX 8.8 cp.async.bulk multicast and mbarrier.arrive.expect_tx forms:
// https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
namespace cg = cooperative_groups;
using u64 = std::uint64_t;
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
// Every field is written at the corresponding path boundary; the B2 audit
// must independently check that synchronization occurs before done/release.
__device__ __forceinline__ unsigned* life_row(unsigned* life,unsigned i,unsigned item){return life+(u64(blockIdx.x)*i+item)*12;}
__device__ __forceinline__ void life_start(unsigned* life,unsigned i,unsigned item,unsigned C,unsigned rank,unsigned mask,bool recv){
 if(threadIdx.x==0){auto* p=life_row(life,i,item);p[0]=rank;p[1]=C;p[2]=item;p[3]=mask;p[4]=recv?1:0;p[5]=0;p[6]=0;p[7]=0;p[8]=0;p[9]=0;p[10]=0;p[11]=0;}
}
// Mode0/1: local/remote read. Mode2/3: local/remote write.
template<unsigned C,unsigned Mode>
__device__ __forceinline__ void dsm_form(unsigned iterations,unsigned seed,unsigned* trace,
    unsigned* sums,unsigned* guards,FormStamp* stamps,u64* completion,unsigned* lifecycle) {
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
    life_start(lifecycle,iterations,item,C,rank,0,true);
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
    if(threadIdx.x==0){auto* p=life_row(lifecycle,iterations,item);p[9]=1;p[10]=1;p[11]=1;}
  }
  end_stamp(stamp,stamps);
  cluster.sync(); // independent exit gate; no remote access after this point
  sums[u64(blockIdx.x)*threads+threadIdx.x]=sum;export_guards(storage,dsm_words,guards);
  if(threadIdx.x==0) {completion[blockIdx.x*6]=iterations;completion[blockIdx.x*6+1]=rank;completion[blockIdx.x*6+2]=C;completion[blockIdx.x*6+3]=1;completion[blockIdx.x*6+4]=1;completion[blockIdx.x*6+5]=0;}
}
template<unsigned C>
__device__ __forceinline__ void cluster_sync_form(unsigned iterations,FormStamp* stamps,u64* completion,unsigned* lifecycle) {
  auto cluster=cg::this_cluster();cluster.sync();
  FormStamp stamp=start_stamp();__syncthreads();
  #pragma unroll 1
  for(unsigned item=0;item<iterations;++item){life_start(lifecycle,iterations,item,C,cluster.block_rank(),0,false);cluster.sync();if(threadIdx.x==0){auto* p=life_row(lifecycle,iterations,item);p[10]=1;p[11]=1;}}
  end_stamp(stamp,stamps);cluster.sync();
  if(threadIdx.x==0) {completion[blockIdx.x*6]=iterations;completion[blockIdx.x*6+1]=cluster.block_rank();completion[blockIdx.x*6+2]=C;completion[blockIdx.x*6+3]=1;completion[blockIdx.x*6+4]=1;completion[blockIdx.x*6+5]=0;}
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
    unsigned seed,unsigned* trace,unsigned* guards,FormStamp* stamps,u64* tokens,u64* completion,unsigned* lifecycle) {
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
    life_start(lifecycle,iterations,item,C,rank,mask,receiver);
    if(receiver) {
      for(unsigned word=threadIdx.x;word<bulk_words;word+=threads)
        tile[word]=~(17u*((blockIdx.x/C*32+item%32)*bulk_words+word)+seed);
      if(threadIdx.x==0) {
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
          :"=l"(token):"r"(barrier),"r"(16384):"memory");
        tokens[u64(blockIdx.x)*iterations+item]=token;auto* p=life_row(lifecycle,iterations,item);p[5]=1;p[6]=16384; // opaque, not exact-numeric evidence
      }
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
    }
    __syncthreads();cluster.sync(); // armed gate: every selected receiver ready
    if(rank==0&&threadIdx.x==0) {
      life_row(lifecycle,iterations,item)[7]=1;
      const unsigned* source=global+(u64(blockIdx.x/C)*32+item%32)*bulk_words;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
        ::"r"(shared_address(tile)),"l"(source),"r"(16384),"r"(barrier),"h"(mask):"memory");
    }
    if(receiver&&threadIdx.x==0){wait_receiver(barrier,token,completion+u64(blockIdx.x)*6+5);life_row(lifecycle,iterations,item)[8]=1;}
    __syncthreads(); // receiver acquire -> local consumers
    for(unsigned word=threadIdx.x;word<bulk_words;word+=threads)
      trace[(u64(blockIdx.x)*iterations+item)*bulk_words+word]=reinterpret_cast<volatile unsigned*>(tile)[word];
    if(threadIdx.x==0)life_row(lifecycle,iterations,item)[9]=1;
    __syncthreads();cluster.sync(); // consumer-done -> slot release, includes nonreceivers
    if(threadIdx.x==0){auto* p=life_row(lifecycle,iterations,item);p[10]=1;p[11]=1;}
    if(receiver&&threadIdx.x==0)++received;
  }
  end_stamp(stamp,stamps);
  cluster.sync(); // all async requests and consumers complete before invalidate/exit
  if(receiver&&threadIdx.x==0)asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(barrier):"memory");
  __syncthreads();export_guards(storage,bulk_words,guards);
  if(threadIdx.x==0) {
    const u64 at=u64(blockIdx.x)*6;completion[at]=iterations;completion[at+1]=rank;
    completion[at+2]=C;completion[at+3]=1;completion[at+4]=1;completion[at+5]=0;
  }
}
#define CLUSTER_FORMS(C) \
extern "C" __global__ void s17_local_read_c##C(unsigned i,unsigned s,unsigned* t,unsigned* sums,unsigned* guards,FormStamp* stamps,u64* done,unsigned* life){dsm_form<C,0>(i,s,t,sums,guards,stamps,done,life);} \
extern "C" __global__ void s17_dsm_read_c##C(unsigned i,unsigned s,unsigned* t,unsigned* sums,unsigned* guards,FormStamp* stamps,u64* done,unsigned* life){dsm_form<C,1>(i,s,t,sums,guards,stamps,done,life);} \
extern "C" __global__ void s17_local_write_c##C(unsigned i,unsigned s,unsigned* t,unsigned* sums,unsigned* guards,FormStamp* stamps,u64* done,unsigned* life){dsm_form<C,2>(i,s,t,sums,guards,stamps,done,life);} \
extern "C" __global__ void s17_dsm_write_c##C(unsigned i,unsigned s,unsigned* t,unsigned* sums,unsigned* guards,FormStamp* stamps,u64* done,unsigned* life){dsm_form<C,3>(i,s,t,sums,guards,stamps,done,life);} \
extern "C" __global__ void s17_cluster_sync_c##C(unsigned i,FormStamp* stamps,u64* done,unsigned* life){cluster_sync_form<C>(i,stamps,done,life);} \
extern "C" __global__ void s17_bulk_single_c##C(const unsigned* g,unsigned i,unsigned s,unsigned* t,unsigned* guards,FormStamp* stamps,u64* tokens,u64* done,unsigned* life){multicast_form<C,false>(g,i,s,t,guards,stamps,tokens,done,life);} \
extern "C" __global__ void s17_bulk_all_c##C(const unsigned* g,unsigned i,unsigned s,unsigned* t,unsigned* guards,FormStamp* stamps,u64* tokens,u64* done,unsigned* life){multicast_form<C,true>(g,i,s,t,guards,stamps,tokens,done,life);}
CLUSTER_FORMS(2) CLUSTER_FORMS(4) CLUSTER_FORMS(8)
#include "../common/probe_runtime.cuh"
#include "../common/word_artifacts.hpp"
#include "../common/cluster_dsm_reference_v1.hpp"
#include <algorithm>
#include <limits>
#include <string>
#include <vector>
struct ClusterCase{const char* form;unsigned C,mode;const char* symbol;const void* function;};
#define CASES(C) \
 {"local_read",C,0,"s17_local_read_c" #C,reinterpret_cast<const void*>(s17_local_read_c##C)}, \
 {"dsm_read",C,1,"s17_dsm_read_c" #C,reinterpret_cast<const void*>(s17_dsm_read_c##C)}, \
 {"local_write",C,2,"s17_local_write_c" #C,reinterpret_cast<const void*>(s17_local_write_c##C)}, \
 {"dsm_write",C,3,"s17_dsm_write_c" #C,reinterpret_cast<const void*>(s17_dsm_write_c##C)}, \
 {"cluster_sync",C,4,"s17_cluster_sync_c" #C,reinterpret_cast<const void*>(s17_cluster_sync_c##C)}, \
 {"bulk_single_target",C,5,"s17_bulk_single_c" #C,reinterpret_cast<const void*>(s17_bulk_single_c##C)}, \
 {"bulk_all_targets",C,6,"s17_bulk_all_c" #C,reinterpret_cast<const void*>(s17_bulk_all_c##C)}
static const ClusterCase cluster_cases[]={CASES(2),CASES(4),CASES(8)};
struct Capability{int supported=0,potential=0,active=0;cudaError_t error=cudaSuccess;cudaFuncAttributes attr{};};
static cudaLaunchConfig_t launch_config(unsigned C,unsigned clusters,cudaLaunchAttribute& attribute){
 cudaLaunchConfig_t config{};config.gridDim=dim3(C*clusters);config.blockDim=dim3(128);
 attribute={};attribute.id=cudaLaunchAttributeClusterDimension;attribute.val.clusterDim.x=C;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;
 config.attrs=&attribute;config.numAttrs=1;return config;
}
static Capability capability(const ClusterCase& c){
 Capability x;int device=0;x.error=cudaGetDevice(&device);if(x.error!=cudaSuccess)return x;
 x.error=cudaDeviceGetAttribute(&x.supported,cudaDevAttrClusterLaunch,device);if(x.error!=cudaSuccess||!x.supported)return x;
 x.error=cudaFuncGetAttributes(&x.attr,c.function);if(x.error!=cudaSuccess)return x;
 cudaLaunchAttribute attribute{};// A large finite query grid avoids clipping the occupancy query to one cluster.
 // It is not launched and is not an estimate of physical residency.
 auto config=launch_config(c.C,1024,attribute);
 x.error=cudaOccupancyMaxPotentialClusterSize(&x.potential,const_cast<void*>(c.function),&config);if(x.error!=cudaSuccess)return x;
 if(x.potential<int(c.C))return x;
 x.error=cudaOccupancyMaxActiveClusters(&x.active,const_cast<void*>(c.function),&config);return x;
}
static void emit_capability(const ClusterCase& c,const Capability& x){
 std::cout<<"{\"type\":\"cluster_capability\",\"kernel_symbol\":"<<gh::quote(c.symbol)<<",\"cluster_size\":"<<c.C<<",\"cluster_launch_supported\":"<<x.supported<<",\"potential_cluster_size\":"<<x.potential<<",\"active_cluster_capacity\":"<<x.active<<",\"cuda_error\":"<<int(x.error)<<",\"registers_per_thread\":"<<x.attr.numRegs<<",\"static_smem_bytes\":"<<x.attr.sharedSizeBytes<<",\"local_size_bytes\":"<<x.attr.localSizeBytes<<",\"occupancy_is_upper_bound\":true}\n";
}
struct Artifact{std::string path,sha;std::vector<gh::u64> shape;};
struct Check{gh::u64 checked=0,opaque=0;std::vector<Artifact> artifacts;};
static void save(Check& c,unsigned n,const char* leaf,const std::vector<unsigned>& data,std::vector<gh::u64> shape){
 gh::u64 count=1;for(auto x:shape){if(x==0||count>std::numeric_limits<gh::u64>::max()/x)throw std::runtime_error("S17 shape overflow");count*=x;}
 if(count!=data.size())throw std::runtime_error("S17 artifact shape");
 std::string p="cluster_"+std::to_string(n)+"_"+leaf+".u32le";c.artifacts.push_back({p,word_artifacts::write_words(p,data),shape});
}
int main(int argc,char** argv)try{
 const auto device=gh::device();gh::emit_device(device);
 if(argc==2&&std::string(argv[1])=="device")return 0;
 if(argc==2&&std::string(argv[1])=="cluster-device"){for(const auto& c:cluster_cases)emit_capability(c,capability(c));return 0;}
 if(argc!=5||std::string(argv[1])!="validate-only"||std::string(argv[3])!="cluster_short_1_2_5_v1"||gh::integer(argv[4],0,4294967295ull)!=3)throw std::runtime_error("S17 only finite grouped validate-only CASE cluster_short_1_2_5_v1 3; formal disabled");
 const ClusterCase* choice=nullptr;bool all=false;std::string id=argv[2];
 for(const auto& c:cluster_cases)for(unsigned scope=0;scope<2;++scope)if(id==std::string(c.form)+"_c"+std::to_string(c.C)+(scope?"_cluster_grid":"_one_cluster")){choice=&c;all=scope;}
 if(!choice)throw std::runtime_error("S17 unknown finite coordinate");const auto& c=*choice;auto cap=capability(c);
 if(cap.error!=cudaSuccess||!cap.supported||cap.potential<int(c.C)||cap.active<1||cap.attr.localSizeBytes)throw std::runtime_error("S17 capability_reject_before_launch; cuda_error="+std::to_string(int(cap.error)));
 const unsigned G=all?unsigned(cap.active):1,B=G*c.C;const unsigned W=c.mode>=5?4096:c.mode<4?1024:0;
 if(G>1024)throw std::runtime_error("S17 bounded cluster count exceeds reference domain");
 const gh::u64 ring_words=c.mode>=5?gh::u64(G)*32*4096:1,trace_words=W?gh::u64(B)*5*W:1;
 const gh::u64 needed=4*(ring_words+trace_words+gh::u64(B)*128+B*8+gh::u64(B)*5*12)+8*(gh::u64(B)*5+B*6)+B*sizeof(FormStamp);
 size_t available=0,total=0;GH_CUDA(cudaMemGetInfo(&available,&total));if(needed>available)throw std::runtime_error("S17 global allocation bound");
 unsigned *global=nullptr,*trace=nullptr,*sums=nullptr,*guards=nullptr,*life=nullptr;u64 *tokens=nullptr,*done=nullptr;FormStamp* stamps=nullptr;
 GH_CUDA(cudaMalloc(&global,ring_words*4));GH_CUDA(cudaMalloc(&trace,trace_words*4));GH_CUDA(cudaMalloc(&sums,gh::u64(B)*128*4));GH_CUDA(cudaMalloc(&guards,B*8*4));GH_CUDA(cudaMalloc(&life,gh::u64(B)*5*12*4));GH_CUDA(cudaMalloc(&tokens,gh::u64(B)*5*8));GH_CUDA(cudaMalloc(&done,B*6*8));GH_CUDA(cudaMalloc(&stamps,B*sizeof(FormStamp)));
 const unsigned lengths[]={1,2,5},seeds[]={0,3,0xffffffffu};std::vector<Check> checks;
 cudaEvent_t end;GH_CUDA(cudaEventCreate(&end));
 for(unsigned launch=0;launch<3;++launch){unsigned i=lengths[launch],seed=seeds[launch];Check ch;
  std::vector<unsigned> ring(ring_words);for(gh::u64 at=0;at<ring_words;++at)ring[at]=cluster_reference::bulk(at/(32*4096),at/4096%32,at%4096,seed);
  GH_CUDA(cudaMemcpy(global,ring.data(),ring_words*4,cudaMemcpyHostToDevice));
  GH_CUDA(cudaMemset(trace,0xff,trace_words*4));GH_CUDA(cudaMemset(sums,0xff,gh::u64(B)*128*4));GH_CUDA(cudaMemset(guards,0xff,B*8*4));GH_CUDA(cudaMemset(life,0xff,gh::u64(B)*i*12*4));GH_CUDA(cudaMemset(tokens,0,gh::u64(B)*i*8));GH_CUDA(cudaMemset(done,0xff,B*6*8));GH_CUDA(cudaMemset(stamps,0xff,B*sizeof(FormStamp)));
  cudaLaunchAttribute attribute{};auto config=launch_config(c.C,G,attribute);
  void* args_dsm[]={&i,&seed,&trace,&sums,&guards,&stamps,&done,&life};void* args_sync[]={&i,&stamps,&done,&life};void* args_bulk[]={&global,&i,&seed,&trace,&guards,&stamps,&tokens,&done,&life};
  GH_CUDA(cudaLaunchKernelExC(&config,c.function,c.mode<4?args_dsm:c.mode==4?args_sync:args_bulk));GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));
  std::vector<unsigned> values(W?gh::u64(B)*i*W:0),gs(W?B*8:0),ss(c.mode<4?B*128:0),ls(gh::u64(B)*i*12);
  std::vector<u64> ds(B*6),ts(c.mode>=5?gh::u64(B)*i:0);std::vector<FormStamp> timing(B);
  if(W){GH_CUDA(cudaMemcpy(values.data(),trace,values.size()*4,cudaMemcpyDeviceToHost));GH_CUDA(cudaMemcpy(gs.data(),guards,gs.size()*4,cudaMemcpyDeviceToHost));}
  if(c.mode<4)GH_CUDA(cudaMemcpy(ss.data(),sums,ss.size()*4,cudaMemcpyDeviceToHost));
  GH_CUDA(cudaMemcpy(ls.data(),life,ls.size()*4,cudaMemcpyDeviceToHost));GH_CUDA(cudaMemcpy(ds.data(),done,ds.size()*8,cudaMemcpyDeviceToHost));GH_CUDA(cudaMemcpy(timing.data(),stamps,B*sizeof(FormStamp),cudaMemcpyDeviceToHost));
  if(c.mode>=5){GH_CUDA(cudaMemcpy(ts.data(),tokens,ts.size()*8,cudaMemcpyDeviceToHost));std::vector<unsigned> back(ring_words);GH_CUDA(cudaMemcpy(back.data(),global,ring_words*4,cudaMemcpyDeviceToHost));if(back!=ring)throw std::runtime_error("S17 source ring changed");}
  auto equal=[&](unsigned a,unsigned b){++ch.checked;if(a!=b)throw std::runtime_error("S17 full value mismatch");};
  for(unsigned b=0;b<B;++b){for(unsigned k=0;k<6;++k){u64 expected=k==0?i:k==1?b%c.C:k==2?c.C:k<5?1:0;equal(unsigned(ds[b*6+k]),unsigned(expected));equal(unsigned(ds[b*6+k]>>32),0);}
   for(unsigned item=0;item<i;++item)for(unsigned f=0;f<12;++f)equal(ls[(gh::u64(b)*i+item)*12+f],cluster_reference::life(c.mode,c.C,b,item,f));
   if(W){for(unsigned k=0;k<8;++k)equal(gs[b*8+k],0xd15ea5e0u+k);
    for(unsigned item=0;item<i;++item)for(unsigned w=0;w<W;++w){unsigned v;
     if(c.mode>=5){bool recv=cluster_reference::mask(c.mode==6,c.C)&(1u<<(b%c.C));v=cluster_reference::bulk(b/c.C,recv?item:0,w,seed);if(!recv)v=~v;}
     else v=cluster_reference::dsm(c.mode>=2,c.mode==1||c.mode==3,c.C,b,item,w,seed);
     equal(values[(gh::u64(b)*i+item)*W+w],v);
    }
   }
   if(c.mode<4)for(unsigned t=0;t<128;++t){unsigned expected=0;for(unsigned item=0;item<i;++item)for(unsigned j=0;j<8;++j)expected+=cluster_reference::dsm(c.mode>=2,c.mode==1||c.mode==3,c.C,b,item,t+128*j,seed);equal(ss[b*128+t],expected);}
   const auto& time=timing[b];u64 sentinel=~u64(0);if(time.begin_ns==sentinel||time.end_ns==sentinel||time.begin_cycle==sentinel||time.end_cycle==sentinel||time.end_ns<time.begin_ns||time.end_cycle<time.begin_cycle||time.smid==~unsigned(0))throw std::runtime_error("S17 time interval/physical SMID");ch.checked+=10;
   if(c.mode>=5){bool recv=cluster_reference::mask(c.mode==6,c.C)&(1u<<(b%c.C));for(unsigned item=0;item<i;++item){if(recv)ch.opaque+=2;else{equal(unsigned(ts[gh::u64(b)*i+item]),0);equal(unsigned(ts[gh::u64(b)*i+item]>>32),0);}}}
  }
  std::vector<u64> encoded;for(const auto& t:timing)encoded.insert(encoded.end(),{t.begin_ns,t.end_ns,t.begin_cycle,t.end_cycle,t.smid});
  if(W){save(ch,launch,"trace",values,{B,i,W});save(ch,launch,"guards",gs,{B,8});}save(ch,launch,"completion",word_artifacts::split_u64(ds),{B,6,2});save(ch,launch,"lifecycle",ls,{B,i,12});save(ch,launch,"stamps",word_artifacts::split_u64(encoded),{B,5,2});
  if(c.mode<4)save(ch,launch,"sums",ss,{B,128});if(c.mode>=5){save(ch,launch,"tokens",word_artifacts::split_u64(ts),{B,i,2});save(ch,launch,"source_ring",ring,{G,32,4096});ch.checked+=ring_words;}
  checks.push_back(std::move(ch));
 }
 const char* reference_sha="b65cac3b2ed65ff881b9cfadec619731791960899dd68b2011c8ae2cc220cf4d";
 std::cout<<"{\"schema_version\":2,\"validation_schema_version\":1,\"type\":\"validation\",\"case_id\":"<<gh::quote(id)<<",\"profile_id\":\"cluster_short_1_2_5_v1\",\"seed\":3,\"scope\":"<<gh::quote(all?"cluster_grid":"one_cluster")<<",\"threads\":128,\"blocks\":"<<B<<",\"errors\":0,\"performance_eligible\":false,\"warmup_executed\":false,\"pilot_executed\":false,\"target_launches\":[";
 for(unsigned n=0;n<3;++n)std::cout<<(n?",":"")<<"{\"launch_index\":"<<n<<",\"iterations\":"<<lengths[n]<<",\"input_profile\":\"nonuniform_uint32_cluster_paired_seeds_v1\",\"threads\":128,\"blocks\":"<<B<<"}";
 std::cout<<"],\"checks\":[";
 for(unsigned n=0;n<3;++n){const auto& ch=checks[n];std::cout<<(n?",":"")<<"{\"launch_index\":"<<n<<",\"reference_model\":\"cluster_dsm_word_reference_v1\",\"reference_sha256\":"<<gh::quote(reference_sha)<<",\"comparison\":\"exact\",\"tolerance_id\":null,\"completed\":true,\"errors\":0,\"checked_elements\":"<<ch.checked<<",\"expected_elements\":"<<ch.checked<<",\"verified_CTA_ids\":[";for(unsigned b=0;b<B;++b)std::cout<<(b?",":"")<<b;std::cout<<"],\"output_artifacts\":[";
  for(unsigned a=0;a<ch.artifacts.size();++a){const auto& x=ch.artifacts[a];std::cout<<(a?",":"")<<"{\"path\":"<<gh::quote(x.path)<<",\"sha256\":"<<gh::quote(x.sha)<<",\"dtype\":\"uint32\",\"evidence_kind\":\"full_values\",\"shape\":[";for(unsigned j=0;j<x.shape.size();++j)std::cout<<(j?",":"")<<x.shape[j];std::cout<<"]}";}std::cout<<"]}";
 }
 std::cout<<"],\"resource_identity\":{\"kernel_symbol\":"<<gh::quote(c.symbol)<<",\"registers_per_thread\":"<<cap.attr.numRegs<<",\"static_smem_bytes\":"<<cap.attr.sharedSizeBytes<<",\"dynamic_smem_bytes\":0,\"local_size_bytes\":"<<cap.attr.localSizeBytes<<",\"occupancy_limit_ctas_per_sm\":0,\"extensions\":{\"cluster_launch_supported\":"<<cap.supported<<",\"potential_cluster_size\":"<<cap.potential<<",\"active_cluster_capacity\":"<<cap.active<<",\"cluster_size\":"<<c.C<<",\"clusters\":"<<G<<",\"mask\":"<<(c.mode>=5?cluster_reference::mask(c.mode==6,c.C):0)<<",\"opaque_token_words_preserved_domain_only\":[";for(unsigned n=0;n<3;++n)std::cout<<(n?",":"")<<checks[n].opaque;std::cout<<"],\"occupancy_is_upper_bound\":true}}}\n";
 GH_CUDA(cudaEventDestroy(end));GH_CUDA(cudaFree(stamps));GH_CUDA(cudaFree(done));GH_CUDA(cudaFree(tokens));GH_CUDA(cudaFree(life));GH_CUDA(cudaFree(guards));GH_CUDA(cudaFree(sums));GH_CUDA(cudaFree(trace));GH_CUDA(cudaFree(global));return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
