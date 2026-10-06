#define main s17_legacy_short_main
#include "cluster_dsm_multicast_v1.cu"
#undef main

// Candidate measured variants: capture=false omits intermediate trace/lifecycle,
// keeps DSM checksum loads and all synchronization; final tile export is after stop.
// Mode0/1: local/remote read. Mode2/3: local/remote write.
template<unsigned C,unsigned Mode>
__device__ __forceinline__ void measured_dsm_form(unsigned iterations,unsigned seed,unsigned* trace,
    unsigned* sums,unsigned* guards,FormStamp* stamps,u64* completion,unsigned* lifecycle,bool capture) {
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
    if(capture)life_start(lifecycle,iterations,item,C,rank,0,true);
    if constexpr(Mode<2) {
      #pragma unroll
      for(unsigned j=0;j<8;++j) {
        const unsigned word=threadIdx.x+threads*j;
        const unsigned value=destination[word];sum+=value;
        if(capture)trace[(u64(blockIdx.x)*iterations+item)*dsm_words+word]=value;
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
        if(capture)trace[(u64(blockIdx.x)*iterations+item)*dsm_words+word]=value;
      }
      __syncthreads();cluster.sync(); // consumers done before next remote overwrite
    }
    if(capture&&threadIdx.x==0){auto* p=life_row(lifecycle,iterations,item);p[9]=1;p[10]=1;p[11]=1;}
  }
  end_stamp(stamp,stamps);
  if(!capture)for(unsigned word=threadIdx.x;word<dsm_words;word+=threads)
    trace[u64(blockIdx.x)*dsm_words+word]=reinterpret_cast<volatile unsigned*>(tile)[word];
  cluster.sync(); // independent exit gate; no remote access after this point
  sums[u64(blockIdx.x)*threads+threadIdx.x]=sum;export_guards(storage,dsm_words,guards);
  if(threadIdx.x==0) {completion[blockIdx.x*6]=iterations;completion[blockIdx.x*6+1]=rank;completion[blockIdx.x*6+2]=C;completion[blockIdx.x*6+3]=1;completion[blockIdx.x*6+4]=1;completion[blockIdx.x*6+5]=0;}
}
template<unsigned C>
__device__ __forceinline__ void measured_cluster_sync_form(unsigned iterations,FormStamp* stamps,u64* completion,unsigned* lifecycle,bool capture) {
  auto cluster=cg::this_cluster();cluster.sync();
  FormStamp stamp=start_stamp();__syncthreads();
  #pragma unroll 1
  for(unsigned item=0;item<iterations;++item){if(capture)life_start(lifecycle,iterations,item,C,cluster.block_rank(),0,false);cluster.sync();if(capture&&threadIdx.x==0){auto* p=life_row(lifecycle,iterations,item);p[10]=1;p[11]=1;}}
  end_stamp(stamp,stamps);cluster.sync();
  if(threadIdx.x==0) {completion[blockIdx.x*6]=iterations;completion[blockIdx.x*6+1]=cluster.block_rank();completion[blockIdx.x*6+2]=C;completion[blockIdx.x*6+3]=1;completion[blockIdx.x*6+4]=1;completion[blockIdx.x*6+5]=0;}
}
__device__ __forceinline__ void measured_wait_receiver(unsigned barrier,u64 token,u64* failure) {
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
__device__ __forceinline__ void measured_multicast_form(const unsigned* global,unsigned iterations,
    unsigned seed,unsigned* trace,unsigned* guards,FormStamp* stamps,u64* tokens,u64* completion,unsigned* lifecycle,bool capture) {
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
    if(capture)life_start(lifecycle,iterations,item,C,rank,mask,receiver);
    if(receiver) {
      if(capture)for(unsigned word=threadIdx.x;word<bulk_words;word+=threads)
        tile[word]=~(17u*((blockIdx.x/C*32+item%32)*bulk_words+word)+seed);
      if(threadIdx.x==0) {
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
          :"=l"(token):"r"(barrier),"r"(16384):"memory");
        if(capture){tokens[u64(blockIdx.x)*iterations+item]=token;auto* p=life_row(lifecycle,iterations,item);p[5]=1;p[6]=16384;} // opaque, not exact-numeric evidence
      }
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
    }
    __syncthreads();cluster.sync(); // armed gate: every selected receiver ready
    if(rank==0&&threadIdx.x==0) {
      if(capture)life_row(lifecycle,iterations,item)[7]=1;
      const unsigned* source=global+(u64(blockIdx.x/C)*32+item%32)*bulk_words;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
        ::"r"(shared_address(tile)),"l"(source),"r"(16384),"r"(barrier),"h"(mask):"memory");
    }
    if(receiver&&threadIdx.x==0){measured_wait_receiver(barrier,token,completion+u64(blockIdx.x)*6+5);if(capture)life_row(lifecycle,iterations,item)[8]=1;}
    __syncthreads(); // receiver acquire -> local consumers
    if(capture)for(unsigned word=threadIdx.x;word<bulk_words;word+=threads)
      trace[(u64(blockIdx.x)*iterations+item)*bulk_words+word]=reinterpret_cast<volatile unsigned*>(tile)[word];
    if(capture&&threadIdx.x==0)life_row(lifecycle,iterations,item)[9]=1;
    __syncthreads();cluster.sync(); // consumer-done -> slot release, includes nonreceivers
    if(capture&&threadIdx.x==0){auto* p=life_row(lifecycle,iterations,item);p[10]=1;p[11]=1;}
    if(receiver&&threadIdx.x==0)++received;
  }
  end_stamp(stamp,stamps);
  if(!capture)for(unsigned word=threadIdx.x;word<bulk_words;word+=threads)
    trace[u64(blockIdx.x)*bulk_words+word]=reinterpret_cast<volatile unsigned*>(tile)[word];
  cluster.sync(); // all async requests and consumers complete before invalidate/exit
  if(receiver&&threadIdx.x==0)asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(barrier):"memory");
  __syncthreads();export_guards(storage,bulk_words,guards);
  if(threadIdx.x==0) {
    const u64 at=u64(blockIdx.x)*6;completion[at]=iterations;completion[at+1]=rank;
    completion[at+2]=C;completion[at+3]=1;completion[at+4]=1;completion[at+5]=0;
  }
}
#define MEASURED_CLUSTER_FORMS(C) \
extern "C" __global__ void s17m_local_read_c##C(unsigned i,unsigned s,unsigned* t,unsigned* sums,unsigned* guards,FormStamp* stamps,u64* done,unsigned* life,bool capture){measured_dsm_form<C,0>(i,s,t,sums,guards,stamps,done,life,capture);} \
extern "C" __global__ void s17m_dsm_read_c##C(unsigned i,unsigned s,unsigned* t,unsigned* sums,unsigned* guards,FormStamp* stamps,u64* done,unsigned* life,bool capture){measured_dsm_form<C,1>(i,s,t,sums,guards,stamps,done,life,capture);} \
extern "C" __global__ void s17m_local_write_c##C(unsigned i,unsigned s,unsigned* t,unsigned* sums,unsigned* guards,FormStamp* stamps,u64* done,unsigned* life,bool capture){measured_dsm_form<C,2>(i,s,t,sums,guards,stamps,done,life,capture);} \
extern "C" __global__ void s17m_dsm_write_c##C(unsigned i,unsigned s,unsigned* t,unsigned* sums,unsigned* guards,FormStamp* stamps,u64* done,unsigned* life,bool capture){measured_dsm_form<C,3>(i,s,t,sums,guards,stamps,done,life,capture);} \
extern "C" __global__ void s17m_cluster_sync_c##C(unsigned i,FormStamp* stamps,u64* done,unsigned* life,bool capture){measured_cluster_sync_form<C>(i,stamps,done,life,capture);} \
extern "C" __global__ void s17m_bulk_single_c##C(const unsigned* g,unsigned i,unsigned s,unsigned* t,unsigned* guards,FormStamp* stamps,u64* tokens,u64* done,unsigned* life,bool capture){measured_multicast_form<C,false>(g,i,s,t,guards,stamps,tokens,done,life,capture);} \
extern "C" __global__ void s17m_bulk_all_c##C(const unsigned* g,unsigned i,unsigned s,unsigned* t,unsigned* guards,FormStamp* stamps,u64* tokens,u64* done,unsigned* life,bool capture){measured_multicast_form<C,true>(g,i,s,t,guards,stamps,tokens,done,life,capture);}
MEASURED_CLUSTER_FORMS(2) MEASURED_CLUSTER_FORMS(4) MEASURED_CLUSTER_FORMS(8)

int main(int argc,char** argv){return s17_legacy_short_main(argc,argv);}
