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

#define MEASURED_CASES(C) \
 {"local_read",C,0,"s17m_local_read_c" #C,reinterpret_cast<const void*>(s17m_local_read_c##C)}, \
 {"dsm_read",C,1,"s17m_dsm_read_c" #C,reinterpret_cast<const void*>(s17m_dsm_read_c##C)}, \
 {"local_write",C,2,"s17m_local_write_c" #C,reinterpret_cast<const void*>(s17m_local_write_c##C)}, \
 {"dsm_write",C,3,"s17m_dsm_write_c" #C,reinterpret_cast<const void*>(s17m_dsm_write_c##C)}, \
 {"cluster_sync",C,4,"s17m_cluster_sync_c" #C,reinterpret_cast<const void*>(s17m_cluster_sync_c##C)}, \
 {"bulk_single_target",C,5,"s17m_bulk_single_c" #C,reinterpret_cast<const void*>(s17m_bulk_single_c##C)}, \
 {"bulk_all_targets",C,6,"s17m_bulk_all_c" #C,reinterpret_cast<const void*>(s17m_bulk_all_c##C)}
static const ClusterCase measured_cases[]={MEASURED_CASES(2),MEASURED_CASES(4),MEASURED_CASES(8)};

static void save_identity(const std::string& path,const std::string& text) {
 int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
 if(fd<0)throw std::runtime_error("exclusive S17 identity creation failed");
 struct Close{int fd;~Close(){::close(fd);}}close{fd};std::size_t at=0;
 while(at<text.size()){auto n=::write(fd,text.data()+at,text.size()-at);if(n<0&&errno==EINTR)continue;
  if(n<=0)throw std::runtime_error("S17 identity write failed");at+=n;}
 if(::fsync(fd))throw std::runtime_error("S17 identity fsync failed");
}

int formal_main(int argc,char** argv)try{
 const auto device=gh::device();gh::emit_device(device);
 if(argc==2&&std::string(argv[1])=="measured-cluster-device"){
  for(const auto& c:measured_cases)emit_capability(c,capability(c));return 0;
 }
 const bool pilot=argc>1&&std::string(argv[1])=="pilot-only";
 if(argc!=5||(!pilot&&std::string(argv[1])!="formal-only"))throw std::runtime_error("S17 pilot-only/formal-only CASE N SEED required");
 unsigned i=gh::integer(argv[3],128,pilot?128:65536),seed=gh::integer(argv[4],0,4294967295ull);
 const ClusterCase* selected=nullptr;bool all=false;const std::string id=argv[2];
 for(const auto& c:measured_cases)for(unsigned scope=0;scope<2;++scope)
  if(id==std::string(c.form)+"_c"+std::to_string(c.C)+(scope?"_cluster_grid":"_one_cluster")){selected=&c;all=scope;}
 if(!selected)throw std::runtime_error("unknown finite S17 coordinate");const auto& c=*selected;const auto cap=capability(c);
 if(cap.error!=cudaSuccess||!cap.supported||cap.potential<int(c.C)||cap.active<1||cap.attr.localSizeBytes)
  throw std::runtime_error("S17 capability_reject_before_launch");
 const unsigned G=all?unsigned(cap.active):1,B=G*c.C,W=c.mode>=5?4096:c.mode<4?1024:0;
 if(G>1024)throw std::runtime_error("bounded cluster grid domain");
 const gh::u64 ring_words=c.mode>=5?gh::u64(G)*32*4096:1,final_words=W?gh::u64(B)*W:1;
 const gh::u64 required=4*(ring_words+final_words+gh::u64(B)*128+B*8)+8*(B*6)+B*sizeof(FormStamp);
 size_t free_bytes=0,total_bytes=0;GH_CUDA(cudaMemGetInfo(&free_bytes,&total_bytes));
 if(required>free_bytes)throw std::runtime_error("S17 full final-output memory budget");
 unsigned *global=nullptr,*trace=nullptr,*sums=nullptr,*guards=nullptr,*life=nullptr;
 u64 *tokens=nullptr,*done=nullptr;FormStamp* stamps=nullptr;
 GH_CUDA(cudaMalloc(&global,ring_words*4));GH_CUDA(cudaMalloc(&trace,final_words*4));
 GH_CUDA(cudaMalloc(&sums,gh::u64(B)*128*4));GH_CUDA(cudaMalloc(&guards,B*8*4));
 GH_CUDA(cudaMalloc(&done,B*6*8));GH_CUDA(cudaMalloc(&stamps,B*sizeof(FormStamp)));
 std::vector<unsigned> initial(ring_words),values(W?gh::u64(B)*W:0),gs(W?B*8:0),ss(c.mode<4?B*128:0),back(ring_words);
 if(c.mode>=5)for(gh::u64 at=0;at<ring_words;++at)initial[at]=cluster_reference::bulk(at/(32*4096),at/4096%32,at%4096,seed);
 std::vector<u64> completion(B*6);std::vector<FormStamp> timing(B);
 cudaEvent_t begin,end;GH_CUDA(cudaEventCreate(&begin));GH_CUDA(cudaEventCreate(&end));
 bool capture=false;unsigned invocation=0;Check final_check;
 auto execute=[&](bool persist){
  ++invocation;GH_CUDA(cudaMemcpy(global,initial.data(),ring_words*4,cudaMemcpyHostToDevice));
  GH_CUDA(cudaMemset(trace,0xff,final_words*4));GH_CUDA(cudaMemset(sums,0xff,gh::u64(B)*128*4));
  GH_CUDA(cudaMemset(guards,0xff,B*8*4));GH_CUDA(cudaMemset(done,0xff,B*6*8));GH_CUDA(cudaMemset(stamps,0xff,B*sizeof(FormStamp)));
  cudaLaunchAttribute attribute{};auto config=launch_config(c.C,G,attribute);
  void* args_dsm[]={&i,&seed,&trace,&sums,&guards,&stamps,&done,&life,&capture};
  void* args_sync[]={&i,&stamps,&done,&life,&capture};
  void* args_bulk[]={&global,&i,&seed,&trace,&guards,&stamps,&tokens,&done,&life,&capture};
  GH_CUDA(cudaEventRecord(begin));GH_CUDA(cudaLaunchKernelExC(&config,c.function,c.mode<4?args_dsm:c.mode==4?args_sync:args_bulk));
  GH_CUDA(cudaGetLastError());GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));
  gh::Observation o;float ms=0;GH_CUDA(cudaEventElapsedTime(&ms,begin,end));o.event_ms=ms;
  GH_CUDA(cudaMemcpy(timing.data(),stamps,B*sizeof(FormStamp),cudaMemcpyDeviceToHost));
  GH_CUDA(cudaMemcpy(completion.data(),done,B*6*8,cudaMemcpyDeviceToHost));
  if(W){GH_CUDA(cudaMemcpy(values.data(),trace,values.size()*4,cudaMemcpyDeviceToHost));GH_CUDA(cudaMemcpy(gs.data(),guards,gs.size()*4,cudaMemcpyDeviceToHost));}
  if(c.mode<4)GH_CUDA(cudaMemcpy(ss.data(),sums,ss.size()*4,cudaMemcpyDeviceToHost));
  if(c.mode>=5)GH_CUDA(cudaMemcpy(back.data(),global,ring_words*4,cudaMemcpyDeviceToHost));
  auto equal=[&](unsigned actual,unsigned expected){++o.checked_elements;o.errors+=actual!=expected;};
  for(unsigned b=0;b<B;++b){
   for(unsigned f=0;f<6;++f){u64 expected=f==0?i:f==1?b%c.C:f==2?c.C:f<5?1:0;
    equal(unsigned(completion[b*6+f]),unsigned(expected));equal(unsigned(completion[b*6+f]>>32),0);}
   if(W){for(unsigned f=0;f<8;++f)equal(gs[b*8+f],0xd15ea5e0u+f);
    for(unsigned w=0;w<W;++w){unsigned expected;
     if(c.mode>=5){bool receiver=cluster_reference::mask(c.mode==6,c.C)&(1u<<(b%c.C));
      expected=cluster_reference::bulk(b/c.C,receiver?i-1:0,w,seed);if(!receiver)expected=~expected;}
     else if(c.mode<2)expected=17u*(b*1024u+w)+seed;
     else expected=cluster_reference::dsm(true,c.mode==3,c.C,b,i-1,w,seed);
     equal(values[gh::u64(b)*W+w],expected);
    }
   }
   if(c.mode<4)for(unsigned t=0;t<128;++t){u64 expected=0;
    for(unsigned j=0;j<8;++j)expected+=cluster_reference::dsm(c.mode>=2,c.mode==1||c.mode==3,c.C,b,0,t+128*j,seed);
    expected=expected*i+(c.mode>=2?u64(31)*8*i*(i-1)/2:0);equal(ss[b*128+t],unsigned(expected));
   }
   const auto& t=timing[b];const auto sentinel=~u64(0);
   o.errors+=t.begin_ns==sentinel||t.end_ns==sentinel||t.begin_cycle==sentinel||t.end_cycle==sentinel||t.smid==~unsigned(0)
       ||t.end_ns<=t.begin_ns||t.end_cycle<=t.begin_cycle;o.checked_elements+=10;
   o.stamps.push_back({t.begin_ns,t.end_ns,t.begin_cycle,t.end_cycle,t.smid});
  }
  if(c.mode>=5)for(gh::u64 n=0;n<ring_words;++n)equal(back[n],initial[n]);
  o.method="S17_complete_final_tiles_sums_guards_and_cluster_exit_v1";
  o.input_conditions="nonuniform_uint32_cluster_pattern;capture_false;all participants live through exit";
  std::string failure;try{gh::envelope(o);}catch(const std::exception& e){failure=e.what();}
  if(persist||!failure.empty()){
   Check ch;ch.checked=o.checked_elements;
   if(W){save(ch,invocation,"final_tiles",values,{B,W});save(ch,invocation,"guards",gs,{B,8});}
   save(ch,invocation,"completion",word_artifacts::split_u64(completion),{B,6,2});
   std::vector<u64> encoded;for(const auto& t:timing)encoded.insert(encoded.end(),{t.begin_ns,t.end_ns,t.begin_cycle,t.end_cycle,t.smid});
   save(ch,invocation,"stamps",word_artifacts::split_u64(encoded),{B,5,2});
   if(c.mode<4)save(ch,invocation,"sums",ss,{B,128});if(c.mode>=5)save(ch,invocation,"source_ring",back,{G,32,4096});
   std::ostringstream identity;identity<<"{\"case_id\":"<<gh::quote(id)<<",\"iterations\":"<<i<<",\"seed\":"<<seed
    <<",\"invocation\":"<<invocation<<",\"blocks\":"<<B<<",\"clusters\":"<<G<<",\"cluster_size\":"<<c.C
    <<",\"capture\":false,\"errors\":"<<o.errors<<",\"diagnostic\":"<<gh::quote(failure)<<"}";
   save_identity("cluster_"+std::to_string(invocation)+"_identity.json",identity.str()+"\n");if(persist)final_check=std::move(ch);
  }
  if(!failure.empty())throw std::runtime_error(failure);return o;
 };
 gh::Warmup warm;if(!pilot)warm=gh::warmup([&](){return execute(false);});const auto observed=execute(true);
 const gh::u64 requested=cluster_reference::requested(c.mode,c.C,G,i),work=c.mode==4?gh::u64(G)*i:requested;
 const unsigned receivers=c.mode==6?c.C:c.mode==5?1:0;
 std::ostringstream extra;extra<<"\"phase\":"<<gh::quote(pilot?"pilot":"formal")<<",\"performance_eligible\":"<<(pilot?"false":"true")
  <<",\"warmup_executed\":"<<(pilot?"false":"true")<<",\"capture\":false,\"kernel_symbol\":"<<gh::quote(c.symbol)
  <<",\"cluster_size\":"<<c.C<<",\"clusters\":"<<G<<",\"form\":"<<gh::quote(c.form)<<",\"mask\":"<<(c.mode>=5?cluster_reference::mask(c.mode==6,c.C):0)
  <<",\"source_request_bytes\":"<<(c.mode>=5?requested:0)<<",\"total_receiver_bytes\":"<<(c.mode>=5?requested*receivers:0)
  <<",\"required_write_readback_bytes\":"<<(c.mode==2||c.mode==3?requested:0)
  <<",\"registers_per_thread\":"<<cap.attr.numRegs<<",\"static_smem_bytes\":"<<cap.attr.sharedSizeBytes<<",\"dynamic_smem_bytes\":0,\"local_size_bytes\":"<<cap.attr.localSizeBytes
  <<",\"potential_cluster_size\":"<<cap.potential<<",\"active_cluster_capacity\":"<<cap.active<<",\"occupancy_is_upper_bound\":true"
  <<",\"post_timing_export_bytes\":"<<gh::u64(B)*W*4<<",\"final_invocation\":"<<invocation<<",\"full_output_artifacts\":[";
 for(unsigned n=0;n<final_check.artifacts.size();++n){const auto& a=final_check.artifacts[n];extra<<(n?",":"")<<"{\"path\":"<<gh::quote(a.path)<<",\"sha256\":"<<gh::quote(a.sha)<<",\"dtype\":\"uint32\",\"shape\":[";
  for(unsigned k=0;k<a.shape.size();++k)extra<<(k?",":"")<<a.shape[k];extra<<"]}";}extra<<"]";
 gh::emit_trial(id,i,seed,128,all?"cluster_grid":"one_cluster",c.mode==4?"cluster_phase":"byte",work,
     c.mode!=4?requested:0,c.mode==2||c.mode==3?requested:0,observed,warm,extra.str());
 GH_CUDA(cudaEventDestroy(begin));GH_CUDA(cudaEventDestroy(end));GH_CUDA(cudaFree(stamps));GH_CUDA(cudaFree(done));
 GH_CUDA(cudaFree(guards));GH_CUDA(cudaFree(sums));GH_CUDA(cudaFree(trace));GH_CUDA(cudaFree(global));return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}

int main(int argc,char** argv){
 if(argc>1&&(std::string(argv[1])=="device"||std::string(argv[1])=="cluster-device"||std::string(argv[1])=="validate-only"))return s17_legacy_short_main(argc,argv);
 return formal_main(argc,argv);
}
