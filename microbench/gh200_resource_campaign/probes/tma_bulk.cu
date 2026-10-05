#include "../common/probe_runtime.cuh"
#include "../common/tma_bulk_reference.hpp"
#include "../common/word_artifacts.hpp"

constexpr unsigned tb_threads=128;
constexpr unsigned tb_slots=32;
__device__ __forceinline__ unsigned tb_shared(const void* p){return static_cast<unsigned>(__cvta_generic_to_shared(p));}
__device__ __forceinline__ gh::u64 tb_ns(){gh::u64 x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)::"memory");return x;}
__device__ __forceinline__ gh::u64 tb_cycle(){gh::u64 x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}

__device__ __forceinline__ gh::u64 tb_wait(unsigned barrier,gh::u64 token,std::uint64_t* failure) {
  const gh::u64 start=tb_ns();gh::u64 attempts=0;
  for(;;) {
    unsigned done;
    asm volatile("{ .reg .pred p; mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2, 64; selp.b32 %0, 1, 0, p; }"
      :"=r"(done):"r"(barrier),"l"(token):"memory");
    ++attempts;if(done)return attempts;
    if(tb_ns()-start>=1000000000ull) {
      // Best-effort failure marker. A poisoned CUDA context may make readback
      // impossible; the controlled runner still retains stderr and failure.
      *reinterpret_cast<volatile std::uint64_t*>(failure)=1;__threadfence();
      asm volatile("trap;":::"memory");
    }
  }
}

template<unsigned Q,bool G2S> __device__ __forceinline__ void tb_transport(
    unsigned* global,int iterations,unsigned seed,bool capture,unsigned* trace,
    gh::Stamp* stamps,std::uint64_t* completion) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto* tile=reinterpret_cast<unsigned*>(storage);constexpr unsigned words=Q/4;
  const unsigned shared=tb_shared(tile),barrier=tb_shared(storage+Q);
  for(unsigned word=threadIdx.x;word<words;word+=tb_threads) {
    const unsigned value=G2S?17u*unsigned((gh::u64(blockIdx.x)*tb_slots)*words+word)+seed:
                                29u*unsigned(gh::u64(blockIdx.x)*words+word)+seed;
    tile[word]=G2S?~value:value;
  }
  if constexpr(G2S)if(threadIdx.x==0)asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(barrier):"memory");
  // All generic tile producers publish before the async proxy. Thread0 also
  // publishes the barrier initialization; this preparation is outside timing.
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  __syncthreads();
  gh::Stamp stamp{};gh::u64 completed=0,attempts=0;
  if(threadIdx.x==0) {
    asm volatile("mov.u32 %0, %%smid;":"=r"(stamp.smid));stamp.begin_ns=tb_ns();stamp.begin_cycle=tb_cycle();
  }
  __syncthreads();
  #pragma unroll 1
  for(int iteration=0;iteration<iterations;++iteration) {
    const gh::u64 offset=(gh::u64(blockIdx.x)*tb_slots+unsigned(iteration)%tb_slots)*words;
    if(threadIdx.x==0) {
      if constexpr(G2S) {
        asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"::"r"(barrier),"r"(Q):"memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
          ::"r"(shared),"l"(global+offset),"r"(Q),"r"(barrier):"memory");
        gh::u64 token;asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];":"=l"(token):"r"(barrier):"memory");
        attempts+=tb_wait(barrier,token,completion+gh::u64(blockIdx.x)*5+2);
      } else {
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"::"l"(global+offset),"r"(shared),"r"(Q):"memory");
        asm volatile("cp.async.bulk.commit_group;":::"memory");
        asm volatile("cp.async.bulk.wait_group 0;":::"memory");
      }
      ++completed;
    }
    __syncthreads();
    if(capture) {
      for(unsigned word=threadIdx.x;word<words;word+=tb_threads) {
        const unsigned value=G2S?reinterpret_cast<volatile unsigned*>(tile)[word]:reinterpret_cast<volatile unsigned*>(global)[offset+word];
        trace[(gh::u64(blockIdx.x)*iterations+iteration)*words+word]=value;
      }
      // Every consumer finishes before a new phase can overwrite the shared
      // destination or revisit a global slot. Absent in formal capture=false.
      __syncthreads();
    }
  }
  if(threadIdx.x==0){stamp.end_cycle=tb_cycle();stamp.end_ns=tb_ns();stamps[blockIdx.x]=stamp;}
  if constexpr(G2S) {
    if(!capture)for(unsigned word=threadIdx.x;word<words;word+=tb_threads)trace[gh::u64(blockIdx.x)*words+word]=tile[word];
    __syncthreads();
    if(threadIdx.x==0)asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(barrier):"memory");
  }
  if(threadIdx.x==0) {
    const gh::u64 base=gh::u64(blockIdx.x)*5;completion[base]=completed;completion[base+1]=attempts;
    completion[base+2]=0;completion[base+3]=0;completion[base+4]=G2S?0:completed;
  }
}

template<unsigned Q> __device__ __forceinline__ void tb_source_release(unsigned* global,unsigned seed,
    unsigned* source_after,gh::Stamp* stamps,std::uint64_t* completion,std::uint64_t* clocks) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto* tile=reinterpret_cast<unsigned*>(storage);constexpr unsigned words=Q/4;
  for(unsigned word=threadIdx.x;word<words;word+=tb_threads)tile[word]=29u*unsigned(gh::u64(blockIdx.x)*words+word)+seed;
  asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();
  gh::Stamp stamp{};
  if(threadIdx.x==0){asm volatile("mov.u32 %0, %%smid;":"=r"(stamp.smid));stamp.begin_ns=tb_ns();stamp.begin_cycle=tb_cycle();}
  __syncthreads();gh::u64 start=0,released=0,full=0;
  if(threadIdx.x==0) {
    start=tb_cycle();
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
      ::"l"(global+gh::u64(blockIdx.x)*tb_slots*words),"r"(tb_shared(tile)),"r"(Q):"memory");
    asm volatile("cp.async.bulk.commit_group;":::"memory");
    asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");released=tb_cycle();
  }
  __syncthreads();
  for(unsigned word=threadIdx.x;word<words;word+=tb_threads) {
    const unsigned original=29u*unsigned(gh::u64(blockIdx.x)*words+word)+seed;
    asm volatile("st.shared.u32 [%0], %1;"::"r"(tb_shared(tile+word)),"r"(~original):"memory");
  }
  __syncthreads();
  if(threadIdx.x==0){asm volatile("cp.async.bulk.wait_group 0;":::"memory");full=tb_cycle();}
  __syncthreads();
  if(threadIdx.x==0) {
    stamp.end_cycle=tb_cycle();stamp.end_ns=tb_ns();stamps[blockIdx.x]=stamp;
    const gh::u64 c=gh::u64(blockIdx.x)*5;completion[c]=1;completion[c+1]=0;completion[c+2]=0;completion[c+3]=1;completion[c+4]=1;
    const gh::u64 t=gh::u64(blockIdx.x)*3;clocks[t]=start;clocks[t+1]=released;clocks[t+2]=full;
  }
  for(unsigned word=threadIdx.x;word<words;word+=tb_threads)source_after[gh::u64(blockIdx.x)*words+word]=tile[word];
}

#define TB_FORMS(Q) \
extern "C" __global__ void tb_g2s_q##Q(unsigned* global,int iterations,unsigned seed,bool capture,unsigned* trace,gh::Stamp* stamps,std::uint64_t* completion){tb_transport<Q,true>(global,iterations,seed,capture,trace,stamps,completion);} \
extern "C" __global__ void tb_s2g_q##Q(unsigned* global,int iterations,unsigned seed,bool capture,unsigned* trace,gh::Stamp* stamps,std::uint64_t* completion){tb_transport<Q,false>(global,iterations,seed,capture,trace,stamps,completion);} \
extern "C" __global__ void tb_release_q##Q(unsigned* global,unsigned seed,unsigned* source_after,gh::Stamp* stamps,std::uint64_t* completion,std::uint64_t* clocks){tb_source_release<Q>(global,seed,source_after,stamps,completion,clocks);}
TB_FORMS(1024)
TB_FORMS(4096)
TB_FORMS(8192)
TB_FORMS(16384)
TB_FORMS(32768)
TB_FORMS(65536)

struct TbCase {const char* id;unsigned bytes;bool g2s,all_gpu;const char* symbol;const void* function;const char* release_symbol;const void* release_function;};
#define TB_CASES(Q,K) \
 {"gmem_to_smem_" #K "kib_one_cta",Q,true,false,"tb_g2s_q" #Q,reinterpret_cast<const void*>(tb_g2s_q##Q),nullptr,nullptr}, \
 {"gmem_to_smem_" #K "kib_all_gpu",Q,true,true,"tb_g2s_q" #Q,reinterpret_cast<const void*>(tb_g2s_q##Q),nullptr,nullptr}, \
 {"smem_to_gmem_" #K "kib_one_cta",Q,false,false,"tb_s2g_q" #Q,reinterpret_cast<const void*>(tb_s2g_q##Q),"tb_release_q" #Q,reinterpret_cast<const void*>(tb_release_q##Q)}, \
 {"smem_to_gmem_" #K "kib_all_gpu",Q,false,true,"tb_s2g_q" #Q,reinterpret_cast<const void*>(tb_s2g_q##Q),"tb_release_q" #Q,reinterpret_cast<const void*>(tb_release_q##Q)}
static const TbCase tb_cases[]={TB_CASES(1024,1),TB_CASES(4096,4),TB_CASES(8192,8),TB_CASES(16384,16),TB_CASES(32768,32),TB_CASES(65536,64)};
struct TbProfile {const char* id;unsigned iterations,seed;bool release;const char* input;};
static const TbProfile tb_profiles[]={
 {"bulk_short_1_seed0_v1",1,0,false,"nonuniform_word_pattern"},
 {"bulk_short_2_seed3_v1",2,3,false,"nonuniform_word_pattern"},
 {"bulk_short_33_seed4294967295_v1",33,4294967295u,false,"nonuniform_word_pattern"},
 {"bulk_source_release_one_request_v1",1,3,true,"original_pattern_then_source_complement"}};
struct TbArtifact {std::string path,sha;std::vector<gh::u64> shape;};
struct TbCheck {gh::u64 errors=0,checked=0;bool completed=false;std::vector<unsigned> ctas;std::vector<TbArtifact> artifacts;};
static void tb_save(TbCheck& check,const char* name,const std::vector<unsigned>& words,std::vector<gh::u64> shape) {
  gh::u64 count=1;for(auto n:shape)count=tma_bulk_reference::multiply(count,n);
  if(count!=words.size())throw std::runtime_error("S14 artifact shape mismatch");
  const std::string digest=word_artifacts::write_words(name,words);check.artifacts.push_back({name,digest,std::move(shape)});
}
static void tb_emit(const TbCase& c,const TbProfile& p,unsigned blocks,const cudaFuncAttributes& attr,int occupancy,
    gh::u64 allocation,const TbCheck& check,bool failed) {
  const gh::u64 words=c.bytes/4,expected=p.release?gh::u64(blocks)*33*words+8:
    gh::u64(blocks)*p.iterations*words+(c.g2s?0:gh::u64(blocks)*32*words)+8;
  const char* completion=p.release?"separate_source_release_then_full_wait":c.g2s?"mbarrier_acquire_then_CTA_gate":"bulk_wait_group_0_then_CTA_gate";
  std::cout<<"{\"schema_version\":2,\"validation_schema_version\":1,\"type\":\"validation\",\"case_id\":"<<gh::quote(c.id)
    <<",\"profile_id\":"<<gh::quote(p.id)<<",\"seed\":"<<p.seed<<",\"scope\":"<<gh::quote(c.all_gpu?"all_gpu":"one_cta")
    <<",\"threads\":128,\"blocks\":"<<blocks<<",\"errors\":"<<check.errors+(failed?1:0)
    <<",\"performance_eligible\":false,\"warmup_executed\":false,\"pilot_executed\":false,\"target_launches\":[{\"launch_index\":0,\"iterations\":"<<p.iterations
    <<",\"input_profile\":"<<gh::quote(p.input)<<",\"threads\":128,\"blocks\":"<<blocks<<"}],\"checks\":[{\"launch_index\":0"
    <<",\"reference_model\":\"tma_bulk_ring_word_reference_v1\",\"reference_sha256\":\"98d20cfdfbdf5fc627976764ce87ceba032c94f09341bdac0e41ac95ec7bcf7b\",\"comparison\":\"exact\",\"tolerance_id\":null"
    <<",\"checked_elements\":"<<check.checked<<",\"expected_elements\":"<<expected<<",\"errors\":"<<check.errors
    <<",\"completed\":"<<(check.completed?"true":"false")<<",\"verified_CTA_ids\":[";
  for(size_t i=0;i<check.ctas.size();++i)std::cout<<(i?",":"")<<check.ctas[i];
  std::cout<<"],\"output_artifacts\":[";
  for(size_t i=0;i<check.artifacts.size();++i) {
    const auto& a=check.artifacts[i];std::cout<<(i?",":"")<<"{\"path\":"<<gh::quote(a.path)<<",\"sha256\":"<<gh::quote(a.sha)<<",\"dtype\":\"uint32\",\"shape\":[";
    for(size_t j=0;j<a.shape.size();++j)std::cout<<(j?",":"")<<a.shape[j];
    std::cout<<"],\"evidence_kind\":\"full_values\"}";
  }
  std::cout<<"]}],\"resource_identity\":{\"kernel_symbol\":"<<gh::quote(p.release?c.release_symbol:c.symbol)
    <<",\"registers_per_thread\":"<<attr.numRegs<<",\"static_smem_bytes\":"<<attr.sharedSizeBytes<<",\"dynamic_smem_bytes\":"<<c.bytes+32
    <<",\"local_size_bytes\":"<<attr.localSizeBytes<<",\"occupancy_limit_ctas_per_sm\":"<<occupancy
    <<",\"extensions\":{\"payload_bytes\":"<<c.bytes<<",\"global_slots_per_cta\":32,\"global_allocation_bytes\":"<<allocation
    <<",\"completion_kind\":"<<gh::quote(completion)<<",\"validation_role\":"<<gh::quote(p.release?"source_release":"short_transport")<<"}}}\n";
}

int main(int argc,char** argv) try {
  const auto device=gh::device();gh::emit_device(device);
  if(argc==2&&std::string(argv[1])=="device")return 0;
  // No formal execution until the existing framework has a reviewed whole-family
  // 84-receipt gate and calibrated-iteration binding. Never warm up a first case.
  if(argc!=5||std::string(argv[1])!="validate-only")throw std::runtime_error("S14 currently requires validate-only CASE PROFILE SEED; formal family gate/calibration not enabled");
  const TbCase* selected=nullptr;for(const auto& c:tb_cases)if(c.id==std::string(argv[2]))selected=&c;
  const TbProfile* planned=nullptr;for(const auto& p:tb_profiles)if(p.id==std::string(argv[3]))planned=&p;
  if(!selected||!planned)throw std::runtime_error("unknown S14 case/profile");const auto& c=*selected;const auto& p=*planned;
  unsigned seed=gh::integer(argv[4],0,4294967295ull);if(seed!=p.seed||p.release&&c.g2s)throw std::runtime_error("S14 frozen profile/seed/direction mismatch");
  const void* target=p.release?c.release_function:c.function;const size_t shared=c.bytes+32;
  GH_CUDA(cudaFuncSetAttribute(target,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  cudaFuncAttributes attr{};GH_CUDA(cudaFuncGetAttributes(&attr,target));int occupancy=0;
  GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,target,tb_threads,shared));
  if(occupancy<1||attr.localSizeBytes||attr.sharedSizeBytes+shared>device.prop.sharedMemPerBlockOptin)throw std::runtime_error("S14 unsupported allocation/spill/occupancy");
  const unsigned blocks=c.all_gpu?std::min(4,occupancy)*device.prop.multiProcessorCount:1;
  const gh::u64 words=c.bytes/4,allocation=tma_bulk_reference::allocation_bytes(blocks,c.bytes),ring_words=(allocation-32)/4;
  const gh::u64 capture_words=p.release?gh::u64(blocks)*words:gh::u64(blocks)*p.iterations*words;
  const gh::u64 capture_bytes=tma_bulk_reference::multiply(capture_words,4),control_bytes=gh::u64(blocks)*(sizeof(gh::Stamp)+5*sizeof(std::uint64_t)+(p.release?3*sizeof(std::uint64_t):0));
  size_t free_bytes=0,total_bytes=0;GH_CUDA(cudaMemGetInfo(&free_bytes,&total_bytes));
  if(allocation>free_bytes||capture_bytes>free_bytes-allocation||control_bytes>free_bytes-allocation-capture_bytes)throw std::runtime_error("S14 insufficient GPU allocation budget");
  unsigned *device_global=nullptr,*device_capture=nullptr;gh::Stamp* stamps=nullptr;std::uint64_t *completion=nullptr,*clocks=nullptr;
  GH_CUDA(cudaMalloc(&device_global,allocation));GH_CUDA(cudaMalloc(&device_capture,capture_bytes));
  GH_CUDA(cudaMalloc(&stamps,blocks*sizeof(gh::Stamp)));GH_CUDA(cudaMalloc(&completion,blocks*5*sizeof(std::uint64_t)));
  if(p.release)GH_CUDA(cudaMalloc(&clocks,blocks*3*sizeof(std::uint64_t)));
  {
    std::vector<unsigned> initial(ring_words+8);
    for(unsigned i=0;i<4;++i){initial[i]=0xd15ea5e0u+i;initial[ring_words+4+i]=0xd15ea5e4u+i;}
    for(gh::u64 index=0;index<ring_words;++index) {
      const gh::u64 block=index/(tb_slots*words),slot=(index/words)%tb_slots,word=index%words;
      initial[index+4]=c.g2s?tma_bulk_reference::global_input_word(block,slot,word,blocks,c.bytes,seed):
        tma_bulk_reference::output_poison(tma_bulk_reference::shared_input_word(block,word,blocks,c.bytes,seed));
    }
    GH_CUDA(cudaMemcpy(device_global,initial.data(),allocation,cudaMemcpyHostToDevice));
  }
  {
    std::vector<unsigned> poison(capture_words);
    for(gh::u64 index=0;index<capture_words;++index) {
      const gh::u64 block=index/((p.release?1:p.iterations)*words),iteration=(index/words)%(p.release?1:p.iterations),word=index%words;
      const unsigned original=c.g2s?tma_bulk_reference::global_input_word(block,iteration,word,blocks,c.bytes,seed):tma_bulk_reference::shared_input_word(block,word,blocks,c.bytes,seed);
      poison[index]=p.release?original:~original;
    }
    GH_CUDA(cudaMemcpy(device_capture,poison.data(),capture_bytes,cudaMemcpyHostToDevice));
  }
  GH_CUDA(cudaMemset(stamps,0xff,blocks*sizeof(gh::Stamp)));GH_CUDA(cudaMemset(completion,0xff,blocks*5*sizeof(std::uint64_t)));
  if(p.release)GH_CUDA(cudaMemset(clocks,0xff,blocks*3*sizeof(std::uint64_t)));
  TbCheck check;cudaEvent_t event;GH_CUDA(cudaEventCreate(&event));unsigned details=0;
  auto mismatch=[&](unsigned actual,unsigned expected,const char* label,gh::u64 index) {
    if(actual!=expected){++check.errors;if(details++<8)std::cerr<<"S14 "<<label<<" index="<<index<<" actual="<<actual<<" expected="<<expected<<'\n';}++check.checked;
  };
  try {
    unsigned* payload=device_global+4;int iterations=p.iterations;bool capture=true;
    if(p.release){void* args[]={&payload,&seed,&device_capture,&stamps,&completion,&clocks};GH_CUDA(cudaLaunchKernel(target,dim3(blocks),dim3(tb_threads),args,shared));}
    else {void* args[]={&payload,&iterations,&seed,&capture,&device_capture,&stamps,&completion};GH_CUDA(cudaLaunchKernel(target,dim3(blocks),dim3(tb_threads),args,shared));}
    GH_CUDA(cudaEventRecord(event));GH_CUDA(cudaEventSynchronize(event));
    std::vector<gh::Stamp> host_stamps(blocks);std::vector<std::uint64_t> counts(blocks*5);
    GH_CUDA(cudaMemcpy(host_stamps.data(),stamps,blocks*sizeof(gh::Stamp),cudaMemcpyDeviceToHost));
    GH_CUDA(cudaMemcpy(counts.data(),completion,blocks*5*sizeof(std::uint64_t),cudaMemcpyDeviceToHost));
    bool completed=true;
    for(unsigned block=0;block<blocks;++block) {
      const auto& stamp=host_stamps[block];const auto poison=std::numeric_limits<gh::u64>::max();
      const gh::u64 offset=gh::u64(block)*5;
      const bool stamp_ok=stamp.begin_ns!=poison&&stamp.end_ns!=poison&&stamp.begin_cycle!=poison&&stamp.end_cycle!=poison&&stamp.smid!=~unsigned(0)&&stamp.end_ns>=stamp.begin_ns&&stamp.end_cycle>=stamp.begin_cycle;
      const bool count_ok=counts[offset]==p.iterations&&(c.g2s?counts[offset+1]>=p.iterations&&counts[offset+1]!=poison:counts[offset+1]==0)&&counts[offset+2]==0&&counts[offset+3]==unsigned(p.release)&&counts[offset+4]==(c.g2s?0:p.iterations);
      if(!stamp_ok||!count_ok){completed=false;++check.errors;if(details++<8)std::cerr<<"S14 CTA="<<block<<" incomplete stamp/lifecycle counters\n";}else check.ctas.push_back(block);
    }
    tb_save(check,"bulk_completion.u32le",word_artifacts::split_u64(counts),{blocks,5,2});
    {
      std::vector<unsigned> values(capture_words);GH_CUDA(cudaMemcpy(values.data(),device_capture,capture_bytes,cudaMemcpyDeviceToHost));
      for(gh::u64 index=0;index<capture_words;++index) {
        const gh::u64 block=index/((p.release?1:p.iterations)*words),iteration=(index/words)%(p.release?1:p.iterations),word=index%words;
        unsigned expected=c.g2s?tma_bulk_reference::global_input_word(block,iteration,word,blocks,c.bytes,seed):tma_bulk_reference::shared_input_word(block,word,blocks,c.bytes,seed);
        if(p.release)expected=~expected;mismatch(values[index],expected,p.release?"source_after":"trace",index);
      }
      if(p.release)tb_save(check,"bulk_source_after.u32le",values,{blocks,words});else tb_save(check,"bulk_trace.u32le",values,{blocks,p.iterations,words});
    }
    if(!c.g2s) {
      std::vector<unsigned> ring(ring_words);GH_CUDA(cudaMemcpy(ring.data(),payload,ring_words*4,cudaMemcpyDeviceToHost));
      for(gh::u64 index=0;index<ring_words;++index) {
        const gh::u64 block=index/(tb_slots*words),slot=(index/words)%tb_slots,word=index%words;
        mismatch(ring[index],tma_bulk_reference::destination_word(block,slot,word,blocks,c.bytes,p.iterations,seed),"ring",index);
      }
      tb_save(check,"bulk_ring.u32le",ring,{blocks,tb_slots,words});
    }
    std::vector<unsigned> guards(8);GH_CUDA(cudaMemcpy(guards.data(),device_global,16,cudaMemcpyDeviceToHost));GH_CUDA(cudaMemcpy(guards.data()+4,payload+ring_words,16,cudaMemcpyDeviceToHost));
    for(unsigned i=0;i<8;++i)mismatch(guards[i],0xd15ea5e0u+i,"guard",i);tb_save(check,"bulk_guards.u32le",guards,{2,4});
    if(p.release) {
      std::vector<std::uint64_t> times(blocks*3);GH_CUDA(cudaMemcpy(times.data(),clocks,blocks*3*sizeof(std::uint64_t),cudaMemcpyDeviceToHost));
      for(unsigned block=0;block<blocks;++block) {
        const auto offset=gh::u64(block)*3;
        if(times[offset]>times[offset+1]||times[offset+1]>times[offset+2]||times[offset+2]==std::numeric_limits<std::uint64_t>::max()){
          completed=false;++check.errors;if(details++<8)std::cerr<<"S14 CTA="<<block<<" invalid source-release clocks\n";
        }
      }
      tb_save(check,"bulk_release_clocks.u32le",word_artifacts::split_u64(times),{blocks,3,2});
    }
    check.completed=completed;
    if(check.errors)throw std::runtime_error("S14 full-word/lifecycle validation failed");
  } catch(const std::exception& error){tb_emit(c,p,blocks,attr,occupancy,allocation,check,true);std::cerr<<error.what()<<'\n';return 2;}
  tb_emit(c,p,blocks,attr,occupancy,allocation,check,false);
  GH_CUDA(cudaEventDestroy(event));if(clocks)GH_CUDA(cudaFree(clocks));GH_CUDA(cudaFree(completion));GH_CUDA(cudaFree(stamps));GH_CUDA(cudaFree(device_capture));GH_CUDA(cudaFree(device_global));return 0;
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
