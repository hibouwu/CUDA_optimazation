#include "../common/probe_runtime.cuh"
#include "../common/global_duplex_reference.hpp"

constexpr unsigned gd_threads=256;

__device__ __forceinline__ uint4 gd_load(const uint4* address,bool ca) {
  uint4 value;
  if(ca)asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(value.x),"=r"(value.y),"=r"(value.z),"=r"(value.w):"l"(address):"memory");
  else asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(value.x),"=r"(value.y),"=r"(value.z),"=r"(value.w):"l"(address):"memory");
  return value;
}
__device__ __forceinline__ void gd_store(uint4* address,uint4 value) {
  asm volatile("st.global.wb.v4.u32 [%0],{%1,%2,%3,%4};"::"l"(address),"r"(value.x),"r"(value.y),"r"(value.z),"r"(value.w):"memory");
}
template<int Ratio> __device__ __forceinline__ gh::u64 gd_slot(gh::u64 group,unsigned request,gh::u64 groups) {
  gh::u64 slot=group*Ratio+request;
  // 0 <= slot < Ratio*groups: at most Ratio-1 wraps. Avoid variable division.
  #pragma unroll
  for(int wrap=0;wrap<Ratio-1;++wrap)if(slot>=groups)slot-=groups;
  return slot;
}
template<int Reads,int Writes,bool CacheAll,bool Copy>
__device__ __forceinline__ void gd_body(const uint4* input,uint4* output,gh::u64 vectors,
    int iterations,unsigned seed,gh::Stamp* stamps,unsigned* sums) {
  __shared__ volatile unsigned drain[gd_threads];
  const gh::u64 total_threads=gh::u64(gridDim.x)*gd_threads;
  const gh::u64 thread=gh::u64(blockIdx.x)*gd_threads+threadIdx.x;
  const gh::u64 groups=vectors/total_threads;
  gh::Stamp stamp{};
  if(threadIdx.x==0) {
    asm volatile("mov.u32 %0, %%smid;":"=r"(stamp.smid));
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(stamp.begin_ns));
    asm volatile("mov.u64 %0, %%clock64;":"=l"(stamp.begin_cycle));
  }
  __syncthreads();
  unsigned sum=0;
  #pragma unroll 1
  for(int iteration=0;iteration<iterations;++iteration) {
    #pragma unroll 1
    for(gh::u64 group=0;group<groups;++group) {
      // The frozen issue order is r reads, then w independent writes. Copy
      // instead stores the actual loaded vector immediately, at the same index.
      #pragma unroll
      for(int request=0;request<Reads;++request) {
        const gh::u64 index=gd_slot<Reads>(group,request,groups)*total_threads+thread;
        const uint4 value=gd_load(input+index,CacheAll);
        sum+=value.x+value.y+value.z+value.w;
        if constexpr(Copy)gd_store(output+index,value);
      }
      if constexpr(!Copy) {
        #pragma unroll
        for(int request=0;request<Writes;++request) {
          const gh::u64 index=gd_slot<Writes>(group,request,groups)*total_threads+thread;
          const unsigned word=unsigned(index*4);
          const uint4 value=make_uint4(29u*word+seed,29u*(word+1)+seed,29u*(word+2)+seed,29u*(word+3)+seed);
          gd_store(output+index,value);
        }
      }
    }
  }
  drain[threadIdx.x]=sum;
  if constexpr(Writes>0)__threadfence();
  __syncthreads();
  if(threadIdx.x==0) {
    asm volatile("mov.u64 %0, %%clock64;":"=l"(stamp.end_cycle));
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(stamp.end_ns));
    stamps[blockIdx.x]=stamp;
  }
  sums[thread]=sum;
}
#define GD_KERNEL(NAME,R,W,CA,COPY) \
extern "C" __global__ void NAME(const uint4* input,uint4* output,gh::u64 vectors,int iterations,unsigned seed,gh::Stamp* stamps,unsigned* sums) {gd_body<R,W,CA,COPY>(input,output,vectors,iterations,seed,stamps,sums);}
GD_KERNEL(gd_read_ca_r1_w0,1,0,true,false)
GD_KERNEL(gd_read_cg_r1_w0,1,0,false,false)
GD_KERNEL(gd_write_wb_r0_w1,0,1,false,false)
GD_KERNEL(gd_copy_cg_r1_w1,1,1,false,true)
GD_KERNEL(gd_independent_cg_r1_w1,1,1,false,false)
GD_KERNEL(gd_independent_cg_r2_w1,2,1,false,false)
GD_KERNEL(gd_independent_cg_r4_w1,4,1,false,false)
GD_KERNEL(gd_independent_cg_r1_w2,1,2,false,false)
GD_KERNEL(gd_independent_cg_r1_w4,1,4,false,false)

struct GdCase {const char* id;int reads,writes;bool large,copy;const char* symbol;const void* function;};
#define GD_CASES(ID,NAME,R,W,COPY) \
 {ID "_small",R,W,false,COPY,#NAME,reinterpret_cast<const void*>(NAME)}, \
 {ID "_large",R,W,true,COPY,#NAME,reinterpret_cast<const void*>(NAME)}
static const GdCase gd_cases[]={
 GD_CASES("read_ca",gd_read_ca_r1_w0,1,0,false),
 GD_CASES("read_cg",gd_read_cg_r1_w0,1,0,false),
 GD_CASES("write",gd_write_wb_r0_w1,0,1,false),
 GD_CASES("dependent_copy",gd_copy_cg_r1_w1,1,1,true),
 GD_CASES("independent_r1_w1",gd_independent_cg_r1_w1,1,1,false),
 GD_CASES("independent_r2_w1",gd_independent_cg_r2_w1,2,1,false),
 GD_CASES("independent_r4_w1",gd_independent_cg_r4_w1,4,1,false),
 GD_CASES("independent_r1_w2",gd_independent_cg_r1_w2,1,2,false),
 GD_CASES("independent_r1_w4",gd_independent_cg_r1_w4,1,4,false)
};
struct GdCheck {int iterations;gh::Observation observation;bool completed=false;std::vector<unsigned> verified_ctas;};
static void gd_resource(const GdCase& c,const cudaFuncAttributes& attr,int occupancy,
    gh::u64 requested,gh::u64 bytes,gh::u64 aggregate) {
  std::cout<<"{\"kernel_symbol\":"<<gh::quote(c.symbol)<<",\"registers_per_thread\":"<<attr.numRegs
    <<",\"static_smem_bytes\":"<<attr.sharedSizeBytes<<",\"dynamic_smem_bytes\":0,\"local_size_bytes\":"<<attr.localSizeBytes
    <<",\"occupancy_limit_ctas_per_sm\":"<<occupancy<<",\"extensions\":{\"requested_array_bytes\":"<<requested
    <<",\"array_bytes\":"<<bytes<<",\"aggregate_array_bytes\":"<<aggregate<<",\"read_requests_per_group\":"<<c.reads
    <<",\"write_requests_per_group\":"<<c.writes<<"}}";
}
static void gd_validation(const GdCase& c,unsigned seed,unsigned blocks,const cudaFuncAttributes& attr,int occupancy,
    gh::u64 requested,gh::u64 bytes,gh::u64 aggregate,const std::vector<GdCheck>& checks,bool failed) {
  gh::u64 errors=failed?1:0;for(const auto& check:checks)errors+=check.observation.errors;
  const gh::u64 expected=(c.reads?gh::u64(blocks)*gd_threads:0)+(c.writes?bytes/4:0);
  std::cout<<"{\"schema_version\":2,\"validation_schema_version\":1,\"type\":\"validation\",\"case_id\":"<<gh::quote(c.id)
    <<",\"profile_id\":\"short_global_duplex_1_2\",\"seed\":"<<seed<<",\"scope\":\"all_gpu\",\"threads\":256,\"blocks\":"<<blocks
    <<",\"errors\":"<<errors<<",\"performance_eligible\":false,\"warmup_executed\":false,\"pilot_executed\":false,\"target_launches\":[";
  for(size_t i=0;i<checks.size();++i)std::cout<<(i?",":"")<<"{\"launch_index\":"<<i<<",\"iterations\":"<<checks[i].iterations
    <<",\"input_profile\":\"address_seed_nonuniform\",\"threads\":256,\"blocks\":"<<blocks<<"}";
  std::cout<<"],\"checks\":[";
  for(size_t i=0;i<checks.size();++i) {
    const auto& check=checks[i];std::cout<<(i?",":"")<<"{\"launch_index\":"<<i
      <<",\"reference_model\":\"global_duplex_modular_sweep_v1\",\"reference_sha256\":\"e6b29111204b1f379188737712aa75334cf9ae1c813a51f02efea44b20c5a696\",\"comparison\":\"exact\",\"tolerance_id\":null"
      <<",\"checked_elements\":"<<check.observation.checked_elements<<",\"expected_elements\":"<<expected
      <<",\"errors\":"<<check.observation.errors<<",\"completed\":"<<(check.completed?"true":"false")<<",\"verified_CTA_ids\":[";
    for(size_t j=0;j<check.verified_ctas.size();++j)std::cout<<(j?",":"")<<check.verified_ctas[j];
    std::cout<<"],\"output_artifacts\":[]}";
  }
  std::cout<<"],\"resource_identity\":";gd_resource(c,attr,occupancy,requested,bytes,aggregate);std::cout<<"}\n";
}

int main(int argc,char** argv) try {
  const auto device=gh::device();gh::emit_device(device);
  if(argc==2&&std::string(argv[1])=="device")return 0;
  const bool short_check=argc>1&&std::string(argv[1])=="validate-only";
  if(argc!=(short_check?5:4))throw std::runtime_error("CASE_ID 16 SEED | validate-only CASE_ID short_global_duplex_1_2 SEED");
  if(short_check&&std::string(argv[3])!="short_global_duplex_1_2")throw std::runtime_error("unknown S13 short profile");
  const int iterations=short_check?1:gh::integer(argv[2],16,16);
  unsigned seed=gh::integer(argv[short_check?4:3],0,4294967295ull);
  const GdCase* selected=nullptr;
  for(const auto& c:gd_cases)if(std::string(argv[short_check?2:1])==c.id)selected=&c;
  if(!selected)throw std::runtime_error("unknown S13 case");const auto& c=*selected;
  cudaFuncAttributes attr{};GH_CUDA(cudaFuncGetAttributes(&attr,c.function));int occupancy=0;
  GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,c.function,gd_threads,0));
  if(occupancy<1||attr.sharedSizeBytes!=1024||attr.localSizeBytes!=0)throw std::runtime_error("S13 unexpected drain allocation/spill/occupancy");
  const unsigned blocks=std::min(4,occupancy)*device.prop.multiProcessorCount;
  const gh::u64 total_threads=gh::u64(blocks)*gd_threads;
  const gh::u64 l2=device.prop.l2CacheSize;
  const gh::u64 requested=c.large?4*l2:l2/4;
  const gh::u64 bytes=global_duplex_reference::aligned_array_bytes(requested,total_threads);
  const gh::u64 aggregate=bytes*((c.reads>0)+(c.writes>0));
  if(!c.large&&aggregate>=l2)throw std::runtime_error("S13 small aggregate is not below L2");
  size_t free_bytes=0,total_bytes=0;GH_CUDA(cudaMemGetInfo(&free_bytes,&total_bytes));
  const gh::u64 bookkeeping=total_threads*sizeof(unsigned)+gh::u64(blocks)*sizeof(gh::Stamp);
  if(aggregate>free_bytes||bookkeeping>free_bytes-aggregate)throw std::runtime_error("S13 insufficient free GPU memory");
  uint4 *input=nullptr,*output=nullptr;unsigned* sums=nullptr;gh::Stamp* stamps=nullptr;
  if(c.reads) {
    std::vector<unsigned> host_input(bytes/4);
    for(gh::u64 word=0;word<bytes/4;++word)host_input[word]=global_duplex_reference::word_value(word,17,seed);
    GH_CUDA(cudaMalloc(&input,bytes));GH_CUDA(cudaMemcpy(input,host_input.data(),bytes,cudaMemcpyHostToDevice));
  }
  if(c.writes)GH_CUDA(cudaMalloc(&output,bytes));
  GH_CUDA(cudaMalloc(&sums,total_threads*sizeof(unsigned)));GH_CUDA(cudaMalloc(&stamps,blocks*sizeof(gh::Stamp)));
  cudaEvent_t begin,end;GH_CUDA(cudaEventCreate(&begin));GH_CUDA(cudaEventCreate(&end));
  std::vector<GdCheck> checks;unsigned details=0;
  auto execute=[&](int length,bool capture) {
    gh::Observation formal;gh::Observation& result=capture?checks.back().observation:formal;
    if(c.writes) {
      std::vector<unsigned> poison(bytes/4);
      for(gh::u64 word=0;word<bytes/4;++word)poison[word]=global_duplex_reference::word_value(word,c.copy?17:29,seed)^~unsigned(0);
      GH_CUDA(cudaMemcpy(output,poison.data(),bytes,cudaMemcpyHostToDevice));
    }
    std::vector<unsigned> sum_poison(total_threads);
    for(gh::u64 thread=0;thread<total_threads;++thread)
      sum_poison[thread]=global_duplex_reference::read_checksum(bytes,total_threads,thread,c.reads,length,seed)^~unsigned(0);
    GH_CUDA(cudaMemcpy(sums,sum_poison.data(),total_threads*sizeof(unsigned),cudaMemcpyHostToDevice));
    GH_CUDA(cudaMemset(stamps,0xff,blocks*sizeof(gh::Stamp)));
    gh::u64 vectors=bytes/16;void* arguments[]={&input,&output,&vectors,&length,&seed,&stamps,&sums};
    GH_CUDA(cudaEventRecord(begin));GH_CUDA(cudaLaunchKernel(c.function,dim3(blocks),dim3(gd_threads),arguments,0));
    GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));
    float elapsed=0;GH_CUDA(cudaEventElapsedTime(&elapsed,begin,end));result.event_ms=elapsed;
    result.stamps.resize(blocks);GH_CUDA(cudaMemcpy(result.stamps.data(),stamps,blocks*sizeof(gh::Stamp),cudaMemcpyDeviceToHost));
    bool completed=true;
    for(unsigned block=0;block<blocks;++block) {
      const auto& stamp=result.stamps[block];const auto poison=std::numeric_limits<gh::u64>::max();
      if(stamp.begin_ns==poison||stamp.end_ns==poison||stamp.begin_cycle==poison||stamp.end_cycle==poison||stamp.smid==~unsigned(0)) {
        ++result.errors;completed=false;if(details++<8)std::cerr<<"length="<<length<<" CTA="<<block<<" incomplete stamp\n";
      } else if(capture)checks.back().verified_ctas.push_back(block);
    }
    std::vector<unsigned> got(total_threads);GH_CUDA(cudaMemcpy(got.data(),sums,total_threads*sizeof(unsigned),cudaMemcpyDeviceToHost));
    for(gh::u64 thread=0;thread<total_threads;++thread) {
      const unsigned expected=global_duplex_reference::read_checksum(bytes,total_threads,thread,c.reads,length,seed);
      if(got[thread]!=expected) {
        ++result.errors;if(!c.reads)completed=false;
        if(details++<8)std::cerr<<"length="<<length<<" thread="<<thread<<" checksum actual="<<got[thread]<<" expected="<<expected<<'\n';
      }
      // Empty pure-write checksums are completion only, not data coverage.
      if(c.reads)++result.checked_elements;
    }
    if(c.writes) {
      std::vector<unsigned> got_output(bytes/4);GH_CUDA(cudaMemcpy(got_output.data(),output,bytes,cudaMemcpyDeviceToHost));
      for(gh::u64 word=0;word<bytes/4;++word) {
        const unsigned expected=global_duplex_reference::word_value(word,c.copy?17:29,seed);
        if(got_output[word]!=expected) {
          ++result.errors;if(details++<8)std::cerr<<"length="<<length<<" word="<<word<<" actual="<<got_output[word]<<" expected="<<expected<<'\n';
        }
        ++result.checked_elements;
      }
    }
    result.method="all_nonempty_thread_read_checksums_and_all_destination_words";
    result.input_conditions="read=uint32(17*word_index+seed); independent_write=uint32(29*word_index+seed); copy=source; poison each launch";
    if(capture)checks.back().completed=completed;
    if(result.errors)throw std::runtime_error("S13 correctness/completion errors="+std::to_string(result.errors));
    return result;
  };
  if(short_check) {
    try {for(int length:{1,2}){checks.push_back(GdCheck{length});execute(length,true);}}
    catch(const std::exception& error){gd_validation(c,seed,blocks,attr,occupancy,requested,bytes,aggregate,checks,true);std::cerr<<error.what()<<'\n';return 2;}
    gd_validation(c,seed,blocks,attr,occupancy,requested,bytes,aggregate,checks,false);
  } else {
    auto measured=[&](){return execute(iterations,false);};const auto warmup=gh::warmup(measured);const auto result=measured();
    const gh::u64 read=gh::u64(iterations)*bytes*c.reads,write=gh::u64(iterations)*bytes*c.writes;
    std::ostringstream extra;
    extra<<"\"requested_array_bytes\":"<<requested<<",\"array_bytes\":"<<bytes<<",\"aggregate_array_bytes\":"<<aggregate
      <<",\"read_requests_per_group\":"<<c.reads<<",\"write_requests_per_group\":"<<c.writes
      <<",\"registers_per_thread\":"<<attr.numRegs<<",\"static_smem_bytes\":"<<attr.sharedSizeBytes
      <<",\"dynamic_smem_bytes\":0,\"local_size_bytes\":"<<attr.localSizeBytes<<",\"occupancy_limit_ctas_per_sm\":"<<occupancy
      <<",\"kernel_symbol\":"<<gh::quote(c.symbol);
    gh::emit_trial(c.id,iterations,seed,gd_threads,"all_gpu","byte",read+write,read,write,result,warmup,extra.str());
  }
  GH_CUDA(cudaEventDestroy(begin));GH_CUDA(cudaEventDestroy(end));
  if(input)GH_CUDA(cudaFree(input));if(output)GH_CUDA(cudaFree(output));GH_CUDA(cudaFree(sums));GH_CUDA(cudaFree(stamps));return 0;
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
