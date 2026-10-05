#include "../common/probe_runtime.cuh"
#include "../common/async_copy_reference.hpp"

constexpr unsigned ac_threads = 128;
constexpr gh::u64 ac_array_bytes = 8388608;

__device__ __forceinline__ unsigned ac_shared(const void* pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

template<int Bytes, bool CacheGlobal>
__device__ __forceinline__ void ac_issue(const unsigned* input, unsigned* tile,
                                        gh::u64 step, unsigned slot) {
  const gh::u64 source = ((gh::u64(blockIdx.x) * ac_threads + threadIdx.x +
                          step * gridDim.x * ac_threads) * Bytes) % ac_array_bytes;
  const unsigned destination = ac_shared(tile) + (slot * ac_threads + threadIdx.x) * Bytes;
  const char* address = reinterpret_cast<const char*>(input) + source;
  if constexpr (CacheGlobal) {
    static_assert(Bytes == 16, "cg requires sixteen bytes");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(destination), "l"(address) : "memory");
  } else if constexpr (Bytes == 4) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(destination), "l"(address) : "memory");
  } else if constexpr (Bytes == 8) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(destination), "l"(address) : "memory");
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(destination), "l"(address) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int Stages>
__device__ __forceinline__ void ac_wait(gh::u64 remaining) {
  const unsigned keep = remaining - 1 < Stages - 1 ? unsigned(remaining - 1) : Stages - 1;
  if (keep == 3) asm volatile("cp.async.wait_group 3;" ::: "memory");
  else if (keep == 2) asm volatile("cp.async.wait_group 2;" ::: "memory");
  else if (keep == 1) asm volatile("cp.async.wait_group 1;" ::: "memory");
  else asm volatile("cp.async.wait_group 0;" ::: "memory");
}

template<int Bytes, int Stages, bool CacheGlobal>
__global__ void ac_pipeline(const unsigned* input, int iterations, gh::Stamp* stamps,
                             unsigned* sums, unsigned* final_slots, unsigned* trace) {
  extern __shared__ __align__(16) unsigned tile[];
  constexpr unsigned words = Bytes / 4;
  const gh::u64 steps = gh::u64(iterations) * 8;
  for (unsigned i = threadIdx.x; i < Stages * ac_threads * words; i += ac_threads)
    tile[i] = 0xdeadbeefu;
  __syncthreads();
  gh::Stamp stamp{};
  if (threadIdx.x == 0) {
    asm volatile("mov.u32 %0, %%smid;" : "=r"(stamp.smid));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamp.begin_ns));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(stamp.begin_cycle));
  }
  __syncthreads();
  #pragma unroll
  for (unsigned stage = 0; stage < Stages; ++stage)
    ac_issue<Bytes, CacheGlobal>(input, tile, stage, stage);
  unsigned sum = 0;
  const unsigned neighbor = (threadIdx.x + 1) % ac_threads;
  #pragma unroll 1
  for (gh::u64 step = 0; step < steps; ++step) {
    ac_wait<Stages>(steps - step);
    __syncthreads();
    #pragma unroll
    for (unsigned word = 0; word < words; ++word) {
      const unsigned value = reinterpret_cast<volatile unsigned*>(tile)[
          (unsigned(step % Stages) * ac_threads + neighbor) * words + word];
      sum += value;
      if (trace) {
        const gh::u64 index = ((gh::u64(blockIdx.x) * ac_threads + threadIdx.x) * steps + step) * words + word;
        trace[index] = value;
      }
    }
    __syncthreads();
    if (step + Stages < steps)
      ac_issue<Bytes, CacheGlobal>(input, tile, step + Stages, unsigned(step % Stages));
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  asm volatile("" : "+r"(sum) :: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(stamp.end_cycle));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamp.end_ns));
    stamps[blockIdx.x] = stamp;
  }
  sums[gh::u64(blockIdx.x) * ac_threads + threadIdx.x] = sum;
  #pragma unroll
  for (unsigned stage = 0; stage < Stages; ++stage) {
    #pragma unroll
    for (unsigned word = 0; word < words; ++word) {
      const gh::u64 destination = ((gh::u64(blockIdx.x) * Stages + stage) * ac_threads + threadIdx.x) * words + word;
      final_slots[destination] = tile[(stage * ac_threads + threadIdx.x) * words + word];
    }
  }
}

struct AcCase { const char* id; int bytes, stages; bool global_cache, all_gpu; const void* function; };
#define AC_CASE(CACHE, BYTES, STAGES, CG) \
 {#CACHE "_w" #BYTES "_stages" #STAGES "_one_cta", BYTES, STAGES, CG, false, reinterpret_cast<const void*>(ac_pipeline<BYTES,STAGES,CG>)}, \
 {#CACHE "_w" #BYTES "_stages" #STAGES "_all_gpu", BYTES, STAGES, CG, true, reinterpret_cast<const void*>(ac_pipeline<BYTES,STAGES,CG>)}
static const AcCase ac_cases[] = {
 AC_CASE(ca,4,1,false), AC_CASE(ca,4,2,false), AC_CASE(ca,4,4,false),
 AC_CASE(ca,8,1,false), AC_CASE(ca,8,2,false), AC_CASE(ca,8,4,false),
 AC_CASE(ca,16,1,false), AC_CASE(ca,16,2,false), AC_CASE(ca,16,4,false),
 AC_CASE(cg,16,1,true), AC_CASE(cg,16,2,true), AC_CASE(cg,16,4,true)
};

struct AcValidationCheck {
  int iterations;
  gh::Observation observation;
  bool completed = false;
  std::vector<unsigned> verified_ctas;
};

static std::string ac_symbol(const AcCase& c) {
  return "_Z11ac_pipelineILi"+std::to_string(c.bytes)+"ELi"+std::to_string(c.stages)+
    "ELb"+(c.global_cache?"1":"0")+"EEvPKjiPN2gh5StampEPjS5_S5_";
}

static void ac_emit_validation(const AcCase& c, unsigned seed, unsigned blocks,
    const cudaFuncAttributes& attributes, size_t shared, int occupancy,
    const std::vector<AcValidationCheck>& checks, bool failed) {
  gh::u64 errors=failed?1:0;
  for(const auto& check:checks)errors+=check.observation.errors;
  std::cout<<"{\"schema_version\":2,\"validation_schema_version\":1,\"type\":\"validation\",\"case_id\":"<<gh::quote(c.id)
    <<",\"profile_id\":\"short_copy_1_2\",\"seed\":"<<seed<<",\"scope\":"<<gh::quote(c.all_gpu?"all_gpu":"one_cta")
    <<",\"blocks\":"<<blocks<<",\"threads\":128,\"errors\":"<<errors
    <<",\"performance_eligible\":false,\"warmup_executed\":false,\"pilot_executed\":false,\"target_launches\":[";
  for(size_t i=0;i<checks.size();++i)
    std::cout<<(i?",":"")<<"{\"launch_index\":"<<i<<",\"iterations\":"<<checks[i].iterations
      <<",\"input_profile\":\"address_seed_nonuniform\",\"threads\":128,\"blocks\":"<<blocks<<"}";
  std::cout<<"],\"checks\":[";
  for(size_t i=0;i<checks.size();++i) {
    const auto& check=checks[i];
    const gh::u64 expected=gh::u64(blocks)*ac_threads*(gh::u64(check.iterations)*8*(c.bytes/4)+1+c.stages*(c.bytes/4));
    std::cout<<(i?",":"")<<"{\"launch_index\":"<<i
      <<",\"reference_model\":\"async_copy_modular_address_v1\",\"reference_sha256\":\"110adb7b6f54088ef79e275e5c20f1193020b7504c9d3969cbde5bd2f3a3b1f7\""
      <<",\"comparison\":\"exact\",\"tolerance_id\":null,\"checked_elements\":"<<check.observation.checked_elements
      <<",\"expected_elements\":"<<expected<<",\"errors\":"<<check.observation.errors
      <<",\"completed\":"<<(check.completed?"true":"false")<<",\"verified_CTA_ids\":[";
    for(size_t j=0;j<check.verified_ctas.size();++j)std::cout<<(j?",":"")<<check.verified_ctas[j];
    std::cout<<"],\"output_artifacts\":[]}";
  }
  std::cout<<"],\"resource_identity\":{\"kernel_symbol\":"<<gh::quote(ac_symbol(c))
    <<",\"registers_per_thread\":"<<attributes.numRegs<<",\"static_smem_bytes\":"<<attributes.sharedSizeBytes
    <<",\"dynamic_smem_bytes\":"<<shared<<",\"local_size_bytes\":"<<attributes.localSizeBytes
    <<",\"occupancy_limit_ctas_per_sm\":"<<occupancy<<",\"extensions\":{}}}\n";
}

int main(int argc, char** argv) try {
  const auto device = gh::device();
  gh::emit_device(device);
  if (argc == 2 && std::string(argv[1]) == "device") return 0;
  const bool short_check = argc > 1 && std::string(argv[1]) == "validate-only";
  if (argc != (short_check ? 5 : 4)) throw std::runtime_error("usage: probe CASE 8192 SEED | validate-only CASE short_copy_1_2 SEED");
  if (short_check && std::string(argv[3]) != "short_copy_1_2") throw std::runtime_error("unknown short profile");
  const int iterations = short_check ? 1 : gh::integer(argv[2],8192,8192);
  const unsigned seed = gh::integer(argv[short_check ? 4 : 3],0,4294967295ull);
  const AcCase* choice = nullptr;
  for (const auto& candidate : ac_cases) if (candidate.id == std::string(argv[short_check ? 2 : 1])) choice = &candidate;
  if (!choice) throw std::runtime_error("unknown S12 case");
  const auto& c = *choice;
  const unsigned words = c.bytes / 4;
  const size_t shared = size_t(c.stages) * ac_threads * c.bytes;
  cudaFuncAttributes attributes{}; GH_CUDA(cudaFuncGetAttributes(&attributes,c.function));
  int occupancy = 0;
  GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,c.function,ac_threads,shared));
  if (occupancy < 1 || shared + attributes.sharedSizeBytes > device.prop.sharedMemPerBlockOptin)
    throw std::runtime_error("S12 resource-infeasible configuration");
  const unsigned blocks = c.all_gpu ? std::min(4,occupancy)*device.prop.multiProcessorCount : 1;
  std::vector<unsigned> host_input(ac_array_bytes/4);
  for (size_t i=0;i<host_input.size();++i) host_input[i]=unsigned(17*gh::u64(i)+seed);
  unsigned *input=nullptr,*sums=nullptr,*final_slots=nullptr,*trace=nullptr;
  gh::Stamp* stamps=nullptr;
  const size_t sum_count=size_t(blocks)*ac_threads, final_count=sum_count*c.stages*words;
  const size_t trace_capacity=short_check?sum_count*16*words:0;
  GH_CUDA(cudaMalloc(&input,ac_array_bytes));
  GH_CUDA(cudaMemcpy(input,host_input.data(),ac_array_bytes,cudaMemcpyHostToDevice));
  GH_CUDA(cudaMalloc(&sums,sum_count*sizeof(unsigned)));
  GH_CUDA(cudaMalloc(&final_slots,final_count*sizeof(unsigned)));
  GH_CUDA(cudaMalloc(&stamps,blocks*sizeof(gh::Stamp)));
  if(short_check)GH_CUDA(cudaMalloc(&trace,trace_capacity*sizeof(unsigned)));
  cudaEvent_t begin,end;GH_CUDA(cudaEventCreate(&begin));GH_CUDA(cudaEventCreate(&end));
  std::vector<AcValidationCheck> checks;
  unsigned failure_details=0;
  auto execute=[&](int length,bool capture) {
    gh::Observation formal_result;
    gh::Observation& result=capture?checks.back().observation:formal_result;
    GH_CUDA(cudaMemset(sums,0xa5,sum_count*sizeof(unsigned)));
    GH_CUDA(cudaMemset(final_slots,0xa5,final_count*sizeof(unsigned)));
    GH_CUDA(cudaMemset(stamps,capture?0xff:0,blocks*sizeof(gh::Stamp)));
    if(capture)GH_CUDA(cudaMemset(trace,0xa5,trace_capacity*sizeof(unsigned)));
    unsigned* active_trace=capture?trace:nullptr;
    void* arguments[]={&input,&length,&stamps,&sums,&final_slots,&active_trace};
    GH_CUDA(cudaEventRecord(begin));
    GH_CUDA(cudaLaunchKernel(c.function,dim3(blocks),dim3(ac_threads),arguments,shared));
    GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));
    float ms=0;GH_CUDA(cudaEventElapsedTime(&ms,begin,end));result.event_ms=ms;
    result.stamps.resize(blocks);GH_CUDA(cudaMemcpy(result.stamps.data(),stamps,blocks*sizeof(gh::Stamp),cudaMemcpyDeviceToHost));
    if(capture) {
      for(unsigned block=0;block<blocks;++block) {
        const auto& stamp=result.stamps[block];
        const auto poison=std::numeric_limits<gh::u64>::max();
        if(stamp.begin_ns==poison || stamp.end_ns==poison || stamp.begin_cycle==poison ||
           stamp.end_cycle==poison || stamp.smid>=unsigned(device.prop.multiProcessorCount)) {
          if(failure_details++<8)std::cerr<<"length="<<length<<" CTA="<<block<<" incomplete stamp\n";
          ++result.errors;
        } else checks.back().verified_ctas.push_back(block);
      }
      if(result.errors)throw std::runtime_error("S12 incomplete CTA stamp");
    }
    std::vector<unsigned> actual_sums(sum_count),actual_final(final_count),actual_trace(capture?sum_count*length*8*words:0);
    GH_CUDA(cudaMemcpy(actual_sums.data(),sums,sum_count*sizeof(unsigned),cudaMemcpyDeviceToHost));
    GH_CUDA(cudaMemcpy(actual_final.data(),final_slots,final_count*sizeof(unsigned),cudaMemcpyDeviceToHost));
    if(capture)GH_CUDA(cudaMemcpy(actual_trace.data(),trace,actual_trace.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
    auto compare=[&](unsigned actual,unsigned expected,unsigned block,unsigned thread,
                     const char* kind,gh::u64 step,unsigned word) {
      if(actual!=expected) {
        ++result.errors;
        if(capture && failure_details++<8)std::cerr<<"length="<<length<<" CTA="<<block<<" thread="<<thread
          <<" kind="<<kind<<" stage_or_step="<<step<<" word="<<word<<" actual="<<actual<<" expected="<<expected<<'\n';
      }
      ++result.checked_elements;
    };
    const gh::u64 steps=gh::u64(length)*8;
    for(unsigned block=0;block<blocks;++block)for(unsigned thread=0;thread<ac_threads;++thread) {
      compare(actual_sums[size_t(block)*ac_threads+thread],async_copy_reference::checksum(c.bytes,blocks,block,thread,steps,seed),block,thread,"sum",0,0);
      for(unsigned stage=0;stage<unsigned(c.stages);++stage)for(unsigned word=0;word<words;++word) {
        const auto step=async_copy_reference::final_step(stage,c.stages,steps);
        const size_t index=((size_t(block)*c.stages+stage)*ac_threads+thread)*words+word;
        compare(actual_final[index],async_copy_reference::payload_word(c.bytes,blocks,block,thread,step,word,seed),block,thread,"final_slot",stage,word);
      }
      if(capture)for(gh::u64 step=0;step<steps;++step)for(unsigned word=0;word<words;++word) {
        const size_t index=((size_t(block)*ac_threads+thread)*steps+step)*words+word;
        compare(actual_trace[index],async_copy_reference::payload_word(c.bytes,blocks,block,(thread+1)%ac_threads,step,word,seed),block,thread,"trace",step,word);
      }
    }
    result.method=capture?"all_consumed_words_final_slots_and_modular_checksum":"modular_checksum_and_all_final_slots";
    result.input_conditions="uint32(17*word_index+seed); neighbor=(thread+1)%128; poison reset each launch";
    if(capture)checks.back().completed=true;
    if(result.errors)throw std::runtime_error("S12 numerical mismatch: "+std::to_string(result.errors));
    return result;
  };
  if(short_check) {
    try {
      for(int length:{1,2}) {
        checks.push_back(AcValidationCheck{length});
        execute(length,true);
      }
    } catch(const std::exception& error) {
      ac_emit_validation(c,seed,blocks,attributes,shared,occupancy,checks,true);
      std::cerr<<error.what()<<'\n';
      return 2;
    }
    ac_emit_validation(c,seed,blocks,attributes,shared,occupancy,checks,false);
  } else {
    auto measured=[&](){return execute(iterations,false);};
    const auto warmup=gh::warmup(measured);const auto result=measured();
    const gh::u64 payload=gh::u64(blocks)*ac_threads*iterations*8*c.bytes;
    std::ostringstream fields;
    fields<<"\"consumer_read_bytes\":"<<payload<<",\"request_bytes_per_thread\":"<<c.bytes<<",\"stages\":"<<c.stages
      <<",\"global_array_bytes\":"<<ac_array_bytes<<",\"registers_per_thread\":"<<attributes.numRegs
      <<",\"static_smem_bytes\":"<<attributes.sharedSizeBytes<<",\"dynamic_smem_bytes\":"<<shared
      <<",\"local_size_bytes\":"<<attributes.localSizeBytes<<",\"occupancy_limit_ctas_per_sm\":"<<occupancy
      ;
    gh::emit_trial(c.id,iterations,seed,ac_threads,c.all_gpu?"all_gpu":"one_cta","byte",payload,payload,payload,result,warmup,fields.str());
  }
  GH_CUDA(cudaEventDestroy(begin));GH_CUDA(cudaEventDestroy(end));
  if(trace)GH_CUDA(cudaFree(trace));GH_CUDA(cudaFree(stamps));GH_CUDA(cudaFree(final_slots));GH_CUDA(cudaFree(sums));GH_CUDA(cudaFree(input));
  return 0;
} catch(const std::exception& error) {
  std::cerr<<error.what()<<'\n';return 2;
}
