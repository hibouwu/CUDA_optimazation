// Original target device functions and short entry are included verbatim.
#define main tma_bulk_original_short_main
#include "tma_bulk.cu"
#undef main

int main(int argc,char** argv) try {
  const bool grouped=argc==5&&std::string(argv[1])=="validate-only"&&std::string(argv[3])=="formal_final_1_2_33_v1";
  if(!grouped&&argc>=2&&(std::string(argv[1])=="device"||std::string(argv[1])=="validate-only"))
    return tma_bulk_original_short_main(argc,argv);
  const auto device=gh::device();gh::emit_device(device);
  const bool short_mode=grouped||(argc==3&&std::string(argv[1])=="validate-formal");
  if(argc!=4&&!short_mode)throw std::runtime_error("S14 requires CASE ITERS SEED or validate-only CASE formal_final_1_2_33_v1 3");
  if(grouped&&gh::integer(argv[4],0,4294967295ull)!=3)throw std::runtime_error("S14 formal-path group requires top seed3");
  const TbCase* selected=nullptr;for(const auto& c:tb_cases)if(c.id==std::string(argv[short_mode?2:1]))selected=&c;
  if(!selected)throw std::runtime_error("unknown S14 formal case");const auto& c=*selected;
  int iterations=short_mode?1:gh::integer(argv[2],1,65536);unsigned seed=short_mode?0:gh::integer(argv[3],0,4294967295ull);
  unsigned short_index=0;std::vector<TbCheck> short_checks;
  const size_t shared=c.bytes+32;const void* target=c.function;
  GH_CUDA(cudaFuncSetAttribute(target,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  cudaFuncAttributes attr{};GH_CUDA(cudaFuncGetAttributes(&attr,target));int occupancy=0;
  GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,target,tb_threads,shared));
  if(occupancy<1||attr.localSizeBytes||attr.sharedSizeBytes+shared>device.prop.sharedMemPerBlockOptin)
    throw std::runtime_error("S14 unsupported formal allocation/spill/occupancy");
  const unsigned blocks=c.all_gpu?std::min(4,occupancy)*device.prop.multiProcessorCount:1;
  const gh::u64 words=c.bytes/4,allocation=tma_bulk_reference::allocation_bytes(blocks,c.bytes),ring_words=(allocation-32)/4;
  const gh::u64 final_words=c.g2s?gh::u64(blocks)*words:1,final_bytes=final_words*4;
  const gh::u64 control_bytes=gh::u64(blocks)*(sizeof(gh::Stamp)+5*sizeof(std::uint64_t));
  size_t free_bytes=0,total_bytes=0;GH_CUDA(cudaMemGetInfo(&free_bytes,&total_bytes));
  if(allocation>free_bytes||final_bytes>free_bytes-allocation||control_bytes>free_bytes-allocation-final_bytes)
    throw std::runtime_error("S14 insufficient formal memory budget");
  unsigned *global=nullptr,*final_tile=nullptr;gh::Stamp* stamps=nullptr;std::uint64_t* completion=nullptr;
  GH_CUDA(cudaMalloc(&global,allocation));GH_CUDA(cudaMalloc(&final_tile,final_bytes));
  GH_CUDA(cudaMalloc(&stamps,blocks*sizeof(gh::Stamp)));GH_CUDA(cudaMalloc(&completion,blocks*5*sizeof(std::uint64_t)));
  cudaEvent_t begin,end;GH_CUDA(cudaEventCreate(&begin));GH_CUDA(cudaEventCreate(&end));
  auto execute=[&]() {
    std::vector<unsigned> initial(ring_words+8);
    for(unsigned i=0;i<4;++i){initial[i]=0xd15ea5e0u+i;initial[ring_words+4+i]=0xd15ea5e4u+i;}
    for(gh::u64 index=0;index<ring_words;++index) {
      const gh::u64 block=index/(tb_slots*words),slot=(index/words)%tb_slots,word=index%words;
      initial[index+4]=c.g2s?tma_bulk_reference::global_input_word(block,slot,word,blocks,c.bytes,seed):
        tma_bulk_reference::output_poison(tma_bulk_reference::shared_input_word(block,word,blocks,c.bytes,seed));
    }
    GH_CUDA(cudaMemcpy(global,initial.data(),allocation,cudaMemcpyHostToDevice));
    std::vector<unsigned> final_poison(final_words,0xffffffffu);
    if(c.g2s)for(gh::u64 index=0;index<final_words;++index)
      final_poison[index]=~tma_bulk_reference::global_input_word(index/words,iterations-1,index%words,blocks,c.bytes,seed);
    GH_CUDA(cudaMemcpy(final_tile,final_poison.data(),final_bytes,cudaMemcpyHostToDevice));
    GH_CUDA(cudaMemset(stamps,0xff,blocks*sizeof(gh::Stamp)));GH_CUDA(cudaMemset(completion,0xff,blocks*5*sizeof(std::uint64_t)));
    unsigned* payload=global+4;bool capture=false;int length=iterations;
    void* args[]={&payload,&length,&seed,&capture,&final_tile,&stamps,&completion};
    GH_CUDA(cudaEventRecord(begin));GH_CUDA(cudaLaunchKernel(target,dim3(blocks),dim3(tb_threads),args,shared));
    GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));
    gh::Observation result;result.stamps.resize(blocks);TbCheck persisted;
    auto save=[&](const char* leaf,const std::vector<unsigned>& values,std::vector<gh::u64> shape){
      if(short_mode){const std::string name="formal_"+std::to_string(short_index)+"_"+leaf;tb_save(persisted,name.c_str(),values,std::move(shape));}
    };std::vector<std::uint64_t> counts(blocks*5);
    GH_CUDA(cudaMemcpy(result.stamps.data(),stamps,blocks*sizeof(gh::Stamp),cudaMemcpyDeviceToHost));
    GH_CUDA(cudaMemcpy(counts.data(),completion,blocks*5*sizeof(std::uint64_t),cudaMemcpyDeviceToHost));
    float elapsed=0;GH_CUDA(cudaEventElapsedTime(&elapsed,begin,end));result.event_ms=elapsed;
    save("completion.u32le",word_artifacts::split_u64(counts),{blocks,5,2});
    const auto poison=std::numeric_limits<gh::u64>::max();
    for(unsigned block=0;block<blocks;++block) {
      const auto& stamp=result.stamps[block];const gh::u64 offset=gh::u64(block)*5;
      if(stamp.begin_ns==poison||stamp.end_ns==poison||stamp.begin_cycle==poison||stamp.end_cycle==poison||stamp.smid==~unsigned(0)||stamp.end_ns<stamp.begin_ns||stamp.end_cycle<stamp.begin_cycle||
        counts[offset]!=unsigned(iterations)||(c.g2s?(counts[offset+1]<unsigned(iterations)||counts[offset+1]==poison):counts[offset+1]!=0)||
        counts[offset+2]!=0||counts[offset+3]!=0||counts[offset+4]!=(c.g2s?0:unsigned(iterations)))
        throw std::runtime_error("S14 formal incomplete CTA stamp/lifecycle");
    }
    auto check=[&](unsigned actual,unsigned expected){++result.checked_elements;if(actual!=expected)++result.errors;};
    if(c.g2s) {
      std::vector<unsigned> values(final_words);GH_CUDA(cudaMemcpy(values.data(),final_tile,final_bytes,cudaMemcpyDeviceToHost));
      for(gh::u64 index=0;index<final_words;++index)
        check(values[index],tma_bulk_reference::global_input_word(index/words,iterations-1,index%words,blocks,c.bytes,seed));
      save("final_tile.u32le",values,{blocks,words});
    } else {
      std::vector<unsigned> ring(ring_words);GH_CUDA(cudaMemcpy(ring.data(),payload,ring_words*4,cudaMemcpyDeviceToHost));
      for(gh::u64 index=0;index<ring_words;++index)
        check(ring[index],tma_bulk_reference::destination_word(index/(tb_slots*words),(index/words)%tb_slots,index%words,blocks,c.bytes,iterations,seed));
      save("ring.u32le",ring,{blocks,32,words});
    }
    unsigned guards[8];GH_CUDA(cudaMemcpy(guards,global,16,cudaMemcpyDeviceToHost));
    GH_CUDA(cudaMemcpy(guards+4,payload+ring_words,16,cudaMemcpyDeviceToHost));
    for(unsigned i=0;i<8;++i)check(guards[i],0xd15ea5e0u+i);
    save("guards.u32le",std::vector<unsigned>(guards,guards+8),{2,4});
    if(result.errors)throw std::runtime_error("S14 formal full-word/guard mismatch");
    result.method="host_final_tile_or_full_ring_and_guards_lifecycle";
    result.input_conditions="32-slot nonuniform modular word pattern; poison reset every launch; capture=false";
    if(!short_mode)gh::envelope(result);
    if(short_mode){
      std::vector<std::uint64_t> encoded;for(const auto& t:result.stamps){encoded.push_back(t.begin_ns);encoded.push_back(t.end_ns);encoded.push_back(t.begin_cycle);encoded.push_back(t.end_cycle);encoded.push_back(t.smid);}
      save("stamps.u32le",word_artifacts::split_u64(encoded),{blocks,5,2});
      persisted.checked=result.checked_elements;persisted.completed=true;short_checks.push_back(std::move(persisted));
    }
    return result;
  };
  if(short_mode) {
    const unsigned lengths[]={1,2,33},seeds[]={0,3,4294967295u};
    for(short_index=0;short_index<3;++short_index){iterations=lengths[short_index];seed=seeds[short_index];execute();}
    std::cout<<"{\"schema_version\":2,\"validation_schema_version\":1,\"type\":\"validation\",\"case_id\":"<<gh::quote(c.id)
      <<",\"profile_id\":\"formal_final_1_2_33_v1\",\"seed\":3,\"scope\":"<<gh::quote(c.all_gpu?"all_gpu":"one_cta")
      <<",\"warmup_executed\":false,\"pilot_executed\":false,\"performance_eligible\":false,\"blocks\":"<<blocks<<",\"threads\":128,\"errors\":0,\"target_launches\":[";
    for(unsigned i=0;i<3;++i)std::cout<<(i?",":"")<<"{\"launch_index\":"<<i<<",\"iterations\":"<<lengths[i]<<",\"input_profile\":\"paired_nonuniform_final_tile\",\"threads\":128,\"blocks\":"<<blocks<<"}";
    std::cout<<"],\"checks\":[";
    for(unsigned i=0;i<3;++i) {
      const auto& check=short_checks[i];std::cout<<(i?",":"")<<"{\"launch_index\":"<<i
        <<",\"reference_model\":\"tma_bulk_ring_word_reference_v1\",\"reference_sha256\":\"98d20cfdfbdf5fc627976764ce87ceba032c94f09341bdac0e41ac95ec7bcf7b\",\"comparison\":\"exact\",\"tolerance_id\":null"
        <<",\"completed\":true,\"errors\":0,\"checked_elements\":"<<check.checked<<",\"expected_elements\":"<<check.checked<<",\"verified_CTA_ids\":[";
      for(unsigned b=0;b<blocks;++b)std::cout<<(b?",":"")<<b;std::cout<<"],\"output_artifacts\":[";
      for(size_t j=0;j<check.artifacts.size();++j){const auto& a=check.artifacts[j];std::cout<<(j?",":"")<<"{\"path\":"<<gh::quote(a.path)<<",\"sha256\":"<<gh::quote(a.sha)<<",\"dtype\":\"uint32\",\"evidence_kind\":\"full_values\",\"shape\":[";for(size_t k=0;k<a.shape.size();++k)std::cout<<(k?",":"")<<a.shape[k];std::cout<<"]}";}
      std::cout<<"]}";
    }
    std::cout<<"],\"resource_identity\":{\"kernel_symbol\":"<<gh::quote(c.symbol)<<",\"registers_per_thread\":"<<attr.numRegs<<",\"static_smem_bytes\":"<<attr.sharedSizeBytes<<",\"dynamic_smem_bytes\":"<<shared<<",\"local_size_bytes\":"<<attr.localSizeBytes<<",\"occupancy_limit_ctas_per_sm\":"<<occupancy
      <<",\"extensions\":{\"payload_bytes\":"<<c.bytes<<",\"global_slots_per_cta\":32,\"global_allocation_bytes\":"<<allocation<<",\"completion_kind\":"<<gh::quote(c.g2s?"mbarrier_acquire_then_CTA_gate":"bulk_wait_group_0_then_CTA_gate")<<",\"validation_role\":\"formal_final_transport\",\"capture_enabled\":false}}}\n";

  } else {
  const auto warmup=gh::warmup(execute);const auto result=execute();
  const gh::u64 q=tma_bulk_reference::completed_payload(blocks,iterations,c.bytes);
  std::ostringstream extra;extra<<"\"payload_bytes\":"<<c.bytes<<",\"global_slots_per_cta\":32,\"global_allocation_bytes\":"<<allocation
    <<",\"capture_enabled\":false,\"completed_requests\":"<<gh::u64(blocks)*iterations<<",\"timeout_count\":0"
    <<",\"registers_per_thread\":"<<attr.numRegs<<",\"static_smem_bytes\":"<<attr.sharedSizeBytes<<",\"dynamic_smem_bytes\":"<<shared
    <<",\"local_size_bytes\":"<<attr.localSizeBytes<<",\"occupancy_limit_ctas_per_sm\":"<<occupancy<<",\"kernel_symbol\":"<<gh::quote(c.symbol);
  gh::emit_trial(c.id,iterations,seed,tb_threads,c.all_gpu?"all_gpu":"one_cta","byte",q,c.g2s?q:0,c.g2s?0:q,result,warmup,extra.str());
  }
  GH_CUDA(cudaEventDestroy(begin));GH_CUDA(cudaEventDestroy(end));GH_CUDA(cudaFree(completion));GH_CUDA(cudaFree(stamps));GH_CUDA(cudaFree(final_tile));GH_CUDA(cudaFree(global));return 0;
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
