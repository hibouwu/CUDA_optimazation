// Preserve all39 original device bodies and the original formal entry.
#define main synchronization_original_main
#include "synchronization.cu"
#undef main
#include "feasibility/warp_sync_divergent_pair_v2.cu"
#include "../common/word_artifacts.hpp"
#include "../common/synchronization_role_bindings_r4.hpp"
#include "../common/synchronization_pair_reference_v2.hpp"

struct SyncShortArtifact {std::string path,sha;std::vector<gh::u64> shape;};
struct SyncShortCheck {int iterations;unsigned words;std::vector<SyncShortArtifact> artifacts;};
struct SyncShortResource {const SyncRoleBinding* binding;cudaFuncAttributes attrs;int occupancy;};
static void sync_short_write(SyncShortCheck& check,const std::string& name,const std::vector<unsigned>& words,std::vector<gh::u64> shape){check.artifacts.push_back({name,word_artifacts::write_words(name,words),std::move(shape)});}
static SyncShortResource sync_short_query(const void* target,const SyncRoleBinding& b,unsigned threads){
 SyncShortResource r{&b,{},{}};GH_CUDA(cudaFuncGetAttributes(&r.attrs,target));GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.occupancy,target,threads,0));
 if(r.attrs.numRegs!=int(b.regs)||r.attrs.sharedSizeBytes!=b.shared||r.attrs.localSizeBytes||r.occupancy<1)throw std::runtime_error("S11 short actual role resources differ from frozen candidate");return r;
}
static void sync_short_emit(const char* id,const char* profile,unsigned threads,const char* input,const char* model,const char* reference,
 const std::vector<SyncShortCheck>& checks,const SyncShortResource& primary,const std::string& extensions){
 std::cout<<"{\"schema_version\":2,\"validation_schema_version\":1,\"type\":\"validation\",\"case_id\":"<<gh::quote(id)<<",\"profile_id\":"<<gh::quote(profile)<<",\"seed\":3,\"scope\":\"one_cta\",\"threads\":"<<threads<<",\"blocks\":1,\"errors\":0,\"performance_eligible\":false,\"warmup_executed\":false,\"pilot_executed\":false,\"target_launches\":[";
 for(size_t i=0;i<checks.size();++i)std::cout<<(i?",":"")<<"{\"launch_index\":"<<i<<",\"iterations\":"<<checks[i].iterations<<",\"input_profile\":"<<gh::quote(input)<<",\"threads\":"<<threads<<",\"blocks\":1}";
 std::cout<<"],\"checks\":[";
 for(size_t i=0;i<checks.size();++i){const auto& c=checks[i];std::cout<<(i?",":"")<<"{\"launch_index\":"<<i<<",\"reference_model\":"<<gh::quote(model)<<",\"reference_sha256\":"<<gh::quote(reference)<<",\"comparison\":\"exact\",\"tolerance_id\":null,\"checked_elements\":"<<c.words<<",\"expected_elements\":"<<c.words<<",\"errors\":0,\"completed\":true,\"verified_CTA_ids\":[0],\"output_artifacts\":[";
  for(size_t j=0;j<c.artifacts.size();++j){const auto& a=c.artifacts[j];std::cout<<(j?",":"")<<"{\"path\":"<<gh::quote(a.path)<<",\"sha256\":"<<gh::quote(a.sha)<<",\"dtype\":\"uint32\",\"evidence_kind\":\"full_values\",\"shape\":[";for(size_t k=0;k<a.shape.size();++k)std::cout<<(k?",":"")<<a.shape[k];std::cout<<"]}";}std::cout<<"]}";
 }
 std::cout<<"],\"resource_identity\":{\"kernel_symbol\":"<<gh::quote(primary.binding->symbol)<<",\"registers_per_thread\":"<<primary.attrs.numRegs<<",\"static_smem_bytes\":"<<primary.attrs.sharedSizeBytes<<",\"dynamic_smem_bytes\":0,\"local_size_bytes\":"<<primary.attrs.localSizeBytes<<",\"occupancy_limit_ctas_per_sm\":"<<primary.occupancy<<",\"extensions\":"<<extensions<<"}}\n";
}
static std::vector<unsigned> sync_short_stamp(const gh::Stamp& t){std::vector<std::uint64_t> fields{t.begin_ns,t.end_ns,t.begin_cycle,t.end_cycle};auto words=word_artifacts::split_u64(fields);words.push_back(t.smid);return words;}
static void sync_short_stamp_check(const gh::Stamp& t){const auto max=std::numeric_limits<gh::u64>::max();if(t.begin_ns==max||t.end_ns==max||t.begin_cycle==max||t.end_cycle==max||t.end_ns<t.begin_ns||t.end_cycle<t.begin_cycle||t.smid==~unsigned(0))throw std::runtime_error("S11 incomplete short CTA stamp");}
int main(int argc,char** argv) try {
 if(argc<2||std::string(argv[1])!="validate-only")return synchronization_original_main(argc,argv);
 const auto device=gh::device();gh::emit_device(device);
 if(argc!=5||gh::integer(argv[4],0,4294967295ull)!=3)throw std::runtime_error("S11 validate-only CASE PROFILE seed3 required");
 const std::string case_id=argv[2],profile=argv[3];unsigned seed=3;
 const bool pair=case_id=="warp_divergent_shared_pair"||case_id=="cta_cross_half_shared_pair";
 std::vector<SyncShortCheck> checks;
 if(pair){
  if(profile!="sync_divergent_pair_1_2_33_v2")throw std::runtime_error("S11 unknown pair profile");const unsigned which=case_id=="warp_divergent_shared_pair"?0:1;
  const void* target=which?reinterpret_cast<const void*>(cta_divergent_pair_v2):reinterpret_cast<const void*>(warp_divergent_pair_v2);const auto resource=sync_short_query(target,sync_pair_bindings[which],32);
  unsigned* output=nullptr;std::uint64_t* clocks=nullptr;GH_CUDA(cudaMalloc(&output,128*sizeof(unsigned)));GH_CUDA(cudaMalloc(&clocks,2*sizeof(std::uint64_t)));
  for(int length:{1,2,33}){
   SyncShortCheck check{length,132,{}};std::vector<unsigned> poison(128);
   for(unsigned tid=0;tid<32;++tid){poison[tid*4]=~synchronization_pair_reference_v2::initial(tid,seed);poison[tid*4+1]=~synchronization_pair_reference_v2::peer_value(tid,length*8-1,seed);poison[tid*4+2]=~synchronization_pair_reference_v2::checksum(tid,length,seed);poison[tid*4+3]=~unsigned(length*8);}
   GH_CUDA(cudaMemcpy(output,poison.data(),128*sizeof(unsigned),cudaMemcpyHostToDevice));GH_CUDA(cudaMemset(clocks,0xff,2*sizeof(std::uint64_t)));
   unsigned mask=0xffffffffu;void* args[]={&length,&mask,&seed,&output,&clocks};GH_CUDA(cudaLaunchKernel(target,dim3(1),dim3(32),args,0));GH_CUDA(cudaDeviceSynchronize());
   std::vector<unsigned> values(128);std::vector<std::uint64_t> times(2);GH_CUDA(cudaMemcpy(values.data(),output,128*sizeof(unsigned),cudaMemcpyDeviceToHost));GH_CUDA(cudaMemcpy(times.data(),clocks,2*sizeof(std::uint64_t),cudaMemcpyDeviceToHost));
   for(unsigned tid=0;tid<32;++tid)if(values[tid*4]!=synchronization_pair_reference_v2::initial(tid,seed)||values[tid*4+1]!=synchronization_pair_reference_v2::peer_value(tid,length*8-1,seed)||values[tid*4+2]!=synchronization_pair_reference_v2::checksum(tid,length,seed)||values[tid*4+3]!=unsigned(length*8))throw std::runtime_error("S11 pair full-thread output mismatch");
   if(times[0]>times[1]||times[0]==~std::uint64_t(0)||times[1]==~std::uint64_t(0))throw std::runtime_error("S11 pair incomplete clocks");const auto i=checks.size();sync_short_write(check,"sync_pair_threads_"+std::to_string(i)+".u32le",values,{32,4});sync_short_write(check,"sync_pair_clocks_"+std::to_string(i)+".u32le",word_artifacts::split_u64(times),{2,2});checks.push_back(std::move(check));
  }
  std::ostringstream extension;extension<<"{\"pairs_per_iteration\":8,\"barriers_per_pair\":2,\"warp_mask\":4294967295,\"target_identity\":{\"kernel_symbol\":"<<gh::quote(resource.binding->symbol)<<",\"instruction_count\":"<<resource.binding->instructions<<",\"target_identity_sha256\":"<<gh::quote(resource.binding->identity)<<"}}";
  sync_short_emit(case_id.c_str(),profile.c_str(),32,"divergent_cross_half_tokens","sync_divergent_peer_sum_v2","c011a9162da5b40de38918d74c6c7c62efcce3502e6ce539b3dc56994ede3fc0",checks,resource,extension.str());GH_CUDA(cudaFree(clocks));GH_CUDA(cudaFree(output));return 0;
 }
 const std::vector<SyncCase> cases={
  make_case<0,32,false>("warp_t32_aligned"),make_case<0,32,true>("warp_t32_fixed_work_skew"),make_case<1,128,false>("cta_t128_aligned"),make_case<1,128,true>("cta_t128_fixed_work_skew"),make_case<1,256,false>("cta_t256_aligned"),make_case<1,256,true>("cta_t256_fixed_work_skew"),make_case<2,128,false>("mbarrier_t128_aligned"),make_case<2,128,true>("mbarrier_t128_fixed_work_skew"),make_case<2,256,false>("mbarrier_t256_aligned"),make_case<2,256,true>("mbarrier_t256_fixed_work_skew"),make_case<3,32,false>("fence_cta_no_outstanding_work"),make_case<4,32,false>("fence_gpu_no_outstanding_work"),make_case<5,32,false>("proxy_async_no_outstanding_work")};
 const SyncCase* selected=nullptr;const SyncCaseBinding* binding=nullptr;for(const auto& c:cases)if(case_id==c.id)selected=&c;for(const auto& b:sync_case_bindings)if(case_id==b.id)binding=&b;if(!selected||!binding)throw std::runtime_error("S11 unknown original case");const auto& c=*selected;
 const bool measured=profile=="sync_measured_1_2_33_v1";if(!measured&&profile!="sync_aux_correctness2_arrival1_v1")throw std::runtime_error("S11 unknown original short profile");
 const SyncKernel kernels[]={c.measured,c.correctness,c.arrival};std::vector<SyncShortResource> resources;
 for(unsigned role=0;role<3;++role)resources.push_back(sync_short_query(reinterpret_cast<const void*>(kernels[role]),binding->roles[role],c.threads));
 gh::Stamp* stamps=nullptr;SyncThread* output=nullptr;GH_CUDA(cudaMalloc(&stamps,sizeof(gh::Stamp)));GH_CUDA(cudaMalloc(&output,c.threads*sizeof(SyncThread)));
 const std::vector<int> lengths=measured?std::vector<int>{1,2,33}:std::vector<int>{2,1};
 for(unsigned i=0;i<lengths.size();++i){int length=lengths[i];const unsigned purpose=measured?0:i+1;const gh::u64 phases=measured?gh::u64(length)*8:length;SyncShortCheck check{length,unsigned(c.threads*14+9),{}};
  std::vector<SyncThread> poison(c.threads);for(unsigned tid=0;tid<unsigned(c.threads);++tid){auto& p=poison[tid];p.value=~synchronization_reference::expected(c.mode,c.skew,tid,c.threads,seed,phases);p.errors=~0u;p.timeout=~0u;p.smid=~0u;p.completed=~phases;p.attempts=~gh::u64(0);p.checked=~gh::u64(purpose==1?2:0);p.arrival_cycle=~gh::u64(0);p.departure_cycle=~gh::u64(0);}
  GH_CUDA(cudaMemcpy(output,poison.data(),c.threads*sizeof(SyncThread),cudaMemcpyHostToDevice));GH_CUDA(cudaMemset(stamps,0xff,sizeof(gh::Stamp)));void* args[]={&length,&seed,&stamps,&output};GH_CUDA(cudaLaunchKernel(reinterpret_cast<const void*>(kernels[purpose]),dim3(1),dim3(c.threads),args,0));GH_CUDA(cudaDeviceSynchronize());
  std::vector<SyncThread> got(c.threads);gh::Stamp stamp{};GH_CUDA(cudaMemcpy(got.data(),output,c.threads*sizeof(SyncThread),cudaMemcpyDeviceToHost));GH_CUDA(cudaMemcpy(&stamp,stamps,sizeof(stamp),cudaMemcpyDeviceToHost));sync_short_stamp_check(stamp);std::vector<unsigned> words;
  for(unsigned tid=0;tid<unsigned(c.threads);++tid){const auto& v=got[tid];const auto expected=synchronization_reference::expected(c.mode,c.skew,tid,c.threads,seed,phases);const gh::u64 minimum=phases*(purpose==1?2:1);
   if(v.value!=expected||v.errors||v.timeout||v.smid!=stamp.smid||v.completed!=phases||v.checked!=(purpose==1?2:0)||(c.mode==2?(v.attempts<minimum||v.attempts==~gh::u64(0)):v.attempts!=0)||(purpose==2?(v.arrival_cycle>v.departure_cycle||v.departure_cycle==~gh::u64(0)):(v.arrival_cycle!=0||v.departure_cycle!=0)))throw std::runtime_error("S11 short complete-thread/lifecycle mismatch");
   words.insert(words.end(),{v.value,v.errors,v.timeout,v.smid});const std::vector<std::uint64_t> fields{v.completed,v.attempts,v.checked,v.arrival_cycle,v.departure_cycle};const auto wide=word_artifacts::split_u64(fields);words.insert(words.end(),wide.begin(),wide.end());
  }
  sync_short_write(check,"sync_threads_"+std::to_string(i)+".u32le",words,{gh::u64(c.threads),14});sync_short_write(check,"sync_stamp_"+std::to_string(i)+".u32le",sync_short_stamp(stamp),{9});checks.push_back(std::move(check));
 }
 std::ostringstream extension;extension<<"{\"role_resources\":{";const char* names[]={"measured","correctness","arrival"};
 for(unsigned role=0;role<3;++role){extension<<(role?",":"")<<gh::quote(names[role])<<':';const auto& r=resources[role];extension<<"{\"kernel_symbol\":"<<gh::quote(r.binding->symbol)<<",\"registers_per_thread\":"<<r.attrs.numRegs<<",\"static_smem_bytes\":"<<r.attrs.sharedSizeBytes<<",\"dynamic_smem_bytes\":0,\"local_size_bytes\":"<<r.attrs.localSizeBytes<<",\"occupancy_limit_ctas_per_sm\":"<<r.occupancy<<'}';}
 extension<<"},\"role_target_identities\":{";for(unsigned role=0;role<3;++role){const auto& b=binding->roles[role];extension<<(role?",":"")<<gh::quote(names[role])<<":{\"kernel_symbol\":"<<gh::quote(b.symbol)<<",\"instruction_count\":"<<b.instructions<<",\"target_identity_sha256\":"<<gh::quote(b.identity)<<'}';}extension<<"}}";
 sync_short_emit(c.id,profile.c_str(),c.threads,measured?"measured_nonuniform_lcg":"paired_auxiliary_nonuniform_tokens","sync_affine_thread_phase_and_lifecycle_v1","1d773dde0cc26c31729e2280ccca949c6e30005b34ffaf280ec7094f383ba87d",checks,resources[measured?0:1],extension.str());GH_CUDA(cudaFree(output));GH_CUDA(cudaFree(stamps));return 0;
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
