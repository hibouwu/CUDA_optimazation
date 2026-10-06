#include <fstream>
#define main s19_legacy_main
#include "s19_probe.cu"
#undef main

int s19_formal_main(int argc,char** argv)try{
 if(argc!=5)throw std::runtime_error("pilot-only/formal-only CASE ITERATIONS SEED");
 const bool pilot=std::string(argv[1])=="pilot-only";
 if(!pilot&&std::string(argv[1])!="formal-only")throw std::runtime_error("unknown formal command");
 const auto device=gh::device();gh::emit_device(device);
 const auto c=s19::parse(argv[2]);const int iterations=int(gh::integer(argv[3],128,65536));
 if(pilot&&iterations!=128)throw std::runtime_error("own pilot requires128 iterations");
 const unsigned seed=unsigned(gh::integer(argv[4],0,4294967295ull));
 cudaFuncAttributes attributes{};GH_CUDA(cudaFuncGetAttributes(&attributes,s19_pipeline));
 int occupancy=0;GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,s19_pipeline,128,32768));
 if(occupancy<1||attributes.localSizeBytes!=0)throw std::runtime_error("legal occupancy and no spill required");
 auto input=s19::input(c,seed,false);auto expected=s19::expected(c,iterations,seed,false,false);
 Buffer in(input.size()),out(1024),digest(128),slots(8192),trace_input(0),trace_c(0);
 gh::Stamp* stamp=nullptr;GH_CUDA(cudaMalloc(&stamp,sizeof(gh::Stamp)));
 cudaEvent_t begin,end;GH_CUDA(cudaEventCreate(&begin));GH_CUDA(cudaEventCreate(&end));
 unsigned invocation=0;std::vector<Artifact> final_artifacts;
 auto execute=[&](bool keep){
  ++invocation;
  for(Buffer* buffer:{&in,&out,&digest,&slots,&trace_input,&trace_c}){
   std::fill(buffer->host.begin()+8,buffer->host.end()-8,s19::poison);
   for(unsigned k=0;k<8;++k){buffer->host[k]=0xd1900000u+k;buffer->host[buffer->host.size()-8+k]=0xd1910000u+k;}
  }
  std::copy(input.begin(),input.end(),in.host.begin()+8);
  for(Buffer* buffer:{&in,&out,&digest,&slots,&trace_input,&trace_c})buffer->upload();
  GH_CUDA(cudaMemset(stamp,0xff,sizeof(gh::Stamp)));
  GH_CUDA(cudaEventRecord(begin));
  s19_pipeline<<<1,128,32768>>>(in.data(),out.data(),digest.data(),slots.data(),nullptr,nullptr,stamp,c.mode,c.stages,c.tiles,iterations);
  GH_CUDA(cudaGetLastError());GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));
  gh::Observation observed;float event_ms=0;GH_CUDA(cudaEventElapsedTime(&event_ms,begin,end));observed.event_ms=event_ms;
  observed.stamps.resize(1);GH_CUDA(cudaMemcpy(observed.stamps.data(),stamp,sizeof(gh::Stamp),cudaMemcpyDeviceToHost));
  auto original=in.download(),output=out.download(),checksums=digest.download(),final_slots=slots.download();
  trace_input.download();trace_c.download();
  observed.method="S19_complete_input_output_digest_slots_guards_stamp_v1";
  observed.input_conditions="periodic_nonuniform_exact_float_input;trace_null;oneCTA";
  observed.checked_elements=input.size()+1024+128+8192+96+10;
  std::string failure;
  try{
   for(const Buffer* buffer:{&in,&out,&digest,&slots,&trace_input,&trace_c})buffer->check_guards();
   compare(original,input,"input");compare(output,expected.output,"output");compare(checksums,expected.digest,"digest");compare(final_slots,expected.slots,"slots");
   gh::envelope(observed);
  }catch(const std::exception& error){observed.errors=1;failure=error.what();}
  if(keep||observed.errors){
   std::vector<Artifact> artifacts;const std::string prefix="s19_"+std::to_string(invocation)+"_";
   save(artifacts,prefix+"input.u32le",original);save(artifacts,prefix+"output.u32le",output);
   save(artifacts,prefix+"digest.u32le",checksums);save(artifacts,prefix+"slots.u32le",final_slots);
   std::vector<unsigned> guards;
   for(const Buffer* buffer:{&in,&out,&digest,&slots,&trace_input,&trace_c}){auto g=buffer->guards();guards.insert(guards.end(),g.begin(),g.end());}
   save(artifacts,prefix+"guards.u32le",guards);
   const auto hs=observed.stamps[0];std::vector<unsigned> words;
   for(gh::u64 value:{hs.begin_ns,hs.end_ns,hs.begin_cycle,hs.end_cycle}){words.push_back(unsigned(value));words.push_back(unsigned(value>>32));}
   words.push_back(hs.smid);words.push_back(s19::bits(float(observed.event_ms)));save(artifacts,prefix+"stamp.u32le",words);
   std::ofstream identity(prefix+"identity.json");identity<<"{\"case_id\":"<<gh::quote(c.id)<<",\"iterations\":"<<iterations<<",\"seed\":"<<seed
    <<",\"invocation\":"<<invocation<<",\"errors\":"<<observed.errors<<",\"diagnostic\":"<<gh::quote(failure)<<",\"trace_enabled\":false}\n";
   identity.flush();identity.close();
   if(!identity)throw std::runtime_error("identity output failed");
   final_artifacts=artifacts;
  }
  if(observed.errors)throw std::runtime_error(failure);
  return observed;
 };
 gh::Warmup warm;if(!pilot)warm=gh::warmup([&](){return execute(false);});const auto measured=execute(true);
 const gh::u64 fma=c.mode==1?0:65536ull*c.tiles*iterations;
 const gh::u64 read=c.mode==0?0:8192ull*c.tiles*iterations,write=c.mode==4?4096ull*iterations:0,epilogue=c.mode==4?2048ull*iterations:0;
 std::ostringstream extra;extra<<"\"phase\":"<<gh::quote(pilot?"pilot":"formal")<<",\"performance_eligible\":"<<(pilot?"false":"true")
  <<",\"warmup_executed\":"<<(pilot?"false":"true")<<",\"trace_enabled\":false,\"input_profile\":\"periodic\",\"final_invocation\":"<<invocation
  <<",\"mode\":"<<c.mode<<",\"stages\":"<<c.stages<<",\"tiles\":"<<c.tiles<<",\"fma_flop\":"<<fma<<",\"epilogue_flop\":"<<epilogue
  <<",\"kernel_symbol\":\"s19_pipeline\",\"registers_per_thread\":"<<attributes.numRegs<<",\"static_smem_bytes\":"<<attributes.sharedSizeBytes
  <<",\"dynamic_smem_bytes\":32768,\"local_size_bytes\":"<<attributes.localSizeBytes<<",\"occupancy_limit_ctas_per_sm\":"<<occupancy
  <<",\"post_timing_output_export_bytes\":"<<(c.mode==4?0:4096)<<",\"post_timing_diagnostic_export_bytes\":33280,\"full_output_artifacts\":[";
 for(size_t n=0;n<final_artifacts.size();++n){const auto& a=final_artifacts[n];extra<<(n?",":"")<<"{\"path\":"<<gh::quote(a.name)<<",\"sha256\":"<<gh::quote(a.sha)<<",\"dtype\":\"uint32\",\"shape\":["<<a.words<<"]}";}extra<<"]";
 gh::emit_trial(c.id,iterations,seed,128,"one_cta",c.mode==1?"byte":"FLOP",c.mode==1?read:fma+epilogue,read,write,measured,warm,extra.str());
 GH_CUDA(cudaEventDestroy(begin));GH_CUDA(cudaEventDestroy(end));GH_CUDA(cudaFree(stamp));return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}

int main(int argc,char** argv){
 if(argc>1&&(std::string(argv[1])=="device"||std::string(argv[1])=="capability"||std::string(argv[1])=="validate-only"||std::string(argv[1])=="diagnose"))return s19_legacy_main(argc,argv);
 return s19_formal_main(argc,argv);
}
