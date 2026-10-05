// S18 finite short host. No formal sampling, warmup, pilot or calibration entry.
#include "../common/auxiliary_device_v1.cuh"
#include "../common/auxiliary_cases_v1.hpp"
#include "../common/auxiliary_setmax_image_v1.hpp"
#include "../common/auxiliary_host_reference_v1.hpp"
#include "../common/probe_runtime.cuh"
#include "../common/word_artifacts.hpp"
#include <cuda.h>
#include <cstring>
#include <limits>
#include <vector>

static void driver_check(CUresult code,const char* operation) {
  if(code!=CUDA_SUCCESS)throw std::runtime_error(std::string(operation)+": CUresult="+std::to_string(int(code)));
}
template<class Function> static Function driver_entry(const char* name) {
  void* address=nullptr;cudaDriverEntryPointQueryResult status{};
  GH_CUDA(cudaGetDriverEntryPointByVersion(name,&address,12090,cudaEnableDefault,&status));
  if(status!=cudaDriverEntryPointSuccess||!address)throw std::runtime_error(std::string("CUDA12.9 driver entry unavailable: ")+name);
  return reinterpret_cast<Function>(address);
}
struct SetmaxModule {
  CUmodule module{};CUfunction function{};
  decltype(&cuModuleLoadData) load=driver_entry<decltype(&cuModuleLoadData)>("cuModuleLoadData");
  decltype(&cuModuleGetFunction) get_function=driver_entry<decltype(&cuModuleGetFunction)>("cuModuleGetFunction");
  decltype(&cuModuleUnload) unload=driver_entry<decltype(&cuModuleUnload)>("cuModuleUnload");
  decltype(&cuFuncGetAttribute) attribute=driver_entry<decltype(&cuFuncGetAttribute)>("cuFuncGetAttribute");
  decltype(&cuOccupancyMaxActiveBlocksPerMultiprocessor) occupancy=driver_entry<decltype(&cuOccupancyMaxActiveBlocksPerMultiprocessor)>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
  decltype(&cuLaunchKernel) launch=driver_entry<decltype(&cuLaunchKernel)>("cuLaunchKernel");
  SetmaxModule() {
    word_artifacts::Sha256 hash;hash.update(auxiliary_setmax_image,sizeof(auxiliary_setmax_image));
    if(hash.finish()!=auxiliary_setmax_image_sha256)throw std::runtime_error("embedded offline cubin SHA");
    int ordinal=0;GH_CUDA(cudaGetDevice(&ordinal));GH_CUDA(cudaSetDevice(ordinal));
    driver_check(load(&module,auxiliary_setmax_image),"load exact offline cubin");
    driver_check(get_function(&function,module,"aux_setmax_initial64_v1"),"get setmax entry");
  }
  ~SetmaxModule(){if(module)unload(module);}
};
struct Capability {
  int registers=0,local=0,static_smem=0,max_threads=0,occupancy=0;
  bool supported=false;std::string reason;
};
static Capability capability(const AuxiliaryCase& c,SetmaxModule& setmax) {
  Capability x;cudaDeviceProp device{};int ordinal=0;
  GH_CUDA(cudaGetDevice(&ordinal));GH_CUDA(cudaGetDeviceProperties(&device,ordinal));
  if(c.kind==3) {
    driver_check(setmax.attribute(&x.registers,CU_FUNC_ATTRIBUTE_NUM_REGS,setmax.function),"setmax actual registers");
    driver_check(setmax.attribute(&x.local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,setmax.function),"setmax actual local");
    driver_check(setmax.attribute(&x.static_smem,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,setmax.function),"setmax actual SMEM");
    driver_check(setmax.attribute(&x.max_threads,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,setmax.function),"setmax thread limit");
    driver_check(setmax.occupancy(&x.occupancy,setmax.function,c.threads,0),"setmax occupancy upper bound");
    if(x.registers!=64)throw std::runtime_error("setmax initial actual REG64 condition not realized");
  } else {
    cudaFuncAttributes attributes{};GH_CUDA(cudaFuncGetAttributes(&attributes,c.function));
    x.registers=attributes.numRegs;x.local=int(attributes.localSizeBytes);
    x.static_smem=int(attributes.sharedSizeBytes);x.max_threads=attributes.maxThreadsPerBlock;
    if(c.dynamic+unsigned(x.static_smem)>device.sharedMemPerBlockOptin) {
      x.reason="actual per-CTA SMEM limit";return x;
    }
    GH_CUDA(cudaFuncSetAttribute(c.function,cudaFuncAttributePreferredSharedMemoryCarveout,c.carveout));
    if(c.dynamic)GH_CUDA(cudaFuncSetAttribute(c.function,cudaFuncAttributeMaxDynamicSharedMemorySize,c.dynamic));
    GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&x.occupancy,c.function,c.threads,c.dynamic));
    if(c.kind==0&&c.operation==2&&x.registers>32)
      throw std::runtime_error("pressure requested maxnreg32 condition not realized");
  }
  if(c.threads>unsigned(x.max_threads)||c.threads>unsigned(device.maxThreadsPerBlock))x.reason="actual thread limit";
  else if(gh::u64(x.registers)*c.threads>unsigned(device.regsPerMultiprocessor))x.reason="actual registers exceed SM capacity";
  else if(x.occupancy<1)x.reason="occupancy API reports no legal CTA";
  else x.supported=true;
  return x;
}
static void emit_capability(const AuxiliaryCase& c,const Capability& x) {
  std::cout<<"{\"type\":\"auxiliary_capability\",\"case_id\":"<<gh::quote(c.id)
    <<",\"kernel_symbol\":"<<gh::quote(c.symbol)<<",\"supported\":"<<(x.supported?"true":"false")
    <<",\"reason\":"<<gh::quote(x.reason)<<",\"threads\":"<<c.threads<<",\"registers_per_thread\":"<<x.registers
    <<",\"static_smem_bytes\":"<<x.static_smem<<",\"dynamic_smem_bytes\":"<<c.dynamic<<",\"local_size_bytes\":"<<x.local
    <<",\"occupancy_limit_ctas_per_sm\":"<<x.occupancy<<",\"occupancy_is_upper_bound\":true,\"requested_carveout\":"<<c.carveout
    <<",\"actual_carveout\":null,\"GPU_target_launches\":0}\n";
}
struct Artifact {std::string path,sha;std::vector<gh::u64> shape;};
struct Check {gh::u64 checked=0;double event_ms=0;std::vector<Artifact> artifacts;};
static void save(Check& check,unsigned stage,const char* name,const std::vector<unsigned>& data,std::vector<gh::u64> shape) {
  gh::u64 count=1;for(auto n:shape){if(!n||count>std::numeric_limits<gh::u64>::max()/n)throw std::runtime_error("S18 output shape overflow");count*=n;}
  if(count!=data.size())throw std::runtime_error("S18 output shape");
  auto path="auxiliary_"+std::to_string(stage)+"_"+name+".u32le";
  check.artifacts.push_back({path,word_artifacts::write_words(path,data),shape});check.checked+=data.size();
}
struct Buffer {
  unsigned* pointer=nullptr;std::vector<unsigned> host;
  explicit Buffer(std::size_t words):host(words+16,0) {
    for(unsigned k=0;k<8;++k){host[k]=0xdac00000u+k;host[host.size()-8+k]=0xdac10000u+k;}
    GH_CUDA(cudaMalloc(&pointer,host.size()*4));
  }
  ~Buffer(){if(pointer)cudaFree(pointer);}
  unsigned* data(){return pointer+8;}
  void upload(){GH_CUDA(cudaMemcpy(pointer,host.data(),host.size()*4,cudaMemcpyHostToDevice));}
  void download(){GH_CUDA(cudaMemcpy(host.data(),pointer,host.size()*4,cudaMemcpyDeviceToHost));
    for(unsigned k=0;k<8;++k)if(host[k]!=0xdac00000u+k||host[host.size()-8+k]!=0xdac10000u+k)throw std::runtime_error("S18 complete output guards");}
  std::vector<unsigned> values()const{return {host.begin()+8,host.end()-8};}
  std::vector<unsigned> guards()const{std::vector<unsigned> out(host.begin(),host.begin()+8);out.insert(out.end(),host.end()-8,host.end());return out;}
};
static void equal(unsigned actual,unsigned expected){if(actual!=expected)throw std::runtime_error("S18 full numerical output mismatch");}
static Check execute(const AuxiliaryCase& c,SetmaxModule& setmax,unsigned stage,unsigned iterations,unsigned seed) {
  using namespace auxiliary_host_reference;
  const std::size_t count=c.kind==0?c.threads*c.words:c.kind==1?c.threads*c.words*2:c.kind==2?c.words:iterations*2*128*60;
  Buffer output(count),inputs(c.kind==0?c.threads*c.words:c.kind==3?128*60:c.kind==2?128:1),smem(c.dynamic?c.dynamic/4:1);
  for(unsigned t=0;t<(c.kind==3?128:c.threads);++t)for(unsigned j=0;j<(c.kind==0||c.kind==3?c.words:1);++j)
    if(c.kind==0||c.kind==3)inputs.host[8+t*c.words+j]=initial(seed,t,j);
  if(c.kind==2)for(unsigned t=0;t<128;++t)inputs.host[8+t]=initial(seed,0,t);
  const auto original_inputs=inputs.values();inputs.upload();output.upload();smem.upload();
  AuxiliaryStamp* stamp=nullptr;AuxiliaryStamp got{};
  if(c.kind!=3){GH_CUDA(cudaMalloc(&stamp,sizeof(AuxiliaryStamp)));GH_CUDA(cudaMemset(stamp,0xff,sizeof(AuxiliaryStamp)));}
  cudaEvent_t begin,end;GH_CUDA(cudaEventCreate(&begin));GH_CUDA(cudaEventCreate(&end));
  unsigned *in=inputs.data(),*out=output.data(),*shared_out=smem.data(),smem_words=c.dynamic/4;
  int length=int(iterations);GH_CUDA(cudaEventRecord(begin));
  if(c.kind==0){void* args[]={&in,&out,&shared_out,&smem_words,&seed,&length,&stamp};GH_CUDA(cudaLaunchKernel(c.function,dim3(1),dim3(c.threads),args,c.dynamic,nullptr));}
  else if(c.kind==1){void* args[]={&seed,&length,&out,&stamp};GH_CUDA(cudaLaunchKernel(c.function,dim3(1),dim3(c.threads),args,0,nullptr));}
  else if(c.kind==2){void* args[]={&in,&out,&seed,&length,&stamp};GH_CUDA(cudaLaunchKernel(c.function,dim3(1),dim3(128),args,0,nullptr));}
  else {CUdeviceptr input_pointer=reinterpret_cast<CUdeviceptr>(in),output_pointer=reinterpret_cast<CUdeviceptr>(out);void* args[]={&input_pointer,&output_pointer,&iterations};driver_check(setmax.launch(setmax.function,1,1,1,128,1,1,0,nullptr,args,nullptr),"fixed setmax launch");}
  GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));float elapsed=0;GH_CUDA(cudaEventElapsedTime(&elapsed,begin,end));
  Check check;check.event_ms=elapsed;if(!std::isfinite(elapsed)||elapsed<0)throw std::runtime_error("S18 event completion/time");
  output.download();inputs.download();smem.download();auto values=output.values();
  if(c.kind==0)for(unsigned t=0;t<c.threads;++t)for(unsigned j=0;j<c.words;++j)equal(values[t*c.words+j],word(seed,t,j,iterations));
  else if(c.kind==1)for(unsigned t=0;t<c.threads;++t)for(unsigned j=0;j<c.words;++j){auto expected=service(c.operation,seed,t,j,iterations);auto at=2*(t*c.words+j);equal(values[at],unsigned(expected));equal(values[at+1],unsigned(expected>>32));}
  else if(c.kind==2)for(unsigned j=0;j<c.words;++j)equal(values[j],atomic(seed,j,c.words==1,iterations));
  else for(unsigned i=0;i<iterations;++i)for(unsigned phase=0;phase<2;++phase)for(unsigned t=0;t<128;++t)for(unsigned j=0;j<60;++j)equal(values[((i*2+phase)*128+t)*60+j],initial(seed,t,j));
  if(c.kind==2&&c.operation<2){auto global=inputs.values();for(unsigned j=0;j<128;++j)equal(global[j],c.words==1&&j>0?initial(seed,0,j):atomic(seed,j,c.words==1,iterations));}
  else if(inputs.values()!=original_inputs)throw std::runtime_error("S18 immutable input changed");
  if(c.dynamic){auto payload=smem.values();for(unsigned j=0;j<payload.size();++j)equal(payload[j],initial(seed,0,j));save(check,stage,"smem",payload,{payload.size()});}
  if(c.kind==0)save(check,stage,"values",values,{c.threads,c.words});
  else if(c.kind==1)save(check,stage,"values",values,{c.threads,c.words,2});
  else if(c.kind==2)save(check,stage,"values",values,{c.words});
  else save(check,stage,"values",values,{iterations,2,128,60});
  save(check,stage,"guards",output.guards(),{2,8});save(check,stage,"inputs",inputs.values(),{inputs.values().size()});save(check,stage,"input_guards",inputs.guards(),{2,8});
  if(c.kind!=3) {
    GH_CUDA(cudaMemcpy(&got,stamp,sizeof(got),cudaMemcpyDeviceToHost));
    if(got.begin_smid!=got.end_smid||got.begin_smid==~0u||got.end_ns<got.begin_ns||got.end_cycle<=got.begin_cycle||got.end_ns==~0ull||got.end_cycle==~0ull)
      throw std::runtime_error("S18 positive same-SM CTA time domain");
    // Short numerical evidence retains timer quantization; B4 must qualify metering.
    // Do not infer rates or reject correct values using an uncalibrated ns/event ratio.
    save(check,stage,"stamps",word_artifacts::split_u64({got.begin_ns,got.end_ns,got.begin_cycle,got.end_cycle,got.begin_smid,got.end_smid}),{6,2});GH_CUDA(cudaFree(stamp));
  }
  GH_CUDA(cudaEventDestroy(begin));GH_CUDA(cudaEventDestroy(end));return check;
}
static void emit_work(const AuxiliaryCase& c) {
  const unsigned lengths[]={1,2,5};const char* service_units[]={"PTX_add_u64","PTX_mad_wide_u32","PTX_cvt_rn_f16_f32","PTX_cvt_f32_f16"};
  const char* unit=c.kind==0?"u32_recurrence":c.kind==1?service_units[c.operation]:c.kind==2?"u32_atomic_logical_update":"setmax_legality_no_service";
  std::cout<<",\"logical_target_unit\":"<<gh::quote(unit)<<",\"logical_target_requests\":[";
  for(unsigned n=0;n<3;++n){std::cout<<(n?",":"");if(c.kind==3)std::cout<<"null";else std::cout<<gh::u64(c.threads)*(c.kind<2?c.words:1)*lengths[n];}
  std::cout<<"],\"explicit_local_request_bytes\":[";
  for(unsigned n=0;n<3;++n)std::cout<<(n?",":"")<<(c.kind==0&&c.operation==1?gh::u64(c.threads)*c.words*lengths[n]*8:0);
  std::cout<<"],\"logical_atomic_updates\":[";
  for(unsigned n=0;n<3;++n)std::cout<<(n?",":"")<<(c.kind==2?gh::u64(c.threads)*lengths[n]:0);
  std::cout<<"],\"expected_native_atomic_updates_from_frozen_SASS\":[";
  for(unsigned n=0;n<3;++n)std::cout<<(n?",":"")<<(c.kind==2?gh::u64(c.operation%2?4:128)*lengths[n]:0);
  std::cout<<"]";
}

int main(int argc,char** argv)try {
  const auto device=gh::device();gh::emit_device(device);
  if(argc==2&&std::string(argv[1])=="device")return 0;
  SetmaxModule setmax;
  if(argc==2&&std::string(argv[1])=="auxiliary-device") {for(const auto& c:auxiliary_cases)emit_capability(c,capability(c,setmax));return 0;}
  if(argc!=5||std::string(argv[1])!="validate-only"||std::string(argv[3])!="auxiliary_short_1_2_5_v1"||gh::integer(argv[4],0,4294967295ull)!=3)
    throw std::runtime_error("S18 finite validate-only CASE auxiliary_short_1_2_5_v1 3 required; formal disabled");
  const AuxiliaryCase* selected=nullptr;for(const auto& c:auxiliary_cases)if(std::string(c.id)==argv[2])selected=&c;
  if(!selected)throw std::runtime_error("S18 unknown finite coordinate");const auto& c=*selected;auto cap=capability(c,setmax);
  if(!cap.supported)throw std::runtime_error("S18 resource rejection before target: "+cap.reason);
  unsigned lengths[]={1,2,5},seeds[]={0,3,0xffffffffu};std::vector<Check> checks;
  for(unsigned stage=0;stage<3;++stage)checks.push_back(execute(c,setmax,stage,lengths[stage],seeds[stage]));
  const char* reference_sha="f6c401eb8e544b88f13054d4abf0db37081f7ac3ffeb1a88e336d8ae5c35af1f";
  std::cout<<"{\"schema_version\":2,\"validation_schema_version\":1,\"type\":\"validation\",\"case_id\":"<<gh::quote(c.id)<<",\"profile_id\":\"auxiliary_short_1_2_5_v1\",\"seed\":3,\"scope\":\"one_cta\",\"threads\":"<<c.threads<<",\"blocks\":1,\"errors\":0,\"performance_eligible\":false,\"warmup_executed\":false,\"pilot_executed\":false,\"target_launches\":[";
  for(unsigned n=0;n<3;++n)std::cout<<(n?",":"")<<"{\"launch_index\":"<<n<<",\"iterations\":"<<lengths[n]<<",\"input_profile\":\"auxiliary_nonuniform_paired_seeds_v1\",\"threads\":"<<c.threads<<",\"blocks\":1}";
  std::cout<<"],\"checks\":[";
  for(unsigned n=0;n<3;++n) {
    const auto& check=checks[n];std::cout<<(n?",":"")<<"{\"launch_index\":"<<n<<",\"reference_model\":\"auxiliary_word_reference_v1\",\"reference_sha256\":"<<gh::quote(reference_sha)<<",\"comparison\":\"exact\",\"tolerance_id\":null,\"completed\":true,\"errors\":0,\"checked_elements\":"<<check.checked<<",\"expected_elements\":"<<check.checked<<",\"verified_CTA_ids\":[0],\"output_artifacts\":[";
    for(unsigned j=0;j<check.artifacts.size();++j){const auto& item=check.artifacts[j];std::cout<<(j?",":"")<<"{\"path\":"<<gh::quote(item.path)<<",\"sha256\":"<<gh::quote(item.sha)<<",\"dtype\":\"uint32\",\"evidence_kind\":\"full_values\",\"shape\":[";for(unsigned k=0;k<item.shape.size();++k)std::cout<<(k?",":"")<<item.shape[k];std::cout<<"]}";}std::cout<<"]}";
  }
  std::cout<<"],\"resource_identity\":{\"kernel_symbol\":"<<gh::quote(c.symbol)<<",\"registers_per_thread\":"<<cap.registers<<",\"static_smem_bytes\":"<<cap.static_smem<<",\"dynamic_smem_bytes\":"<<c.dynamic<<",\"local_size_bytes\":"<<cap.local<<",\"occupancy_limit_ctas_per_sm\":"<<cap.occupancy<<",\"extensions\":{\"occupancy_is_upper_bound\":true,\"requested_carveout\":"<<c.carveout<<",\"actual_carveout\":null,\"requested_pressure_register_limit\":"<<(c.kind==0&&c.operation==2?32:0)<<",\"setmax_offline_cubin_sha256\":"<<gh::quote(c.kind==3?auxiliary_setmax_image_sha256:"")<<",\"no_formal_admission\":true,\"timer_policy\":\"numeric_short_positive_cycles_nondecreasing_ns_event_no_rate\",\"CUDA_event_ms\":[";for(unsigned n=0;n<3;++n)std::cout<<(n?",":"")<<std::setprecision(17)<<checks[n].event_ms;std::cout<<"]";emit_work(c);std::cout<<"}}}\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
