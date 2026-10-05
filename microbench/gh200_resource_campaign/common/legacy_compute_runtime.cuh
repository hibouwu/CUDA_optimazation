#pragma once
#include "probe_runtime.cuh"
#include "legacy_compute_reference.hpp"
#include <functional>

namespace legacy_compute {
using Kernel = void(*)(int,unsigned,bool,gh::Stamp*,double*);
struct Case {
  const char* id;const char* kind;Kernel kernel;
  int threads,chains,batch,lanes,m,n,k,width;
  bool all_gpu,calibrated;int fixed_iterations,max_iterations;
};
inline std::vector<double> reference(const Case& c,int iterations,unsigned seed,bool nonuniform) {
  int outputs=c.width==1?c.lanes:c.m*c.n/c.width;
  std::vector<double> out(std::size_t(c.threads)*c.chains*outputs);
  std::vector<double> logical;
  if(c.width!=1)logical=legacy_reference::matrix(c.m,c.n,c.k,c.threads/c.width,c.chains,
                                              iterations,c.batch,seed,nonuniform);
  for(int t=0;t<c.threads;++t)for(int chain=0;chain<c.chains;++chain)for(int e=0;e<outputs;++e) {
    double value;
    if(c.width==1)value=legacy_reference::fma(c.kind,t,chain,e,seed,iterations,c.batch,nonuniform);
    else {
      auto rc=legacy_reference::output_coordinate(c.width,c.m,t%c.width,e);
      value=logical[(((t/c.width)*c.chains+chain)*c.m+rc.first)*c.n+rc.second];
    }
    out[(t*c.chains+chain)*outputs+e]=value;
  }
  return out;
}
inline int run(int argc,char**argv,const std::vector<Case>& cases) {
  auto device=gh::device();gh::emit_device(device);
  if(argc==2&&std::string(argv[1])=="device")return 0;
  if(argc!=4)throw std::runtime_error("CASE_ID ITERATIONS SEED required");
  int iterations=gh::integer(argv[2],0,1048576);
  unsigned seed=gh::integer(argv[3],0,4294967295ull);
  auto found=std::find_if(cases.begin(),cases.end(),[&](const Case& c){return c.id==std::string(argv[1]);});
  if(found==cases.end())throw std::runtime_error("unknown case");const Case& c=*found;
  if(c.calibrated ? (iterations<8192||iterations>c.max_iterations) : iterations!=c.fixed_iterations)
    throw std::runtime_error("fixed/calibrated iteration domain mismatch");
  cudaFuncAttributes attr{};GH_CUDA(cudaFuncGetAttributes(&attr,reinterpret_cast<const void*>(c.kernel)));
  if(attr.localSizeBytes)throw std::runtime_error("local memory/spill forbidden");
  int occupancy=0;GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,c.kernel,c.threads,0));
  if(occupancy<1)throw std::runtime_error("no resident CTA");
  int blocks=c.all_gpu?device.prop.multiProcessorCount*std::min(4,occupancy):1;
  auto uniform=reference(c,iterations,seed,false);
  const std::size_t per_cta=uniform.size(),elements=per_cta*blocks;
  gh::Stamp* stamps;double* output;
  GH_CUDA(cudaMalloc(&stamps,blocks*sizeof(gh::Stamp)));GH_CUDA(cudaMalloc(&output,elements*sizeof(double)));
  cudaEvent_t begin,end;GH_CUDA(cudaEventCreate(&begin));GH_CUDA(cudaEventCreate(&end));
  std::vector<double> got(elements);
  auto execute=[&](int length,bool nonuniform,const std::vector<double>& expected) {
    // NaN poison and allocation initialization precede the outer CUDA event.
    GH_CUDA(cudaMemset(output,0xff,elements*sizeof(double)));
    void* args[]={&length,&seed,&nonuniform,&stamps,&output};
    GH_CUDA(cudaEventRecord(begin));
    GH_CUDA(cudaLaunchKernel(reinterpret_cast<const void*>(c.kernel),dim3(blocks),dim3(c.threads),args,0,nullptr));
    GH_CUDA(cudaGetLastError());GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));
    gh::Observation o;o.stamps.resize(blocks);float ms=0;GH_CUDA(cudaEventElapsedTime(&ms,begin,end));o.event_ms=ms;
    // Internal nonuniform checks always have positive work, including for an empty formal case.
    o.timer_policy=gh::control_timer_policy(length==0&&!nonuniform&&!c.calibrated&&c.fixed_iterations==0,!c.all_gpu);
    GH_CUDA(cudaMemcpy(o.stamps.data(),stamps,blocks*sizeof(gh::Stamp),cudaMemcpyDeviceToHost));
    GH_CUDA(cudaMemcpy(got.data(),output,elements*sizeof(double),cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<elements;++i)if(!std::isfinite(got[i])||got[i]!=expected[i%per_cta])++o.errors;
    o.checked_elements=elements;o.method="uniform_and_nonuniform_full_output_v1";
    o.input_conditions="legacy uniform constants; separate seeded nonuniform validation; D reset each launch";
    if(c.all_gpu) {
      std::set<unsigned> ids;for(auto stamp:o.stamps)ids.insert(stamp.smid);
      if(int(ids.size())!=device.prop.multiProcessorCount)throw std::runtime_error("incomplete SM coverage");
    }
    gh::envelope(o);return o;
  };
  // The same arithmetic kernel receives distinct operand setup, outside formal timing.
  for(int length:{1,2})execute(length,true,reference(c,length,seed,true));
  auto launch_uniform=[&](){return execute(iterations,false,uniform);};
  auto warm=gh::warmup(launch_uniform);auto observation=launch_uniform();
  gh::u64 work=gh::u64(blocks)*iterations*c.batch*c.chains;
  if(c.width==1)work*=c.threads*c.lanes*2;
  else work*=gh::u64(c.threads/c.width)*2*c.m*c.n*c.k;
  std::ostringstream extension;
  extension<<"\"timing_model\":\"v2_cta_start_gate_result_drain_v1\",\"phase\":"
    <<gh::quote(iterations?"measure":"empty_control")
    <<",\"registers_per_thread\":"<<attr.numRegs<<",\"static_smem_bytes\":"<<attr.sharedSizeBytes
    <<",\"local_size_bytes\":"<<attr.localSizeBytes<<",\"occupancy_limit_ctas_per_sm\":"<<occupancy
    <<",\"max_abs_error\":0,\"output_elements_checked\":"<<elements
    <<",\"nonuniform_validation\":{\"iterations\":[1,2],\"checked_elements\":"<<elements*2
    <<",\"input_seed\":"<<seed<<",\"errors\":0,\"reference_model\":"
    <<gh::quote(c.width==1?"integer_dyadic_rne_fma_v1":"integer_dyadic_logical_matrix_v1")<<"}";
  gh::emit_trial(c.id,iterations,seed,c.threads,c.all_gpu?"all_gpu":"one_cta","FLOP",work,0,0,
                 observation,warm,extension.str());
  GH_CUDA(cudaEventDestroy(begin));GH_CUDA(cudaEventDestroy(end));
  GH_CUDA(cudaFree(stamps));GH_CUDA(cudaFree(output));return 0;
}
}
