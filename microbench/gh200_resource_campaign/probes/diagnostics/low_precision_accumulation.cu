// Reuse the reviewed device source verbatim. Never call its renamed host main.
#define main s08_original_host_main_unused
#include "../low_precision.cu"
#undef main
#include <cstring>

static void emit_resource(const cudaFuncAttributes& attributes, int occupancy) {
  std::cout << "{\"kernel_symbol\":\"lp_wgmma_e4m3_g1\",\"registers_per_thread\":" << attributes.numRegs
    << ",\"static_smem_bytes\":" << attributes.sharedSizeBytes
    << ",\"dynamic_smem_bytes\":0,\"local_size_bytes\":" << attributes.localSizeBytes
    << ",\"occupancy_limit_ctas_per_sm\":" << occupancy << ",\"extensions\":{}}";
}

int main(int argc, char** argv) try {
  if (argc != 2 || (std::string(argv[1]) != "device" && std::string(argv[1]) != "collect"))
    throw std::runtime_error("usage: accumulation_probe device|collect; fixed coordinate only");
  const auto device = gh::device();
  const void* target = reinterpret_cast<const void*>(lp_wgmma_e4m3_g1);
  cudaFuncAttributes attributes{};GH_CUDA(cudaFuncGetAttributes(&attributes,target));
  int occupancy=0;
  GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,target,128,0));
  gh::emit_device(device);
  if (std::string(argv[1]) == "device") {
    std::cout << "{\"type\":\"resource\",\"resource_identity\":";
    emit_resource(attributes,occupancy);std::cout << "}\n";
    return 0;
  }
  constexpr size_t count=8192;
  int iterations=8192;unsigned seed=3;bool nonuniform=false;
  double* output=nullptr;gh::Stamp* stamps=nullptr;
  GH_CUDA(cudaMalloc(&output,count*sizeof(double)));
  GH_CUDA(cudaMalloc(&stamps,sizeof(gh::Stamp)));
  GH_CUDA(cudaMemset(output,0xff,count*sizeof(double)));
  GH_CUDA(cudaMemset(stamps,0xff,sizeof(gh::Stamp)));
  cudaEvent_t begin,end;GH_CUDA(cudaEventCreate(&begin));GH_CUDA(cudaEventCreate(&end));
  void* arguments[]={&iterations,&seed,&nonuniform,&stamps,&output};
  GH_CUDA(cudaEventRecord(begin));
  GH_CUDA(cudaLaunchKernel(target,dim3(1),dim3(128),arguments,0));
  GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));
  float event_ms=0;GH_CUDA(cudaEventElapsedTime(&event_ms,begin,end));
  std::vector<double> values(count);gh::Stamp stamp{};
  GH_CUDA(cudaMemcpy(values.data(),output,count*sizeof(double),cudaMemcpyDeviceToHost));
  GH_CUDA(cudaMemcpy(&stamp,stamps,sizeof(stamp),cudaMemcpyDeviceToHost));
  if (stamp.begin_ns==~gh::u64(0)||stamp.end_ns==~gh::u64(0)||
      stamp.begin_cycle==~gh::u64(0)||stamp.end_cycle==~gh::u64(0)||stamp.smid==~unsigned(0)||
      stamp.end_ns<stamp.begin_ns||stamp.end_cycle<stamp.begin_cycle)
    throw std::runtime_error("incomplete or reversed diagnostic stamp");
  if (!std::isfinite(event_ms)||event_ms<0)throw std::runtime_error("invalid completion event");
  for(size_t index=0;index<count;++index) {
    if(!std::isfinite(values[index])||double(float(values[index]))!=values[index])
      throw std::runtime_error("nonfinite/poison/non-FP32 output at index "+std::to_string(index));
  }
  std::cout<<std::setprecision(17)<<std::showpoint
    <<"{\"schema_version\":1,\"type\":\"long_accumulation_diagnostic\","
    <<"\"profile_id\":\"wgmma_e4m3_uniform_8192_full_output\",\"case_id\":\"wgmma_e4m3_g1_one_cta\","
    <<"\"target_symbol\":\"lp_wgmma_e4m3_g1\",\"iterations\":8192,\"seed\":3,\"threads\":128,\"blocks\":1,"
    <<"\"target_launches\":1,\"explicit_auxiliary_launches\":0,\"warmup_executed\":false,\"pilot_executed\":false,"
    <<"\"performance_eligible\":false,\"family_B3_eligible\":false,\"numerical_kernel_qualification\":false,"
    <<"\"mathematical_expected\":16384,\"resource_identity\":";
  emit_resource(attributes,occupancy);
  std::cout<<",\"stamp\":{\"begin_ns\":"<<stamp.begin_ns<<",\"end_ns\":"<<stamp.end_ns
    <<",\"begin_cycle\":"<<stamp.begin_cycle<<",\"end_cycle\":"<<stamp.end_cycle
    <<",\"smid\":"<<stamp.smid<<"},\"event_ms\":"<<event_ms<<",\"outputs\":[";
  for(size_t index=0;index<count;++index) {
    const float value=float(values[index]);unsigned bits=0;static_assert(sizeof(bits)==sizeof(value));
    std::memcpy(&bits,&value,sizeof(bits));
    std::cout<<(index?",":"")<<"{\"index\":"<<index<<",\"value\":"<<values[index]<<",\"f32_bits\":"<<bits<<'}';
  }
  std::cout<<"]}\n"<<std::flush;
  GH_CUDA(cudaEventDestroy(begin));GH_CUDA(cudaEventDestroy(end));
  GH_CUDA(cudaFree(output));GH_CUDA(cudaFree(stamps));return 0;
} catch(const std::exception& error) {
  std::cerr<<error.what()<<'\n';return 2;
}
