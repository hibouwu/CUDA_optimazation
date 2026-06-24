#include "gemm_common.cuh"
#include "v7_candidate_registry.cuh"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

int floor_div_or_zero(int numerator, int denominator) {
  if (denominator <= 0) return 0;
  return numerator / denominator;
}

std::string limiting_resources(int thread_limit, int warp_limit, int smem_limit,
                               int reg_limit, int api_blocks) {
  std::string result;
  auto append = [&](const char* name) {
    if (!result.empty()) result += "|";
    result += name;
  };
  if (api_blocks == thread_limit) append("threads");
  if (api_blocks == warp_limit) append("warps");
  if (api_blocks == smem_limit) append("shared_memory");
  if (api_blocks == reg_limit) append("registers_estimate");
  if (result.empty()) return "api_or_other";
  if (result.find('|') != std::string::npos) return "multiple_or_uncertain:" + result;
  return result;
}

}  // namespace

int main() {
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  const int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
  const auto candidates = v7_probe_candidates();

  std::filesystem::create_directories("scripts/v7_tuning/results");
  std::ofstream csv("scripts/v7_tuning/results/v7_resource_probe.csv");
  csv << "candidate_id,BM,BN,BK,WM,WN,TM,TN,WNITER,WMITER,WSUBM,WSUBN,"
         "NumThreads,warps_per_block,accumulators,reg_m,reg_n,R_static,"
         "shared_bytes_model,numRegs,sharedSizeBytes,localSizeBytes,"
         "maxThreadsPerBlock,activeBlocksPerSM,activeWarpsPerSM,"
         "theoreticalOccupancy,threadLimitBlocks,warpLimitBlocks,"
         "sharedLimitBlocks,registerLimitBlocksEstimate,limitingResources\n";

  std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor
            << ")\n";
  std::cout << "warpSize=" << prop.warpSize
            << ", maxThreadsPerSM=" << prop.maxThreadsPerMultiProcessor
            << ", maxWarpsPerSM=" << max_warps_per_sm
            << ", SMs=" << prop.multiProcessorCount << "\n";

  for (const auto& c : candidates) {
    cudaFuncAttributes attr{};
    CHECK_CUDA(cudaFuncGetAttributes(&attr, c.kernel));

    int active_blocks_per_sm = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, c.kernel, c.NumThreads, 0));

    const int active_warps_per_sm =
        active_blocks_per_sm * (c.NumThreads / prop.warpSize);
    const double occupancy =
        max_warps_per_sm > 0
            ? static_cast<double>(active_warps_per_sm) / max_warps_per_sm
            : 0.0;

    const int thread_limit =
        floor_div_or_zero(prop.maxThreadsPerMultiProcessor, c.NumThreads);
    const int warp_limit =
        floor_div_or_zero(max_warps_per_sm, c.NumThreads / prop.warpSize);
    const int smem_limit =
        attr.sharedSizeBytes > 0
            ? floor_div_or_zero(prop.sharedMemPerMultiprocessor,
                                static_cast<int>(attr.sharedSizeBytes))
            : 0;
    const int reg_limit =
        attr.numRegs > 0
            ? floor_div_or_zero(prop.regsPerMultiprocessor,
                                attr.numRegs * c.NumThreads)
            : 0;
    const std::string limit =
        limiting_resources(thread_limit, warp_limit, smem_limit, reg_limit,
                           active_blocks_per_sm);

    std::cout << c.candidate_id << ": numRegs=" << attr.numRegs
              << ", sharedSizeBytes=" << attr.sharedSizeBytes
              << ", localSizeBytes=" << attr.localSizeBytes
              << ", activeBlocksPerSM=" << active_blocks_per_sm
              << ", activeWarpsPerSM=" << active_warps_per_sm
              << ", occupancy=" << std::fixed << std::setprecision(4)
              << occupancy << ", limitingResources=" << limit << "\n";

    csv << c.candidate_id << "," << c.BM << "," << c.BN << "," << c.BK
        << "," << c.WM << "," << c.WN << "," << c.TM << "," << c.TN
        << "," << c.WNITER << "," << c.WMITER << "," << c.WSUBM << ","
        << c.WSUBN << "," << c.NumThreads << "," << c.warps_per_block
        << "," << c.accumulators << "," << c.reg_m << "," << c.reg_n
        << "," << c.R_static << "," << c.shared_bytes_model << ","
        << attr.numRegs << "," << attr.sharedSizeBytes << ","
        << attr.localSizeBytes << "," << attr.maxThreadsPerBlock << ","
        << active_blocks_per_sm << "," << active_warps_per_sm << ","
        << occupancy << "," << thread_limit << "," << warp_limit << ","
        << smem_limit << "," << reg_limit << "," << limit << "\n";
  }

  return 0;
}
