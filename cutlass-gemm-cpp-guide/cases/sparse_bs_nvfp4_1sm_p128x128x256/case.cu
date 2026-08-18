// SPDX-License-Identifier: BSD-3-Clause
#include "guide/cutlass_sparse_case.hpp"

static_assert(sizeof(guide::SparseNvfp4CandidateConfig::Gemm) > 0,
              "instantiate the CUTLASS sparse SM110 candidate");

int main(int argc, char** argv) {
  guide::CaseDescriptor descriptor{
      GUIDE_CASE_ID, "Structured-sparse NVFP4 block-scaled GEMM candidate",
      {128, 128, 256, 1}, {128, 128, 256}, {128, 128, 64}, {1, 1, 1},
      1, 16, "sparse-nvfp4(e2m1,ue4m3)", "nvfp4(e2m1,ue4m3)", "fp32", "bf16",
      "sparse-SS+scale-TMEM", "KernelSparseTmaWarpSpecialized1SmNvf4Sm100"};
  return guide::run_case_main(argc, argv, descriptor, [&](std::uint64_t seed) {
    return guide::sparse_runtime_not_closed(seed);
  });
}
