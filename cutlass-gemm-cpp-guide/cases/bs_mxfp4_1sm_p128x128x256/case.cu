// SPDX-License-Identifier: BSD-3-Clause
#include "guide/cutlass_blockscaled_case.hpp"

#include <cutlass/float_subbyte.h>

using Config = guide::BlockScaledGemmConfig<
    cutlass::mx_float4_t<cutlass::float_e2m1_t>,
    cutlass::mx_float4_t<cutlass::float_e2m1_t>,
    cutlass::bfloat16_t,
    cute::Shape<cute::_128, cute::_128, cute::_256>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmMxf4Sm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm,
    32, 32, 8>;

int main(int argc, char** argv) {
  guide::CaseDescriptor descriptor{
      GUIDE_CASE_ID, "Native MXFP4 block-scaled GEMM with K tile 256",
      {128, 128, 256, 1}, {128, 128, 256}, {128, 128, 64}, {1, 1, 1},
      1, 32, "mxfp4(e2m1,e8m0)", "mxfp4(e2m1,e8m0)", "fp32", "bf16",
      "SS+scale-TMEM", "KernelTmaWarpSpecialized1SmMxf4Sm100"};
  return guide::run_case_main(argc, argv, descriptor, [&](std::uint64_t seed) {
    return guide::verify_blockscaled_cutlass<Config>(descriptor, seed, 1.0, 0.1);
  });
}
