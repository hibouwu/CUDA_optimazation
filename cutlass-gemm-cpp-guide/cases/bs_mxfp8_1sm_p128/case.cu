// SPDX-License-Identifier: BSD-3-Clause
#include "guide/cutlass_blockscaled_case.hpp"

#include <cutlass/float_subbyte.h>

using Config = guide::BlockScaledGemmConfig<
    cutlass::mx_float8_t<cutlass::float_e4m3_t>,
    cutlass::mx_float8_t<cutlass::float_e4m3_t>,
    cutlass::bfloat16_t,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm,
    16, 16, 8>;

int main(int argc, char** argv) {
  guide::CaseDescriptor descriptor{
      GUIDE_CASE_ID, "MXFP8 block-scaled GEMM 128x128x128 triple-path case",
      {128, 128, 128, 1}, {128, 128, 128}, {128, 128, 32}, {1, 1, 1},
      1, 32, "mxfp8(e4m3,e8m0)", "mxfp8(e4m3,e8m0)", "fp32", "bf16",
      "SS+scale-TMEM", "KernelTmaWarpSpecialized1SmMxf8f6f4Sm100"};
  return guide::run_case_main(argc, argv, descriptor, [&](std::uint64_t seed) {
    return guide::verify_blockscaled_cutlass<Config>(descriptor, seed, 0.5, 0.05);
  });
}
