// SPDX-License-Identifier: BSD-3-Clause
#include "guide/cutlass_dense_case.hpp"

#include <cutlass/half.h>

using Config = guide::DenseGemmConfig<
    cutlass::half_t, cutlass::half_t, float,
    cute::Shape<cute::_256, cute::_128, cute::_64>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::TmaWarpSpecialized2Sm,
    8, 8, 4>;

int main(int argc, char** argv) {
  guide::CaseDescriptor descriptor{
      GUIDE_CASE_ID, "FP16 dense GEMM 256x128x128 using a two-CTA MMA",
      {256, 128, 128, 1}, {256, 128, 64}, {256, 128, 16}, {2, 1, 1},
      2, 0, "fp16", "fp16", "fp32", "fp32", "SS",
      "KernelTmaWarpSpecialized2SmSm100"};
  return guide::run_case_main(argc, argv, descriptor, [&](std::uint64_t seed) {
    return guide::verify_dense_cutlass<Config>(descriptor, seed, 0.01, 0.01);
  });
}
