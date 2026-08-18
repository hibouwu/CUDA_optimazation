// SPDX-License-Identifier: BSD-3-Clause
#include "guide/cutlass_dense_case.hpp"

#include <cutlass/bfloat16.h>

using Config = guide::DenseGemmConfig<
    cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm,
    8, 8, 4>;

int main(int argc, char** argv) {
  guide::CaseDescriptor descriptor{
      GUIDE_CASE_ID, "BF16 dense GEMM 128x128x128 on one SM",
      {128, 128, 128, 1}, {128, 128, 64}, {128, 128, 16}, {1, 1, 1},
      1, 0, "bf16", "bf16", "fp32", "fp32", "SS",
      "KernelTmaWarpSpecialized1SmSm100"};
  return guide::run_case_main(argc, argv, descriptor, [&](std::uint64_t seed) {
    return guide::verify_dense_cutlass<Config>(descriptor, seed, 0.05, 0.02);
  });
}
