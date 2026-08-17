// SPDX-License-Identifier: BSD-3-Clause
#include "guide/cutlass_bias_relu_case.hpp"

int main(int argc, char** argv) {
  guide::CaseDescriptor descriptor{
      GUIDE_CASE_ID, "FP16 dense GEMM with fused bias and ReLU epilogue",
      {128, 128, 128, 1}, {128, 128, 64}, {128, 128, 16}, {1, 1, 1},
      1, 0, "fp16", "fp16", "fp32", "fp16", "SS",
      "KernelTmaWarpSpecialized1SmSm100"};
  return guide::run_case_main(argc, argv, descriptor, [&](std::uint64_t seed) {
    return guide::verify_bias_relu_f16(descriptor, seed);
  });
}
