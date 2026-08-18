// SPDX-License-Identifier: BSD-3-Clause
#include "guide/cutlass_dense_case.hpp"

#include <cutlass/half.h>

using BaseConfig = guide::DenseGemmConfig<
    cutlass::half_t, cutlass::half_t, float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm,
    8, 8, 4>;

struct Config : BaseConfig {
  static auto stride_a(int m, int, int) {
    return StrideA{int64_t(128), cute::_1{}, int64_t(m) * 128};
  }
  static auto stride_b(int, int n, int) {
    return StrideB{int64_t(136), cute::_1{}, int64_t(n) * 136};
  }
  static auto stride_c(int m, int, int) {
    return StrideC{int64_t(132), cute::_1{}, int64_t(m) * 132};
  }
  static auto stride_d(int m, int, int) {
    return StrideD{int64_t(132), cute::_1{}, int64_t(m) * 132};
  }
};

int main(int argc, char** argv) {
  guide::CaseDescriptor descriptor{
      GUIDE_CASE_ID, "FP16 dense GEMM with logical M/N/K tails and padded strides",
      {130, 129, 127, 1}, {128, 128, 64}, {128, 128, 16}, {1, 1, 1},
      1, 0, "fp16", "fp16", "fp32", "fp32", "SS",
      "KernelTmaWarpSpecialized1SmSm100"};
  return guide::run_case_main(argc, argv, descriptor, [&](std::uint64_t seed) {
    return guide::verify_dense_cutlass<Config>(descriptor, seed, 0.01, 0.01);
  });
}
