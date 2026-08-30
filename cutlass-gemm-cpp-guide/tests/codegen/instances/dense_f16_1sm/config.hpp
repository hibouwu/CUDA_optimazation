// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "guide/cutlass_dense_case.hpp"

#include <cutlass/half.h>

namespace guide::codegen::dense_f16_1sm {

using Config = guide::DenseGemmConfig<
    cutlass::half_t, cutlass::half_t, float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm,
    8, 8, 4>;

}  // namespace guide::codegen::dense_f16_1sm
