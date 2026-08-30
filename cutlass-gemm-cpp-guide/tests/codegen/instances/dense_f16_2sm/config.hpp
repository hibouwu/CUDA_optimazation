// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "guide/cutlass_dense_case.hpp"

#include <cutlass/half.h>

namespace guide::codegen::dense_f16_2sm {

using Config = guide::DenseGemmConfig<
    cutlass::half_t, cutlass::half_t, float,
    cute::Shape<cute::_256, cute::_128, cute::_64>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::TmaWarpSpecialized2Sm,
    8, 8, 4>;

}  // namespace guide::codegen::dense_f16_2sm
