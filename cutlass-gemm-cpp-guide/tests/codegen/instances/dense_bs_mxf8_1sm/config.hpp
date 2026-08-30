// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "guide/cutlass_blockscaled_case.hpp"

#include <cutlass/float_subbyte.h>

namespace guide::codegen::dense_bs_mxf8_1sm {

using Config = guide::BlockScaledGemmConfig<
    cutlass::mx_float8_t<cutlass::float_e4m3_t>,
    cutlass::mx_float8_t<cutlass::float_e4m3_t>,
    cutlass::bfloat16_t,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm,
    16, 16, 8>;

}  // namespace guide::codegen::dense_bs_mxf8_1sm
