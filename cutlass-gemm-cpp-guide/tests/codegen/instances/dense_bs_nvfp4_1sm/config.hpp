// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "guide/cutlass_blockscaled_case.hpp"

#include <cutlass/float_subbyte.h>

namespace guide::codegen::dense_bs_nvfp4_1sm {

using Config = guide::BlockScaledGemmConfig<
    cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    cutlass::bfloat16_t,
    cute::Shape<cute::_128, cute::_128, cute::_256>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm,
    32, 32, 8>;

}  // namespace guide::codegen::dense_bs_nvfp4_1sm
