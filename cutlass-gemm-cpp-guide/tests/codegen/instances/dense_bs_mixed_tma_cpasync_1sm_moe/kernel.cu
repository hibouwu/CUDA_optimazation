// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include <cutlass/device_kernel.h>
using GemmKernel = guide::codegen::dense_bs_mixed_tma_cpasync_1sm_moe::Config::GemmKernel;
template __global__ void cutlass::device_kernel<GemmKernel>(
    CUTLASS_GRID_CONSTANT GemmKernel::Params const params);
