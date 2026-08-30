// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"

#include <cutlass/device_kernel.h>

using GemmKernel = guide::codegen::dense_bs_nvfp4_1sm::Config::GemmKernel;

template __global__ void cutlass::device_kernel<GemmKernel>(
    CUTLASS_GRID_CONSTANT GemmKernel::Params const params);
