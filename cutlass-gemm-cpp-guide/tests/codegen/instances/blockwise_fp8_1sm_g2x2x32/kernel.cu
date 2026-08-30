// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include <cutlass/device_kernel.h>
using GemmKernel = guide::codegen::blockwise_fp8_1sm_g2x2x32::Config::GemmKernel;
template __global__ void cutlass::device_kernel<GemmKernel>(
    CUTLASS_GRID_CONSTANT GemmKernel::Params const params);
