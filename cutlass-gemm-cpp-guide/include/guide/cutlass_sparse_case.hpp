// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "guide/case_runner.hpp"
#include "guide/test_support.hpp"

#include <cute/tensor.hpp>

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/float_subbyte.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

namespace guide {

// This type is intentionally instantiated even before the runtime harness is
// closed. It catches SM110a/CUTLASS API drift without pretending that template
// acceptance is sparse numerical evidence.
struct SparseNvfp4CandidateConfig {
  using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementC = void;
  using ElementD = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_256>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledSparseTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, float,
      ElementC, LayoutC, 8,
      ElementD, LayoutD, 8,
      cutlass::epilogue::TmaWarpSpecialized1SmNvf4>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledSparseTensorOp,
      ElementPairA, LayoutA, 64,
      ElementPairB, LayoutB, 32,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      cutlass::gemm::KernelSparseTmaWarpSpecialized1SmNvf4Sm100>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

inline VerificationResult sparse_runtime_not_closed(std::uint64_t seed) {
  VerificationResult result = require_sm110_device();
  if (result.status == Status::skip) return result;
  // Keep a runtime-dependent reference to initialize() so nvcc emits the
  // candidate device kernel for PTX/SASS inspection. The ordinary test seed
  // never enters this branch; it is not a substitute for valid arguments.
  if (seed == 0) {
    typename SparseNvfp4CandidateConfig::Gemm::Arguments arguments{};
    SparseNvfp4CandidateConfig::Gemm gemm;
    (void)gemm.initialize(arguments, nullptr);
  }
  // The independent host 2:4 contract is tested in tests/host. The remaining
  // bridge is CUTLASS compressor metadata -> full runtime GEMM -> logical CPU
  // output. Keep this explicitly NOT_RUN until that bridge is hardware-checked.
  result.status = Status::not_run;
  result.message = "sparse candidate compiled; independent compressor/metadata runtime closure is pending";
  return result;
}

}  // namespace guide
