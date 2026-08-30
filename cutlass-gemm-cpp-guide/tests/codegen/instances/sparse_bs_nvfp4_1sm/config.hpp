// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cute/tensor.hpp>

#include <cutlass/bfloat16.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/float_subbyte.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

namespace guide::codegen::sparse_bs_nvfp4_1sm {

struct Config {
  using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementA = ElementPairA;
  using ElementB = ElementPairB;
  using ElementC = void;
  using ElementD = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 64;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentD = 8;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledSparseTensorOp;
  using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_256>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using MainloopSchedule = cutlass::gemm::KernelSparseTmaWarpSpecialized1SmNvf4Sm100;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1SmNvf4;
  using ProblemShape = cute::Shape<int, int, int, int>;
  using TileScheduler = void;
  using EpilogueBuilder = cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, float,
      ElementC, LayoutC, 8,
      ElementD, LayoutD, 8,
      EpilogueSchedule>;
  using CollectiveEpilogue = typename EpilogueBuilder::CollectiveOp;
  using StagePolicy =
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>;
  using MainloopBuilder = cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementPairA, LayoutA, AlignmentA,
      ElementPairB, LayoutB, AlignmentB,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      StagePolicy,
      MainloopSchedule>;
  using CollectiveMainloop = typename MainloopBuilder::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

}  // namespace guide::codegen::sparse_bs_nvfp4_1sm
