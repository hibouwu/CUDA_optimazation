// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cutlass/arch/mma_sm100.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>


namespace guide::codegen::blockwise_fp8_1sm_g2x2x32 {

using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<2,2,32,cute::UMMA::Major::MN,cute::UMMA::Major::K>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
struct Config {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = cutlass::float_e4m3_t;
  using ElementD = cutlass::float_e4m3_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 16;
  static constexpr int AlignmentB = 16;
  static constexpr int AlignmentC = 16;
  static constexpr int AlignmentD = 16;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using MainloopOperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueOperatorClass = cutlass::arch::OpClassTensorOp;
  using BuilderElementA = cutlass::float_e4m3_t;
  using BuilderElementB = cutlass::float_e4m3_t;
  using BuilderLayoutA = cute::tuple<cutlass::layout::RowMajor,LayoutSFA>;
  using BuilderLayoutB = cute::tuple<cutlass::layout::ColumnMajor,LayoutSFB>;
  using EpilogueElementC = cutlass::float_e4m3_t;
  using EpilogueElementD = cutlass::float_e4m3_t;
  using EpilogueLayoutC = cutlass::layout::RowMajor;
  using EpilogueLayoutD = cutlass::layout::RowMajor;
  using MmaTileShape = cute::Shape<cute::_128,cute::_128,cute::_128>;
  using ClusterShape = cute::Shape<cute::_1,cute::_1,cute::_1>;
  using ClusterDefaultShape = cute::Shape<cute::_1,cute::_1,cute::_1>;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
  using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
  using FusionOperation = void;
  using ProblemShape = cute::Shape<int,int,int,int>;
  using TileScheduler = void;

  using EpilogueBuilder = cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, EpilogueOperatorClass, MmaTileShape, ClusterShape,
      EpilogueTile, ElementAccumulator, ElementCompute,
      EpilogueElementC, EpilogueLayoutC, AlignmentC,
      EpilogueElementD, EpilogueLayoutD, AlignmentD,
      EpilogueSchedule>;
  using CollectiveEpilogue = typename EpilogueBuilder::CollectiveOp;
  using StagePolicy = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
  using MainloopBuilder = cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, MainloopOperatorClass,
      BuilderElementA, BuilderLayoutA, AlignmentA,
      BuilderElementB, BuilderLayoutB, AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      StagePolicy, MainloopSchedule>;
  using CollectiveMainloop = typename MainloopBuilder::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

}  // namespace guide::codegen::blockwise_fp8_1sm_g2x2x32
