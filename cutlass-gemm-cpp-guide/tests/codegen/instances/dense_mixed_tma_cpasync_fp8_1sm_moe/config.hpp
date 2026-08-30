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


namespace guide::codegen::dense_mixed_tma_cpasync_fp8_1sm_moe {


struct Config {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentA = 16;
  static constexpr int AlignmentB = 16;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using MainloopOperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueOperatorClass = cutlass::arch::OpClassTensorOp;
  using BuilderElementA = cutlass::float_e4m3_t;
  using BuilderElementB = cutlass::float_e4m3_t;
  using BuilderLayoutA = cutlass::layout::RowMajor;
  using BuilderLayoutB = cutlass::layout::ColumnMajor;
  using EpilogueElementC = cutlass::half_t;
  using EpilogueElementD = cutlass::half_t;
  using EpilogueLayoutC = cutlass::layout::ColumnMajor;
  using EpilogueLayoutD = cutlass::layout::ColumnMajor;
  using MmaTileShape = cute::Shape<cute::_128,cute::_16,cute::_128>;
  using ClusterShape = cute::Shape<cute::_1,cute::_1,cute::_1>;
  using ClusterDefaultShape = cute::Shape<cute::_1,cute::_1,cute::_1>;
  using MainloopSchedule = cutlass::gemm::KernelMixedTmaCpAsyncWarpSpecialized1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
  using FusionOperation = cutlass::epilogue::fusion::LinearCombination<cutlass::half_t,float>;
  using ProblemShape = cutlass::gemm::MoEProblemShape<cute::Shape<int,int,int>>;
  using TileScheduler = void;

  using EpilogueBuilder = cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, EpilogueOperatorClass, MmaTileShape, ClusterShape,
      EpilogueTile, ElementAccumulator, ElementCompute,
      EpilogueElementC, EpilogueLayoutC, AlignmentC,
      EpilogueElementD, EpilogueLayoutD, AlignmentD,
      EpilogueSchedule,
      FusionOperation>;
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

}  // namespace guide::codegen::dense_mixed_tma_cpasync_fp8_1sm_moe
