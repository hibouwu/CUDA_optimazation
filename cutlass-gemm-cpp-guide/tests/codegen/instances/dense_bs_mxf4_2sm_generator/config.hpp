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


namespace guide::codegen::dense_bs_mxf4_2sm_generator {


struct Config {
  using ElementA = cutlass::float_e2m1_t;
  using ElementB = cutlass::float_e2m1_t;
  using ElementC = void;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentC = 4;
  static constexpr int AlignmentD = 4;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using MainloopOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using EpilogueOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using BuilderElementA = cute::tuple<cutlass::float_e2m1_t, cutlass::float_ue8m0_t>;
  using BuilderElementB = cute::tuple<cutlass::float_e2m1_t, cutlass::float_ue8m0_t>;
  using BuilderLayoutA = cutlass::layout::RowMajor;
  using BuilderLayoutB = cutlass::layout::ColumnMajor;
  using EpilogueElementC = void;
  using EpilogueElementD = float;
  using EpilogueLayoutC = cutlass::layout::RowMajor;
  using EpilogueLayoutD = cutlass::layout::RowMajor;
  using MmaTileShape = cute::Shape<cute::_256,cute::_128,cute::_256>;
  using ClusterShape = cute::Shape<cute::_2,cute::_1,cute::_1>;
  using ClusterDefaultShape = cute::Shape<cute::_2,cute::_1,cute::_1>;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmMxf4Sm100;
  using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized2Sm;
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

}  // namespace guide::codegen::dense_bs_mxf4_2sm_generator
