#pragma once

// CUTLASS calls these policies "Sm100" because they describe the first
// Blackwell TCGen05 programming model. CUDA 13 / CUTLASS 4.5.2 enable the same
// TCGen05 feature family when this translation unit is compiled for sm_110a.

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/util/packed_stride.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace gemm_sm110::cutlass_backend {

template <
    class MainloopSchedule, class EpilogueSchedule,
    class TileSchedulerTag = void,
    class TileShape = cute::Shape<cute::_128, cute::_128, cute::_64>,
    class ClusterShape_ = cute::Shape<cute::_1, cute::_1, cute::_1>>
struct Sm110GemmConfig {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using MmaTileShape = TileShape;
  using ClusterShape = ClusterShape_;

  using EpilogueOperation = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD, EpilogueSchedule,
          EpilogueOperation>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB,
          LayoutB, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(
                  sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop,
      CollectiveEpilogue, TileSchedulerTag>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Official CUTLASS Blackwell example 71 defaults:
//   KernelScheduleAuto + EpilogueScheduleAuto
//   256x128x64 MMA tile + 2x2x1 cluster
// With an even cluster-M, CUTLASS's auto policy selects the 2-SM MMA path.
// The operand/output layouts remain row-major to match this benchmark's
// existing input allocation and cuBLAS reference.
using CutlassOfficialConfig = Sm110GemmConfig<
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::collective::EpilogueScheduleAuto, void,
    cute::Shape<cute::_256, cute::_128, cute::_64>,
    cute::Shape<cute::_2, cute::_2, cute::_1>>;

template <class Config>
class Runner {
 public:
  using Gemm = typename Config::Gemm;
  using Arguments = typename Gemm::Arguments;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  Runner(const half* a, const half* b, float* d, int m, int n, int k)
      : gemm_(new Gemm) {
    const auto stride_a =
        cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    const auto stride_b =
        cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    const auto stride_c =
        cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    const auto stride_d =
        cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

    cutlass::KernelHardwareInfo hw_info;
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    hw_info.device_id = device;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);

    arguments_ = new Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {reinterpret_cast<const typename Config::ElementA*>(a), stride_a,
         reinterpret_cast<const typename Config::ElementB*>(b), stride_b},
        {{}, d, stride_c, d, stride_d},
        hw_info};
    arguments_->epilogue.thread.alpha = 1.0f;
    arguments_->epilogue.thread.beta = 0.0f;

    const std::size_t workspace_size = Gemm::get_workspace_size(*arguments_);
    if (workspace_size != 0) {
      check_cuda(cudaMalloc(&workspace_, workspace_size),
                 "cudaMalloc(CUTLASS workspace)");
    }

    check_status(gemm_->can_implement(*arguments_), "can_implement");
    check_status(gemm_->initialize(*arguments_, workspace_), "initialize");
  }

  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  ~Runner() {
    delete arguments_;
    delete gemm_;
    if (workspace_ != nullptr) {
      cudaFree(workspace_);
    }
  }

  void launch() { check_status(gemm_->run(), "run"); }

 private:
  static void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA failure in CUTLASS %s: %s\n", where,
                   cudaGetErrorString(status));
      std::abort();
    }
  }

  static void check_status(cutlass::Status status, const char* where) {
    if (status != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "CUTLASS failure in %s: %s\n", where,
                   cutlassGetStatusString(status));
      std::abort();
    }
  }

  Gemm* gemm_ = nullptr;
  Arguments* arguments_ = nullptr;
  void* workspace_ = nullptr;
};

}  // namespace gemm_sm110::cutlass_backend
