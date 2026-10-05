// Compile-only representative Dense recipe. Do not call force_dense_instantiation.
// Recipe copied structurally from the hash-verified dense_baseline.cu, not its runtime main.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <typeinfo>
#include <cxxabi.h>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/device_kernel.h"

using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
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
constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 4, AlignmentD = 4;
using MmaTileShape = cute::Shape<cute::_256, cute::_128, cute::_64>;
using ClusterShape = cute::Shape<cute::_2, cute::_2, cute::_1>;
using EpilogueOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, float, cutlass::FloatRoundStyle::round_to_nearest>;
using EpilogueBuilder = cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto, EpilogueOperation>;
using CollectiveEpilogue = typename EpilogueBuilder::CollectiveOp;
using StagePolicy = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
using MainloopBuilder = cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB, ElementAccumulator,
    MmaTileShape, ClusterShape, StagePolicy,
    cutlass::gemm::collective::KernelScheduleAuto>;
using CollectiveMainloop = typename MainloopBuilder::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using TiledMma = typename CollectiveMainloop::TiledMma;
using MmaAtom = typename TiledMma::Atom;
using MmaOp = typename TiledMma::MMA_Op;
using ExpectedOp = cute::SM100_MMA_F16BF16_2x1SM_SS<
    ElementA, ElementB, float, 256, 128,
    cute::UMMA::Major::K, cute::UMMA::Major::MN>;
static_assert(cute::is_same_v<MmaOp, ExpectedOp>, "Dense Auto must select the verified F16 2x1SM operation");
static_assert(cute::is_same_v<MmaAtom, cute::MMA_Atom<ExpectedOp>>, "Selected Atom must wrap the selected operation");
static_assert(MainloopBuilder::UmmaMajorA == cute::UMMA::Major::K);
static_assert(MainloopBuilder::UmmaMajorB == cute::UMMA::Major::MN);
static_assert(MainloopBuilder::is_2sm);
static_assert(cute::size<0>(typename GemmKernel::CtaShape_MNK{}) == 128);
static_assert(cute::size<1>(typename GemmKernel::CtaShape_MNK{}) == 128);
static_assert(cute::size<2>(typename MmaAtom::Shape_MNK{}) == 16);

#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ != 1100
#error This probe is specifically compiled for the sm_110a binary target.
#endif
#if !defined(CUTLASS_ARCH_MMA_SM110A_ENABLED) || !defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
#error The original snapshot architecture configuration must enable the real F16 TCGen05 branch.
#endif
#endif

#if !defined(PROBE_HOST_TYPES)
// This uncalled host function only forces real device-kernel template emission.
// Its launch configuration is deliberately NOT a validated runtime launch recipe.
void force_dense_instantiation(typename GemmKernel::Params params) {
  cutlass::device_kernel<GemmKernel><<<dim3(1), dim3(GemmKernel::MaxThreadsPerBlock),
      sizeof(typename GemmKernel::SharedStorage)>>>(params);
}
#else
template<class T>
void show_type(char const* label) {
  int status = 0;
  char* demangled = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
  std::printf("%s=%s\n", label, status == 0 ? demangled : typeid(T).name());
  std::free(demangled);
}
int main() {
  // Host-only type/sizeof inspection. No CUDA API, allocation, module loading or launch call here.
  show_type<MainloopBuilder>("Mainloop.Builder");
  show_type<StagePolicy>("Mainloop.StagePolicy");
  show_type<typename CollectiveMainloop::DispatchPolicy>("Mainloop.DispatchPolicy");
  show_type<CollectiveMainloop>("Mainloop.FullType");
  show_type<TiledMma>("Mainloop.TiledMma");
  show_type<MmaAtom>("Mainloop.MMA_Atom");
  show_type<MmaOp>("Mainloop.MMA_Op");
  show_type<typename MmaAtom::Traits>("Mainloop.MMA_Traits");
  show_type<typename MmaAtom::Shape_MNK>("Mainloop.AtomShape_MNK");
  show_type<typename GemmKernel::CtaShape_MNK>("Kernel.CtaShape_MNK");
  show_type<typename GemmKernel::TileScheduler>("Kernel.TileScheduler");
  show_type<CollectiveEpilogue>("Epilogue.FullType");
  std::printf("Builder.is_2sm=%d\n", int(MainloopBuilder::is_2sm));
  std::printf("Builder.UmmaMajorA=%d\n", int(MainloopBuilder::UmmaMajorA));
  std::printf("Builder.UmmaMajorB=%d\n", int(MainloopBuilder::UmmaMajorB));
  std::printf("Mainloop.PipelineStages=%d\n", int(MainloopBuilder::PipelineStages));
  std::printf("Mainloop.SchedulerStages=%u\n", unsigned(MainloopBuilder::SchedulerPipelineStageCount));
  std::printf("Mainloop.AccumulatorStages=%u\n", unsigned(MainloopBuilder::AccumulatorPipelineStageCount));
  std::printf("Epilogue.SharedStorage.bytes=%zu\n", sizeof(typename CollectiveEpilogue::SharedStorage));
  std::printf("Kernel.SharedStorage.bytes=%zu\n", sizeof(typename GemmKernel::SharedStorage));
}
#endif
