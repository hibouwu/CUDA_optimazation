#include "r00_common.hpp"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>

using namespace cute;
using KernelTile = Shape<_128, _256, _64>;
using KernelCluster = Shape<_2, _1, _1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, KernelTile, KernelCluster,
    Shape<_128, _32>, float, float, void, cutlass::layout::RowMajor, 4, float,
    cutlass::layout::RowMajor, 4, cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cutlass::half_t, cutlass::layout::RowMajor,
    8, cutlass::half_t, cutlass::layout::RowMajor, 8, float, KernelTile, KernelCluster,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
inline void cutlass_check(cutlass::Status status, const char* operation = "run") {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string("CUTLASS ") + operation + ": " +
                             cutlassGetStatusString(status) +
                             "; CUDA: " + cudaGetErrorString(cudaGetLastError()));
}
int main(int argc, char** argv) {
  try {
    Options o(argc, argv);
    if (o.dtype != "fp16" || o.backend != "cutlass")
      throw std::runtime_error("fixed CUTLASS kernel requires fp16/cutlass");
    DeviceBuffer<__half> a(size_t(o.m) * o.k), b(size_t(o.k) * o.n);
    DeviceBuffer<float> d(size_t(o.m) * o.n);
    auto sa = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{},
                                               make_shape(o.m, o.k, 1));
    auto sb = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{},
                                               make_shape(o.n, o.k, 1));
    auto sd = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{},
                                               make_shape(o.m, o.n, 1));
    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                       {o.m, o.n, o.k, 1},
                                       {reinterpret_cast<cutlass::half_t*>(a.pointer), sa,
                                        reinterpret_cast<cutlass::half_t*>(b.pointer), sb},
                                       {{1.f, 0.f}, nullptr, sd, d.pointer, sd}};
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::cerr << "fixed kernel SMEM=" << sizeof(typename Kernel::SharedStorage)
              << " bytes, device CTA limit=" << properties.sharedMemPerBlockOptin
              << ", threads=" << Kernel::MaxThreadsPerBlock << '\n';
    Gemm gemm;
    cutlass_check(gemm.can_implement(arguments), "can_implement");
    DeviceBuffer<unsigned char> workspace(std::max<size_t>(1, Gemm::get_workspace_size(arguments)));
    cutlass_check(gemm.initialize(arguments, workspace.pointer), "initialize");
    std::string metadata =
        "{\"tile\":[128,256,64],\"cluster\":[2,1,1],\"stages\":4,\"threads\":" +
        std::to_string(Kernel::MaxThreadsPerBlock) +
        ",\"smem_bytes\":" + std::to_string(sizeof(typename Kernel::SharedStorage)) +
        ",\"epilogue_tile\":[128,32],\"schedule\":\"TmaWarpSpecializedCooperative\",\"layout_adapted\":false}";
    return measure_gemm(
        o, a.pointer, b.pointer, d.pointer, false, false, [&]() { cutlass_check(gemm.run()); },
        metadata);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
