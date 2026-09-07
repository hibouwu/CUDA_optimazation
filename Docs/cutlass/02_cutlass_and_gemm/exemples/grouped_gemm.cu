// Grouped GEMM：Example 75 的 1SM 类型；文档中的三组问题合为一次调用。
// 对应文档：../01-thor-gemm-programming-guide_zh-CN.md，第六章。
// CUTLASS 固定版本：8f50b052e1099fb982392a622caab69b97b63128
// CUDA 13.x，compute_110a/sm_110a；编译与适用范围见 README.md。
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

// 6.2 为 Mainloop 与 Epilogue 同时选择逐组地址

using ArchTag = cutlass::arch::Sm100;  // C++ 实现族；二进制目标仍为 sm_110a。
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = LayoutC;
constexpr int AlignmentA = 16, AlignmentB = 16;  // 元素数。
constexpr int AlignmentC = 8, AlignmentD = 8;  // 元素数。
using MmaTileShape = cute::Shape<cute::_128, cute::_256, cute::_128>;  // Collective M/N/K。
using ClusterShape = cute::Shape<int32_t, int32_t, cute::_1>;  // CTA 个数；int 维度在运行时指定。
using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,             // ArchTag：架构实现族。
    OperatorClass,       // OpClass：计算引擎类别。
    MmaTileShape,        // TileShape_MNK：与 Mainloop 相同的局部尺寸。
    ClusterShape,        // ClusterShape_MNK：CTA Cluster。
    cutlass::epilogue::collective::EpilogueTileAuto,  // 后处理子分块。
    ElementAccumulator,  // 累加值类型。
    ElementCompute,      // 后处理计算类型。
    ElementC,            // C 的存储类型。
    LayoutC*,             // C 布局，* 选择逐组寻址。
    AlignmentC,          // C 对齐，以元素数计。
    ElementD,            // D 的存储类型。
    LayoutD*,             // D 布局，* 选择逐组寻址。
    AlignmentD,          // D 对齐，以元素数计。
    EpilogueSchedule,    // 结果读取与写回方式。
    cutlass::epilogue::fusion::LinearCombination<
        ElementD,        // 输出类型。
        ElementCompute,  // 线性组合计算类型。
        ElementC>        // 源矩阵类型；标量类型默认随 ElementCompute。
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,             // ArchTag：架构实现族。
    OperatorClass,       // OpClass：计算引擎类别。
    ElementA,            // A 的输入表示类型。
    LayoutA*,             // A 布局，* 选择逐组寻址。
    AlignmentA,          // A 对齐，以元素数计。
    ElementB,            // B 的输入表示类型。
    LayoutB*,             // B 布局，* 选择逐组寻址。
    AlignmentB,          // B 对齐，以元素数计。
    ElementAccumulator,  // K 维归约累加类型。
    MmaTileShape,        // TileShape_MNK：局部 M/N/K。
    ClusterShape,        // ClusterShape_MNK：CTA Cluster。
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
                         // StageCountType：扣除 Epilogue 共享内存后推导级数。
    MainloopSchedule     // KernelScheduleType：局部加载与计算方式。
>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,        // 问题描述类型。
    CollectiveMainloop,  // 已生成的 Mainloop。
    CollectiveEpilogue   // 已生成的 Epilogue；Tile Scheduler 使用默认选择。
>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    GemmKernel           // 被包装的设备 Kernel。
>;

static void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static void check(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(cutlassGetStatusString(status));
}
static cutlass::KernelHardwareInfo get_hardware_info() {
  cutlass::KernelHardwareInfo info{};
  check(cudaGetDevice(&info.device_id));  // 输出当前设备编号。
  info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
      info.device_id);  // 查询该设备的 SM 数量。
  return info;
}
// 辅助函数只复用 Dense 的调用顺序；Workspace 在同步完成前一直有效。
template<class GemmType>
static void run_gemm(typename GemmType::Arguments const& args, cudaStream_t stream) {
  GemmType gemm;
  check(gemm.can_implement(args));  // 本次参数与已生成 Kernel 的兼容性。
  cutlass::DeviceAllocation<uint8_t> workspace(
      GemmType::get_workspace_size(args));  // 所需辅助存储字节数。
  check(gemm.initialize(
      args,             // 问题、输入和 Epilogue 参数。
      workspace.get(),  // 设备 Workspace 地址。
      stream));         // 初始化与执行使用同一流。
  check(gemm.run(stream));
  check(cudaStreamSynchronize(stream));  // 返回前确保输出与 Workspace 不再被使用。
}
static bool close_value(double actual, double reference, double atol, double rtol) {
  return std::isfinite(actual) && std::isfinite(reference) &&
         std::abs(actual - reference) <= atol + rtol * std::abs(reference);
}

// -----------------------------------------------------------------------------
// 6.2 每组问题产生自己的输出 Tile；用 Shape、地址和 Stride 三类数组描述问题
// -----------------------------------------------------------------------------
int main() {
  try {
    check(cudaSetDevice(0));
    using Shape = typename ProblemShape::UnderlyingProblemShape;
    using StrideA = typename GemmKernel::InternalStrideA;  // 单组步长，不是数组指针。
    using StrideB = typename GemmKernel::InternalStrideB;
    using StrideC = typename GemmKernel::InternalStrideC;
    using StrideD = typename GemmKernel::InternalStrideD;
    const int G = 3;
    const int ms[G] = {128, 256, 64};
    const int ns[G] = {512, 256, 768};
    const int ks[G] = {128, 128, 256};
    std::vector<Shape> shapes;
    std::vector<StrideA> strides_a;
    std::vector<StrideB> strides_b;
    std::vector<StrideC> strides_c;
    std::vector<StrideD> strides_d;
    std::vector<size_t> offsets_a(G+1, 0), offsets_b(G+1, 0), offsets_c(G+1, 0);
    for (int g=0; g<G; ++g) {
      shapes.push_back(cute::make_shape(ms[g], ns[g], ks[g]));  // 本组 M/N/K。
      strides_a.push_back(cutlass::make_cute_packed_stride(
          StrideA{}, cute::make_shape(ms[g], ks[g], 1)));  // A：(m,k,l)。
      strides_b.push_back(cutlass::make_cute_packed_stride(
          StrideB{}, cute::make_shape(ns[g], ks[g], 1)));  // B：(n,k,l)。
      strides_c.push_back(cutlass::make_cute_packed_stride(
          StrideC{}, cute::make_shape(ms[g], ns[g], 1)));  // C：(m,n,l)。
      strides_d.push_back(cutlass::make_cute_packed_stride(
          StrideD{}, cute::make_shape(ms[g], ns[g], 1)));  // D：(m,n,l)。
      offsets_a[g+1] = offsets_a[g] + size_t(ms[g])*ks[g];
      offsets_b[g+1] = offsets_b[g] + size_t(ks[g])*ns[g];
      offsets_c[g+1] = offsets_c[g] + size_t(ms[g])*ns[g];
    }
    std::vector<ElementA> h_a(offsets_a.back());
    std::vector<ElementB> h_b(offsets_b.back());
    std::vector<ElementC> h_c(offsets_c.back());
    std::vector<ElementD> h_d(offsets_c.back());
    for (int g=0; g<G; ++g) {
      for (int m=0; m<ms[g]; ++m)
        for (int k=0; k<ks[g]; ++k)
          h_a[offsets_a[g]+size_t(m)*ks[g]+k] =
              ElementA(float((m+3*k+g)%11-5)/8);
      for (int n=0; n<ns[g]; ++n)
        for (int k=0; k<ks[g]; ++k)
          h_b[offsets_b[g]+size_t(n)*ks[g]+k] =
              ElementB(float((2*n+k+g)%9-4)/8);  // ColumnMajor B。
      for (int n=0; n<ns[g]; ++n)
        for (int m=0; m<ms[g]; ++m)
          h_c[offsets_c[g]+size_t(n)*ms[g]+m] =
              ElementC(float((m+n+g)%3-1)/4);  // ColumnMajor C/D。
    }
    cutlass::DeviceAllocation<ElementA> d_a(h_a.size());
    cutlass::DeviceAllocation<ElementB> d_b(h_b.size());
    cutlass::DeviceAllocation<ElementC> d_c(h_c.size());
    cutlass::DeviceAllocation<ElementD> d_d(h_d.size());
    d_a.copy_from_host(h_a.data());
    d_b.copy_from_host(h_b.data());
    d_c.copy_from_host(h_c.data());

    // 描述数组中的指针指向 Device Buffer，不能填 Host vector.data()。
    std::vector<ElementA const*> ptrs_a(G);
    std::vector<ElementB const*> ptrs_b(G);
    std::vector<ElementC const*> ptrs_c(G);
    std::vector<ElementD*> ptrs_d(G);
    for (int g=0; g<G; ++g) {
      ptrs_a[g] = d_a.get()+offsets_a[g];
      ptrs_b[g] = d_b.get()+offsets_b[g];
      ptrs_c[g] = d_c.get()+offsets_c[g];
      ptrs_d[g] = d_d.get()+offsets_c[g];
    }
    cutlass::DeviceAllocation<Shape> d_shapes(G);
    cutlass::DeviceAllocation<ElementA const*> d_ptr_a(G);
    cutlass::DeviceAllocation<ElementB const*> d_ptr_b(G);
    cutlass::DeviceAllocation<ElementC const*> d_ptr_c(G);
    cutlass::DeviceAllocation<ElementD*> d_ptr_d(G);
    cutlass::DeviceAllocation<StrideA> d_stride_a(G);
    cutlass::DeviceAllocation<StrideB> d_stride_b(G);
    cutlass::DeviceAllocation<StrideC> d_stride_c(G);
    cutlass::DeviceAllocation<StrideD> d_stride_d(G);
    d_shapes.copy_from_host(shapes.data());
    d_ptr_a.copy_from_host(ptrs_a.data());
    d_ptr_b.copy_from_host(ptrs_b.data());
    d_ptr_c.copy_from_host(ptrs_c.data());
    d_ptr_d.copy_from_host(ptrs_d.data());
    d_stride_a.copy_from_host(strides_a.data());
    d_stride_b.copy_from_host(strides_b.data());
    d_stride_c.copy_from_host(strides_c.data());
    d_stride_d.copy_from_host(strides_d.data());

    // -------------------------------------------------------------------------
    // 6.2 将问题数组交给一次调用
    // -------------------------------------------------------------------------
    auto hw = get_hardware_info();
    hw.cluster_shape = dim3(4,2,1);           // 动态 Cluster 的首选值。
    hw.cluster_shape_fallback = dim3(2,1,1);  // 回退值，沿用 Example 75。
    const float alpha=1.0f, beta=0.5f;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,  // 多问题模式。
        {G,              // num_groups：问题组数。
         d_shapes.get(), // problem_shapes：设备上的 Shape 数组。
         shapes.data()}, // host_problem_shapes：对应的 Host Shape 数组。
        { // mainloop：各数组相同的 g 对应同一组。
          d_ptr_a.get(),     // 各组 A 地址组成的设备数组。
          d_stride_a.get(),  // 各组 A 步长组成的设备数组。
          d_ptr_b.get(),     // 各组 B 地址组成的设备数组。
          d_stride_b.get()   // 各组 B 步长组成的设备数组。
        },
        { // epilogue：各组 C/D 使用匹配的逐组寻址。
          {alpha,            // 所有组共用的乘积系数。
           beta},            // 所有组共用的 C 系数。
          d_ptr_c.get(),     // 各组 C 地址组成的设备数组。
          d_stride_c.get(),  // 各组 C 步长组成的设备数组。
          d_ptr_d.get(),     // 各组 D 地址组成的设备数组。
          d_stride_d.get()   // 各组 D 步长组成的设备数组。
        },
        hw  // 设备信息、首选与回退的动态 Cluster。
    };
    run_gemm<Gemm>(args, nullptr);
    d_d.copy_to_host(h_d.data());

    // -------------------------------------------------------------------------
    // 6.2 区分问题集合与地址数组：逐组验证各自的真实边界
    // -------------------------------------------------------------------------
    bool passed=true;
    for (int g=0; g<G; ++g) {
      bool group_passed=true;
      double max_error=0;
      for (int n=0; n<ns[g]; ++n) {
        for (int m=0; m<ms[g]; ++m) {
          double acc=0;
          for (int k=0; k<ks[g]; ++k)
            acc += double(float(h_a[offsets_a[g]+size_t(m)*ks[g]+k])) *
                   double(float(h_b[offsets_b[g]+size_t(n)*ks[g]+k]));
          const size_t index=offsets_c[g]+size_t(n)*ms[g]+m;
          // 比较相同低精度输入，并计入最终 FP16 输出转换。
          const double ref=float(ElementD(float(alpha*acc+beta*float(h_c[index]))));
          const double actual=float(h_d[index]);
          max_error=std::max(max_error, std::abs(actual-ref));
          group_passed &= close_value(actual, ref, 1e-3, 1e-3);
        }
      }
      passed &= group_passed;
      std::cout << "group=" << g << " " << (group_passed?"PASS":"FAIL")
                << " max_abs_error=" << max_error << '\n';
    }
    return passed?0:1;
  } catch (std::exception const& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 2;
  }
}
