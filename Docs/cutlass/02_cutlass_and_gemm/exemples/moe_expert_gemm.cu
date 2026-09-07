// MoE Expert GEMM：Example 92 的固定槽位路径，加固定 Top-1 路由的 Gather/恢复。
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

// 6.3 从矩阵方向理解 Token Count 为什么位于 N：对应类型配置

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
using MmaTileShape = cute::Shape<cute::_128, cute::_16, cute::_128>;  // Collective M/N/K。
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;  // CTA 个数；int 维度在运行时指定。
using ProblemShape = cutlass::gemm::MoEProblemShape<cute::Shape<int, int, int>>;
using MainloopSchedule = cutlass::gemm::KernelMixedTmaCpAsyncWarpSpecialized1SmSm100;
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,             // ArchTag：架构实现族。
    OperatorClass,       // OpClass：计算引擎类别。
    MmaTileShape,        // TileShape_MNK：与 Mainloop 相同的局部尺寸。
    ClusterShape,        // ClusterShape_MNK：CTA Cluster。
    cutlass::epilogue::collective::EpilogueTileAuto,  // 后处理子分块。
    ElementAccumulator,  // 累加值类型。
    ElementCompute,      // 后处理计算类型。
    ElementC,            // C 的存储类型。
    LayoutC,             // C 布局。
    AlignmentC,          // C 对齐，以元素数计。
    ElementD,            // D 的存储类型。
    LayoutD,             // D 布局。
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
    LayoutA,             // A 布局。
    AlignmentA,          // A 对齐，以元素数计。
    ElementB,            // B 的输入表示类型。
    LayoutB,             // B 布局。
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
// 6.3 从 Token 路由得到 Expert GEMM
// 本例提供一个固定 Top-1 路由，不实现 Router 网络或 Top-k 权重合并。
// -----------------------------------------------------------------------------
int main() {
  try {
    check(cudaSetDevice(0));
    const int E=3, M=128, K=128, max_n=32;
    const std::vector<int32_t> token_counts={8,17,32};  // 包含 N=17 的尾部 Tile。
    const int total_tokens=8+17+32;
    const size_t slot_a=size_t(M)*K, slot_b=size_t(max_n)*K, slot_d=size_t(M)*max_n;

    std::vector<ElementB> tokens(size_t(total_tokens)*K);
    for (int t=0; t<total_tokens; ++t)
      for (int k=0; k<K; ++k)
        tokens[size_t(t)*K+k]=ElementB(float((t+2*k)%13-6)/8);

    std::vector<ElementA> h_a(E*slot_a);
    std::vector<ElementB> h_b(E*slot_b, ElementB(0.0f));
    std::vector<ElementC> h_c(E*slot_d, ElementC(0.0f));
    std::vector<ElementD> h_d(E*slot_d);
    std::vector<std::vector<int>> token_ids(E);
    int prefix=0;
    for (int e=0; e<E; ++e) {
      // A_e=W_e^T，按 M×K RowMajor 保存每个 Expert 的权重。
      for (int m=0; m<M; ++m)
        for (int k=0; k<K; ++k)
          h_a[e*slot_a+size_t(m)*K+k]=ElementA(float((m+3*k+e)%11-5)/8);
      for (int n=0; n<token_counts[e]; ++n) {
        const int token=total_tokens-1-(prefix+n);  // 固定乱序映射，覆盖每个 Token 一次。
        token_ids[e].push_back(token);
        // Gather：X_e 的一行等于 B_e=X_e^T 的一列。
        for (int k=0; k<K; ++k)
          h_b[e*slot_b+size_t(n)*K+k]=tokens[size_t(token)*K+k];
      }
      prefix += token_counts[e];
    }
    cutlass::DeviceAllocation<ElementA> d_a(h_a.size());
    cutlass::DeviceAllocation<ElementB> d_b(h_b.size());
    cutlass::DeviceAllocation<ElementC> d_c(h_c.size());
    cutlass::DeviceAllocation<ElementD> d_d(h_d.size());
    cutlass::DeviceAllocation<int32_t> d_counts(E);
    d_a.copy_from_host(h_a.data());
    d_b.copy_from_host(h_b.data());
    d_c.copy_from_host(h_c.data());
    d_counts.copy_from_host(token_counts.data());

    // -------------------------------------------------------------------------
    // 6.3 从矩阵方向理解 Token Count 为什么位于 N
    // D_e = W_e^T X_e^T，实际 Shape 为 (M, token_counts[e], K)。
    // 6.3 用最大尺寸和 Token Count 描述每个 Expert
    // -------------------------------------------------------------------------
    using StrideC=typename GemmKernel::StrideC;
    using StrideD=typename GemmKernel::StrideD;
    auto stride_c=cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(M,max_n,E));  // ColumnMajor C：(1,M,M*max_n)。
    auto stride_d=cutlass::make_cute_packed_stride(
        StrideD{}, cute::make_shape(M,max_n,E));  // D 使用同样的最大槽位步长。
    ProblemShape problem{
        M,              // max_m：固定输出特征数。
        max_n,          // max_n：每个槽位的 Token 容量。
        K,              // max_k：固定输入特征数。
        E,              // Expert 数量。
        d_counts.get()  // 每个 Expert 的实际 Token Count，设备地址。
    };
    auto hw=get_hardware_info();
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,  // 多 Expert 问题。
        problem,  // 最大尺寸、Expert 数量和实际 Token Count。
        { // mainloop：最大槽位基址，不是指针数组。
          d_a.get(),  // 所有 Expert 权重槽位的基址。
          d_b.get()   // 所有 Expert 激活槽位的基址。
        },
        { // epilogue：alpha=1、beta=0，对应 Y_e=X_e W_e。
          {1.0f,       // alpha：乘积系数。
           0.0f},      // beta：不叠加 C。
          d_c.get(),   // 所有 C 槽位的基址。
          stride_c,    // C 的组内与跨 Expert 步长。
          d_d.get(),   // 所有 D 槽位的基址。
          stride_d     // D 的组内与跨 Expert 步长。
        },
        hw  // 设备信息；Cluster 已静态固定为 (1,1,1)。
    };
    run_gemm<Gemm>(args, nullptr);
    d_d.copy_to_host(h_d.data());

    // -------------------------------------------------------------------------
    // 6.3 从 Expert 输出恢复 Token 输出
    // -------------------------------------------------------------------------
    std::vector<float> restored(size_t(total_tokens)*M, 0.0f);
    std::vector<int> visits(total_tokens,0);
    for (int e=0; e<E; ++e) {
      for (int n=0; n<token_counts[e]; ++n) {
        const int token=token_ids[e][n];
        ++visits[token];
        for (int m=0; m<M; ++m)
          restored[size_t(token)*M+m]=float(h_d[e*slot_d+size_t(n)*M+m]);
      }
    }

    // -------------------------------------------------------------------------
    // 6.3 分别验证 Expert 计算与路由结果
    // 参考读取聚集前的 Token 和路由索引，不使用 h_b，避免掩盖 Gather 错误。
    // -------------------------------------------------------------------------
    bool passed=true;
    double max_error=0;
    for (int e=0; e<E; ++e) {
      bool expert_passed=true;
      for (int n=0; n<token_counts[e]; ++n) {
        const int token=token_ids[e][n];
        expert_passed &= visits[token]==1;
        for (int m=0; m<M; ++m) {
          double acc=0;
          for (int k=0; k<K; ++k)
            acc += double(float(tokens[size_t(token)*K+k])) *
                   double(float(h_a[e*slot_a+size_t(m)*K+k]));
          const double ref=float(ElementD(float(acc)));
          const double expert_value=float(h_d[e*slot_d+size_t(n)*M+m]);
          const double token_value=restored[size_t(token)*M+m];
          expert_passed &= close_value(expert_value,ref,1e-3,1e-3);
          expert_passed &= close_value(token_value,ref,1e-3,1e-3);
          max_error=std::max(max_error,std::abs(token_value-ref));
        }
      }
      passed &= expert_passed;
      std::cout << "expert=" << e << " tokens=" << token_counts[e]
                << " " << (expert_passed?"PASS":"FAIL") << '\n';
    }
    for (int count:visits) passed &= count==1;
    std::cout << "top1_restore=" << (passed?"PASS":"FAIL")
              << " max_abs_error=" << max_error << '\n';
    return passed?0:1;
  } catch (std::exception const& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 2;
  }
}
