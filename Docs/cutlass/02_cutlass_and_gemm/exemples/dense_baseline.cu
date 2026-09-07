// CUTLASS Thor Dense FP16 GEMM：正文第五章的完整示例。
// 对应文档：../01-thor-gemm-programming-guide_zh-CN.md
// CUTLASS：8f50b052e1099fb982392a622caab69b97b63128
// 编译目标：CUDA 13.x，compute_110a/sm_110a。
// 本文件按正文小节划分；编译与运行命令见第五章“完整示例与编译运行”。

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
// CuTe 类型与 CUTLASS 的组件构造接口。
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
// 示例使用的设备内存管理与紧密步长工具。
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

// -----------------------------------------------------------------------------
// 5.1 固定输入类型与局部计算尺寸
// -----------------------------------------------------------------------------
// C++ 使用 Sm100 配方；Thor 二进制仍编译为 sm_110a。
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;  // K 维乘加的累加类型。
using ElementCompute = float;      // Epilogue 的计算类型。
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
// Alignment 以元素为单位，此处四个矩阵均对应 16 字节。
constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 4, AlignmentD = 4;
// Collective 的局部 (M,N,K)，不是完整问题尺寸。
using MmaTileShape = cute::Shape<
    cute::_256,  // M 维的局部 Tile 长度。
    cute::_128,  // N 维的局部 Tile 长度。
    cute::_64  // K 维的局部 Tile 长度。
>;
using ClusterShape = cute::Shape<
    cute::_2,  // M 方向的 CTA 个数。
    cute::_2,  // N 方向的 CTA 个数。
    cute::_1  // K 方向的 CTA 个数。
>;

// -----------------------------------------------------------------------------
// 5.2 先构造 Epilogue，再确定 Mainloop 的存储预算
// -----------------------------------------------------------------------------
// 线性组合：D = alpha * Acc + beta * C。
using EpilogueOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD,  // ElementOutput_：输出 D 的类型。
    ElementCompute,  // ElementCompute_：线性组合的计算类型。
    ElementC,  // ElementSource_：源矩阵 C 的类型。
    float,  // ElementScalar_：alpha/beta 的类型。
    cutlass::FloatRoundStyle::round_to_nearest  // RoundStyle_：输出转换的舍入规则。
>;
// 先确定 Epilogue，后面据此预留共享内存。
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,  // ArchTag：架构实现族。
    OperatorClass,  // OpClass：计算引擎类别。
    MmaTileShape,  // TileShape_MNK：与 Mainloop 匹配的 Collective Tile。
    ClusterShape,  // ClusterShape_MNK：与 Mainloop 匹配的 CTA Cluster。
    cutlass::epilogue::collective::EpilogueTileAuto,  // EpilogueTileType：Epilogue 子分块；Auto 在编译期选择。
    ElementAccumulator,  // ElementAccumulator：Mainloop 提供的累加类型。
    ElementCompute,  // ElementCompute：后处理使用的计算类型。
    ElementC,  // ElementC：源矩阵 C 的存储类型。
    LayoutC,  // GmemLayoutTagC：C 的全局内存布局标签。
    AlignmentC,  // AlignmentC：C 的访问对齐，以元素数计。
    ElementD,  // ElementD：输出矩阵 D 的存储类型。
    LayoutD,  // GmemLayoutTagD：D 的全局内存布局标签。
    AlignmentD,  // AlignmentD：D 的访问对齐，以元素数计。
    cutlass::epilogue::collective::EpilogueScheduleAuto,  // EpilogueScheduleType：结果读取、后处理与写回方式。
    EpilogueOperation  // FusionOpOrCallbacks：融合操作或回调类型。
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,  // ArchTag：架构实现族。
    OperatorClass,  // OpClass：计算引擎类别。
    ElementA,  // ElementA：A 的输入存储类型。
    LayoutA,  // GmemLayoutA：A 的全局内存布局标签。
    AlignmentA,  // AlignmentA：A 的访问对齐，以元素数计。
    ElementB,  // ElementB：B 的输入存储类型。
    LayoutB,  // GmemLayoutB：B 的全局内存布局标签。
    AlignmentB,  // AlignmentB：B 的访问对齐，以元素数计。
    ElementAccumulator,  // ElementAccumulator：K 维归约的累加类型。
    MmaTileShape,  // TileShape_MNK：完整 Collective 的局部 M/N/K 尺寸。
    ClusterShape,  // ClusterShape_MNK：CTA Cluster 的形状，以 CTA 个数计。
    // StageCountType：Mainloop 流水级数或自动推导策略。
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))  // carveout_bytes：为 Epilogue 预留的共享内存字节数。
    >,
    cutlass::gemm::collective::KernelScheduleAuto  // KernelScheduleType：Mainloop 的加载、乘加与协作方式。
>::CollectiveOp;  // 编译期选择实现。

// 四个 int 为运行时 M/N/K/L 留出位置。
// TileScheduler_ 未显式指定，使用该 Kernel 的默认选择。
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<
        int,  // M 维的运行时整数类型。
        int,  // N 维的运行时整数类型。
        int,  // K 维的运行时整数类型。
        int  // L 维的运行时整数类型。
    >,  // ProblemShapeOrThreadblockMma_：3.x 的问题描述类型。
    CollectiveMainloop,  // CollectiveMainloopOrEpilogue_：3.x 的 Mainloop 类型。
    CollectiveEpilogue  // CollectiveEpilogueOrThreadblockSwizzle_：3.x 的 Epilogue 类型。
>;
// 将设备 Kernel 包装为主机调用句柄。
using GemmHandle = cutlass::gemm::device::GemmUniversalAdapter<
    GemmKernel  // GemmKernel_：被包装的设备端 Kernel 类型。
>;

// -----------------------------------------------------------------------------
// 公共辅助：错误检查（定义放在 main 之前，正文在参考比较后说明）
// -----------------------------------------------------------------------------
// 统一将 CUDA 与 CUTLASS 的失败状态转换为异常。
static void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
static void check(cutlass::Status result) {
  if (result != cutlass::Status::kSuccess)
    throw std::runtime_error(cutlassGetStatusString(result));
}

int main() {
  try {
    // -----------------------------------------------------------------------------
    // 5.3 用 Shape 与 Stride 描述本次矩阵
    // -----------------------------------------------------------------------------
    // 从当前 Kernel 取得匹配的 Stride 类型。
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;

    int M = 256, N = 256, K = 128, L = 1;
    auto problem_shape = cute::make_shape(
        M,  // 输出行数。
        N,  // 输出列数。
        K,  // 归约长度。
        L   // 批次数，此处为单问题。
    );
    // 按紧密存储生成步长；带行尾填充时应改用实际步长。
    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{},  // 当前 Kernel 要求的 A 步长类型。
        cute::make_shape(M, K, L)  // A 的 (行数, 归约长度, 批次数)。
    );
    // B 的坐标顺序是 (n,k,l)，数学形状仍为 K×N。
    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{},  // 当前 Kernel 要求的 B 步长类型。
        cute::make_shape(N, K, L)  // B 的 (输出列数, 归约长度, 批次数)。
    );
    auto stride_C = cutlass::make_cute_packed_stride(
        StrideC{},  // 当前 Kernel 要求的 C 步长类型。
        cute::make_shape(M, N, L)  // C 的 (行数, 列数, 批次数)。
    );
    auto stride_D = cutlass::make_cute_packed_stride(
        StrideD{},  // 当前 Kernel 要求的 D 步长类型。
        cute::make_shape(M, N, L)  // D 的 (行数, 列数, 批次数)。
    );

    // -----------------------------------------------------------------------------
    // 5.4 准备矩阵数据并保留参考输入
    // -----------------------------------------------------------------------------
    check(cudaSetDevice(0));  // 参数 0 为设备编号；在分配设备内存前选择。
    std::vector<ElementA> h_A(size_t(M)*K);
    std::vector<ElementB> h_B(size_t(K)*N);
    std::vector<ElementC> h_C(size_t(M)*N);
    std::vector<ElementD> h_D(size_t(M)*N);
    // A/B 写入时即转为 FP16，参考计算也读取这些实际存储值。
    for (int m=0; m<M; ++m)
      for (int k=0; k<K; ++k)
        h_A[size_t(m)*K+k] = ElementA(float((m+2*k)%7-3)/4);
    for (int k=0; k<K; ++k)
      for (int n=0; n<N; ++n)
        h_B[size_t(k)*N+n] = ElementB(float((3*k+n)%5-2)/4);
    for (int m=0; m<M; ++m)
      for (int n=0; n<N; ++n)
        h_C[size_t(m)*N+n] = float((m+n)%3-1)/4;

    // 分配大小以元素为单位，D 由本次 GEMM 写入。
    cutlass::DeviceAllocation<ElementA> d_A(h_A.size());  // ElementA：存储类型；size()：元素数。
    cutlass::DeviceAllocation<ElementB> d_B(h_B.size());  // ElementB：存储类型；size()：元素数。
    cutlass::DeviceAllocation<ElementC> d_C(h_C.size());  // ElementC：存储类型；size()：元素数。
    cutlass::DeviceAllocation<ElementD> d_D(h_D.size());  // ElementD：存储类型；size()：元素数。
    d_A.copy_from_host(h_A.data());  // 源 Host 地址；默认复制整个 d_A。
    d_B.copy_from_host(h_B.data());  // 源 Host 地址；默认复制整个 d_B。
    d_C.copy_from_host(h_C.data());  // 源 Host 地址；默认复制整个 d_C。

    // -----------------------------------------------------------------------------
    // 5.5 将参数交给 Adapter 并等待输出完成
    // -----------------------------------------------------------------------------
    // 记录已选择的设备，供 Kernel 配置使用。
    cutlass::KernelHardwareInfo hardware_info{};
    check(cudaGetDevice(&hardware_info.device_id));  // 输出参数：当前设备编号的保存地址。
    hardware_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
        hardware_info.device_id  // 要查询 SM 数量的设备编号。
    );
    float alpha = 1.0f, beta = 0.5f;
    typename GemmHandle::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,  // mode：普通 GEMM 模式。
        problem_shape,  // problem_shape：本次 M/N/K/L。
        { // mainloop：输入矩阵。
          d_A.get(),  // A 的设备地址。
          stride_A,   // A 的 (m,k,l) 步长。
          d_B.get(),  // B 的设备地址。
          stride_B    // B 的 (n,k,l) 步长。
        },
        { // epilogue：线性组合与输出。
          {
            alpha,  // Acc 的缩放系数。
            beta    // C 的缩放系数。
          },
          d_C.get(),  // 源矩阵 C 的设备地址。
          stride_C,   // C 的 (m,n,l) 步长。
          d_D.get(),  // 输出矩阵 D 的设备地址。
          stride_D    // D 的 (m,n,l) 步长。
        },
        hardware_info  // hw_info：设备编号与 SM 数量。
    };

    GemmHandle gemm;
    check(gemm.can_implement(
        arguments  // 要检查的本次调用参数。
    ));
    size_t workspace_bytes = GemmHandle::get_workspace_size(
        arguments  // 按本次问题和调度参数计算辅助存储需求。
    );
    cutlass::DeviceAllocation<uint8_t> workspace(
        workspace_bytes  // 分配元素数；uint8_t 的元素数等于字节数。
    );
    cudaStream_t stream = nullptr;  // 使用默认 stream。
    check(gemm.initialize(
        arguments,        // args：本次问题、矩阵和后处理参数。
        workspace.get(),  // workspace：辅助存储的设备地址。
        stream            // stream：初始化所用执行流。
    ));
    // 使用 initialize 保存的参数提交执行，不表示结果已经就绪。
    check(gemm.run(
        stream  // Kernel 的执行流。
    ));
    check(cudaStreamSynchronize(
        stream  // 等待此流上的工作完成。
    ));
    d_D.copy_to_host(
        h_D.data()  // 目标 Host 地址；默认复制整个 d_D。
    );

    // -----------------------------------------------------------------------------
    // 5.6 用相同输入计算参考结果
    // -----------------------------------------------------------------------------
    double max_error = 0;
    bool passed = true;
    for (int m=0; m<M; ++m) {
      for (int n=0; n<N; ++n) {
        // 将同一份 FP16 输入提升为 double，计算独立参考值。
        double accum = 0;
        for (int k=0; k<K; ++k)
          accum += double(float(h_A[size_t(m)*K+k])) * double(float(h_B[size_t(k)*N+n]));
        double reference = double(alpha)*accum + double(beta)*double(h_C[size_t(m)*N+n]);
        double actual = h_D[size_t(m)*N+n];
        double error = std::abs(actual-reference);
        max_error = std::fmax(max_error,error);
        // 同时检查有限值、绝对误差与相对误差。
        passed &= std::isfinite(actual) && error <= 1e-5 + 1e-5*std::abs(reference);
      }
    }

    std::cout << (passed ? "PASS" : "FAIL")
              << " max_abs_error=" << max_error << '\n';
    return passed ? 0 : 1;
  } catch (std::exception const& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 2;
  }
}

// -----------------------------------------------------------------------------
// 5.7 将基线用于另一项矩阵问题
// -----------------------------------------------------------------------------
// 本节是迁移说明，不增加另一轮计算。
// 只改尺寸、地址或可变步长：重新准备 Buffer、Arguments 和所需 Workspace，再初始化。
// 改 Element、Layout、Tile、Cluster 等类型：重新构造 Builder、Kernel 与 Adapter。
// 不能只改 Host arguments 后直接 run；完整解释见正文对应小节。
