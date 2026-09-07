// 分离式 Attention：两个 CUTLASS FP16 Tensor Core GEMM，加独立 CUDA Softmax。
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

// 6.4 先把阶段依赖表示为两个独立 GEMM：分别固定两次乘法的输入布局。
namespace qk {

using ArchTag = cutlass::arch::Sm100;  // C++ 实现族；二进制目标仍为 sm_110a。
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;
constexpr int AlignmentA = 8, AlignmentB = 8;  // 元素数。
constexpr int AlignmentC = 4, AlignmentD = 4;  // 元素数。
using MmaTileShape = cute::Shape<cute::_256, cute::_128, cute::_64>;  // Collective M/N/K。
using ClusterShape = cute::Shape<cute::_2, cute::_2, cute::_1>;  // CTA 个数；int 维度在运行时指定。
using ProblemShape = cute::Shape<int, int, int, int>;
using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;
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
}
namespace pv {

using ArchTag = cutlass::arch::Sm100;  // C++ 实现族；二进制目标仍为 sm_110a。
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
using LayoutD = LayoutC;
constexpr int AlignmentA = 8, AlignmentB = 8;  // 元素数。
constexpr int AlignmentC = 4, AlignmentD = 4;  // 元素数。
using MmaTileShape = cute::Shape<cute::_256, cute::_128, cute::_64>;  // Collective M/N/K。
using ClusterShape = cute::Shape<cute::_2, cute::_2, cute::_1>;  // CTA 个数；int 维度在运行时指定。
using ProblemShape = cute::Shape<int, int, int, int>;
using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;
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
}

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
// 6.4 先把阶段依赖表示为两个独立 GEMM
// 这是独立的教学 Softmax：每个 CUDA 线程处理一行，不是高性能融合实现。
// -----------------------------------------------------------------------------
__global__ void causal_softmax_to_half(
    float const* scores,       // 已乘 1/sqrt(d) 的 RowMajor 分数。
    cutlass::half_t* p,         // 输出 FP16 概率，供第二个 Tensor Core GEMM 使用。
    int rows, int cols) {      // 本例 rows==cols，因果条件为 j<=i。
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=rows) return;
  float max_score=-INFINITY;
  for (int j=0; j<cols; ++j)
    if (j<=i) max_score=fmaxf(max_score,scores[size_t(i)*cols+j]);
  float denom=0;
  for (int j=0; j<cols; ++j)
    if (j<=i) denom += expf(scores[size_t(i)*cols+j]-max_score);
  for (int j=0; j<cols; ++j) {
    const float probability=j<=i ? expf(scores[size_t(i)*cols+j]-max_score)/denom : 0.0f;
    p[size_t(i)*cols+j]=cutlass::half_t(probability);
  }
}

int main() {
  try {
    check(cudaSetDevice(0));
    // 单 Batch、单 Head，等长 Q/K；Value 特征宽度可以与 Q/K 不同。
    const int Sq=256,Sk=256,d=128,dv=256;
    using Half=cutlass::half_t;
    std::vector<Half> h_q(size_t(Sq)*d),h_k(size_t(Sk)*d),h_v(size_t(Sk)*dv);
    for (int i=0; i<Sq; ++i)
      for (int k=0; k<d; ++k) h_q[size_t(i)*d+k]=Half(float((i+2*k)%11-5)/8);
    for (int j=0; j<Sk; ++j)
      for (int k=0; k<d; ++k) h_k[size_t(j)*d+k]=Half(float((3*j+k)%13-6)/8);
    for (int j=0; j<Sk; ++j)
      for (int v=0; v<dv; ++v) h_v[size_t(j)*dv+v]=Half(float((j+5*v)%9-4)/8);
    // 两个 GEMM 都取 beta=0，但仍提供明确初始化的 C。
    std::vector<float> zeros(size_t(Sq)*std::max(Sk,dv),0.0f);
    cutlass::DeviceAllocation<Half> q(h_q.size()),k(h_k.size()),v(h_v.size());
    cutlass::DeviceAllocation<Half> p(size_t(Sq)*Sk);
    cutlass::DeviceAllocation<float> scores(size_t(Sq)*Sk),o(size_t(Sq)*dv),c(zeros.size());
    q.copy_from_host(h_q.data());
    k.copy_from_host(h_k.data());
    v.copy_from_host(h_v.data());
    c.copy_from_host(zeros.data());
    auto hw=get_hardware_info();
    const float scale=1.0f/std::sqrt(float(d));

    // GEMM 1：Q(Sq,d) × Kᵀ(d,Sk)，B 是 ColumnMajor，无须复制转置 K。
    auto sa=cutlass::make_cute_packed_stride(
        qk::GemmKernel::StrideA{},cute::make_shape(Sq,d,1));  // Q：(i,k,l)。
    auto sb=cutlass::make_cute_packed_stride(
        qk::GemmKernel::StrideB{},cute::make_shape(Sk,d,1));  // Kᵀ：(j,k,l)。
    auto sc=cutlass::make_cute_packed_stride(
        qk::GemmKernel::StrideC{},cute::make_shape(Sq,Sk,1));
    auto sd=cutlass::make_cute_packed_stride(
        qk::GemmKernel::StrideD{},cute::make_shape(Sq,Sk,1));
    qk::Gemm::Arguments qk_args{
        cutlass::gemm::GemmUniversalMode::kGemm,  // 普通 GEMM。
        {Sq,  // M：Query 序列长度。
         Sk,  // N：Key 序列长度。
         d,   // K：Q/K 的特征宽度。
         1},  // L：单 Batch、单 Head。
        {q.get(), // Q 的设备地址。
         sa,      // Q 的步长。
         k.get(), // Kᵀ 的设备地址，与原始 RowMajor K 共用存储。
         sb},     // Kᵀ 的步长。
        {{scale,  // alpha：1/sqrt(d)。
          0.0f},  // beta：不叠加 C。
         c.get(), // 零 C 的设备地址。
         sc,      // C 的步长。
         scores.get(), // FP32 分数的设备地址。
         sd},          // 分数矩阵的步长。
        hw  // 设备信息。
    };
    run_gemm<qk::Gemm>(qk_args,nullptr);

    // Mask + Softmax：S 先按因果关系屏蔽，再归一化，最后显式转为 FP16 P。
    causal_softmax_to_half<<<(Sq+127)/128,128>>>(scores.get(),p.get(),Sq,Sk);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(nullptr));

    // GEMM 2：P(Sq,Sk) × V(Sk,dv)，两个输入均为 RowMajor FP16。
    auto pa=cutlass::make_cute_packed_stride(
        pv::GemmKernel::StrideA{},cute::make_shape(Sq,Sk,1));
    auto pb=cutlass::make_cute_packed_stride(
        pv::GemmKernel::StrideB{},cute::make_shape(dv,Sk,1));  // B 仍按 (n,k,l)。
    auto pc=cutlass::make_cute_packed_stride(
        pv::GemmKernel::StrideC{},cute::make_shape(Sq,dv,1));
    auto pd=cutlass::make_cute_packed_stride(
        pv::GemmKernel::StrideD{},cute::make_shape(Sq,dv,1));
    pv::Gemm::Arguments pv_args{
        cutlass::gemm::GemmUniversalMode::kGemm,  // 普通 GEMM。
        {Sq,  // M：Query 序列长度。
         dv,  // N：Value 的特征宽度。
         Sk,  // K：Key 序列长度。
         1},  // L：单 Batch、单 Head。
        {p.get(), // FP16 概率的设备地址。
         pa,      // P 的步长。
         v.get(), // V 的设备地址。
         pb},     // V 的步长。
        {{1.0f,   // alpha：直接累加 P*V。
          0.0f},  // beta：不叠加 C。
         c.get(), // 零 C 的设备地址。
         pc,      // C 的步长。
         o.get(), // FP32 输出的设备地址。
         pd},     // O 的步长。
        hw  // 设备信息。
    };
    run_gemm<pv::Gemm>(pv_args,nullptr);
    std::vector<float> h_scores(size_t(Sq)*Sk),h_o(size_t(Sq)*dv);
    std::vector<Half> h_p(size_t(Sq)*Sk);
    scores.copy_to_host(h_scores.data());
    p.copy_to_host(h_p.data());
    o.copy_to_host(h_o.data());

    // -------------------------------------------------------------------------
    // 6.4 按阶段与边界条件验证结果
    // 参考独立从 Q/K 计算分数；Softmax 用 double，再模拟 P 的 FP16 存储。
    // -------------------------------------------------------------------------
    bool score_ok=true,probability_ok=true,output_ok=true;
    double max_score_error=0,max_probability_error=0,max_output_error=0;
    std::vector<double> ref_scores(Sk),weights(Sk);
    std::vector<Half> ref_p(Sk);
    for (int i=0; i<Sq; ++i) {
      double row_max=-INFINITY;
      for (int j=0; j<Sk; ++j) {
        double acc=0;
        for (int f=0; f<d; ++f)
          acc += double(float(h_q[size_t(i)*d+f]))*float(h_k[size_t(j)*d+f]);
        ref_scores[j]=double(scale)*acc;
        const double actual=h_scores[size_t(i)*Sk+j];
        score_ok &= close_value(actual,ref_scores[j],1e-4,1e-4);
        max_score_error=std::max(max_score_error,std::abs(actual-ref_scores[j]));
        if (j<=i) row_max=std::max(row_max,ref_scores[j]);
      }
      double sum=0;
      for (int j=0; j<Sk; ++j) {
        weights[j]=j<=i?std::exp(ref_scores[j]-row_max):0;
        sum+=weights[j];
      }
      double actual_row_sum=0;
      for (int j=0; j<Sk; ++j) {
        ref_p[j]=Half(float(weights[j]/sum));
        const double actual=float(h_p[size_t(i)*Sk+j]);
        const double expected=float(ref_p[j]);
        probability_ok &= close_value(actual,expected,2e-5,2e-3);
        if (j>i) probability_ok &= actual==0;
        actual_row_sum+=actual;
        max_probability_error=std::max(max_probability_error,std::abs(actual-expected));
      }
      // P 已转为 FP16，行和不要求逐位等于 1。
      probability_ok &= std::abs(actual_row_sum-1.0)<=2e-3;
      for (int f=0; f<dv; ++f) {
        double end_to_end_ref=0,pv_ref=0;
        for (int j=0; j<Sk; ++j) {
          const double value=float(h_v[size_t(j)*dv+f]);
          end_to_end_ref+=double(float(ref_p[j]))*value;
          pv_ref+=double(float(h_p[size_t(i)*Sk+j]))*value;
        }
        const double actual=h_o[size_t(i)*dv+f];
        output_ok &= close_value(actual,pv_ref,1e-4,1e-4);  // 独立检查第二个 GEMM。
        output_ok &= close_value(actual,end_to_end_ref,2e-3,2e-3);
        max_output_error=std::max(max_output_error,std::abs(actual-end_to_end_ref));
      }
    }
    std::cout << "QK=" << (score_ok?"PASS":"FAIL") << " max_abs_error=" << max_score_error << '\n';
    std::cout << "softmax_fp16=" << (probability_ok?"PASS":"FAIL")
              << " max_abs_error=" << max_probability_error << '\n';
    std::cout << "PV_and_end_to_end=" << (output_ok?"PASS":"FAIL")
              << " max_abs_error=" << max_output_error << '\n';
    // 在线 Softmax、Correction 与融合接口由正文后续小节讲解，本文件不实现它们。
    return score_ok && probability_ok && output_ok ? 0 : 1;
  } catch (std::exception const& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 2;
  }
}
