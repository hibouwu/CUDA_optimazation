// NVFP4 Block-Scaled：CPU 量化、packed Payload/Scale 布局、GEMM 和两类误差检查。
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
#include "cutlass/util/host_tensor.h"

// 6.1 NVFP4 与 Example 72a 的类型配置

using ArchTag = cutlass::arch::Sm100;  // C++ 实现族；二进制目标仍为 sm_110a。
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;
constexpr int AlignmentA = 32, AlignmentB = 32;  // 元素数。
constexpr int AlignmentC = 8, AlignmentD = 8;  // 元素数。
using MmaTileShape = cute::Shape<cute::_256, cute::_256, cute::_256>;  // Collective M/N/K。
using ClusterShape = cute::Shape<cute::_2, cute::_4, cute::_1>;  // CTA 个数；int 维度在运行时指定。
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

using Payload=typename ElementA::DataType;
using Scale=typename ElementA::ScaleFactorType;
using PackedPayload=cutlass::HostTensor<Payload,cutlass::layout::PackedVectorLayout>;
using ScaleBuffer=cutlass::HostTensor<Scale,cutlass::layout::PackedVectorLayout>;
using ScaleConfig=typename CollectiveMainloop::Sm1xxBlkScaledConfig;
constexpr int V=ScaleConfig::SFVecSize;
static_assert(V==16, "This example quantizes dense NVFP4 blocks of 16 elements");

struct Quantization {
  float tensor_scale=1;
  std::vector<float> block_scales;  // 逻辑 (outer, k/V)，供独立参考读取。
  double mse=0;
};

// -----------------------------------------------------------------------------
// 6.1 从原始张量得到 Tensor Scale、Block Scale 和 Payload
// CPU 教学量化器：有限输入，沿 K 每 V 个元素共享 Scale。
// A 的 outer 是 m；ColumnMajor B 的 outer 是 n。
// -----------------------------------------------------------------------------
template<class ScaleLayout>
static Quantization quantize(
    std::vector<float> const& original,  // 原始浮点值，按 outer×K 保存。
    int outer, int K,                    // 外维长度和归约长度，K 必须整除 V。
    PackedPayload& packed,              // 子字节打包目标，不是 vector<FP4>。
    ScaleBuffer& scale_buffer,          // 按 Kernel 物理布局分配的 Scale 存储。
    ScaleLayout const& scale_layout) {  // 元素坐标 (outer,k,l) 到 Scale 位置。
  if (K%V || original.size()!=size_t(outer)*K)
    throw std::runtime_error("quantizer shape mismatch");
  Quantization q;
  float amax=0;
  for (float x:original) {
    if (!std::isfinite(x)) throw std::runtime_error("quantizer requires finite input");
    amax=std::max(amax,std::abs(x));
  }
  q.tensor_scale=amax==0 ? 1.0f : amax/(6.0f*448.0f);
  if (!(q.tensor_scale>0) || !std::isfinite(q.tensor_scale))
    throw std::runtime_error("tensor scale underflow or overflow");
  q.block_scales.resize(size_t(outer)*(K/V));
  for (int o=0; o<outer; ++o) {
    for (int b=0; b<K/V; ++b) {
      float block_amax=0;
      for (int u=0; u<V; ++u)
        block_amax=std::max(block_amax,
            std::abs(original[size_t(o)*K+b*V+u]/q.tensor_scale));
      // 全零块取 Scale=1；非零块使用实际舍入后的 UE4M3 Scale。
      const Scale stored_scale(block_amax==0 ? 1.0f : block_amax/6.0f);
      const float sf=float(stored_scale);
      if (!(sf>0) || !std::isfinite(sf))
        throw std::runtime_error("block scale underflow or overflow");
      q.block_scales[size_t(o)*(K/V)+b]=sf;
      scale_buffer.host_data(scale_layout(o,b*V,0))=stored_scale;
      for (int u=0; u<V; ++u) {
        const size_t index=size_t(o)*K+b*V+u;
        const Payload value(original[index]/q.tensor_scale/sf);
        // host_data(index) 返回支持 FP4 的引用；host_data()[index] 不等价。
        packed.host_data(index)=value;
        const double reconstructed=double(q.tensor_scale)*sf*float(value);
        const double error=reconstructed-original[index];
        q.mse += error*error;
      }
    }
  }
  q.mse /= original.size();
  return q;
}

int main() {
  try {
    check(cudaSetDevice(0));
    // -------------------------------------------------------------------------
    // 6.1 NVFP4 由哪些数据组成；Scale Layout 由配方与 ProblemShape 共同确定
    // -------------------------------------------------------------------------
    const int M=256, N=1024, K=256;
    auto problem=cute::make_shape(M,N,K,1);  // 单问题，L=1。
    using StrideA=typename GemmKernel::StrideA;
    using StrideB=typename GemmKernel::StrideB;
    using StrideC=typename GemmKernel::StrideC;
    using StrideD=typename GemmKernel::StrideD;
    auto stride_a=cutlass::make_cute_packed_stride(StrideA{},cute::make_shape(M,K,1));
    auto stride_b=cutlass::make_cute_packed_stride(StrideB{},cute::make_shape(N,K,1));
    auto stride_c=cutlass::make_cute_packed_stride(StrideC{},cute::make_shape(M,N,1));
    auto stride_d=cutlass::make_cute_packed_stride(StrideD{},cute::make_shape(M,N,1));
    auto layout_sfa=ScaleConfig::tile_atom_to_shape_SFA(problem);
    auto layout_sfb=ScaleConfig::tile_atom_to_shape_SFB(problem);
    const int64_t count_sfa=cute::size(cute::filter_zeros(layout_sfa));
    const int64_t count_sfb=cute::size(cute::filter_zeros(layout_sfb));

    PackedPayload a,b;
    ScaleBuffer sfa,sfb;
    a.reset(cutlass::make_Coord(int64_t(M)*K));   // 逻辑 FP4 元素数，内部按位宽分配。
    b.reset(cutlass::make_Coord(int64_t(N)*K));
    sfa.reset(cutlass::make_Coord(count_sfa));    // 保留 Kernel Layout 的物理填充。
    sfb.reset(cutlass::make_Coord(count_sfb));
    for (int64_t i=0; i<count_sfa; ++i) sfa.host_data(i)=Scale(1.0f);
    for (int64_t i=0; i<count_sfb; ++i) sfb.host_data(i)=Scale(1.0f);

    std::vector<float> original_a(size_t(M)*K),original_b(size_t(N)*K);
    for (int m=0; m<M; ++m)
      for (int k=0; k<K; ++k)
        original_a[size_t(m)*K+k]=m==0 ? 0.0f : float((7*m+11*k)%31-15)/8;
    for (int n=0; n<N; ++n)
      for (int k=0; k<K; ++k)
        original_b[size_t(n)*K+k]=float((5*n+3*k)%29-14)/8;
    auto qa=quantize(original_a,M,K,a,sfa,layout_sfa);
    auto qb=quantize(original_b,N,K,b,sfb,layout_sfb);
    a.sync_device();
    b.sync_device();
    sfa.sync_device();
    sfb.sync_device();

    std::vector<ElementC> h_c(size_t(M)*N);
    std::vector<ElementD> h_d(size_t(M)*N);
    for (int m=0; m<M; ++m)
      for (int n=0; n<N; ++n)
        h_c[size_t(m)*N+n]=ElementC(float((m+n)%3-1)/4);
    cutlass::DeviceAllocation<ElementC> d_c(h_c.size());
    cutlass::DeviceAllocation<ElementD> d_d(h_d.size());
    d_c.copy_from_host(h_c.data());

    // -------------------------------------------------------------------------
    // 6.1 在 GEMM 中分别放置两级 Scale；用四组输入构造 Arguments
    // -------------------------------------------------------------------------
    const float alpha_original=1.0f,beta=0.5f;
    const float alpha_effective=alpha_original*qa.tensor_scale*qb.tensor_scale;
    auto hw=get_hardware_info();
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,  // mode：普通 GEMM。
        problem,  // problem_shape：M/N/K/L。
        { // mainloop：两组 Payload 与两组 Block Scale。
          a.device_data(),   // A 的 packed FP4 设备地址。
          stride_a,          // A 的 (m,k,l) 步长。
          b.device_data(),   // B 的 packed FP4 设备地址。
          stride_b,          // B 的 (n,k,l) 步长。
          sfa.device_data(), // A 的 Block Scale 设备地址。
          layout_sfa,        // A 的元素坐标到 Scale 物理位置的映射。
          sfb.device_data(), // B 的 Block Scale 设备地址。
          layout_sfb         // B 的元素坐标到 Scale 物理位置的映射。
        },
        { // epilogue：Tensor Scale 只计入乘积系数。
          {alpha_effective,  // 已包含 g_A*g_B 的乘积系数。
           beta},            // C 的系数，不并入 g_A*g_B。
          d_c.get(),         // C 的设备地址。
          stride_c,          // C 的步长。
          d_d.get(),         // D 的设备地址。
          stride_d           // D 的步长。
        },
        hw  // 设备信息；Cluster 由静态类型确定。
    };
    run_gemm<Gemm>(args,nullptr);
    d_d.copy_to_host(h_d.data());

    // -------------------------------------------------------------------------
    // 6.1 反量化与误差比较
    // 参考从 packed Payload 解码，但按独立的逻辑数组读取 Scale。
    // Kernel 验证不把输入量化误差混入容差。
    // -------------------------------------------------------------------------
    std::vector<float> local_a(size_t(M)*K),local_b(size_t(N)*K);
    for (int m=0; m<M; ++m)
      for (int k=0; k<K; ++k) {
        const Payload value=a.host_data(size_t(m)*K+k);
        local_a[size_t(m)*K+k]=float(value)*qa.block_scales[size_t(m)*(K/V)+k/V];
      }
    for (int n=0; n<N; ++n)
      for (int k=0; k<K; ++k) {
        const Payload value=b.host_data(size_t(n)*K+k);
        local_b[size_t(n)*K+k]=float(value)*qb.block_scales[size_t(n)*(K/V)+k/V];
      }
    bool passed=true;
    double max_error=0,output_quantization_mse=0;
    for (int m=0; m<M; ++m) {
      for (int n=0; n<N; ++n) {
        double quantized_acc=0,original_acc=0;
        for (int k=0; k<K; ++k) {
          quantized_acc += double(local_a[size_t(m)*K+k])*local_b[size_t(n)*K+k];
          original_acc += double(original_a[size_t(m)*K+k])*original_b[size_t(n)*K+k];
        }
        const size_t index=size_t(m)*N+n;
        const double quantized_ref=double(alpha_effective)*quantized_acc+beta*float(h_c[index]);
        const double ref=float(ElementD(float(quantized_ref)));  // BF16 输出转换。
        const double actual=float(h_d[index]);
        passed &= close_value(actual,ref,1e-2,1e-2);
        max_error=std::max(max_error,std::abs(actual-ref));
        const double original_ref=alpha_original*original_acc+beta*float(h_c[index]);
        const double quant_error=quantized_ref-original_ref;
        output_quantization_mse += quant_error*quant_error;
      }
    }
    std::cout << "input_mse_a=" << qa.mse << " input_mse_b=" << qb.mse
              << " output_quantization_mse=" << output_quantization_mse/(double(M)*N) << '\n';
    std::cout << (passed?"PASS":"FAIL") << " max_abs_error=" << max_error << '\n';
    return passed?0:1;
  } catch (std::exception const& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 2;
  }
}
