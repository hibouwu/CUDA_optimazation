# 官方接口速查

固定提交 `8f50b052e1099fb982392a622caab69b97b63128`；库版本 4.6.0，采用其中的 3.x API 体系。以下仅为会议精选接口，不是完整 API 清单。

每项保留独立身份、原始完整声明、仓库相对路径和固定行号。在线原文仅作可选核对；离线源码和已有详细图无需联网。具体语义说明适用于会议包所列 Dense 配置。

<a id="ref-host"></a>

## Host 调用

<a id="api-host.can_implement"></a>

### can_implement(Arguments const&)

独立检查请求能否实现；initialize 不会自动调用它。

`include/cutlass/gemm/device/gemm_universal_adapter.h:231`

[离线源码](../site/source/include/cutlass/gemm/device/gemm_universal_adapter.h.html#L231) · [完整接口与独立关系](../site/index.html?api=host.can_implement) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/device/gemm_universal_adapter.h#L231)

限定名：`cutlass::gemm::device::GemmUniversalAdapter<
  GemmKernel_,
  cute::enable_if_t<gemm::detail::IsCutlass3GemmKernel<GetUnderlyingKernel_t<GemmKernel_>>::value>>::can_implement`

```cpp
static Status
  can_implement(Arguments const& args)
```

<a id="api-host.get_workspace_size"></a>

### get_workspace_size(Arguments const&)

返回所需字节数，不负责分配。查询、分配、初始化和回收分别核对。

`include/cutlass/gemm/device/gemm_universal_adapter.h:242`

[离线源码](../site/source/include/cutlass/gemm/device/gemm_universal_adapter.h.html#L242) · [完整接口与独立关系](../site/index.html?api=host.get_workspace_size) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/device/gemm_universal_adapter.h#L242)

限定名：`cutlass::gemm::device::GemmUniversalAdapter<
  GemmKernel_,
  cute::enable_if_t<gemm::detail::IsCutlass3GemmKernel<GetUnderlyingKernel_t<GemmKernel_>>::value>>::get_workspace_size`

```cpp
static size_t
  get_workspace_size(Arguments const& args)
```

<a id="api-host.initialize"></a>

### initialize(args, workspace, stream, cuda_adapter)

初始化 workspace 后重建 params_，按条件设置共享内存属性；不是执行 GEMM。

`include/cutlass/gemm/device/gemm_universal_adapter.h:312`

[离线源码](../site/source/include/cutlass/gemm/device/gemm_universal_adapter.h.html#L312) · [完整接口与独立关系](../site/index.html?api=host.initialize) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/device/gemm_universal_adapter.h#L312)

限定名：`cutlass::gemm::device::GemmUniversalAdapter<
  GemmKernel_,
  cute::enable_if_t<gemm::detail::IsCutlass3GemmKernel<GetUnderlyingKernel_t<GemmKernel_>>::value>>::initialize`

```cpp
Status
  initialize(
    Arguments const& args,
    void* workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr)
```

<a id="api-host.update"></a>

### update(args, workspace)

不调用 initialize_workspace，也不保证轻量更新。变更问题或存储之前要重新确认前提。

`include/cutlass/gemm/device/gemm_universal_adapter.h:359`

[离线源码](../site/source/include/cutlass/gemm/device/gemm_universal_adapter.h.html#L359) · [完整接口与独立关系](../site/index.html?api=host.update) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/device/gemm_universal_adapter.h#L359)

限定名：`cutlass::gemm::device::GemmUniversalAdapter<
  GemmKernel_,
  cute::enable_if_t<gemm::detail::IsCutlass3GemmKernel<GetUnderlyingKernel_t<GemmKernel_>>::value>>::update`

```cpp
Status
  update(Arguments const& args, void* workspace = nullptr)
```

<a id="api-host.run.stream"></a>

### run(stream, cuda_adapter, launch_with_pdl)

使用成员 params_ 的实例重载；与下面的静态 run 保持独立。

`include/cutlass/gemm/device/gemm_universal_adapter.h:610`

[离线源码](../site/source/include/cutlass/gemm/device/gemm_universal_adapter.h.html#L610) · [完整接口与独立关系](../site/index.html?api=host.run.stream) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/device/gemm_universal_adapter.h#L610)

限定名：`cutlass::gemm::device::GemmUniversalAdapter<
  GemmKernel_,
  cute::enable_if_t<gemm::detail::IsCutlass3GemmKernel<GetUnderlyingKernel_t<GemmKernel_>>::value>>::run`

```cpp
Status
  run(
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr,
    bool launch_with_pdl = false)
```

<a id="api-host.run.params"></a>

### static run(Params&, stream, cuda_adapter, launch_with_pdl)

提交设备任务并检查启动/API 状态；返回成功不能证明设备完成。

`include/cutlass/gemm/device/gemm_universal_adapter.h:374`

[离线源码](../site/source/include/cutlass/gemm/device/gemm_universal_adapter.h.html#L374) · [完整接口与独立关系](../site/index.html?api=host.run.params) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/device/gemm_universal_adapter.h#L374)

限定名：`cutlass::gemm::device::GemmUniversalAdapter<
  GemmKernel_,
  cute::enable_if_t<gemm::detail::IsCutlass3GemmKernel<GetUnderlyingKernel_t<GemmKernel_>>::value>>::run`

```cpp
static Status
  run(Params& params,
      cudaStream_t stream = nullptr,
      CudaHostAdapter *cuda_adapter = nullptr,
      bool launch_with_pdl = false)
```

<a id="api-host.cluster.launch"></a>

### ClusterLauncher::launch_with_fallback_cluster(...)

当前已选分支的 Cluster launch 包装。不要推广到其他架构、PDL 或 Host Adapter 分支。

`include/cutlass/cluster_launch.hpp:259`

[离线源码](../site/source/include/cutlass/cluster_launch.hpp.html#L259) · [完整接口与独立关系](../site/index.html?api=host.cluster.launch) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/cluster_launch.hpp#L259)

限定名：`cutlass::ClusterLauncher::launch_with_fallback_cluster`

```cpp
static inline CUTLASS_HOST
  Status launch_with_fallback_cluster(
      dim3 const grid_dims,
      dim3 const preferred_cluster_dims,
      dim3 const fallback_cluster_dims,
      dim3 const block_dims,
      size_t const smem_size,
      cudaStream_t cuda_stream,
      void const* kernel,
      void** kernel_params,
      bool launch_with_pdl = false)
```

<a id="api-host.device_kernel"></a>

### device_kernel<Operator>(Operator::Params const params)

CUDA 运行时调度的设备入口；入口内才调用设备 functor。

`include/cutlass/device_kernel.h:112`

[离线源码](../site/source/include/cutlass/device_kernel.h.html#L112) · [完整接口与独立关系](../site/index.html?api=host.device_kernel) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/device_kernel.h#L112)

限定名：`cutlass::device_kernel`

```cpp
template <typename Operator>
CUTLASS_GLOBAL
#ifdef __CUDACC__
// Enclosing this in __CUDACC__ suppresses MSVC warnings.
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
#endif // __CUDACC__
void device_kernel(CUTLASS_GRID_CONSTANT typename Operator::Params const params)
```

<a id="ref-build"></a>

## 编译期选择

<a id="api-types.builder_entry"></a>

### CollectiveBuilder 主模板入口

GEMM CollectiveBuilder 主模板入口；完整模板参数如下。没有适用实现的组合在编译期拒绝。

`include/cutlass/gemm/collective/collective_builder_decl.hpp:93`

[离线源码](../site/source/include/cutlass/gemm/collective/collective_builder_decl.hpp.html#L77) · [完整接口与独立关系](../site/index.html?api=types.builder_entry) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/collective_builder_decl.hpp#L93)

限定名：`cutlass::gemm::collective::CollectiveBuilder`

```cpp
template <
  class ArchTag,
  class OpClass,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType,
  class Enable = void
>
struct CollectiveBuilder
```

<a id="api-types.builder_sm100"></a>

### SM100 Dense UMMA Builder 偏特化

已选 SM100 偏特化；特化条件决定适用性，CollectiveOp 是产生的类型，不是运行时返回对象。

`include/cutlass/gemm/collective/builders/sm100_umma_builder.inl:169`

[离线源码](../site/source/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl.html#L155) · [完整接口与独立关系](../site/index.html?api=types.builder_sm100) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl#L169)

限定名：`cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,    // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    ClusterShape_MNK, // Static cluster shape or dynamic (int, int, _1)
    StageCountType,
    BuilderScheduleTag,
    cute::enable_if_t<
      (cute::is_same_v<ArchTag, arch::Sm100> 
      ) &&
      not cute::is_tuple_v<ElementA>   && not cute::is_tuple_v<ElementB> &&
      not cute::is_complex_v<ElementA> && not cute::is_complex_v<ElementB> &&
      // Dense Gemm / PtrArrayDenseGemm
      (
       (not cute::is_same_v<KernelMixedTmaCpAsyncWarpSpecialized1SmSm100, BuilderScheduleTag>) && 
       (not cute::is_same_v<KernelMixedTmaCpAsyncWarpSpecialized2SmSm100, BuilderScheduleTag>) && 
       (not cute::is_same_v<KernelWarpSpecialized1SmSm100, BuilderScheduleTag>) && 
       (cute::is_base_of_v<KernelScheduleSm100DenseGemm, BuilderScheduleTag> ||
        cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>)) &&
      // Alignment check
      detail::sm1xx_gemm_is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, BuilderScheduleTag>()>>`

```cpp
template <
  class ArchTag,
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class BuilderScheduleTag
>
struct CollectiveBuilder<
    ArchTag,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,    // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    ClusterShape_MNK, // Static cluster shape or dynamic (int, int, _1)
    StageCountType,
    BuilderScheduleTag,
    cute::enable_if_t<
      (cute::is_same_v<ArchTag, arch::Sm100> 
      ) &&
      not cute::is_tuple_v<ElementA>   && not cute::is_tuple_v<ElementB> &&
      not cute::is_complex_v<ElementA> && not cute::is_complex_v<ElementB> &&
      // Dense Gemm / PtrArrayDenseGemm
      (
       (not cute::is_same_v<KernelMixedTmaCpAsyncWarpSpecialized1SmSm100, BuilderScheduleTag>) && 
       (not cute::is_same_v<KernelMixedTmaCpAsyncWarpSpecialized2SmSm100, BuilderScheduleTag>) && 
       (not cute::is_same_v<KernelWarpSpecialized1SmSm100, BuilderScheduleTag>) && 
       (cute::is_base_of_v<KernelScheduleSm100DenseGemm, BuilderScheduleTag> ||
        cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>)) &&
      // Alignment check
      detail::sm1xx_gemm_is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, BuilderScheduleTag>()>>
```

<a id="api-types.atom_call"></a>

### MMA_Atom::call(D,A,B,C)

MMA_Atom 的一次操作接口。操作数表示必须符合具体 Atom/Traits，不能从 Fragment 名字直接推出存储位置。

`include/cute/atom/mma_atom.hpp:94`

[离线源码](../site/source/include/cute/atom/mma_atom.hpp.html#L88) · [完整接口与独立关系](../site/index.html?api=types.atom_call) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cute/atom/mma_atom.hpp#L94)

限定名：`cute::MMA_Atom<MMA_Traits<MMAOperation, Args...>>::call`

```cpp
template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr
  void
  call(Tensor<TD, DLayout>      & D,
       Tensor<TA, ALayout> const& A,
       Tensor<TB, BLayout> const& B,
       Tensor<TC, CLayout> const& C) const
```

<a id="api-types.fma"></a>

### SM100_MMA_F16BF16_2x1SM_SS::fma

当前类型选择最终使用的 2SM 硬件包装；其约束不自动适用于其他硬件。

`include/cute/arch/mma_sm100_umma.hpp:563`

[离线源码](../site/source/include/cute/arch/mma_sm100_umma.hpp.html#L562) · [完整接口与独立关系](../site/index.html?api=types.fma) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cute/arch/mma_sm100_umma.hpp#L563)

限定名：`cute::SM100_MMA_F16BF16_2x1SM_SS::fma`

```cpp
CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
```

<a id="ref-handoff"></a>

## 参数和职责交接

<a id="api-host.kernel.arguments"></a>

### GemmKernel::Arguments

面向调用方的参数，包括 mode、problem_shape、mainloop、epilogue、hw_info、scheduler。

`include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:215`

[离线源码](../site/source/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp.html#L215) · [完整接口与独立关系](../site/index.html?api=host.kernel.arguments) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L215)

限定名：`cutlass::gemm::kernel::GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag_,
  cute::enable_if_t<
    cute::disjunction_v<cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelTmaWarpSpecializedSm100>,
    cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelTmaWarpSpecializedBlockScaledSm100>>>>::Arguments`

```cpp
struct Arguments
```

字段定义（固定源码）：

```cpp
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };
```

<a id="api-host.kernel.params"></a>

### GemmKernel::Params

设备所用参数，嵌套字段已转换为各组件的 Params；不是 Arguments 的别名。

`include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:225`

[离线源码](../site/source/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp.html#L225) · [完整接口与独立关系](../site/index.html?api=host.kernel.params) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L225)

限定名：`cutlass::gemm::kernel::GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag_,
  cute::enable_if_t<
    cute::disjunction_v<cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelTmaWarpSpecializedSm100>,
    cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelTmaWarpSpecializedBlockScaledSm100>>>>::Params`

```cpp
struct Params
```

字段定义（固定源码）：

```cpp
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    TileSchedulerParams scheduler{};
    KernelHardwareInfo hw_info{}; 
  };
```

<a id="api-host.kernel.initialize_workspace"></a>

### GemmKernel::initialize_workspace(args, workspace, stream, cuda_adapter)

Kernel 将 workspace 初始化交给具体组件；与下面的参数转换操作不同。

`include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:354`

[离线源码](../site/source/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp.html#L354) · [完整接口与独立关系](../site/index.html?api=host.kernel.initialize_workspace) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L354)

限定名：`cutlass::gemm::kernel::GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag_,
  cute::enable_if_t<
    cute::disjunction_v<cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelTmaWarpSpecializedSm100>,
    cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelTmaWarpSpecializedBlockScaledSm100>>>>::initialize_workspace`

```cpp
static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr)
```

<a id="api-host.kernel.lower"></a>

### GemmKernel::to_underlying_arguments(args, workspace)

划分并对齐 workspace，分别调用三个转换接口；当前实现的 mainloop_workspace 为 nullptr。

`include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:255`

[离线源码](../site/source/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp.html#L255) · [完整接口与独立关系](../site/index.html?api=host.kernel.lower) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L255)

限定名：`cutlass::gemm::kernel::GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag_,
  cute::enable_if_t<
    cute::disjunction_v<cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelTmaWarpSpecializedSm100>,
    cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelTmaWarpSpecializedBlockScaledSm100>>>>::to_underlying_arguments`

```cpp
static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace)
```

<a id="api-host.mainloop.lower"></a>

### CollectiveMainloop::to_underlying_arguments(problem_shape, args, workspace, hw_info)

转换 Mainloop 输入；TMA 描述符内部构造尚未由本会议图完整展开。

`include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp:353`

[离线源码](../site/source/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp.html#L353) · [完整接口与独立关系](../site/index.html?api=host.mainloop.lower) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp#L353)

限定名：`cutlass::gemm::collective::CollectiveMma<
    MainloopSm100TmaUmmaWarpSpecialized<
      Stages,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape,
      ArchTag_>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>::to_underlying_arguments`

```cpp
template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
    ProblemShape const& problem_shape,
    Arguments const& args,
    [[maybe_unused]] void* workspace,
    cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{})
```

<a id="api-host.epilogue.lower"></a>

### CollectiveEpilogue::to_underlying_arguments(problem_shape, args, workspace)

转换 Epilogue 输入；C/D、融合参数和空间依赖需要独立核对。

`include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp:299`

[离线源码](../site/source/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp.html#L299) · [完整接口与独立关系](../site/index.html?api=host.epilogue.lower) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp#L299)

限定名：`cutlass::epilogue::collective::CollectiveEpilogue<
    Sm100TmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_>,
    CtaTileShape_,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    FusionCallbacks_,
    CopyOpT2R_,
    CopyOpG2S_,
    SmemLayoutAtomC_,
    CopyOpS2R_,
    CopyOpS2G_,
    SmemLayoutAtomD_,
    CopyOpR2S_,
    CopyOpR2R_
>::to_underlying_arguments`

```cpp
template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace)
```

<a id="api-host.scheduler.lower"></a>

### TileScheduler::to_underlying_arguments(problem_shape, tile, atom_thr, cluster, hw_info, args, workspace)

转换工作分配参数；它不等于 Mainloop Schedule，也不是 MMA Atom 选择器。

`include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp:135`

[离线源码](../site/source/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp.html#L135) · [完整接口与独立关系](../site/index.html?api=host.scheduler.lower) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp#L135)

限定名：`cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100::to_underlying_arguments`

```cpp
template <class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShape>
  static Params
  to_underlying_arguments(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape_mnk,
      AtomThrShape atom_thr_shape_mnk,
      ClusterShape cluster_shape_mnk,
      KernelHardwareInfo const& hw_info,
      Arguments const& args,
      void* workspace = nullptr
    )
```

<a id="api-host.kernel.operator"></a>

### GemmKernel::operator()(Params const&, char*)

设备 functor 收到 Params 与共享内存地址，组织角色和各组件执行。

`include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:403`

[离线源码](../site/source/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp.html#L403) · [完整接口与独立关系](../site/index.html?api=host.kernel.operator) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L403)

限定名：`cutlass::gemm::kernel::GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag_,
  cute::enable_if_t<
    cute::disjunction_v<cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelTmaWarpSpecializedSm100>,
    cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelTmaWarpSpecializedBlockScaledSm100>>>>::operator()`

```cpp
CUTLASS_DEVICE
  void
  operator() (Params const& params, char* smem_buf)
```

<a id="ref-sync"></a>

## 同步与资源使用

<a id="api-contract.api.input_acquire"></a>

### PipelineTmaUmmaAsync::producer_acquire

取得当前输入 stage 的可写条件；参与角色、stage/phase 与事务计数必须一致。

`include/cutlass/pipeline/sm100_pipeline.hpp:684`

[离线源码](../site/source/include/cutlass/pipeline/sm100_pipeline.hpp.html#L683) · [完整接口与独立关系](../site/index.html?api=contract.api.input_acquire) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/pipeline/sm100_pipeline.hpp#L684)

限定名：`cutlass::PipelineTmaUmmaAsync::producer_acquire`

```cpp
CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain})
```

<a id="api-contract.api.input_wait"></a>

### PipelineTmaUmmaAsync::consumer_wait

等待当前输入 stage 可读；发起搬运与完成搬运是不同事件。

`include/cutlass/pipeline/sm100_pipeline.hpp:720`

[离线源码](../site/source/include/cutlass/pipeline/sm100_pipeline.hpp.html#L719) · [完整接口与独立关系](../site/index.html?api=contract.api.input_wait) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/pipeline/sm100_pipeline.hpp#L720)

限定名：`cutlass::PipelineTmaUmmaAsync::consumer_wait`

```cpp
CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain})
```

<a id="api-contract.api.input_release"></a>

### PipelineTmaUmmaAsync::consumer_release

把相关 MMA 完成和 empty 通知联系起来；发出通知不等于下一轮 acquire 已成功。

`include/cutlass/pipeline/sm100_pipeline.hpp:725`

[离线源码](../site/source/include/cutlass/pipeline/sm100_pipeline.hpp.html#L724) · [完整接口与独立关系](../site/index.html?api=contract.api.input_release) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/pipeline/sm100_pipeline.hpp#L725)

限定名：`cutlass::PipelineTmaUmmaAsync::consumer_release`

```cpp
CUTLASS_DEVICE
  void consumer_release(PipelineState state)
```

<a id="api-contract.api.store_wait"></a>

### cute::tma_store_wait<Count>

实际 PTX 是 wait_group.read；源 SMEM 读完不等于全局 D 写完。

`include/cute/arch/copy_sm90_tma.hpp:1248`

[离线源码](../site/source/include/cute/arch/copy_sm90_tma.hpp.html#L1246) · [完整接口与独立关系](../site/index.html?api=contract.api.store_wait) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cute/arch/copy_sm90_tma.hpp#L1248)

限定名：`cute::tma_store_wait`

```cpp
template <int Count>
CUTE_HOST_DEVICE static void
tma_store_wait()
```

<a id="api-contract.api.allocate"></a>

### TMEM::Allocator2Sm::allocate

双 CTA TMEM 分配有参与 warp 与配对条件；不能把它当作任意线程均可调用的通用分配器。

`include/cute/arch/tmem_allocator_sm100.hpp:135`

[离线源码](../site/source/include/cute/arch/tmem_allocator_sm100.hpp.html#L134) · [完整接口与独立关系](../site/index.html?api=contract.api.allocate) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cute/arch/tmem_allocator_sm100.hpp#L135)

限定名：`cute::TMEM::Allocator2Sm::allocate`

```cpp
CUTE_HOST_DEVICE void
  allocate(int num_columns, uint32_t* dst_ptr)
```

<a id="api-contract.api.free"></a>

### TMEM::Allocator2Sm::free

释放前要满足拥有者与协作完成条件；单侧 free 发出不等于配对释放完成。

`include/cute/arch/tmem_allocator_sm100.hpp:159`

[离线源码](../site/source/include/cute/arch/tmem_allocator_sm100.hpp.html#L157) · [完整接口与独立关系](../site/index.html?api=contract.api.free) · [固定提交原文（联网）](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cute/arch/tmem_allocator_sm100.hpp#L159)

限定名：`cute::TMEM::Allocator2Sm::free`

```cpp
__device__
  void
  free(uint32_t tmem_ptr, int num_columns)
```

## 同步接口之外还要查什么

`Allocator2Sm::release_allocation_lock()` 只是放弃分配许可，不是释放已分配的 TMEM。其声明见 [固定源码第 174 行](../site/source/include/cute/arch/tmem_allocator_sm100.hpp.html#L174)。Kernel 在调用 `free()` 前的独立等待、双 CTA 握手与条件分支见 [第 783 行起](../site/source/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp.html#L783) 和 [TMEM 协议详细图](../site/index.html?protocol=dense.protocol_draft.tmem_lifetime)。

`cudaStreamSynchronize(stream)` 是本参考应用的显式等待，不在 Adapter 内部。见 [已保存的示例调用位置](../site/evidence/6c1d941eb5a1b64f-dense_baseline.cu.html#L250)。其他实现应说明自己的等价完成条件；不能只检查执行接口的返回值。

本包引用已发布的静态编译材料，但没有新增 GPU 执行。完整 Copy 分派、全部描述符/调度/融合行为和其他配置尚不能由这些接口证明。未展开事项在会议记录中另立问题，不由简图补猜。
