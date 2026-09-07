**CUTLASS 3.x (2)：Thor GEMM 的 Collective、Kernel 与 Device 执行过程**

[第一篇](01-thor-gemm-programming-guide_zh-CN.md)已经从矩阵问题出发，构造并调用一份 Dense GEMM。本篇继续展开同一份 Kernel：先说明两个 Collective 怎样组织局部计算，再说明 Kernel 怎样安排协作线程和领取工作，最后回到 Device 接口，观察一次调用的参数与资源如何进入设备执行。[第三篇](03-cutlass-principled-abstractions_zh-CN.md)再展开 Layout、Tensor、Atom、TiledMMA 和 Fragment 的空间构造。

# 1. 沿用第一篇的 Dense GEMM 基线

## 1.1 本篇解释哪一个 Kernel

本篇沿用 [dense_baseline.cu](exemples/dense_baseline.cu) 的计算：

$$D=\alpha AB+\beta C$$

A/B 使用 FP16，Accumulator、Epilogue 计算以及 C/D 使用 FP32，四个矩阵均按 RowMajor 保存。编译期的 Collective Tile 为 `(256,128,64)`，Cluster 为 `(2,2,1)`；Mainloop 与 Epilogue Schedule 使用 Auto，Mainloop Stage 通过 Epilogue 的共享内存需求推导。运行时基线为 `(M,N,K,L)=(256,256,128,1)`，`alpha=1`、`beta=0.5`。

> 阅读基线：源码统一固定到 CUTLASS `8f50b052e1099fb982392a622caab69b97b63128`。C++ 的 `cutlass::arch::Sm100` 选择 Blackwell 实现族，二进制目标为 `compute_110a/sm_110a`。本文的具体类型与资源常量已通过该配置的编译期实例化核对；这些证据用于解释执行机制，不代替 Thor 上的启动、数值验证或性能测量。

这组配置选中 Dense TMA、2SM MMA、共享内存操作数和 TMEM 累加路径。一个完整的 `256×128` Collective 输出区域由一对 CTA 协作完成，每个 CTA 对应 `128×128` 输出位置。Cluster 包含四个 CTA，因而既要区分完整 Collective Tile 与 CTA 工作坐标，也要区分 MMA 的 CTA pair 与整个 Cluster。后面的工作领取以 CTA 坐标描述，数值归约则按协作范围解释。

第一篇的基线尺寸足以验证一次矩阵乘加，但可分配的工作很少。讨论 Persistent 的连续工作领取时，可以增大运行时 M/N，让同一个 Kernel 类型面对更多输出 Tile；Element、Layout、Tile、Cluster 和数值操作保持不变。静态 Persistent 和 K 维拆分会在基础路径讲完后作为对照引入，贯穿实例仍使用默认的 CLC 调度。

## 1.2 编译期类型与运行时执行的关系

编译期首先确定一组能够协同工作的类型。Epilogue Builder 给出结果处理方式和共享内存需求，Mainloop Builder 据此构造加载与乘加组件，`GemmUniversal` 再把它们与 Tile Scheduler 组合为设备 Kernel。这里确定的是操作形式、空间尺寸、缓冲级数和协作协议；本次矩阵的实际地址、尺寸和标量随后由 `Arguments` 提供。

运行时，主机把 `Arguments` 转换为 `Params` 并启动 Kernel。设备线程按角色分别调用加载、MMA、Epilogue 和工作领取接口，利用 Pipeline 中的完成状态保持数据依赖。因此，本篇按“Collective → Kernel → Device”解释组件的组成关系，而实际调用从 Device 进入 Kernel。两个方向描述同一份程序的不同侧面。

本篇把 TiledMMA、TiledCopy 和 Layout 作为已经形成的空间对象使用，追踪它们指向的数据、占用的资源和参与的执行阶段。它们怎样由布局代数、切片和参与者映射构造，留到第三篇。先建立这个边界，就可以从下一节的具体类型进入运行过程，而不必在每个 Layout 表达式处中断主线。

# 2. Collective 怎样完成局部计算

Kernel 为当前工作确定输出坐标和 K 范围后，Mainloop 负责形成相应的累加结果，Epilogue 负责消费这些结果并写回 D。两者分别提供加载与计算接口，实际由哪些线程调用、如何取得下一项工作，在第三部分统一说明。这里先跟踪一份输入 Stage 和一个输出 Subtile 的数据与状态。

## 2.1 从 Builder 配置得到具体 Collective

### 展开 Dense Mainloop 与 Epilogue 类型

第一篇已经逐项解释两个 Builder 的输入，本节从它们的输出继续。`CollectiveMainloop`、`CollectiveEpilogue` 是文章定义的别名，`CollectiveMma`、`CollectiveEpilogue` 则是 CUTLASS 中承载具体实现的模板。`::CollectiveOp` 将输入配置落实为其中一个具体类型。

下面的图表达“输入配置 → Builder 偏特化 → 局部组件 → Collective”的关系。图中下方多组 Policy 表示候选组合，用于说明分派结构；本篇 Dense 基线的具体常量以随后展开的类型为准。

![Mainloop CollectiveBuilder 的编译期分派策略选择与类型生成路径](Imgaes/cutlass-3-gemm-abstractions/whiteboards/03-collective-builder/feishu-latest.jpg)

Mainloop Builder 根据 `ArchTag`、`OpClass`、A/B 表示、布局、对齐和 Schedule 匹配 [SM100 Dense Builder](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl#L167-L245)。当前 M Tile 为 256，Cluster 的 M 方向允许 CTA pair，Auto 因而选中 2SM 的 TiledMMA。A 的 MMA Major 为 K，B 的 MMA Major 为 MN，这与本例两张输入均为 RowMajor 并不矛盾：A 按 `(m,k,l)`、B 按 `(n,k,l)` 接收坐标，各自的连续维不同。Major 和 Layout 的内部转换由第三篇展开。

在这份配置中，A 的加载类型为 `SM100_TMA_2SM_LOAD_MULTICAST`，B 为 `SM100_TMA_2SM_LOAD`。Builder 继续选择共享内存布局与流水线，再把结果写入 [CollectiveMma 类型](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl#L302-L345)。下面是本例的编译期类型展开，沿用第一篇已有的别名与配置：

**展开当前 Dense 配置的两个 DispatchPolicy**

```cpp
using DenseMainloopPolicy =
    cutlass::gemm::MainloopSm100TmaUmmaWarpSpecialized<
        8,             // A/B 共享内存 Pipeline 的 Stage 数。
        2,             // Scheduler Pipeline 的 Stage 数。
        4,             // Accumulator Pipeline 的 Stage 数。
        ClusterShape,  // (2,2,1)，以 CTA 个数计。
        ArchTag        // cutlass::arch::Sm100。
    >;

using DenseEpiloguePolicy =
    cutlass::epilogue::Sm100TmaWarpSpecialized<
        4,      // StagesC：C Load 的缓冲级数。
        2,      // StagesD：D Store 的在途深度参数。
        16,     // FragmentSize：一次 visit 处理的元素数。
        true,   // ReuseSmemC：C/D 复用共享存储。
        false   // DelayTmaStore：本例不延后一轮发射 Store。
    >;

static_assert(cute::is_same_v<
    DenseMainloopPolicy, typename CollectiveMainloop::DispatchPolicy>);
static_assert(cute::is_same_v<
    DenseEpiloguePolicy, typename CollectiveEpilogue::DispatchPolicy>);
```

这些常量由 Builder 产生，修改类型、Tile 或 Epilogue 后应从新生成的类型重新读取。Mainloop 的 [Policy 声明](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/dispatch_policy.hpp#L1021-L1035)把 Stage、Cluster 和架构写入算法类型；Epilogue 的 [Policy 声明](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/dispatch_policy.hpp#L217-L232)则约束结果处理的分块与缓冲。至此，两个 Collective 的执行形式已经确定，接下来需要解释这些选择怎样产生。

### DispatchPolicy 怎样确定执行方式

Mainloop Builder 的 `KernelScheduleType` 是用户提供的选择条件，生成的 `DispatchPolicy::Schedule` 则是内部执行类型。本例最终进入 [`MainloopSm100TmaUmmaWarpSpecialized` 对应的 CollectiveMma 偏特化](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp#L58-L123)。这一偏特化接收已经选好的 TiledMMA、TMA Copy、SMEM Layout、Element 与 Stride，再据此提供 `load_init`、`load`、`mma_init`、`mma` 和 `load_tail` 等运行时接口。

编译器在类体中继续检查组合条件：Collective Tile 与 TiledMMA 的覆盖关系必须一致，SMEM Layout 必须能表示对应操作数，2SM MMA 要匹配支持 CTA pair 的加载形式。基础 FP16 SS 路径将 `SmemCopyAtomA/B` 设为 `void`，`TransformA/B` 采用 `cute::identity`；A/B 留在共享内存，由 MMA 的操作数表示定位。由此形成的局部计算过程不需要额外的输入量化或经典的 SMEM→RMEM 操作数复制。

两个 Builder 的声明顺序来自资源依赖。`sizeof(CollectiveEpilogue::SharedStorage)` 在本例中为 33792 字节，Mainloop 的 `StageCountAutoCarveout` 使用它预留 Epilogue 空间。Builder 还扣除 Kernel 级同步和调度资源，再按单个 A/B Stage 的开销计算可容纳的级数。下面的公式描述 [Stage 推导函数](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl#L77-L102)的计算关系：

$$B_{\mathrm{stage}}
=B_A+B_B+B_{\mathrm{pipeline},1},\qquad
N_{\mathrm{stage}}=
\left\lfloor
\frac{B_{\mathrm{capacity,reduced}}-B_{\mathrm{epilogue}}}
     {B_{\mathrm{stage}}}
\right\rfloor$$

这里的 $B_A/B_B$ 是一个 CTA 实际分配的单 Stage 输入存储，来自生成的局部 SMEM Shape。对于 2SM 路径，它们不能直接用完整 Collective 的 M/N 尺寸代入估算。当前每 Stage 的 A 为 8192 个 FP16、B 为 4096 个 FP16，分别占 16384 和 8192 字节；加上该 Stage 的 Pipeline 存储后，当前预算允许八级。具体容量和附加预留由 [Dense Builder 的资源推导](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl#L245-L300)确定。

### 区分三类 Stage

同一个 Kernel 中有三种主要的循环复用资源，它们的 Stage 数描述不同对象：

- **A/B Stage：**保存某一段 K 输入及其同步状态，连接 MainloopLoad 与 MMA。本例为八级，由共享内存预算推导。
- **Accumulator Stage：**保存某项工作的累加结果，连接 MMA 与 Epilogue。本例为四级，由 TMEM 容量、CTA 输出尺寸和 Builder 的级数上限共同确定。
- **Scheduler Stage：**保存工作领取的响应及其同步状态，连接 Scheduler 与工作描述的消费者。本例为两级，由当前 Dense 分支的 Builder 规则确定。

[Builder](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl#L259-L285)把三种数量放入同一 Policy，但它们在运行时分别推进。一个 A/B Stage 被释放，只表明那段输入可以覆盖；一个 Accumulator Stage 被释放，表明结果消费者已经读完；Scheduler Stage 被释放，则表示各参与者已经取得工作描述。后面分别跟踪这些资源的交接。

## 2.2 Mainloop：沿 K 维加载与累加

### 当前工作对应哪些输入与 K 范围

Kernel 将本次 `Params`、CTA 坐标和 K 迭代范围交给 Mainloop。Mainloop 的用户参数只包含本次 A/B 地址及 Stride；[`to_underlying_arguments`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp#L251-L403)将这些实际值与已经确定的类型组合，构造本次运行使用的 TMA 对象。Builder 决定加载形式，参数转换把矩阵地址和尺寸写入对应的运行描述。

设备端 [`load_init`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp#L479-L541)建立全局输入与共享内存目标的分区，`load` 再按工作坐标选择实际的 A/B 区域。A 的选择依赖 M 坐标，B 的选择依赖 N 坐标，二者沿同一 K 迭代器前进。CTA pair 的成员保持相应的分工，完整的 `256×128` 输出区域沿 K 归约；每个 CTA 对应 `128×128` 输出。

当前 $K=128$、$T_K=64$，所以一次工作需要两次 K Tile 迭代。实际 MMA Atom 的 Shape 为 `(256,128,16)`，一轮 K Tile 内再依次进行四次 K Block 乘加。这里的两轮 K Tile、每轮四个 K Block 和八个物理 A/B Stage 分别描述工作量、基本乘加粒度和缓冲容量。工作量不足八轮时，部分缓冲不被本项工作使用，Stage 数仍是类型中固定的八。

### A/B 缓冲与 Pipeline 的组织

[`SmemLayoutA/B`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp#L172-L194)在局部操作数形状上追加 PIPE 维，使同一空间划分对应多份可复用的输入存储。下面的图先表达这些静态存储怎样与运行时游标、Barrier 联系；Producer 和 Consumer 的实际执行可以重叠。

![TiledMma、SMEM Layout 与 Pipeline State 共同构成多阶段 A/B SMEM Pipeline](Imgaes/cutlass-3-gemm-abstractions/whiteboards/02-mainloop/feishu-latest.jpg)

A/B 数值保存在共享内存中的 `TensorStorage::smem_A/smem_B`，每个物理 Stage 有对应的 Full 与 Empty Barrier。`PipelineStorage` 保存共享 Barrier，Producer 和 Consumer 各自的 `PipelineState` 保存本地游标，两者不是同一个对象。Kernel 将这些共享资源放进实际分配的 SMEM，参与线程分别构造访问它们的轻量接口对象。

下面根据 [Mainloop 的类型与存储声明](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp#L151-L160)整理接口，并省略不影响本段的数据布局构造。输入为已经确定的 Stage、Cluster 和 CTA pair 类型，输出是后续双方共用的同步协议类型。

**定义 A/B Pipeline 与双方的本地状态**

```cpp
using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
    DispatchPolicy::Stages,  // 本例为 8；保护 A/B 的循环缓冲。
    ClusterShape,           // 本例为 (2,2,1)。
    AtomThrShapeMNK          // 本例为 (2,1,1)，表示 MMA 的 CTA pair。
>;
using MainloopPipelineState = typename MainloopPipeline::PipelineState;

// 以下为 Kernel 中两个角色各自持有的局部状态。
MainloopPipelineState consumer_state{};
MainloopPipelineState producer_state =
    cutlass::make_producer_start_state<MainloopPipeline>();
```

`PipelineState` 中的 `index` 选择物理 Stage，`phase` 区分同一 Stage 的不同复用代次，`count` 记录逻辑推进次数。索引从末级回到零时，phase 翻转；Producer 的起始 phase 由 `make_producer_start_state` 调整，使首次 acquire 能把初始空缓冲与后续复用区分开。真正决定可读、可写的仍是 Barrier 的完成状态，本地游标只确定此次等待哪一代状态。[状态推进实现](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/pipeline/sm90_pipeline.hpp#L156-L242)给出了 index、phase 与 count 的更新关系。

### Producer 与 Consumer 怎样推进

MainloopLoad Warp 作为 Producer 取得下一份可写 Stage。`producer_try_acquire` 返回等待状态的 token，`producer_acquire` 根据该 token 完成必要的等待；这里的 token 用于避免重复等待，并不表示可以忽略尚未满足的 Empty 条件。获得 Stage 后，Producer 取得它的 Transaction Barrier，把 A/B 的 TMA 操作与该 Barrier 绑定。

下面是根据 [`load`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp#L594-L624)整理的单次加载节选。输入 View 和多播掩码已经由 `load_init` 建立；片段只放大“锁定 Stage、取得 Barrier、发出当前 K Tile 搬运”的关系。

**将当前 A/B 搬运绑定到可写 Stage**

```cpp
mainloop_pipeline.producer_acquire(
    producer_state, barrier_token);  // 等待当前 Stage 的 Empty 条件。
auto* barrier = mainloop_pipeline.producer_get_barrier(producer_state);
int write_stage = producer_state.index();

if (cute::elect_one_sync()) {
  copy(tma_a.with(*barrier, mcast_mask_a),  // A 的 TMA 对象、完成 Barrier 与多播范围。
       global_a(_, *k_tile_iter),         // 当前 K Tile 的全局源。
       shared_a(_, write_stage));         // 当前共享内存 Stage。
  copy(tma_b.with(*barrier, mcast_mask_b),
       global_b(_, *k_tile_iter),
       shared_b(_, write_stage));
}
// 省略：下一 Stage 的提前探测及循环游标更新。
```

源码中的局部 View 名称较长，上面用 `global_a/shared_a` 等别名表示相同角色。Producer 在发出异步请求后可以准备后续 Stage；当前数据何时可读，由 TMA 完成事件决定。Kernel 将 `TmaTransactionBytes` 写入 Pipeline 参数，本例为 49152 字节，覆盖 CTA pair 的 A/B 搬运约定。该数量与单个 CTA 的 24576 字节输入 Stage 不同，来源中的 CTA 协作因子必须保留。[事务字节数](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp#L232-L239)和 [Pipeline 参数绑定](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L455-L474)共同规定这一完成条件。

MMA Warp 作为 Consumer 等待 Full Barrier，确认当前代次的 A/B 已就绪后，沿 Stage 内的四个 K Block 发出 `cute::gemm`，由对应 Atom 调用 TCGen05 MMA。本例的 Atom 使用 [`SM100_MMA_F16BF16_2x1SM_SS`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cute/arch/mma_sm100_umma.hpp#L548-L590)，将共享内存操作数与 TMEM 累加地址交给 `tcgen05.mma.cta_group::2.kind::f16`；Kernel 由 CTA pair 的 leader 进入实际乘加分支。第一次基本乘加使用 `UMMA::ScaleOut::Zero` 建立累加结果，后续使用 `One` 保留旧累加值。这里的 Zero/One 控制是否累加已有结果，与量化的 Tensor Scale 或 Block Scale 无关。

MMA 指令是异步操作，Consumer 在源码中走到 `consumer_release`，并不表示可以立即覆盖输入。该函数通过 [`umma_arrive_multicast_2x1SM`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/pipeline/sm100_pipeline.hpp#L725-L756)把前序 MMA 完成与 Empty Barrier 通知联系起来。Producer 只有在相应完成状态满足后，才能再次 acquire 同一 Stage。输入生命周期因而包含两条不同的硬件完成边：

```text
Empty → Producer 取得 Stage → TMA 写入
                              │ TMA 完成预期事务
                              ▼
                            Full → MMA 读取并计算
                                      │ 前序 MMA 完成后通知 Empty
                                      └────────────────────→ Stage 可复用
```

当工作很多时，Producer 可以加载未来 K Tile，Consumer 同时计算已经就绪的 Tile；首个 Full 一旦就绪即可开始消费，不要求先填满所有 Stage。停止发出新的输入后，[`load_tail`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp#L627-L638)等待已使用的 Stage 被释放，保护 Cluster 中仍可能访问这些共享资源的参与者。PipelineState 可跨工作项继续推进，切换到下一输出 Tile 时沿用 Kernel 返回和保存的状态。

### 累加结果何时可以交给 Epilogue

A/B Pipeline 保护输入，Accumulator Pipeline 保护结果。MMA 在向某个 TMEM Stage 写入之前，先取得该结果 Stage 的 Producer 权限；当本项工作的所有 K Block 都已发出后，Kernel 调用 `accumulator_pipeline.producer_commit`。这条提交通过 MMA 完成通知建立结果 Ready 条件，Epilogue 等待该条件后才能读取 TMEM，而不是仅以主线程已经返回 `mma` 为依据。

本例的 Accumulator Pipeline 使用四级缓冲，与两 CTA 的 MMA 协作范围绑定：

**声明 MMA 与 Epilogue 之间的结果交接协议**

```cpp
using AccumulatorPipeline = cutlass::PipelineUmmaAsync<
    4,                // 本例的 Accumulator Stage 数。
    AtomThrShapeMNK   // 两 CTA 共同参与结果生产与消费。
>;
```

结果交接由 [Kernel 的 MMA 分支](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L750-L779)和 [Pipeline 的完成通知](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/pipeline/sm100_pipeline.hpp#L204-L275)共同实现。A/B Stage 的释放与 Accumulator Stage 的提交保护不同资源，不能用其中一条完成状态替代另一条。

至此，当前输出区域的归约结果具有明确的消费条件。MMA 可以在其他可用结果 Stage 上继续工作；当前 Stage 则留给 Epilogue，直到它完成必要的 TMEM 读取。下一节从这份 Ready 的累加结果继续。


## 2.3 Epilogue：消费累加结果并写回输出

### Accumulator、C 与 D 的关系

Epilogue 接收当前工作的 TMEM 累加结果、源矩阵 C 和线性组合参数。在基线中，C 通过 `beta=0.5` 参与计算，EpilogueLoad 因而需要读取 C；Accumulator 则由 MMA 产生，与 C 是两条独立输入。Epilogue 将二者汇合为 `alpha * Acc + beta * C`，最后以 FP32 写回 D。

[CollectiveEpilogue 的偏特化](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp#L64-L127)由已生成的 Policy、CTA Tile、EpilogueTile、C/D 类型与步长、FusionCallbacks 和各段 Copy 操作组成。C/D 地址及步长进入 Epilogue 的 `Arguments`，其中的 `thread` 保存融合计算参数；[参数转换](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp#L226-L318)产生本次运行所需的融合参数和 C/D TMA 对象。

下面的图表达三路输入、计算与写回的依赖关系。它是整个 Epilogue 的结构图；当前 Dense 路径只使用 Accumulator、C、alpha/beta，不引入额外 Aux 输入或归约。

![SM100 CollectiveEpilogue 的编译期类型与运行时数据流](Imgaes/cutlass-3-gemm-abstractions/whiteboards/04-epilogue/feishu-latest.jpg)

Kernel 为 EpilogueLoad 分配一个 Warp，为结果处理分配四个 Warp，即 `CollectiveEpilogue::ThreadCount=128`。前者调用 `load` 准备 C，后者调用 `store` 消费 C 与 Accumulator。它们使用相同的输出工作坐标，但由各自的 PipelineState 推进加载和写回，因此 C 的预取不必等到 MMA 全部完成后才开始。

### Epilogue Subtile 与数据搬运

本例每个 CTA 对应 `128×128` 输出区域，生成的 `EpilogueTile` 为 `128×16`，因此一次 Epilogue 沿 N 分成八个 Subtile。这里按 M/N 切分已经完成归约的输出；Mainloop 中沿 K 的四次基本乘加则是在构造累加结果，两者的循环维度和目的不同。

[`SmemLayoutC/D`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp#L151-L219)为 Subtile 追加缓冲维。C 的每个 `128×16` Subtile 占 8192 字节，四个 C Stage 提供 32768 字节张量存储。基线 `ReuseSmemC=true`，C 和 D 利用同一片共享存储；`SharedStorage` 还包括同步和对齐要求，因此整体大小为前面取得的 33792 字节。

Epilogue 线程先把当前 C Subtile 从共享内存读入寄存器，再把对应的 TMEM 累加分区读入寄存器。FusionCallbacks 以 Fragment 为单位生成 D；本例 `FragmentSize=16` 描述一次 `visit` 处理的元素数，不是整个 Subtile 的元素数。随后生成的 D 分区进入共享内存，TMA Store 再从共享内存写回全局 D。于是一个 Subtile 的数据路径为：

```text
GMEM C → C 的 SMEM Stage → 寄存器 C ─┐
                                    ├→ Fusion → 寄存器 D → SMEM → TMA Store → GMEM D
TMEM Accumulator → 寄存器 Acc ───────┘
```

上述路径中，TiledCopy 为各参与者提供相应分区，Collective 负责何时执行这些 Copy。具体分区和 Fragment 的构造在第三篇展开；本篇继续跟踪保护这些数据的完成条件。

### 各条 Pipeline 怎样推进

Epilogue 同时使用三条 Pipeline。C Load Pipeline 连接 EpilogueLoad 与 Epilogue 线程，保护 C 的共享内存 Stage；Accumulator Pipeline 连接 MMA 与 Epilogue，保护 TMEM 结果 Stage；D Store Pipeline 跟踪在途 TMA Store，保证待写出的共享内存不会被提前覆盖。这三条状态各自推进，即使缓冲复用了相同存储，也不能把它们理解为同一个游标。

根据 [Pipeline 类型定义](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp#L195-L219)，基线的加载与写回类型可展开为：

**展开当前 Epilogue 的 C Load 与 D Store 协议**

```cpp
using CLoadPipeline = cutlass::PipelineTransactionAsync<
    4  // 本例 StagesC，保护四个 C 缓冲。
>;
using DStorePipeline = cutlass::PipelineTmaStore<
    4, // ReuseSmemC=true，沿用 C/D 共享缓冲的环形级数。
    1  // StagesD-1：等待后最多仍在途的已提交 Store 数。
>;
```

C 的 Producer 取得空 Stage 后发出 TMA Load，硬件完成使对应 Full 条件满足。Epilogue 对每个 Subtile 等待 C 就绪；对当前输出工作，则在首个 Subtile 读取前等待 Accumulator Ready。之后各 Subtile 使用同一份完整的累加结果，不需要重新发起该工作的 K 维归约。

当前 `ReuseSmemC=true`，所以将 C 读入寄存器后，暂时不能直接把这个共享缓冲交还给 C Load。Epilogue 还要把 D 写入该空间，并等待对应的 TMA Store 不再读取它，才能让输入 Producer 覆盖。源码的 [`tma_store_fn`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp#L764-L800)先建立共享内存写入对 TMA 的可见性，再发出 Store、提交 Store Pipeline、等待在途数量下降，最后释放已经安全回收的 Load Stage。这个等待保护的是 C/D 共享缓冲，不是 TMEM。

TMEM 的释放更早。完成最后一次必要的 TMEM Load 后，所有后续计算都可以使用寄存器中的值，Accumulator Stage 因而可以交还给 MMA。下面根据 [Subtile 循环中的释放点](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp#L865-L901)整理这一段，省略 C Load、访问谓词和后续写回：

**在最后一次 TMEM 读取后归还结果 Stage**

```cpp
copy(tiled_t2r, tTR_tAcc_mn, tTR_rAcc);  // 当前 Subtile 的 TMEM→寄存器读取。

if (do_acc_release) {  // 当前基线在最后一个 Subtile 为真。
  cutlass::arch::fence_view_async_tmem_load();
  acc_pipeline.consumer_release(acc_pipe_consumer_state);
  ++acc_pipe_consumer_state;
}

// 后续 Fusion 与 D Store 使用已经读入寄存器的结果。
```

因此，Accumulator Stage 可复用与 D 已经写回全局内存，是两个不同的完成时刻。基线 `DelayTmaStore=false`，当前 Subtile 处理完就发出其 Store；其他配方可以延后一轮 Store，但必须继续满足存储复用约束。末尾的 [`store_tail`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp#L958-L990)等待剩余写回并补齐延迟的 Stage 释放。局部输出处理至此闭合，下一节再说明 Fusion 如何接入这个已建立的过程。

## 2.4 Fusion：在输出过程中组合数值操作

### LinearCombination 与 FusionCallbacks

基线的 `LinearCombination` 表达数值关系，`FusionCallbacks` 则是该关系在具体 Epilogue 调度中的实现。Epilogue Builder 的最后一个输入可以是 Operation Tag，也可以是已经组织好的 EVT 或回调类型。[`CallbacksBuilder`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/collective/collective_builder.hpp#L75-L111)根据输入类型，生成与当前 Policy 和 Subtile 匹配的回调，或沿用调用者提供的回调类型。

默认线性组合的叶节点取得 Accumulator、C 和标量，计算节点生成 `alpha*Acc+beta*C`。回调的 `begin_loop`、`previsit`、`visit`、`reduce`、`postreduce`、`tma_store` 等入口被放在 Collective 的固定位置，因而融合逻辑能够在正确的数据就绪和存储复用条件下执行。对本例，核心工作是逐 Fragment 的线性组合；复杂 Operation 才会引入额外加载、归约或 Aux 写出。

D 的常规存储路径仍由 CollectiveEpilogue 安排。给回调增加一项数学运算，并不意味着回调接管了整个 C Load、TMEM Load 或 D Store Pipeline；它是在前面已经建立的数据流中增加输入或计算节点。这样，局部数学表达式可以变化，而基本的同步与资源组织仍由同一类 Epilogue 实现承载。

### 预定义融合操作与自定义 EVT

下面保留预定义 Operation 的接口索引，方便在基线之外选择数值关系。表格只比较数学操作和新增输入，不表示所有类型、布局和架构组合都支持同一组回调；具体组合仍由所选 Builder 匹配。

| Fusion Operation | 数学表达式 | 相比基础 Accumulator 增加的输入 | 主要用途 |
|-|-|-|-|
| `ScaledAcc` | `D = alpha * Acc` | 标量 `alpha` | 不读取 C，只对 Mainloop 累加器进行缩放并写回 D |
| `LinearCombination` | `D = alpha * Acc + beta * C` | 标量 `alpha`、`beta` 和源矩阵 C | 最基础、最常用的 GEMM Epilogue |
| `LinCombEltAct` | `D = activation(alpha * Acc + beta * C)` | `alpha`、`beta`、C 和 `ActivationFn` | 在线性组合之后融合 ReLU、GELU、SiLU、Clamp 等逐元素激活函数 |
| `LinCombPerRowBias` | `D[m,n] = alpha * Acc[m,n] + beta * C[m,n] + Bias[m]` | Per-row Bias | 为输出矩阵的每一行广播不同的 Bias |
| `LinCombPerColBias` | `D[m,n] = alpha * Acc[m,n] + beta * C[m,n] + Bias[n]` | Per-column Bias | 为输出矩阵的每一列或输出通道广播不同的 Bias |
| `LinCombPerRowBiasEltAct` | `D[m,n] = activation(alpha * Acc[m,n] + beta * C[m,n] + Bias[m])` | Per-row Bias 和 `ActivationFn` | 融合行 Bias 与逐元素激活函数 |
| `LinCombPerColBiasEltAct` | `D[m,n] = activation(alpha * Acc[m,n] + beta * C[m,n] + Bias[n])` | Per-column Bias 和 `ActivationFn` | 常用于 GEMM + 通道 Bias + Activation |
| `PerRowLinCombPerRowBiasEltAct` | `D[m,n] = activation(alpha[m] * Acc[m,n] + beta[m] * C[m,n] + Bias[m])` | Per-row `alpha`、Per-row `beta`、Per-row Bias 和 `ActivationFn` | 每一行使用独立 scale、residual scale 和 Bias |
| `PerColLinCombPerColBiasEltAct` | `D[m,n] = activation(alpha[n] * Acc[m,n] + beta[n] * C[m,n] + Bias[n])` | Per-column `alpha`、Per-column `beta`、Per-column Bias 和 `ActivationFn` | 每个输出通道使用独立 scale、residual scale 和 Bias，适合 per-channel quantization |
| `ScaledLinCombPerRowBiasEltAct` | `Z[m,n] = scale_a * scale_b * alpha * Acc[m,n] + scale_c * beta * C[m,n] + Bias[m]`；普通输出：`D = activation(Z)`；FP8 输出：`D = scale_d * activation(Z)` | `scale_a`、`scale_b`、`scale_c`、`scale_d`、Per-row Bias 和 `ActivationFn` | 融合输入反量化比例、C 的比例、Bias、Activation 和输出缩放 |
| `ScaledLinCombPerColBiasEltAct` | `Z[m,n] = scale_a * scale_b * alpha * Acc[m,n] + scale_c * beta * C[m,n] + Bias[n]`；普通输出：`D = activation(Z)`；FP8 输出：`D = scale_d * activation(Z)` | `scale_a`、`scale_b`、`scale_c`、`scale_d`、Per-column Bias 和 `ActivationFn` | 适合带 per-channel Bias、输出缩放和 Activation 的量化 GEMM |

这些 Operation 的定义集中在 [`fusion/operations.hpp`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/fusion/operations.hpp#L70-L392)，具体执行由 [Blackwell 回调配方](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp#L35-L110)连接到前面的 Epilogue。`Sm90EVT` 等名称仍可作为复用的 Visitor API 出现；它们的名称不改变本例的 Blackwell MMA 与存储路径。

预定义 Operation 无法表达目标数据流时，可以用 EVT 组合输入叶节点和计算节点。`Sm90AccFetch` 取得累加 Fragment，`Sm90SrcFetch` 取得 C，`Sm90ScalarBroadcast` 或张量加载节点取得标量与辅助输入。父节点消费子节点已经生成的 Fragment，使数据依赖与数学表达保持一致。这里需要明确输入从哪里来、结果交给谁；节点内部的 CuTe 分区不在本篇继续展开。

### INT8 输出转换与 Requant 的区别

此处才从 Dense FP32 输出切换到量化输出的对照。只改变线性组合的输出类型，会让它按转换规则写出整数，但尚未定义一套新的缩放方案。下面是局部接口示例，它使用独立别名，不改变本篇前面的 `ElementD=float` 基线：

**为 INT8 输出定义独立的线性组合操作**

```cpp
using Int8LinearCombination =
    cutlass::epilogue::fusion::LinearCombination<
        int8_t,       // 输出整数类型。
        float,        // 线性组合计算类型。
        ElementC,     // 源矩阵 C 的类型。
        float,        // alpha/beta 标量类型。
        cutlass::FloatRoundStyle::round_to_nearest
    >;
```

如果需要对输出重新量化，应另外确定输出尺度、零点、舍入和饱和关系。例如，设正的输出 Scale 为 `s_out`，定义 `requant_scale=1/s_out`，则可以把目标写为：

$$D=\operatorname{saturate}_{[-128,127]}
\left(\operatorname{round}_{\mathrm{nearest}}
\left(\mathrm{requant\_scale}\,(\alpha\,\mathrm{Acc}+\beta C)
+\mathrm{zero\_point}\right)\right)$$

这里的 `zero_point` 是输出整数域的平移项。它与第一篇 Mixed-input 中作为 `qs+t` 加法项传入的参数，应分别按各自公式解释。下面是 Requant 的数值数据流草图，不是可以直接实例化的 C++ 类型：

**把输出量化关系分解为 EVT 的输入与计算节点**

```text
OutputConvert<int8_t, round, saturate>
└── Add
    ├── Multiply
    │   ├── requant_scale
    │   └── Add
    │       ├── Multiply(alpha, Acc)
    │       └── Multiply(beta, C)
    └── zero_point
```

标量输出 Scale 可以使用广播节点；逐行、逐列 Scale 需要相应坐标下的输入节点。输出为 NVFP4 等块缩放格式时，还要定义块级范围、Scale 生成和 Scale 写出，不能只替换根节点的目标类型。第一篇已讨论输入 Block Scaling 与输出 requant 的区别，相关量化背景可继续参考[《Cutlass NVFP4 GEMM 技术分享》](https://xiaopeng.feishu.cn/wiki/S8N2wn26piQePNkBjiNcwFRCnRc)。

到这里，Mainloop 已能交付 Ready 的结果 Stage，Epilogue 已能消费并写回它。两者的接口与资源依赖已经明确，下面将这些局部过程放回完整 Kernel，说明各执行角色怎样围绕工作描述协作。


# 3. Kernel 怎样组织协作与分配工作

前一部分已经建立输入 Stage、累加 Stage 和输出 Subtile 的局部依赖。Kernel 把这些过程映射到同一 CTA/Cluster 内的线程角色，并决定当前工作和下一项工作。这里先确定工作描述，再比较静态 Persistent 与默认 CLC 路径，最后讨论 K 维拆分；各角色使用的 Dense Mainloop 和线性组合保持前面的定义。

## 3.1 GemmUniversal 组合了哪些组件

`GemmUniversal` 的类型由问题描述、两个 Collective 和 Tile Scheduler 选择共同形成。第一篇省略了第四个模板参数，本例等价于显式填写 `void`。下面是类型组合的局部接口示例，接续已经生成的 Collective：

**保留 Dense 计算组件并显式写出默认调度选择**

```cpp
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>, // 运行时 M/N/K/L 的描述类型。
    CollectiveMainloop,             // 已生成的 Dense Mainloop。
    CollectiveEpilogue,             // 已生成的 FP32 线性组合 Epilogue。
    void                            // 当前架构的默认 Tile Scheduler。
>;
```

第一条类型分派来自 Mainloop Policy 的内部 Schedule，它使 Kernel 匹配 [SM100 TMA Warp-Specialized 实现](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L61-L76)。第二条分派由 [`TileSchedulerSelector`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L115-L132)完成，本例将 `void` 映射到 `PersistentTileSchedulerSm100<ClusterShape,2>`。前一条确定局部加载与计算的协作形式，后一条确定工作怎样取得。

下面的图把类型组合、共享资源和运行角色放在一起。当前路径沿 Dense、默认 Persistent Scheduler 和完整 K 归约展开；图中的其他候选用于定位扩展分支。

![GemmUniversal 的编译期组合、运行时资源与 Warp 角色状态机](Imgaes/cutlass-3-gemm-abstractions/whiteboards/05-kernel/feishu-latest.jpg)

Kernel 从两个 Collective 取得 TensorStorage、Pipeline 类型和线程数，再组织 Kernel 级 `SharedStorage`。实例化这份 Dense 类型得到的 `SharedStorageSize` 为 230400 字节；它由当前类型决定，不随单次调用的 M/N/K 线性增长。下一节把完整矩阵问题转成可交给这些角色处理的工作。

## 3.2 WorkTileInfo 描述什么工作

`WorkTileInfo` 是各角色取得当前任务的共同依据。默认 Dense 路径保存 CTA 级的 M/N Tile 坐标、批次坐标与有效标志；K 范围由 Scheduler 接口根据本次 ProblemShape 取得。下面按 [当前调度器使用的工作描述](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/static_tile_scheduler.hpp#L55-L77)整理字段，省略辅助查询函数：

**用工作描述定位 CTA 输出区域**

```cpp
struct WorkTileInfo {
  int32_t M_idx;       // CTA 输出 Tile 的 M 坐标，不是元素行号。
  int32_t N_idx;       // CTA 输出 Tile 的 N 坐标，不是元素列号。
  int32_t L_idx;       // 当前批次。
  bool is_valid_tile; // 是否仍有有效工作。
};
```

对基线的 `CtaShape_MNK=(128,128,64)`，CTA 的 M 坐标相差一，对应的输出区域沿 M 前进 128 行。2SM MMA 的两个 peer CTA 各有自己的工作坐标，但共同完成一个更大的 Collective Tile；Mainloop 根据协作关系选择输入，Epilogue 根据本 CTA 的输出区域写回。这样，工作的编号、协作范围和数值输出范围保持一致。

Scheduler 的 [`work_tile_to_cta_coord`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp#L369-L389)给出各组件使用的 CTA 坐标，`get_k_tile_iterator` 与 `get_work_k_tile_count` 给出本项工作的 K 起点和迭代数量。Full-K 情况下，这个范围覆盖完整 K；采用 K 分解时，相同 M/N 坐标还需要区分不同的归约区间。

## 3.3 Warp Role 与共享状态

[`operator()`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L403-L454)取得动态共享内存基址，将它解释为 Kernel 的 SharedStorage，并让各线程按当前角色构造 Collective 与 Pipeline 接口。共享的是 Buffer、Barrier 和响应存储；各角色的工作描述、状态游标和接口对象沿各自分支推进。

基线每个 CTA 有 256 个线程，角色分为五类：

- **MMA：**Warp 0 参与结果存储管理和 MMA 路径；CTA pair 的 leader 发出当前协作乘加，并提交 Accumulator Stage。
- **Scheduler：**Warp 1 负责默认路径中的 CLC 查询；只有 Cluster 中第一个 CTA 的这个角色实际参与调度分支。
- **MainloopLoad：**Warp 2 根据当前 WorkTileInfo 加载 A/B，并推进 A/B Pipeline。
- **EpilogueLoad：**Warp 3 负责 C 或 Aux 的加载；本例 beta 非零，因此参与 C Load。
- **Epilogue：**Warp 4—7 消费累加结果与 C，执行 Fusion、必要的 Fixup 和 D 写回。

这些角色并发存在，列举次序不表示它们逐个执行。它们通过 [Kernel 的共享资源](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L181-L210)建立依赖：Mainloop Pipeline 保护 A/B，Accumulator Pipeline 保护结果，Epilogue Load Pipeline 保护 C，CLC Pipeline 保护工作响应。`tmem_base_ptr` 放在共享内存中供角色取得地址，累加值本身仍由前面的结果 Stage 保存。

Kernel 还维护 LoadOrder 和 CLCThrottle。前者安排 A/B Prologue 与 C/Aux 加载的启动关系；后者限制调度查询相对于实际输入加载的超前程度。它们分别约束不同执行流的推进，不能代替 A/B 数据 Ready 或 Accumulator Ready。Barrier 初始化完成后，Kernel 通过 Cluster 范围的初始化同步使各参与者看到有效状态，再进入工作循环。[初始化与角色参数](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L450-L614)给出了每条协议的生产者、消费者和参与计数。

## 3.4 静态 Persistent：按既定规则推进工作

### 首项工作与后续工作的坐标

Persistent 表示一个驻留的 Worker 可以连续处理多项工作，Worker 可以按当前 Kernel 的协作范围由 CTA 或 Cluster 构成。先看静态 Persistent 对照：首项工作由启动坐标确定，后续工作按既定映射规则推进。算法仍在运行时使用实际问题尺寸，“静态”描述工作分配规则，并不要求 M/N/K 都是编译期常量。

固定源码的 [静态调度基类](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/static_tile_scheduler.hpp#L132-L194)为 CTA 保存当前线性工作索引和 Grid 中的 CTA 总数。下面是下一项工作的源码节选，输入是当前索引，输出是同一规则下的后继索引：

**按 Grid 步长推进静态工作索引**

```cpp
void advance_to_next_work(uint32_t advance_count = 1) {
  current_work_linear_idx_ +=
      total_grid_size_ * uint64_t(advance_count);
}
```

取得新的线性索引后，调度器检查是否超出工作总数，再按批次、Cluster 排列、Raster Order 和 Swizzle 转回 M/N/L 坐标。`Raster Order` 决定沿哪个输出方向组织遍历，调度中的 `Swizzle` 调整 Tile 编号次序；它们改变工作访问顺序，不改变单个 Tile 的数学关系，也不同于共享内存 Layout 内部的 Swizzle。相关坐标还原由 [SM100 静态适配器](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_static_tile_scheduler.hpp#L73-L114)完成。

下面仅改变 Kernel 的调度选择，作为默认 CLC 路径的类型对照；输入类型与两个 Collective 沿用基线：

**选择静态 Persistent 调度器**

```cpp
using StaticGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,  // 同一问题描述。
    CollectiveMainloop,              // 同一局部乘加组件。
    CollectiveEpilogue,              // 同一结果处理组件。
    cutlass::gemm::StaticPersistentScheduler
>;
```

选择器据此绑定静态实现。新的类型可以沿第一篇的 Adapter 接口调用，但应按这个 Kernel 重新取得参数与资源需求；本篇贯穿实例没有因此改成静态调度。

### 静态分配的适用条件与局限

静态推进使后续工作能够由当前索引直接计算，不必为每个 Tile 查询一个动态领取结果。只要各 Worker 按同一规则推进，工作集合就可以被一致地覆盖。当资源可用性与各工作耗时比较均匀时，这种规则容易推演，也便于分析访问顺序。

工作完成速度并不总是均匀。不同问题、边界 Tile 或其他并发 Kernel 都可能改变 Worker 的推进速度；已经完成自己分配序列的 Worker，不能仅靠原来的静态步长接手另一个序列中的剩余任务。CLC 接下来改变的正是“下一项工作怎样取得”，局部 Mainloop 和 Epilogue 的数据计算保持既定形式。

## 3.5 CLC：动态取得尚未启动的工作

### CLC 查询、接管与返回结果

Cluster Launch Control（CLC）提供对尚未启动工作的动态接管机制。Grid 中的一项工作可以在资源允许时作为新的 Worker 启动，也可以被已经运行的 Worker 成功接管。接管成功后，原本尚未启动的执行被取消，其工作坐标交给请求者；被取消的是那次待启动工作，不是正在执行中的矩阵计算。[CLC 的编程模型](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html)描述了这两种互补的工作去向。

当前默认调度器的首项工作来自已启动的 `blockIdx`，随后经过 Raster/Swizzle 映射得到本 CTA 的工作坐标。Worker 要继续处理后续工作时，Scheduler 发出 `clusterlaunchcontrol.try_cancel`，请求一个待处理的 Cluster 工作。当前 Cluster 为 `2×2×1`，成员需要共同取得并解释这次领取结果，而不是四个 CTA 各自独立改变到互不相关的位置。

[`issue_clc_query`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp#L391-L409)使用异步 CLC 指令，将 16 字节响应写入指定的共享存储，并把完成与 Transaction Barrier 关联。响应可能表示接管成功，也可能表示没有取得新的工作；调度器解析有效标志后，才使用返回坐标推进。因而“查询已经发出”“响应已经可读”“还有有效工作”是连续的三个判断。

### 工作标识怎样转换为 WorkTileInfo

CLC 返回的是启动网格中的工作标识，不直接等于最终矩阵元素坐标。当前 [`fetch_next_work`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp#L453-L477)等待响应 Stage 就绪，解析取消结果和首个 CTA 的标识，归还响应 Stage，再结合本 CTA 在 Cluster 内的位置执行 Raster/Swizzle 转换，最终给出新的 WorkTileInfo。

下面是根据这一接口整理的消费节选。输入为当前响应游标和本地工作描述，输出为下一项工作；响应地址来自 Kernel 预先分配的 CLC 缓冲。

**从已就绪 CLC 响应取得下一项工作**

```cpp
auto [next_work, advance_state] = scheduler.fetch_next_work(
    work_tile_info,          // 当前工作；某些分解策略还会从它继续推进。
    clc_pipeline,            // 保护共享响应的 Pipeline。
    clc_pipe_consumer_state  // 当前角色期待的响应 Stage 与代次。
);
if (advance_state) {
  ++clc_pipe_consumer_state;
}
work_tile_info = next_work;
// 后续先检查 is_valid()，再由 work_tile_to_cta_coord 取得计算坐标。
```

这里释放的是响应存储，不是刚领取工作的计算资源。每个角色把需要的工作信息保存到本地后，就可以归还它对该响应 Stage 的消费份额，随后继续自己的加载或计算。只有参与者都完成必要的消费，Scheduler 才能复用这个 Stage 写入后续响应。数据计算是否结束，则继续由 Mainloop、Accumulator 和 Epilogue 的完成条件保护。

### CLC Pipeline 怎样向各执行角色交付工作

基线的 CLC Pipeline 为两级，与两个响应存储位置对应。Cluster 中第一个 CTA 的 Scheduler Warp 是 Producer，MMA、MainloopLoad、参与的 EpilogueLoad 和 Epilogue 角色都是 Consumer；Scheduler 自己也消费响应，以判断是否继续发出下一次查询。这些角色读取同一代工作描述，但可以在不同时间到达读取位置。

[Kernel 的 CLC 参数](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L499-L520)固定一个发射参与者，将响应事务字节数设为 16，并根据实际参与的 Warp 数计算 Consumer 到达计数。当前 beta 非零，EpilogueLoad 需要 C，因此也在消费集合中。这样，只有实际会读响应的角色才计入回收条件，响应不会在某个角色尚未取得工作信息时被覆盖。

CLCThrottle 进一步把查询进度与 MainloopLoad 的推进联系起来。MainloopLoad 为下一次查询释放一个推进许可，Scheduler 消费该许可后再 acquire CLC 响应 Stage 并发出查询。[Scheduler 分支](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L679-L723)随后读取这次结果，更新自身的工作状态。Throttle 控制调度超前，CLC Pipeline 保护响应读写，二者服务于不同的依赖。

这一过程可以用两条相互配合的通路理解；下图是状态草图，不规定各角色的实际执行时长：

```text
Scheduler：取得推进许可 → 取得空响应 Stage → 发出 CLC 查询
                                                   │ 硬件写入并完成响应
                                                   ▼
各角色：等待同代响应 → 保存本地 WorkTileInfo → 归还响应消费份额
                           │
                           └→ 继续各自的加载、MMA 或 Epilogue
```

查询没有取得新工作时，各角色在处理完自己已有的工作后离开领取循环，并完成各自的 Tail。Scheduler 等待响应消费者退出，MainloopLoad 等待输入 Stage 释放，MMA 与 Epilogue 完成结果存储的交接。基线只含少量输出 Tile，可能很快读到无效响应；观察多轮领取时应增大运行时工作集合，而不用改变当前 Dense 类型。

## 3.6 K 维拆分与结果合并

### Full-K 与部分 K 工作

静态推进与 CLC 描述工作从哪里来，Full-K 与 K 拆分则描述一项工作计算多少归约范围。默认 Dense 工作覆盖整个 K，Mainloop 完成后，Epilogue 可以直接使用完整 Accumulator。设 I、J 为当前局部输出区域的行、列坐标集合。把它的 K 范围划分为多个区间后，各项工作只得到部分和：

$$P_j=A[I,K_j]B[K_j,J],\qquad
\mathrm{Acc}_{I,J}=\sum_j P_j$$

这里的部分积只覆盖当前 I×J 区域，各 Split 在自己的工作范围内生成和保存贡献。K 拆分使多个工作可以共同完成一个输出 Tile，同时增加部分结果存储、同步与合并。分解策略需要平衡这些额外操作与增加的并行工作量。

### Split-K、Stream-K 与 Fixup

Split-K 通常把一个输出 Tile 的 K 范围划成若干份；Stream-K 则沿输出 Tile 和 K 迭代形成的工作空间分配计算单元，一个单元可能包含部分 Tile，也可能继续到另一 Tile。在固定 SM100 路径中，[`PersistentTileSchedulerSm100StreamK`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp#L46-L159)在 CLC 调度器上组合 K 分解，运行参数进一步提供 `splits`、`decomposition_mode` 和 `reduction_mode`。这里改变的是工作范围与合并方式，不重新定义 A/B 的局部 MMA 数据通路。

Stream-K 的 WorkTileInfo 增加 K 起点和 K Tile 数，使 Mainloop 可以从指定的 K Tile 开始处理有限范围。一次 CLC 响应对应的调度单元还可能在本地继续推进，因此每处理一个输出片段不一定都要读取新响应。前面的 `advance_state` 就用来区分这两种情况，避免错误推进 CLC Pipeline 的代次。

部分结果需要 [`fixup`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp#L636-L728)合并。当前 TMEM 累加路径先将需要合并的 Fragment 读入寄存器，再利用 Scheduler 的 Workspace 和同步状态交换、归约各 Split 的贡献；承担最终输出的工作将合并结果写回供 Epilogue 消费的 TMEM 表示。[`tmem_fixup`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp#L931-L1008)把结果加载、合并和必要的回写连接起来，跨 CTA 的部分和并不是直接在另一 CTA 的 TMEM 上相加。

### 谁负责最终 Epilogue

`compute_epilogue(work_tile_info)` 决定该工作是否承担最终输出。这里的“最终”由分解策略定义，不等于哪个 CTA 在墙钟时间上最后结束。只有完整贡献合并后，才执行一次含 C、Bias、Activation 或量化转换的最终 Epilogue；否则，逐 Split 执行完整后处理可能重复计入 C，非线性操作也不能通过简单求和恢复。

对选择关系，可以保留下面这张小表作为本节的索引。它比较同一层的工作领取与 K 范围，不把第一篇的数值类型混入调度分类：

| Kernel 调度选择 | 后续工作来源 | 当前工作与结果处理 |
|---|---|---|
| `StaticPersistentScheduler` | 静态 Grid 步长与坐标映射 | 基础工作覆盖完整 K，直接进入 Epilogue |
| 默认 `void` / `PersistentScheduler` / `DynamicPersistentScheduler` | CLC 响应 | 基础工作覆盖完整 K，直接进入 Epilogue |
| `StreamKScheduler` | CLC 单元与单元内部推进 | 按分解参数处理完整或部分 K，必要时 Fixup 后输出 |

Grouped 的 `GroupScheduler` 还需要确定当前问题编号，再按本组 Shape 和地址解释工作；相应输入组织已在第一篇展开。本篇到这里建立的是“领取方式、K 范围和最终输出条件”的区分，具体分组或数值变体按发生变化的部分继续组合。

## 3.7 将工作领取与局部计算连起来

把前面的状态放回一个默认 Full-K 工作：各角色先取得同代的 WorkTileInfo，MainloopLoad 根据它选择 A/B 和 K 范围，EpilogueLoad 根据同一输出坐标选择 C。两条加载可以与已有工作交错推进，但都通过各自的 Full/Empty 条件控制缓冲访问。MMA 在输入就绪后形成结果，Accumulator Pipeline 再使 Epilogue 能够消费对应的完整归约。

Epilogue 读完当前 Accumulator 后归还结果 Stage，继续用寄存器与共享内存完成 D 写回。与此同时，已经取得后续工作描述的角色可以准备下一项工作。角色可以处于不同工作项的不同阶段，但在每次资源交接处必须匹配工作次序和 Pipeline 代次；它们不是绕过同步后各自任意前进。

这里还有两个资源释放层次。归还某个 Accumulator Stage，允许后续工作复用该 Stage；释放整个 TMEM 分配，则发生在工作循环结束、相关消费者完成之后。[Kernel 的退出路径](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L780-L804)等待尾部结果释放，并协调 CTA pair 的最终回收。完成这些 Tail 后，Kernel 才结束对该调用资源的使用。

至此，局部计算、跨角色交接和跨工作推进已经连成设备端执行过程。下一部分回到主机端，说明第一篇的参数如何形成这些运行状态，并启动这里分析的 Kernel。


# 4. Device 怎样准备并启动 Kernel

前面的 Kernel 已经确定局部计算组件、工作领取方式和共享资源。本节从主机调用者提供的矩阵地址与尺寸出发，将它们转换成这份 Kernel 使用的参数，并说明启动前后各项资源的生命周期。完整数据初始化与数值参考继续使用第一篇的 [Dense 程序](exemples/dense_baseline.cu)，这里只展开调用背后的机制。

## 4.1 Arguments 怎样转换为 Params

`GemmUniversalAdapter<GemmKernel>` 复用 Kernel 定义的 `Arguments` 和 `Params`，并将初始化后的 Params 保存为成员 `params_`。Arguments 面向调用者，保存问题尺寸、矩阵指针、Stride、融合参数和调度选项；Params 面向设备执行，保存各组件已经准备好的访问与调度描述。Adapter 不需要为每次 `run(stream)` 重新从 Host Arguments 推导这些对象，因此可以复用已经初始化的状态。

下面根据 [Kernel 参数声明](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L214-L232)整理两级结构，省略字段默认初始化。类型名称来自已经选定的三个组件：

**保留调用语义并降低各组件的内部参数**

```cpp
struct Arguments {
  GemmUniversalMode mode;          // 本次调用模式。
  ProblemShape problem_shape;     // 实际 M/N/K/L。
  MainloopArguments mainloop;     // A/B 地址与 Stride。
  EpilogueArguments epilogue;     // alpha/beta、C/D 地址与 Stride。
  KernelHardwareInfo hw_info;     // 设备与相关硬件信息。
  TileSchedulerArguments scheduler;
};

struct Params {
  GemmUniversalMode mode;
  ProblemShape problem_shape;
  MainloopParams mainloop;        // 当前输入的 TMA 等访问描述。
  EpilogueParams epilogue;        // 当前 C/D 与 Fusion 的访问和计算参数。
  TileSchedulerParams scheduler;  // 工作网格、映射和分解参数。
  KernelHardwareInfo hw_info;
};
```

[`GemmKernel::to_underlying_arguments`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L254-L298)按顺序分配外部 Workspace 子区域，并分别调用 Mainloop、Epilogue 和 Scheduler 的转换函数。Mainloop 结合本次矩阵地址、Shape 和已生成的局部类型准备输入访问；Epilogue 准备 C/D 访问和 Fusion 参数；Scheduler 准备逻辑工作网格与映射。Device 将这些结果保存下来，Kernel 启动时接收的便是这份 Params。

下面的图把参数降低、资源准备和执行结果的检查放在一起。上半部分描述准备好的对象如何进入启动，下半部分描述复用、错误和结果验证；这些关系沿用前面已经建立的 Kernel。

![Device 层的 Arguments、Workspace、Params、Launch 与验证边界](Imgaes/cutlass-3-gemm-abstractions/whiteboards/07-device/feishu-latest.jpg)

## 4.2 Workspace、共享内存与资源生命周期

Workspace 是调用者在设备全局内存中分配的辅助缓冲。`get_workspace_size(args)` 汇总本次 Epilogue 与 Scheduler 的需求，并按 Kernel 要求对齐。默认路径、额外 Fusion 或 K 分解可能需要不同内容；未使用外部辅助状态时，查询结果可以为零。第一篇的程序始终按查询结果分配，而不把某个固定字节数作为通用常量。

Kernel 的共享内存是另一类资源。当前类型的 `SharedStorageSize=230400` 在每个 CTA 的执行环境中使用，包含输入与输出缓冲及同步状态；它由类型、Stage 与对齐等选择决定。Workspace 则由主机提供地址并按调用需求准备。两者在 [Kernel 的 Workspace 查询与初始化](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L337-L375)和 Adapter 的启动准备中分别处理。

A/B/C/D、Workspace 和 stream 由调用者管理，Adapter 保存相关参数而不接管这些资源的所有权。程序可以在提交 Kernel 后继续进行主机工作，但在对应的设备使用结束之前，应保留 Buffer 和已建立的执行依赖。更换输入地址或复用 Workspace 时，也要先处理此前使用它们的异步操作。

## 4.3 can_implement、initialize 与 update

[`can_implement(args)`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/device/gemm_universal_adapter.h#L230-L239)检查本次运行参数与当前 Kernel 类型的兼容性。Kernel 将检查分派给 Mainloop、Epilogue 和 Scheduler，并验证模式、对齐、布局及 Cluster 等条件。它没有执行矩阵计算；裸指针背后的分配容量、数据内容和最终结果仍分别由调用者的数据准备与参考比较保证。

[`initialize(args,workspace,stream)`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/device/gemm_universal_adapter.h#L313-L355)先初始化所需 Workspace，再构造并保存 Params，最后设置必要的 Kernel 属性。本例共享内存需求超过 48 KiB，因此初始化还需要通过 `cudaFuncSetAttribute` 请求相应的动态共享内存容量。参数类型能够实例化，只解决了类型构造问题；实际设备也需要支持这份 Kernel 和它请求的资源。

下面是根据 Adapter 源码整理的初始化节选。输入为已经检查过的本次 Arguments 和 Workspace，输出为成员 Params 及完成的启动准备：

**初始化辅助状态并保存设备参数**

```cpp
Status status = GemmKernel::initialize_workspace(args, workspace, stream, cuda_adapter);
if (status != Status::kSuccess) {
  return status;
}
params_ = GemmKernel::to_underlying_arguments(args, workspace);
// 后续根据 GemmKernel::SharedStorageSize 设置所需的 Kernel 属性。
// 省略：CudaHostAdapter 分支和属性设置的错误处理。
```

[`update(args,workspace)`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/device/gemm_universal_adapter.h#L358-L370)重建 Params，但不执行同样的 Workspace 初始化，也不替调用者重新进行兼容性检查。使用它之前，需要确认新参数不要求尚未完成的资源准备，并且此前的执行已经满足复用条件。对于第一次运行或问题、调度、Workspace 需求有实质变化的调用，重新走检查与 initialize 可以保持准备过程完整。

只修改 Host 端 `arguments` 而不更新 Adapter，随后调用不带 Arguments 的 `run(stream)`，使用的仍是已保存 Params。第一篇“将基线用于另一项矩阵问题”中的操作顺序，正是由这一状态关系决定的。

## 4.4 Grid、Cluster 与异步启动

`run` 从 Kernel 取得 Block Shape、Grid Shape 和动态共享内存需求。Block 的线程数来自前面的 Warp Role，本例为 256；Grid 则由 Scheduler 根据工作空间与 Cluster 约束计算，不能只看完整 Collective 的 M/N 分块。当前基线的 CTA 级输出网格为 `2×2`，恰好组成一个 `2×2×1` Cluster，四个 CTA 协作完成这次小尺寸调用。

静态 Persistent 的 Grid 表示一组按固定规则推进的 Worker，后续工作由各自的 Grid 步长计算；CLC 路径的逻辑 Grid 则覆盖可启动的工作，运行中的 Worker 还可以接管其中尚未启动的部分。所以逻辑工作网格大小与实际新启动了多少 Worker，不是同一项计数。[调度器的 Grid 计算](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp#L192-L261)与 [Adapter 的启动入口](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/device/gemm_universal_adapter.h#L372-L449)在这里汇合。

本例具有非平凡的静态 Cluster，Adapter 使用支持 Cluster 的启动方式；`1×1×1` Cluster 可以使用普通启动，动态 Cluster 则还要从参数中取得首选与回退形状。这里的 Cluster Launch 负责提交具备协作范围的 Kernel，而 CLC 负责 Kernel 执行中的后续工作领取，两个名称相近的机制位于不同阶段。

实际 CUDA 设备由调用者当前的 device/context 和 stream 决定，`hw_info.device_id` 只是传入的运行信息。第一篇先调用 `cudaSetDevice` 再分配设备数据，确保 Buffer 和执行设备相符；Workspace 初始化和 Kernel 使用同一 stream，或通过明确的跨 stream 依赖连接。完成这些准备后，run 才提交真正的设备执行。

## 4.5 分别检查启动、执行和数值结果

一次调用的结果需要按发生位置分别判断。Adapter 返回的状态说明参数准备或立即启动是否成功；stream 同步使调用者等待设备执行结束，并暴露执行期间的错误；最后的参考比较判断已写回 D 是否符合相同输入与数值规则。任何一层的成功，都有对应的检查范围。

第一篇的完整程序已经包含这条调用链。本篇用下面的状态草图回收各项检查的作用，而不再复制一份含占位参数的“完整程序”：

```text
Arguments 与 Kernel 兼容
        ↓
Workspace 初始化、Params 与启动属性准备
        ↓
run 返回立即启动状态
        ↓
stream 同步确认设备使用结束
        ↓
回读 D，与相同输入的参考计算比较
```

基线输入先保存为 FP16，CPU 参考读取相同的已存储值并使用更高精度归约，再应用同一 alpha/beta。这样，比较对象与实际 Kernel 输入一致。若更换量化方式或输出格式，还需要计入相应的重建与舍入规则；具体数值路径已经在第一篇按场景说明。

本篇给出的源码分析和类型实例化能够解释参数、资源与执行协议。Thor 上的成功启动、数值误差和性能仍应分别以对应运行结果为依据。版本和编译方法集中在第一篇与 [示例目录](exemples/README.md)，不作为本篇执行机制的额外分支。

# 5. 从执行过程进入 CuTe 的空间构造

现在可以沿同一条路径复述这份 Kernel：Builder 固定局部计算类型和缓冲配置，Device 把本次输入降低为 Params，Kernel 为各角色交付工作描述，Mainloop 在输入就绪后形成累加结果，Epilogue 消费结果并写回 D。静态 Persistent 与 CLC 改变后续工作怎样取得，K 分解改变当前工作承担的归约范围；输入、结果、响应和写回各有自己的完成条件。

第二篇在这里停止于执行协议。TiledMMA 如何从 Atom 和布局组成，`partition_*` 如何得到参与者的局部 View，Fragment 如何与某个操作兼容，以及修改布局、构造视图和执行搬运之间的区别，继续交给[第三篇的 CuTe 空间模型](03-cutlass-principled-abstractions_zh-CN.md)。

第三篇后半使用单 CTA、单 A/B Buffer 的教学 Kernel，将这些空间对象显式构造出来。它保留矩阵乘加关系，但为了观察对象依赖而简化时间组织；其 Tile、Cluster 和部分布局因此与本篇不同。理解这条简化路径后，再回看本篇的八级输入 Pipeline、结果交接与 CLC 工作领取，就能把空间对象与生产级执行组织重新联系起来。
