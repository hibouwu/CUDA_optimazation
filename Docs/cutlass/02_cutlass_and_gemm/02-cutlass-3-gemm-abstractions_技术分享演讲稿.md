# CUTLASS 3.x GEMM 抽象层次技术分享演讲稿

> 对应材料：[《CUTLASS 3.x：面向 GEMM 内核设计的正交、可复用、可组合的抽象结构》](02-cutlass-3-gemm-abstractions_zh-CN.md)  
> 建议时长：50 分钟讲解 + 10 分钟问答  
> 时长组成：口头正文约 30～35 分钟，图示停留与示例 71/74 源码 walkthrough 约 15～20 分钟  
> 目标听众：了解 CUDA、GEMM、shared memory 和 Tensor Core，但没有系统掌握 CUTLASS 3.x 的开发者  
> 分享目标：让听众建立 CUTLASS 3.x 的五层心智模型，知道一个 GEMM 配置如何从硬件指令逐层组合成可启动内核，并理解 Thor/SM110a 路径中的实现复用与验证边界

## 一、时间安排

| 时间 | 内容 | 对应材料 |
|---:|---|---|
| 0～5 分钟 | 开场：为什么 CUTLASS 看起来复杂，什么叫正交组合 | 摘要、问题背景 |
| 5～12 分钟 | 五层概念模型与一条完整 GEMM 数据流 | 图 1、图 2 |
| 12～17 分钟 | 贯穿示例、参数选择与 SM110a 关键边界 | 完整 Builder 示例 |
| 17～28 分钟 | Collective Mainloop、Builder 与示例 71 源码 walkthrough | 图 3、图 4 |
| 28～34 分钟 | Collective Epilogue、EVT 与数据类型边界 | 图 5 |
| 34～43 分钟 | Kernel、tile scheduler、CLC、Stream-K 与示例 74 | 图 6、图 7 |
| 43～47 分钟 | Device 层、失败路径和启动生命周期 | 图 8 |
| 47～50 分钟 | SM110a 验证阶梯与总结 | 验证清单、总结页 |
| 50～60 分钟 | 集中问答 | 高概率问答、源码索引 |

## 二、完整演讲稿

### 0～5 分钟：开场——CUTLASS 为什么看起来这么复杂

大家好，今天分享的主题是 CUTLASS 3.x 的 GEMM 抽象层次。

很多人第一次看 CUTLASS 3.x，最直接的感受是模板特别多。一个 GEMM 还没有真正开始运行，我们就已经看到了 `CollectiveBuilder`、`CollectiveMma`、`CollectiveEpilogue`、`GemmUniversal`、`GemmUniversalAdapter`，以及各种 tile shape、cluster shape 和 schedule。

如果只是把这些模板逐个记下来，CUTLASS 会显得非常复杂。但它真正想解决的问题其实很清楚：GPU GEMM 的优化不是一个单一决策，而是一组彼此相对独立的决策。

例如，我们需要决定：

- 使用哪一条 Tensor Core 指令；
- 一条指令覆盖多大的矩阵；
- 线程和数据怎样做空间划分；
- GMEM 到 SMEM 的拷贝怎样与 MMA 重叠；
- epilogue 如何融合缩放、bias 和 activation；
- CTA 或 cluster 怎样领取整个问题中的输出 tile；
- 主机端怎样检查、初始化和启动内核。

CUTLASS 3.x 的核心做法，不是把这些问题藏在一个巨大的 kernel 里，而是把它们拆成多个可组合层次。

这里标题里有三个词：正交、可复用、可组合。我们先把这三个词说清楚。

“正交”不是说各部分完全没有关系，而是说它们首先表达不同维度的决策。例如，MMA 指令选择描述计算原语，epilogue 描述后处理，tile scheduler 描述 grid 级工作分配。理论上，我们可以在保持其他部分接口不变的情况下替换其中一部分。

“可复用”是说同一个低层组件可以被多个高层组件使用。例如，同一种 TiledMma 可以进入不同的 mainloop schedule；同一种 fusion operation 可以进入不同的 epilogue；同一个 `GemmUniversalAdapter` 接口可以包裹多种不同 kernel。

“可组合”则意味着每一层不仅完成自己的工作，还向上一层提供稳定接口。TiledMma 不是单独结束，而是被 Collective 消费；Collective 不是单独结束，而是被 Kernel 组合；Kernel 最终由 Device 层暴露给主机。

不过，正交不等于资源上完全独立。tile shape 改大以后，可能增加寄存器和 SMEM 占用；epilogue 使用更多 shared storage，可能迫使 mainloop 减少 pipeline stage；cluster shape 也可能影响合法 schedule 和 occupancy。

所以更准确的说法是：

> CUTLASS 让各类决策在接口和语义上尽量正交，再由 Builder 在编译期处理它们之间的资源与合法性约束。

这也解释了为什么 CUTLASS 代码看起来模板很多。模板参数并不只是语法负担，它们是在编译期描述一个 kernel 的物理计划。

所以今天我希望大家不要记住所有类型名，而是先记住一个判断方法：

> 看到一个 CUTLASS 类型时，先问它解决的是“硬件原语”“work tile 内部的空间划分”“一个 work tile 的 load/MMA pipeline”“全局 work-tile 分配”，还是“主机启动”。

只要能回答这个问题，大部分 CUTLASS 代码就不会再是一团模板。

### 5～12 分钟：五层概念模型与完整 GEMM 数据流

请看图 1。CUTLASS 3.x 把 GEMM 组织成五层：Atom、Tiled MMA/Copy、Collective、Kernel 和 Device。

这五层可以分别用五个问题来理解。

第一层 Atom 回答的是：**底层执行什么硬件操作？**

它包装具体的 MMA 或 copy 指令，以及执行这条指令所需的线程—数据元信息。例如，某条 MMA 指令要求哪些线程共同参与，A、B 和 accumulator 分别使用什么 shape 和 layout。

第二层 Tiled MMA/Copy 回答的是：**一个 work tile 内部，Atom、线程和值怎样做空间划分？**

一个 Atom 往往只覆盖很小的矩阵区域。Tiled MMA 定义 Atom 在 M、N、K 方向怎样重复、交织或重排，以及线程和值怎样 partition；Tiled Copy 定义一次 tile copy 中线程和值的空间划分。它们共同给出 work tile 内部的空间执行计划。

这里需要精确一点：完整 work tile 的外部 shape 合同通常由 Collective 的 `TileShape` 或 Builder 的 `MmaTileShape` 参数给出；TiledMma 负责说明在这个 tile 内部如何执行 MMA。因此不能简单理解为“TiledMma 单独决定完整 work tile”。

前两层主要属于 CuTe。它们关心的是线程和数据在空间上如何组织，因此可以统称为空间微内核。

第三层 Collective 回答的是：**怎样把 TMA/TiledCopy、SMEM layout 和 TiledMma 组装成处理一个 work tile 的 pipeline？**

这里开始出现 TMA pipeline、SMEM stage、warp specialization、barrier 和 phase。Collective 以 `TileShape` 作为 work-tile 合同，把 GMEM→SMEM load、wait、TiledMma、stage release 等操作按时间组装起来。Collective Mainloop 负责加载和 MMA，Collective Epilogue 负责后处理和输出存储。

第四层 Kernel 回答的是：**怎样让一个 collective 覆盖完整问题？**

它把 mainloop 和 epilogue 组合成设备端 kernel，并决定 CTA 或 cluster 如何遍历整个输出矩阵。tile scheduler 也是在这一层接入。

第五层 Device 回答的是：**主机怎样使用这个 kernel？**

它负责参数管理、`can_implement`、workspace 分配、初始化，以及最终的 kernel launch。

所以这五层不是五套互相重叠的 API，而是一条组合链：

```text
硬件指令
  → 空间微内核
  → CTA/cluster 内的时间微内核
  → 覆盖整个问题的设备 kernel
  → 主机端可复用句柄
```

这里最关键的分界是：Atom 定义原语；Tiled MMA/Copy 定义一个 work tile 内部的空间执行方式；Collective 把 load、MMA、同步和 stage 生命周期组装成处理一个 work tile 的 pipeline；Kernel 再把 work tile 分配到整个 grid。

为了把这五层和真实 GEMM 对上，我们再沿着一条数据流看一遍。

输入矩阵 A 和 B 最初位于 GMEM。内核要先决定每个 CTA 或 cluster 负责哪一个输出 tile，这属于 Kernel 和 tile scheduler 的范围。拿到 tile 之后，Collective Mainloop 使用 TiledCopy 或 TMA 把 A、B tile 搬到 SMEM；等待数据就绪后，再使用 TiledMma 发射一组 MMA Atom。计算结果进入 accumulator，在 Blackwell 上可能位于 TMEM。最后，Collective Epilogue 把 accumulator 转成 D，执行缩放、融合操作和输出写回。

因此，一条完整路径可以写成：

```text
GMEM A/B
  → TiledCopy / TMA
  → SMEM stage
  → TiledMma / MMA Atom
  → accumulator
  → Collective Epilogue
  → GMEM D
```

五层分别在这条路径上做不同决策：

- Atom 决定最后一条数学或搬运指令长什么样；
- Tiled 决定一个 work tile 内部的 Atom、线程和值怎样划分；
- Collective 接收 `TileShape`，把 TMA/TiledCopy、SMEM stage、wait、TiledMma 和 release 组装成一个 work-tile pipeline；
- Kernel 决定哪一个 CTA 或 cluster 执行哪一份 tile 工作；
- Device 把问题尺寸、指针和调度参数交给 Kernel。

这里可以停下来问听众一个问题：`partition_A` 属于哪一层？它主要是空间划分，所以属于 TiledMma/CuTe 这一侧。TMA 的 producer/consumer barrier 属于哪一层？它描述时间依赖，所以属于 Collective。

这个判断练习比背类名更有用。以后遇到陌生类型，也可以先判断它作用于空间、时间、grid，还是主机接口。

还要注意，五层是概念层次，不代表每个源码文件只出现一层。一个具体 Kernel 实现文件可能同时调用 Collective、pipeline 和 scheduler；一个完整示例也会同时实例化 Builder、Kernel 和 Device。分层的价值是帮助我们理解职责，而不是要求源码目录绝对隔离。

### 12～17 分钟：贯穿示例、参数选择与 SM110a 的关键边界

为了避免后面每一层都换一个例子，全文使用同一组 GEMM 配置。

A 和 B 使用 FP16，累加和输出使用 FP32；MMA tile 是 256×128×64，cluster shape 是 2×2×1。目标设备是 NVIDIA Thor，也就是 SM110a。这组配置可以对照 CUTLASS 固定版本的 [Blackwell 示例 71：Collective Builder + EVT GEMM](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu) 阅读。

先看最小化后的实际配置：

```cpp
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementAccumulator = float;
using ElementC = float;
using ElementD = float;

using MmaTileShape = cute::Shape<cute::_256, cute::_128, cute::_64>;
using ClusterShape = cute::Shape<cute::_2, cute::_2, cute::_1>;
```

这里已经同时出现了五类约束。

`ElementA/B` 决定输入表示和可选择的 MMA 操作；`ElementAccumulator` 决定累加精度；`MmaTileShape` 决定 collective 每次处理的工作区域；`ClusterShape` 表示 2×2 的 CTA cluster；`ArchTag` 和 `OperatorClass` 决定 Builder 应从哪个实现族选择 Tensor Core 配方。

再给一个 alignment 的具体例子。FP16 元素宽度是 16 bit，如果要求 128-bit 对齐，那么每次自然向量化访问包含 8 个元素：

```cpp
static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
// AlignmentA = AlignmentB = 8
```

这意味着 A/B 指针和 leading dimension 必须满足相应的 8 元素对齐约束。如果调用时 stride 或地址不满足，代码可能仍然成功编译，但 `can_implement` 会拒绝这个问题。

所以 Builder 参数不是单纯的类型装饰；它们把数据类型、内存访问、tile、cluster 和硬件配方连接在一起。

这里有一个非常容易误解的点，我建议大家重点记住。

代码中的：

```cpp
using ArchTag = cutlass::arch::Sm100;
```

与编译参数中的：

```text
-gencode arch=compute_110a,code=sm_110a
```

并不矛盾。

`cutlass::arch::Sm100` 在这里选择的是 CUTLASS 源码中可复用的 Blackwell TCGen05 配方族。真正决定最终二进制目标架构的，是 `compute_110a → sm_110a`。

换句话说：

> `Sm100` 是源码实现族标签，`sm_110a` 是最终编译目标。

不能因为源码文件名中出现 `sm100`，就说最终 kernel 是为 B200 编译；也不能因为某个公共参数类型带有 `Sm90`，就说 kernel 退回了 Hopper。

但反过来，源码可以复用，也不等于运行时天然正确。Thor 路径仍然需要逐层验证：

1. Builder 的静态约束是否通过；
2. `ptxas` 是否接受目标指令；
3. 生成的 SASS 是否确实属于 SM110a；
4. Thor 实机数值结果是否正确。

所以今天讲的是一套组合方法，而不是“只要写 `Sm100` 就自动支持 SM110a”。

举一个具体的判断例子。假设代码成功实例化了 `KernelTmaWarpSpecialized2SmSm100`，并且用 `sm_110a` 编译成功。我们最多可以先得到两个结论：第一，CUTLASS 的静态约束接受了这组类型；第二，工具链接受了目标 PTX。此时还不能直接说“2SM GEMM 已经在 Thor 上正确工作”。

下一步必须检查生成函数范围内是否确实出现预期的 SM110a Tensor Core、TMA 和同步指令，再用固定输入与独立参考实现比较结果。如果数值正确，才得到运行时正确性的证据；如果还要讨论性能，则需要进一步测量实际吞吐、访存和 occupancy。

也就是说，下面四句话不能互相替代：

```text
模板可以实例化
PTX/SASS 可以生成
Thor 上数值正确
Thor 上性能优秀
```

技术分享中把这四层证据分开，会比直接说“支持 SM110”更严谨。

接下来我们看这组参数如何沿五层向下推导。

### 17～28 分钟：Collective Mainloop、Builder 与示例 71

请看图 3。Collective Mainloop 以 `TileShape` 定义的 work tile 为处理单位，把这个 tile 所需的加载、同步和 MMA 组装成 pipeline。对应的公共声明位于 [`collective_mma_decl.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/collective/collective_mma_decl.hpp)。

它需要组合几类低层对象：

- 一个 `TiledMma`，决定执行什么 MMA，以及线程和数据如何分区；
- A、B 各自的 GMEM 到 SMEM `TiledCopy`；
- SMEM layout；
- 如果 MMA 从寄存器读取操作数，还需要 SMEM 到 RMEM 的 copy atom；
- 最后还要有 dispatch policy，指定架构实现和调度策略。

可以把它理解成一个 work-tile pipeline 组装器。`TileShape` 告诉它要处理多大的 work tile；TiledMma 和 TiledCopy/TMA 告诉它 tile 内有哪些空间操作；Collective Mainloop 决定这些 load、wait、MMA 和 release 按什么顺序发生、怎样形成 pipeline。

我们用一个双 stage 的简化时间线来具体说明。假设 stage 0 已经装入第 k 个 A/B tile，stage 1 正在装入第 k+1 个 tile，那么理想情况是：

```text
时间 t0：TMA 写 stage 0
时间 t1：MMA 读 stage 0，同时 TMA 写 stage 1
时间 t2：MMA 读 stage 1，同时 TMA 复用 stage 0 写下一块
```

这条时间线需要至少两组依赖。

第一组依赖保证 MMA 不能在 TMA 写完之前读取 stage；第二组依赖保证 TMA 不能在 MMA 消费完之前覆盖 stage。Blackwell 的具体实现还要处理异步 MMA completion、warp 角色和 cluster 内同步。

现在就能看出空间层和时间层的区别：`TiledCopy` 描述“一次拷贝由哪些线程搬哪些元素”，而 pipeline 描述“第几次拷贝与第几次 MMA 在什么时候重叠”。

再看 cluster shape 的具体作用。我们的 `ClusterShape = Shape<_2,_2,_1>` 表示一个 cluster 在 M 方向有 2 个 CTA，在 N 方向有 2 个 CTA。对 A tile，同一 M 坐标的多个输出 tile 可能复用 A；对 B tile，同一 N 坐标的多个输出 tile 可能复用 B。TMA multicast 可以让 cluster 中相关 CTA 共享一次全局内存加载结果，从而减少重复 GMEM 流量。

但是 cluster shape 不是越大越好。它会影响：

- 一个 cluster 需要同时调度多少 CTA；
- cluster 能否在目标 GPU 上驻留；
- multicast mask 怎样构造；
- tile scheduler 以 CTA 还是 cluster 为工作单位；
- 2SM MMA 对 peer CTA ownership 的要求。

所以 `ClusterShape` 同时约束 mainloop、Kernel 和 scheduler，是“语义正交但资源耦合”的典型例子。

这里最重要的参数之一是 dispatch policy。它不是一个普通枚举，而是选择某一类 mainloop 实现，例如 TMA、多阶段流水线、warp specialization、1SM 或 2SM TCGen05 路径。

如果用户直接实例化 `CollectiveMma`，控制力很强，但需要手工给出大量低层类型。这就是 `CollectiveBuilder` 存在的原因。

请看图 4。Builder 接受的是更接近用户需求的参数。现场如果只打开一个完整实例，建议打开前面的 [示例 71](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu)，观察 `CollectiveEpilogue`、`CollectiveMainloop`、`GemmKernel` 的构造顺序：

- 架构配方与 operator class；
- A/B 和 accumulator 的数据类型；
- GMEM layout 和 alignment；
- tile shape；
- cluster shape；
- stage count；
- kernel schedule。

然后它在编译期推导出 TiledMma、TiledCopy、SMEM layout 和具体 Collective 实现。

这里可以展示一段真正值得讲的 Builder 代码，而不需要展开全部类型：

```cpp
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        sizeof(typename CollectiveEpilogue::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

现场讲这段时，不要从左到右逐项念。可以把它分成四组框起来：

```text
硬件：ArchTag + OperatorClass
数据：Element / Layout / Alignment
空间：MmaTileShape + ClusterShape
时间：StageCount + KernelSchedule
```

Builder 的输出只有一个：`CollectiveOp`。但这个输出背后已经确定了 dispatch policy、TiledMma、GMEM copy、SMEM layout 和 pipeline 类型。

这里要强调，Builder 不是运行时 autotuner。`StageCountAuto` 和 `KernelScheduleAuto` 的含义，是让 CUTLASS 在编译期根据已知参数选择一个合法、预设的实现，不是保证它一定是所有问题尺寸上的性能最优解。

在我们的例子中，还使用了 `StageCountAutoCarveout`。原因是 mainloop 和 epilogue 会共享有限的 SMEM。先确定 epilogue 的 `SharedStorage` 大小，再把剩余 SMEM 交给 mainloop 计算 stage 数，才能避免两边独立分配后超过硬件容量。

给一个具体的资源例子。这里不使用某张 GPU 的固定数字，只看关系。假设一个 CTA 可用的动态 SMEM 预算为 S，epilogue 的共享存储需要 E，每个 mainloop stage 的 A/B buffer 共需要 P，那么理论 stage 上限近似为：

```text
stages ≤ floor((S - E) / P)
```

如果我们给 epilogue 增加更复杂的融合操作，使 E 变大，Builder 可能把 mainloop 从 4 stage 降到 3 stage。Epilogue 的数学功能虽然没有直接修改 mainloop 代码，却通过共享资源改变了 mainloop 的可选计划。

这正是 `StageCountAutoCarveout` 的含义：不是简单选择“越多越好”，而是在扣除 epilogue carveout 后选择合法 stage 数。

#### 现场源码 walkthrough：示例 71

这一段建议实际打开示例 71，按搜索而不是按行号浏览。

第一步搜索 `CollectiveEpilogue`。指出 CUTLASS 先实例化 epilogue，因为 mainloop 的 stage carveout 需要知道 epilogue `SharedStorage`。

第二步搜索 `CollectiveMainloop`。让听众看到 `StageCountAutoCarveout` 如何引用刚才得到的 `CollectiveEpilogue::SharedStorage`。

第三步搜索 `GemmKernel`。指出这里没有重新描述 MMA 和 TMA，只是组合两个已经完整定义的 collective。

第四步搜索 `GemmUniversalAdapter`。说明同一个设备 Kernel 最终被包装成主机可调用句柄。

第五步如果时间允许，搜索 EVT callback 或 fusion operation，说明替换后处理不需要重新手写整条 mainloop。

这个 walkthrough 的目的不是看懂示例 71 每一行，而是在一个真实文件里验证五层组合链确实存在。

因此 Builder 的价值可以概括为：

> 它把“手工拼装低层类型”变成“根据高层约束在编译期推导低层类型”。

### 28～34 分钟：Collective Epilogue、EVT 与数据类型边界

请看图 5。GEMM 的 epilogue 不是简单地把 accumulator 写回 GMEM。

对于 AI 工作负载，GEMM 后面通常还会有：

```text
D = activation(alpha × AB + beta × C)
```

这里可能涉及缩放、读取 C、bias、activation、类型转换和最终 store。如果这些操作拆成多个 kernel，就会产生额外的全局内存流量。

Collective Epilogue 的作用，就是把输出搬运和逐元素后处理组织成另一个时间微内核。

先用一个具体例子说明为什么要区分多种数据类型。假设 A、B 是 FP16，Tensor Core 使用 FP32 accumulator，最终 D 仍然写成 FP16，那么 epilogue 至少涉及三种角色：

```text
ElementAccumulator = float   // MMA 累加结果
ElementCompute     = float   // alpha/beta、bias、activation 的计算类型
ElementD           = half    // 最终写回类型
```

`ElementAccumulator` 决定 mainloop 交给 epilogue 的输入；`ElementCompute` 决定后处理计算精度；`ElementD` 决定输出存储格式。三者可以相同，也可以不同。

例如，把 FP32 accumulator 直接截断为 FP16，与先在 FP32 中执行 `alpha * accumulator + beta * C`、再做 activation、最后舍入到 FP16，数值语义并不相同。Epilogue Builder 必须把这种数据类型路径表达清楚。

它通常需要指定：

- C 和 D 的数据类型与布局；
- accumulator 和中间计算类型；
- epilogue tile；
- GMEM store 和 SMEM staging 使用的 TiledCopy；
- epilogue schedule；
- fusion operation。

CUTLASS 使用 Epilogue Visitor Tree，也就是 EVT，把多个后处理操作组合起来。它的价值不是“多写几层模板”，而是把后处理表达成可组合的数据流图，让 CUTLASS 在同一个 epilogue 中完成计算和写回。常见融合节点定义在 [`operations.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/epilogue/fusion/operations.hpp)，Blackwell warp-specialized EVT callback 配方位于 [`sm100_callbacks_tma_warpspecialized.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp)。

举两个具体的组合例子。

第一个是普通线性组合：

```text
D = alpha × accumulator + beta × C
```

第二个是在同一条路径上增加 bias 和 ReLU：

```text
D = ReLU(alpha × accumulator + beta × C + bias)
```

从算法角度看，第二个例子只是增加后处理节点；从 kernel 角度看，如果另起 kernel，就需要先把中间结果写到 GMEM，再读回来执行 bias 和 ReLU。EVT 的目标是在 epilogue 中直接完成这些节点。

因此，“替换 epilogue，而不重写 mainloop”是正交组合的一个真实例子。A/B 的 TMA pipeline、TCGen05 MMA 和 tile scheduler 可以保持不变，变化集中在 accumulator 如何转成 D。

在 Blackwell 上还要注意 accumulator 位于 TMEM。Epilogue 不能把它简单当成普通线程寄存器数组；具体实现需要按照 TiledCopy 把 TMEM 数据装入线程可处理的 fragment，再执行融合计算与 store。图 5 中从 TMEM accumulator 到 EVT 再到 D 的路径，表达的就是这个过程。

这里可以问听众：如果我们把 `ElementD` 从 FP32 改成 FP16，会不会改变 MMA 指令？通常 MMA 和 accumulator 路径可以保持不变，主要变化发生在 epilogue 的计算、舍入和 store。但如果资源占用或 schedule 合法性发生变化，Builder 仍可能选择不同实现。

#### 具体例子：只替换 fusion operation

假设示例 71 已经构造出一个合法的 `CollectiveMainloop`。第一版 epilogue 使用普通 `LinearCombination`；第二版改成带 activation 的 EVT callback。我们希望保留：

```text
ArchTag
MmaTileShape
ClusterShape
CollectiveMainloop
GemmUniversal 的问题遍历方式
```

只替换：

```text
EpilogueOperation / FusionCallbacks
CollectiveEpilogue
```

随后重新实例化 `GemmKernel`。这就是“低层对象可复用，高层组合点可替换”的具体含义。

在 Thor/SM110a 路径中，C++ Builder 复用的是 Blackwell `sm100_callbacks_tma_warpspecialized.hpp` 中的 EVT callback 配方，而不是 Hopper 的 `sm90_callbacks...`。

这再次说明，阅读 CUTLASS 源码时不能只看文件名中最大的架构数字，要沿着 Builder、dispatch policy、编译目标和最终 SASS 一起判断。

### 34～43 分钟：Kernel、tile scheduler、CLC 与 Stream-K

Collective 已经定义了一个 CTA 或 cluster 如何用一条 pipeline 处理一个 work tile。接下来，Kernel 层要解决这些 work tile 如何覆盖整个问题空间。

请看图 6。[`GemmUniversal`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/kernel/gemm_universal_decl.h) 把 `CollectiveMainloop` 和 `CollectiveEpilogue` 组合成一个设备函数，并提供统一的参数接口、grid/block 计算、workspace 需求和 `operator()` 实现。Blackwell TMA warp-specialized 的具体 Kernel 实现可以在 [`sm100_gemm_tma_warpspecialized.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp) 中查看。

最小组合形式非常简单：

```cpp
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;
```

这里 rank-4 的问题 shape 表示 M、N、K、L，其中 L 可以表示 batch。若使用 rank-3 shape，则表示普通非批处理 GEMM。`GemmUniversal` 自身不保存某一次调用的 A/B 指针或 M/N/K，因此说它“无状态”；这些运行时信息稍后由 `Arguments` 提供。

这个例子也说明 Kernel 层并没有重新定义 mainloop。它只要求两个 collective 的接口能够正确衔接：mainloop 产生的 accumulator 必须能被 epilogue 消费，tile shape、cluster shape 和 schedule 必须彼此兼容。

这里需要区分两个经常混淆的概念：kernel schedule 和 tile scheduler。

kernel schedule 决定一个 collective 内部怎样执行，例如使用哪种 TMA/TCGen05 pipeline、1SM 还是 2SM、是否 warp specialized。

tile scheduler 决定不同 collective 之间怎样分配整个输出矩阵中的工作 tile。

一句话概括：

> kernel schedule 管内部，tile scheduler 管全局分工。

请看图 7。最基本的是 DataParallel：一份输出 tile 分配给一个 CTA 或 CTA cluster。

给一个具体的 wave quantization 例子。假设 GPU 同时可以运行 8 个工作单位，而问题一共有 17 个输出 tile。

DataParallel 会形成三波：

```text
第 1 波：8 个 tile
第 2 波：8 个 tile
第 3 波：1 个 tile
```

最后一波只有一个工作单位在运行，其他计算资源空闲。这就是 wave quantization 的直观表现。

如果每个 tile 的工作量完全相同，persistent scheduler 不能消除总工作量，但可以减少反复启动和交接成本，并允许驻留的 CTA/cluster 连续领取工作。如果 tile 工作量不同，动态领取还可以改善静态分配造成的负载不均衡。

另一类是 persistent scheduler。它不会为每个工作 tile 都依赖一次新的 CTA 启动，而是让已经驻留的 CTA 或 cluster 连续领取工作。在 Blackwell 上，CUTLASS 可以利用 Cluster Launch Control，也就是 CLC，通过硬件的 query/cancel 机制取得尚未启动 cluster 的工作。对应实现可查看 [`PersistentTileSchedulerSm100`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp) 和 [Blackwell CLC 文档](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/media/docs/blackwell_cluster_launch_control.md)。

CLC 和“所有 CTA 对同一个全局原子计数器做 fetch-add”不是一回事。后者是软件实现的动态调度，CLC 则利用 Blackwell 的硬件启动控制能力。

例如，预定 grid 中某个 cluster 还没有真正启动，已经驻留的 cluster 可以通过 CLC 尝试取消这个待启动 cluster，并取得它原本负责的 tile 坐标。这样，第一波 cluster 可能持续执行更多工作，而不需要所有预定 cluster 都实际启动。

但 CLC 也不是无成本的万能负载均衡器。调度请求本身有延迟；pipeline 预取过多工作可能又让不同 cluster 的队列重新失衡；并发 kernel 和抢占也会影响取消是否成功。因此 CLC 的 stage 数和请求时机仍然是调度设计的一部分。

Stream-K 解决的是另一个问题：wave quantization 和负载不均衡。当 M、N 方向的输出 tile 数不能很好地填满所有 SM 时，Stream-K 可以把部分工作沿 K 维拆开，让多个 CTA 协作完成同一个输出 tile。它不是 Hopper 独占算法；CUTLASS 已经提供 [Blackwell SM100 示例 74](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/74_blackwell_gemm_streamk/blackwell_gemm_streamk.cu) 和 [`PersistentTileSchedulerSm100StreamK`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp)。

但这里必须区分 SM100 Blackwell 与 Thor/SM110a。当前 CUTLASS 源码中的示例 74明确以 SM100 为例；其 CUDA 13 运行时提示提到 compute capability 100 或 110，但示例目录的 `CMakeLists.txt` 只为 `100a/100f/101a/101f/103a/103f` 添加可执行目标，没有包含 `110a/110f`。因此，该示例证明 CUTLASS 有 Blackwell Stream-K 实现，但不能单独证明 Thor/SM110a 已经开箱可用。

如果要在 Thor 上复用，应把它视为待验证工程路径：显式以 `compute_110a/sm_110a` 实例化和编译，检查 Builder 与 `ptxas`，核对 SM110a SASS，再做数值与并发归约测试。演讲中不要把 SM100 示例的存在直接表述为 SM110a 的硬件验证结果。

但沿 K 拆分意味着会产生部分和，因此还需要：

- workspace；
- 锁或同步状态；
- reduction；
- 确定性或非确定性归约策略。

继续用刚才 17 个 tile 的例子。假设每个完整 tile 沿 K 方向有很长的循环。Stream-K 可以不把最后一波只留给一个 CTA，而是把一部分 K 范围分给原本空闲的 CTA。多个 CTA 分别计算同一个输出 tile 的部分和，再做归约。

这样做的收益是提高尾部利用率，代价是：

- 需要写入和读取 partial accumulator；
- CTA 之间需要同步；
- 归约顺序可能影响浮点可重复性；
- 小问题上归约开销可能超过负载均衡收益。

因此 CUTLASS 提供 Heuristic 模式，根据问题形状选择 DataParallel、SplitK 或 StreamK。这里的 Heuristic 仍然不是对所有设备和问题的性能证明，而是一套内置决策规则。

CUTLASS 的公共 `StreamKScheduler` 只是把这种调度能力接到 `GemmUniversal` 上。它不会自动替换前面的 TMA/TCGen05 mainloop，也不会把 1SM kernel 自动变成 2SM kernel。

这就是 CUTLASS 3.x 所说的正交性：

> mainloop 决定一个 tile 怎么算，epilogue 决定结果怎么处理，tile scheduler 决定这些 tile 分给谁。

三者可以组合，但职责不同。

#### 具体例子：保持 mainloop 不变，只接入 Stream-K

普通 `GemmUniversal`：

```cpp
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;
```

接入公共 Stream-K scheduler 后：

```cpp
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StreamKScheduler
>;
```

这里没有替换 `CollectiveMainloop` 和 `CollectiveEpilogue`。变化发生在 Kernel 的第四个模板参数，以及运行时 scheduler arguments 和 workspace。

这就是演讲时最值得展示的“正交替换”例子：计算一个 tile 的方法不变，变化的是不同 collective 如何分配 tile 和 K 范围。

#### 现场源码 walkthrough：示例 74

打开示例 74 后，先搜索 `KernelTmaWarpSpecialized2SmSm100`，确认该实例显式使用 2SM mainloop，而不是由 `StreamKScheduler` 自动决定 2SM。

再搜索 `StreamKScheduler`，确认它作为 `GemmUniversal` 的 tile scheduler 参数出现。

然后搜索 `scheduler_args` 或 decomposition mode，指出 DataParallel、SplitK、StreamK、Heuristic 是运行参数的一部分。

最后搜索 `get_workspace_size`，让听众看到 Stream-K 不只是改变 grid 次序，还可能需要用于 partial result、锁和归约的 workspace。

示例 71 和 74 放在一起看，可以形成非常清楚的对比：示例 71重点展示 Builder 和 EVT 的组合；示例 74重点展示 tile scheduler 和归约路径。

### 43～47 分钟：Device 层与启动生命周期

最后看图 8。到了 Device 层，我们已经有了一个完整 `GemmKernel`，但它还是设备端类型。主机需要一个稳定的使用入口，这就是 [`GemmUniversalAdapter`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/device/gemm_universal_adapter.h)。

标准生命周期可以分成四步。

第一步，构造 `Arguments`。这里包含问题 shape、A/B/C/D 指针和 stride、alpha/beta、硬件信息以及 scheduler 参数。

可以展示下面这个简化实例：

```cpp
Arguments args {
    cutlass::gemm::GemmUniversalMode::kBatched,
    cute::make_shape(M, N, K, L),
    {A, stride_A, B, stride_B},
    {{alpha, beta}, C, stride_C, D, stride_D},
    hw_info,
    scheduler_args
};
```

注意，编译期 kernel 类型没有包含某一次运行的 M/N/K 和指针；这些都通过 `Arguments` 进入。这是“编译期物理计划”和“运行时问题实例”的分界。

第二步，调用 `can_implement`。这一步不能省略。它检查当前问题的 alignment、shape、数据类型和硬件能力是否满足已选 kernel 的约束。

例如，Builder 按 128-bit 对齐构造了 FP16 A/B load，因此 `AlignmentA=AlignmentB=8`。如果传入的 A 指针只满足 2 字节自然对齐，或者 leading dimension 不是 8 个 FP16 元素的倍数，那么 `can_implement` 可能返回失败。

另一个例子是 2SM kernel 对 cluster shape 有额外要求。如果问题 shape、tile shape 或启动配置不能形成合法 CTA pair，`can_implement` 也应该拒绝，而不是带着不合法配置进入 kernel。

现场建议把失败处理也展示出来：

```cpp
cutlass::Status status = GemmHandle::can_implement(args);
if (status != cutlass::Status::kSuccess) {
  std::cerr << "problem not supported by this kernel\n";
  return;
}
```

不要把这一步讲成普通防御式编程。它是运行时参数与编译期 kernel 合同之间的正式检查点。

第三步，查询并分配 workspace，再调用 `initialize`。对于普通 DataParallel GEMM，workspace 可能很小；对于 Stream-K 或其他需要归约和全局同步的路径，workspace 会参与算法正确性。

一个常见错误是只在第一次运行时按某个问题 shape 分配 workspace，后面换了更大的 M/N/K 或不同 decomposition mode，却继续复用旧大小。正确流程是根据当前 `Arguments` 重新查询 `get_workspace_size`，或者明确证明已分配容量覆盖后续调用。

第四步，调用 `run`，必要时传入 CUDA stream 或 host adaptor。

`initialize` 和 `run` 也要区分。`initialize` 把 `Arguments` 转换为 kernel 参数、准备 workspace 状态；`run` 才真正发起 kernel。可复用句柄的价值在于可以更新参数后多次运行，但必须遵守该实现对初始化和 workspace 的要求。

#### 具体失败路径

假设代码能够编译，但 `can_implement` 返回失败，我们首先检查 alignment、leading dimension、tile 整除关系、cluster shape 和数据类型支持，而不是立刻怀疑 Tensor Core 指令。

假设 `can_implement` 成功、`initialize` 失败，则重点检查 workspace 地址、容量、硬件信息和 scheduler 参数。

假设 launch 成功但数值错误，则进入另外一条验证路径：输入布局、stride、alpha/beta、epilogue 类型、pipeline 同步和目标 SASS。不能用 `can_implement` 成功来替代数值测试。

假设数值正确但性能很差，才进入性能诊断：tile shape、stage 数、occupancy、TMA 命中、cluster 调度、wave quantization 和 epilogue 开销。

这四类失败处在不同层次，按层诊断会比在整个模板栈中盲目搜索更有效。

所以 Device 层的主要价值是：

> 把一个复杂的编译期 kernel 类型，封装成可检查、可初始化、可复用的主机端对象。

### 47～50 分钟：验证阶梯与总结

在总结之前，我想单独给出一条 SM110a 验证阶梯。因为在跨架构复用 CUTLASS 配方时，“能写出类型”与“硬件上正确高效”之间还有很长距离。

第一层是 Builder 和类型系统。

我们检查模板是否能够实例化，`static_assert` 是否通过，A/B/C/D layout、alignment、tile shape、cluster shape 和 schedule 是否构成合法组合。这一层证明的是 CUTLASS 的建模合同成立。

第二层是编译器中间结果和 PTX。

确认翻译单元确实使用 `compute_110a`，检查 PTX 中是否出现预期的 TMA、TCGen05 和同步语义。这里可以发现宏条件、架构分支或 intrinsic 选择错误。

第三层是 `ptxas` 和函数范围内的 SASS。

只看到整个二进制中存在某条指令还不够，需要把 SASS 归属到目标 kernel 函数，确认 mainloop 和 epilogue 真正生成了预期的加载、MMA、commit/wait 和 store 路径。

第四层是目标硬件数值测试。

使用固定随机种子、明确 shape/stride/alpha/beta，与独立参考实现比较。需要覆盖：

- 首个和非首个 K tile，检查 accumulate 语义；
- 边界 shape 和 alignment；
- 1SM 与 2SM 路径；
- 不同 epilogue 和输出类型；
- Stream-K 需要归约的情况。

第五层才是性能证据。

性能分析应区分：

- Tensor Core 计算吞吐；
- TMA/GMEM/SMEM 数据供给；
- stage 和 occupancy；
- epilogue 开销；
- tile scheduler 的负载均衡；
- 全 GEMM 结果，而不是单个微基准峰值。

举一个完整闭环的例子：

```text
示例 71 类型实例化成功
  → 使用 compute_110a 编译
  → ptxas 生成 SM110a code
  → kernel 函数内出现预期 TCGen05/TMA 指令
  → Thor 上与参考 GEMM 数值一致
  → 最后测量不同 M/N/K 下的吞吐和稳定性
```

只有走完这条链，才能从“源码配方可复用”推进到“目标 kernel 在 Thor 上被验证”。

这一段也能回答为什么文章会同时提到 Builder、PTX/SASS 和实机：它们不是重复验证，而是在证明不同层次的命题。

最后总结三个结论。

第一，CUTLASS 3.x 的复杂性来自问题本身的模块化，而不是为了模板而模板。

五层分别回答：

```text
Atom               执行什么指令
Tiled MMA/Copy      怎样定义 work tile 内部的空间划分
Collective          怎样把 TMA/copy 与 MMA 组装成 work-tile pipeline
Kernel              怎样覆盖完整问题并分配工作 tile
Device              怎样从主机检查、初始化和启动
```

第二，要区分空间微内核和时间微内核。

CuTe 的 Atom、Layout、TiledMma 和 partition 主要解决 work tile 内线程与数据的空间组织；CUTLASS Collective 接收 work-tile `TileShape`，进一步把 TMA/TiledCopy、SMEM stage、MMA、barrier、warp specialization 和 epilogue 组装为时间 pipeline。

第三，源码实现复用不等于目标架构相同，更不等于运行时正确。

在 Thor/SM110a 上，`cutlass::arch::Sm100` 可以作为 TCGen05 源码配方标签，但最终编译目标必须是 `compute_110a/sm_110a`，并且还要经过 Builder、`ptxas`、SASS 和实机数值验证。

如果今天只记住一句话，我希望是：

> 阅读 CUTLASS 时，不要先问“这个模板怎么背”，而要先问“这一层负责哪一种决策，它组合了哪些下层对象，又把什么接口交给上一层”。

我的分享就到这里，谢谢大家。

## 三、现场讲解建议

### 1. 不要逐行读模板

展示代码时，只标出真正需要听众关注的参数：

```text
ArchTag / OperatorClass
ElementA / ElementB / ElementAccumulator
TileShape / ClusterShape
StageCount
KernelSchedule
```

其余参数告诉听众“它们属于操作数布局、对齐或 epilogue 配置”即可。逐个读模板参数会迅速消耗听众注意力。

### 2. 每讲一层，都回到同一个问题

建议反复使用下面五句话：

```text
Atom：我选哪条硬件指令？
Tiled：一个 work tile 内，Atom、线程和值怎样划分？
Collective：怎样把 TMA/copy、SMEM stage 和 MMA 组装成 pipeline？
Kernel：怎样把 work tile 分配到整个问题？
Device：我怎样从主机启动它？
```

重复不是冗余，而是在帮助听众稳定五层心智模型。

### 3. `Sm100` 与 `sm_110a` 要单独停顿说明

这是最容易被问到、也最容易引起误解的地方。建议把这句话单独放在一页：

```text
Sm100：CUTLASS C++ 源码配方族
sm_110a：最终 CUDA 编译目标
```

随后立即补充：配方复用仍然需要静态编译、SASS 和硬件数值验证。

### 4. Tile scheduler 不要与 kernel schedule 混讲

可以使用这个对比：

```text
kernel schedule：一个厨房内部怎样备料和烹饪
tile scheduler：不同厨房分别领取哪些订单
```

如果希望完全避免类比，也可以直接说：

```text
kernel schedule：collective 内部
tile scheduler：collective 之间
```

### 5. 把性能承诺说清楚

分享中不要说 `Auto` 一定选择最优 kernel，也不要说源码复用就证明 SM110a 正确。更准确的表达是：

- Builder 负责推导合法的预设实现；
- Auto 是编译期启发式选择，不是全问题空间 autotuning；
- `can_implement` 只证明参数满足 kernel 约束，不证明性能最优；
- SASS 证明目标指令确实生成，但不单独证明数值正确；
- 最终正确性仍需要目标硬件上的独立数值测试。

## 四、CUTLASS 官方源码与实例索引

### 1. 分享时最值得打开的两个完整实例

| 优先级 | CUTLASS 实例 | 分享时重点观察 |
|---:|---|---|
| 1 | [示例 71：Blackwell GEMM with Collective Builder](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu) | `CollectiveEpilogue → CollectiveMainloop → GemmKernel → GemmUniversalAdapter` 的完整组合顺序；自定义 EVT；SM100/TCGen05 配方 |
| 2 | [示例 74：Blackwell SM100 GEMM Stream-K](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/74_blackwell_gemm_streamk/blackwell_gemm_streamk.cu) | 显式 2SM mainloop、公共 `StreamKScheduler`、scheduler arguments、workspace 与归约模式；它是 SM100 Blackwell 实例，不是 SM110a 开箱支持证明 |

这两个实例都固定在 CUTLASS commit `e05f953a5b3d38adc240df2ff928e0421c2abba3`，适合技术分享时复现，不依赖 `main` 分支后续变化。

### 2. 按五层结构查找源码

| 层次或主题 | 具体源码链接 | 应该看什么 |
|---|---|---|
| CUTLASS 3.x 总体层次 | [`gemm_api_3x.md`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/media/docs/gemm_api_3x.md) | Atom、Tiled、Collective、Kernel、Device 的官方职责定义 |
| CuTe MMA Atom/TiledMMA 文档 | [`0t_mma_atom.md`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/media/docs/cpp/cute/0t_mma_atom.md) | Operation、`MMA_Traits`、`MMA_Atom` 和 `TiledMMA` 怎样逐层组成 |
| CuTe Atom/TiledMMA 核心实现 | [`mma_atom.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cute/atom/mma_atom.hpp) | `MMA_Atom`、`TiledMMA`、`make_tiled_mma`、partition/fragment 接口 |
| Blackwell MMA Traits | [`mma_traits_sm100.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cute/atom/mma_traits_sm100.hpp) | `SM100_MMA_*` 操作对应的 traits、线程/CTA group 和 fragment 元信息 |
| Collective 公共声明 | [`collective_mma_decl.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/collective/collective_mma_decl.hpp) | `CollectiveMma` 的模板组成和低层组合点 |
| GEMM dispatch policy | [`dispatch_policy.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/dispatch_policy.hpp) | Mainloop dispatch policy、架构族和 schedule 类型 |
| Collective 实现目录 | [`include/cutlass/gemm/collective`](https://github.com/NVIDIA/cutlass/tree/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/collective) | 不同架构和数据通路的 collective 实现族 |
| Epilogue Builder 公共接口 | [`collective_builder.hpp`](https://github.com/NVIDIA/cutlass/blob/62750a2b75c802660e4894434dc55e839f322277/include/cutlass/epilogue/collective/collective_builder.hpp) | Epilogue Builder 的统一模板接口 |
| 常见 EVT 操作 | [`operations.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/epilogue/fusion/operations.hpp) | LinearCombination、activation 等常见融合节点 |
| Blackwell EVT callback | [`sm100_callbacks_tma_warpspecialized.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp) | Blackwell TMA warp-specialized epilogue 的 callback 配方 |
| Kernel 公共入口 | [`gemm_universal_decl.h`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/kernel/gemm_universal_decl.h) | `GemmUniversal` 如何组合 mainloop、epilogue 和 scheduler |
| Blackwell GEMM Kernel | [`sm100_gemm_tma_warpspecialized.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp) | 具体 `operator()`、warp role、pipeline 与 kernel 执行逻辑 |
| Persistent/CLC scheduler | [`sm100_tile_scheduler.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp) | `PersistentTileSchedulerSm100` 和 CLC query/cancel 路径 |
| CLC 官方说明 | [`blackwell_cluster_launch_control.md`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/media/docs/blackwell_cluster_launch_control.md) | Blackwell CLC 的概念、启动控制和调度动机 |
| Stream-K scheduler | [`sm100_tile_scheduler_stream_k.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp) | DataParallel、SplitK、StreamK、Heuristic 与 reduction/workspace |
| Device Adapter | [`gemm_universal_adapter.h`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/device/gemm_universal_adapter.h) | `can_implement`、workspace、initialize、run 的主机端封装 |

### 3. 建议的现场源码浏览顺序

如果现场只安排 3～5 分钟看代码，建议按以下顺序打开：

1. 先看[示例 71](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu)，在一个文件里指出五层组合关系；
2. 跳到 [`sm100_gemm_tma_warpspecialized.hpp`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp)，说明 Builder 最后会落到具体 Kernel 实现；
3. 再看[示例 74](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/74_blackwell_gemm_streamk/blackwell_gemm_streamk.cu)，对比只更换 tile scheduler 后新增的 scheduler arguments 和 workspace；
4. 最后打开 [`gemm_universal_adapter.h`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/include/cutlass/gemm/device/gemm_universal_adapter.h)，回到 `can_implement → initialize → run` 的主机生命周期。

不要在现场从头阅读大型实现文件；只用搜索定位 `CollectiveBuilder`、`GemmUniversal`、`StreamKScheduler`、`can_implement` 和 `operator()` 等关键符号。

## 五、高概率问答

### Q1：为什么目标是 SM110a，`ArchTag` 还写 `Sm100`？

因为当前 CUTLASS C++ Builder 复用 `Sm100` 命名的 Blackwell TCGen05 实现族。它是源码配方选择，不是最终二进制目标。最终目标由 `-gencode arch=compute_110a,code=sm_110a` 决定。是否真正可用，还要继续检查 Builder、`ptxas`、SM110a SASS 和 Thor 数值结果。

### Q2：`KernelScheduleAuto` 是否一定选择最快方案？

不是。它根据 CUTLASS 内置规则和已知编译期参数选择一个合法的预设 schedule。它不是运行时 autotuner，也不能保证对所有 M/N/K 和数据分布都最优。

### Q3：Collective 与 Kernel 的本质区别是什么？

Collective 接收 work-tile `TileShape`，描述一个 CTA 或 cluster 如何通过 load/MMA pipeline 完成这个 tile；Kernel 把 work tile 扩展到整个问题空间，并接入 tile scheduler 和 grid 级资源管理。

### Q4：TiledMma 与 CollectiveMma 有什么区别？

`TiledMma` 描述一个 work tile 内 MMA Atom 的重复、交织以及线程—数据 partition；完整 work-tile shape 由 Collective 的 `TileShape` 合同给出。`CollectiveMma` 再把 TMA/TiledCopy、SMEM layout、stage、同步和 TiledMma 执行组装成处理该 work tile 的 pipeline。

### Q5：CLC 与普通 persistent kernel 的全局原子计数器有什么区别？

普通软件动态调度通常让 CTA 对全局计数器执行原子操作来领取工作；Blackwell CLC 使用硬件的 cluster launch query/cancel 机制，让已启动 cluster 接管尚未启动 cluster 的工作。两者目标相似，但实现路径和硬件支持不同。

### Q6：Stream-K 为什么需要 workspace？

因为 Stream-K 可能把一个输出 tile 的 K 范围分给多个 CTA。各 CTA 产生部分和，需要通过 workspace、锁和归约协议合并，最后才能得到完整输出。

### Q7：Epilogue Visitor Tree 的价值是什么？

它把缩放、C 读取、bias、activation、类型转换和 store 表达成可组合后处理图，使这些操作可以融合进 GEMM epilogue，避免额外 kernel 和 GMEM 往返。

### Q8：`can_implement` 成功是否说明 kernel 正确？

它只说明当前参数满足已选 kernel 的静态和运行时约束，例如 shape、alignment、数据类型和硬件能力。它不证明编译器生成完全符合预期，也不证明数值正确或性能最优。

### Q9：如何验证 SM110a 路径？

建议按以下顺序：

1. Builder 和静态断言；
2. 编译到 typed IR/PTX；
3. `ptxas` 成功；
4. 函数范围内核对 SM110a SASS；
5. Thor 实机数值对照；
6. 最后再做性能分析。

### Q10：这套五层结构只适用于 GEMM 吗？

文章中的具体类型围绕 GEMM，但“硬件原语 → 空间微内核 → 时间微内核 → grid 级 kernel → 主机接口”的分层思想具有更广泛适用性。FlashAttention 等复杂 GPU kernel 也会复用 CuTe 的布局、Atom 和 tiled 抽象，不过其高层 collective 和调度逻辑会有所不同。

### Q11：Stream-K 是不是只有 Hopper 才有，Blackwell 已经没有了？

不是。Stream-K 是 GEMM 工作分解和负载均衡方法，不是 Hopper 特有指令。CUTLASS 有明确的 Blackwell SM100 实现：示例 74 使用 `StreamKScheduler`，底层映射到 `PersistentTileSchedulerSm100StreamK`。但这不能自动外推到 Thor/SM110a：当前示例的 CMake 构建门禁没有列出 110，且缺少独立的 `arch::Sm110` Stream-K selector。Thor 路径可以研究复用 `arch::Sm100` scheduler，但必须重新经过编译、SASS、数值和归约并发验证，不能仅凭运行时提示字符串宣布已支持。

## 六、10 分钟压缩版取舍

如果现场只有 10 分钟，建议只讲：

1. GPU GEMM 优化为什么需要模块化；
2. 五层分别回答什么问题；
3. `Sm100` 配方标签与 `sm_110a` 编译目标的区别；
4. Collective Mainloop、Epilogue、Kernel scheduler 三者的职责；
5. Device 层的 `can_implement → workspace → initialize → run` 生命周期；
6. 最后三条总结。

删去：

- 完整 Builder 模板参数；
- EVT callback 文件细节；
- 示例 71、74 的源码路径；
- Stream-K 的 reduction mode 细节；
- 大部分问答，只保留 Q1、Q2、Q3。
