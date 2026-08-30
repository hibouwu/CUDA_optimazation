# CUTLASS CUDA Core / SIMT GEMM Codegen 形式化合同验证

CUTLASS 不只能够生成 Tensor Core kernel。把 `OperatorClass` 设为
`cutlass::arch::OpClassSimt` 后，同一套 `MMA_Atom → TiledMMA → Collective → Kernel`
抽象也可以组织 CUDA Core FMA。区别在于，Tensor Core 用一条矩阵指令完成一块计算，
SIMT 路径则把标量或向量 FMA 分给线程，再由 warp 和 CTA 共同覆盖输出 Tile。

分析从 CUTLASS 的高层 GEMM 配置出发，逐层展开 Builder、Collective、`TiledMMA` 和
FMA Atom，最后落到 `sm_110a` 的 PTX/SASS。要回答的问题是：用户在高层设置的数据
类型、Layout、CTA Tile、Stage 和 Schedule，分别选择了什么实际数据路径，又对应哪些
CUDA Core 搬运、计算、同步和写回指令。

```text
GemmUniversal / CollectiveBuilder 配置
→ Builder 偏特化与 DispatchPolicy
→ CollectiveMainloop / CollectiveEpilogue
→ Stage、Copy、Layout 与 TiledMMA
→ FMA Atom / Wrapper
→ 目标 PTX .entry
→ 目标 SASS function
```

类型展开和函数级 PTX/SASS 归属是证明这组高低层对应关系的方法。这里的“形式化合同
验证”以静态代码生成为边界；Thor launch、数值校验和性能实验分别提供后续证据。

> 阅读基线：目标固定为 `compute_110a/sm_110a`，CUTLASS 固定为
> `e05f953a5b3d38adc240df2ff928e0421c2abba3`。当前仓库尚未建立 SIMT 实例，因此下面各层
> 的结构已经确定，具体类型和指令结果仍为 `NOT_CHECKED`。首个贯穿实例将是一份 FP32
> dense SIMT GEMM；只有在固定 submodule 上重新编译和反汇编后，才填入正式结果。

## 1. 高层配置与底层指令的对应关系

一份 SIMT GEMM 源码能够通过编译，只说明模板组合没有在当前工具链中立即失败。还需要
沿着高层配置向下确认：Builder 命中了哪条偏特化，生成了什么 `CollectiveMma` 和
`TiledMMA`，最后由哪个 wrapper发出 FMA。声明配置、解析类型、目标 PTX 和目标 SASS 必须
绑定到同一个 kernel symbol，才能说明某个高层选项确实对应了当前看到的底层指令。

```text
Declared Gemm / Builder Config
→ Builder Specialization / DispatchPolicy
→ Collective / TiledMMA / Atom
→ CompileArtifact
→ PtxFunction
→ SassFunction
→ STATIC_PASS
```

失败也需要有明确位置。Builder 预期拒绝、NVCC 失败、PTXAS 失败和目标函数归属失败分别
记录，而不是合并成一个 `FAIL`。这样后续看到缺失 opcode 时，才能判断是实例根本没有生成，
还是检查工具抓错了函数。

## 2. CUDA Core FMA Atom

SIMT 路径的最低计算单元是线程执行的 FMA operation；矩阵级 TCGen05 属于另一条 Tensor
Core 路径。CuTe 使用 `MMA_Atom` 包装 FMA，并通过 `MMA_Traits` 给出 A、B、C、D 的值类型、寄存器合同和逻辑
`Shape_MNK`。这里的 shape 只表示一次 Atom 调用覆盖多少个逻辑乘加位置，不应与 Tensor
Core instruction shape混为一谈。

这一节最终要得到一个具体合同：当前 `OpClassSimt` 实例命中了哪个 FMA Atom，Atom 从哪些
寄存器读取 A/B/C、把 D 写到哪里，以及源码中的调用最终生成哪类 PTX FMA。目标 SASS
必须在本实例对应的函数中走 CUDA Core 计算路径；同一 binary 中其他 kernel 的
`tcgen05.mma` 与本实例无关。

正式实验将保存下面这些结果：

| 对象 | 需要保存的数据 |
|---|---|
| Builder 输入 | Element、Layout、Alignment、CTA Tile、Stage、Schedule |
| FMA Atom | operation 类型、`Shape_MNK`、A/B/C/D register contract |
| PTX | 目标 `.entry` 中的 FMA required/forbidden 结果 |
| SASS | 目标 function 中的 CUDA Core 指令族与可选计数 |

目前这一层没有固定 commit 下的 type dump 和函数产物，因此结论保持 `NOT_CHECKED`。

## 3. 从线程级 FMA 到 CTA Tile

Atom 只描述一次线程级计算，还不足以覆盖 GEMM 输出块。`TiledMMA` 在 Atom 之上加入
Thread Layout 和 Value Layout：先决定每个线程持有哪些 A/B/Accumulator fragment，再把
线程组织成 warp，最后让多个 warp 覆盖 CTA Tile。沿这条链可以同时看到“单次 FMA 的逻辑
shape”和“一个 CTA 负责的矩阵范围”，而不会把二者写成同一个 shape。

这一节选择同一个 FMA Atom，至少构造两个合法 CTA Tile。对每个实例，文档会给出线程到
数据的映射、Warp Shape、CTA Shape 和逻辑 FMA 数，并与实际 PTX/SASS 统计对照。逻辑覆盖
数用于解释空间分区；最终指令条数会受到循环、展开、谓词化和 ptxas 调度影响，因此作为
实验数据保存，不进入跨版本恒定的 gate。

## 4. SIMT Mainloop 的数据路径

`TiledMMA` 确定了“谁计算哪些元素”，`CollectiveMainloop` 再补上“数据怎样到达这些线程”。
对于 SIMT 路径，需要从 `CollectiveBuilder` 的输入追踪到具体 `DispatchPolicy`、GMEM Copy、
SMEM Layout、SMEM→RMEM Copy 和 `CollectiveMma`。完整主路径应当能够写成：

```text
GMEM A/B
→ 异步或普通全局加载
→ SMEM A/B
→ 线程寄存器 Fragment
→ CUDA Core FMA
→ 寄存器 Accumulator
```

源码可能复用名称中带有旧架构标识的 DispatchPolicy；这只说明 CUTLASS 复用了实现。真正的
binary target仍由 `compute_110a/sm_110a` 和目标函数反汇编确认。本节结束时应当拿到
一份确定的 `CollectiveMma<...>` 类型，并能指出每个关键模板参数由用户指定、Builder 推导
还是偏特化固定。

当前 guide 还没有 SIMT `CollectiveBuilder` 实例，所以 Copy/Layout、DispatchPolicy 和
Mainloop 函数数据都尚未生成。

## 5. Mainloop 的 Stage 与流水线

前一节得到的是一份 A/B 空间布局。Stage 在这套布局后增加时间维度，使 Producer 可以准备
后续 K Tile，而 Consumer 同时计算已经就绪的数据。每增加一个 Stage，SMEM 中就多一份
A/B Buffer，也会增加对应的 pipeline state 和同步成本。因此，结果需要给出
`StageCountAuto`，还要记录 Builder 最终解析出的 Stage、每个 Stage 的字节数、完整
SharedStorage 和 SMEM `PIPE` 维。

SIMT 路径首先检查实际使用的同步原语。若解析后的 Mainloop 使用 `cp.async`，就记录它的
commit/wait group 和 Stage 轮转；只有源码中确实存在 barrier/phase 时，才把 barrier 与
phase 纳入合同。SASS 中重复出现的 Load/FMA 数量只作记录，因为编译器可以
保留循环，也可以展开和重排。

目前 manifest 没有 SIMT Stage 数据。第一轮会先闭合 Auto 或默认 Stage，再选择一个合法的
显式 Stage 做对照。这一节只解释代码生成和资源变化，性能差异留到后续实验。

## 6. 从寄存器累加器到 D

Mainloop 完成当前输出 Tile 的全部 K 维累加后，SIMT accumulator仍位于线程寄存器。
`CollectiveEpilogue` 接过这些 fragment，执行基础的 `alpha × Acc + beta × C`、必要的类型
转换和 D 写回。第一版只保留一个简单 LinearCombination，使主路径容易追踪；复杂 EVT 和
量化输出等到基础 Epilogue 闭合后再增加。

这一节会记录 accumulator fragment 与 Epilogue 输入的对应关系、C 是否读取、D 使用哪种
Store 路径，以及目标 PTX/SASS 中哪些片段属于 Epilogue。静态代码生成可以说明类型和路径
已经生成，但 alpha/beta、舍入和最终矩阵是否正确仍要交给 runtime correctness 实验。

## 7. 从 CTA Tile 到完整 GEMM

Mainloop 和 Epilogue共同定义了一个输出 Tile 内部的工作，`GemmUniversal` 则把它们放进
完整问题空间。这里必须区分两个名字相近的对象：Mainloop `KernelSchedule` 决定一个 Tile
内部怎样加载和计算；Kernel Tile Scheduler 决定 M/N/L/K 空间中的 Work Tile 由哪些 CTA
领取。

首个 SIMT control 使用最简单的默认或 DataParallel scheduler，先把基本链路闭合。若后续
加入 Persistent、Stream-K 或 Split-K，则分别记录 scheduler 类型、Arguments/Params、
workspace 和 fixup/reduction helper。Scheduler 不一定对应一条独有 opcode，静态验证落在
类型和函数路径。

## 8. 函数级 PTX/SASS 归属

`GemmUniversalAdapter` 把具体 `GemmKernel` 暴露给主机端，也给最终产物提供了可定位的
kernel symbol。每个编译期实例最好生成独立 executable；如果多个实例必须共存，则至少保证
symbol 稳定且唯一。检查工具先提取目标 PTX `.entry`，再提取同名 SASS function，禁止在
整个 binary 中无归属地搜索 mnemonic。

每条结果还要绑定下面的身份信息：

```text
CUTLASS commit
CUDA / nvcc / ptxas / nvdisasm version
compile command
source hash
binary hash
kernel symbol
PTX entry
SASS function
```

`can_implement`、`initialize` 和 `run` 属于动态接口，不进入本篇的 `STATIC_PASS` 判定。

## 9. 现有数据

仓库中的 `static-20260817` 静态快照覆盖了 10 个 Tensor Core/TCGen05 实例，没有独立的
`OpClassSimt` GEMM。因此，当前数据只说明“SIMT 尚未进入既有分母”，Atom、Tile、Stage
和 SASS 的具体结论仍待生成。

| Instance | Atom | CTA Tile | Stage | PTX | SASS | 当前状态 |
|---|---|---|---:|---|---|---|
| `simt_f32_control` | 待固定 commit 解析 | 待定义 | — | NOT_CHECKED | NOT_CHECKED | `NOT_CHECKED` |

第一轮实验先闭合这一行，再扩展 CTA Tile 和 Stage。它表示一条尚未生成静态证据的主路径，
不属于失败实例。

## 10. 当前结论

目前已经确定验证边界和类型链，具体 SIMT 实例的代码生成结果仍待实验确认。下一步直接在
固定 submodule 上建立 FP32 control，生成解析后的类型、
PTX 和 SASS，然后沿 Atom、TiledMMA、Collective 和 Stage 四个位置解释实际数据。

当这条主路径闭合后，结果分析主要回答三个问题：上层 Tile 是否命中了预期 FMA Atom；
Stage 与 SharedStorage 是否一致；目标函数是否始终保持 CUDA Core 主计算路径。只有这三个
问题都有函数级证据，SIMT 实例才会从 `NOT_CHECKED` 升级为 `STATIC_PASS`。
