# CUTLASS Tensor Core / TCGen05 GEMM Codegen 形式化合同验证

Tensor Core 路径把一次 GEMM 拆成两类工作。Mainloop 沿 K 维加载 A/B Tile，并用
`tcgen05.mma` 把结果累加到 TMEM；Epilogue 再把 TMEM accumulator读回寄存器，完成线性
组合或融合计算，最后写回 D。CUTLASS 的 Builder、Collective 和 Kernel 抽象，就是把这条
数据流组织成一份可实例化的 C++ 类型。

这篇文档要回答的问题很直接：在 CUTLASS 中写下一份高层 GEMM 配置后，底下实际落成了
哪些类型、数据通路和 PTX/SASS 指令？分析沿一个真实 1SM FP16 实例向下展开，依次追踪
数据类型、Layout、`MmaTileShape`、Cluster、Stage 和 Schedule 如何参与 Builder 分派。
能够生成代码的实例继续追到该实例唯一的 `sm_110a` kernel 函数；静态拒绝项则停在能够
稳定复现的失败层，不会虚构并不存在的 PTX/SASS。

```text
GemmUniversal / CollectiveBuilder 配置
→ Builder 偏特化与 DispatchPolicy
→ CollectiveMainloop / CollectiveEpilogue
→ Stage、TMA/Copy、Layout 与 TiledMMA
→ TCGen05 MMA Atom / Wrapper
→ 目标 PTX / SASS
```

贯穿实例来自 [`dense_f16_1sm_p128`](../cases/dense_f16_1sm_p128/)：A/B 使用 FP16，
Accumulator 和 D 使用 FP32，A 为 RowMajor，B 为 ColumnMajor，`MmaTileShape` 为
`128×128×64`，Cluster 为 `1×1×1`，Mainloop 使用
`KernelTmaWarpSpecialized1SmSm100`，Epilogue 使用 `NoSmemWarpSpecialized1Sm`。目标固定为
`compute_110a/sm_110a`，CUTLASS 固定为
`e05f953a5b3d38adc240df2ff928e0421c2abba3`。

这里的 `STATIC_PASS` 只表示当前实例生成了预期的函数级 PTX/SASS，而且两份产物归属于
同一个 kernel symbol。真实 Thor launch、数值正确性和性能分别属于后续实验。

## 1. 高层配置与底层指令的对应关系

代码生成由一组能够唯一决定 kernel type 的编译期配置开始。数据类型、
Layout、Alignment、MMA Tile、Cluster、Stage、Mainloop Schedule、Epilogue Schedule 和 Tile
Scheduler 共同决定 Builder 会命中哪一条路径。运行时 shape 与编译期实例分开记录：

```text
实现路径：Dense / Block-scaled / Sparse、SS / TS、1SM / 2SM
编译期实例：MmaTileShape、Stage、Epilogue、Tile Scheduler
运行样本：M/N/K/L、Stride、alpha/beta、seed
```

贯穿全文的 FP16 1SM 是一份编译期实例。改变运行时 M/N/K 不会自动产生新的
代码生成实例；改变 `MmaTileShape`、Stage 或 Scheduler 才会生成不同 kernel type。分析从
这些高层差异出发，依次记录它们命中的 Builder 偏特化、Collective、TiledMMA、MMA Atom
和最终函数指令：

```text
Declared Gemm / Builder Config
→ Builder Specialization / DispatchPolicy
→ Collective / TiledMMA / Atom
→ CompileArtifact
→ PtxFunction
→ SassFunction
→ STATIC_PASS
```

## 2. TCGen05 MMA Atom 与 Opcode

Builder 根据数值类型、操作数来源和 CTA 协作范围选择 TCGen05 MMA Atom。对于当前 FP16
SS 实例中，A/B 由 TMA 放入 SMEM，MMA 通过 SMEM descriptor读取两个操作数，累加器位于
TMEM；`cta_group::1` 表示一个 CTA 承担这次 MMA。Atom 的 instruction shape、A/B 类型、
Accumulator 类型和 PTX opcode共同构成一条 MMA 的合同。

Phase 1 已经把这条链补齐。FP16 1SM 实际解析为
`SM100_MMA_F16BF16_SS<half, half, float, 128, 128, ...>`，Atom shape 为
`128×128×16`；目标函数中有 4 条 `tcgen05.mma.cta_group::1.kind::f16`，对应 4 条不带
`.2CTA` 的 `UTCHMMA`。2SM 实例则解析为 `SM100_MMA_F16BF16_2x1SM_SS`，Atom shape 扩展为
`256×128×16`，PTX 的 4 条 MMA 全部是 `cta_group::2`，SASS 的 4 条 MMA 全部是
`UTCHMMA.2CTA`。

Block-scaled 和 Sparse 也不是从 Tag 名猜 opcode。NVFP4、MXFP4、MXFP8 与 Sparse NVFP4
分别解析出自己的 `MMA_Atom`；函数内 PTX 再独立确认 `mxf4nvf4.block_scale.block16`、
`mxf4.block_scale.block32`、`mxf8f6f4.block_scale` 和 `mma.sp...mxf4nvf4.block_scale.block16`。
Sparse 的 SASS MMA family 本身不能单独区分稀疏与非稀疏，正式结论来自
`SparseConfig + metadata layout + PTX mma.sp + 目标 SASS function` 的联合证据。

## 3. 从 MMA Atom 到 `MmaTileShape`

一条 TCGen05 MMA 只覆盖 instruction shape。`TiledMMA` 在其上增加 Atom Layout 和 CTA
slice，使一个或两个 CTA 能够共同覆盖更大的 `MmaTileShape`。当前 1SM 实例中，
`cta_group::1` 只有一个 CTA slice；2SM 扩展则会引入 peer CTA、`cta_group::2` 和不同的
Cluster/TMEM 分配关系。

这条类型链包含五种不同的 shape：instruction/Atom shape、TiledMMA shape、
`MmaTileShape`、Cluster/work tile 和运行时 problem shape。Phase 1 的六个点已经同时保存前
四类静态值：FP16 1SM 是 `128×128×16 → 128×128×64`，FP16 2SM 是
`256×128×16 → 256×128×64`；NVFP4、MXFP4、MXFP8、Sparse NVFP4 的 Atom K 分别为
64、64、32、128，对应的 MmaTile K 分别为 256、256、128、256。

这些比例解释了一个编译期 tile 如何由 Atom 沿 K 维重复覆盖，却不被当作稳定 SASS 条数公式。
当前仍只是六个离散点，尚未形成同一 family 下的完整 shape surface；后续扩展 Tile 时会继续
复用相同的 Atom/TiledMMA 记账方式。

## 4. Tensor Core Mainloop 的数据路径

`TiledMMA` 确定空间分区以后，`CollectiveBuilder` 继续选择数据路径。当前 Dense 1SM SS
实例使用 TMA 把 A/B 从 GMEM 放入多阶段 SMEM Buffer，TCGen05 再通过 descriptor直接读取
SMEM。`SmemCopyAtomA/B` 在这条路径中不承担经典的 SMEM→RMEM copy，Accumulator 则由
TMEM 持有。

固定源码和声明配置给出的候选数据流是：

```text
GMEM A/B
→ TMA
→ SMEM Stage
→ TCGen05 MMA
→ TMEM Accumulator
```

普通 Dense SS 不需要为了“指令齐全”强制出现 `tcgen05.cp`。Block-scaled 的 scale ingress
是否使用 `tcgen05.cp`，TS 操作数怎样进入 TMEM，都由各自解析后的路径决定。Sparse 还会
增加 compressed operand和 metadata。它们属于不同 Builder 分支，不能从 Dense 的指令合同
直接外推。

Fresh witness 现在同时保存 `MainloopBuilder`、`Builder::CollectiveOp`、`DispatchPolicy`、
`TiledMMA`、A/B Copy 与 SMEM Layout。Dense 1SM/2SM 的 `SmemCopyAtomA/B` 都是 `void`，
目标函数也没有 `tcgen05.cp`；这与 SS descriptor 直接供给 MMA 的路径一致。三条 Dense
block-scaled 路径分别观察到 8/4/2 条函数内 `tcgen05.cp`，Sparse NVFP4 观察到 5 条，
同时 type witness 闭合了 scale copy/layout 和 sparse metadata layout。这里的计数是当前
实例观测，不是跨 Tile、跨工具链的固定公式。

`operand_source` 不再由实例手填。通用 witness 直接读取 `TiledMma::FrgTypeA/B`：六个 fresh
实例中，普通 Dense 和三条 Dense block-scaled 路径都是 `SMEM_DESCRIPTOR / SMEM_DESCRIPTOR`；
Sparse NVFP4 是 `SPARSE_SMEM_DESCRIPTOR / SMEM_DESCRIPTOR`。MixedInput pilot 则实际解析为
`TMEM_FRAGMENT / SMEM_DESCRIPTOR`。这组结果说明 SS/TS 应从 MMA fragment 类型确认，不能只
看 Schedule Tag 的命名。

## 5. Stage 的来源与多级流水线

当前实例的 A/B 空间布局在末尾追加 `PIPE` 维，从而在 SMEM 中形成多份可循环复用的物理
Buffer。Mainloop Producer 在一个 Stage 上发射 TMA，MMA Consumer 同时读取另一个 Ready
Stage。Stage 数因此同时影响 A/B 存储量、Barrier 数量、SharedStorage 和 Producer/Consumer
状态轮转。

`StageCountAutoCarveout` 先为 Epilogue SharedStorage 预留空间，再根据单个 A/B Stage 的
开销和可用 SMEM 推导合法 Mainloop Stage。Accumulator Stage、Scheduler Stage 和 Epilogue
C/D Stage分别属于不同 pipeline，不能共用一个模糊的 Stage 数字。

旧 `case.json` 中的 `pipeline_stages=0` 仍只是历史占位值；fresh witness 已经读取真实
`DispatchPolicy::Stages`。FP16 1SM 为 7/2/4，FP16 2SM 为 8/2/4，三个 Dense
block-scaled 和 Sparse NVFP4 都为 6/2/2；三个数字依次是 Mainloop、Scheduler 和
Accumulator Stage。

声明 policy 与解析 policy 分开保存。FP16 2SM 的 Epilogue SharedStorage 为 34816 字节，
因此解析为 `StageCountAutoCarveout<34816>`；No-SMEM 路径的 Epilogue SharedStorage 为
32 字节，解析为 `StageCountAutoCarveout<32>`。Sparse 使用
`StageCountAutoCarveoutEpi<resolved CollectiveEpilogue>`。这些数据证明静态容量推导结果，
不说明增加或减少 Stage 会提高性能。

## 6. 从 TMEM Accumulator 到 D

Mainloop 完成当前输出 Tile 的全部 K 维累加后，Accumulator Pipeline 把一个 Ready 的 TMEM
Stage交给 Epilogue。Epilogue先把 accumulator fragment从 TMEM 读到寄存器，再执行
LinearCombination或 FusionCallbacks，最后按 resolved Epilogue Schedule选择 D Store 路径。
No-SMEM Epilogue 与 TMA Store Epilogue 分别建立合同，D Store 路径以实际 Schedule 为准。

Fresh 结果已经区分两类 Epilogue。FP16 1SM 和三个 Dense block-scaled 实例解析为
`Sm100NoSmemWarpSpecialized`；FP16 2SM 与 Sparse NVFP4 解析为带 TMA load/store 的
`Sm100TmaWarpSpecialized`。Sparse 原先写成 No-SMEM 的声明也据此修正为
`TmaWarpSpecialized1SmNvf4`。历史 Bias+ReLU 实例还没有按新合同重放，因此 activation
仍只属于历史快照；静态 Epilogue 类型也不证明 alpha/beta、舍入或最终数值正确。

## 7. 从 Work Tile 到完整 GEMM

Collective 只负责一个输出 Tile 内部的计算。`GemmUniversal` 把 Mainloop 和 Epilogue组合成
完整 kernel，再由 Tile Scheduler 将 M/N/L/K 空间中的 Work Tile 分给 CTA 或 Cluster。
这里的 Tile Scheduler 与 Mainloop `KernelSchedule` 处于不同层：前者分配整题工作，后者
选择一个 Tile 内部的 TMA/TCGen05 实现。

六个 fresh instance 都把 `TileScheduler` 固定为 CUTLASS 默认的 `void`，并由
`GemmKernel` cross-view 确认 Mainloop 与 Epilogue 确实组合进同一个 kernel。它们闭合了默认
Scheduler 控制面，却没有形成 Scheduler 对比轴。Persistent/CLC、Stream-K 和 Split-K
改变的是 Work Tile、workspace、fixup 或 reduction path，未必对应唯一 opcode；后续仍要从
resolved Scheduler 类型、Arguments/Params 和 helper function归属建立独立实例。

## 8. 函数级 PTX/SASS 归属

不同 `MmaTileShape`、Stage 或 Scheduler 都会生成不同 kernel type。本文把稳定唯一的 kernel
symbol 作为归属键，从同一 symbol 提取 PTX `.entry` 和 SASS function，再执行
required/forbidden 合同。整个 binary 中发现某条指令与当前实例之间没有直接归属关系。

每条结果必须绑定：

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

Phase 1 为六个实例各生成一次同时包含 PTX 与 CUBIN 的 seeded FATBIN。每份结果都要求一个
PTX entry、一个 ELF `STO_ENTRY` 和一个同名 nvdisasm function；六个 CUBIN 各含 4 个
nvdisasm function，其中只有一个是目标 entry，另外三个 guardrail 不参与 opcode 合同。
完整产物保存在忽略目录，Git 中保存函数摘录、resolved type、合同报告、hash 和 journal。
随后从源码重新编译、重提取、重反汇编的 deep replay 6/6 通过。

## 9. 59 个显式 Schedule Tag 的覆盖合同

固定 CUTLASS commit 在 `dispatch_policy.hpp` 中声明了 59 个显式 SM100 Tensor Core
Mainloop Schedule Tag。本项目把这 59 个 Tag 作为完整静态分母：每个 Tag 选取一个固定的
基准实例。能够编译的实例沿完整类型链追到函数级 PTX/SASS，静态拒绝项则保存停止层和
编译诊断。最终状态只有三种：成功闭合记为
`STATIC_PASS`，有稳定源码约束和编译诊断的合法候选记为 `EXPECTED_STATIC_REJECT`，经过
合法上层实例与目标架构诊断确认的情形记为 `UNSUPPORTED_SM110A`。`NOT_CHECKED`、
`HISTORICAL_STATIC_PASS`、`UNEXPECTED_COMPILE_FAIL` 和 `ATTRIBUTION_FAIL` 都是执行中的
状态，收尾时必须归零。

机器可读清单位于
[`sm110a_tensor_schedule_tags.json`](../tests/codegen/sm110a_tensor_schedule_tags.json)。
当前状态由清单和 result record 联合重算。6 个历史 Tag 已经由新的 canonical instance 完成
fresh 重放，其余 53 个仍未执行：

| 清单状态 | 数量 |
|---|---:|
| `NOT_CHECKED` | 53 |
| `HISTORICAL_STATIC_PASS` | 0 |
| `STATIC_PASS` | 6 |
| `EXPECTED_STATIC_REJECT` | 0 |
| `UNSUPPORTED_SM110A` | 0 |
| `UNEXPECTED_COMPILE_FAIL` | 0 |
| `ATTRIBUTION_FAIL` | 0 |

完整分母见附录 A。`KernelScheduleAuto` 是自动选择入口，不代表第 60 个显式实现，因此
单独作为控制项验证，不计入 59。

固定源码清点为每个 Tag 找到了构造基准实例的起点。来源清单记录在
[`sm110a_schedule_reference_inventory.json`](../tests/codegen/sm110a_schedule_reference_inventory.json)，
当前统计如下：

| 基准实例的来源线索 | 数量 | 这项数据说明什么 |
|---|---:|---|
| 官方 C++ 显式或条件式引用 | 39 | test/example 中能找到目标 Tag；条件分支仍需固定实参后编译 |
| 官方注释中的 Auto 候选映射 | 1 | 实际 Builder 输入是 `KernelScheduleAuto`，注释只提供候选 Tag |
| `generator.py` 可还原配置 | 5 | 生成器给出了还原类型、Tile、Cluster 和 Schedule 所需的配置 |
| 沿单一变化轴的源码派生 | 14 | 从相邻合法配置只改变一个概念轴，再用编译结果判断 |

59 个显式 Tag 的基准实例来源分成四类：官方 C++ 代码中的显式或条件式引用、官方注释中的
Auto 候选映射、可从生成器还原的配置，以及沿一个明确变化轴得到的源码派生项。这组数字只
说明待编译实例从哪里来，不表示任何一项已经被 `sm_110a` 编译器接受。

其中，`KernelTmaWarpSpecialized1SmMxf4Sm100` 的来源线索只有 Auto 注释映射；注释本身不
计入 `STATIC_PASS`。Phase 1 另行固定了一个显式 Mxf4 instance，并用完整类型链和函数产物
得到 fresh 结果，因此该 Tag 当前的状态不再依赖那行注释。对应的 Auto 行为仍留给附录 B
中的独立控制项。

14 个派生项都记录了直接父项和唯一变化轴。校验器还会确认父子项使用同一来源锚点，并拒绝
循环依赖。当前最明确的边界是
`KernelTmaWarpSpecialized2SmMixedInputSmemSm100`：固定源码的 2SM mixed-input helper 没有
对应的 Smem/SS 返回路径，因此它是 `EXPECTED_STATIC_REJECT` 候选。这个判断目前属于源码
假设，后续仍要保存独立编译诊断才能成为正式终态。

59/59 的含义也在这里固定下来：每个 Tag 至少闭合一个基准实例。Fast FP32 等 Tag
可以因输入类型不同命中多个 Builder 偏特化，generic block-scaled Tag 也可能选择不同 MMA
分支；这些组合域不由一行 Tag 结果自动覆盖。

`KernelScheduleAuto` 的控制项按相同的 11 个实现分组单独记录。每一项从本组一个合法显式
Tag 的配置出发，只把 Builder 输入改为 Auto。固定源码目前给出 4 项“预计可构造”和 7 项
“预计静态拒绝”，11 项的正式状态仍全部是 `NOT_CHECKED`。Auto 成功时记录实际命中的 Builder
偏特化和内部 `DispatchPolicy::Schedule`，不要求物化成 59 个 public leaf Tag 之一。机器清单
位于 [`sm110a_auto_control_inventory.json`](../tests/codegen/sm110a_auto_control_inventory.json)，
完整对照项见附录 B。

## 10. 历史静态数据留下了什么

仓库保留的 `static-20260817` 快照使用 CUTLASS
`e05f953a5b3d38adc240df2ff928e0421c2abba3`、NVCC/PTXAS 13.0.88、GCC 13.3 和
`sm_110a`。它记录了 10 个实例的聚合结果：

```text
source_present  = 10/10
compile_passed  = 10/10
ptx_verified    = 10/10
sass_verified   = 10/10
runtime_correct = 0/10
```

| 实现路径 | 现有实例 | MmaTileShape | CTA Group | 静态结果 | 运行结果 |
|---|---|---|---:|---|---|
| Dense FP16 | `dense_f16_1sm_p128` | `128×128×64` | 1 | `HISTORICAL_STATIC_PASS` | `NOT_RUN` |
| Dense BF16 | `dense_bf16_1sm_p128` | `128×128×64` | 1 | `HISTORICAL_STATIC_PASS` | `NOT_RUN` |
| Dense FP8 | `dense_fp8_1sm_p128` | `128×128×128` | 1 | `HISTORICAL_STATIC_PASS` | `NOT_RUN` |
| Dense FP16 2SM | `dense_f16_2sm_p256x128x128` | `256×128×64` | 2 | `HISTORICAL_STATIC_PASS` | `NOT_RUN` |
| MXFP8 | `bs_mxfp8_1sm_p128` | `128×128×128` | 1 | `HISTORICAL_STATIC_PASS` | `NOT_RUN` |
| MXFP4 | `bs_mxfp4_1sm_p128x128x256` | `128×128×256` | 1 | `HISTORICAL_STATIC_PASS` | `NOT_RUN` |
| NVFP4 | `bs_nvfp4_1sm_p128x128x256` | `128×128×256` | 1 | `HISTORICAL_STATIC_PASS` | `NOT_RUN` |
| Sparse NVFP4 | `sparse_bs_nvfp4_1sm_p128x128x256` | `128×128×256` | 1 | `HISTORICAL_STATIC_PASS` | `NOT_RUN` |
| Bias+ReLU | `epilogue_bias_relu_f16_p128` | `128×128×64` | 1 | `HISTORICAL_STATIC_PASS` | `NOT_RUN` |
| FP16 tail sample | `tail_dense_f16_p130x129x127` | `128×128×64` | 1 | `HISTORICAL_STATIC_PASS` | `NOT_RUN` |

这些行仍是 `static-20260817` 的历史记录：它们说明固定工具链曾经为多种 TCGen05 路径生成
过预期指令，但完整函数产物没有归档，旧 inspector 也不能排除跨函数串证据。Phase 1 另用
六个 canonical instance 把对应的 6 个 Schedule Tag 升级成 fresh `STATIC_PASS`；BF16、FP8、
Bias+ReLU 和 tail 等历史变体没有因此自动升级。`runtime_correct=0` 表示没有 Thor 数值证据，
并非十个数值失败。

## 11. 当前结果能够支持的结论

现在既有横向历史宽度，也有六条纵向 fresh 闭环。对这六个 canonical instance，可以确认
Builder 输入、`CollectiveOp`、Dispatch、Stage、Copy/Layout、TiledMMA、Atom 和同一目标
函数内 PTX/SASS 的对应关系；六份结果还通过了从源码开始的独立 deep replay。

结论仍以实例为单位，不能写成“这 6 个 Tag 的全部类型、Tile 和 Builder 分支都已覆盖”。
其余 53 个显式 Tag 与 11 个 Auto 对照项仍未形成正式终态；下一阶段从 39 个有官方 C++
显式或条件式引用的 Tag 开始扩展。整个文档仍没有 Thor launch、数值或性能证据。

## 附录 A：59 个显式 Schedule Tag

下面的完整台账只回答“每个显式 Tag 是否已经建账并获得静态终态”。一个 Tag 能接受的全部
数据类型、Tile、Cluster 和 Builder 分支不在这张表的覆盖范围内。

### A.1 普通 Dense（5）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelTmaWarpSpecialized1SmSm100` | `STATIC_PASS` | `dense_f16_1sm_p128`、`dense_bf16_1sm_p128`、`dense_fp8_1sm_p128`、`epilogue_bias_relu_f16_p128`、`tail_dense_f16_p130x129x127` |
| `KernelTmaWarpSpecialized2SmSm100` | `STATIC_PASS` | `dense_f16_2sm_p256x128x128` |
| `KernelWarpSpecialized1SmSm100` | `NOT_CHECKED` | — |
| `KernelMixedTmaCpAsyncWarpSpecialized1SmSm100` | `NOT_CHECKED` | — |
| `KernelMixedTmaCpAsyncWarpSpecialized2SmSm100` | `NOT_CHECKED` | — |

### A.2 Pointer-array Dense（2）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelPtrArrayTmaWarpSpecialized1SmSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized2SmSm100` | `NOT_CHECKED` | — |

### A.3 Blockwise（4）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelTmaWarpSpecializedBlockwise1SmSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecializedBlockwise2SmSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100` | `NOT_CHECKED` | — |

### A.4 Planar Complex（4）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelTmaWarpSpecialized1SmPlanarComplexSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized2SmPlanarComplexSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized1SmPlanarComplexSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized2SmPlanarComplexSm100` | `NOT_CHECKED` | — |

### A.5 Fast FP32 / 9xBF16（8）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelTmaWarpSpecialized1SmFastFP32Sm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized2SmFastFP32Sm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized1SmFastFP32SmemSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized2SmFastFP32SmemSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized1SmFastFP32Sm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized2SmFastFP32Sm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized1SmFastFP32SmemSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized2SmFastFP32SmemSm100` | `NOT_CHECKED` | — |

### A.6 Mixed-input（4）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelTmaWarpSpecialized1SmMixedInputSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized1SmMixedInputSmemSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized2SmMixedInputSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized2SmMixedInputSmemSm100` | `NOT_CHECKED` | — |

### A.7 Interleaved Complex TF32（4）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelTmaWarpSpecialized1SmInterleavedComplexTF32Sm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized2SmInterleavedComplexTF32Sm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized1SmInterleavedComplexTF32Sm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized2SmInterleavedComplexTF32Sm100` | `NOT_CHECKED` | — |

### A.8 普通 Sparse（2）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelSparseTmaWarpSpecialized1SmSm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized2SmSm100` | `NOT_CHECKED` | — |

### A.9 Dense Block-scaled（10）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelTmaWarpSpecialized1SmBlockScaledSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized2SmBlockScaledSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized1SmNvf4Sm100` | `STATIC_PASS` | `bs_nvfp4_1sm_p128x128x256` |
| `KernelTmaWarpSpecialized2SmNvf4Sm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized1SmMxf4Sm100` | `STATIC_PASS` | `bs_mxfp4_1sm_p128x128x256` |
| `KernelTmaWarpSpecialized2SmMxf4Sm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` | `STATIC_PASS` | `bs_mxfp8_1sm_p128` |
| `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` | `NOT_CHECKED` | — |
| `KernelMixedTmaCpAsyncWarpSpecialized1SmBlockScaledSm100` | `NOT_CHECKED` | — |
| `KernelMixedTmaCpAsyncWarpSpecialized2SmBlockScaledSm100` | `NOT_CHECKED` | — |

### A.10 Pointer-array Block-scaled（8）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelPtrArrayTmaWarpSpecialized1SmBlockScaledSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized2SmBlockScaledSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized1SmMxf4Sm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized2SmMxf4Sm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100` | `NOT_CHECKED` | — |

### A.11 Sparse Block-scaled（8）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelSparseTmaWarpSpecialized1SmBlockScaledSm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized2SmBlockScaledSm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized1SmNvf4Sm100` | `STATIC_PASS` | `sparse_bs_nvfp4_1sm_p128x128x256` |
| `KernelSparseTmaWarpSpecialized2SmNvf4Sm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized1SmMxf4Sm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized2SmMxf4Sm100` | `NOT_CHECKED` | — |

性能和 runtime correctness 使用独立分母，不由这张静态表推导。

## 附录 B：11 个 `KernelScheduleAuto` 对照项

读取固定 Builder 源码后，11 个 Auto 对照项形成了一组待编译推测：4 项预计能够完成类型
构造，7 项预计会在 Builder 或 kernel 组合阶段被拒绝。11 项当前仍全部是 `NOT_CHECKED`；
只有重新编译并完成函数归属检查后，才能写入正式静态终态。

| Auto 对照项 | 实现分组 | 显式 seed Tag | 源码推测 | 当前状态 |
|---|---|---|---|---|
| `auto_dense_canonical` | `dense` | `KernelTmaWarpSpecialized1SmSm100` | 预计可构造 | `NOT_CHECKED` |
| `auto_ptr_array_dense_canonical` | `ptr_array_dense` | `KernelPtrArrayTmaWarpSpecialized1SmSm100` | 预计静态拒绝 | `NOT_CHECKED` |
| `auto_blockwise_canonical` | `blockwise` | `KernelTmaWarpSpecializedBlockwise1SmSm100` | 预计静态拒绝 | `NOT_CHECKED` |
| `auto_planar_complex_canonical` | `planar_complex` | `KernelTmaWarpSpecialized1SmPlanarComplexSm100` | 预计静态拒绝 | `NOT_CHECKED` |
| `auto_fast_fp32_canonical` | `fast_fp32` | `KernelTmaWarpSpecialized1SmFastFP32Sm100` | 预计静态拒绝 | `NOT_CHECKED` |
| `auto_mixed_input_canonical` | `mixed_input` | `KernelTmaWarpSpecialized2SmMixedInputSm100` | 预计静态拒绝 | `NOT_CHECKED` |
| `auto_interleaved_complex_tf32_canonical` | `interleaved_complex_tf32` | `KernelTmaWarpSpecialized1SmInterleavedComplexTF32Sm100` | 预计可构造 | `NOT_CHECKED` |
| `auto_sparse_canonical` | `sparse` | `KernelSparseTmaWarpSpecialized1SmSm100` | 预计可构造 | `NOT_CHECKED` |
| `auto_dense_block_scaled_canonical` | `dense_block_scaled` | `KernelTmaWarpSpecialized1SmMxf4Sm100` | 预计可构造 | `NOT_CHECKED` |
| `auto_ptr_array_block_scaled_canonical` | `ptr_array_block_scaled` | `KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100` | 预计静态拒绝 | `NOT_CHECKED` |
| `auto_sparse_block_scaled_canonical` | `sparse_block_scaled` | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` | 预计静态拒绝 | `NOT_CHECKED` |
