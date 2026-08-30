# CUTLASS Tensor Core / TCGen05 GEMM Codegen 形式化合同验证

Tensor Core 路径把一次 GEMM 拆成两类工作。Mainloop 沿 K 维加载 A/B Tile，并用
`tcgen05.mma` 把结果累加到 TMEM；Epilogue 再把 TMEM accumulator读回寄存器，完成线性
组合或融合计算，最后写回 D。CUTLASS 的 Builder、Collective 和 Kernel 抽象，就是把这条
数据流组织成一份可实例化的 C++ 类型。

分析沿一个真实 1SM FP16 实例，从高层配置向下追踪代码生成。核心问题是：Builder 中的
数据类型、Layout、`MmaTileShape`、Cluster、Stage
和 Schedule，最终分别选择了什么数据路径与 TCGen05 指令。

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

这里的 `STATIC_PASS` 只表示当前实例生成了预期的函数级 PTX/SASS。真实 Thor launch、
数值正确性和性能分别属于后续实验。

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

这一节最终要把 `MMA_Atom` 与 `MMA_Traits` 展开到具体类型，再在目标函数中确认
`tcgen05.mma...kind::f16`。BF16、FP8、Block-scaled 和 Sparse 会命中不同 descriptor 或
opcode family；这些变化在 FP16 主路径闭合后作为扩展讨论。

现有历史静态快照已经记录 FP16/BF16/FP8、Block-scaled 和 Sparse 的聚合 PTX/SASS PASS，
但没有保存完整 Atom type dump 和当前可逐函数重查的产物。因此这一节仍需要重新生成，
把历史状态补成可复核的 V2 证据。

## 3. 从 MMA Atom 到 `MmaTileShape`

一条 TCGen05 MMA 只覆盖 instruction shape。`TiledMMA` 在其上增加 Atom Layout 和 CTA
slice，使一个或两个 CTA 能够共同覆盖更大的 `MmaTileShape`。当前 1SM 实例中，
`cta_group::1` 只有一个 CTA slice；2SM 扩展则会引入 peer CTA、`cta_group::2` 和不同的
Cluster/TMEM 分配关系。

这一节需要并排展示五种 shape：instruction shape、TiledMMA shape、`MmaTileShape`、
Cluster/work tile 和运行时 problem shape。它们来自不同层，文档将分别给出具体值，而不使用
“128³”同时指代多个 shape。逻辑覆盖倍数可以由 Tile 与 instruction shape 的比例得到，但实际 SASS 条数还
受到循环、展开和调度影响，只作为观测数据。

历史静态快照目前只有少量单点：1SM 覆盖 `128×128×64/128/256`，2SM 只有
`256×128×64`。同一实现路径下 3～4 个合法 `MmaTileShape` 的覆盖面尚未形成，解析后的
TiledMMA 和 CTA slice也没有归档。下一步沿 shape 轴补实例，数值类型保持 FP16 不变。

## 4. Tensor Core Mainloop 的数据路径

`TiledMMA` 确定空间分区以后，`CollectiveBuilder` 继续选择数据路径。当前 Dense 1SM SS
实例使用 TMA 把 A/B 从 GMEM 放入多阶段 SMEM Buffer，TCGen05 再通过 descriptor直接读取
SMEM。`SmemCopyAtomA/B` 在这条路径中不承担经典的 SMEM→RMEM copy，Accumulator 则由
TMEM 持有。

这一节从 Builder 输入追踪到具体 `DispatchPolicy`、GMEM Copy、SMEM Layout、Transform 和
`CollectiveMma`，最终形成：

```text
GMEM A/B
→ TMA
→ SMEM Stage
→ TCGen05 MMA
→ TMEM Accumulator
```

普通 Dense SS 不需要为了“指令齐全”强制出现 `tcgen05.cp`。Block-scaled 的 scale ingress
是否使用 `tcgen05.cp`，TS 操作数怎样进入 TMEM，都由各自解析后的路径决定。Sparse 还会
增加 compressed operand和 metadata。主路径闭合后，只需要解释这些扩展在哪一层发生变化。

现有静态快照已经为这些代表路径保存 required PTX/SASS 聚合结果，但没有完整
`DispatchPolicy`、Copy/Layout、Transform 和函数产物；这部分仍要重新生成并归档。

## 5. Stage 的来源与多级流水线

当前实例的 A/B 空间布局在末尾追加 `PIPE` 维，从而在 SMEM 中形成多份可循环复用的物理
Buffer。Mainloop Producer 在一个 Stage 上发射 TMA，MMA Consumer 同时读取另一个 Ready
Stage。Stage 数因此同时影响 A/B 存储量、Barrier 数量、SharedStorage 和 Producer/Consumer
状态轮转。

`StageCountAutoCarveout` 先为 Epilogue SharedStorage 预留空间，再根据单个 A/B Stage 的
开销和可用 SMEM 推导合法 Mainloop Stage。文档最终给出解析后的 Stage，而不止写
`Auto`。Accumulator Stage、Scheduler Stage 和 Epilogue C/D Stage分别属于不同 pipeline，
各自记录对应数字和资源。

现有实例清单中的 `pipeline_stages=0` 是未解析占位值，不是真实 Stage。因此 Stage 是
当前最明确的证据缺口。重新生成时需要保存解析后的
`DispatchPolicy::Stages`、各类 Pipeline 类型、SMEM Layout、SharedStorage 和对应函数片段。
这里分析代码生成和资源合同，Stage 的性能影响留到性能实验。

## 6. 从 TMEM Accumulator 到 D

Mainloop 完成当前输出 Tile 的全部 K 维累加后，Accumulator Pipeline 把一个 Ready 的 TMEM
Stage交给 Epilogue。Epilogue先把 accumulator fragment从 TMEM 读到寄存器，再执行
LinearCombination或 FusionCallbacks，最后按 resolved Epilogue Schedule选择 D Store 路径。
No-SMEM Epilogue 与 TMA Store Epilogue 分别建立合同，D Store 路径以实际 Schedule 为准。

当前主线使用基础 No-SMEM Epilogue，先确认 TMEM readback、输出类型转换和 D 写回。历史
静态快照还包含一份 Bias+ReLU 实例；基础路径解释完成后，再用它说明 Fusion 带来的变化。
静态产物可以证明某种 Epilogue 已经实例化，bias 轴、activation 和最终数值则留给 runtime
reference。

## 7. 从 Work Tile 到完整 GEMM

Collective 只负责一个输出 Tile 内部的计算。`GemmUniversal` 把 Mainloop 和 Epilogue组合成
完整 kernel，再由 Tile Scheduler 将 M/N/L/K 空间中的 Work Tile 分给 CTA 或 Cluster。
这里的 Tile Scheduler 与 Mainloop `KernelSchedule` 处于不同层：前者分配整题工作，后者
选择一个 Tile 内部的 TMA/TCGen05 实现。

第一轮先固定默认或 DataParallel scheduler。Persistent/CLC、Stream-K 和 Split-K 会改变
Work Tile、workspace、fixup 或 reduction path，因此需要生成独立实例。它们未必对应唯一
opcode，静态验证应关注 Scheduler 类型、Arguments/Params 和 helper function归属。现有
10 个实例没有覆盖这些路径，所以 Scheduler 轴目前是 `NOT_CHECKED`。

## 8. 函数级 PTX/SASS 归属

不同 `MmaTileShape`、Stage 或 Scheduler 都会生成不同 kernel type。为了避免实例串扰，每个
编译期实例最好生成独立 executable；至少也要有稳定唯一的 symbol。检查工具从该 symbol
提取 PTX `.entry` 和 SASS function，再执行 required/forbidden 合同。整个 binary 中发现某条
指令与当前实例之间没有直接归属关系。

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

历史静态快照保存了工具链身份和每个实例的 binary/PTX hash，但没有把完整 PTX/SASS
提交到 Git。它作为历史结果保留，当前逐函数结论由重新生成的产物支持。

## 9. 59 个显式 Schedule Tag 的覆盖合同

固定 CUTLASS commit 在 `dispatch_policy.hpp` 中声明了 59 个显式 SM100 Tensor Core
Mainloop Schedule Tag。本项目把这 59 个 Tag 作为完整静态分母：每个 Tag 都必须有唯一清单
条目，并最终得到 `STATIC_PASS`、`EXPECTED_STATIC_REJECT`、`UNSUPPORTED_SM110A`、
`UNEXPECTED_COMPILE_FAIL` 或 `ATTRIBUTION_FAIL`。没有结果的条目保持
`NOT_CHECKED`；`OUT_OF_SCOPE` 不作为跳过理由。

机器可读清单位于
[`sm110a_tensor_schedule_tags.json`](../tests/codegen/sm110a_tensor_schedule_tags.json)。
当前 6 个 Tag 有历史静态结果，另外 53 个尚未生成。下面逐项列出全部分母。
`KernelScheduleAuto` 是自动选择入口，不代表第 60 个显式实现，因此单独作为控制项验证，
不计入 59。

### 9.1 普通 Dense（5）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelTmaWarpSpecialized1SmSm100` | `HISTORICAL_STATIC_PASS` | `dense_f16_1sm_p128`、`dense_bf16_1sm_p128`、`dense_fp8_1sm_p128`、`epilogue_bias_relu_f16_p128`、`tail_dense_f16_p130x129x127` |
| `KernelTmaWarpSpecialized2SmSm100` | `HISTORICAL_STATIC_PASS` | `dense_f16_2sm_p256x128x128` |
| `KernelWarpSpecialized1SmSm100` | `NOT_CHECKED` | — |
| `KernelMixedTmaCpAsyncWarpSpecialized1SmSm100` | `NOT_CHECKED` | — |
| `KernelMixedTmaCpAsyncWarpSpecialized2SmSm100` | `NOT_CHECKED` | — |

### 9.2 Pointer-array Dense（2）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelPtrArrayTmaWarpSpecialized1SmSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized2SmSm100` | `NOT_CHECKED` | — |

### 9.3 Blockwise（4）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelTmaWarpSpecializedBlockwise1SmSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecializedBlockwise2SmSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100` | `NOT_CHECKED` | — |

### 9.4 Planar Complex（4）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelTmaWarpSpecialized1SmPlanarComplexSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized2SmPlanarComplexSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized1SmPlanarComplexSm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized2SmPlanarComplexSm100` | `NOT_CHECKED` | — |

### 9.5 Fast FP32 / 9xBF16（8）

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

### 9.6 Mixed-input（4）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelTmaWarpSpecialized1SmMixedInputSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized1SmMixedInputSmemSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized2SmMixedInputSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized2SmMixedInputSmemSm100` | `NOT_CHECKED` | — |

### 9.7 Interleaved Complex TF32（4）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelTmaWarpSpecialized1SmInterleavedComplexTF32Sm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized2SmInterleavedComplexTF32Sm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized1SmInterleavedComplexTF32Sm100` | `NOT_CHECKED` | — |
| `KernelPtrArrayTmaWarpSpecialized2SmInterleavedComplexTF32Sm100` | `NOT_CHECKED` | — |

### 9.8 普通 Sparse（2）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelSparseTmaWarpSpecialized1SmSm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized2SmSm100` | `NOT_CHECKED` | — |

### 9.9 Dense Block-scaled（10）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelTmaWarpSpecialized1SmBlockScaledSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized2SmBlockScaledSm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized1SmNvf4Sm100` | `HISTORICAL_STATIC_PASS` | `bs_nvfp4_1sm_p128x128x256` |
| `KernelTmaWarpSpecialized2SmNvf4Sm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized1SmMxf4Sm100` | `HISTORICAL_STATIC_PASS` | `bs_mxfp4_1sm_p128x128x256` |
| `KernelTmaWarpSpecialized2SmMxf4Sm100` | `NOT_CHECKED` | — |
| `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` | `HISTORICAL_STATIC_PASS` | `bs_mxfp8_1sm_p128` |
| `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` | `NOT_CHECKED` | — |
| `KernelMixedTmaCpAsyncWarpSpecialized1SmBlockScaledSm100` | `NOT_CHECKED` | — |
| `KernelMixedTmaCpAsyncWarpSpecialized2SmBlockScaledSm100` | `NOT_CHECKED` | — |

### 9.10 Pointer-array Block-scaled（8）

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

### 9.11 Sparse Block-scaled（8）

| Schedule Tag | 当前状态 | 已有关联实例 |
|---|---|---|
| `KernelSparseTmaWarpSpecialized1SmBlockScaledSm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized2SmBlockScaledSm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized1SmNvf4Sm100` | `HISTORICAL_STATIC_PASS` | `sparse_bs_nvfp4_1sm_p128x128x256` |
| `KernelSparseTmaWarpSpecialized2SmNvf4Sm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized1SmMxf4Sm100` | `NOT_CHECKED` | — |
| `KernelSparseTmaWarpSpecialized2SmMxf4Sm100` | `NOT_CHECKED` | — |

这张表只回答“59 个 Tag 是否全部进入测试分母”。每个 Tag 的 Builder 参数、合法数据类型、
Tile、Cluster、Stage、Epilogue 和函数级 PTX/SASS 仍由后续实例逐项补齐。性能和 runtime
correctness 使用独立分母，不由这张静态表推导。

## 10. 现有静态数据

仓库当前保留的 `static-20260817` 静态快照使用 CUTLASS
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
| Dense FP16 | `dense_f16_1sm_p128` | `128×128×64` | 1 | PASS | NOT_RUN |
| Dense BF16 | `dense_bf16_1sm_p128` | `128×128×64` | 1 | PASS | NOT_RUN |
| Dense FP8 | `dense_fp8_1sm_p128` | `128×128×128` | 1 | PASS | NOT_RUN |
| Dense FP16 2SM | `dense_f16_2sm_p256x128x128` | `256×128×64` | 2 | PASS | NOT_RUN |
| MXFP8 | `bs_mxfp8_1sm_p128` | `128×128×128` | 1 | PASS | NOT_RUN |
| MXFP4 | `bs_mxfp4_1sm_p128x128x256` | `128×128×256` | 1 | PASS | NOT_RUN |
| NVFP4 | `bs_nvfp4_1sm_p128x128x256` | `128×128×256` | 1 | PASS | NOT_RUN |
| Sparse NVFP4 | `sparse_bs_nvfp4_1sm_p128x128x256` | `128×128×256` | 1 | PASS | NOT_RUN |
| Bias+ReLU | `epilogue_bias_relu_f16_p128` | `128×128×64` | 1 | PASS | NOT_RUN |
| FP16 tail sample | `tail_dense_f16_p130x129x127` | `128×128×64` | 1 | PASS | NOT_RUN |

这张表表示历史静态状态。完整函数产物尚未归档，后续重新生成后才能写入逐层结果；
`runtime_correct=0` 表示没有 Thor 数值证据，并非十个数值失败。

## 11. 这些结果说明了什么

现有数据已经横向覆盖多种 TCGen05 路径，却还没有纵向解释任何一条实例。FP16 1SM、
FP16 2SM、FP8、Block-scaled 和 Sparse 都有聚合 PASS，但解析后的 Atom、TiledMMA、Stage、
Scheduler 和完整函数归属没有进入同一条可复核链。因此下一步保持现有精度范围，先把
FP16 1SM 贯穿实例写深。

最明显的三个缺口是：Stage 仍是占位值；1SM/2SM 没有形成 3～4 个 Tile 的 shape surface；
Scheduler 轴完全没有实例。等 FP16 1SM 的 Atom、TiledMMA、Collective、Stage 和函数产物闭合
以后，再沿一个轴增加 2SM 或新的 Tile。这样每次扩展都能回答“哪一层发生了变化”，而不是
只在表格里再多一行 `PASS`。
