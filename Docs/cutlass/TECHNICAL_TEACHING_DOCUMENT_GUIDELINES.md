# 技术教学文档写作规范

> 适用范围：GPU 架构、CUDA、CUTLASS、编译器、算子与其他需要结合源码、代码和技术图进行讲解的教学文档。
>
> 参考文档：[CUTLASS 3.x GEMM 抽象与 Blackwell 实现分析](https://xiaopeng.feishu.cn/wiki/Wod4wss3rirbjXkUcyjcUGJEn3e)
>
> 参考版本：飞书文档 revision 2171。

## 1. 核心原则

技术教学文档的目标是帮助读者建立可以迁移的技术模型。正文应围绕问题、因果关系、数据流、状态和所有权组织，形成以下理解路径：

```text
问题是什么
→ 整体编程模型是什么
→ 当前实例采用什么配置
→ 编译期如何构造类型
→ 运行时如何组织数据、状态和角色
→ 最终如何启动、验证并交付结果
```

文字、代码、图和源码链接分别承担不同职责：

| 载体 | 主要职责 | 不应承担的职责 |
|---|---|---|
| 正文 | 解释原因、输入输出、契约、状态变化和适用边界 | 大段逐行复述代码 |
| 代码 | 给出精确的类型、参数和调用形式 | 独自承担全局关系说明 |
| 技术图 | 展示拓扑、顺序、分支、同步、状态和所有权 | 容纳全部架构例外和源码细节 |
| 表格 | 比较多个候选在固定维度上的差异 | 表达连续因果链或并发状态机 |
| 源码链接 | 为具体结论提供可验证证据 | 替代正文解释 |
| 提示框 | 集中声明版本、适用范围和关键边界 | 在每个小节重复出现 |

完整性由主路径的因果闭环决定。读者应当能够回答：输入是什么、谁负责处理、状态如何变化、输出交给谁。异常分支和工程细节的数量不能替代这条闭环。

## 2. 文档定位与全局文字约束

### 2.1 教学正文与工程过程分离

教学正文围绕理解路径组织。README、维护手册、测试报告和实现日志所需的信息，应根据用途进入附录或代码仓库说明。

以下内容只有直接影响当前技术结论时才进入正文：

- 环境安装和依赖配置；
- 仓库目录树和文件清单；
- 完整构建命令；
- 修改过程和维护记录；
- 测试矩阵、执行日志和状态统计；
- 全部异常分支和错误码；
- 兼容列表、排障步骤和维护责任。

需要复现时，建议集中到附录：

- 附录 A：版本、依赖和编译命令；
- 附录 B：最小可运行示例；
- 附录 C：输入、参考实现和验证方法；
- 附录 D：扩展路径和已知限制；
- 附录 E：源码入口和精确行号索引。

判断某段工程信息是否进入正文时，可以询问：删除这段信息后，读者是否仍然能够理解当前抽象的输入、状态变化和输出？如果答案为是，这段内容通常进入附录。

### 2.2 主路径优先

正文先讲清一条当前实例的主路径，再介绍扩展路径。Dense FP16 主路径尚未形成完整模型时，不同时展开 Sparse、Block-scaled、Pointer-array、Grouped、Stream-K 和 Dynamic Cluster 等所有变体。

推荐在文章开头固定一套贯穿实例，例如：

```text
数据类型：A/B FP16，Accumulator/D FP32
Builder ArchTag：cutlass::arch::Sm100
实际二进制目标：compute_110a / sm_110a
MMA Tile：256 × 128 × 64
Cluster：2 × 2 × 1
Mainloop：TMA + TCGen05
Accumulator：TMEM
```

每解释一个通用机制，应在紧邻位置说明当前配置命中了哪条路径，以及由此产生了什么具体类型或行为。

### 2.3 正向职责先于边界

边界需要准确，表达顺序应当是：当前对象负责什么、为什么这样组织、适用范围是什么。

不推荐：

> `can_implement` 不会选择 Kernel，也不会执行设备代码，不能替代 launch、同步和数值验证。

推荐：

> `can_implement(args)` 建立当前运行参数与已编译 Kernel 之间的启动前兼容性。随后，`run()` 返回立即启动状态，stream 同步暴露设备执行期错误，Reference/Tolerance Check 最终验证数值结果。

全局文字应减少以下表达习惯：

- 反复使用“不是……而是……”；
- 以“不要”“不能”作为正常机制段落的第一句；
- 频繁使用“需要注意的是”；
- 先罗列所有例外，再介绍主路径；
- 使用疑问句自问自答；
- 为每个正常行为附带一组免责声明；
- 为显示严谨而持续使用“可能、一般、通常、某种意义上”。

版本、架构、硬件路径和例外应集中到“阅读基线”或“扩展与边界”小节。后续章节只有偏离基线时才再次说明。

### 2.4 术语与中文说明

API、类型名、枚举名、字段名、指令名和业内通用缩写保持源码拼写。正文第一次引入一个 API 时，先给出中文职责，再给出原始名称：

> `cutlass::gemm::collective::CollectiveBuilder` 是 Mainloop 的编译期构造入口。它接收架构、数据、Tile、Cluster 和 Schedule 约束，输出 `::CollectiveOp` 类型。

必须区分：

- CUTLASS 原生类型，例如 `CollectiveMma`、`GemmUniversalAdapter`；
- 文档为了可读性定义的别名，例如 `CollectiveMainloop`、`GemmKernel`；
- 概念性名称，例如 Mainloop、Epilogue、Producer、Consumer。

每个长段围绕一个核心对象展开，其他 API 只作为它的输入、输出或调用者出现。正文不连续堆叠大量英文名词，也不机械重复中英文双写。

## 3. 整篇文档的 Section 架构

### 3.1 Section 类型

| Section 类型 | 必须回答的问题 | 推荐内容 | 完成条件 |
|---|---|---|---|
| 引言 / 问题定义 | 要解决什么问题？输入输出是什么？ | 数学公式、数据流、读者基线、文章范围 | 读者知道后文在解释什么 |
| 编程模型总览 | 系统由哪些层组成？各层边界是什么？ | 总览图、平级定义、代表 API、基线代码 | 读者能够复述整体模型 |
| 编译期构造 | 用户输入如何变成具体类型？ | Builder 参数、决策树、DispatchPolicy、类型展开 | 输出一个明确的具体类型契约 |
| 运行时机制 | 类型在运行时如何组织数据和状态？ | Storage、Pipeline、角色、同步、生命周期 | 输出状态和所有权明确 |
| 接口与生命周期 | 用户参数怎样进入设备执行？ | Arguments、Params、workspace、initialize、run | 从主机输入到 Kernel 启动形成闭环 |
| 当前实例落地 | 当前配置命中哪条路径？ | 数据类型、Tile、Cluster、Schedule、偏特化 | 通用规则与具体实例建立映射 |
| 对比与选择 | 多个候选有什么区别？ | 固定维度的比较表、选择条件、当前实例 | 读者知道何时选择哪一种 |
| 完整示例 | 读者如何组合已经解释的组件？ | 契约完整代码、关键参数、验证入口 | 完成当前教学任务所需的完整链路 |
| 扩展与边界 | 主路径之外有哪些重要变体？ | 少量架构分支、版本边界、适用范围 | 不打断主线，也不污染基线定义 |
| 总结 | 全文建立了什么模型？ | 层间交接、关键不变量、后续优化方向 | 不再引入新的核心概念 |

### 3.2 一个主要 Section 的标准结构

一个主要 Section 通常包含以下部分。

#### 入口段

说明上一节已经产出了什么、本节从哪个状态或类型继续、本节负责什么，以及最终交付什么。

例如：

> 前一节已经把 Mainloop Builder 的输出展开为一个具体的 `CollectiveMma` 类型。本节从该类型的类体继续分析，说明它如何组织 A/B 的多阶段 SMEM Pipeline、TMA Producer、TCGen05 Consumer 和 TMEM Accumulator。章节结束时，当前输出 Tile 的完整 K 维累加结果应处于一个 Ready 的 TMEM Accumulator Stage。

#### 总览段或总览图

先给出本节内部的整体关系。组件超过三个，或者存在分支、并发、状态回环和所有权交接时，优先使用图。

#### 类型与资源契约

解释：

- 关键模板参数；
- Builder 或上层传入的类型；
- 本层生成的成员类型；
- SharedStorage、TensorStorage 和 PipelineStorage；
- 当前实例的具体值。

#### 运行时机制

沿一条主路径解释：

```text
入口对象
→ 调用或匹配动作
→ 状态变化
→ 数据位置或所有权变化
→ 产生的结果
```

#### 代码证据

选择只回答当前问题或展示目标结构的最小代码块，并提供固定源码链接。

#### 当前实例

说明本文配置命中了哪条偏特化、Schedule、CTA 范围或 Pipeline 形态。

#### 边界与变体

只解释与当前结论直接相关的例外。其他扩展在主路径之后集中说明。

#### 出口段

说明本节结束时系统处于什么状态、哪个对象拥有结果、下一节从哪里继续。

### 3.3 标题层级

标题层级建议控制在三层：

- H1：主要抽象层或大主题；
- H2：这一层需要解决的核心问题；
- H3：具体机制、状态或证据。

例如：

```text
H1 Collective 层
  H2 Collective 层：Mainloop
    H3 CollectiveMma 偏特化
    H3 空间布局到多阶段 SMEM Pipeline
    H3 Arguments、Params 与 Tiled 数据划分
    H3 Producer/Consumer 状态机与 TCGen05 MMA
    H3 TMEM Accumulator 与 Epilogue 交接
```

标题应描述技术问题或行为，避免使用“介绍”“相关代码”“其他”“补充”“注意事项”“示例”和“一些细节”等泛化名称。

创建新的 H3，应至少满足一个条件：

- 主体对象发生变化；
- 从编译期切换到运行时；
- 从布局切换到同步；
- 数据所有权发生交接；
- 出现一个可以独立回答的新问题。

源码中出现一个新类或函数，不能自动成为创建小节的理由。

### 3.4 CUTLASS 各 Section 的完成要求

| Section | 入口 | 必须讲清楚 | 出口 |
|---|---|---|---|
| Builder | 问题配置与 Mainloop/Epilogue 两阶段模型 | Builder 输入、Epilogue 优先构造、SMEM carveout、KernelSchedule、DispatchPolicy、最终 Collective 类型 | 具体 `CollectiveMma` 和 `CollectiveEpilogue` 类型 |
| Mainloop | Builder 输出的 `CollectiveMma` | 偏特化、A/B Layout、SMEM Stage、PipelineStorage、PipelineState、TMA Producer、MMA Consumer、TMEM 交接 | Ready 的 TMEM Accumulator Stage |
| Epilogue | Ready 的 TMEM Accumulator Stage | C/D Layout、三条 Pipeline、Arguments/Params、Warp 分工、Subtile 循环、FusionCallbacks/EVT、D Store | D 写回并释放 Accumulator Stage |
| Kernel | Mainloop 和 Epilogue 两个 Collective | 两条分派轴、SharedStorage、Arguments/Params、五类 Warp Role、Tile Scheduler、WorkTileInfo | 完整设备端状态机、Grid/Block 和资源需求 |
| Device | 已编译的 `GemmKernel` | Adapter 状态、所有权、workspace、Arguments→Params、can_implement/initialize/update/run、Cluster Launch、错误与验证边界 | 一次 GEMM 调用的主机侧闭环 |

### 3.5 Section 完成检查

- [ ] 入口状态明确；
- [ ] 核心问题只有一个；
- [ ] 输入、输出和职责明确；
- [ ] 编译期和运行时没有混写；
- [ ] 当前实例命中的路径明确；
- [ ] 数据存储位置明确；
- [ ] 状态和所有权变化明确；
- [ ] 关键代码或源码证据存在；
- [ ] 通用机制与当前实例可以区分；
- [ ] 例外集中在主路径之后；
- [ ] 出口状态明确；
- [ ] 下一节能够自然接续。

## 4. 技术细节深度与证据门槛

### 4.1 技术深度的定义

技术深度不能用代码行数、API 数量、图的节点数量或变体数量衡量。本文采用下面的定义：

> 技术深度是从一个可观察问题出发，向下展开到足以解释该结论的类型、数据、状态和架构机制，再把这些机制连接回可观察输出的完整程度。

下面这段文字虽然具有表面上的输入、处理者和输出，但仍然属于浅层摘要：

> Builder 生成 `CollectiveMma`；TMA 把 A/B 搬到 SMEM；MMA 把结果累加到 TMEM；Epilogue 把 D 写回 GMEM。

它尚未解释偏特化选择、模板参数来源、Tile 划分、Layout、Stage、Barrier、状态轮转、硬件完成事件和所有权交接。技术机制 Section 必须达到对应的最低深度，不能只完成名词覆盖。

### 4.2 L0～L5 技术深度模型

| 深度 | 名称 | 必须回答的问题 | 可验收结果 |
|---|---|---|---|
| L0 | 问题与可观察语义 | 数学操作、输入输出、Shape、数据类型、Layout/Stride、融合和正确性语义是什么？ | 读者可以构造合法输入并说明期望输出 |
| L1 | 概念职责与层间边界 | 当前组件接收什么、负责什么、输出什么、谁消费输出、相邻层负责什么？ | 读者可以画出层间输入、输出和唯一交接边 |
| L2 | 精确类型与 API 契约 | 官方类型是什么、模板实参是什么、参数由谁指定或推导、别名最终展开为什么？ | 读者可以指出当前实例命中的具体类型和参数来源 |
| L3 | 数据、布局与资源映射 | 数据如何划分、采用什么 Shape/Layout/Stride、位于哪个存储空间、Buffer 多大、谁拥有、何时失效？ | 读者可以追踪一个 Tile 或 Subtile 的位置、布局、owner 和 lifetime |
| L4 | 状态、时序与并发协议 | 有哪些并发角色、维护哪些状态、什么事件触发转移、谁等待、谁释放、如何避免提前读取和覆盖？ | 读者可以推演一份资源从可写到再次可写的完整生命周期 |
| L5 | 架构原语与硬件实现 | 上层动作落到哪个 Wrapper/PTX 原语、谁发射、谁参与、操作数在哪、协作范围和完成语义是什么？ | 读者可以把关键硬件动作映射到具体原语和参与范围 |

#### L0：问题与可观察语义

必须明确：

- 数学操作；
- 输入和输出；
- Shape、数据类型、Layout 和 Stride；
- 融合、量化和舍入语义；
- 什么结果算正确；
- 当前实例固定的参数。

例如，Requant 不能只写 `D = Requant(Acc)`。至少需要明确：

```text
D = saturate_round(
    requant_scale × (alpha × Acc + beta × C)
    + zero_point)
```

并说明 scale 的方向、zero-point、rounding、saturation 和目标数据类型。

#### L1：概念职责与层间边界

必须明确当前层的输入、主要职责、输出、下游消费者和相邻层边界。读者应能区分：

- Builder 是编译期类型构造入口，不是运行时对象；
- Mainloop 负责一个输出 Tile 内沿 K 维的 A/B 加载与累加；
- Tile Scheduler 在 Kernel 层分配 Work Tile；
- Device 负责主机参数、workspace、Params 和 Launch，不拥有用户张量。

#### L2：精确类型与 API 契约

必须区分参数来源：

- 用户显式指定；
- Builder 根据约束推导；
- 命中的偏特化固定；
- 运行时通过 `Arguments` 提供；
- 经过降低后保存在 `Params`。

Builder 不能停在“`KernelSchedule` 生成 `DispatchPolicy`”。至少需要追踪：

```text
CollectiveBuilder<..., KernelScheduleAuto>::CollectiveOp
→ 命中具体的 SM100 Builder 偏特化
→ 选择具体 KernelSchedule
→ 构造 DispatchPolicy
→ 生成 CollectiveMma<DispatchPolicy, ...>
```

同时说明 `Stages`、`SchedulerStages`、`AccumulatorStages`、`ClusterShape`、`TiledMma` 等关键参数的来源。

#### L3：数据、布局与资源映射

每条主要数据边必须说明：

- 逻辑坐标和物理存储之间的映射；
- Shape、Layout、Stride 和 Alignment；
- GMEM、SMEM、TMEM 或 RMEM；
- Buffer 数量和容量；
- owner 和 lifetime；
- Stage 数或资源大小的来源；
- 当前实例的具体值，或者负责推导该值的组件。

Mainloop 的最低数据链包括：

```text
A/B CTA Tile
→ TiledMma partition
→ MMA 所需的 A/B 空间形状
→ append PIPE 维
→ 多 Stage SmemLayoutA/B
→ smem_A[stage] / smem_B[stage]
→ TCGen05 MMA
→ TMEM Accumulator
```

Epilogue 至少追踪一个 Subtile：

```text
TMEM Accumulator
→ RMEM Fragment
→ Fusion/EVT
→ SMEM D Staging
→ TMA Store
→ GMEM D
```

C 和 Aux 的读取路径需要单独说明，不能只给出最终汇合点。

#### L4：状态、时序与并发协议

每条主要同步边必须说明：

- Producer；
- 被保护的资源；
- 完成事件；
- Consumer；
- 等待条件；
- 释放动作；
- `index`、`phase` 和其他状态的推进；
- Prologue、Steady State 和 Drain 的结束条件。

Mainloop 的最低状态闭环是：

```text
producer_acquire Empty
→ 选择 Stage i
→ TMA 绑定该 Stage 的 Transaction Barrier
→ 硬件完成预期字节
→ Full Barrier Ready
→ consumer_wait
→ TCGen05 MMA
→ consumer_release Empty
→ index/phase 推进
→ Stage 再次可写
```

读者应能说明为什么不会发生 read-before-ready 或 overwrite-before-release。

#### L5：架构原语与硬件实现

必须明确：

- 对应的 CUTLASS/CuTe Wrapper 或 PTX 原语；
- 发射者和参与者；
- 操作数所在的存储空间；
- Warp、CTA、CTA pair、1SM、2SM 或 Cluster 协作范围；
- 硬件完成如何反映到同步对象；
- Alignment、Tile、Cluster、容量和 binary target 约束；
- `ArchTag` 与最终 `compute_xxx/sm_xxx` 的关系。

Mainloop 示例至少应解释 TMA Descriptor、Transaction Barrier、当前 `tcgen05.mma` Wrapper、1SM/2SM 协作范围、TMEM 累加器和 Accumulator Pipeline 的交接。

### 4.3 V0～V5 证据轴

技术深度和证据强度是两条独立轴。解释到 L5，不代表已经获得运行、数值或性能证据。

| 证据等级 | 能支持的声明 |
|---|---|
| V0 | 个人解释或概念草图，只能作为待验证假设 |
| V1 | 固定版本的官方接口、源码声明或文档契约 |
| V2 | 精确偏特化或类体源码，加上当前实例的类型推导或编译接受证据 |
| V3 | Kernel 成功启动并完成设备执行 |
| V4 | 与 Reference/Tolerance Check 对比后数值正确 |
| V5 | 固定硬件、输入、版本和测量方法下的性能结论 |

声明与最低证据应当绑定：

| 声明 | 最低证据 |
|---|---|
| 某 API 声明了特定契约 | V1 |
| 当前实例会生成某个具体类型 | V2 |
| Kernel 可以启动并完成设备执行 | V3 |
| 计算结果正确 | V4 |
| 某个 Schedule 或实现更快 | V5 |

精确 GitHub 链接只能提供 V1/V2 的一部分，不能代替编译实例、实际运行、数值验证或性能测量。

### 4.4 各类 Section 的最低深度

| Section | 最低深度 | 必须形成的可验收结果 | 停止位置 |
|---|---|---|---|
| 引言 / 问题定义 | L0～L1 | 数学、Shape、类型、Layout、当前实例和阶段边界 | 问题契约无歧义后停止，不展开模板内部 |
| 编程模型总览 | L1，辅以代表性 L2 | 每层输入、职责、输出、交接和代表 API | 读者可复述整体链路后停止 |
| Builder | L1～L3，至少 V2 | Builder 输入到具体 Schedule、DispatchPolicy 和 Collective 类型的推导；参数来源分类；资源约束 | 输出类型及参数来源完全确定后停止，不展开运行时状态机 |
| Mainloop | L2～L5，至少 V2 | 一个 K Stage 从 GMEM A/B、SMEM、Barrier、MMA 到 TMEM 的完整数据与状态闭环 | Stage 回到可写，Accumulator Ready 并交给 Epilogue |
| Epilogue | L0、L2～L5，至少 V2 | 一个 Subtile 从 Acc/C/Aux 经 Fusion 到 D Store 的数值、数据和状态闭环 | D Store 完成且 Accumulator Stage 释放 |
| Kernel | L2～L4，选择性 L5，至少 V2 | 一个 Work Tile 从 Scheduler 分配、Warp 协作到完成或退出的状态机 | Work Tile 生命周期和资源边界闭合 |
| Device | L2～L4；运行声明至少 V3；正确声明至少 V4 | `Arguments→Params→workspace→launch→sync→validation` 的所有权和异步边界 | 主机调用闭环完成，不展开 CUDA Driver 内部 |
| 对比与选择 | 深入到真正产生差异的最低层 | 候选沿相同维度比较，给出可执行选择规则 | 读者能根据新配置作出选择 |
| 完整示例 | L2～L4；称为“正确示例”至少 V4 | 可以构建、运行并验证结果 | 不声称性能时无需 V5 |
| 扩展与边界 | 只覆盖相对基线发生变化的层 | 明确哪些层改变、哪些层保持不变 | 差异闭合后停止，不复述整条主路径 |
| 总结 | 回收 L0～L4 的关键不变量 | 问题、类型、数据、状态和层间交接串联 | 不引入新类型、新机制和新证据 |

### 4.5 代表性工作单元推演

每个机制 Section 都应指定一个读者可以手工推演的代表性工作单元：

- Builder：一组当前配置如何落到具体 `CollectiveMma` 类型；
- Mainloop：一个 K Stage 如何从 GMEM A/B 进入 SMEM，完成 MMA 并产生 TMEM Accumulator；
- Epilogue：一个 Subtile 如何从 TMEM/C/Aux 经 Fusion 写回 D；
- Kernel：一个 Work Tile 如何被领取、计算、尾处理并完成；
- Device：一次 GEMM 调用如何从 `Arguments` 走到同步和数值验证。

如果读者只能复述组件名称，不能推演代表性工作单元，该 Section 的技术深度不合格。

### 4.6 关键参数与最小代码的判据

一个参数只要影响以下任意一项，就属于关键参数：

- 偏特化或 Schedule 选择；
- Shape、Layout、Stride 或 Alignment；
- Buffer 容量或 Pipeline Stage 数；
- CTA、Warp 或 SM 参与范围；
- 同步协议和状态推进；
- 数值语义；
- SharedStorage、TMEM 或动态共享内存需求；
- Grid、Block、Cluster 或 Launch 路径；
- 可观察行为或性能结论。

代码节选应满足双向约束：既删除与当前结论无关的实现噪声，也保留支撑完整因果链所需的类型、分支、字段和调用。不能以“避免代码堆砌”为理由删除决定行为的关键路径。

### 4.7 停止下钻条件

达到以下条件后停止继续展开：

1. 当前 Section 的最低 L 层已经达到；
2. 所有会改变当前路径的参数都已解释来源和影响；
3. 主路径的数据边、同步边和所有权边已经闭合；
4. 读者可以推演一个代表性工作单元；
5. 中心结论已经绑定匹配强度的证据；
6. 继续进入下一层不会改变当前结论或读者决策；
7. 更低层只剩指令编码、微架构延迟、未命中分支或工程实现细节。

一个低层细节只有影响类型选择、数据布局、资源容量、同步协议、参与范围、数值语义、可观察行为或性能结论时，才进入正文。没有 ISA、SASS 或性能声明时，不要求继续展开指令编码、流水槽、延迟和发射吞吐。

### 4.8 深度验收清单

- [ ] 本节达到对应 Section 的最低技术深度；
- [ ] 中心结论完成了适用的“输入约束→类型/决策→数据与资源→状态推进→架构原语→可观察输出”链路；
- [ ] 所有影响路径的参数都标明来源：用户指定、Builder 推导、偏特化固定或运行时提供；
- [ ] 主要数据边标明存储空间、Shape/Layout、owner 和 lifetime；
- [ ] 主要同步边标明 Producer、完成事件、Consumer、释放动作和状态轮转；
- [ ] 架构结论说明具体 Wrapper/原语、参与范围、操作数空间和完成语义；
- [ ] 当前实例给出具体值，或明确指出负责推导该值的组件；
- [ ] 中心结论的声明强度与证据等级匹配；
- [ ] 读者能够推演本节的代表性工作单元；
- [ ] 已达到停止下钻条件，没有无目的展开源码；
- [ ] 技术深度来自因果链，不来自 API 数量、代码长度、图的节点数或变体数量。

## 5. 正文文字的分类与写法

### 5.1 问题定义段

用途是建立问题、输入输出和阅读范围。

```text
目标问题
→ 当前实例
→ 主要阶段
→ 本文覆盖范围
```

问题定义共同构成一个心智模型，应使用连续普通段落，不拆成术语列表。

### 5.2 概念定义段

回答：它是什么、接收什么、负责什么、输出什么，以及与相邻层的边界在哪里。

推荐使用“一句定位 + 输入 + 主要职责 + 输出 + 相邻层边界”的完整段落。

### 5.3 源码机制分析段

用于解释类型、状态和硬件行为之间的因果关系，是技术教学正文的主体。

推荐内部顺序：

```text
源码入口
→ 类型或对象怎样形成
→ 运行时如何推进
→ 数据位置或所有权如何变化
→ 最终结果与意义
```

例如，下面这条链应写成连续机制段并配合状态图：

```text
producer_acquire
→ 取得 Empty Stage
→ TMA 绑定 Transaction Barrier
→ 硬件完成预期字节并使 Full Barrier Ready
→ consumer_wait
→ cute::gemm
→ consumer_release
```

将其拆成普通项目符号会把异步 Pipeline 表达成串行操作清单。

### 5.4 当前实例落地段

推荐结构：

```text
通用规则
→ 当前配置
→ 命中的具体路径
→ 产生的类型或行为
```

### 5.5 过渡段

过渡段用于连接相邻 Section，必须包含真实的状态或所有权变化，不能只写“下一节介绍某某”。

### 5.6 边界提示

边界提示集中说明固定 commit、CUDA Toolkit、目标硬件、binary target、`ArchTag`、贯穿实例、版本边界和证据范围。建议在引言后设置一个“阅读基线”提示框，后续章节只在偏离基线时补充。

## 6. 什么时候可以分点

解释性分点只用于平级、同维度、顺序无关的信息。

使用前执行四项检查：

1. 所有项目是否回答同一个问题？
2. 每一项是否可以套用相同句式？
3. 调换顺序后是否不改变含义？
4. 每项是否能在一至两句话内讲清？

四项全部满足，才使用分点。

一个更直接的测试是：如果项目可以自由重排，通常可以分点；如果重排后因果关系或时序被破坏，应使用段落、编号或图。

适合分点：

- CUTLASS 五层抽象；
- 五类 Warp Role；
- 多种平级 Scheduler Tag；
- 一个类型内彼此平行的资源成员；
- 若干互不依赖的静态约束；
- 预定义 Fusion Operation 的类别。

分点规范：

- 通常控制在 3～7 项；
- 两个对象具有明确对偶关系时可以分两点；
- 嵌套不超过两层；
- 每项以“对象：职责或差异”开始；
- 每项能够独立阅读；
- 列表前先给出总领句；
- 列表后如有结论，再用普通段落收束。

## 7. 什么时候不能分点

以下内容不能为了页面整齐而机械拆成项目符号。

### 7.1 因果链

例如：

```text
KernelSchedule
→ DispatchPolicy
→ CollectiveMma 偏特化
```

解释各对象的平级职责时可以分点；解释编译期如何完成分派时，应使用连贯段落和决策图。

### 7.2 状态机和异步 Pipeline

Producer、Consumer、Barrier、Stage、index 和 phase 之间存在状态依赖和并发关系，应使用连续机制段和流程图。

### 7.3 不同维度的混合信息

一个列表如果同时包含 Tile 划分、计算类型、Fusion Operation 和 Schedule，这些项目应拆成不同段落，或者整理为“参数 / 控制内容 / 当前配置”的表格。

### 7.4 每项需要完整机制分析

列表项超过两句话，并且还需要源码、例外和结论时，应升级为 H3 小节。

### 7.5 比较矩阵

多个候选在搬运方式、协作范围、Cluster 约束和适用场景等重复维度上比较时，应使用表格。

### 7.6 并发流程

多个 Warp Role 同时执行时，不使用普通编号列表。编号会暗示上一项完全结束后下一项才开始。

### 7.7 仅用于视觉结构化

不能把一个完整段落按句号拆成项目符号。页面上的列表数量不能作为结构质量指标。

## 8. 段落、编号与表格

### 8.1 连续长段

用于源码机制、因果链、状态机、所有权变化、异步重叠和设计约束。通常包含 4～7 个相互依赖的句子，但不设置机械字数上限。

满足以下任一条件时拆段：

- 主语发生变化；
- 从 Builder 切换到 Kernel；
- 从编译期切换到运行时；
- 从数据布局切换到同步状态；
- 从主路径切换到扩展路径；
- 从机制解释切换到适用范围。

不要仅因为段落视觉上较长就拆开。

### 8.2 普通段落

用于相对独立的定义、事实或局部解释，通常由 2～4 句组成。

### 8.3 短段落

只用于引出代码、引出图、声明公式假设、总结和过渡。连续出现多个一句话段落会产生明显的碎片感。

### 8.4 编号步骤

编号只用于严格先后关系，例如：

1. 调用 `can_implement`；
2. 查询并分配 workspace；
3. 调用 `initialize`；
4. 调用 `run`；
5. 同步 stream；
6. 执行 Reference/Tolerance Check。

编译期联合匹配和运行时并发 Pipeline 不使用编号步骤。

### 8.5 表格

至少三个候选，并且需要沿两个以上固定维度比较时，使用表格。表格单元格保持短小；需要机制分析的内容在表格后展开。

## 9. 代码规范

### 9.1 代码块分类

每个代码块应明确属于以下一种：

| 类型 | 用途 |
|---|---|
| 完整调用示例 | 读者需要照着构造或运行 |
| 用户接口示例 | 展示普通用户填写的模板或运行参数 |
| 源码节选 | 证明偏特化、字段或调用顺序 |
| 类型展开 | 展示 Builder 最终生成的类型链 |
| 执行伪代码 | 展示循环、角色和时序，不要求可编译 |
| 状态或数据草图 | 展示 PipelineState、形状、树或类型流 |
| 局部 API 示例 | 精确放大一个调用或转换 |

### 9.2 代码块的教学单元

代码前必须说明：

1. 这段代码要回答什么问题；
2. 它属于哪一类代码；
3. 输入和输出是什么；
4. 如果来自源码，固定 commit 和具体位置是什么。

代码后必须说明：

- 关键输入进入了哪个字段或类型；
- 这一步为什么存在；
- 改变的是编译期类型、运行时参数还是硬件状态；
- 哪些内容被省略；
- 下一步由谁消费结果。

代码后不逐行复述。

### 9.3 Caption

命名公式：

```text
动作 + 核心对象 + 关键结果或边界
```

有效示例：

- 构造 CollectiveMainloop：预留 Epilogue SMEM 并推导 Stage；
- 展开 Builder 输出：显式 CollectiveMma 类型契约；
- 消费当前 SMEM Stage 并执行 TCGen05 MMA；
- CollectiveEpilogue 的 Arguments → Params 参数降级；
- Requant EVT：后序求值树与量化输入节点。

避免使用“示例代码”“相关代码”“CUTLASS 源码”“Mainloop 代码”和“代码 1”等名称。

每个 Caption 在全文中应当唯一。只阅读 Caption 列表，也应能够大致复原文章的执行链。

### 9.4 完整代码、源码节选和流程骨架

只有读者下一步需要照着使用时才贴完整代码。“完整”指完成当前教学任务所需的契约完整，不要求粘贴整个官方 Example。

如果代码中仍有 `/* 调用者提供 */`、Reference Check 占位或省略的资源初始化，应命名为“流程骨架”，不能声称为完整可运行示例。

源码节选只保留证明当前结论的最小范围。不相关分支使用有意义的省略注释：

```cpp
// 省略：Sparse、Block-scaled 和 Pointer-array 分支
```

不要使用含义不明的裸 `...`。

### 9.5 代码注释

中文注释解释：

- 参数语义；
- 单位；
- 所有权；
- 存储位置；
- 编译期或运行时；
- 当前 Warp Role；
- 被省略的分支；
- 采用当前写法的原因。

源码原文保持逐字符一致时，中文说明放在代码块外。只要调整了源码结构、删除了分支或增加了中文注释，就标为“根据源码整理的节选”。

### 9.6 源码链接

具体实现结论使用固定 commit 和精确行范围：

```text
https://github.com/NVIDIA/cutlass/blob/<commit>/<path>#Lstart-Lend
```

要求：

- 一篇文档固定一个主 commit；
- 同一类型链不混用 `main` 和多个 commit；
- 链接覆盖支持当前结论的最小范围；
- 偏特化、Builder 生成位置和运行时函数分别链接；
- 伪代码引用多个源码片段时，分别列出精确链接；
- 官方 Example 同样固定 commit 和行号。

### 9.7 避免代码堆砌

- 一个代码块只回答一个教学问题；
- 两个代码块之间原则上有解释性正文；
- 同一组基础类型只完整定义一次，后续示例展示差异；
- 只需要证明五行逻辑时，不贴一百行函数；
- 组件超过三个、出现分支、并发或回环时，先用图说明关系，再用代码放大接口；
- 每个代码块在后文都应被引用和解释；
- 删除后不影响理解的代码块通常没有保留价值。

## 10. 技术图规范

### 10.1 图的分类

| 类型 | 主要回答的问题 |
|---|---|
| 编译期决策图 | 输入参数怎样选择偏特化并生成类型 |
| 数据布局和 Pipeline 图 | Buffer、Stage 和 Barrier 如何组织 |
| 多路数据汇合图 | 多条 Pipeline 在哪里等待、计算和写回 |
| Kernel 状态机图 | Collective、SharedStorage 和 Warp Role 如何协作 |
| Device 生命周期图 | Arguments、Workspace、Params、Launch 和验证如何衔接 |

简单事实和单步调用不需要画图。存在三个以上组件、多阶段状态变化、分支、并发、所有权或异步边界时，优先使用图。

### 10.2 方框的两层表达

每个功能框由两部分组成。

方框上方使用中文职责说明，回答“这个组件在当前流程里做什么”。推荐使用：

```text
动作 + 操作对象 + 结果或边界
```

方框内部保留 CUTLASS 类型、API、关键参数和理解关系所需的最小代码，回答“源码中具体是什么”。

### 10.3 框内文字

- 一个方框内部只使用一个多行文本对象；
- 根据参数组、调用阶段和语义单元换行；
- 不使用大量空格模拟布局；
- 不把每一行拆成独立的绝对坐标文字；
- 不使用 SVG `tspan` 或多个叠放文字排版；
- 框内左对齐，职责说明居中；
- 方框宽度适应关键标识符，不持续缩小字体；
- API 和缩写保留英文，解释句和关系句使用中文。

### 10.4 箭头与 Connector

箭头文字使用中文动宾关系，例如：

- 追加流水级维度；
- 等待 TMEM 累加器就绪；
- 传递工作块信息并接收 CLC 返回；
- 保存内核参数；
- 选择启动路径。

要求：

- 文字属于 Connector 自身的 Caption；
- 不使用独立悬浮文字模拟箭头标题；
- 一个箭头表达一个主要关系；
- 主方向尽量从左到右；
- Connector 不穿过方框和职责说明；
- 使用单层飞书原生 Connector；
- 不叠加 SVG 线条形成双线。

### 10.5 颜色语义

| 颜色 | 推荐语义 |
|---|---|
| 浅蓝 | 编译期类型、空间布局、参数构造和静态骨架 |
| 浅绿 | 运行时资源、SharedStorage、Arguments/Params 和中间状态 |
| 浅黄 | Warp 行为、同步、状态推进、验证和错误边界 |

颜色用于表达语义类别。同一张图中已经用颜色表示编译期和运行时后，应使用区域标题、边框或位置区分当前实例和候选分支。

### 10.6 画布与主题

- 使用透明画布；
- 不铺整张白色背景；
- 框上职责说明使用主题自适应文字；
- 浅色方框内可以使用固定深色代码文字；
- 飞书亮色和暗色模式都必须检查；
- 画布四周保留必要边距，不因对齐产生巨量空白。

### 10.7 同一画布中的通用流程、实例和候选分支

允许将三者放在同一画布，但必须满足：

1. 通用流程形成主干；
2. 当前实例形成一条可连续追踪的唯一路径；
3. 候选偏特化从明确的决策节点分叉；
4. 当前实例和候选区域有明确标签；
5. 候选分支能够追溯到选择条件；
6. 当前实例有唯一出口；
7. 颜色不同时承担“编译期/运行时”和“当前/候选”两套含义。

信息密度高本身不是拆图理由。出现大量交叉线、回头线、语义无法通过泳道区分，或者找不到唯一入口和出口时再拆图。

### 10.8 图与正文、代码的协同

图前说明图的范围、输入、输出、当前架构路径和阅读方式。图后说明最重要的不变量、状态和所有权交接，以及图中未容纳的关键约束。

正文不逐框复述图。图也不能替代异步边界、资源复用和正确性约束的明确说明。源码链接放在正文相应 API 上，不把长 URL 放进方框或箭头。

## 11. 跨文字、代码和图的技术一致性

必须逐项核对：

- 编译期类型生成与运行时执行；
- GMEM、SMEM、TMEM、RMEM；
- 数据 Buffer 与 `PipelineState`；
- 共享 Barrier 与 Warp 本地状态；
- Producer、Consumer 与硬件完成事件；
- Mainloop、Accumulator Pipeline 与 Epilogue 的所有权交接；
- 启动前检查、异步启动、同步执行和数值验证；
- `ArchTag`、PTX 指令协作范围和最终 binary target；
- CUTLASS 原生类型、文档别名和概念名称。

证据层级也应分开：

```text
源码声明或实现
→ 编译器接受
→ Kernel 成功启动
→ 设备执行完成
→ 数值结果正确
→ 性能结论
```

前一层不能自动证明后一层。

## 12. 整篇文档验收清单

### 12.1 全局

- [ ] 文章以问题和数据流开头；
- [ ] 有一套贯穿全文的当前实例；
- [ ] 主路径完整，扩展路径没有抢占主线；
- [ ] 正文保持教学文章形态，没有退化成 README；
- [ ] 工程步骤和复现信息集中在附录；
- [ ] 版本和适用范围集中声明；
- [ ] 正向职责先于边界和例外；
- [ ] 没有过度防御化表达；
- [ ] 没有实现过程、修改记录和汇报式文字。

### 12.2 Section

- [ ] 每节有明确入口和出口；
- [ ] 标题描述技术问题或行为；
- [ ] 开头定义本节职责；
- [ ] 中间形成因果闭环；
- [ ] 当前实例与通用机制建立映射；
- [ ] 结尾完成状态或所有权交接；
- [ ] H3 由独立问题驱动，而不是由类名驱动；
- [ ] 达到对应 Section 的最低 L0～L5 技术深度；
- [ ] 读者能够推演本节的代表性工作单元；
- [ ] 关键数据边、同步边和所有权边已经闭合；
- [ ] 中心结论与对应 V0～V5 证据等级匹配；
- [ ] 已达到停止下钻条件，没有无目的展开源码。

### 12.3 文字

- [ ] 长段用于因果、状态和所有权；
- [ ] 分点只用于平级、同维度、可重排信息；
- [ ] 编号只用于严格先后；
- [ ] 表格只用于固定维度比较；
- [ ] 没有把段落按句号机械拆成列表；
- [ ] 没有连续多个一句话段落；
- [ ] API 第一次出现时有中文职责；
- [ ] 没有连续堆叠大量英文术语。

### 12.4 代码

- [ ] 每个代码块类型明确；
- [ ] Caption 说明动作、对象和结果；
- [ ] 代码前说明问题、输入、输出和来源；
- [ ] 代码后说明机制、边界和下一环节；
- [ ] 源码链接固定到统一 commit 和精确行号；
- [ ] 省略内容有语义明确的注释；
- [ ] 语言类型与代码内容一致；
- [ ] 流程骨架没有被声称为完整可运行代码；
- [ ] 最小代码仍保留支撑当前因果结论的完整关键路径；
- [ ] 每个影响行为的参数都说明了来源和作用。

### 12.5 技术图

- [ ] 图有明确入口、主要路径和出口；
- [ ] 方框上方有中文职责；
- [ ] 方框内部保留源码对象；
- [ ] 箭头使用中文关系；
- [ ] 颜色表达固定语义；
- [ ] 透明背景支持明暗主题；
- [ ] 使用单层原生 Connector；
- [ ] 没有文字溢出、遮挡和双线；
- [ ] 图、代码和正文的内存空间与状态关系一致；
- [ ] 图包含当前 Section 最低深度要求的关键数据边和同步边；
- [ ] 图中的每个关键硬件动作都能映射到正文中的 Wrapper、原语或源码证据。

## 13. 飞书 CLI 文档修改执行协议

本节供使用 `lark-cli` 修改飞书文档的执行者或自动化 Agent 使用。它属于编辑工作流，不进入技术教学正文。

### 13.1 总体流程

飞书文档修改必须遵循：

```text
Observe
→ Diagnose
→ Patch Plan
→ Dry Run
→ Patch
→ Refetch
→ Verify
```

具体要求：

1. 读取当前目录、目标 Section、block ID 和 revision；
2. 诊断目标问题，并识别必须原样保留的代码、图片、画板、引用和其他资源；
3. 将修改拆成最小的 block 操作；
4. 对写入命令执行 `--dry-run`；
5. 使用刚读取的 `revision_id` 写入；
6. 检查返回的 `ok`、`result`、`warnings` 和新 revision；
7. 重新 fetch 修改范围，并按新 block ID 验证；
8. 核对非目标内容和资源完整性。

### 13.2 开始前读取当前工具规范

在第一次操作飞书文档前，读取当前安装版本提供的 Skill：

```bash
lark-cli skills read lark-doc
lark-cli skills read lark-doc references/lark-doc-fetch.md
lark-cli skills read lark-doc references/lark-doc-update.md
lark-cli skills read lark-doc references/lark-doc-xml.md
```

文档操作默认显式使用 `--as user`。如果资源来自 `--as bot` 链路，后续命令必须沿用同一身份，不能通过切换身份绕过权限或可见性边界。

### 13.3 Observe：读取最新状态

先读取目录和最新 revision：

```bash
lark-cli docs +fetch \
  --as user \
  --doc '<DOC_URL_OR_TOKEN>' \
  --scope outline \
  --max-depth 4 \
  --detail with-ids \
  --doc-format xml \
  --format json
```

定位目标标题后，读取完整 Section：

```bash
lark-cli docs +fetch \
  --as user \
  --doc '<DOC_URL_OR_TOKEN>' \
  --scope section \
  --start-block-id '<SECTION_BLOCK_ID>' \
  --detail full \
  --doc-format xml \
  --revision-id '<REVISION_ID>' \
  --format json
```

只有模糊关键词时，先定位最小范围：

```bash
lark-cli docs +fetch \
  --as user \
  --doc '<DOC_URL_OR_TOKEN>' \
  --scope keyword \
  --keyword '关键词1|关键词2' \
  --context-before 1 \
  --context-after 1 \
  --detail with-ids \
  --doc-format xml \
  --format json
```

如果返回 `<excerpt>`，说明只取得了容器或表格中的节选。修改前应使用其 `top-block-id` 重新读取完整 Section 或 range。

### 13.4 Diagnose：生成修改清单

写入前必须形成显式修改清单：

- 目标 block ID；
- 当前 revision；
- 计划使用的 update command；
- 要替换或插入的 XML；
- 必须原样保留的 `<pre>`、`<img>`、`<whiteboard>`、`<cite>`、`<source>`、`<sheet>`、`<bitable>` 和 `reference_map`；
- 修改前后的预期结构；
- 验证方式；
- 是否会导致 block ID 变化。

技术修改还应说明：

- 结论达到哪个 L0～L5 深度；
- 声明需要哪个 V0～V5 证据；
- 新内容与当前 Section 的入口、出口和代表性工作单元如何衔接。

### 13.5 Patch Plan：选择最小操作

| 命令 | 适用场景 | 约束 |
|---|---|---|
| `str_replace` | 唯一、简单的行内文本替换 | 先确认匹配唯一；不用于多行、资源或结构修改 |
| `block_replace` | 替换一个完整段落、标题、代码块、表格或容器 | 一次合并同一 block 的所有修改；替换后旧 block ID 可能失效 |
| `block_insert_after` | 在已知锚点后新增 Section 或 block | 使用最新锚点 ID；插入后获取新 block ID |
| `block_move_after` | 调整已有 block 顺序 | 移动后重新读取 Section，不能沿用旧 range 语义 |
| `block_delete` | 删除明确授权的一个或多个 block | 删除前核对精确 block ID 和影响范围 |
| `overwrite` | 用户明确要求完全重建整篇文档 | 默认禁止；可能丢失评论和暂不支持的资源 |

简单且唯一的行内替换：

```bash
lark-cli docs +update \
  --as user \
  --doc '<DOC_URL_OR_TOKEN>' \
  --command str_replace \
  --pattern '旧文本' \
  --content '新文本' \
  --revision-id '<REVISION_ID>' \
  --format json
```

在已知 block 后插入新内容：

```bash
lark-cli docs +update \
  --as user \
  --doc '<DOC_URL_OR_TOKEN>' \
  --command block_insert_after \
  --block-id '<ANCHOR_BLOCK_ID>' \
  --content @./patches/new-section.xml \
  --doc-format xml \
  --revision-id '<REVISION_ID>' \
  --format json
```

移动已有 block：

```bash
lark-cli docs +update \
  --as user \
  --doc '<DOC_URL_OR_TOKEN>' \
  --command block_move_after \
  --block-id '<DESTINATION_ANCHOR_BLOCK_ID>' \
  --src-block-ids '<SOURCE_BLOCK_ID_1,SOURCE_BLOCK_ID_2>' \
  --revision-id '<REVISION_ID>' \
  --format json
```

删除已经明确授权的 block：

```bash
lark-cli docs +update \
  --as user \
  --doc '<DOC_URL_OR_TOKEN>' \
  --command block_delete \
  --block-id '<BLOCK_ID_1,BLOCK_ID_2>' \
  --revision-id '<REVISION_ID>' \
  --format json
```

上述写入命令同样需要先附加 `--dry-run` 检查请求，确认后再执行正式写入。

普通局部编辑禁止使用 `overwrite`。文档中存在图片、画板、附件、评论或其他资源时，必须使用 block 级修改。

### 13.6 使用 XML 编写补丁

飞书文档的精确编辑默认使用 XML。多行内容保存为当前工作目录下的相对文件，例如 `./patches/mainloop-section.xml`，并通过 `@./...` 传入。

示例：

```xml
<h3>Producer/Consumer 状态机与 TCGen05 MMA</h3>
<p>MainloopLoad Warp 取得可写 Stage 后，将 TMA 事务与该 Stage 的 Barrier 绑定。</p>
<pre lang="cpp" caption="消费当前 SMEM Stage 并执行 TCGen05 MMA"><code>cute::gemm(...);</code></pre>
```

要求：

- 标签本身不转义；
- 文本中的 `<`、`>` 和 `&` 分别写成 `&lt;`、`&gt;` 和 `&amp;`；
- 代码必须位于 `<pre><code>...</code></pre>` 中；
- 代码语言和 Caption 必须准确；
- 复制既有资源时保留真实 token 和 `reference_map`；
- 不能把 `<img>`、`<whiteboard>`、`<cite>` 等资源降级为纯文本占位符。

### 13.7 Dry Run

正式写入前先检查请求：

```bash
lark-cli docs +update \
  --as user \
  --doc '<DOC_URL_OR_TOKEN>' \
  --command block_replace \
  --block-id '<TARGET_BLOCK_ID>' \
  --content @./patches/target-block.xml \
  --doc-format xml \
  --revision-id '<REVISION_ID>' \
  --dry-run \
  --format json
```

Dry Run 必须确认：

- URL 指向目标文档；
- command 与计划一致；
- block ID 正确；
- XML 内容完整；
- revision 与刚读取的版本一致；
- 没有无意包含其他 Section 或资源。

### 13.8 Patch：执行最小写入

Dry Run 通过后，移除 `--dry-run`：

```bash
lark-cli docs +update \
  --as user \
  --doc '<DOC_URL_OR_TOKEN>' \
  --command block_replace \
  --block-id '<TARGET_BLOCK_ID>' \
  --content @./patches/target-block.xml \
  --doc-format xml \
  --revision-id '<REVISION_ID>' \
  --format json
```

只有同时满足以下条件才视为写入成功：

```text
ok == true
result == success
warnings 为空
document.revision_id 已更新
```

如果响应包含 `updated_blocks_count`，其值应与预期修改数量一致；如果响应包含 `tips`，应阅读并判断是否需要额外验证。

如果 revision 冲突、权限变化或返回部分成功，应停止后续写入，重新 fetch 最新状态并重新制定 Patch Plan。

### 13.9 Refetch：使用新 revision 和新 block ID

`block_replace`、`block_delete`、插入和移动操作可能使旧 block ID 或旧 range 语义失效。每轮结构性写入后都要重新 fetch：

```bash
lark-cli docs +fetch \
  --as user \
  --doc '<DOC_URL_OR_TOKEN>' \
  --scope section \
  --start-block-id '<SECTION_BLOCK_ID_OR_NEW_ID>' \
  --detail full \
  --doc-format xml \
  --revision-id '<NEW_REVISION_ID>' \
  --format json
```

如果被替换的对象就是 Section 标题，并且 update 响应没有返回新 block ID，应重新执行 `--scope outline` 或 `--scope keyword` 定位新标题，不能猜测或继续使用旧 ID。

不能在连续多轮写入中反复使用旧 block ID。修改同一 block 的多处内容时，应合并成一次 `block_replace`。

### 13.10 Verify：验收目标和非目标内容

目标内容检查：

- 标题、段落、列表、表格、代码 Caption 和链接符合预期；
- 技术结论达到对应 L 层；
- 声明与 V 层证据匹配；
- Section 的入口、代表性工作单元和出口保持闭环；
- 代码正文、语言类型和源码链接没有意外变化；
- 新增内容没有把主路径改写成 README 或工程日志。

非目标完整性检查：

- 标题数量和顺序；
- 图片 token；
- 白板 token；
- 代码块数量、正文和语言类型；
- `<cite>`、附件、表格和 `reference_map`；
- 未修改 Section 的文本；
- 直达链接依赖的 block ID 是否变化。

对于代码块 Caption 修改，应比较更新前后的 `<code>` 内容是否逐字符一致。对于画板更新，应同时回读 raw nodes 和 preview。

### 13.11 白板和其他内嵌资源

更新已有白板时必须复用原 whiteboard token，不能通过新建空白画板替代目标画板。写入前先导出当前 raw 和 preview；整板覆盖需要用户明确授权。

导出当前白板：

```bash
lark-cli whiteboard +export \
  --as user \
  --whiteboard-token '<WHITEBOARD_TOKEN>' \
  --output-type raw \
  --output ./whiteboard-before.json \
  --overwrite \
  --format json

lark-cli whiteboard +export \
  --as user \
  --whiteboard-token '<WHITEBOARD_TOKEN>' \
  --output-type preview \
  --output ./whiteboard-before.jpg \
  --overwrite \
  --format json
```

使用原生节点数据覆盖既有白板前，先执行 Dry Run：

```bash
lark-cli whiteboard +update \
  --as user \
  --whiteboard-token '<WHITEBOARD_TOKEN>' \
  --input_format raw \
  --source @./whiteboard-openapi.json \
  --overwrite \
  --idempotent-token '<UNIQUE_IDEMPOTENT_TOKEN>' \
  --dry-run \
  --format json
```

确认整板覆盖范围后，移除 `--dry-run` 正式写入，并再次导出 raw 和 preview 做线上回读。`--overwrite` 只用于用户已经授权重建该目标白板的场景。

白板更新完成后检查：

- 节点和 Connector 数量；
- Caption 是否属于 Connector；
- 图片是否裁切；
- 明暗主题是否可读；
- 是否出现重复线、旧节点或全屏白色背景；
- 文档正文和其他资源是否保持不变。

图片、附件、Sheet、Base、评论和思维笔记应使用各自对应的 `lark-cli` 能力，不能通过普通文本替换伪造资源。

### 13.12 飞书 CLI 执行检查表

- [ ] 已读取当前 `lark-doc`、fetch、update 和 XML 规范；
- [ ] 全程显式使用正确身份；
- [ ] 已获取最新目录、目标 Section、block ID 和 revision；
- [ ] 已形成显式修改清单；
- [ ] 已识别必须保留的代码、图片、画板、引用和其他资源；
- [ ] 使用最小 block 操作，没有普通局部编辑使用 `overwrite`；
- [ ] 多行 XML 通过相对 `@./file` 传入；
- [ ] 已完成 Dry Run；
- [ ] 写入结果为 `ok == true`、`result == success` 且无 warning；
- [ ] 已使用新 revision 重新 fetch；
- [ ] 没有复用失效的旧 block ID；
- [ ] 已验收目标内容和非目标完整性；
- [ ] 技术深度、证据等级和停止条件符合本规范；
- [ ] 临时补丁文件已清理或明确归档。

## 14. 参考文档的后续核对项

根据本规范只读检查参考文档 revision 2171，仍有以下后续核对项。本节记录的是特定版本快照，不属于通用规范本身。

1. CUTLASS GitHub 链接仍混用 `main` 和多个 commit。当前 97 个相关链接中，51 个精确到行号，46 个没有行号。后续应统一主 commit，并优先补齐 Builder 和 Mainloop 的精确行链接。
2. “构造 CollectiveMainloop：预留 Epilogue SMEM 并推导 Stage”实际是 C++，代码块语言当前仍为 `Markdown`，应改为 `cpp`。
3. Device 的“执行参数检查、Workspace 初始化、异步启动与结果验证”仍含 stream 和 Reference Check 占位。在补齐前，更准确的 Caption 是“启动与验证流程骨架”。
4. Mainloop 附近存在“TensorStorage 在 TMEM 上按 SmemLayoutA/B 分配多阶段 A/B Buffer”的表述。该上下文和技术图描述的是 A/B 的 SMEM Buffer，这里的 `TMEM` 很可能应为 `SMEM`，应结合固定源码位置确认。

## 15. 可复用的总要求

> 技术教学文档先建立问题和编程模型，再沿一条具体实例解释类型、数据、状态、所有权和架构原语。每个机制 Section 必须达到对应的 L0～L5 最低技术深度，让读者能够推演一个代表性工作单元；中心结论必须绑定匹配强度的 V0～V5 证据，并在数据边、同步边、所有权边闭合后停止下钻。因果机制使用连贯正文，平级对象才使用分点，严格先后才使用编号，固定维度比较才使用表格。代码提供精确落点，图展示关系与时序，源码链接建立证据。正文保持教学主线，工程复现和异常分支集中放入附录。使用飞书 CLI 修改时，必须读取最新 revision，以最小 block 操作执行 Dry Run、Patch、Refetch 和 Verify，保护代码、图片、白板、引用与其他非目标内容。
