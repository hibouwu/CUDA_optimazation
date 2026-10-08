# GH200 微架构机制与建模证据审阅

日期：2026-10-08。对象：用户提供的三份材料——微架构文献综述、寄存器服务专题、当前模型缺口分析，以及本仓库的 EXP-04–20、R00–R19、B01、V07/V08 和预测代码。审阅后 EXP-04、R04、R07、R17 已按维护位置并入其他页面，下表链接指向新位置。具体实验设计见 [PLAN](PLAN.md)，现有条件规则见 [RULES](RULES.md)。本报告记录证据与缺口，不新增测量结果。

## 1. 审阅结论与研究范围

当前已经覆盖较多资源、指令和组合窗口，也有测前冻结的完整 GEMM 预测。主要不足是：条件时间表尚未普遍转成由请求、依赖、占用和释放事件驱动的规则。改变寄存器映射、地址序列、在途组织或并发来源时，模型仍经常需要重新校准聚合参数。对完整预测而言，最直接的缺口是 V08 暴露的每 SM 操作数供给、swizzle 补齐 tile、输出窗口与长窗口频率；PLAN 第 1 节把它们列为主线，先于本报告第 4 节的机制项。

本次用户明确要求完整机制规划，范围扩展为 **GH200、SM90a、单 GPU 计算路径的微架构服务模型**。RF 读写、reuse、返回竞争和调度具有独立研究价值，不再以“先证明能改善现有 GEMM 误差”为启用条件。范围不包括图形、光追、所有 GPU 世代或跨 GPU 互连；Grace CPU 与 GPU 间的数据迁移也不混入当前片上服务参数。

“覆盖全面”在此指每类机制都有已有证据、未识别部分与对应实验，不表示所有机制已经被唯一识别，也不要求一次运行全部参数的笛卡尔积。已有条件结果继续使用，新增对照在原实验内扩展，重点分离竞争解释；EXP/R/B 编号、原文档和运行入口继续沿用。真实硬件上难以区分的结构，可以保留等价的服务模型和适用范围，不强行命名物理端口或队列。

[V07](V07-rule-validation.md) 在限定条件内的总时间误差为中位 3.06%、最大 8.71%；[V08](V08-wider-validation.md) 扩大到 36 个新条件后为 2.61% / 35.34%，整体未通过。V08 的 21 个限定条件为 1.90% / 6.04%，属于测后子集分析。这些结果支持继续改进机制模型，不能把子集精度视为任意地址、配置和设备上的保证。

## 2. 证据怎样使用

| 证据类型 | 能支持的内容 | 在报告中的写法 |
|---|---|---|
| 公开语义 | 操作合法性、依赖顺序、等待涵盖的完成事件、作用域 | “PTX 规定……”；不由此推导未公开的吞吐或内部容量 |
| 本卡条件观察 | 固定机器码、输入组织、资源与计时边界下的数值关系 | 写明配置、窗口、变化因素，以及是否包含控制和消费者 |
| 候选物理解释 | 能解释现象、但尚有其他解释的结构或仲裁方式 | “支持/符合此解释”；列出仍不能区分的替代机制 |
| 不足以识别 | 多个未知量产生同一时间曲线，或观测扰动盖住差异 | 输出条件服务、范围或上/下界；不同时拟合多个不可辨识参数 |

例如同样有效字节的 TMA 变慢，可能是请求量变多，也可能是接收、传输或目标端服务降低。只有完整时间时，`时间≈请求量/服务率` 不能分别确定两项。计数器也必须注明所在层级、计量单位和其他流量来源，不能把全 kernel 的 L2 sector 全部归给某一次 TMA。

类似地，自依赖链的平均周期不能同时给出接收间隔、结果就绪与 RF 写回时间；多个 stage 出现吞吐平台也不能唯一给出物理队列容量。新的识别应围绕模型将要使用的事件和状态，保留这些可辨识性限制。

## 3. 当前模型实际表示了什么

[模型接口](../../../model/interfaces.md) 已定义 `eligible / issue / complete / visible / release`。当前 [v08_model.py](../../../../../../microbench/gh200_resource_campaign/access_rules/v08_model.py) 的实现主要是聚合主循环 `L=l0+l1×Ktile`、首段供给 `S`、输出 `E`、边界修正与 CTA 事件递推，再通过 `T=F+κ·C/f` 转换成微秒。它尚未显式保存 RF 源读取、reuse 内容、请求名额或逐请求缓存状态。

因此，新的机制识别有两种用途：能映射到现有事件的规则逐步替换聚合服务项；粒度更细的规则先形成可独立预测的指令或资源模块。无需为每个微基准强行增加一个完整 GEMM 修正系数。若用新规则计算了某段等待，就不能再把包含相同等待的旧校准时间完整叠加。

[v06_model.py](../../../../../../microbench/gh200_resource_campaign/access_rules/v06_model.py) 的 `dram_bytes()` 以每轮 A/B panel 集合近似访问，按固定顺序更新 `OrderedDict`，超阈值时清空。这不是逐请求 L2 模型。其中历史 `L2_CAP=30 MiB` 是估计器参数，不是本卡物理 L2 的 60 MiB 容量。代码变量名不能代替硬件证据。

## 4. 机制覆盖与缺口

下表的 M01–M25 仅作机制与跳转索引，不是新实验编号。每项在 PLAN 中都指定了原 EXP/R/B 实验的扩展位置与代码入口；“缺口”指尚不能从已有证据迁移的部分，不能读成整类实验从未做过。

| 机制与原实验扩展入口 | 已有依据 | 尚需识别的关系 |
|---|---|---|
| [M01 依赖与结果可用](PLAN.md#m01) | [R01](R01-readiness.md)、[EXP-18](../EXP-18-occupancy-aux.md) 的依赖/独立流窗口 | 生产者、消费者及源位置的就绪间隔；控制开销与真实依赖分开 |
| [M02 RF 读服务](PLAN.md#m02) | [R02](R02-register-service.md) 固定 40 reg/thread 后仍有 5.26% 源形式差异 | 普通/uniform/立即数、物理编号、源位置、宽度如何改变服务；bank/端口结构未定 |
| [M03 RF reuse 与缓存状态](PLAN.md#m03) | R02 已保存复用与轮换 SASS | 缓存建立、命中、消耗、覆盖与失效；重用距离和 warp 交错的影响 |
| [M04 写回、旁路、返回竞争](PLAN.md#m04) | R02 的保留结果、消费尾部和混合返回 | 返回相撞影响发出还是消费；目标编号与路径的关联；旁路和物理写入的区分 |
| [M05 warp 调度与子分区](PLAN.md#m05) | [R06](R06-issue-residency.md)、EXP-18 的 warp 数与资源扫描 | 活跃 warp 的竞争等价类、ready 集合与局部发出；不先指定映射公式 |
| [M06 算术、特殊、warp 交换服务](PLAN.md#m06) | [EXP-05](../EXP-05-fma.md)、[EXP-08](../EXP-08-fp8-int8.md)、[EXP-10](../EXP-10-ldmatrix-shfl.md)、EXP-18 | 各类指令的依赖与接收服务、数据宽度和执行路径组合；补缺而非重复峰值表 |
| [M07 前端、I-cache、分歧](PLAN.md#m07) | R06 中的前端条件扩展，现有循环 SASS | 静态足迹、PC 序列、分支/谓词与 fetch 竞争；代码展开的因果分离 |
| [M08 静态资源、驻留、local/spill](PLAN.md#m08) | EXP-18 已有 local 数组、spill 压力、寄存器/SMEM；[R16](R16-residency-quota.md) 有驻留对照 | 多 CTA 实际驻留、分配粒度、local 请求组织及 spill 与调度的组合代价 |
| [M09 动态 setmaxnreg](PLAN.md#m09) | R16 已验证先释放/后释放、配额算术和等待 | 多个合法申请如何由不同释放满足；等待与仲裁的条件关系 |
| [M10 普通 global/local LSU 请求](PLAN.md#m10) | [原 EXP-04](../EXP-13-global-rw.md#exp-04)、[R03](R03-access-demand.md)、EXP-18 | 地址、宽度、mask 到需求；在途合并、分批、返回及 local 路径影响 |
| [M11 SMEM 标量/向量/矩阵布局](PLAN.md#m11) | [EXP-09](../EXP-09-smem.md) 已测 stride/广播；EXP-10、R03 有矩阵与宽访问 | 访问拆分、不同 word 竞争和矩阵片段映射；不同路径能否共用同一规则 |
| [M12 cp.async](PLAN.md#m12) | [EXP-12](../EXP-12-cp-async.md)、[R05](R05-async-lifecycle.md) 的拷贝与流水 | 请求/commit/等待组织、部分复制、接收限制与跨线程消费；不与 TMA 合并参数 |
| [M13 TMA 描述符与请求形成](PLAN.md#m13) | [EXP-15](../EXP-15-tma-2d.md)、R03/R05、V08 行距对照 | stride、box、对齐、swizzle、有效区域到服务需求；请求放大仍待独立证据 |
| [M14 TMA 在途与排空](PLAN.md#m14) | [EXP-14](../EXP-14-tma-1d.md)、[EXP-16](../EXP-16-tma-pipeline.md)、R05 | 发出、后端完成、资源恢复与共享范围；软件槽数不是硬件队列数 |
| [M15 WGMMA 组与累加器](PLAN.md#m15) | [EXP-07](../EXP-07-wgmma.md)、R05、[R13](R13-async-retirement.md) | 指令数/group 数/累加器与输入槽分别产生的限制；第二 WG 的覆盖能力 |
| [M16 同步与 buffer 生命周期](PLAN.md#m16) | [EXP-11](../EXP-11-sync.md)、R05、R13、[R14](R14-stage-handoff.md) | 等待传播、phase、proxy 可见性、最后消费者释放；负载下 fence 的窗口 |
| [M17 缓存、合并与替换](PLAN.md#m17) | [R08](R08-waves-l2-reuse.md)、[R10](R10-layout-cache.md)、[B01](B01-bandwidth-cache.md) | 真实复访顺序、冲突/容量失效、在途同地址合并、输出挤出；不只看 footprint |
| [M18 地址翻译](PLAN.md#m18) | 当前缺少独立的翻译识别结果 | 页跨度、复访与并发翻译的条件效应；与 cache、物理地址分布分开 |
| [M19 L2 互连、DSM、多播](PLAN.md#m19) | [EXP-17](../EXP-17-cluster.md)、[R18](R18-cluster-boundary.md) | 来源/目标服务类别、跨流干扰、多播源读取与接收 payload、目标端节奏 |
| [M20 HBM 读写服务](PLAN.md#m20) | [EXP-13](../EXP-13-global-rw.md) 已有 7 种读写比、依赖/独立复制 | 固定比值下交错粒度、地址分布、并发流如何改变服务；无需重做简单比例覆盖 |
| [M21 原子](PLAN.md#m21) | EXP-18 已测 global/shared 同址与异址 | 竞争地址数、返回值消费、作用域和排序语义对并发服务的影响 |
| [M22 多路径共享资源](PLAN.md#m22) | [原 R04](R11-mixed-issue.md#r04)、[R11](R11-mixed-issue.md)、[R12](R12-smem-path-contention.md) | 配比、重叠时段、线程/CTA 位置改变时的联合服务；固定百分比不能通用 |
| [M23 真实 epilogue 与最终完成](PLAN.md#m23) | R05 与 [R15](R15-output-service.md) 已区分源读完/完整写出；V06/V08 使用真实输出窗口 | STSM/转换/整理与并发输出 CTA、槽数、最终排空的关系；标量 STS 帧不能替代 |
| [M24 CTA/cluster 调度及 OOB](PLAN.md#m24) | [原 R17](R18-cluster-boundary.md#r17)、R18、[R19](R19-critical-cta-tail.md)、V08 | 物理 tile、启动、工作分配、边界路径与共享服务如何决定全 kernel 包络 |
| [M25 频率、计时与观测扰动](PLAN.md#m25) | [原 R07](R09-inkernel-clock-stages.md#r07)、[R09](R09-inkernel-clock-stages.md)、[V03](V03-clock-rule.md)、V07/V08 | 周期到微秒的迁移、短长调用差异、打点偏差；最长周期 CTA 与最晚退出 CTA 分开 |

## 5. 需要保留的解释边界

### RF：源供给、结果可用和写回分别建模

R02 的 40 reg/thread 固定帧证明，分配数量相同仍存在稳定差异。但分支的物理源映射、reuse、控制字段和排程还不同，不能由 5.26% 直接得到“八对源的 RF 惩罚”。旧 22% 与新 5.26% 还来自不同设备/编译帧，不能视为同卡逐项归因后的差额。

保留结果数改变后，`T_tail=8+4P` 描述的是指定串行 FADD 消费代码。消费者可正确读取的时刻，不等于主 RF 已完成物理写入。混合返回变慢也可能来自输入地址、执行路径和发出竞争。因此 RF 应分别输出源服务状态、消费者就绪边与条件返回竞争；只有多个独立模式约束了同一结构，才尝试物理 bank/port 命名。

Huerta 等的公开稿对读端口、控制字段和 RFC 提出细化模型，同时承认所拟读策略未覆盖全部测试。其公开稿主要验证 Ampere/Turing，正式版与作者项目扩展到 Blackwell；这些结果为 GH200 提供候选解释，不提供已测 Hopper 参数。研究方案中的 BOW、Memento、多级 RF 与专利也只能说明“可能怎样设计”。

CuAssembler 上游列出的支持为 Pascal/Volta/Turing/Ampere，不能默认 SM90a 控制位可安全修改。位级实验需要先确认限定指令集合可往返组装且语义一致；如果做不到，就保存最终 SASS 和条件结果，不声称已做仅改 reuse 的因果对照。

### 访存：逻辑需求、路径流量和完成事件分别计量

有效字节、请求地址覆盖、L2 sector、DRAM 字节和多播接收 payload 不是同一个量。对于普通 global load，lane 地址可以给出 sector 覆盖；缓存命中、合并与写回会继续改变下层流量。Nsight 的 request/sector/wavefront 定义具有 L1TEX 路径语境，不能直接套到 TMA 内部。

TMA 的合法参数以 tensor-map API 为准：通常 base/stride 至少满足 16 B 对齐，特定模式更严格；box、元素宽度与 swizzle 也有关联。对齐到 128 B 可作性能对照，不是通用合法性门槛。不同 box 形状比较须注明实际请求与目的布局，否则相同有效字节仍可能不是相同服务需求。

R05、R15 已区分 TMA 输出源读完和完整写出。缺的是该区别如何进入真实 epilogue 与不同并发量的递推，不是从头证明两种事件不同。EXP-18 的显式 local、spill 与原子结果，EXP-13 的多比例读写结果也应作为已测基线；其循环窗口不能重新命名为裸指令延迟或内存控制器常数。

### 异步与同步：公开协议约束实验合法性

PTX 定义的 group 等待是完成顺序约束，不是内部队列容量。WGMMA 累加器消费必须等待覆盖对应操作的 group；矩阵指令的全 warp/warpgroup 一致执行要求也限制了 mask 扫描。[PTX 8.8](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)

TMA 输出 `.read` 只等待源读取，完整等待还涉及目的写及对发出线程可见。`mbarrier` phase 完成同时要求 arrival 与 tx-count 清零；等待者返回还包含观察成本。`setmaxnreg` 在 CTA 池内调节，新的配额须合法并初始化，不能假设跨 CTA 动态转借。generic/async proxy 的发布顺序须保留；fence 不能统一当作全设备排空。[同一 PTX 规范的 bulk wait、mbarrier、setmaxnreg 与 proxy 章节](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)

R13 的 ready/retire/reuse 事件对来自不同调用，不足以拼成一次完整绝对时间线。跨消费者等待传播需要同调用的必要事件，而不是把多份中位数当作可相减时间戳。

### 全 kernel：本地服务最长不一定决定最晚退出

同一 SM 内 `clock64` 差值适合表示本地窗口；不同 SM 的启动偏移、频率与局部窗口长度共同决定整卡结束时刻。最长 `clock64` CTA 不必是 `globaltimer` 最晚退出的 CTA。CTA ID、warp ID 也不能未经验证直接当作物理拓扑坐标。

改变 cluster 或 scheduler swizzle 可能改变补齐工作、CTA 数和轮次，所以性能差不能全部归到 multicast 或 OOB。当前 PLAN 的有效零/OOB 对照应固定物理工作列表；L2/NoC 研究则先输出可复现的干扰类别，之后才判断是否支持特定分区图。

计数器采集与普通计时分开解释。trace 扰动不只是总时间比例，也可能改变关键 CTA 或阶段重叠。冻结模型的成绩保留原含义；用 V08 失配开发的新规则只能在新留出条件上报告预测能力。

## 6. 规则接入的最小表示

每条候选规则须说明的内容只在 [PLAN 的共同模型接口](PLAN.md#model-interface)维护，这里不重复列出。各机制可共用现有事件接口，不必先建设统一仿真框架。

例如 TMA 在途规则可以先保存可观测的请求/字节占用和恢复关系，无须立即认定某个内部队列有固定 Q 项；RF 可以先输出编号组合与源位置的服务等价类，无须先指定取模 bank 公式。机制模块至少需要一个未参与识别的组织来检验迁移；完整 GEMM 还需要检查模块组合是否重复计账、是否选中正确的关键路径。

## 7. 参考资料与使用范围

以下为针对本次机制问题的阅读入口。官方文档用于语义，逆向论文用于候选结构与识别方法，开源代码用于检查实验或模型如何实现。读取这些资料不意味着采用其全部常数或部署整套模拟器。

1. **[PTX ISA 8.8 / CUDA 12.9.1](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)**。M09、M12–M16、M21：确认指令、等待、作用域与 proxy 语义；不是吞吐、端口数和物理队列规格。
2. **[Tensor Map Driver API 12.9.1](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)**。M13：描述符的维度、对齐、stride、box 与 swizzle 约束；合法性不代表不同描述符等速。
3. **[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)**。M10–M14、M17、M25：请求计量及分析边界；确认 metric 的层级与 replay 行为后再使用。
4. **[CUDA Best Practices 12.9.1](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html)**。M08、M10、M11：合并访存、SMEM bank 与读取广播、资源取舍；一般规则不能替代矩阵路径的实际分解。
5. **[General-Purpose Graphics Processor Architectures](https://link.springer.com/book/10.1007/978-3-031-01759-9)**，Aamodt、Fung、Rogers，2018。第 3 章指令/寄存器数据流，第 4 章存储系统；用于整理结构问题，历史微架构需与现代逆向证据比较。
6. **[Dissecting and Modeling the Architecture of Modern GPU Cores](https://upcommons.upc.edu/entities/publication/e31cdd46-ce96-41ba-8ba0-1919bb7adeb4)**，MICRO 2025；[早期公开稿 v1](https://arxiv.org/html/2503.20481v1)。M01–M07：控制字段、RF 与前端；版本和测试架构必须注明，不能称为 GH200 已验证实现。
7. **[作者的 modern-gpu-simulator-micro-2025](https://github.com/upc-arco/modern-gpu-simulator-micro-2025)**。查看模型状态和 SASS 控制字段如何落实到代码；其配置与实现不是厂商硬件规格，作者不承诺长期维护。
8. **[Hopper 微基准论文 v1](https://arxiv.org/html/2501.12084v1)** 与 [NVIDIA-Hopper-Benchmark](https://github.com/HPMLL/NVIDIA-Hopper-Benchmark)。M11、M13–M15、M19：借鉴多层次测试；主卡 H800 PCIe，TMA 窗口含事务设置与等待，不能把论文数字当成本卡裸延迟。
9. **[Uncovering Real GPU NoC Characteristics](https://people.ece.ubc.ca/aamodt/publications/papers/realgpu-noc.micro2024.pdf)**，MICRO 2024，与 [GPUNetBench](https://github.com/chrirocca/GPUNetBench)。M19：延迟、服务率与干扰分别研究；位置相关延迟不自动意味着处处存在相同比例的带宽差。
10. **[Optimal Software Pipelining and Warp Specialization / Twill](https://arxiv.org/html/2512.18134v1)**。M14–M16、M22：依赖图与资源预约约束；“最优”相对于给定机器模型，当前循环控制和 tile 选择仍有限制。
11. **[Accel-Sim 2026 论文](https://arxiv.org/abs/2608.22602)** 与 [官方项目](https://github.com/accel-sim/accel-sim-framework)。异步执行和分区存储建模参照；运行轨迹驱动与只根据源/配置预测是不同任务。
12. **[A Compile-Time Managed Multi-Level Register File Hierarchy](https://research.nvidia.com/publication/2011-12_compile-time-managed-multi-level-register-file-hierarchy)**，MICRO 2011。M03、M04：分层 RF、短寿命结果与编译器管理的设计思路；不能证明 Hopper 采用了该方案。
13. **[BOW: Breathing Operand Windows](https://www.cs.ucr.edu/~nael/pubs/micro20-bow.pdf)**，MICRO 2020。M03、M04：操作数旁路与减少读写的候选设计；用于构造可区分解释，不提供本卡旁路常数。
14. **[Memento](https://upcommons.upc.edu/entities/publication/04d929ad-6bb3-46b3-ba5e-186704b951e2)**，ISCA 2024。M03、M05：复用距离、RF cache 与调度协同；研究方案与真实产品逆向结论分开。
15. **[MaxAs SGEMM](https://github.com/NervanaSystems/maxas/wiki/SGEMM)** 与 [Control Codes](https://github.com/NervanaSystems/maxas/wiki/Control-Codes)。M01–M04：手工布局、reuse 与控制因素的对照方法；主要 Maxwell 背景，不移植 bank 数或位域结论。
16. **[CuAssembler](https://github.com/cloudcores/CuAssembler)** 与 [教程](https://github.com/cloudcores/CuAssembler/blob/master/Tutorial.md)。M01–M04：SASS 对照工具候选；先验证 SM90a 限定指令可用性，不能将 README 的“可能扩展”当成已支持。
17. **[gpu-arch-microbenchmark](https://github.com/sjfeng1999/gpu-arch-microbenchmark)**。M02、M03：`reg_bankconflict` 等对照起点；其 Turing 等测试结果不作为 GH200 规则。
18. **[RRZE-HPC/gpu-benches](https://github.com/RRZE-HPC/gpu-benches)**。M10、M13、M17、M20：地址、cache、吞吐微基准实现；复用前对齐工作量和计时端点。
19. **[Colfax TMA 教程](https://research.colfax-intl.com/tutorial-hopper-tma/)** 与 [GEMM 流水线教程](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)。M13–M16、M23：把生产、消费和槽生命周期对应到 CUTLASS/CuTe 代码；当前实验仍固定自己的 CUTLASS 版本。
20. **[NVIDIA operand collector 专利](https://patents.google.com/patent/US7834881B2/en)** 与 [source operand caching 专利](https://patents.google.com/patent/US8639882B2/en)。M02–M04：结构与仲裁的候选设计；专利存在不证明 GH200 实现了它，也不能据此否定新的逆向结果。
