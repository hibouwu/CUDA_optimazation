# 机制模型

更新：2026-10-18。机制模型由[程序图](program/cutlass_program_graph.md)、[硬件服务规则](hardware_rules/README.md)和[组合](composition.md)构成，目标与验收见 [GOALS §3](../../GOALS.md#3-机制模型)。本页把 cfg_b（128×128×64 pingpong、cluster 1×1、S6/S4）这类 TMA＋WGMMA GEMM 拆成机制，逐项写明它给程序图的哪些事件定价、已有证据的类型、规则辨识到什么程度、组合里现用什么模块，以及还缺什么。数值与判定以链接的研究页和实验页为准，本页只写摘要。

## 1. 执行路径

```text
CTA 启动                                                  ① 调度与启动
  entry → producer_first_work；setmaxnreg 调整配额          ⑩ 寄存器配额与驻留
  按工作列表依次取输出 tile j

每个 Ktile g（slot = g % S）
  producer_acquire(g) ← consumer_release(g−S)（g≥S；前 S 组初始槽为空）               ⑥ 槽释放与再取得
  3 条 UTMALDG.3D：A 16 KiB，B0/B1 各 8 KiB                  ② TMA 请求
    源：L2 命中，或 miss 后读 HBM                           ③ 源服务
    目的：写入本 CTA 的 SMEM 槽                              ② TMA 请求、⑨ SMEM 共享
  full：1 次 arrival＋32768 B 事务 → consumer_wait 返回      ④ 输入同步
  fence＋8 条 HGMMA.64x128x16 → commit                       ⑤ WGMMA 计算、⑨ SMEM 共享
  每输出首 Ktile 只 commit；后续 wait<1> → consumer_release(g−1)       ⑥ 槽释放与再取得

每个输出 tile j（两个数学 WG 交替）
  order barrier.wait → mma → arrive → mma_tail（wait<0>）    ⑦ 数学 WG 交接
    mma_tail 后释放该输出最后槽                              ⑥ 槽释放与再取得
  8 个 64×32 子块：STS.64 → fence／同步 → TMA store → store_tail
  与另一 WG 的下一个主循环重叠                               ⑧ 输出、⑨ SMEM 共享

全程                                                       ⑪ 有效频率：cycle → ns
观察                                                       ⑫ 包络 → plain 时间
```

操作、依赖与槽代次的来源见[程序图 §2–3](program/cutlass_program_graph.md#2-对象操作与依赖)。程序图不含时间；①–⑫ 给这些事件定价。

## 2. 证据类型与规则状态

| 证据类型 | 含义 | 用法 |
|---|---|---|
| 零负载 | 目标服务单独运行，窗口只含该服务 | 竞争规则的基线 |
| 复合 | 窗口还含循环、等待、同步或消费 | 只作上界或条件服务，不当作单个事件的延迟 |
| 吞吐 | 指定并发与组织下的持续速率 | 上限参照 |
| 结构 | 排除或确立一种组织方式 | 约束候选形式 |

规则状态：**已核对**指程序事实与源码、SASS 或执行语义一致；**获得支持**、**被反证**、**信息不足**按 [C2](../../GOALS.md#35-c2-机制主张) 判定；**工作假设**是已采用、未通过区分检验的候选。组合模块的资格按[组合 §2](composition.md#2-模块与资格)的五种定义。各事件现有的零负载点、复合上界与缺口见 [GH200 L0 零负载服务点](../../hardware/gh200_sm90/L0.md#零负载服务点)。

## 3. 机制分解

| 机制 | 定价的事件 | 已有证据 | 已辨识 | 组合现用 | 缺口 |
|---|---|---|---|---|---|
| ① 调度与启动（[程序图](program/cutlass_program_graph.md)、[R19](../../experiments/gh200_sm90/Q01-scheduling/R19-critical-cta-tail.md)） | 工作列表、CTA→SM 放置、`entry → producer_first_work` | 结构：工作映射与 739011 的 12 条件、120 个 dual 进程逐 CTA 一致；R19 中 P2 的最晚 CTA 漏选来自 CTA 内部服务排序，入口偏斜不能解释。复合：V09 前缀 P0=1608 ns | 已核对：工作映射。前缀未拆分 | 工作映射：已核对；入口前缀：经验替代 | 前缀的组成（生产者到首个 A 请求、消费者准备）；CTA 启动分布；原 12 条件的最晚 CTA 命中 1/120 |
| ② TMA 请求：发出、在途与目的投递（[多 CTA 共享服务](hardware_rules/TMA-SHARED-SERVICE.md#4-结论)、[候选与反例](hardware_rules/TMA-GEMM-HYPOTHESES.md)） | 每 Ktile 3 条 `UTMALDG.3D`（A 128×64，B0/B1 64×64，SW128）的发出 → 写入目的 SMEM → 事务计入 full | 吞吐：整卡 tensor 读、写在 payload ≥8 KiB 时 3.8–3.9 TB/s（[EXP-15](../../experiments/gh200_sm90/Q02-tma-request/EXP-15-tma-2d.md)）；单 CTA 读随缓冲容量从 14.7 升至 42.3 B/cycle（[EXP-16](../../experiments/gh200_sm90/Q02-tma-request/EXP-16-tma-pipeline.md)）。复合：EXP-16 的 S1R1 每轮一个 16 KiB 1D bulk 请求约 1110 cycle，含发出、完成等待与 CTA 同步，只是单请求往返的上界，指令路径与 cfg_b 的 tensor 请求不同。结构：同 GPC 对照窗口增加 21.8%–22.1%，跨组接近零。cfg_b 原生三请求的单 CTA 短链（[COMPUTE-01 D](../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#d-measured)）数值与路径通过，trace 扰动超限。[固定分配行距轮](../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261017-Q02-tma-stride-002/EXECUTION.md)完成18数值臂与18plain臂，Q4下B32响应近零、B80/B112约增加4.3%–5.4%，Q1包络响应不稳定同向 | 获得支持：同 GPC 的 SM 共享服务；每 GPC 2 槽服务通过 N32 的 6 条新条件；本地计算干扰响应，排除仅频率解释。被反证：同质 FIFO＋统一入口间隙；每条 tensor 独立经 GPC core 收费（G3 约 79 µs，明显高估）；2 槽服务迁移到 cfg_b 源坐标的 4 个订阅版本（最大误差约 45%）。按请求大小分道的 2 个版本未通过筛查 | TMA 输入服务：假设 | 零负载：EXP-15 只有同 box 形状、单请求、无 SW128 的复合窗口（16 KiB 约 766、8 KiB 约 644 cycle），缺三请求同时在途的窗口。交互：活跃 CTA 1/16/132 下的就绪延迟；在途容量及其共享范围；源信用在源就绪还是目的完成时归还 |
| ③ 源服务：L2 与 HBM（[L2 服务](hardware_rules/L2-SERVICE.md)、[HBM 服务](hardware_rules/HBM-SERVICE.md)） | 源请求命中 L2，或 miss 后读 HBM → 源就绪；输出 D 的写回 | 吞吐：大工作集读 3.52、写 3.79 TB/s（[EXP-13](../../experiments/gh200_sm90/Q03-source-service/EXP-13-global-rw.md)）。结构：1:1 读写合计 3.43 TB/s，读、写速率不能相加。[L2-01](../../experiments/gh200_sm90/Q03-source-service/L2-01-saturation.md) 固定工作集并发曲线，开发 48 进程 | 获得支持：两域映射 GPC{0,4,5,6}/{1,2,3,7}，每域 66 SM；近期源状态（只交换暖机次序，GPC2/3 早窗口缩短 41.30%/39.53%）；局部域容量。被反证：全局有效容量；L2 并发曲线的 6 个预置形式（稳定单调反例）；HBM 混合窗口的 max 与 sum 两种外推。[CPU源隔离](../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-service-rules/20261018-Q02-source-isolation-001/score.json)中B32/B80的源就绪、投递及最终窗口响应均为0，现有源服务本身未产生行距响应 | L2 与源地址状态：假设；外存读 miss 后端（可选）：假设 | 零负载：L2 命中与 HBM miss 的单请求延迟。按访问序列更新的缓存规则（GPC5 仍失配）；真实 tensor 请求下的 miss 与写回服务；行距响应：A+16 B 实测 +28.94%、预测 +0.09%，B+80 B 实测 +18.13%、预测 0 |
| ④ 输入同步：full 完成 → wait 返回（[同步服务](hardware_rules/SYNC-SERVICE.md#3-旧观测约束与剩余缺口)） | full（1 次 arrival＋32768 B 事务）满足 → `consumer_wait` 返回 | 复合：mbarrier arrive＋等待相位 218/226 cycle（128/256 线程齐步，[EXP-11](../../experiments/gh200_sm90/Q04-sync-slot/EXP-11-sync.md)）；barrier_return 三轮 46 条件：入口晚于全部信号时等待调用窗口 78–174 cycle，入口早于信号时最后必要信号→返回区间 [47,82]–[94,128] cycle，含调用组织，无 TMA 与 WGMMA。零负载：与 `consumer_wait` 同形式（try_wait 后按需 wait，1 个 arriver、128 个 waiter）为 78 cycle，阻塞时 [47,82] cycle | 已核对：计数、phase 与槽代次。观测支持把调用形式与进入状态（已就绪／阻塞）作为变量。被反证：B/C 零额外控制的具体形式（plain 响应，[COMPUTE-01 B/C](../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#bc-measured)） | 输入同步与槽释放：假设 | GEMM 中两条返回路径的价格 `r_path`，即 full 完成与 wait 返回之间的关系 |
| ⑤ WGMMA 计算（[计算服务](hardware_rules/COMPUTE-SERVICE.md)） | `compute_batch`（fence＋8 条 HGMMA.64x128x16）→ commit → `compute_complete` | 吞吐：m64n64k16 SS 在 wait≥3 时 4095.9 FLOP/cycle（[EXP-07](../../experiments/gh200_sm90/Q05-wgmma/EXP-07-wgmma.md)），N128 为 4095.1–4095.5（R00-B）。复合：依赖链 T(n)=555+105n cycle，是单条往返上界；[EXP-20](../../experiments/gh200_sm90/Q14-composition/EXP-20-bf16-wgmma-tile.md) 平均每条约 150 cycle，含 fence、commit、wait0 与循环。复合：cfg_b 驻留短链 T=769+512·Kt cycle，4/13 Ktile 开发、32 Ktile 留出通过（COMPUTE-01 A）；[Q05两wait冻结长度验证](../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261018-Q05-compute-wait-002/EXECUTION.md)的每组完成滞后形式最大误差0.020405%，固定截距形式wait0误差11.156922% | 512 cycle/Ktile 的条件长度外推通过；A首轮名义与条件两种解释预测重合，C2 信息不足；Q05每组完成滞后形式通过匹配驻留窗口的两wait长度验证，固定截距形式被反证；计算量干预的增量接近名义工作量。769 cycle及Q05的677.5 cycle均为探针复合固定项，不迁移 | WGMMA 计算服务：已测 | issue、commit、wait 的拆分；256 cycle 提交提前量（工作假设）；与供给、同步联合后的持续服务 |
| ⑥ 槽释放与再取得（[同步服务](hardware_rules/SYNC-SERVICE.md)、[程序图 §3](program/cutlass_program_graph.md#3-槽代次与两种调度)） | `consumer_release(g)`（由后续 Ktile 的 `wait<1>` 或末尾 `wait<0>` 触发，empty 128 次 arrival）→ `producer_acquire(g+S)` | 结构：计算量干预支持有限槽复用对停顿的影响。零负载：与 `producer_acquire` 同形式（直接 wait＋后继事务登记，128 个 arriver、1 个 waiter）为 174 cycle，阻塞时最后信号→返回 [87,121] cycle | 已核对：`release(g) → acquire(g+S)`。当前组合保留 220 cycle 的释放→复用延迟工作假设；原筛查中加 220 cycle 时 S4 比 S6 慢约 5%，空间跨度仍只有约 55 ns | 程序依赖与槽代次：已核对；释放价格归输入同步与槽释放：假设 | empty 最后到达 → producer 返回在 GEMM 中的价格，零负载点的迁移未验证；stage6→4 响应模型 3.90%、实测 10.47% |
| ⑦ 数学 WG 交接（[程序图 §3](program/cutlass_program_graph.md#3-槽代次与两种调度)） | `math_wg_order_barrier.wait → mma → arrive → mma_tail`；下一组在第一次 arrive 后进入 MMA | 结构：真实 pingpong 单 CTA 三输出中，output1 的 FIRST 早于 output0 的 MAIN_END 47 cycle，晚于其 mma_return 421–430 cycle，只解释 trace 程序（[COMPUTE-01 E](../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#e-measured)）；[R14](../../experiments/gh200_sm90/Q06-handoff-output/R14-stage-handoff.md)：中等 K 与大输出时交接实为重叠 | 已核对：程序顺序。观测支持区分数学交接与尾部完成边界 | 聚合模式的交接参数 gm、h：经验替代 | order barrier 交接与 `mma_tail` 尾部的价格 |
| ⑧ 输出（[R15](../../experiments/gh200_sm90/Q06-handoff-output/R15-output-service.md)、[SMEM 服务](hardware_rules/SMEM-SERVICE.md)） | 每输出 8 个 64×32 子块：STS.64 写 8 KiB → fence／同步 → TMA store → `store_tail`；两个源槽，延迟提交前一子块 | 结构：STS.64 完整 lane 地址与两源槽已核对（[SMEM-01](../../experiments/gh200_sm90/Q07-smem/SMEM-01-replay.md)）；只延后其他 CTA 的输出仍使 stage4 早段 10/10 缩短，局部服务依赖全卡执行组织。复合：R15 的输出窗口条件观测 | 无可迁移规则；V09 末次输出分项误差中位／最大 3.73%/35.60% | 输出服务：聚合模式经验替代（E、r、w、we、Etail），细事件[独立输出链](hardware_rules/SMEM-SERVICE.md#2-可执行写块假设)：假设；写块价格 q：假设 | 写块、TMA store 与源复用的服务；与下一主循环的重叠，比例 r 没有程序依据 |
| ⑨ SMEM 多路共享（[SMEM 服务](hardware_rules/SMEM-SERVICE.md)、[联合服务](hardware_rules/joint_service.md)、[m11](hardware_rules/M11-SMEM.md)） | TMA 写入（每 Ktile 32 KiB）、WGMMA 操作数引用（每 Ktile 48 KiB）、STS.64（每子块 8 KiB）、TMA store 源读取 | 吞吐：8/16 B 读、写各 128 B/cycle。结构：同时读写合计仍约 128 B/cycle，不能相加（[EXP-09](../../experiments/gh200_sm90/Q07-smem/EXP-09-smem.md)）。复合：独立区域下 TMA＋WGMMA 接近完全重叠，WGMMA＋STS 额外约 10–11%，TMA＋STS 额外约 6–8%（[R12](../../experiments/gh200_sm90/Q07-smem/R12-smem-path-contention.md)）。[Q07真实四组织](../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261018-Q07-input-output-001/EXECUTION.md)12进程合格，同seed并发比串行少918–955 cycle、比单独窗口max多730–742 cycle；共同窗口含计时内fragment准备及两阶段join | 被反证：独立流直接取 max（部分配对）。m11 普通 LDS/STS 两次冻结留出未通过。共享约束保留，未定位端口或仲裁；Q07为开发联合窗口，机制信息不足 | SMEM 需求与输出写块：假设 | 准备完成后的输入／输出联合响应；实际STS.64与WGMMA的匹配联合及GEMM迁移 |
| ⑩ 寄存器配额与驻留（[寄存器服务](hardware_rules/register_service.md)、[驻留](hardware_rules/residency.md)） | 启动时 producer 释放到 40、consumer 申请到 232 reg/thread，共 12 个 warp；S4/S6 均 1 CTA/SM | 结构：384×168 = 128×40＋256×232 = 64512，无 spill。复合：R16 中 consumer 申请窗口 144.5 cycle，把 64/256 次 producer 动作移到释放前后变为 2106.5/7866.5 cycle | 被反证：配额申请是固定成本。工作假设：申请由可满足的释放事件约束，R16 支持该条件依赖 | 启动寄存器配额（可选）：假设；SM 容量与活动：已核对 | 其余 producer warp 释放与 consumer 申请的时长，只出现在启动段 |
| ⑪ 有效频率（[频率主线](hardware_rules/FREQUENCY-MECHANISM.md#4-结论)） | 各服务的 cycle → ns；G3 主循环 S6 首／后段 1.537/1.750 GHz，S4 为 1.572/1.766 GHz | 结构：其他 131 个 SM 改变活动时，目标 SM 驻留服务的周期变化 −1.78%至+0.03%，ns 变化 +8.31%至+10.73%；空闲后首阶段升频 52.10%–53.92%（d044） | 获得支持：驻留服务按 cycle 计，排除固定 ns；空闲后升频。半经验：运行前饱和响应式通过 cfg_b 的 8 条 C3 留出，时间误差中位 0.81%、最大 2.96%。未达标：一阶、双松弛、阻尼二阶状态方程；CTA 阶段活动驱动（S6/S4 最大阶段误差 88.36%/24.07%） | 有效频率：经验替代 | 阶段频率的状态规则，区分当前活动与前序状态 |
| ⑫ 包络到 plain 时间（[MODEL-CANDIDATES](../semi_empirical/MODEL-CANDIDATES.md#v09-model)） | dual 包络 W → plain 完整时间 | 复合：`T_plain = F + κ·W` 由参照条件的 dual 包络与 plain event 拟合；参照的 event−包络在 V09 为 3.86–3.98 µs，C1-01 两次作业为比 V09 多 0.85–3.2 µs 及 3.73–4.72 µs（[C1-01](../../experiments/gh200_sm90/Q16-config-selection/C1-01-selection.md#恢复前要定的事)）；COMPUTE-01 B–E 的 trace 扰动超过 5% 界限 | 无 | 包络到 plain 时间：经验替代 | event−包络的组成及其作业间变化；trace 端点迁移到 plain 的条件 |

cfg_b 不经过的路径：

| 路径 | 使用者 | 已有证据 | 缺口 |
|---|---|---|---|
| cluster 多播与 DSM | cfg_a（2×1）、cfg_c（1×2）的跨 rank 多播；程序图已展开多播投递与跨 rank empty 到达 | 吞吐：多播接收量 5.06/4.77/5.05 TB/s（C=2/4/8），DSM 读 2.2–2.5 TB/s（[EXP-17](../../experiments/gh200_sm90/Q12-cluster-dsm/EXP-17-cluster.md)） | 多播请求的服务规则 |
| 普通 L1／LSU | cfg_b 动态普通 global／local 需求为 0；79 处静态 LDC／ULDC 由现有复合服务覆盖 | [L1／LSU 服务](hardware_rules/L1-LSU-SERVICE.md)、EXP-13 小工作集 `.ca`／`.cg`；[原8留出24进程](../../../../../results/gh200_resource_campaign/access_rules/Q13-l1-lsu/L1-01-minimal/20261017-Q13-l1-minimal-002/EXECUTION.md)的快复用规则在声明的普通LSU条件内通过 | TMA GEMM 主循环不经过；实际物理cache分区未知 |

## 4. 现状

当前完整假设版本见[10-20同版交付](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261020-Q14-current-system-001/README.md)：保存当前源码、12份输入、①–⑫的具体接法、三个完整条件的时间线和全部开发评分；局部未知处按工作假设执行，独立C2/C3仍未通过。

一项机制在组合中由已核对或已测的模块提供，且规则在 GEMM 工况下按 C2 获得支持，才算建成；完整时间按 [C3](../../GOALS.md#53-c3-完整时间预测) 独立检验。

- 已核对或已测：① 的工作映射，⑥⑦ 的程序依赖与槽代次，⑤ 的计算体 512 cycle/Ktile。
- 有结构支持、没有量化规则：② 的同 GPC 共享，③ 的两域映射与近期源状态，⑪ 的周期域与空闲升频。
- 其余为假设或经验替代。cfg_b 原 12 条件的 plain 绝对误差中位／最大 10.225%/25.422%，stage 响应 3.90%（实测 10.47%），最晚 CTA 命中 1/120（[COMPOSITION-01](../../experiments/gh200_sm90/Q14-composition/COMPOSITION-01-replay.md#2-回归与原评分)）。
