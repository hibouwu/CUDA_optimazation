# 机制组合：首版

更新：2026-10-17。**聚合、请求级候选、阶段频率、经验前缀和 plain 映射已接通；细事件组合尚未通过 C2/C3。** 本页维护第一版组合的范围、模块来源与资格、连接方式、输出和完成判据；目标与验收见 [GOALS §3.4](../../GOALS.md#34-组合)，通用的图组织与层级见[接口 §7](../common/interfaces.md#7-程序图的组织与组合)，输入字段见[共同输入](../common/inputs.md)。

## 1. 范围

cfg_b：128×128×64 pingpong、cluster 1×1、6 stage；同一组合另跑 4 stage 构建，用来检查 stage 响应。FP16 输入、FP32 累加与输出，α=1、β=0，持久调度，原参考卡 GPU-43269fbc，plain 协议。先只做 cfg_b，原因是它没有 cluster 跨 rank 依赖，且已有 V09 的 stage6→4 反例（实测慢 10.47%，冻结候选响应为零）。

## 2. 模块与资格

### 两种开发模式

| 模式 | 主循环来源 | 能输出的事件 | 用途 |
|---|---|---|---|
| 聚合模式 | V09 的整段主循环 L = b + K_t·max(q/f, θᵀx)（`supply_predict`），已含计算下限，整段替代下表的 TMA 与 WGMMA 两个模块 | 每个输出 tile 的主循环与输出端点 | 接通程序图、调度、输出与时间映射，检查端点对照；不检验 stage 响应的机制 |
| 细事件模式 | 具备请求完成语义的 TMA 服务（每个请求的就绪时刻）与按 Ktile 的计算服务 | 每个 Ktile 的发出、就绪、消费、释放，以及 tile 端点 | 检验 stage 响应等机制；已有候选接线供开发，服务参数待辨识 |

两种模式不混用：聚合模式不再单独计算 WGMMA，也不从整段 L 拆出内部时间线。

资格只取五种：**已核对**（与源码或程序语义核对过）、**已测**（有微基准测量，组合内未验证）、**待核对**（已在使用，尚未登记核对结果）、**假设**（候选，参数未辨识）、**经验替代**（暂用半经验的拟合价格）。输出中每个时间分量都标明来自哪一类模块。

| 模块 | 首版来源 | 资格 |
|---|---|---|
| 工作映射 | `v08_model.scheduled_work` 的持久调度移植（含 swizzle、raster、补齐与 OOB tile），grid 取 setup | 已核对：739011 的 12 个 cfg_b 条件、120 个成功 dual 进程，逐 CTA 工作坐标与 pingpong 消费者归属一致；2 次失败尝试保留在报告中 |
| 程序依赖与槽代次 | [CUTLASS 程序图](program/cutlass_program_graph.md) §2–3 的 pingpong 顺序：`math_wg_order_barrier.wait → mma → arrive → mma_tail → wait → store → store_tail → arrive`；`release(g) → acquire(g+S)` | 已核对：cfg_b 已有独立图导出，S6/S4 请求与槽规则对应冻结源码和构建；单CTA的B/C、D已有逐组软件端点，E已有真实三输出端点；观察扰动不支持迁移到plain，原全GEMM逐Ktile内部时刻未获对应验证 |
| 入口前缀 | `entry→producer_first_work` 使用冻结 V09 的 `events_ns.cfg_b.P0=1608 ns`，每 CTA、每调用一次；只连接生产者首路径，消费者许可保留原依赖 | 经验替代：[固定前缀接入](../../../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-prefix-001/README.md)；PROD→首 A 及消费者准备沿用原未单独定价假设 |
| 整段主循环（聚合模式） | V09 的 first/later 主循环 L（[supply_model](../../../../../microbench/gh200_resource_campaign/access_rules/supply_model.py)），按输出 tile 给出，已含计算下限 | 经验替代：只给 tile 端点，不提供请求就绪时刻 |
| TMA 输入服务（细事件模式） | [多 CTA 共享服务](hardware_rules/TMA-SHARED-SERVICE.md)的共享范围候选，以及[候选与反例](hardware_rules/TMA-GEMM-HYPOTHESES.md)中保留的“有限槽＋源共享＋独立目的投递”；[阶段4接口](hardware_rules/TMA-SHARED-SERVICE.md#stage-4-service)可独立接入异步源、目的投递和信用定时事件 | 假设：请求完成端点与路径约束已接通；旧证据已回放，[D开发路径](../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#d-measured)数值核对完成，服务价格尚未辨识 |
| L2与源地址状态 | `ProgramGemmSource`的两域缓存、128 B行、LRU、pending fill、远端返回及每GPC两槽服务；实际case行距、分配和CTA分组进入同一对象 | 假设：当前地址状态与服务候选仍有反例。[L2并发曲线](hardware_rules/L2-SERVICE.md)开发完成，六形式按预置单调反例停止，无冻结或请求级替换参数 |
| 外存读miss后端（可选） | [HBM服务](hardware_rules/HBM-SERVICE.md)的共享方向FIFO；`ProgramGemmSource.input → shared_stage_events.input_source_service`接收实际程序地址，新128 B miss替换原外存前沿；显式写回可共用同一对象 | 假设：调用方给定方向服务率，六个小程序核对替换；当前缓存没有dirty写回策略，默认GEMM参数保持 |
| WGMMA 计算服务 | [EXP-07](../../experiments/gh200_sm90/Q05-wgmma/EXP-07-wgmma.md)：单 warpgroup m64n64k16 SS 在 wait≥3 时 4095.9 FLOP/cycle；依赖链每轮约 105 cycle。cfg_b 每 Ktile 名义 2·128·128·64/4096 = 512 cycle | 已测：[A驻留短链](hardware_rules/COMPUTE-SERVICE.md)的条件循环量q512通过独立长度留出；A固定项不迁移，256提交提前量仍为假设。当前接线保留原 `compute_ns` 与提交／完成参数；issue、commit、wait的独立价格及与供给联合的GEMM持续服务尚未辨识 |
| 输入同步与槽释放 | [同步服务](hardware_rules/SYNC-SERVICE.md)核对full(1、32768 B)、empty(128)与槽代次；请求级接口可显式给定已就绪／尚未就绪两条返回价格，等待经计算与release影响复用 | 假设：计数与程序依赖已核对；旧46条件回放一致；B/C零额外控制具体形式已被开发响应反证，独立返回价格未校准，默认组合结果保持 |
| SMEM需求与输出写块 | [SMEM服务](hardware_rules/SMEM-SERVICE.md)：S4/S6实际STS.64的完整lane／word与两源槽；共同程序另列TMA投递、WGMMA引用和输出源读取的覆盖关系；调用方给定q/f替换`epilogue_write_ns` | 假设：地址已核对；同价复合写块及准备／写入可加性未由匹配短链定价，原m11失败保留 |
| 启动寄存器配额（可选） | [寄存器服务](hardware_rules/register_service.md)：CTA保留池的12warp配额账本，`register_quota=True`展开启动节点与对应首次工作依赖 | 假设：需求和软件依赖已核对；节点未单独定价，producer warp0释放已含前缀，其余warp释放及consumer申请时长待测 |
| 普通L1／LSU需求（可选） | [L1／LSU服务](hardware_rules/L1-LSU-SERVICE.md)：`ordinary_lsu=True`保存冻结alpha条件下的普通global／local需求及静态常量读取覆盖 | 已核对：源码与分支推导普通global动态请求为0，local为0；参数控制保留现有复合时间，六图原端点保持 |
| SM容量与活动（可选） | [驻留服务](hardware_rules/residency.md)：`sm_activity=True`核对S4/S6容量并输出已求解的计算服务、输出准备／写块和输入在途窗口；[联合服务](hardware_rules/joint_service.md)提供完整条件合同 | 已核对：两构建均1 CTA/SM；活动窗沿原假设，六图原端点保持。cfg_b无匹配联合合同；E已取得真实三输出trace端点，但未形成可迁移的相位价格，入口分布仍待测 |
| 输出服务 | V09 的 pingpong 输出：E 与 `epilogue_end` 的重叠服务，参数 E、r、w、we、Etail 取自冻结的 `events_ns.cfg_b`，递推见[程序图 §4](program/cutlass_program_graph.md#recurrence)。single/multi、real/OOB 规则只校准了 cfg_a、cfg_c，[output_model](../../../../../microbench/gh200_resource_campaign/access_rules/output_model.py) 拒绝将该规则用于非 cooperative 调度，不接入。细事件开发接线使用 `shared_stage_events` 的八个 D 子块、两源缓冲与共享服务候选 | 经验替代：聚合模式；细事件模式为假设 |
| 有效频率 | 聚合模式使用 V09 的输入相关时钟（[clock_model](../../../../../microbench/gh200_resource_campaign/access_rules/clock_model.py)），与包络联立求解；可替换为运行前预测器。细事件模式可使用[阶段频率开发模型](../../../../../results/gh200_resource_campaign/access_rules/20261015-phase-frequency-development-001/README.md)，输入为 case、静态工作列表和输出序号 | 经验替代 |
| 包络到 plain 时间 | 两种模式均可使用 V09 的 `T_plain = F + κ·W`，F、κ 由参照条件的 dual 包络与 plain event 拟合（[MODEL-CANDIDATES](../semi_empirical/MODEL-CANDIDATES.md#v09-model)） | 经验替代 |

模块级交付与有限实测按[微基准地图](hardware_rules/MICROBENCH-ROADMAP.md)记录；其中 §4 给出 cfg_b 实际计算／同步短链，§5 对应当前源服务、目的投递、计算和控制接口。新增结果按相同外部端点替换上表对应复合服务，模块资格仍按已有证据登记。

## 3. 连接方式

| 项 | 规定 |
|---|---|
| 事件推进 | 聚合模式按 CTA 与输出 j 推进，沿用程序图 §4 的 pingpong 递推（主循环交接 gm、h，输出间隔 w、we），主循环取整段 L。细事件模式按 CTA、输出 j、逻辑 Ktile g 展开：producer 在 `acquire(g)` 满足后发出 A/B 请求，就绪时刻由 TMA 输入服务给出；`consumer_wait` 在对应槽就绪后返回；WGMMA 的发出、commit、wait 由计算服务定价；`warpgroup_wait` 满足后 `consumer_release`，再允许 `acquire(g+S)`；两个数学 warpgroup 由 order barrier 交替；输出按输出服务推进 |
| 模块间状态 | TMA 服务的共享状态跨 CTA，共享范围由所选候选决定。聚合模式的频率与包络联立求解，输出重叠使用 V09 参数；细事件默认读取指定的 ns 服务价格；启用阶段频率时，每个输出的计算与提交价格由周期换算，其他服务维持原声明单位，输出由同一请求候选推进 |
| 时间单位 | 程序图不带时间。计算服务的微基准依据以 cycle 计；供给、输出服务按其版本的单位给出；频率模块把 cycle 换算为 ns。细事件默认参数使用 ns；阶段频率模式将每 Ktile 的计算周期与提交提前量分别除以该输出阶段的 GHz，再送入事件引擎；全局事件时间用 ns |
| 结束条件 | 每个 CTA 的最后一个输出完成 `store_tail`；包络取最早入口到最晚退出 |
| 观察端点 | 模型事件绑定 dual trace 的实际打点：CTA 入口、每个 tile 的主循环起止与输出起止、CTA 退出；不把聚合事件当成某一条指令。plain 时间只经由上表的时间映射得到 |

## 4. 输出

聚合模式给出逐 CTA、逐输出 tile 端点、包络 W、最晚 CTA 与并列集合、有效频率和 plain 完整时间。细事件开发接线给出指定服务价格下的请求、计算完成和槽释放时间线、消费者跨度与最晚 CTA；增加 `--phase-frequency` 后，同时输出各阶段预测 GHz、包络、dual event 与 plain 时间。每个分量附带其模块资格，经验替代部分单列。

九阶段的同版入口为[cfg_b_system.py](../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_system.py)：case／setup、相对分配、CTA分组、声明服务和调用协议经共同程序进入状态化源与请求引擎，输出逐CTA／Ktile时间线、配额／普通LSU／SM活动、阶段频率及plain。同版复现保留原零额外控制时间与输出复合服务；B/C中q512／lead256加零额外控制的具体形式已停止，未获得可替换价格。新短链观测没有改变原12条件预测或原阈值评分。旧`--candidate shared/unshared`入口继续复放它各自的历史参数，原12矩阵的同版复现使用此系统入口。

```bash
python -B microbench/gh200_resource_campaign/access_rules/cfg_b_system.py \
  --repo . --case v09d_cfg_b_G3_base --output /tmp/cfg-b-system
python -B microbench/gh200_resource_campaign/access_rules/cfg_b_system.py \
  --input /tmp/cfg-b-system/inputs.json --output /tmp/cfg-b-system-replay
```

[同版交付](../../../../../results/gh200_resource_campaign/access_rules/20261017-cfg-b-system-001/README.md)保存完整输入及55份源码。G3_base、S4、A16各11调用，152064个目标请求的原节点、完成与输出时刻精确保持；其余九项复用旧开发成绩，G6未重跑。67项CPU检查通过；原12条件误差仍为10.225%／25.422%，原门槛和独立验收状态保持，详见[组合实验](../../experiments/gh200_sm90/Q14-composition/COMPOSITION-01-replay.md)。

CPU 入口为 [cfg_b_composition.py](../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_composition.py)。在仓库根目录执行：

```bash
# 12 个冻结条件的全部 16 个字段，以及全部历史 dual 进程
python -B microbench/gh200_resource_campaign/access_rules/cfg_b_composition.py \
  --all-traces --output /tmp/cfg-b-composition-report.json

# 导出 G3 S4 的程序实例和 CTA 55 的操作依赖；S6 将 case 改为 v09d_cfg_b_G3_base
python -B microbench/gh200_resource_campaign/access_rules/cfg_b_composition.py \
  --case v09d_cfg_b_G3_stage4 --cta 55 --trace --output /tmp/cfg-b-G3-S4-program.json

# CPU 回归、槽生命周期、源码/SASS 对照和服务替换检查
PYTHONPATH=microbench/gh200_resource_campaign/access_rules \
  python -B -m unittest test_cfg_b_composition test_v09_ledger_integration -v
```

默认读取 `20261009-V09-heldout-job739011`，可用 `--archive` 指定路径。输出包含冻结/源码/SASS 绑定、当前实现哈希、Python/NumPy/SciPy 版本、模块版本与资格，以及逐条件预测。浮点比较按规范 JSON 精确匹配，不设误差容限；元组序列化为数组。`--trace` 只回放每条件的 `dual-00`，`--all-traces` 回放全部 dual 记录并保留失败尝试。观察输出含四个 tile 端点的 ns 残差，以及 V09 原有六个分项的相对误差。

2026-10-14 CPU 验收：12 个条件的 16 个字段精确复现；120 个成功进程共 14,800 个 CTA、58,000 个输出 tile 的坐标和消费者归属一致；2 次历史失败尝试单列。8 项测试通过，包含原有 29 个条件的时钟工作量回归。G3 S6/S4 仍同为 `50.79100947497063 µs`，保留原冻结模型对 stage 响应的零预测。

Python 接口为 `predict(program, calibration, services={...})`。替换项及端点如下：

| 替换项 | 默认实现 | 接口 |
|---|---|---|
| `mainloop` | `V09Mainloop` | `intervals(frequency_ghz)` → 全体 CTA 的逐 tile L（ns）与支持标记；已含计算下限 |
| `events` | `V09Events` | `parameters(cta)` 提供聚合前缀/交接/尾差；`output_end(params, start_ns, next_mainloop)` 提供输出结束 ns |
| `frequency` | `V09Frequency` | `solve(envelope_us)` → GHz、包络 µs 与闭合结果；工作量和零活动混合口径由此模块提供 |
| `observation` | `V09Observation` | `project(solved)` → dual/plain µs |

聚合递推从程序实例读取工作数量和顺序，独立执行 pingpong 递推；测试禁止调用旧完整预测器和旧 CTA 递推。所有默认时间模块均为经验替代。请求级接线独立求解整段执行，使用程序图的 TMA 完成、计算完成与槽释放端点；跨 CTA 共享状态由同一个服务实例管理。

请求级 CPU 入口为 [cfg_b_request_model.py](../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_request_model.py)，Python 接口为 `simulate(program, parameters, share_inflight_sources=..., detailed_ctas=...)`。输入服务参数统一使用 ns；控制节点未单独定价时取前驱最晚时刻，并在报告中列出。当前接入完整输入/输出 tile、全部 CTA 活跃、每输出至少两个 Ktile 的未干预 cfg_b 程序。

```bash
# 原样复用已失败的历史共享参数，导出普通 G3 程序的接线诊断；S4 改用 v09d_cfg_b_G3_stage4
python -B microbench/gh200_resource_campaign/access_rules/cfg_b_request_model.py \
  --case v09d_cfg_b_G3_base --candidate shared --cta 55 \
  --output /tmp/cfg-b-G3-S6-requests.json

PYTHONPATH=microbench/gh200_resource_campaign/access_rules \
  python -B -m unittest test_cfg_b_request_model -v
```

2026-10-14 请求级 CPU 检查：G3 S6/S4 各 132 个 CTA、50,688 个输入请求/投递、357,588 条依赖边通过检查；导出的 CTA 55 含 128 个槽代次与 384 个请求。5 项测试覆盖源身份、跨 CTA 共享、计算/释放依赖、服务替换及原反例。两套历史参数在原四个干预条件下的完成跨度仍低于全部实测进程，判定沿用[候选与反例](hardware_rules/TMA-GEMM-HYPOTHESES.md#2-假设)。本轮未新增测量或拟合。

阶段频率与完整时间入口：

```bash
python -B microbench/gh200_resource_campaign/access_rules/cfg_b_request_model.py \
  --case v09d_cfg_b_G3_base --candidate shared --cta 55 \
  --phase-frequency results/gh200_resource_campaign/access_rules/20261015-phase-frequency-development-001/fit-001/model.json \
  --output /tmp/cfg-b-G3-S6-phased-requests.json
```

Python 接口为 `predict(program, parameters, frequency_model, calibration, ...)`，`simulate` 与 `predict` 可传入 `producer_prefix_ns=1608.` 接入上述前缀，默认值0保留原预测。默认计算服务为 512 cycle/Ktile，提交提前量为 256 cycle；后者保留为候选参数，可通过 `--commit-lead-cycles` 替换。阶段频率使用 `FIRST_MMA→MAIN_END` 窗口的经验预测。计算与提交以外的控制、输入和输出价格仍由服务参数给出。V09 的 F、κ 与 dual event 额外时间均从冻结校准读取。

2026-10-15 接线检查：19 项相关测试通过；176 行阶段频率推理逐位复现冻结开发实现，恒定阶段价格逐事件复现原标量引擎；改变两个阶段的频率时，请求数量和依赖不变，计算时长与时间映射按预期改变。原失败共享参数接入阶段频率后，G3 S6/S4 的开发 plain 预测为 46.248/51.349 µs；这些接线输出尚未经过独立组合验收。

输入服务可通过 `--tensor-source 参数JSON --tensor-candidate banked_resident_fifo` 替换，或向 `simulate` 传入 `tensor_source`。探针和 GEMM 共用 `TensorSourceService`；后者使用程序账本的源地址和槽释放，不接入探针控制价格。重构前后第二轮 8 条冻结预测的完整时间线一致，新增测试核对两个前端的逐请求服务与 GEMM 依赖。首次迁移的 stage 零响应及下一步见[迁移记录](../../../../../results/gh200_resource_campaign/access_rules/20261014-tma-tensor-development/gemm-transfer-001/README.md)。

[整组组合回归](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-combination-regression-001/README.md)保留原12个cfg_b条件及规格/V09对照。当前状态化源适配器覆盖其中G3 S6/S4两项，101,376条请求及715,176条依赖完成绑定；既有参数的端点逐位复现。对746679的40进程重放，plain误差为−5.31%/−12.12%，stage变慢幅度预测4.04%、实测12.10%；第二MAIN跨度仅30.45/46.01ns，实测1184/2832ns。下一主要阻塞是共享源完成、独立目的投递与槽复用产生的CTA差异；覆盖缺口和模块资格在报告中单列。按[当前阶段限制](../../GOALS.md#current-scope)只推进这一组合阻塞，其他支线暂缓。

[局部生产者延迟筛查](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-local-producer-candidate-001/README.md)已在同一矩阵检查40/80cycle的计算活跃期发出延迟；S4平均L有所改善，空间跨度仍低于全部实测，两组参数不接入，停止这一形式的扩扫。

[既有搬运空间分布复核](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-sharing-spatial-audit-001/README.md)重放746797的80个进程：独立源Q4的同GPC持续时间跨度为3792ns，旧同质候选为352.68ns；共享源Q4实测约32ns。该数据要求后续服务同时解释两种分布。K起始相位干预的32次CPU筛查未区分共享候选，不提交该测量；本轮新增GPU判别为0。

[状态化服务接入回放](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-stateful-probe-replay-001/README.md)已完成8条件、88次CPU调用。目标调用在模型内全部为本地行命中，数据状态新增等待为0，预测退化为上述旧候选，独立源跨度反例仍在。下一项限定为驻留数据的竞争与完成规则，先检查独立源与共享源的共同约束，再回到G3 S6/S4；不扩展暖机或频率研究。

[等待分解与固定轮转筛查](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-resident-arbitration-001/README.md)确认独立源Q4的快慢CTA差主要落在等待：总差中位7553cycle，扣除等待后每对仅−13至27cycle。保留原两通道和价格的每CTA32KiB固定轮转候选仍低估跨度，且共享条件时间误差达38%–63%；该形式不接入，不扩扫额度或初始相位。

[固定源集合的共享范围判别](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-source-scope-discrimination-001/RESULTS.md)747595完成8条件×7进程，GPC4/Q1实测+24.82%，超出全部冻结候选；Q4零响应控制也未通过，按原判据记信息不足。相同源多重集合的重排使Q1/Q4空间跨度由32ns变为2624/1568ns，新增周期主要在等待。下一服务须联合解释该响应与既有跨度，再回到G3；已用1/2轮GPU判别，暂不提交第2轮。

[单价联合检查](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-service-price-check-001/README.md)固定服务形式，仅检查原价至两倍的五个价格点；均未修复空间跨度，提高25%已使原六条件最大误差由3.64%升至5.84%，停止单价扩扫。[共同程序地址适配器](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-program-source-adapter-001/README.md)已实现按共同程序读取几何，12条件的选定地址检查通过，G3 S6/S4完整时间线逐字段复现。下一步补齐原12条件的状态化运行接线与整组回放，硬件规则的失败记录保留。

[原12条件状态化回放](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-full-matrix-001/README.md)已接通共同程序、分配、放置和频率输入，原12条件全部完成。前11项绑定3,341,552条依赖，G6保存组级事件、未执行逐atom绑定；[最终比较](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-full-matrix-001/comparison-final.md)保留规格/V09对照和六项残差。组合plain绝对误差中位17.378%、最大39.797%，G6为+1.880%；本版尚无独立组合C2/C3通过资格。A16及B80/B112的行距响应未被当前源服务预测，短条件另有entry→FIRST缺口。G3两种stage预测逐位复现旧版，本次统一对照739011原观测；下一服务须联合满足行距、源重排和空间跨度约束。

[地址无关回传信用筛查](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-return-credit-candidate-001/README.md)因保留零行距响应而停止矩阵扫描。[异步完成接线](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-async-source-interface-001/README.md)已提供`input_source_async`与`input_destination_service`：等待信用的请求先排队，目的实际完成后按`demand_id`回调源并归还信用。6项合成反例测试通过，20组同步完整结果逐字段复现原快照；后续地址服务候选使用该接口。

[32 B 服务与 128 B 信用候选](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-dual-granule-service-001/RESULTS.md)的三个固定容量均未通过旧必过条件：2048/4096/8192 的等待窗口误差为−17.78%/−6.34%/−6.34%，原门槛为5%。共运行3项，其余51项停止；本候选不接入组合，未新增GPU判别。原12条件已使用既有同版服务完成回放，失败与剩余缺口保留在整组报告。

[按请求大小分道的两个固定版本](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-size-lanes-service-001/RESULTS.md)均未通过首项旧必过约束，本轮停止，未替换组合。

[固定经验前缀轮次](../../../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-prefix-001/RESULTS.md)已完成：五项各11次CPU调用逐位保留内部结果，790080条新图依赖通过；其余七项由保存时间线投影，G6未重跑。原12条件plain绝对误差中位/最大由17.378%/39.797%变为10.225%/25.422%。stage响应仍只有3.903%（实测10.467%），A16与B80/B112仍失配，最晚CTA命中仍为1/120；本轮无新增独立C2/C3成绩，按登记上限收口。

## 5. 完成判据

1. **可运行：** 聚合模式对 cfg_b 冻结预测逐位复现 V09，作为后续每次替换模块的回归基线；对 V09 的 cfg_b 开发条件（含 G3 的 stage6 与 stage4）生成 tile 端点，并与 dual trace 逐 tile 对照；细事件模式另生成 Ktile 时间线。这一步是开发诊断，不计成绩。
2. **结构检查（细事件模式）：** stage6→4 的响应由槽释放与供给规则产生，不加 stage 专属修正；同时核对[候选与反例](hardware_rules/TMA-GEMM-HYPOTHESES.md)表中已排除的形式没有被重新引入。
3. **冻结与留出：** 冻结后按 GOALS §3.4 分别判定：组合机制主张按 C2，完整时间按 C3（阈值在冻结前写定），给定候选集合时按 C1。经验替代模块不进入 C2 结论。
4. **结论与退出：** 结论按 GOALS 的三种记录。冻结预测与精度足够的观测明确冲突时，组合主张记为被反证；只排除了这组参数、没有排除结构时，记该参数版本未通过并保留结构候选；只有观测无法区分或参数不可辨识时记为信息不足。后两种情况按端点分流：工作列表或最晚 CTA 不对 → [程序图](program/cutlass_program_graph.md)；`prefix`、`prefill`、`first_L`、`later_L` 的残差 → 供给（[多 CTA 共享服务](hardware_rules/TMA-SHARED-SERVICE.md)）；cycle 与 ns 的变化不一致 → [频率](hardware_rules/FREQUENCY-MECHANISM.md)；`last_E`、`tail` → 输出服务；包络与 plain 的差 → 时间映射。分流只用于选择下一项判别实验，归因要由实验确定。

## 6. 依赖

| 依赖 | 状态 |
|---|---|
| 程序图静态导出与整图核对 | cfg_b 的 CPU 导出和历史工作映射已交付；细化范围与剩余观察缺口见[程序图 §5](program/cutlass_program_graph.md#5-给机制组合的交付与缺口) |
| TMA 共享服务候选的参数辨识 | tensor 原生正确性、Q2 校准与首批机制判别完成；首批与源分区候选未通过，工作集状态干预获得支持，量化服务规则开发中；细事件模式须提供 cfg_b 实际 tensor 请求的服务：每个 Ktile 为 A 一次 128×64（16 KiB）、B 沿 N 分两次 64×64（各 8 KiB），TMA SMEM swizzle 为 128B，见[程序请求](program/cutlass_program_graph.md#2-对象操作与依赖)；完整编码字段与实际 launch params 的绑定见[746953 描述符查询](../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-descriptor-query-001/RESULTS.md) |
| 计算完成与输入槽释放的服务（细事件模式） | [746824计算量干预](../../../../../results/gh200_resource_campaign/access_rules/20261015-g3-compute-credit-discrimination-003/RESULTS.md)支持有限槽复用对停顿的影响；首版保留复合计算服务，issue、commit、wait的拆分随证据再细化 |
| 运行前频率预测器 | RBF 通过首次 8 条频率留出后，完整时间留出仅 4/8 通过；按工作分布修正的饱和响应式在下一轮 8 条 C3 全部通过，最大时间误差 2.96%；`cfg_b_frequency.StaticFrequency` 已可替换聚合频率模块，版本与支持域见[频率结论](hardware_rules/FREQUENCY-MECHANISM.md#4-结论) |

依赖未完成时，可先以经验替代模块接通流程，检查程序图与事件推进的正确性；此时的输出只作开发诊断。

没有 NCU 时，程序操作继续用源码、SASS 和已有 trace 核对；硬件服务按对应路径页的竞争假设设计干预，分别测请求就绪、计算完成/槽释放、频率和输出端点。首版计算服务保留复合节点，获得区分证据后再拆 issue、commit、wait；未能区分的机制按 GOALS 记录为信息不足。
