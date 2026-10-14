# 机制组合：首版

更新：2026-10-13。**未实现。** 本页定义第一版组合的范围、模块来源与资格、连接方式、输出和完成判据；目标与验收见 [GOALS §3.4](../../GOALS.md#34-组合)，通用的图组织与层级见[接口 §7](../common/interfaces.md#7-程序图的组织与组合)，输入字段见[共同输入](../common/inputs.md)。

## 1. 范围

cfg_b：128×128×64 pingpong、cluster 1×1、6 stage；同一组合另跑 4 stage 构建，用来检查 stage 响应。FP16 输入、FP32 累加与输出，α=1、β=0，持久调度，原参考卡 GPU-43269fbc，plain 协议。先只做 cfg_b，原因是它没有 cluster 跨 rank 依赖，且已有 V09 的 stage6→4 反例（实测慢 10.47%，冻结候选响应为零）。

## 2. 模块与资格

### 两种开发模式

| 模式 | 主循环来源 | 能输出的事件 | 用途 |
|---|---|---|---|
| 聚合模式 | V09 的整段主循环 L = b + K_t·max(q/f, θᵀx)（`supply_predict`），已含计算下限，整段替代下表的 TMA 与 WGMMA 两个模块 | 每个输出 tile 的主循环与输出端点 | 接通程序图、调度、输出与时间映射，检查端点对照；不检验 stage 响应的机制 |
| 细事件模式 | 具备请求完成语义的 TMA 服务（每个请求的就绪时刻）与按 Ktile 的计算服务 | 每个 Ktile 的发出、就绪、消费、释放，以及 tile 端点 | 检验 stage 响应等机制；TMA 服务到位前不启用 |

两种模式不混用：聚合模式不再单独计算 WGMMA，也不从整段 L 拆出内部时间线。

资格只取五种：**已核对**（与源码或程序语义核对过）、**已测**（有微基准测量，组合内未验证）、**待核对**（已在使用，尚未登记核对结果）、**假设**（候选，参数未辨识）、**经验替代**（暂用半经验的拟合价格）。输出中每个时间分量都标明来自哪一类模块。

| 模块 | 首版来源 | 资格 |
|---|---|---|
| 工作映射 | `v08_model.scheduled_work` 的持久调度移植（含 swizzle、raster、补齐与 OOB tile），grid 取 setup | 待核对：已用于 V08/V09 冻结预测；R18/R19 的 trace 逐 CTA 记录实际 tile 坐标，首版须对 cfg_b 条件逐项比对 |
| 程序依赖与槽代次 | [CUTLASS 程序图](program/cutlass_program_graph.md) §2–3 的 pingpong 顺序：`math_wg_order_barrier.wait → mma → arrive → mma_tail → wait → store → store_tail → arrive`；`release(g) → acquire(g+S)` | 已核对：源码与现有递推的对应；整图导出与 SASS/trace 验收未完成 |
| 整段主循环（聚合模式） | V09 的 first/later 主循环 L（[supply_model](../../../../../microbench/gh200_resource_campaign/access_rules/supply_model.py)），按输出 tile 给出，已含计算下限 | 经验替代：只给 tile 端点，不提供请求就绪时刻 |
| TMA 输入服务（细事件模式） | [多 CTA 共享服务](hardware_rules/TMA-SHARED-SERVICE.md)的共享范围候选，以及[候选与反例](hardware_rules/TMA-GEMM-HYPOTHESES.md)中保留的“有限槽＋源共享＋独立目的投递” | 假设：尚无具备请求完成语义、参数已辨识的服务 |
| WGMMA 计算服务 | [EXP-07](../../experiments/gh200_sm90/EXP-07-wgmma.md)：单 warpgroup m64n64k16 SS 在 wait≥3 时 4095.9 FLOP/cycle；依赖链每轮约 105 cycle。cfg_b 每 Ktile 名义 2·128·128·64/4096 = 512 cycle | 已测：微基准吞吐，只用于细事件模式；这些吞吐与复合窗口不足以分别定价 issue、commit、wait，三者的拆分仍待辨识；与供给联合时的持续服务在 GEMM 内未验证 |
| 输出服务 | V09 的 pingpong 输出：E 与 `epilogue_end` 的重叠服务，参数 E、r、w、we、Etail 取自冻结的 `events_ns.cfg_b`，递推见[程序图 §4](program/cutlass_program_graph.md#recurrence)。single/multi、real/OOB 规则只校准了 cfg_a、cfg_c，[output_model](../../../../../microbench/gh200_resource_campaign/access_rules/output_model.py) 拒绝将该规则用于非 cooperative 调度，不接入 | 经验替代 |
| 有效频率 | V09 的输入相关时钟（[clock_model](../../../../../microbench/gh200_resource_campaign/access_rules/clock_model.py)），与包络联立求解；[有效频率任务](hardware_rules/FREQUENCY-MECHANISM.md)交付后替换 | 经验替代 |
| 包络到 plain 时间 | V09 的 `T_plain = F + κ·W`，F、κ 由同调用 dual 包络对 plain event 拟合（[MODEL-CANDIDATES](../semi_empirical/MODEL-CANDIDATES.md#v09-model)） | 经验替代 |

## 3. 连接方式

| 项 | 规定 |
|---|---|
| 事件推进 | 聚合模式按 CTA 与输出 j 推进，沿用程序图 §4 的 pingpong 递推（主循环交接 gm、h，输出间隔 w、we），主循环取整段 L。细事件模式按 CTA、输出 j、逻辑 Ktile g 展开：producer 在 `acquire(g)` 满足后发出 A/B 请求，就绪时刻由 TMA 输入服务给出；`consumer_wait` 在对应槽就绪后返回；WGMMA 的发出、commit、wait 由计算服务定价；`warpgroup_wait` 满足后 `consumer_release`，再允许 `acquire(g+S)`；两个数学 warpgroup 由 order barrier 交替；输出按输出服务推进 |
| 模块间状态 | TMA 服务的共享状态跨 CTA，共享范围由所选候选决定；频率由整卡活动决定，与包络联立求解，口径同 V09；输出与后续主循环的重叠暂按 V09 的 pingpong 输出参数 |
| 时间单位 | 程序图不带时间。计算服务以 cycle 计；供给、输出服务按其版本的单位给出；频率模块把 cycle 换算为 ns；全局事件时间用 ns |
| 结束条件 | 每个 CTA 的最后一个输出完成 `store_tail`；包络取最早入口到最晚退出 |
| 观察端点 | 模型事件绑定 dual trace 的实际打点：CTA 入口、每个 tile 的主循环起止与输出起止、CTA 退出；不把聚合事件当成某一条指令。plain 时间只经由上表的时间映射得到 |

## 4. 输出

聚合模式给出每个 CTA、每个输出 tile 的端点；细事件模式另给每个 Ktile 的事件时刻（带槽号与代次）。两种模式都给出包络 W、最晚 CTA 与并列集合；有效频率；plain 完整时间。每个分量附带其模块资格，经验替代部分单列。

## 5. 完成判据

1. **可运行：** 聚合模式对 V09 的 cfg_b 开发条件（含 G3 的 stage6 与 stage4）生成 tile 端点，并与 dual trace 逐 tile 对照；细事件模式另生成 Ktile 时间线。这一步是开发诊断，不计成绩。
2. **结构检查（细事件模式）：** stage6→4 的响应由槽释放与供给规则产生，不加 stage 专属修正；同时核对[候选与反例](hardware_rules/TMA-GEMM-HYPOTHESES.md)表中已排除的形式没有被重新引入。
3. **冻结与留出：** 冻结后按 GOALS §3.4 分别判定：组合机制主张按 C2，完整时间按 C3（阈值在冻结前写定），给定候选集合时按 C1。经验替代模块不进入 C2 结论。
4. **结论与退出：** 结论按 GOALS 的三种记录。冻结预测与精度足够的观测明确冲突时，组合主张记为被反证；只排除了这组参数、没有排除结构时，记该参数版本未通过并保留结构候选；只有观测无法区分或参数不可辨识时记为信息不足。后两种情况回到[多 CTA 共享服务](hardware_rules/TMA-SHARED-SERVICE.md)补充辨识。

## 6. 依赖

| 依赖 | 状态 |
|---|---|
| 程序图静态导出与整图核对 | [程序图 §5](program/cutlass_program_graph.md#5-给机制组合的交付与缺口)的下一步，未开始 |
| TMA 共享服务候选的参数辨识 | 任务设计完成，未新增测量 |
| 运行前频率预测器 | 任务设计完成，未新增测量 |

依赖未完成时，可先以经验替代模块接通流程，检查程序图与事件推进的正确性；此时的输出只作开发诊断。
