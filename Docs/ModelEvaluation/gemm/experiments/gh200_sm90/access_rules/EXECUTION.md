# GH200 分工与当前状态

问题见 [PLAN](PLAN.md)，当前目标见 [GOALS §5.4](../../../GOALS.md#current-scope)，顺序与依赖见[项目 PLAN](../../../PLAN.md#1-近期顺序)；测量与评分规则只在 [RESEARCH-GUIDE](RESEARCH-GUIDE.md) 维护。各轮登记、结果与结论在对应实验记录中，入口为[按问题索引](../README.md#research-questions)；原始轮次见[结果目录索引](../../../../../../results/gh200_resource_campaign/access_rules/README.md)。

## 分工

| 写入者 | 归属 | 相关的现行任务 |
|---|---|---|
| 主集成 | PLAN/RULES/EXECUTION、公共 probe/runner、v09_model、集成与冻结评分；公共文件串行维护 | 项目与执行 PLAN、冻结与评分 |
| A：时间输入 | R09、clock_model、r09 专属分析与测试；使用 B 定义的源工作口径 | [有效频率](../../../model/mechanism/hardware_rules/FREQUENCY-MECHANISM.md) |
| B：供给请求 | R10/R13、R14 输入生命周期、supply_model；接入 C 提出的跨 rank 前驱 | [多 CTA TMA](../../../model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md)、[S16](../../../model/mechanism/hardware_rules/TMA-S16-STAGE-COST.md)、[GEMM 候选与反例](../../../model/mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md)；`tma_input_events.py`、`analyze_tma_s16_stage_cost.py` |
| C：输出边界 | R15/R18/R19、output_model；m11 专属 probe/模型/runner 与 R03/EXP-09 | m11 已停止，见[重启条件](../../../model/mechanism/hardware_rules/M11-SMEM.md#4-结论与重启条件) |
| C1-01 | `results/.../20261013-C1-design`、`-C1-models`、`-C1-dev`、`-C1-baselines` | [C1-01](../Q16-config-selection/C1-01-selection.md) |

用户已授权主集成与上述专业组之间的有界协调。具体对话身份保留在已有授权与执行记录中。

<a id="research-chat"></a>

2026-10-16 对话分工：主集成在本对话维护 cfg_b 程序、组合、针对已知反例的有限 CPU 建模、模块接入与冻结评分；[GH200 可选机制与待批实验](thread://01a14190-8d4a-7e03-8c58-f1929dc6d19a)承接可选逆向、证据缺口评估和待批执行方案。A/B/C 继续表示文件责任分工，执行许可统一见 [GOALS §5.4](../../../GOALS.md#current-scope)。

<a id="status"></a>

## 当前状态

更新：2026-10-28。当前重点、五阶段推进依赖和后续候选的进入条件见[项目 PLAN](../../../PLAN.md#1-近期顺序)；各轮登记、结果与结论见[按问题索引](../README.md#research-questions)；执行许可与暂停范围以 [GOALS §5.4](../../../GOALS.md#current-scope) 为准。a044 预约已于 2026-10-27 22:10 结束，见 [a044 阶段说明](../Q10-time-mapping/A044-REMEASUREMENT.md)。

暂停或已停止的工作流，重启条件写在各自页面：S16 与 C1-01 见 GOALS §5.4，m11 见[重启条件](../../../model/mechanism/hardware_rules/M11-SMEM.md#4-结论与重启条件)，长 K 见[主线](../../../model/semi_empirical/MAINLINE-LONG-K.md)，V10 与跨卡见[验证顺序](RESEARCH-GUIDE.md#order)。

旧 worktree、分支、准备包和结果目录暂时保留，本次没有删除。原清理审计留在[结果记录](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261012-repository-consolidation/)，需要清理时按当时 Git 与文件状态决定，不维护过期清单。

<a id="counter-access"></a>

计数器：免费 ROMEO 节点上的 NCU 计数最近一次复查仍限管理员。[已有诊断](../../../../../../microbench/gh200_l0/results/20260930-counter-access/README.md)记录权限受限，2026-10-20 零 GPU 作业 764001 复查同节点仍为 `RmProfilingAdminOnly=1`（[查询记录](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261020-Q02-source-base-replay-001/ncu-access.txt)）。a044 预约期间为 `RmProfilingAdminOnly:0`，10-26、10-27 的 NCU 补测、利用率、L1 与 TMA 来源拆分都在该预约内完成，见 [10-26 补测](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-pending-20261026-report.md)和 [a044 阶段说明](../Q10-time-mapping/A044-REMEASUREMENT.md)。预约结束后再使用计数器，须先复查所在节点的权限。两项授权测量的作业与身份见[交付记录](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261012-authorized-measurement-integration/authorization.json)。
