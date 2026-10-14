# GH200 分工与当前状态

问题见 [PLAN](PLAN.md)，顺序与依赖见[项目 PLAN](../../../PLAN.md)；测量、数据和验证规则只在 [RESEARCH-GUIDE](RESEARCH-GUIDE.md)维护；任务详情见[主线](../../../model/semi_empirical/MAINLINE-LONG-K.md)、[m11](../../../model/mechanism/hardware_rules/M11-SMEM.md)及下表各任务页。

## 分工

| 写入者 | 归属 | 相关的现行任务 |
|---|---|---|
| 主集成 | PLAN/RULES/EXECUTION、公共 probe/runner、v09_model、集成与冻结评分；公共文件串行维护 | 项目与执行 PLAN、冻结与评分 |
| A：时间输入 | R09、clock_model、r09 专属分析与测试；使用 B 定义的源工作口径 | [有效频率](../../../model/mechanism/hardware_rules/FREQUENCY-MECHANISM.md) |
| B：供给请求 | R10/R13、R14 输入生命周期、supply_model；接入 C 提出的跨 rank 前驱 | [多 CTA TMA](../../../model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md)、[S16](../../../model/mechanism/hardware_rules/TMA-S16-STAGE-COST.md)、[GEMM 候选与反例](../../../model/mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md)；`tma_input_events.py`、`analyze_tma_s16_stage_cost.py` |
| C：输出边界 | R15/R18/R19、output_model；m11 专属 probe/模型/runner 与 R03/EXP-09 | m11 已停止，见[重启条件](../../../model/mechanism/hardware_rules/M11-SMEM.md#4-结论与重启条件) |
| C1-01 | `results/.../20261013-C1-design`、`-C1-models`、`-C1-dev`、`-C1-baselines` | [C1-01](C1-01-selection.md) |

用户已授权主集成与上述专业组之间的有界协调。具体对话身份保留在已有授权与执行记录中。

<a id="status"></a>

## 当前状态

更新：2026-10-13。所有提交保留本地 main，不推送、不登录 GitHub。

| 工作 | 当前结论 | 下一步 |
|---|---|---|
| S16 探针控制成本 | 历史离线诊断完成；token 控制（H1）与组状态阻塞（H2）对现有窗口预测相同 | 准备同一二进制、活动 S2/S4 的 A/B，冻结预测后做一次小规模判别 |
| 多 CTA TMA 共享服务 | 任务设计完成，未新增测量 | 复用多 CTA 记录，以并发 CTA 数×每 CTA 请求并发为主轴 |
| 有效频率 | 任务设计完成，未新增测量；C1-01 开发数据尚未采集 | 先用 V09/R09 与 C1-01 冒烟数据做离线开发 |
| C1-01 | 构建 743011、冒烟 743016、遥测对照 743051 完成；开发作业 743973–743976 按用户要求在运行前取消 | 先定短 kernel 样本数与作业间漂移的处理，再重新提交；[恢复条件](C1-01-selection.md#恢复前要定的事) |
| m11 | 741153 完成 48 条件×3 进程；旧需求与新的最忙 bank 标量均有反例，首个新静态汇总形式得到离线反例。随后 v1（741915）、v2（742266）两次冻结留出均未通过（v2 中位 0.44%、最大 11.60%） | 不新增单 warp 任务；剩余留出预算不转为开发；[重启条件](../../../model/mechanism/hardware_rules/M11-SMEM.md#4-结论与重启条件)、[m11推导](R03-access-demand.md#m11-bank-vector) |
| 长 K 主线 | 校准 741159 在旧参照门槛处停止；120 个参照采样完成，540 个开发槽位未执行；新提交 1/1，无恢复。失败修正未接入公共模型。cfg_b G6 的频率迁移转入有效频率任务 | 待定参数和最少对照明确后，再决定是否批准新校准；频率预测器可用后先判断局部 L 残差是否需要测量；[主线](../../../model/semi_empirical/MAINLINE-LONG-K.md) |
| 供给与其他机制 | 供给部分转为上面两项 TMA 任务；寄存器、联合服务、驻留等其他机制暂停 | 不自动恢复；达到[项目 PLAN](../../../PLAN.md#3-后续任务的进入条件)的进入条件再立项 |
| V10 与跨卡 | 未冻结、未采样，联合验证未完成 | 有候选后才进入最终 SHA 确认；[验证顺序](RESEARCH-GUIDE.md#order) |

旧 worktree、分支、准备包和结果目录暂时保留，本次没有删除。原清理审计留在[结果记录](../../../../../../results/gh200_resource_campaign/access_rules/20261012-repository-consolidation/)，需要清理时按当时 Git 与文件状态决定，不维护过期清单。

<a id="counter-access"></a>

计数器：[已有诊断](../../../../../../microbench/gh200_l0/results/20260930-counter-access/README.md)记录 NCU 权限受限，仍待用户联系 ROMEO 管理员，本次未尝试获取权限。两项测量的作业与身份见[交付记录](../../../../../../results/gh200_resource_campaign/access_rules/20261012-authorized-measurement-integration/authorization.json)。
