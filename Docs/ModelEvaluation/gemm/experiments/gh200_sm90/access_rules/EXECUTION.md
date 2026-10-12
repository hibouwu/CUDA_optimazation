# GH200 分工与当前状态

问题见 [PLAN](PLAN.md)，测量、数据和验证规则只在 [RESEARCH-GUIDE](RESEARCH-GUIDE.md)维护；任务详情见[主线](MAINLINE-LONG-K.md)与[m11](M11-SMEM.md)。

## 分工

| 写入者 | 归属 |
|---|---|
| 主集成 | PLAN/RULES/EXECUTION、公共 probe/runner、v09_model、集成与冻结评分；公共文件串行维护 |
| A：时间输入 | R09、clock_model、r09 专属分析与测试；使用 B 定义的源工作口径 |
| B：供给请求 | R10/R13、R14 输入生命周期、supply_model；接入 C 提出的跨 rank 前驱 |
| C：输出边界 | R15/R18/R19、output_model；m11 专属 probe/模型/runner 与 R03/EXP-09 |

用户已授权主集成与上述专业组之间的有界协调。具体对话身份保留在已有授权与执行记录中。

<a id="status"></a>

## 当前状态

更新：2026-10-12。所有提交保留本地 main，不推送、不登录 GitHub。文档整理已结束；m11 首个新静态汇总形式得到离线反例，未拟合，GPU 0。

| 工作 | 当前结论 | 下一步 |
|---|---|---|
| 长 K 主线 | 校准 741159 在旧参照门槛处停止；120 个参照采样完成，540 个开发槽位未执行；新提交 1/1，无恢复。失败修正未接入公共模型 | 待定参数和最少对照明确后，再决定是否批准新校准；[主线](MAINLINE-LONG-K.md) |
| m11 | 741153 完成 48 条件×3 进程；旧需求与新的最忙 bank 标量均有反例，无候选、未冻结或留出；新提交仍为1/2，无恢复 | bank分布／广播描述须先满足已有等价约束；剩余留出预算不转为开发；[m11推导](R03-access-demand.md#m11-bank-vector) |
| 供给与其他机制 | 暂停 | 不自动恢复 |
| V10 与跨卡 | 未冻结、未采样，联合验证未完成 | 有候选后才进入最终 SHA 确认；[验证顺序](RESEARCH-GUIDE.md#order) |

旧 worktree、分支、准备包和结果目录暂时保留，本次没有删除。原清理审计留在[结果记录](../../../../../../results/gh200_resource_campaign/access_rules/20261012-repository-consolidation/)，需要清理时按当时 Git 与文件状态决定，不维护过期清单。

<a id="counter-access"></a>

计数器：[已有诊断](../../../../../../microbench/gh200_l0/results/20260930-counter-access/README.md)记录 NCU 权限受限，仍待用户联系 ROMEO 管理员，本次未尝试获取权限。两项测量的作业与身份见[交付记录](../../../../../../results/gh200_resource_campaign/access_rules/20261012-authorized-measurement-integration/authorization.json)。
