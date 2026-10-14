# GH200 实验执行计划

更新：2026-10-13。

## 范围

本页维护 GH200 上各任务已知的结论、开放问题和测量安排；顺序与依赖见[项目 PLAN](../../../PLAN.md)，验收与停止规则见 [GOALS](../../../GOALS.md)，分工与作业状态见 [EXECUTION](EXECUTION.md#status)。

半经验线（V09）的目标不变：在运行前，由程序、静态执行条件、参考卡冻结参数和声明的暖机协议，预测 cfg_a/b/c 的完整时间并解释变化。目标、暖机或前序调用的实测值只用于诊断。支持范围按校准覆盖预先声明，非 128 B 对齐行距、非默认 stage 和原拒绝项仍不支持；具体边界见[细则](RESEARCH-GUIDE.md#prediction-boundary)。

## 现在知道什么

| 对象 | 结论与证据 |
|---|---|
| V09 | 整体未通过：29 个可评分条件的绝对误差中位数 4.11%、最大 35.46%，另 1 个拒绝；分项和关键 CTA 也有错误。已转为开发数据，冻结成绩不改。[V09](V09-component-validation.md) |
| 长 K | 时钟扩展和局部修正均未得到候选。必要校准只完成参照，因旧 event-extra 门槛终止，540 个开发槽位未执行；新参数能否独立定价仍未知。cfg_b G6 代入实测频率后误差由 +35.46% 降到 +1.71%，频率部分转由有效频率任务负责。[R09](R09-inkernel-clock-stages.md)、[长 K](../../../model/semi_empirical/MAINLINE-LONG-K.md) |
| 有效频率 | V09 实测 1.07–1.85 GHz，随配置与输入变化：同形状 random 输入使 cfg_a/b 比 dyadic 低 16–19%，cfg_c 低 7%。尚无运行前预测器。[频率](../../../model/mechanism/hardware_rules/FREQUENCY-MECHANISM.md) |
| TMA 单 CTA | S16 中 stage4 每组约 54.8 cycle 的额外成本，token 控制（H1）与组状态阻塞（H2）对现有窗口给出相同预测，判别实验未执行。[S16](../../../model/mechanism/hardware_rules/TMA-S16-STAGE-COST.md) |
| 计时协议 | 10-13 在原参考卡上，冒烟作业的参照 plain 比 V09 冻结值高 0.4%–6.6%，40 分钟后的作业只差 −0.2%–+1.6%；30–100 µs 参照的进程间极差 2%–4%，8 µs kernel 达 11%–15%；遥测轮询无一致影响。[C1-01](C1-01-selection.md#恢复前要定的事) |
| 普通 SMEM | 开发阶段：stride 冲突关系在开发条件上吻合；旧需求和新最忙 bank 标量都不能覆盖宽广播／排列，仍缺广播与依赖描述。之后两次冻结留出均未通过：v1（741915）中位 4.55%、最大 34.82%；v2（742266）47 个受支持条件中位 0.44%、最大 11.60%，唯一超标为 8 B、1 warp 独立读的 random 布局，8 warp 独立读写全部 ≤1.83%。[R03](R03-access-demand.md#m11-current)、[EXP-09](../EXP-09-smem.md)、[v2 结果](R03-access-demand.md#m11-holdout-v2-result) |

<a id="mainline"></a><a id="smem-contract"></a>

## 开放问题

1. **长 K 的后续 L：** 能否用不依赖 G6 的新形状，区分局部成本不足与 CTA／包络组合偏差？必要开发数据尚未采到。频率替换后 cfg_b G6 仍有 +1.71%，cfg_a G6 的代表 CTA 早于实测最晚 CTA 结束；是否测量待运行前频率预测器可用后再定。[主线任务](../../../model/semi_empirical/MAINLINE-LONG-K.md)
2. **SMEM 的单 warp 时间线：** v2 需求规则与所测 8 warp 独立读写子集相容，失败集中在 1 warp 独立读的 `P1(D)`（低 D 偏长、高 D 偏短）。冻结主张已判定未通过；当前微架构首切片是 TMA→本地 SMEM，单 warp 线暂不新立 C2 主张，重新立项时以 v1/v2 留出为开发数据、另取新留出。[v2 结果](R03-access-demand.md#m11-holdout-v2-result)
3. **完整时间转换：** event−包络增量 0.688–0.848 µs 影响哪些预测用途？原因尚未识别；局部 L、SMEM 和完整时间的有效性分别检查。10-13 冒烟的增量为 0.85–3.2 µs，同日另一作业又与 V09 相近；C3 评分需要在同一作业内保留参照块，并按冻结前写定的范围判断漂移。[测量检查](RESEARCH-GUIDE.md#measurement-checks)

<a id="measurement-next"></a>

## 下一步选择

当前顺序与依赖统一见[项目 PLAN](../../../PLAN.md)：S16 限额收口，多 CTA TMA 与有效频率并行，C1-01 按原协议推进；其开发作业已取消，恢复前先定短 kernel 样本数与漂移处理。m11 的两次冻结失败已记录，暂不新增单 warp 任务；候选与重启条件见[M11-SMEM](../../../model/mechanism/hardware_rules/M11-SMEM.md#4-结论与重启条件)。旧任务未使用预算不自动转为开发。

已结束测量的预算与终点见[主线](../../../model/semi_empirical/MAINLINE-LONG-K.md#measurement-20261012)与[m11 历史安排](../../../model/mechanism/hardware_rules/M11-SMEM.md#measurement-20261012)。后续验证使用新冻结与新留出，具体身份和执行清单由对应任务维护；V10 的最终 SHA 确认与门槛仍按[原验证细则](RESEARCH-GUIDE.md#validation)。

模型接口、按用途划分的停止条件和留出要求见[研究细则](RESEARCH-GUIDE.md)。正文先写发现；身份和传输记录放 manifest。保留同卡配对、提交前 SASS 检查、原始数据及失败，不靠删除条件或重做旧留出来宣称通过。

<a id="m01"></a>

机制储备与已有证据见[微架构审阅](../../../model/mechanism/hardware_rules/MICROARCH-REVIEW.md#m01)。现行公式见 [MODEL-CANDIDATES](../../../model/semi_empirical/MODEL-CANDIDATES.md#v09-model)，已通过独立留出的规则见 [RULES](RULES.md)。
