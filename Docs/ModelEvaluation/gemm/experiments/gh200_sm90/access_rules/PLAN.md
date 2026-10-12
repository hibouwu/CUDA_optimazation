# GH200 GEMM 时间预测计划

更新：2026-10-12。

## 目标

在运行前，由程序、静态执行条件、参考卡冻结参数和声明的暖机协议，预测 cfg_a/b/c 的完整时间并解释变化。目标、暖机或前序调用的实测值只用于诊断。支持范围按校准覆盖预先声明，非 128 B 对齐行距、非默认 stage 和原拒绝项仍不支持；具体边界见[细则](RESEARCH-GUIDE.md#prediction-boundary)。

## 现在知道什么

| 对象 | 结论与证据 |
|---|---|
| V09 | 整体未通过：29 个可评分条件的绝对误差中位数 4.11%、最大 35.46%，另 1 个拒绝；分项和关键 CTA 也有错误。已转为开发数据，冻结成绩不改。[V09](V09-component-validation.md) |
| 长 K | 时钟扩展和局部修正均未得到候选。必要校准只完成参照，因旧 event-extra 门槛终止，540 个开发槽位未执行；新参数能否独立定价仍未知。[R09](R09-inkernel-clock-stages.md) |
| 普通 SMEM | stride 冲突关系在开发条件上吻合；旧需求和新最忙 bank 标量都不能覆盖宽广播／排列，仍缺广播与依赖描述。没有独立留出成绩。[R03](R03-access-demand.md#m11-current)、[EXP-09](../EXP-09-smem.md) |

<a id="mainline"></a><a id="smem-contract"></a>

## 开放问题

1. **长 K 的后续 L：** 能否用不依赖 G6 的新形状，区分局部成本不足与 CTA／包络组合偏差？必要开发数据尚未采到。[主线任务](MAINLINE-LONG-K.md)
2. **SMEM 的阶段与广播：** 哪种静态的 lane、字和广播描述能区分已有反例，并解释依赖执行？原模型已经按 128 B lane 子组算冲突，简单重用它不足以解决密集布局反例。[m11 任务](M11-SMEM.md)
3. **完整时间转换：** event−包络增量 0.688–0.848 µs 影响哪些预测用途？原因尚未识别；局部 L、SMEM 和完整时间的有效性分别检查。[测量检查](RESEARCH-GUIDE.md#measurement-checks)

<a id="measurement-next"></a>

## 下一步选择

继续用已有反例约束 m11 的 bank 分布和广播描述；首个新标量已被广播对照否定，[离线推导](R03-access-demand.md#m11-bank-vector)保留等价约束。先解释这些条件，再决定是否拟合或测量。主线补测先明确待定参数和最少对照，另申请预算；代码收敛另做小范围任务。

已结束测量的预算与终点分别见[主线](MAINLINE-LONG-K.md#measurement-20261012)与[m11](M11-SMEM.md#measurement-20261012)。未使用的 m11 条件留出预算不能转作开发。得到候选后，V10 使用新形状：先校准、冻结全部预测和支持范围，用户确认最终 SHA，再采留出；完整时间目标中位 ≤5%、最大 ≤10%，分项与关键 CTA 按各自冻结标准判断。[验证细则](RESEARCH-GUIDE.md#validation)

模型接口、按用途划分的停止条件和留出要求见[研究细则](RESEARCH-GUIDE.md)。正文先写发现；身份和传输记录放 manifest。保留同卡配对、提交前 SASS 检查、原始数据及失败，不靠删除条件或重做旧留出来宣称通过。

<a id="m01"></a>

机制储备与已有证据见[微架构审阅](MICROARCH-REVIEW.md#m01)。现行公式见 [MODEL-CANDIDATES](MODEL-CANDIDATES.md#v09-model)，已通过独立留出的规则见 [RULES](RULES.md)。
