# GH200 GEMM 时间预测计划

更新：2026-10-12。两条测量线均已结束，没有新候选；当前整理计划与检查用途，GPU 0。提交保留本地 main，不推送。执行状态见 [EXECUTION](EXECUTION.md#status)。

<a id="prediction-boundary"></a>

## 目标

在运行前，由程序、静态执行条件、参考卡冻结参数和声明的暖机协议，预测 cfg_a/b/c 的完整时间并解释变化。目标、暖机或前序调用的实测值只用于诊断。支持范围按校准覆盖预先声明，非 128 B 对齐行距、非默认 stage 和原拒绝项仍不支持；具体边界见[细则](RESEARCH-GUIDE.md#prediction-boundary)。

## 现在知道什么

| 对象 | 结论与证据 |
|---|---|
| V09 | 整体未通过：29 个可评分条件的绝对误差中位数 4.11%、最大 35.46%，另 1 个拒绝；分项和关键 CTA 也有错误。已转为开发数据，冻结成绩不改。[V09](V09-component-validation.md) |
| 长 K | 时钟扩展和局部修正均未得到候选。必要校准只完成参照，因旧 event-extra 门槛终止，540 个开发槽位未执行；新参数能否独立定价仍未知。[R09](R09-inkernel-clock-stages.md) |
| 普通 SMEM | stride 冲突关系在开发条件上吻合；宽广播及 lane 排列反证了旧需求描述，依赖执行仍有缺口。没有独立留出成绩。[R03](R03-access-demand.md#m11-current)、[EXP-09](../EXP-09-smem.md) |

<a id="mainline"></a><a id="smem-contract"></a>

## 开放问题

1. **长 K 的后续 L：** 能否用不依赖 G6 的新形状，区分局部成本不足与 CTA／包络组合偏差？必要开发数据尚未采到。[主线任务](MAINLINE-LONG-K.md)
2. **SMEM 的阶段与广播：** 哪种静态的 lane、字和广播描述能区分已有反例，并解释依赖执行？原模型已经按 128 B lane 子组算冲突，简单重用它不足以解决密集布局反例。[m11 任务](M11-SMEM.md)
3. **完整时间转换：** event−包络增量 0.688–0.848 µs 影响哪些预测用途？原因尚未识别；局部 L、SMEM 和完整时间的有效性分别检查。[测量检查](RESEARCH-GUIDE.md#measurement-checks)

<a id="order"></a><a id="measurement-next"></a>

## 下一步选择

本次先完成文档收缩；新的研究尚未启动。建议先用现有反例推导 m11 的静态预测，再决定是否需要新测量；主线补测先明确待定参数和必要对照，重新申请作业预算。代码收敛另做小范围任务，先合并活跃分析器的重复逻辑。

两项已授权测量的预算与终点见[已结束的安排](MEASUREMENT-20261012.md)。未使用的 m11 条件留出预算不能转作开发。得到候选后，V10 使用新形状：先校准、冻结全部预测和支持范围，用户确认最终 SHA，再采留出；完整时间目标中位 ≤5%、最大 ≤10%，分项与关键 CTA 按各自冻结标准判断。[验证细则](RESEARCH-GUIDE.md#validation)

<a id="model-interface"></a><a id="7-验证规则"></a><a id="full-validation"></a>

模型接口、按用途划分的停止条件和留出要求见[研究细则](RESEARCH-GUIDE.md)。正文先写发现；身份和传输记录放 manifest。保留同卡配对、提交前 SASS 检查、原始数据及失败，不靠删除条件或重做旧留出来宣称通过。

<a id="mechanism-backlog"></a>

<a id="m01"></a><a id="m02"></a><a id="m03"></a><a id="m04"></a><a id="m05"></a><a id="m06"></a><a id="m07"></a><a id="m08"></a><a id="m09"></a><a id="m10"></a><a id="m11"></a><a id="m12"></a><a id="m13"></a><a id="m14"></a><a id="m15"></a><a id="m16"></a><a id="m17"></a><a id="m18"></a><a id="m19"></a><a id="m20"></a><a id="m21"></a><a id="m22"></a><a id="m23"></a><a id="m24"></a><a id="m25"></a>

机制储备 m01–m25 移至[清单](MECHANISM-BACKLOG.md)；旧入口保留，详细任务与预算不再占本页。现行公式见 [MODEL-CANDIDATES](MODEL-CANDIDATES.md#v09-model)，已通过独立留出的规则见 [RULES](RULES.md)。

<a id="1-目标与预测边界"></a><a id="2-主线合同长-k-时钟与事件组合修正"></a><a id="起点与问题"></a><a id="预算检查点与交付"></a><a id="支持评分与采用规则"></a><a id="终点"></a><a id="3-之后的顺序与工作规则"></a><a id="顺序"></a><a id="工作规则"></a><a id="4-研究范围与共同模型接口"></a><a id="共同的模型接口"></a><a id="5-实验共用条件"></a><a id="6-机制储备清单"></a><a id="m11-第一轮合同普通-smem-访问模型"></a><a id="m11-后续离线任务需求聚合诊断"></a><a id="整理后的测量合同2026-10-12-用户授权"></a><a id="每条机制规则先有自己的留出"></a><a id="组合验证检查误差是否抵消"></a><a id="v10-留出的覆盖组织"></a><a id="8-文档职责"></a>

原各节入口对应[预测与验证细则](RESEARCH-GUIDE.md)、[主线任务](MAINLINE-LONG-K.md)、[m11 任务](M11-SMEM.md)与[已结束的测量安排](MEASUREMENT-20261012.md)。
