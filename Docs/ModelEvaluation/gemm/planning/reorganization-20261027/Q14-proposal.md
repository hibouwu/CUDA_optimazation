# Q14 实验族候选清单

历史草案：本页仅在原 Q14 内拆族，已由 [全 Q14 问题归属调整清单](Q14-question-ownership.md) 取代。先审阅哪些轮次归回 Q01–Q13，再确定目标问题下的实验族；本页不能作为迁移依据。

本表只提出路径，不执行迁移。原 71 个轮次逐一登记；族名和端点仍供审阅，不是已确认的分类。
按对象、变量、端点先提出 29 个候选族；8 个主题只作页面导航。相关模型回放、必要界和准备记录已明确标注，不伪装成新增实机测量。部分较宽候选的边界在下文单列，仍需审阅。
同一轮次名保持原样，避免无证据地补卡名。等值源 001–004、失败构建和换卡历史仍在同族。

## compute-completion-lag（1 轮）

对象：组合中的计算完成。变量：完成滞后参数。端点：G3 S6/S4 的 L 与 plain 预测。

归属依据／待审边界：CPU 局部完成规则迁移；与 wait0 实机干预分开。

- [20261018-Q14-compute-completion-transfer-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-compute-completion-transfer-001/README.md)

## wgmma-wait-policy（2 轮）

对象：GEMM 正常计算等待。变量：正常 wait1/wait0。端点：S6/S4 的 L、plain 与对应预测。

归属依据／待审边界：预测和实测围绕同一等待干预。

- [20261018-Q14-gemm-wait-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-gemm-wait-001/README.md)
- [20261018-Q14-gemm-wait-prediction-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-gemm-wait-prediction-001/README.md)

## full-wait-policy（4 轮）

对象：GEMM consumer full 等待。变量：blocking/polling，含行距对照。端点：plain 响应；具备 dual 的轮次补充 L。

归属依据／待审边界：14 只有 plain；若要求所有辅助端点也完全相同，应继续拆分。

- [20261018-Q14-full-wait-policy-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-full-wait-policy-001/README.md)
- [20261018-Q14-runtime-full-wait-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-runtime-full-wait-001/README.md)
- [20261020-Q14-full-wait-plain-002](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261020-Q14-full-wait-plain-002/README.md)
- [20261023-Q14-stride-wait-policy-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-stride-wait-policy-001/README.md)

## wait0-tail-observer（2 轮）

对象：末尾 wait0 返回至结束。变量：循环外时戳观察实现。端点：边界时戳与观察资格。

归属依据／待审边界：包含失败实现，不能按 runtime 字样另分。

- [20261019-Q14-runtime-tail-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261019-Q14-runtime-tail-001/README.md)
- [20261019-Q14-wait0-boundary-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261019-Q14-wait0-boundary-001/README.md)

## mainloop-active-repeat（3 轮）

对象：固定物理 S6 的主循环。变量：活动槽额度及已测重复计算。端点：FIRST_MMA→MAIN_END、阶段响应。

归属依据／待审边界：6/12 为已有数据诊断，8 为程序接入；是否并入原 G3 族待核对。

- [20261018-Q14-program-budget-scope-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-program-budget-scope-001/README.md)
- [20261018-Q14-runtime-intervention-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-runtime-intervention-001/README.md)
- [20261019-Q14-stage-scope-screen-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261019-Q14-stage-scope-screen-001/README.md)

## mainloop-compute-repeat（4 轮）

对象：GEMM 行距代价的隐藏。变量：单遍/三遍及其实现修复。端点：CTA 两段 L 的行距增量、plain 桥接。

归属依据／待审边界：恢复执行路径和静态 dispatch 属于同一干预的实现演进。

- [20261021-Q14-direct-repeat-path-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-direct-repeat-path-001/README.md)
- [20261021-Q14-static-stage-repeat-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-static-stage-repeat-001/README.md)
- [20261021-Q14-stride-compute-discrimination-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-stride-compute-discrimination-001/README.md)
- [20261022-Q14-static-repeat-dispatch-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-static-repeat-dispatch-001/README.md)

## gemm-active-peers（1 轮）

对象：同行 CTA 对目标 CTA 的影响。变量：132/1 活动 CTA。端点：CTA0 两段 L 与 plain。

归属依据／待审边界：不能并入仅修改源地址或输出 copy 的族。

- [20261022-Q14-active-peer-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-active-peer-001/README.md)

## epilogue-defer-order（1 轮）

对象：输出与下一数学 WG 交接。变量：E0 permit 延后依赖。端点：C16−C1、M−C16、E0、plain。

归属依据／待审边界：计划和 EXECUTION 代替缺失的 README；保留独立。

- [20261018-Q14-epilogue-defer-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-epilogue-defer-001/EXECUTION.md)

## output-shared-service（2 轮）

对象：输出后台工作与输入共享。变量：输出源读取后的共享占用规则。端点：早段/后段响应及 plain 回归。

归属依据／待审边界：5 是 9 的同版完整时间回归，标为关联回归而非新增物理测量。

- [20261018-Q14-output-write-gemm-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-output-write-gemm-001/EXECUTION.md)
- [20261018-Q14-shared-output-service-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-shared-output-service-001/EXECUTION.md)

## gemm-peer-output（1 轮）

对象：同行 SMEM→global 输出 TMA。变量：同行输出 copy 开关。端点：CTA0 行距周期增量与 plain。

归属依据／待审边界：输出保留/屏蔽与 epilogue 顺序不是同一种操纵。

- [20261022-Q14-peer-output-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-peer-output-001/README.md)

## gemm-source-sharing（8 轮）

对象：真实 GEMM 的跨 CTA 源重用。变量：等值地址折叠/逐 CTA 等值副本。端点：CTA0 两段 L、plain 及对应预测。

归属依据／待审边界：001–004 换卡、换序与修复保持同族；44/50 是该干预的程序接入。

- [20261022-Q14-equal-value-source-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-001/README.md)
- [20261022-Q14-equal-value-source-002](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-002/README.md)
- [20261022-Q14-equal-value-source-003](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-003/README.md)
- [20261022-Q14-equal-value-source-004](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-004/README.md)
- [20261022-Q14-program-intervention-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-program-intervention-001/README.md)
- [20261022-Q14-source-fold-replay-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-source-fold-replay-001/README.md)
- [20261023-Q14-private-source-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-private-source-001/README.md)
- [20261023-Q14-private-source-transfer-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-private-source-transfer-001/README.md)

## input-cache-organization（2 轮）

对象：输入服务中的源缓存状态。变量：分区组相联/端口受理后查询。端点：G3 两段 L、plain、full 状态。

归属依据／待审边界：缓存组织候选；不与实机等值源改址直接混合。

- [20261022-Q14-banked-source-cache-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-banked-source-cache-001/README.md)
- [20261023-Q14-port-ordered-cache-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-port-ordered-cache-001/README.md)

## input-line-merge（4 轮）

对象：输入源行发出与订阅。变量：32/128B 合并粒度、域范围、合并截止。端点：G3 两段 L、plain、请求工作计数。

归属依据／待审边界：共同主端点为 G3 行距响应；每轮具体合并规则必须保留。

- [20261022-Q14-bounded-sector-merge-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-bounded-sector-merge-001/README.md)
- [20261022-Q14-domain-line-pool-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-domain-line-pool-001/README.md)
- [20261022-Q14-line-subscription-screen-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-line-subscription-screen-001/README.md)
- [20261022-Q14-source-line-pool-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-source-line-pool-001/README.md)

## input-read-reply-network（6 轮）

对象：输入读取与逐接收者应答。变量：分区路由、在途共享、订阅上限与服务价格。端点：G3 L/plain，部分轮次另读纯 TMA release 窗口。

归属依据／待审边界：较宽候选：69 是参数标定，纯 TMA 与 G3 双端点需审阅是否再拆。

- [20261022-Q14-address-routed-input-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-address-routed-input-001/README.md)
- [20261022-Q14-request-reply-network-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-request-reply-network-001/README.md)
- [20261023-Q14-bounded-read-fanout-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-bounded-read-fanout-001/README.md)
- [20261023-Q14-public-fold-correction-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-public-fold-correction-001/README.md)
- [20261023-Q14-read-sharing-network-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-read-sharing-network-001/README.md)
- [20261023-Q14-source-only-calibration-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-source-only-calibration-001/README.md)

## tma-formation-return（3 轮）

对象：请求形成与 producer 返回。变量：形成服务位置、返回反馈及地址绑定修复。端点：纯 TMA release/早晚窗口；55 另做 GEMM 迁移。

归属依据／待审边界：53 为同址修正，保留同一次模型谱系；不能归成实机发布时戳实验。

- [20261023-Q14-address-matched-return-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-address-matched-return-001/README.md)
- [20261023-Q14-formation-before-cache-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-formation-before-cache-001/README.md)
- [20261023-Q14-formation-return-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-formation-return-001/README.md)

## full-entry-state（2 轮）

对象：consumer 进入等待前的 full 状态。变量：位图观察实现/overlay 修复。端点：各 Ktile ready/unready 位图及扰动资格。

归属依据／待审边界：实现1未采到机制状态，失败轮次仍保留。

- [20261022-Q14-full-entry-mask-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-full-entry-mask-001/README.md)
- [20261022-Q14-full-entry-mask-002](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-full-entry-mask-002/README.md)

## operand-completion-set（1 轮）

对象：g43 分操作数完成集合。变量：A/B 请求分组的观察。端点：A-only/B-only/A+B 未完成集合。

归属依据／待审边界：端点区别于整组 full 位图，不强并单轮。

- [20261023-Q14-operand-completion-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-operand-completion-001/README.md)

## producer-publication-progress（4 轮）

对象：g43 查询处 producer 发布进度。变量：发布标记的全局/CTA 本地实现。端点：B1 后发布计数、full 未完成状态。

归属依据／待审边界：45 是对应计数和时间线的 CPU 核对。

- [20261022-Q14-producer-publication-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-001/README.md)
- [20261022-Q14-producer-publication-002](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-002/README.md)
- [20261022-Q14-producer-publication-003](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-003/README.md)
- [20261022-Q14-publication-time-alignment-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-publication-time-alignment-001/README.md)

## request-query-age（3 轮）

对象：B1 后发布至 full 查询/完成边界。变量：软件计时原点及返回规则对齐。端点：年龄区间、query−FIRST、MAIN−query、full 状态。

归属依据／待审边界：57 修正上界原点；不得沿用已撤销的 issue 年龄推断。

- [20261022-Q14-published-request-age-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-published-request-age-001/README.md)
- [20261023-Q14-full-boundary-screen-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-full-boundary-screen-001/README.md)
- [20261023-Q14-query-return-connection-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-query-return-connection-001/README.md)

## request-age-raster（3 轮）

对象：请求年龄的空间分布。变量：源组/逻辑组与 raster 对应。端点：中心化年龄预测/空间投影。

归属依据／待审边界：28 为干预前诊断，48 为预注册可分性，47 为实机尝试。

- [20261022-Q14-completion-spatial-screen-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-completion-spatial-screen-001/README.md)
- [20261022-Q14-raster-age-discrimination-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-raster-age-discrimination-001/README.md)
- [20261022-Q14-raster-contrast-prediction-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-raster-contrast-prediction-001/README.md)

## gemm-local-completion-release（4 轮）

对象：局部完成/继续/释放规则接回组合。变量：固定联合规则替换旧 reuse 项。端点：G3及开发条件的 L/plain、关键 CTA。

归属依据／待审边界：G6 补测不另分族；18 为同版时间线分析。

- [20261020-Q14-local-bundle-G6-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261020-Q14-local-bundle-G6-001/README.md)
- [20261020-Q14-local-bundle-regression-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261020-Q14-local-bundle-regression-001/README.md)
- [20261020-Q14-local-bundle-transfer-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261020-Q14-local-bundle-transfer-001/README.md)
- [20261021-Q14-critical-path-current-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-critical-path-current-001/README.md)

## producer-preparation-cost（1 轮）

对象：producer 逐轮请求准备。变量：两档已测准备价格替换软件 gap。端点：G3/A16 的 L、plain、CTA 结束跨度。

归属依据／待审边界：准备成本与消费者完成/释放的对象不同，保留独立。

- [20261021-Q14-producer-preparation-transfer-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-producer-preparation-transfer-001/README.md)

## input-compute-service-bound（1 轮）

对象：TMA 与 WGMMA 共同服务必要容量。变量：全逻辑引用/相邻 B 复用的假设。端点：原生 D 窗口与服务下界。

归属依据／待审边界：下界推导与 GEMM 时间预测不同，保留独立候选。

- [20261022-Q14-joint-smem-bound-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-joint-smem-bound-001/README.md)

## gemm-input-compute-sharing（2 轮）

对象：GEMM 输入与计算共同服务。变量：逻辑字节/源片段引用计费。端点：G3 两段 L、plain、后续 full 未完成数。

归属依据／待审边界：同一共同资源模型的需求计数变化。

- [20261023-Q14-gather-joint-service-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-gather-joint-service-001/README.md)
- [20261023-Q14-local-joint-service-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-local-joint-service-001/README.md)

## gpc-copy-service（2 轮）

对象：逐 CTA 副本的 GPC 投递。变量：固定价格的 FIFO/工作守恒约束。端点：完整时间预测/必要下界、L 响应。

归属依据／待审边界：29 是 37 的必要下界诊断；若只收测量轮次则作为关联分析保留。

- [20261022-Q14-copy-work-bound-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-copy-work-bound-001/README.md)
- [20261022-Q14-gpc-delivery-bound-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-gpc-delivery-bound-001/README.md)

## tma-read-service-bound（1 轮）

对象：独立 TMA 读取/应答容量。变量：按实际地址与资源归属统计工作。端点：每片段/每16KiB服务时间上界。

归属依据／待审边界：与 copy 下界对象不同；不能归入 input-counter。

- [20261023-Q14-independent-service-bounds-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-independent-service-bounds-001/README.md)

## gemm-input-sector-counter（1 轮）

对象：真实 GEMM 输入路径工作量。变量：base/A16 行距。端点：NCU L2 sector 比值及采集资格。

归属依据／待审边界：实际计数未取得；plain 有效不等于 counter 实验成功。

- [20261023-Q14-input-counter-localization-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-input-counter-localization-001/README.md)

## gemm-repeatability-control（1 轮）

对象：同二进制桥接的可重复性。变量：相邻重复/顺序。端点：plain、L cycle/ns 配对偏差。

归属依据／待审边界：Q10 跨问题候选；本表暂保留 Q14 等待确认。

- [20261022-Q14-bridge-null-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-bridge-null-001/README.md)

## compute-handoff-contract（1 轮）

对象：跨输出计算模块状态。变量：上一输出 service_end 状态传递。端点：CPU 逐节点一致性和默认回归。

归属依据／待审边界：软件契约检查，Q06 或关联实现记录候选；暂保留 Q14。

- [20261019-Q14-compute-handoff-contract-001](../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261019-Q14-compute-handoff-contract-001/README.md)
