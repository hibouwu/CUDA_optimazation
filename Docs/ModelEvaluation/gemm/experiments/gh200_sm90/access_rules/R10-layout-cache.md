# R10：A/B 行距、地址覆盖与执行规模

## 当前结论

A/B 行距的时间响应依赖配置、执行方向、规模和起步状态，不能只按地址覆盖加一个共同罚项。局部冻结两候选均失败；V09 行距失败保持原成绩。当前没有接入正式预测的新行距价格。

静态账本已区分名义 payload、有效源地址、覆盖并集、源身份、目的投递与输出义务。AlongM/N 的局部完整工作张量可以相同而实测响应不同；single36/126 的前36个 CTA 工作相同而响应不同。因此局部覆盖计数本身不足以定价，源身份和其他 CTA 如何改变状态仍是缺口。[完整工作审查](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/B-20261011-request-work-audit-v3/B-request-work-audit.json)。

源共享、独立请求路径和共享队列的具体候选没有同时通过 first/later、方向、规模及空间约束；闭区间证书只排除声明的固定图与价格域，不排除全部共享服务模型。原始尾部的 fill、完整 MMA、释放和目的完成义务保留，不能按有效源量删掉实际工作。

## 作业索引

2026-10-11，B的静态工作统一为 [R13供给账本](R13-async-retirement.md#supply-work-ledger)。`gh200-supply-work-ledger/1`保留每个CTA/输出的descriptor box、源身份和multicast目的前驱，不再只返回七列聚合覆盖；R10的两个raster与36/126/132执行规模已进入57条件守恒复核。名义源payload、有效源地址、去重源身份、目的投递和D输出分别记账。该接口仅补齐组织信息，没有给方向或B32另加价格；旧几何B32变慢、新几何未见同量级惩罚仍是同一供给状态候选必须同时解释的约束。A16和stage4不用于另拟行距价，原局部冻结失败不变。


| 作业 | 固定对象 | 结果入口 |
|---|---|---|
| 735876 | 三配置，2600×3000×2000，B 行距 6000/6016/6144 B | [v2 完整时间](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v2/analysis-formal/summary.json)、[v4 cfg_b 局部周期](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/analysis-formal/summary.json)、[独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reviews/ROOT-independent.md) |
| 737322，V08F | A/B/D 行距、K/N 尾部；84 个解释性条件 | [原汇总](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/followup-v1/followup.json)、[供给残差](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/A-20261009-supply-v3/summary.json) |
| 738101 | cfg_a、2304×3072×4096，grid 32/64/96/132 × 三种行距 | [独立回放](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-sm-job738101/reanalysis/manager-formal-replay/summary.json) |
| 738169 | cfg_b h03 四角，2400×3000×1000，同容量 | [逐 trial 联合项与扰动](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-joint-pitch-job738169/reanalysis/B-20261009-matched-joint-final/joint-pitch.json) |
| 738203 | cfg_b B 行距五点，2304×3072×1024，局部冻结检验 | [冻结](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/frozen/r10-b-pitch.json)、[原评分](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/b-pitch-score-v1/b-pitch-score.json)、[位置与扰动](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/B-20261009-frozen-negative/localization.json) |
| 739011，V09 | 新几何 A16/B32/B80/B112 与 stage4 | [完整留出](V09-component-validation.md)、[逐例复核](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/manager-results-v1/cases.csv) |
| 739571 | 同几何AlongM/N×B0/B32 | [方向与行距差分](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-raster-pitch.json) |
| 739585 | 同输入map，36/126单输出CTA与132持久CTA，各配B0/B32 | [同源组压力对照](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-source-pressure.json) |
| 739598 | B基址+0/+256 B × 行距+0/+32 B，分配容量相同 | [位置与相邻面板对应](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-base-shift.json) |
| 739608 | CTA55/91整条工作序列交换 × B0/B32，逐轮工作集合不变 | [工作与物理位置对应](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-work-swap.json) |
| 739619 | 固定CTA55工作与SM，producer起步0/512 ns × B0/B32 | [实际时移与行距交互](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-producer-delay.json) |

<a id="source-event-transfer"></a>

## 推导与历史证据索引

本页只保留当前结论和入口；原有局部条件、公式、反证与逐例证据移到各自的 reanalysis README，原锚点继续可用。整理不修改采样、参数、冻结成绩或资格。

| 原问题 | 详情 |
|---|---|
| <a id="作业与证据索引"></a>作业与证据索引 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#作业与证据索引) |
| <a id="source-event-transfer"></a><a id="真实输出次数与源身份仍不足以迁移b32响应"></a>真实输出次数与源身份仍不足以迁移B32响应 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#真实输出次数与源身份仍不足以迁移b32响应) |
| <a id="不同源box的地址覆盖部分重叠"></a>不同源box的地址覆盖部分重叠 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#不同源box的地址覆盖部分重叠) |
| <a id="地址重叠反馈到ready后当前参数仍未暴露首输出停顿"></a>地址重叠反馈到ready后，当前参数仍未暴露首输出停顿 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#地址重叠反馈到ready后当前参数仍未暴露首输出停顿) |
| <a id="强响应在native-cycles中仍存在统一换单位不能修复零响应"></a>强响应在native cycles中仍存在，统一换单位不能修复零响应 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#强响应在native-cycles中仍存在统一换单位不能修复零响应) |
| <a id="startup-absorption"></a><a id="当前在途共享候选吸收了512ns起步变化"></a>当前在途共享候选吸收了512ns起步变化 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#当前在途共享候选吸收了512ns起步变化) |
| <a id="每需求最短路径可以打破吸收但固定常数仍不足"></a>每需求最短路径可以打破吸收，但固定常数仍不足 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#每需求最短路径可以打破吸收但固定常数仍不足) |
| <a id="共享队列的位置必须同时解释迟发与原首窗"></a>共享队列的位置必须同时解释迟发与原首窗 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#共享队列的位置必须同时解释迟发与原首窗) |
| <a id="request-path-complete"></a><a id="独立请求路径接入后一个前缀可行点仍未通过联合约束"></a>独立请求路径接入后，一个前缀可行点仍未通过联合约束 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#独立请求路径接入后一个前缀可行点仍未通过联合约束) |
| <a id="request-path-interval"></a><a id="single126在原必要价格闭区间内始终无首窗停顿"></a>single126在原必要价格闭区间内始终无首窗停顿 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#single126在原必要价格闭区间内始终无首窗停顿) |
| <a id="request-work-identifiability"></a><a id="覆盖工作向量本身没有解释方向差异"></a>覆盖工作向量本身没有解释方向差异 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#覆盖工作向量本身没有解释方向差异) |
| <a id="原-b-行距对照配置之间不能共用一个罚项"></a>原 B 行距对照：配置之间不能共用一个罚项 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#原-b-行距对照配置之间不能共用一个罚项) |
| <a id="legacy-boundary-ledger"></a><a id="原始尾部的有效源目的fill与d范围"></a>原始尾部的有效源、目的fill与D范围 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#原始尾部的有效源目的fill与d范围) |
| <a id="legacy-persistent-binding"></a><a id="原始调度输出序号与输入ring绑定"></a>原始调度、输出序号与输入ring绑定 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#原始调度输出序号与输入ring绑定) |
| <a id="legacy-tail-lifecycle"></a><a id="最后k16的完整mma工作与stage释放"></a>最后K16的完整MMA工作与stage释放 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#最后k16的完整mma工作与stage释放) |
| <a id="legacy-completion-ports"></a><a id="零源盒与目的完成端口"></a>零源盒与目的完成端口 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#零源盒与目的完成端口) |
| <a id="legacy-new-common-interface"></a><a id="旧尾部与新完整tile的共同接口"></a>旧尾部与新完整tile的共同接口 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#旧尾部与新完整tile的共同接口) |
| <a id="v08-stride"></a><a id="v08f分开行距和尺寸尾部"></a>V08F：分开行距和尺寸尾部 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#v08f分开行距和尺寸尾部) |
| <a id="v08-stride-next"></a><a id="配置与位置反例"></a>配置与位置反例 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#配置与位置反例) |
| <a id="requested-sm-pitch"></a><a id="固定形状改变持久-grid行距效应随工作位置变化"></a>固定形状改变持久 grid：行距效应随工作位置变化 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#固定形状改变持久-grid行距效应随工作位置变化) |
| <a id="h03-联合行距正交互不等于物理请求相互放大"></a>h03 联合行距：正交互不等于物理请求相互放大 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#h03-联合行距正交互不等于物理请求相互放大) |
| <a id="bcurve-negative"></a><a id="局部冻结负结果job738203"></a><a id="b-覆盖的冻结检验两候选均失败"></a>B 覆盖的冻结检验：两候选均失败 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#b-覆盖的冻结检验两候选均失败) |
| <a id="p64-失败位于多个后续输出位置"></a>p64 失败位于多个后续输出位置 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#p64-失败位于多个后续输出位置) |
| <a id="coverage-identifiability"></a><a id="覆盖组合的可辨识性与失败"></a>覆盖组合的可辨识性与失败 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#覆盖组合的可辨识性与失败) |
| <a id="v09-pitch-comparison"></a><a id="旧-b32b64-与-v09-b32先比较几何再谈对齐"></a>旧 B32/B64 与 V09 B32：先比较几何，再谈对齐 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#旧-b32b64-与-v09-b32先比较几何再谈对齐) |
| <a id="同工作配对差异确实出现在主循环"></a>同工作配对：差异确实出现在主循环 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#同工作配对差异确实出现在主循环) |
| <a id="遍历方向足以改变b32响应"></a>遍历方向足以改变B32响应 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#遍历方向足以改变b32响应) |
| <a id="相同b源组在不同执行规模下的代价不同"></a>相同B源组在不同执行规模下的代价不同 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#相同b源组在不同执行规模下的代价不同) |
| <a id="shared-service-candidate"></a><a id="从分组响应到逐cta两参数候选仍缺状态"></a>从分组响应到逐CTA：两参数候选仍缺状态 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#从分组响应到逐cta两参数候选仍缺状态) |
| <a id="base-shift"></a><a id="基址平移没有让响应整体跟随相邻b面板"></a>基址平移没有让响应整体跟随相邻B面板 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#基址平移没有让响应整体跟随相邻b面板) |
| <a id="历史推导与复核"></a>历史推导与复核 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#历史推导与复核) |
| <a id="work-swap"></a><a id="交换工作序列后响应仍留在原物理位置"></a>交换工作序列后，响应仍留在原物理位置 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#交换工作序列后响应仍留在原物理位置) |
| <a id="producer-delay"></a><a id="固定工作与sm起步状态仍改变行距代价"></a>固定工作与SM，起步状态仍改变行距代价 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#固定工作与sm起步状态仍改变行距代价) |
| <a id="producer-delay-median-estimands"></a><a id="局部中位数保留联合端点与配对定义"></a>局部中位数保留联合端点与配对定义 | [原推导与证据](reanalysis/20261012-consolidation/R10/README.md#局部中位数保留联合端点与配对定义) |
