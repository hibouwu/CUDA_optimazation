# R09：调用内时钟、输入与时间换算

## 当前结论

V09 r2 的长 K 时钟迁移未通过；后续八小时合同以“缺一个必要测量”结束，GPU 使用为 0，没有采用新系数或修正。固定 29 条件上的第一形式把 cfg_a G6 的 plain 误差从 −3.698% 改成 −16.739%，留掉 G6 形状后为 −36.507%；其他配置的改善不能抵消这一退化。[失败候选与独立复放](../../../../../../results/gh200_resource_campaign/access_rules/20261011-long-k-event-bounded/README.md)。

包络误差必须分开局部窗口与 CTA 组合。G6 实测最晚 CTA 属于模型的 36 个最大值并列候选，10/10 次命中；代表编号在 max 之后选择，单改编号不会改变完整时间。后续 L 是主要残差位置，但尚未识别物理原因；读取实测频率的结果只是诊断。

缺的是**参考卡、相同 dual 协议下，不依赖 G6 定价的长 K 后续 L 校准，以及匹配的 ends/dual 桥接**。已有高覆盖 ends 不能替代局部 dual 服务价格；旧 cfg_c K16384 dual 只有一个输出，也不能给 later L 定价。必要补测已预列 8192×6144、3072×16384，两形状各 K2048/4096/32768、三配置，共18条件；加桥接与参照共660进程。[测量方案与静态核对](../../../../../../results/gh200_resource_campaign/access_rules/20261011-long-k-event-bounded/measurement-v2/README.md)。用户已批准整理后执行这一校准作业；它不是 V10 留出。

局部周期换算仍需匹配端点、执行历史和观察协议。完整 event 接近不能覆盖局部分项失败，边际中位数也不能直接组合成逐调用的最后完成时间。此前各项反证、未辨识项及原误差合同保留在下方索引。

## 作业索引

| 实验问题 / 作业 | 一句结果 | 证据入口 |
|---|---|---|
| 调用内和调用后时钟，735634/735669 | 调用后探针导致旧20%开销误判 | [R07修正](../../../../../../results/gh200_resource_campaign/access_rules/20261007-ndebug-clock-fixedcost/analysis-r2/summary.json)、[R09报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-r09-clock-stages/report.md) |
| V03/V08频率和周期抵消 | 只换频率使cfg_b周期低估更明显 | [V08诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/C-20261009-clock-diagnostic-v2/diagnostic.json) |
| 公共数值检查，738097 | K65536 random失败保留 | [逐点误差](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-R13-shared-smoke-job738097/reanalysis/C-20261009-random-error-v1/diagnostic.json) |
| 长K输入，738110 | zero可升频而不加速，K20480 random失败 | [比较](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-clock-input-job738110/reanalysis/C-20261009-clock-input-v2/comparison.json) |
| 宽几何输入，738197 | 同形状random慢27%–39%，周期并非不变 | [九条件](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-wide-input-job738197/reanalysis/C-20261009-wide-input-v1/summary.json) |
| 短中窗口及长桥接，738376 | 21条训练支持输入相关候选，跨几何仍有误差 | [source-clock](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-input-clock-calibration-job738376/reanalysis/C-20261009-source-clock-dev-v2/source-clock.json) |
| 等Q/S跨K，738459 | cfg_a非零输入响应失败，zero不支持统一D项 | [完整配对](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-equal-work-job738459/reanalysis/C-20261009-equal-work-clock-v1/equal-work-clock.json) |
| R18部分零活动，738296/738307 | 固定z只能解释部分频率低估 | [八条件](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R18-input-map-job738296/reanalysis/C-20261009-zero-activity-clock-v1/zero-activity-clock.json) |
| V09迁移，739011 | cfg_b长K失败，拒绝项与分项独立保留 | [逐例诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/manager-results-v1/summary.json) |
| V09三层分解，复用739011 | cfg_b条件时钟失配，删源项无益，缺同条件ends/dual对照 | [300调用与删项结果](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/A-clock-failure.json) |
| 观察协议，739095 | 新卡长K无同量级系统频率差，plain映射仍有偏差 | [两档K、60进程](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/A-controls.json) |
| 次序配对，739539 | 固定Q/S/D后sw8仍升频并减少周期，时长项不充分 | [sw1/sw8](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-swizzle.json) |
| 共同前序，739552 | 共同前序下频率差缩小，周期收益保留 | [四条件、40进程](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-common-history.json) |
| 连续历史，739579 | 周期即时响应与频率历史分离，单极留段无优势 | [700次调用与同期对照](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-history-series.json) |
| 镜像源前序，739590 | 目标源未被前序重复计算仍表现晚段成本；未出现共同重置/追上轨迹 | [40进程、1400调用](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-mirror-history.json) |

离线复核使用各结果中的原分析器或现有 `r09_analyze.py --shared`、`r09_input_source_clock.py`、`r09_equal_work_clock.py`、`r09_zero_activity_clock.py`；参数和命令见对应归档及[固定旧正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md)。新输出写新目录，旧summary/frozen只读；本页不复制准备包启动步骤。

## 推导与历史证据索引

本页只保留当前结论和入口；原有局部条件、公式、反证与逐例证据移到各自的 reanalysis README，原锚点继续可用。整理不修改采样、参数、冻结成绩或资格。

| 原问题 | 详情 |
|---|---|
| <a id="用哪个窗口换算周期与时间"></a>用哪个窗口换算周期与时间？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#用哪个窗口换算周期与时间) |
| <a id="stage-phase-clock"></a><a id="首输出和后续输出能共用换算率吗"></a>首输出和后续输出能共用换算率吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#首输出和后续输出能共用换算率吗) |
| <a id="stage-order-decomposition"></a><a id="阶段逆序保留在哪些观测分量中"></a>阶段逆序保留在哪些观测分量中？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#阶段逆序保留在哪些观测分量中) |
| <a id="s-prefill-alignment"></a><a id="首段-s-与预填如何对齐"></a>首段 S 与预填如何对齐？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#首段-s-与预填如何对齐) |
| <a id="r07"></a><a id="other-clock-evidence"></a><a id="empty-window"></a><a id="调用后频率曾怎样误导主循环解释"></a>调用后频率曾怎样误导主循环解释？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#调用后频率曾怎样误导主循环解释) |
| <a id="v08-clock-reanalysis"></a><a id="总时间接近时周期和频率是否都准确"></a>总时间接近时，周期和频率是否都准确？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#总时间接近时周期和频率是否都准确) |
| <a id="input-modes"></a><a id="改变输入值会只改变频率吗"></a>改变输入值会只改变频率吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#改变输入值会只改变频率吗) |
| <a id="r09-clock-input-job738110"></a><a id="长-kzero-的频率上升为什么没有统一加速"></a>长 K：zero 的频率上升为什么没有统一加速？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#长-kzero-的频率上升为什么没有统一加速) |
| <a id="wide-input-preparation"></a><a id="wide-input-job738197"></a><a id="短-k大几何输入差会扩大吗"></a>短 K、大几何：输入差会扩大吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#短-k大几何输入差会扩大吗) |
| <a id="input-clock-job738376"></a><a id="短中长窗口的输入差是固定偏移吗"></a>短、中、长窗口的输入差是固定偏移吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#短中长窗口的输入差是固定偏移吗) |
| <a id="source-clock-dev-v1"></a><a id="当前输入时钟候选能解释到哪里"></a>当前输入时钟候选能解释到哪里？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#当前输入时钟候选能解释到哪里) |
| <a id="clock-form-candidates"></a><a id="input-clock-mode-fit"></a><a id="equal-work-job738459"></a><a id="qs相同跨k响应是否也能预测"></a>Q/S相同，跨K响应是否也能预测？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#qs相同跨k响应是否也能预测) |
| <a id="r18-zero-activity-mixture"></a><a id="必零乘积比例能解释边界频率吗"></a>必零乘积比例能解释边界频率吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#必零乘积比例能解释边界频率吗) |
| <a id="v09-frequency-domain"></a><a id="v09长k失败约束了什么"></a>V09长K失败约束了什么？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#v09长k失败约束了什么) |
| <a id="v09-three-layer-diagnostic"></a><a id="观测包络能否消除时钟误差"></a>观测包络能否消除时钟误差？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#观测包络能否消除时钟误差) |
| <a id="观察协议是否造成长窗口频率偏差"></a>观察协议是否造成长窗口频率偏差？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#观察协议是否造成长窗口频率偏差) |
| <a id="只改频率响应的形式能修复跨k迁移吗"></a>只改频率响应的形式，能修复跨K迁移吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#只改频率响应的形式能修复跨k迁移吗) |
| <a id="源项是否只是在代替输出活动"></a>源项是否只是在代替输出活动？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#源项是否只是在代替输出活动) |
| <a id="工作量不变只改swizzle频率与服务会变化吗"></a>工作量不变，只改swizzle，频率与服务会变化吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#工作量不变只改swizzle频率与服务会变化吗) |
| <a id="共同前序以后次序收益还在吗"></a>共同前序以后，次序收益还在吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#共同前序以后次序收益还在吗) |
| <a id="连续调用能定出一个频率时间常数吗"></a>连续调用能定出一个频率时间常数吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#连续调用能定出一个频率时间常数吗) |
| <a id="镜像源前序目标地址重复不是必要条件"></a>镜像源前序：目标地址重复不是必要条件 | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#镜像源前序目标地址重复不是必要条件) |
| <a id="time-input-contract"></a><a id="时间预测必须声明哪些输入"></a>时间预测必须声明哪些输入？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#时间预测必须声明哪些输入) |
| <a id="局部分量的中位数能直接组合吗"></a>局部分量的中位数能直接组合吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#局部分量的中位数能直接组合吗) |
| <a id="忽略历史固定点至少错多少"></a>忽略历史，固定点至少错多少？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#忽略历史固定点至少错多少) |
| <a id="条件换算与自由组合分别错多少"></a>条件换算与自由组合分别错多少？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#条件换算与自由组合分别错多少) |
| <a id="单一调用换算率需要多大的端点解释"></a>单一调用换算率需要多大的端点解释？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#单一调用换算率需要多大的端点解释) |
| <a id="同plain前序接口的两进程冒烟证明了什么"></a>同plain前序接口的两进程冒烟证明了什么？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#同plain前序接口的两进程冒烟证明了什么) |
| <a id="大的常数率残差一定要求大的局部计时误差吗"></a>大的常数率残差一定要求大的局部计时误差吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#大的常数率残差一定要求大的局部计时误差吗) |
| <a id="跨sm共同的可变周期率允许多少端点误差才相容"></a>跨SM共同的可变周期率允许多少端点误差才相容？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#跨sm共同的可变周期率允许多少端点误差才相容) |
| <a id="740068首输出控制的共同时间变化能直接叫频率变化吗"></a>740068首输出控制的共同时间变化能直接叫频率变化吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#740068首输出控制的共同时间变化能直接叫频率变化吗) |
| <a id="只使用先前调用能预测下一次时间吗"></a>只使用先前调用，能预测下一次时间吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#只使用先前调用能预测下一次时间吗) |
| <a id="静态预测比值能修正首个swizzle切换吗"></a>静态预测比值能修正首个swizzle切换吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#静态预测比值能修正首个swizzle切换吗) |
| <a id="一个共同切换倍率能跨批使用吗"></a>一个共同切换倍率能跨批使用吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#一个共同切换倍率能跨批使用吗) |
| <a id="cfg_c跨k的周期接近四倍时间也能直接取四倍吗"></a>cfg_c跨K的周期接近四倍，时间也能直接取四倍吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#cfg_c跨k的周期接近四倍时间也能直接取四倍吗) |
| <a id="一个调用级换算倍率的误差下界是多少"></a>一个调用级换算倍率的误差下界是多少？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#一个调用级换算倍率的误差下界是多少) |
| <a id="同卡mma参考的cycle与时间能一起迁移吗"></a>同卡MMA参考的cycle与时间能一起迁移吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#同卡mma参考的cycle与时间能一起迁移吗) |
| <a id="前序event能给出不读取目标的局部换算吗"></a>前序event能给出不读取目标的局部换算吗？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#前序event能给出不读取目标的局部换算吗) |
| <a id="使用动态前序输入时怎样冻结局部候选"></a>使用动态前序输入时怎样冻结局部候选？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#使用动态前序输入时怎样冻结局部候选) |
| <a id="前序投影还须检查哪些静态声明"></a>前序投影还须检查哪些静态声明？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#前序投影还须检查哪些静态声明) |
| <a id="gt前后读取能给中央cycle什么时间边界"></a>GT前后读取能给中央cycle什么时间边界？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#gt前后读取能给中央cycle什么时间边界) |
| <a id="真实gt三元组怎样约束同cta首次输出"></a>真实GT三元组怎样约束同CTA首／次输出？ | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#真实gt三元组怎样约束同cta首次输出) |
| <a id="作业与复核索引"></a>作业与复核索引 | [原推导与证据](reanalysis/20261012-consolidation/R09/README.md#作业与复核索引) |
