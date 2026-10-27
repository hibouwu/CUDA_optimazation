# 输入／输出联合窗口

8 个目录（10-18 至 10-21）：同一 kernel 的输入单独、输出单独、串行、并发四种组织，输出使用 cfg_b 原 Epilogue 工厂。

## 结论

- 首轮四个窗口为 6292/7154/8809/7890 cycle，计时内含 fragment 准备与两阶段 join（751927）。
- “准备完成后”版本的准备写使用 `lane×128+i`，同 warp 的 32 个 word 落在同一 bank，探针自身就有 32 路冲突，窗口约 1.5–1.7 万 cycle（752311）。改为 `i×128+lane` 后，四个窗口为 2981/3839/5415/4553 cycle（752322）。两版正式窗口另有 17 处指令编码差异，原记录未把新旧差值单独归因于 bank 排列。
- 并发比较大单窗多 704–727 cycle，比串行少 851–909 cycle；max 与 serial 两个简单形式都不在 5% 内。
- 输出相位观察的桥接偏差 5.5%–6.4%（753514）；输入组数对照因暖机 CV 3.8% 停在 7/15 个进程（756133）。
- 同批条件化检验774013完成12开发进程；input1/seed3暖机与input4周期稳定性未过，未冻结或采并发单组留出，两个联合形式仍未判定。
- 重叠收益存在，但幅度是该程序的联合性质，不拆成逐指令价格；cfg_b 输出仍用 V09 复合服务。见 [SMEM-SERVICE](../../../model/mechanism/hardware_rules/SMEM-SERVICE.md)。

## 轮次

按日期排列；全部结果目录见[Q07-smem/input-output-joint](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 输出状态转换能否独立执行并保留组合行为（CPU） | 提取既有八子块／双源槽转换，四组织含132CTA；固定全部价格与边界 | 四组全部字段及observer顺序保持，七个独立输出回放一致；跨输出前驱元数据修正，时刻不变。60调用、一修复、拟合／GPU0，原硬件判定保持 | [模块、回归与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261018-Q07-local-output-chain-001/EXECUTION.md) |
| 输入／输出联合窗口的哪种组数干预能区分候选（CPU，关联问题7） | 旧752322数据不变，额外窗取跨seed共用中位；比较固定J与J×输入组数/4，沿原双单位逐seed5%带 | 1组可分、2组重叠，选择5条件15进程的最小后续清单；48预测和8代数界两地一致，当前构建／GPU0 | [两形式、共用参数与可分条件](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261019-Q07-joint-count-design-001/README.md) |
| 输入组数能否区分联合额外窗口（关联问题7） | 固定J与J×groups/4；只改输入4→1组，其他链与资源固定 | 756133的7进程数值及资源保持；input1/seed22暖机CV3.827328%超原2%，停止、无留出、两形式未判定 | [计划、原始失败与复核](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261019-Q07-joint-count-001/README.md) |
| 同批条件化联合额外窗口能否外推输入组数 | 已审查binary不改；四组对照导出共同J，固定两形式再预测一组并发；4开发+1留出、各三进程 | 774013完成12开发进程、数值983424值通过；input1/seed3暖机失败，input4周期范围6.591%超门；未冻结、未采留出，信息不足，本轮收口 | [计划、原始数据与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261021-Q07-joint-count-same-batch-001/README.md) |
| a044 跨卡复核（mechanism-r3，780769） | 五轮共 57 次调用，原见证检查与评分 | prepared-joint 的 `max` 通过 6/6，串行 4/6；bank-layout 输入窗口三进程范围 5.23%（cycle）/6.38%（ns），资格失败；joint-count 7 次计时通过，但开发集合仍不全；同批 joint-count 的 input1 ns 范围 6.45%，未通过 | [内存服务报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/mechanism-storage-stage-report.md) |
