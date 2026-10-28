# GEMM 源工作集与分配

4 个目录（10-15、10-23）：G3 原程序分配与 scratch 准备查询、真实输入路径的 L2 读取计数、逐 CTA 等值私有源副本及其 CPU 接回。由 Q14 移入。

## 结论

- 逐 CTA 等值私有源：源覆盖 34→396 MiB，四个 L 的副本／原源周期比 2.38–4.70，支持采前的大源工作集响应，行距增长类别未分（780343）；不据此声称缓存容量已单独识别。
- 接回同一输入服务：68 调用，私有源 plain 误差 −0.77%／−0.03%，但两段 L 及原源行距仍失败，不采用该固定服务。
- 输入计数：原两次 NCU 未采集到计数；a044 补测 L2 读 A16/base 为 1.39–1.50（3 pass）与 1.44–1.50（单 pass），DRAM 读比 0.94–0.95，A16 多出的读取由 L2 供给，按原判据仍不定（见补充说明）。

## 轮次

按日期排列；全部结果目录见[Q03-source-service/gemm-source-workset](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/gemm-source-workset/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| G3 原程序分配与 scratch 准备 | 746714 原生构建四个查询版本，只在setup输出中增加host指针。S6 plain/dual、S4 plain/dual的全部设备指令分别1960、2216、1944、2224条，与V09冻结SASS逐条相同；没有新增设备操作。各版本运行三次setup查询，共12份。 | 见原记录 | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/gemm-source-workset/20261015-g3-allocation-query-001/README.md) |

## 补充说明

[本地输入／计算共同服务](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261023-Q14-local-joint-service-001/README.md)与[受理后查询缓存](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261023-Q14-port-ordered-cache-001/README.md)各完成34调用，固定版本均未解释A16持续full未完成，后续条件未求解。[真实输入路径计数定位](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/gemm-source-workset/20261023-Q14-input-counter-localization-001/README.md)四plain进程合格；两次NCU因参数／kernel过滤错误未采集计数，信息不足，未改变两种工作量解释或独立验收结论。

[逐CTA等值源副本](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/gemm-source-workset/20261023-Q14-private-source-001/README.md)：保持请求形状、逻辑输入528MiB及计算值，源覆盖34→396MiB。780264的24次初始化失败保留；780301修正host绑定后，780343有24/24有效，四个L副本／原源周期比2.3842–4.6979，支持采前大源工作集响应，行距增长类别未分。累计GPU25秒，模型／拟合0。[CPU接回](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/gemm-source-workset/20261023-Q14-private-source-transfer-001/plan.json)只接实际地址及暖机，不调价格。

[同一输入服务的四条件回放](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/gemm-source-workset/20261023-Q14-private-source-transfer-001/README.md)完成68调用：实际源副本与暖机接线准确，原源预测逐位保持，私有源plain误差−0.765%／−0.033%；两L周期／纳秒及原源行距仍失败，故不采用该固定服务。引擎调用68、拟合／GPU0；原12及独立C2/C3保持。

输入计数定位按原计划在 a044 补测（作业 785361）：修正 kernel 过滤，取第 17 次 GEMM，3 个 trial 的 base/A16 配对。plain 的 A16/base 为 1.339–1.389。L2 读 sector 的 A16/base 比值：两个合格 trial 为 1.391 和 1.498；trial 0 因 A16 第三次应用重放时暖机 CV>2% 不合格，比值为 1.512。按原判据为信息不足。三次观测都与小幅读工作带（0.95–1.20）不相容。增加的读取由 L2 供给：A16 的 DRAM 读少 6–7%，L2 命中率约 92.8%。base 的 L2 读在三个 trial 间相差 8.5%，可能受 3-pass replay 的缓存状态影响，尚未单独检验。见[补测报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-pending-20261026-report.md)。

10-27 单 pass 补测（作业 785959、785974）：每个 profile 只采一个指标，共 5 个 trial，其中 4 个合格。

- L2 读 A16/base 为 1.440–1.498，仍高于大幅读工作带的上界 1.40，按原判据为不定。

- DRAM 读比值为 0.94–0.95：A16 多出来的读取由 L2 供给，额外量稳定在 +44%–50%。

- 4 个合格 trial 中，base 的 L2 读量分散为 4.1%（3 pass 时为 8.5%），多 pass 重放只能解释一部分。

见[结果](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-followup-20261027/RESULTS.md)。
