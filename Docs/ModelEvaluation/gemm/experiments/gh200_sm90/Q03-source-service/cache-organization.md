# 缓存组织与状态回放

9 个目录（10-19 至 10-23）：在 D 链与 G3 回放中替换缓存组织、查询时点、填充插入与替换、源基址映射等规则。由 Q02 source-organization、Q14 与 HBM-01 移入。

## 结论

- 只在 home 保留行：并集约 17.33 MiB／域、无淘汰，0 响应未过旧 ±5 个百分点，停止。
- D 链回放：输出写入状态、受理后发起数据分支、实际分配基址、填充完成插入四个固定形式，八个绝对端点均失败，停止；其中实际基址改变了 B 的 home 归属，但 528 个 CTA 结束时刻逐值保持。
- 完成时 LRU 更新只执行了 FIFO／132 源一组（13 调用），首组两个端点失败后停止，其余源数、排序、S4 与完整 GEMM 未执行。
- G3 回放：显式分区组相联缓存的两段 L 增长为 0，464 个后续组仍提前就绪，固定版本停止；受理后查询缓存（34 调用）未解释 A16 的持续 full 未完成，后续条件未求解。
- 现用两域缓存、LRU、pending fill 仍为假设，见 [L2-SERVICE](../../../model/mechanism/hardware_rules/L2-SERVICE.md)；与之反向的“数据就绪约束 GPC 准入”留在 [source-organization](../Q02-tma-request/source-organization.md)。

## 轮次

按日期排列；全部结果目录见[Q03-source-service/cache-organization](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 取消接收副本是否兼容旧近期源响应（CPU） | 只在home保留行，固定完全相联30MiB／域、映射／初始化／价格；先核全访问并集及两次序就绪 | 并集约17.33MiB／域，无淘汰，0响应未过旧±5个百分点，停止固定形式；12模块流、两地一致、GPU0。758367零GPU只读核NCU仍为管理员专用 | [容量界、旧约束与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261019-Q03-home-only-screen-001/README.md) |
| D输出写入是否遗漏共享源状态（CPU） | 固定66／132源、两排序及全部旧价格，只接每调用8.25MiB输出清写与CTA计时后导出；44调用 | 八个绝对端点失败，FIFO66源多1744.975ns且跨度更小；132源与CTA优先稳态保持，停止固定接线，不扩大S4或完整组合 | [计划、评分与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261020-Q02-output-state-replay-001/README.md) |
| 数据分支是否应在GPC受理后开始（CPU） | 固定所有价格及服务范围，只将缓存查找／新miss提交移到既有service_start；66／132源、两排序及零服务等价检查，46调用 | 零服务精确保持、5148查找按时间推进；八个绝对端点失败，响应带保持，停止固定连接，不接S4／完整组合 | [计划、评分与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261020-Q02-admitted-data-replay-001/README.md) |
| 实际分配基址是否改变现用home映射（CPU） | 固定服务／映射规则，只将旧A0/B152MiB换成保存描述符的实际基址；66／132源×两排序，44调用 | 48组编码对应通过；B有106496次home变化，但528个CTA结束逐值保持，八绝对端点仍失败，保留实际输入及反例，不扩大 | [输入、回放与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261020-Q02-source-base-replay-001/README.md) |
| 填充完成后的缓存插入能否修复共享推进 | 固定实际基址、容量、全部价格及并发组织；未完成填充单列，ready时插入及替换；46次CPU调用 | 零延迟单CTA精确保持；132源暖机后全命中，源计数响应退化为0，八绝对端点失败；停止固定形式，不接S4或完整模型 | [规则、保存预测与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261020-Q02-fill-completion-replay-001/README.md) |
| 保留在途占位后，完成时LRU更新能否修复推进 | 容量及发出占位保持，只在fill完成时将尚存条目移到MRU；FIFO／132源13调用 | 零延迟等价保持；132CTA和5148请求逐值不变，首项包络与跨度失败即停；未扩大，不改完整模型 | [规则、反例与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261020-Q02-fill-recency-replay-001/README.md) |
| 公开分区折叠的转写修复 | 计划与原始来源、改动源码、评分、同条件比较。重放：`python run.py --output /tmp/cfg-b-public-fold-replay`；system/tensor引擎复用前轮快照。 | 读取价固定为纯TMA标定的2.602903143cycle/片段，应答仍为8cycle/16KiB。56次引擎调用、CPU153.906秒、拟合／GPU0。unique/global1 Q4最后释放中位数为29911.154／25770.944ns；G3基础／A16 plain为46.849919／46.869613µ | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261023-Q14-public-fold-correction-001/README.md) |

## 补充说明

- 实际每调用清写和导出的8.25MiB输出未接入旧D源回放；沿用现有写分配规则补入后，132源稳态保持、66源的FIFO预测过慢，八个绝对端点均失败，停止该固定接线，见[CPU回放](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261020-Q02-output-state-replay-001/README.md)。

- 用保存描述符的实际A/B基址替换旧假设后，B的106496/219648次行访问home归属改变，但四组528个CTA结束时刻逐值保持，仍未修复残差，见[输入核对](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261020-Q02-source-base-replay-001/README.md)。

- 未完成填充不占缓存容量、完成时才插入的固定形式，使132源在暖机后也全命中，66／132源响应变为0；原冻结响应及八个绝对端点均失败，停止该形式，见[填充完成回放](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261020-Q02-fill-completion-replay-001/README.md)。

- 保留发出时占位、只为完成时尚存条目增加MRU更新，FIFO／132源仍全miss，132个CTA及5148请求逐值保持；13调用即在首项停止，不继续扩展LRU更新，见[完成时替换顺序](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261020-Q02-fill-recency-replay-001/README.md)。

[有限同址sector合并](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-input-network/20261022-Q14-bounded-sector-merge-001/README.md)：每GPC最多4份在途订阅替换无限整atom合并，34调用、CPU781.065秒、GPU/拟合0；两L增长仍0，本固定版本停止。[显式分区与组相联缓存](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261022-Q14-banked-source-cache-001/README.md)也已完成三项检查、34调用、CPU808.098秒，两L增长0，固定版本停止；464个后续组仍提前就绪。价格和其他模块保持，未扩大原12或独立验收。

[本地输入／计算共同服务](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261023-Q14-local-joint-service-001/README.md)与[受理后查询缓存](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/cache-organization/20261023-Q14-port-ordered-cache-001/README.md)各完成34调用，固定版本均未解释A16持续full未完成，后续条件未求解。[真实输入路径计数定位](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/gemm-source-workset/20261023-Q14-input-counter-localization-001/README.md)四plain进程合格；两次NCU因参数／kernel过滤错误未采集计数，信息不足，未改变两种工作量解释或独立验收结论。
