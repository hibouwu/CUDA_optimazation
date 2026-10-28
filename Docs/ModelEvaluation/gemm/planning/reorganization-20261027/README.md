# 实验目录整理：Q14 初审

当前审阅入口为 [Q14 问题归属调整清单](Q14-question-ownership.md) 和 [113 行逐轮归属表](Q14-question-ownership.tsv)。按最新反馈，族按主题划分，族内用表格逐轮列出对象、变量、端点；COMPUTE-01–04 的 15 个目录整组留在 Q14。当前为 68 个目录建议归回 Q01–Q13、35 个保留 Q14、1 个归 Q15、8 个非实验支撑、1 个混合批次，没有悬空的待定目录。Q02 当前 213 个加本次 45 个，暂计 258 个；其主题分族与本次外移需一起确认，Docs 结论分流和双侧索引同步实施。下文是上一阶段仅针对 gemm-connection 的历史盘点记录；29 族候选和旧迁移 TSV 已被新归属清单取代，不能用来执行搬迁。

本次只新增盘点和方案文件，没有移动实验轮次、修改历史链接、改写冻结包或执行实验。审阅范围是 Q14/gemm-connection 的 71 个轮次；其余问题只盘点目录，未完成逐轮分类。没有把本轮新增文件自动提交到 Git。

## 审阅入口

- [Q14 分族说明与逐轮入口](Q14-proposal.md)：29 个候选族，按 8 个主题导航；导航不产生目录层级。
- [逐轮迁移候选 TSV](Q14-move-proposal.tsv)：71 行，包含旧路径、候选新路径、对象、变量、端点、理由、来源与深度检查。
- [族级摘要 TSV](Q14-family-proposal.tsv)：查看哪些候选仍然过宽。
- [来源正文快照](Q14-source-evidence.jsonl)：各轮 README、plan、EXECUTION 的正文和 SHA，缺失 README 的三轮使用计划与执行记录。
- [全目录轮次盘点](all-rounds-inventory.tsv)：611 个三级目录项。目录项不一定都是科学实验，也未把 a044 批次内部的 476 项直接算成 476 个实验。

71 个候选路径均为 `Q14-composition/<族>/<原轮次名>`，没有遗漏、重复目标或目标冲突，前后深度均为三层。卡名尚未逐轮核实，因此本次不改轮次名。候选均未获执行确认，也不作为自动搬迁输入。

## 分类中需要修正的地方

“只有一轮”不是合并依据。NCU 输入 sector 计数、共同服务必要下界和 CPU 模块契约，虽然都只有一轮，其对象或端点不同，应先保留。29 是保守候选数量，不是最优数量，也不是全部通过严格三项判据后的最终数量。

较宽的候选已在 TSV 的 reason 中标注：

- `full-wait-policy` 以 plain 等待策略响应为共同主端点，但 plain-only 轮次缺少其他轮次的 dual 辅助端点。
- `input-read-reply-network` 仍包含结构候选、参数标定和两套程序的端点。若要求完整端点集合完全相同，还需要拆分。
- `output-shared-service`、`gpc-copy-service` 和局部规则迁移，含同一假设的回归或下界诊断。它们不是新增实机测量；若实验族只允许同端点测量，应把关联分析另外登记，而不能伪造相同的操纵变量。
- `producer-publication-progress` 读发布进度；`request-query-age` 读时间区间；`request-age-raster` 读空间投影。这三类不因都涉及“发布”而合并。

等值源 001–004 的换卡、换序及修复保留在同族；full 位图实现失败和 overlay 修复也保留在同族。软件名字中的 runtime、program 不决定分类。

`gemm-repeatability-control` 可考虑归 Q10，`compute-handoff-contract` 可考虑归 Q06 或关联实现记录。本轮候选路径均留在 Q14，跨 Q 移动尚未决定。

Q16 的四分法暂不作为第二个执行试点：design、baseline、model 实现、trace 容量验证和配置完整时间测量并不都具有相同的实验属性。先定好 Q14 对“关联分析/实现修复”的容纳方式，再一致地处理 Q16。Q02→Q03、Q09→Q15 也需要按轮次的实际对象确认，不能按 source 或 frequency 字样决定。

## 文件身份与已有副本

大清单放在 `/data`，避免向源码仓库增加约 405 MiB 的审计输出：

- [results 文件 SHA-256 清单](/data/jianyeshi/cuda-reorganization-audit-20261027/results-sha256.tsv)
- [目录清单](/data/jianyeshi/cuda-reorganization-audit-20261027/directories.json)
- [软链接与实际目标](/data/jianyeshi/cuda-reorganization-audit-20261027/symlinks.tsv)
- [审计统计](/data/jianyeshi/cuda-reorganization-audit-20261027/audit-summary.json)

本次读取 1,143,882 个普通文件，逻辑大小 140,410,137,445 B；`du` 的实际磁盘占用约 124 GiB。每个文件计算哈希前后的大小与 mtime 均未变化。这是遍历期间的身份清单，不是原子快照，也不是可恢复的数据副本。

另登记 630 个软链接，检查时全部目标存在，其中 293 个解析到仓库外。清单不递归复制或哈希这些链接的目标；不能据此宣称 a044 外部数据已完整备份。

`/data/jianyeshi/romeo-scratch-backup-20261026` 中发现 75 个 tar 文件，共 24,100,362,240 B，抽查成员路径确实是远端恢复目录；尚未完整校验归档成员或证明其覆盖现有 results。`/data/jianyeshi/gh200-a044-20261023` 是正在使用的数据目录，不能把它与指向它的软链接算成两份独立副本。未创建新的完整数据备份；正式移动之前仍需建立并核对可恢复副本。

## 链接、冻结与可运行性

[失效内联链接清单](/data/jianyeshi/cuda-reorganization-audit-20261027/broken-inline-links.tsv)扫描了 access_rules 下 1,371 个实体 Markdown 文件，跳过 fenced code block，统计 Markdown 内联本地路径，不检查引用式链接、HTML、锚点有效性或代码中的所有路径。此口径下有 3,605 条失效引用；其中 2,675 条按旧 MOVED-PATHS 可以提出存在的目标。它们尚未逐条批准替换，也没有与旧 3,441 条报告做逐项对账。

[路径和冻结候选清单](/data/jianyeshi/cuda-reorganization-audit-20261027/path-and-freeze-review.tsv)记录 1,405 处相对深度/旧视图依赖命中，以及 1,690 个 manifest/freeze/binding 名称候选。名称只是筛查线索；尚未解析所有绑定关系或运行 manifest 校验，因此不能报告“冻结 SHA 全通过”。正式改链接前，应按绑定内容决定哪些文件不可改，未决的文件先按不可改处理。

当前结果目录首页仍要求 `flat_layout_view.py` 旧布局视图。保持三层只避免再次改变深度，不会修复已有 `parents[n]`、硬编码平铺路径或跨族相对引用。一个冻结脚本如果绑定旧路径，单纯搬目录不能同时满足“原件字节不变”和“新位置原入口直接运行且不借助兼容布局”。

验收应分别报告原件身份、非冻结入口可运行性、冻结入口的运行方式。对确实不可移植的冻结入口，可另建带来源和补丁 SHA 的运行副本；原件与旧失败保持。是否采用这种方式需要在试点执行前确定，不能通过更改原冻结包或静默软链接来宣称已修复。

## 下一步边界

先审阅 Q14 的分族粒度和关联分析的归属，再形成可执行的最终对应表。之后完成备份覆盖核对、冻结范围识别与逐入口预检，才移动已确认轮次。此次没有开始 Q14/Q16 迁移，也没有修复旧链接；不能把初审清单当作全部整理工作已完成。

## 执行状态（2026-10-28）

已按 [TOTAL-MAPPING.md](TOTAL-MAPPING.md) 执行，d332 改归 Q16；变更记录与检查见 [reorganization-20261028](../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/reorganization-20261028/README.md)。本目录中的其余方案文件（29 族草案、主题即族草案、Q02 初版）为审阅过程记录，已被总表取代。
