# 请求完成状态与时刻的观测

16 个目录（10-18、10-22、10-23）：用单点 barrier、旁路 warp、逐 Ktile full 位图、发布计数与发布后年龄，观察请求在消费者查询时是否已经完成。这些轮次读的是请求完成状态，没有单独测 full 完成后 wait 的返回耗时（后者见 [Q04](../Q04-sync-slot/full-ready-return.md)）。

## 结论

- 探针上的单点观察均未过资格：ready-snapshot 的 CTA→SM 映射轮换、ready-fixed-ports 的包络范围 14%、full-side-warp 的稳定/扰动门。full-side-warp 取得 125–128 cycle 的完成可见区间，没有转成价格。
- G3 S6：A16 后续 full 未完成率 91.8%，基础为 4.1%，当前全 ready 预测被反证（774830）；第一版实现没有观察语义。
- g43 的局部定位（775049）：六个输出条件中三个合格，其 306 次 full 未完成均已发布，且后续三组也已发布。“六条件均由发布前未完成主导”被反证；发布后主导只在这三个条件获得支持，六条件整体主张尚未验证。
- 发布后年龄：五个合格输出的 530 次未完成，年龄下界均超过旧完成上界，“保留完成规则、仅查询提前”的形式被排除。
- 年龄的空间分布与逻辑 CTA 组绑定；raster 转置的实测中原 raster 的六项空间形状控制全部失败，按计划停止，along_m 未测，源组与逻辑组尚未区分。
- a044 复核中 full-entry-mask-001/002 因资格降为信息不足（[修复报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/mechanism-repair-r2-report.md)）。

## 轮次

按日期排列；全部结果目录见[Q02-tma-request/completion-observation](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 同源两遍的单点barrier完成状态 | 同binary/plain/g8/g40；保留原wait、phase及三请求 | 753444数值三臂零错；CTA→SM轮换，映射资格失败，正式0，收口 | [执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261018-Q02-ready-snapshot-001/EXECUTION.md) |
| 固定每SM源/起步的单点完成 | 保持32CTA与三观察条件；按预声明SM/source_rank配对 | 753464身份/数值通过；plain范围14.063657%、逐SM差17.903066%，原5%失败，收口 | [执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261018-Q02-ready-fixed-ports-001/EXECUTION.md) |
| 同代次旁路warp完成区间 | plain/g8/g40同binary三条件，原consumer wait保持 | 183/192取得125–128cycle区间；原稳定/扰动资格失败，信息不足并停止 | [执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261018-Q02-full-side-warp-001/EXECUTION.md) |
| 后续full未完成能否解释持续行距响应 | 原R及同binary关闭／开启，S6 G3/A16八条件各三进程；原预测及5%门槛固定 | 实现1的24进程无观察语义，信息不足；修复2的774830完成24有效进程，桥接／扰动通过，A16未完成率91.823%对基础4.099%，当前全ready预测被反证 | [失败实现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261022-Q14-full-entry-mask-001/README.md)、[修复与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261022-Q14-full-entry-mask-002/README.md) |
| g43的full未完成位于producer发布前还是发布后 | 固定S6 G3/A16与服务；原R、已合格F及同binary发布关／开，8条件24进程；只采g43前后发布数 | 774863完成18有效进程，时间／行距保持通过；N0基础一输出17/132超10%，未采N1，信息不足；稀疏g43形式停止 | [冻结计划、构建与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261022-Q14-producer-publication-001/README.md) |
| 保留逐Ktile full观察后定位g43发布位置 | R／F／同binary发布关开，固定G3/A16、S6，8条件24进程；不改模型价 | 775011完成24有效进程，F/N0六前置条件通过；全局原子N1的g43仅41/792未完成、六位置资格失败，形式停止 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261022-Q14-producer-publication-002/README.md) |
| 用CTA本地发布计数定位g43 | 将全局原子换成本地SMEM原子，查询／程序和原门槛保持；8条件24进程 | 775039的N0六进程misaligned；修复775049有23/24有效，三合格输出的306次未完成均已发布且领先三组，发布前全域主导类被反证；其余资格失败保留 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261022-Q14-producer-publication-003/README.md) |
| 2026-10-22 发布与查询时间线CPU核对 | 同一g43的发布进度、查询与full完成位置 | 18调用逐位复现旧预测，全部CTA的47/111进度吻合；A16 full提前639–881cycle仍与观测冲突 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261022-Q14-publication-time-alignment-001/README.md) |
| 2026-10-22 同组B1后至查询年龄 | 保留完成规则与较早查询，或发布后完成延迟 | 24有效进程；五合格输出530次未完成的年龄下界均超旧完成上界，保留完成／仅查询提前形式被排除 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261022-Q14-published-request-age-001/README.md) |
| 发布后年龄更随源坐标还是SM位置变化 | 复用775061合格输出，按输出做留一进程分组比较 | 1452项坐标吻合；A16的A组留一进程误差改善8%/18%，但与逻辑CTA组绑定；模型／拟合／GPU0 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261022-Q14-completion-spatial-screen-001/README.md) |
| Raster能否区分A源组与逻辑CTA组 | 264输出与每CTA两输出保持，冻结年龄形状投影与重复性区间 | 每输出66CTA，两带均不重叠，空间差RMS234/240cycle；模型／拟合／GPU0 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261022-Q14-raster-contrast-prediction-001/README.md) |
| Raster转置后的年龄形状跟随 | 同binary heuristic/along_m、年龄关/开，两布局各三进程 | 775150完成12有效进程，时间检查全过、最大扰动2.1304%；原raster六项空间形状控制全失败，按计划停止，along_m未测，形状迁移解释未区分 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261022-Q14-raster-age-discrimination-001/README.md) |

## 补充说明

- [请求返回与查询位置对齐](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261023-Q14-query-return-connection-001/README.md)：复用g43时钟区间完成18调用，查询单列为程序端点；[g43分请求完成集合](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261023-Q14-operand-completion-001/README.md)完成18进程、17有效，仅一个trial／输出共同合格，信息不足收口。

- [full完成边界筛查](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/completion-observation/20261023-Q14-full-boundary-screen-001/README.md)：六份原始记录、1320条合格观察；FIRST参考上界705条、下界69条违例。逻辑issue与B1后戳分开，跨试次缺失证据保留；引擎／拟合／GPU0。
