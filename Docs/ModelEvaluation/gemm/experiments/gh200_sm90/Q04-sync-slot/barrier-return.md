# mbarrier 等待返回

3 轮（10-15）开发测量。

## 结论

- CUTLASS 的已完成检查为 78–81 cycle，阻塞返回上界为 80–85 cycle，不支持大额唤醒成本。
- 事务登记只增加 0–3 cycle，解释不了约 220 cycle 的旧端点。
- producer 直接 wait 的阻塞上界为 121 cycle，已完成检查加登记为 174 cycle，形不成统一价格，单独的 barrier 扫描到此停止。
- 条件返回接口见 [SYNC-01](SYNC-01-replay.md)。

## 轮次

按日期排列；全部结果目录见[Q04-sync-slot/barrier-return](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/barrier-return/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 等待返回是否有大额唤醒成本（746687，开发） | 固定单CTA与128个等待线程，改变信号相对起点及1/128次arrival或32 KiB事务完成，比较按需等待与主动测试。 | CUTLASS已完成检查为78–81 cycle、阻塞返回上界80–85 cycle，未支持大额唤醒成本，下一轮补事务登记。 | [记录](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/barrier-return/20261015-barrier-return-development-001/README.md)、[逐线程区间分析](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/barrier-return/20261015-barrier-return-development-001/analysis.json) |
| 事务登记能否解释约220 cycle端点（746693，开发） | 沿用前置 try_wait 后按需 wait，开关 arrive_and_expect_tx(32768)，并改变arrival与等待者数。 | 登记只增加0–3 cycle，未解释旧端点；源码复核发现producer直接wait，下一轮改为实际调用方式对照。 | [记录及调用路径复核](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/barrier-return/20261015-barrier-return-development-002/README.md)、[分析](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/barrier-return/20261015-barrier-return-development-002/analysis.json) |
| producer与consumer能否共用返回价格（746697，开发） | 固定128次必要arrival、一个等待者，在同一二进制内比较按需wait、主动test_wait、直接wait与事务登记开关。 | 直接wait阻塞上界121 cycle、已完成检查加登记174 cycle，未形成统一价格，停止单独barrier扫描。 | [记录与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/barrier-return/20261015-barrier-return-development-003/README.md)、[分析](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/barrier-return/20261015-barrier-return-development-003/analysis.json) |
