# full 就绪、首次等待入口与返回

6 个目录（10-18 至 10-20）：区分输入就绪与 consumer 阻塞返回，观察首组 full 完成到返回，并修复首次等待入口的接线。由 local-sync-chain 移入。

## 结论

- 驻留 B 与 D 的首次等待延迟响应为 4131／3358 cycle、差 773，0 与 908 两个形式失败（755666）。
- 输入就绪与阻塞返回在现有窗口内是 773 cycle 的连续等价族，位置未辨识。
- 旁路观察：相位完成到 consumer 返回不超过 77–98 cycle，首次 TRYWAIT 总是成功，未取得判别（758556）。
- 按 blocked 入口逐次收 275 cycle 迁移到 G3：S4 plain 误差 9.6%→3.3%，但 first_L/later_L 过冲、stage 响应 18.3%（实测 10.5%），已停止。
- 首次等待入口不再被同步适配器覆盖（接口修复）。

## 轮次

按日期排列；全部结果目录见[Q04-sync-slot/full-ready-return](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-ready-return/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 275cycle条件full返回能否迁移至G3（CPU开发） | ready0／blocked275cycle按既有阶段频率换算，其他模块固定；S6三门槛通过后才跑S4，共22调用 | S4 plain改善但first_L／later_L及stage响应恶化，逐blocked入口收费形式停止 | [迁移、调用及端点比较](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-ready-return/20261018-Q04-conditional-full-return-001/EXECUTION.md) |
| 首组full旁路观察的构建资格 | S6/K4槽不复用；空闲warp观察full[0]，consumer仅采首组返回，原end后导出；原source／宏关闭／观察版三构建 | 宏关闭设备码精确复现，三版REG167／cuobjdump SHARED2176B／无spill；原3TMA／8WGMMA及关键序列保持，23位置核对通过，当轮仅CPU | [构建、位置及下一有限配对](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-ready-return/20261019-Q04-first-full-observer-build-001/README.md) |
| full完成与blocked返回能否由现有窗口区分（CPU） | 固定旧consumer推进及全部预算，只移动就绪／返回分配；六形状／入口与旧时间线逐端点比较 | 返回、计算、释放及producer推进相同，首组就绪相差772cycle；保留连续等价族，位置未辨识；24调用、两地一致、GPU0 | [可辨识性与下一观测](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-ready-return/20261019-Q04-ready-return-identifiability-001/README.md) |
| 驻留B首次等待延迟匹配 | 固定旧价格，S6/K13旧B0／新B0／D0三个桥接，再采B4096／D4096；冻结ΔB−ΔD为0或908cycle | 755666完成15进程，19项资格全过；B／D响应4131／3358cycle、差773，0及908两形式失败，先检查D自身复合段分配 | [计划、构建与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-ready-return/20261019-Q04-first-entry-control-001/EXECUTION.md) |
| 首组full完成与consumer返回的旁路观察 | 原plain／新关闭／新开启，S6/K4三个条件各三进程；两次检查修复保留原binary／条件／时间门槛 | 758556九进程24资格通过；完成至返回上界98／83／77cycle，首次TRYWAIT均成功，冻结排除条件未满足；观察合格、位置未区分 | [原始结果、两次修复、时间界与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-ready-return/20261020-Q04-first-full-observer-001/README.md) |
| 显式首次等待入口与producer前缀的换算（CPU） | 固定程序及服务；S4/S6×入口0/8/4096，比较无钩子、零价及条件返回 | 修复5704→1672ns的零价入口覆盖；9测试、六条件、empty延迟共用及八默认通过，异目录逐字一致，82调用、GPU0 | [有限轮次与交付](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-ready-return/20261020-Q04-wait-entry-prefix-contract-001/README.md) |
