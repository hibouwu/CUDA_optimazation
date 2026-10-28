# full 等待方式

6 个目录（10-18 至 10-23）：在 TMA 探针、驻留短链与真实 cfg_b 上比较阻塞、轮询（try_wait／test_wait）等 full 等待方式。由 local-sync-chain 与 Q14 移入。

## 结论

- TMA 探针上 try_wait 与 test_wait 轮询：B80 增量分别为 5411 与 3366 cycle，两个简单形式都被反证（753648）。
- 驻留短链的同 binary 运行时等待开关使基线改变 5.85%，桥接失败（753962）。
- 真实 cfg_b：分别编译的阻塞／轮询形式因输出算术改变停止；同 binary 作业 753707 排队时取消；763812 只完成 11 个合格 plain 进程，信息不足。
- cfg_b 行距与等待方式（780627）：两输出均排除 Q04 比例迁移，policy 无关形式在第二输出获开发支持，第一输出信息不足。

## 轮次

按日期排列；全部结果目录见[Q04-sync-slot/full-wait-policy](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-wait-policy/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 驻留短链的full等待方式是否改变延迟暴露 | S6/K13×D0/4096×原binary／runtime阻塞／轮询，各三进程；普通arrival、驻留SMEM及阻塞暖机固定 | 753962完成18进程；新阻塞D0偏原binary5.85% cycle／6.06% ns，桥接失败，信息不足；两修复用完、停止本实现 | [原始记录、两次失败构建与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-wait-policy/20261018-Q04-local-wait-001/EXECUTION.md) |
| 实际tensor行距增量是否依赖阻塞等待 | 同kernel、同分配与长hint暖机，try_wait／test_wait×base／B80，四开发条件各三进程 | 753648数值／原5%资格全过；增量5411→3366cycle，保持／全消失两极端均未命中，源与等待仍须联合辨识 | [冻结、原始数据及判定](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-wait-policy/20261018-Q04-wait-policy-001/EXECUTION.md) |
| 原始cfg_b的stage响应是否依赖full等待方式 | S6/S4×原阻塞／test_wait轮询，四开发条件；不加活动槽、重复循环或内部打点，固定其余路径 | 753686/753688 CPU构建完成；阻塞机器字精确，轮询另改输出算术／排程，停止分别编译形式，GPU0 | [源码、构建及控制反例](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-wait-policy/20261018-Q14-full-wait-policy-001/EXECUTION.md) |
| 同一binary的full等待能否解释原stage响应 | S6/S4×原始／runtime阻塞／runtime轮询，六开发条件各plain/dual三进程；阻塞暖机保持 | 753702 CPU构建通过，frame/实际指令及同binary控制已核；753707在排队阶段取消、GPU运行时间0，905输入绑定保留；按新顺序延后至局部等待／释放连接检验后再决定 | [有限清单与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-wait-policy/20261018-Q14-runtime-full-wait-001/EXECUTION.md) |
| 同binary full等待是否改变真实cfg_b的plain stage响应 | 复用753702构建，S6/S4×原始／runtime阻塞／轮询，六plain条件各三进程；十次阻塞暖机、无内部打点 | 763812共12次尝试，11合格，p23暖机CV2.050824%超门槛后停止，余6未测；桥接／响应三重复不齐，信息不足，无重试／改价 | [计划、原始输出与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-wait-policy/20261020-Q14-full-wait-plain-002/README.md) |

## 补充说明

- [cfg_b行距与full等待方式](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/full-wait-policy/20261023-Q14-stride-wait-policy-001/README.md)：复用同binary的24进程，22有效；两输出各两共同trial，均排除Q04比例迁移，policy无关形式第二输出获开发支持、第一输出信息不足。780627分配1GPU、15秒，无构建／拟合或补跑。
