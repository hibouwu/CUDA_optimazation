# 计算—等待—释放局部链

16 个目录（10-17 至 10-20）：原生驻留计算链（无 TMA、单 CTA、S4/S6、K4–K32）上的 full 等待方式、empty 反馈与暴露控制，以及接口修复。

## 结论

**冻结外推获支持**
- empty 反馈 275 cycle：由 S6/K13 的 D4096 响应 2067（零控制版为 1792）得到。外推到 S4/K13 与 S6/K32，四个窗口和四个响应的误差都在 1.5% 以内；S4 延迟响应预测 3091、实测 3045 cycle（754144）。
- 正常 wait0：响应实测 2512 cycle，d0 预测 0、d95 预测 1045，两者都被反证（754575）。三种放置等价的每组暴露控制 1467/11 = 133.4 cycle 用于外推，S4/K13 与 S6/K32 预测 10084／24150.9、实测 10123／24087 cycle，误差 0.39%／0.27%（754593）。
- 现用局部规则为 q512、d95、133.4、275 与公共项 546 cycle。
- 不注入延迟时，S4/S6 对照预测 0、实测 23–39 cycle，属于零响应检验；注入延迟后，S4/S6 的差别也在 5% 内预测对了，局部反馈预测类保留；反馈位置（empty 反馈或 blocked full 返回）仍未区分。

**被反证或信息不足**
- TMA 探针上 try_wait 与 test_wait 轮询：B80 增量分别为 5411 与 3366 cycle，两个简单形式都被反证（753648）。
- 同 binary 运行时等待开关使基线改变 5.85%，桥接失败（753962）。
- 旁路观察：完成区间宽约 230–258 cycle，相位完成到返回不超过 77–98 cycle；首次 TRYWAIT 总是成功，未取得判别（758556）。
- 输入就绪与阻塞返回在现有窗口内是 773 cycle 的连续等价族。

**接口修复**
- empty 的“发布”与“完成”分开；首次等待入口不再被同步适配器覆盖。

**迁移**
- 按 blocked 入口逐次收 275 cycle 时，S4 的 plain 误差改善，但 first_L/later_L 与 stage 响应（18.3%，实测 10.5%）都变差，已停止（[gemm-connection](../Q14-composition/gemm-connection.md)）。

## 轮次

按日期排列；全部结果目录见[Q04-sync-slot/local-sync-chain](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 原生反馈连接能否外推stage与长度 | 原生incremental binary与q512／lead256固定；两旧S6/K13同批对照，S4/K13及S6/K32各D0/4096四留出，各三进程 | 754144四窗口／四响应过原5%，窗口最大误差0.330164%；保留反馈预测类，两放置等价、位置未辨识 | [冻结、原始数据、同条件比较与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-native-feedback-001/EXECUTION.md) |
| 计算完成d95能否与局部反馈共用（CPU） | 固定q512／lead256与反馈275；两个放置仅重算旧D0公共项，七既有条件与独立计算许可流交叉核对 | 七窗口／原响应保持，两放置等价；完成与末释放后移95cycle，非末释放保持。56调用、搜索／修复／GPU0，按开发连接保留 | [参数替换、逐端点与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-compute-feedback-connection-001/EXECUTION.md) |
| 原生正常wait0能否区分d0／d95完成连接 | 同批原binary／新wait1／新wait0的S6/K13各三进程；两个候选失败后未采S4/K13、S6/K32 | 754575九进程全部资格通过，增量2512cycle对冻结0／1045cycle均失败，停止两个局部连接；原始复核及异目录评分一致 | [测前冻结、构建与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-native-compute-wait-001/README.md) |
| wait0暴露的控制能否保留旧局部响应（CPU） | 固定q512／lead256／反馈275；按同一K13响应推导消费者推进、ready full、B条件完成三种放置 | 三形式均通过八旧窗口与原响应，预测相同；保留133.363636cycle复合量，位置信息不足。60调用、搜索／修复／GPU0 | [静态位置、时间线及复现](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-exposed-control-001/README.md) |
| 暴露控制预测能否外推stage与长度 | 同批三个S6/K13对照合格后，采S4/K13、S6/K32 wait0各三进程；已审核binary不重编译 | 754593全部15进程合格；留出10123／24087cycle对冻结10084／24150.909091cycle及原响应全过门，保留局部预测类，位置仍等价 | [有限清单、冻结与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-exposed-control-002/EXECUTION.md) |
| 驻留短链的full等待方式是否改变延迟暴露 | S6/K13×D0/4096×原binary／runtime阻塞／轮询，各三进程；普通arrival、驻留SMEM及阻塞暖机固定 | 753962完成18进程；新阻塞D0偏原binary5.85% cycle／6.06% ns，桥接失败，信息不足；两修复用完、停止本实现 | [原始记录、两次失败构建与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-local-wait-001/EXECUTION.md) |
| 普通arrival局部连接能否解释递增回绕响应（CPU） | 保留q512／lead256与d0／驻留d95，导出S4/S6短长链；只重核12个已有incremental进程，未新增测量 | 计算／等待／释放／再取得槽可执行；两版过长度增长、未过延迟响应，保留接口、停止向GEMM扩大 | [模型、原始输入与逐位复放](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-local-handoff-001/README.md) |
| 实际tensor行距增量是否依赖阻塞等待 | 同kernel、同分配与长hint暖机，try_wait／test_wait×base／B80，四开发条件各三进程 | 753648数值／原5%资格全过；增量5411→3366cycle，保持／全消失两极端均未命中，源与等待仍须联合辨识 | [冻结、原始数据及判定](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-wait-policy-001/EXECUTION.md) |
| 275cycle条件full返回能否迁移至G3（CPU开发） | ready0／blocked275cycle按既有阶段频率换算，其他模块固定；S6三门槛通过后才跑S4，共22调用 | S4 plain改善但first_L／later_L及stage响应恶化，逐blocked入口收费形式停止 | [迁移、调用及端点比较](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-conditional-full-return-001/EXECUTION.md) |
| 首组full旁路观察的构建资格 | S6/K4槽不复用；空闲warp观察full[0]，consumer仅采首组返回，原end后导出；原source／宏关闭／观察版三构建 | 宏关闭设备码精确复现，三版REG167／cuobjdump SHARED2176B／无spill；原3TMA／8WGMMA及关键序列保持，23位置核对通过，当轮仅CPU | [构建、位置及下一有限配对](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261019-Q04-first-full-observer-build-001/README.md) |
| full完成与blocked返回能否由现有窗口区分（CPU） | 固定旧consumer推进及全部预算，只移动就绪／返回分配；六形状／入口与旧时间线逐端点比较 | 返回、计算、释放及producer推进相同，首组就绪相差772cycle；保留连续等价族，位置未辨识；24调用、两地一致、GPU0 | [可辨识性与下一观测](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261019-Q04-ready-return-identifiability-001/README.md) |
| 驻留B首次等待延迟匹配 | 固定旧价格，S6/K13旧B0／新B0／D0三个桥接，再采B4096／D4096；冻结ΔB−ΔD为0或908cycle | 755666完成15进程，19项资格全过；B／D响应4131／3358cycle、差773，0及908两形式失败，先检查D自身复合段分配 | [计划、构建与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261019-Q04-first-entry-control-001/EXECUTION.md) |
| 计算完成与槽释放接口是否一致（CPU） | 固定已验证B/C服务及10条件，将同一full返回序列接独立计算入口，再复算empty完成和producer取得槽 | 2116端点关系一致，三项冻结wait0窗口逐位保持；14输入及两地报告保持，50调用、修复／拟合／GPU0，原价格保留 | [接口闭环与价格归属](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261019-Q04-compute-release-contract-001/README.md) |
| 首组full完成与consumer返回的旁路观察 | 原plain／新关闭／新开启，S6/K4三个条件各三进程；两次检查修复保留原binary／条件／时间门槛 | 758556九进程24资格通过；完成至返回上界98／83／77cycle，首次TRYWAIT均成功，冻结排除条件未满足；观察合格、位置未区分 | [原始结果、两次修复、时间界与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261020-Q04-first-full-observer-001/README.md) |
| empty软件发布与完成的完整图接线（CPU） | 固定事件引擎及价格；正释放延迟下消费者沿发布继续，producer按完成复用槽 | 修复g2.consumer_wait前驱误拒；23测试、104端点、四CTA跨输出及八零默认报告通过，异目录逐字一致，107调用、GPU0 | [有限轮次与交付](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261020-Q04-empty-completion-contract-001/README.md) |
| 显式首次等待入口与producer前缀的换算（CPU） | 固定程序及服务；S4/S6×入口0/8/4096，比较无钩子、零价及条件返回 | 修复5704→1672ns的零价入口覆盖；9测试、六条件、empty延迟共用及八默认通过，异目录逐字一致，82调用、GPU0 | [有限轮次与交付](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261020-Q04-wait-entry-prefix-contract-001/README.md) |
| a044 跨卡复核（mechanism-r3，780769） | 阻塞/轮询 12 进程、24 配对臂，B 行距 2304/2344 | 行距响应：阻塞 5373 cycle、轮询 3266 cycle（旧卡 5411/3366）；“全部保留 5429”与“降为 0”两候选都不匹配，信息不足 | [Q04 报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/q04-returned-stage-report.md) |
| a044 保序修复（785518–785575） | exposed-control 15、local-wait 18、native-compute-wait 9、native-feedback 18、first-entry-control 15、first-full-observer 12 次，原顺序与环境，原评分器 | 7 个根中 6 个判定不变。first-full-observer/repair-001：旧卡停在描述符一致性门，a044 通过该门，但原轮次只采了 2 个 context，变为 awaiting_processes；repair-002 判定不变 | [修复报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/mechanism-repair-r2-report.md#q02q04q05-原评分器重评) |
