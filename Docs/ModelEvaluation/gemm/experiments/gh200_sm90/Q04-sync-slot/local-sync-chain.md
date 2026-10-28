# 计算—等待—释放局部链

8 个目录（10-17 至 10-20）：原生驻留计算链（无 TMA、单 CTA、S4/S6、K4–K32）上的 empty 反馈与暴露控制，以及接口修复。等待方式见 [full-wait-policy](full-wait-policy.md)，就绪与返回见 [full-ready-return](full-ready-return.md)。

## 结论

**冻结外推获支持**
- empty 反馈 275 cycle：由 S6/K13 的 D4096 响应 2067（零控制版为 1792）得到。外推到 S4/K13 与 S6/K32，四个窗口和四个响应的误差都在 1.5% 以内；S4 延迟响应预测 3091、实测 3045 cycle（754144）。
- 正常 wait0：响应实测 2512 cycle，d0 预测 0、d95 预测 1045，两者都被反证（754575）。三种放置等价的每组暴露控制 1467/11 = 133.4 cycle 用于外推，S4/K13 与 S6/K32 预测 10084／24150.9、实测 10123／24087 cycle，误差 0.39%／0.27%（754593）。
- 现用局部规则为 q512、d95、133.4、275 与公共项 546 cycle。
- 不注入延迟时，S4/S6 对照预测 0、实测 23–39 cycle，属于零响应检验；注入延迟后，S4/S6 的差别也在 5% 内预测对了，局部反馈预测类保留；反馈位置（empty 反馈或 blocked full 返回）仍未区分。

**接口修复**
- empty 的“发布”与“完成”分开。首次等待入口的修复见 [full-ready-return](full-ready-return.md)。

## 轮次

按日期排列；全部结果目录见[Q04-sync-slot/local-sync-chain](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 原生反馈连接能否外推stage与长度 | 原生incremental binary与q512／lead256固定；两旧S6/K13同批对照，S4/K13及S6/K32各D0/4096四留出，各三进程 | 754144四窗口／四响应过原5%，窗口最大误差0.330164%；保留反馈预测类，两放置等价、位置未辨识 | [冻结、原始数据、同条件比较与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-native-feedback-001/EXECUTION.md) |
| 计算完成d95能否与局部反馈共用（CPU） | 固定q512／lead256与反馈275；两个放置仅重算旧D0公共项，七既有条件与独立计算许可流交叉核对 | 七窗口／原响应保持，两放置等价；完成与末释放后移95cycle，非末释放保持。56调用、搜索／修复／GPU0，按开发连接保留 | [参数替换、逐端点与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-compute-feedback-connection-001/EXECUTION.md) |
| 原生正常wait0能否区分d0／d95完成连接 | 同批原binary／新wait1／新wait0的S6/K13各三进程；两个候选失败后未采S4/K13、S6/K32 | 754575九进程全部资格通过，增量2512cycle对冻结0／1045cycle均失败，停止两个局部连接；原始复核及异目录评分一致 | [测前冻结、构建与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-native-compute-wait-001/README.md) |
| wait0暴露的控制能否保留旧局部响应（CPU） | 固定q512／lead256／反馈275；按同一K13响应推导消费者推进、ready full、B条件完成三种放置 | 三形式均通过八旧窗口与原响应，预测相同；保留133.363636cycle复合量，位置信息不足。60调用、搜索／修复／GPU0 | [静态位置、时间线及复现](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-exposed-control-001/README.md) |
| 暴露控制预测能否外推stage与长度 | 同批三个S6/K13对照合格后，采S4/K13、S6/K32 wait0各三进程；已审核binary不重编译 | 754593全部15进程合格；留出10123／24087cycle对冻结10084／24150.909091cycle及原响应全过门，保留局部预测类，位置仍等价 | [有限清单、冻结与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-exposed-control-002/EXECUTION.md) |
| 普通arrival局部连接能否解释递增回绕响应（CPU） | 保留q512／lead256与d0／驻留d95，导出S4/S6短长链；只重核12个已有incremental进程，未新增测量 | 计算／等待／释放／再取得槽可执行；两版过长度增长、未过延迟响应，保留接口、停止向GEMM扩大 | [模型、原始输入与逐位复放](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-local-handoff-001/README.md) |
| 计算完成与槽释放接口是否一致（CPU） | 固定已验证B/C服务及10条件，将同一full返回序列接独立计算入口，再复算empty完成和producer取得槽 | 2116端点关系一致，三项冻结wait0窗口逐位保持；14输入及两地报告保持，50调用、修复／拟合／GPU0，原价格保留 | [接口闭环与价格归属](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261019-Q04-compute-release-contract-001/README.md) |
| empty软件发布与完成的完整图接线（CPU） | 固定事件引擎及价格；正释放延迟下消费者沿发布继续，producer按完成复用槽 | 修复g2.consumer_wait前驱误拒；23测试、104端点、四CTA跨输出及八零默认报告通过，异目录逐字一致，107调用、GPU0 | [有限轮次与交付](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261020-Q04-empty-completion-contract-001/README.md) |
| a044 跨卡复核（mechanism-r3，780769） | 阻塞/轮询 12 进程、24 配对臂，B 行距 2304/2344 | 行距响应：阻塞 5373 cycle、轮询 3266 cycle（旧卡 5411/3366）；“全部保留 5429”与“降为 0”两候选都不匹配，信息不足 | [Q04 报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/q04-returned-stage-report.md) |
| a044 保序修复（785518–785575） | exposed-control 15、local-wait 18、native-compute-wait 9、native-feedback 18、first-entry-control 15、first-full-observer 12 次，原顺序与环境，原评分器 | 7 个根中 6 个判定不变。first-full-observer/repair-001：旧卡停在描述符一致性门，a044 通过该门，但原轮次只采了 2 个 context，变为 awaiting_processes；repair-002 判定不变 | [修复报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/mechanism-repair-r2-report.md#q02q04q05-原评分器重评) |
