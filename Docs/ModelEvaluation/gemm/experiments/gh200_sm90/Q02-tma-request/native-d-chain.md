# 原生 TMA D 链（单 CTA）

24 个目录（10-18 至 10-21）：把真实三请求 TMA（A 16 KiB、B0/B1 各 8 KiB、SW128）接入驻留计算链，研究搬运→就绪→full 返回的局部连接。

## 结论

**已测事实**
- K4→K13 的长度增长：D 为 4585 cycle，驻留 B 为 4584 cycle。两者相同，保留“真实输入被隐藏、另加共同固定项”的开发形式。D−B 窗口差为 883／884 cycle，其中输入启动、事务等待返回与程序固定项尚未分开（754824）。
- 冻结外推：S4/K13 与 S6/K32 预测 8480/18208 cycle，实测 8478/18203 cycle；stage 差预测 0、实测 23 cycle，属于零响应检验（754854）。
- 首次等待前加软件延迟：R4096 的两种放置均被反证，实测 3379 cycle，预测 4096／3188（755491）。首入口控制对照的两种形式也失败（755666）。
- R1536 外推：D 的响应预测 798、实测 849 cycle，误差 6.4%，判失败；其余端点通过（755770）。
- 直接端点（S6/K4，观察开启的探针程序）：
  - 首组最后 B 请求发出到 full 完成，测后联合边界为 425–607 cycle：起点取 B 请求对的前后戳，完成上界取旁路首次成功与 consumer 返回后戳中的较小者（758626）。起点不是整组请求的 issue。
  - A 请求发出区间为 412–475 cycle；若各进程取共同常数，A→full 可行区间为 384–660 cycle（758758）。原程序的 A 起点与观察干预尚未分开。
  - 相位完成到 consumer 返回不超过 77–98 cycle（758556）。

**不可辨识**
- 首组 full 返回前的 773 cycle 无法由现有窗口拆分：输入就绪与阻塞返回之间存在连续等价族。

- empty后准备0→640的同binary对照：九进程资格通过，R→N0桥接通过；窗口响应6618–6699cycle，名义模型2176cycle，冻结成本域无法解释。停止本校准形式、S4未采（774214／774221）。随后逐轮准备类通过独立长度响应：K4为2153–2163、K32为16044–16166cycle，首段形式被排除（774347）。N0−R随长度增长，固定J及P／源残量的分配仍未解释，见[本轮结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-preparation-length-001/README.md)。

**接口交付**
- 目的服务、源受理与信用归还、程序请求合同。旧 producer 控制合同因夹具失败停止；当前轮已接入 empty 后准备、请求受理与 acquire／关键路径的对应关系。

**判断**
- 单 CTA 链表现为计算受限的递推加一个共同固定项。空载、观察开启时，首组最后 B 请求发出到 full 的联合边界为 425–607 cycle。
- 这里没有负载，得不到完整 GEMM 所需的满载搬运延迟。

## 轮次

按日期排列；全部结果目录见[Q02-tma-request/native-d-chain](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 固定控制与释放能否接入D三请求（CPU） | 固定B局部价格、generation源与三请求；显式排程empty完成，四组合／三默认D回归及四零源匹配 | 默认报告逐位保持，四D图与保存B逐端点相同；14项测试及异目录复现通过，64调用、拟合／修复／GPU0；匹配TMA服务仍缺观测 | [接线、逐端点及收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261018-Q02-control-connection-001/EXECUTION.md) |
| 原生输入是否只增加共同固定窗口 | 固定已验证B控制与D原生三请求，B／D×S6/K4、K13各三进程；比较长度增长 | 754824完成12进程，D增长4585对冻结4608cycle，B/D增长差1cycle，资格全过；保留隐藏输入开发形式，固定项未拆分 | [输入、原始评分及收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261018-Q02-native-input-growth-001/EXECUTION.md) |
| D三请求完成与full就绪的局部连接（CPU） | 复用D实际请求、32768B事务及六物理槽；显式服务输入，旧九plain边界、三合成时序及六图 | 接线核对通过；旧D仍用runtime索引，S4/S6的A行距512/768B，下一轮先固定匹配程序与描述符 | [实现、边界证据、时间线与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261018-Q02-local-input-chain-001/EXECUTION.md) |
| 输入窗口放置与跨stage／长度外推 | 固定B规则，比较共同1454与共同546＋就绪908两放置；冻结S4/K13、S6/K32 | 两放置预测完全相同；754854的15进程完成，S4／长链窗口8478／18203cycle及响应通过原门槛，局部预测类获支持，位置仍等价 | [等价证明](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261019-Q02-input-window-identifiability-001/README.md)、[独立外推与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261019-Q02-native-input-validation-001/EXECUTION.md) |
| D自身启动段分配（CPU） | 停止迁移B公共项，按D窗口与匹配差固定三种已有放置，回放旧长度／stage并预测R1536 | 三放置各过14项开发检查、预测相同；保留consumer推进，位置仍等价，48调用、GPU0 | [参数、时间线与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261019-Q02-local-startup-allocation-001/EXECUTION.md) |
| 软件入口与隐藏输入能否分开归因（CPU） | 固定局部计算／控制放置，两批R4096／1536响应求必要界，回放改δ或改H两见证及旧stage／长度 | 原δ35/H773不相容；两见证各过17开发检查，归因仍未区分，36调用、GPU0，不选新价格 | [必要界、逐端点与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261019-Q02-entry-input-identifiability-001/EXECUTION.md) |
| 源就绪与目的完成能否在D局部模块分别接入（CPU） | 固定源／计算／同步，只透传完整引擎已有目的服务；旧4形状及18见证，三合成组织 | 接口已接通；210端点一致、旧报告逐位保持，76/100调用、GPU0，硬件价格不变 | [交付、反例与复核](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261019-Q02-destination-connection-001/README.md) |
| 异步源受理与信用归还能否在D局部模块接入（CPU） | 固定完整引擎、服务价与旧回归；受理／就绪两配对、信用归还两配对、通知乱序共五合成条件 | 接口已接通；418端点一致、旧报告逐位保持，21测试及异目录通过，97/120调用、GPU0，硬件规则未新判定 | [交付、反例与复核](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261019-Q02-source-credit-connection-001/README.md) |
| D局部与cfg_b调用方的请求合同是否一致（CPU） | 两匹配源几何及G3／G5五实际条件；固定现有服务、价格与程序 | 1048576地址／512端点一致、2304实际请求核对通过；D活动S4保留物理S6差异，8/16调用、GPU0，未迁移价格 | [合同、差异与复核](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261019-Q02-program-source-contract-001/README.md) |
| 首次等待局部预测类独立外推 | 固定上一轮参数与原B／D binary，B0／D0桥接后测B1536／D1536，各三进程 | 755770完成12进程且资格通过；B／D响应1592／849、匹配差743cycle，D对798偏6.390977%，其余四端点通过，完整形式停止 | [测前计划与冻结输入](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261019-Q02-startup-prediction-validation-001/README.md) |
| 原生首次等待延迟的局部判别 | 固定S6/K13与三请求，旧0／新0／新4096各三开发，新2048三留出；同binary改变首次软件延迟 | 755491九开发进程资格通过，响应3379cycle使4096／3188两版本均失败，R2048未采；先核对匹配驻留B干预 | [测前输入与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261019-Q02-first-wait-native-001/EXECUTION.md) |
| 首次等待时序能否区分输入窗口放置（CPU） | 固定两个价格放置与producer发出，只延迟首次consumer full进入；S4/S6、K13、R0/4096 | 4096／3188cycle响应带分开，默认报告保持，15项测试及异目录复现通过；尚未实现新GPU干预 | [时间线、端点与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261019-Q02-first-wait-entry-001/README.md) |
| 旧H773预算能否映射真实请求推进（CPU） | 保存六局部图与三份新端点；闭式筛查零发出族，再用既有回调接一组按操作区分的producer推进 | 统一平移及ready／return分配均不能修复旧物理映射；一组代数见证保持886项下游关系与六窗口，满足21项新端点约束，价格未辨识 | [反例、可执行接线与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261020-Q02-issue-budget-connection-001/README.md) |
| 首组A实际发出与producer推进 | 固定B／full观察，只增首代A工厂前后戳；旧版／新版关闭／新版开启 | 758758九进程60资格通过；观察程序A412–475cycle、A→B21–179cycle，旧见证的直接映射及统一平移失败；原程序A起点与观察干预未分开 | [有限轮次](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261020-Q02-first-a-issue-001/README.md) |
| producer程序控制与源准入分工（CPU） | 固定源／计算／同步价格；分开empty后准备与各请求受理返回 | 73／81条producer指令、八零默认报告保持；新完整入口夹具失败，两修复用完并恢复活动代码，候选未接入；186请求372端点字段核对通过 | [有限轮次](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261020-Q02-producer-control-contract-001/README.md) |
| 首组三请求issue至full完成能否约束局部分配 | 在合格full旁路观察上增加B0/B1请求对前后戳；S6/K4旧版／新版关闭／新版开启，各三进程 | 758626九进程36资格通过；联合成功返回后戳，三个issue→full区间为425–607／425–609／425–599cycle，1／773两具体价格映射均不相容；原冻结评分保留 | [原始评分、联合边界及复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261020-Q02-first-issue-completion-001/README.md) |
| 生产者节流能否区分已有控制放置（CPU） | 固定S6/K13、两解释及H必要界端点，仅把组间隔0改为1024cycle | 八调用、拟合/GPU0；两解释均预测plain增6144cycle，存在相同可行响应，不补测 | [计划、预测与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-control-placement-intervention-001/README.md) |
| g11等待路径身份能否区分已有控制放置（CPU） | 固定S6/K13、两个解释与H两端，只比较full就绪相对wait进入 | 四调用、拟合/GPU0；四份图均预测快等待，布尔量不可分，停止该观察选题 | [计划、四份事件与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-wait-path-screen-001/README.md) |
| empty 后程序准备与请求受理接线（CPU） | 固定原硬件价格，准备7/5/3ns及名义640cycle干预；同步／异步与局部／完整入口同序 | 八默认报告逐字保持，47项测试通过；140/160调用、GPU0。两个解释的S4−S6响应带可分，实机程序成本待下一轮约束 | [接口、反例与推导](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-program-preparation-001/README.md) |
| empty后准备能否区分两控制解释 | 固定旧硬件价格与H；S6原程序／同binary0／640各三进程，成本与资格通过后冻结S4同配对 | 774214两有效进程后执行器失败，774221按同计划继续剩余七个；九进程资格通过。响应6618–6699cycle对名义2176，校准超域，未测S4 | [测前计划与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-preparation-delay-001/README.md) |
| 原有准备是否逐轮限制请求推进（CPU） | 固定H总预算、计算/同步；两个控制形式各16相关见证，核旧六条件与新两条件；首段类另作必要界 | 256/256核对通过；逐轮准备P=240–309.75cycle可保持旧六窗口并解释新响应；首段类上界3737cycle低于观测6617 | [计划与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-recurrent-preparation-001/README.md) |
| 逐轮准备能否预测独立长度 | 固定S6及旧R/N binary，K4／K32各三臂三进程；K4通过才测K32；四个留出条件、两个R桥接条件 | 774347十八进程资格／桥接通过；K4响应2153–2163、K32响应16044–16166cycle，均在冻结门槛内；首段形式被排除，N0−R长度差未解 | [冻结预测与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-preparation-length-001/README.md) |
| a044 跨卡复核（mechanism-r3，780769） | native-input-growth 12 次，原评分器 | 开发判定“输入被覆盖形式通过”保持 | [Q02 报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/q02-returned-report.md) |
| a044 保序修复（785518–785575） | first-wait-native 9、native-input-validation 15、startup-prediction 12、first-a-issue 9、first-issue-completion 9 次，原顺序与环境，原评分器 | 4 轮判定不变。first-issue-completion：seed 3 的 issue→completion 区间 273–587 cycle，与 ready_773 预测带 734–812 不相容（seed 22/41 仍相容），保留形式变为空；layout 程序查询 7 次逐字节相同 | [修复报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/mechanism-repair-r2-report.md#q02q04q05-原评分器重评) |
