# 多 CTA 共享源的固定形式筛查

18 个目录（10-18），基于既有 Q1/Q2 起步偏移（skew）多 CTA 探针及新测的相位与源移动；单点完成观察已移到 [completion-observation](completion-observation.md)。

## 结论

**已测事实**
- 每组软件控制四段固定为 328/137/85/127 cycle，同源跨遍的窗口变化全部落在 wait。
- skew8 下，每八组窗口中位从 512 ns 降到 416 ns，跨 CTA 发出跨度则从 304 ns 增大到 3680 ns。
- 保持源多重集合不变、只重排 CTA 与源坐标的对应，CTA37/73 的 plain 周期增加约 27%。
- 相同 plain 重复三次（[plain-repeat](../Q10-time-mapping/plain-repeat.md)）：包络在 ±3.3% 内，逐 SM 完整窗口相差 11.3%–17.4%。

**被排除的固定形式**
- 每 GPC 一份总速率：所需总工作量是实测窗口的 2.44 倍。
- 单 FIFO 汇合。
- 局部/共享汇合：六个 prior 通过，grouped/mixed 失败。
- 共同尾队列；35 信用的在途容量；动态分摊；共享预算重估；短请求优先；匿名共享初态；局部轮转完成；共享未完持有局部槽；active 持有到源就绪（最快界仍高 20%）；串行尾。

**信息不足**
- source-rotation：同一条件的配对符号不一致。
- 单点完成观察（ready-snapshot、ready-fixed-ports、full-side-warp）的资格均未过，见 [completion-observation](completion-observation.md)。

**判断**
- 共享源条件下，逐 CTA、逐 SM 的窗口在相同运行间本身就不可复现，用确定性固定形式去对齐逐组窗口并不稳妥。
- 共享服务的检验应以包络和结束时刻分布为对象，并允许服务随请求相位演化。本组没有接受任何形式。

## 轮次

按日期排列；全部结果目录见[Q02-tma-request/shared-source-forms](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 共享热源迁移 | 每声明GPC PS2/waiting1、真实地址工作与同源generation，先旧六prior | 11独立CTA工作守恒排除固定每GPC总速率迁移，未建adapter | [必要界](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-gpc-hot-migration-001/EXECUTION.md) |
| 局部推进与共享源汇合 | 每CTA有界PS在实际受理时进入旧GPC服务，两支ready取max；先六prior再重排 | 六prior通过，重排响应与mixed空间跨度失败，停止固定连接 | [执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-local-shared-join-001/EXECUTION.md) |
| 共享仲裁与局部完成 | 保留局部完成支，共享两条τ前沿改成单FIFO τ/2，吞吐预算不变 | 首prior双5%失败，停止其余七项 | [反例](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-shared-fifo-join-001/EXECUTION.md) |
| 原价串联完成尾部 | local_end后接原core，两套原价不变；先查单CTA/Q1/A-only最快界 | 下界31605.175cycle超过允许29004.211，停止固定连接 | [必要界](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-serial-tail-bound-001/EXECUTION.md) |
| 共同尾部与共享排队 | local完整预算保持，local_end后只增加core排队 | native保存时刻保持，首prior双5%失败，停止其余七项 | [结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-common-tail-queue-001/EXECUTION.md) |
| 共享代有限在途 | 按吞吐与共同尾推导35信用，source_ready归还 | 信用生效但首prior主窗口不变，原双5%失败，停止固定容量 | [结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-inflight-credit-001/EXECUTION.md) |
| 旧共享约束的协议与时间分布 | 84份旧raw的早晚端点、双时钟及trace/plain；三个保存模型对照 | 实测early→late下降集中于wait，现用模型未复现；原评分保持 | [结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-protocol-profile-001/README.md) |
| 时间状态必要筛查 | q1-skew8七进程的逐CTA首遍/次遍/访问者退出边界，热缓存预算与局部界 | 首遍且访问者未退出时已下降；热缓存接线等价，局部401.526ns界仍相容 | [结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-temporal-state-screen-001/README.md) |
| 共享服务动态分摊 | 保持原2/τ吞吐、完成尾与局部支；仲裁改为活跃请求等分 | 早段过门，后段高估35.698%ns，停止固定版本 | [结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-gpc-sharing-arbitration-001/EXECUTION.md) |
| 同源相位与软件推进约束 | 三Q1起步×七进程，配对同CTA同源跨遍及循环控制 | 软件四段恒定；同源等待随起步改变，保留共享完成／相位缺口 | [保存证据](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-source-phase-controls-001/EXECUTION.md) |
| 短请求优先选择 | 两共享槽及全部价格保持，只改空闲时的请求选择 | 首prior低估19.135%ns，停止固定版本 | [结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-short-request-priority-001/EXECUTION.md) |
| 局部更新后的共享预算重估 | 保留两FIFO槽/max连接，只一次重估共享τ/L | 达到12次求值上限未收敛；早段失败，停止参数版本 | [有限拟合](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-shared-budget-refit-001/EXECUTION.md) |
| 自身源不变的背景响应与慢流归属 | 48份Q1 raw，固定自身源CTA37/73；源共享20trace及旧七plain锚点 | 自身源/SM未变仍增加约27%；控制恒定，进入plain源移动判别 | [复核](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-source-following-audit-001/EXECUTION.md) |
| plain源移动的冻结判别 | 同REG40 binary，source_tile_shift0/1各三进程；不加内部观察 | 753347完成12进程；原对照桥接及shift1符号失败，信息不足，收口 | [执行与等待](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-source-rotation-001/EXECUTION.md) |
| 局部量化源与轮转共享完成 | 旧r10仲裁接实际局部受理/max完成；首prior早晚 | 早窗通过、晚窗高25.217514%，固定连接停止 | [执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-local-rr-completion-001/EXECUTION.md) |
| 每调用匿名共享初态 | 两槽按事前seed17/29/43匿名优先序；价格/局部/完成连接固定 | 早窗低18.451419%、晚窗过门；具体形式停止 | [执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-call-order-state-001/EXECUTION.md) |
| 共享未完是否通过局部槽反馈（CPU开发） | 原2active/1waiting及两套价格；进入active才预约core，占用到max(local_end,core.end) | 六prior过原双5%；重排15.413612%及跨度574.935495ns仍失败，固定连接停止 | [八条件、87552请求审计与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-shared-held-local-slot-001/EXECUTION.md) |
| 两active能否持有到源就绪（CPU必要界） | 原局部工作/尾及Q1三请求，取零共享争用最快界 | 612.557674ns／1221.930048cycle仍超原双5%允许上界，固定解释停止 | [闭式、事件核对与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/shared-source-forms/20261018-Q02-ready-held-slot-bound-001/EXECUTION.md) |
