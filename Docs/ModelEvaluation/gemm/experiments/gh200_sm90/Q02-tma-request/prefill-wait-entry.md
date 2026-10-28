# 首次预填、在途组数与等待入口

10 个目录（10-18、10-22）：在单 CTA TMA 探针上定位行距增量出现在首次预填还是逐组推进，以及推迟 full 等待入口能否隐藏它。原 row-stride 族的一部分。

## 结论

- Q4 的 A16 增量 363–513 cycle 全部出现在首次预填，之后为 0；Q1 初始增量 0、完整增量中位 6806 cycle（752176）。
- 初始在途组数：Q2 预填增量 0、Q4 为 498 cycle，R=0（752300）。共同预填代码路径的对照资格超 5%，信息不足（752334）。
- 每组 full 等待入口推迟 8192 cycle 后，Q1 的 A16 增量 6737→0 cycle，支持可隐藏位置形式（752191）；B32 的同类判别资格失败，信息不足（752208）。
- 预填组间间隔：循环版 gap0 增量 59 cycle 低于冻结门槛，信息不足（775192）；四组展开版 A16 预填增量 377→0、R=0，支持间隔隐藏（775199）。
- 正式等待 hint 由长改短，增量 5429／5388 cycle，与 null 相差 0.44%（752118）。

## 轮次

按日期排列；全部结果目录见[Q02-tma-request/prefill-wait-entry](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| Q1正式等待hint的开发定位 | 正式Q1/G32仅改hint 10000000→32ns，两行距配对；暖机固定长hint/Q4/G64，4条件12进程 | 001未执行因代码状态混杂停止；002改单kernel，752118完成，long／short增量5429／5388cycle，控制合格，short距冻结null仅0.443459%，本轮收口 | [双kernel准备停止](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/20261018-Q02-wait-hint-001/EXECUTION.md)；[单kernel结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/20261018-Q02-wait-hint-002/EXECUTION.md) |
| 首次预填的稀疏观测定位 | Q1/Q4×A0/A16×plain0/sparse−1，固定分配与统一暖机；同进程四臂，三seed，首预填末单点且end后写回 | 752176的12进程资格通过；Q1初始增量0、完整增量中位6806cycle；Q4初始与完整增量均513cycle，约束发出反馈；752175的Popen前执行位故障保留 | [冻结清单](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/20261018-Q02-prefill-observer-001/README.md)；[执行与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/20261018-Q02-prefill-observer-001/EXECUTION.md) |
| Q1等待入口的可隐藏位置判别 | 每组full wait前D0／8192cycle×A0/A16，固定真实三请求、Q1/G32及统一暖机，三seed同进程四臂 | 752191完成6进程，全部资格通过；A16增量中位6737→0cycle、3424→0ns，双单位R=0，支持可隐藏位置形式 | [冻结清单](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/20261018-Q02-wait-entry-001/README.md)；[结果与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/20261018-Q02-wait-entry-001/EXECUTION.md) |
| B32等待入口的可隐藏位置判别（752208） | 复用已批准binary，A0固定，B0/B32×D0/8192四条件，同进程配对，先3verify后3plain | 数值、稳定性与完整窗控制通过；D0配对效应偏旧值5.163853%／9.090909%，资格失败，结论信息不足，不重采或替换组合 | [执行与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/20261018-Q02-b32-wait-entry-001/EXECUTION.md) |
| 预填组间间隔的发出反馈判别 | 计划Q4/G32、gap0／8192×A0/A16×plain／sparse八开发条件，同进程配对、统一gap0暖机 | CPU构建与输入绑定未通过，两次修复预算用尽；未运行GPU，结论信息不足，两个响应形式未判定 | [执行记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/20261018-Q02-prefill-gap-001/EXECUTION.md) |
| 初始在途组数的预填响应判别（752300） | 复用752255原binary，正式Q2／4×A0/A16×plain／sparse八开发条件，统一Q4暖机；12进程／48臂 | 全部资格通过，prefill增量0／498cycle、R=0，保留Q2初始零响应形式，停止1/3与1；Q2完整plain仍增长231cycle／128ns，默认组合保留 | [执行与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/20261018-Q02-prefill-active-001/EXECUTION.md) |
| 共同预填代码路径（752334） | 私有CU只改首次guard，实际Q2／Q4共用首两组机器码；八开发条件、12进程／48臂，固定原Q4暖机 | 完整窗控制、稳定性及扰动通过；R=0，但Q4 prefill两窗及增量偏旧值12.167027%、10.545840%、6.024096%，超原5%，信息不足；本形式停止，不重采、不替换组合 | [执行与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/20261018-Q02-prefill-shared-prefix-001/EXECUTION.md) |
| 原预填间隔的同二进制执行 | 复用752276已构建程序，不以动态循环静态变化作废；原八条件、最多3数值＋3正式进程 | 775192的48臂有效、最大扰动0.284%；gap0增量59低于冻结69.45cycle，信息不足，循环版收口；无构建 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/20261022-Q02-prefill-gap-execution-001/README.md) |
| 四组展开的预填间隔 | 仅恢复首次四组显式调用，间隔0/8192×A0/A16×plain/sparse原八条件；最多6进程 | 775199的48臂有效、三个seed全过原资格；A16预填增量377→0cycle、R=0，支持本版间隔隐藏，排除保留形式；观察扰动最大0.136426%，无拟合或组合替换 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/20261022-Q02-prefill-gap-unrolled-001/README.md) |
