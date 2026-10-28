# 请求形状与次序的响应

20 个目录（10-15 至 10-22）：在 TMA 探针上改变 B/A 行距、长度、在途组数、描述符分解、请求次序与源排列，测完整窗口与配对增量；另含 cfg_b 实际 tensor-map 描述符查询。原 row-stride 族中的实测响应部分；预填与等待入口见 [prefill-wait-entry](prefill-wait-entry.md)，规则形式检验见 [request-service-rules](request-service-rules.md)。

## 结论

问题来自完整 cfg_b：A16、B80、B112 的 plain 实测分别慢 28.9%、18.1%、18.6%，B32 为 −0.8%，组合模型对四者都预测 0。

- 132 CTA、Q4 探针：B32 接近零，B80/B112 约 +4.3%–5.4%（20261017-Q02-tma-stride-002）。
- Q4（4 组在途）：B80 只多一次性约 295 cycle，与长度无关，冻结长度判别支持（752028）。
- Q1（1 组在途）：B80 增量为 294 + 165·(G−1) cycle，冻结留出 G16/G64 误差 3.08%／1.00%（752094）。Q1/G32 下 B32/B80/B112 分别 +2002/5398/5530 cycle，A16 +6802 cycle，两个按覆盖比例的形式均被反证。
- A16 响应与 K 步长（R=1.006）、描述符外层分解（R=1）、源次序排列（R=1.002）无关；同覆盖计数下 A16／A80 的不变性被反证（R≈0.889，778301）。
- A-only 与双 B 的响应不能简单相加（MA≈0.90，MB≈1.31，752232）。
- 单 CTA 下，Q1 每组持续增加约 165 cycle；这些单 CTA 观测不能直接解释完整 GEMM 中 18%–29% 的响应。完整 GEMM 的行距响应取决于哪些 CTA 同时读取重叠的源行，须在保留真实工作排列的多 CTA 条件下研究（见 [gemm-input-network](gemm-input-network.md)、[gemm-source-reuse](gemm-source-reuse.md)）。

## 轮次

按日期排列；全部结果目录见[Q02-tma-request/request-shape-response](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 实际 CUTLASS tensor-map 查询 | 补齐程序图的实际描述符字段：在当前原生源副本中输出 `cuTensorMapEncodeTiled` 的全部参数和返回描述符字节，再从实际 launch params 取 A/B 描述符，通过字节与源地址匹配到各自编码记录。 | 当前为准备状态，尚未编译或执行。后续完成原生查询后，核对 rank、数据类型、global 形状/字节步长、box、element stride、swizzle、L2 promotion、OOB fill 和原始 128 B 描述符；再按实际范围更新程序图来源。 | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261015-cfg-descriptor-query-001/README.md) |
| 固定分配的实际tensor行距定位（751907） | 132 CTA、A16＋B0/B1各8 KiB、SW128、全分配固定；B四行距×Q4，另配Q1端点，三seed同进程配对 | Q4的B32周期响应近零，B80/B112约+4.4%至+5.4%；Q1包络未稳定增长，保留地址与在途组织连接 | [实测](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261017-Q02-tma-stride-002/EXECUTION.md)；[准备](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261017-Q02-tma-stride-001/README.md) |
| 单CTA行距响应的长度冻结判别 | 暖机64组、SM124、原二进制与两行距固定，只改正式16／32组；4未采条件、每条件3进程 | 752028的12进程同卡同SM合格；16／32组增量中位均295cycle，固定边界获支持、逐组形式被反证 | [冻结与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-tma-length-001/EXECUTION.md) |
| 固定Q4暖机的Q1/Q4行距判别 | 固定Q4／G64暖机，只改正式G32的active1／4，两行距配对，4条件12进程 | 752057完成；Q1增量5412cycle、Q4为294，控制及稳定性通过，两冻结具体形式均被反证 | [冻结与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-tma-active-001/EXECUTION.md) |
| Q1行距增量的长度判别 | 固定Q1、原binary与Q4/G64暖机；G32同批控制，G16／G64四条件采前留出，共18进程 | 比较Q相关一次5412cycle与共同边界294加持续量5118/31两形式，752088在Popen前缺执行位停止；752094完成18进程，两个新长度支持持续量形式、反证一次边界，独立评分一致 | [冻结包](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-tma-active-length-001/README.md)；[执行结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-tma-active-length-002/EXECUTION.md) |
| Q1源覆盖尺度的冻结判别 | 原binary、Q1/G32及暖机固定，四行距同分配；base/B80控制，B32/B112两个未采留出 | 752126完成，B32／B112为2002／5530cycle，控制合格；B32误差44.51%／56.84%，两冻结比例形式被反证并停止 | [冻结与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-tma-granule-001/EXECUTION.md) |
| A真实行距的分支判别 | 固定Q1/G32及同分配，A lda4096/4104×B ldb2304/2320，四条件三seed；测量kernel保持 | 752146完成，控制与资格通过；B32配对增量2014→1860cycle，R=0.923535，两个冻结形式均被反证；保留A共同增量、B剩余增量与负交互三约束 | [冻结清单与预测](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-a-stride-001/README.md)；[结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-a-stride-001/EXECUTION.md) |
| A／B独立请求响应分解（752232） | 新binary运行时mask1／6／7；8开发条件，统一三请求暖机，先3verify后3plain | 数值、控制与稳定性通过；MA约0.902／0.906、MB约1.314／1.290，停止0或1的两个位置形式，默认组合保留 | [执行与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-operand-split-001/EXECUTION.md) |
| 相邻Ktile源地址状态判别（752255） | 固定Q4/G32和真实三请求，step1／2×A0/A16×plain／sparse八开发条件，统一step1暖机；6进程／48臂 | 全部资格通过，prefill与完整窗A16增量中位498／501cycle，R=1.006024；停止相邻共享单独解释，邻接无关形式保留开发相容性，默认组合保留 | [执行与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-source-step-001/EXECUTION.md) |
| A描述符外层分解（752363） | 固定源／目的字节、三请求及SW128，仅A box `[64,128,1]`→`[64,2,64]`；Q4八开发条件、6进程／48臂 | 全部资格通过，两factor预填及plain A16增量均490cycle、R=1；保留本次分解不变形式的开发相容性，未接新价格；两构建和Popen前执行位失败归执行历史 | [执行与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-descriptor-factor-001/EXECUTION.md) |
| 三请求发出次序（752379） | 同A/B描述符与VA，A→B0→B1／B0→B1→A，Q4八开发条件、6进程／48臂 | 数值、稳定性及扰动通过；原次序完整窗、prefill及增量七项控制均超5%，信息不足；不以R=0反证原形式，不重采或替换组合 | [执行与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-request-order-001/EXECUTION.md)；三CPU构建与两修复归执行历史 |
| Ktile异质性的必要筛查（CPU） | 752146同批四角×三seed；固定价格，复算逐组max的聚合界，枚举真实地址的既有home／P2特征 | 同质零响应停止；异质界全部相容，1152请求中映射负载随Ktile变化，总覆盖恒定。未拟合或替换组合 | [必要界、特征与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-ktile-heterogeneity-001/README.md) |
| 同集合源次序判别：752416 | Q1/G64 step1／57×A0／A16，保持源集合与slot／phase；Q4/G32两臂桥接 | 数值、桥接与稳定性全部通过；Rcycle=1.002271395、Rns=1.004694836，保留本次排列相容性，未给新价格或替换组合 | [执行与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-source-permutation-001/EXECUTION.md) |
| 单B／双B连接判别：752488 | 四开发控制，最多两候选；两次构建修复核实内部rest_val单B切片、8192B事务及SASS | 四窗口稳定、12桥接全过、2359296值零错；原基础必软件门未过，信息不足，未采留出 | [原登记、评分与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-single-b-discrimination-001/EXECUTION.md) |
| 同覆盖计数的A16／A80不变性（778301） | 复用二进制、单CTA Q1/G32、A0/A16/A80同分配，三数值＋三plain；六次CPU预测相等，冻结R=1±.05 | 六进程有效、4718592值零错，Rcycle=0.889266／Rns=0.887850，反证该不变性；地址／片段／历史未分，未给新价 | [冻结、实测与原始回放](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261022-Q02-row-phase-invariance-001/README.md) |
| 同集合次序的A80扩展（778441） | Q1/G64 step1／33×A0/A16/A80，暖机step1，六进程／36臂；设备SASS保持 | 18874368值零错；双单位窗比均入1±5%，A80/A16增长比仍约0.889。A0/A16重复752416；不追加同类扫描，见[原始数据与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261022-Q02-fullset-order-001/README.md) |
| 132CTA源地址与目的组织判别 | G3真实两工作列表、SMID固定源坐标；Q1/Q4×A16/A80×A SW128/NONE，八条件各三进程；只比较同二进制 | 779181构建、779198十二有效进程，3321888768值零错；Q1/Q4均信息不足，不补测；现行模型保持 | [计划与预测](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261022-Q02-shared-swizzle-001/README.md) |
| a044 跨卡复核（mechanism-r3，780769） | 25 个来源的原二进制、参数与评分器；只换设备身份，不重拟合 | Q1 行距增量 G16/32/64 为 2681/5352/10818 cycle，边界加持续成本误差 ≤3.229%；operand-split 桥接误差 5.08%/6.67%，由排除改为信息不足；行首相位不变性仍被反证，固定集合次序不变性保留；单 B 与 swizzle 仍信息不足，其余原结论不变 | [Q02 报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/q02-returned-report.md) |
