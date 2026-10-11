# R10：A/B 行距、地址覆盖与执行规模

**行距响应依赖配置、遍历方向、几何和输出位置，现有结果不支持通用的 32 B 或 128 B 对齐罚项。** 旧 job738203 的 B32/B64 虽满足 32 B 对齐仍变慢；V09 另一几何下 B32 未见明显惩罚，二者同时保留。同卡固定旧几何后，沿M的B32惩罚约16.94%，沿N仅1.20%，执行顺序参与了迁移失败。单输出对照进一步确认，同一B源组在不同全局执行组织下的行距代价不同，跨输出预取不是大幅惩罚的必要条件。+256 B基址平移没有让响应整体跟随相邻B面板；交换CTA55/91的完整工作序列后，强弱响应仍留在原物理CTA/SM。随后固定工作与SM、延迟CTA55 producer起步，行距响应仍改变，固定位置常数同样不足；需要保留时间状态，尚未唯一归因到达、预取或流水初态。地址覆盖只是软件需求，不是物理流量。有效供给与 stage 关系见 [R13](R13-async-retirement.md)，OOB/显式零的边界路径见 [R18](R18-cluster-boundary.md)；现行公式由 [RULES](RULES.md#v09-model)维护。

## 作业与证据索引

2026-10-11，B的静态工作统一为 [R13供给账本](R13-async-retirement.md#supply-work-ledger)。`gh200-supply-work-ledger/1`保留每个CTA/输出的descriptor box、源身份和multicast目的前驱，不再只返回七列聚合覆盖；R10的两个raster与36/126/132执行规模已进入57条件守恒复核。名义源payload、有效源地址、去重源身份、目的投递和D输出分别记账。该接口仅补齐组织信息，没有给方向或B32另加价格；旧几何B32变慢、新几何未见同量级惩罚仍是同一供给状态候选必须同时解释的约束。A16和stage4不用于另拟行距价，原局部冻结失败不变。


| 作业 | 固定对象 | 结果入口 |
|---|---|---|
| 735876 | 三配置，2600×3000×2000，B 行距 6000/6016/6144 B | [v2 完整时间](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v2/analysis-formal/summary.json)、[v4 cfg_b 局部周期](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/analysis-formal/summary.json)、[独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reviews/ROOT-independent.md) |
| 737322，V08F | A/B/D 行距、K/N 尾部；84 个解释性条件 | [原汇总](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/followup-v1/followup.json)、[供给残差](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/A-20261009-supply-v3/summary.json) |
| 738101 | cfg_a、2304×3072×4096，grid 32/64/96/132 × 三种行距 | [独立回放](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-sm-job738101/reanalysis/manager-formal-replay/summary.json) |
| 738169 | cfg_b h03 四角，2400×3000×1000，同容量 | [逐 trial 联合项与扰动](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-joint-pitch-job738169/reanalysis/B-20261009-matched-joint-final/joint-pitch.json) |
| 738203 | cfg_b B 行距五点，2304×3072×1024，局部冻结检验 | [冻结](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/frozen/r10-b-pitch.json)、[原评分](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/b-pitch-score-v1/b-pitch-score.json)、[位置与扰动](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/B-20261009-frozen-negative/localization.json) |
| 739011，V09 | 新几何 A16/B32/B80/B112 与 stage4 | [完整留出](V09-component-validation.md)、[逐例复核](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/manager-results-v1/cases.csv) |
| 739571 | 同几何AlongM/N×B0/B32 | [方向与行距差分](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-raster-pitch.json) |
| 739585 | 同输入map，36/126单输出CTA与132持久CTA，各配B0/B32 | [同源组压力对照](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-source-pressure.json) |
| 739598 | B基址+0/+256 B × 行距+0/+32 B，分配容量相同 | [位置与相邻面板对应](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-base-shift.json) |
| 739608 | CTA55/91整条工作序列交换 × B0/B32，逐轮工作集合不变 | [工作与物理位置对应](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-work-swap.json) |
| 739619 | 固定CTA55工作与SM，producer起步0/512 ns × B0/B32 | [实际时移与行距交互](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-producer-delay.json) |

<a id="source-event-transfer"></a>

### 真实输出次数与源身份仍不足以迁移B32响应

继续同一工作/状态定义，`cfg_b_source_program`从账本生成每CTA全部K-group的A/B0/B1源身份，不再限于2816×1536的两输出。`shared_stage_events(output_counts=...)`接入实际1/3/4输出，输入环不重置；两WG交替复用accumulator，j≥2须等自身E_(j−2)返回，E_j又须等E_(j−1)释放共享输出SMEM。该前驱来自原pingpong的store_tail、epilogue次序到达和下一次scheduler loop，不把其它WG的math返回直接当自身可开始。两输出默认的13组旧核心结果精确重现；事件实现通过不代表价格成立。

本次只用GPU-ef8692f3的739571/739585，复放100个dual原记录并逐CTA核对工作与SHA。固定[R13已保存共享参数点](R13-async-retirement.md#source-candidate-replay)，每个源atom费用为原名义payload服务加`π×B_extra128_KiB`，目的投递和D工作不变。只以AlongM later的全部匹配窗口B32/B0比例中位18.4003%定位一个π；Brent在[0,4]ns/KiB内按价格容差定位至π=0.973627275，共27次求值，无额外起点、A16价、stage4价或方向/位置常数。定位点的响应残差仍为+7.42029127×10⁻⁵，价格收敛不证明存在精确响应零点；这是近似开发点。其它参数来自既有探索，未证明其选择历史未使用这些数据；本轮不是独立留出，也不声称π唯一或物理可辨识。

| 检查窗口 | 预测B32响应 | 实测中位响应 | 结论 |
|---|---:|---:|---|
| AlongM首轮，T3/T4 | 0% / 0% | +21.63% / +27.68% | first迁移失败 |
| AlongM j1，T3/T4 | +17.71% / +18.29% | +19.94% / +20.12% | 接近已定值的later目标，不替代其它窗口 |
| AlongM j2，T3/T4 | +19.47% / +18.52% | +19.35% / +18.52% | 接近同一开发目标 |
| AlongM末j3/T4 | −2.16% | +5.30% | 末轮方向失败 |
| AlongN全部七个j/T | +64.53%～+70.62% | −0.59%～+2.87% | raster迁移失败 |
| single36首36 CTA | 0% | 0% | 本窗口相容，不推广到更大规模 |
| single126首36 CTA | 0% | +23.36% | 无下一输出也存在的规模响应仍漏失 |
| persistent132首36 CTA | 0% | +20.40% | 首窗仍漏失 |

全部j/T的基线、响应、十次范围和逐CTA预测保存在[结果](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/B-20261011-source-event-transfer-v3/B-r10-event-transfer.json)。v3仅澄清定位点的非零响应残差；价格、27次求值、全部预测/观察/空间字段与v2精确一致。空间检验同样失败：AlongM B32首轮T3/T4的预测局部L跨度为7.18/0ns，实测中位1504/1648ns，对应原生cycle跨度2569.5/2735；single126 B32预测约0ns，实测1344ns。两个时钟域分别保存；本候选的ns价尚无可靠first周期映射，不用目标实测f换算预测。不是只缺一个完整时间常数。所拟价格只改变源服务时长，模型source/delivery工作守恒，不能把错误解释成应再删目的请求。

本具体参数点不采用。原始432卡B曲线及V09 G5没有接收ef卡的时间价格；当前卡的raster干预已提供旧/新几何差异中的一个同卡因果桥，另一卡的数值迁移仍未完成。源身份、输入slot与accumulator前驱是必须保留的结构，但这套固定共同服务、同时入口和源订阅图仍不足以解释首/后输出、方向、规模及空间的联合响应。不据此排除全部源共享模型，不追加方向常数、A16价或stage4价。复核用 `analyze_r10_supply_events.py --data-root <主checkout结果根> --output <新的B目录>`，没有GPU操作或原冻结修改。

### 不同源box的地址覆盖部分重叠

整box身份相同与地址覆盖重叠不同。B32使N64×K64的相邻box共享部分128B覆盖单元；完整box请求及目的投递仍分别保留。`analyze_r10_granule_overlap.py`按descriptor stride、origin、extent和显式B offset形成有效地址集合，声明分配基址128B对齐，分别计算32/128B覆盖。SW128仅描述目的布局，不改全局源地址。这些集合不是物理读取次数，也不证明硬件按该粒度合并。

下表仅为软件首轮的一个K-group，不声称所有请求同时在途。全部12几何的其它软件轮、A侧和32B对照在[结构筛选](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/B-20261011-granule-overlap-v1/B-source-granule-overlap.json)。

| 几何/组织，B32 | 不重复box有效源 KiB | 逐box的128B覆盖 KiB | 覆盖并集 KiB | 并集额外覆盖 KiB |
|---|---:|---:|---:|---:|
| 2304×3072，AlongM | 128 | 224 | 134 | 6 |
| 同形状，AlongN | 384 | 672 | 386 | 2 |
| V09 G5，2304×2304 | 288 | 504 | 290 | 2 |
| 单输出36 CTA | 32 | 56 | 38 | 6 |
| 单输出126 CTA | 112 | 196 | 118 | 6 |

32B对照没有额外覆盖或不同box之间的附加订阅。128B并集能表达方向和旧/新几何的覆盖差异，但36/126的额外并集都为6KiB，而已见首36 CTA的L响应为0%/23.36%；单靠这一并集不能作为时间规则。G5仅做静态几何，不接收ef卡价格。

另以旧失败图自身生成的发起/ready时间，固定所有时间重新检查不同box的严格在途订阅，不引入观察时间。10个R10条件中，B32首轮追加订阅后均达到该轮的静态并集极限：AlongM额外覆盖为每K-group6KiB，AlongN为2KiB，36/126也都为6KiB。该固定时间表没有产生额外的规模状态差异。若减少服务工作，ready会提前、订阅窗口会缩短；必须让这类变化反馈到有限stage、消费和输出图中重新求解，不能用本次计数直接替换服务时长。未拟新价格，原R10/V09失败及全部源/目的/D账目保持不变。

### 地址重叠反馈到ready后，当前参数仍未暴露首输出停顿

`shared_stage_events`新增显式的地址单元批服务候选：先检查整box在途共享，新box代次的各单元再按严格`issue<owner-ready`订阅。新单元作为一个FIFO批，服务为单元数乘显式单价，批完成后加read latency；已订阅单元沿用原owner的ready，整个box取两者最大。单元到ready即过期，没有驻留缓存。每个完整输入需求、目的投递和D仍保留，服务变化反馈到ready、订阅、stage释放、消费及输出。批完成不等于逐单元流式完成，也不是物理事务解释。

[未拟合反馈计算](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/B-20261011-address-feedback-v3/B-address-feedback.json)固定原GPU-ef的ns参数，32/128B单元价仅由对齐16KiB源服务价换算，分别为`service_ns/512`和`service_ns/128`；没有沿用失败的B-extra覆盖价。10个条件各计算两粒度。五个B0条件的全部MAIN端点与旧图精确相同；旧覆盖价格模式的10组完整预测也逐字段未变。原100进程数值复放的观察汇总只读复用，其数据/metadata SHA重新核对，不把本次模型重算写成再次执行数值实验。

| B32配对窗口 | 128B反馈预测L响应 | 实测中位 |
|---|---:|---:|
| AlongM，首输出T3/T4 | 0% / 0% | 21.63% / 27.68% |
| AlongM，后输出各j/T | 0.154%～0.798% | 5.30%～20.12% |
| single36，首36 CTA | 0% | 0% |
| single126，首36 CTA | 0% | 23.36% |
| persistent132，首36 CTA | 0% | 20.40% |

32B对照所有L响应均为0。128B反馈已改变后续ready与槽复用，不能再归咎于时间表没有重算；但该参数点仍未解释首/后、规模和空间响应，不采用。首输出并非没有full等待：AlongM的最大累计ready等待从46.971升到49.725ns，single126从39.626升到42.380ns；在本候选的TC服务前沿之前，最小ready余量仍分别94.275/101.620ns，所有首输出的TC服务空档均为0。等待变化被已有TC服务覆盖，L不变。B32首输出空间L跨度预测近零，AlongM实测中位为1504/1648ns。这里约束的是这组既有参数及批完成假设，不排除全部地址共享服务；也不为匹配首输出追加方向、规模或first专属项。

### 强响应在native cycles中仍存在，统一换单位不能修复零响应

[双时钟复放](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/B-20261011-clock-consistency-v1/B-r10-clock-consistency.json)重新读取100进程、复算409600个D值，17组原汇总和逐trial结果精确复现。9720个B0/B32窗口按CTA、输出j/T、工作坐标和consumer role匹配，原配对SM全部相同；每个native窗口在单调用、单SM的同role内作差，ns使用直接FIRST/MAIN戳。没有跨SM减clock64，也没有以ns插值cycles。

AlongM首输出T3/T4的native L响应中位为22.951%/29.652%，十trial中位全部为正，1320个逐窗口cycle增量也全部为正；single126首36 CTA为26.004%〔24.064%,27.319%〕，360窗口全部正，而single36中位仍为0。persistent132首36为20.356%〔16.102%,24.164%〕。这些强响应在两个时钟域均存在，仍是候选需解释的开发约束；这不识别具体资源，也不补足未扰动执行的局部观察资格。

AlongN第二后输出j2/T3则不同：ns中位+2.657%，native中位−0.203%，十trial的native中位全部负；不能把这个ns小增量直接拟为相同的cycle供给罚项。逐窗口还保留`(1+r_ns)×(1+r_proxy)=1+r_cycles`，其中proxy为两臂`cycle/ns`的比值变化；此恒等式不对不同中位数成立，proxy也不是物理频率或DVFS测量，不用于修正数据或预测。

对当前事件图，所有时间参数统一乘正数λ，`max`、加法、严格订阅先后和同刻插入次序均保持，端点也乘λ，B32/B0的L比值不变。因此统一从ns换算cycles不能修复本候选首输出的零响应；回归检查以精确二倍缩放核对事件、服务工作和订阅代次。改变真实频率且源/目的/TC服务不等比例变化属于另一未识别机制，不包含在这个换单位结论中。新的first-output控制失败保留，既不升级旧R10观察资格，也不自动抹去其native响应。

<a id="startup-absorption"></a>

### 当前在途共享候选吸收了512ns起步变化

在相同2304×3072×1024、AlongM、stage6、GPU-ef源程序上，固定旧覆盖价候选及32/128B地址单元候选的既有参数，分别计算B0/B32的原入口、仅CTA55理想后移512ns、全部CTA共同后移512ns。共18次模型计算、373248个逻辑输入需求，没有拟价、扫描入口或使用实测起步戳。六个原入口结果的全部432输出窗口精确复现此前报告；共同平移后九类消费/释放端点的最大平移误差为2.92×10⁻¹⁰ns以内，仅作浮点算术核对。

三候选的CTA55最初六个stage均有18个完整输入需求，它们全部订阅CTA1/54已在途的源。最小订阅余量为768.945ns，后移512ns仍严格早于同一owner的截止；首slot释放相对原预填发出结束的余量为1220.862～1284.343ns，producer也没有因此晚于首次refill。重算后，全部132CTA的ready、TC推进、commit、退休、释放及MAIN数组均精确不变，源代次记录也相同。这里描述的是这些固定模型的状态，不是硬件容量或任意入口扰动区间的证明。

| CTA55首输出变化，ns | 三候选B0/B32均预测 | 739619实测B0，中位〔范围〕 | 739619实测B32，中位〔范围〕 |
|---|---:|---:|---:|
| entry→FIRST | 0 | +1168〔1024,1344〕 | +1328〔1120,1408〕 |
| entry→MAIN | 0 | +1312〔1216,1536〕 | +768〔512,928〕 |
| producer→MAIN，即S+L | −512 | +816〔736,960〕 | +288〔0,416〕 |
| FIRST→MAIN，即L | 0 | +160〔−32,320〕 | −512〔−864,−448〕 |

模型中的首full不变，故首组`issue_end→full`缩短512ns。相关739653独立首代可见性区间却给出`issue_end→ready`增量：B0十次均为正，中位下/上界704/832ns；B32亦十次均正，为752/912ns。这是不同观察协议的额外开发约束，不能和739619合成同binary验收或物理源延迟。native L的739619配对变化B0为+377cycle〔248,580〕、B32为−696.5cycle〔−1274,−580〕，原时钟域分别保留，不用目标频率换算模型价格。

[起步吸收复核](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-producer-delay/reanalysis/B-20261011-startup-absorption-v1/B-startup-absorption.json)保存每个初始订阅的owner、截止与延后到达时刻，以及全部原端点汇总。800份既有数据输入重新核对SHA，数值/观察报告在核对后复用，本次没有重新执行数值重放。完整目的与D工作保持；同刻事件插入顺序仍使少量后续D源buffer完成时间变化1.592～58.758ns，报告逐项保留，没有把消费数组不变扩大为全状态不变或D全局可见。

当前参数点的缺口已出现在首次供给，不能只补MAIN末尾常量或把原零L响应解释为TC覆盖。下一状态机制还须解释producer迟发后full为何变晚，而不是仍被共同在途订阅吸收。该结果只检验三组既有候选及一个理想512ns根变化，不排除其它入口、缓存或仲裁状态；实际计时循环、跨进程初态及观察资格的限制继续保留，无新GPU或独立留出。

### 每需求最短路径可以打破吸收，但固定常数仍不足

`shared_stage_events`新增可选`input_min_latency_ns=τ`：目的FIFO入口为`max(source_ready+destination_latency, demand_issue+τ)`。它保留共享源与本需求自己的两条前驱；晚订阅者可在源已完成后继续等自己的路径，但不延长整box或地址单元的pending，也不增加源服务工作。默认τ=0保持旧结果。三需求的语义反例中，共享源完成分别为30/30/65ns，τ=20使目的入口为30/45/65ns；一个共同目的后程无法同时保持这些值。因此它与此前已证等价的源/目的共同延迟搬移不同，仍不代表硬件已存在该最短时间。

先令τ等于既有`read_latency_ns=754.255`，不另拟价格。三候选×B0/B32×0/512ns起步共12次模型、248832逻辑需求，首full/FIRST及MAIN开始响应迟发：B0三候选均增加437.099ns；B32的覆盖价/32B/128B候选分别增加373.618/437.099/426.082ns。首L保持在浮点精度内不变，producer→FIRST反而缩短74.901～138.382ns，仍没有解释739619中S增加512～896ns的方向。

这一不足不需要再拟τ。设初始三份源完成为R_A/R_B0/R_B1，需求发出偏移为a_A/a_B0/a_B1，目的服务为A72ns、B各36ns。在原初始源池和每CTA FIFO组织下，A/B0/B1顺序保留，其它初始copy均在B1之后进入，得到：

\[
V=\max(R_A+144,R_{B0}+72,R_{B1}+36),\qquad
H=\max(a_A+144,a_{B0}+72,a_{B1}+36)=144,
\]
\[
\mathrm{FIRST}(\delta,\tau)=\max(V,\delta+\tau+H),\qquad
0\le\Delta\mathrm{FIRST}\le\delta,\quad\Delta S=\Delta\mathrm{FIRST}-\delta\le0.
\]

该必要界覆盖任意非负**共同常数τ**，限定根变化`0≤δ<580.200ns`：δ加上六stage最后一份初始发出必须早于754.255ns的最小源延迟；18个需求也都在原owner截止前，consumer反馈不能改变该初始源池。739619的20个原配对producer变化均在这个条件域内，S在ns与native cycles中却20/20为正。实测起步戳只用于核对必要界的条件域，没有作为模型预测入口；真实干预与跨进程隐藏初态的限制继续保留。

[每需求路径与必要界](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-producer-delay/reanalysis/B-20261011-input-latency-v1/B-input-latency.json)保存初始源、FIFO表达式与全部配对条件。809份输入SHA核对，旧数值/观察证据只读复用；另以真实前版e541c478动态加载对照12份默认完整结果字典，包含trace、源、目的、actor和输出，全部相同。当前固定源池下的共同常数τ族仍不采用，不拟τ或扩GPU矩阵。下一机制需要改变源/投递的共享忙碌状态或其它前驱；本界不排除这些变化，也不扩展为全部供给模型的不可能性证明。

### 共享队列的位置必须同时解释迟发与原首窗

下一步先做必要前缀检查，没有直接增加完整模拟器参数。保留B0、原GPU-ef参数及每CTA的A72/B各36ns本地投递；新的全局FIFO对每个目的需求分别支付σ×名义工作，σ单位为ns/16KiB逻辑目的工作，不是物理带宽。比较两种位置：回程队列在`max(source_ready, issue+τ)`之后；独立请求路径队列在issue之后，先串行完成，再加τ，与共享源完成汇合后进入本地投递。后者没有推迟原源发起，仍是两条并行前驱，不称已识别的硬件descriptor队列。

初始六stage共2376需求、1584个16KiB工作单位。全部初始投递入口最晚1900.081ns，最早refill源完成的保守下界为2148.393ns；Ktile6的源身份也与初始集合分离。因此任何非负σ下，未来refill都不能插入此前缀。对原首输出的每个初始组g，还必须满足`L≥full_ready(g)−FIRST+(16−g)×288ns`。只用既有B0首L的最大值5504ns作宽松上界，两个位置均给出必要价格域：

\[
0\le\sigma\le\frac{292}{165}=1.769697\ \mathrm{ns}/16\mathrm{KiB}.
\]

高价格分支中，第六full与首full相隔1320σ，故首L下界为1320σ+3168ns；超过上述价格就先违反原首窗。所有低价格分支也由精确分段上包络检查，未用网格扫描代替全域。

| 队列位置 | 512ns理想迟发下，价格域内最大S增量 | 同时满足已见S范围512～832ns的必要价格区间 |
|---|---:|---|
| 源完成之后的回程FIFO | 344.582ns | 空集 |
| 与源并行的独立请求路径FIFO，τ取原read latency | 1979.782ns | 0.698602～0.917595ns/16KiB |

回程FIFO的结论还覆盖`0≤τ≤read_latency`：原入口的初始需求均由源完成约束，改变该范围内的τ不会改变原入口；迟发目标的copy顺序保持，增大τ只能推迟其首full，因此τ取上限给出最大响应。另逐一把十个B0实际producer位移放入条件界，允许任意域内σ，最大S仍分别比对应记录少115.200～487.418ns。实际位移只用于后验必要界检查，没有作为自由预测输入，也没有把不同进程当作隐藏初态相同的硬件证明。

[共享路径前缀分析](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-producer-delay/reanalysis/B-20261011-shared-path-prefix-v1/B-shared-path-prefix.json)保留全部2376需求、价格区间与各试次条件上界。计算以有理数精确表示已保存模型浮点值，在所有分段边界/内点作236次独立直接FIFO复算，全部一致；806份输入SHA核对，旧数值证据只读复用。这是给定开发参数的数学检查，不授予硬件计时精度或观察资格。

该前缀分析排除了当前回程FIFO族；独立请求路径只通过必要筛选，完整生命周期检验见下节。τ超过原read latency、不同队列分区/仲裁和源初态变化仍不在回程排除范围内，无新增GPU请求。

<a id="request-path-complete"></a>

### 独立请求路径接入后，一个前缀可行点仍未通过联合约束

`shared_stage_events`新增可选`input_path_service_ns=σ`，让每个逻辑输入需求在issue时进入独立全局FIFO，源订阅者也分别支付σ×名义需求工作。路径完成再加`input_min_latency_ns`，与源分支取max后进入本地投递；源发起及pending截止不被新队列直接推迟，后续发起只经目的ready、消费及有限stage释放反馈。新队列不服务D输出，源读取、请求路径、目的投递和D继续分账；默认σ=0保持旧结果。

先固定此前必要区间的算术中点σ=0.808098282ns/16KiB，τ仍取旧read latency，源使用已有128B地址单元批服务。这个开发点只由已保存B0前缀区间选取，没有用后面的B32、方向、规模或空间结果回调。完整计算包括10个R10条件、两个CTA55理想起步512ns条件及R13两stage×预等开关，共16次模型、384192逻辑需求。

| 检验 | 该固定点预测 | 已有观察约束 |
|---|---|---|
| B0起步512ns，S变化 | +684.238ns，符合独立前缀表达式 | +656ns〔512,832〕；仅开发相容 |
| B32起步512ns，S/L/S+L变化 | +673.220/约0/+673.220ns | +848/−512/+288ns；L十次均缩短448～864ns |
| AlongM首输出T3/T4的B32 L响应 | 0%/0% | 21.63%/27.68% |
| single126/persistent132首36 CTA的B32 L响应 | 0%/0% | 23.36%/20.40% |
| AlongM后输出各j/T的B32 L响应 | 0.062%～0.888% | 5.30%～20.12% |
| AlongM B32首输出T3/T4的L空间跨度 | 浮点精度内为0 | 中位1504/1648ns，native为2569.5/2735cycle |

请求路径确实改变了起步响应，不能再说新队列没有反馈到ready；但首输出TC仍覆盖了该点的后续输入变化，强B32响应与空间离散继续缺失。[完整生命周期报告](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/B-20261011-input-path-v1/B-input-path.json)保存全部17组配对、逐CTA first/later和空间跨度；同一价格的[R13联合检查](R13-async-retirement.md#request-path-joint)也失败。因此本点不采用，B0起步局部相容不能抵消其它失败。这个单点结果本身不能排除整个前缀价格区间；下节另给保守界。

新增7项语义测试核对每需求独立付费、并行分支、源先于路径过期、D不进入新队列及有限stage；172项R13和40项cfg_a/c/R10回归通过。真实前版043d06b8与新版零路径服务的24份完整字典一致，覆盖两种floor、两种起步和三种源候选；另10个原地址候选的2916窗口精确复现。1023份输入SHA核对，旧数值/观察证据在核对后复用，没有新增数值重放、GPU、观察资格或独立留出。

<a id="request-path-interval"></a>

### single126在原必要价格闭区间内始终无首窗停顿

固定上一节的源状态、128B批服务、τ和全部旧参数，只让σ遍历原B0必要闭区间〔0.698601908678,0.917594654788〕ns/16KiB。single126 B32给出了不依赖价格采样的条件排除：126个首输出均有L=16×288=4608ns，而同图B0的L不能小于4608ns，因此该模型不能产生已有single126正B32响应。结论只针对这个固定请求FIFO族及原必要区间。

初始六组的发起时间按原图取`g*(2*operand_gap+producer_gap)+atom*operand_gap`，两个gap都保留10.2385407698ns。初始源服务、独立请求FIFO和每CTA目的FIFO写成精确max仿射包络；每个分段交点和闭区间端点都纳入极值计算。初始g1～g5的`FIRST+g*288−ready[g]`全区间最小值为56.766146993ns；FIRST的统一界为968.372182732～1030.465681997ns。aligned B0初始128B单元在不同描述符间无部分共享，每个A/B盒分别含128/64单元，故此前整盒前缀必要区间仍适用于当前源批服务。

后续g≥6用首次停顿反证。若此前没有TC空闲，则普通释放固定后续A发起为`FIRST+(g−4)*288−144`，B1晚两个operand gap；`producer_gap`已计入上一组赶得上释放的条件，首refill相对初始producer_free仍有1216.078ns余量。不同CTA同一K组的全部发起跨度至多82.570580805ns，小于754.254959883ns的源订阅下界。16个K组地址单元两两不交，因此同组每单元至多付费一次，费用至多3248×unit_price=186.380392981ns；这只是模型中的地址单元账本，不是物理流量。

| 供给上界所需条件 | 整个闭区间的保守余量 |
|---|---:|
| 源服务每组小于288ns | 101.619607019ns |
| 请求路径每组至多252σ，小于288ns | 56.766146993ns |
| 本地目的投递每组144ns，小于288ns | 144ns |
| 初始源／请求队列在最早refill前清空 | 282.089824844／12.969064692ns |
| 后续组ready赶在FIRST+g×288前 | 83.940606305ns |
| 最早D发起晚于全部首输出输入发起 | 1501.429419195ns |

设Fmax/Fmin为上述FIRST统一界，Vs/Vp为源／请求每组费用上界，d=72、o为operand gap。两个全局FIFO每组费用均小于288ns，且目的FIFO每组费用也小于一个计算周期，故后续组有

\[
ready_{c,g}\le F_{max}+(g-4)288-144+2o+\max(V_s,V_p)+\tau+2d.
\]

同组跨度小于288ns，保证不同K组的发起不交错；两条全局FIFO及互不相交的源地址又保证下一组的准入不早于当前组的全部准入。因此，较后组不能插入当前组的目的投递，目的FIFO每组144ns的递推上界成立。右侧比`Fmin+g*288`至少早83.940606305ns，与首次TC停顿矛盾。证明包含有限stage、源共享、请求及目的积压；D尚未排入源队列，不能改变这段供给。它无需固定同组源所有者或全部全局事件排序。

[闭区间证书](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/B-20261011-input-path-bound-v4/B-input-path-interval.json)保存630项初始余量、全部包络、12项保守条件及主线两端完整图的独立核对。计算采用既有二进制输入常量的精确有理数；两端逐CTA FIRST/初始余量与原浮点图误差小于1e−7ns，中点亦匹配，后者不被当成IEEE全区间舍入证明。新增4项源到期／非零producer gap／包络／失败保护测试，176项R13测试通过。

此前完整事件总序只证明中点附近宽8.42e−7的一段，偏序实验也仅得到局部cell；它们及全部未覆盖域原样保留，未靠省略排序约束取得本结论。新的保守界另行完成闭区间证明。这仍是已揭示数据上的固定模型筛选，没有升级观察资格、独立留出或其它请求工作／队列分区／仲裁／源初态家族。

<a id="request-work-identifiability"></a>

### 覆盖工作向量本身没有解释方向差异

静态复核14个当前／旧R10／已揭示G5条件，遍历全部CTA、输出、K组的305856个逻辑输入需求。34176项描述符×宽度检查用不相交行跨度计数与现有32/128B地址集合互核；源有效负载、目的投递和D账本不变，没有接入新计费API、选择服务价格或运行新时间候选。

| 每完整K组，A／B0／B1 | aligned | B32 |
|---|---:|---:|
| 名义有效负载，KiB | 16／8／8 | 16／8／8 |
| 32B覆盖单元数 | 512／256／256 | 512／256／256 |
| 128B覆盖单元数 | 128／64／64 | 128／112／112 |
| 按逻辑需求求和的128B覆盖，KiB | 32 | 44 |

B32的64行B盒中，16行起点对齐128B，另48行跨两个单元，故每盒112个。44KiB是每个逻辑需求各自覆盖后求和，包含不同盒和不同CTA的重叠；它不等于源并集或物理流量，不能据此直接确定硬件请求数。

AlongM／AlongN在B0下的完整工作张量逐项相同，B32下也逐项相同；比较包含每个CTA的总输出数、全部输出、16个K组和三输入原子，并非只比较中位特征。已有首窗T3/T4的ns响应分别为AlongM 21.633%／27.684%、AlongN 0.286%／0.289%；native cycle为22.951%／29.652%与−0.216%／−0.168%。因此，只用这些工作张量和相同初态的确定性规则无法区分方向。两方向的源身份关联不同，带源身份、发起顺序或共享状态的完整图不在这个排除范围内。

single36与single126在B0、B32下，前36个CTA的完整16K组局部源描述符逐项相同，两边每CTA都只有一个输出；已有首窗ns响应却为0%／23.365%，native为0%／26.004%。当前局部地址和工作量相同仍不够，必须表达其它CTA请求怎样改变共享状态；该对照不唯一定位缓存、请求分区或SM仲裁。

旧Bcurve与当前持久132条件的工作计数张量相同；G5每K组也有相同的覆盖计数，但K、每CTA输出数、源关系和卡均有区别。这里仅连接工作定义，旧卡响应与ef卡保持分开，不转移价格或回填V09。原始2600×3000×2000还有M/N/K尾部，当前`cfg_b_source_program`要求完整Ktile、有效单播盒和完整输出，仍未支持该旧几何；不能通过覆盖计数删掉fill或部分D工作。

[静态审查与完整工作张量](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/B-20261011-request-work-audit-v3/B-request-work-audit.json)保存逐条件账本、碰撞证明、旧响应及未支持项。下一机制需要明确源身份和其它CTA请求如何改变请求形成／队列状态，再给出能区分已有方向、规模或起步干预的预测；静态覆盖本身没有确定这个映射。本节没有新增候选评分、GPU、数值重放、观察资格或独立留出。


## 原 B 行距对照：配置之间不能共用一个罚项

固定 M×N×K=2600×3000×2000，FP16 输入、FP32 累加/输出，α=1、β=0；A 的 lda=2000、D 的 ldd=3000，只改 B 的 ldb。job735876，a053/GPU-54896349，CUDA12.9、sm_90a、CUTLASS3.9.2。每点十个独立 plain 进程。

| 配置 | tile / schedule / cluster / stage | ldb3000 µs | ldb3008 µs | ldb3072 µs |
|---|---|---:|---:|---:|
| cfg_a | 128×128×64 / cooperative / 2×1 / 6 | 66.400 | 62.944 | 63.328 |
| cfg_b | 128×128×64 / pingpong / 1×1 / 6 | 76.272 | 58.272 | 57.136 |
| cfg_c | 256×128×64 / cooperative / 1×2 / 4 | 54.592 | 54.544 | 54.016 |

实际字节行距为 6000/6016/6144 B。cfg_b 从 ldb3000 到3008 减少 18.000 µs，cfg_a 较小，cfg_c 几乎不变；不能把差值单独命名为 TMA 等待，也不支持所有配置统一需要 128 B 行距对齐。

v2 全网格 trace 的 cfg_b3008/3072 有负向扰动超过 5%，不采用其周期；cfg_a/c 七点合格。最终 cfg_b 用 v4 对称准备、CTA0–3/consumer0 的 last-tile 局部 trace：

| ldb | v4 plain µs | 最大绝对配对扰动 | 周期/Ktile：局部池化中位 | 周期/Ktile：先按进程中位再汇总 |
|---:|---:|---:|---:|---:|
| 3000 | 76.160 | 2.068% | 854.531 | 869.906 |
| 3008 | 57.904 | 2.913% | 671.359 | 671.492 |
| 3072 | 57.456 | 3.775% | 659.047 | 662.812 |

局部周期与同档完整时间方向一致，但不替代全 grid 分布，也不与 v2 池化。v3 的3072仍有6.202%扰动，仅作诊断。完整时间包含 warm GEMM 和输出；trace 从首个 full-barrier 返回后至 mma_tail 完成，除以32个 Ktile，包含等待/循环/排空，首次输入等待在窗外。

实际 stride 同时用于初始化和 tensor map，B padding 填65504且计时外完整检查。v2/v3/v4 共300进程、1228800个保存值经未参与实现者逐K整数参考复算，误差0；数值见证不声明任意输入误差标准。[独立证据清单](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reviews/ROOT-independent.json)。历史准备、v1–v4检查及命令见[固定旧正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R10-layout-cache.md#时间抽样与数值边界)。

<a id="legacy-boundary-ledger"></a>

### 原始尾部的有效源、目的fill与D范围

`cfg_b_logical_boundary_ledger`单独展开旧2600×3000×2000的逻辑输出坐标和K组，不给它们指定历史CTA、输出序号或全局stage。原`cfg_b_source_program`的完整tile约束保留，旧几何仍未进入时间图。归档源码确认128×128×64主循环tile、64×32 FP32 epilogue tile；60个plain/trace记录的六stage、132CTA和215040B SMEM与旧独立审查哈希一致，只复用已有数值状态。

每条件为21×24=504个逻辑输出、每输出32个K组，最后一组只有16个有效K元素。三种B行距的以下账本相同，单位B：

| 项目 | 每次调用 |
|---|---:|
| 逻辑输入原子A／B0／B1总数 | 48384个 |
| 源有效字节，保留不同逻辑需求的重复 | 501600000 |
| 目的填充字节 | 26882304 |
| 目的完整事务期待字节 | 528482304 |
| 去重逻辑源盒的有效字节 | 22400000 |
| D有效全局输出 | 31200000 |
| D名义盒范围／超出全局输出范围 | 33030144／1830144 |

每条件672个B1逻辑盒没有有效源地址，仍保留8192B目的填充义务。最后M/N/K角点的A／B0／B1有效字节为1280／1792／0，合计3072B；目的fill为29696B，与有效部分共同保留32768B组事务。D有178个逻辑子盒完全超出有效输出；名义盒范围不证明这些OOB盒都发起或读取SMEM，D源缓冲与完成图仍需另外绑定。

[PTX8.8](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)规定tensor边界零填充和mbarrier事务完成语义；归档CUTLASS的`TmaTransactionBytes`按完整SMEM布局计算，经kernel参数传给`arrive_and_expect_tx`，未按最后K组或M/N尾部缩减。这里把有效源、目的fill和软件期待量分开，不把有效源字节当作full barrier预算。

旧行距还有一个不能转移新几何证明的差别：普通完整K组的128B覆盖计数A／B0／B1为ldb3000的224／120／120、ldb3008和3072的224／64／64。A行距4000B使相邻K组的地址覆盖并集共享1950个128B单元；三个布局均如此，B对应交集为0。这只是几何重叠，不证明实际发起时仍可订阅或驻留，但已不满足前文新几何的跨K单元不交前提。归档probe又按`K*ldb`分配B，三档容量不同，不能作为固定容量的新卡控制来转移价格。

源坐标中的K组可以归零，实际ring状态不能随输出归零。归档producer按`k_tile_count`推进、consumer按两WG跨度推进；绑定公式为`global_group=output_ordinal*32+logical_k_group`。六stage下，假设首输出的group31位于slot1，下一输出group32应在slot2，不能把它重新当成group0。这里仅保存未绑定模板，没有推测旧CTA的真实输出顺序。

[边界账本](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/B-20261011-boundary-ledger-v4/B-boundary-ledger.json)保存三条件逐盒数据、源身份、目的义务、D范围、归档源码位置和全部哈希。36项普通／角点逐FP16元素枚举核对了覆盖和有效字节，5项新增边界／完整tile兼容性测试通过；181项R13测试通过，剔除新增函数后现有模块AST与6c5015b2完全相同。未拟合源/fill/D价格、未绑定就绪／释放／退出时间，也未升级GPU证据、观察资格或独立留出。

<a id="legacy-persistent-binding"></a>

### 原始调度、输出序号与输入ring绑定

在上述未绑定账本之外，`cfg_b_persistent_boundary_ledger`要求显式调度参数后再赋予CTA、输出序号、consumer WG和输入ring状态。归档FP16／FP16 builder选择SS主循环，不交换A/B；probe省略scheduler模板与参数，归档默认链为`void → PersistentTileSchedulerSm90`、`max_swizzle_size=1`、`Heuristic`。21个M tile、24个N tile因此取`AlongM`，cluster1不增加padding；保存的132CTA与grid公式共同确定`(132,1,1)`。这里从归档默认值推导，没有借用后续R18的显式sw1设置。

源码绑定为`linear=c+j*132`、`(mi,ni)=(linear%21,linear//21)`。CTA0–107各有4个输出，CTA108–131各有3个；WG0消费偶数j，WG1消费奇数j。边界在各输出序号上的分布如下，单位为输出tile个数：

| 输出序号j | 完整M/N | 仅M尾部 | 仅N尾部 | M/N双尾部 |
|---:|---:|---:|---:|---:|
| 0 | 126 | 6 | 0 | 0 |
| 1 | 126 | 6 | 0 | 0 |
| 2 | 126 | 6 | 0 | 0 |
| 3 | 82 | 5 | 20 | 1 |

旧trace只保存CTA0–3的WG0最后一次主循环，因此对应j=2，源码推导坐标依次为`(12,12)`、`(13,12)`、`(14,12)`、`(15,12)`。它们都是完整M/N输出，每个输出含1024000B有效输入和24576B目的fill，fill只来自最后K组。旧trace没有坐标字段；120个保存CTA的WG0计数均为2，与该绑定兼容，但计数2本身无法区分总输出数3与4。SMID没有参与坐标或并发推断。

每个保存窗对应global K组64–95，而非重新开始的0–31。六stage下首组为slot4、generation10、full／empty wait参数0／1；末组为slot5、generation15、wait参数1／0。producer每输出推进32组，两个consumer分别从0／32开始、各次推进64组。这些是软件等待参数，不是实测barrier相位，也不提供就绪或释放时间。旧trace起点仍位于首个full-barrier返回后，首次输入等待在窗外。

[静态绑定报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/B-20261011-boundary-binding-v1/B-boundary-binding.json)保存三条件完整压缩账本、各输出序号的边界分布和全局K组源覆盖并集／交集。15份实际使用的归档源码与原`source_hashes.json`一致，60个记录SHA与旧独立审查一致；三条件48384组ring参数通过独立逐次推进核对，源身份／有效字节／fill／D范围逐盒保留。新增5项测试后186项R13测试通过；剔除新增绑定函数后，供给模块AST与f521a01f相同。旧几何仍未进入时间图，D源缓冲与完成依赖也没有因此绑定。

<a id="legacy-tail-lifecycle"></a>

### 最后K16的完整MMA工作与stage释放

归档SS builder对cfg_b选择`m64n128k16` FP16／FP32 atom，`AtomLayoutMNK`为1×1×1；128×128×64主循环tile展开为2个M片×1个N片×4个K片，共8次atom调用后commit一组。展开尺寸来自静态tile，不随有效M/N/K缩小。最后K组虽只有16个有效K元素，仍保留8次调用，其中6次覆盖填充的K区域；M40/N56/K16角点仅1次调用含有效数学工作，其余调用也不跳过。这里统计源码调用工作，不把它当作新实测吞吐或硬件周期。

每个逻辑输出仍执行32组，以下计数不因三种B行距或边界变化：

| 软件工作／义务 | 每输出 |
|---|---:|
| 完整MMA atom调用／commit组 | 256／32 |
| full等待API调用，每consumer线程 | 32 |
| 普通`wait_group<1>`后释放前一组 | 31 |
| `mma_tail`中`wait_group<0>`后释放末组 | 1 |
| 每K组full事务期待量 | 32768B |
| 每组empty完成所需的线程到达 | 所属WG的128次 |

`K_PIPE_MMAS=1`。q0的prologue提交后不释放上一输出的组；q1–31各自在提交当前组、`wait_group<1>`返回后释放前一组。最后q31由`mma_tail`的`wait_group<0>`排空后释放。[PTX8.8的WGMMA等待语义](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)保证比最近N组更早的组完成，N=0排空全部组。empty barrier的128次到达来自当前拥有该组的一个WG，不应按两个交替WG改成256次。

源码在`mma`返回后先通知下一WG的MMA顺序，再调用`mma_tail`；MMA和epilogue使用ordered barrier的两个独立stage。S6/Kt32下，下一输出q0–4的acquire分别依赖前输出q26–30的普通释放，q5才依赖前输出q31的tail释放。旧j2末组g95因此限制下一输出g101的同slot复用，不能把它默认添加为g96首组的MMA顺序前驱。这里没有宣称实际发生重叠；首组仍需满足自己的full等待。

既有`shared_stage_events`中的`ready/starts/commits/completed`、普通`wait1_return/released`、分开的`mma_returns/main_ends`和同WG的j+2输出约束可以复用其结构。数值含义仍有缺口：零源B1仍需目的fill，但现有地址单元路径要求非空源单元；释放时刻也尚未独立分开128次empty到达与acquire返回。`compute_ns`、commit lead、retire lag与handoff价格不能由完整调用数确定。

归档producer在最终持续状态再检查六个empty slot。T3的检查覆盖组90–95，T4覆盖122–127；更早的释放已经被后续同slot acquire要求，二者合起来恰好覆盖全部输入组。既有时间图没有显式producer-tail／输入安全退出节点，不能把其返回的end直接当成安全退出证明。输入stage释放也不等于D完成：C接口仍需明确自身`mma_tail`与epilogue顺序之后的入口、每输出`store_tail`返回、128次epilogue顺序到达、D源读取完成与全局可见，以及同WG的j+2许可。

[生命周期审计](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/B-20261011-boundary-lifecycle-v2/B-boundary-lifecycle.json)保存10份归档源码的原manifest核对、既有阶段图位置、T3/T4释放／尾部检查账本和MMA角点展开。每条件16128组、129024次atom调用，其中15624组普通释放、504组tail释放；有效数学量为31200000000FLOP，完整tile名义量为33822867456FLOP。新增6项静态测试后192项R13测试通过；供给模块和原full-tile guard未改，旧几何仍明确拒绝进入原时间入口。未重做局部计量归一化、未定价、未执行GPU或新增资格判定。

<a id="legacy-completion-ports"></a>

### 零源盒与目的完成端口

`completion_fragment`把已绑定账本展开为无时间的完成条件。每组从外部`empty_acquired`端口开始，经`arrive_expect_tx`和`acquire_return`发起A、B0、B1；A发起先于B，两个B盒的相对顺序不作断言。每盒有有效数据时保留availability／valid delivery条件，有fill时保留fill delivery条件，二者共同构成该盒的destination completion。full合流始终保留`arrive_expect_tx`及三个目的完成义务。

零源B1不建立有效数据分支，但仍保留`submit → fill_delivered → destination_complete → full`，其中fill为8192B。四个事件的完成时间均未赋值；valid／fill拆分只表达逻辑条件，不声明独立硬件队列或可分别观测的微事件。[PTX8.8](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)的tensor零填充和mbarrier完成语义，结合归档CUTLASS的完整事务期待量，要求保留该目的义务。

每条件48384个目的盒中，有47712个非空有效源义务、3558个fill义务，含672个零源B1盒。两种看似简化的转换与归档账本不一致：

| 转换 | 少计的K组数 | 少计目的字节 |
|---|---:|---:|
| 将full预算改为有效源字节 | 1868 | 26882304 |
| 删除零源目的盒 | 672 | 5505024 |

三个行距抹去实际地址／stride后，CTA／输出／K／ring状态、完成拓扑、valid／fill工作和2208个逻辑源身份的等价类完全一致。相同参数与初始状态、只读取这份投影的模型不能区分三个行距；这一结构结论不排除地址相关的工作、共享、仲裁或其他状态。完整源身份仍保存在各盒中，相同身份不会自动合并完成节点或确定物理读取owner。

[完成端口报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/B-20261011-boundary-completion-v2/B-boundary-completion.json)保存三份逐组压缩端口账本、预算反例和相同的工作／身份拓扑投影哈希。4份实际使用的归档源码与原manifest一致，12个输入SHA全部复核；新增6项测试后198项R13测试通过，供给模块与ff4a7423逐字节相同。端口没有闭合empty到达、consumer释放、producer tail或D完成；主集成负责后续依赖组合，无价节点不能按零时间求解，旧时间入口与guard保持不变。

<a id="v08-stride"></a>

## V08F：分开行距和尺寸尾部

job737322，a043/GPU-099dda56，CUDA12.9.41、CUTLASS3.9.2、sm_90a、NDEBUG。V08 留出后追加84条件，plain/stamped/ends各十进程，属于开发解释，不回填 V08 判定；不与上文不同卡数据池化。参照 M=2304、N=3072，每次只改一项。

| 改动 | cfg_a 完整时间变化 | cfg_b | cfg_c |
|---|---:|---:|---:|
| K=1000，A 行距仍2048 B | +0.3% | −0.6% | −0.7% |
| N=3000，B/D 行距仍对齐 | −0.1% | +4.7% | −1.2% |
| D 行距12320 B | +0.1% | +0.2% | +1.9% |
| A +16 B，K1024 / K4096 | +7.7% / +8.7% | +15.0% / +16.4% | +0.2% / +0.3% |
| B +16 B，K1024 / K4096 | +4.7% / +6.2% | +28.2% / +33.7% | −0.4% / +0.8% |
| A 行距2000 B＋K尾部 | +7.3% | +17.1% | +0.1% |
| B/D行距6000/12000 B＋N尾部 | +2.7% | +31.5% | −0.2% |

这组数据里尾部本身和 D 行距影响较小，A/B 行距显著影响 cfg_b、其次 cfg_a，cfg_c 普通 tile 接近计算下界。这里的尾部是 M/N/K 不整除 tile，不是 epilogue 或最后完成 CTA。

同 cfg/同 K 对齐参照下，后续主循环 L 的进程内均值再跨进程中位变化如下；first 是每 CTA 首输出 tile，另列：

| 单轴条件 | cfg_a | cfg_b | cfg_c |
|---|---:|---:|---:|
| A16，K1024 | +2.00% | +12.88% | −0.03% |
| A16，K4096 | +6.00% | +13.99% | −0.004% |
| B16，K1024 | +0.74% | +36.11% | +0.01% |
| B16，K4096 | +6.12% | +33.74% | +0.001% |

微小正负值只表示当前精度内未分辨。cfg_a 的 K 依赖同时改变输入工作集，不能归为固定请求放大；cfg_b 的 B16 代价比 A16 大，不能共用操作数单价。

<a id="v08-stride-next"></a>

### 配置与位置反例

K4096 对齐/A16/B16 的 L/Ktile 为 cfg_a **548.933/581.891/582.535**，cfg_b **546.456/622.912/730.853**，cfg_c **1029.457/1029.420/1029.472**。cfg_a/b 计算项相同、逻辑源需求却为24/32 KiB；把 cfg_a 超额全解释成同一供给单价会把 cfg_b 推到约724.49 cycle/Ktile，与观测不符。共享单价缺少条件，不能从该矛盾选择某个 bank 机制。

cfg_a K4096 的位置分组更直接：

| 行距 | j1，132 CTA | j2，132 CTA | j3，36 CTA |
|---|---:|---:|---:|
| 对齐 | 579.27 | 526.39 | 515.77 |
| A16 | 629.96 | 553.29 | 515.78 |
| B16 | 627.00 | 557.61 | 515.77 |

单位 cycle/Ktile，逐进程对该位置求均值再取中位。最后一组已接近计算下界；j3 的 CTA 集合也不同，表中 CTA 数不是实际并发 SM 数。旧拟议 A/B64 等单轴点没有据此全部执行；详细候选与失败见 [R13](R13-async-retirement.md#v08-supply-fit)和[旧供给分析](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/A-20261009-supply-v3/summary.json)。

<a id="requested-sm-pitch"></a>

## 固定形状改变持久 grid：行距效应随工作位置变化

738101，a058/GPU-ef8692f3；cfg_a、2304×3072×4096、6 stages、sw1、dyadic seed17、重复同一 A/B。12条件为请求 SM 数32/64/96/132 × aligned/A16/B16，共240成功进程，983040个保存值通过回放。最大进程CV1.328%，完整时间打点变化最大2.343%；它不证明各局部区间无扰动。输入逻辑42 MiB、输出27 MiB不是 L2 驻留量。

| 请求/峰值 SM | CTA 总输出数 T | 末轮 CTA 数 | CTA 窗口平均重叠 | 包络 cycle/ns |
|---:|---|---:|---:|---:|
| 32 | 16 CTA×13；16 CTA×14 | 16 | 30.64–30.69 | 1.8038–1.8055 |
| 64 | 16 CTA×6；48 CTA×7 | 48 | 58.98–60.95 | 1.7193–1.7262 |
| 96 | 48 CTA×4；48 CTA×5 | 48 | 81.74–84.95 | 1.6717–1.6802 |
| 132 | 96 CTA×3；36 CTA×4 | 36 | 104.43–105.13 | 1.6647–1.6699 |

窗口重叠由同 SM CTA 生命周期求并集后计算，不是并发 TMA 数；包络 cycle/ns 也不是逐 tile 频率。grid 同时改变工作轮次、访问历史和频率，因此主比较为同档内按 CTA/j/T/坐标匹配的行距差。

| 请求 SM | j1 的两个 T | A16−aligned，cycle/Ktile | B16−aligned，cycle/Ktile |
|---:|---|---:|---:|
| 32 | 13 / 14 | −1.93 / −1.95 | −0.03 / −0.17 |
| 64 | 6 / 7 | +12.48 / +15.04 | +9.10 / +12.31 |
| 96 | 4 / 5 | +1.48 / +13.29 | +4.04 / +16.12 |
| 132 | 3 / 4 | +64.22 / +70.13 | +49.20 / +52.30 |

132档、T4 的 A16 在 j1/j2/j3 为 **70.13/17.43/0.009**，B16 为 **52.30/24.14/0.019 cycle/Ktile**。相同行距、K和总输出数仍非固定罚项。

96/132档在 T4、j1和j2各有24个相同当前 M/N 坐标。差分 `(variant−aligned)_132−(variant−aligned)_96` 的 A16 j1/j2中位为 **73.85/18.91**，B16为 **48.19/23.99 cycle/Ktile**；十次范围分别48.30～80.19、13.46～22.02、40.75～62.02、16.28～30.43。此前坐标和其他 CTA 访问历史仍不同，不能唯一定位 HBM/L2/SM 本地路径。候选拟合与低压力训练秩亏见 [R13](R13-async-retirement.md#requested-sm-pitch)。

## h03 联合行距：正交互不等于物理请求相互放大

738169，a043/GPU-099；cfg_b、2400×3000×1000、sw1，D ldd=3000固定，A/B容量固定为2400×1024和1000×3072 FP16元素；四角的实际 lda/ldb 为1024或1000、3072或3000，数值、尾部和grid一致。四条件×plain/wide/stamped/dual×十进程，655360个保存值通过参考回放。前一版738162按实际stride分配，容量不匹配，只保留[诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-joint-pitch-job738162/reanalysis/allocation-diagnostic-v2/joint-pitch.json)。

| A/B 条件 | plain µs | wide µs | dual µs | dual later cycle/Ktile | dual later ns/Ktile |
|---|---:|---:|---:|---:|---:|
| 对齐/对齐 | 29.888 | 29.840 | 30.960 | 566.379 | 334.935 |
| 未对齐/对齐 | 36.160 | 36.128 | 36.496 | 717.251 | 415.515 |
| 对齐/未对齐 | 37.920 | 38.048 | 38.736 | 779.464 | 447.485 |
| 未对齐/未对齐 | 46.672 | 46.832 | 47.216 | 998.478 | 561.861 |

后续窗口每进程先均值并除16，再跨十进程中位。wide/plain中位为−0.161%～+0.343%；dual相对同缓冲wide中位为+0.820%～+3.753%，但对齐最大单对+8.073%、仅B未对齐+5.378%，不能称所有trace通过5%。分项只属于dual观察窗口。

逐trial先算 `both−A_only−B_only+aligned`，十次均为正：plain中位 **2.528 µs**（范围1.344～3.008），dual later中位 **69.1169 cycle/Ktile、32.5586 ns/Ktile**。不能对四条件各取中位后再把差称为联合项中位。

trial0可由没有 A×B 交互项的计算/供给 max 解释：对齐被335.0556 ns计算锚点截住，隐含线性供给基线299.1235 ns，加A/B分别115.6296/146.7778 ns即可重现四点，正联合项恰为335.0556−299.1235=35.9321 ns。它是四点的可行事后解释，不是物理请求放大或新预测验证。全trial及实际坐标见[配对结果](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-joint-pitch-job738169/reanalysis/B-20261009-matched-joint-final/joint-pitch.json)。

<a id="bcurve-negative"></a>
<a id="局部冻结负结果job738203"></a>

## B 覆盖的冻结检验：两候选均失败

738203，a057/GPU-43269fbc；cfg_b、2304×3072×1024、A/D对齐、6 stages、sw1。五点共同分配 `alloc_lda=1024, alloc_ldb=3136`，仅B行距改变；每点四变体各十进程，共200成功进程、819200个保存值通过回放。plain/wide/stamped/dual均168寄存器、16条静态HGMMA、无spill/C7510。

| B增量 | ldb | 用途 | 额外32 B覆盖行数 | 额外128 B覆盖行数 | dual later ns/Ktile |
|---|---:|---|---:|---:|---:|
| 0 B | 3072 | 校准 | 0 | 0 | 329.9967 |
| 16 B | 3080 | 校准 | 32 | 56 | 439.1767 |
| 32 B | 3088 | 校准 | 0 | 48 | 385.9533 |
| 64 B | 3104 | 留出 | 0 | 32 | 365.7233 |
| 128 B | 3136 | 留出 | 0 | 0 | 331.7933 |

覆盖按64行×256 B B-box的有效地址跨度计算，不是物理事务数。目标为每trial后续 j≥1 主循环直接ns均值/16，再取十trial中位；首tile、完整时间和实测频率不进入目标。两候选分别使用32/128 B覆盖，形式为历史冻结的 `max(C,q0+q1*x)`，C只由对齐校准给出。只读p0/p16/p32后冻结，再采p64/p128；要求每候选的三个校准和两个留出最大绝对误差均≤5%。参数不唯一时按完整预测区间最差端点评分。

| 候选 | 校准最大误差 | p64预测/实测 ns | p64误差 | p128预测/实测 ns | p128误差 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 32 B覆盖 | 14.4983% | 329.9967 / 365.7233 | −9.7688% | 329.9967 / 331.7933 | −0.5415% | 失败 |
| 128 B覆盖 | 0% | 329.9967 / 365.7233 | −9.7688% | 329.9967 / 331.7933 | −0.5415% | 失败 |

32 B参数非唯一，但两候选留出预测均唯一，不能测后选择有利解。p128接近锚点不改变整组失败；被否定的是该冻结形式，不是识别出物理cache line。冻结SHA256 `6fd226ef90f1a9d9224d1d6db769744eaece3befd69e98abf4352c4676d072c5`保持原样；时序、身份和原只读位见[独立评分](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/B-20261009-frozen-negative/b-pitch-score.json)。

dual/wide完整时间中位扰动p0/p16/p32/p64/p128为+4.860%/+2.381%/+1.885%/+3.006%/+4.997%，单对仍有超5%，p0/p128最大+8.037%/+8.945%。所有有效trial保留，局部冻结只验证约定dual窗口。

### p64 失败位于多个后续输出位置

工作列表、CTA/j/T/坐标匹配后，每trial先取组均值再作p64−p0差；两个采样阶段不是同时调用。

| j/T | 每进程窗口数 | cycle/Ktile增量 | ns/Ktile增量 | 十trial ns范围 |
|---|---:|---:|---:|---:|
| 1/3 | 96 | 66.3281 | 37.2396 | 32.8333～42.1458 |
| 1/4 | 36 | 64.0026 | 35.8611 | 26.1111～40.3333 |
| 2/3 | 96 | 70.1533 | 37.7396 | 33.7708～40.8333 |
| 2/4 | 36 | 88.5694 | 47.2222 | 39.6667～61.6667 |
| 3/4 | 36 | 17.6771 | 9.4167 | 4.0556～12.6111 |

五组每trial的cycle/ns差均为正；j1/j2约贡献后续总增量的96.8%，不是只有最后tile变慢。p128同阶段接近p0；位置分组不足以命名物理请求放大、带宽下降或某个等待机制。

<a id="coverage-identifiability"></a>

## 覆盖组合的可辨识性与失败

以下均为测后开发，未改变上述冻结失败。[组合报告](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-composed-coverage-v5/summary.json)分别定值每张卡、每种观察协议，保存参数范围和逐窗口残差。

同时使用32/128 B覆盖时，原p0/p16/p32只提供两个独立服务等式，三个非负服务参数不能唯一确定；p64预测范围为 **329.9967～367.3011 ns/Ktile**，实测365.7233在其中。揭晓p64后能选定系数，只是已有可行集选点，不追认旧预测通过。已见0/+16 B的完整box恒有extra128=7×extra32；小拟合误差不能据此支持新覆盖比例。

| 数据 | 后续目标 | A/B共用覆盖单价最大误差 | A/B分开最大误差 | 活跃秩/参数数：共用；分开 |
|---|---|---:|---:|---|
| 432卡 B曲线 dual | 25组 | 16.33% | 16.33% | 4/4；4/6 |
| ef卡 cfg_a SM扫描 stamped | 144组 | 4.80% | 3.78% | 3/4；4/6 |
| 099卡 h03 dual | 20组 | 30.63% | 19.22% | 4/4；5/6 |
| 099卡 V08F stamped | 51个聚合窗口 | 18.14% | 8.66% | 5/6；6/8 |

h03 j1/T3 的 B-only 覆盖少于 A-only，实测却为833.682>700.956 cycle/Ktile，共用非负单价模型至少有8.65%最大相对误差；V08F对应下界9.33%。分开A/B后仍未消除位置差异。h03 aligned j3/T4比j2/T3更慢的反例还混入B部分OOB：前者60个窗口中19个含N尾部，后者没有；几何分组和填零解释见 [R18](R18-cluster-boundary.md#h03-partial-fill)，不能只据有效地址少而推定服务更快。

<a id="v09-pitch-comparison"></a>

## 旧 B32/B64 与 V09 B32：先比较几何，再谈对齐

两批都在a057/GPU-432，cfg_b、六stage、sw1、dyadic seed17，且各自组内固定容量、输入值与D行距；**跨批 M/N/K、总输出tile和波次数并不相同**：

| 条件 | 738203 B曲线 | V09 G5 |
|---|---|---|
| M×N×K | 2304×3072×1024 | 2304×2304×4096 |
| Ktile数 | 16 | 64 |
| 输出tile/grid | 432 / 132 CTA | 324 / 132 CTA |
| 软件工作轮次 | 96 CTA×3；36 CTA×4 | 72 CTA×2；60 CTA×3 |
| 对齐B行距 | 6144 B | 4608 B |
| B32实际行距 | 6176 B | 4640 B |
| B预留ldb | 3136 FP16元素 | 2360 FP16元素 |
| plain基线→B32 | 28.704→33.008 µs，+14.99% | 69.536→68.992 µs，−0.78% |
| B64 | 6208 B；31.440 µs，+9.53% | 未测 |

波次数由实际tile/grid分配计数，是软件工作组织，不是瞬时并发。旧B32/B64的dual later窗口也分别慢16.96%/10.83%；V09 B32的−0.78%只表示未见明显惩罚，不判为显著提速。这些结果否定“输入只要32 B对齐就不会慢”，却不能单凭跨几何对照认定32 B因素发生因果翻转。

V09 G3的cfg_b A16在2816×1536×4096下实测慢28.94%，模型只预计1.40%；G5 B32预计慢6.53%，实测未出现。两项说明当前条件供给迁移失败，不是改一个布尔对齐门槛就已解决。[EXP-15/R05-E](../EXP-15-tma-2d.md#r05-e)的32 B结果针对TMA输出，不直接迁入A/B输入。stage4响应归 [R13](R13-async-retirement.md)。

### 同工作配对：差异确实出现在主循环

复用738203、738496与739011的dual原记录，每批内部按trial、CTA、j/T及坐标匹配。先计算每个窗口的变化率，取进程内中位，再取十进程中位；不同几何不直接相减，也不混合拟合。这与上文“后续窗口均值/每Ktile”统计不同。

| 配对 | first L，ns变化 | later L，ns变化 | first L，cycle变化 | later L，cycle变化 |
|---|---:|---:|---:|---:|
| 738203 B32，相对同批对齐 | +17.57% | +16.96% | +18.37% | +19.67% |
| 738203 B64，相对同批对齐 | +8.62% | +10.44% | +9.40% | +11.80% |
| V09 G5 B32，相对同批对齐 | +1.47% | −0.66% | +0.19% | −0.92% |
| 738496 cfg_b A16，相对同批对齐 | +10.31% | +21.22% | +16.41% | +15.39% |
| V09 G3 cfg_b A16，相对同批对齐 | +22.97% | +31.12% | +29.68% | +32.42% |

旧B32的惩罚分布在所有输出位置，j1/j2约17%–21%，最后j3/T4仍为4.82%；V09 B32的三个后续位置为−1.16%、−1.46%、+0.04%，未出现同量级惩罚。因此新B32的完整时间接近基线，不是一个很大的主循环惩罚被输出抵消。V09 B80/B112的later L仍分别慢19.70%/18.76%；这不恢复通用32 B对齐规则，因为旧B32/B64的反例仍成立。

A16也不是频率或输出可以补救：G3的两个输出位置都变慢，later L的十trial中位变化范围为29.82%–32.69%。校准批j1/T3、j1/T4的A16惩罚其实已有28.04%/27.62%，接近G3的31.12%；将各位置聚合成21.22%会遮住这点。模型却因G3的j1已经是末轮，把软件源需求从4224降至3828 KiB，拟合供给从371.12降至336.32 ns/Ktile，靠近计算下界，因而低估响应。首轮的两批静态请求向量及338.60 ns/Ktile供给价格完全相同，实测惩罚仍不同；只加末轮修正也不充分。源重访、几何和足迹尚未隔离，不能命名为某一层缓存带宽。

以上分析支持先修正供给状态假设，不默认追加B64/B96矩阵。新点应能区分一个尚未被否定的原因；原冻结与V09判定不变。[分析与完整17组成对结果](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/B-analysis.md)、[逐trial/位置](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/B-paired-windows.csv)、[静态需求与来源](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/B-summary.json)。

## 遍历方向足以改变B32响应

CUTLASS 3.9.2 `get_rasterization_order` 在 N tile数大于M时选AlongM，否则AlongN。旧B曲线是18×24 tile、默认AlongM；V09 G5是18×18 tile、默认AlongN，原比较还混入了遍历方向。

739571在a058/GPU-ef8692f3固定旧形状2304×3072×1024、132 CTA、stage6、sw1、dyadic seed17与全部分配容量，只做AlongM/AlongN×B增量0/32 B四点。四条件plain/dual各十进程，先采两个plain参照；共100成功进程、无失败。静态资源均168寄存器、16条HGMMA、零spill。每次dual实际坐标与显式raster调度逐CTA一致。

| 方向 | plain aligned / B32，µs | plain B32变化 | first L，ns变化 | later L，ns变化 | later L，cycle变化 |
|---|---:|---:|---:|---:|---:|
| AlongM | 27.904 / 32.800 | +16.94% | +22.35% | +18.40% | +20.90% |
| AlongN | 28.000 / 28.480 | +1.20% | +0.29% | 0.00% | −0.33% |

变化列先按trial配对，L先匹配同方向的CTA/j/T/坐标并求进程内中位，再跨十进程汇总。AlongN的later L十次范围为−0.58%～+0.59%，未见AlongM的同量级惩罚。两方向的B32完整时间增量再作差，十次全部为负，中位−4.368µs、范围−5.984～−3.360µs。全CTA有效频率的B32变化仅约+0.71%/−0.74%，周期窗口也保留上述差异。

方向改变了每个软件轮的源分组：AlongM前三轮各有18个A/8个B面板，AlongN各有6个A/24个B面板；432个输出tile、96 CTA×3和36 CTA×4的工作量不变，grid形状随raster转置。结果支持**执行顺序与源分组是B32迁移条件的一部分**，而非仅由K和足迹变化造成；没有测得物理请求数、缓存命中或某个L2资源，不能把面板分组直接命名为物理带宽。它也不意味着为AlongM单独添加一个B32常数就能迁移到其他行距和形状。

当前供给模型对两方向生成的432行`(CTA,j,T,Ktile,X,compute_cycles)`逐项相同，B0和B32各自的输入SHA也完全相等。实际坐标列表虽不同，七列汇总请求却丢失了源分组。因此仅重拟合现有七列的系数，无法表达这次方向响应；下一候选需要保留访问组织，不能把失败归于参数没调好。

dual/plain完整时间扰动中位为2.60%～5.76%，最大8.91%，故不把局部窗口当裸服务价格；plain自身已独立显示方向×行距效应。同卡参照相对739556中位变化−1.41%/−0.61%，旧432卡的绝对参数未混入拟合。原冻结和V09成绩保持不变。[四点、逐进程和差分](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-raster-pitch.json)、[工作位置表](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-raster-pitch-positions.csv)。

## 相同B源组在不同执行规模下的代价不同

源分组不能简化成另一个标量罚项。739571中，唯一B字节更少的AlongM反而更慢；AlongM首末轮的同B使用者数相同、该B面板均未在此前软件轮出现，惩罚却不同。`活动CTA数×同B使用者数`的非减响应也给错顺序。首轮再按相同M坐标、A/B使用者数量和此前轮成员逐窗口匹配，T4的行距周期增量仍比T3多384.375 cycle，十进程范围207.75～643.5；两组B地址及后续工作不同，故只否定计数充分性，不否定任意源身份/历史规则。[无参数反证与实际窗口重叠](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-source-organization.json)。

739585固定input map、初始化与分配容量为2304×3072×1024、AlongM、stage6、dyadic seed17。逻辑N=256/896分别启动36/126个单输出CTA；前36个CTA的当前坐标、两组B面板及每组18个使用者完全相同，没有下一输出的producer预取。另同期复测原132 CTA、每CTA 3/4输出的B0/B32。每档先在内部匹配行距配对，再比较前36个CTA首输出L的响应：

| 执行组织 | plain aligned / B32，µs | plain B32变化 | 首36 CTA的L，ns变化 | L，cycle变化 | 配对L的ns增量〔十进程范围〕 |
|---|---:|---:|---:|---:|---:|
| 单输出36 CTA | 11.344 / 11.792 | +2.77% | 未分辨 | 未分辨 | 0〔0～0〕 |
| 单输出126 CTA | 12.720 / 13.952 | +9.24% | +23.36% | +26.00% | 1200〔1120～1248〕 |
| 持久132 CTA | 28.256 / 32.944 | +16.91% | +20.40% | +20.36% | 1120〔832～1248〕 |

各变化列先按trial匹配、在该进程36个窗口内取中位，再跨进程汇总；零指进程中位在32 ns分辨率内未分辨，不表示每个窗口或物理请求零代价。126减36的行距响应差十次均为正，中位1200 ns、2204.75 cycle；局部cycle/ns的行距变化约1.94%，不能解释23%的时间差。B32条件下，目标窗口实际重叠的同B伙伴中位为16.65/16.36，其它CTA则为34.32/118.58；每个peer CTA的L区间先求并集，未把pingpong两个输出重复计入。它仍不是TMA发起并发数。

这说明**当前B面板的使用者、源布局与逻辑坐标相同，改变其它源组和A侧共享组织，仍会改变行距响应**；跨输出预取和126～132间的窄阈值都不是这次大幅惩罚的必要条件。逻辑输出量、epilogue写出者、预热工作及频率随执行组织变化，不能进一步唯一归因某层缓存、TMA队列或全卡物理带宽。下一候选须保留共享需求与每个源组的竞争关系；已有首末轮反例仍要求状态，不能直接给36/126或方向加常数。

作业在同a058/GPU-ef8692f3完成140个成功独立进程，另有一个暖机CV未收敛的失败plain尝试，按原规则重试且原件保留。数值回放通过，两个二进制均168寄存器、16条HGMMA、零spill，实际SM数为36/126/132；参照相对739571变化+1.43%/+0.035%。dual/plain扰动中位2.59%～5.77%、单次最大10.43%，所以局部窗口不当作裸服务常数；plain的126档十次也均变慢4.22%～15.65%。[条件](../../../../../../microbench/gh200_resource_campaign/access_rules/configs/r10-source-pressure.json)与[完整逐进程分析](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-source-pressure.json)保留分项、来源和失败记录；V09评分不变。

<a id="shared-service-candidate"></a>

## 从分组响应到逐CTA：两参数候选仍缺状态

[RULES中的共享响应候选](RULES.md#source-sharing-dev)只用739585的single36/single126约束两个非负参数。额外用aligned L限制共同基线，防止响应相减藏住两边都错的部分；参数仍为相关分片域，不能随CTA选不同价格。域内确有一对参数使739571的(j,T)分组中位目标误差为0.87%/7.12%，同批未训练persistent组为3.82%/7.67%。这只是**实测B0窗口锚定的分组响应诊断**，好点也在看完迁移后才被注意到，不能称作事前选中的完整预测。

逐CTA复核改变了判断。30个配对进程、12960个窗口分别计算各自的B使用者数，再用同一对参数；不把窗口当独立样本。低a好点在AlongM和persistent的每进程最大局部误差达到约22%～29%，分组中位掩盖了稳定位置差异。更强的检验是对**完整参数域**求每个位置最有利的响应区间，再与该位置十个进程的中位目标比较：

| 配对 | 中位目标必超10%的位置 | 最差位置 | ΔL预测区间，cycle | 实测ΔL中位〔十次范围〕 | 整L误差下界 |
|---|---:|---|---:|---:|---:|
| 739571 AlongM | 74/432 | CTA91/j0，M/N tile=(1,5) | 2041.5～2686.9 | 366〔23～485〕 | 18.15% |
| 739585 persistent | 68/432 | 同一位置 | 2041.5～2686.9 | 273〔48～494〕 | 19.62% |
| 739571 AlongN | 0/432 | — | 按位置保留 | — | 最差3.27% |

即使将训练条件放宽到每进程5%的整L容许范围，两批仍有47/46个位置必超10%，CTA91下界为15.25%/16.64%。所以问题不只是经验范围过窄或一次噪声极值，缩小当前参数域、补中间CTA规模不能救回这些位置。

CTA55/j0与CTA91/j0的M坐标、T3工作量、同B使用者数相同。部分代价确实移到了首次输入等待：91相对55的行距响应差，在739571的首段为+240 ns、L为−1200 ns，合起来仍为−1024 ns〔−1280～−832〕；739585合起来仍为−576 ns〔−928～−448〕。每次调用分别核对入口→FIRST_MMA→MAIN_END闭合，再做同期B0/B32配对，未拼接不同调用的绝对时轴。首次full等待只能解释部分L差。SMID回放还显示，两批十次B0/B32中CTA55始终在SM55、CTA91始终在SM87，沿N时也保持该映射；源位置与硬件位置当前绑定，稳定差值不能直接归因源地址。源身份、时间状态及硬件派发/位置仍未隔离；下一步不能直接给CTA91或某个tile编号加常数。[完整参数域](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-shared-service-candidate.json)、[逐CTA、关键CTA窗口和端点闭合](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-shared-service-windows.json)。

<a id="base-shift"></a>

## 基址平移没有让响应整体跟随相邻B面板

739598在同a058/GPU-ef保持2304×3072×1024、AlongM、132 CTA、stage6、sw1与dyadic seed17，仅改变B的相对基址0/256 B及行距增量0/32 B。两边同样预留256 B，移动后的指针同时用于初始化、tensor map与数值检查，设备侧没有新增按tile变换地址的逻辑。100个成功进程、无失败；plain/dual均168寄存器、16条HGMMA、零spill。每进程payload为3211264个FP16元素，总容量为3211392个元素。

先在各offset内按同CTA/j/T/坐标求B32−B0，再比较两offset的响应。下列统计先在每个trial内部计算，再取十个进程配对的中位。相邻面板只匹配同M的ni+1，不绕末列；两种对应使用相同目标子集。

| 对应关系 | 每进程窗口数 | 周期响应Pearson相关中位 | 响应差RMS，cycle |
|---|---:|---:|---:|
| 同工作位置，匹配上下文的共同子集 | 270 | 0.804 | 606 |
| 相邻B面板，同j/T/共享条件 | 270 | 0.339 | 1011 |
| 同工作位置，只看首轮T3 | 72 | 0.864 | 410 |
| 相邻B面板，只看首轮T3 | 72 | 0.216 | 847 |

纳秒统计方向相同；因此不只是把不同软件轮的平均值混在一起造成对应。plain的B32减速为17.31%/17.69%。CTA55的周期增量从2291变为1870.5，CTA91从453.5变为509，原有大小关系保留；两位置的配对offset交互范围都跨零，不能只看中位数断言稳定局部效应。反过来，整个432位置中有17个在十次比较的cycle/ns交互都为正、11个都为负，也不能写成基址条件完全不影响局部窗口；这是本批观察到的同号计数，不是多重检验后的显著性结论。[响应空间图](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-base-shift-response.png)。

40个dual进程只有一种CTA→SM映射，同CTA与同SM对应因此相同，仍不能认定SM机制。B分配VA原点却有三个，相差512 MiB/1 GiB；只有trial5四条件原点全同，该旁证也更接近同位置对应，但不替代全批结果、不证明同PA。每进程自身的指针差严格为0/256 B；本结果只限制这一次相对基址干预，不能排除所有地址依赖。dual/plain中位扰动为2.07%～4.28%、单次最大7.18%；两个同卡参照相对739585变化−2.03%/+1.22%，未把跨批绝对值并入拟合。

此前的到达定位也未提供替代规则：CTA91−55的行距入口响应变化中位为0 ns，producer first-work只有−32/−16 ns；FIRST_MMA与L的负相关部分来自共用端点，入口→MAIN_END的净差仍保留。first-work不能当请求到达，尚缺其它stage是否ready和已发Ktile数。[60个dual进程的到达分析](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-arrival-localization.md)。

后续[工作序列交换](#work-swap)直接解除逻辑工作与原物理CTA的绑定，不继续扫描基址。

## 历史推导与复核

行距入口为 `run_r10.py`，joint/coverage重算为 `analyze_r10.py --joint-pitch`、`--score-b-pitch`，grid扫描为 `run_r13_sm.py` / `analyze_r13_sm.py`。已有reanalysis优先使用上表链接；采样前冻结算法、完整命令与准备过程见[固定 cee978a 旧正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R10-layout-cache.md)。历史参数只用于复核旧判定，现行规则不在本页复制。

<a id="work-swap"></a>

## 交换工作序列后，响应仍留在原物理位置

739608沿用739598的卡、几何、容量、数据和B0/B32配对，B相对基址固定0。仅在持久调度器初始索引交换55/91，之后仍每次加132：`(1,3)→(7,10)→(13,17)`与`(1,5)→(7,12)→(13,19)`互换。每轮工作集合、A/B分组人数、T3和M序列不变。40个dual进程逐CTA坐标完全符合置换，CTA55/91仍在SM55/87；这两条数学工作序列与原物理执行位置因此不再绑定。

下表先在每个模式、每个trial内求B32−B0，再取十进程中位；不是跨SM相减clock64。

| 工作分配 | 物理CTA | 首轮逻辑坐标 | ΔL，cycle〔进程范围〕 | ΔL，ns | Δ入口→MAIN_END，ns |
|---|---:|---|---:|---:|---:|
| 原分配 | 55 | (1,3) | 2312.5〔1876,2624〕 | 1312 | 1360 |
| 原分配 | 91 | (1,5) | 360〔291,579〕 | 80 | 496 |
| 交换 | 55 | (1,5) | 2144.5〔1708,2676〕 | 1264 | 1312 |
| 交换 | 91 | (1,3) | 485〔323,736〕 | 160 | 464 |

仅用受干预的两个CTA作主要对应：首轮响应按原物理CTA匹配的RMS为276.5 cycle，按原逻辑工作匹配为1828.7；三个输出共六窗口为385.5与1616.6。首轮入口→MAIN_END纳秒响应的RMS也分别为152.1/891.5，所以不只是FIRST_MMA边界移动。此结果反对仅按这条B/D工作序列收费；支持执行位置或其形成的时序是缺失条件，**尚不能命名为SM硬件差异**。

未交换工作的130个CTA另作内部对照，不混入上述对应分数。按进程先取首轮中位，swap1−swap0的L变化在B0/B32为−32/+8 ns，入口→MAIN_END为+32/+48 ns；各自十进程范围都跨零。它们没有出现足以解释两代表位置反差的共同平移。但两模式虽同binary，SASS的swap0会跳过初始55/91比较，不能声称动态指令路径完全相同。

100个采样进程全部成功、无失败；四次独立正确性比较覆盖全部7077888个输出，交换前后逐位相同且无非有限值，每次交换前重新填充输出哨兵。plain/dual均168寄存器、215040 B SMEM、16 HGMMA、零spill。plain行距惩罚原分配19.07%、交换18.42%；dual/plain中位0.63%～3.73%，单次最大7.57%，局部价仍只属于本观察协议。参照相对739598变化+0.46%/+0.31%。各独立进程出现三个B分配VA原点，无一trial四条件原点全同，不假定相同物理页。

[本批同源组分析](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-work-swap-arrival.md)仍无法区分SM与先后状态。B32下，CTA55的producer在18个同B伙伴内的零起平均秩从1.75变0，较组内中位早128→224 ns；CTA91的秩从11变15，较中位晚8→80 ns。也就是说，换了B面板以后，早/晚的软件位置仍跟着物理CTA保留。B3/B5未换work的各17个伙伴，行距响应交互的组内mean L再取十进程中位只有+6.6/+45.2 ns；减去其余96CTA共同背景后为−45.9/+12.4 ns，十进程范围均跨零，没有稳定的约1 µs整组外溢。

后续[起步延迟对照](#producer-delay)固定工作和SM，只改变时间状态；producer first-work仍不能直接当TMA请求到达。

<a id="producer-delay"></a>

## 固定工作与SM，起步状态仍改变行距代价

739619保持前批的卡、工作映射、容量和地址布局，在CTA55 mainloop producer首个work循环前加入0/512 ns等待，配B0/B32。延迟前tensor map prefetch已经发出；延迟后才记录原producer first-work、获取stage并发出TMA。因而检验的是时间状态参与，不能唯一归为同B请求排队。plain与dual共用运行时开关，四个端点只由dual记录，侧缓冲容量相同。

干预确实改变了软件先后：零档的实测等待中位32 ns，512档为512/528 ns；配对增量分别480/496 ns。等待结束到原producer打点约160 ns，两档之差中位0。CTA55在18个同B伙伴内，producer/FIRST由约第5～8名变为全部20个延迟进程的第18名，实际工作和SM55保持不变。

先按同delay、同trial求B32−B0，再比较两档，得到CTA55首输出的交互：

| 分项 | 512档相对零档的行距响应变化，中位〔十进程范围〕 |
|---|---:|
| L，cycle | −1076〔−1555,−902〕 |
| L，ns | −704〔−960,−448〕 |
| S，ns | +128〔64,352〕 |
| S+L，ns | −576〔−864,−320〕 |
| 入口→MAIN_END，ns | −576〔−832,−384〕 |

**相同工作和SM下，行距响应随起步干预变化；不能再给固定CTA/SM一个无状态的罚项。** S增加只抵消部分L变化，完整首输出的交互仍稳定为负，因此不只是移动FIRST_MMA边界。

这不代表延迟提高性能。直接比较512档与零档，B0的入口→MAIN_END增加1312 ns〔1216,1536〕，其中P0增加512 ns、S+L增加816 ns〔736,960〕；B32完整窗口增加768 ns〔512,928〕，S+L增加288 ns〔0,416〕。B32虽有L缩短，S的增加更大。各列独立取中位，端点闭合在逐进程验证。17个未干预同B伙伴的L变化中位仅16/16 ns，其余130CTA为0/24 ns，各自进程范围跨零；没有同等幅度的共同平移。

100个采样进程全部成功、无失败；四次独立全输出比较均7077888元素逐位相同，无非有限值。plain/dual均168寄存器、215040 B SMEM、16 HGMMA、零spill。plain整调用行距惩罚为19.10%/18.58%，延迟本身的整调用效应范围跨零；不把单CTA观测当整GEMM优化。dual/plain中位2.72%～4.85%，单次最大6.61%，局部价仍限本观察协议。同卡参照变化+0.57%/+0.10%；跨进程VA原点有三个，不假定同PA。沿用自适应暖机，等待也用于本条件暖机；结果约束该协议下的时间状态，不唯一定位最后一次调用的局部初态。

按同CTA本地周期复核，B0的P0增加974 cycle〔833,1052〕，入口→MAIN_END增加2099.5〔1947,2326〕；逐试次减掉P0后仍增加1156.5〔1047,1356〕。在事件前驱、同域边成本和其它根时刻不变的max-plus递推中，单根后移δ不能使完成点增加超过δ。因此本批反对简单平移固定局部时间线，不依赖统一GHz；这不排除带可变仲裁、等待返回和初态的有限stage模型，也不把跨进程配对当成隐藏状态完全相同。

[R13首次供给分解](R13-async-retirement.md#first-supply)已将新增时间定位到首对A/B copy的软件发出标记之后：prod→issue_end基本不变，consumer进入首次等待也未明显变晚。尚需区分异步完成与等待返回，FIRST不等于硬件首次ready；不继续扫描延迟或将名次拟合成服务价格。
