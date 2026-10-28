# 探针控制预算与源参数（单 CTA）

29 个目录（10-18、10-20）。研究对象是 REG52 单 CTA TMA 探针自身的控制推进、首次预填与源完成预算。

## 结论

**已测事实**
- 零搬运控制每组 565 cycle：Q1 为 470+565·G，Q4 为 441+565·G；G64 冻结留出误差 0.17%（752449）。
- 零搬运前缀：Q1 为 433 cycle，Q4 为 952 cycle。
- Q4 的 A16 一次性 +363 cycle 全部落在首次四组预填内，后缀为 0（752532）。
- 每组等待 8192 cycle 时，加载与零搬运之差为 5876 cycle，冻结支持持续成本形式（752644）。
- 首次预填：A 为 493 cycle，双 B 为 540 cycle。
- 双 B 与 A 的完整窗之差：G2 为 108 cycle，G32 为 1998 cycle；两长度的增长差为 1890 cycle。

**冻结预测**
- 通过：Q4 长度外推，四个新窗口误差 1.9%–2.7%（752560）；持续成本（5876）。
- 未通过：Q4 行距外推，预测 206、实测 363 cycle（752522）；两请求短窗，误差 12.6%（759059）；统一请求数的首次净量，预测 54.5、实测 47 cycle（759547）。

**按操作拟合的复合预算**
- A、双 B、混合三类操作的首次、持续、终段预算分别为 60/61/−12、107/124/−14、167/185/−26 cycle，属于探针程序，明确不迁移到 GEMM。
- 那次代数校准后的 13 项“控制检查一致”，所用量基本就是校准输入；唯一的样本外检查是原生 A2 窗口，偏差 12.5%。

**与直接端点矛盾**
- 由完整窗口拟合的源参数和完成尾，与直接测到的端点不符：按旧参数，A→full 的最快界为 772 cycle；加入地址局部性尾部后仍为 705 cycle；观察开启的探针程序中实测上界只有 660–668 cycle。

**判断**
- 这一组给探针定价，不改变 GEMM 的判断。除非某个完整模型残差明确依赖它，否则停止这条线（GOALS §5.5）。

## 轮次

按日期排列；全部结果目录见[Q02-tma-request/probe-control](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| plain零搬运共同控制窗口：752444／752449 | 同一752232设备程序，mask0的Q1／Q4 G16／32及加载控制六开发条件；只用Q1定共同斜率，冻结后测两G64留出 | 565 cycle／284 ns的复合推进长度规则通过；G64周期精确、ns误差0.173913%，旧九价与请求成本仍待分配，默认组合保留 | [执行、评分与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-control-floor-001/EXECUTION.md) |
| 连续单CTA源服务（CPU） | 共同覆盖价、跨组两槽PS、source_start返回；十请求Q1／零搬运、同版Q4基础与真实S6/S4接线 | Q1实现通过，三控制分配Q4均慢6.121–6.832%，停止固定版本；A16缺测 | [后端、回放与反例](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-continuous-source-001/EXECUTION.md) |
| 同版Q4基础/A16独立外推 | 固定queue_acceptance/k565及原价、保存native与分配；三verify＋三formal的基础/A16配对 | 752522观测合格；两窗通过，增长206.059预测对363实测失败，固定版本停止 | [独立结果与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-q4-stride-heldout-001/EXECUTION.md) |
| 同代码Q4首次预填定位 | 固定REG52 binary，A0/A16×trace0/−1；原始四臂同进程比较首次四组发完前后 | 752532观测合格，ΔP363／ΔR0cycle，发出推进需替换；控制分配仍缺锚点 | [开发定位与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-q4-prefill-localization-001/EXECUTION.md) |
| 同代码无搬运前缀锚点 | A0固定，mask0/7×普通/单点；保留c565并定位无搬运控制及加入copy后的净前缀 | 752539合格，P0/P7=952/1606、R0/R7=17569/22723cycle；控制分布得到锚点 | [开发定位与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-q4-control-prefill-001/EXECUTION.md) |
| 共同控制与返回必要界 | 用零窗和P0区分首次预填／refill／retire，固定源价检验一个waiting信用或source_start返回 | 固定旧价两连接均不满足base P；保留有界队列结构 | [全区间必要界](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-control-return-bound-001/EXECUTION.md) |
| 预填／循环控制连接 | 按展开与runtime补发区分软件context，固定源价，以base/A16前缀校准后回放Q1/Q4 | 两端点后由base后缀全区间矛盾停止，未二分或新增GPU | [最小筛查](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-prefill-control-connection-001/EXECUTION.md) |
| 共同发出推进与有界准入 | 用base前缀约束共同h54.5，复用A16 prefix括区，仅校准f后回放原Q1/Q4 | Q1/Q4全部开发约束通过，待四条件独立长度外推 | [开发回归](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-shared-issue-control-001/EXECUTION.md) |
| Q4独立长度外推 | 冻结共同h/f与源价；A0/A16×G16/G64四新plain，G32同批控制 | 752560独立九检查全过，最大2.951270%；完整GEMM未验收 | [结果与回放](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-q4-length-heldout-001/EXECUTION.md) |
| 有界源与实际程序接线 | 将waiting1从探针调用方移入可选源，保留真实GEMM控制；四个单CTA程序同价对照 | 两Q4 trace与四程序依赖通过，容量反馈改变推进、四终点相同；分配元数据另存纠正 | [结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-bounded-program-connection-001/EXECUTION.md) |
| 短等待入口与完成连接 | 同REG52，D0/384×zero/A0/A16六条件；控制由同批zero约束 | 完整窗误差小于2%，行距增长误差25.085%，固定连接停止 | [结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-short-wait-001/EXECUTION.md) |
| 大等待入口的持续成本留出 | D8192三个新条件；5232/5876cycle两预测测前冻结 | 两loaded−zero均支持5876，返回预算固定接入仍失败 | [独立结果与接入筛查](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-ready-control-heldout-001/EXECUTION.md) |
| 首预填／循环补发context | h_p54.5、h_r61.424731固定；原Q1四共同源价一次校准，再查单B | Q1十三项过门，单B基础窗5.000802%越原门，停止参数版本 | [CPU筛查](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-refill-context-001/EXECUTION.md) |
| 持续成本的独立长度判别 | 同binary/D8192，G4/16×zero/loaded，G32同批参照；固定调用与runtime两预测 | cycle净量更接近runtime形式；G4净ns资格失败，正式信息不足，不追加长度 | [结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-ready-cost-length-001/EXECUTION.md) |
| 同版Q1首预填控制 | REG52/Q1/G32/D8192，mask0/7×trace0/-1四开发条件 | 观察资格过门；净前缀167、净后缀5709cycle，局部软件预算锚点补齐 | [开发结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-q1-control-prefill-001/EXECUTION.md) |
| 实测软件预算与源连接 | 固定首预填/runtime/终段，组后推进代替扩大请求间距；一次四共同源价校准 | 保存Q1/Q4/单B/短等待/长度全部过门，开发版本保留 | [同版回归](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-measured-control-budget-001/EXECUTION.md) |
| 新源参数接实际cfg_b程序 | 四单CTA S6/S4×A832/840，仅换源价及L，GEMM其他模块固定 | 依赖/容量通过；基础stage响应仍零，转回多CTA共享连接 | [程序时间线与比较](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-measured-source-program-001/EXECUTION.md) |
| 大等待入口是否隐藏加载程序额外成本 | 与问题2共用同binary的zero/A0/A16及D8192独立留出 | 持续复合成本获得支持；从旧尾移到返回的固定预算接法失败 | [同一轮证据](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261018-Q02-ready-control-heldout-001/EXECUTION.md) |
| 已存局部源参数的首组完成必要界（CPU） | 固定旧源价及共同尾、PS2总服务率；对758758三进程真实请求与区间做守恒筛查 | 最快771.865956cycle仍超观测上界668／661／660及原5%，停止固定迁移；4源模块流含异目录复现，完整引擎／拟合／GPU0 | [必要界、原始约束与交付](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261020-Q02-source-tail-bound-001/README.md) |
| 完成尾随地址局部性缩放是否足够（CPU） | 固定旧源工作、PS2及目的价；4KiB区间规则，旧Q4 base/A16及新S6/K4三seed | 旧192请求尾保持，A路径最快704.804924cycle仍超三份观测及原5%，固定候选停止；2模块流含异目录复现，GPU0 | [必要界与交付](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261020-Q02-locality-tail-001/README.md) |
| 匹配A路径的行距与长度对照 | 复用752232二进制；旧A-only控制＋同分配4096/8192B×1/32请求 | 758857完成12进程，控制过门；G1的ns极差5.556%超原5%，信息不足，G32未采 | [有限轮次](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261020-Q02-matched-a-path-001/RESULTS.md) |
| A路径最短窗口改为2请求 | 固定原binary和双单位5%门；控制＋两行距×2/32请求 | 759059完成18进程，原资格全过；两短窗口偏12.623%／12.801%失败，长窗口与增长通过 | [有限轮次](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261020-Q02-a-path-length-002/RESULTS.md) |
| A路径固定项与完成尾预算 | 复用2/32请求窗口与首组完成端点；分别只改一次性项或共同A尾 | 两种单独修正均停止；当前约束见[路径页](../../../model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md#a-path-budget) | [CPU轮次](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261020-Q02-a-path-budget-001/README.md) |
| 匹配A控制等待 | 原binary、G2/G32×每组wait前延迟0/8192 | 759280完成24进程，原资格全过；旧控制下界相容，完整价格未定 | [有限轮次](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261020-Q02-a-control-wait-001/RESULTS.md) |
| 匹配A首次prefill | 原binary、G32、同分配plain／单点观察 | 759358完成6进程，13项资格全过；493cycle与旧487.5相容 | [有限轮次](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261020-Q02-a-prefill-001/RESULTS.md) |
| A首次／复用完成尾 | 四物理槽、两点代数确定尾，保留源工作／控制／目的预算 | 首个旧单B窗口偏8.128882%，停止固定组合；未提交G4补测 | [CPU筛查](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261020-Q02-a-completion-state-001/README.md) |
| 共同完成尾与终段联合界 | 固定原源工作／速率和控制／目的预算，同时放开共同尾与终段平移 | 消去终段项后全区间仍短缺1829.877177cycle，停止整个联合修正族 | [CPU必要界](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261020-Q02-joint-tail-end-bound-001/README.md) |
| 同版双B与A的plain联合界 | 完整binary一致；同进程A／双B窗口及原生A长度差 | 必要分支的Bpair−A预测为负、三seed实测均正；排除不依赖构建版本或首组观察 | [同版复核](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261020-Q02-matched-pair-budget-bound-001/README.md) |
| 等总量A／双B推进 | 原binary、16KiB、G2/G32、每组wait前8192cycle及单点prefill | 48资格全过；首次47cycle未过54.5预测，持续增长1890对1850通过，两完整形式均未全过 | [759547／759553](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261020-Q02-mask-control-001/RESULTS.md) |
| A／双B复合推进规则 | 一次代数解首次／持续／终段预算，源价与默认组合固定 | 13项控制一致；A2仍偏12.492176%，共同尾／终段族仍与旧正配对相反，固定迁移停止 | [CPU规则与回放](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261020-Q02-operand-control-model-001/README.md) |
| a044 跨卡复核（mechanism-r3，780769） | 14 个来源，原冻结预测与门槛 | Q4 长度留出 9 项最大误差 2.9513%，冻结外推通过；零搬运长度 G64 ns 误差 0.1739%，通过；Q4 行距冻结形式、短 wait 连接、mask-control 两形式仍被反证，ready-cost 仍信息不足 | [Q02 报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/q02-returned-report.md) |
