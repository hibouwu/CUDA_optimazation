# 源组织：共享源与独立源

26 个目录（10-19 至 10-21）：在原生 D 链（真实三请求）上把 CTA 数扩到 132，并固定分配，只改变源 batch 的共享身份；另含后续代次观察的准备与采前核对。

## 结论

**已测事实**
- 共享源（并集 416 KiB）与独立源（53.625 MiB）下，S6 的 CTA 窗口中位分别为 5.088 µs 与 17.888 µs（757119）。
- 按 CTA 窗口中位，两种模式都没有超过阈值的额外 stage 响应。全共享的 132 CTA 链也没有负载相关的 stage 价（756885）。
- 独立源的结束时刻跨度：S4 为 2112 ns，S6 为 4768 ns。固定源到 SM 的绑定后，S6−S4 的跨度差仍有 2272–3840 ns，只有两个 seed（757444）。
- 固定132 CTA／S6和全部分配，将源batch由SMID改为SMID%66后，源并集从53.625 MiB降至26.8125 MiB；三seed包络下降53.42%–54.23%，通过测前冻结的源状态响应带，排除零响应形式（763587）。三进程、六条件观测均合格；绝对窗口及结束跨度仍有反例。

**现用源服务的回放**
- 共享与独立的差值预测误差 3.8%，但绝对窗口低估 8%–20%。
- 结束跨度预测为 0 ns（共享）与 1197 ns（独立），实测为 384–416 ns 与 2112–5952 ns。
- 只改仿真里同时刻事件的排序约定，预测跨度就变为 2898／3584 ns，说明跨度预测依赖求解约定。

**未完成**
- 66源响应已获局部支持，现用服务的绝对价格及CTA结束分布仍未解释；[同条件比较](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-source-count-build-002/score.json)保留两种排序的残差。
- 实际每调用清写和导出的8.25MiB输出未接入旧D源回放；沿用现有写分配规则补入后，132源稳态保持、66源的FIFO预测过慢，八个绝对端点均失败，停止该固定接线，见[CPU回放](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-output-state-replay-001/README.md)。
- 将缓存／HBM发起移到既有GPC受理时刻后，132源仍全miss、66源仍全命中，包络只变−2.320／+25.094ns，八个绝对端点仍失败，停止该固定连接。
- 用保存描述符的实际A/B基址替换旧假设后，B的106496/219648次行访问home归属改变，但四组528个CTA结束时刻逐值保持，仍未修复残差，见[输入核对](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-source-base-replay-001/README.md)。
- 未完成填充不占缓存容量、完成时才插入的固定形式，使132源在暖机后也全命中，66／132源响应变为0；原冻结响应及八个绝对端点均失败，停止该形式，见[填充完成回放](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-fill-completion-replay-001/README.md)。
- 保留发出时占位、只为完成时尚存条目增加MRU更新，FIFO／132源仍全miss，132个CTA及5148请求逐值保持；13调用即在首项停止，不继续扩展LRU更新，见[完成时替换顺序](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-fill-recency-replay-001/README.md)。
- 受理后查找同时反馈producer后继，FIFO／132源末调用4723次延后，包络／跨度只变−45.444／+13.953ns，13调用即停止；439296条源线互不重复，原价后端连续忙15.681µs，见[固定受理反馈](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-admission-feedback-replay-001/README.md)。
- 数据准备的超额延迟改为约束GPC服务进入，末调用5148请求全部生效，132源包络／跨度仍未通过；13调用收口，停止该固定连接，见[数据准备约束共享服务](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-data-gated-service-001/README.md)。
- wait0 返回与尾部的观察实现改变了循环或未过扰动门。
- 后续槽代次的旁路观察经用户授权追加零GPU构建772273后编译成功，但REG167→174、两条主循环改变，停止当前观察实现；full／empty端点仍缺，见[追加构建](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-late-slot-observer-001/rebuild-20261021/README.md)。
- 同二进制N0／N1实际扰动及向R桥接在773099、cycle-only版773128均通过，各S6九进程。寄存器变化未使这两次观察失效。full→empty联合区间由2032–2608cycle缩至2132–2628cycle，宽576/496cycle，仍覆盖两个冻结预测族；两轮均未测S4，轮询形式收口，见[原双单位观察](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261021-Q02-late-slot-switch-001/README.md)、[cycle-only观察](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261021-Q02-late-slot-cycle-observer-001/README.md)。
- 132源g11跨CTA观察774105的九进程数值、暖机与唯一SM通过，总窗口与consumer完成戳扰动通过；CTA结束跨度N1−N0为−10.811%至+14.286%，向R桥接也未过。N1的full跨度为5184–5824ns，原R结束跨度自身范围/中位21.642%；信息不足，停止本轮，不选择两种源排序，见[原始数据与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261021-Q02-late-full-span-001/README.md)。

**判断**
- 源组织决定绝对时间（这两个极端相差 3.5 倍）。真实 G3 的情况介于两者之间：唯一源 34 MiB，每个 box 被 11–12 个 CTA 读取。
- 结束时刻分布是尚未解决的问题。

## 轮次

按日期排列；全部结果目录见[Q02-tma-request/source-organization](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 共享负载是否放大D局部槽复用响应 | 固定K13、物理S6、共享generation32源及计算释放，CTA1/132×活动S4/S6，旧单CTA桥接 | 756885完成18进程及31项资格；额外stage响应−58.5cycle在±423.85带内，本工况不加负载stage价；两地评分一致 | [有限执行与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-local-cta-sharing-001/README.md) |
| D/E是否覆盖真实cfg_b的源工况 | 固定程序与价格，逐CTA核请求并集／共享，复核E交接资格 | D416KiB、E832KiB与G3/G5的34/36MiB不同；选择固定分配／描述符的源batch对照；25输入两地一致，GPU0 | [CPU覆盖筛查](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-source-footprint-screen-001/README.md) |
| 固定源分配后，共享／独立源是否改变stage响应 | 固定CTA132、计算／等待／释放和行距，同binary切batch0／CTA编号 | 757119完成18进程、桥接通过；绝对窗口变长，额外stage响应136cycle／0ns在原带内，不加该价；两地原始评分一致 | [固定源batch对照](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-source-batch-001/README.md) |
| 现用共享源服务能否迁移到固定batch局部链 | 固定旧价格、分组／地址相位、计算／释放及协议，只接实际请求 | 源模式增量误差3.776163%过门，CTA结束分散失败，固定迁移停止；88调用，两地复现，GPU0 | [现用服务迁移](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-source-batch-service-001/README.md) |
| 源batch完成分散是否对应相同输入和端点 | 固定12份正式raw，核计时前驱、完整描述符和CTA映射 | 六组均124CTA换GPC；end含wait0后checksum与汇合，先隔离绑定与端点；两地复现，GPU0 | [端点与输入核对](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-endpoint-boundary-001/README.md) |
| 固定分配和源到SM绑定后是否仍有stage推进差 | 同一分配内CTA／SM绑定×S4/S6，旧两条件桥接 | 757444完成14记录，写盘错误使第三配对未采；两SM跨度响应2592／3840ns，信息不足；两修复收口 | [固定源绑定stage配对](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-fixed-source-pair-001/README.md) |
| SM绑定程序是否用了对应服务域（CPU） | 按SM→GPC映射给源batch分组，固定旧服务／频率／地址假设与协议；两个stage预测后评分 | 114归属改正，stage跨度差仍预测0，固定迁移停止；44调用、两地复现，GPU0 | [计划、评分与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-sm-source-domain-001/README.md) |
| 零耗时事件的同刻排序是否影响共享轨迹（CPU） | 只在副本将FIFO改为同刻CTA优先，固定服务／价格／输入；单CTA和S6/S4局部回放 | 单CTA保持、12408端点关系通过；跨度显著变化但stage响应方向仍错，保留排序敏感性，未选硬件规则；52调用、两地一致、GPU0 | [排序、端点与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-zero-time-order-001/README.md) |
| 固定分配缩减源并集能否区分现用缓存规则与零响应 | 132CTA／S6不变，只将源SMID映射成SMID%66；两个同刻顺序均保留 | 冻结窗口下降55.38%–56.40%；原构建缺共用头未测，763560补齐后机器码／资源核对通过；763587三seed响应−53.42%至−54.23%，现用状态响应获限定支持，零响应被反证；绝对时间与结束跨度仍失败 | [冻结预测](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-source-count-prediction-001/README.md)、[原构建历史](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-source-count-build-001/README.md)、[构建接续、原始数据与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-source-count-build-002/README.md) |
| 多CTA是否接收与独立模块相同的producer控制输入（CPU） | 只透传既有组内／组间间隔；四形状单／组合及双CTA受理返回核对 | 8默认报告保持，452关系及27测试通过；异目录一致，80调用、拟合／修复／GPU0，硬件价格未选 | [接口反例、接线及收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-producer-control-input-001/README.md) |
| 独立搬运输入能否表达源batch对照 | 固定服务，只补batch坐标与分配内偏移，核矩阵地址及端点接线 | 20592请求地址核对、四默认报告及23测试通过；两地复现，GPU0 | [源batch输入合同](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-source-batch-input-001/RESULTS.md) |
| 多CTA D是否共用一次源服务状态 | 固定价格，将CTA与batch映射接入既有引擎 | 四默认报告、132CTA同次状态及52916端点约束通过；26测试，两地复现，GPU0 | [多CTA独立入口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-cta-group-entry-001/README.md) |
| D输出写入是否遗漏共享源状态（CPU） | 固定66／132源、两排序及全部旧价格，只接每调用8.25MiB输出清写与CTA计时后导出；44调用 | 八个绝对端点失败，FIFO66源多1744.975ns且跨度更小；132源与CTA优先稳态保持，停止固定接线，不扩大S4或完整组合 | [计划、评分与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-output-state-replay-001/README.md) |
| 数据分支是否应在GPC受理后开始（CPU） | 固定所有价格及服务范围，只将缓存查找／新miss提交移到既有service_start；66／132源、两排序及零服务等价检查，46调用 | 零服务精确保持、5148查找按时间推进；八个绝对端点失败，响应带保持，停止固定连接，不接S4／完整组合 | [计划、评分与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-admitted-data-replay-001/README.md) |
| 实际分配基址是否改变现用home映射（CPU） | 固定服务／映射规则，只将旧A0/B152MiB换成保存描述符的实际基址；66／132源×两排序，44调用 | 48组编码对应通过；B有106496次home变化，但528个CTA结束逐值保持，八绝对端点仍失败，保留实际输入及反例，不扩大 | [输入、回放与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-source-base-replay-001/README.md) |
| 首full完成跨度能否区分两种同刻推进 | 固定两源数及旧价格，K13暖机后正式K4，不复用slot0；先44次CPU预测，再核无consumer打点的旁路区间观察 | 44次CPU回放的两顺序预测可分；四次零GPU构建均未进入测量，当前旁路形式停止，两顺序均信息不足 | [有限计划](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-first-full-span-001/README.md) |
| 填充完成后的缓存插入能否修复共享推进 | 固定实际基址、容量、全部价格及并发组织；未完成填充单列，ready时插入及替换；46次CPU调用 | 零延迟单CTA精确保持；132源暖机后全命中，源计数响应退化为0，八绝对端点失败；停止固定形式，不接S4或完整模型 | [规则、保存预测与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-fill-completion-replay-001/README.md) |
| 保留在途占位后，完成时LRU更新能否修复推进 | 容量及发出占位保持，只在fill完成时将尚存条目移到MRU；FIFO／132源13调用 | 零延迟等价保持；132CTA和5148请求逐值不变，首项包络与跨度失败即停；未扩大，不改完整模型 | [规则、反例与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-fill-recency-replay-001/README.md) |
| 状态化源受理能否通过producer后继改变共享推进 | 受理后查找保持，只将service_start接回producer返回；固定实际基址、D程序及旧价格；13调用 | 零排队等价保持；4723请求反馈生效，但包络／跨度仍失败；首项即停，不改完整模型 | [连接、工作守恒与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-admission-feedback-replay-001/README.md) |
| 后续代次的full／empty旁路区间能否实现（CPU） | S4/S6、K13，读取既有generation后只观察最后使用的slot0；用户另准一次零GPU构建、最多10分钟 | 765495缺参数失败保留；追加772273编译成功、宏关闭精确，但REG167→174且两循环改变，原门槛失败，停止当前实现；GPU／模型／拟合0 | [原有限构建](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-late-slot-observer-001/README.md)、[追加构建与差异](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-late-slot-observer-001/rebuild-20261021/README.md) |
| 数据准备是否应约束共享服务进入（CPU） | 保留旧数据查找时点与全部价格，以超出local hit的就绪延迟推迟GPC队列进入；先local-hit等价再132源 | 13次CPU调用；local-hit逐值保持，132源包络17111.935ns／跨度1335.226ns均失败，停止固定连接，未扩展，GPU／拟合0 | [有限连接与判定](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-data-gated-service-001/README.md) |
| g11同二进制观察能否区分已有控制放置 | 固定S6／S4、K13、CTA1；已有消费者继续／ready full返回两形式，H全必要界传播；R／N0／N1有限执行 | 16次CPU调用复现六窗口，预测族间距87.449cycle；773094修正构建通过，773099的S6九进程观察与桥接通过，区间宽576cycle，信息不足；未测S4，不重复相同轮询 | [计划、预测、源码与构建日志](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261021-Q02-late-slot-switch-001/README.md) |
| g11相位的cycle-only观察能否分开已有控制放置 | 固定原程序与冻结两预测，只去掉旁路相位的globaltimer读取；R/N0/N1同批资格，先S6后按门槛S4 | 773126构建通过；773128的S6九进程观察及桥接通过，区间由576缩至496cycle，仍覆盖两解释；信息不足，未测S4，不继续微调 | [计划、源码与原始记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261021-Q02-late-slot-cycle-observer-001/README.md) |
| 132源g11完成跨度能否定位共享推进分散 | 固定S6/K13、SMID源绑定、真实三请求；44次CPU核旧频率范围，两顺序冻结；R/N0/N1各三进程 | 774095构建通过；774105九进程完成，窗口／consumer戳通过，结束跨度扰动及桥接未过，信息不足；未扩大 | [计划、预测与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261021-Q02-late-full-span-001/README.md) |
| a044 跨卡复核（mechanism-r3，780769） | late-slot 两观察 9+9 次，另 9 次 CPU 程序查询 | switch 与 cycle-observer 的 full→empty 范围为 1964–2596 / 2120–2628 cycle，两候选仍相容，信息不足；程序查询新旧字节相同，CUDA 调用 0 | [Q02 报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/q02-returned-report.md) |
| a044 保序修复（785518–785575） | local-cta-sharing 18、source-batch 18、fixed-source-pair 8、late-full-span 9 次，原顺序与环境，原评分器 | local-cta-sharing、source-batch、late-full-span 判定不变。fixed-source-pair：paired seed 3 的 cta_s4 暖机 CV 为 2.17%（旧卡 1.96%），按原 2% 门槛停止，两卡都没有完整配对 | [修复报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/mechanism-repair-r2-report.md#q02q04q05-原评分器重评) |
