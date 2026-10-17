# GEMM TMA 供给：硬件候选与反例

状态：已有候选的开发诊断；未形成通过独立干预留出的组合规则。新实验顺序和预算只在[多 CTA 主线](TMA-SHARED-SERVICE.md)维护；S16 控制成本只在[S16](TMA-S16-STAGE-COST.md)维护。软件依赖见[程序图](../program/cutlass_program_graph.md)。

## 1. 现象

cfg_b 的 stage、生产者起步、预填与其他 CTA 输出干预同时影响首段等待、后段主循环和 CTA 离散度。主要证据为[R13](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md)、[R10](../../../experiments/gh200_sm90/Q03-source-service/R10-layout-cache.md)、[R09](../../../experiments/gh200_sm90/Q09-frequency/R09-inkernel-clock-stages.md)、[R15](../../../experiments/gh200_sm90/Q06-handoff-output/R15-output-service.md)。源为本 GPU global 分配，目标为本地 SMEM；多播/输出配对按对应记录区分，逻辑覆盖量不作物理流量。

## 2. 假设

| 候选 | 当前判定 | 重新研究或保留的条件 |
|---|---|---|
| 固定边成本、只平移入口的图 | 不能解释延迟放大及已有响应 | 仅在图/观测边界改变且重写可检验预测时重新立项 |
| 同质 FIFO 服务＋统一入口间隙 | 受保序反例排除，不继续调统一 gap | 不把排除推广为所有排队模型被否定 |
| 有限槽＋源共享＋独立目的投递 | 保留结构候选；已测两套参数均失败 | 同一组参数须同时解释干预方向、幅度和 CTA 跨度 |
| 仅改变源订阅截止 | 已有有界拟合不能同时解释方向与跨度 | 当前不扩展截止扫描；不宣称全部函数族被排除 |
| 独立公共请求路径＋原有源/目的服务 | R10 前缀选出的参数点在 R13 将 stage4 预等 early 预测为增长，与实测缩短相反 | 该参数点不采用；其他规则须同时接受行距迁移、预等响应与空间跨度检验，见[原比较](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#request-path-joint) |
| 本地 epilogue 排他 | 只能解释部分响应，默认关闭 | 若实际依赖使两者相交，再用独立干预确认 |
| 统一输入/输出工作集状态＋在途源共享 | 输入访问集合与先前 global 写入的局部干预均获支持；G3 stage 响应改善，但空间跨度与预等响应仍失败 | 保留存储状态证据，下一版须解释逐请求或逐执行位置的差异，见[工作集联合回放](../../../../../../results/gh200_resource_campaign/access_rules/20261014-tma-tensor-development/working-set-001/README.md) |
| 每 GPC 有限服务＋在途源订阅 | 2 槽服务通过 N32 的六个局部新条件；迁移到 cfg_b 源坐标后，四个订阅版本均失败 | 须保留近期源状态，见[匹配源判别](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gemm-source-discrimination-001/README.md) |
| 两域容量＋副本状态 | 746641 的 GPC2/3 工作重排响应支持局部域容量候选；完整缓存候选仍失配 GPC5 和 GEMM 的 stage/空间响应 | 占位时点、初始化副本、固定替换优先级与scratch准备回放均未修复反例，见[开发入口](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gemm-source-development-001/README.md) |
| 释放到生产者复用的统一延迟 | 增加220cycle可使S4比S6慢约5%，空间跨度仍只有约55ns；三个独立barrier探针未得到可直接迁移到GEMM的价格 | 保留[敏感性结果](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gemm-source-development-001/gemm-release-sensitivity-001/README.md)，暂不扩展barrier扫描 |
| 每条 tensor 独立通过 GPC core 服务 | 保留原价格并取消整请求合并后，G3 S6/S4均约79µs，明显高估且stage响应接近零 | 该参数版本不采用；746797固定小分配的源共享对照中，无打点Q1/Q4周期下降约19.54%/28.58%，完全独立收费未能解释；[共享对照](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-source-sharing-development-001/RESULTS.md)保留量化失配；见[回放](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gemm-source-development-001/gemm-core-sharing-001/README.md) |
| 本地计算干扰与跨 CTA 服务共同作用 | 746720两个新端点支持冻结的局部干扰响应，排除仅频率解释；完整组合仍未通过 | 区分本地指令发出、SMEM访问与队列反压，见[局部干扰留出](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-local-interference-heldout-001/README.md) |

<a id="bounded-supply-candidates"></a>

**2026-10-15 有限新候选：按请求大小分道。** 本轮只有下面两个固定版本，已实现并完成[提前筛查](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-size-lanes-service-001/RESULTS.md)：均在`q1-skew8`违反原ns/cycle门槛，停止，不接入组合；信用归还时点仍未区分。执行顺序与预算只在[本轮安排](TMA-SHARED-SERVICE.md#bounded-supply-round)维护。

旧[两通道服务](../../../../../../microbench/gh200_resource_campaign/access_rules/tma_gpc_service.py)按最早空闲前沿分配 atom，每条前沿在该 atom 的 `service_end` 即可复用。[上轮双粒度实现](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-dual-granule-service-001/model.py)改用单 sector FIFO 和半价；`q1-skew8` 从旧预测493.3378901113151ns变成479.5544069486027ns，差13.7834831627124ns，与旧A服务价的一半相差不足1e-9ns。4096/8192两个容量均没有信用等待，三容量的失败结论见[筛查结果](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-dual-granule-service-001/RESULTS.md)。

单独恢复原仲裁、再比较两种信用归还时点仍有解析反例：747595的GPC4有16个CTA，Q1每CTA最多一组256条line，占用上界4096；四个Ktile的不同源box均128B对齐，384种源身份之间没有line交集。同GPC同源CTA的请求与释放锁步，line订阅退化为原整atom合并，两种归还规则都会保留[原预测](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-source-scope-discrimination-001/predictions.json)的+15.545586%，与[实测](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-source-scope-discrimination-001/RESULTS.md)的+24.816997%相差9.271411个百分点，超过原主端点容差。这两个只改信用的形式不进入批量筛查。

| 版本 | 共同服务规则 | 唯一区别：128B行代次的信用归还 |
|---|---|---|
| `size_ready` | 逻辑16KiB请求固定进通道0，逻辑8KiB请求固定进通道1，各自FIFO，不借另一通道；32B计新增源工作，128B计在途信用 | 该行实际源就绪事件 |
| `size_destination` | 同上 | 该代所有已登记订阅的最后一个实际目的完成事件 |

分类读取逻辑请求大小，行距增加的覆盖量只改变工作量。保留两个服务前沿及[原两通道价格](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gpc-heldout-002/predictions.json)：τ=27.566966323761964ns/16KiB、L=263.21313675301536ns、D=32.96152869828381ns/16KiB。每GPC固定4096条行信用，沿用上轮CPU筛查取点；本轮不扫描容量、价格、通道数或路由函数。目的沿用每CTA FIFO与D，GEMM的缓存/准备访问、频率、计算/槽释放、输出及plain映射沿用原12条件版本。

每个实际入场且有新增sector的atom，先取得全部新增行信用，再在指定通道上安排服务：`start=max(admit,frontier[channel])`，`end=start+τ×新增sector数/512`，该前沿更新为`end`。该atom新增的sector统一在`end+L`源就绪；GEMM再与既有缓存数据就绪路径取最大值。完全复用请求只订阅已有sector；同GPC同line最新代按`t < line.service_end`接受订阅，`line.service_end`取该代所有sector服务结束的最大值。缺失sector由本次atom服务追加，先前订阅只等待自己需要的sector。两版的atom服务前沿都在`service_end`复用，信用按上表另行归还。

接线须提供实际line就绪定时事件：追加sector若推迟该代就绪，旧定时事件失效，按新时刻归还。源就绪版归还后保留目的引用至完成；目的版在最后目的回调归还。每次归还都重新计算FIFO队首的可复用行并尝试入场。定时事件已通过[事件引擎](../../../../../../microbench/gh200_resource_campaign/access_rules/supply_model.py)的`input_source_bind_events`接入，按时调用`callback(time)`。

这组候选检验两件事：按请求大小分道能否改变固定源重排的等待与空间响应；在同一路由下，信用归还时点能否改变Q2/Q4及GEMM槽复用。旧模型`q2-skew32`轨迹按atom的`source_ready`或目的`ready`归还全部line，峰值分别为4032/4224，夹住4096；数据在[原请求时间线](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gpc-heldout-002/evidence/c2-q2-skew32.json)。这是旧atom轨迹的信用计账，新路由后的端点和占用由事件执行产生。每个版本还须同时面对A16、B32、B80/B112及S6/S4；响应幅度在CPU筛查后逐项输出。

## 3. 实验验证

下面记录已完成的开发干预及其可排除范围；新实验依[多 CTA 主线](TMA-SHARED-SERVICE.md)执行。测过的干预已属开发证据，不能重命名为新候选的独立留出。

[746679 同卡复测](../../../../../../results/gh200_resource_campaign/access_rules/20261015-g3-phase-reproduction-001/README.md)原样使用 V09 的 G3 四个二进制，S6/S4 的 plain 中位数为48.384/54.240µs，S4慢12.10%。首/后主循环中，S6约33145/33369cycle，S4约35767/39777cycle；两者均重现首段频率低于后段。后续分别检验新增周期与阶段换算率。

[746729 同一 S6 二进制的活动槽对照](../../../../../../results/gh200_resource_campaign/access_rules/20261015-g3-active-stages-development-001/README.md)固定215,040B SMEM，只改变运行时4/5/6个输入槽：4槽比6槽慢9.85%，5槽的第二L周期只高0.99%。[有限递推核对](../../../../../../results/gh200_resource_campaign/access_rules/20261015-g3-active-stages-development-001/analysis-stage-audit/README.md)未发现槽状态或释放顺序错误；阶段频率改变复用环的可隐藏时延，调用平均频率会丢失该阈值响应。当前条件回放仍低估CTA空间跨度；[阶段频率与源状态联合回放](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gemm-source-development-001/gemm-phase-state-001/README.md)仍只有30–46ns的第二MAIN跨度，下一组合须同时保留阶段换算与非同质服务。

[746824计算量干预](../../../../../../results/gh200_resource_campaign/access_rules/20261015-g3-compute-credit-discrimination-003/RESULTS.md)保持同一S6布局和每Ktile的TMA/commit/wait/release次数，将WGMMA工作改为三遍。两段4槽−6槽差值的三遍/单遍比为−0.00150/0.01260，均支持有限信用候选，排除固定不可隐藏的stage附加周期。原程序的CTA空间差异继续由地址与执行位置干预辨识。

[746841地址换位](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-address-placement-discrimination-001/RESULTS.md)的冻结后段窗口排除了两种固定响应候选；未插桩全段仍随源坐标交换翻号。旧新作业在相同虚拟地址描述符下也发生基线翻号。[746864同分配暖机历史配对](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-warm-history-paired-001/RESULTS.md)的两种顺序均低于冻结分辨率；[746888虚拟/物理分配对照](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-virtual-physical-development-001/RESULTS.md)的两项主效应也未达到分辨率，均未复现翻号。[746982 页内偏移](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-source-offset-development-001/RESULTS.md)中，A +1 MiB的配对效应12/12为正，中位2.053个百分点，但未达冻结4个百分点分辨率；B +4 KiB及交互也未通过。本轮停止偏移扫描，不添加固定地址组价格；跨作业状态仍未辨识。

另在cfg_a S6的640×1984×7360、zero条件中，[746997固定调用次数判别](../../../../../../results/gh200_resource_campaign/access_rules/20261015-d226-fixed-history-discrimination-001/RESULTS.md)支持局部两状态响应：奇目标比偶目标慢8.20%，S4控制为1.15%。28个dual进程的CTA0–7按奇偶切换两套SM放置，其余88个CTA放置不变；后者仍有完成跨度变化。[747022工作放置判别](../../../../../../results/gh200_resource_campaign/access_rules/20261015-d226-work-placement-discrimination-001/RESULTS.md)已通过冻结响应：互换前四个cluster的工作到SM对应后，S6奇偶差由+8.66%反转为−10.20%，S4控制在±3%内。未移动CTA的完成时刻也改变；[端点分解](../../../../../../results/gh200_resource_campaign/access_rules/20261015-d226-work-placement-discrimination-001/endpoint-constraints.md)将主要增量定位在FIRST→MAIN。[747049局部A2/A4交换](../../../../../../results/gh200_resource_campaign/access_rules/20261015-d226-a-source-discrimination-001/RESULTS.md)保持B源坐标后，CTA28主循环周期下降9.21%，完整轮换下降13.16%，响应比0.700；两种冻结定量形式均被反证。三种工作图已绑定全部端点，下一候选须联合解释局部源关系与背景工作。

[本地生产者延迟的组合筛查](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-local-producer-candidate-001/README.md)只在本CTA计算活跃时增加40/80cycle的逐指令发出延迟，两stage共用参数。80cycle使S4首/后L误差降至−1.36%/−3.01%，但第二MAIN跨度仍仅50.65ns，低于全部实测1728–3456ns；S6跨度仍为30.45ns。两组参数均不接入，停止固定延迟扩扫，本轮未使用GPU判别预算。后续继续检查共享源完成与独立目的投递规则，并回到同一组合矩阵。

### 下一候选的联合约束入口

已有[逐项结论](../../../experiments/gh200_sm90/Q02-tma-request/reanalysis/20261012-consolidation/R13/README.md#已有约束逐项结论)和[15 份报告的机器可读汇总](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/B-20261011-lifecycle-v3/B-existing-constraints.json)。2026-10-14 复核这 15 份来源的 SHA256 均一致；从既有逐进程导出重算了预等响应与完成跨度，没有重新解析原始 GPU 日志。下面按需要解释的问题连接证据与程序端点，原统计继续在实验页维护。

| 需要同时解释的问题 | 现有证据与约束 | 请求级组合中的对应与缺口 |
|---|---|---|
| 延迟发生的位置：就绪之前或消费者返回之后 | [739653 ready 区间](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#ready-observer)：40/40 次绑定首代事务；两种行距的 20 个配对均要求 issue→ready 增量为正，ready→FIRST 增量区间均含零 | 逐请求完成汇合到 `full_ready`，之后才到 `consumer_wait`。实验 `issue_end` 是 copy 后的软件标记，比较时还需其观察位置；barrier 可见区间保留上下界 |
| 预等对局部 early 与累计推进的不同影响 | [739790 同批配对](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#second-prefill)：同时检查 early、post、预等开始→C16/M；S4/S6 分列 | `early=C16−C1`、`post=M−C16`，其中 M 在当前候选中对应 `math_handoff`；预等开始是另一个软件端点。共同程序当前只展开未干预分支，预等与输出延期仍由原引擎回放 |
| 跨 CTA 完成时刻的离散 | [同批空间跨度](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#source-candidate-replay)：共享参数虽接近 S4 的 early 响应，两套参数在四条件下的跨度仍低于全部进程 | 此处取全 CTA 第二 WG 的 `mma` 返回，当前候选映射到 `j1.math_handoff`。另一个 [739095 排序约束](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#stage-order-invariant)用 `FIRST0→MAIN1`，末端对应 `j1.main_end`，两种端点分别评分 |
| 局部供给对其他 CTA 的响应 | [739740 背景输出延期](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#stage-background-baseline)：固定本 CTA 处理后，早段缩短而后段变长；处理也改变背景输入请求 | 对照整卡输入、输出请求的时间顺序及本 CTA 的 ready/计算/释放，寻找第一处响应差异；仅本地 E0 排他已不足以解释 |
| 空间差与地址、执行位置及时间状态的关系 | [A 源轮换](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#a-source-rotation-preparation)与[R09 周期分解](../../../experiments/gh200_sm90/Q09-frequency/R09-inkernel-clock-stages.md#stage-order-decomposition)：固定地址组价格未获支持，S4/S6 的周期与 ns 分解也不同 | 保留源身份、CTA 工作与实际 SM 放置；比较原生 cycle 和同调用 ns。下一实验从局部并发与背景负载区分开始，顺序见[多 CTA 判别步骤](TMA-SHARED-SERVICE.md#近期判别顺序) |

近期比较先使用上述原条件和端点：分别报告局部响应、累计窗口、空间跨度/排序，以及跨条件迁移；某项改善不抵消另一项失败。请求级接线入口与适用域见[组合 §4](../composition.md#4-输出)。

### 有限槽、交接与局部端点

[入口条件检验](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#stage-conditioned)用后段slot关系排除了旧ℓ=2257.5用于本批。[两stage共同开发](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#stage-reuse)得到后段复用组合Z约1755～1773 cycle；S−1来自软件释放依赖，不是物理队列深度。完整继承子类仍缺stage4早段约2079 cycle，不能作为完整预测交付，也不能给early另加专属常数。任意单一producer边界间隙仍不能突破该子类早段界；更宽分支的[联合端点约束](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#stage-joint)只排除部分逐调用解释，不证明中央代表轨迹不可行。[同调用整卡上下文](../../../experiments/gh200_sm90/Q06-handoff-output/R15-output-service.md#stage-output-context)还显示重叠面积本身不能定价，需保留stage输入复用与入口相位。诊断参数与实测入口均未进入正式预测。 739717改变本CTA输出顺序后仍有早段缺口；739740进一步固定本CTA处理，仅延后其它CTA输出，stage4早段又10/10缩短，配对中位−1008.5 cycle，但后段变长、plain未改善。局部供给必须允许全卡组织的影响；该处理也改变其它CTA输入请求，未识别唯一物理资源，未加入并发价格。延后的E0进入后段，故只能用选定无E0覆盖窗口约束有效c/T，不能将整段flag2作为清洁基线，见[R13条件迁移](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#stage-background-baseline)。见[R15同调用背景配对](../../../experiments/gh200_sm90/Q06-handoff-output/R15-output-service.md#stage-output-background)。

后续候选必须保留时间状态。[同位置起步干预](../../../experiments/gh200_sm90/Q03-source-service/R10-layout-cache.md#producer-delay)在工作/SM不变时改变了行距响应，而且B0在扣除实际producer后移量后，首输出仍多耗816 ns、1156.5 cycle。固定事件前驱、固定同域边成本且其它根时刻不变的max-plus图，不能把单根延迟放大；仅平移旧时间线或增加固定位置罚项不充分。该约束不排除带可变仲裁、等待返回与初态的有限stage模型。producer/FIRST软件名次也不是物理请求到达，不直接拟合为服务价格。 [首次供给端点](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#first-supply)把额外量定位在copy发出标记到FIRST之间，首次发射前成本与consumer进入等待时刻未明显增长；仍不能把此段当裸TMA延迟。[释放前端点约束的ready观察](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#ready-observer)在新批40/40次绑定首代事务：延迟干预的issue→ready增量下界在两种行距各十次均正，ready→FIRST增量区间全部含零，排除本协议下仅consumer返回变慢的解释。请求排队、数据服务与完成可见性仍不可分；B32的L缩短也大多兼容等待移入S，不能直接提高稳态供给价。

[第二WG联合初态对照](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#second-prefill)进一步支持保留交接状态：同时预等S槽、排空前WG并改变相位，使stage4 early十对均缩短，stage6基本不变，完整调用未稳定改善。候选须由槽ready与未完成计算依赖预测该响应；全预填后，恒定reduced复用关系的无参数窗口界仍在stage4十次全部失败，故仅修初态不够。此界不覆盖宽提交/退休分支；后续off-only宽拟合虽能使部分窗口RMS降至0.28%，仍失配局部端点和early干预，不采用。内部量只能辨识u=ℓ+a，交接δ也存在训练量不变方向；不能把分开的拟合系数解释为硬件价格。从预等开始核算，stage4到C16/M均未稳定缩短，stage6十次均变晚；early收缩包含等待重分配，不采用early专属价格。FIRST在barrier返回后、首HGMMA前；统一TC起动偏移在q≥v时只重参数化q/λ，不能修复当前响应。尚不能分离三个作用。stage6后段与更多E0重叠而周期基本不变，也排除了直接按重叠数收费的解释。

<a id="finite-source"></a>

### 共享请求与源身份

[supply_model.shared_stage_events](../../../../../../microbench/gh200_resource_campaign/access_rules/supply_model.py)给当前cfg_b两输出组织实现一个纯事件候选。对软件前驱允许发出的请求j，FIFO服务结束为`F_j=max(a_j,F_(j−1))+τv_j`，就绪/源复用为`R_j=F_j+ℓ_type`。每Ktile的A16KiB取v=1、B0/B1各8KiB取v=1/2，输出8KiB取v=η/2；这只是逻辑服务，不命名为物理L2/HBM。输入S槽、普通/尾释放、输出8子tile和双缓冲都显式保留，E0相位由图生成。

当前同质请求及无本地TC竞争的版本不再作为待调参候选：[R13](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#stage-shared-service)的联合窗口最大残差虽只有3.32%，但全卡M跨度过窄、本CTA输出干预效应不足。两低误差校准解也不唯一。它不参与V09预测；测后E0、入口或完成排序不进入参数。末端仅覆盖源读取尾，不是全局D可见或完整CUDA时间。[请求形成](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#stage-request-formation)已纠正B的请求粒度；[分阶段计时](../../../experiments/gh200_sm90/Q09-frequency/R09-inkernel-clock-stages.md#stage-phase-clock)显示首/次输出换算率不同，当前统一ns服务价仍是待检验假设。观测PROD仅在单列诊断中作输入；两个未标定的有限gap示例消除了微小起步偏移的全卡分布跳变，说明零间隔并列处理影响旧结果。现有61cycle组外窗口没有分离A→B0、B0→B1和跨组控制工作，尚不能给它们定价。更强的[保序反例](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#stage-order-invariant)表明：即使允许任意producer起步偏移，同质FIFO图也不能产生实测的FIRST0→MAIN1严格逆序。因此停止用统一gap或入口调参修复这张图；下一候选必须解释非同质推进，不能据此宣称识别某种硬件仲裁。[分量替换](../../../experiments/gh200_sm90/Q09-frequency/R09-inkernel-clock-stages.md#stage-order-decomposition)只用于诊断：stage4保留较强周期成本差，stage6排序更依赖观测周期率；没有将这些观测值或比例接入预测。

在途源身份扩展仍是开发候选，未接入正式预测：[两套具体参数复放](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#source-candidate-replay)中，共享版本能给出stage4预等early缩短约745ns、stage6不变的方向，但四条件的全CTA mma返回跨度都低于全部实测进程；不共享参数同时失配stage4累计响应。保留“源服务可共享、目的投递独立”的候选结构，拒绝将这两套参数当作可组合规则。三种同时请求插入顺序未修复跨度；下一步用既有逐CTA阶段与时钟记录约束非同质推进，不拟合stage专属价，不用观测入口/频率修正正式预测。

[源组分解](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#source-spatial)进一步限定缺口：739095的stage4首/后输出中，A行组均值解释周期离散的94.6%/96.5%，组均值跨度约2356/2463cycle；stage6周期差更小而ns成组性仍强。A组、CTA与SM集合在这20进程完全绑定，不能定为A源服务价格。当前源键相等关系模型对键的一一重命名严格不变；这一不变量可用于设计地址/位置对照，不加入组编号罚项。

同一共享参数的[20进程producer切口](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#source-offsets)也失败：原幅起步偏移下stage4/6的第二MAIN跨度中位仅129.9/318.3ns，实测2624/1136ns。trial0内部记录显示每个A/B源generation都被完整同组订阅，共同ready强烈同步了请求；这不是硬件复用率。有限gap下微小起步扰动未复现旧并列事件跳变，停止为这套参数调入口。非同质推进仍需由前驱、目的服务或源状态解释，不能转成组标签价格。

[A源单独轮换](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#a-source-rotation-preparation)740033的首窗口10/10配对通过同SM/无E0覆盖门槛，但固定位置对应仅8/10优于跟随地址，两个方向残差仍大；同档重复的组曲线也不稳定。不支持将空间差直接定成A地址组价格，也未识别唯一位置机制。后窗口10/10不能替代主窗口；完整时间变化范围跨零，不能当成严格零响应验证。保留共享状态与局部推进的候选边界，不追加地址矩阵。

[源订阅截止检验](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#source-join-boundary)未提供可采用的新参数：服务结束/开始截止虽减弱同步，但有界两坐标重拟后仍将stage4预等early预测为增长311/319ns，与十次实测缩短相反；ready截止保留方向而空间跨度仅123ns。默认模型保持原ready边界，其余选择仅作诊断。局部拟合退化、分支不光滑及单起点范围均保留，不宣称全族排除；下一步约束局部发射/消费与共享服务的组合，不再扩大截止时间扫描。

[本地E0排他](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#local-epilogue-exclusion)沿用既有prepare/write时长，不新增数值参数；stage4输出延期响应最多扩大到约−0.86%，仍不足以解释旧批约−6.82%的周期响应。defer2下本CTA全部输入与MMA起动先于E0，局部排他严格不改变图，不能靠放大E0价格修复剩余stage4缺口。该候选保留为开发诊断，默认关闭；下一专业供给轮次由B负责。

## 4. 结论

当前采用“需要时间状态、局部推进与跨 CTA 条件”的工作方向，不采用上述失败参数作为硬件价格。备选与反证范围见上表；只有能够同时预测现有反例并对未测干预给出不同响应的候选，才进入下一次冻结。半经验 B 源响应式仍在[MODEL-CANDIDATES](../../semi_empirical/MODEL-CANDIDATES.md#source-sharing-dev)维护，两者不相互授予通过资格。
