# 三类模型与机制研究推进计划

更新：2026-10-23。目标、选题与验收规则见[GOALS](GOALS.md)，模型差别见[对照表](mechanistic_modeling.md)。本页维护顺序和依赖；各轮登记与结果在对应实验记录中，入口为[按问题索引](experiments/gh200_sm90/README.md#research-questions)，当前分工见[EXECUTION](experiments/gh200_sm90/access_rules/EXECUTION.md)。

## 1. 近期顺序

**当前主交付（2026-10-22）**：整体重建输入供给服务，保留已验证的程序、计算／释放及其余模块。源读取与复用、每CTA应答、投递完成及回压共同决定时间，不再要求新结构继承旧core价格或与其取max的完成接法。先用[132CTA源／目的判别](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-shared-swizzle-001/plan.json)决定需要的状态；下一版同时面对持续行距增长、三遍隐藏、同行关闭和源折叠反增，再按原C2/C3验收。单CTA swizzle补测和旧固定价格下的队列／容量微调不另开轮次。

132CTA判别已完成，Q1/Q4均信息不足，见[结果](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-shared-swizzle-001/README.md)。下一交付直接做输入服务整体替代：至多两种结构分别表示带地址路由的共享读取／应答、本地组装／投递与计算的联合推进；同一结构内的输入服务参数共同约束，不继承旧core价格的资格。先在G3基础／A16检查完整时间线、两段主循环及full状态，再检查已有三遍隐藏、同行关闭与源折叠响应；通过必要约束后进入S4、B80／B112及原12。程序与已有计算／释放、频率、输出保持，局部辨识不作为实现的前置门槛。

2026-10-19 起按下表的五阶段推进，覆盖范围仍为九个方向；九方向及 10-17 首版十阶段的交付见[微基准地图 §6](model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)，授权与单轮要求见 [GOALS §5.4–5.6](GOALS.md#current-scope)。

| 阶段 | 交付与复用 | 进入下一步的依据 |
|---|---|---|
| 1. 固定真实程序事实 | 复用[cfg_b程序图](model/mechanism/program/cutlass_program_graph.md)、源码／SASS核对、工作列表、真实三请求、槽代次与输出顺序 | 已完成的程序事实直接使用；新增干预只核对实际改变的程序部分 |
| 2. 建立最小模块 | 分别交付搬运、计算、同步／释放和输出的独立运行入口及端点时间线，接口与端点一致性按[GOALS的模块交付要求](GOALS.md#current-scope)；复用已有接口、A/Q05、B/C、TMA与输出证据，标明当前假设及缺口 | 所需外部行为可以独立运行、对照局部证据。寄存器／驻留先提供容量和启动依赖，普通L1／LSU保持独立收口 |
| 3. 验证最小连接 | 接通输入就绪与full返回，并闭合计算完成、WGMMA等待返回、empty发布／槽释放和producer再取得槽；一次接入一条连接，固定其余模块；复用[shared_stage_events](../../../microbench/gh200_resource_campaign/access_rules/supply_model.py)、[ConditionSyncService](../../../microbench/gh200_resource_campaign/access_rules/mbarrier_service.py)及[短链生命周期](../../../microbench/gh200_resource_campaign/access_rules/cfg_b_handoff_chain.py) | 同一规则在已有短／长链及S4/S6上通过登记的必要约束；失败定位到新增连接，信息不足时列出最小缺失观测，继续不依赖它的工作 |
| 4. 加入共享与竞争 | 相关局部连接通过所需约束后，接共享源和输出重叠；复用多CTA源状态、两域、源重排及输入／输出四组织证据 | 分别检验独立组合与联合服务，明确新增共享状态及其端点响应；多CTA研究的独立进入条件见[当前执行范围](GOALS.md#current-scope) |
| 5. 完整GEMM验收 | 相关模块、局部连接和交互完成登记的检验后，接回真实工作列表，完成原12条件的同版开发回归；再冻结并取得独立C2/C3证据，交付执行时间线、关键端点、stage／行距／CTA推进及plain预测 | 按原C2/C3标准验收；经验替代和失败条件保留 |

当前状态（2026-10-23）：[逐CTA等值源副本与同服务接回](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-private-source-transfer-001/README.md)已完成，源大响应能够预测，但阶段端点及原源行距仍失败；[片段共同服务](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-gather-joint-service-001/README.md)也未形成持续full未完成，固定版本停止。[请求形成前置](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-formation-before-cache-001/README.md)完成432次CPU调用：原源四L最大误差降到9.338%，plain仍超门；纯TMA Q1、full状态及私有源阶段端点失败，当前参数版本停止。下一交付共同约束输入返回、full就绪与消费者查询位置，保留持续行距响应；原源和私有源两种工况同时检验。此前最新G3 S6／S4／A16关键路径核对已完成，见[局部规则接回GEMM](experiments/gh200_sm90/Q14-composition/gemm-connection.md)。后续g11观察的实际开关扰动与向R桥接通过，但两种观察组织的区间仍覆盖两个控制解释，见[源组织](experiments/gh200_sm90/Q02-tma-request/source-organization.md)。132源g11跨CTA观察的结束跨度资格未过，信息不足，同见源组织；输入/输出同批开发资格未通过，见[联合窗口](experiments/gh200_sm90/Q07-smem/input-output-joint.md)。完整模型参数及独立C2/C3状态保持。

本轮收口：生产者节流CPU的两解释响应相同；773128局部相位区间仍宽496cycle；774013输入/输出同批检验未过开发资格，未冻结或采留出。随后[g11跨CTA观测](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261021-Q02-late-full-span-001/README.md)完成44次CPU核对并冻结可分预测，774095构建通过；774105九进程数值／暖机通过，总窗口与consumer戳扰动通过，CTA结束跨度的扰动和向R桥接未过，信息不足，不扩大。旧首full形式与整体失败记录保持。另[等待身份CPU筛查](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-wait-path-screen-001/README.md)四调用均预测g11快等待，停止布尔观察选题，未构建／补测。

当前待解为输入就绪→full返回→计算／释放的持续响应：producer控制与请求受理已接通，逐轮准备的局部长度响应已获支持，旧局部组合未产生GEMM行距增长；请求形成前置候选已产生增长，full状态及跨程序端点仍失败。CTA结束跨度本批原R范围/中位为21.642%；下一次跨度判别先用已有重复性定区间，不重复本轮5%跨度观察。程序与输入、固定/可改部分、预测和有限清单在新一轮执行前登记，沿用GOALS要求。

程序准备接线已收口，见[本轮实现与预测](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-program-preparation-001/README.md)：八默认报告逐字保持、47项测试通过，140/160次CPU调用、拟合/GPU0。两旧控制解释在名义640cycle干预下的跨stage响应带分开，实机成本校准结果见下段。

准备延迟轮已收口，见[原始评分与结论](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-preparation-delay-001/README.md)：S6九进程资格与关闭臂桥接通过，响应6618–6699cycle超出冻结校准域，未测S4，不扩展该形式。随后CPU区分了原有producer准备的逐轮推进与首段放置，复用本轮响应、旧长度／stage及首A约束；受理／就绪与计算／释放固定，没有把反推的991–995cycle当价格，也未重复未变化的原12回放。

逐轮准备CPU轮已完成256项核对，见[结果](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-recurrent-preparation-001/report.json)。[独立长度外推](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-preparation-length-001/README.md)774347完成十八进程，资格／桥接通过；K4／K32响应均落在冻结门槛内，首段形式被排除。N0−R随长度增长，固定J及P／源残量分配未解；原两个consumer控制解释仍未区分。

[逐轮准备替换旧gap](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-producer-preparation-transfer-001/README.md)已完成：G3/A16固定输入和源／输出／频率，P240与309.75两档正式44调用，另11调用为补齐保存依赖前的实现失败。两档行距响应均约0.0931%（实测28.9371%），first_L／later_L增长均0；停止两参数版本，不扩展stage／原12。下一项从已有匹配源／行距证据选取会改变持续服务的局部判断，不继续准备价格扫描。

计算间隔与A16行距对照已收口，见[结果](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-stride-compute-discrimination-001/README.md)。774436单遍十二进程资格通过，两段周期增量低于冻结1000cycle下限，三遍未测；两响应假设尚未判定。下一项先恢复改造程序的单遍原始执行路径，并检验是否保留G3/A16反例，再决定计算量判别。

[恢复单遍路径](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-direct-repeat-path-001/README.md)774459完成24进程；同批原R重现两段约9.9k／11.8kcycle行距增量，新N1为66.5–271.5／2459.5–2818cycle，基础plain慢约22%，桥接未过、N3未采。[固定原S6推进](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-static-stage-repeat-001/README.md)774550完成24合格进程，仍未保留反例，N3未采；SASS发现运行时开关汇合路径的额外ARRIVE和特殊HGMMA。[循环外选择静态计算量](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-static-repeat-dispatch-001/README.md)由host选择同一文件的静态单遍／三遍实例；774570资源解析失败后，修复1的774581构建成功。774591完成36合格进程、R／N1桥接通过，两段行距增量比均落入冻结可隐藏带，排除本条件的固定不可隐藏加项；未拟合控制价格。下一项CPU同时约束单遍正增长与三遍近零，固定计算／释放／频率／输出，选择有依据的持续供给或输入就绪连接；不扩展干预构建。

[在途源行订阅CPU筛查](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-line-subscription-screen-001/README.md)已完成8合成例、44引擎调用。原两atom前沿与价格固定，32B／128B版本的A16工作分别增1.716%／12.012%，两L增长仍0；单遍必要门失败即停止，三遍未求解，GPU／拟合0。新增工作进入源就绪后被消费推进遮蔽。下一规则须解释单遍持续暴露的供给状态，并保留三遍隐藏响应；不继续扫描行粒度、τ或合并窗口。 后续先核对R12与已有COMPUTE-01 D链的工作比、源复用及联合边界；R12的TMA／WGMMA为1KiB／条，cfg_b为4KiB／条，不能直接迁移近完全重叠关系，出处见本轮记录。

[SMEM联合服务必要下界](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-joint-smem-bound-001/README.md)已收口：原生D的K32窗口排除全部逻辑引用共同128B/cycle形式；相邻B复用共同服务与独立路径保留。R12配比偏低，但D已覆盖真实4KiB／条配比。模型引擎／拟合／构建／GPU0；下一项处理跨CTA源共享／次序如何形成持续状态，不新增局部配比或操作数次序扫描。

[后续full进入状态位图](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-full-entry-mask-001/plan.json)：先冻结当前局部组合在G3／A16全部CTA后续Ktile的ready预测，CPU上限22调用／15分钟。可分后，用原等待前的非阻塞test_wait累积位图、循环后导出；同binary关闭／开启及原R同批核扰动和桥接。仅S6，8开发条件各三进程，上限一次零GPU构建10分钟、一次GPU5分钟，两次实现修复；不改价、不扩S4或三遍。

full位图实现1已按SASS收口：774786／774800的24进程数值／时间门通过，但编译使用的overlay头没有位图逻辑，全零无机制效力，见[语义核对](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-full-entry-mask-001/semantic-audit.json)。[实现修复2](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-full-entry-mask-002/plan.json)只把相同观察接到实际overlay，复用原冻结预测与门槛，CPU模型调用0；先核目标SASS的查询、开关、累积和导出，再测同8条件24进程，最多一次零GPU构建10分钟／一次GPU5分钟，无追加修复。

[full状态实机判别收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-full-entry-mask-002/README.md)：774830的24进程全部有效，R→N0／N0→N1端点最大变化3.111%／2.999%，行距增量保持；A16后续查询未完成率91.823%，基础4.099%，反证当前全ready预测。下一项固定计算／释放／频率／输出，先以位图及原三遍隐藏响应约束持续供给、组完成与full可见的连接；位图不用于反推等待价格。本修复无模型调用或拟合，组合原12成绩及独立C2/C3状态保持。

[g43的producer发布与full未完成](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-001/plan.json)：固定G3/A16、S6及所有服务价，先冻结两种发布位置主导类；只在g43的full查询前后读取producer的B1后发布数。三次原R／已合格full观察F／同binary关闭N0／开启N1，共8条件24进程；实际端点扰动与桥接沿5%，未完成状态须保留基础≤10%、A16≥50%。原子发布只提供软件位置，不作TMA物理准入。CPU模型／拟合0，最多一次零GPU构建10分钟、一次GPU5分钟、两次定向修复；资格失败收口，不扩S4或三遍。

[g43发布观察收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-001/README.md)：774863完成R／F／N0共18有效进程，三组端点最大变化3.955%／3.918%／4.986%，行距保持通过；N0基础trial2第二输出17/132未完成，超过冻结10%，故未采N1，发布前／后仍未区分。稀疏g43形式停止，不放宽门槛或补跑；F同批基础全ready、A16持续未完成保持。最小缺口仍为full查询相对B1后发布的位置；如更换观察，保留F逐Ktile查询骨架、只在g43取发布快照，另行冻结有限清单。CPU模型／拟合0，组合成绩保持。

[保留逐Ktile查询的发布位置观察](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-002/plan.json)：依据774863的F／稀疏N0差异，恢复F完整查询位图，仅g43读取B1后发布数；三请求、计算／释放、价格及时间门槛保持。R／F／同binary N0／N1，G3基础/A16各三进程，上限8条件24进程、一次构建10分钟、一次GPU5分钟、两次定向修复，模型／拟合0。按GOALS逐输出条件保留资格；六个A16条件齐全且各位置比例≥75%才判整体主导类。

[CTA本地发布观察](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-003/plan.json)：775011的全局原子N1将A16 g43未完成降至41/792、状态保持失败，停止该形式。仅将计数与查询前后读取改为本地SMEM原子，保留逐Ktile查询；利用原__syncthreads初始化，只在加载循环结束导出。R/F/N0/N1共8条件24进程，一次构建10分钟／一次GPU5分钟、两次定向修复；原门槛及逐条件资格保持，模型／拟合0。

[本地发布实现的对齐修复](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-003/repair-001/plan.json)：775039原R/F十二进程有效，新N0六进程均报misaligned address、N1未采。保留失败实现，新增共享字改为1024B对齐块以恢复原动态缓冲起始对齐；同一冻结预测和8条件清单，另一次有限构建10分钟／GPU5分钟、最多24进程。属于实施修复，不新增硬件候选；本次不扩大条件或改门槛。

[逐Ktile全局原子形式收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-002/README.md)：775011完成24有效进程，F/N0六前置条件通过；N1将A16 g43未完成从681/792降为41/792，随后g44也改变，六位置资格均失败。停止全局原子形式；已转本地SMEM观察及其对齐修复，不改模型价格。

[本地发布定位收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-003/README.md)：775049的24尝试中23有效，R基础trial1暖机9次被剔除；另一个输出未过状态保持。三合格输出的306次full未完成均已发布，且查询前已发后续三组，反证预列发布前全域主导类。冻结六条件主张仍未全部验证，不补齐批次。下一项CPU核模型的发布进度与full查询／完成位置，固定其他模块，不把已发指令当物理准入或用次数定等待价；模型调用／拟合0，原12成绩保持。

[发布与查询时间线CPU核对](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-publication-time-alignment-001/plan.json)：固定原输入、九次调用协议及全部服务，只对G3基础/A16导出g43发布、查询及full完成位置；上限18引擎调用／15分钟，GPU与拟合0。先复现旧预测，再定位首次状态分歧；不据计数增加等待价格。

[发布与查询CPU核对收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-publication-time-alignment-001/README.md)：18调用逐位复现旧预测，两布局全部CTA均已发47/111组，与合格观测一致；A16仍预测full提前639–881cycle完成。下一项最小判别量为同组B1后至查询的周期区间，先区分完成过早与查询过晚，不重算未变化模型。

[同组B1后至full查询的周期区间](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-published-request-age-001/plan.json)：由保存时间线冻结两输出完成年龄及查询年龄；保留逐Ktile查询，仅每输出g43的B1后与查询前后读clock64。R/F/同binary N0/N1两布局各三进程，24进程、一次构建10分钟／GPU5分钟；原时间和状态门槛保持，按合格输出判别完成过早与查询过晚，模型／拟合0。

[同组年龄定位收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-published-request-age-001/README.md)：775061完成24有效进程，同binary最大扰动2.668%；五合格输出的530次full未完成，B1后查询下界1815–3493cycle，均超过旧完成上界1102/1232。排除这些条件仅查询提前、保留完成价格的形式；第六项桥接失败保留，不补齐。下一项只选发布后完成连接的有依据规则，固定计算／释放／频率／输出；保留原单遍正增长／三遍隐藏约束，不把年龄下界直接作价格。

[发布后年龄的空间分组核对](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-completion-spatial-screen-001/plan.json)：复用775061五个合格A16输出及基础对照，核raw工作坐标后，分别比较A/B源组与实际SM的跨进程年龄形状；留一进程比较只选下一干预，不回填预测。CPU上限2分钟，模型调用／拟合／GPU0。普通.cg的8.3TB/s不直接迁移为tensor服务价。

[年龄空间分组收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-completion-spatial-screen-001/README.md)：1452项raw坐标与程序吻合；A16的A组在全部留一进程中改善空间误差约8%/18%，实际SM轴未稳定改善。A组与逻辑CTA组绑定，下一项先计算改变raster后两解释的预测是否分开；不把A组均值作为模型价格。引擎／拟合／GPU0。

[Raster对照的预测可分性](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-raster-contrast-prediction-001/plan.json)：固定旧观测与全部服务，将raster改along_m；先核输出需求及每CTA工作数，再比较源A组跟随/逻辑CTA组跟随的投影预测带。重叠则停止该干预，不构建或采样。CPU上限2分钟，引擎／拟合／GPU0。

[Raster年龄判别](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-raster-age-discrimination-001/plan.json)：两预测带在旧重复性下分开，固定需求和每CTA工作数，只改host raster参数。两raster各做年龄关/开，基础/A16各三进程，8条件24进程；先核heuristic重复性，通过后采along_m。一次构建10分钟／GPU5分钟，原5%时间门保持；模型/拟合0。

[Raster判别当前执行状态](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-raster-age-discrimination-001/README.md)：775150完成12有效进程，36项时间检查全过、最大扰动2.1304%；原raster六项控制均未重复固定空间形状，按冻结条件停止，along_m未测。停止固定A组均值迁移形式；已找到10月18日的[实际A行距证据](experiments/gh200_sm90/Q02-tma-request/row-stride.md)：单CTA Q1持续增长、Q4只在预填增长，重复测量取消。[旧额外覆盖规则的连续预填迁移](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-coverage-prefill-001/README.md)四调用完成，Q4预测677.08对实测513cycle，误差31.98%，停止该固定参数形式。[有界入队返回](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-publication-queue-001/README.md)的容量3/6八调用保留Q1，但Q4增量670.19/382.69仍未过513cycle原带，两形式停止。775148／775150／775151均已结束。

- 阶段 1、2 已交付：程序事实，以及搬运、计算、同步／释放、输出的独立入口与端点时间线。
- 阶段 3 部分完成：计算完成→WGMMA 等待返回→empty 发布／槽释放→producer 再取得槽，在驻留链上通过冻结长度／stage 外推，见[局部同步链](experiments/gh200_sm90/Q04-sync-slot/local-sync-chain.md)、[计算完成与提交](experiments/gh200_sm90/Q05-wgmma/local-compute-chain.md)。搬运→输入就绪→full 返回在单 CTA 原生 D 链上通过长度／stage 外推，但首组 full 返回前 773 cycle 在输入就绪与阻塞返回之间的分配不可辨识，R1536 外推未通过，见[原生 D 链](experiments/gh200_sm90/Q02-tma-request/native-d-chain.md)。
- 阶段 4 已开始：共享／独立源对照见[源组织](experiments/gh200_sm90/Q02-tma-request/source-organization.md)，固定形式筛查见[共享源固定形式](experiments/gh200_sm90/Q02-tma-request/shared-source-forms.md)。源并集由 53.6 MiB 减到 26.8 MiB 的冻结响应获限定支持（763587），绝对窗口和 CTA 结束时刻分布仍未解释；10-20 的六个 CPU 固定接线及实际基址修正均未修复这两项。
- 阶段 5：默认版仍为 10-16 前缀版的 10.225%／25.422%；局部同步链规则作为显式开发选项为 8.706%／25.334%，stage 响应 6.7%（实测 10.5%），行距响应仍为 0；独立 C2/C3 未通过，见[cfg_b 组合](experiments/gh200_sm90/Q14-composition/cfg-b-composition.md)、[局部规则接回 GEMM](experiments/gh200_sm90/Q14-composition/gemm-connection.md)。

[预填间隔执行收口](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-prefill-gap-execution-001/README.md)：775192复用752276，6进程48臂有效、12582912值零错，最大扰动0.284%；循环版gap0响应59cycle低于冻结69.45cycle，信息不足，原R0/1判定不变。下一项只复用已有四组展开源码恢复预填组织，保持相同八条件与间隔，先核本版基线响应；不新增价格或容量值。775192/775193均结束。

[四组展开的间隔判别收口](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-prefill-gap-unrolled-001/README.md)：775198构建核对真实请求、gap与导出语义，REG56、零溢出；775199完成6进程48臂、12582912值零错。三个seed的A16预填增量均377→0cycle，原69.45基线及5%实际资格全过，R=0，支持本版间隔隐藏、反证保留形式。最大观察扰动0.136426%；旧绝对预填变化单列，不迁入控制价。GPU5秒、三作业结束，模型/拟合0。下一项以此约束发布反馈与完成连接，保留原Q1/Q2/Q4、单遍增长及三遍隐藏；源/前端/目的反馈位置尚未分开，不重采本八条件或扩容量。

[GPC副本投递有限筛查](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-gpc-delivery-bound-001/plan.json)：原本地投递已按每CTA计字节；本轮仅增加按每份副本计费的GPC共享约束，原源合并和其余模块保持。服务价取六份既有共享短链包络给出的必要上界，拟合0；先G3/A16单遍必要门，通过才三遍。最多44引擎调用、四合成例、CPU10分钟，GPU0，不扩容量/价格。 已收口：4合成例通过、22调用、53.9秒、GPU0；两L增量96.31／37.19 cycle，plain102.25／100.52 µs。固定上界FIFO版本未过单遍必要门，三遍未测；其他GPC投递机制未判定，见[评分](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-gpc-delivery-bound-001/score.json)。

[保留工作列表的同行CTA定位](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-active-peer-001/plan.json)：G3/S6、原grid132和CTA0两输出保持，新132/1两臂先固定16次全量前缀，仅目标调用在入口放行132/1CTA。先同批原R与新132桥接，通过后测1；两布局、plain/dual各三进程，最多36进程。开发分流区间已保存，模型/拟合0。775562仅排队后取消；775564构建、775581测量及775596回收均完成。36/36有效、零失败、三trial全SM124；桥接最大4.020351%，观察扰动最大4.049213%。新132两L增量12536–15401／11740–11943cycle，新1为8–117／1206–1362；两比值区间均入冻结同行依赖带。支持目标调用的同行工作影响大部分A16增长，源／前端／输出位置仍未分；开发定位，模型／拟合0，原12及独立验收状态保持。节点exclusive保留4GPU，测量step用1GPU，分配16秒；[原始回放](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-active-peer-001/replayed-score.json)逐值一致。

[同行输出TMA定位](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-peer-output-001/plan.json)：保留132CTA两输出的真实输入／计算、CTA0程序及全部输出STS／屏障／提交等待，只屏蔽同行SMEM→global TMA copy。新臂共同16次全写前缀，仅目标调用全写／CTA0写切换；先同批全写桥接，通过后测屏蔽臂。两布局plain／dual各三进程，最多36进程，一次零GPU构建10分钟／一次GPU5分钟、两次定向修复，模型／拟合0。两L分别按已冻结的响应保留比区间分流，开发定位；不扩stage／活跃规模或价格。 已收口：775646的gate未生效由SASS发现，定向修复1的775647通过；775662完成17秒，35/36有效，原暖机CV失败一进程未补。全写桥接三trial通过，单写保留trial0／2；两L响应比分别[0.9601194284,1.052762746]／[0.8954191188,1.118705036]，均入冻结保留带。同行全局输出TMA不是当前大部分增长的必要条件，132CTA输入／计算仍可维持；源／前端物理服务未定价。原12及独立C2/C3状态保持，见[结果](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-peer-output-001/README.md)。

[等值A源折叠定位](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-002/plan.json)：保留132CTA原两输出、计算值、B源和同步，仅正式调用把A源m17–21折叠到m0–4；CTA0 m0/11不变。原dyadic周期17保证逐值相同，先同批原源单写桥接，通过后测折叠。base/A16、plain/dual各三进程，最多36；一次零GPU构建10分钟／GPU5分钟、两次定向修复，模型／拟合0。两L按冻结响应保留比独立分流；只约束五个第二输出panel的地址组织，不预设容量或物理归因。[001](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-001/README.md)的775816未启动即取消、测量0进程；002复用775812构建，改用ROMEO分配的132SM GH200，UUID执行前绑定，三臂同批同卡；原区间、5%桥接与其余门槛保持。002的776745在romeo-a021完成14秒，24/24进程数值、暖机及观察合格；桥接仅trial0通过，未满足至少两项共同trial，Nfold未采，信息不足收口，原12及独立C2/C3状态保持。 [003邻接交换顺序](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-003/plan.json)：按原最多48进程／5分钟GPU、0构建执行，24进程桥接后收口。 003的776879同卡24/24有效但0组过桥，折叠未采；[同binary负对照](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-bridge-null-001/README.md)777026在romeo-a056完成21秒，47/48进程有效；N1 A16 plain trial0暖机CV超2%保留，未补跑。R同二进制重复3/3、N完整配对2/2通过原门；旧、新二进制等价桥接仍未通过，同二进制对照可用。

2026-10-22 按用户要求推进结构性改进：跨二进制桥接修复收口，依据同二进制负对照与 GOALS §5.6，执行[004 原源／折叠](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-004/plan.json)，24进程、5分钟、0构建。保持真实请求与计算，只改同行A源地址重用。改变增长时优先研究有需求量的共享源服务；增长保留时转向输入投递与计算的联合服务及回压。先以干预区分方向，再替换一组服务并回归；不追加桥接、主机控制或观察开销轮次。 004已完成：777113在romeo-a033运行13秒，24/24有效；两输出保留比分别为[0.912199733,1.016899078]与[1.324059076,1.484525922]，第一输出保持，第二输出超出两个冻结类别。按原判据信息不足，保留“源地址重用增加后第二输出行距增长反增”的新反例；模型／拟合0，独立C2/C3状态保持。

[源折叠组合回放](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-source-fold-replay-001/plan.json)：固定原服务，接入16前缀、正式调用源地址与缓存输出写入干预；4条件、68引擎调用已完成，CPU202.34秒，0拟合/GPU。源状态变化仍给出两L行距增量0；核对发现同行输出TMA聚合时间未关闭，正式输出服务2112条而程序应为16条，完整程序分支匹配未完成，见该轮README。

近期主交付改为一版可替换的供给服务整体：源请求、共享地址状态、订阅及目的完成由同一个事件生命周期连接。第一步把已测干预的实际源身份和输出请求数量放入程序输入，消除程序与服务各自改坐标的接法；随后最多保留“地址历史与同址订阅服务”“本地投递与计算联合推进”两个结构方向。源状态、在途读取与各CTA目的完成分开计数，其他模块固定。先用匹配TMA已有端点约束价格；筛查单CTA可隐藏、132CTA增长、同行global输出关闭仍增长、三遍计算可隐藏及004第二输出反增。通过必要约束才回归S4/S6、B80/B112及原12。各实现/回放轮执行前登记有限清单；不恢复旧固定τ逐副本服务，不以新价格名称继承原参数资格。

[程序干预接线](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-program-intervention-001/plan.json)：修复源折叠回放的2112／16输出请求错配，由程序提供实际源身份与global copy开关，保留本地STS和控制。四条件、68次完整引擎调用、30分钟CPU、最多两次定向实现修复，拟合/GPU0；固定价格回放后决定供给服务替代。 已完成68调用、CPU166.37秒；原源/折叠前16次各2112条输出，目标16条，202752输入请求的地址/身份一致，默认入口保持。两L行距增量仍预测0cycle，旧供给结构未解释实测；后续转已登记的整组服务替代。



[GPC逐副本服务必要下界](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-copy-work-bound-001/plan.json)：保留旧固定τ，仅检查工作守恒是否值得实现；真实工作列表给最忙GPC副本工作，按原F＋κW映射比较base原10%门。CPU2分钟、引擎／GPU／拟合0；必要下界失败即停止该固定价格形式，不建队列或调价格。 已完成：4608份16KiB×固定τ给56.713846µs，不计／计P0的plain下界59.470203／61.091820µs，均超原上限54.313600µs；该固定价格的工作守恒版本停止，见[重放与来源](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-copy-work-bound-001/README.md)。

[源片段与目的行流水](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-fragment-stream-001/plan.json)：替换整atom源PS及目的FIFO，按实际源片段与独立目的行逐行完成、归还订阅；先用八个已保存Q1条件约束五个共同局部参数，再筛Q4及间隔隐藏。固定两请求流，不扫描容量；一次拟合、最多2000局部调用、CPU15分钟、GPU0，其他模块和原12保持。 已完成760局部调用、一次拟合达到24nfev上限；15项约束中10项失败，停止本参数版本，Q4／GEMM未跑。

[源流与目的行解耦](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-fragment-decoupled-001/plan.json)：前版逐行阻塞的760次局部调用未过Q1，停止该参数版本。本轮只切断目的组装到下一源行的等待；源流在源片段到达时前进，片段缓冲仍持有至目的订阅归还。同八局部条件、一次24nfev拟合、最多2000调用、四个Q4验证输入、GPU0，不扩容量或价格。 已完成424调用、17nfev局部收敛；八窗全过但四项配对／预填失败，停止本参数版本，Q4／GEMM未跑。

[片段读取与组装双级流水](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-fragment-dual-stage-001/plan.json)：源／目的解耦版八完整窗全过，但A16/B80配对增长及预填未过。保留独立信用归还，将目的单FIFO改为两个按请求身份保持、只处理已到达行的共享速率流；同八局部条件、一次24nfev、最多2000调用，GPU0。这是本批第三种结构，不扩容量／价格；局部未过即不接GEMM。 已完成680调用、24nfev；八窗全过但三项配对／预填失败，停止本参数版本；另8调用从保存参数评分逐值一致，Q4／GEMM未跑。

[A16／A80同覆盖不变性](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-row-phase-invariance-001/plan.json)：同一A形状及逐行覆盖计数，只改实际片段比例与地址序列；六次CPU检查确认前三候选预测相等，冻结增长比R=1±0.05。复用775198二进制，单CTA Q1/G32、A0/A16/A80三个条件、各三进程，先三数值后最多三plain，GPU5分钟、零构建／拟合。开发结构判别，不重采间隔八条件或扩容量。 已完成：778301六进程全有效、4718592值零错；Rcycle/ns=0.889266/0.887850，冻结不变性预测被反证。下一服务比较保留每段实际源范围，区分片段比例与跨组/跨调用地址状态，先确定哪项产生新响应；不再把请求覆盖总数当闭合规则。CPU候选调用6、拟合0，原12及独立验收保持。

[源字节与目的片段流水](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-byteflow-gather-001/plan.json)：旧局部规则的A16/A80不变性已被778301反证。本轮把整行就绪改为实际32B目的片段依赖，源128B按四节拍到达、引用按目的包归还；两侧请求流固定2。复用旧八窗及同批A0/A16/A80，最多一次20nfev／2500调用／CPU20分钟，GPU0；初始仍相等则不拟合，局部未过不接GEMM，不加逐相位价格。 已完成四项语义检查、374调用及一次拟合；11窗全过，六项配对／预填／比例失败，A80/A16预测1.001298而实测0.889266。本参数版本停止，Q4／GEMM未跑。

[完整请求集合的次序判别](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-fullset-order-001/plan.json)：Q1/G64固定64个Ktile各一次，只改正式kt步长1／33，暖机恒1；A0/A16/A80六条件同binary各三次。冻结逐请求无次序状态的窗比1±5%；先三数值后最多三plain，一次零GPU构建10分钟、GPU5分钟、拟合0。主机仅放行33，kernel及同步不改；先核SASS与请求多重集合，再冻结执行。 已完成778440/778441，三seed六进程有效、18874368值零错，kernel与775198相等；三个窗比中位最大偏离0.08489%，A80/A16仍约0.889。752416已有A0/A16同集合结论，本轮该部分重复，新增A80扩展；不继续排列扫描。

[源行共享发出替换整atom服务](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-source-line-pool-001/plan.json)：移除旧GPC整请求core，用真实128B地址的两个全局发出槽及发出前订阅提供服务。原TMA条件价按行分摊，仅作工作假设；其他模块固定。直接回放G3/base与A16、16前缀及CTA0输出，最多34引擎调用／CPU15分钟，GPU／拟合0；两L未过>1000cycle必要门即停止固定版本，不扫描价格或容量。 已完成34调用、CPU120.883秒；两L增量201.124/229.027cycle，未过原>1000必要门；plain55.453/55.793µs，停止固定版本。

[缓存副本域独立发出](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-domain-line-pool-001/plan.json)：前版全局同址合并的两L响应仅201/229cycle。本轮按已有两域副本分别发出，key含接收域，每域一流、总槽数仍2；粒度、价格与其他模块保持。三项语义检查后直接G3/base与A16，34调用／CPU15分钟，GPU／拟合0；两L未过>1000必要门即停止固定版本。 已完成三项语义检查、34调用、CPU142.951秒；两L增量21.985/302.653cycle，plain87.912/88.341µs，固定版本停止。新发出源行工作仅增0.5214%，后续区分共享源获取与逐请求地址处理／重组；不继续给这两个源行版本调容量或价格。

[有限同址sector合并替换整atom服务](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-bounded-sector-merge-001/plan.json)：公开H100研究给出同GPC最多4份读合并；本轮将其作为GH200 tensor工作假设，源条目按实际32B覆盖、最多4份订阅并保留至应答。替换旧core，缓存填充与逐CTA目的仍分开；每GPC两流及原τ/512不变。两G3条件、34调用／CPU20分钟，GPU／拟合0；先核第五份独立工作、跨GPC隔离和返回后重收费，两L未过>1000cycle必要门即停止固定版本。 已完成三项语义检查、34调用、CPU781.065秒；plain46.687/46.731µs，两L增量0，停止本固定版本。

[显式分区与组相联源缓存](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-banked-source-cache-001/plan.json)：有限合并仍未形成增长；本轮按公开H200的同容量结构，将每域完全相联缓存改为48分区×256组×20路，IPOLY+MODULO分区和去分区后的XOR组索引。保留home/副本、替换更新时间及其余服务；两G3条件34调用／CPU30分钟，GPU／拟合0，不扫描映射或价格。映射/容量/组隔离通过后，仍以两L均>1000cycle作为后续联合约束的必要门。 已完成三项检查、34调用、CPU808.098秒；plain46.686950/46.728959µs，两L增长0，停止本固定版本。两版保存时间线的464后续组均提前就绪；A16最小余量494.127/376.308ns。下一项须区分逐请求处理／共享应答与返回投递，先给出能改变这些后续依赖的预测，不继续微调这两版。

[请求返回与查询位置对齐](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-query-return-connection-001/plan.json)：复用g43查询时钟夹持与原五个合格输出，按八暖机、全CTA输出重放G3基础／A16。冻结上一版全部服务，只对照查询推进、B1发布后年龄与full状态；已完成2条件、18调用、120.977秒、拟合／GPU0：基础两种B1参考年龄数值接近而full状态相反，issue与后戳的边界核对见下方新轮次，A16查询推进已偏晚；仅后移query的修复停止。查询节点已单列，既有时间精确保持，见[结果](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-query-return-connection-001/README.md)。

[g43分请求完成集合](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-operand-completion-001/plan.json)：查询年龄对齐后，下一项只分辨A／B0／B1在full未完成时的状态。每输出仅g43使用独立事务barrier，保留empty和计算／释放；同binary统一／分离、原F参考，两布局各三进程，最多18进程、一次零GPU构建10分钟／GPU5分钟、两次定向修复。冻结主导集合及原5%实际资格后完成780500的18进程，17有效；仅一个trial／输出共同合格，信息不足收口，不重开桥接扫描，见[结果](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-operand-completion-001/README.md)。

[请求返回同地址比较](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-address-matched-return-001/plan.json)：旧形成轮使用trace0基址预测trace1混合中位数，现有地址分区映射随该偏移改变。固定冻结参数，按被评分trace1的实际A/B地址逐组回放，复用三份相同输入预测，只新增四组；旧轮未保存的引擎依赖先做一项同输入核对，执行前将上限登记为55调用／CPU15分钟、拟合与GPU0。已完成55调用、108.582秒，同输入完整时间线逐值一致；校正后unique Q1／Q4仍超原门槛，参数版本继续停止，见[同条件比较](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-address-matched-return-001/README.md)。

[cfg_b行距与full等待方式](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-stride-wait-policy-001/plan.json)：复用753702的S6同binary，固定实际请求／计算／释放、A分配大小和blocking暖机；仅改变base／A16实际行距与目标blocking／polling。四程序条件各plain／dual、三进程，共24；0构建／拟合、一次免费ROMEO GPU分配最多5分钟。先冻结增量比1与Q04迁移比例3366/5411的两条不重叠预测带，实际资格只剔除受影响配对；至少两共同trial且blocking两L增长均大于1000cycle后判别。780627完成24进程、22有效、1GPU分配15秒，两输出各有两个共同合格trial：Q04比例迁移均被反证，policy无关形式第二输出获开发支持、第一输出信息不足，见[结果](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-stride-wait-policy-001/README.md)。本轮收口，不扩S4或等待hint。

[full完成边界筛查](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-full-boundary-screen-001/plan.json)：只读基础/A16六进程及五个合格输出，固定原模型与价格；核对B1后戳和逻辑issue的基准，以FIRST相对边界逐CTA评分。CPU60秒、引擎／拟合／GPU0；跨试次差异按实际已记录的输入判定，不重采样。已完成1320条原始核对：FIRST参考上界违例705条、下界69条；逻辑issue与后戳差未测，跨试次实际地址和短窗口ns缺失，不据此选择仲裁。见[结果](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-full-boundary-screen-001/README.md)。

[形成服务的返回反馈](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-formation-return-001/plan.json)：仅将当前Formation PS2的开始／结束接到producer后继，固定全部价格与控制。先核8次小图、11次缺省同输入，再各11次回放unique-Q1 r00；最多41调用／CPU15分钟、拟合／GPU0。原三端点10%门及等待／控制分解决定是否继续，失败不扩Q4或GEMM。

本轮[已收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-formation-return-001/README.md)：41调用、79.6秒、拟合／GPU0；基线／start／end最大误差33.266%／33.590%／31.312%，24分项全部低估，两形式停止。基线控制已吻合（0.997），等待和仅为实测0.617；返回反馈只把停顿从等待移到控制。Q1每组往返实测约675–750ns、预测约571–588ns，缺口在低在途往返本身；下一结构区分固定往返与可重叠服务量，须同时保持unique／global1 Q4窗口，不再调整反馈位置或单一读取价。

## 2. 三类模型与程序图的接入依赖

cfg_b 组合原第 1、2 步的 CPU 交付已完成，验收与入口见[组合 §4](model/mechanism/composition.md#4-输出)。请求级候选已接通共同程序的请求、计算完成和槽释放端点，以及阶段频率、经验前缀和 plain 映射；原12条件的结果与失败反例已保留。§1 的历史前缀轮次已收口；替换供给规则时仍检查[联合约束](model/mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md#下一候选的联合约束入口)，该整组回归不作为阶段 2–4 转段的门槛。

| 工作 | 推进顺序 | 接口或交付入口 |
|---|---|---|
| 规格基线 | 独立计算入口已完成；比较时记录产品/参考频率口径 | [规格基线计算](model/spec/spec_baseline.md)；C1-01继续使用原冻结口径 |
| 半经验模型 | 保留V09冻结版本与C1-01已有结果，先对照当前组合 | C1-01已经分别列出纯规格、wave_roofline、structural与TileSight，完整时间归C3 |
| 程序图 | cfg_b S6/S4 已可导出并核对历史工作映射；请求级完成事件已接通候选，cfg_a S4/S6和cfg_c S4的cluster整图与指定历史端点核对已完成；cfg_c1也已完成程序核对；其余配置SASS对照与TileLang前端暂缓 | [CUTLASS程序图](model/mechanism/program/cutlass_program_graph.md)、[操作与事件接口](model/common/interfaces.md) |
| 硬件服务候选 | 按九个方向逐项交付条件服务或显式工作假设及接入端点，方向与首版交付见[模块地图 §6](model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)；已有规则和反例保留 | 各阶段实验与结论见[按问题索引](experiments/gh200_sm90/README.md#research-questions) |
| 机制组合 | 所需局部规则和程序图可用即做必要接线；阶段 5 汇成同版 cfg_b 模型并检查原12条件响应和完整时间（首版对应原第 10 阶段） | 完整输入→时间线→plain 与原阈值报告齐全才作该阶段交付；C2/C3 未通过项照实保留，旧版预测继续对照 |

频率沿用已有版本与声明范围，供给规则由请求级组合检验；整组残差决定是否需要改变两者的连接。

## 3. 后续任务的进入条件

以下是本次五阶段推进与九方向覆盖之外的候选方向，当前不执行；本次队列内按授权依次推进，队列外的新阶段由用户选择，按 [GOALS 当前阶段限制](GOALS.md#current-scope)处理。

| 候选方向 | 何时重新选题 |
|---|---|
| 多播、DSM、WGMMA其他路径 | 当前C1/C3残差需要该规则，或形成具有新增区分价值且可执行的独立C2课题 |
| FlashAttention、MoE | GEMM阶段已有可复用接口，且明确新算子需要增加或检验的机制 |
| 算子图、端到端与其他硬件 | 单kernel组合有可用范围，目标工作负载或新设备已明确 |

以上为候选方向，不是自动展开的任务队列。具体预算与终止条件放在各任务登记中。

## 4. 协作与资料入口

- TileLang方向合作者提供降级IR到程序图的前端；本仓库维护服务候选、参数证据与组合验证。首轮人工核对图不等待自动前端。
- 历史复用入口：[TMA盘点](model/mechanism/hardware_rules/TMA-HISTORY-REUSE.md)、[首版输入模型](model/mechanism/hardware_rules/TMA-INPUT-EVENT-MODEL.md)、[模型候选与反例](model/semi_empirical/MODEL-CANDIDATES.md)。
- 用户有两次向 NVIDIA 微架构研究人员请教的机会；保留为候选分歧明确后的咨询资源，不作为当前工作的前置条件。

[请求应答网络整组替代](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-request-reply-network-001/plan.json)：源cache就绪后进入GPC共享PS2，再独立CTA投递；移除旧core/max并共同估计基本／跨行工作价格。G3基础/A16两条件拟合，至多六验证条件；一次拟合10nfev、1292引擎调用、CPU45分钟、GPU0，其他模块固定。

本轮[已收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-request-reply-network-001/README.md)：新结构产生持续增长，第一L周期误差10.801%未过门，停止本参数版本，后续六条件未求解。下一结构让地址状态参与共享处理／投递的到达和仲裁；频率换算失配随端点评分单列，原12及独立验收状态保持。

[地址路由的共享读取](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-address-routed-input-001/plan.json)：固定IPOLY+MODULO与每域48分区；实际片段进入独立读取FIFO，再接GPC应答／CTA投递，删除统一跨行价格。两共同服务价一次有限拟合6nfev，上限952引擎调用／CPU50分钟／8条件，GPU0；原10%端点门槛及独立验收保持。

本轮[已收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-address-routed-input-001/README.md)：两开发plain误差−3.410%／−3.432%，第一L周期误差11.160%未过门；联合标定未收敛，六验证条件未求解。纳秒端点和频率换算的误差已分解，下一项先以现有独立服务证据核对读取／应答价格及计时单位；不追加本轮价格搜索，原12及独立验收保持。

[独立TMA服务工作量约束](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-independent-service-bounds-001/plan.json)：先用已保存8个trace0条件×5进程的完整globaltimer窗口，按真实源片段和逐接收者应答累计资源工作，检查固定地址读取候选的必要价格上界；合并只列理想工作下界。CPU20分钟、模型／拟合／新增条件／GPU0；本轮不重拟合G3。

[在途源读取共享、应答独立](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-read-sharing-network-001/plan.json)：上一逐接收者读取参数超出独立global1-Q4完成窗口。新结构只合并同域尚未完成的同址128B读取，保留每接收者应答／投递；同一服务先运行两纯tensor工况，再跑两G3，过原端点门后至多四干预。旧价格固定，172引擎调用／CPU24分钟／8条件／0拟合／0GPU，最多两次定向实现修复。

本轮收口：独立TMA的40个完成窗口排除了原逐接收者读取参数与global1-Q4的组合；在途同址读取共享版完成56调用，同一服务的两纯tensor布局均预测25770.944ns，实测trace1释放为29968／26080ns；G3两L增长0，固定参数点停止。下一项先用纯tensor共享／独立源的响应约束共同服务可行域，再以同价检验G3；不追加本点的G3价格扫描。原12、默认版及独立验收保持。

[纯TMA约束读取价并检验GEMM迁移](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-source-only-calibration-001/plan.json)：复用在途共享读取结构，不新增候选；只由unique-Q4释放／早／晚窗口约束一项读取价，冻结应答8cycle及其余模块，读取上界由独立完整窗口给出。一次6nfev、300引擎调用、CPU20分钟、最多8条件、GPU0。随后复核三项纯tensor并直接跑两G3；局部失败也检查目标迁移，原门槛及独立验收保持，不恢复G3两价拟合。

[公开分区折叠转写修复](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-public-fold-correction-001/plan.json)：地址12288B在旧函数为15，引用源码折叠后恢复原bank位应为14。只修该映射，冻结纯TMA得到的读取价和其余模块；两纯tensor及两G3先做同条件比较，过原G3门后至多四干预。172调用／CPU20分钟／8条件／0拟合／0GPU；历史快照不改。

[有限读取订阅](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-bounded-read-fanout-001/plan.json)：无限合并版纯TMA标定／公开映射修复后，G3仍全ready。新结构只把现有128B在途读取限制为含首次最多4接收者，第5个另付读取工作；固定纯TMA价格与其余模块。两纯tensor／两G3先回放，过原门后检查all132输出的单遍／三遍隐藏配对。最多8条件、172调用、CPU20分钟、拟合／GPU0；不扫合并容量。

本批三轮已收口：纯TMA一价标定155调用，unique-Q4总窗口接近但分项13.086%、Q1迁移30.432%未过，G3仍零增长；公开折叠修复56调用、132CTA两L增长均为零；固定4订阅读取56调用，A16仅1/126后续full阻塞、两L仍零增长。预定三遍配对未求解，执行范围与原验收保持。下一项转向本页已列的本地投递／计算联合结构，复用746720的12.10%／20.50%本地周期干扰和原生D共同服务下界；先核对控制发出与目的服务能否用同一规则区分，不再扩读取价或合并容量。

[本地输入／计算共同服务](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-local-joint-service-001/plan.json)：替换独立CTA投递，以128 B/cycle共同服务处理每Ktile的32 KiB TMA写入和相邻B复用后的32 KiB计算读；计算按四个8 KiB操作数对与原512cycle节拍推进，双方先到先预留可用区间，争用延迟传至计算完成／释放。源状态、读取／应答、频率及输出固定，旧投递价由新服务替换。一个结构、零拟合；先核资源守恒和双向争用，直接跑G3基础／A16，与保存的独立版比较。四L误差≤10%且两增长>1000cycle后，才做all132输出单遍／三遍配对；最多6条件、102引擎调用、CPU20分钟、GPU0。首门失败则停止该固定结构，不扫服务率或节拍。

本轮[已收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-local-joint-service-001/README.md)：34调用、169.021秒、GPU／拟合0；两L增长516／1896cycle，A16端点仍偏31.866%／18.020%，后续full仅1/126阻塞。双向连接已可执行，固定共同SMEM版停止，不调服务率／节拍。下一结构须使输入本身持续未完成，而非只延长计算。

[读取受理后查询缓存](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-port-ordered-cache-001/plan.json)：现版在程序发出请求时改变全部源行状态，读取排队发生在数据就绪之后。新结构让128B行先取得对应读取分区，再在服务时刻查询缓存；合并条目保留至该次读取的数据返回，最多4份订阅，逐接收者应答与投递保持。沿既有lookup时更新状态，不恢复填充完成插入／MRU。冻结全部价格及其余模块，一个结构、零拟合，先G3两条件；过原四L／增长门后至多六源折叠与三遍配对。最多8条件、136引擎调用、CPU30分钟、GPU0；不扫描价格、共享域或条目数。

本轮[已收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-port-ordered-cache-001/README.md)：34次完成调用，含中断的总上界51，续跑341.430秒；两L增长0，A16仍仅1/126后续full阻塞。固定结构停止，后续六项未跑。下一项只读核实现有NCU与计数权限，再决定最小的真实cfg_b输入路径定位，不追加同版参数搜索。

[真实输入路径计数定位](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-input-counter-localization-001/plan.json)：复用004的plain二进制、16次原源全输出暖机及正式单CTA输出，普通／A16同分配、S6、132CTA不变。一次免费ROMEO GPU分配≤5分钟，零构建，最多三trial的6次plain＋6次NCU；首个硬件计数明确拒绝即停止。每trial plain行距比≥1.15且原数值／暖机合格才用对应计数定位；NCU只取第17个匹配GEMM，application replay、cache-control none、clock-control none。预列读取sector响应1.30–1.40与0.95–1.20两带，区间落入者用于选择工作量或联合推进下一规则；中间区间保留未区分。开发定位，模型／拟合0，原12和独立验收保持。

本轮[已收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-input-counter-localization-001/README.md)：780060及一次实现修复重试780099共分配18秒、六次进程尝试，四plain合格。NCU先因无效`--units`退出，删除后因`GemmUniversal`未匹配真实`cutlass::device_kernel<Kernel>`而未采集kernel，信息不足；权限尚未检验。原三trial门槛保留，不再追加本轮执行修复。下一次计数需按返回的真实符号选择第17次调用；当前先停止无新观测依据的读取价格、容量及局部SMEM节拍变体，已有固定版本保持停止。

[逐CTA等值源副本](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-private-source-001/plan.json)：零GPU查询返回管理员计数策略，先用请求源身份干预。原dyadic周期17不支持全源等值折叠，改为每CTA逐字节复制原A/B，仅用同binary运行时开关切换描述符基址；布局、请求数、计算值和槽序保持。每臂16次本臂源的全输出暖机，正式仅CTA0输出；base/A16×原源/独立源、plain/dual各三进程，共24。冻结地址盲1±5%与源状态压力≥20%的两类Lcycle响应，行距保持比分项报告；原源两L增长须>1000cycle、至少两共同trial、原CV及5%观察门保留。不以静态寄存器变化作废观察。一次零GPU构建10分钟、一次GPU5分钟、两次定向实现修复，模型／拟合0；不扫描副本数，结果仅约束此真实G3干预。

首次[执行收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-private-source-001/README.md)：780223／780254完成构建及一次描述符路径修复，plain／dual均REG168、无spill；780264分配11秒，24进程均在`private_a_maps`符号绑定处退出，timed GEMM 0。ELF符号存在，日志20091-D指向`const void*`重载误选；两处显式模板host修正已准备，尚未重建或运行。两次执行修复及进程额度用完，主张未检验，不追加本轮测量。几何覆盖34→396MiB、逻辑输入528MiB保持；恢复仍须先取得预列主循环响应，原12和独立验收保持。

[同一实验执行恢复](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-private-source-001/execution-recovery.json)按§5.6的实现故障规则执行；原24失败保留，新增一次零GPU构建10分钟、一次GPU5分钟，最多24进程，累计尝试上限48、GPU分配时长上界311秒。仅验证两处host绑定修正，不改设备操作或冻结预测。首个正式original/base/plain兼作实现检查，共有初始化或数值错误即停止，不追加修复。模型／拟合0，不新增机制候选。

[私有源程序接回同一服务](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-private-source-transfer-001/plan.json)：780343四个L的独立源／原源周期比均超过冻结1.20，24进程有效。先修复程序对private_source的忽略，接实际每CTA源偏移及本臂16次暖机；冻结全部价格和其他模块，只运行原源／私有源×base/A16四条件。逻辑分配扩大，并按保持现有home／48分区映射的高位周期给B／D平移以隔离，实际GPU VA未导出，映射作为工作假设。最多68引擎调用／CPU30分钟／GPU与拟合0；四L、四plain及共享响应分项比较，不加专属误差项。

[源片段整理与计算共同服务](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-gather-joint-service-001/plan.json)：原共同服务只计逻辑输入32KiB，A16后续full仍几乎全ready；新形式按真实源128B片段引用给逐CTA整理／投递收费。基础256片段／Ktile，A16为368；计算保持256片段，128 B/cycle及所有其他模块固定。源片段占用本地服务为工作假设，不拟合。仅G3两条件34调用／CPU15分钟／GPU0，以原四L≤10%、两增长>1000cycle筛查；失败停止，不扫服务率。

该固定片段版本[已收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-gather-joint-service-001/README.md)：含恢复51调用、GPU／拟合0；两L增长7551／5972cycle，第一L仍低17.828%，A16后续full仍仅1/126未完成。停止本参数点，不扫局部服务率；私有源同服务四条件结果见下段，分别检查源压力与阶段时间分配。

私有源[同服务回放已完成](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-private-source-transfer-001/README.md)：四条件68调用、1629.615秒、拟合／GPU0。源副本大响应可以预测，两plain误差−0.765%／−0.033%，但阶段端点与原源持续行距响应仍失败；私有源第二L纳秒端点也超全CTA预测包络。下一版输入完成服务须同时接受这两个工作集工况及原full状态约束；不以带宽价格或频率修正替代端点响应。

[请求形成先于缓存查询](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-formation-before-cache-001/plan.json)：替换旧后置GPC应答，每接收者源tag引用先经共享形成服务，再查询缓存／读取分区。固定其余模块，三纯TMA与两G3先检验结构响应；产生两L持续增长后，只用纯TMA共同约束两服务价，再冻结检验G3及必要的两私有源。至多7条件、550引擎调用、一次4nfev、CPU1小时、GPU0；初值不产生增长或冻结参数未过原10%端点门时收口。

请求形成前置轮[已收口](../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-formation-before-cache-001/README.md)：432调用、2119.952秒、一次4nfev标定、GPU0。原源两L增长14171／14455cycle，四L最大误差34.422%→9.338%；plain最大误差仍14.428%，基础CTA0后续full未完成124/126，纯TMA Q1最大分项误差32.224%，私有源阶段端点仍失败。停止当前参数，不扩原12或追加同点拟合；下一轮以请求返回、full就绪和查询位置的联合约束选择规则，固定部分及有限清单另行登记。
