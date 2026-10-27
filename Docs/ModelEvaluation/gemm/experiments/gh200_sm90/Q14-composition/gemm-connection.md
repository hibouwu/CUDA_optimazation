# 局部规则接回完整 GEMM

38 个目录（10-18 至 10-22）：把局部计算、同步、输出规则与程序干预接回 G3 S6/S4 的请求级组合，并准备真实 GEMM 的同 binary 干预。

## 结论

**局部规则迁移**
- 计算完成 d95：S6/S4 plain 为 47.491／49.342 µs，stage 响应 3.899%（实测 10.467%），基本没有变化。
- 局部联合规则按阶段频率迁移：d95／消费者继续133.364／empty完成275，替换旧reuse220，S4 plain误差9.639%→7.065%，stage响应3.903%→6.738%；S6只保留d95的变化，A16与CTA分散仍失败。通过三条件开发筛查；其余条件及G6回归均已完成，见下条。
- 联合规则其余八条件回归完成，新增88调用，复用三项。11项中位／最大误差10.346%／25.334%，同11项旧版10.810%／25.422%；5/11在10%内，关键CTA命中0/110，random后续L略恶化。正式入口可从保存profile启用，G1新增11调用逐值一致、80测试通过。G6另列11调用已完成，同12条件中位8.706%／最大25.334%、关键CTA1/120；G6为组级事件，默认版本保持。
- 保存事件的行距配对：源就绪有响应，新增等待只在CTA首组；各输出主循环长度保持，而实测A16／B80／B112的first_L／later_L增长2744–5920ns。下一项区分首组之后的输入就绪与full返回，未选新价格，见[逐事件分析](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261020-Q14-local-bundle-regression-001/README.md#行距残差定位)。
- 逐轮准备P240／309.75cycle替换旧软件gap后，G3／A16行距响应仍仅0.0931%，实测28.9371%；first_L／later_L增长均为0，实测5008／5920ns。两参数版本停止，不扩展stage／原12，见[有限迁移](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-producer-preparation-transfer-001/README.md)。
- 真实G3/S6的静态单遍／三遍对照通过同批R桥接；36进程支持计算间隔可隐藏A16行距增量，排除固定不可隐藏加项。尚未区分源服务、等待返回与反压位置，组合成绩保持，见[冻结预测与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-static-repeat-dispatch-001/README.md)。
- 按 blocked 入口逐次收 275 cycle：S4 plain 误差由 9.6% 降到 3.3%，但 S4 的 first_L/later_L 过冲，stage 响应变为 18.3%，已停止。

**真实 GEMM 的 wait0 干预**
- 冻结预测：d0 规则下 S6 不变；d95 规则下 S6 增加 15.6%。
- 752070 只完成 21/24 个进程：S6 实测 +21.6%，S4 +15.5%。因一个进程的暖机 CV 为 2.096%（门槛 2%）记为信息不足，但 d0 的“S6 不变”已明显不符。
- 同 binary 的 full 等待作业 753707 在排队时取消，原因是 10-18 调整为模块优先。
- 10-20复用合格构建，只测六plain条件：763812完成11个合格进程，第12个暖机CV2.050824%超原2%门槛后停止，余6未测；桥接／policy三重复不齐，判信息不足，零响应未作判定。

**输出与背景**
- 首输出延后：模型的早段、后段响应为 0，旧实测为 −544／−128 ns。
- 输出源读取后的共享占用只复现了部分方向；A16 误差保持 25.422%。

**程序干预**
- 活动 4/6 槽 × 1/3 遍计算四个条件，误差 24.7%、24.8%、9.8%、10.5%。
- 重复计算改造程序中约 1.25 万 cycle 只属于该改造（program-budget-scope），不作为原 cfg_b 的价格。
- 固定物理 S6 时已有 stage 响应：第一、二段分别 +2647.5、+5635 cycle，物理布局变化不是必要条件。
- 计算量开关的原binary在单遍下，A16的两段周期增量仅105–133／587–687.5cycle，低于冻结1000cycle辨识下限；774436十二进程资格通过后按计划停止，三遍未测。此改造程序没有保留原cfg_b的持续行距反例，见[计算间隔对照](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-stride-compute-discrimination-001/README.md)。

**判断**
- 默认原12条件成绩保持；本轮联合规则改善了S4的开发预测，原12条件同版开发回归已完成，独立验收未通过。共享源、full完成／返回及CTA推进仍有未解连接，见 [g3-stage-credit](../Q04-sync-slot/g3-stage-credit.md) 与 [source-organization](../Q02-tma-request/source-organization.md)。

## 轮次

按日期排列；全部结果目录见[Q14-composition/gemm-connection](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 同一S6的活动槽／计算量能否由同版组合回放 | 746824已有四条件；修活动4／6槽与1／3遍程序事实，物理资源、服务价及运行前频率固定 | 默认行为保持，三遍stage差消失方向可回放；单遍第一L差50.2对624ns、plain仍失败，保留改造程序的主循环残差 | [改前反例、程序接入及四条件](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-runtime-intervention-001/EXECUTION.md) |
| 改造程序的共同预算是否也存在于原始cfg_b | 八既有条件48份dual，从同SM原始周期核对原S4/S6、活动槽与重复循环程序，核对两份plain SASS | 原S6剩376／584cycle，重复体剩约1.25万；停止向普通cfg_b迁移，下一项回原stage等待连接 | [原始记录、SASS与判定](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-program-budget-scope-001/EXECUTION.md) |
| 原始cfg_b的stage响应是否依赖full等待方式 | S6/S4×原阻塞／test_wait轮询，四开发条件；不加活动槽、重复循环或内部打点，固定其余路径 | 753686/753688 CPU构建完成；阻塞机器字精确，轮询另改输出算术／排程，停止分别编译形式，GPU0 | [源码、构建及控制反例](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-full-wait-policy-001/EXECUTION.md) |
| 同一binary的full等待能否解释原stage响应 | S6/S4×原始／runtime阻塞／runtime轮询，六开发条件各plain/dual三进程；阻塞暖机保持 | 753702 CPU构建通过，frame/实际指令及同binary控制已核；753707在排队阶段取消、GPU运行时间0，905输入绑定保留；按新顺序延后至局部等待／释放连接检验后再决定 | [有限清单与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-runtime-full-wait-001/EXECUTION.md) |
| 真实GEMM正常wait0迁移预测（CPU） | G3 S6／S4，正常wait1改wait0；d0／d95原服务及其他模块固定，首组与tail保留 | 44调用完成；S6的d0／d95预测增量0／7.412µs可区分，S4两形式均会变慢；实机判别见下一轮 | [私有预测包](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-gemm-wait-prediction-001/README.md) |
| 真实GEMM正常wait0的四条件判别（信息不足） | G3 S6／S4×wait1／wait0，fixed10暖机、同seed17；plain／原dual各三进程，正常wait之外的kernel与资源保持 | 752070构建通过；20/24进程合格，p20暖机CV2.095854%超门槛后停止，不完整批不作候选判定 | [冻结与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-gemm-wait-001/EXECUTION.md) |
| d95接入能否修复G3 stage（CPU迁移） | 原默认S6回归11调用、显式d95的S6/S4各11调用；其余模块与原频率固定 | 默认JSON精确；普通释放、CTA跨度保持，stage响应3.898566%对实测10.466624%；停止扩大并保留默认 | [同条件比较与版本](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-compute-completion-transfer-001/EXECUTION.md) |
| 真实GEMM首输出延后能否由现用服务表达（CPU开发回放） | S4/S6×flag1/2，各8次flag0暖机；只接E0等待下一WG mma返回，固定服务与价格 | 默认完整报告精确，四目标早段差0对旧配对−544／−128ns；保留背景交互缺口 | [接线、四目标与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-epilogue-defer-001/EXECUTION.md) |
| 输出源读取后的共享占用是否产生背景响应（CPU开发） | 原两槽/τ，8KiB异步工作在source_read_done后占0.5单位；四旧条件、其它模块固定 | S4 early/post−198.369185/+2.582738ns、S6均0；必要门过，中央幅度未解释，保留假设 | [固定规则、四条件及复现](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-shared-output-service-001/EXECUTION.md) |
| 输出共享假设能否修复原A16最大误差（CPU回归） | 原A16输入/11调用，固定上一轮规则和全部价格；超10%即停止余项 | 源就绪及46个CTA MAIN改变，关键CTA/plain保持，25.422%仍失败；剩余两条件停止 | [完整报告、端点比较与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-output-write-gemm-001/EXECUTION.md) |
| 固定物理S6是否仍有活动槽响应（CPU） | 746729活动4/6的10份保存raw，固定源、物理资源及binary | 2640窗复现原值，4−6槽两段L增2647.5／5635cycle；暂缓新增物理布局对照，模型／GPU0 | [复核与研究决定](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261019-Q14-stage-scope-screen-001/README.md) |
| wait0返回与完整结束的分工（CPU） | 保留原四臂、请求与释放；Record／既有SMEM暂存的循环外端点 | 两构建关闭宏逐位保持；启用版本消费者循环改变，两修复后停止、活动源码恢复，GPU0 | [有限计划、SASS与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261019-Q14-wait0-boundary-001/README.md) |
| 同binary能否观察最后wait0 | runtime开关只控制循环外单时戳，两模式共用K循环；保持原begin/end | 757920构建与两地复核通过；757982两上下文／八臂数值及完整窗口桥接通过，跨度资格失败，余四上下文停止，两位置未判定。两修复用完；保留联合端点，节点5秒／预约20GPU秒 | [构建与控制路径](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261019-Q14-runtime-tail-001/README.md)、[有限观察验证](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261019-Q14-runtime-tail-001/measurement/README.md) |
| 独立计算模块能否延续跨输出服务状态（CPU） | 固定完整引擎和全部价格，核源码交接顺序；S6／S4、K2和晚输入四组织各三输出 | 原入口在提前交接时丢失进度；增加显式初始进度后762端点相同，九旧独立例及四完整引擎报告保持，90调用、GPU0 | [反例、接口与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261019-Q14-compute-handoff-contract-001/EXECUTION.md) |
| 同binary full等待是否改变真实cfg_b的plain stage响应 | 复用753702构建，S6/S4×原始／runtime阻塞／轮询，六plain条件各三进程；十次阻塞暖机、无内部打点 | 763812共12次尝试，11合格，p23暖机CV2.050824%超门槛后停止，余6未测；桥接／响应三重复不齐，信息不足，无重试／改价 | [计划、原始输出与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261020-Q14-full-wait-plain-002/README.md) |

| 局部联合预测类能否迁移cfg_b（CPU） | 固定既有价格；d95／继续133.364／empty275替换旧reuse220，按阶段频率换算；最多S6／S4／A16三条件33完整调用 | 26小调用／33完整调用完成，50688组接线符合价格；S4误差9.639%→7.065%、stage响应6.738%，保留开发迁移假设，A16／CTA分散仍失败 | [计划与接法](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261020-Q14-local-bundle-transfer-001/README.md) |

| 局部联合规则能否覆盖其余原条件（CPU回归） | 固定上一轮规则及原输入，八条件各11调用，复用三条件；G6待单独回归 | 八条件88调用完成，11项中位10.346%／最大25.334%，关键CTA0/110；正式入口新增11调用逐值一致，80测试通过；G6另列完成后同12项中位8.706%／最大25.334%，独立C2/C3未完成 | [计划与产物](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261020-Q14-local-bundle-regression-001/README.md) |

| 联合规则的G6长K回归（CPU） | 同一规则、原G6输入／11调用协议；CPU上限70分钟，保存组级事件 | 11调用完成、2637.795秒，plain5320.468µs、误差+1.912811%；同12项中位8.706%／最大25.334%，关键CTA1/120，完整请求完成图仍缺 | [计划与进度](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261020-Q14-local-bundle-G6-001/README.md) |
| 最新联合规则的关键路径能否支持行距／stage归因（保存轨迹分析） | 固定全部输入与价格，读取G3 S6／S4／A16三条保存时间线，只分析各自最晚CTA；完整模型／拟合／GPU均0 | 节点重建通过1e-6ns门槛；同组等待进入差≤7.28e-12ns。修复无关producer尾部被标零时差；S4原plain响应6.738%保持，后续full／empty观察缺口保持 | [计划、反例与当前分解](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-critical-path-current-001/README.md) |
| 逐轮程序准备能否改善持续行距响应（CPU迁移） | 用P240／309.75cycle替换旧producer／operand gap；G3/A16各11调用，源／计算／输出／频率固定 | 正式44调用完成，另11调用为缺依赖的实现失败；行距响应仍约0.0931%，L增长均0，每档关键CTA0/20，两参数版本停止 | [计划与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-producer-preparation-transfer-001/README.md) |
| 增加计算间隔能否隐藏A16行距代价 | 原746824 binary、活动S6及G3，基础／A16×一遍／三遍；plain／dual各三进程 | 774436单遍十二进程资格通过，周期增量低于1000cycle辨识下限，三遍未测、两假设未判定；恢复单遍路径后再立项 | [计划与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-stride-compute-discrimination-001/README.md) |
| 去掉动态重复循环能否恢复单遍目标反例 | 原单遍gemm后按runtime开关直接加两遍；固定源/槽/提交/释放，同批原V09为R | 774459完成24进程，R重现反例，N1基础慢约22%且行距响应塌缩，桥接未过、N3未采 | [计划与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-direct-repeat-path-001/README.md) |
| 固定原S6槽推进后能否保留行距反例 | PipelineState及producer_tail恢复原文件，保留直接额外两遍开关，同批R／N1先桥接 | 774523构建成功，774550完成24合格进程；仍未过R桥接，N3未采，该实现停止 | [计划与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-static-stage-repeat-001/README.md) |
| 循环外选择静态计算量能否保持计算组组织并隐藏行距增量 | host选择单遍／三遍静态类型，Ktile内去掉runtime重复分支 | 774570资源解析漏冒号退出；修复1的774581构建成功。774591完成36合格进程，四静态实例组织及R／N1桥接通过；两L增量比落入冻结可隐藏带，固定不可隐藏加项被排除 | [计划与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-static-repeat-dispatch-001/README.md) |
| 在途源行订阅能否修复A16持续响应（CPU） | 保留原两atom前沿与价格，整请求合并改32B／128B实际源行订阅；其余模块固定 | 8合成核对通过，44引擎调用完成；工作量增加而两L增长仍0，两个单遍必要门失败，三遍未求解 | [计划、评分与定位](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-line-subscription-screen-001/README.md) |
| 逻辑操作数是否都可通过共同SMEM服务收费（CPU） | 固定原生D三个长度、A及R12；比较全部引用共同128B/cycle、相邻B复用共同服务与独立路径，不拟合 | D的K32窗口低于第一形式必要下界，即使沿原5%门槛也失败；其余两形式保留，模型引擎／GPU0 | [下界、原始窗口与复算](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-joint-smem-bound-001/README.md) |
| 后续full未完成能否解释持续行距响应 | 原R及同binary关闭／开启，S6 G3/A16八条件各三进程；原预测及5%门槛固定 | 实现1的24进程无观察语义，信息不足；修复2的774830完成24有效进程，桥接／扰动通过，A16未完成率91.823%对基础4.099%，当前全ready预测被反证 | [失败实现](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-full-entry-mask-001/README.md)、[修复与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-full-entry-mask-002/README.md) |
| g43的full未完成位于producer发布前还是发布后 | 固定S6 G3/A16与服务；原R、已合格F及同binary发布关／开，8条件24进程；只采g43前后发布数 | 774863完成18有效进程，时间／行距保持通过；N0基础一输出17/132超10%，未采N1，信息不足；稀疏g43形式停止 | [冻结计划、构建与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-001/README.md) |
| 保留逐Ktile full观察后定位g43发布位置 | R／F／同binary发布关开，固定G3/A16、S6，8条件24进程；不改模型价 | 775011完成24有效进程，F/N0六前置条件通过；全局原子N1的g43仅41/792未完成、六位置资格失败，形式停止 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-002/README.md) |
| 用CTA本地发布计数定位g43 | 将全局原子换成本地SMEM原子，查询／程序和原门槛保持；8条件24进程 | 775039的N0六进程misaligned；修复775049有23/24有效，三合格输出的306次未完成均已发布且领先三组，发布前全域主导类被反证；其余资格失败保留 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-003/README.md) |
| 2026-10-22 发布与查询时间线CPU核对 | 同一g43的发布进度、查询与full完成位置 | 18调用逐位复现旧预测，全部CTA的47/111进度吻合；A16 full提前639–881cycle仍与观测冲突 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-publication-time-alignment-001/README.md) |
| 2026-10-22 同组B1后至查询年龄 | 保留完成规则与较早查询，或发布后完成延迟 | 24有效进程；五合格输出530次未完成的年龄下界均超旧完成上界，保留完成／仅查询提前形式被排除 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-published-request-age-001/README.md) |
| 发布后年龄更随源坐标还是SM位置变化 | 复用775061合格输出，按输出做留一进程分组比较 | 1452项坐标吻合；A16的A组留一进程误差改善8%/18%，但与逻辑CTA组绑定；模型／拟合／GPU0 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-completion-spatial-screen-001/README.md) |
| Raster能否区分A源组与逻辑CTA组 | 264输出与每CTA两输出保持，冻结年龄形状投影与重复性区间 | 每输出66CTA，两带均不重叠，空间差RMS234/240cycle；模型／拟合／GPU0 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-raster-contrast-prediction-001/README.md) |
| Raster转置后的年龄形状跟随 | 同binary heuristic/along_m、年龄关/开，两布局各三进程 | 775150完成12有效进程，时间检查全过、最大扰动2.1304%；原raster六项空间形状控制全失败，按计划停止，along_m未测，形状迁移解释未区分 | [计划](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-raster-age-discrimination-001/README.md) |
| 每CTA副本的GPC共享投递能否产生持续行距响应（CPU） | 源合并、本地sink及其余模块固定；服务取六份短链的必要上界12.307692ns／16KiB，到达FIFO含本地阻塞；先G3/A16单遍 | 4合成例通过、22调用、53.9秒、GPU0；两L增量96.31／37.19cycle，plain102.25／100.52µs；固定上界FIFO版本失败，三遍未测，其他GPC投递机制未判定 | [计划、评分与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-gpc-delivery-bound-001/README.md) |
| 保留原工作列表后，持续A16增长是否依赖同行CTA | 固定G3/S6、grid132、CTA0两输出；新132/1均先16次全量前缀，仅目标调用放行132/1；先同批R桥接 | 775581有36/36有效、零失败；三trial同SM124、桥接通过。两L增量由12536–15401／11740–11943降至8–117／1206–1362cycle，两比值入冻结同行依赖带；支持同行工作影响大部分增长，源／前端／输出位置未分；模型／拟合0，开发定位 | [计划、原始回放与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-active-peer-001/README.md) |
| 同行输出TMA是否构成当前共享影响 | 保留132CTA输入／计算与CTA0程序，只屏蔽同行global TMA copy；先全写桥接 | 775646未生效gate修复后775647通过；775662完成17秒、35/36有效，一暖机CV失败不补。有效trial0／2两响应比[0.960119,1.052763]／[0.895419,1.118705]入冻结保留带；同行全局输出TMA不是大部分增长必要条件，物理服务未定价；开发定位、模型／拟合0 | [计划、原始回放与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-peer-output-001/README.md) |
| 等值A源身份折叠是否改变持续行距增长 | 132CTA输入／计算、仅CTA0写及16次原源全输出前缀保持，只折叠m17–21到m0–4，CTA0地址不变；先原源桥接 | 775812零GPU构建及实际SASS通过，REG168、无溢出、无修复；[001](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-001/README.md)的775816未启动即取消、测量0进程，775818回收已取消；002同型号132SM GH200迁移，UUID执行前绑定，三臂同批同卡，原区间、5%桥接与其余门槛保持；776745在romeo-a021完成14秒，24/24进程数值、暖机及观察合格；桥接仅trial0通过，未满足至少两项共同trial，Nfold未采，信息不足收口；[003邻接交换顺序](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-003/README.md)按原最多48进程／5分钟GPU、0构建执行，24进程桥接后收口；模型／拟合0，原12及独立C2/C3状态保持 | [冻结计划、源等值与重叠依据](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-002/README.md)  003的776879同卡24/24有效但0组过桥，折叠未采；[同binary负对照](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-bridge-null-001/README.md)777026在romeo-a056完成21秒，47/48进程有效；N1 A16 plain trial0暖机CV超2%保留，未补跑。R同二进制重复3/3、N完整配对2/2通过原门；旧、新二进制等价桥接仍未通过，同二进制对照可用。 |
| 同二进制等值A源折叠对两输出的影响 | 原源／折叠×base/A16，24进程；固定132CTA、真实请求数、计算值、B源、同步及CTA0地址 | 777113在romeo-a033完成13秒，24/24有效；第一输出响应比[0.912199733,1.016899078]保持，第二输出[1.324059076,1.484525922]超出两个冻结类别。整体信息不足，新增地址重用增加而第二输出行距增长反增的反例；物理归因未定，模型／拟合0，独立C2/C3状态保持 | [004计划、评分与回放](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-004/README.md) |
| 程序源折叠和输出copy开关是否准确进入组合 | 16原源全输出前缀后正式原源／折叠×base/A16，固定原服务 | 68引擎调用、166.37秒、GPU/拟合0；正式输出2112→16条，STS与控制保留，202752输入身份一致，默认回归保持；两L行距增量仍为0，原供给结构未解释实测 | [实现、四条件评分](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-program-intervention-001/README.md) |
| 消除FIFO空洞能否挽救原固定GPC副本价格 | 保留每副本单服务器、τ12.307692ns及原F／κ，读取真实工作列表，不建队列 | 最忙GPC4608份16KiB给服务下界56.713846µs，plain不计／计P0下界59.470203／61.091820µs，均超base原10%上限54.313600µs；同价格工作守恒版本停止，引擎／GPU／拟合0 | [公式、来源与重放](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-copy-work-bound-001/README.md) |

2026-10-22 整条供给替代直接回放真实G3：全局128B源行与两域副本发出两个固定版本共68调用、GPU/拟合0，均未过两L>1000cycle必要门。两域版源工作增0.5214%，大量重叠覆盖被共享；后续区分源获取与逐请求地址处理／重组。见[全局源行](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-source-line-pool-001/README.md)、[两域副本](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-domain-line-pool-001/README.md)。原12及独立C2/C3保持。

[有限同址sector合并](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-bounded-sector-merge-001/README.md)：每GPC最多4份在途订阅替换无限整atom合并，34调用、CPU781.065秒、GPU/拟合0；两L增长仍0，本固定版本停止。[显式分区与组相联缓存](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-banked-source-cache-001/README.md)也已完成三项检查、34调用、CPU808.098秒，两L增长0，固定版本停止；464个后续组仍提前就绪。价格和其他模块保持，未扩大原12或独立验收。

[本地输入／计算共同服务](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-local-joint-service-001/README.md)与[受理后查询缓存](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-port-ordered-cache-001/README.md)各完成34调用，固定版本均未解释A16持续full未完成，后续条件未求解。[真实输入路径计数定位](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-input-counter-localization-001/README.md)四plain进程合格；两次NCU因参数／kernel过滤错误未采集计数，信息不足，未改变两种工作量解释或独立验收结论。

[逐CTA等值源副本](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-private-source-001/README.md)：保持请求形状、逻辑输入528MiB及计算值，源覆盖34→396MiB。780264的24次初始化失败保留；780301修正host绑定后，780343有24/24有效，四个L副本／原源周期比2.3842–4.6979，支持采前大源工作集响应，行距增长类别未分。累计GPU25秒，模型／拟合0。[CPU接回](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-private-source-transfer-001/plan.json)只接实际地址及暖机，不调价格。

[源片段整理与计算共同服务](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-gather-joint-service-001/README.md)将每CTA输入工作由逻辑payload改为实际源128B片段引用；零拟合产生7551／5972cycle行距增长，第一L仍低17.828%，后续full仅1/126未完成。固定版本停止；含一次A16执行恢复共51调用，GPU0，原12和独立验收保持。

[同一输入服务的四条件回放](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-private-source-transfer-001/README.md)完成68调用：实际源副本与暖机接线准确，原源预测逐位保持，私有源plain误差−0.765%／−0.033%；两L周期／纳秒及原源行距仍失败，故不采用该固定服务。引擎调用68、拟合／GPU0；原12及独立C2/C3保持。

[请求形成先于缓存查询](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-formation-before-cache-001/README.md)已完成432调用、2119.952秒、GPU0：只用三纯TMA条件标定两服务价，再检验两G3和两私有源。原源两L增长14171／14455cycle，四L最大误差34.422%→9.338%；plain仍高14.428%，基础CTA0后续full未完成124/126；纯TMA Q1最大误差32.224%，私有源阶段端点仍失败。停止当前参数版本，默认与原12成绩保持。

- [请求返回与查询位置对齐](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-query-return-connection-001/README.md)：复用g43时钟区间完成18调用，查询单列为程序端点；[g43分请求完成集合](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-operand-completion-001/README.md)完成18进程、17有效，仅一个trial／输出共同合格，信息不足收口。

- [请求返回同地址比较](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-address-matched-return-001/README.md)：固定形成轮参数，更正预测与评分进程的地址绑定；55调用、108.582秒、拟合／GPU0，unique Q1／Q4仍失败，当前参数版本继续停止。

- [cfg_b行距与full等待方式](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-stride-wait-policy-001/README.md)：复用同binary的24进程，22有效；两输出各两共同trial，均排除Q04比例迁移，policy无关形式第二输出获开发支持、第一输出信息不足。780627分配1GPU、15秒，无构建／拟合或补跑。

- [full完成边界筛查](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-full-boundary-screen-001/README.md)：六份原始记录、1320条合格观察；FIRST参考上界705条、下界69条违例。逻辑issue与B1后戳分开，跨试次缺失证据保留；引擎／拟合／GPU0。

- [形成服务的返回反馈](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-formation-return-001/README.md)：producer后继接Formation PS2开始／结束两形式，unique Q1最大误差33.590%／31.312%（基线33.266%），24分项全部低估，均停止。控制已吻合、等待和仅0.57–0.62，缺口在低在途请求往返；41调用、拟合／GPU0。

## a044 复测（2026-10-23～26）

输入计数定位按原计划在 a044 补测（作业 785361）：修正 kernel 过滤，取第 17 次 GEMM，3 个 trial 的 base/A16 配对。plain 的 A16/base 为 1.339–1.389。L2 读 sector 的 A16/base 比值：两个合格 trial 为 1.391 和 1.498；trial 0 因 A16 第三次应用重放时暖机 CV>2% 不合格，比值为 1.512。按原判据为信息不足。三次观测都与小幅读工作带（0.95–1.20）不相容。增加的读取由 L2 供给：A16 的 DRAM 读少 6–7%，L2 命中率约 92.8%。base 的 L2 读在三个 trial 间相差 8.5%，可能受 3-pass replay 的缓存状态影响，尚未单独检验。见[补测报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-pending-20261026-report.md)。

10-27 保序修复，作业 785518–785610：21 个原轮次按原进程顺序和原环境变量在 a044 上重跑，其中 19 轮（共 20 个根）已用原评分器评分。
- 每个根先用原 `measure.replay()` 复现保存的旧评分，再对同位置的 a044 输出评分。
- 15 个根判定不变，包括 same_binary_repeatability、static-repeat-dispatch 的 hidden_supply、private-source 的大工作集响应，以及几个 partial_localization 和信息不足。
- 5 个根判定变化：
  - active-peer：A16 dual 相位 bridge 超过 5% 门槛（5.6%–6.6%），降为信息不足。
  - peer-output：只有一个同 SM 配对 trial，降为信息不足。
  - full-entry-mask-001：观察器扰动未过，降为信息不足。
  - full-entry-mask-002：一个进程暖机次数不合格，降为信息不足。
  - equal-value-source-002：bridge 由失败变为通过。

这些变化都来自资格条件或 5% 门槛，没有出现资格满足后结论相反的情况。runtime-tail 和 input-counter 尚未用原评分器评分。见[修复报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/mechanism-repair-r2-report.md)。

10-27 单 pass 补测（作业 785959、785974）：每个 profile 只采一个指标，共 5 个 trial，其中 4 个合格。
- L2 读 A16/base 为 1.440–1.498，仍高于大幅读工作带的上界 1.40，按原判据为不定。
- DRAM 读比值为 0.94–0.95：A16 多出来的读取由 L2 供给，额外量稳定在 +44%–50%。
- 4 个合格 trial 中，base 的 L2 读量分散为 4.1%（3 pass 时为 8.5%），多 pass 重放只能解释一部分。

见[结果](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-followup-20261027/RESULTS.md)。
