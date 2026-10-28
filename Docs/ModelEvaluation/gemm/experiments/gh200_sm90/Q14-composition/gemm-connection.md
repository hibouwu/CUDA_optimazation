# 完整 GEMM 中的跨路径交互

15 个目录（10-18 至 10-22）：在真实 G3 上检验一条路径怎样改变另一条路径的时间——输出推进与输入／计算、计算量与输入等待和槽复用、多项局部规则的联合迁移，以及同行 CTA 活动。单机制的接回已移到各自问题：计算完成与 wait0 见 [Q05](../Q05-wgmma/local-compute-chain.md)，等待方式见 [Q04](../Q04-sync-slot/full-wait-policy.md)，请求共享、完成观测与源复用见 [Q02](../Q02-tma-request/gemm-input-network.md)，缓存与源工作集见 [Q03](../Q03-source-service/cache-organization.md)。

## 结论

**局部规则迁移**
- 局部联合规则按阶段频率迁移：d95／消费者继续133.364／empty完成275，替换旧reuse220，S4 plain误差9.639%→7.065%，stage响应3.903%→6.738%；S6只保留d95的变化，A16与CTA分散仍失败。通过三条件开发筛查；其余条件及G6回归均已完成，见下条。
- 联合规则其余八条件回归完成，新增88调用，复用三项。11项中位／最大误差10.346%／25.334%，同11项旧版10.810%／25.422%；5/11在10%内，关键CTA命中0/110，random后续L略恶化。正式入口可从保存profile启用，G1新增11调用逐值一致、80测试通过。G6另列11调用已完成，同12条件中位8.706%／最大25.334%、关键CTA1/120；G6为组级事件，默认版本保持。
- 保存事件的行距配对：源就绪有响应，新增等待只在CTA首组；各输出主循环长度保持，而实测A16／B80／B112的first_L／later_L增长2744–5920ns。下一项区分首组之后的输入就绪与full返回，未选新价格，见[逐事件分析](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261020-Q14-local-bundle-regression-001/README.md#行距残差定位)。
- 真实G3/S6的静态单遍／三遍对照通过同批R桥接；36进程支持计算间隔可隐藏A16行距增量，排除固定不可隐藏加项。尚未区分源服务、等待返回与反压位置，组合成绩保持，见[冻结预测与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-static-repeat-dispatch-001/README.md)。

**输出与背景**
- 首输出延后：模型的早段、后段响应为 0，旧实测为 −544／−128 ns。
- 输出源读取后的共享占用只复现了部分方向；A16 误差保持 25.422%。

**程序干预**
- 活动 4/6 槽 × 1/3 遍计算四个条件，误差 24.7%、24.8%、9.8%、10.5%。
- 重复计算改造程序中约 1.25 万 cycle 只属于该改造（program-budget-scope），不作为原 cfg_b 的价格。
- 计算量开关的原binary在单遍下，A16的两段周期增量仅105–133／587–687.5cycle，低于冻结1000cycle辨识下限；774436十二进程资格通过后按计划停止，三遍未测。此改造程序没有保留原cfg_b的持续行距反例，见[计算间隔对照](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-stride-compute-discrimination-001/README.md)。

**判断**
- 默认原12条件成绩保持；本轮联合规则改善了S4的开发预测，原12条件同版开发回归已完成，独立验收未通过。共享源、full完成／返回及CTA推进仍有未解连接，见 [g3-stage-credit](../Q04-sync-slot/g3-stage-credit.md) 与 [source-organization](../Q02-tma-request/source-organization.md)。

## 轮次

按日期排列；全部结果目录见[Q14-composition/gemm-connection](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 同一S6的活动槽／计算量能否由同版组合回放 | 746824已有四条件；修活动4／6槽与1／3遍程序事实，物理资源、服务价及运行前频率固定 | 默认行为保持，三遍stage差消失方向可回放；单遍第一L差50.2对624ns、plain仍失败，保留改造程序的主循环残差 | [改前反例、程序接入及四条件](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-runtime-intervention-001/EXECUTION.md) |
| 改造程序的共同预算是否也存在于原始cfg_b | 八既有条件48份dual，从同SM原始周期核对原S4/S6、活动槽与重复循环程序，核对两份plain SASS | 原S6剩376／584cycle，重复体剩约1.25万；停止向普通cfg_b迁移，下一项回原stage等待连接 | [原始记录、SASS与判定](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-program-budget-scope-001/EXECUTION.md) |
| 真实GEMM首输出延后能否由现用服务表达（CPU开发回放） | S4/S6×flag1/2，各8次flag0暖机；只接E0等待下一WG mma返回，固定服务与价格 | 默认完整报告精确，四目标早段差0对旧配对−544／−128ns；保留背景交互缺口 | [接线、四目标与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-epilogue-defer-001/EXECUTION.md) |
| 输出源读取后的共享占用是否产生背景响应（CPU开发） | 原两槽/τ，8KiB异步工作在source_read_done后占0.5单位；四旧条件、其它模块固定 | S4 early/post−198.369185/+2.582738ns、S6均0；必要门过，中央幅度未解释，保留假设 | [固定规则、四条件及复现](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-shared-output-service-001/EXECUTION.md) |
| 输出共享假设能否修复原A16最大误差（CPU回归） | 原A16输入/11调用，固定上一轮规则和全部价格；超10%即停止余项 | 源就绪及46个CTA MAIN改变，关键CTA/plain保持，25.422%仍失败；剩余两条件停止 | [完整报告、端点比较与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-output-write-gemm-001/EXECUTION.md) |

| 局部联合预测类能否迁移cfg_b（CPU） | 固定既有价格；d95／继续133.364／empty275替换旧reuse220，按阶段频率换算；最多S6／S4／A16三条件33完整调用 | 26小调用／33完整调用完成，50688组接线符合价格；S4误差9.639%→7.065%、stage响应6.738%，保留开发迁移假设，A16／CTA分散仍失败 | [计划与接法](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261020-Q14-local-bundle-transfer-001/README.md) |

| 局部联合规则能否覆盖其余原条件（CPU回归） | 固定上一轮规则及原输入，八条件各11调用，复用三条件；G6待单独回归 | 八条件88调用完成，11项中位10.346%／最大25.334%，关键CTA0/110；正式入口新增11调用逐值一致，80测试通过；G6另列完成后同12项中位8.706%／最大25.334%，独立C2/C3未完成 | [计划与产物](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261020-Q14-local-bundle-regression-001/README.md) |

| 联合规则的G6长K回归（CPU） | 同一规则、原G6输入／11调用协议；CPU上限70分钟，保存组级事件 | 11调用完成、2637.795秒，plain5320.468µs、误差+1.912811%；同12项中位8.706%／最大25.334%，关键CTA1/120，完整请求完成图仍缺 | [计划与进度](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261020-Q14-local-bundle-G6-001/README.md) |
| 最新联合规则的关键路径能否支持行距／stage归因（保存轨迹分析） | 固定全部输入与价格，读取G3 S6／S4／A16三条保存时间线，只分析各自最晚CTA；完整模型／拟合／GPU均0 | 节点重建通过1e-6ns门槛；同组等待进入差≤7.28e-12ns。修复无关producer尾部被标零时差；S4原plain响应6.738%保持，后续full／empty观察缺口保持 | [计划、反例与当前分解](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-critical-path-current-001/README.md) |
| 增加计算间隔能否隐藏A16行距代价 | 原746824 binary、活动S6及G3，基础／A16×一遍／三遍；plain／dual各三进程 | 774436单遍十二进程资格通过，周期增量低于1000cycle辨识下限，三遍未测、两假设未判定；恢复单遍路径后再立项 | [计划与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-stride-compute-discrimination-001/README.md) |
| 去掉动态重复循环能否恢复单遍目标反例 | 原单遍gemm后按runtime开关直接加两遍；固定源/槽/提交/释放，同批原V09为R | 774459完成24进程，R重现反例，N1基础慢约22%且行距响应塌缩，桥接未过、N3未采 | [计划与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-direct-repeat-path-001/README.md) |
| 固定原S6槽推进后能否保留行距反例 | PipelineState及producer_tail恢复原文件，保留直接额外两遍开关，同批R／N1先桥接 | 774523构建成功，774550完成24合格进程；仍未过R桥接，N3未采，该实现停止 | [计划与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-static-stage-repeat-001/README.md) |
| 循环外选择静态计算量能否保持计算组组织并隐藏行距增量 | host选择单遍／三遍静态类型，Ktile内去掉runtime重复分支 | 774570资源解析漏冒号退出；修复1的774581构建成功。774591完成36合格进程，四静态实例组织及R／N1桥接通过；两L增量比落入冻结可隐藏带，固定不可隐藏加项被排除 | [计划与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-static-repeat-dispatch-001/README.md) |
| 保留原工作列表后，持续A16增长是否依赖同行CTA | 固定G3/S6、grid132、CTA0两输出；新132/1均先16次全量前缀，仅目标调用放行132/1；先同批R桥接 | 775581有36/36有效、零失败；三trial同SM124、桥接通过。两L增量由12536–15401／11740–11943降至8–117／1206–1362cycle，两比值入冻结同行依赖带；支持同行工作影响大部分增长，源／前端／输出位置未分；模型／拟合0，开发定位 | [计划、原始回放与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-active-peer-001/README.md) |
| 同行输出TMA是否构成当前共享影响 | 保留132CTA输入／计算与CTA0程序，只屏蔽同行global TMA copy；先全写桥接 | 775646未生效gate修复后775647通过；775662完成17秒、35/36有效，一暖机CV失败不补。有效trial0／2两响应比[0.960119,1.052763]／[0.895419,1.118705]入冻结保留带；同行全局输出TMA不是大部分增长必要条件，物理服务未定价；开发定位、模型／拟合0 | [计划、原始回放与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-peer-output-001/README.md) |
| 程序源折叠和输出copy开关是否准确进入组合 | 16原源全输出前缀后正式原源／折叠×base/A16，固定原服务 | 68引擎调用、166.37秒、GPU/拟合0；正式输出2112→16条，STS与控制保留，202752输入身份一致，默认回归保持；两L行距增量仍为0，原供给结构未解释实测 | [实现、四条件评分](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/integration/20261022-Q14-program-intervention-001/README.md) |

## a044 复测（2026-10-23～26）

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
