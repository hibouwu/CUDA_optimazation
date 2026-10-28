# 请求准备、发出受理与返回反馈

14 个目录（10-18 至 10-23）：producer 的逐轮准备、源受理返回与形成服务的返回反馈，在原生 D 链、TMA 探针与 G3 回放上检验。

## 结论

- empty 后准备 0→640 的同 binary 对照：响应 6618–6699 cycle，名义模型 2176 cycle，冻结成本域无法解释（774214／774221）。逐轮准备类通过独立长度响应：K4 为 2153–2163、K32 为 16044–16166 cycle，首段形式被排除（774347）；N0−R 的长度差仍未解释。
- 逐轮准备 P240／309.75 cycle 接回 G3 后，A16 行距响应仍仅 0.0931%（实测 28.94%），两参数版本停止。
- 返回连接已交付：同步 issue_return、异步准入与就绪分离、队列受理返回；队列受理返回 k565 的 Q4 误差 4.08%，保留开发假设。
- 请求形成先于缓存查询：四个 L 的最大误差 34.4%→9.3%，plain 仍高 14.4%，停止当前参数版本；形成服务的两种返回反馈 unique Q1 最大误差 33.6%／31.3%，均停止；同地址评分修正后仍失败。

## 轮次

按日期排列；全部结果目录见[Q02-tma-request/request-preparation-return](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 同步源的发出返回连接（CPU） | 固定全部价格，只透传可选issue_return_ns并约束producer后继；三合成例、G3 S6/S4两系统回归 | 三合成核对及S6/S4全部报告／时间线精确回归通过，共25引擎调用；只交付连接，未选择硬件返回事件或新增价格 | [执行与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261018-Q02-issue-return-001/EXECUTION.md) |
| 异步源实际准入返回接线（CPU） | 准入与就绪分开，容量2 PS第三请求等待及后到延后先到；旧async／sync／零价完整报告对照 | 准入10、ready30/100时B0均11；第三请求4发出／22准入，旧报告逐值保持 | [接线与回归](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261018-Q02-async-admission-001/README.md) |
| 队列受理返回（CPU） | 只改入队即返回，原工作价／两槽PS／尾与FIFO固定；三预定k分配、Q1及真实程序接线 | k565 Q4误差4.077727%通过、Q1保持，保留开发假设；另两分配停止 | [同条件比较、选择与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261018-Q02-queue-return-001/EXECUTION.md) |
| producer程序控制与源准入分工（CPU） | 固定源／计算／同步价格；分开empty后准备与各请求受理返回 | 73／81条producer指令、八零默认报告保持；新完整入口夹具失败，两修复用完并恢复活动代码，候选未接入；186请求372端点字段核对通过 | [有限轮次](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261020-Q02-producer-control-contract-001/README.md) |
| 状态化源受理能否通过producer后继改变共享推进 | 受理后查找保持，只将service_start接回producer返回；固定实际基址、D程序及旧价格；13调用 | 零排队等价保持；4723请求反馈生效，但包络／跨度仍失败；首项即停，不改完整模型 | [连接、工作守恒与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261020-Q02-admission-feedback-replay-001/README.md) |
| 生产者节流能否区分已有控制放置（CPU） | 固定S6/K13、两解释及H必要界端点，仅把组间隔0改为1024cycle | 八调用、拟合/GPU0；两解释均预测plain增6144cycle，存在相同可行响应，不补测 | [计划、预测与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261021-Q02-control-placement-intervention-001/README.md) |
| empty 后程序准备与请求受理接线（CPU） | 固定原硬件价格，准备7/5/3ns及名义640cycle干预；同步／异步与局部／完整入口同序 | 八默认报告逐字保持，47项测试通过；140/160调用、GPU0。两个解释的S4−S6响应带可分，实机程序成本待下一轮约束 | [接口、反例与推导](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261021-Q02-program-preparation-001/README.md) |
| empty后准备能否区分两控制解释 | 固定旧硬件价格与H；S6原程序／同binary0／640各三进程，成本与资格通过后冻结S4同配对 | 774214两有效进程后执行器失败，774221按同计划继续剩余七个；九进程资格通过。响应6618–6699cycle对名义2176，校准超域，未测S4 | [测前计划与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261021-Q02-preparation-delay-001/README.md) |
| 原有准备是否逐轮限制请求推进（CPU） | 固定H总预算、计算/同步；两个控制形式各16相关见证，核旧六条件与新两条件；首段类另作必要界 | 256/256核对通过；逐轮准备P=240–309.75cycle可保持旧六窗口并解释新响应；首段类上界3737cycle低于观测6617 | [计划与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261021-Q02-recurrent-preparation-001/README.md) |
| 逐轮准备能否预测独立长度 | 固定S6及旧R/N binary，K4／K32各三臂三进程；K4通过才测K32；四个留出条件、两个R桥接条件 | 774347十八进程资格／桥接通过；K4响应2153–2163、K32响应16044–16166cycle，均在冻结门槛内；首段形式被排除，N0−R长度差未解 | [冻结预测与执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261021-Q02-preparation-length-001/README.md) |
| 逐轮程序准备能否改善持续行距响应（CPU迁移） | 用P240／309.75cycle替换旧producer／operand gap；G3/A16各11调用，源／计算／输出／频率固定 | 正式44调用完成，另11调用为缺依赖的实现失败；行距响应仍约0.0931%，L增长均0，每档关键CTA0/20，两参数版本停止 | [计划与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261021-Q14-producer-preparation-transfer-001/README.md) |

## 补充说明

- empty后准备0→640的同binary对照：九进程资格通过，R→N0桥接通过；窗口响应6618–6699cycle，名义模型2176cycle，冻结成本域无法解释。停止本校准形式、S4未采（774214／774221）。随后逐轮准备类通过独立长度响应：K4为2153–2163、K32为16044–16166cycle，首段形式被排除（774347）。N0−R随长度增长，固定J及P／源残量的分配仍未解释，见[本轮结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261021-Q02-preparation-length-001/README.md)。

- 受理后查找同时反馈producer后继，FIFO／132源末调用4723次延后，包络／跨度只变−45.444／+13.953ns，13调用即停止；439296条源线互不重复，原价后端连续忙15.681µs，见[固定受理反馈](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261020-Q02-admission-feedback-replay-001/README.md)。

- 逐轮准备P240／309.75cycle替换旧软件gap后，G3／A16行距响应仍仅0.0931%，实测28.9371%；first_L／later_L增长均为0，实测5008／5920ns。两参数版本停止，不扩展stage／原12，见[有限迁移](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261021-Q14-producer-preparation-transfer-001/README.md)。

[请求形成先于缓存查询](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261023-Q14-formation-before-cache-001/README.md)已完成432调用、2119.952秒、GPU0：只用三纯TMA条件标定两服务价，再检验两G3和两私有源。原源两L增长14171／14455cycle，四L最大误差34.422%→9.338%；plain仍高14.428%，基础CTA0后续full未完成124/126；纯TMA Q1最大误差32.224%，私有源阶段端点仍失败。停止当前参数版本，默认与原12成绩保持。

- [请求返回同地址比较](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261023-Q14-address-matched-return-001/README.md)：固定形成轮参数，更正预测与评分进程的地址绑定；55调用、108.582秒、拟合／GPU0，unique Q1／Q4仍失败，当前参数版本继续停止。

- [形成服务的返回反馈](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261023-Q14-formation-return-001/README.md)：producer后继接Formation PS2开始／结束两形式，unique Q1最大误差33.590%／31.312%（基线33.266%），24分项全部低估，均停止。控制已吻合、等待和仅0.57–0.62，缺口在低在途请求往返；41调用、拟合／GPU0。
