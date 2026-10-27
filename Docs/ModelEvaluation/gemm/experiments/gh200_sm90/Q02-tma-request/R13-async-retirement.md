# R13：有效供给、阶段与可辨识性

当前半经验参数见[MODEL-CANDIDATES](../../../model/semi_empirical/MODEL-CANDIDATES.md)，硬件候选见[TMA 假设](../../../model/mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md)，软件槽代次见[程序图](../../../model/mechanism/program/cutlass_program_graph.md)；本页及 reanalysis 保留测量和历史推导。

## 当前结论

first/later 的有效供给模型是 V09 r2 的候选来源，整体及非默认 stage 的局部迁移未通过。软件 stage、源 ready、MMA 返回、槽释放和下一请求的允许时刻分别建模；当前没有可采用的新 stage 时间价格。

工作账本与生命周期接口已补齐 cfg_a/c 的源 atom、multicast 目的和跨 rank 依赖；结构与守恒复核通过，只说明依赖接线成立，不授予未测端点、时间或跨卡资格。cfg_b 的释放标记尚未定位全部128次 empty arrival，局部观察失败保留。

固定启动差、恒定每 Ktile 罚项、同质 FIFO、固定源订阅截止和本地 E0 排他等具体候选均保留失败证据。统一发射间隔不能修复实测排序；空间离散尚未分开源、CTA 与 SM；低联合窗口误差不能覆盖端点或全卡跨度失败。源组标签、目标实测入口和频率都未作为正式预测价格。

供给、输出干预和时钟换算的支持范围分别维护；m11 普通 LDS/STS 的开发条件关系见 [R03](../Q07-smem/R03-access-demand.md#m11-current)，不迁移到 TMA/WGMMA 或有限 stage 模型。

## 作业索引

| 证据 | 条件与作用 | 可复核入口 |
|---|---|---|
| R05-D，735059 | 立即释放消费者，stage 2/4；共享小源与独立大源 | [正式报告](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/R03-access-demand/20261006-c-job735059/r05-formal-v2/report.md) |
| R13，735876/735985/736169 | 18 条件的 plain 服务率、ready/retire/reuse 事件对 | [ready](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R13-async-retirement/20261007-R13-job735876-v8-formal-ready-review-B/report.md)、[ready 补查](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R13-async-retirement/20261007-R13-job735985-v9-ready-recheck-review-B/report.md)、[retire](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R13-async-retirement/20261007-R13-job735985-v9-formal-retire-review-B/report.md)、[reuse](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R13-async-retirement/20261007-R13-job736169-v11-formal-reuse-review-B/report.md) |
| V08F，737322 | 51 条后续窗口拟合；14 条尾部/D 行距诊断 | [候选与留组结果](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V01-V08-validation/20261008-V08F-job737322-v1/reanalysis/A-20261009-supply-v3/summary.json) |
| SM 扫描，738101 | cfg_a 四档 grid、三个 pitch，固定 K4096 | [软件波次](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R13-async-retirement/20261009-R13-sm-job738101/reanalysis/shared-cap-candidates-v3/summary.json)、[有限预填](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R13-async-retirement/20261009-R13-sm-job738101/reanalysis/prefill-cap-candidates-v1/summary.json) |
| 同卡 B13，738203/738296/738307 | B 行距和 A/B source-map 共 13 条件，供给与填零组合开发 | [需求/填零候选](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/R10-layout-cache/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-valid-fill-v4/summary.json)、[CTA 组合](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/R10-layout-cache/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-cta-composition-v3/summary.json) |
| 三配置，738496 | 29 条件的 first/later 定值；cfg_b 后与旧 B13 有条件联合 | [phase 模型](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R13-async-retirement/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-phase-supply-v1/models.json)、[cfg_b 联合报告](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R13-async-retirement/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-b-joint-supply-v1/summary.json) |
| 首轮填零，738707/738865 | 只补 A/C first；六组训练行与拒绝范围 | [模型](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R13-async-retirement/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/models.json)、[验证](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R13-async-retirement/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/verification.json)、[支持定位](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R13-async-retirement/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/support-diagnosis.json) |
| V09，739011 | 冻结后的新几何、行距和 stage 迁移 | [完整留出与判定](../Q15-full-time/V09-component-validation.md) |
| stage×K，739095 | 同卡四条件；固定启动差和恒定每Ktile罚项均不充分 | [成对窗口、协议扰动与缩放残差](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/B-controls.json) |
| 同次stage事件，739556 | CTA0第二输出Ktile16；额外L在mma返回前，局部等待与发射区间互相转移 | [窗口与累计进度](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-events.md) |
| consumer累计进度，739578 | 同几何stage4/6；相邻commit分解与早/后段累计差，区分循环推进和最终drain | [同次端点与递推对照](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-progress.md) |
| 组内服务递推，离线 | 仅stage6三个局部量校准；独立局部约束的q剖面仍高估stage4，推导有限窗口下界 | [参数族与逐分项比较](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-admission.md) |
| 交接不可辨识性，离线 | 不同ready/交接组合给出相同stage6局部量及L，却产生不同stage4响应 | [同调用下界与参数反例](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-handoff.md) |
| 同次计算组交接，739686 | 原两stage×plain/progress/handoff；交接非零，stage差主要在首full等待，局部观察扰动明显 | [三段闭合与累计推进](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-handoff-samples.md) |
| 固定入口，离线 | 旧延迟违反后段必要界；stage4必须有内部停顿，不能只移动入口或末端领先量 | [条件重放与参数无关界](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-conditioned.md) |
| 共同复用组合，离线 | 有限窗口与任意单一producer间隙的条件界；宽分支须联立端点 | [中位区间与早段上界](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-reuse.md)、[边界间隙与适用限制](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-boundary.md)、[联合端点](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-joint.md) |
| 前一输出顺序，739717 | 消除早段E0重叠后stage4早段缩短，完整继承子类仍有缺口 | [R15统一分析](../Q06-handoff-output/R15-output-service.md#next-mainloop-overlap)、[逐进程结果](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-epilogue-order.md) |
| 输出背景，739740 | 固定CTA0处理、仅改变其它CTA首输出；stage4早段缩短、后段变长 | [R15统一分析](../Q06-handoff-output/R15-output-service.md#stage-output-background)、[同调用背景与进程差](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-background.md) |
| 第二WG初态，739790 | 仅CTA0预等S槽，stage4/6同binary开关；early与完整时间分列 | [同批配对与限制](#second-prefill)、[逐进程结果](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-second-prefill.md) |
| 背景基线，离线 | 区分早/后段输出相位；仅两个无E0覆盖窗口约束c/T，固定后迁移 | [条件范围与未定项](#stage-background-baseline)、[逐进程包络](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-baseline.md) |
| 共享服务候选，离线 | 真实软件依赖自行生成E0；低窗口残差仍伴随相位/本地响应失败 | [候选边界](#stage-shared-service)、[拟合与全卡对照](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-shared-service.md) |
| 首次供给，739629 | 同binary新增打点开关 × 起步0/512 ns × B0/B32；定位S增长 | [同批观察扰动与端点](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R13-first-supply.md) |
| 独立ready，739645 | 只读phase0观察开关 × 起步0/512 ns × B0/B32；10/40可绑定、30/40边界跨FIRST | [观察扰动与区间](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R13-ready-observer.md) |
| 释放界ready，739653 | 同四条件、同binary联合观察开关；40/40可绑定，ready前增长不能由consumer返回单独解释 | [逐进程区间与模型必要界](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R13-ready-release.md) |

<a id="supply-work-ledger"></a>

## 推导与历史证据索引

本页只保留当前结论和入口；原有局部条件、公式、反证与逐例证据移到各自的 reanalysis README，原锚点继续可用。整理不修改采样、参数、冻结成绩或资格。

| 原问题 | 详情 |
|---|---|
| <a id="作业与分析索引"></a>作业与分析索引 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#作业与分析索引) |
| <a id="supply-work-ledger"></a><a id="2026-10-11统一工作账本与局部生命周期复核"></a>2026-10-11：统一工作账本与局部生命周期复核 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#2026-10-11统一工作账本与局部生命周期复核) |
| <a id="局部图必须分开的事件"></a>局部图必须分开的事件 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#局部图必须分开的事件) |
| <a id="已有约束逐项结论"></a>已有约束逐项结论 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#已有约束逐项结论) |
| <a id="cfg-a-source-composition"></a><a id="cfg_a源atom与跨rank依赖的组合接口"></a>cfg_a源atom与跨rank依赖的组合接口 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#cfg_a源atom与跨rank依赖的组合接口) |
| <a id="release-prerequisites"></a><a id="cfg_c源切片与闭合rank依赖影响范围已核对服务时间仍未定"></a>cfg_c源切片与闭合rank依赖：影响范围已核对，服务时间仍未定 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#cfg_c源切片与闭合rank依赖影响范围已核对服务时间仍未定) |
| <a id="cfg_b的释放标记没有定位全部128次empty-arrival"></a>cfg_b的释放标记没有定位全部128次empty arrival | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#cfg_b的释放标记没有定位全部128次empty-arrival) |
| <a id="四warp-smoke保留了明显的局部观察差"></a>四warp smoke保留了明显的局部观察差 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#四warp-smoke保留了明显的局部观察差) |
| <a id="ef卡冻结研究未通过上游门槛"></a>ef卡冻结研究未通过上游门槛 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#ef卡冻结研究未通过上游门槛) |
| <a id="second-prefill"></a><a id="第二wg的初态是否影响早段推进"></a>第二WG的初态是否影响早段推进？ | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#第二wg的初态是否影响早段推进) |
| <a id="source-candidate-replay"></a><a id="在途共享能解释局部预等方向但当前参数仍失配空间完成分布"></a>在途共享能解释局部预等方向，但当前参数仍失配空间完成分布 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#在途共享能解释局部预等方向但当前参数仍失配空间完成分布) |
| <a id="request-path-joint"></a><a id="独立请求路径的前缀选点未复现stage4预等改善"></a>独立请求路径的前缀选点未复现stage4预等改善 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#独立请求路径的前缀选点未复现stage4预等改善) |
| <a id="source-spatial"></a><a id="stage4周期离散集中在a行组但源与位置尚未分开"></a>stage4周期离散集中在A行组，但源与位置尚未分开 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#stage4周期离散集中在a行组但源与位置尚未分开) |
| <a id="source-offsets"></a><a id="实测producer起步差仍未让共享候选产生足够分散"></a>实测producer起步差仍未让共享候选产生足够分散 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#实测producer起步差仍未让共享候选产生足够分散) |
| <a id="a-source-rotation-preparation"></a><a id="a源行组轮换未给出稳定的地址跟随规律"></a>A源行组轮换未给出稳定的地址跟随规律 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#a源行组轮换未给出稳定的地址跟随规律) |
| <a id="source-join-boundary"></a><a id="缩短源订阅窗口没有同时修复空间分散与预等响应"></a>缩短源订阅窗口没有同时修复空间分散与预等响应 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#缩短源订阅窗口没有同时修复空间分散与预等响应) |
| <a id="local-epilogue-exclusion"></a><a id="共同读延迟全部移到目的后程等价于已有service_end截止"></a>共同读延迟全部移到目的后程，等价于已有service_end截止 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#共同读延迟全部移到目的后程等价于已有service_end截止) |
| <a id="本地e0排他只能解释部分响应不能修复defer2缺口"></a>本地E0排他只能解释部分响应，不能修复defer2缺口 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#本地e0排他只能解释部分响应不能修复defer2缺口) |
| <a id="stage-background-baseline"></a><a id="没有e0覆盖的窗口能约束什么"></a>没有E0覆盖的窗口能约束什么？ | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#没有e0覆盖的窗口能约束什么) |
| <a id="stage-shared-service"></a><a id="一条共享服务队列能解释输出干预吗"></a>一条共享服务队列能解释输出干预吗？ | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#一条共享服务队列能解释输出干预吗) |
| <a id="stage-order-invariant"></a><a id="统一发射间隔为什么不能修复排序"></a>统一发射间隔为什么不能修复排序？ | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#统一发射间隔为什么不能修复排序) |
| <a id="stage-request-formation"></a><a id="起步分散与请求粒度"></a>起步分散与请求粒度 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#起步分散与请求粒度) |
| <a id="序列供给消费者与在途组织会改变结果"></a>序列供给：消费者与在途组织会改变结果 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#序列供给消费者与在途组织会改变结果) |
| <a id="r05-d"></a><a id="readyretire-与槽复用的观测边界"></a>ready、retire 与槽复用的观测边界 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#readyretire-与槽复用的观测边界) |
| <a id="v08-supply"></a><a id="v08-supply-fit"></a><a id="有效供给函数受到哪些约束"></a>有效供给函数受到哪些约束 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#有效供给函数受到哪些约束) |
| <a id="requested-sm-pitch"></a><a id="软件波次和有限预填能解释多少"></a>软件波次和有限预填能解释多少 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#软件波次和有限预填能解释多少) |
| <a id="composed-coverage"></a><a id="地址覆盖与填零需求的边界"></a>地址覆盖与填零需求的边界 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#地址覆盖与填零需求的边界) |
| <a id="valid-fill-candidate"></a><a id="fill-cta-composition"></a><a id="firstlater-定值与非唯一预测"></a>first/later 定值与非唯一预测 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#firstlater-定值与非唯一预测) |
| <a id="cfg_b-19-条件的有条件联合"></a>cfg_b 19 条件的有条件联合 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#cfg_b-19-条件的有条件联合) |
| <a id="ac-首轮填零后仍有拒绝范围"></a>A/C 首轮填零后仍有拒绝范围 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#ac-首轮填零后仍有拒绝范围) |
| <a id="v09-stage-failure"></a><a id="v09-后当前缺口"></a>V09 后当前缺口 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#v09-后当前缺口) |
| <a id="stage参数为什么没有改变预测"></a>stage参数为什么没有改变预测？ | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#stage参数为什么没有改变预测) |
| <a id="同卡k对照固定启动差和恒定每ktile罚项均不充分"></a>同卡K对照：固定启动差和恒定每Ktile罚项均不充分 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#同卡k对照固定启动差和恒定每ktile罚项均不充分) |
| <a id="有限缓冲递推拟合略好仍不能迁移"></a>有限缓冲递推：拟合略好，仍不能迁移 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#有限缓冲递推拟合略好仍不能迁移) |
| <a id="同次稀疏事件差异不在最终drain单点等待也不能代表整l"></a>同次稀疏事件：差异不在最终drain，单点等待也不能代表整L | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#同次稀疏事件差异不在最终drain单点等待也不能代表整l) |
| <a id="局部约束进一步否定了哪些递推简化"></a>局部约束进一步否定了哪些递推简化？ | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#局部约束进一步否定了哪些递推简化) |
| <a id="累计进度把额外时间定位到哪里"></a>累计进度把额外时间定位到哪里？ | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#累计进度把额外时间定位到哪里) |
| <a id="stage-admission"></a><a id="组内发射重叠与共享ready延迟的迁移"></a>组内发射重叠与共享ready延迟的迁移 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#组内发射重叠与共享ready延迟的迁移) |
| <a id="stage-handoff"></a><a id="ready与计算组交接不能由现有校准分开"></a>ready与计算组交接不能由现有校准分开 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#ready与计算组交接不能由现有校准分开) |
| <a id="stage-handoff-measured"></a><a id="同调用补测交接非零stage差主要在首full等待"></a>同调用补测：交接非零，stage差主要在首full等待 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#同调用补测交接非零stage差主要在首full等待) |
| <a id="stage-conditioned"></a><a id="固定入口后仍须解释第二输出内部的停顿"></a>固定入口后仍须解释第二输出内部的停顿 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#固定入口后仍须解释第二输出内部的停顿) |
| <a id="stage-reuse"></a><a id="共同复用组合能约束后段尚不能解释启动"></a>共同复用组合能约束后段，尚不能解释启动 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#共同复用组合能约束后段尚不能解释启动) |
| <a id="stage-boundary"></a><a id="输出边界间隙不能替代缺失的早段关系"></a>输出边界间隙不能替代缺失的早段关系 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#输出边界间隙不能替代缺失的早段关系) |
| <a id="stage-joint"></a><a id="联合端点约束的是哪一层模型"></a>联合端点约束的是哪一层模型 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#联合端点约束的是哪一层模型) |
| <a id="stage-wait-sum"></a><a id="早段stage差在累计full等待与等待外反向变化"></a>早段stage差在累计full等待与等待外反向变化 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#早段stage差在累计full等待与等待外反向变化) |
| <a id="first-supply"></a><a id="首次供给增长发生在copy发出标记之后"></a>首次供给增长发生在copy发出标记之后 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#首次供给增长发生在copy发出标记之后) |
| <a id="ready-observer"></a><a id="首个ready随起步状态变晚consumer返回不能单独解释"></a>首个ready随起步状态变晚，consumer返回不能单独解释 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#首个ready随起步状态变晚consumer返回不能单独解释) |
| <a id="l缩短不等于持续供给变快"></a>L缩短不等于持续供给变快 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#l缩短不等于持续供给变快) |
| <a id="复核入口"></a>复核入口 | [原推导与证据](reanalysis/20261012-consolidation/R13/README.md#复核入口) |

## 相关轮次

以下各轮原登记在访问规则索引，按日期排列；全部结果目录见[Q02-tma-request/R13-async-retirement](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R13-async-retirement/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 739629：首对copy发出与等待入口：起步干预后的首供给增长发生在首次准备、等待入口还是发出之后？ | 固定CTA55工作/SM、B0/B32与0/512ns起步对照；同binary开关新增issue_end与wait_begin。 | prod→issue_end近乎不变，增长在issue_end→FIRST，consumer没有更晚入wait；异步完成与等待返回仍未分开，局部扰动保留。 | [原结果目录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R13-async-retirement/20261010-R13-first-supply)、[原页判定](#first-supply) |
