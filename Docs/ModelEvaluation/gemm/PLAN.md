# 三类模型与机制研究推进计划

更新：2026-10-28。目标、选题与验收规则见[GOALS](GOALS.md)，模型差别见[对照表](mechanistic_modeling.md)。本页维护当前重点、推进依赖和后续候选的进入条件；各轮登记与结果在对应实验记录中，入口为[按问题索引](experiments/gh200_sm90/README.md#research-questions)，当前分工见[EXECUTION](experiments/gh200_sm90/access_rules/EXECUTION.md)。

## 1. 近期顺序

### 当前重点

a044 预约已于 2026-10-27 22:10 结束，重测范围与评分见[a044 阶段说明](experiments/gh200_sm90/Q10-time-mapping/A044-REMEASUREMENT.md)。当前优先复用已有数据，核对输入服务模型在纯 TMA 与完整 GEMM 中的响应，检验能同时解释低在途请求往返、持续行距增长和私有源阶段端点的候选。需要新观测的问题保留为 §3 的测量候选；免费 ROMEO 上的已登记测量按 [GOALS 当前执行范围](GOALS.md#current-scope)执行。

10-22 起的主交付是整体重建输入供给服务，保留已验证的程序、计算／释放、频率和输出模块，不要求新结构继承旧 core 价格或取 max 的完成接法。10-22 至 10-23 共试过十余种固定结构，均已停止，逐轮结果见[GEMM 输入网络](experiments/gh200_sm90/Q02-tma-request/gemm-input-network.md)、[缓存组织](experiments/gh200_sm90/Q03-source-service/cache-organization.md)、[准备与返回](experiments/gh200_sm90/Q02-tma-request/request-preparation-return.md)、[输入／输出联合窗口](experiments/gh200_sm90/Q07-smem/input-output-joint.md)。

此前一轮[请求形成先于缓存查询](../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-preparation-return/20261023-Q14-formation-before-cache-001/README.md)第一次让原源两段 L 产生持续增长（14171／14455 cycle），四 L 最大误差由 34.422% 降到 9.338%。但 plain 最大误差仍为 14.428%，基础 CTA0 后续 full 未完成 124/126，纯 TMA Q1 最大分项误差 32.224%，私有源阶段端点仍失败，当前参数版本停止。下一轮以请求返回、full 就绪和查询位置的联合约束选择规则，固定部分及有限清单另行登记。

下列证据用于下一版候选的联合检验，数值与结论限于各自原实验的条件和资格范围；跨卡复测未取得合格观测的项目，保留原卡结论与复测缺口。

| 约束 | 证据与适用范围 | 出处 |
|---|---|---|
| 低在途请求往返 | 纯 TMA Q1 每组往返实测约 675–750 ns，现有候选约 571–588 ns；控制已吻合，缺口在往返本身，移动返回反馈位置无效。同时须保持 unique／global1 Q4 窗口 | [准备与返回](experiments/gh200_sm90/Q02-tma-request/request-preparation-return.md) |
| 持续行距增长 | G3 A16 两段 L 增长须 >1000 cycle；旧局部组合及多数固定结构为 0；单 CTA Q1 持续增长、Q4 只在预填增长；A80/A16 增长比约 0.889，按覆盖总数的闭合规则被反证 | [请求形状](experiments/gh200_sm90/Q02-tma-request/request-shape-response.md)、[GEMM 输入网络](experiments/gh200_sm90/Q02-tma-request/gemm-input-network.md) |
| 输入本身持续未完成 | A16 后续 full 查询未完成 91.823%、基础 4.099%（774830）；B1 发布后的查询年龄下界 1815–3493 cycle，超过旧完成上界。只延长计算的共同 SMEM 服务不能产生该状态 | [完成观测](experiments/gh200_sm90/Q02-tma-request/completion-observation.md)、[联合窗口](experiments/gh200_sm90/Q07-smem/input-output-joint.md) |
| 单遍增长、三遍隐藏 | 774591 的两段 A16 增量比落入冻结可隐藏带，排除固定不可隐藏加项 | [局部规则接回 GEMM](experiments/gh200_sm90/Q14-composition/gemm-connection.md) |
| 同行依赖 | 目标调用的同行工作影响大部分 A16 增长（775581）；屏蔽同行 global 输出 TMA 后增长保留（775662） | 同上 |
| 源工作集 | 逐 CTA 私有源的 L 周期比 2.38–4.70；同服务接回后 plain 误差 −0.765%／−0.033%，阶段端点仍失败 | [GEMM 源工作集](experiments/gh200_sm90/Q03-source-service/gemm-source-workset.md) |
| 源折叠反增 | 004 同二进制折叠后第二输出的增长保留比 [1.324, 1.485]，超出两个冻结类别 | [源复用](experiments/gh200_sm90/Q02-tma-request/gemm-source-reuse.md) |

a044 上的 cfg_b 单 pass 输入计数比值为 1.440–1.498，落在预列的两条响应带之间，未区分工作量与联合推进两种解释，见[NCU 补测结果](../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-followup-20261027/RESULTS.md)。

已停止、不再另开轮次的形式：旧固定价格下的缓存／队列／容量微调与单 CTA swizzle 补测；准备价格扫描；行粒度、τ 与合并窗口扫描；读取价与合并容量扩展；局部 SMEM 服务率与节拍扫描；返回反馈位置与单一读取价调整；旧固定 τ 逐副本服务；用带宽价格或频率修正代替端点响应。

### 五阶段推进

2026-10-19 起按下表推进，覆盖范围仍为九个方向；九方向及 10-17 首版十阶段的交付见[微基准地图 §6](model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)，授权与单轮要求见 [GOALS §5.4–5.6](GOALS.md#current-scope)。

| 阶段 | 交付与复用 | 进入下一步的依据 |
|---|---|---|
| 1. 固定真实程序事实 | 复用[cfg_b程序图](model/mechanism/program/cutlass_program_graph.md)、源码／SASS核对、工作列表、真实三请求、槽代次与输出顺序 | 已完成的程序事实直接使用；新增干预只核对实际改变的程序部分 |
| 2. 建立最小模块 | 分别交付搬运、计算、同步／释放和输出的独立运行入口及端点时间线，接口与端点一致性按[GOALS的模块交付要求](GOALS.md#current-scope)；复用已有接口、A/Q05、B/C、TMA与输出证据，标明当前假设及缺口 | 所需外部行为可以独立运行、对照局部证据。寄存器／驻留先提供容量和启动依赖，普通L1／LSU保持独立收口 |
| 3. 验证最小连接 | 接通输入就绪与full返回，并闭合计算完成、WGMMA等待返回、empty发布／槽释放和producer再取得槽；一次接入一条连接，固定其余模块；复用[shared_stage_events](../../../microbench/gh200_resource_campaign/access_rules/supply_model.py)、[ConditionSyncService](../../../microbench/gh200_resource_campaign/access_rules/mbarrier_service.py)及[短链生命周期](../../../microbench/gh200_resource_campaign/access_rules/cfg_b_handoff_chain.py) | 同一规则在已有短／长链及S4/S6上通过登记的必要约束；失败定位到新增连接，信息不足时列出最小缺失观测，继续不依赖它的工作 |
| 4. 加入共享与竞争 | 相关局部连接通过所需约束后，接共享源和输出重叠；复用多CTA源状态、两域、源重排及输入／输出四组织证据 | 分别检验独立组合与联合服务，明确新增共享状态及其端点响应；多CTA研究的独立进入条件见[当前执行范围](GOALS.md#current-scope) |
| 5. 完整GEMM验收 | 相关模块、局部连接和交互完成登记的检验后，接回真实工作列表，完成原12条件的同版开发回归；再冻结并取得独立C2/C3证据，交付执行时间线、关键端点、stage／行距／CTA推进及plain预测 | 按原C2/C3标准验收；经验替代和失败条件保留 |

- 阶段 1、2 已交付：程序事实，以及搬运、计算、同步／释放、输出的独立入口与端点时间线。
- 阶段 3 部分完成：计算完成→WGMMA 等待返回→empty 发布／槽释放→producer 再取得槽，在驻留链上通过冻结长度／stage 外推，见[局部同步链](experiments/gh200_sm90/Q04-sync-slot/local-sync-chain.md)、[计算完成与提交](experiments/gh200_sm90/Q05-wgmma/local-compute-chain.md)。搬运→输入就绪→full 返回在单 CTA 原生 D 链上通过长度／stage 外推，但首组 full 返回前 773 cycle 在输入就绪与阻塞返回之间的分配不可辨识，R1536 外推未通过，见[原生 D 链](experiments/gh200_sm90/Q02-tma-request/native-d-chain.md)。
- 阶段 4 进行中：源并集由 53.6 MiB 减到 26.8 MiB 的冻结响应获限定支持（763587），绝对窗口和 CTA 结束时刻分布仍未解释，见[源组织](experiments/gh200_sm90/Q02-tma-request/source-organization.md)、[共享源固定形式](experiments/gh200_sm90/Q02-tma-request/shared-source-forms.md)。输入供给服务的整体替代见上文当前重点。
- 阶段 5：默认版仍为 10-16 前缀版的 10.225%／25.422%；局部同步链规则作为显式开发选项为 8.706%／25.334%，stage 响应 6.7%（实测 10.5%），行距响应仍为 0；独立 C2/C3 未通过，见[cfg_b 组合](experiments/gh200_sm90/Q14-composition/cfg-b-composition.md)、[局部规则接回 GEMM](experiments/gh200_sm90/Q14-composition/gemm-connection.md)。

## 2. 三类模型与程序图的接入依赖

cfg_b 组合原第 1、2 步的 CPU 交付已完成，验收与入口见[组合 §4](model/mechanism/composition.md#4-输出)。请求级候选已接通共同程序的请求、计算完成和槽释放端点，以及阶段频率、经验前缀和 plain 映射；原12条件的结果与失败反例已保留。替换供给规则时仍检查[联合约束](model/mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md#下一候选的联合约束入口)，该整组回归不作为阶段 2–4 转段的门槛。

| 工作 | 推进顺序 | 接口或交付入口 |
|---|---|---|
| 规格基线 | 独立计算入口已完成；比较时记录产品/参考频率口径 | [规格基线计算](model/spec/spec_baseline.md)；C1-01继续使用原冻结口径 |
| 半经验模型 | 保留V09冻结版本与C1-01已有结果，先对照当前组合；a044 上 V09 只评了 34 个完整条件的子集 | C1-01已经分别列出纯规格、wave_roofline、structural与TileSight，完整时间归C3；a044 结果见[阶段说明](experiments/gh200_sm90/Q10-time-mapping/A044-REMEASUREMENT.md) |
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

a044 预约结束后留下的事项，按性质区分：

| 事项 | 性质 | 状态与进入条件 |
|---|---|---|
| 后 6 个资源家族 | 测量候选 | 5271 次调用；tensor2D 一块原始约 55.7 GiB，超过 scratch 配额。TMA2D 首块（780806）因节点磁盘写满中断：已回传 152 个 validate-only 数值检查，完整性核验通过，原数值 oracle 尚未运行；性能采样未完成，不计该家族完成，见[存储中断记录](../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/storage-incident-20261023/README.md)。恢复需管理员提高配额，或改为在节点上做 oracle、只回传压缩结果；后者改变证据保留方式，由用户决定 |
| legacy r5 历史回归 | 测量候选 | 26,769 次，原预算约 17.5 h，需要新的预约 |
| C1-01 | 暂停中的已登记任务 | a044 的 30 个开发块已回传并通过完整性检查。模型比较、选择和冻结可用现有开发数据在 CPU 上完成，120 条件独立留出才需要 GPU。按 GOALS 这条线仍暂停，是否恢复由用户决定，不因数据齐备自动重启 |
| V09 两个 G1_base 条件 | 保留的观测缺口 | 原协议重试已耗尽（暖机末五次 CV > 2%），本轮决定不补这 28 个槽；再测需要新的研究目的，见[子集报告](../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/v09-complete-subset-report.md) |
| 交换 AB／A／B 被 NCU profile 的 kernel 启动顺序 | 未登记的新实验候选 | 786546 中 AB 每次都是第一个被 profile 的 launch，“AB 请求本身多读”与“首个被 profile 的 launch 多读”未分开；与 kernel 内 TMA 指令的发出顺序无关。需要计数器权限，见[阶段说明](experiments/gh200_sm90/Q10-time-mapping/A044-REMEASUREMENT.md) |

## 4. 协作与资料入口

- TileLang方向合作者提供降级IR到程序图的前端；本仓库维护服务候选、参数证据与组合验证。首轮人工核对图不等待自动前端。
- 历史复用入口：[TMA盘点](model/mechanism/hardware_rules/TMA-HISTORY-REUSE.md)、[首版输入模型](model/mechanism/hardware_rules/TMA-INPUT-EVENT-MODEL.md)、[模型候选与反例](model/semi_empirical/MODEL-CANDIDATES.md)。
- 用户有两次向 NVIDIA 微架构研究人员请教的机会；保留为候选分歧明确后的咨询资源，不作为当前工作的前置条件。
