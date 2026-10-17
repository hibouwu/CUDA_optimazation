# 三类模型与机制研究推进计划

更新：2026-10-17。目标、选题与验收规则见[GOALS](GOALS.md)，模型差别见[对照表](mechanistic_modeling.md)。本页维护顺序和依赖；具体测量状态与资源安排见[实验执行计划](experiments/gh200_sm90/access_rules/PLAN.md)。

## 1. 近期顺序

目标模式已按用户授权开启，依次完成九个模块阶段及最终 cfg_b 组合；阶段顺序和交付条件只在[微基准地图 §6](model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md)维护。本页登记当前阶段、有限轮次和依赖；范围与执行许可见 [GOALS §5.4](GOALS.md#current-scope)。

九阶段首批离线证据／接口与第10阶段同版cfg_b组合已交付。免费ROMEO的有限实测已完成L2、A、B/C、D、E：L2六形式停止，A驻留长度留出通过，B/C零额外控制具体形式停止，D/E完成开发数值与路径核对；原L1开发清单实测及独立评分也已完成，本轮九阶段首版交付已齐；原完整模型失败保持，不租用付费资源。准备、实测、条件规则和未解决项分别登记；局部阶段不要求全部 C2/C3 通过才转段，原模型验收标准保留。

A–E准备、实测与原判定的当前结论统一见[COMPUTE-01](experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md)。A的条件循环量只支持匹配驻留体；B/C反例及D/E复合窗口未产生可迁移的独立控制／供给／输出价格，原12条件组合预测与评分保持。

| 优先级 | 任务 | 当前阶段与下一步 | 依赖与收口 |
|---|---|---|---|
| 阶段1本轮收口 | [L2 有限登记](#stage-1-l2)、[模块级微基准地图](model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md) | 两个先导通过，[开发48进程](../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-execution-002/dev/summary.json)全部合格；目标CUDA12.9.41九模板审核及20项CPU检查通过 | [六形式](../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-fit-001/service.json)按S32／J4→8稳定单调反例停止；未拟合、未冻结、4留出未采，不搜索新形式；本轮收口 |
| 阶段2 A本轮收口 | [计算A登记](#stage-2-compute-a)、[COMPUTE-01](experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md) | [开发及留出](experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#a实测与长度留出)18进程同卡合格，冻结后的32 Ktile窗口长度外推通过原5%标准 | [评分](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-heldout-001/score.json)保留两解释重合不可区分；769 cycle固定项不接GEMM、256提前量未辨识；B/C已按原登记完成开发筛查并停止具体形式 |
| 阶段3独立离线交付 | [同步登记](#stage-3-sync)、[SYNC-01](experiments/gh200_sm90/Q04-sync-slot/SYNC-01-replay.md) | phase／计数、条件返回与槽反压接口已接通；[回放包](../../../results/gh200_resource_campaign/access_rules/20261017-SYNC-01-replay-001/README.md)独立核对旧46条件／138进程／4416轮，21项CPU回归通过 | B/C零额外控制具体形式已停止，D实际路径已观测；独立返回价格尚未辨识，旧值不新增到GEMM参数 |
| 阶段4独立离线交付 | [异步搬运登记](#stage-4-transfer)、[TRANSFER-01](experiments/gh200_sm90/Q02-tma-request/TRANSFER-01-replay.md) | 源／目的接口与路径约束修复已接通；[回放包](../../../results/gh200_resource_campaign/access_rules/20261017-TRANSFER-01-replay-001/README.md)保存三个旧证据族及63项CPU检查，默认六图和两异步轨迹保持 | D原生三请求的开发数值／路径核对完成，无正式预测／留出，独立供给价格仍未辨识 |
| 阶段5独立离线交付 | [HBM登记](#stage-5-hbm)、[HBM-01](experiments/gh200_sm90/Q03-source-service/HBM-01-replay.md) | 共享外存假设与读miss替换接口已实现；[回放包](../../../results/gh200_resource_campaign/access_rules/20261017-HBM-01-replay-001/README.md)核对旧18条件／180进程与两种纯方向外推；25项CPU检查通过，显式写回与读共用服务对象 | 真实物理流量、写回时点及TMA迁移待测；下一独立阶段为SMEM |
| 阶段6独立离线交付 | [SMEM登记](#stage-6-smem)、[SMEM-01](experiments/gh200_sm90/Q07-smem/SMEM-01-replay.md) | [回放包](../../../results/gh200_resource_campaign/access_rules/20261017-SMEM-01-replay-001/README.md)保存完整lane／word输入、S4/S6实际STS.64两槽地址与写块替换；4096地址零差异、19项CPU检查通过，768份旧记录保留原失败 | 真实写块周期与联合迁移待测；下一独立阶段为寄存器 |
| 阶段7独立离线交付 | [寄存器登记](#stage-7-register)、[REG-01](experiments/gh200_sm90/Q08-register-residency/REG-01-replay.md) | [回放包](../../../results/gh200_resource_campaign/access_rules/20261017-REG-01-replay-001/README.md)核对539份旧记录与实际资源；逐warp CTA池及可选启动依赖已接通，21项CPU检查通过，六图原端点保持 | 申请时长与实际消费者迁移待测；下一独立阶段为L1／LSU |
| 阶段8独立离线交付 | [L1／LSU登记](#stage-8-l1-lsu)、[L1-01](experiments/gh200_sm90/Q13-l1-lsu/L1-01-replay.md) | [交付包](../../../results/gh200_resource_campaign/access_rules/20261017-L1-01-replay-001/README.md)核对117份旧记录与实际普通访存分支；12开发／8留出探针准备及本地静态编译完成，30项CPU检查通过 | cfg_b有效普通global／local需求为0，常量控制保留原覆盖；原开发实测与独立评分完成，8留出未启；仅为条件响应筛查，无新价格或独立C2/C3 |
| 阶段9独立离线交付 | [发射／驻留登记](#stage-9-sm)、[SM-01](experiments/gh200_sm90/Q08-register-residency/SM-01-replay.md) | [交付包](../../../results/gh200_resource_campaign/access_rules/20261017-SM-01-replay-001/README.md)回放267份旧记录、复用174份R12；S4/S6容量、36份完整合同及可选活动报告已接通，29项CPU检查通过 | E真实三输出trace端点已观测，尚无可迁移相位价格；启动分布待测，原时间保持 |
| 阶段10离线交付 | [同版组合登记](#stage-10-composition)、[COMPOSITION-01](experiments/gh200_sm90/Q14-composition/COMPOSITION-01-replay.md) | [交付包](../../../results/gh200_resource_campaign/access_rules/20261017-cfg-b-system-001/README.md)保存运行前输入到plain的统一入口、三条件原端点精确回归及原12评分；67项CPU检查通过 | 后续L2及A–E有限实测已收口，尚无新匹配价格替换；原12数值、失败及独立C2/C3缺口保留 |
| 最终组合依赖 | [cfg_b 组合](model/mechanism/composition.md) | 程序图、V09 聚合复现和请求级接线已完成。原12条件同版回放全部完成，plain绝对误差中位17.378%、最大39.797%；[最终比较](../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-full-matrix-001/comparison-final.md)显示A16、B80/B112的主循环和行距响应仍失配，短条件另有前缀缺口；G6时间误差+1.880%，空间跨度仍失配。固定32 B服务/128 B信用候选的三个容量均未通过旧必过条件，[筛查已停止](../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-dual-granule-service-001/RESULTS.md)；固定P0接入后原12条件plain绝对误差为中位10.225%、最大25.422%，[前缀轮次已完成](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-prefix-001/RESULTS.md)。本版独立组合验收尚未完成 | 各阶段保留规则或显式工作假设；第 10 阶段接入同版模型，交付完整时间线、plain 时间、已知反例和原阈值评分 |
| 异步搬运阶段的既有证据 | [多 CTA TMA](model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md) | 已有局部两槽与源状态证据；保留迁移失败。上一组固定候选已淘汰，原12条件报告已完成；用户允许的[两个新供给版本](model/mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md#bounded-supply-candidates)已实现；按请求大小分道、分别在源就绪或最后目的完成归还行信用的两版均在首项旧约束失败，[本轮已停止](../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-size-lanes-service-001/RESULTS.md)，未接回GEMM。保留现有组合与未解决的供给缺口，已停止轮次不追加候选 | 旧候选[顺序与上限](model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md#bounded-supply-round)保持；[独立对话](experiments/gh200_sm90/access_rules/EXECUTION.md#research-chat)已完成离线判别设计，尚缺可用服务形式和可观测差异下界。原 GPU 判别 1/2 轮只属于旧具体候选；新阶段单独登记条件与成本，不恢复旧窗口迭代 |
| 组合内的时间输入 | [有效频率](model/mechanism/hardware_rules/FREQUENCY-MECHANISM.md) | 使用已交付预测器及支持范围；独立状态方程和前序状态扫描暂停 | 在组合报告中分项评分，本阶段不另开课题 |
| 暂缓，保留对照 | [C1-01](experiments/gh200_sm90/Q16-config-selection/C1-01-selection.md) | 使用已有合格结果对照组合；暂停剩余测量、模型搜索和独立留出续跑 | 恢复由用户决定；组合推进使用现有基线 |
| 暂缓 | [S16](model/mechanism/hardware_rules/TMA-S16-STAGE-COST.md)独立收尾、暖机异常扩展、其他配置与前端、通用工具扩展 | 保留现有证据、准备包和已完成实现；九阶段内探针、服务与组合所需工具按有限登记推进，其余不恢复旧“下一步” | 重启条件统一见 GOALS 当前阶段限制 |

<a id="stage-1-l2"></a>

第 1 阶段有限登记（2026-10-17）：独立 `l2_saturation.cu`／`l2_saturation.py` 测热数据路径随发出并发及活跃 CTA 条件的饱和响应，协议与执行预算见[L2-01](experiments/gh200_sm90/Q03-source-service/L2-01-saturation.md)。固定每数组 16.5 MiB、128 线程、16 次迭代，独立加载批量 `J` 取 1、4、8；实际驻留和 SM 分布由运行时 SMID 核对，不将 CTA 数直接当作 SM 数。只实现这一探针结构，不沿用 EXP-13 的 checksum／threadfence 窗口作为裸 L2 带宽。

| 分组 | 模式 | active 参数 | J | 条件数 |
|---|---|---|---|---|
| 开发 | read | 1、8、32、132 | 1、4、8 全配 | 12 |
| 开发 | write、mixed | 32、132 | 4 | 4 |
| 留出 | read | 16、66 | 4、8 全配 | 4 |

每条件固定 3 个独立进程，开发 48 个、留出 12 个进程目标，共 60 个，另有 L2-01 登记的两个先导，总上限 62 进程；首轮 GPU 分配上限 10 分钟，实现修复最多两次，暖机与重跑边界以 L2-01 为准。源／目的数组各自保持固定容量。核对工作量、缓存准备、计时起止、写完成边界与实际生成指令；条件吞吐以逻辑字节计，未获得对应物理流量前只声明路径服务。开发用于形成条件曲线及解释并发受限与吞吐平台，留出不参与选择参数或修改判定；正式测量前冻结预测、观察量、判定与采样身份，不能按留出结果扩点。准备阶段 GPU 作业数为 0；用户已允许在免费ROMEO按上述预算执行L2，之后推进计算A。实际测量不足时登记最小证据缺口和工作假设，保留本阶段未完成的观测。

本轮已结束：两个先导及[开发48进程](../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-execution-002/dev/summary.json)全部合格，[回传318文件／231098176 B](../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-execution-002/transport-verified.json)全SHA验证通过。S32下J4窗口353856–354592 ns、J8窗口355328–355552 ns，中位窗口约0.253%逆向，按原单调规则使[六形式全部停止](../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-fit-001/service.json)；未拟合参数／误差，`selected=null`，未生成冻结、未采四留出。保留完整数据与未辨识的共享平台，不将小幅逆向归因某个物理资源；不扩模型搜索，按既定队列继续计算A。

<a id="stage-2-compute-a"></a>

第 2 阶段 A 驻留短链的离线准备登记（2026-10-17）：阶段1采样等待期间独立推进，GPU作业为0。只实现 cfg_b 实际 FP16→FP32、128×128×64 Ktile的单CTA／单128线程WG计算体；固定六槽预驻留环（196608 B），A K-major、B MN-major、SW128。每 Ktile 按四个K16偏移、各两个不同M64子块执行8条WGMMA，首Ktile的各累加片段第一次使用ScaleOut::Zero；首组commit、后续commit→wait1、末尾wait0。只做4／13／32 Ktile，plain与最后Ktile的单点trace配对；trace保存发出前、commit后、wait1返回和尾wait0返回，结果写出在排空后。

本轮竞争解释固定为共同开销加名义512 cycle/Ktile、共同开销加待辨识条件循环服务q两种；复用现有计算递推生成工作假设预测，不从旧长循环导出短链价格。先核对三种长度的工作量、M/K／六槽覆盖、完整非均匀输出参考、group依赖，再初次本地编译与SASS核对，最多两次定向修复。交付一版派生探针、三个长度的静态程序与已有假设预测、源码快照、CPU检查和编译证据；本轮不拟合、不执行GPU，实际计算服务仍等待采样。4／13为开发，32为冻结后的留出；后续采样限6个plain／trace条件、每条件3进程，开发12、留出6个，共18个进程，先导只复用其中一项；暖机和目标构建按实验页登记，后续免费ROMEO采样已获用户许可，在L2后按原预算执行。B–E、S4/S6配对、其他形状／精度／wait组织不在本轮展开。

A实测本轮已结束：[开发12进程](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-execution-002/dev/summary.json)及[冻结留出6进程](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-execution-002/heldout/summary.json)同卡合格，完整数值与目标构建已核对。[冻结](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-fit-001/dev-freeze.json)的两个解释完全重合，[独立长度评分](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-heldout-001/score.json)通过原5%窗口标准；具体数据及trace资格统一见[实验结果](experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#a实测与长度留出)。保留匹配驻留体的条件循环量证据，不把A固定项或trace段差值新增到GEMM价格；B/C后续开发筛查与停止结果见下文。

<a id="compute-bc-preparation"></a>

B/C匹配短链的CPU准备轮（2026-10-17）：复用A的实际8条WGMMA计算体，增加一个producer WG与full(1)／empty(128)的普通arrival协议；固定物理六槽196608 B，逻辑S4／S6。首S槽在窗口前就绪，后续producer等待empty后发布下一代；B为正常推进，C只在第一个复用组`tile=stages`取得empty后延迟显式D个clock64周期。复用后的B不预设每次都已就绪，按信号与等待区间保留实际路径。普通wait1释放较早组，tail wait0释放末组；窗口内保持各参与者独立推进。

只准备S6的4 Ktile B、13／32 Ktile B/C及S4的13 Ktile B/C，各plain／trace，共14个条件形状；每条件后续最多3进程，合计42进程，其中32 Ktile的12进程留出。CPU示例D=2048只用于参数与事件核对，正式D及数值预测在A开发观测后冻结，本轮GPU作业为0。固定驻留数据、计算顺序、提交组织与源布局；不接TMA、真实输出或新服务价格。交付一版派生探针、准备／构建／离线校验入口、完整输出与槽代次参考、逐端点schema及本地SASS；初次编译及最多两次定向修复，新增资源单独记录。已有A与组合回归不重跑；D/E、定价、实机快／慢路径辨识及独立C2/C3另待对应依赖。

B/C实测轮登记（2026-10-17，A留出收口后）：沿上述14形状与42进程清单，正式`D_delay=8q=4096 cycle`，一次作用于首次复用；先导复用S6／4／B／plain／seed3。固定A已测条件循环量q512、原commit_lead256，completion_lag／wait与full、empty额外控制服务为0的工作假设，A的769固定项不迁移。用既有lifecycle的producer节点与E/C/W递推冻结相对首次计算的端点，预测包见[有限预测](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-02-prediction-001/prediction.json)。主判别为plain窗口C−B：13 Ktile下S6预测1792、S4预测2816 cycle，跨stage响应差1024；各差值的绝对预测误差≤预测幅度5%为本轮相容标准。trace另核同seed窗口扰动≤5%，超限时只解释trace程序。B共同截距不拟合，完整begin→end只报告实测；trace核对relative端点、实际ready／blocked和最后必要release区间。

开发30进程通过完整数值、目标构建、性能资格与原冻结响应筛查后，才运行32的12留出；32的响应与端点在开发前一并冻结，留出评分不重拟合。规则不通过即保留反例、结束该具体形式，D的已登记独立路径继续；不追加候选或条件。免费ROMEO目标CUDA12.9，单卡、单CTA；累计分配上限10分钟（开发最多8、留出最多2），实现定向修复最多两次。只采未执行条件，原A／L2／组合不重跑；执行与失败收据存[新执行包](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-02-execution-001/)，不写GPFS历史目录。

B/C本轮已结束：[独立评分](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-02-score-001/score.json)核对开发完整资格，但三项冻结plain响应均未通过；按原规则停止q512／lead256加零额外控制的具体形式，12留出未执行，不增加拟合或候选。trace扰动全部超限，端点仅解释trace程序；具体数字统一见[实验结果](experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#bc-measured)。

<a id="compute-d-preparation"></a>

D原生TMA短链的CPU准备轮（2026-10-17）：保留B/C的256线程、物理六槽196608 B、逻辑S4／S6及消费者8条WGMMA／commit／wait1／tail wait0／128次empty arrival，producer改用实际A16 KiB与B0/B1各8 KiB的三次TMA，full登记一次arrival与32768 B事务。采用CUTLASS 3.9.2 StageCount6工厂的描述符与SMEM布局，逐98304个half坐标核对与B/C布局一致。global性能输入按原槽公式初始化，源`tile%stages`；另一完整正确性调用沿独立source tile使用单调正dyadic输入，检查真实重新填槽。

唯一主窗口从首个producer acquire／expect_tx／TMA之前开始，到tail wait0、末组释放、结果消费与CTA会合后结束；全部Ktile的TMA都在窗口内，4 Ktile短链也包含初始供给。缓存协议固定为重复调用暖机，不命名为纯L2命中；不增加预填窗口版本，不叠加P0。trace从同次记录另报首等待返回／首计算前缀及各组发出、等待与释放；请求发出括号与full完成上界分别保留。

只准备S6的4／13／32 Ktile及S4的13 Ktile，各plain／trace，共8形状；每条件后续最多3进程，开发18、32 Ktile留出6，总上限24进程。每进程先做1／2／`stages+1`与目标长度的代次非均匀完整正确性调用，再按原8–60次暖机协议采性能；不扩点、不定价、不拟合。本轮GPU作业0、初次编译及最多两次定向修复；交付一版派生源、CPU布局查询、完整整数参考、请求／事务／槽代次见证、记录校验与静态SASS，正式描述符编码身份、缓存状态及数值／时间／迁移等待实机证据。B/C和组合未变化，不重跑原回归；E另行登记。

本轮已完成：[D准备包](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-03-tma-preparation-001/README.md)通过15项CPU检查；工厂与B/C的98304个SMEM half地址零差异，本地SASS为每组8条HGMMA与三条TMA循环体。原失败构建和两次定向修复已保存；准备轮当时未有正式descriptor与GPU数值／时间，后续开发结果见下文。

D实测开发轮登记（2026-10-17，B/C原零额外成本形式筛查停止后）：原生TMA路径独立继续，仅执行原清单18开发进程，先导复用S6／4／plain／seed3；32的6留出不采。源、程序与数值协议沿原准备包，目标CUDA12.9的[薄执行包](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-03-execution-001/README.md)先审核三请求／8条MMA／wait1及尾wait0／资源和布局，再在同A卡执行。新增观测为实际descriptor、重新填槽正确性、plain周期／ns与逐组trace及其扰动；本轮为开发诊断，不拟合供给参数，`model_prediction=null`。先区分完整数值与路径资格，再报告与B的同条件整体程序响应；B/C被否定的时间连接不用于D的独立验收。累计GPU分配上限8分钟，目标实现定向修复最多两次；回传只含新证据及全文件SHA，冻结source／plan不重复回传。原A、B/C与组合不重跑，不加条件或候选。

D本轮开发诊断已结束：[独立评分](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-03-score-001/score.json)核对全部开发进程、ring与独立代次数值、实际descriptor源字段及opaque实例身份。trace扰动全部超限，无正式时间预测或拟合，6留出未执行；[实验结果](experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#d-measured)保留完整程序响应，不将D−B拆成供给价格。

<a id="compute-e-preparation"></a>

E真实pingpong输出的CPU准备轮（2026-10-17）：薄封装原CUTLASS 3.9.2 cfg_b工厂，固定S6、M384×N128×K832、调度SM数1，单CTA处理3个输出tile，每输出13 Ktile。使用原384线程、两个数学WG、`OrderedSequenceBarrier<2,2>`及真实64×32 FP32 epilogue；覆盖WG0→WG1→WG0。消费者保留`wait→mma→arrive→mma_tail→wait→store→store_tail→arrive`顺序；输入绝对组号`g=13j+k`连续推进，第二输出从slot1／generation2开始。实际资源由构建与运行分别记录。

只准备plain／trace两形状，后续各3开发进程，共6进程，不增加S4、长度或延迟扫描。输入采用已有period17非均匀dyadic，检查3×128×128全部49152个输出及分配首尾sentinel；使用独立整数点积参考。GPU event计完整kernel窗口，trace另报FIRST、mma返回、数学交接、MAIN_END、epilogue许可、store返回、store_tail与输出交接的cycle/ns；源读取等待边界与stream完成后的全局输出检查分别保存。E包含真实启动、调度和13组global源，与D六源环不同，差值按整体程序响应处理。暖机沿8–60次、末5次CV≤2%协议。

本轮GPU作业0，不拟合、不加服务价格；初次本地编译及最多两次定向修复。交付最薄派生源、固定工作列表、39输入组／117逻辑输入请求／312条WGMMA／24输出子块、跨输出槽与order相位见证、完整数值参考、记录校验、静态SASS及复现包。复用现有工作列表与输出地址函数，不另建事件求解器；当前A–D及组合未变，不重跑原条件。正式服务预测、输出重叠／时序与迁移判定等待A–D开发观测，独立留出以后另行登记。

本轮已完成：[E准备包](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-04-pingpong-preparation-001/README.md)通过14项CPU检查，真实工厂与两种本地SASS已核对；准备轮当时未有完整global输出与八端点的GPU观测。A–E短链首批CPU准备已齐；后续实测沿各轮有限登记执行。

E实测开发轮登记（2026-10-17，在D开发诊断收口后）：沿原S6／M384×N128×K832／调度SM数1与三输出条件，只执行plain／trace各3进程，共6开发进程；先导复用plain／seed3。使用原工厂与overlay、完整49152输出及32个guard、原暖机和8端点校验。目标CUDA12.9先核资源、WGMMA／输入与输出组织及overlay身份，再在同A卡采样。输出为完整kernel时间、三个输出的mma返回／MAIN_END／输出交接与观察扰动；trace同seed窗口扰动≤5%时讨论plain端点迁移，超限保留trace程序结论。本轮无正式完整时间预测，不拟合、不留出；用A–D已有开发证据定位向真实pingpong迁移的最小缺口。累计GPU分配上限8分钟，目标实现定向修复最多两次；薄执行包见[COMPUTE-04](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-04-execution-001/README.md)。原A–D、原12与G6不重跑，不增加配置或候选。

E本轮开发诊断已结束：[独立评分](../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-04-score-001/score.json)核对真实工厂、完整输出／guard及三输出工作列表。观察扰动不满足向plain迁移端点的原条件；[实验结果](experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#e-measured)保留数学交接与尾部完成的真实软件端点，无正式完整时间预测、拟合或留出验收，原组合价格保持。

<a id="stage-3-sync"></a>

第 3 阶段同步／释放的离线登记（2026-10-17）：先复用三轮barrier_return的46条件×3进程，不新增独立barrier扫描。固定cfg_b的full到达数1、事务32768 B、empty到达数128，以及producer初phase1／consumer初phase0、wait1释放较早组和尾wait0释放末组。交付可执行phase／计数生命周期、请求级组合的条件等待返回接口、旧记录的独立重算与条件区间；不从旧R13观察段导出通用价格。

有限实现为一个同步服务模块与一个旧证据回放入口。先检验完整128次arrival、事务计数、三代phase、跨输出槽复用与因果顺序，再用4／13／32 Ktile×S4／S6六个小程序检查接口反压、默认结果及显式零成本退化。工作假设为`wait_return=max(enter,ready)+r_path`，path按进入时已就绪或尚未就绪区分；r_fast/r_blocked由调用方显式给定，当前GEMM不新增校准价格。计算、供给、频率和输出规则固定，释放arrival额外耗时仍沿原假设。三轮138份旧raw只独立回放一次并与原分析核对，不拟合／搜索、不运行GPU；实现最多两次定向修复。原数据的调用形式／程序上下文分别绑定，保留不能区分的完成—返回区间。后续B/C和D已取得实际短链路径观测；B/C零额外控制具体形式停止，独立返回价格与向GEMM迁移仍未辨识。

同步稀疏补测登记（2026-10-17，用户授权继续未完成机制实测）：针对B/C逐组trace扰动29%–42%的缺口，只改变观测组织，runtime槽索引、generation检查、驻留计算与full(1)／empty(128)协议固定。首批S6／13 Ktile，首次复用取得empty后的延迟D0／8192 cycle，各plain／稀疏、seed3／22／41，共12开发进程；全部数值、代次、资源、暖机合格且每seed配对的cycle与ns扰动绝对值均≤5%，才补同样S4的12进程。循环内仅记录full组0／S、producer取得组S／S+1及128人释放组0／1的clock64括号，归并与导出移到计时结束后。预填full为已就绪对照，D8192用于制造等待入口早于信号的条件，实际路径按信号与入口区间判定；依赖顺序、完整输出和5%观测扰动门槛先冻结。本轮为开发观测，不拟合绝对返回价格，不启留出，不恢复已淘汰的时间形式；计算、供给、频率、输出及原12组合保持。使用免费ROMEO，本轮配对固定同一张GH200；前两次分配因指定UUID约束停止，第三次因SASS审查器未去尾空白误判停止，均未执行探针，原启动保留；本轮按实际分配身份绑定配对，各卡数据分开统计；累计分配上限8分钟，最多两次目标实现定向修复，超限或资格失败即收口；执行身份、原始数据和独立重算保存于[稀疏观测包](../../../results/gh200_resource_campaign/access_rules/20261017-SYNC-02-sparse-004/README.md)。

同步稀疏本轮已收口：[独立评分](../../../results/gh200_resource_campaign/access_rules/20261017-SYNC-02-sparse-004/score.json)核对12个S6开发进程全部数值及性能资格，但6个配对的cycle与ns扰动均超过原5%，按登记未扩S4。新端点只描述稀疏程序，不新增返回／释放价格，原12条件组合与原失败保持；下一项缺口为匹配实际程序控制组织后，取得可迁移的同步／释放连接约束。

<a id="stage-4-transfer"></a>

第4阶段异步搬运的离线登记（2026-10-17）：复用现有tensor服务、异步源回调及目的投递，给cfg_b组合提供独立服务参数；源完成、目的就绪、等待返回、信用归还与软件槽释放分别保留。现有源完成归还／最后目的完成归还只作两条可执行工作假设的接口对照，不新增价格、容量或仲裁候选，不恢复旧淘汰形式。固定计算、输出、频率和P0规则；L2整窗读服务仍等待请求级迁移证据。

先修复已复现的外接源跳过路径延迟约束，再用4／13／32 Ktile×S4／S6六个小程序核对三请求、跨输出代次、default／零价退化与显式源／目的服务组合；单信用的两种归还轨迹只用于CPU接线检查。旧证据限首批tensor的12条件／168进程、S16已发布的12个匹配坐标和cp.async的12个匹配坐标，重算原量与端点，不重新拟合或执行S16全量payload审查。每个版本、每个条件正式回放一次；最多两次定向实现修复，不启动GPU。交付保存源码、分项时间线、原评分／发布资格、默认回归及剩余缺口；不能迁移的价格留在原证据。后续D联合短链开发数值／路径已完成，未产生独立供给价格或正式留出验收；当前组合接口与原参数保持。

<a id="stage-5-hbm"></a>

第5阶段HBM的离线登记（2026-10-17）：复用EXP-13的9个large条件（每条件10个历史进程），另以9个small条件作同协议参照；核对原C审查／COMPLETE身份、原始字节与窗口、精确统计及发布数值。纯方向参数固定取同工作集的read.cg／write均值，检验无额外混合项的独立方向`max(Qr/Br,Qw/Bw)`与共享串行`Qr/Br+Qw/Bw`两种完整窗口外推；失败形式保存反例，不扫描混合修正项。由均值速率取倒数得到的量明确记作等效窗口。

交付一个参数由调用方显式提供的共享外存服务假设，全部CTA的读miss／显式写回按可发出事件进入同一对象。只替换状态化源已有的HBM前沿，不改缓存映射／替换、home返回、TMA前端、计算、同步、频率、输出与P0；缓存store保留原行为，写回由调用方另行声明。先用读写竞争、在途合并／命中和128 B线计账的小例子，再做4／13／32 Ktile×S4／S6六个组合接线与默认回归。旧证据回放一次、两种外推各一次，不拟合、不开GPU，最多两次定向实现修复。交付源码、输入／端点／资格、旧证据与剩余缺口；真实miss字节、写回时点、并发平台及global窗口到TMA的迁移等待新观测，下一独立阶段为SMEM。

<a id="stage-6-smem"></a>

第6阶段SMEM的离线登记（2026-10-17）：复用m11两次冻结留出的126条件／378进程及lane对照48条件／144进程，保持原失败和判定；另核R03的18条件／72份记录、R12的24正式条件与18阶段／空控制，按原窗口回放，不新增拟合或需求候选。普通LDS/STS需求保留每次访问的lane、word、bank、同址关系和依赖组织，旧标量及v1/v2规则只按原资格重算。

以冻结cfg_b实际SASS为起点：输出为64条STS.64，每个8 KiB子块由128线程各8次8 B写组成；继续核对lane地址及两输出源槽。交付程序SMEM需求与覆盖关系、调用方显式给价的输出写块接法；输入TMA目的、WGMMA隐式取数、普通输出STS和TMA输出源读取分别登记，已含在复合服务中的操作不另加价格。先核对需求／别名／字节守恒与SASS，再做4／13／32 Ktile×S4／S6六图的默认退化、写块价格响应及槽复用检查。每个旧证据族重放一次，最多两次定向实现修复；不启动GPU、不恢复已淘汰的m11形式搜索。保存源码、完整输入与有限结论、旧评分和最小实机缺口，下一独立阶段为寄存器。

<a id="stage-7-register"></a>

第7阶段寄存器的离线登记（2026-10-17）：固定cfg_b S4/S6的384线程、初始168 reg/thread及producer40／两个consumer232配额，核对128个FP32累加器／consumer线程、无spill及启动时的释放／申请依赖。程序图可选展开4个producer warp释放和8个consumer warp申请成功，CTA池按初始64512个32-bit寄存器守恒；后续输出与Ktile复用配额。软件依赖按本warp释放／本WG申请完成表达，资源先后由CTA池判断。先检验consumer先申请受阻、一个producer warp释放4096后两个consumer warp可各取得2048，再检查4／13／32 Ktile×S4／S6六图的依赖、默认结果与启用后的原端点相等。现有ENTRY→FIRST_WORK已包含producer warp0释放，其余warp释放和consumer申请时长待测；时间价格固定，不将旧R16窗口新增到前缀或计算服务。

旧证据限R16六条件／66进程，R02 v8的32服务条件与当前382记录（原380加两条进位正确性控制），R02固定帧82进程，R01正式FFMA三个长度各3进程，共539份记录。逐一核对源码／二进制／SASS身份、数值见证、原窗口与原评分；每证据块正式回放一次，最多两次定向实现修复，不重新拟合、不新增RF bank／端口候选、不运行GPU。竞争解释沿寄存器服务页：分配数量已不足以解释固定帧差异，源字段／reuse／排程与RF服务仍待区分。交付绑定原条件的服务等价类、可执行配额生命周期、实际程序需求和可复现包；真实申请时长及实际输出消费者的匹配观测另列，下一独立阶段为L1／LSU。

<a id="stage-8-l1-lsu"></a>

第8阶段L1／普通LSU的有限登记（2026-10-17）：先核对冻结cfg_b S4/S6主kernel的普通global／local／常量指令、分支与实际参数。固定r18的alpha=1、beta=0及空alpha指针；两处静态scalar LDG的动态需求由源码与分支推导为0，A/B及D仍归TMA，参数LDC／ULDC按实际程序位置保留现有复合服务覆盖。只增加可选需求报告，4／13／32 Ktile×S4／S6六图的原时间线与端点保持。

旧证据限R01普通global的10条件／37进程、EXP-13的small／large read.ca／read.cg四条件／40进程、EXP-18 local／spill32／128 word四条件／40进程，共18条件／117进程。逐一核对原身份、同体cache对照、数值见证与完整窗口，回放一次，不拟合。新探针固定单CTA×128线程、16 B向量节点、J=1依赖链或J=4分开的依赖链、每次kernel内先遍历同一私有工作集，再计时256次完整遍历；两种cache policy仅改.ca／.cg，preferred carveout只取0／100，记录请求值及构建资源。开发12条件：16／256 KiB、hint100、两policy×两J的8点，加64 KiB、J4、两hint×两policy的4点；留出128 KiB、两hint×两policy×两J共8点。后续每条件3独立进程，共60；首轮分配上限10分钟，先导复用开发点，最多两次定向实现修复。该准备轮仅做CPU准备与本地编译，GPU作业为0。

两项待判别响应固定为.ca与.cg随工作集／carveout的差别、J增加对依赖返回受限窗口的改善；原准备预案拟在先导后冻结幅度预测；现行开发轮改为下文登记的描述响应筛查，不按先导编造幅度。交付实际需求、绑定原窗口的条件证据、探针源码／校验／静态编译及复现包；未获得新观测前不导出L1命中率或通用load价格。local／spill窗口保留独立来源；原子、地址翻译及多SM放置本轮不展开。实际L1响应与生成指令已按下文有限开发轮核对；8留出未启。

L1本轮开发实测登记（2026-10-17）：仅采原12开发形状×seed3／22／41，共36进程；8留出不启。先导`w16_h100_cg_j1`／seed3复用其中一项。目标CUDA12.9先核对同体.ca／.cg及J链组织、实际资源与单CTA，再在同A的GPU UUID执行；每进程沿原协议核768个checksum、SMID、暖机与计时资格。主指标为clock64内窗，globaltimer伴随；跨W使用`cycle/(256W)`，CUDA event保留完整kernel边界。同点3进程范围一并报告，固定5%相对内窗变化仅作为描述响应门槛。两项假设方向筛查不从旧117份观测或先导编造绝对幅度／置信界，不称C2数值判别或独立验收。累计分配上限600秒，含目标构建及开发；实现定向修复最多两次，不追加条件／候选、不拟合。结果与执行入口维护在[L1-01](experiments/gh200_sm90/Q13-l1-lsu/L1-01-replay.md)。

L1本轮已结束：[独立评分](../../../results/gh200_resource_campaign/access_rules/20261017-L1-01-score-001/score.json)核对原开发全部进程、完整checksum、目标静态审查及响应，与原摘要一致；条件响应和同点范围统一见[L1-01结果](experiments/gh200_sm90/Q13-l1-lsu/L1-01-replay.md#2-新探针条件与观察量)。无绝对时间预测或拟合，实际carveout仍未知，8留出未采；本轮不新增独立C2/C3或组合价格。

<a id="stage-9-sm"></a>

第9阶段SM发射／驻留的有限登记（2026-10-17）：固定cfg_b S4/S6的384线程、168 reg/thread及完整SMEM149504／215040 B，复用阶段7配额与阶段8构建审查；以显式GH200设备条件核算寄存器、SMEM及线程容量，并对照历史occupancy查询。另仅解码G3 S6/S4各一份dual-00的旧SMID与活动窗口，静态容量、持久工作映射和实际窗口分别报告，不新增放置或启动分布候选。

旧回放限R06展开版6条件的39正式＋12短记录、R16驻留12条件的120正式＋24短记录，以及R11 v6的FFMA＋IMAD.WIDE／WGMMA＋IMAD两组：12正式条件、36正式＋24短＋12角色控制，共267记录。R16配额已在阶段7回放，不重复。R12复用阶段6已保存的24正式／18控制条件及174记录的重算结果、身份与已知CV差异，不再解压全值。每个证据族回放一次；CPU回放预算20分钟，最多两次定向实现修复，不重新拟合、不恢复旧排除形式、不运行GPU。

竞争解释固定为occupancy倒数缩放与容量只约束合法并发、所有独立流一律max与绑定组织的条件联合窗口。保留原反例与原资格；交付只覆盖当前S4/S6的容量核对、完整合同的联合窗口服务、实际程序覆盖与同版活动报告。联合合同保存工作、角色、消费者、同步及SASS身份，返回外部完整窗口，不把差值拆成opcode罚值；cfg_b尚无匹配合同，当前默认独立推进与原价格固定。先核对容量算术、合同单位／覆盖／边界，再做4／13／32 Ktile×S4／S6六图的报告接线和原结果回归。最小迁移依赖是实际cfg_b相位配对与内部完成／释放端点；本轮完成后转最终同版组合，不新增CUDA探针。

<a id="stage-10-composition"></a>

第10阶段同版组合的有限登记（2026-10-17）：先贯穿G3_base、G3_stage4、G3_A16三个普通／stage／行距条件，固定原12矩阵的运行前case／setup、相对分配与放置假设、频率预测器、状态化源、计算、释放、输出、P0及plain映射。三条件各最多11次CPU调用，与已保存前缀版逐CTA端点、内部窗口及plain逐位比较；不重跑G6。其余九条件复用已核对同版的保存结果，全部原评分和失败保留。

交付一条可序列化的运行前输入→共同程序→请求级时间线→plain路径，九方向分别标出已核对需求、现用假设／经验替代、可选接口及未定价项。先修复已复现的状态化source／output bound method残留在report参数导致JSON导出失败，只增加现有回调的独立透传；不改服务规则。再做4／13／32 Ktile×S4／S6六图，联合开启配额、普通LSU、活动报告及零价同步，并检查等价SMEM替换与HBM读miss后端。真实参数未辨识的接口保持显式给价；新曲线和旧完整窗口不转换成逐请求价格。

竞争项只核对“可选接口组合后与原时间相同”及“显式替换覆盖旧价而非重复计费”，不增加硬件候选、拟合或阈值。上限一版接线入口、三项正式回放、原12评分复用、CPU20分钟、最多两次定向实现修复；不启动GPU、不重复未变化的旧数值回放。保存源码、输入身份、同条件比较、三类模型对照、模块覆盖与最小证据缺口。后续有限实测已取得A–E证据，但尚未完成真实服务价格迁移或全GEMM独立C2/C3验收。

第10阶段定向复核追加登记：首轮三项11调用的plain与内部窗口保持；G3_base CTA0的`g6.B0.issue`由3008.1879500193613变为3008.187950019361 ns，定位为零价同步在entry／producer原点之间加减前缀的浮点舍入。第二次定向修复保留未增加成本时的原端点值，并将入口声明的A/B/D基址完整透传；原12条件A=0不受后一项影响。不加浮点容差、不改服务价格。仅受影响的三条件各追加一次11调用回归，共33调用，仍在20分钟CPU预算；首轮源码及输出另存，其他九项不重算模型。

已结束的研究窗口：**2026-10-16T00:36:59+02:00 至 2026-10-16T08:36:59+02:00（Europe/Paris）**。下列轮次从主循环、stage、行距与 CTA 推进反例选择具体规则；窗口授权见 [GOALS §5.4](GOALS.md#current-scope)，启动记录见[窗口记录](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/window.json)，最终交付与剩余缺口见[收口状态](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/STATUS.md)。各轮保留原登记、筛查与回归结果。

窗口第1轮：共享域返回服务（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r01-domain-return/RESULTS.md)）。失败端点为固定源重排的 GPC4 Q1 release 响应，以及组合的 A16/B80/B112 主循环、stage 响应和 CTA 跨度。假设为已有两域映射内的在途返回工作可跨 GPC 合并，并受共同吞吐预算约束；相同全局源集合在重排后，两域独立源由16/16增为31/30，支持检验接收域需求。唯一候选保留原每GPC两槽与整atom在途合并、CTA目的FIFO和已有价格，增加每域8槽的返回服务（4个GPC×原2槽，作为组合假设）；atom工作为实际32B覆盖数/512，域内同源只在服务结束前合并，就绪取原服务与返回服务的较晚者。可改共享范围与返回工作计量；固定程序、控制、消费者、计算/释放、缓存参数、频率、输出、P0与plain映射，不拟合新价格。先跑`prior-q1-skew8`，通过原ns/cycle 5%门槛才补其余五项；再跑`permutation-grouped-q1`/`permutation-mixed-q1`并保留原十次交替暖机，GPC4响应沿用4个百分点开发相容性门槛。任一失败即结束具体形式；通过后才接入缓存与GEMM，按G3 S6/S4/A16、G5 base/B80/B112六项回归。上限为1候选、8个探针条件及6个GEMM条件，各一次正式CPU回放、每项最多11次调用；实现修复最多2次定向复核，不重复未变化版本。证据和结果写入[窗口交付](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/window.json)，GPU作业为0。

窗口第2轮：输入与背景输出共用GPC接入（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r02-output-admission/RESULTS.md)）。739740固定CTA0处理、只延后其他CTA输出时，S4早段10/10缩短而后段变长；外接输入服务后，当前输出FIFO与GPC输入服务分离。本轮唯一候选保留原地址缓存、GPC两槽输入服务和全局输出FIFO，输出源请求还须占本GPC最早可用槽：开始为请求、全局输出前沿和GPC槽前沿的最大值，按原输出价格结束，同时推进两条前沿；缓存写回仍在原输出源完成端点登记。相对第1轮，改变的是输入/输出双向接入依赖，域返回8槽候选不接入。可改该连接及对应接口；计算/释放、输入/输出价格、频率、P0和plain映射固定，不新增数值参数。先做G3 S4的defer1/2两项，再在S4 early缩短、post变长的方向均成立时补S6两项；最多4项CPU回放，各10次flag0暖机加1次目标。739740为另一参考卡与自适应暖机协议，本轮只将其方向与幅度作为开发迁移对照。方向失败即停止该形式；通过后才跑当前原12矩阵的G3 S6/S4、G3 A16、G5 base/B80/B112六项，固定原11调用。上限为1候选、4项结构干预和6项GEMM条件，修复最多2次定向复核。无输出的探针图结构不受此连接影响，复用原结果并保留747595重排失败，新增GPU作业为0。

窗口第3轮：共享域在途行信用（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r03-domain-credit/RESULTS.md)）。第1轮同等总吞吐的域返回只能改变约0.141个百分点，转查在途容量的共享范围。保留原GPC两槽整atom服务、原价与目的FIFO，增加按接收域计量的128B行信用；容量固定沿用4096作为组合假设，不扫描容量，不替换为sector FIFO或A/B硬分道。grouped与mixed的域源集合不同，须先按实际128B覆盖核对静态需求，再由事件计算同时在途需求。最多两版，分别在实际源数据就绪、该代最后目的完成归还信用；同域同物理行在该代service_end之前共用信用，超过截止创建新代次；完整覆盖的订阅保留该代截止与就绪时间，新增sector才由原GPC服务更新截止与就绪时间。新atom须获得所需新增信用后才进入原GPC服务；缺信用时FIFO等待，完全复用已有行者可直接订阅。固定控制、消费者、价格、计算/释放、缓存参数、频率、输出、P0和plain映射。每版先`prior-q1-skew8`，通过原ns/cycle 5%才补旧六项；再测grouped/mixed Q1原4个百分点相容性；失败版停止，幸存才接入缓存与GEMM。上限2候选、每版8探针条件，幸存版最多6项G3 S6/S4/A16、G5 base/B80/B112；各一次正式回放、每项最多11调用，暖机skew按原协议为0；实现修复每版最多2次定向复核。GPU作业为0。

窗口第4轮：整atom的32B覆盖计价（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r04-sector-weight/RESULTS.md)）。失败端点为A16与B80/B112的首/后L及plain行距响应。EXP-15的padding读支持检查实际sector工作；旧双粒度和大小分道同时改变了仲裁，本轮保留原GPC两槽整atom仲裁和service_end合并，只把新源generation的core工作改为该atom实际32B覆盖数/512。A16的A为576sector、B80/B112的每B为288sector，三atom总工作均由2变为2.125；B32仍为2。目的仍按逻辑payload投递，缓存128B路径、程序、价格、计算/释放、频率、输出、P0与plain映射固定，不叠加前3轮候选。先正式回放G3_A16，plain与首后L绝对误差均改善且行距响应为正才跑G5_B80/B112；否则停止。上限1候选、3项各11调用、实现修复最多2次定向复核。原12项中其余9项的sector工作与旧值逐位相同，保存结果直接复用；aligned旧六项及固定源重排的未变失败保留。独立C2/C3资格不变，GPU作业为0。具体输入与复现写入[本轮计划](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r04-sector-weight/plan.json)。

窗口第5轮：计算服务占用门控生产者（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r05-compute-issue-gate/RESULTS.md)）。失败端点为G3 S6/S4的第二MAIN跨度、首后L和stage响应。746720将本地增量定位到发出与补发区间；当前计算只经槽释放影响producer，旧每指令40/80cycle延迟未修复空间跨度。本轮只增加一条动态排他连接：issue落入本CTA的真实计算服务区间`[max(mma_start,前批service_frontier),当前service_frontier)`时，推迟到区间末端并重试；不把排队或完成滞后算为占用，不添加固定罚时。先跑G3_base/G3_stage4，第二MAIN跨度两stage均进入各自旧观测范围，且stage与首后L误差均改善才补A16及G5 base/B80/B112；否则停止。原程序、TMA源/缓存/目的、价格、512/256cycle计算与提交、220cycle复用、频率、输出、P0和plain映射固定，不叠加前4轮形式。上限1候选、6项各11调用、实现修复最多2次定向复核，GPU作业为0。资格为排他边界的开发假设，具体条件在[本轮计划](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r05-compute-issue-gate/plan.json)。

窗口第6轮：提交前活动门控（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r06-commit-issue-gate/RESULTS.md)）。第5轮完整TC服务排他高估时间，S4空间仍失败；746723移除A的SMEM读取未改变主要干扰，746785较慢位置随issuer移动，支持区分发出活动与异步计算占用。本轮只把producer排他区间改为程序`[mma_start,commit)`，提交后TC仍执行时允许发出；只检查已经登记区间，不新增周期参数，不叠加前4轮候选。固定其余程序、源/缓存/目的、价格、512/256cycle计算与提交、220cycle复用、频率、输出、P0和plain映射。先跑G3_base/G3_stage4，相对窗口开始版首后L与stage响应误差均改善、第二MAIN跨度两stage均进入原观测范围才补A16及G5 base/B80/B112；否则停止。上限1候选、6项各11调用、修复最多2次定向复核，GPU作业为0。具体输入与复现见[本轮计划](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r06-commit-issue-gate/plan.json)。

窗口第7轮：逐sector订阅截止（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r07-sector-subscription/RESULTS.md)）。第4轮实际覆盖改价改变了请求完成，但A16评分端点不变；旧source-offsets也显示共同ready强同步。保留GPC原两条最早空闲atom通道和原价，改为每个32B sector在自己的service_end关闭订阅；新atom按描述符行序为未复用sector预约连续批服务，完成取所需sector的max，完整复用不占新通道。该规则保留整atom仲裁，区别于旧单sector FIFO、固定大小分道及第4轮wholeatom截止。固定缓存128B路径、程序/放置、计算/释放、频率、输出、P0和plain映射，不叠加前轮候选。先跑G3 base/A16/stage4；A16 plain与首后L绝对误差均改善、行距响应更接近实测，且两stage主循环不劣化、第二MAIN跨度均改善才补G5 base/B32/B80/B112，否则停止。七项通过才做原12同版回归，已完成项复用；上限1候选、12项各11调用、修复最多2次定向复核。旧对齐探针同源cohort同步，逐sector规则解析退化原预测，旧六项复用、固定源重排失败保留。资格与输入见[本轮计划](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r07-sector-subscription/plan.json)，GPU作业为0。

窗口第8轮：整atom的128B tag覆盖计价（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r08-tag-weight/RESULTS.md)）。失败端点为A16、B80/B112的行距响应，并保留B32近零响应作约束。第4轮只按32B覆盖计价；本轮检验源服务工作是否由去重128B tag覆盖数决定，原两槽整atom仲裁、service_end订阅与τ不变。A16的A为240tag，B32的单B为112tag，B80/B112为120tag，工作取tag数/128；对齐请求仍为1或1/2。依据为实际地址覆盖及EXP-15 padding输入的覆盖响应；128B L2 promotion只用于核对DRAM填充属性，不作为hit服务价格的证明。固定缓存读写、逻辑目的工作、程序/放置、计算/释放、频率、输出、P0和plain映射，不叠加前7轮候选。先跑G3_A16，plain与首后L绝对误差均改善、行距响应更接近实测才跑G5_B32；B32相对G5_base的plain响应误差不劣于窗口开始版、首后L误差不劣化，才补G5_B80/B112。失败即结束该形式；四项通过后与其余8项已保存的逐位等价结果组合评分。上限1候选、4项各11调用、实现修复最多2次定向复核。旧对齐探针与stage响应逐位不变，保留旧重排及stage失败；资格为tag工作假设，GPU作业为0。

窗口第9轮：源服务后启动缓存访问（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r09-cache-after-core/RESULTS.md)）。失败端点为主循环、行距与CTA推进。旧cold-route的有效慢路延迟加在GPC服务结束之后，当前组合却在logical issue时访问缓存，再与排队后的core就绪取max；本轮检验这一连接是否过度隐藏了慢路。唯一候选在原core.service_end事件执行每个logical请求的cache read、LRU及HBM登记，命中时严格复用原core.source_ready，慢路只加相对hit的超出量；整atom订阅仍在service_end严格截止，价格、缓存容量/映射/填充规则、逻辑目的、程序/放置、计算/释放、频率、输出、P0和plain映射固定，不叠加前8轮候选。先核对纯热命中逐位等价及事件时序，再回放原746641的original/domain_local×last0/1四项、各11调用，仅保留原两域完全相联形式；GPC2/3原布局响应沿原−34.029%±12pp、新布局沿0±10pp筛查。通过才跑G3 base/A16，首后L与plain绝对误差不劣化且A16行距响应改善，再补S4检验stage与第二MAIN跨度；失败保存反例停止。三项通过后扩展G5 base/B32/B80/B112，再补原12同版回归。上限1候选、4探针条件和12 GEMM条件、各11调用，修复最多2次定向复核。旧冷路绝对时间失败与aligned重排失败保留，本轮资格为组合边界假设，GPU作业为0。

窗口第10轮：空闲槽上的CTA轮转仲裁（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r10-cta-round-robin/RESULTS.md)）。失败端点为固定源重排响应及GEMM空间推进；旧保序反例要求检查不同于同质FIFO的动态服务次序。唯一候选保留每GPC两槽与原τ/L/D，在槽实际空闲时从有待服务工作的CTA队首循环选择，空CTA跳过，初始cursor按CTA升序；请求不在logical issue时预订未来槽。整atom同源代次在排队及其实际service_end前共享，源job由首个请求CTA持有，订阅者仍独立投递。相对旧CyclicService，空窗可借用、无固定32KiB时窗和奇偶分道；相对前9轮，改变的是动态仲裁而非价格、覆盖或信用。固定程序、控制、计算/释放、缓存参数、频率、输出、P0和plain映射，GEMM缓存仍从logical issue访问，不叠加第9轮。先原六项prior，沿原ns/cycle 5%逐项筛查，再grouped/mixed Q1沿原4pp响应门槛；失败结束具体形式。通过才G3 S6/S4/A16及G5 base/B80/B112，主循环与plain误差不劣化、stage/行距和第二MAIN跨度改善才补原12同版回归。上限1候选、8探针条件和12 GEMM条件、各最多11调用，修复最多2次定向复核；资格为未辨识仲裁的开发假设，GPU作业为0。

窗口第11轮：仅HBM miss占用的有限池（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r11-miss-only-pool/RESULTS.md)）。失败端点为A16/B80/B112的持续主循环缺口及空间推进；保存时间线显示这些行距的模型TC空闲中位仍为0。现有状态模型合并未完成HBM miss，却无限制地预约新miss；Sim-FA的miss tracking提供结构参照，本机容量尚无上下界。最多三个固定开发点K=2048/4096/8192，每个归属域各有K个128B miss槽；只有新的HBM miss占位，至原home_ready归还，同line订阅不新增或延长占用。命中、远端已缓存与逻辑目的不占槽；按原全局HBM发出FIFO预约，缺本域槽的队首可阻塞后续域的HBM发出。保留原cache put时点、GPC两槽/整atom分享、τ/L/D及hit/home/link/HBM价格，不叠加前10轮；计算/释放、频率、输出、P0与plain映射固定。各点先original/domain_local×last0/1四项原746641协议，沿第9轮所用原响应门槛；通过才G3 base/A16，首后L与plain绝对误差不劣化且A16响应改善再S4；S4的三项绝对误差及stage响应误差不劣于窗口开始版，才进入G5。未产生改善或违反旧约束的点停止，不扩扫K；幸存点才G5 base/B32/B80/B112及原12同版回归。上限3候选、各4探针和12 GEMM条件、各11调用，修复每点最多2次定向复核。K不是本机容量测量，旧冷路绝对时间与重排失败保留，GPU作业为0。

窗口第12轮：有信用请求优先的HBM发出（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r12-eligible-hbm/RESULTS.md)）。第11轮K2048使G3首L小幅改善，却把G5_base后L从19457.33推至20841.25ns，绝对误差由1.61%增至3.50%；该形式已停止。区别在于第11轮预订未来HBM前沿，会让一个满池域阻塞另一个有空槽的域。本轮只改为事件仲裁：新miss进入归属域FIFO，有空信用时立即准入并占槽，至home_ready归还；满池请求等待。HBM实际空闲时，从已准入的各域队首中选最早到达者，满池域未准入的请求不阻塞其他域。只有已经到达的miss参与选择，不预订未来服务；同line订阅不增槽。唯一容量沿用K2048，不新增价格。缓存仍在logical issue占位，未来ready由实际HBM发出求得；原GPC两槽整atom服务、缓存容量/映射/价格、逻辑目的、程序/放置、计算/释放、频率、输出、P0及plain映射固定，不叠加其他轮次。先用小图核对单域退化、跨域跳过、同行订阅、信用精确归还及跨调用状态；再以第11轮已保存的四探针零等待输入流证明与原模型等价，证明不足才各回放11调用。接着G3 base/A16，三项绝对误差不劣且A16响应改善才S4；S4三项绝对误差与stage响应不劣才G5 base，分项不劣才B32/B80/B112，七项通过后补原12同版回归。失败结束具体形式。上限1候选、4探针和12 GEMM条件、各11调用，修复最多2次定向复核；原冷路、重排失败及独立验收资格保留，GPU作业为0。

窗口第13轮：按需sector的远端副本（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r13-remote-sector-replica/RESULTS.md)）。失败端点为A16与B80/B112的行距主循环；第4/8轮覆盖计价未触发持续等待。实际描述符的128B promotion决定HBM填充，官方NCU的128B行/32B sector组织仍允许检查远端副本的有效范围。本轮唯一候选保持home的新HBM填充为完整128B，只让跨域返回在接收域登记本次请求覆盖的32B sector；再次请求同tag的缺失sector时，经原home/pending路径取回，命中、home、link与HBM价格不变。区别于已否决的“HBM只填请求sector”、第7轮core订阅截止和第9轮cache访问时点：本轮只改远端副本状态，容量仍按128B tag计，LRU仍logical issue更新。以实际描述符行序生成mask，不按case/stage加价；原GPC两槽/整atom分享、逻辑目的、程序/放置、计算/释放、频率、输出、P0和plain映射固定，不叠加第11/12轮。先核对home填充、跨域缺sector、pending订阅以及完整mask与原模型逐位等价；对齐的8个GEMM与旧探针可据等价证明复用，保留stage和重排旧失败。先G3_A16，plain与首后L绝对误差均改善且行距响应更接近实测，才G5_B32；B32响应及首后L不劣才B80/B112，失败结束具体形式。四项通过后与8项等价结果合成原12同版评分。上限1候选、4 GEMM条件各11调用、修复最多2次定向复核，资格为未辨识的远端副本粒度假设，GPU作业为0。

窗口第14轮：dirty owner淘汰共用HBM（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r14-dirty-owner-writeback/RESULTS.md)）。失败端点为暖状态下A16/B80/B112的主循环与空间推进。当前缓存淘汰直接丢行；NCU记录L2的dirty eviction会产生DRAM写回。唯一候选在原home保存dirty owner，store仍按原端点在home和writer登记，writer副本为clean；home dirty行淘汰时，在原全局HBM FIFO追加一次128B写回，输入miss与之共用带宽，回写价沿原b128，读写等价为开发假设。read分配为clean，已有dirty行命中保留状态；初始化和记录写入也按同一store规则处理，不专门给D补时间。淘汰使用实际logical访问时刻，缓存容量/映射/LRU、原GPC两槽/整atom服务、逻辑目的、程序/放置、计算/释放、频率、输出、P0及plain映射固定，不叠加其他轮次。先小图核对owner只写回一次、clean副本不写回、写回与miss FIFO及跨调用状态，再跑original/domain_local×last0/1四探针，沿第9轮原响应门槛；通过才G3 base/A16，三绝对误差不劣且A16响应改善才S4；S4三绝对与stage响应不劣才G5 base/B32/B80/B112，再补原12同版回归。失败结束具体形式。上限1候选、4探针与12 GEMM条件、各11调用，修复最多2次定向复核；输出源读完成到home写入的原假设资格及旧失败保留，GPU作业为0。


窗口第15轮：缓存慢路反压源槽（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r15-cache-source-backpressure/RESULTS.md)）。失败端点为主循环、行距与CTA推进；第9轮把cache访问移到core.end仍未改善A16，源槽却始终在τ服务结束即释放。已有GPC有限服务和两域源状态证据支持检查二者连接，慢路是否占住源槽仍为假设。本轮唯一候选在真实core.end执行原逐logical缓存访问，以`extra=max(0,data_ready-(core.end+hit))`延后该槽可重用时刻，最终ready为原core_ready加extra；只延后慢路超出量，hot退化原两槽。整atom订阅仍严格在service_end截止，槽归还另取该代全部订阅lookup的最大extra；动态FIFO只在槽真实空闲后发出，未发出请求不预约未来槽。区别于第9轮无反压、全L占槽以及第11/12轮home miss信用，不叠加这些形式或第14轮写回。可改cache→源槽连接与事件调度；原价格、缓存容量/映射/LRU/填充、目的、程序/放置、计算/释放、频率、输出、P0及plain映射固定。先核对hot逐位等价、迟到subscriber、同刻lookup与slot归还、队列及跨调用状态；再original-last0/last1成对筛查，沿原GPC2/3响应−34.029%±12pp，通过才domain_local两项0±10pp，各保留十次交替暖机。通过后G3 base/A16，base三项绝对误差不劣且A16三项及响应改善才S4；S4三项与stage响应不劣且第二MAIN跨度改善才G5 base/B32/B80/B112；关键七项通过再补原12同版回归。失败结束具体形式。上限1候选、4探针和12 GEMM条件、各11调用，修复最多2次定向复核；旧冷路和重排失败及独立验收资格保留，GPU作业为0。


窗口第16轮：复合提交与完成端点接入（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r16-composite-completion-boundary/RESULTS.md)）。失败端点为S6/S4主循环、stage响应、第二MAIN跨度和末commit→MAIN；[完成边界复核](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/completion-evidence.json)在512cycle复合服务下给出共同q下界407cycle，既有末commit→MAIN为495/500cycle。最多两点`(q,λ)=(500,0)/(407,93)`，93由500−407得到，500作为含软件观察路径的有效联合尾；资格为复合提交/完成服务假设。区别于已停止的ready2257.5固定供给剖面，本轮接原动态地址缓存与请求服务，不叠加第15轮或其他失败形式。可改复合计算模块的commit/complete参数及按既有阶段频率换算的λ接口；512cycle计算服务、220cycle复用、其余软件价格、程序/放置、源/目的/缓存、频率预测器、输出、P0与plain映射固定，不补每stage或case误差。先核对λ≤512−q时普通wait1不加尾、末槽和跨阶段完成依赖、缺省阶段参数退化旧版；旧纯TMA探针不使用该模块，复用原结果并保留重排失败。每点先G3_base，plain与首后L绝对误差均不劣才G3_stage4；S4三项也不劣、stage响应误差改善、两stage第二MAIN跨度误差均改善才A16及G5 base/B32/B80/B112。七项通过后补原12同版回归；任一失败停止该点，不补扫q。上限2候选、每点12 GEMM条件各11调用，最小筛查共最多4条件44调用，修复每点最多2次定向复核。原500cycle有效量的来源与物理不可分解性保留，不新增独立C2/C3资格，GPU作业为0。


窗口第17轮：按home域路由的两条源通路（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r17-home-routed-source/RESULTS.md)）。失败端点为747595固定源重排、行距与CTA推进；[状态缓存解析筛查](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/source-state-screen/proof.json)表明该探针每域保守需求仅17.154MiB，接入原缓存仍为全热命中。现有两条可互换源槽未解释重排，已测两域归属映射支持检验地址路由。本轮唯一候选把每GPC的两条最早空闲通路改为home0/home1各一条FIFO；按实际描述符覆盖的32B sector及原home hash分配工作，各通路沿原τ/512每sector计价，atom就绪取两部分结束的最大值加原L。整atom仍在两部分最晚service_end前共享，订阅不新增工作；目的按逻辑payload和原D投递。两通路与home的关联是组合假设，未作为物理端口事实；区别于第1轮追加域返回、旧A/B大小分道与第4轮只改总工作量。可改GPC源路由与分部工作；缓存访问/容量/映射/价格、程序/控制、计算/释放、频率、输出、P0与plain映射固定，不叠加其他轮次，不扫描路由或新价格。先核对实际基址归一化、home分部、总工作守恒与整atom截止；再沿旧六prior的ns/cycle双5%逐项筛查，通过才grouped/mixed Q1（原4pp主响应门槛；GPC4两布局跨度各至少达到对应七次观测范围下界，作为本轮开发必要条件），失败停止。通过才G3 base/S4/A16及G5 base/B32/B80/B112，要求主循环与plain误差不劣、stage/行距与第二MAIN跨度改善；再补原12同版回归，已完成项复用。上限1候选、8探针和12 GEMM条件、各最多11调用，修复最多2次定向复核；旧实验信息不足结论和独立验收资格保留，GPU作业为0。

窗口第18轮：调用前trace清零接入（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r18-trace-preparation/RESULTS.md)）。[协议复核](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/protocol-preparation-screen/proof.json)确认当前GEMM源只初始化A/B/D，实机每次计时前还清零trace；dual为每CTA1296个uint64，132CTA共1368576B。唯一候选把该写入按原初始化的home-only store规则接入每次begin之前，不计入event区间；trace基址沿现有2MiB连续相对分配假设放在D分配之后，不搜索地址。先用dual的运行前布局与逐case实际warmup_calls；多数条件8暖机，A16九进程8次、一进程10次，最多一条11调用链保存对应目标端点。可改该准备动作与协议读取；运行中trace写入、原源/缓存/目的价格及状态规则、程序/放置、计算/释放、频率、输出、P0及plain映射固定，不叠加其他轮次。暖机次数的旧保存端点等价已核对，不单独重跑旧版。先G3 base/A16，plain与首后L绝对误差不劣且A16响应改善才S4；S4三项与stage响应不劣、两stage第二MAIN跨度改善才G5 base/B32/B80/B112，七项通过才原12同版回归。失败停止此准备连接形式；上限1候选、12条件各最多11调用、修复最多2次定向复核。trace地址和home-only写入沿用模型假设资格；旧纯TMA探针程序未改，保存结果复用，独立C2/C3仍待证据，GPU作业为0。

窗口第19轮：整atom接入与home分部重叠汇合（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r19-core-home-join/RESULTS.md)）。第17轮移除整atom接入后首prior低估6.34%；本轮保留原两槽接入，检验地址路由竞争可否与其重叠。唯一候选先由原两槽得到S/E，再从S预约每GPC home0/home1各一条FIFO，以实际32B覆盖和原τ/512计工作；完成M=max(E,各part.end)，ready=M+L，整atom订阅至M，原接入槽在E释放。孤立对齐atom保留原完成；A/B同时进入时存在τ/4排队差，故不恒等旧模型。home路由与start接入均为组合假设，不作为已测端口事实；与第1轮域八槽及第17轮替换接入不同。可改这项重叠汇合；缓存、目的、价格、程序/控制、计算/释放、频率、输出、P0与plain映射固定，不叠加其他轮次。先原六prior双单位5%，再grouped/mixed Q1原4pp响应及GPC4两布局跨度各达到七进程范围下界；失败停止。通过才G3 base/S4/A16及G5 base/B32/B80/B112，绝对时间与首后L不劣、stage/行距及第二MAIN跨度改善，再补原12同版回归。上限1候选、8探针和12GEMM条件、各11调用、修复最多2次定向复核，GPU作业为0。

窗口第20轮：GPC共享逻辑目的服务（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r20-gpc-destination/RESULTS.md)）。747595 Q4 mixed/GPC4同源CTA64/68在g3发出同刻、wait_return分开128ns，要求检查共享源之后的逻辑目的服务；两通道沿原D仅工作下界33752.605ns，已超过grouped Q4观测25920ns，解析排除。唯一候选保留源服务与原D，把独立CTA目的FIFO改为同时占用本CTA与本GPC三槽之一的同一服务区间；三为原D下未被容量下界排除的最小整数，是共享范围假设，不新增串联D成本。每份逻辑payload独立服务，不随源合并；在真实source_ready到达后排队，只从已到达且CTA目的空闲的队首选择，GPC空闲槽按到达FIFO发出，不预订未具备条件的未来服务。区别于第1轮可合并域返回及旧dual-granule同时修改源与目的。可改目的共享范围与实际发出事件；源/缓存、价格、程序/控制、计算/释放、频率、输出、P0及plain映射固定，不叠加其他轮次。先原六prior双单位5%，通过才grouped/mixed Q1原4pp响应及GPC4两布局跨度下界；再两项Q4核对总时间与同源失锁：GPC4最早begin至最晚g63 release落在原七进程范围扩展±32ns的观测量化区间，mixed同源跨度中位>32ns、grouped≤32ns；原Q4信息不足结论保留。通过才G3 base/S4/A16及G5 base/B32/B80/B112，绝对时间与首后L不劣、stage/行距及第二MAIN跨度改善，再补原12同版回归。上限1候选、10探针和12GEMM条件、各11调用、修复最多2次定向复核，GPU作业为0。


窗口第21轮：GPC内有限在途行信用（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r21-gpc-credit/RESULTS.md)）。旧六prior保存轨迹的GPC峰值复核为Q1≤2816行、Q2分别4032/4224至5632行；747595 Q1 grouped/GPC4同步4个源为1024行，mixed首组13个源在最早ready之前已发完，实际同时3328行。该差异支持检验GPC内有限容量，区别于第3轮跨四GPC接收域信用、旧sector FIFO及A/B大小分道。最多两版沿原两槽整atom服务与价格，信用范围改为单GPC，容量固定3072条128B行（12个32KiB请求组），分别在源就绪或该代最后目的完成归还；不扫描容量。同物理行只在原service_end之前共享，完整订阅不延长截止，缺新增信用时FIFO等待；保留原逻辑目的服务。可改信用范围与准入连接，程序/控制、源仲裁、缓存/价格、计算/释放、频率、输出、P0及plain映射固定，不叠加其他轮次。先用保存Q1流及同源cohort归纳证明三项prior零阻塞且逐位退化原模型；证明不足才正式回放。每版先q2-skew32，再q2-skew0/16，沿原ns/cycle双5%；通过才grouped/mixed Q1原4pp响应及GPC4两布局跨度下界，再Q4原两布局总时间量化区间与同源失锁条件。失败停止该版。幸存才G3 base/S4/A16与G5 base/B32/B80/B112，主循环/plain绝对误差不劣、stage/行距及第二MAIN跨度改善后补原12同版回归。上限2候选、每版10探针（含三项拟以等价证明复用）与12GEMM条件、各最多11调用，修复每版最多2次定向复核；3072为有限容量假设，不作为硬件容量测量，GPU作业为0。


窗口第22轮：跨域副本返回的有限吞吐（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r22-remote-return/RESULTS.md)）。现有两域驻留与暖机次序干预支持保留远端副本路径；当前read对每条远端行只加固定link延迟，未计同方向返回的共享工作。第1轮对全部源追加接收域服务未解释热源重排；本轮只改变实际缺少接收副本的跨域128B填充，本域hit与已有pending订阅不新增传输。唯一候选为每返回方向一条FIFO，按logical read到达顺序排队，队首等待home数据就绪；每128B服务价取原τ/(128×8)，8由该接收域4GPC×原2槽的吞吐预算组合而来，随后加原link延迟。该预算是组合假设，不作为已测链路速率；不新增拟合价格，不借用DSM参数。缓存仍在原logical read登记未来副本，后续pending合并沿原规则。可改远端返回吞吐与ready连接；源两槽/整atom截止、缓存容量/映射/LRU/填充、程序/控制、计算/释放、频率、输出、P0及plain映射固定，不叠加其他轮次。先小图核对本域hit、同line只传一次、方向独立、FIFO等待home和跨调用前沿；热源Q1及旧六prior的全热路径解析复用，原重排失败保留。先original-last0/last1沿原GPC2/3响应−34.029%±12pp，再domain_local两项0±10pp；通过后G3 base/A16，base三项绝对误差不劣且A16三项及行距响应改善才S4；S4三项与stage响应不劣且第二MAIN跨度改善才G5 base/B32/B80/B112，再补原12同版回归。失败停止此固定FIFO预算，不扫描吞吐。上限1候选、4探针与12GEMM条件，各最多11调用，修复最多2次定向复核，GPU作业为0。


窗口第23轮：128B轮转的共享目的服务（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r23-destination-chunks/RESULTS.md)）。第20轮整atom占用三槽在首prior低估5.225%，事件复核确认新增目的等待改变了后续源相位；747595 Q4同源CTA在同刻发出后失锁仍要求辨别目的仲裁。唯一新形式保持三槽与原D的总吞吐和单CTA速率，把槽租期由完整atom改为128B逻辑传输块：每块服务D/128，16/8KiB请求为128/64块，同CTA一次至多一块、按atom FIFO；每GPC空槽从已就绪CTA队列轮转选择，块完成后未结束请求回队尾。最后一块完成才投递完成；不新增串联D或价格。128B是本轮传输粒度假设，三槽沿原工作下界，不当作硬件端口。区别于第20轮不可抢占atom及第10轮源仲裁，不叠加它们。可改目的仲裁/连续占用粒度及必要的实际busy计账；源/缓存、程序/控制、计算/释放、频率、输出、P0及plain映射固定。先核对单CTA退化、总块/字节/busy守恒、私有atom顺序、轮转和三槽上限；再旧六prior双5%，幸存才Q1原4pp及GPC4跨度下界，再Q4原量化时间与同源失锁条件。失败停止此128B/三槽具体形式，不扫描块大小或槽数。幸存再接G3 base/S4/A16与G5 base/B32/B80/B112，沿第20轮主循环/plain不劣和stage/行距/第二MAIN跨度改善，关键七项通过才原12同版回归。上限1候选、10探针与12GEMM条件，各最多11调用，修复最多2次定向复核；GPU作业为0。

窗口第24轮：显式wait1与release软件窗口（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r24-explicit-release/RESULTS.md)）。739556的普通wait1观察窗口中位约6cycle，release段为S4/S6的20/22cycle；当前引擎wait_overhead与loop_advance为0，释放与wait返回同刻。512cycle来自饱和计算吞吐，旧220cycle是未分解的释放复用敏感性点，未证明这些软件段已包含或未包含。唯一候选将观察窗口可迁移作为假设，两stage共同使用wait1=6、release=21cycle，按已有输出阶段频率换算；普通W=max(C,Dprev)+6/f，Rprev=W+21/f，下一consumer与producer均须经过Rprev，producer在Rprev后仍沿用220cycle；末次普通循环的math_handoff也经过Rprev，末slot在wait0后增加21cycle，MAIN在末release之后。该形式把220明确解释为release后复用路径，不把27cycle加到512计算吞吐；与第16轮q/λ边界修改不同。可改这些软件端点及必要绑定，固定程序工作/源/缓存/目的/价格、512/256计算、频率预测、输出、P0和plain映射，不叠加其他轮次。先核对k=0无wait1、普通释放前驱、末slot与main_end顺序及默认零开销逐位退化；正式先G3 base再S4，两者plain/first_L/later_L绝对误差不劣、stage响应与两stage第二MAIN跨度误差严格改善才A16与G5 base/B32/B80/B112，再补原12。失败停止此6/21形式，不扫描软件价格。上限1候选、12条件各最多11调用、实现修复最多2次定向复核；纯TMA图未改，保存旧约束与重排失败，GPU作业为0。

窗口第25轮：按请求计量的目的完成服务（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r25-request-completion/RESULTS.md)）。第20轮按payload计D的三槽不可抢占服务在首prior低估5.225%；第23轮只改块轮转仍失败。现有cfg_b及匹配探针每组固定A16KiB、B0/B1各8KiB，源之后的观察又未分开数据写入、完成通知与等待恢复，支持把服务工作单位作为待区分假设。唯一新形式保留三槽、实际source_ready到达FIFO与CTA私有互斥，把每条逻辑tensor请求的服务改为共同2D/3，三个请求总busy仍为原2D；不再按payload比例给A/B不同服务。该服务计量是请求完成候选，未识别为物理通知单元；不新增串联D，不拟合新价格，不叠加分块轮转。可改目的工作单位与实际busy计账，固定源两槽/整atom截止、缓存/控制、计算/释放、频率、输出、P0及plain映射。先小图核对A/B等成本、每组三请求busy守恒、三槽上限及完成回调；再旧六prior双5%，通过才Q1原4pp响应/GPC4跨度下界，再Q4原量化总时间与同源失锁条件。失败停止此固定计量形式，不扫描通知价格或槽数。通过才按第20轮七项组合门槛接入G3 base/S4/A16与G5 base/B32/B80/B112，之后补原12同版回归。上限1候选、10探针与12GEMM条件，各最多11调用，修复最多2次定向复核；GPU作业为0。

窗口第26轮：本地请求准备后进入GPC服务（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r26-local-setup/RESULTS.md)）。现有GPC竞争与TMAU专利FIG.6的SM侧descriptor/setup、请求生成阶段支持检验私有准备与共享服务的串联；旧候选只在GPC服务之后保留一个复合延迟。新形式对每条逻辑tensor请求先经过本CTA准备槽，再在真实准备完成事件进入原GPC两槽服务；准备占用δ=L−hit，GPC服务后的剩余延迟改为hit，孤立请求总延迟保持原L，GEMM缓存查询也移到准备完成时，源订阅按真实GPC到达/service_end判断。δ的拆分与占用是本轮假设，L/hit沿既有价格，未识别实际setup时长。固定两个容量形式：每CTA1槽、2槽；前者先由保存G3 base/S4的Ktile数、输入槽数与(kt−S)×3δ/容量下界筛查，若超过已测FIRST→MAIN即解析停止，不执行模型；两槽只在未被同一必要界排除时回放。每条TMA均需准备，源合并只省后续共享服务；每CTA即当前一SM一CTA，不外推多CTA驻留。区别于第19轮共享home重叠汇合及旧两个GPC串行阶段；不叠加其他轮次。可改准备准入、实际到达与复合延迟位置；固定总价格、缓存容量/映射/LRU、程序/控制、目的、计算/释放、频率、输出、P0和plain映射。先小图核对孤立延迟、local与GPC共享范围、晚到请求不预占未来共享槽和零准备逐位退化，再原六prior双5%，通过才Q1原4pp/GPC4跨度下界及Q4原量化时间/失锁门槛；失败停止该容量形式，不扫描δ或容量。幸存按第20轮七项组合门槛接回，再补原12。上限2容量形式（1槽优先解析筛查），每版10探针与12GEMM条件、各最多11调用、修复每版最多2次定向复核；GPU作业为0。

窗口第27轮：共享目的槽允许同CTA的独立atom并发（已结束，[结果](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-window-001/r27-parallel-destination/RESULTS.md)）。第20/23/25轮均令目的服务同时占CTA私有槽与GPC三槽，原τ/L/D又由每CTA独立目的FIFO联合拟合；程序中A、B0、B1写不同SMEM区、共同mbarrier只汇合总字节，没有跨atom的软件串行边。本轮保留原GPC两槽整atom源与service_end订阅，目的改为GPC三个共享槽按真实source_ready到达FIFO，不再占CTA私有互斥槽，同CTA不同atom可并发。三槽为固定拓扑假设；不是新硬件容量结论。旧dual-granule曾有GPC四目的槽且无CTA互斥，但同时更换sector源/信用且未重辨识，本轮不恢复该具体形式。只用746541的N32/Q1 independent `w0-m0-t1`、`w0-m64-t1`两目标起步（暖机均0）和746425的N1/16/132×Q1/Q4共享源六条件重新辨识全局τ/L/D；校准cycle按原参考1.9951034959906244GHz统一，原ns/cycle同时保留，固定原控制/消费者及实际CTA分组；不把747595、六prior或GEMM放进拟合。八条件等权，各条件相对残差除以端点数平方根，原5%仍逐端点验收；1起点、非负三参数、least_squares最多24次主步且120次残差调用，加最多10次局部秩/±10%参数敏感性核对，总计最多130×8=1040次单调用校准回放；无跨调用状态的探针只回放目标，不重复暖机。先核对原数据与程序、同CTA并发、GPC三槽/FIFO、工作量/排空及默认路径退化，再冻结校准输入和实现。校准各原端点须过原5%，局部数值秩为3才冻结唯一拟合参数；未达则信息不足或被反证，不换起点或扫描槽数。该辨识仍属经验参数，不宣称唯一物理价格。幸存先原六prior双5%，再Q1原4pp/两跨度下界、Q4原量化时间/同源失锁；全部通过才接G3 base/S4/A16与G5 base/B32/B80/B112。七项plain及首后L的绝对误差中位/最大不劣于窗口开始版，stage和A16/B80/B112响应误差及两stage第二MAIN跨度误差均改善才补原12同版回归，原C2/C3验收保持。固定缓存/地址/容量/LRU、程序/控制、计算/释放、频率、输出、P0和plain映射，不叠加其他窗口形式。上限1规则版本、10判别探针及12GEMM条件各最多11调用、实现修复最多2次定向复核，GPU作业为0；具体输入与复现保存在r27共享目的并发包。

当前 CPU 轮次（2026-10-16）已完成：接入固定P0，五项各正式回放一次、其余七项投影，未重跑G6；[交付与复现](../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-prefix-001/README.md)。本轮上限已用完，原登记顺序保留如下。

1. 复用原12条件的时间线、原始观测和六分项评分；重点核对 G1_base、G2_base、G4_base、G3_base、G3_stage4 五项的 `entry→producer_first_work→FIRST` 软件边界与模型端点。原评分的 `prefix` 为 `entry→FIRST`、`prefill` 为 `producer_first_work→FIRST`，模型后者起点取首个 A issue；核对现有前缀价格覆盖的操作及来源。
2. 只修改前缀的程序/观测接线或接入有现成证据依据的前缀服务，最多一版。现成候选为[冻结校准](../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/frozen/calibration.json)的 cfg_b `P0=1608 ns`，覆盖 `entry→producer_first_work`，接入时标为经验替代；`S=1192 ns` 覆盖 producer 到 FIRST 的整段，不与现有请求就绪服务叠加。固定输入与 CTA 放置假设、TMA 源/目的规则、计算与槽释放、频率、输出及 plain 映射。五项各最多一次正式 CPU 回放，保留原调用协议；不搜索新价格、入口偏移或 stage 专属修正。
3. 有可接入修正时，补齐原12条件的同版回归，继续报告 A16、B80/B112、stage 响应和最晚 CTA；同版已完成的五项直接复用，其余条件各最多一次。未受影响且能核对同版等价的条件复用已存结果；G6 仅在事件过程受影响且无法从保存时间线推导时回放一次。无可接入修正时，复用原整组结果收口。
4. 交付端点对应、修正依据、前后结果及剩余证据缺口。没有匹配现成证据的可接入修正、仍需新成本测量，或本轮上限用完时，结束本轮并给出待批方案；原 C2/C3 未通过项保留。

## 2. 三类模型与程序图的接入依赖

cfg_b 组合原第 1、2 步的 CPU 交付已完成，验收与入口见[组合 §4](model/mechanism/composition.md#4-输出)。请求级候选已接通共同程序的请求、计算完成和槽释放端点，以及阶段频率、经验前缀和 plain 映射；原12条件的结果与失败反例已保留。§1 的历史前缀轮次已收口；替换供给规则时仍检查[联合约束](model/mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md#下一候选的联合约束入口)，该整组回归不作为前九个局部阶段转段的门槛。

| 工作 | 推进顺序 | 接口或交付入口 |
|---|---|---|
| 规格基线 | 独立计算入口已完成；比较时记录产品/参考频率口径 | [规格基线计算](model/spec/spec_baseline.md)；C1-01继续使用原冻结口径 |
| 半经验模型 | 保留V09冻结版本与C1-01已有结果，先对照当前组合 | C1-01已经分别列出纯规格、wave_roofline、structural与TileSight，完整时间归C3 |
| 程序图 | cfg_b S6/S4 已可导出并核对历史工作映射；请求级完成事件已接通候选，cfg_a S4/S6和cfg_c S4的cluster整图与指定历史端点核对已完成；cfg_c1也已完成程序核对；其余配置SASS对照与TileLang前端暂缓 | [CUTLASS程序图](model/mechanism/program/cutlass_program_graph.md)、[操作与事件接口](model/common/interfaces.md) |
| 硬件服务候选 | 按[模块地图 §6](model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md)的九阶段逐项交付条件服务或显式工作假设及接入端点，已有规则和反例保留 | 当前第 1 阶段条件见[有限登记](#stage-1-l2)；具体实验按 GOALS 登记 |
| 机制组合 | 所需局部规则和程序图可用即做必要接线；第 10 阶段汇成同版 cfg_b 模型并检查原12条件响应和完整时间 | 完整输入→时间线→plain 与原阈值报告齐全才作该阶段交付；C2/C3 未通过项照实保留，旧版预测继续对照 |

频率沿用已有版本与声明范围，供给规则由请求级组合检验；整组残差决定是否需要改变两者的连接。

## 3. 后续任务的进入条件

以下是本次九阶段与组合队列之外的候选方向，当前不执行；本次队列内按授权依次推进，队列外的新阶段由用户选择，按 [GOALS 当前阶段限制](GOALS.md#current-scope)处理。

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
