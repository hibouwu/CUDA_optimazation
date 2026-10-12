# GH200 实验执行规范

实验问题见 [PLAN](PLAN.md)，现行候选见 [RULES](RULES.md#v09-model)，各项证据写对应实验页。本页只维护分工、共享代码、存储、确认边界与当前状态。已完成批次的提交、启动文本和故障经过见 [cee978a固定版本](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/EXECUTION.md)。

## 分工与交回

管理对话统一维护公共模型、代码集成、GPU分配和最终评分；执行组按原实验承担完整的小目标，同卡对照独立完成后交回。每个文件/运行目录只有一个写入者。

| 唯一写入者 | 对话ID | 文件与任务归属 |
|---|---|---|
| 主集成 | `01a129c0-b787-7e03-adf1-fcfdb2afa149` | 公共probe/trace、CUTLASS覆盖头与runner、`v09_model.py`、`analyze_v09_ablation.py`等总分析入口、PLAN/RULES/EXECUTION、GPU统一调度、V01/V10/跨卡及最终评分 |
| A：时间输入 | `01a11da3-a045-7ae3-b2b5-f5762e3bddf5` | R09及时间/输入文档、`clock_model.py`、`r09_input_source_clock.py`等专属分析和测试 |
| B：供给请求 | `01a11da3-a02c-7f93-8290-a82d81df0795` | R10/R13、R14输入槽与安全释放、`supply_model.py`、`analyze_r10.py`、`analyze_r13_supply.py`等供给分析和测试 |
| C：输出边界 | `01a11da3-a02c-7f93-8290-a83b838ab41a` | R15/R18/R19文档、`output_model.py`、`analyze_r18.py`等专属分析和测试；共享replay接口变更先交主集成协调 |

2026-10-11用户批准恢复以上三个既有专业工作树；实际同步与恢复由规划对话负责，主集成不重复checkout或派发启动任务。当前供给轮次在[本地E0排他检验](R13-async-retirement.md#local-epilogue-exclusion)处交接，不以先解决R13为前提。交接后主集成停止并行修改专业模块；跨组接口由主集成合入。源工作口径由B定义、A使用；跨rank依赖由C提出、B接入供给前驱。

规划对话已确认三个旧worktree均从干净状态切到共同交接提交 `94f9bcfaa991c2a9fb5af4c4dfeb4844d8c4feb7`，旧分支HEAD原样保留：

| 专业组 | 原worktree标识 | 恢复后的分支 |
|---|---|---|
| A | `9cce` | `codex/gh200-clock-parallel-20261011` |
| B | `0785` | `codex/gh200-supply-parallel-20261011` |
| C | `9e4a` | `codex/gh200-boundary-parallel-20261011` |

规划对话已从三个create_goal原始工具结果核实A/B/C目标均为active、无token预算，并核对各自turn_context的工作树与权限；三组已开始离线研究。主集成的wait_threads也确认三组当前turn均在执行，A另已交回恢复与首轮分析范围。主集成统一接收专业组交回及发送集成反馈；并行恢复完成不代表研究目标完成。

专业组只写各自原实验的独占分析输出，不运行会覆盖共同`candidate-development-v1`的总分析。旧共同目录作为只读证据输入；需要公共probe/runner或总模型改动时提交有界接口需求。用户已授权主集成与上述三个专业对话互发有界协调消息，不能扩展到其它对话或外部人员。全组cfg_a/b/c范围、完整时间中位≤5%/最大≤10%、分项/关键CTA、新冻结留出与跨卡要求不变，V10采样授权边界不变。

[本次交接结果索引](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-a-row-rotation/handoff-manifest.json)列出可复放命令、已保存的失败候选、未决问题以及仍留在主checkout的忽略数据路径。上层DAG规划的三个文件保持原未提交状态，不纳入本次研究交接提交。

主集成已复放[冻结V09联合验收缺口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-V09-gate-audit/README.md)：分项先取绝对值与共同globaltimer关键终点差单独报告；历史总时间评分不变，下一次冻结前仍需确定有效观测及联合门槛。

B首版静态账本 `abc8179` 已由主集成合入为 `c7509ef`。总模型通过 `gh200-supply-work-ledger/1` 的 `totals.source_payload_KiB/132` 读取r2名义源工作量；不同版本或valid/distinct口径不能套旧校准。接入前后30个V09完整序列化预测逐项相等，29个受支持条件的时钟输入与冻结记录一致；拒绝项保留。此项完成接口统一，不改变供给候选尚未通过的结论。

A首批时间契约 `8634da9` 已合入为 `a98cb3f`。主集成在新临时目录独立执行 `r09_time_contract.py` 的完整与 `--stage-endpoints` 两条复放：七份主JSON与一份端点JSON均与A-v3逐项一致，5项时间契约测试通过。历史固定点下界及首/后换算端点约束仅作开发证据，未采用新时钟候选。A提出的bracket观察尚未冻结：现有common-sw1-16调用当前变体kernel，不满足三变体相同plain前序；需补精确区间判定和扰动标准后才进入公共probe准备。

公共plain前序的共享库源码已准备：`probes/r09_plain_prefix.cu/.h`复用原cfg_b/stage6 kernel定义，通过C ABI接收调用方A/B/D地址，在初始化阶段创建workspace，run只入队同一个plain kernel。ABI 1仅接受A观察请求的2816×1536×4096、132SM目标，不扩充正式模型范围。[740042构建检查](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-plain-prefix-build/README.md)已用CUDA12.9.41通过：仅4CPU/16GiB、无GPU分配；共享库可加载，非法ABI拒绝，GEMM的1280条SASS与同批及冻结V09 plain一致。[740048 host检查](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-prefix-host-check-r5/README.md)已完成plain/dual各一次共享plain前序＋目标调用，原容差4096抽样与padding检查通过；两目标kernel SASS分别与冻结V09一致。四次准备失败保留，实际grid更正为[1,132,1]。尚未验证bracket观察资格或开展正式采样。普通r18路径只增加不启用的library main guard，默认正文逐字节保持。

C首版依赖图 `e097a6b` 已合入为 `2878c1a`。主集成9项专用测试通过；100进程独立复放产生13200CTA、146880阶段行和91760区间行，两图及五份CSV内容精确相同。主JSON仅因已合入B账本而改变supply_model源码哈希，诊断一致。图无服务价，370条轨迹可行不能作因果或预测通过证明。

B的释放前驱分析 `1042b17` 与输入校验 `c786438` 已合入为 `a05437e`/`843fce4`。主集成20进程复放的逐项观测、两组非唯一分解和摘要与B-v3一致，5项测试通过；source位置/修订哈希单列。现有leader到acquire窗口不能当作最后128个arrival已到之后的价格，四warp观察已完成实现与740051 CPU构建，740054全D及单次轨迹smoke通过；正式40进程及扰动门槛尚未完成。

B的可变输出供给图 `fd7d2f9` 已合入为 `d08a5a4`，65项专用测试及13组旧/新默认图逐字段比较通过。[R10独立复放](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/main-20261011-source-event-transfer-v1/README.md)的100进程/17组响应与B一致；AlongN方向、首输出和空间响应排除当前条件价格点，不引入新正式价。

B的cfg_a原子源组合 `52f18ce` 已合入 `94992c8`：[主线复放](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R18-cluster-fill-discrimination/reanalysis/main-20261011-cfg-a-source-composition-v1/README.md)核对22条件/220进程/28200CTA/12模板，74项测试通过。每group源48KiB、目的64KiB守恒；C输出保证升级之外的绑定/边/工作量不变。C的实际emit接口 `d91e65e` 已合入 `4a31199`，[独立CPU枚举](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R18-cluster-fill-discrimination/reanalysis/main-20261011-empty-arrival-routes-v1/README.md)与C完整同文；8条发送路径和emit/可见/acquire_return细分保留12份组合图旧依赖，无新服务价。

A的专属离线链已串行合入至 `f950ca5`，63项R09测试通过；[历史预测基线独立复放](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/main-20261011-history-forecast-v1/README.md)的80进程/2320 event预测/601920角色分解完整同文，switch最大40.62%保留。局部可变时钟入口对旧740048 smoke触发当前前序身份拒绝，[失败记录](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/main-20261011-local-clock-consistency-v1/README.md)保留，专属历史诊断修复4f39ab6已合入af06c62，65项R09测试通过；[v2独立复放](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/main-20261011-local-clock-consistency-v2/README.md)的21进程完整JSON与A一致，旧smoke仍qualified=false，不放松公共解析或提升旧协议资格。

C的输出buffer generation接口 `287930d` 已合入为 `6854ee0`。主线[独立R14复放](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R18-cluster-fill-discrimination/reanalysis/main-20261011-output-lifecycle-v1/README.md)复核852份源码、两二进制及40进程数值；20份trace的90个保证、70个跨输出中间窗口与C一致。只规范化当前工作目录绝对前缀后，JSON和两个CSV完整文本相同。中间permit→保证36598.5cycle包含下一输出推进，下一permit→保证657.5cycle；不将前者收费或迁为ns。

[740056逐线程pilot](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-release-threads-correctness/README.md)已完成（a057/GPU432，4CPU/16GiB/1GPU，134秒）。新r13-release-threads-v2保留128线程各自原arrival前后戳；全D与严格解码通过，两stage on为128/128、最大区间25/22cycle，SASS保留原发出且无区间内分支/写回。旧四leader不能替代该覆盖；active-mask读数本身也不提供后续收敛保证。每stage仅一对、非独占且非ef，paired gate仍不合格，无新价格或V10。

740054的数值/记录通过不等于局部观察合格：非独占单对smoke的stage4 H16→wait1差368cycle，超过原单对192cycle门槛；保留预警，不从本批采用机制区间。四warp正式配对门槛在采样前明确按 `median(abs(delta))` 和 `max(abs(delta))` 判定；有符号中位数仅作诊断，不允许抵消。

[740050前序身份构建](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-prefix-identity-build/README.md)已通过：setup/call显式区分模块plain前序及清零准备，解析器拒绝未声明身份。旧r5仅保留数值smoke，不升级为同准备或观察有效性通过；正式dual/bracket必须另核对模块及准备容量一致。

长期执行对话用独立worktree；同目录临时agent按明确文件边界工作。共享文件需求交管理者合成，不能各自复制cfg_a/b/c框架。各组只提交本组文件，管理者串行集成；不reset/stash他人修改、不追改冻结包。已完成的证据可读取复用，单组完成不代表整轮预测通过。

交回只需：问题与一句结论、改动/提交身份、job及GPU UUID（如有）、结果路径与复核命令、适用范围/失败、影响别组的变化。完整批次或明确阻塞时同步，不增加第二套任务数据库、交付目录或分钟级日志。跨对话消息沿用用户对本轮指定对话的已有授权，不扩展到无关对话或外部联系人。

<a id="shared-gemm"></a>

## 共享代码与计时边界

| 公共对象 | 维护要求 |
|---|---|
| `r18.cu`、`r18_trace.hpp`、共享CUTLASS覆盖头、`r00_common.hpp`、`gaps_common.hpp` | 管理者维护输入/seed、正确性、静态资源与事件定义；改动后检查受影响的数值、SASS/资源和观察扰动 |
| `run_r18.py`、`run_v08.py`、`run_v09.py` | 保持同一构建/采样逻辑；公共补丁发布明确SHA，新批次使用新run-id |
| `v09_model.py`、供给/输出/时钟模块、既有CTA递推 | 预测只读静态输入与冻结参数；训练、归档解析与评分留在分析入口 |
| 实验专用runner/analyzer/config | 所属组维护；开始承担共享行为时由管理者明确归属 |
| 原frozen、旧run、families冻结源码 | 只读，保留旧版本用于重放 |

正式时间换算使用公共cfg_a/b/c与实际输入模式。旧R09的128×256×64、4stage探针仅作机制对照，常数不直接移植。plain/stamped/ends/wide/dual分别保留身份和边界；整调用扰动小不能替代局部分项检查。

历史公共包6338653、输出打点d8eb6a0、行距eb49b87/c690f61、输入map 1f175c2各自绑定既有运行，不用当前checkout追改来源。现行纯预测入口为 `v09_model.predict_components`，供给、输出、时钟接口及冻结参数见 [RULES](RULES.md#v09-model)。

## 数据与GPU资源

- 本地归档根为 `results/gh200_resource_campaign/access_rules`；worktree读取主checkout的明确路径，不复制整份历史results。旧原始数据、summary和frozen不覆盖；新分析写独立目录，新采样写新run-id。
- 新批次从准备到收取结果共用一个逻辑run目录；job ID记录在元数据，远端`/tmp`仅作必要中转，不再永久保留一份重复准备包。失败重试归该run的attempt记录，各自保留来源；不同源码、条件或协议不能因此拼成成功样本。
- 清理准备副本时逐文件核对source、build、cases和独有记录。10月9日的20个重复准备目录已清理；对应作业及失败记录保留，两个早期R09版本的9个独有文本与目录对应表保存在[准备历史附件](../../../../../../results/gh200_resource_campaign/access_rules/preparation-history-20261009.tar.gz)。
- 正式GPU计时只在Slurm获配设备上运行。最多三个独立节点各一计时流，实际按配额缩减；同节点多卡正式计时须先有互扰对照。整节点exclusive可能分配全部GPU，以AllocTRES计资源，不把三个计时流写成只占三卡。
- 已准备好代码与条件再申请GPU，不占分配等待开发或人工确认，不为同一批次重复提交副本，不取消其他研究作业。共享GPU的计时串行。
- 本轮历史home/scratch已近硬配额，曾改用登录/计算节点本地`/tmp`中转；这是当次状态，新批次先核对当前余量。临时盘不当长期归档，编译与GPU检查在计算节点完成。
- 先确认本地完整副本再按已授权范围清理本轮临时副本；旧数据不删除。不新增交付系统或重跑全部历史检查，只验证受影响且证据不足的部分。

<a id="reference-gpu"></a>

## 参考卡

V09参考卡为romeo-a057的 **GPU-43269fbc-449d-3e0f-908a-9c81229546d3**；V07/V08的GPU-099dda56是另一张历史参考卡。节点名不构成GPU身份或当前可用性证明。[共同参照跨卡复放](../../../../../../results/gh200_resource_campaign/access_rules/20261011-V09-crosscard-references/README.md)已核实两卡cfg_a/b四个条件的plain/dual二进制及SASS一致并复放160进程；仅为历史对照，cfg_c及新冻结跨卡留出仍缺失，不能据此判定迁移通过。

指定卡必须属于实际job/step分配，随后才收窄`CUDA_VISIBLE_DEVICES`到完整UUID；不能用全节点设备清单或覆盖环境变量取得未分配设备。参考卡分配可含四GPU、只启动一个计时流，仍按实际四卡额度记录。新卡需重做受影响校准；同卡跨作业也先跑预列参照，不把运行状态自动视为一致。

<a id="v09-freeze"></a>

## V09确认边界（已经完成）

用户明确要求V09留出前核对完整条件、预测与两份SHA；确认绑定具体版本，不是对任意未来预测的许可。流程为：

1. 作业一按固定程序完成参照/校准、拟合、预测、只读冻结与回传，然后释放分配；留出只取静态setup，不做暖机或pilot。
2. 管理者展示完整留出表、逐例预测、预测SHA、清单SHA、GPU UUID、支持范围和跨作业参照标准，由用户确认指定版本。
3. 作业二重新取得同卡，核对确认身份与参照后采留出、独立评分；不符则停止留出。重校准/改预测须另行确认，不能沿用旧许可。

[V09 r2](V09-component-validation.md)已获用户确认并由739011测完；本轮不再请求同一确认。原r1因支持判定修订被替代，未经确认采样。确认的预测SHA为 `f4df39e00cf1d2ab61947c911e699bd5bea4e533609d93f726de43077515d81e`，清单SHA为 `09da85d124eb697a95c4695bd05281b07c37877890d0eca18afbdd2cf213f2ab`；完整身份与参照结果以V09页为准。

这条历史要求只约束已声明的V09新完整留出，不给每次离线分析或原实验开发新增审批。后续如产生新验证版本，先明确该轮范围与用户授权。

<a id="counter-access"></a>

## 计数器权限

[已有诊断](../../../../../../microbench/gh200_l0/results/20260930-counter-access/README.md)记录普通/独占作业均受`RmProfilingAdminOnly=1`限制，NCU返回`ERR_NVGPUCTRPERM`；独占或换partition不等于取得权限。物理流量分解在受支持入口确认前保持未完成，计时研究可继续。

ROMEO账户持有人联系管理员；管理者可准备诊断、指标和窗口，未经另行明确授权不代发邮件/工单、不修改驱动或功率设置。管理员给出入口后只做一次针对性验证，记录节点与授权范围。

<a id="status"></a>

## 最新独立复核

[740061同卡配对](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-release-threads-ef-study/README.md)保留全部60进程及两次无样本启动失败，exclusive按4GPU计资源；上游异常定位于consumer full_wait，尚未确定机制。A跨SM共同变化时钟9ec0a87的完整JSON复放一致，84份可行/不可行证书另逐边核验；C首输出切点ee27859的JSON逐字节一致。两项仅交付离线约束和待实施的观测接口，不改变预测或留出资格。

C的cfg_c输出结构f0c3674合入f6e27c4，26项测试通过；主集成baseline复放40进程/23592960值，两CSV及完整JSON与C一致。cfg_a回归40进程/11796480值，仅规范化仓库路径后CSV一致、JSON仅代码SHA改变，T1/T2/T8完整rank图与旧版相同。cfg_c八子tile本地输出图与P2逻辑字节守恒已覆盖，legacy时间不迁入。A有界时钟读fab91fd合入841f5d0，6测试及本地CUDA13一处/两处PTX fixture独立编译检查通过；目标CUDA12.9/CUTLASS及硬件观察资格仍未验证，尚未接入公共probe。

[首输出五切点构建](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R18-first-output-build/README.md)已落实：派生prepare/helper/严格decoder已提交，C完整调用配对约束c8f6f76合入，16项CPU测试通过。740064头文件重复定义失败保留；740065成功后经C审查修复host模式入口，740066最终CPU构建通过，SASS与740065相同，168regs/2barrier/零spill，无GPU分配。五戳延迟写回，原HGMMA/TMA/arrival/wait数量相同；全D和独立控制pilot仍待执行，不判观察资格或新价格。

[740067首输出数值检查](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R18-first-output-correctness/README.md)已完成：a057/GPU432、4CPU/16GiB/1GPU、6秒。两map各2703360全D逐位置解析复核精确相同，off/on及跨map完整D相同，四调用记录通过，两个on各24角色。没有性能/观察资格；独立控制pilot与正式ns门槛仍待完成。

B的cc29004已合入63134e5。主集成独立复放60进程冻结release研究和12几何/10条件覆盖单元筛查，仅规范化工作树路径后完整JSON一致，257/11输入SHA重新核对一致，7新增测试通过。40/40 producer issue标记早于full_wait，stage4 off自身波动保留，原观察资格失败不变。36/126的覆盖并集均额外6KiB，固定旧图时间表不能解释响应差；工作变化对ready/订阅的反馈尚未求解，无新供给价格。

[740068首输出共同L控制](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R18-first-output-control/README.md)完成20样本，ef卡exclusive按288CPU/4GPU/16GiB、19秒记账。数值/记录/同SM身份通过，24角色绝对差中位128..352ns、最大448..800ns，全部违反冻结64/256ns门槛。整调用0.40%/2.62%及offCV0.98%通过不能抵消局部失败；不采map960、不放宽、不授内部端点或机制资格。

[740068事后共同变化检查](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R18-first-output-control/reanalysis/main-common-mode-v1/README.md)复核原20trace全132CTA的5280首输出角色窗口与2640原配对时长差，全部同SM。trial3/8/9的数百ns变化在未新增打点的CTA上同样出现，对应cycle中位近零。它限制仅用局部读开销解释失败，尚不证明物理频率或读取偏差；不扣共同项、不改变冻结门槛或关键CTA集合。A的guarded读请求面向cfg_b，其实际帧构建即使通过也不能自动修复本cfg_a控制失败。

[740070 guarded实际帧构建](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-guarded-frame-build/README.md)完成：cfg_b stage6三target+plain模块，CUDA12.9，130秒4CPU/16GiB，无GPU。四kernel168regs/2barrier/零spill，计算/TMA/arrival/wait静态数量相同，FIRST/MAIN两处64bit比较→BRA及有限重试保留。前序/target统一12word布局，原三overlay逐字节未变。尚未完成reader、动态occupancy或运行时包围/扰动资格，不迁为cfg_a观察通过；0c9afd和3aefacb已分别合入2a53633/8a3aec0。

[地址服务反馈独立复放](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/main-20261011-address-feedback-v1/README.md)完成，af3a8ac合入f7f4414。10条件/17响应完整报告一致，214输入SHA核对，157相关测试通过；10组旧默认完整字典及旧覆盖预测不变，5组B0 MAIN精确不变。新候选工作变化反馈ready/订阅/stage，但首输出TC idle仍0，不能解释沿M与single126等20%以上响应，固定参数点不采用，不追加方向/规模价。

guarded记录读取边界已补一层：A的bfe384d合入5876c7d，新增主线decode_r09_clock_observer packet校验三arm、12word布局、1638912byte共同准备、16前序、同进程setup/call缓冲一致及全132CTA×2角色的坐标/SM/窗口；局部错误保留原因，不回退dual10。13项helper/packet测试通过，qualified始终false。此层不替代归档source/binary/module/raw SHA绑定，也尚未在740070真实运行数据上复放；数值、动态occupancy与观察资格仍未验证。

A的shared-clock实数边界修正6e36df0合入0a90ea6，9项测试通过：无cap只能报告下确界闭界且未证明可取得，撤回旧uncapped开下界表述；旧整数预算/证书不改，历史归档追加说明而不覆盖。

C后续输出诊断已独立复放，原20进程/5280角色路径/2640配对的JSON与CSV完全一致，首L的ns差保留到原FINAL，未由后续输出整体抵消。B原生cycle复核0a1ef33合入529da26，100进程/9720窗口与原报告一致，211当前SHA核对；analyze_r18实际版本差单列。沿M/single126强响应在cycle域仍存在，AlongN小ns响应与cycle异号，不混成统一时钟修正。旧观察失败、V09评分及价格保持。

[740071 guarded三臂最小检查](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-guarded-record-check/README.md)完成：复用740070，GPU432获配单卡，三臂共12288抽样值精确，matched/guarded各264角色协议有效、plain无trace。三target动态smem215040、API最大CTA/SM=1；源码/binary/module/raw/snapshot身份链已复核。FIRST/MAIN括号最大130/138cycle只作记录检查，无全D、硬件包围或扰动资格，不启动30进程或V10。

[740071条件括号算术](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-guarded-record-check/reanalysis/conditional-brackets-v1/README.md)在假设读包围成立时，用原128ns总区间预算得到0/132同CTA常数周期率交集；独立Fraction逐对复核精确最小预算1149.598..1242.163ns。单对guarded首段cycle差中位-761.5，不能替代原10进程扰动评分；保留inconclusive/false，不调预算、不定价、不证明物理频率。

[740071头字段加严回放](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-guarded-record-check/reanalysis/main-packet-header-v1/README.md)：修复A发现的entry/producer缺失与固定设备SM越界接受缺口，补充两个时钟域事件顺序检查。冻结三臂原记录仍全部通过，matched/guarded各264角色；10个原raw副本破坏检查均拒绝，9项reader测试通过。原快照不改，物理观察资格仍false。

## 当前状态

2026-10-11，按[任务 A 绑定计划](../../../../../../results/gh200_resource_campaign/access_rules/20261011-V10-taskA/A-clock-v2/source/PLAN.md#mainline)完成任务A，终点3：没有满足本轮形式与采用规则的可冻结候选。扩大训练的时钟候选及组件留形状诊断均失败；首轮cluster项对P2和G6三个配置恒为零，无法补救。主集成独立复算固定29条件、300个dual进程的绝对分项与关键CTA，另验证12组完整预测相等及cfg_a G6的频率/时间采用矛盾。GPU作业数为0，不请求本轮补测，不进入V10或新研究。计划绑定SHA为`3f1f34a0eec29fda257fce710bda0d84228a140b8677543a4fb76f65122cdd32`。

A时间与C边界的本次普通有界任务均已交付并停止；旧持续目标仍为paused，工具拒绝新建目标，未绕过限制或标记旧目标完成。B供给保持暂停。A/C专属代码分别保存在未推送提交`6ee1c0bd`、`8b80aa76`及交付源码快照；用户已将主集成任务 A 代码、已审查文档与原 R09 修改合并提交为`3b53dbf7`，11个文件范围与代码/原R09哈希已按归属核对；保留该提交，不拆分历史。原r2默认预测和原冻结评分保持。任务结论、比较表、参数/源码来源与复放入口见[任务A交付](../../../../../../results/gh200_resource_campaign/access_rules/20261011-V10-taskA/README.md)。

以下启动段落记录 2026-10-11 的原合同；m11 的现行预算与截止以本节 2026-10-12 修订为准。

用户已批准新的[八小时主线合同](PLAN.md#mainline)与[m11 普通 SMEM 合同](PLAN.md#smem-contract)。两个独立工作树从同一经审查提交`3b53dbf721d7d30828d6b78b92103a214be1fbfe`启动；两个新目标均已启用为active。主线巴黎时间2026-10-11 23:07:35开始，最晚2026-10-12 07:07:35停止，归因检查点最迟02:37:35；SMEM于23:07:45开始、07:07:45截止。SMEM先按普通任务开始，用户清除旧目标后于23:12:20成功创建新目标，截止未延长。启动核对时两组新GPU提交均为0；后续SMEM提交以其作业清单更新。用户合并提交作为本次启动基点的例外已确认，后续分别提交各自修改。路径、只读输入、PLAN/源码SHA、预算与启动消息见[准备交付](PARALLEL-START-20261011.md)和[绑定清单](PARALLEL-START-20261011.json)；实际启动证据见[启动清单](../../../../../../results/gh200_resource_campaign/access_rules/20261011-parallel-bounded-preparation/launch.json)。

| 新任务 | 工作树 / 分支 | 准备状态 | 执行预算 |
|---|---|---|---|
| 长 K 时钟与事件组合 | `/home/jianyeshi/.codex/worktrees/gh200-long-k-event-20261011/CUDA_optimazation` / `codex/gh200-long-k-event-20261011` | 本有界主线完成，终点2：必要测量；停止继续开发 | 最多8小时，实际GPU 0 |
| m11 普通 SMEM | `/home/jianyeshi/.codex/worktrees/gh200-smem-lds-sts-20261011/CUDA_optimazation` / `codex/gh200-smem-lds-sts-20261011` | 本轮研究与交付复核完成；形式被反例否定，停止；模型未通过 | 截止2026-10-12巴黎18:00，GPU已提交2/3 |

当前主集成负责新主线目标；“GH200 输出与边界”负责新SMEM目标，每次命令显式选择新工作树。用户清除旧机制目标后工具已确认goal=null并成功创建新目标，原研究范围未恢复。时间与供给对话保持暂停。两工作树中的基点EXECUTION为历史副本；公共当前状态只在本主checkout维护。

新主线[交付](../../../../../../results/gh200_resource_campaign/access_rules/20261011-long-k-event-bounded/README.md)：归因检查点先于实现，六参考+三十开发条件/360 dual进程完成时间账；固定后续K迭代成本与dual-only时钟组合使cfg_a G6 plain误差−3.70%→−16.74%，组件留掉G6形状后−36.51%，不采用。超L2源覆盖形式在独立后续L证据中不能定价；旧cfg_c长K dual仅单输出已核对，未漏用可定价的后续窗口。150预测/2160恒等式与八项原测试复放通过，默认29预测不变；仅交付18新校准条件/660进程、一个参考卡GPU作业的提案，未提交、未冻结V10，不自动开第四轮开发。失败实现与复放脚本独立提交`a4ca14c`，不合入公共预测器、不推送。

以下三段保留修订前的失败、离线交付和 blocked 记录，原两作业/八小时限制已被后述人类修订取代。

m11作业740708失败完整保留：CUDA12.9编译成功，独立读及原CUTLASS读被PTXAS移出重复循环，动态LDS工作量不符，采样未开始。当前尚无GPU校准/留出数值；CUDA13 device-only修复SASS只是准备，不替代12.9运行资格。第二次提交未执行，原合同要求第二作业前冻结参数与预测；是否将其改为开发重试已向用户提出异步选择，答复前仅离线准备，不自改预算或将校准称留出。

2026-10-12巴黎时间00:13后，主集成核对[m11离线修复交付](../../../../../../results/gh200_resource_campaign/access_rules/20261011-m11-lds-sts-bounded/run-20261011T210745Z/README.md)及242项文件SHA，清单`1055066c4fbfd24cdd64a67e0959c8fd019fc47d67a71ebc4c9510fe0de5dcbe`一致；源提交`b555c60`、`3397207`及干净工作树核对通过，七个修改文件全部在专属范围，未推送。九个普通LDS/STS内核的CUDA13 device-only静态检查通过，native/CUDA12.9修正版执行仍待验证；新计时样本、校准参数及独立留出均为零/未完成。独立复核记录见[公共交付核对](../../../../../../results/gh200_resource_campaign/access_rules/20261011-parallel-bounded-preparation/smem-checkpoint-review-20261012.json)。截止仍为07:07:45；当前没有再次提交授权，不据等待时间自动执行。

随后m11执行对话确认，同一第二提交授权阻碍已连续出现三轮，`update_goal`实际返回`blocked`，停止无效自动继续。未标为complete；校准和独立留出仍未完成，唯一740708失败、无存活GPU作业、第二提交未执行。人类明确允许改为开发重试后，才可在原07:07:45截止与剩余一次提交内恢复；不重新计时，不追加第三作业。此前启动/交付清单的active字段是历史快照，当前状态以本表为准。

2026-10-12 人类在“GH200 输出与边界”中直接批准累计 GPU 提交上限改为3，并将截止延长到当天巴黎18:00（UTC16:00）；主集成已核对原答复，授权身份和原合同见[修订记录](../../../../../../results/gh200_resource_campaign/access_rules/20261011-parallel-bounded-preparation/smem-amendment-20261012.json)。原开始时间、失败740708及冻结快照保留；失败算第一次提交，剩余两次依次用于开发/校准和全量冻结后的独立留出。先在 CPU 上用 CUDA12.9 检查完整源码和最终 SASS，不提交额外编译作业；校准与留出保持同一 GPU UUID 和协议，指标、局部资格、native覆盖及禁止第四次提交不变。执行对话已收到答复并恢复普通任务工作，目标工具状态以该对话回传为准，不新建重复执行者。主集成本次只读查询 `squeue -u hibouwu` 返回空队列；此时累计仅740708，尚无新 GPU 作业。主线必要测量提案仍未获执行授权，保持停止和GPU0。

2026-10-12 巴黎09:55前，主集成核对本地[CPU预检查](../../../../../../results/gh200_resource_campaign/access_rules/20261011-m11-lds-sts-bounded/run-20261011T210745Z/continuation-20261012/cpu-preflight/result.json)及最终SASS审计：CUDA12.9.41的9个普通和2个真实CUTLASS内核均保留循环内每轮8条目标LDS/STS，宽度对应4/8/16B；归档SHA `d3780bc500cc1042f336852f12162742a175f0b283e096fafbfe7ac02a83ff96`匹配。静态检查不授予运行时正确性或校准资格。[提交账本](../../../../../../results/gh200_resource_campaign/access_rules/20261011-m11-lds-sts-bounded/run-20261011T210745Z/submissions.json)记录第二开发作业740961；实时squeue确认RUNNING于romeo-a048，申请4 GPU/独占节点，计时流1。累计2/3，剩余1次只供全量冻结后的独立留出；目标工具仍blocked，执行以直接人类修订的普通有界任务继续。新开发结果与模型冻结尚未完成。

2026-10-12 巴黎10:19前，第二开发作业740961已COMPLETED/0:0，耗时18m18s；提交账本记录552条件/1656进程、失败0。主集成加载本轮归档模型并独立复算540条件比较：绝对相对误差中位0.16212%、最大83.83293%，最大误差门槛失败。直接反例为16B、8warp、独立读、8192重复：stride1与broadcast均被当前需求形式压为每轮256，给出同一预测2088213.390cycle，但各三次记录分别恒为2097378/1135930cycle，SMID均124、数值错误0。任意共同预测值的最小最大相对误差为240362/808327=29.73574%，故不能仅调该形式的价格达到10%；这否定当前需求到时间的形式，不证明物理事务数或唯一硬件机制。见[独立反例复核](../../../../../../results/gh200_resource_campaign/access_rules/20261011-parallel-bounded-preparation/smem-counterexample-review-20261012.json)。按m11反证终点停止新增测量；累计2/3，未冻结、未提交第三留出、未授予模型通过或跨卡资格。专属最终报告/manifest正在整理，公共最终交付审查仍待完成。

2026-10-12 巴黎10:28，主集成完成[m11最终交付](../../../../../../results/gh200_resource_campaign/access_rules/20261011-m11-lds-sts-bounded/run-20261011T210745Z/continuation-20261012/README.md)复核：清单SHA `891ad38a7ed246dd914b14dedd8805c7188b3ec441f99a6400164dacae8d5665`及10299文件SHA/大小全部匹配，七个专属仓库文件SHA和修订账本匹配，原交付manifest保持。最终文档提交`a3e5f7ce`只改R03/EXP-09，两工作树干净、未推送；未将失败候选合入公共模型。75条>5%与24条>10%开发失败全保留，66条预列留出未执行；native仅静态SASS通过，未执行数值重放。见[最终交付审查](../../../../../../results/gh200_resource_campaign/access_rules/20261011-parallel-bounded-preparation/smem-final-delivery-review-20261012.json)。本轮研究与交付完成，模型资格仍false；目标工具结束状态由执行对话单独回传，不回写创建时blocked的历史manifest。主线和机制均不自动启动下一轮。随后执行对话实际调用 `update_goal(status=complete)` 返回complete，[工具终态](../../../../../../results/gh200_resource_campaign/access_rules/20261011-m11-lds-sts-bounded/run-20261011T210745Z/continuation-20261012/goal-completed.json)绑定上述已审查manifest；完成指有界研究反证终点，不授予模型通过。工具累计活动时间3926秒（约1小时5分），不含工具blocked期间的普通续做，不当作完整墙钟耗时。

原分项、关键CTA及跨卡联合要求仍未完成；V09仅作开发，拟支持范围的成绩不改写原冻结评分。以下表格与后续逐批记录保存此前研究状态，本轮任务A结果单列于[新分析目录](../../../../../../results/gh200_resource_campaign/access_rules/20261011-V10-taskA/)。

| 对象 | 已有结果 | 当前工作/边界 |
|---|---|---|
| 管理 | 接续对话`01a129c0-b787-7e03-adf1-fcfdb2afa149`；740061完成同ef卡40配对样本及20参照；数值与身份通过，两stage上游扰动门槛失败 | `gh200-v09-failure-analysis`持续开发；未进入V10；结论归[R13](R13-async-retirement.md#a-source-rotation-preparation) |
| 供给 | R13在途源共享、订阅截止与局部E0排他均保留失败；740033 A源轮换40进程、主观测10/10有效但未唯一识别机制 | 主线已升级r13-release-threads-v2；128线程全D及严格解码已通过；740061上游绝对差stage4为64.5/848、stage6为8.5/1020cycle，观察资格失败，不采用条件机制结论。B新R10候选失败已独立复核；现行预测仍未通过 |
| 输出/边界 | [R18](R18-cluster-boundary.md)同cluster轨迹相关；源码证明full/empty跨rank依赖 | 将依赖作为候选约束，不直接收费；P2最后cluster仍在冻结候选外 |
| 时间/输入 | [R09](R09-inkernel-clock-stages.md)700次连续调用显示服务与频率均有历史响应；single也复现变慢 | 单极状态留段劣于末值基线，不采用；镜像前序后目标仍呈晚段成本，重复目标地址不是必要条件；批次封顶计数不能定位调用；同调用首/次输出周期率不同，已与供给组联查；本组暂无独立GPU任务 |
| V09 | 739011完成30条件；29受支持预测4.11%/35.46%，1冻结前拒绝 | **整体未通过**，冻结/判定不改；新候选须另用新留出检验 |
| 计数器 | 既有权限拒绝证据可复用 | 受支持入口仍未确认，账户持有人联系管理员 |

当前状态只在本节维护；具体实验分析归原页。历史A/B/C长期对话ID、工作树、提交表和故障记录见[固定状态表](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/EXECUTION.md#status)，不把旧绑定或旧“待补测”状态当作当前任务。

主集成已接入并独立重算[740068共同时钟分析](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R18-first-output-control/reanalysis/main-clock-consistency-v2/README.md)：20进程的160证书与98560次反演通过，第一L无上限模型仍需97..113ns单端点容差，原观察失败不变。另接入[ cfg_c源/full端口](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/main-cfg-c-source-ports-v1/README.md)，26静态条件、150历史调用、18960CTA、614400值及19个压缩绑定/模板独立复核；cfg_a22绑定不变。cfg_c完整rank生命周期仍未闭合，开放端口不当作零成本；C继续补源码依赖，无新GPU或价格。

[740070/740071证据连接](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-guarded-record-check/reanalysis/main-frame-smoke-evidence-v1/README.md)已主集成审查并执行三份复核：860源/四binary/SASS、真实三臂12288抽样值、当前reader10个破坏副本拒绝。编译1024B静态shared与运行215040B动态shared分列；旧reader缺口与修复版本同时保留。物理包围及局部/空间扰动仍未资格化，无新增设备运行。

[cfg_c跨K原角色诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/main-cfg-c-cross-k-v1/README.md)新增60历史进程/245760值/12960匹配窗口对复核；18组C4096/(4C1024)−1中位为−1.952%..+1.178%，ns首轮−7.165%..−4.189%、后轮+10.899%..+15.070%。保留全部角色/工作量，不拟合价格；K伴随A行距变化且观察资格未建立，只作为闭合生命周期后需解释的开发约束。

[guarded共同可变时钟直接反例](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-guarded-record-check/reanalysis/main-guarded-common-curve-v1/README.md)：主集成从原raw复核528戳，138证书/1584交集通过。SM57/68相同18880ns起止读数给出[33043,33198]/[33573,33720]cycle，零额外端点平移下相隔375cycle；无需常数率或跨SMclock相减即可解释共同F负闭环。全模型最小整数额外单端点容差有/无cap为187/177ns，仍不证明硬件包围、物理频率或观察资格。

[cfg_c角色边界分解](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/main-cfg-c-role-boundaries-v1/README.md)完成同60原raw、12960同CTA角色对精确恒等式复核。aligned FIRST差中位−7..+6cycle，MAIN差−635.25..−529.5；源码MAIN在mma_tail最终wait<0>/release之后，角色L差不能当裸WGMMA服务差。完整逐窗口与源码SHA保留，观察资格仍false。

[cfg_c跨K全窗口补查](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/main-cfg-c-cross-k-v1/README.md#全窗口条件迁移检查)：12960窗口以4×K1024实测作条件迁移，按K4096目标分母的cycle误差最大4.809%，无>5%；ns最大17.378%，3921窗口>10%。精确分数验证，不以分组中位掩盖尾部；这种条件输入仍不是静态自由预测或验收通过。

[cfg_c完整rank图主集成复核](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/main-cfg-c-rank-dependencies-v2/README.md)完成：接入C d6422c9，44测试通过，12完整图byte同、341输入SHA匹配，覆盖26静态条件/1556cluster/7148rank输出。接入前cfg_a三完整图、cfg_c三本地图和cfg_a arrival routes逐字段不变。源到四stage复用/两WG尾部/输出完成逻辑闭合，entry与服务仍未定价；不增加预测或观察资格。

[cfg_c跨K输入可用时点](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/main-cross-k-input-availability-v1/README.md)：独立查60raw/30配对，12对所选参考晚于目标开始、6对warmup不同、2目标无任何先前同pitch小K参考。原误差不改、无替换参考；限制为离线条件诊断。A245760数值及12960条换算恒等式复验通过，30条host差与主集成同。

[cfg_c闭合图历史顺序检查](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/main-cfg-c-observed-order-v1/README.md)：150既有dual调用、9480物理cluster、6492800可达标记先后关系均无GT反转，含跨rank约束。仅零lag偏序相容性，无内部服务价、完备性或观察资格结论；V09未读取计时。

[源/目的延迟重分配独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R13-stage-events/reanalysis/main-latency-relocation-v1/README.md)：8对模型405504需求与19字段/全actor/目的时线一致，6SHA匹配；另从真实前版加载比较8个默认完整字典全部同，165测试通过。整box共同L搬到目的后程等价旧service_end截止，继承其固定候选失败，不作为新机制或追加GPU区分；无新价格/资格。

[cfg_c完成边界不可识别见证](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/main-cfg-c-completion-identifiability-v1/README.md)：同真实cluster全部36端点固定，未定价图producer_tail或D_visible仍无上界，三份逐边合法赋值给出不同完成时间。任意1000ns只作数学见证，不称实际延迟；consumer FINAL不能替代全部完成边界。另发现绝对GT加浮点容差导致假冲突，脚本先整数减原点规避，C继续修复比较。

[大整数bounds修复独立回归](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/main-large-integer-bounds-v1/README.md)：接入C e6eda283，先lower−upper再比1e-9；47测试通过。预定150调用各首cluster，940350节点区间绝对/相对整数平移逐字段一致；旧696e0b5对这150绝对值全误拒，相对值全通过。原直接整数偏序及减原点完成见证不受影响，未改变图边/价格/资格。

[cfg_c producer-tail源码准备](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-preparation/README.md)：新增独立prepare，固定S4/2304×3072×4096，thread0于load_tail返回后写head14/15；同binary16次前序关闭新增observer。856源SHA与唯一hook投影验证通过，trace分配不变；目标编译/SASS、decoder、全D与扰动资格仍待完成，无设备执行。

cfg_c producer-tail补齐[严格decoder与7项合成拒绝测试](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-preparation/README.md#解码接口补充)：固定132CTA双WG完整覆盖、前序和on/off尾戳，保留CTA错误；不强制tail与consumer FINAL顺序。尚无目标编译或设备记录资格，源归档身份不变。

[cfg_c elected-tail目标构建740082](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-build-v2/README.md)完成，4CPU16GiB无GPU。按C反馈撤回初版thread0端点证明，v2同elected lane于pipeline.producer_tail之后采样；SASS四槽快/慢等待→flag→CLOCK/GT→head14/15核对通过。168reg/2barrier/零spill，856源身份保留；无设备数值/全D/扰动资格，下一步先补正确性入口。

[cfg_c producer-tail正确性740088](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-correctness/README.md)完成：740087补host全D入口，SASS/资源byte同740082；GPU-fd5317a4获配单卡，7077888全D off/on逐位且独立参考精确，两warm各132CTA严格有效、8192抽样值精确。身份与实际卡分别保存；不计跨卡性能或扰动资格，不将单对当正式对照。

[cfg_c闭合供给影响范围独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/main-cfg-c-closed-response-v1/README.md)：12图byte同、349SHA匹配、30测试通过；拓扑最短距离独立核对126查询/100路径/26无后续full。B完成q=0最早remote full为g+4，q>0为g+3；B issue为g+1。短K可仅影响producer退出而不达伙伴D，可达仍不是服务时间或正响应证据。

[guarded联合读取界独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-guarded-record-check/reanalysis/main-guarded-robustness-v1/README.md)：接入A37b10506，1091输入SHA/528原raw端点/46证书/25110边/10560交集通过，相关100测试通过。ε=32ns时四端点证明κ至少105cycle，完整图在此可行；容差是条件相容变量，不是硬件误差测量或校正量。物理时钟与读取误差仍不可区分，原观察资格及预算不变，无GPU运行。

[迟发响应独立重算](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-producer-delay/reanalysis/main-startup-absorption-v1/README.md)：接入B e541c478，18次模型/373248需求、2592旧窗口复现；803共同SHA及4路径不同模块SHA匹配，完整结果相同。+512ns后初始订阅仍余256.945ns，首复用仍余708.862..772.343ns，三候选首full/FIRST/MAIN不变、producer→MAIN预测−512ns，与旧记录非负变化不符。失败定位到首次供给吸收迟发，保留后续D时刻差；无新价格、数值重放、GPU或资格。

[producer-tail独立正确性审查重跑](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-correctness/reanalysis/main-producer-tail-correctness-review-v1/README.md)：C脚本主线执行结果逐字段同，856源/7077888参考值/8192样本/四拒绝路径通过。另完成[增量扰动对照规格](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-control-preparation/README.md)：20个交替配对命令、全CTA/角色/输出共同端点及L门槛明确，尚待采集判定器和冻结，未执行GPU；通过也不自动授予tail精度或基础dual资格。

[producer-tail对照判定器准备](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-control-preparation/README.md)：逐系列分析及串行采集器已实现；两旧smoke包各2556系列/132CTA读取通过，不评分单对。14项相关测试覆盖共同平移相消、单次超限、event/CV及缺失/错序等拒绝；采集前实际设备setup检查。仍待独立规格审查、独占远端脚本及完整冻结，无新增GPU、无观察资格。

producer-tail冻结前按C反馈补齐W/E/GAP/TAIL直接差门槛，防止两个共同端点分别通过而派生窗口超限；现每进程4116系列，15测试及三种错误分配拒绝通过。远端命令固定a058独占/4GPU，脚本检查288CPU/4GPU/OverSubscribe=NO及实际UUID；仍未冻结或执行新GPU。见[对照准备](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-control-preparation/README.md)。

producer-tail按C最终审查再补2256条每调用min-entry共同原点ns系列，防止各CTA单独归零隐藏跨CTA相位移动；当前6372系列/进程、16相关测试通过，禁止跨SM clock相减，整调用launch边界仍不获资格。所有修改均在冻结和新采样前。

[cfg_c producer-tail对照740091](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-control/README.md)：b11fae8冻结后参考ef/a058独占执行20进程，81920样本、完整记录/身份通过，CTA→SM配对无变化。6372系列4771失败，432条L全失败（逐条max320..1280ns）；event中位0.446%/最大1.136%、off CV0.310%通过。增量观察资格仍失败，不补抽/调门槛/校正抵消；未识别差异物理原因，无V10或跨卡性能结论。

[每需求最小延迟独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-producer-delay/reanalysis/main-input-latency-v1/README.md)：接入B24b85629，12场景/248832需求全字段重现，804共同SHA+5模块SHA匹配；真实前版6dce665默认12完整字典相同，172测试通过。独立分数FIFO递推给出FIRST=max(V,δ+τ+144)，固定初始池/有界δ条件下任意共同τ均有ΔFIRST≤δ，仍无法解释旧20对正S；停止拟τ，不增加GPU或资格。

[历史静态归一化重算](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/main-static-history-equivalence-v2/README.md)三个完整JSON内容与A相同，80raw/2120调用/1160末值预测匹配；静态输入签名相同，T0比值退化为1，跨卡拒绝保留。[共同倍率同批下界](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-common-history/reanalysis/main-switch-multiplier-bound-v1/README.md)直接复核十份原raw，trial9/5使10%倍率区间无交集，精确minimax10.521266%；只排除共同末值倍率，不排除完整历史状态，无新GPU或资格。

[740091两域/轮次失败诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-control/reanalysis/main-window-domain-diagnosis-v1/README.md)：4320窗口精确换算恒等式、1680同CTA/WG双输出对保留。pair1的168角色全首后轮ns差异号，观测C/N配对变化的轮次对比中位+2.926%；pair9为−2.012%，cycle分组中位多近零但不授资格。C独立冻结/raw/数值/门槛复核归档，原失败不变；无物理频率或tail开销归因，无校正/新GPU。

[tail与首轮MAIN因果边界](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-control/reanalysis/main-tail-causal-boundary-v1/README.md)：T1/T2共16源码可达查询，T2首MAIN→本cluster两tail成立，T1无此顺序。740091十on的最早tail均来自T1，仅14..48/264首MAIN读数早于全部tail，不能由本cluster顺序排除跨cluster影响。保留物理读取边界，不归因、不校正、不改失败；若补测，优先相同流程两臂off基线辨别配对背景变化。

[两臂off基线740092](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/README.md)：759fa55冻结，同binary/ef/前序/门槛20进程，实际flag全0、2640CTA尾槽0、81920样本及身份/解码通过，SM无变化。4884/6372系列失败、432L全失败，event中位0.272%/最大1.711%、arm0 CV0.794%通过。新增tail不是本次局部差异出现的必要条件；不能据此证明无扰动或扣除740091失败。17测试及旧完整判定回归通过，无资格升级。

[前序event统一比例诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/main-prefix-scalar-diagnostic-v1/README.md)：740092末次前序预测整event中位0.462%/最大2.021%；但即使使用两臂目标cycle和arm0目标ns，432条L条件残差仍全超原64/256尺度。4320精确恒等式、全部16项前序变化及源码可用时点保留；仅排除这一个零参数比例代理，非自由局部预测/物理频率结论，无新GPU或资格。

[off/off原生cycle重复性](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/main-native-cycle-repeatability-v1/README.md)：L4320配对相对差中位0.1965%/最大2.5100%；W绝对差21/195cycle但相对中位12.7119%，E/GAP/TAIL尾差保留。逐坐标常数oracle仅诊断，不能追认cycle资格或替换ns失败。C独立740092冻结、arm/flag、6372系列/81920数值及门槛报告归档，无新GPU/拟价。

[cfg_c共同计算窗口静态参考](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/main-cta-compute-envelope-v1/README.md)：tile工作268435456 FLOP/1024逻辑m64n128k16跨两WG，条件4096 FLOP/SM-cycle给65536cycle。4320同CTA minFIRST→maxMAIN为65776/65943/67553cycle，参考误差中位0.617%/最大2.986%；非冻结预测或普遍下界。历史N128-Kmajor/另一卡与MN-N256证据缺口明确，不移植实测价格，仍需独立同卡形状/组织参考。

[cfg_c MMA参考740094构建](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-mma-reference-preparation/README.md)：派生独立N128/MN/two-WG/eight-MMA组入口，保留R00原probe；无GPU编译841源，SASS八HGMMA/两链、ARRIVE/wait1/final0核对，154reg/零spill。驻留重复操作数与checksum计时边界差异明确，未执行数值或计时；下一步固定64/1024两长度×3进程同卡参考，无新GEMM矩阵。

[同卡MMA参考740095](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-mma-reference/README.md)：0f357f4冻结两长度×3进程，ef/a058独占、实际SM124；196608正式FP32值独立精确，坐标短检查通过。64组均66339cycle，1024组均1049379cycle，持续4092.865682 FLOP/SM-cycle；各比4096条件参考多803，不拟/扣截距。只补同卡指令参考，不转成完整GEMM/全SM/跨卡或观察资格。

[同卡静态compute候选](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-mma-reference/reanalysis/main-static-compute-candidate-v1/README.md)：独立长窗口参考给K4096预测1049379/16cycle，无目标残差/截距；740092全4320开发窗口误差中位0.541%/最大2.912%。15旧cfg_c多K/行距条件实为432卡，入口全部拒绝直接ef价格迁移。3语义测试通过，仅compute实验候选，不授观察/留出或完整范围资格。

[共享FIFO前缀独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-producer-delay/reanalysis/main-shared-path-prefix-v1/README.md)：B043d06b8重算报告/2376需求同，804SHA+2模块匹配（演进工作树源码由原Git blob恢复）。独立未聚合FIFO70检查及高价上界292/165通过；回程512ns迟发最大S344.582ns不足，独立请求路径保留必要σ区间0.698602..0.917595，未选价/完整采用。无新GPU或资格。

[共同倍率两种精度界复核](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/main-common-converter-bound-v1/README.md)：接入A229cd678，119SHA/40raw/12960窗口/8640端点/40最优证书通过，主线独立区间交集及两窗口下界同。部分pair绝对oracle仍>256ns，但最优最大相对残差on/off≤1.721%、off/off≤0.920%；不能据绝对gate失败强制增加ordinal状态或否定全时间5%/10%。仍是读取目标的条件oracle，不采用/改gate/授资格。

[MMA参考独立审查重放](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-mma-reference/reanalysis/main-mma-reference-review-v1/README.md)：C报告重跑SHA完全相同，交付及全部输入SHA核对；841源、196608正式值、4320旧同CTA窗口通过。明确起始描述符设置、末尾128 FADD/同步及ScaleOut/producer组织差异，不把参考窗口等同GEMM服务，不扣803cycle，不授观察或完整预测资格；无新GPU。

[冻结r2包络主线重算](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/reanalysis/main-envelope-contract-v1/README.md)：接入A30abc0ba，三个完整JSON重算相同，14SHA/30条件/3516CTA/1113分支/45包络段及三负例通过；360数值核对最大相对差2.0e-15。固定公式正根唯一，不能换根修复误差；29预测/1不支持保持，不推广到新供给/历史或新卡，无新GPU或资格。

[请求路径完整图重算](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/main-input-path-v1/README.md)：接入B fad0d6b1，16图/384192需求结果全同，1017共同SHA+6模块匹配，179测试通过；独立旧版32全字典/1152 FIFO需求检查通过。固定中点B0 S响应出现但B32首窗、空间跨度和R13 S4预等失败，不采用该点。1116首轮窗口模型空闲全0、L全4608ns，下一步检查原必要区间的停顿分支而非另拟标签价格，无新GPU或资格。

[请求路径必要区间端点诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/main-input-path-endpoints-v1/README.md)：仅原上下端20完整图/279936需求，2232首CTA及33480供给组；首轮空闲全0，最小ready余量下端66.731/上端45.755ns，L仍4608ns。排除中点独有现象，不证明区间内部；B继续全区间证书，主线不扫描拟价，无新GPU或资格。

[同卡cycle/ns迁移主线复放](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-mma-reference/reanalysis/main-cycle-time-transfer-v1/README.md)：接入A1b667800，三个完整JSON重算相同；165SHA/6参考/40目标/8640整数窗口与有标签分项/80同SM窗口、两负例通过。工作量缩放周期最大误差2.912%，直接参考C/N换算ns最大15.075%；同SM也不直接迁移。后验参考与端点/负载差异保留，不归因物理频率，不加校正/资格，无GPU。

[请求路径最小余量分支核对](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/main-input-path-witness-v1/README.md)：仅重放原两端极小窗口，FIRST/余量与原报告逐值同。下端g1由源完成限制，ready间隔221.269ns；上端由请求路径限制，间隔264σ=242.245ns，距compute288余45.755ns。两端限制分支不同，不能线性外推或据聚合吞吐作全区间证明，无新价/GPU/资格。

[请求路径区间比较预审](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/main-input-path-interval-review-v1/README.md)：对B未提交源码快照独立检查300比较/2520严格内部点/20相等单点/6堆同刻排序通过。仅比较基础机制，不是全图证书；最终仍需覆盖全部开区间及单独边界，保留分段上限后的剩余域，无GPU/资格。

[区间分段成本诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/main-input-path-cell-cost-v1/README.md)：仅增加计数、原v2中点证书逐字段相同；259409非恒定比较中236882为heap排序（91.3%），最终两侧界来自heap弹出，当前cell只占原域0.000384347%。不能据此估计总cell数，也不能跳过资源排序；建议用有证明的偏序/保守界缩短路径，原未覆盖域保留，无GPU/资格。

[single126源并集独立核对](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/main-single126-source-union-v1/README.md)：B0/B32共1024不同描述符/6553600有效FP16元素独立地址枚举与128B覆盖函数同；每组总单元3200/3248，16组各120组对不重叠。仅静态工作前提，单次服务界仍须证明同组订阅不失效，不称物理流量或全区间结论，无GPU/资格。

[前序event局部候选主线重放](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/main-prefix-ordinal-clock-v1/README.md)：接入A736ac190，五份完整内容重算相同；167SHA/2160校准/4320评分/8640有标签分项与输入拒绝通过。共同系数自由局部误差3.541%/6.446%，两序号0.798%/3.437%，保留简单共同候选，不称序号必要；仅后验开发，同cfg_c协议，仍需新有效观察与独立冻结采集，公共组合未采用，无GPU/资格。

[请求路径闭区间排除主线验收](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/main-input-path-bound-replay-v1/README.md)：接入B689edf8f，1029SHA/完整重跑一致、183测试通过；独立756包络/1722闭段/630余量/2542断点及后十组队列递推通过，原producer_gap保留。必要区间全域single126 B32首L=4608ns，无法给出正B32响应；停止该固定模型的同区间调价。仅有理数模型条件排除，不扩大队列家族、不授资格，无GPU。

[前序输入与预测保存时点核对](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/main-prefix-availability-v1/README.md)：绑定原构建源码，prefix event在目标前host可用、目标后report才输出；现binary未执行数值预测预提交。优先保持binary、预冻结确定性规则后仅用前序字段评价新数据，明确后处理时点；不等同观测有效性，C独立审查最小控制证据，无GPU/门槛变更。

[共同系数规则入口主线复放](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/main-prefix-clock-local-rule-v1/README.md)：接入A7d67c306，五文件逐字节同，20前序/40预测、23负例、固定SHA与拒绝覆盖/隔离目录重放通过。规则只读setup+16前序，不改binary；预冻结规则后处理不等于数值预提交，旧观察与新采集缺口保留，无GPU。

[前序输入提取边界](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/main-prefix-extraction-v1/README.md)：新增固定raw身份的静态/前序投影工具，20已知记录目标毒值不影响输入，异buffer/无效pointer/缺历史及SHA/覆盖拒绝通过；目标正确性和trace独立后验检查，不能据预测成功筛样。接入A271e86de并重核时点证据，规则未变；尚未新实验冻结，无GPU/资格。

[前序投影协议一致性修复](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/main-prefix-extraction-v2/README.md)：按A独立发现补22项call/setup静态声明检查；20原记录440矛盾/440缺失拒绝，目标毒值仍不影响输入，与255e93a旧版20合法输入逐值相同。固定raw身份不能替代协议自洽；规则未变，C继续原gate残差否决检查，无GPU/资格。

[请求工作静态审查主线重算](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/main-request-work-audit-v1/README.md)：接入B6c5015b2，14条件/305856需求/34176计数检查及1318SHA通过，完整报告/张量一致。两方向全计数张量相同却响应异，single36/126前36局部源程序同却规模响应异；需要源关联/共享影响信息，不能仅换覆盖计数定价。保留旧尾部不支持和跨卡计时分离，无新模型/候选/GPU/资格。

[前序规则观测否决重放](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/main-prefix-observation-review-v1/README.md)：C结果重跑同、33SHA核对，216共同系列扣除冻结预测配对变化后全超原64/256诊断尺；相对预测通过不代表绝对背景受控。GT→clock→GT目标内区间只是待审实现提案，需采样顺序、独立shadow及扰动/null证据，不授新资格。另[A投影修复审查](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/main-prefix-projection-review-v1/README.md)两完整报告重跑同，260协议反例由接受变拒绝，合法预测/数值失败保留，无GPU。

<a id="goals"></a>

原目标模式启动文本保留在[固定历史版本](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/EXECUTION.md#goals)。文档中的旧模板不激活任务或自动授权GPU；当前执行以用户在对话中已确认的具体范围为准。

[目标内 GT 读取依赖构建](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-bracket-build/README.md)：740156 CPU-only 编译通过，固定 cfg_c FIRST/MAIN 的 GT→clock→GT 真实结果依赖在两处 SASS 保留；原 event 槽不变，独立 shadow 9504bytes，16次prefix关闭，target同binary开关，producer-tail恒关。168regs/2barriers/0spill，原856文件仅host/实际trace头变化；740155错改未引用副本的失败保留。仅编译链证据，尚无新GPU记录、reader或观测资格；旧prefix binary约束不可直接复用，下一步先严格解码与最小数值/区间诊断，不进入V10。

[GT读取区间同卡最小诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-bracket-smoke-v2/README.md)：86fd7a8预冻结后740160排他a058/原ef卡，7077888个D全精确且off/on逐位同，两个warm各完整132CTA/4096数值。864stamp的GT域宽度中位32、max128（58个0不等于物理零误差），432窗口与[A精确区间模块](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/main-gt-bracket-contract-v1/README.md)重算一致，全部资格false。新旧reader+A共21测试；[C排序独立审查](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-bracket-build/reanalysis/main-ordering-review-v1/README.md)883SHA重放结果同。740159 setup参数错误在启动CUDA前退出并保留；下一步同binary完整扰动/null控制，未启动V10或复用旧binary校准。

[原始R10边界账本主线重放](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-boundary-ledger-v1/README.md)：B f521a01f接入e92f3566，71SHA（三份完整ledger）重算相同、188R13测试通过、既有供给模块AST除新静态函数外不变。旧2600×3000×2000每条件有效源501600000B、目的fill26882304B、完整期待528482304B；672全零源B1盒仍保留目的义务。独立形状守恒复核通过；原A stride4000的跨K覆盖重叠不能沿用新几何不交前提。仅静态边界账本，旧CTA/ordinal/global-stage与时间图支持未完成，无新候选/GPU/资格。

[GT observer与null必要控制失败](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/README.md)：79a86b8预冻结后740161同ef/a058排他，40进程/163840数值/全部132CTA记录有效。off_on4912/6372、off_off4729/6372点系列失败，两组L均432/432；event与CV通过、SM映射不变。on8640读取宽度max96仍不解除局部失败；off缺GT_before，不能补零构造paired interval或以批次相减。独立raw槽位复核8640配对L与event/CV同；26测试通过，全部资格false，C独立完整复核已请求。不重复同控制或采on/on以求通过，不启动V10。

[读取区间跨次可行性](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-repeat-interval-v1/README.md)：只复用740161十个enabled目标，432系列/4320窗口在零附加GT误差条件下均无共同精确时长；最小被迫跨度192/中位576/max1344，闭合缺口的单GT误差预算下限48/144/336仅为条件可行性阈值。事后每坐标常数覆盖全部盒的相对半径最大1.764%，故绝对gate失败不推导5/10目标不可能；无新价格/预测/采样/资格。[C生命周期适配重放](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-bracket-build/reanalysis/main-lifecycle-brackets-v1/README.md)18测试、24共同+9迁移SHA和3完整payload一致；864锚点/66cluster可行，但FINAL/source_safe_exit/CUDA完成仍无上界，不能据此缩小最后者集合。

[旧R10调度与ring绑定主线复核](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-boundary-binding-v1/README.md)：0b17f8b2接入92ce1aac，88SHA、3gzip完整重算同，193R13测试；独立逆向矩阵与元素地址核对1512输出/48384组/768覆盖cohort/762邻组交集。旧CTA0–3 lastWG0是j2/(12..15,12)，只有K尾部，global64..95从slot4开始；不把四局部窗口当首输出/MN尾部服务。A跨输出K末→首仍交1950units，仅几何关系，无新定价/时间支持。[740161完整C审查重放](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-control-review-v1/README.md)137SHA、完整6372×两批结果同、163840独立数值及26测试通过，原局部失败与资格false不变。

[同进程首/次输出率区间主线复核](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-bracket-smoke-v2/reanalysis/main-gt-window-audit-v1/README.md)：A c3759a7接入c37b8caa，740160的三份完整分析及独立verify逐字节同；17输入/14冻结/858源码/648窗口/252配对和4负例通过。84共同与168同WG首/次率区间均分离，共同保守比值差9.745..10.969%；仅条件精确常数率拒绝，不授物理频率/观测资格、不直接否定5/10误差目标，不覆盖740161控制失败。

[旧R10窗口工作口径核对](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-window-work-normalization-v1/README.md)：41SHA/30旧trace进程/120窗口重现原cycles/32及完整统计。j2内部输出有效K2000/密集tile K2048工作分别65536000/67108864 FLOP，换算固定128/125，仅改变计量单位，不改变三行距响应。首full等待在窗外、后续等待/control/尾release在窗内，不作裸MMA/SM吞吐或跨卡服务价格。无GPU、新价、模型修改或资格；B继续独立核对完整MMA与阶段依赖语义。

[十进程同调用率区间复验](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-output-rate-interval-v1/README.md)：A b118af97接入bb310223，三完整分析及独立verify逐字节同，67输入/16冻结/6480窗口/2520配对/4负例通过。840共同+1680同WG首/次率区间均分离，共同比值差7.500..11.193%（中位9.529%）；仅条件精确常率排除，原控制失败不变，无物理频率/预测资格、新价或GPU。冻结程序bundle含独立verify，主线按原布局复制并核对公共模块身份，不跳过SHA。

[旧R10完整MMA与输入退出复核](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-boundary-lifecycle-v1/README.md)：B ff4a7423接入8a5413a0，两侧18SHA、完整报告重算同、199R13测试；独立轴工作守恒/672释放指针/6槽位退出证书通过。每K组固定8 atom，K16仍保留6个零有效K调用；每输出31普通释放+1tail释放，empty须128到达。输出切换q5才依赖前输出tail，producer_tail仅关闭输入阶段；零源fill、empty完成与旧pingpong输出接口仍未定价，无GPU/新资格。

[旧首输出记录分段重放](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R18-first-output-correctness/reanalysis/main-record-partitions-v1/README.md)：C交付7SHA、各37输入SHA及完整报告/4数据文件一致；48角色/288分段/144有符号差、96调用内和48差值闭合、16384数值通过。740067为432卡非排他correctness采集，各map的prefix沿用自身目标参数，不能称相同历史状态或map因果性能比较。保留全部原记录、控制失败和资格false，无新GPU/模型/价格。

[零宽GT联合双点反例主线复核](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-gt-zero-width-witness-v1/README.md)：A 30e5a211接入ef02d2e5，8交付/88输入SHA、2完整分析及独立verify逐字节同。11进程9504三元组/29040同SM有序双点中55个零宽GT同坐标却CLOCK不同（54×8cycles、1×12）；拒绝e=0精确物理盒与同SM单值非递减CLOCK的联合前提，不识别乱序/实际GT误差或拟合e。边际代数与零lag生命周期可行性均不能认证联合物理锚点，原控制失败/资格false保留，无新GPU/价格。

[零源目的完成接口主线重放](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-boundary-completion-v1/README.md)：B 12032695接入01592905，两侧12SHA、完整报告及3压缩端口数据一致，205R13测试。每条件48384目的盒保留3558fill义务（含672零源B1），完整528482304B预算不按有效源缩减；移除地址/stride后的工作与源身份等价投影三行距相同，无服务价格或新时间支持。

[旧R10输入必要依赖组合](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-input-composition-v1/README.md)：27SHA/三行距相同必要图/拓扑无环，132CTA的16128已用组均被producer退出依赖覆盖，504尾部、372输出切换及240同WG复用关系通过祖先核对；672零源fill路径保留。消费者端口为128线程聚合界，输出内部仍待C绑定，服务/D可见/kernel时间均None，不作零lag预测、关键CTA或时间资格。

B对上述组合完成限定源码边审查，无行动项；16项定向父边/祖先检查通过，主线核对脚本及保存图身份并归档审查回执。`release_emit_all`固定为实际arrival发出之max，不能换成返回后打点；输入producer退出仍不代表最终D或kernel完成。

[旧pingpong D接口主线重放](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-legacy-D-lifecycle-v1/README.md)：采用C正式v3，10交付SHA、各15输入SHA、完整报告与3ledger一致，18CPU测试通过。主线指出的逐线程own mma_tail→epilogue wait返回已修复；每条件4032子块含178全越界，仍保留源协议与独立D可见义务，不迁移cooperative输出图或价格。

[旧R10输入/输出必要依赖组合](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-lifecycle-composition-v1/README.md)：3156对应聚合端点连接、132CTA全图无环；129024源码线程读路径被源生命周期要求覆盖，3854有效D可见义务仅进入CUDA完成要求，178全越界不虚构有效写。372次下一WG MMA/D解耦、372次epilogue顺序和240次同WG复用检查通过；129024线程级输入端口仍为符号，所有时间/服务未知，无GT锚点/预测资格。

[C端点映射审查主线重放](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-lifecycle-alias-review-v1/README.md)：每侧9输入SHA、5共同+4重定位程序相同，完整报告除路径一致，无行动项。CTA0/107/108精确图并集及12符号顺序扩展保留线程提前到达、下一WG MMA与前D tail分离、D全局可见晚于源关闭的可能性；不把这些必要图见证称作物理时间或联合执行证明。

[旧／新R10接口对应主线重放](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/main-boundary-transfer-v1/README.md)：B 59b1ee18接入3ac0938d，各37输入SHA、完整报告与4账本相同；12核心头文件匹配各自manifest且字节相同。Kt32→16后持续状态重建，源keys与旧full-tile API一致；新quartet纯工作/ring相同，raster改变源等价关系，pitch只改实际地址而不改等价关系。未重复未改helper测试，不转移价格、观察或时间资格。

[同 SM 原生 L 一致性诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-native-L-consistency-v1/README.md)：40原进程/45输入SHA/8640配对差匹配原分析；off/off相对原生周期差最大2.484611%，不将全部差异归因于ns换算。原64/256ns门槛及两批L各432/432失败保留；四个执行的合成反例区分点差、均值、中位数、尾部及准确度，不作为GPU证据。未设新门槛、拟合、启动GPU或授予资格；未来统计对象须单独明确。

[cfg_c G2共享供给参数与时钟联合区间](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/reanalysis/main-g2-coupled-bounds-v1/README.md)：冻结source/15输入身份，64个单输出CTA共享θ的163次LP并求时钟闭合，plain条件区间17.099059–17.442896µs（宽2.010851%）。一维可行线段/仿射交点独立核对端点差≤2.65e−10µs，21组原预测器重放通过；两端关键集合分别64CTA并列与62/63并列，仅代数结果。独立读job739011十plain/20SHA得中位17.872000µs，整个区间仍低估2.401–4.325%；仅等价供给参数无法消除偏差，不包含其他组件/观测不确定性。原unsupported及评分不变，无新GPU/模型/资格。

[中位数采样能力诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-median-sampling-capacity-v1/README.md)：复用45SHA/864系列/8640配对；精确二项尾与1024符号枚举核对。在未插值次序统计量+Bonferroni、示例95%且采样假设成立的条件下，全432 L一批/两批有限区间至少每臂16/17次；非通用最小量，也不保证精度。旧十次逐坐标两臂ns区间均含零，宽度中位800/1120ns，不能判等价或全坐标通过；中位数之差也不同于配对差中位数。采样独立同分布未证明，原门槛/失败/资格false保留，无新GPU或新标准。

[局部中位数等价候选协议](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-median-protocol-v1/README.md)：落实全部6372系列×两批×两臂的数值判定器，候选95%同时覆盖、±64ns中位数差容差、64 block共256进程随机交错表；仅提案未冻结/执行，样本量不保证精度。52SHA/40原raw重提取，12744原点控制逐项复现4912/4729失败；n10回放全部覆盖不足，不追认通过。8定向测试通过；统计等价与观察/时间戳/关键CTA资格分开，全部资格false。B/C/A继续限定joint端点与统计组合审查，未改公共模型或启动GPU。

[R10联合端点统计主线复核](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R10-producer-delay/reanalysis/main-paired-estimands-v1/README.md)：B 1eb0b5ec接入ad34fe27；两侧90SHA（86共同、3重定位字节同、1新增其他trace分支）核对，完整报告除输入路径/模块身份同、两gzip账本同。独立原始字复验5280CTA/52800时长/13200响应向量通过。B32分段响应中位数之和336ns而联合窗口288ns；B0配对响应816ns而条件中位数差832ns。已接入候选协议联合记录/独立预列估计对象要求，未改统计目标、容差、旧控制或资格，无GPU。

[中位数组合精确反例主线复放](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-median-composition-contract-v1/README.md)：A 46690750接入1ce87c36，9交付SHA、完整报告及有限CDF验证字节同；四合成总体/6换算状态、6种rate和3种stage列置换，4错误组合主张拒绝。只证明边际不能一般决定C/r、sum、max及逐次最后者，不是GPU误差或新价格。

[中位数输出组合边界主线复放](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-median-output-contract-v1/README.md)：C 11交付SHA、完整报告除路径同、4数据文件及独立验证字节同，无实现行动项；完整6372系列合成边际全等价仍有FINAL包络median差H，67584合成关系核对。旧17280角色/8640merged同调用恒等式保留；统计不交换，不授逐次DAG/排序/完成资格。

[联合进程记录提取器](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-median-protocol-v1/README.md)：新增raw身份→完整setup/前序/同调用series→固定block数组的提取路径，拒绝重复、缺项、错误臂与SM映射变化；13定向测试。正式joint-replay-v3复放40raw，独立字复核17280窗口/69120复合标量/40 consumer FINAL包络，完整median输出与旧回放一致。复合量另存不入原6372覆盖族，不称CUDA或全局D完成；A/C限定审查已交接，无新GPU/资格。另B两版解码完整40replay/5280CTA/17280tile一致性补查重放字节同，新增warmup检查边界明确。

[联合记录采集身份绑定修复](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-median-protocol-v1/README.md)：接入A对正式v3的归属错配发现，旧verifier在最终旧统计比较时拒绝，非端到端静默通过。v4要求独立固定回收包SHA，解码/统计前逐字核对归档manifest、records、complete和raw；两种真实记录错配入口拒绝且不产出统计，另联合block后改在verifier拒绝。19定向测试、合法40进程/17280窗口/69120复合标量/40包络重放通过，原4912/4729失败与全部资格false不变。A限定修复审查待回交；无新GPU、模型或留出成绩。

[联合记录限定审查闭合](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-joint-interface-review-v1/README.md)：A v4归属复审完整报告主线重放逐字同，合法40进程及两种入口拒绝通过；R09提交281fae5e接入b2c7b7c0。C六文件交付及55输入SHA复核，40同调用event/10560FINAL/8640可恢复括号的完整报告除重定位路径同；v3/v4联合及统计gzip逐字同，仅复用历史端点语义，不把FINAL当D可见/CUDA完成或event差值当尾部。原观测资格仍false。

[64次中位数草案的规划诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-median-protocol-v1/README.md)：在明确假想的各臂旧十样本经验IID总体下，14/51秩的等价必要宽度要求38/64样本落128ns区间。1728 L臂坐标的精确二项尾并集界中，off/off最强全族通过概率上界4.1878531e-11、off/on 1.0582152e-6；9个小总体穷举及独立十进制递推核对。非真实GPU功效/噪声下界，不改阈值、样本数、失败或资格；当前不直接冻结执行256进程草案，先明确局部服务所需观察设计。无新GPU。

[有限stage的最小缺失观测](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R13-stage-progress/reanalysis/main-minimal-service-observable-v1/README.md)：B85输入SHA及20progress/20sparse直接trace槽复核，C16−C1的stage4−6配对中位2941.5cycle保留。缺同调用group2..16 full-wait累计量Q_early，现单点与非均匀软件路径不能重建；不把单点乘15、跨协议拼接或追加统一gap。先辨别等待内/外，未新增probe/GPU或服务价。上述中位数规划必要界经A整数多项式独立复算，主线同文有理数核对。

[R13早段累计等待实现与构建](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-early-wait-sum-build/README.md)：固定cfg_b第二输出local2..16累计full try/wait原生cycles，同调用保留C1/C16、15count/mask和H2/B16。740238双头重复定义失败保留，canonical include修复后740240 CPU-only CUDA12.9两stage编译通过（168regs/2barriers/0spill/16HGMMA），SASS快/慢等待均在H/B间；B后38条静态collector指令不能视为无扰动。6CPU测试及B128组覆盖/完整报告复放通过，A同线程包含漏检已修复。四目标数值/trace smoke已准备，尚无GPU采样或观察/服务/预测资格。

[R13累计等待最小正确性采集](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-early-wait-sum-correctness/README.md)：5375438预冻结后740259原ef/a058排他仅采stage4/off，binary0但collector因实际四包/二维cluster格式缺陷退出1，原失败保留。修复reader并新增真实raw回归，a8f5e14冻结剩余3目标后740276成功，不重采原目标；总共原四进程16384独立精确样本，同SM124，两个on皆15group/mask/端点包含有效。on记录stage4 D/Q/外为10739/4633/6106cycle，stage6为8582/1301/7281；仅各一次且两job，不估效应/扰动/服务价格。7CPU测试过；两归档22冻结输入及完整四记录重新解析通过。B源码和SASS报告、A旧时间边界报告、C首记录完整整数dot均主线复放一致；新packet包装适用版本单列。无全D/局部观察/预测/新留出资格，cfg_a/b/c目标不变。

[R13累计等待分组诊断740334](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-early-wait-sum-diagnostic/README.md)：96dc9b8预冻结，原ef/a058排他16随机组×4臂，64目标及262144独立精确抽样值、12冻结输入/完整raw复放通过，全SM124。逐组stage4−6的on D/Q/R中位2593.5/3519.5/−1045.5cycle，Q全16组正、R全16组负；stage6 on−off D中位670.5，条件区间[349,820]。八量按预列3..14秩、仅IID-block条件下解释，中位数不相加、不把含零称等价；B独立128差/24摘要值一致。下一步只辨别累计观察对后续等待的反馈，不追加本协议样本、不把Q定价或授新观察/预测/留出资格。A包装格式限定复审已重放并接入daa12023，原失败保留。

[R13四模式反馈probe](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-wait-feedback-build/README.md)：740367 CPU-only CUDA12.9两stage编译通过，统一mask选择OFF/ALL15/ONLY2/ONLY16，同一H/try/wait/B，新增Q2/Q16但single完整Q/R为None；168regs/0spill，B后53静态指令非定价，5新CPU测试通过。740368因copyfile丢执行位在setup启动前失败、0target，原包保留；相同binary字节的新attempt冻结执行位后740369完成8目标/32768独立精确抽样值。

[R13累计反馈诊断740370](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-wait-feedback-diagnostic/README.md)：34c747b预冻结16组×6臂共96目标，ef/a058排他全部SM124，12输入/执行位/393216值通过，原slot独立核96个联合差。stage4 L16/L2/F中位−159.5/−5.5/−116cycle，预列区间均包含0（L16上界恰0）；stage6分别0/−3.5/19，F条件区间[3,37]、其余含0。只按IID-block条件解释六量3..14秩，不支持统一负反馈价、不判无反馈、不加样或将旧38指令协议混池。下一步在既有cycle递推分开full H/B/MMA与未定价的pre/post观察输入，默认公共预测不改；新模型、观察、留出及跨卡仍未通过。
