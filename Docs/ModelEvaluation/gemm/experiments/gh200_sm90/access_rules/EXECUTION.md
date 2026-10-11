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

持续目标以新的独立冻结验证和跨卡评估为终点；完成一批分析、采样或提交只完成一个里程碑。沿同一预测主线继续选择最有辨识力的下一步，不自动展开全部机制矩阵。新完整留出仍先明确范围与采样授权，等待时推进独立工作。

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

<a id="goals"></a>

原目标模式启动文本保留在[固定历史版本](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/EXECUTION.md#goals)。文档中的旧模板不激活任务或自动授权GPU；当前执行以用户在对话中已确认的具体范围为准。
