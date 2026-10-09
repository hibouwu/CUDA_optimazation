# GH200 实验执行规范

实验问题见 [PLAN](PLAN.md)，现行候选见 [RULES](RULES.md#v09-model)，各项证据写对应实验页。本页只维护分工、共享代码、存储、确认边界与当前状态。已完成批次的提交、启动文本和故障经过见 [cee978a固定版本](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/EXECUTION.md)。

## 分工与交回

管理对话统一维护公共模型、代码集成、GPU分配和最终评分；执行组按原实验承担完整的小目标，同卡对照独立完成后交回。每个文件/运行目录只有一个写入者。

| 角色 | 负责范围 | 交回内容 |
|---|---|---|
| 管理 | PLAN、EXECUTION、README、RULES、公共cfg_a/b/c框架、组合模型、V09及后续完整验证 | 集成版本、范围、唯一全组判定与下一步 |
| A：供给与请求 | R10/R13、对应专用配置和分析 | 请求/供给关系、first/later差异、残差、不可辨识部分 |
| B：输出与边界 | R15/R18/R19、对应专用配置和分析 | 输出端点、real/OOB、边界配对与关键CTA |
| C：时间与输入 | R09、旧R07/V03证据及专用分析 | 同调用cycle/ns、输入与数值范围、时钟迁移 |

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

V09参考卡为romeo-a057的 **GPU-43269fbc-449d-3e0f-908a-9c81229546d3**；V07/V08的GPU-099dda56是另一张历史参考卡。节点名不构成GPU身份或当前可用性证明。

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

## 当前状态

| 对象 | 已有结果 | 当前工作/边界 |
|---|---|---|
| 管理 | 对话`01a11cfc-87f0-76e3-b53e-dc2b5e241844`；V09采样、复核完成 | 文档收敛、纯模型等价拆分和[消融](V01-validation.md#v09-ablation)已完成；后续先解释实验反例，不新增GPU任务 |
| 供给 | [R10/R13](R13-async-retirement.md) first/later已接入r2；首轮A/B填零补点完成 | 行距、stage响应与代数支持仍有失败/拒绝，不能称供给规律已通过 |
| 输出/边界 | [R15](R15-output-service.md)双角色ns及single/multi、real/OOB候选已接入；[R18](R18-cluster-boundary.md)保留路径对照 | partial、cfg_c普通middle与关键CTA仍有限制 |
| 时间/输入 | [R09](R09-inkernel-clock-stages.md)21点时钟、等工作配对和零比例诊断完成 | cfg_b长K迁移失败，首段与局部cycle/ns继续离线定位 |
| V09 | 739011完成30条件；29受支持预测4.11%/35.46%，1冻结前拒绝 | **整体未通过**，冻结/判定不改；新候选须另用新留出检验 |
| 计数器 | 既有权限拒绝证据可复用 | 受支持入口仍未确认，账户持有人联系管理员 |

当前状态只在本节维护；具体实验分析归原页。历史A/B/C长期对话ID、工作树、提交表和故障记录见[固定状态表](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/EXECUTION.md#status)，不把旧绑定或旧“待补测”状态当作当前任务。

<a id="goals"></a>

原目标模式启动文本保留在[固定历史版本](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/EXECUTION.md#goals)。文档中的旧模板不激活任务或自动授权GPU；当前执行以用户在对话中已确认的具体范围为准。
