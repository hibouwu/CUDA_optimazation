# GH200 实验执行规范

问题与预算见 [PLAN](PLAN.md)，现行候选见 [MODEL-CANDIDATES](MODEL-CANDIDATES.md#v09-model)，可采用规则见 [RULES](RULES.md)。本页维护分工、边界与当前状态。历史复核、故障和启动记录见[整理前固定版本](https://github.com/hibouwu/CUDA_optimazation/blob/7e29f5b51cad3e9685f805e3573583ef946c7f19/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/EXECUTION.md)。

## 分工与交回

每个文件和运行目录只有一个写入者；主集成维护公共模型、计划、GPU分配与最终评分。专业组只写独占研究产物，公共接口由主集成合入。用户已授权主集成与以下三个专业对话互发有界协调消息。

| 写入者 | 对话ID | 归属 |
|---|---|---|
| 主集成 | `01a129c0-b787-7e03-adf1-fcfdb2afa149` | PLAN/RULES/EXECUTION、公共probe/runner、v09_model、集成与冻结评分 |
| A：时间输入 | `01a11da3-a045-7ae3-b2b5-f5762e3bddf5` | R09、clock_model、r09专属分析与测试 |
| B：供给请求 | `01a11da3-a02c-7f93-8290-a82d81df0795` | R10/R13、R14输入生命周期、supply_model及供给分析 |
| C：输出边界 | `01a11da3-a02c-7f93-8290-a83b838ab41a` | R15/R18/R19、output_model；本次m11专属probe/模型/runner与R03/EXP-09 |

最多并行一个主线和一个机制任务，各用独立分支/worktree/结果目录。整理阶段仅主集成写入，GPU预算0；旧专业目标保持暂停或完成状态，不启动供给研究。当前待删除对象见[整理清单](CONSOLIDATION-20261012.md)，未经用户确认不删除。

<a id="shared-gemm"></a>

## 共享代码与计时边界

- 公共cfg_a/b/c、CUTLASS覆盖头、trace ABI和最终预测由主集成维护。专属分析器属于相应组；跨rank前驱由C提出、B接入，源工作口径由B定义、A使用。
- frozen源码、旧run、原始记录和summary只读；每次运行保存独立身份，新输出使用新目录。
- 正式换算绑定实际tile、cluster、schedule、stage、输入、暖机及完成端点；旧R09的128×256×64、4stage常数不能直接移植。
- plain/stamped/ends/wide/dual分别维护边界；完整调用扰动小不覆盖局部观察失败。使用目标或前序实测的诊断不进入V10正式预测。
- 被否决的长K接口隔离为 `v09_long_k_rejected.py`，仅供失败候选复现；公共 `v09_model.py` 未接入该成本。普通SMEM模型亦只用于研究，没有采用资格。

## 数据与GPU资源

本地归档根为 `results/gh200_resource_campaign/access_rules`。worktree读取主checkout的明确只读输入路径，不复制整份历史results。新run下保留各attempt的源码、失败和独有记录，不拼接不同协议为成功样本。结果目录的空间与独立证据清单见[整理清单](CONSOLIDATION-20261012.md#results)。

正式计时仅在Slurm获配GPU上运行，每卡串行；整节点exclusive按实际AllocTRES计资源。先完成准备再申请，不占分配等待开发或人工确认，不重复提交同批副本，不取消其他研究作业。远端临时盘仅中转，先收齐本地再按明确授权清理；历史home/scratch近配额的状态不能当当前余量。

<a id="reference-gpu"></a>

## 参考卡

V09为romeo-a057 / **GPU-43269fbc-449d-3e0f-908a-9c81229546d3**；V07/V08的GPU-099dda56是另一卡。新主线校准须绑定原参考卡和预列参照；新卡必须重做受影响校准，不能只换UUID。节点名不能证明设备身份或当前可用。

m11开发740961绑定romeo-a048 / **GPU-009a8880-f309-a55d-d256-748fb9adad74**，SMID124；后续同协议桥接、校准和留出使用同卡并记录实际SMID。指定UUID必须先属于job/step分配，再收窄CUDA_VISIBLE_DEVICES。跨卡资格仍未完成。

<a id="v09-freeze"></a>

## V09确认边界（已经完成）

V09 r2已经用户确认、job739011测完；冻结预测SHA为 `f4df39e00cf1d2ab61947c911e699bd5bea4e533609d93f726de43077515d81e`，清单SHA为 `09da85d124eb697a95c4695bd05281b07c37877890d0eca18afbdd2cf213f2ab`。原拒绝项、支持范围与失败分数永久保留；[完整记录](V09-component-validation.md)。

V10先确定候选和新条件，必要校准完成后回传最终预测与manifest SHA，由用户确认这一版本，随后才采留出。参数或预测变化不能沿用原确认。当前新许可只有主线校准，不包含尚未产生的V10预测版本。

<a id="counter-access"></a>

## 计数器权限

[已有诊断](../../../../../../microbench/gh200_l0/results/20260930-counter-access/README.md)记录普通/独占作业均受RmProfilingAdminOnly=1限制，NCU返回ERR_NVGPUCTRPERM。受支持入口仍待用户联系ROMEO管理员；没有计数器时不能从时间拟合宣称识别物理流量、事务或唯一硬件端口。

<a id="status"></a>

## 当前状态

更新：2026-10-12，仓库整理阶段。整理不做新科学分析、不提交GPU作业；既有两个研究都已达到终点。

| 工作 | 交付与终点 | 下一步 |
|---|---|---|
| 主线8小时合同 | 缺必要测量；第一修正cfg_a G6 −3.698%→−16.739%，未采用；GPU0 | 用户已批准整理后一个参考卡校准作业，18新条件加参照/桥接共660进程；[R09](R09-inkernel-clock-stages.md) |
| m11第一轮 | 740708失败、740961开发完成；宽广播反证，无冻结；历史用2/3次 | 普通探针与分析器收口，独立留出未执行；[R03](R03-access-demand.md#m11-current) |
| m11后续离线 | 两形式无候选；全warp形式相对门槛过而绝对周期/留布局失败；GPU0 | 用户批准48条件开发对照并追加独立留出预算；[新合同](PLAN.md#measurement-next) |
| 供给与其他机制 | 保持暂停 | 本次整理或补测不自动恢复 |
| V10与跨卡 | 未冻结、未采样，联合验收未完成 | 校准后有候选才进入指定版本确认 |

冻结与旧研究身份不回写。两条线后续执行以[新合同](PLAN.md#measurement-next)为准，实际启动与job状态写新run，不把过去“active”或“未授权”文字当当前状态。

<a id="goals"></a>

## 目标模式

目标完成指有界合同达到明确终点，不等于模型通过。文档模板不激活目标；已完成的反证目标不重开，后续测量用新记录保留原失败、开始时间和作业计数。
