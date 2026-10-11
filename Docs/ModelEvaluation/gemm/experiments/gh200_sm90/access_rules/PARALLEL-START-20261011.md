# 两个有界目标的启动准备

本文件保存启动准备与目标消息；实际运行状态见[EXECUTION](EXECUTION.md#status)。准备完成时两个目标未启动、GPU 提交为 0。八小时从各目标实际启用时分别计时；GPU 排队计入上限。输入与源码 SHA、路径、分支及预算见[绑定清单](PARALLEL-START-20261011.json)。

## 基点与路径

共同基点：`3b53dbf721d7d30828d6b78b92103a214be1fbfe`，由用户将此前已审查的 11 个文件合并提交；任务 A 四个 Python 文件与交付哈希一致，原 R09 修改完整保留。本次保留用户提交，不拆分或改写历史，后续按文件归属分别提交。

| 对象 | 工作树与分支 | 新结果根 | 预算 |
|---|---|---|---|
| 主线 | `/home/jianyeshi/.codex/worktrees/gh200-long-k-event-20261011/CUDA_optimazation`；`codex/gh200-long-k-event-20261011` | `/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-long-k-event-bounded` | 最多八小时，GPU 0 |
| SMEM | `/home/jianyeshi/.codex/worktrees/gh200-smem-lds-sts-20261011/CUDA_optimazation`；`codex/gh200-smem-lds-sts-20261011` | `/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-m11-lds-sts-bounded` | 最多八小时，GPU 作业 2 |

两工作树的代码和 PLAN 与共同基点相同。公共执行状态以主 checkout 的[EXECUTION](EXECUTION.md#status)为准；新工作树内旧 EXECUTION 只反映基点历史，不能把其中旧“未提交/未准备”描述当实时状态。未移动现有对话的默认 cwd，执行者每次 shell 调用须显式使用上表工作树。

## 数据与归属

现有数据只读根：`/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules`。使用绝对 `--base`/输入路径，不复制整个 results，不把主结果树软链接为新工作树可写目录。新输出使用各自结果根下尚不存在的 run/reanalysis 子目录；结果根已含准备记录，不能直接作为要求目录不存在的 runner 输出。

主线负责组合分析、`v09_model`、需要修正的服务/事件组件及 R09/R13/R18/R19 证据扩展。主集成串行维护公共 PLAN/RULES/EXECUTION、共享 runner、最终冻结/评分。SMEM 只改专属模型/probe、R03 runner 的新增模式和 R03/EXP-09 扩展，不改公共预测或共享计时头；需要公共接口时先在交付中说明。时间与供给对话保持暂停。双方不覆盖旧原始、冻结、失败或准备记录。

SMEM 旧对照已绑定 R03 job735059 的源码/环境/协议与 EXP-09 的原共享内存 probe、合同及报告。CUTLASS 子模块未初始化；静态序列核对使用已有冻结运行中的 3.9.2 快照：`/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/source/cutlass`。它是只读源，不是新的编译/运行资格，也不以根仓库 HEAD 冒充 CUTLASS 提交。

## 主线启动消息

> 在工作树 `/home/jianyeshi/.codex/worktrees/gh200-long-k-event-20261011/CUDA_optimazation`、分支 `codex/gh200-long-k-event-20261011` 启用一个新的有界目标：最多八小时完成长 K 时钟与事件模型的一次联合修正。先读取主 checkout 的本文件、绑定清单与 EXECUTION，以及工作树 PLAN 第 2 节；核对基点 `3b53dbf721d7d30828d6b78b92103a214be1fbfe` 与 PLAN SHA `2af12d00f5519fb6322caedaea5c6f9f03827ce4bbbfc702c34dadcb568c29dc`。以 cfg_a G6 为入口区分局部误差与组合误差，用 G6 三配置和固定 29 条开发条件检查推广。GPU 预算为 0，最多检验两种事先写定的最小形式；归因完成时、最迟 3.5 小时先写检查点，再进入实现，无需重复等用户确认。不得读取目标/前序实测作为正式预测输入，不按 G6 编号修补；留掉 G6 后参数无法由其他数据或静态约束确定时交付必要测量方案，不填入全数据参数。候选、必要测量或无候选任一终点即停止，不自动开第四轮开发，不执行 V10 或新 GPU。启动时登记开始时间、八小时截止、计划与源码身份，使用绑定清单中的只读输入和独立结果根；公共状态只由主集成维护。

## SMEM 启动消息

> 在工作树 `/home/jianyeshi/.codex/worktrees/gh200-smem-lds-sts-20261011/CUDA_optimazation`、分支 `codex/gh200-smem-lds-sts-20261011` 启用一个新的有界目标：最多八小时完成 PLAN m11 第一轮普通 SMEM 访问模型。先读取主 checkout 的本文件、绑定清单与 EXECUTION，以及工作树 PLAN 的 m11 合同；核对共同基点与 PLAN SHA `2af12d00f5519fb6322caedaea5c6f9f03827ce4bbbfc702c34dadcb568c29dc`。限定单 CTA、实际 SASS 确认的 LDS/STS 32/64/128 bit，矩阵指令、多 CTA 与 TMA/WGMMA 不在范围。最多两个提交作业，含失败与重试：第一作业协议/旧结果对照/开发校准，第二作业仅在预测、参数、条件、seed、协议和 SHA 冻结后做留出；同 GPU UUID、同协议，记录实际 SMID。先用兼容原窗口的对照检查 EXP-09 stride 与广播旧条件，再区分地址需求和发射/依赖限制，新窗口不机械套旧速率。报告时间相对误差中位 ≤5%、最大 ≤10%，同时报告绝对周期误差、粒度、固定开销及离散程度，局部容差留出前写定；服务批次仅为待检验推导。留出使用预列生成规则的新地址映射和真实 CUTLASS epilogue 中受支持 LDS/STS 序列的隔离重放，范围外指令明确记录。不得删除预列失败或用留出回调同轮参数。通过、反证/不可辨识或资源/检查不足时如实交付并停止，不自动扩展或接入主线。只改专属文件，公共维护交给主集成；启动时登记起止与提交作业清单，每次 shell 调用显式使用本工作树。

准备阶段未发送启动消息或创建目标。随后用户明确授权启动：当前主集成创建新主线目标，机制复用“GH200 输出与边界”；用户清除其旧暂停目标后，新SMEM目标也成功启用。原普通任务启动记录与目标工具启用记录分别保存，八小时截止不重计；实际时间与状态只在 EXECUTION 维护。本轮未创建额外对话，时间与供给对话保持暂停。
