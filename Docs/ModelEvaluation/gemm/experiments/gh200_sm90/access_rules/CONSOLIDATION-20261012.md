# 2026-10-12 GH200 仓库收口

## 范围与检查

本次仅整理已有代码、文档与证据并修复CPU回归的历史依赖入口，GPU预算0，不作新科学分析或产生新模型结论。起点为 `7e29f5b51cad3e9685f805e3573583ef946c7f19`；fetch后的origin/main为 `7e11ced62a56b610da5f9ca3f84842ac842c4c55`，落后当前263提交。整理前完整快照、文件SHA、分支补丁表、忽略文件保存记录与回归输出在[本地审计目录](../../../../../../results/gh200_resource_campaign/access_rules/20261012-repository-consolidation/)。保留备份分支 `codex/gh200-cleanup-before-20261012`。

R09/R10/R13正文缩为当前结论、作业和证据索引；原推导分别移到本目录下reanalysis/20261012-consolidation的README，原锚点保留。EXECUTION只维护分工、规则与当前状态；原文保留在起点提交。V09候选公式与未获资格的条件关系移到MODEL-CANDIDATES，RULES不把开发关系写成通过规则。

新收口代码包含任务A的时钟/边界工具、R10身份敏感性复放、普通LDS/STS探针与runner，以及隔离的长K失败库。没有修改公共v09_model.py；默认36条件（含参照）的完整输出与整理前完全一致，冻结calibration/predictions/manifest逐字节不变。长K隔离库已独立复放150份旧预测、2160项时间账，保持全部采用失败。

最终CPU回归全部通过：access_rules 549项、历史资源套件482项、auditor 7项，共1038项，无失败或跳过。迁移路径故障与CPU fixture修复见[旧路径兼容](../HISTORICAL-PATHS.md)，日志和[最终核对](../../../../../../results/gh200_resource_campaign/access_rules/20261012-repository-consolidation/final-checks.json)保留之前失败与重跑。链接检查0错误；整理前381条证据路径继续可打开，原results文件SHA保持；SMEM原交付10299项SHA不变。测试通过不授予旧GPU实现新资格。

## 分支按内容核对

git cherry的“+”只表示补丁不等价，不自动表示缺合；对GH200逐文件比较后，再保留已有后续修订。完整文件判定见[内容表](../../../../../../results/gh200_resource_campaign/access_rules/20261012-repository-consolidation/branch-content-comparison.json)与[决策表](../../../../../../results/gh200_resource_campaign/access_rules/20261012-repository-consolidation/branch-decisions.json)。

| 分支 | 未匹配/等价补丁 | 处理 |
|---|---:|---|
| `backup/main-before-large-file-cleanup-20261005T172438Z` | 1/1 | 大文件清理前历史备份；有意排除，不复活旧大文件 |
| `codex/cutlass-thor-autotune` | 8/7 | SM110/Thor独立研究；不属于本次GH200收口，保留 |
| `codex/gh200-boundary-parallel-20261011` | 1/8 | 遗漏独占工具已接入，兼容性修订保留；旧失败结果未采用 |
| `codex/gh200-clock-input-20261009` | 6/9 | 相同旧文件已在主线；其余为后续接口抽取、扩展或文档更新覆盖，不反向恢复旧版本 |
| `codex/gh200-clock-parallel-20261011` | 1/38 | 遗漏独占工具已接入，兼容性修订保留；旧失败结果未采用 |
| `codex/gh200-long-k-event-20261011` | 1/0 | 三脚本与失败库已归档接入，150预测复放通过；公共预测器不采用 |
| `codex/gh200-output-boundary-20261009` | 7/5 | 相同旧文件已在主线；其余为后续接口抽取、扩展或文档更新覆盖，不反向恢复旧版本 |
| `codex/gh200-smem-lds-sts-20261011` | 3/0 | 全部普通探针、runner、模型、分析器与实验文档已接入；保留未通过资格 |
| `codex/gh200-supply-offline-0785` | 5/7 | 相同旧文件已在主线；其余为后续接口抽取、扩展或文档更新覆盖，不反向恢复旧版本 |
| `codex/gh200-supply-parallel-20261011` | 2/30 | 遗漏独占工具已接入，兼容性修订保留；旧失败结果未采用 |
| `codex/sm110a-codegen-bootstrap` | 0/1 | SM110/Thor独立研究；不属于本次GH200收口，保留 |
| `codex/sm110a-codegen-closure` | 11/7 | SM110/Thor独立研究；不属于本次GH200收口，保留 |
| `gh200-kernel-prediction` | 0/0 | 已为当前历史祖先，保留或确认后清理 |
| `gh200-mechanism-integration` | 0/0 | 已为当前历史祖先，保留或确认后清理 |
| `gh200-model-simplification` | 2/0 | 相同旧文件已在主线；其余为后续接口抽取、扩展或文档更新覆盖，不反向恢复旧版本 |
| `main` | 0/0 | 已为当前历史祖先，保留或确认后清理 |

## worktree与分支待清理清单

**没有执行删除。** 列“可清理候选”表示代码和独有忽略文件已保存，仍须用户确认删除。主checkout、整理备份和无关研究保留。归属/哈希见[原worktree表](../../../../../../results/gh200_resource_campaign/access_rules/20261012-repository-consolidation/worktree-audit-before.json)与[忽略文件逐项保存表](../../../../../../results/gh200_resource_campaign/access_rules/20261012-repository-consolidation/worktree-ignored-preservation.json)。

| 路径/分支 | 建议 | 保留依据或条件 |
|---|---|---|
| 主checkout / main及本轮分支 | 保留 | 集成与历史证据根 |
| 0785 / gh200-supply-parallel-20261011 | 可清理候选 | 独占工具已收口；58项忽略文件逐项保存，未删除 |
| 9e4a / gh200-boundary-parallel-20261011 | 可清理候选 | 独占工具已收口；32项忽略文件逐项保存，未删除 |
| gh200-long-k-event-20261011 | 可清理候选 | 失败源码隔离保留、复放通过；6项忽略文件逐项保存 |
| gh200-smem-lds-sts-20261011 | 可清理候选 | 代码与实验页收口；4项忽略文件逐项保存；后续任务启动时可继续复用 |
| 9cce / gh200-clock-parallel-20261011 | 保留，不列立即删除 | R09仍有未提交说明，patch及46项忽略文件已保存，工作树原修改未动 |
| d710 detached | 保留 | SM110的多项未提交/未跟踪改动，无关本次整理 |
| sm110a-codegen-closure | 保留 | 独立研究与4886项忽略产物，未触碰 |
| /tmp/cutlass-thor-autotune | 可确认后清理陈旧元数据 | 实际目录已不存在；其分支研究仍保留，不等于已合入 |
| gh200-kernel-prediction、gh200-mechanism-integration | 分支可清理候选 | 当前分支祖先，无独有未合补丁 |
| codex/sm110a-codegen-bootstrap | 分支可清理候选 | git cherry补丁全部等价；closure独立分支继续保留 |
| 10月9日旧GH200分支、gh200-model-simplification | 暂保留分支 | 内容已被当前修订覆盖，但其历史不强制改写或删除 |

<a id="results"></a>

## 结果目录清单（只列，不删除）

整个access_rules结果根目录du约15G。按顶层名称日期计，20261010/11/12分别20/44/2个目录，共66个，清单采集时约2.37GiB、56498文件；其中包含本次审计目录，清单及运行日志后续增加不属于新采样。“两天45目录”与此口径不同。完整大小、包、manifest和证据入口见[JSON清单](../../../../../../results/gh200_resource_campaign/access_rules/20261012-repository-consolidation/results-inventory.json)。

下列独立性按已有作业/报告身份列出，源码/构建资格、correctness、计时、离线复核与独立留出不互相替代。“未确认”不是无价值或可删除。准备与失败attempt都可能有独有SASS、错误和源码，应先归档而非直接丢弃。

| 目录 | MiB | 文件 | 独立证据/现有入口 |
|---|---:|---:|---|
| `20261010-R09-R13-R18-failure-controls` | 122.10 | 1449 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R09-common-history` | 32.20 | 973 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R09-history-series` | 175.28 | 1135 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R09-mirror-history` | 37.57 | 960 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R09-swizzle-order` | 36.01 | 1044 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R10-base-shift` | 34.16 | 1080 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R10-producer-delay` | 41.71 | 1146 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R10-raster-pitch` | 63.20 | 1208 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R10-source-pressure` | 34.79 | 1162 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R10-work-swap` | 34.19 | 1084 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R13-epilogue-order` | 39.11 | 1094 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R13-epilogue-order/README.md) |
| `20261010-R13-first-supply` | 35.96 | 1170 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R13-output-background` | 39.18 | 1098 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R13-output-background/README.md) |
| `20261010-R13-ready-observer` | 35.99 | 1170 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R13-ready-release` | 36.00 | 1170 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R13-second-prefill` | 76.01 | 1984 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R13-stage-events` | 76.85 | 1929 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R13-stage-handoff` | 44.21 | 1052 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R13-stage-progress` | 37.84 | 1021 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261010-R18-cluster-fill-discrimination` | 62.79 | 1335 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261011-R09-guarded-frame-build` | 41.10 | 907 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-guarded-frame-build/README.md) |
| `20261011-R09-guarded-record-check` | 19.32 | 275 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-guarded-record-check/README.md) |
| `20261011-R09-plain-prefix-build` | 8.35 | 22 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-plain-prefix-build/README.md) |
| `20261011-R09-prefix-host-check` | 7.49 | 18 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261011-R09-prefix-host-check-r2` | 8.25 | 18 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261011-R09-prefix-host-check-r3` | 12.41 | 23 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261011-R09-prefix-host-check-r4` | 12.45 | 23 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261011-R09-prefix-host-check-r5` | 13.49 | 41 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-prefix-host-check-r5/README.md) |
| `20261011-R09-prefix-identity-build` | 12.42 | 24 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-prefix-identity-build/README.md) |
| `20261011-R13-a-row-rotation` | 90.95 | 2856 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-a-row-rotation/handoff-manifest.json) |
| `20261011-R13-early-wait-sum-build` | 90.90 | 2759 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-early-wait-sum-build/README.md) |
| `20261011-R13-early-wait-sum-correctness` | 15.66 | 113 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-early-wait-sum-correctness/README.md) |
| `20261011-R13-early-wait-sum-diagnostic` | 10.89 | 255 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-early-wait-sum-diagnostic/README.md) |
| `20261011-R13-release-threads-correctness` | 41.05 | 906 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-release-threads-correctness/README.md) |
| `20261011-R13-release-threads-ef-study` | 19.30 | 398 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-release-threads-ef-study/README.md) |
| `20261011-R13-release-warps-build` | 25.36 | 859 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |
| `20261011-R13-release-warps-build-r2` | 40.77 | 890 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-release-warps-build-r2/README.md) |
| `20261011-R13-release-warps-correctness` | 41.00 | 905 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-release-warps-correctness/README.md) |
| `20261011-R13-wait-feedback-build` | 59.34 | 1755 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-wait-feedback-build/README.md) |
| `20261011-R13-wait-feedback-correctness` | 15.51 | 108 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-wait-feedback-correctness/README.md) |
| `20261011-R13-wait-feedback-diagnostic` | 12.72 | 348 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R13-wait-feedback-diagnostic/README.md) |
| `20261011-R18-first-output-build` | 39.34 | 924 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R18-first-output-build/README.md) |
| `20261011-R18-first-output-control` | 23.81 | 301 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R18-first-output-control/README.md) |
| `20261011-R18-first-output-correctness` | 25.42 | 51 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R18-first-output-correctness/README.md) |
| `20261011-V09-crosscard-references` | 0.13 | 2 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-V09-crosscard-references/README.md) |
| `20261011-V09-gate-audit` | 1.13 | 2 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-V09-gate-audit/README.md) |
| `20261011-V10-taskA` | 55.41 | 97 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-V10-taskA/README.md) |
| `20261011-cfg-c-gt-bracket-build` | 37.07 | 940 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-bracket-build/README.md) |
| `20261011-cfg-c-gt-bracket-build-failed-header-740155` | 27.76 | 879 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-bracket-build-failed-header-740155/README.md) |
| `20261011-cfg-c-gt-bracket-smoke` | 3.98 | 43 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-bracket-smoke/README.md) |
| `20261011-cfg-c-gt-bracket-smoke-v2` | 34.03 | 82 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-bracket-smoke-v2/README.md) |
| `20261011-cfg-c-gt-controls` | 76.68 | 681 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/README.md) |
| `20261011-cfg-c-mma-reference` | 5.33 | 79 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-mma-reference/README.md) |
| `20261011-cfg-c-mma-reference-preparation` | 29.44 | 884 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-mma-reference-preparation/README.md) |
| `20261011-cfg-c-producer-tail-build-v2` | 31.07 | 891 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-build-v2/README.md) |
| `20261011-cfg-c-producer-tail-check-build` | 31.05 | 885 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-check-build/README.md) |
| `20261011-cfg-c-producer-tail-control` | 11.36 | 121 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-control/README.md) |
| `20261011-cfg-c-producer-tail-control-preparation` | 1.56 | 23 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-control-preparation/README.md) |
| `20261011-cfg-c-producer-tail-correctness` | 32.80 | 54 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-correctness/README.md) |
| `20261011-cfg-c-producer-tail-off-off` | 13.66 | 250 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/README.md) |
| `20261011-cfg-c-producer-tail-preparation` | 27.59 | 869 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-preparation/README.md) |
| `20261011-long-k-event-bounded` | 22.68 | 33 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-long-k-event-bounded/README.md) |
| `20261011-m11-lds-sts-bounded` | 190.64 | 10546 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-m11-lds-sts-bounded/run-20261011T210745Z/delivery-manifest.json) |
| `20261011-parallel-bounded-preparation` | 0.13 | 10 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261011-parallel-bounded-preparation/smem-amendment-20261012-source/manifest.json) |
| `20261012-m11-demand-offline-v1` | 11.14 | 62 | 离线独立复核/研究输出，非独立采样；[入口](../../../../../../results/gh200_resource_campaign/access_rules/20261012-m11-demand-offline-v1/README.md) |
| `20261012-repository-consolidation` | 0.70 | 22 | 准备/结果身份保存；是否独立采样未由顶层入口确认，待保留核对；无顶层报告 |

压缩包逐字节SHA发现一组完全相同的7份：cfg-c-gt-controls原output.tar.gz与A-joint-extractor-audit-v1/v4的六个fixture副本。它们不是7次独立采样；这些副本是原分析器的验证fixture，是否改为共用归档需先检查复放引用，本次不删。全部SHA与路径见[重复包表](../../../../../../results/gh200_resource_campaign/access_rules/20261012-repository-consolidation/duplicate-packages.json)。其他包与展开目录只列为可能副本，未做完整等价核验，不能直接归为可删除。

## 本地交付与推送状态

代码收口提交 `32c12ca`，文档收口提交 `17447ed3`，均已快进合入本地main。HTTPS推送失败（无法读取用户名），GitHub CLI未登录，SSH也被拒绝；用户明确选择“先保留本地结果”，故没有更新远端。origin/main仍为 `7e11ced6`，本次原计划的推送未完成，不能把本地合入写成已发布。

## 整理后测量

用户已批准一个主线必要校准，以及m11开发对照后追加独立留出预算，见[新合同](PLAN.md#measurement-next)。整理本身不提交作业；用户选择暂留本地结果后，两项测量仍未启动、未提交GPU，后续授权与条件表保留。被反证的形式不能只为用完预算而冻结，V10仍须确认最终指定SHA。
