# 联合对应表：Q14 外移与 Q02 重组

[JOINT-MAPPING.tsv](JOINT-MAPPING.tsv) 覆盖 Q02 现有的 213 个目录和 Q14 现有的 113 个目录，共 326 行。
- 依据：
  - 去往 Q02 的目录，按 [Q02 分族方案](Q02-family-proposal.md)；
  - 其余目录，按 [Q14 归属清单](Q14-question-ownership.md)；
  - 两表重叠的 45 个目录，以 Q02 方案为准。
- 每行写出当前路径、新路径、是否移动、目标问题和目标族。

## 核对结果

- 326 行的旧路径互不重复，新路径也互不重复；所有目录都存在；新旧路径都是三层，轮次名不变。
- 需要移动 163 个目录。它们的目标路径目前都不存在，也不与任何现有目录重名。

## 移动去向

| 目标 | 目录 | 目标族 |
|---|---:|---|
| Q02 | 124 | 17 个族中的 8 个新族及原族；包括 Q14 归入的 45 个，以及 Q02 族之间调整的 79 个 |
| Q03 | 12 | `source-state-candidates`：Q14 归入 7 个，Q02 移出 5 个 |
| 非实验 | 9 | `non-experiment/integration`：程序图 6 个、模块汇总、程序接入；`non-experiment/campaigns`：`cfg-b-window` |
| Q04 | 5 | `local-sync-chain`：等待策略 4 个，与原有的 `Q04-wait-policy-001` 同主题；`g3-stage-credit`：活动槽对照 1 个 |
| Q05 | 3 | `local-compute-chain` |
| Q06 | 3 | `R14-stage-handoff` |
| Q07 | 3 | `input-output-joint` |
| Q01 | 2 | `startup-prefix`（新族，`cfg-b-prefix`）；`work-placement`（`d226-fixed-history`） |
| Q10 | 1 | `plain-repeat`（`bridge-null`） |
| Q15 | 1 | `V09-component-validation`（`V09-gate-audit`） |

Q14 留下的 35 个目录不移动：
- `cfg-b-composition` 5 个：完整组合回归 4 个，加计算活跃对发出的影响；
- `COMPUTE-01-04-resident-chain` 15 个；
- `gemm-connection` 15 个：输出交互、计算与等待的交互、联合规则迁移、同行活动。

`gemm-connection` 外移后，留下的都是“完整 GEMM 中一条路径影响另一条路径”，主题已经一致，所以不改名，族 README 内分四个小节。

新建 11 个族目录：Q02 的 8 个、`Q01/startup-prefix`、`non-experiment/integration`、`non-experiment/campaigns`。

## 执行时要处理的两件事

**1. 三个被清空的族目录**

`Q02/row-stride`、`Q02/gemm-source-candidates` 和 `Q14/program-graph-export` 移空后，各自只剩一个族级 `README.md`。这三份 README 归档到 `non-experiment/records/reorganization-20261028/old-family-readmes/` 后，删除空目录。原内容并入新族的 README。

**2. 四个会断的相对软链接**

这些软链接指向同族中相邻的轮次，搬迁后两边分属不同的族：

| 软链接 | 原指向 | 搬迁后应指向 |
|---|---|---|
| `source-fold-replay-001/source` | `../20261022-Q14-gpc-delivery-bound-001/source` | `../../gemm-input-network/20261022-Q14-gpc-delivery-bound-001/source` |
| `bridge-null-001/{source,build,reference}` | `../20261022-Q14-equal-value-source-002/…` | `../../../Q02-tma-request/gemm-source-reuse/20261022-Q14-equal-value-source-002/…` |

建议做法：按新位置重写这 4 个软链接的指向文本，被指向的文件字节不变，改动记入变更记录。另外 5 个软链接（equal-value-source 003/004 指向 002）随同族一起移动，仍然有效。

全树没有从不移动的位置指向这些轮次的软链接。

## 执行步骤

1. **备份。** `cp -a --reflink=always` 把 `results/gh200_resource_campaign/access_rules` 复制到同一 btrfs 盘的备份目录，核对文件数和总大小相同。
2. **移动。** 按本表逐行同盘 `mv`，先建好 11 个新族目录。处理三个被清空的族目录和四个软链接。
3. **更新对应表。**
   - `MOVED-PATHS-20261020.tsv` 中，凡是“当前路径”等于本表旧路径的行，改为新路径。`flat_layout_view.py` 读的正是这一列，所以历史入口无需改动。
   - 另存本次的变更记录 `MOVED-PATHS-20261028.tsv`。
4. **Docs。** 把指向旧路径的链接按本表替换。结论按 Q14 归属清单分流到各问题页，这一步单独审阅。
5. **验收。**
   - 163 个目标都存在，源路径都已消失；
   - 文件数和总大小与备份一致；
   - 4 个软链接能解析；
   - 抽 4 个旧轮次用 `flat_layout_view.py` 建视图后，入口能运行；
   - Docs 中没有新增失效链接。
