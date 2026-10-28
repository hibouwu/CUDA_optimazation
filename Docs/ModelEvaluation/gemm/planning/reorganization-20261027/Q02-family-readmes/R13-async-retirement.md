# R13-async-retirement（草稿）

主题：R13 异步退役（早期轮次）。

22 个目录，其中 0 个由别处移入（原族列出）。“记录性质”按目录内是否有 Slurm 日志、receipt、submission、allocation 或回传目录判定；“卡”由文本中的 GPU UUID 识别，未注明为 —；“结论摘录”取自各轮正文，最终版按原记录改写。

| 轮次 | 原族 | 问题 | 记录性质 | 卡 | 结论摘录 |
|---|---|---|---|---|---|
| `20261007-R13-job735876-v8-formal-ready` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261007-R13-job735876-v8-formal-ready-review-B` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261007-R13-job735985-v11-formal-reuse-partial-review-B` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261007-R13-job735985-v11-rep-reuse` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261007-R13-job735985-v11-rep-reuse-review-B` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261007-R13-job735985-v9-formal-retire` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261007-R13-job735985-v9-formal-retire-review-B` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261007-R13-job735985-v9-ready-recheck-review-B` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261007-R13-job736169-v11-formal-reuse` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261007-R13-job736169-v11-formal-reuse-review-B` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261009-R13-R15-R18-composition-job738496` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261009-R13-sm-job738101` | R13-async-retirement | （无 README 正文） | 含 GPU 执行 | — |  |
| `20261009-R13-stage4-check-job738583` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261009-R13-stage4-check-job738599` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261010-R13-first-supply` | R13-async-retirement | （无 README 正文） | CPU／已有数据 | — |  |
| `20261010-R13-second-prefill` | R13-async-retirement | （无 README 正文） | 含 GPU 执行 | — |  |
| `20261010-R13-stage-events` | R13-async-retirement | （无 README 正文） | 含 GPU 执行 | — |  |
| `20261010-R13-stage-progress` | R13-async-retirement | （无 README 正文） | 含 GPU 执行 | — |  |
| `20261011-R13-a-row-rotation` | R13-async-retirement | （无 README 正文） | 含 GPU 执行 | — |  |
| `20261011-R13-early-wait-sum-build` | R13-async-retirement | R13 早段累计 full 等待探针 | CPU／已有数据 | — | 严格decoder为decode_r13_wait_sum.py，要求caller固定raw SHA和GPU UUID；数值检查按同次4096个唯一样本独立重算，尚不代表全D正确。四目标正确性smoke现已取得实际Q记录，详见下方链接；尚无扰动/null通过、无新增服务价、无完整 |
| `20261011-R13-early-wait-sum-correctness` | R13-async-retirement | R13 累计等待四目标正确性检查 | 含 GPU 执行 | — | verify.py从两份固定SHA回收包核对各11个冻结输入、原进程归属及binary返回码，加载remaining3固定reader重算四条记录；完整结果见verification.json。复放命令为`python verify.py --output NEW_REPORT` |
| `20261011-R13-early-wait-sum-diagnostic` | R13-async-retirement | R13 等待内外的分组诊断 | 含 GPU 执行 | — | 下一步选择累计观察反馈的最小判别：同binary比较累计15组与只观察group16的该组等待，并用group2作为没有前序累计操作的对照；实现、SASS及正确性须另行确认。本次不增加同协议样本、不拟合公共模型。观察、服务、关键CTA、新留出及跨卡资格保持未通过。 |

## 族内分组（待定稿）

- （按假设或端点分小节，审阅时补）

