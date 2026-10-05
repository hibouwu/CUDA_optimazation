# S14：1D TMA 双向运输的 24 配置短验证

作业 `731626` 的 24 个配置、72 次启动已通过完整数值独审。本次没有执行校准、预热或正式性能采样，因此不产生带宽、延迟或周期服务参数。原始数据和来源见[独立审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S14-session-current-B3-review.json)。

## 实验问题与配置

验证无 tensor-map 的 1D TMA bulk 在两种方向、不同 payload 和执行范围下，数据、保护区及完成事件是否符合约定。对应后续模型中的 GMEM↔SMEM 运输服务；本次结果先证明测量探针的数值行为。

| 坐标 | 本次取值 |
|---|---|
| 方向 | GMEM→SMEM、SMEM→GMEM |
| 每请求有效 payload | 1、4、8、16、32、64 KiB |
| 范围 | 单 CTA、整卡配置；实际 CTA 数逐配置保留 |
| 每配置启动 | `(iterations, seed)=(1,0)、(2,3)、(33,4294967295)` |
| 输入与保存 | 非均匀输入；完整数据数组、保护区、完成记录和时间戳 |
| 分支 | `capture_enabled=false`、`formal_final_transport` |

共 `2×6×2=24` 个配置，每配置三次启动。这里验证最终运输结果及其完成边界；历史 84 次启动的额外源缓冲释放覆盖仍保留原来源，不算成本次重新执行。

## 从原始字段算一个例子

取 `smem_to_gmem_64kib_one_cta`。其实际记录为：

| 原始字段 | 值 |
|---|---:|
| `blocks` | 1 |
| `payload_bytes` | 65,536 |
| `global_slots_per_cta` | 32 |
| ring 数组 shape | `[1,32,16384]`，元素为 `uint32` |
| guard 数组 shape | `[2,4]`，元素为 `uint32` |
| 每次启动的 `checked_elements` | 524,296 |
| `global_allocation_bytes` | 2,097,184 |
| 实际寄存器 / 线程 | 31 |
| 动态 SMEM / CTA | 65,568 B |
| occupancy API 给出的上限 | 3 CTA/SM |

每个槽容纳 `65,536/4=16,384` 个 word，因此完整 ring 与保护区的检查数为：

\[
1\times32\times16\,384+2\times4=524\,296\text{ word}.
\]

对应分配为 `524,296×4=2,097,184 B`，与实际字段一致。另有完成记录和时间戳，各为 `[1,5,2]`，合计 20 个 word，单独审查，不混入这里的 `checked_elements`。

在 33 次迭代的启动中，逻辑运输量是 `33×65,536=2,162,688 B`。它与 ring 分配大小不同：33 次迭代会复用 32 个槽，独立参考核对各槽的最终值；1 次、2 次迭代的启动还会核对其余未访问槽是否保持初始值。本次没有用上述任一字节数除以短验证时间来发布带宽，也没有将逻辑运输量称为物理 HBM 流量。occupancy API 的上限也不代表本次单 CTA 实际驻留了三个 CTA。

## 正确性与实际资源证据

实际设备为 `romeo-a047` 上的 GH200，GPU UUID 为 `GPU-a97903c9-11d5-330a-0635-ec3931cbae6e`。设备记录包含 SM90、132 SM、60 MiB L2，driver/runtime API 版本为 13010/12090。各配置保留自己的实际资源、网格、源码与机器码身份。

| 独立核对项目 | 本次结果 |
|---|---:|
| 配置 / 启动 | 24 / 72 |
| 完整原始数组工件 | 288 |
| 数据与保护区 word | 1,461,766,464 |
| 生命周期与时间戳 word | 365,040 |
| 归档成员，全部检查 EOF 与 SHA | 15,062 |
| 完整独立重放耗时 | 160.896 s |
| 无损压缩包大小 | 85,991,516 B |

数值比较、原始字段与工件形状、18 个目标的完整编码、资源、依赖及串行进程记录分别核对。计数器与正式统计没有在本次生成；这里也不展示性能图表。

## 作业终态与剩余问题

`731626` 完成了全部短测试，并在等待独立审查十分钟后保存 `short_review_timeout_or_budget` 检查点退出，Slurm 记录为 `FAILED 75:0`。这是运行器约定的提前退出码；完整数值独审在窗口后完成。该结果不能声称已在同一分配内完成 preflight 或正式采样。

后续离线 CPU 准入调用还发现 `s14_session_admission_v1.py` 中变量覆盖了原审查记录，产生 `missing gate_files`。问题属于准入源码；[局部修复及真实工件回归](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s14-session-admission-fix-r4/implementation.json)独立于本次数值结果，不能用通过的数值审查替代后续完整准入。

较早的 `731617` 在设备查询程序启动前因缺少可执行权限退出，未运行数值目标；其失败记录保留。`731626` 的第一次收集因 `srun --unbuffered` 引入终端而被 XZ 拒绝，收集端退出、未输出数据。移除该选项后，从同一批原始文件完成无损收集，没有重新启动 GPU 测试。

## 离线复现入口

本次完整归档位于[本地 session-proof](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s14-session-deployment-r3-b/deployment/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s14-current-session/session-proof/collection.json)，包含 `run.tar.xz`、`index.json`、`closure.json`、`candidate-facts.json` 和独立生成的 `facts.json`、`numeric-proof.json`。原始结果目录遵循现有忽略规则，未自动加入 Git。

在当前工作区可离线重算，输出目录应选择尚未使用的新目录：

```sh
cd /home/jianyeshi/Note/CUDA/CUDA_optimazation
S14_SUITE=results/gh200_resource_campaign/20261001-resource-suite-v2
S14_REPO="$PWD/$S14_SUITE/implementation/s14-session-deployment-r3-b/deployment/repo"
S14_PROOF="$S14_REPO/$S14_SUITE/implementation/s14-current-session/session-proof"
PYTHONPATH="$S14_REPO/microbench/gh200_resource_campaign" python3 -B \
  "$S14_SUITE/reviews/evidence/S14-session-numeric-tool-v1/review_current72.py" \
  --repo "$S14_REPO" --archive "$S14_PROOF/run.tar.xz" \
  --index "$S14_PROOF/index.json" --closure "$S14_PROOF/closure.json" \
  --candidate "$S14_PROOF/candidate-facts.json" --expected-job 731626 \
  --output-dir "$S14_PROOF/offline-recheck-job731626"
```

该命令不调用 GPU，不产生新的正式性能资格。后续性能测量还需完成实际运行器接入、当前测量条件核对、preflight、完整 B 审查和正式 C 阶段。
