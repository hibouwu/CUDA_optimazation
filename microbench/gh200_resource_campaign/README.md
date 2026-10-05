# GH200/SM90 资源微基准

目标为 `sm_90a`，在 ROMEO 单 GPU Slurm 分配中运行。v1 入口和原始结果保留；本轮 v2 按独立 A/B/C 审查扩展资源家族和受控组合。

实验范围和后续顺序见 [GH200 实验入口](../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/README.md)。通用模型和硬件参数继续使用 `Docs/ModelEvaluation/gemm/`。

剩余六类实验改为[按家族连续执行](../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/RUN-REMAINING.md)。S15 新连续入口及参数候选报告为：

```sh
bash microbench/gh200_resource_campaign/runners/run_s15_once.sh --help
python3 -B microbench/gh200_resource_campaign/report_tma_tensor_2d.py --help
```

`run_s15_once.sh` 复用完整已签 formal-sampling 源码包，默认一次 GPU 作业连续运行剩余 12 配置，并自动安排无 GPU 的普通 CPU 重算。提交前核完整源码和全实验容量。其他五类尚需接齐各方案列明的正式入口，不把现有短诊断包装为完整测量。独立结果资格仍以最终 C 为准。

## v2 总控

当前计划和教学入口分别为 [PLAN](../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/PLAN.md) 和 [GUIDE](../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/GUIDE.md)。`contracts/` 定义实验，`probes/` 实施 CUDA 探针，`common/` 保存公共协议，`runners/` 管理执行，`auditors/` 独立重算，`tests/` 保存 CPU 检查和负例。

```sh
python -B microbench/gh200_resource_campaign/run_suite.py --help
python -B microbench/gh200_resource_campaign/run_suite.py plan \
  --contract microbench/gh200_resource_campaign/contracts/shared_memory.json
python -B -m unittest discover -s microbench/gh200_resource_campaign/tests -v
```

六个主要入口为 `plan/run/status/resume/audit/report`，另有受审查约束的 `finalize`。新运行按 `results/gh200_resource_campaign/<suite-id>/<family>/<run-id>/` 保存。`run --preflight` 用于 B 阶段设备检查，不能通过恢复变成正式样本；正常 `run` 需要相应独立门禁。并非所有计划家族都已实现，未注册的适配器会明确拒绝。

当前尚未完成全轮交付。实际可重放的 v2 示例、证据边界与原始字段手算见 GUIDE。以下内容保留 v1 的原有用法；v1 的两次预热和恢复规则不适用于 v2。

## v1 文件与结果

| 文件 | 职责 |
|---|---|
| `audit_legacy.py` | 独立重算三批历史数据；不调用旧分析器、不改写旧报告 |
| `resource_probe.cu` | 设备属性、SMEM scalar stride、global load/store/duplex |
| `run_campaign.py` | 冻结配置与源码、编译、独立进程采样、恢复和状态 |
| `audit_campaign.py` | 独立重算字节数、时间、SM 覆盖、统计和工件哈希 |
| `run_on_romeo.sh` | 在已分配节点加载 CUDA 12.9 |
| `plot_results.py` | 使用 Matplotlib 生成分作用域 SVG/PNG 与来源清单 |
| `test_auditors.py` | 验证错误工作量、时钟、正确性和缓存主张能被拒绝 |

新增结果统一放 `results/gh200_resource_campaign/<run-id>/`。旧结果保留在 `microbench/gh200_l0/results/`。这些结果被当前忽略规则排除，本地可读不表示已纳入 Git。

新运行保留配置清单、源码快照、编译命令和日志、二进制、SASS、设备属性、环境、遥测、原始 trial、进程记录、进度、状态、summary、报告和完成哈希。未提交源码按精确文件 SHA-256 冻结，不将旧 Git HEAD 当成新文件版本。

## 当前结果入口

- [旧三批计算复审](../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/EXP-01-compute-audit.md)：1,596 次正式记录、84 次空窗口、81 个目标函数循环检查。
- [SMEM/global 实验与结果](../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/EXP-02-memory-paths.md)：采用 `20261001-memory-c`，12×10 次独立进程。
- `memory-a` 编译失败；`memory-b` 的 SMEM load 外提使请求量与实际执行不符，后审否决该批次建模使用。失败和否决记录均保留。

## 历史复审与计划

```bash
python3 microbench/gh200_resource_campaign/audit_legacy.py \
  --output results/gh200_resource_campaign/NEW-legacy-audit

python3 microbench/gh200_resource_campaign/run_campaign.py --plan
```

复审输出必须是新目录；plan 不编译、不申请 GPU。旧批次的 `integrity_pass` 与 `qualified_like_thor` 分开：算术和工件检查通过，不代表当时已满足外部重复、计数器、预先冻结记录等正式条件。

## 运行与恢复

在已分配 GH200 节点的登录 shell 中执行：

```bash
source microbench/gh200_resource_campaign/run_on_romeo.sh \
  "$PWD/results/gh200_resource_campaign/NEW-memory"
```

申请步骤见 [ROMEO 访问说明](../../docs/romeo_gh200_access.md)。必须有 `SLURM_JOB_ID`，CUDA 仅可见一张 GH200；不在登录节点运行 GPU 工作，不修改频率、功率或权限。

当前共 12 个配置，每配置默认 10 次独立进程，按轮打乱配置。每个进程两次同路径预热后测一次；预热不构成缓存命中或热稳态证明。进程超时 120 秒，编译超时 180 秒，Slurm 也需有限时长；失败保存诊断并停止。

```bash
python3 microbench/gh200_resource_campaign/run_campaign.py \
  --output results/gh200_resource_campaign/EXISTING-memory --resume
```

恢复要求源码、配置、二进制、SASS 和已完成 trial receipt 均一致。出现有 raw 但无 receipt 的试验时，保留失败目录并使用新 run ID。不能静默合并不同设备身份和运行条件。

## 审查、图表与 CPU 检查

```bash
python3 microbench/gh200_resource_campaign/audit_campaign.py \
  results/gh200_resource_campaign/RUN_ID --require-complete
python3 microbench/gh200_resource_campaign/plot_results.py \
  results/gh200_resource_campaign/RUN_ID
python3 -m unittest discover \
  -s microbench/gh200_resource_campaign -p test_auditors.py -v
```

正式复查使用 `--require-complete`：只读重算 summary，核对完成哈希和 progress，并检查目标访存留在计时循环中。SMEM 使用 PTX `ld/st.volatile.shared`，每个未展开的外层循环必须恰有 8 次相应 LDS/STS；仅有 C++ `asm volatile` 不保证 PTX 编译阶段保留每次访存。

结果等级为 `timing_validated`。NCU 独立尝试；未证明时 `cache_residency_proven` 和 `physical_hbm_bytes_proven` 保持 false。图表分别展示单 CTA 的 SMEM B/clock64 cycle 和整卡 global 请求 payload GB/s，误差条为最小至最大样本。CPU 检查验证审查器，不替代 GPU 执行。
