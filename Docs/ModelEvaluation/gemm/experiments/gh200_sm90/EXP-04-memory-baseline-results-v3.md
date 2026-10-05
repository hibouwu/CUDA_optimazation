# EXP-04 实测：访存基线、stride=1 波动与空窗口

本记录对应 suite `20261001-resource-suite-v2` 的 `memory_baseline/formal-v3-a`，ROMEO 作业 `730329`。数据采集与本地离线重放已经完成，独立 C 审查尚未完成，暂不导出合格模型参数。原 [实验约定](EXP-04-memory-baseline-v2.md) 保留为冻结 A 版本；本次使用 [计时修订合同](../../../../../microbench/gh200_resource_campaign/contracts/memory_baseline_timer_v3.json)，单 CTA 空窗口以 clock64 周期报告，其他配置保留各自计量边界。

## 实验问题与对应参数

本实验比较给定线程数、访问模式和工作集下的请求字节服务速率，并检查跨进程可重复性。它给出的是片上访问与全局访问路径的条件参数：SMEM 使用单 CTA 的 B/clock64 cycle，global 使用整卡的请求 payload GB/s。结果不能直接代替完整 GEMM 带宽，也不能将单 CTA 数值乘以 SM 数推得整卡结果。

实际设备为 NVIDIA GH200 120GB，132 SM，设备查询 L2 为 62,914,560 B（60 MiB），CUDA runtime 12.9，CUDA driver API 13.1。完整 UUID、工具链和 Slurm 条件保存在 run 的 `environment/`，本次与 B 阶段使用同一 GPU UUID。驱动 API 版本与驱动软件发行号分别记录，不混为一个版本。

## 配置与计量

| 配置 | 工作集与执行条件 | 指标 |
|---|---|---|
| SMEM read stride=1/2/4/8/16/32 | 1 CTA、256 线程、32 KiB 分配；每线程每轮 8 次 4 B load，8192 轮 | 请求 B/clock64 cycle/CTA |
| SMEM write stride=1 | 同样的 CTA、分配和轮数；互不重叠写入 | 请求 B/clock64 cycle/CTA |
| global read .ca/.cg，8 MiB | 528 CTA、256 线程，64 轮；实际分配按执行网格向上对齐 | 请求 payload GB/s/GPU |
| global read .cg / write，256 MiB | 同样的执行网格，16 轮 | 请求 payload GB/s/GPU |
| global dependent copy，128 MiB | 输入、输出各一块，16 轮；分子计读加写 | 请求 payload GB/s/GPU |
| 单 CTA / 整卡空窗口 | 无访存 payload；整卡空窗口不要求所有 SM 参与 | cycles/window / ns/window |

SMEM 计时分母为同一 CTA 的 `stop_cycle-start_cycle`。整卡计时使用所有 CTA 的最早 globaltimer 起点至最晚终点；不跨 SM 相减 clock64。global 写入的完成边界包含 device fence。初始化、host 传输和数值校验不在设备计时窗口中；空窗口不用于减去实际测量时间。

## 从一条原始记录手算

取 `batches/smem_read_stride1/batch_00/trial_00/attempt_00/raw.jsonl` 的 trial 行：

- `threads=256`、`iterations=8192`，每轮每线程 8 次 4 B 请求，故 `work_count=256×8192×8×4=67,108,864 B`。
- 唯一 CTA 的 `start_cycle=10371768058746`，`stop_cycle=10371768763521`，差为 `704775 cycle`。
- 速率为 `67,108,864 / 704775 = 95.2202674612465 B/clock64 cycle/CTA`，与归档一致。

这里没有把 32 KiB 分配大小再乘一次。相同地址可能被多次读取，每次实际执行的请求均计入分子；因此该数值也不是“不同地址字节数/时间”。这条样本本身通过校验，不代表该配置跨进程稳定。

## 全部采样结果

每个配置每批目标为 10 个独立进程。预热最多 30 次；末 5 次窗口的样本 CV≤2% 才记作收敛。正式 CV>5% 按协议最多采三批；合并所有预热合格的有效样本，不挑选最好批次。共保留 171 条进程试验记录，其中 170 条参加合并统计。

### SMEM 服务

单位：B/clock64 cycle/CTA。下表中位数与 CV 均使用合并样本；CV 用百分数显示。

| 配置 | 样本数 | 中位数 | CV | 统计状态 |
|---|---:|---:|---:|---|
| smem_read_stride1 | 30 | 95.220267 | 9.850950% | 有界复测后仍不稳定 |
| smem_read_stride2 | 10 | 63.982396 | 0.000865% | 稳定，待 C 审查 |
| smem_read_stride4 | 10 | 31.997170 | 0.000377% | 稳定，待 C 审查 |
| smem_read_stride8 | 10 | 16.000206 | 0.000113% | 稳定，待 C 审查 |
| smem_read_stride16 | 10 | 8.000340 | 0.000092% | 稳定，待 C 审查 |
| smem_read_stride32 | 10 | 4.000272 | 0.000046% | 稳定，待 C 审查 |
| smem_write_stride1 | 10 | 127.919240 | 0.000197% | 稳定，待 C 审查 |

### 全局访问服务

单位：请求 payload GB/s/GPU。下表中位数与 CV 均使用合并样本；CV 用百分数显示。

| 配置 | 样本数 | 中位数 | CV | 统计状态 |
|---|---:|---:|---:|---|
| global_read_ca_8m | 10 | 13370.559505 | 0.211106% | 稳定，待 C 审查 |
| global_read_cg_8m | 10 | 7834.052473 | 0.957650% | 稳定，待 C 审查 |
| global_read_cg_256m | 10 | 3515.286489 | 0.310996% | 稳定，待 C 审查 |
| global_write_256m | 10 | 3837.604101 | 0.683582% | 稳定，待 C 审查 |
| global_duplex_128m | 10 | 3411.479138 | 0.173139% | 稳定，待 C 审查 |

### 计时控制

单位：分别标注。下表中位数与 CV 均使用合并样本；CV 用百分数显示。

| 配置 | 样本数 | 中位数 | CV | 统计状态 |
|---|---:|---:|---:|---|
| empty_one_cta (cycles/window) | 10 | 34.000000 | 0.000000% | 稳定，待 C 审查 |
| empty_all_gpu (ns/window) | 20 | 128.000000 | 12.710662% | 有界复测后仍不稳定 |

![SMEM：全部样本、中位数与最小至最大值](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/memory-baseline-formal-v3-a/smem.png)

![全局访问：全部样本、中位数与最小至最大值](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/memory-baseline-formal-v3-a/global.png)

点表示所有参加合并统计的进程样本，菱形表示中位数，误差条表示最小至最大值，**不是置信区间**。数值接近的点会重叠；逐点数据在分析目录的 `samples.csv`，含未合并记录及理由字段。

### stride=1 为什么暂时不能给出一个参数

stride=1 读取的合并 CV 为 9.85%，三个批次后仍超过 5% 阈值。样本主要分布在约 95.22 与 126.99 B/cycle/CTA。各进程预热已收敛，仍出现跨进程差异，说明“本进程末几次窗口稳定”不能证明跨进程服务稳定，更不能证明长期热稳态。

![stride=1 三批全部样本](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/memory-baseline-formal-v3-a/stride1.png)

当前证据不足以把两档归因到特定调度、频率或 bank 行为。该配置保留为分布证据，不选择较快批次或最大值作为模型参数；本协议不再追加第四批。后续若研究原因，应另建受控对照并保留本次结果。

整卡空窗口首批第一个进程预热未收敛，runner 因此停止该批；该记录的值为 128 ns，仍保留在 raw 和 CSV，但不纳入合并统计。后续两批共 20 个合格进程的 CV 仍为 12.71%。这解释了表中 20 个合并样本，而不是悄悄删除某个值。单 CTA 空窗口恒为 34 cycles，仅作为该条件下的计时控制，不从 payload 窗口扣除。

## 正确性、后审与适用范围

SMEM 读取逐线程校验地址公式生成的 checksum；写入使用 poison 初值后校验写入区域。global 读取核对所有 uint4 lane 的 checksum；global 写入用独立验证 kernel 检查全部输出。SASS 检查确认访存在实际循环中；这些检查的独立 B 记录与本次完整证据均保留在 suite 内。

本地重放采用已独立审查的 `bounded_float_replay_v1`，仅允许白名单派生浮点统计相差至多 2 ULP。完整输入哈希、工作量、单位、整数、状态与阈值判定不放宽。本次重放保留 15 个派生数差异记录，独立核对 189 条阈值判定；原冻结严格工具的结果也保留，不能将后审通过描述为原严格比较已通过。

NCU 的本次能力检查返回权限拒绝，因此缓存驻留、命中率和物理 HBM 流量未证明。8 MiB 名义工作集小于设备查询的 L2 容量，但这不足以单独证明命中层级。图中的请求速率约 13,371 GB/s 不能解释为 HBM 带宽；256 MiB 配置的速率也只在既定访问、计时和缓存条件下成立。`global_duplex` 是依赖复制，不能替代 S13 的独立读写竞争实验。

稳定的 11 个 payload 配置是后续条件参数的候选；稳定的单 CTA 空窗口另作计时控制。所有参数都需独立 C 审查后才能获得资格。当前绘图工具不输出合格参数文件。

## 离线复现

从项目根目录运行；原 run 可复制到其他目录，分析输出必须是原 run 外的新目录。所需 Python 依赖为 matplotlib（本次 3.10.8），重放和 CSV 处理使用标准库。

```bash
python -B microbench/gh200_resource_campaign/audit_replay.py \
  results/gh200_resource_campaign/20261001-resource-suite-v2/memory_baseline/formal-v3-a \
  --policy microbench/gh200_resource_campaign/contracts/offline_replay_v1.json \
  > /tmp/s04-replay.json

python -B microbench/gh200_resource_campaign/report_memory_v2.py \
  results/gh200_resource_campaign/20261001-resource-suite-v2/memory_baseline/formal-v3-a \
  --replay /tmp/s04-replay.json --output /tmp/s04-analysis-new
```

分析脚本逐文件核对后审收据中的完整输入清单，从 raw 重算每条速率，保存 CSV、手算字段、PNG/SVG 与来源哈希。换目录后图像及统计可重建；重放收据含新环境信息，收据哈希不要求与先前环境相同。

GPU 重现使用本次归档中 `implementation/job-730329/driver.sh` 对应的 Slurm 条件与冻结合同，新建 run ID，并先检查 B 证据依赖是否仍适用。不得向现有 `formal-v3-a` 追加新协议样本。原始记录受项目忽略规则管理，本地位置为 `results/gh200_resource_campaign/20261001-resource-suite-v2/`，没有自动提交或推送。
