# EXP-06 实测：warp 协同 MMA 的依赖链、线程数与固定长度

本记录对应 `20261001-resource-suite-v2/legacy_mma/formal-v3-b`。55 个配置各完成 10 个独立进程，共 550 条合并样本；全部达到预定统计稳定条件。离线后审已通过，完整独立 C 审查尚未完成，本文与图表暂不授予合格参数导出资格。

原 [MMA 实验约定](EXP-06-legacy-mma.md) 和 [计时修订合同](../../../../../microbench/gh200_resource_campaign/contracts/legacy_mma_timer_v3.json) 保持冻结。本次正式作业为 `730340`，设备是 132 SM 的 GH200，驱动 590.48.01，CUDA 12.9 V12.9.41、目标 `sm_90a`。正式 run 导入已审查的二进制及 24 个校准点的冻结长度，具体身份见 [run_spec.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_mma/formal-v3-b/run_spec.json)。

## 实验条件与工作量

每个 warp 的 32 个线程共同执行 `D=A×B+D`，矩阵片段分布在各线程寄存器中。每个协同 MMA 的工作量是 `2×M×N×K`，不能把每个线程视为独立完成一个矩阵乘法。每条累加链每轮执行 16 个 MMA；不同链有独立的累加寄存器。

| 输入形式 | 累加类型 | M×N×K | 每个 warp 协同 MMA 的 FLOP |
|---|---|---|---:|
| FP16 | FP32 | 16×8×16 | 4096 |
| BF16 | FP32 | 16×8×16 | 4096 |
| TF32 | FP32 | 16×8×8 | 2048 |
| FP64 | FP64 | 8×8×4 | 512 |

因此总工作量为：

`W = CTA数 × 轮数 × 每链每轮MMA数 × 链数 × (每CTA线程数/32) × 2MNK`。

这里无需再乘 FP16/BF16 的寄存器打包 lane 数；矩阵形状已经包含全部有效元素。寄存器操作数准备在计时前完成，外部矩阵读写 payload 记为零；结果归约、转换、shared 排空、同步和循环控制占用时间，但不计入 MMA 的 FLOP 分子。

| 来源组 | 有限配置 | 点数 |
|---|---|---:|
| 旧 initial | 4 种输入形式，32 线程，1/8 条链，单 CTA，固定 128/512/2048 轮 | 24 |
| 旧 sustained | 4 种输入形式，32/128/256 线程，8 条链，单 CTA/全 GPU，各自校准后冻结长度 | 24 |
| 旧 audit | 仅 FP16，256 线程、8 条链，单 CTA/全 GPU，固定 8192/32768/65536 轮 | 6 |
| 空计时窗口 | FP16、256 线程、8 条链的旧单 CTA 控制坐标，0 轮 | 1 |

54 个计算点报告 FLOP 速率；空窗口仅报告 `cycles/window`，不参与速率图或从计算窗口中扣除。同名配置保留各自旧来源与完整 mapping ID。

## 计时范围与实际校准长度

单 CTA 的主分母是同一 CTA 的 `stop_cycle-start_cycle`，单位为 `FLOP/clock64_cycle/CTA`。全 GPU 的主分母是所有 CTA 的 `max(stop_ns)-min(start_ns)`，单位为 `GFLOP/s/GPU`。后者数值等于 `FLOP/ns`，不经固定频率换算成 SM 周期，也不由单 CTA 速率乘 SM 数得到。

计时前完成操作数准备和初始化；thread0 保存起始时间后，全 CTA 再经过入口 barrier。循环后，全部累加值参与 volatile shared 排空，经过结束 barrier 才取终点。窗口包括这些同步与排空成本；每个输出元素的全局写回和主机比较位于窗口外。

sustained 的 24 点使用下表长度，每点所有正式进程保持一致。长度来自 [resolved_cases.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_mma/formal-v3-b/resolved_cases.json) 绑定的 pilot；目标为 100 ms，按事件时间比例截断后限制到 8192–65536 轮。达到上限不表示实际运行了 100 ms。全 GPU 的 CTA 数为 `132×min(4, occupancy_limit_ctas_per_sm)`，该 occupancy 是 API 给出的资源上限。

| 形式 | 每 CTA 线程数 | 单 CTA 实际轮数 | 全 GPU 实际轮数 | 全 GPU CTA 数 |
|---|---:|---:|---:|---:|
| f16 | 32 | 65536 | 65536 | 528 |
| f16 | 128 | 65536 | 59496 | 528 |
| f16 | 256 | 65536 | 39671 | 396 |
| bf16 | 32 | 65536 | 65536 | 528 |
| bf16 | 128 | 65536 | 59497 | 528 |
| bf16 | 256 | 65536 | 39671 | 396 |
| tf32 | 32 | 65536 | 65536 | 528 |
| tf32 | 128 | 65536 | 57901 | 528 |
| tf32 | 256 | 65536 | 38739 | 396 |
| f64 | 32 | 65536 | 65536 | 528 |
| f64 | 128 | 65536 | 24160 | 528 |
| f64 | 256 | 48317 | 12082 | 528 |

## 两条真实记录的手算

单 CTA 例子是 `legacy-ef3663785df2dbe2f8ef3fae0016ce45e694f1777671a54793e5bf1310e702ee` 的 `batch_00/trial_00/attempt_00/raw.jsonl`：FP16、1 CTA、32 线程、1 条链、每轮每链 16 个 MMA、128 轮。

- 工作量：`1×128×16×1×(32/32)×(2×16×8×16)=8,388,608 FLOP`。
- 同一 CTA 的起止 clock64 为 `12,251,766,074,851` 和 `12,251,766,124,202`，相差 `49,351 cycle`。
- `8,388,608/49,351=169.9784806792 FLOP/clock64_cycle/CTA`。

整卡例子是 `legacy-99991e595233dbd36561ecdb7181a48870938eef6439b6d2bc5e6b77f9ec87b8` 的相同进程坐标：FP16、528 CTA、每 CTA 32 线程、8 条链、每轮每链 16 个 MMA、65,536 轮。

- 工作量：`528×65536×16×8×(32/32)×4096=18,141,941,858,304 FLOP`。
- 所有 CTA 的 globaltimer 包络从 `1,790,848,540,226,803,072` 到 `1,790,848,540,254,951,840 ns`，相差 `28,148,768 ns`。
- `18,141,941,858,304/28,148,768=644,502.162876329 GFLOP/s/GPU`。

这两个数来自单条 raw，不是十进程中位数。原始相对路径与 SHA256 见 [worked_examples.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-mma-formal-v3-b/worked_examples.json)。

## 代表结果与完整图表

以下均取 sustained、128 线程、8 条链、每轮每链 16 个 MMA，每行是 10 个独立进程的中位数。两种计时范围分别列出，比较时同时保留矩阵形状和循环长度。

| 形式 | 实际轮数 | FLOP/clock64_cycle/CTA | 样本 CV |
|---|---:|---:|---:|
| f16 | 65536 | 2668.103258 | 0.000000% |
| bf16 | 65536 | 2668.103258 | 0.000000% |
| tf32 | 65536 | 1312.346441 | 0.000000% |
| f64 | 65536 | 127.937235 | 0.000000% |

| 形式 | 实际轮数 | CTA 数 | GFLOP/s/GPU | 样本 CV |
|---|---:|---:|---:|---:|
| f16 | 59496 | 528 | 656830.594241 | 0.243410% |
| bf16 | 59497 | 528 | 653778.600558 | 0.440422% |
| tf32 | 57901 | 528 | 323536.678689 | 0.000130% |
| f64 | 24160 | 528 | 33453.937259 | 0.305472% |

在本次相同单 CTA 条件下，FP16 与 BF16 的周期指标相同；TF32 每个逻辑 MMA 的工作量减半，FP64 的矩阵形状及指令形式又不同。表中速率同时反映这些计算定义与实际服务时间，不能只按类型名字解释为精度转换的收益。CV 为零表示本批主指标记录一致，不表示设备全部状态恒定。

![FP16 单 CTA 条件比较](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-mma-formal-v3-b/f16-one_cta-1.png)

图中 `t` 是每 CTA 线程数，`c` 是每个 warp 的独立累加链数，`q` 是每链每轮逻辑 MMA 数，`n` 是实际轮数。通用绘图字段 `packed_lanes` 的 `lanes1` 标签只用于兼容标量图表字段（scalarLanes），不代表 MMA 只有一个参与 lane，也不作为 MMA 工作量的额外乘数。方括号为 case ID 后八位。

点显示全部预热合格的进程样本，菱形表示中位数，误差条为最小到最大值；它们不是置信区间，重合也不表示只有一次采样。四种输入、两个 scope 的完整 8 张图见 [图表索引](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-mma-formal-v3-b/index.md)。[cases.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-mma-formal-v3-b/cases.csv) 包含全部 55 点和空窗口；[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-mma-formal-v3-b/samples.csv) 保留全部 550 条正式样本、raw 路径及哈希。

## 固定 kernel 的长度拟合

仅对同一实际编译 kernel、同一 scope 和相同配置下的固定长度点拟合 `T(n)=a+b×n`；校准点不进入拟合。分析从冻结源码的 case 表取得 kernel 符号，并要求该符号出现在实际 SASS 审查中。

例如上述 FP16、32 线程、1 条链使用 `lc_847f837734e9012c7b55`。128/512/2048 轮的时间中位数为 49,351/196,807/786,631 cycle，得到：

`T(n)=199+384×n cycle/CTA`。

384 是这个 kernel 每轮 16 个依赖 MMA 加循环控制的增量，199 是当前计时窗口的拟合常数项；不能直接把 `384/16` 宣称为裸 MMA 延迟。不同链数或线程数需要各自观测，不能沿用此直线。

全部 10 组拟合见 [fixed_length_fits.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-mma-formal-v3-b/fixed_length_fits.json)。FP16、256 线程的整卡固定长度组拟合为约 `T(n)=-127083.68+2531.66n ns/GPU`，样本内最大相对误差约 0.0811%；负截距说明它只是有限长度区间内的描述性直线，不能解释成负启动开销，也不能用于预测零长度。所有拟合均没有独立留出验证，不导出通用延迟或完整 GEMM 预测参数。

## 正确性、实际指令与证据边界

正式均匀输入为 `A=B=1/16`、`D0=0`，每个结果元素的参考值为 `轮数×16×K/256`。每次启动重新初始化累加器，检查所有 CTA、warp、链和输出片段。另有 1/2 轮的非均匀输入检查，用独立主机矩阵参考验证片段坐标；这与正式均匀输入的计时分开，不能推广为任意输入分布的误差结论。

[实际 SASS 审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_mma/formal-v3-b/build/sass_audit.json) 记录 FP16/BF16/TF32 的 HMMA 与 FP64 的 DMMA；当前 1 链与 8 链循环分别有 16 和 128 个目标机器指令位置。它们是本次编译的静态循环计数，不能代替硬件计数器的动态发射统计。NCU 复用已有 `permission_denied`、`family_profile=false` 的权限证据，未重新运行家族 profiling；本次没有动态计数器测量。

本地 [有界浮点后审](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/post_audits/bounded-float-replay-v1-r2/legacy_mma-formal-v3-b.json) 通过，独立重算 605 条阈值判定，保留 24 个白名单派生统计的浮点差异，允许范围至多 2 ULP。raw、工作量、长度、单位、哈希与状态保持严格检查。原严格重放的失败记录仍保留，后审没有改写原 run。统计稳定、离线复核与完整 C 资格分别记录；当前状态仍为 `collected_pending_review`。

## 复现

从项目根目录执行；分析输出使用原 run 外的新目录，依赖 matplotlib。先重放归档证据，再生成表、图和手算/拟合记录：

```bash
python -B microbench/gh200_resource_campaign/audit_replay.py \
  results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_mma/formal-v3-b \
  --policy microbench/gh200_resource_campaign/contracts/offline_replay_v1.json \
  > /tmp/mma-replay.json

python -B microbench/gh200_resource_campaign/report_compute_v2.py \
  results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_mma/formal-v3-b \
  --replay /tmp/mma-replay.json --output /tmp/mma-analysis-new
```

原 GPU 作业入口见 [job-730340/driver.sh](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/job-730340/driver.sh)。GPU 重现需有效单 GPU Slurm 分配，新实验新建 run；同一次检查点恢复使用 `resume`，不覆盖原快照。本文只读取既有结果并撰写说明，没有重新编译或启动 GPU。
