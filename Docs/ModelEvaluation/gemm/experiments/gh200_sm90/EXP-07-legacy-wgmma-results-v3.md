# EXP-07 实测：WGMMA 的提交等待、参与组与 SS/RS 来源

本记录对应 `20261001-resource-suite-v2/legacy_wgmma/formal-v3-b`：85 个配置各完成一批 10 个独立进程，共 850 条正式样本，全部达到预定统计稳定条件。当前归档资格仍为 `measurement_evidence_pending_C_review`；浮点后审通过不等于完整 C 审查通过，本文和图表暂不导出合格性能参数。

正式采样由作业 `730344` 启动，并在同一 run 中由 `730347` 恢复完成，设备为 `romeo-a057` 的 132 SM GH200，驱动 590.48.01，CUDA 12.9 V12.9.41，目标 `sm_90a`。源、二进制、环境及审查身份见 [run_spec.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_wgmma/formal-v3-b/run_spec.json)，本次采用 [计时修订合同](../../../../../microbench/gh200_resource_campaign/contracts/legacy_wgmma_timer_v3.json)。[原实验约定](EXP-07-legacy-wgmma.md) 保留其历史设计状态，实测结论以本记录和冻结归档为准。

作业 `730344` 先收集 678 条样本，在 10:13:38 因剩余 139.4 秒不足单进程 120 秒加 20 秒清理余量而保存检查点；`730347` 于 10:17:25 恢复，补齐余下 172 条。恢复沿用同一 snapshot、binary 和 24 个冻结 pilot 长度，没有重新选择长度。独立复核的 850 组 case/batch/trial 坐标全部唯一，采样进程时间区间不重叠，没有失败或重复采样收据。时间见归档 [progress.jsonl](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_wgmma/formal-v3-b/progress.jsonl)。

## 85 个点具体覆盖什么

下表从本次 WGMMA 合同逐项统计，不沿用 FMA/MMA 的来源表。81 个旧点保留 `legacy.key`、原配置 ID 与来源行号；四个新增 BF16 来源对照的 `legacy=null`，在 CSV 中标为 `new representative`。

| 来源 | 本合同保留的条件 | 点数 |
|---|---|---:|
| `20260930-initial` | FP16/BF16；1 warpgroup、chains=1/2、batch=1/4/16、wait0；单 CTA × 128/512/2048 轮 | 36 |
| `20260930-sustained` | FP16/BF16；1/2 warpgroup、chains=2、batch=16、wait0/3/7；单 CTA/全 GPU，各自校准后冻结长度 | 24 |
| `20260930-audit` 计算点 | 仅 FP16；1 warpgroup、chains=2、batch=16、wait0/3/7；两范围 × 8192/32768/65536 轮 | 18 |
| `20260930-audit` 空窗口 | 上述三个 wait 配置，单 CTA，0 轮 | 3 |
| 新 BF16 来源对照 | SS/RS × 单 CTA/全 GPU；1 warpgroup、chains=2、batch=16、wait0、8192 轮 | 4 |

82 个计算点报告 FLOP 速率；三个空窗口只报告 `cycles/window`，中位数依次为 wait0 的 846、wait3 的 849、wait7 的 849 cycle。它们不进入速率图，也不从计算时间中扣除。这里使用 v3 单 CTA 空窗口的 cycle 单位，不能把原设计文档的 `ns/window` 直接移植到本次结果。

## 128 个线程共同完成一次矩阵操作

本实验统一使用 `m64n64k16`、FP16/BF16 输入和 FP32 累加。每个 warpgroup 的 128 个线程共同执行一次 `D=A×B+D`，名义工作量为：

`2×M×N×K = 2×64×64×16 = 131072 FLOP`。

128/256 个 CTA 线程分别对应 1/2 个参与 warpgroup；128 个线程不是各自完成一份完整矩阵乘法。每个 warpgroup 的每条 chain 有独立 D 累加器，每条 chain 每轮执行 batch 次 WGMMA。因此：

`W = blocks × iterations × batch × chains × (threads/128) × 131072`。

例如单 CTA、128 线程、2 chains、batch=16、8192 轮，`W=34,359,738,368 FLOP`。commit、wait、同步、循环控制与结果排空不增加这个分子，但占用计时窗口。计时循环的外部矩阵 payload 记为零不表示没有 SMEM 操作：SS 的 A/B 描述符、RS 的 B 描述符及最终结果排空都涉及 SMEM。

每轮发出 `batch×chains` 条 WGMMA 后执行一次 `wgmma.commit_group`，随后 `wgmma.wait_group 0/3/7`。这里的 wait 值控制本线程所提交 WGMMA group 的完成等待，不是 warpgroup 数，也不是一次提交中的指令条数。wait0 要求已提交组全部完成；wait3/7 允许保留相应上界的未完成组，以便后续发出与完成重叠，不能据此推定硬件队列实际占用了多少项。

initial 每轮 wait0，合同没有额外的循环后 wait。sustained、audit 和四个新来源点在循环后另外执行 wait0，然后读取全部 D 参与结果排空。SS/RS 的操作数准备和 fence 在计时前完成，所有参与线程一致执行对应发出/提交/等待协议；不能把缺少最终排空的窗口与这些实测值比较。

## 计时范围与冻结长度

thread0 保存 globaltimer/clock64 起点后，全 CTA 经过入口 barrier 才进入计算。循环、提交、完成等待和全部累加值参与的 volatile double SMEM 排空都在窗口内；结束 barrier 后才读取终点。操作数准备、输出全局写回及主机比较在窗口外。

单 CTA 的主分母是同一 CTA 的 `stop_cycle-start_cycle`，单位 `FLOP/clock64_cycle/CTA`。全 GPU 使用所有 CTA 的 `max(stop_ns)-min(start_ns)` 包络，单位 `GFLOP/s/GPU`，数值等于 FLOP/ns。全 GPU 包络包括不同 CTA 起止时刻的离散，不能用单 CTA 速率乘 SM 数替代，也不把 ns 按固定频率换成周期。旧计时边界仅保留为历史映射，新旧绝对周期不是同一窗口。

24 个 sustained 点各有原 pilot 收据，见 [resolved_cases.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_wgmma/formal-v3-b/resolved_cases.json) 和 [calibration_origin](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_wgmma/formal-v3-b/calibration_origin)。以 8192 轮的 CUDA event 时长 `t_ms` 计算 `clamp(int(8192×100/max(t_ms,0.01)),8192,65536)`，再绑定 binary、contract、device、pilot raw/receipt 哈希。正式进程只使用导入的冻结值，不在每个进程内重校准；达到 65536 上限不表示实际运行满 100 ms。

| 输入 | warpgroup/CTA | wait | 单 CTA 冻结轮数 | 全 GPU 冻结轮数 | 全 GPU CTA 数 |
|---|---:|---:|---:|---:|---:|
| f16 | 1 | 0 | 65536 | 58837 | 396 |
| f16 | 1 | 3 | 65536 | 58916 | 396 |
| f16 | 1 | 7 | 65536 | 58883 | 396 |
| f16 | 2 | 0 | 65536 | 65536 | 132 |
| f16 | 2 | 3 | 65536 | 65536 | 132 |
| f16 | 2 | 7 | 65536 | 65536 | 132 |
| bf16 | 1 | 0 | 65536 | 58733 | 396 |
| bf16 | 1 | 3 | 65536 | 58916 | 396 |
| bf16 | 1 | 7 | 65536 | 58924 | 396 |
| bf16 | 2 | 0 | 65536 | 65536 | 132 |
| bf16 | 2 | 3 | 65536 | 65536 | 132 |
| bf16 | 2 | 7 | 65536 | 65536 | 132 |

这 24 点均为 SS。整卡网格为 `132×min(4,实际 occupancy_limit_ctas_per_sm)`；128 线程点实际 396 CTA，256 线程点实际 132 CTA。增加每 CTA 的 warpgroup 数同时改变了资源与可驻留 CTA 数，不能只按参与组数解释整卡吞吐差异。

## 两条 raw 的手算

单 CTA 例子为 `legacy-48971f19366030808d95f5d5b652bebf4a4b9845449e56e97686e64daaaffd57` 的 `batch_00/trial_00/attempt_00/raw.jsonl`：FP16、SS、128 线程、1 chain、batch=1、wait0、128 轮。

- 工作量：`1×128×1×1×(128/128)×131072 = 16,777,216 FLOP`。
- CTA clock64 从 `13,173,792,204,550` 到 `13,173,792,218,545`，相差 `13,995 cycle`。
- `16,777,216/13,995 = 1198.8007145409 FLOP/clock64_cycle/CTA`。

全 GPU 例子为 `legacy-81992572e8cdbf0c5353c96140908af4628a45bcf18db900c1443e3df938e708` 的相同进程坐标：FP16、SS、396 CTA、每 CTA 128 线程、2 chains、batch=16、wait0、58,837 轮。

- 工作量：`396×58837×16×2×1×131072 = 97,724,984,721,408 FLOP`。
- globaltimer 包络从 `1,790,849,280,518,059,808` 到 `1,790,849,280,620,826,112 ns`，相差 `102,766,304 ns`。
- `97,724,984,721,408/102,766,304 = 950943.8494684795 GFLOP/s/GPU`。

这两个数是单进程值，不是十进程中位数。精确 raw 路径、SHA256、字段和工作量见 [worked_examples.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/worked_examples.json)。

## wait 对照与固定 kernel 的长度拟合

先选固定长度、同输入、同来源和同参与者的 audit 点看 wait：FP16、SS、1 warpgroup、2 chains、batch=16、8192 轮，十进程中位数如下。

| wait | 单 CTA FLOP/clock64_cycle/CTA | 全 GPU GFLOP/s/GPU | 全 GPU CV |
|---:|---:|---:|---:|
| 0 | 3858.217118 | 980992.230583 | 0.454972% |
| 3 | 4095.536674 | 980027.898727 | 0.346748% |
| 7 | 4095.536674 | 980111.477593 | 0.436378% |

单 CTA 的 wait3/7 在这个条件下相同，且快于逐轮 wait0；整卡三者的中位数接近，不能据此写成增大 wait 就提高整卡吞吐。三种 wait 是不同 kernel 特化，计量对象包括其提交、等待和循环实现；表中没有测量裸指令延迟或硬件队列深度。

长度拟合进一步要求同一实际编译 kernel、scope 和其余条件相同，校准点不参与。分析从冻结源码的 case 表识别符号，再要求实际 SASS 审查包含该符号。仅在同 kernel 的多个固定长度点上拟合 `T(n)=a+b×n`。

FP16、SS、1 chain、batch=1、wait0 的 initial kernel `lc_f5801827fbd02dbe186f`，128/512/2048 轮时间中位数分别为 13,995/54,315/215,595 cycle，得到 `T(n)=555+105n cycle/CTA`。105 是本 kernel 一轮 WGMMA、commit/wait0 和控制流的增量，555 是当前窗口的拟合常数项，不能直接解释为 WGMMA 固有延迟或启动代价。

固定 8192/32768/65536 轮的 FP16 audit 点中，wait0 的 `lc_4a22f4ae00afd590e580` 得到 `T(n)=896+1087n cycle/CTA`；wait3 的 `lc_b212ecb59a0a7a6bd211` 和 wait7 的 `lc_d95a4da569eb33e34fbd` 均得到 `T(n)=949+1024n cycle/CTA`。这与上表的单 CTA 结果一致，但仍是各自提交/等待窗口的受控比较。

全部 18 组拟合见 [fixed_length_fits.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/fixed_length_fits.json)。例如 wait0 的整卡固定长度组为约 `T(n)=-184833.51+1727.812843n ns/GPU`，样本内最大相对误差约 0.7160%。负截距不能解释成负启动开销，亦不可据此预测零长度；全部拟合没有独立留出验证，`parameter_export=false`，不外推成完整 GEMM 模型。

## 新 BF16 SS/RS 代表点

四个新增点固定 128 线程、1 warpgroup、2 chains、batch=16、wait0、8192 轮。SS 的 A/B 都使用 SMEM 描述符；RS 的 A 使用每线程四个 b32 寄存器，B 仍使用 SMEM 描述符。两者使用相同逻辑矩阵和对应独立片段参考，输入准备在计时前完成。因此这些结果没有测量把 A 从全局内存准备成寄存器或 SMEM 的完整代价。

| 范围 | 来源 | CTA 数 | 十进程中位数 | CV |
|---|---|---:|---:|---:|
| 单 CTA | SS | 1 | 3858.217118 | 0.000000% |
| 单 CTA | RS | 1 | 3890.416501 | 0.000000% |
| 全 GPU | SS | 396 | 983912.826603 | 0.390844% |
| 全 GPU | RS | 396 | 980375.937913 | 0.419773% |

单 CTA 两行单位为 `FLOP/clock64_cycle/CTA`，全 GPU 两行为 `GFLOP/s/GPU`，不能跨两种单位排序。单 CTA RS 比 SS 的中位数高约 0.835%；整卡 RS 比 SS 低约 0.359%，同时两组 CV 分别约 0.420% 和 0.391%，样本范围重叠。这里只报告这一配对条件下的观测，未做统计显著性检验，也没有“RS 普遍更快”的结论。

实际资源仍需单列：SS 为 148 registers/thread、5136 B static SMEM；RS 为 146 registers/thread、3088 B static SMEM，两者 local size 都为 0、occupancy 上限都为 3 CTA/SM，因此本次全 GPU 配对均为 396 CTA。相同 occupancy 上限不代表寄存器/SMEM 使用相同，更不表示所有 warpgroup 在任意时刻都活跃。这个 BF16 单形状配对不能推广到 FP16、其他 wait、其他形状或完整 GEMM。

## 图表与完整数据

![BF16 全 GPU：明确标出参与组、wait 和 SS/RS](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/bf16-all_gpu-1.png)

图中 `wg` 为每 CTA 的 warpgroup 数，`c` 为每 warpgroup 独立累加链数，`q` 为每链每轮 WGMMA 数，`wait` 为每轮等待上界，`SS/RS` 为操作数来源，`n` 为实际轮数。方括号保留 case ID 后八位用于定位。CSV 的 `operand_source_form`、`collectives_per_CTA`、`wait` 与完整 case ID 应一起使用，不能仅按 BF16/FP16 标签合并不同来源。

散点保留全部预热合格的正式进程，菱形是中位数，误差条是 min/max；它们不是置信区间，重合不表示只采了一次。当前所有正式点各有 10 条样本，最大样本 CV 为 2.6001%，低于 5% 的正式稳定阈值；CV=0 只表示所记录主指标一致。

[图表索引](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/index.md) 包含 9 张 PNG 与对应 SVG，按 dtype/scope 分页。[cases.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/cases.csv) 保留全部 85 点及空窗口；[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/samples.csv) 保留全部 850 个进程值、raw 路径和哈希。[manifest.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/manifest.json) 绑定输入 summary、后审报告、CSV、图、拟合和冻结绘图脚本。

## 正确性与历史失败记录

正式输入为 A=B=1/16、D0=0；每个 D 元素的参考值为 `iterations×batch×16/256`，与 chain 数和 warpgroup 数无关。每次启动都重新初始化，并检查全部 CTA、参与组、chain 和片段。另有计时外的 1/2 轮非均匀输入检查，独立主机参考验证 SS/RS 的逻辑坐标、输入片段和输出片段；它证明本有限验证范围，不替代任意输入分布或长 GEMM 的精度研究。

[实际 SASS 审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_wgmma/formal-v3-b/build/sass_audit.json) 与冻结 PTX/源码共同说明目标 HGMMA、提交等待、入口同步及结果排空。静态指令位置不等于硬件动态计数器；本次 NCU 仍复用 `permission_denied`、`family_profile=false` 的权限证据，没有新增家族 profiling，不据此声称物理 HBM 流量或缓存命中状态。

此前 `preflight-v3` 的 FP16、SS、1 warpgroup、2 chains、batch=16、wait0、全 GPU、32768 轮点在 30 个预热窗口耗尽后仍未收敛，`warmup_converged=false`。其 [原 raw](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_wgmma/preflight-v3/preflight/legacy-1ee72da3f364208d048834bc2dd81f911814b755afe630993ce5980c4b13c56a/batch_00/trial_00/attempt_00/raw.jsonl) 保留原样；后续 formal-v3-b 的新正式进程稳定，不会把该旧记录改成稳定样本，也不将旧失败混入本次 850 条正式统计。

[有界浮点后审](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/post_audits/bounded-float-replay-v1-r2/legacy_wgmma-formal-v3-b.json) 通过：2550 项派生浮点比较中保留 48 项白名单差异，允许至多 2 ULP；935 条阈值判定独立重算通过。原严格重放仍记录 `summary differs from raw recomputation`，raw、整数工作量、长度、单位、哈希和状态仍严格检查。该后审处理的是跨环境派生统计的末位差异，不是为设备计算结果放宽容差，也没有改写原 run。完整 C 审查仍待完成。

## 精确复现当前分析

本页使用分析目录中 SHA256 为 `aabf960910190418a4d66223fc240b40dbb760983a368ce38e621fc54e65454f` 的冻结 [report_compute_v2.py](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/report_compute_v2.py)，不是未来可能更新的 live 同名脚本。原图记录 matplotlib 3.10.8；使用相同脚本、输入后审与运行环境，在新目录生成本版 CSV/图/手算/拟合：

```bash
python -B results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/report_compute_v2.py \
  results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_wgmma/formal-v3-b \
  --replay results/gh200_resource_campaign/20261001-resource-suite-v2/post_audits/bounded-float-replay-v1-r2/legacy_wgmma-formal-v3-b.json \
  --output /tmp/wgmma-analysis-reproduced
```

输出路径应是新目录，不覆盖本版分析或原 run；精确图像字节还依赖相同 matplotlib/字体环境，数据与手算可通过 manifest 对照。需要重新检查原归档时，可另行运行受审 `audit_replay.py` 并把新后审报告保存到新路径，不能用新报告覆盖这里绑定的历史报告。

GPU 的启动入口见 [job-730344/driver.sh](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/job-730344/driver.sh)，同 run 恢复入口见 [job-730347/driver.sh](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/job-730347/driver.sh)。本文只读取现有证据并撰写说明，没有重新编译、启动 GPU 或签发 C 资格。
