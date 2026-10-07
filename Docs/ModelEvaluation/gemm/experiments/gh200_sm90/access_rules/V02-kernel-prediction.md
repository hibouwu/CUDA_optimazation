# V02：完整 kernel 的规则预测（先冻结、后测量）

[总计划](README.md)。2026-10-07，Slurm 735696，romeo-a050，GPU-3953fe72…（与 R08 同卡，与 R09 的 GPU-e403ae62 不同卡），驱动 590.48.01，CUDA 12.9.41 / sm_90a，CUTLASS v3.9.2，`-DNDEBUG`。GPU 功率上限 900 W，Module Power Limit 680 W（默认 1000 W）。

## 问题

只用 R08/R09 已测得的规则，能否在测量前预测固定 CUTLASS kernel 在 11 个从未测过的尺寸上的单次调用时间？目标：|e| 中位数 ≤10%，最大 ≤20%，\(e=(\widehat T-T)/T\)。

## 对象与计时

- Kernel：FP16→FP32，tile 128×256×64，cluster 2×1×1，TMA warp-specialized cooperative，4 stage，`ElementC=void`，epilogue 128×32，默认持久调度器（raster Heuristic、swizzle 1），132 SM 各 1 CTA。
- 计时：与 R00/R09 warm 相同。同步预热到末 5 次 CV≤2%，再用 CUDA event 包围一次调用。每尺寸 10 个独立进程，每轮随机打乱尺寸顺序；同一尺寸的无打点、打点两版相邻运行，顺序随机。
- 测量值：无打点版（`v02_plain`）event 时间，取 10 个进程的中位数。
- 打点版（`v02_trace`）：沿用 R09 的覆盖头文件，在 7 处记录 `clock64`/`globaltimer`；另在每个 tile 的 `mma_tail` 之后累加该 CTA 的 tile 数，用来直接核对轮数。调用内频率取关键 CTA（tile 数最多的那些 CTA）的“入口→退出”cycle/ns 中位数。
- 正确性：两版所有 220 个进程都做 R00 抽样 FP64 检查（4096 点，含 tile 边界），误差全部为 0。

## 模型

\[
\begin{aligned}
&t_m=\lceil M/128\rceil,\quad t_n=\lceil N/256\rceil,\quad K_t=\lceil K/64\rceil\\
&U=2\lceil t_m/2\rceil\cdot t_n,\qquad R=\lceil U/132\rceil\\
&C=P+R\,(c_k K_t+c_0+E)\qquad\text{（关键 CTA 的 cycle）}\\
&W=C/f(W),\qquad f(W)=a-b\ln(W/\mu s)\ \text{GHz}\\
&\widehat T=h+s+W+p+\tau
\end{aligned}
\]

| 符号 | 值 | 来源 |
|---|---:|---|
| 132 个驻留 CTA；U 先按 2×1 cluster 把奇数 \(t_m\) 补成偶数，再按轮计费 | — | R08 “配置”“结论”（离散波次，`a+t·⌈U/132⌉`） |
| \(c_k\) 主循环 | 1024.03 cycle/Ktile | R09 `summary.json` `fits.stages_m2048.mainloop_cycles.b` |
| \(c_0\) 每 tile 主循环常数 | 186.3 cycle | 同上 `.a` |
| \(P\) 预填，每 CTA 一次 | 4172.5 cycle | R09 `cases.csv` `prefill_cycles_med`，`trace_m2048_k{256..4096}` warm 的中位数 |
| \(E\) 持久 CTA 每 tile 主循环以外的成本 | 5448 cycle | R09 `samples/idle_m8192_k8192` 原始打点：16-tile 与 15-tile CTA 的 cycle 中位数之差 136711，减去 \(128c_k+c_0\) |
| \(p\) store 后 | 0.292 µs | R09 `fits.stages_m2048.post_store.a` |
| \(s\) 入口偏移 | 0.064 µs | R09 `fits.stages_m2048.entry_skew.a` |
| \(\tau\) 尾部 | 0.459 µs | R09 `fits.stages_m2048.tail_ns.a` |
| \(h\) 主机可见间隙（warm） | 3.857 µs | R09 `fits.stages_m2048.host_gap_ns.a` |
| \(a,\ b\) 调用内频率规则 | 2.0087，0.08496 | R09 `cases.csv` warm 的 7 对（`cta_window_ns_median_med`，`window_ghz_median_med`）：M=N=2048、K=256–8192 与 8192³；最大残差 1.7% |

几点选择及依据：

- **预填只计一次。** 任务描述里写的是“每 tile 都含预填”。R09 8192³ 的原始打点显示，第 16 个 tile 只比第 15 个多 136.7k cycle，即主循环 131.3k 加 5.4k；如果每个 tile 都重新预填，差值应约为 139.6k。推断：持久 CTA 的生产者在上一 tile 的 epilogue 期间已经装好了下一 tile 的前几个 stage。
- **主循环按计算下界计。** R08 没有得出 L2 供给规则；R09 在 8192³ 上测到主循环仍为 1024 cycle/Ktile。因此模型里不设供给项。
- **频率只取决于调用窗口长度 W。** 这是 R09 数据能支持的最简形式；它不含活动 SM 数、遍历方式，也不区分不同的卡。
- 各常数都来自 M=N=2048 或 8192³，没有用测试尺寸拟合。用同一套规则重算 5 个已测的 R09 点，误差为 −1.2%～+2.3%（预测文件中的 `replay_prior_points_not_scored`，不计分）。

## 冻结

- 预测文件 `microbench/gh200_resource_campaign/access_rules/configs/v02-predictions.json`：SHA256 `f37a52558ecc9913dd40efb4e3cea55514a6dc93be70debadda0c3669ca1a7de`，生成于 **2026-10-07T06:01:42Z**，随即改为 `chmod 444`。`v02_predict.py`（SHA256 `54a3a3fd…b457`）同时改为只读。
- 作业 735696 于 06:03:28Z 开始，晚于冻结时间；节点上的副本和结果目录中 `source/` 下的副本哈希一致。测量之后预测器与预测文件都没有改动。

## 测试尺寸与冻结预测

这些尺寸都未出现在 R00/R07/R08/R09/V01 的样本中（逐一核对过 `samples.jsonl` 的 m/n/k）。

| 名称 | M×N×K | tile（补齐后 U） | R | Kt | C（cycle） | f GHz | W µs | \(\widehat T\) µs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| partial_wave | 1536×2560×4096 | 120（120） | 1 | 64 | 75,345 | 1.686 | 44.7 | 49.4 |
| pad_crosses_wave | 1600×2496×4096 | 130（140） | 2 | 64 | 146,518 | 1.626 | 90.1 | 94.8 |
| just_over_wave | 2304×2048×3072 | 144（144） | 2 | 48 | 113,749 | 1.649 | 69.0 | 73.7 |
| long_k | 1024×4096×16384 | 128（128） | 1 | 256 | 271,960 | 1.571 | 173.1 | 177.8 |
| edges_odd_pad | 2600×3496×2000 | 294（308） | 3 | 32 | 119,384 | 1.645 | 72.6 | 77.3 |
| multi_tail_a | 3000×5000×4096 | 480（480） | 4 | 64 | 288,864 | 1.565 | 184.5 | 189.2 |
| multi_tail_b | 3072×6144×3072 | 576（576） | 5 | 48 | 278,114 | 1.569 | 177.3 | 181.9 |
| short_k | 4096×4096×512 | 512（512） | 4 | 8 | 59,480 | 1.707 | 34.8 | 39.5 |
| mid_k1024 | 4096×8192×1024 | 1024（1024） | 8 | 16 | 180,326 | 1.608 | 112.2 | 116.8 |
| large_cube | 6144³ | 1152（1152） | 9 | 96 | 939,650 | 1.459 | 643.9 | 648.6 |
| large_flat | 10240×10240×4096 | 3200（3200） | 25 | 64 | 1,783,493 | 1.401 | 1272.7 | 1277.3 |

pad_crosses_wave 的实际 tile 只有 130 个，但奇数 \(t_m=13\) 补齐后 U=140，预测为 2 轮；用来检验补齐规则。

## 结果

| 名称 | 预测 µs | 实测中位数 µs | CV | e | 轮 预测/实测 | 每 CTA tile 数分布（实测） | cycle 误差 | 频率 预测/实测 GHz | 换用实测频率后的 e |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| partial_wave | 49.4 | 49.2 | 0.8% | +0.4% | 1/1 | 1×120 | −2.9% | 1.686/1.698 | −0.3% |
| pad_crosses_wave | 94.8 | 88.8 | 0.4% | +6.7% | 2/2 | 1×124，2×8 | −0.7% | 1.626/1.707 | +1.9% |
| just_over_wave | 73.7 | 69.8 | 0.5% | +5.5% | 2/2 | 1×120，2×12 | −1.5% | 1.649/1.733 | +0.7% |
| long_k | 177.8 | 194.2 | 0.5% | −8.5% | 1/1 | 1×128 | −0.2% | 1.571/1.449 | −1.0% |
| edges_odd_pad | 77.3 | 75.7 | 0.7% | +2.0% | 3/3 | 2×88，3×44 | −2.6% | 1.645/1.707 | −1.5% |
| multi_tail_a | 189.2 | 207.1 | 1.1% | −8.7% | 4/4 | 3×48，4×84 | −0.8% | 1.565/1.453 | −1.8% |
| multi_tail_b | 181.9 | 193.5 | 1.0% | −6.0% | 5/5 | 4×84，5×48 | −1.4% | 1.569/1.504 | −2.0% |
| short_k | 39.5 | 39.9 | 0.7% | −0.9% | 4/4 | 3×16，4×116 | −6.3% | 1.707/1.729 | −2.0% |
| mid_k1024 | 116.8 | 122.3 | 0.5% | −4.5% | 8/8 | 7×32，8×100 | −1.4% | 1.608/1.560 | −1.6% |
| large_cube | 648.6 | 709.2 | 0.2% | −8.5% | 9/9 | 8×36，9×96 | −0.2% | 1.459/1.344 | −0.7% |
| large_flat | 1277.3 | 1423.9 | 0.3% | −10.3% | 25/25 | 24×100，25×32 | −0.3% | 1.401/1.266 | −0.8% |

**|e| 中位数 6.0%，最大 10.3%（large_flat），达到目标（≤10% / ≤20%）。** CV 为 10 个进程的总体标准差除以均值；各尺寸的实测范围在 `cases.csv` 的 `measured_min/max`。

“cycle 误差”是预测 C 与关键 CTA 实测 cycle 中位数之比减 1。“换用实测频率后的 e”是诊断量，不是预测：把冻结的 C 和各固定项原样保留，只把 \(f\) 换成实测值，看剩余误差有多大。

分项（µs 误差占实测的百分比，正值表示预测偏长）：

| 名称 | 末 tile 主循环 cycle/Ktile | 每 tile 实测额外 cycle | 末 tile epilogue cycle | 主机间隙 µs | 尾部 µs | 打点/无打点 −1 | cycle 项 | 频率项 | 主机项 | 其余 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| partial_wave | 1026 | 7678 | 5700 | 3.36 | 0.6 | +1.2% | −2.7 | +0.7 | +1.0 | +1.4 |
| pad_crosses_wave | 1026 | 5930 | 3289 | 3.44 | 43.3 | +1.7% | −0.6 | +4.8 | +0.5 | +2.1 |
| just_over_wave | 1027 | 6288 | 4160 | 3.65 | 31.7 | +0.9% | −1.4 | +4.8 | +0.3 | +1.8 |
| long_k | 1025 | 6067 | 4667 | 3.87 | 1.3 | −0.3% | −0.2 | −7.5 | −0.0 | −0.7 |
| edges_odd_pad | 1027 | 6491 | 3295 | 3.47 | 23.4 | +1.1% | −2.4 | +3.5 | +0.5 | +0.5 |
| multi_tail_a | 1025 | 6012 | 4162 | 3.98 | 3.6 | −0.5% | −0.7 | −6.9 | −0.1 | −0.9 |
| multi_tail_b | 1025 | 6211 | 4165 | 4.53 | 37.6 | −0.3% | −1.3 | −3.9 | −0.3 | −0.4 |
| short_k | 1033 | 6456 | 4846 | 3.49 | 1.0 | +3.6% | −5.8 | +1.1 | +0.9 | +2.9 |
| mid_k1024 | 1028 | 5765 | 4160 | 3.47 | 2.2 | +0.1% | −1.3 | −2.8 | +0.3 | −0.6 |
| large_cube | 1025 | 5653 | 4162 | 4.00 | 6.1 | −0.2% | −0.2 | −7.8 | −0.0 | −0.5 |
| large_flat | 1025 | 5678 | 4158 | 4.06 | 57.2 | −0.2% | −0.3 | −9.5 | −0.0 | −0.4 |

各列含义：

- 每 tile 实测额外 cycle = (关键 CTA cycle − P)/R − (\(c_kK_t+c_0\))，模型取值为 E=5448。单轮的尺寸里，这一项还包含 store 后的约 550 cycle。
- 主机间隙 = 打点版 event 时间减去包络。
- 尾部 = 最后一个 CTA 退出时间减去所有 CTA 退出时间的中位数；当一部分 CTA 比别的多跑一轮时，它约等于一个 tile 的时间。这与模型中的 \(\tau\)（同一轮内各 CTA 的退出抖动）不是同一个量；模型的包络直接取关键 CTA，所以不受影响。
- “其余”包含打点扰动以及 \(s,p,\tau\) 的偏差。

## 误差归因

1. **结构项全部正确。** 11 个尺寸的轮数、每个 CTA 的 tile 数分布、关键 CTA 个数都与预测完全一致，10 个进程之间也相同。pad_crosses_wave 只有 130 个实际 tile，确实因 cluster 补齐跑了 2 轮，其中 8 个 CTA 跑第二个 tile。末 tile 主循环为 1025–1028 cycle/Ktile（short_k 为 1033，因为 Kt=8，减去 \(c_0\) 时的误差会被放大），在 6144³、10240² 上也保持计算下界。
2. **cycle 项误差小。** 关键 CTA 的 cycle 误差为 −0.2%～−2.9%，只有 short_k 为 −6.3%。short_k 每 tile 只有 8 个 Ktile，主循环以外的成本占 tile 时间约 40%，实测为 6456 cycle，比 E 多 1000。单 tile 尺寸的预填也偏大（4698 / 4902 / 5119 cycle，对比 4172）。
3. **频率是主要误差来源。** 把 \(f\) 换成实测值后，11 个尺寸的误差都落在 −2.0%～+1.9%。误差的方向有两类：
   - 预测偏长（+5～+7%）：pad_crosses_wave、just_over_wave、edges_odd_pad。共同点是最后一轮只有 8/12/44 个 CTA 在跑，实测频率比规则高 3.5～5%。推断：最后一轮活动 SM 少、功耗低，频率回升。R09 的规则全部取自满波调用，不含这一因素。只跑 1 轮、120 个 CTA 的 partial_wave 与规则相差 +0.7%。
   - 预测偏短（−4.5～−10%）：W≥117 µs 的调用，实测频率比规则低 3～10%，W 越长差得越多（1411 µs 时为 1.266 GHz，规则给 1.401）。在窗口长度相近的情况下，long_k（1 个 tile、Kt=256，188 µs）为 1.449 GHz，multi_tail_b（5 轮，187 µs）为 1.504 GHz。推断：频率还取决于主循环所占的时间比例（long_k 中途没有 epilogue 间隙）以及负载历史，不只取决于 W。
4. **卡间差异与访存量（推断，未隔离）。** 规则的长调用端只有 R09 卡上的 8192³ 一个点（1561 µs，1.384 GHz）。本卡在 10240²×4096、1411 µs 时只有 1.266 GHz，差 9%，超过常见的 2～3% 卡间差。用同一张卡上的 R08 数据事后核对：按本模型的 cycle 换算，8192²×4096（约 850 µs）的隐含频率随 swizzle 在 1.29～1.43 GHz 之间变化（swizzle 1 低，swizzle 8 高）。这说明在功率封顶时，频率还与片外流量有关。R08 是另一种计时协议（空闲 0.5 s 后连续 5 次调用），只能作旁证。R08 的问题 1 中 W≤185 µs 的点，隐含频率与规则相差 −4%～+8%（中位 +1.7%），与本组短调用的情况一致。
5. **其他项较小。** 主机间隙实测 3.4～4.5 µs，模型取 3.86 µs，影响 ≤±1%。打点版比无打点版快 0.5% 到慢 3.6%，K 越短、调用越短，扰动越大（与 R09 相同）。

## 结论

- 按冻结规则预测 11 个新尺寸：|e| 中位数 6.0%，最大 10.3%，达到初版目标。V01 中 −23%～−28% 的偏差已经消除。
- 时间的结构可以拆成 **轮数 × 每 tile cycle + 每 CTA 一次的预填 + 固定段**，并能正确外推到 1–25 轮、Kt=8–256、非整除边界和 cluster 补齐；换用实测频率后，各尺寸误差在 ±2% 以内。
- 剩下的误差几乎全部来自调用内频率。只用窗口长度的对数规则，有两种情况描述不了：最后一轮未满时频率回升，以及长时间满载（且片外流量大）时频率下降得更多。要达到 ±5%，频率规则至少还需要考虑活动 SM 数、主循环占比或片外流量，而且要在被预测的那张卡上标定。

## 未解决

- 频率规则的形式：需要在同一张卡上测量频率与 W、活动 SM 数、swizzle（片外流量）之间的关系，并在调用内记录功率或降频原因。本组只能指出这些因素的方向，无法给出系数。
- 卡间差异：R09 卡上没有 W 在 100–1500 µs 之间的调用内频率点，所以 10240² 上 9% 的偏差有多少来自卡、多少来自形状，无法区分。
- short_k 每 tile 多出的约 1000 cycle（每 tile 的切换与 epilogue 在短 K 下没有被重叠），以及单 tile 尺寸预填偏大 500–950 cycle 的原因，都未拆分。
- 默认 swizzle 1；swizzle 8 会同时改变频率（R08），本模型没有覆盖。

## 数据与复现

- 结果目录：[20261007-v02-prediction](../../../../../../results/gh200_resource_campaign/access_rules/20261007-v02-prediction/)，包括 [cases.csv](../../../../../../results/gh200_resource_campaign/access_rules/20261007-v02-prediction/cases.csv)、[summary.json](../../../../../../results/gh200_resource_campaign/access_rules/20261007-v02-prediction/summary.json)、[预测与频率图](../../../../../../results/gh200_resource_campaign/access_rules/20261007-v02-prediction/plots/v02_prediction.png)。另有 `samples/<case>/main-NN/`（原始 stdout 含每 CTA 打点，gzip 压缩）、`samples.jsonl`、`build/`（编译命令、ptxas 日志、SASS、二进制哈希；两版都是 8 条 HGMMA，无 C7510）、`source/`（探针、覆盖头文件、冻结预测副本、分析器）、`environment.json`、`nvidia-smi-{before,after-pilot,after-main}.txt`、`job-735696.out`、`v02.sbatch`。
- `source_hashes.json` 在 build 时生成，那时分析器还不存在；`source/v02_analyze.py` 是之后加入的，后来又添加了诊断列 `pred_with_measured_clock_us`。
- 代码：[v02_predict.py](../../../../../../microbench/gh200_resource_campaign/access_rules/v02_predict.py)、[v02_run.py](../../../../../../microbench/gh200_resource_campaign/access_rules/v02_run.py)、[v02_analyze.py](../../../../../../microbench/gh200_resource_campaign/access_rules/v02_analyze.py)、[probes/v02_probe.cu](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/v02_probe.cu)。探针与 R09 相比只改了一处：最后一次计时调用之前，打点版把记录清零，两版都先做一次设备同步。

```bash
python3 microbench/gh200_resource_campaign/access_rules/v02_predict.py --output <预测.json>
python3 microbench/gh200_resource_campaign/access_rules/v02_run.py build --cutlass-root <CUTLASS> \
  --predictions <预测.json> --output <RUN>
python3 microbench/gh200_resource_campaign/access_rules/v02_run.py sample --output <RUN> --set main
python3 microbench/gh200_resource_campaign/access_rules/v02_analyze.py --input <RUN>
```
