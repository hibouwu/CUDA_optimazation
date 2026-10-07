# R07：NDEBUG锚点、负载频率与每kernel固定成本

[总计划](README.md)。2026-10-07，Slurm 735634，romeo-a048，GPU-009a8880…，CUDA 12.9 / sm_90a，CUTLASS v3.9.2。与R00、V01不是同一块卡。

## 问题

1. R00固定CUTLASS未加 `-DNDEBUG`，ptxas C7510使每条MMA后wait0。去掉后锚点吞吐是多少？
2. GEMM持续运行时SM频率多少，受什么限制？
3. M=N=2048、1波时约9 µs的截距由哪些部分组成？

## 配置

- CUTLASS：与[R00](R00-anchor-target.md)相同（FP16→FP32，128×256×64，cluster 2×1×1，cooperative，4 stage，`ElementC=void`，epilogue 128×32），debug与NDEBUG两版只差 `-DNDEBUG`。NDEBUG：C7510消失，主循环wait1，HGMMA数不变。
- cuBLASLt FP16→FP32，首个合法启发式候选，64 MiB workspace，同一分配内复测。
- 输入、布局、抽样检查同R00；全部进程检查误差为0。

## 计时边界

| 名称 | 窗口 | 包含 |
|---|---|---|
| R00计时 | 同步预热至CV≤2%后，event包围一次调用 | host发射与提交延迟 + 设备时间 |
| graph | 一个graph内200次连续调用，总时间/200 | 设备时间 + kernel间切换，不含host发射 |
| host_call | `gemm.run()` 返回的host墙钟时间 | 参数准备与发射API |
| 频率 | NVML `clocks.sm` 50 ms采样；每批调用**之后**另跑 20 µs 的 clock64/globaltimer 探针 | 探针值是探针 block 的中位数（未记录 SM ID）；它是调用后的频率，不是 GEMM 内部平均频率 |

R00计时与K扫描为每点10个独立进程，graph与发射对照5个，频率循环8 s×2进程。

## 结果

**锚点**（R00计时，中位数）

| 尺寸 | CUTLASS debug | CUTLASS NDEBUG | 提升 | cuBLASLt FP16 |
|---|---:|---:|---:|---:|
| 2048³ | 544 TFLOP/s | 587 | +8.0% | 557 |
| 4096³ | 652 | 688 | +5.5% | 697 |
| 8192³ | 621 | 657 | +5.8% | 695 |

本次的debug版与cuBLASLt比R00低约2%–3%；GPU、节点和运行会话均有变化，差异来源未隔离，比较以同会话比例为准。

**负载频率**

| 情形 | SM频率或调用后探针频率 | 首次进入后半段探针中位数 ±1% 的时间 |
|---|---:|---:|
| 单次2048³调用前、后探针 | 1.98；1.96 GHz | — |
| 单次8192³调用（1.6 ms）后探针 | 1.72–1.76 GHz | — |
| R00计时调用后探针：2048³ / 4096³ / 8192³ | 1.93 / 1.66 / 1.73–1.77 GHz | — |
| 持续CUTLASS 2048³ | NVML：1.74–1.75 GHz，556 W | 约0.1–0.2 s |
| 持续CUTLASS 8192³ | NVML：1.39–1.41 GHz，546 W | 约1 s |
| 持续cuBLASLt 8192³ | NVML：1.45 GHz，548 W | 约0.4 s |

负载期间 98.0%–99.4% 的 NVML 样本只有 SW Power Cap（0x4）；采样未出现 thermal、HW slowdown 或 power brake 标志，最高温度为69 °C。配置快照为 GPU 限值 900 W、**Module Power Limit 680 W**（默认 1000 W），GPU 约 550 W 时出现封顶。证据指向模块功率预算限制；没有负载期间的模块总功耗或改变上限的对照，尚未证明是唯一原因。持续段频率统计（后半段调用后探针，n≈75–79，CV 使用总体标准差）：CUTLASS 2048³ 中位 1.74–1.75 GHz（CV 0.4–0.6%），CUTLASS 8192³ 1.39–1.41（CV 0.7–0.8%），cuBLASLt 8192³ 1.45（CV 0.4–0.5%）。上表的首次进入时间不要求后续持续保持在该范围内；8 s 测量不代表长期热稳态。

**固定成本**（CUTLASS NDEBUG，T=a+b·Ktile，Ktile=K/64，K=64–4096）

| 形状 | a | b |
|---|---:|---:|
| M=N=2048，128 CTA | 9.45 µs | 0.637 µs/Ktile |
| M=N=256，2 CTA | 7.63 µs | 0.571 µs/Ktile |

不同 K 区间的拟合（M=N=2048，单调用计时）：K 64–4096：a=9.452、b=0.6366；K 64–2048：9.533、0.6268；K 256–2048：9.530、0.6268（最大绝对残差 0.064 µs）。斜率随区间略变；另测的长 K 调用后探针频率较低，但 K 扫描计时窗口内的频率未知，不能把变化全部归因于降频。

截距 a 在不同协议、形状之间的差值（M=N=2048，K 64–2048）：

| 差值 | 值 | 含义 |
|---|---:|---|
| 单调用截距 − CUDA graph 截距 | ≈3.3 µs | 两种协议的差；同时改变了主机提交、排队与负载持续方式，不等于纯主机提交成本 |
| 同发射形状空 kernel（graph 内每次） | ≈0.6 µs | 直接测量 |
| M=N=256（2 CTA）graph 截距 − 空 kernel | ≈4.33 µs | 小形状的截距差；空 kernel 未匹配实际寄存器与指令路径，尚未拆成预填、输出、尾部 |
| 大、小形状各自扣除空 kernel 后的截距差 | ≈1.24 µs | (6.2167−0.6426)−(4.8565−0.5267)；输出量和驻留条件同时改变，原因未隔离 |

graph 的拟合也依赖区间：M=N=2048、K 64–4096 时 a=5.713 µs、b=0.6963 µs/Ktile，K 64–2048 时为 6.217、0.6357。6.2 µs 是后一区间的 graph 截距，5.6 µs 是再扣除约 0.64 µs 空 kernel 基线的差值。

手算：2×2048³ = 17179869184 FLOP，除以NDEBUG中位数29.248 µs得587.4 TFLOP/s；拟合9.452+0.6366×32 = 29.82 µs，独立K扫描实测29.60 µs。

## 结论

- 固定CUTLASS的锚点应使用NDEBUG构建：2048³ 587、4096³ 688、8192³ 657 TFLOP/s。以debug版标定的规则或预测需要按此修正。
- 持续负载频率不是常数：持续大 GEMM 为 1.40–1.45 GHz，持续 2048³ 约 1.75 GHz，证据指向 680 W 模块功率预算下的 SW power cap。短调用的调用后探针为 1.93–1.98 GHz，但 R00 计时调用内部的平均频率仍未知。模型不能用单一的 1.83 GHz。
- 单调用经验关系 \(T\approx9.45+0.637\,\text{Ktile}\ \mu s\)（M=N=2048，本构建与协议）可直接用于该配置。斜率已包含组合执行的全部成本，使用它时不能再另加“每 Ktile 约 20% 组合开销”；截距也不能与上表的差值重复叠加。
- ~~按约 1.95 GHz 换算为约 1220 cycle、多 20%~~：[R09](R09-inkernel-clock-stages.md) 在调用内直接测得主循环 1024 cycle/Ktile（加约 185 cycle 常数），调用内频率约 1.64–1.70 GHz；0.63 µs/Ktile 的斜率来自频率，不是组合开销。
- “频率降 18%、时间增 9%”混合了短调用与持续调用两种协议，不能据此判断计算瓶颈占比。
- 未解决：graph 截距扣除空 kernel 后约 5.6 µs 的差值未拆成预填、输出、尾部（需 NDEBUG 构建的 kernel 内时间戳）；负载期间模块总功耗未记录；其他节点的模块功率上限未确认。

## 数据

[初版归档报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-ndebug-clock-fixedcost/report.md) · [修正后的统计与拟合 analysis-r2](../../../../../../results/gh200_resource_campaign/access_rules/20261007-ndebug-clock-fixedcost/analysis-r2/summary.json) · [修正后的 cases.csv](../../../../../../results/gh200_resource_campaign/access_rules/20261007-ndebug-clock-fixedcost/analysis-r2/cases.csv) · [K拟合图](../../../../../../results/gh200_resource_campaign/access_rules/20261007-ndebug-clock-fixedcost/analysis-r2/kfit.png) · [频率图](../../../../../../results/gh200_resource_campaign/access_rules/20261007-ndebug-clock-fixedcost/analysis-r2/clock.png)。原 `cases.csv` 的频率行混用了不同窗口的统计量、CV 固定为 0，已在 r2 修正；当前解释以本文件为准，初版报告保留历史表述。代码：[r07_run.py](../../../../../../microbench/gh200_resource_campaign/access_rules/r07_run.py)、[r07_analyze.py](../../../../../../microbench/gh200_resource_campaign/access_rules/r07_analyze.py)、[probes/r07_probe.cu](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/r07_probe.cu)。

analysis-r2 保存了对应分析器，可用 `python3 <归档目录>/analysis-r2/r07_analyze.py --input <归档目录> --output <新目录>` 离线重放。构建归档记录了 CUTLASS 的两个外部头文件哈希；完整外部编译依赖的来源关系尚未核对，不据此承诺独立重编译。
