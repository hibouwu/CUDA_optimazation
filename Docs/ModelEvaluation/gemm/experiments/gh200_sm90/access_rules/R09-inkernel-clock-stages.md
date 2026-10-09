# R09：固定 CUTLASS 的 kernel 内频率与分段时间

[总计划](README.md)。2026-10-07，Slurm 735669，romeo-a050，GPU-e403ae62…（与 R07 的 GPU-009a8880 不是同一块卡），CUDA 12.9 / sm_90a，CUTLASS v3.9.2，`-DNDEBUG`。Module Power Limit 680 W。

## 问题

1. 一次 GEMM 调用**内部**的平均 SM 频率是多少？[R07](#r07) 只有调用后的探针（1.93–1.98 GHz）。
2. M=N=2048 与 M=N=256（2 CTA）单调用时间怎样分成预填、主循环、epilogue、尾部与主机可见间隙？
3. 各段相加能否重现 [R07](#r07) 单调用拟合 \(T\approx9.45+0.637\,\text{Ktile}\ \mu s\)？

## 配置

- CUTLASS 与 [R07](#r07) 相同（FP16→FP32，128×256×64，cluster 2×1×1，cooperative，4 stage，`ElementC=void`，epilogue 128×32）。M=N=2048 为 128 CTA、1 波；M=N=256 为 2 CTA；8192³ 为 132 个持久 CTA，每个 15–16 个 tile。
- 打点构建：对 kernel 与 mainloop collective 两个头文件做覆盖副本，在 7 处由一个线程读 `clock64` 与 `globaltimer` 并写入全局记录；原头文件不改。无打点构建只差覆盖目录与 `-DR09_TRACE`。
- 计时协议同 R00/R07 K 扫描：同步预热到末 5 次 CV≤2%，再用 event 包围一次调用；读出这次调用的记录。冷调用：加载调用后空闲 1.5 s 再计时一次。
- 每点 10 个独立进程；打点与无打点每轮相邻运行、顺序随机。全部输出 4096 点 CPU FP64 抽检误差为 0。

## 计时边界

| 名称 | 起点 → 终点（同一 CTA，`globaltimer`） |
|---|---|
| 预填 | CTA 入口（线程 0）→ 消费者 warpgroup 0 第一次 full barrier 等待返回、首条 HGMMA 之前 |
| 其中生产者准备 | 入口 → 生产者选中线程第一次 `producer_acquire`/TMA 之前 |
| 主循环 | 首 MMA → `mma_tail` 之后（最终 wait0 与 stage 释放完成） |
| 主循环→epilogue | `mma_tail` 之后 → `collective_epilogue.store` 之前 |
| epilogue | `store` 之前 → 线程 128（发 TMA store 的 warp）`store_tail` 返回，即 `cp.async.bulk.wait_group.read 0`：SMEM 源已读完，不证明全局写完成 |
| store 后 | `store_tail` 返回 → 该 CTA 三个角色线程（0/128/256）最后一个到达 kernel 末尾 |
| 包络 | 所有 CTA 最早入口 → 最晚退出 |
| 尾部 | 最晚 CTA 退出 − CTA 退出中位数 |
| 主机可见间隙 | event 时间 − 包络：发射到首个 CTA 入口、最后 CTA 退出到 event 完成（含全局写完成） |
| 窗口频率 | 同一 CTA 的 `clock64` 差 / `globaltimer` 差 |

`globaltimer` 步长 32 ns（单线程自旋实测）。频率比值只取 ≥3.2 µs 的窗口（100 步，量化误差 ≤1%）。包络和尾部跨 SM 相减 `globaltimer`，假定各 SM 读到同一时钟。分段表取每 CTA 值的中位数，再取 10 个进程的中位数；消费者 0 链各段首尾相接，和等于该 CTA 的入口到退出。8192³ 每个 CTA 跑多个 tile，阶段打点只留最后一个 tile，只用其入口→退出频率。

## 结果

### kernel 内频率（打点构建）

| M=N / K | 调用 | event µs（CV） | CTA 窗口 µs | 窗口频率 GHz（CTA 范围） | 主循环窗口 GHz | 调用后探针 GHz |
|---|---|---:|---:|---:|---:|---:|
| 2048 / 512 | 冷 | 16.1（22%） | 10.8 | 1.81（1.78–1.84） | 1.69 | 1.99 |
| 2048 / 512 | 热 | 15.5（2.3%） | 10.4 | 1.82（1.78–1.85） | 1.71 | 1.99 |
| 2048 / 2048 | 冷 | 30.5（13%） | 25.0 | 1.78（1.75–1.80） | 1.73 | 1.99 |
| 2048 / 2048 | 热 | 29.8（2.1%） | 25.3 | 1.72（1.67–1.75） | 1.67 | 1.96 |
| 2048 / 8192 | 冷 | 91.7（4.5%） | 85.0 | 1.67（1.64–1.71） | 1.67 | 1.99 |
| 2048 / 8192 | 热 | 91.5（0.7%） | 86.1 | 1.64（1.62–1.67） | 1.64 | 1.84 |
| 8192 / 8192 | 冷 | 1574（1.4%） | 1537 | 1.41（1.38–1.46） | 1.42 | 1.99 |
| 8192 / 8192 | 热 | 1598（1.3%） | 1561 | 1.38（1.36–1.44） | 1.40 | 1.85 |

K 扫描（热调用）的窗口频率：M=N=2048 从 K=256 的 1.84 降到 K=4096 的 1.68 GHz（主循环 1.70→1.65）；M=N=256（2 CTA）1.84–1.87 GHz（主循环 1.79–1.83）。每点进程间 CV ≤1.3%。

### 分段（打点构建，热调用，µs 中位数）

M=N=2048，128 CTA：

| K | event | 主机间隙 | 入口偏移 | 预填（生产者准备） | 主循环 | 主循环→epi | epilogue | store 后 | 尾部 | 各段和 | event − 和 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 12.46 | 3.89 | 0.06 | 2.14（1.22） | 2.53 | 0.03 | 3.06 | 0.29 | 0.51 | 12.48 | −0.02 |
| 512 | 14.67 | 3.79 | 0.06 | 2.14（1.22） | 4.93 | 0.02 | 3.01 | 0.29 | 0.51 | 14.69 | −0.02 |
| 1024 | 19.62 | 3.89 | 0.06 | 2.14（1.22） | 9.81 | 0.00 | 2.98 | 0.29 | 0.51 | 19.60 | +0.02 |
| 2048 | 29.81 | 3.90 | 0.06 | 2.21（1.23） | 19.79 | 0.00 | 2.94 | 0.26 | 0.51 | 29.86 | −0.05 |
| 4096 | 50.29 | 3.87 | 0.06 | 2.72（1.47） | 39.84 | 0.01 | 2.88 | 0.26 | 0.77 | 50.34 | −0.06 |

M=N=256，2 CTA：

| K | event | 主机间隙 | 预填（生产者准备） | 主循环 | epilogue | store 后 | 尾部 | 各段和 | event − 和 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 10.45 | 3.66 | 1.85（1.11） | 2.45 | 2.21 | 0.20 | 0.00 | 10.43 | +0.02 |
| 512 | 12.61 | 3.65 | 1.85（1.12） | 4.67 | 2.21 | 0.19 | 0.00 | 12.70 | −0.09 |
| 1024 | 17.02 | 3.65 | 1.80（1.12） | 9.15 | 2.21 | 0.19 | 0.00 | 17.02 | 0.00 |
| 2048 | 26.10 | 3.74 | 1.88（1.12） | 18.10 | 2.21 | 0.19 | 0.00 | 26.09 | +0.01 |
| 4096 | 43.97 | 3.73 | 1.85（1.12） | 36.00 | 2.21 | 0.19 | 0.00 | 43.96 | +0.01 |

M=N=256 的入口偏移与主循环→epilogue 都 ≤0.04 µs，未列。用 cycle 计：主循环两种形状都是 **1024.0 cycle/Ktile + 约 185 cycle**（拟合 1024.03 与 1023.46，K=256–4096）；预填约 4160（2048）/ 3690（256）cycle，K=4096 时 2048 形状升到 4980；epilogue 5470–5850 / 4130 cycle。冷调用的主机间隙为 4.7–5.7 µs，热调用为 3.65–3.9 µs。

### 对照 R07（T = a + b·Ktile）

| 来源 | 构建 | K 范围 | a µs | b µs/Ktile |
|---|---|---|---:|---:|
| 各段和，M=2048 | 打点 | 256–4096 | 9.668 | 0.6341 |
| event，M=2048 | 打点 | 256–4096 | 9.666 | 0.6332 |
| event，M=2048 | 无打点 | 256–4096 | 9.209 | 0.6347 |
| event，M=2048 | 无打点 | 64–4096 | 9.315 | 0.6324 |
| R07，M=2048 | 无打点 | 256–4096 / 64–4096 | 9.389 / 9.452 | 0.6379 / 0.6366 |
| 各段和，M=256 | 打点 | 256–4096 | 8.176 | 0.5590 |
| event，M=256 | 打点 / 无打点 | 256–4096 | 8.149 / 7.723 | 0.5597 / 0.5606 |
| R07，M=256 | 无打点 | 256–4096 | 7.878 | 0.5660 |

M=2048 各段的截距（µs）：主机间隙 3.86，入口偏移 0.06，预填 2.03，主循环 −0.06，epilogue 3.04，store 后 0.29，尾部 0.46；斜率几乎全在主循环（0.6227），预填 +0.010、尾部 +0.004。M=256：3.65、—、1.84、0.20、2.21、0.20、0；主循环斜率 0.5593。各段截距直接相加与“各段和”拟合相差 ≤0.06 µs（逐 CTA 中位数相加与中位数之和的差别）。

### 打点扰动与机器码

| M=N | K=256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| 2048：打点/无打点 − 1（相邻配对中位数） | +3.5% | +3.4% | +2.3% | +1.3% | +0.7% |
| 256 | +3.0% | +4.1% | +1.8% | +1.8% | +0.9% |

扰动近似为常数 0.3–0.5 µs：截距 +0.46 µs（2048）/ +0.43 µs（256），斜率不变（0.6332 对 0.6347；0.5597 对 0.5606）。K≤1024 时超过 2%。消融（K=256 两种形状与 2048×K2048，配对中位数）：完整打点 +0.50–0.54 µs，只留入口/退出 −0.03–+0.40，只留中间 5 处 +0.18–+0.37；两者不可加，无打点进程间 CV 1.6–2.9%（0.2–0.3 µs），未能定位到具体打点。

SASS：两版整个 kernel 都是 8 条 HGMMA、wait1 与 wait0 各一次；最内层主循环都是 48 条指令、4 HGMMA、1 次 wait1、3 条 SYNCS，循环内无计时读，操作码多重集合相同；顺序有两处差别：最后一条 HGMMA 提前两条指令，一对 `VIADD`/`IADD3` 互换。寄存器均为 168、无 spill。同会话 R00 驱动对照：2048×K512 14.45 µs、K2048 29.54 µs，与 `r09_plain` 相差 +1.2% / +0.7%，在 CV 内。

## 结论

- **调用内频率明显低于调用后探针。** M=N=2048 单调用窗口 1.68–1.84 GHz、主循环 1.65–1.71 GHz；8192³ 即便是空闲 1.5 s 后的第一次调用，平均也只有 1.41 GHz（热 1.38）；同一次调用后的探针读 1.84–1.99 GHz。调用后探针不能代表调用内频率。
- **主循环在 cycle 上已达计算下界。** 两种形状都是 1024.0 cycle/Ktile（2 个 warpgroup 各 4 条 `m64n256k16`，理想 1024），另加约 185 cycle。R07 的斜率 0.637 µs/Ktile 对应的是频率而不是组合开销：1024 cycle / 0.6227 µs ≈ 1.64 GHz。R07 按 1.95 GHz 换算出的“约 1220 cycle、多 20%”应改为“1024 cycle、频率约 1.64–1.70 GHz”。
- **截距的组成（M=N=2048，无打点 9.21 µs，打点 9.67 µs）**：主机可见间隙约 3.9 µs、epilogue 约 3.0 µs、预填约 2.0 µs（其中生产者准备 1.2 µs、首个 stage 到达 0.9 µs）、尾部约 0.46 µs、store 后约 0.29 µs；打点带来的约 0.46 µs 落在这些项中，无法分配到具体一项。
- **重现 R07。** 各段和与打点 event 逐点相差 ≤0.06 µs（2048）/ ≤0.09 µs（256）；拟合截距相差 +0.002 µs、斜率 +0.0009 µs/Ktile。扣除打点常数后（9.67 − 0.46 = 9.21 µs），与 R07 同区间（K 256–4096）a=9.39 的残差为 −0.18 µs，斜率 0.6341 对 0.6379 为 −0.6%；与 R07 全区间 9.45 / 0.637 的残差为 −0.24 µs / −0.5%。两次不是同一块 GPU。
- **形状差（2048 − 256）** 的截距约 1.5 µs 主要来自 epilogue +0.83 µs 与尾部 +0.46 µs，其次主机间隙 +0.21、预填 +0.19、store 后 +0.10；主循环截距 −0.27 µs 是 2048 形状的频率随 K 下降造成的时间拟合效应（cycle 截距两者都约 185）。斜率差（0.6227 对 0.5593）完全来自频率（1.64 对 1.83 GHz）。
- 推断：2 个 CTA 时模块功率远低于上限，主循环频率仍只有 1.79–1.83 GHz，说明调用内降频不只来自 680 W 模块功率封顶；可能还有与 Tensor Core 活动相关的快速频率调节。本次没有微秒级功率或降频原因记录，无法区分。
- 推断：2048 形状的 epilogue 每 CTA 输出量与 256 形状相同（128 KiB），多出的 0.8 µs 与 128 个 CTA 同时写 16 MiB 共享写回路径一致。

## 未解决

- 打点扰动 0.3–0.5 µs 未定位；各段截距可能各自含其中一部分。
- 降频机制（功率预算还是 Tensor Core 相关的频率策略）未隔离。
- 2048 形状 K=4096 的预填升到 2.72 µs（4980 cycle，生产者准备也升到 1.47 µs），原因未查。
- 消费者 warpgroup 1 的主循环比 0 早结束 0.1–0.37 µs，却最后退出；未分析。
- 包络与尾部依赖各 SM `globaltimer` 一致的假定，未独立验证。
- 冷调用 event CV 13–22%，10 个样本只够给中位数。

## 数据

[报告与手算](../../../../../../results/gh200_resource_campaign/access_rules/20261007-r09-clock-stages/report.md) · [summary.json](../../../../../../results/gh200_resource_campaign/access_rules/20261007-r09-clock-stages/summary.json) · [cases.csv](../../../../../../results/gh200_resource_campaign/access_rules/20261007-r09-clock-stages/cases.csv) · [分段图](../../../../../../results/gh200_resource_campaign/access_rules/20261007-r09-clock-stages/plots/stages.png) · [频率图](../../../../../../results/gh200_resource_campaign/access_rules/20261007-r09-clock-stages/plots/clock.png) · SASS 与编译命令在 `build/`，覆盖头文件在 `source/overlay/`，消融在 `ablation/`。代码：[r09_run.py](../../../../../../microbench/gh200_resource_campaign/access_rules/r09_run.py)、[r09_analyze.py](../../../../../../microbench/gh200_resource_campaign/access_rules/r09_analyze.py)、[probes/r09_probe.cu](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/r09_probe.cu)。

```bash
python3 microbench/gh200_resource_campaign/access_rules/r09_run.py build --cutlass-root <CUTLASS> --output <RUN>
python3 microbench/gh200_resource_campaign/access_rules/r09_run.py sample --output <RUN> --set stages   # 另有 pilot、clock、timer、ablation
python3 microbench/gh200_resource_campaign/access_rules/r09_analyze.py --input <RUN>
```

<a id="r07"></a>

## 前序：调用后频率、持续负载与单调用截距（原 R07，2026-10-07）

2026-10-08 由 R07 迁入；R07 的 NDEBUG 锚点在 [R00](R00-anchor-target.md#ndebug)。Slurm 735634，romeo-a048，GPU-009a8880…，CUDA 12.9 / sm_90a，CUTLASS v3.9.2，与本页上文不是同一块卡。CUTLASS 与上文相同（FP16→FP32，128×256×64，cluster 2×1×1，cooperative，4 stage，`ElementC=void`，epilogue 128×32），NDEBUG 构建；cuBLASLt FP16→FP32 取首个合法启发式候选，64 MiB workspace。全部进程检查误差为 0。

原问题：GEMM 持续运行时 SM 频率多少、受什么限制；M=N=2048、1 波时约 9 µs 的截距由哪些部分组成。上文的调用内频率与分段即为对这两个问题的修正。

| 名称 | 窗口 | 包含 |
|---|---|---|
| R00计时 | 同步预热至CV≤2%后，event包围一次调用 | host发射与提交延迟 + 设备时间 |
| graph | 一个graph内200次连续调用，总时间/200 | 设备时间 + kernel间切换，不含host发射 |
| host_call | `gemm.run()` 返回的host墙钟时间 | 参数准备与发射API |
| 频率 | NVML `clocks.sm` 50 ms采样；每批调用**之后**另跑 20 µs 的 clock64/globaltimer 探针 | 探针值是探针 block 的中位数（未记录 SM ID）；它是调用后的频率，不是 GEMM 内部平均频率 |

R00计时与K扫描为每点10个独立进程，graph与发射对照5个，频率循环8 s×2进程。

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

不同 K 区间的拟合（M=N=2048，单调用计时）：K 64–4096：a=9.452、b=0.6366；K 64–2048：9.533、0.6268；K 256–2048：9.530、0.6268（最大绝对残差 0.064 µs）。斜率随区间略变；另测的长 K 调用后探针频率较低，但 K 扫描计时窗口内的频率未知，不能把变化全部归因于降频。拟合 9.452+0.6366×32 = 29.82 µs，独立 K 扫描实测 29.60 µs。

截距 a 在不同协议、形状之间的差值（M=N=2048，K 64–2048）：

| 差值 | 值 | 含义 |
|---|---:|---|
| 单调用截距 − CUDA graph 截距 | ≈3.3 µs | 两种协议的差；同时改变了主机提交、排队与负载持续方式，不等于纯主机提交成本 |
| 同发射形状空 kernel（graph 内每次） | ≈0.6 µs | 直接测量 |
| M=N=256（2 CTA）graph 截距 − 空 kernel | ≈4.33 µs | 小形状的截距差；空 kernel 未匹配实际寄存器与指令路径，尚未拆成预填、输出、尾部 |
| 大、小形状各自扣除空 kernel 后的截距差 | ≈1.24 µs | (6.2167−0.6426)−(4.8565−0.5267)；输出量和驻留条件同时改变，原因未隔离 |

graph 的拟合也依赖区间：M=N=2048、K 64–4096 时 a=5.713 µs、b=0.6963 µs/Ktile，K 64–2048 时为 6.217、0.6357。6.2 µs 是后一区间的 graph 截距，5.6 µs 是再扣除约 0.64 µs 空 kernel 基线的差值。

**原结论及其修订**

- 持续负载频率不是常数：持续大 GEMM 为 1.40–1.45 GHz，持续 2048³ 约 1.75 GHz，证据指向 680 W 模块功率预算下的 SW power cap。模型不能用单一的 1.83 GHz。
- 单调用经验关系 \(T\approx9.45+0.637\,\text{Ktile}\ \mu s\)（M=N=2048，本构建与协议）可直接用于该配置。斜率已包含组合执行的全部成本，不能再另加“每 Ktile 约 20% 组合开销”；截距也不能与上表的差值重复叠加。
- ~~按约 1.95 GHz 换算为约 1220 cycle、多 20%~~：上文在调用内直接测得主循环 1024 cycle/Ktile（加约 185 cycle 常数），调用内频率约 1.64–1.70 GHz；0.63 µs/Ktile 的斜率来自频率，不是组合开销。
- “频率降 18%、时间增 9%”混合了短调用与持续调用两种协议，不能据此判断计算瓶颈占比。
- 当时未解决：graph 截距扣除空 kernel 后约 5.6 µs 的差值未拆开（上文分段已部分回答）；负载期间模块总功耗未记录；其他节点的模块功率上限未确认。

[初版归档报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-ndebug-clock-fixedcost/report.md) · [修正后的统计与拟合 analysis-r2](../../../../../../results/gh200_resource_campaign/access_rules/20261007-ndebug-clock-fixedcost/analysis-r2/summary.json) · [修正后的 cases.csv](../../../../../../results/gh200_resource_campaign/access_rules/20261007-ndebug-clock-fixedcost/analysis-r2/cases.csv) · [K拟合图](../../../../../../results/gh200_resource_campaign/access_rules/20261007-ndebug-clock-fixedcost/analysis-r2/kfit.png) · [频率图](../../../../../../results/gh200_resource_campaign/access_rules/20261007-ndebug-clock-fixedcost/analysis-r2/clock.png)。原 `cases.csv` 的频率行混用了不同窗口的统计量、CV 固定为 0，已在 r2 修正；当前解释以本节为准，初版报告保留历史表述。代码：[r07_run.py](../../../../../../microbench/gh200_resource_campaign/access_rules/r07_run.py)、[r07_analyze.py](../../../../../../microbench/gh200_resource_campaign/access_rules/r07_analyze.py)、[probes/r07_probe.cu](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/r07_probe.cu)。

analysis-r2 保存了对应分析器，可用 `python3 <归档目录>/analysis-r2/r07_analyze.py --input <归档目录> --output <新目录>` 离线重放。构建归档记录了 CUTLASS 的两个外部头文件哈希；完整外部编译依赖的来源关系尚未核对，不据此承诺独立重编译。

<a id="other-clock-evidence"></a>

## 其他实验中的频率与功率证据

频率机制在本页维护；下列验证页保留各自的冻结规则和判定，这里只汇总机制证据。

- [V03](V03-clock-rule.md#功率nvml诊断)：job735778 在 GPU-572de9c0 上做持续负载诊断。132 个 SM 满载时模块功率最后 1 s 中位数 648–671 W，降频原因只有 SW Power Cap。32 个 SM 时模块功率 472 W、未封顶，NVML 报 1980 MHz，调用内 cycle/ns 仍只有约 1.82 GHz。这是未解释的观测平台，不是已识别的硬件时钟上限。
- V03 同时记录：NVML 报告的 SM 时钟比 clock64/globaltimer 换算值高 0.1–0.2 GHz，原因未查；本页称为“主机可见间隙”的 event 与内核包络之差，在不同卡/节点上为 3.9–6 µs；同一频率规则在另一块卡的满载长调用上低 2.7%–6.5%。
- [V08](V08-wider-validation.md#失败原因测后诊断不改判定)：窗口约 650 µs 的 h06 超过校准最长的约 410 µs，三个配置的频率解低 6%–7%。V08 的配置快照中模组功率限制为 680 W（同一文件中的 900 W 是 GPU 层字段）；这说明配置相同，不证明每次调用都触发了封顶。
- 所有 GEMM 探针输入为同一个 17 级二进分数正确性见证（`probes/r00_common.hpp` 的 `input_value`）。功率与频率结论只适用于这种输入，随机数据下的情况尚未测量。

主机可见间隙按上文定义包含发射到首个 CTA 入口、最后 CTA 退出到 event 完成，以及其间可能尚未完成的全局写入，不能全部归因于主机 launch 开销。

<a id="empty-window"></a>

## 计时空窗口（原 EXP-04 一部分）

2026-10-08 由 EXP-04 迁入；EXP-04 的 SMEM 与 global 部分分别在 [EXP-09](../EXP-09-smem.md#exp-04) 与 [EXP-13](../EXP-13-global-rw.md#exp-04)。资源套件 `20261001-resource-suite-v2`，原始 run `memory_baseline/formal-v3-a`。

| 窗口 | 单位 | 中位数 | CV |
|---|---|---:|---:|
| 单 CTA 空计时窗口 | clock64 cycle | 34 | 0 |
| 整卡空窗口 | globaltimer ns | 128 | 12.7% |

单 CTA 的 34 cycle 可作为 clock64 计时固定开销的参考；整卡空窗口跨进程不稳定。它们是该探针的空窗口，不是本页 CUTLASS 打点的扰动。数据：[samples.csv](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/memory-baseline-formal-v3-a/samples.csv)。

<a id="v08-clock-reanalysis"></a>

## 2026-10-09：V03/V08 时间换算离线复核

本次只读取历史归档，没有 GPU 测量。V08 的 36 个留出点、V03 的 11 个留出点均从原始 trace 复算并与原汇总一致；冻结预测、原评分和判定不变。V08 结果在 [C-20261009-clock-diagnostic-v2](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/C-20261009-clock-diagnostic-v2/diagnostic.json)，V03 在 [C-20261009-v03-residual-v1](../../../../../../results/gh200_resource_campaign/access_rules/20261007-v03-clock-rule/reanalysis/C-20261009-v03-residual-v1/diagnostic.json)。V08 的 v1 为首次计算，v2 补充逐调用 gap，原数值分解未改。

### h06：替换实测频率后还剩什么

保持冻结的周期 C、比例 κ 和固定项 F，仅把 `T=F+κC/(1000f)` 中的 f 换为 ends 的调用内实测 GHz。h06 均为 3584×3584×20480、swizzle=1、原 dyadic 输入。

| 配置 | 实测 ends 包络 µs | f 冻结 / 实测 GHz | 原完整时间误差 | 换实测 f 后 µs / 误差 |
|---|---:|---:|---:|---:|
| cfg_a | 640.336 | 1.485835 / 1.584461 | +5.271% | 637.455 / −1.249% |
| cfg_b | 781.088 | 1.480212 / 1.587056 | −13.570% | 627.598 / −19.358% |
| cfg_c | 624.928 | 1.504003 / 1.620511 | +5.247% | 614.199 / −2.277% |

cfg_a/c 的偏长主要来自频率偏低；cfg_b 的周期低估更大，频率偏低反而掩盖了一部分。对全部 36 点只做同样替换，绝对误差中位数 / 最大值为 1.526% / 37.951%；这是诊断，不能作为新验证成绩。

下面按中位数做代数分解。令 Ĉ=κC，Cₛ、Cₑ 为 stamped/ends 的实测最大 CTA 周期，fₑ 为 ends 频率，Wₑ 为 ends 包络，Tₑ、Tₚ 为 ends/plain event 时间。各项单位为 µs：

| 项 | 计算 | cfg_a | cfg_b | cfg_c |
|---|---|---:|---:|---:|
| 频率替换量 | Ĉ/(1000f)−Ĉ/(1000fₑ) | +42.087 | +45.046 | +47.293 |
| 周期模型差 | κ(C−Cₛ)/(1000fₑ) | −0.641 | −139.661 | −5.036 |
| κ 的迁移差 | (κCₛ−Cₑ)/(1000fₑ) | −4.218 | −18.220 | −6.400 |
| 周期/频率代理与包络差 | Cₑ/(1000fₑ)−Wₑ | −1.414 | +0.862 | −2.989 |
| 固定项差 | F−(Tₑ−Wₑ) | −0.448 | −0.776 | −1.072 |
| ends/plain 进程差 | Tₑ−Tₚ | −1.344 | +7.136 | +1.184 |
| 总差 | 冻结预测−plain | +34.023 | −105.612 | +32.980 |

36 点最大闭合差为 8.53×10⁻¹⁴ µs。这是**分别取中位数后的恒等式，不是同次调用的因果分解**。κCₛ−Cₑ 同时包含打点、编译与不同进程状态差；不能只命名为打点成本。冻结 κ 为 0.991263/0.993114/0.988041，h06 实测周期比为 0.997850/1.016806/0.998314，短窗口的比例不能无条件延伸到长窗口。

fₑ 是每进程中最多 tile 的 CTA 的 cycle/ns 比值中位数，再跨进程取中位数；Cₑ 则先取每进程最大 CTA 周期。`Cₑ/fₑ` 与跨 SM 的 `max(final_ns)−min(entry_ns)` 并非同一统计量，二者之差不能全解释为入口错开。

### 长窗口误差不只是 log 项外推

冻结规则为 `f=a−b·φ·ln(W/µs)−c·D−d·μ`，V08 只重拟合 a、c，b、d 沿用 V03。V08 时间校准的最长 ends 包络分别是 **409.728、401.504、398.176 µs**。h06 在当前卡与 cfg 上超出该范围；V03 虽有毫秒级调用，但使用另一张卡和旧 128×256 配置，不能替代这一缺口。

保持冻结 φ、μ 和模型隐含流量，直接把 W 换成实测包络，得到 1.485008/1.481708/1.504257 GHz，仍低 6.277%/6.638%/7.174%。因此，先修 cfg_b 的周期、进而得到更准确的 W，并不会自动修好频率规则。

进一步只把 `ln W` 限制在各配置校准最长窗口，频率仅提高 **15.62/23.28/15.77 MHz**，仍低 **5.291%/5.171%/6.201%**。它只是敏感性计算，不是新规则；超过校准时长的 log 惩罚不能单独解释全部偏差。

这里的 D 是 LRU 模型估计字节/W，不是计数器实测流量。固定该字节数、φ、μ 后，公式的 f(W) 在 **943.75/921.85/550.15 µs** 达到极大值，此后才下降；h06 的 a/b 还在极大值之前。长期 log 形式没有稳态极限，既不能概括为“时间越长必然越低”，也不能从这条公式切片辨认功率控制机制。仍缺同卡公共配置的长窗口数据、前序暖机配对，以及调用内分段 cycle/ns；现有 trace 的逐 tile 记录只有 cycle。

### 固定项、V03 余差与最终完成

V08 的 F 实际拟合的是 `median_over_cases(median(plain event)−median(ends envelope))`。校准中各配置的 F 为 3.392/3.528/3.696 µs，而 `median_over_cases(median(ends event)−median(ends envelope))` 为 4.048/4.232/3.984 µs；F 已包含跨协议差，不是纯 launch 常数。

h06 同次 ends 调用的 `median(event−envelope)` 为 **4.224/4.032/4.688 µs**；上表使用的两个中位数之差则为 3.840/4.304/4.768 µs。cfg_b 的 `median(plain)−median(ends envelope)=−2.832 µs`，进一步说明不能把跨进程差当作物理上必为正的间隙。

V03 的 11 个留出点换实测频率后，有符号误差中位数 −1.252%，范围 −4.431%～+0.310%；逐调用 event−包络的中位数为 5.296–6.336 µs，比旧 R09 的 3.857 µs 高 1.439–2.479 µs。但 `h_huge_sw1` 实测 1365.728 µs，替换频率后仍少 **60.513 µs**：周期模型差按实测频率折合 −39.370 µs，traced/plain event 中位数相差 −7.712 µs，关键 CTA 周期/频率代理与包络还差 −11.798 µs。固定 gap 的增量只有 2.479 µs，不能解释全部余差。V03 的关键周期是最多 tile CTA 集合的**中位数**，又不同于 V08 的 Cmax。

R09 的尾端取三个角色最后的记录；公共 cooperative final 取消费者中发 TMA 的线程 256 在 `store_tail` 返回后的记录，pingpong 取最后消费者。该 store 等待是 `.read`，其边界为源 SMEM 可复用，不能直接改称目标全局内存写入完成。完整 event 与该包络之差仍可能包含末端后的异步写入；若继续物理解剖 F，才需要匹配的目标写完成端点，当前先保留经验差值。

V03 的三个未解问题仍未被离线数据解决：G=32 的持续诊断调用内频率 1.824308 GHz，而 NVML 为 1980 MHz，模块功率末 1 s 中位数 472.081 W、该段 SW Power Cap 占比 0；不能据此命名一个与功率无关的硬件上限。NVML 是持续段慢采样，cycle/ns 是最终调用窗口，二者相差 155.692 MHz 尚无同窗口对照。不同卡/节点的 gap 差也尚未归因。该记录整进程的 reasons 曾出现 SW Power Cap，应限定为“最后 1 s 未观测到封顶”。

<a id="s-prefill-alignment"></a>

### S 与预填只能在相同事件上对齐

公共 [v06_fit.py](../../../../../../microbench/gh200_resource_campaign/access_rules/v06_fit.py) 定义 `P0=producer_first_work−entry`，`S=first_MMA(tile0)−producer_first_work`。R09 定义 `prefill=first_mma0−entry`，其 `producer_setup` 终点已在 collective load 内的首次 producer acquire 前，晚于公共 producer-first-work。因此只有同 CTA、同首 tile、同首 MMA 端点时，预填才等于 P0+S；R09 固定取消费者 0，公共 cooperative 取两消费者较早的首 MMA，现有汇总常数不能直接相加。R09 的多 tile 阶段槽还会被最后 tile 覆盖。

V08 归档 cooperative overlay 的显式 descriptor prefetch 位于 entry 后、producer-first-work 前，其发射属于 P0；未完成的效果可以延续进 S，但旧端点分不开。不能事后把描述符准备的全部成本记入 S。

既有 2×L2 写驱逐配对给出以下 cycle 差；每格为 **ΔP0 / ΔS**：

| 配对 | cfg_a | cfg_b | cfg_c |
|---|---:|---:|---:|
| g1 | +353.75 / +897.25 | +410.75 / +1278.00 | +337.75 / +1220.25 |
| g4 | +28.50 / +721.00 | +5.50 / +595.25 | +17.50 / +9.25 |
| c4_k2048 | +43.00 / +803.50 | +62.00 / +454.25 | +34.00 / +8.25 |

这支持首段等待与前序缓存准备有关，也显示变化不全在 S 内；它不单独识别描述符、数据缓存或指令前端。最小新增 cycle 端点是首 tile 首次 producer acquire/TMA 前，与已有 producer-first-work、首 MMA 分成两个区间。先复用已有驱逐配对；只有要区分剩余候选时，再分别配对 descriptor prefetch 与暖机历史，避免加一个混合“冷启动常数”。

<a id="input-modes"></a>

### 公共输入开关的数值定义

管理者实现的接口为 `--input-mode dyadic|zero|random`、`--seed`、`--sm-count`。默认仍为 `dyadic`、`seed=17`；三档只改变逻辑 A/B 元素，padding 与 D 的哨兵、alpha=1/beta=0 不变。下面采用已只读核对的管理者 uint32 哈希实现，公共补丁及其最终源码 SHA 由管理者发布；本次没有编译或运行 GPU kernel。

| 模式 | 分布 | 检查 |
|---|---|---|
| dyadic | 原 A=`((7r+13c+3seed)%17−8)/32`，B=`((5r+11c+5seed)%17−8)/32`，17 级值 | 保留原 17 周期参考与零误差判据 |
| zero | 所有逻辑 A/B 元素为 FP16 +0 | 被检查 D 数值精确为零，非有限值与 padding 错误均为 0 |
| random | 下面的逻辑坐标哈希产生 24 位离散均匀 `[-1,1)`，RN 量化到 FP16；量化后范围 `[-1,1]`，包含少量 subnormal | 计时外 CPU double 点积参考，使用下述逐点容差 |

```cpp
uint32_t x = uint32_t(seed) ^ (is_a ? 0xa511e9b3u : 0x63d83595u);
x ^= uint32_t(row) * 0x9e3779b9u;
x ^= uint32_t(column) * 0x85ebca6bu;
x ^= x >> 16;
x *= 0x7feb352du;
x ^= x >> 15;
x *= 0x846ca68bu;
x ^= x >> 16;
float value = float(x >> 8) * 0x1p-23f - 1.0f;
__half stored = __float2half_rn(value);
```

整数运算按 uint32 模 2³²，seed 首轮固定为 17；逻辑坐标不依赖 pitch、grid 或线程遍历次序。可核对的 FP16 位模式：坐标 (0,0)/(0,1)/(1,0)/(127,255)，A 为 `ba2a/a8c9/b3f3/3be4`，B 为 `38b8/3812/3974/bb92`。位模式已用 CPU 独立整数计算及 FP16 RN 转换复核；这里只固定一份伪随机样本，不声称代表所有随机分布。

随机参考从计时外读回的**实际存储 FP16 A/B**转为 double，计算 `ref=ΣA·B`、`s=Σ|A·B|`；不要调用同一个 GPU hash 重生参考。最初提出的工程判据为每个抽检点 `|D−ref|≤2^-20+2^-21·s`，且非有限值和 padding 错误均为 0；以绝对乘积和处理相消，不单靠相对误差。**下文公共 GPU 检查发现 K=65536 不满足，撤回最初 K≤65536 的适用范围建议；随后 job738110 的 K=20480 random 预检也失败。** 本次保留阈值和失败，不为通过而放宽。继续报告原绝对误差、非有限值、padding 错误，并记录最大 error/tolerance。沿用现有 4096 点抽检；小矩阵可全检，抽检不能写成全矩阵正确性证明。输入生成还须核对上述位模式；用实际输入作 GEMM 参考本身不验证生成器。

共同基点的 `r18.cu` 以 `Options check(1, argv)` 固定 seed=17；返回码及 `gaps_common.hpp::print_check` 均使用零误差判据，公共补丁须让实际 seed 与两处模式判断一致。初始化、输入读回和检查均在目标计时外。原 dyadic 的精确性来自小整数格点：每个乘积为整数/1024、整数绝对值≤64；K≤262144 时绝对部分和的整数界≤2²⁴，不能将这种性质推广到任意 FP16 输入。本组没有修改公共头或 kernel。

### 真正缺少的最小配对

旧 R09 的 **128×256×64、cluster 2×1、cooperative、4 stage** 常数不能移入公共配置。cfg_a 是 128×128×64、2×1、cooperative、6 stage；cfg_b 是 128×128×64、1×1、pingpong、6 stage；cfg_c 是 256×128×64、1×2、cooperative、4 stage。计算、供给、输出、stage、最终记录角色与 GPU 均有差别；即使 cfg_c 的名义 1024 cycle/Ktile 相同，也不使其预填和固定项相同。

公共框架的首批需求如下，均为诊断/校准，不能充当 V09 新留出：

- **时间换算：** 三配置各用 `M=N=3584, K∈{1024,8192,20480}`，swizzle=1、紧密对齐行距、dyadic/seed17、原重复暖机；共 9 个几何条件，plain/stamped/ends 成对。K=20480 复用 h06 几何，新卡/新协议下以实测 ends 包络>600 µs 为准；不足时才延长 K。短/中/长分组按实测窗口，不按 K 名称。三点足以检查本次残差能否重现，不足以唯一拟合时长、流量和功率机制。
- **输入：** 只在上述每配置的长窗口补 zero/random，增加 6 个条件，不与全部短中窗口交叉。三档固定布局和暖机，先比周期、调用内频率和 event；功率另用同一持续负载段的 NVML，记录模组与 GPU 字段、时间戳和封顶占比，不用 680 W 配置值证明调用已封顶。
- **前序历史：** 每配置只在长窗口 dyadic 增加一次“完成原暖机后同步空闲 1.5 s，再测首调用”的配对，增加 3 个条件；不把冷调用混入原暖机模型。与重复暖机条件交错，保留原始样本。
- **少量 SM 与 NVML 差：** 先比较 32 个实际活动 SM 的公共 GEMM（WGMMA+TMA）、FFMA、仅 TMA 三种负载，每种各持续约 2 s、分别统计本段末 1 s。NVML 5 ms 轮询与多次同 SM cycle/ns 在各自同一持续段内重叠采样。GEMM 用公共 cfg 的 sm_count 开关；另两种复用原指令/供给探针。如需纯 WGMMA，另用寄存器/SMEM 常驻原探针，不能把公共 GEMM 当成纯计算负载。负载组成、实际 SMID 与窗口同时保留，频率差并不能单凭这组三点唯一归因。
- **S 与最终完成：** 首段只需上文一个 acquire/TMA 前端点，并与既有驱逐配对共用。若只预测 event，可继续实测条件 F；只有继续分离写回与外部间隙时才增加目标写完成对照，不把它设为所有测量的前置条件。

每条件先沿用 10 个独立进程；plain/ends 的事件差、每 CTA cycle/ns 和必要 stamped 区间同时保存，波动不足以区分时才扩大重复。不同输入的连续功率段与正式短调用计时分开；所有新时间常数来自公共 cfg_a/b/c 的同卡配对。

复核入口（`--output` 必须指定不存在的新目录；不写旧 summary 或 frozen）：

```bash
ROOT=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
python3 microbench/gh200_resource_campaign/access_rules/r09_v08_clock.py \
  --run "$ROOT/20261008-V08-job737322-v1" \
  --output "$ROOT/20261008-V08-job737322-v1/reanalysis/C-20261009-clock-replay-<新后缀>"
python3 microbench/gh200_resource_campaign/access_rules/r09_v03_residual.py \
  --run "$ROOT/20261007-v03-clock-rule" \
  --output "$ROOT/20261007-v03-clock-rule/reanalysis/C-20261009-v03-replay-<新后缀>"
```

### 首批 15 条公共条件与长 K 数值失败

2026-10-09，按管理者确认的首批范围，只准备下面 15 条；本批不加入空闲历史、持续功率扫描或新打点。`r09_run.py shared-list` 输出完整条件，均为 `ctrl`，M=N=3584、swizzle=1、evict=0、seed=17、sm_count=0（公共默认全部 SM），行距分别为 K/N/N。

| 每配置的 K | input_mode | 三配置合计 |
|---:|---|---:|
| 1024 | dyadic | 3 |
| 8192 | dyadic | 3 |
| 20480 | dyadic、zero、random | 9 |

公共运行源码固定为 **6338653b5a4127a7cb947b6355c38642310f3dcc**。复用管理者在 job738097、romeo-a057、GPU-43269fbc-449d-3e0f-908a-9c81229546d3 构建的 [source/build 包](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-R13-shared-smoke-job738097/)，不重编、不复制 samples，也不以当前 checkout 的无关公共改动否定旧包。新 run 的 `run_config.json` 分别记录公共源码提交、来源源清单指纹，以及本次 R09 入口/分析器文件 SHA256；`build/origin.json` 记录来源二进制清单指纹。R09 只增加专用入口与分析，不复制随机生成或正确性实现。

**公共包的数值范围有具体失败。** 39 个进程中，M=2304、N=3072、K=1024 的三配置、三输入及 SM32 检查共 36 个通过；另三个 64×64×65536、seed=17、random plain 均为 numeric_error。用 [r09_input_error.py](../../../../../../microbench/gh200_resource_campaign/access_rules/r09_input_error.py) 导入公共 `analyze_r18.random_references`，从原始 4096 个输出复算：

| 指标（每进程 4096 点） | K=1024，seed=20261009 | K=65536，seed=17 |
|---|---:|---:|
| 随机输入进程数 | 9，输出逐点相同 | 3，输出逐点相同 |
| 超出原容差 | 0 | **958（23.3887%）** |
| 绝对误差中位 / p95 / 最大，对 FP64 参考 | 0.000008655 / 0.000026561 / 0.000052521 | **0.004387 / 0.013477 / 0.023849** |
| 原容差中位数 | 0.000122989 | 0.007810505 |
| 最大 error/tolerance | 0.419902 | **3.068338** |
| 误差方向朝零 | 83.4229% | 83.6670% |

长 K 最坏点为 (47,18)：输出 247.32081604003906，FP64 参考 247.34466478994727，误差 −0.023848749908211175，sum_abs=16298.177030993618。原 `max_storage_reference_error=0.0238494873046875` 使用 float(ref)，两种口径均已复现。正/负误差分别 2010/2086 点，不是每个输出减去同一个常数。覆盖这批输出所需的最小 sum_abs 系数为 1.463218628e−6（原系数的 3.06859 倍），**仅是测后描述，不采用为新容差**。长短组同时改变几何和 seed；设备实际输入 buffer 未归档；这些数据不能唯一识别 K、舍入/截断或输入生成机制。结果与逐点表在 [C-20261009-random-error-v1](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-R13-shared-smoke-job738097/reanalysis/C-20261009-random-error-v1/diagnostic.json)。

本批先逐一执行三个 K=20480 random plain 检查，记录在 `samples/check_<case>/`，不计入正式 trial 0。**数值失败只影响该条件**：继续其他已通过检查的 dyadic/zero/random，15 条原矩阵保持不变；`random_checks.json` 和 `sampling.json` 保存数值失败及已采样列表。正式过程仍调用公共 `run_v08.run_one`，沿用原暖机、10 trial 和条件/variant 随机顺序；正式采样中出现 numeric_error 的条件也停止其后续采样，不生成该条件的性能值。其他运行错误按公共 runner 原样报错。

全通过时为 3 个预检查加 450 个正式进程；若三个 random 均失败，则 12 个条件、360 个正式进程，不能写成 15 条全部完成。`r09_analyze.py --shared` 使用公共 `v08_model.summarize_case`，输出 `summary.json`、`metrics.csv`，失败行保留 numeric_error 且不填性能数值。报告 plain/stamped/ends 时间、周期、调用内频率、包络、CV、P0/S、每 Ktile 主循环和相对本批 dyadic 的差；不做冻结评分或功率推断。

最少命令（准备与传输由管理者安排；远端新目录仅用节点本地 /tmp，不写超配额的共享空间）：

```bash
python3 microbench/gh200_resource_campaign/access_rules/r09_run.py shared-prepare \
  --shared-run <完整公共包路径> --output <新的R09目录>
# 获得分配后，在已加载 CUDA 12.9、已绑定获配 UUID 的 step 内：
export V08_GPU=<获配GPU-UUID>
export CUDA_VISIBLE_DEVICES="$V08_GPU"
bash /tmp/<新的R09目录>/run.sh
```

`run.sh` 依次调用公共 setup、R09 shared-sample（含逐条件预检查）、R09 shared 分析。实际依赖为共享包的 9 个 cfg_a/b/c×plain/stamped/ends 二进制、对应完整 source/build 清单、Python 3 与 NumPy、CUDA 12.9 运行环境。构建已经复用，运行时不需要下载 CUTLASS 或向共享目录安装依赖。本节保留采样前的准备记录，实际结果见[下一节](#r09-clock-input-job738110)。

<a id="r09-clock-input-job738110"></a>

### job738110：长窗口与 zero 输入实测

2026-10-09，romeo-a043，**GPU-201f9d2a-b55f-2a3b-b355-3abf04c2bc88**，公共源码 6338653；不是 V08 的 GPU-099dda56，即使节点名相同也不是同卡。本批 15 条计划保留完整：12 条通过、360 个正式进程成功，三个 random 预检查均 numeric_error，无该三条件的正式性能结果。`run.sh` 的采样和分析 exit_code=0；Slurm 外层在完整归档转存并验 SHA 后，由管理者取消传输确认等待以释放资源。外层 CANCELLED 不等于样本丢失，也不能将该 Slurm 作业写成正常 COMPLETED；见 [transfer-note.txt](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-clock-input-job738110/transfer-note.txt)。

本次离线重放 360 个成功进程及三个失败预检查，检查 source/bin/SASS 与原始记录身份；汇总复现原结果。ARM/x86 的两个 CV 值仅差约 2×10⁻¹⁸，比较使用 10⁻¹² 浮点容差。[比较结果](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-clock-input-job738110/reanalysis/C-20261009-clock-input-v2/comparison.json)同时保存逐进程 gap、频率、周期和同 trial 对照；[processes.csv](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-clock-input-job738110/reanalysis/C-20261009-clock-input-v2/processes.csv)保留 360 个进程。没有重新采样或修改旧冻结判定。

**dyadic 长窗口的旧偏差方向重现，但这里只作跨卡诊断。** K=20480 的实测 ends 包络为 638.928/743.072/619.424 µs，均超过 600 µs。直接沿用旧 V08 h06 冻结值，不重拟合：

| 配置 | 本批 plain µs / ends GHz | 旧预测时间误差 | 旧频率误差 | 旧周期预测对 stamped 的误差 | 只换本批实测 f 后的时间误差 |
|---|---:|---:|---:|---:|---:|
| cfg_a | 641.328 / 1.585969 | +5.959% | −6.314% | +0.200% | −0.698% |
| cfg_b | 754.720 / 1.584560 | −10.875% | −6.585% | −16.009% | −16.713% |
| cfg_c | 624.832 / 1.632489 | +5.867% | −7.871% | −0.772% | −2.419% |

cfg_a/c 的周期接近旧预测而频率被低估，cfg_b 同时保留显著的周期低估；后者仍被偏低的频率部分掩盖。本批 long K 的后续主循环为 513.559/596.196/1025.306 cycle/Ktile，cfg_b 仍明显高于其约 512 cycle/Ktile 的计算基准。实际偏差数值不能当作旧参考卡的新验证成绩，也没有得到可直接替换的跨卡 F 或频率常数。

从 K=1024→8192→20480，ends 频率分别为 cfg_a 1.67735→1.57099→1.58597、cfg_b 1.66923→1.55646→1.58456、cfg_c 1.72117→1.60852→1.63249 GHz；中窗口到长窗口没有继续单调下降。K 同时改变足迹、供给和主循环占比，三点不足以唯一识别长时频率机制。

**zero 的频率提高，没有统一变成等比例的时间下降。** 以下是每项各自的跨进程中位数，比较同一 cfg、M=N=3584、K=20480、seed=17 的两种输入：

| 配置 | plain dyadic→zero µs | plain 变化 | ends 最大 CTA 周期变化 | ends 频率变化 | ends 包络变化 |
|---|---:|---:|---:|---:|---:|
| cfg_a | 641.328→619.200 | −3.450% | +5.817% | +9.339% | −3.498% |
| cfg_b | 754.720→770.912 | +2.145% | +11.539% | +9.641% | +0.818% |
| cfg_c | 624.832→575.760 | −7.854% | +0.122% | +8.289% | −7.519% |

cfg_b 的 SM 周期窗口增幅足以抵消频率升幅，不能假定同一几何的 C 不随输入变化。cycle 是 SM 时钟域的窗口长度，不是动态指令条数；更长的周期窗口本身不证明增加了指令或物理流量。本批没有功率或计数器测量，不据此归因于功率、压缩、缓存或某个供给上限。

为定位周期增量，每个 stamped 进程选取本 CTA `end_c−entry_c` 最大者，直接分解 `C=P0+S+L0+Σ后续L+Σ(fm[j]−me[j−1])+(end−最后me)`。逐调用严格闭合，再对 10 个进程取算术均值，cfg_b 的 zero−dyadic 为：

| 区间 | 平均差，cycle |
|---|---:|
| 总 C | +132758.3 |
| P0 / S | −22.8 / −79.0 |
| 首 tile 主循环 L0 | +17597.8 |
| 后续主循环合计 | +113906.6 |
| 主循环之间的交接间隔合计 | +1355.7 |
| 最后主循环完成到端点 | 0.0 |

主循环区间占该平均增量的 **99.06%**，但 L 含流水线等待，不是纯 Tensor Core 指令服务。每次最大周期 CTA 可以不同；这是 stamped CTA 窗口的分解，不是完整 event 时间分解。原汇总中 cfg_b 的 P0/S、最后 E、Etail 基本不变；L0 173848→189432.5 cycle、后续 L 190782.757→212098.946 cycle，是主要变化。cfg_b 为 pingpong，重叠 Efull 不能再与主循环直接相加；原区间汇总的均值/中位数也不能当作同一条精确关键路径。

**固定项和聚合口径。** cfg_b 的 `median(C)/median(f)` 换算时间增长 1.731%；先在每进程计算 C/f 再取中位数，则为 745.403→749.566 µs，仅增长 0.559%。该代理仍混合最大周期 CTA 和最多 tile CTA 的中位频率；直接测得的 ends 包络为 743.072→749.152 µs（+0.818%）。因此不能把任意 C/f 聚合当作实测包络。

同次 ends 调用内的 `median(event−包络)`，cfg_a 为 3.680→3.984 µs、cfg_b 为 **3.808→3.904 µs**、cfg_c 为 4.000→4.000 µs。cfg_b 只增加约 0.096 µs；另算 `median(plain)−median(ends包络)=11.648→21.760 µs` 不能据此声称固定项增加 10.112 µs。后者混合两个进程群，容易受状态和统计口径影响。

**扰动与波动限制。** cfg_b dyadic/zero 的 plain CV 为 1.390%/2.119%，ends CV 为 1.321%/2.171%；ends/plain 差为 −1.054%/−2.356%，stamped/plain 差为 −0.789%/−0.137%。因此较小的总时间增幅不能精确解释为新增的某个固定代价。其 ends 周期 CV 为 1.689%→2.359%，频率 CV 反而为 0.708%→0.365%；波动没有仅表现为频率波动。

按相同 trial 编号对照，cfg_a/c 在 plain、stamped、ends 中均为 10/10 次 zero 更快；cfg_b 的 zero 较慢次数分别为 8/10、8/10、7/10，plain 相对变化范围 −0.864%～+5.525%。trial 是打乱顺序的采样轮，两进程相隔约 0.7–22.5 s，不是同时或紧邻的调用。合适结论是“本批 cfg_b 未观察到随频率上升而下降的总时间”，不能把慢 2.145% 当作可迁移常数。

**random 保持失败。** 三个 K=20480 预检的抽检索引与输出逐点一致，各 **158/4096（3.8574%）**超出原阈值，最大 error/tolerance=1.7895834006；非有限值和 padding 错误为零。对 FP64 参考的绝对误差中位/p95/最大为 0.000759928/0.002315922/0.004383186，原容差中位数为 0.002441932。`max_storage_reference_error=0.0043792724609375` 使用 float(ref)，两种口径均已复算。3434/4096 点误差朝零，与此前长 K 失败的趋势相似，仍不识别数值机制。结果见 [random-error-v1](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-clock-input-job738110/reanalysis/C-20261009-random-error-v1/diagnostic.json)；不放宽容差，不使用失败预检的 event 时间形成性能结论。

复核命令（输出目录使用新后缀）：

```bash
ROOT=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
python3 microbench/gh200_resource_campaign/access_rules/r09_clock_input_compare.py \
  --run "$ROOT/20261009-R09-clock-input-job738110" \
  --v08 "$ROOT/20261008-V08-job737322-v1" \
  --output "$ROOT/20261009-R09-clock-input-job738110/reanalysis/C-clock-replay-<新后缀>"
python3 microbench/gh200_resource_campaign/access_rules/r09_input_error.py \
  --run "$ROOT/20261009-R09-clock-input-job738110" \
  --output "$ROOT/20261009-R09-clock-input-job738110/reanalysis/C-random-replay-<新后缀>"
```

<a id="clock-form-candidates"></a>

### 2026-10-09：固定周期后的时间换算候选

**没有得到可直接替换冻结规则、同时改善频率与完整时间的定值。** 四项全拟合降低了按几何留出的频率误差，但按 K 段留出退化；删除时长项也不稳健。有稳态极限的函数族可保留为局部候选，尚不能把其时间尺度解释成硬件常数。以下仅为离线开发比较，没有新 GPU 采样或公共 model/fit 修改。

只用 V08 原来选入时间换算的 **64 个校准条件**拟合，保留原排除项；V08 与 V08F 的 GPU 都是 GPU-099dda56。周期 C、由其得到的 φ/μ、F、κ 均保持 V08 冻结值，所有模型直接拟合频率 GHz，不以总时间误差反调系数。因此不会通过这一步修改供给周期。

这是**固定周期模型、F、κ 后的条件式频率交叉验证**，不是整个 GEMM 预测管线的交叉验证；这几个固定量原本使用了全部 V08 校准。频率系数另按两种方式留出：21 个 M/N 几何组（同几何的全部配置与 K 一起留出）；三个 K 段（K≤2048、2048<K≤8192、K>8192，分别 39/19/6 点）。每个候选在每个方案下都预测全部 64 个未参与本折频率拟合的点。基线在每个训练折仅重拟合 a/c，b/d 仍用旧 V03 值；已见过所有校准点的原冻结参数另列，不冒充留出基线。

比较下列少量形式，其中 W 用 µs，D 用 TB/s，f 用 GHz：

| 形式 | f(W) | 可调量 |
|---|---|---|
| 原形式基线 | a−bφ ln W−cD−dμ | 只拟合 a/c |
| 四项全拟合 | a−bφ ln W−cD−dμ | a/b/c/d |
| 无时长项 | a−cD−dμ | a/c/d |
| 饱和平均项 | a−bφ·g(W/τ)−cD−dμ，g(x)=1−(1−exp(−x))/x | a/b/c/d 与 τ |

g 对应指数松弛函数在调用窗口内的平均变化，W→∞ 时趋于 1；这是候选形状，不是已经测出的瞬时调频过程。τ 在 0.1–100000 µs 的固定对数网格上作一维剖面，每折只按该折训练集的 GHz 残差选 τ。其余系数用无符号约束的最小二乘；未作频率截断或测后删点。

拟合时允许代入观测 ends 包络 W；**自由预测另行求解** `W=κC/(1000f(W))`、`D=B/(10^6 W)`。φ、μ、模型估计字节 B 来自冻结周期/调度预测，不使用实测周期；求解从原预测器相同的 κC/1600 起点开始。不能把 `f(观测W)` 的误差当作自由预测成绩。校准和 V08F 的 B 使用现有排序 LRU，heldout 的 B 从原冻结预测反解以保留原值；原 a/c 重拟合与冻结常数的微小差异源于旧 LRU 遍历顺序，原冻结基线仍独立保留。

**校准整组留出结果，均为 RMS 相对误差百分比：**

| 形式 | 几何留出：观测W频率 / 自由频率 / 自由时间 | K段留出：观测W频率 / 自由频率 / 自由时间 |
|---|---:|---:|
| 原形式，只拟合 a/c | 2.905 / **2.847** / **4.034** | 2.819 / **2.739** / **4.040** |
| 四项全拟合 | 2.189 / **2.177** / 4.117 | 3.487 / **3.197** / 4.790 |
| 无时长项 | 2.691 / **2.607** / 4.373 | 4.359 / **3.771** / 5.204 |
| 饱和平均项 | 2.216 / **2.187** / 4.185 | 2.760 / **2.733** / 4.615 |

四项全拟合与饱和项的几何留出频率接近；后者在 K 段留出较稳健，但频率误差只是接近原基线，完整时间没有改善。单独留出 K>8192 六点时，自由频率 RMS 分别为 1.160%、2.220%、2.370%、1.916%；饱和项也没有全面优于原形式。总时间中仍有固定周期、κ 和端点代理的误差，不能以这些时间残差反向认定更低的频率预测更正确。

饱和项的自由频率最大误差仍为几何留出 6.105%（cfg_b_c9_thrash）、K 段留出 8.238%（cfg_a_c9_thrash），不能用约 2% 的 RMS 代表每个条件。

全部 64 点拟合的参数如下，仅作为本卡、本 dyadic 条件的开发结果：

| 形式 | a | b | c | d | τ µs |
|---|---:|---:|---:|---:|---:|
| 原冻结 | 2.138361 | 0.035340 | 0.029182 | 0.387158 | — |
| 四项全拟合 | 2.002750 | 0.030165 | 0.025296 | 0.242630 | — |
| 无时长项 | 2.047306 | 0 | 0.052289 | 0.401417 | — |
| 饱和平均项 | 1.925467 | 0.180354 | 0.024944 | 0.181526 | 39.81 |

**可辨识边界。** 所有设计矩阵满秩；列归一化条件数约为四项 19.24、无时长 13.18、饱和最优点 22.17，未出现求解失败。饱和 τ 在几何留出和 K 段留出中均落在 35.48–63.10 µs，支持一个有限曲率的经验形状，但训练频率 RMS 在最优值 +0.1 个百分点以内的 τ 为 **19.95–89.13 µs**；这只是剖面宽度，不是置信区间。τ 与幅度 b、主循环项 d 相互补偿，不能将 39.81 µs 命名为硬件响应时间。四项全拟合在 K 段留出中的流量系数 c 变化为 0.0221–0.0907；这些系数不是独立测得的物理流量或功耗系数。

**开发后外推诊断。** 下表使用全部原校准拟合后直接自由预测。V08 heldout 已参与此前开发分析，V08F 本来就是测后机制对照，因此都不算新验证。

| 形式 | V08 heldout：频率 RMS / 时间 RMS | V08F 84点：频率 RMS / 时间 RMS |
|---|---:|---:|
| 原冻结 | 3.081 / 8.182 | 2.515 / 7.589 |
| 四项全拟合 | 1.964 / 8.275 | 2.087 / 7.988 |
| 无时长项 | 2.449 / 8.227 | 2.270 / 7.906 |
| 饱和平均项 | 1.810 / 8.478 | 1.926 / 8.062 |

饱和项改善了这两组的频率误差，但时间 RMS 都变大。V08 heldout 的时间绝对误差中位/最大从原冻结 2.613%/35.338% 变为 1.928%/36.132%，不能只看中位数宣布改善。

h06 保留原周期误差不变，以下每格为**自由频率误差 / 完整时间误差**：

| 形式 | cfg_a | cfg_b | cfg_c |
|---|---:|---:|---:|
| 原冻结 | −6.225% / +5.271% | −6.732% / −13.570% | −7.190% / +5.247% |
| 四项全拟合 | −3.438% / +2.248% | −3.804% / −16.187% | −4.550% / +2.353% |
| 无时长项 | −1.193% / −0.063% | −1.713% / −17.961% | −1.294% / −1.004% |
| 饱和平均项 | −2.905% / +1.690% | −3.222% / −16.689% | −4.045% / +1.817% |

cfg_b 的冻结周期低估 **18.287%** 始终保留。频率预测更接近实测后，原来被偏低频率掩盖的周期错误更加明显。无时长项在 h06 上看似最接近频率，却在整组 K 留出上最差，不能依据这个开发点选择它。

固定 C 不表示 C 的错误与频率完全隔离：它仍经自洽 W、D 传播到自由频率。例如无时长项的 cfg_b_h06，代入观测 W 时频率误差为 −0.640%，自由预测变成 −1.713%。这也是必须单列观测W拟合与自由预测的原因，不能将自由频率余差全归给时长函数。

**本轮交回：** 不采用无时长项；四项全拟合只能作为比原固定 b/d 更灵活的局部对照，尚无跨 K 优势；可保留饱和平均项的函数族，适用范围先限原卡、dyadic 和校准组织，τ 不作物理定值。现有结果不支持单独替换时钟规则来修复完整时间。另卡 job738110 每配置只有三个 dyadic K 点，仅保留其“中到长窗口频率不单调下降”和 zero 同时改变 C/f 的事实，没有与旧卡混合拟合或用它选 τ；random 失败也不参与任何频率训练。

[candidates.json](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/C-20261009-clock-candidates-v1/candidates.json)保存训练 ID、全部折参数、τ 剖面、输入特征与分组误差，[scores.csv](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/C-20261009-clock-candidates-v1/scores.csv)保存逐点观测W拟合与自由预测。检查了 920 个全数据/开发诊断自由解：将观测 W、频率与 plain 标签分别乘 17、0.73、2 后，预测 W/f 均不变；最大自洽方程残差约 1.03×10⁻¹²。冻结 heldout 时间复算最大相对差约 10⁻¹²；所有分组折均覆盖 64 点且没有自由求解失败。

```bash
ROOT=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
python3 microbench/gh200_resource_campaign/access_rules/r09_v08_clock.py --candidates \
  --run "$ROOT/20261008-V08-job737322-v1" \
  --followup "$ROOT/20261008-V08F-job737322-v1" \
  --trend-run "$ROOT/20261009-R09-clock-input-job738110" \
  --output "$ROOT/20261008-V08-job737322-v1/reanalysis/C-clock-candidates-replay-<新后缀>"
```

<a id="wide-input-preparation"></a>

### 20480²×1024 三档输入：plain/ends 准备

本节保留 2026-10-09 的采样前准备，实际结果见 [job738197](#wide-input-job738197)。最终几何为 **M=N=20480、K=1024**，取代最初拟用的 16384²；cfg_a/b/c×dyadic/zero/random、seed=17，共 9 条。sm_count=0、swizzle=1、evict=0、紧密行距，沿用原暖机和原数值阈值，不增加持续 NVML 或新打点。K=1024 的旧检查通过不保证新几何也通过，新条件逐例检查；旧 K=20480/K=65536 random 失败保留原判定。

**内存与软件调度。** A/B 各 40 MiB，FP32 D 为 1600 MiB；合计 **1,761,607,680 B（1.762 GB / 1.640625 GiB）**。按已知 60 MiB L2，探针即使 evict=0 仍分配 120 MiB eviction buffer；再计入 132 CTA 的 827,904 B trace 和 49,152 B 输出抽检数组，显式申请小计 **1,888,313,856 B（约 1.888 GB）**，另加 workspace、分配粒度和 CUDA 运行时占用。三个配置按进程串行运行，不将九份矩阵同时驻留。random 的主机输入参考另需 A/B 约 80 MiB。

按 132 SM、原 tile/cluster 和 swizzle=1 的软件调度：

| 配置 | 预计 grid | 总输出 tile | CTA 的 tile 数分布 | 最大单个 consumer role 计数 |
|---|---|---:|---|---:|
| cfg_a | 2×66×1 | 25600 | 124 个 CTA×194，8 个×193 | 194 |
| cfg_b | 1×132×1 | 25600 | 124 个 CTA×194，8 个×193 | 97（两 role 交替） |
| cfg_c | 66×2×1 | 12800 | 128 个 CTA×97，4 个×96 | 97 |

尺寸与行距均满足当前 tile/cluster 整除和 128 B 对齐，无尺寸尾部或 swizzle 补齐。上表来自软件调度器，实际 grid、SMID 和计数仍由运行记录确认。总工作为 **858.9934592 GFLOP**；按题设 4096 FLOP/SM-cycle、132 SM、1.98 GHz 的理想满载假设，纯计算约 802.4 µs。它只说明选取更大几何的理由，**不作为任何条件超过 600 µs 的实测证明**。

**64 槽不是 ends 头部计数上限。** 已读取 6338653 归档的 [r18_trace.hpp](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-R13-shared-smoke-job738097/source/probes/r18_trace.hpp)、[r18.cu](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-R13-shared-smoke-job738097/source/probes/r18.cu) 和 `v08_model._ends`：

- `V08_ENDS` 分支的 `v06_begin` 只写 producer 首次工作，`v06_stamp` 为空；不会用 tile 下标访问逐 tile 数组或设置其溢出位。`v06_final` 将 role 总 tile 数作为 uint64 写入头部第 6/9 项。
- 探针仍分配 `16+2×64×6=784` 个 uint64/CTA，host 只检查头部溢出标志；`trace_tile_capacity=64` 描述预留逐 tile 数组容量，不限制 ends 的头部计数。
- `_ends` 读取头部：cooperative 核对两个 role 计数相等，pingpong 将两个 role 计数相加；随后 `observe` 与软件调度的 tile 数逐 CTA 对照，没有截断到 64。
- 这批三个配置的单 role 计数均可超过 64，**不得使用 stamped**。新模式的采样列表明确为 plain/ends，分析也走专门的头部汇总，不调用要求三种 variant 和逐 tile 记录的 `summarize_case`。

[test_r09_ends.py](../../../../../../microbench/gh200_resource_campaign/access_rules/test_r09_ends.py) 对真实 20480² 调度生成 CPU 合成头部，逐 tile 区域全为零，直接调用归档的 `observe/_ends` 并检查新汇总：194/97 计数均通过，故意错误的计数被拒绝，stamped 记录被新入口拒绝。合成时间只测试解析和大于 600 µs 的标记分支，不是 GPU 性能结果。另用 CPU 调用替身核对新模式为 9 个独立预检查和最多 180 个正式调用，不含 stamped；原 15 条模式与 job738110 的 12 成功/3 失败汇总保持兼容。

**准备包与运行。** 使用 [20261009-R09-wide-input-prepared-v2](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-wide-input-prepared-v2/)；v1 是被替代的 16384² 草案，不用于提交。完整复用 6338653 的 source/build 和原清单，不重编；包里保留原二进制归档，但本批只选择 6 个 plain/ends 程序。新增入口与分析器指纹和公共源码版本分别记录。

每条条件先跑 plain 数值检查，9 条均检查；numeric_error 仅排除该条件的后续性能采样，其余继续，原矩阵保留。全部通过时为 9 个预检查加 9×2×10=180 个正式进程。`numeric_checks.json`、`sampling.json` 保存结果；分析输出 plain/ends 时间及 CV、ends 最大周期和有效 GHz、包络、同调用 event−包络，以及相对 dyadic 的变化。`window_ends_gt_600us` 和 `ends_processes_gt_600us` 分别按实测中位数和逐次窗口判断，不预先将全部配置标成“长窗口通过”。不输出本批没有记录的 L/S/E 分解或 stamped/ends κ；输入效应只解释为这些条件下的周期、有效频率和时间关系，不作功率因果判断。

```bash
# 本地 CPU 核对；不会运行 GPU。
python3 microbench/gh200_resource_campaign/access_rules/test_r09_ends.py \
  --shared-run <6338653完整公共包>

# 如需重新生成准备包，目标目录必须不存在。
python3 microbench/gh200_resource_campaign/access_rules/r09_run.py shared-prepare \
  --batch wide-input --shared-run <6338653完整公共包> --output <新目录>

# 管理者安排参考卡后，将准备包放到节点本地 /tmp；在已绑定获配 UUID、
# 设置 V08_GPU/CUDA_VISIBLE_DEVICES 且加载 CUDA 12.9 的 step 中执行：
bash /tmp/<准备包>/run.sh
```

运行依赖仍为公共 CUDA 12.9 二进制、Python 3/NumPy 和同一包中的原始回放函数；不改公共文件、不向超配额共享空间追加运行数据。使用 `bash run.sh`，不要 `source run.sh`；不调用默认会包含 stamped 的公共 `run_v08.py sample`。

<a id="wide-input-job738197"></a>

### job738197：20480²×1024 三档输入实测

2026-10-09，独占 romeo-a057，**GPU-43269fbc-449d-3e0f-908a-9c81229546d3**，公共 CUDA 源码/二进制 6338653，运行 Python 3.11.9、NumPy 1.26.4（[python-runtime.json](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-wide-input-job738197/python-runtime.json)）。9 个预检查与 180 个正式进程全部通过原检查和独立 4096 点回放，作业退出 0；本次离线分析复现归档汇总。实际 grid 与上一节的 132 CTA 软件调度一致，ends 头部成功记录 cfg_a/b 的 193/194 tile 和 cfg_c 的 96/97 tile；没有 stamped 数据。

M=N=20480、K=1024、seed=17，其余沿用上一节。表中时间为跨 10 个进程的中位数；C 为 ends 最大 CTA 周期中位数，f 为最多 tile CTA 的 cycle/ns 比值中位数再跨进程聚合。**ends window 是 CTA 包络，不是 ends 的完整 CUDA event 时间**；stdout 标签已改为 `ends_window`，原汇总字段数值未改。

| 配置 / 输入 | plain event µs | ends window µs | C，百万 cycle | 有效 f，GHz | plain / ends event CV | ends event / plain 变化 |
|---|---:|---:|---:|---:|---:|---:|
| a / dyadic | 1816.784 | 1821.392 | 2.259151 | 1.248908 | 0.736% / 0.491% | +0.472% |
| a / zero | 1381.856 | 1377.936 | 2.439592 | 1.778852 | 0.634% / 0.603% | +0.028% |
| a / random | 2322.208 | 2319.232 | 2.244204 | 0.977298 | 0.544% / 0.708% | +0.054% |
| b / dyadic | 1855.328 | 1855.920 | 1.794688 | 0.965850 | 0.698% / 0.513% | +0.257% |
| b / zero | 1402.416 | 1401.040 | 2.433126 | 1.739571 | 0.952% / 0.784% | +0.230% |
| b / random | 2597.776 | 2567.632 | 1.750070 | 0.681815 | 0.796% / 1.054% | −1.001% |
| c / dyadic | 1673.152 | 1670.256 | 2.158347 | 1.300832 | 0.285% / 0.532% | +0.077% |
| c / zero | 1372.272 | 1364.704 | 2.412724 | 1.766605 | 0.527% / 0.620% | −0.178% |
| c / random | 2133.568 | 2121.872 | 2.131740 | 1.010465 | 0.591% / 0.444% | −0.337% |

**九条件的所有 90 个 ends 窗口都超过 600 µs**，其中最短单次为 1347.424 µs；这次由实测确认，而非使用纯计算估算。完整 event CV 最大 1.054%，ends/plain 中位数变化绝对值最大 1.001%。两 variant 来自不同进程，这个差不能全部认作打点固定成本。

按相同 trial 编号、相同 variant 逐次配对，先算 `当前/dyadic−1`，再取 10 对中位数：

| 配置 / 输入 | plain 时间变化，中位数［范围］ | ends event 时间变化 | ends C 变化 | ends 有效 f 变化 |
|---|---:|---:|---:|---:|
| a / zero | −24.200%［−24.744, −22.342］ | −24.056% | +7.992% | +42.140% |
| a / random | +28.269%［+26.388, +30.055］ | +27.397% | −0.681% | −21.772% |
| b / zero | −24.240%［−26.036, −22.936］ | −24.393% | +35.655% | +80.199% |
| b / random | +39.315%［+38.297, +41.015］ | +39.060% | −2.559% | −29.391% |
| c / zero | −18.144%［−18.584, −17.001］ | −18.083% | +11.892% | +35.681% |
| c / random | +27.464%［+25.344, +28.613］ | +27.043% | −1.241% | −22.442% |

三配置的 random 在 plain/ends 中均为 10/10 对更慢，zero 均为 10/10 对更快。配对进程相隔 0.696–12.296 s；同 trial 是随机化采样轮，不是同时调用。上述逐次比值中位数不同于第一表两个中位数之比，例如 cfg_b random 的 plain 为 +39.315%，而中位数之比为 +40.017%，两者不能混用。

本批 random 较慢与有效 f 下降约 22%–29% 同时出现，C 略降约 0.7%–2.6%；zero 虽然更快，C 却上升约 8%–36%，因此不能假定输入模式只改变 f 而保持周期模型不变。C 是 SM 周期窗口，不是指令数；f 也由 cycle/ns 计算，不能把它当作独立功率证据。仅凭计时不能识别功率封顶、调频策略或其他机制；本批没有持续功率实验，也没有逐 tile 分项。C 与 f 的统计对象还不同，不能简单相除并强求完全重构 plain 时间。

固定项同样要区分口径：cfg_b random 的 `median(plain)−median(ends window)=30.144 µs`，而同次 ends 调用的 `median(event−window)=4.176 µs`。跨进程相减的 30.144 µs 不能命名为新增 launch 或固定开销。本批不与 GPU-099dda56、GPU-201f9d2a 的旧常数混合定值。

**数值与前序失败。** 全部 63 个 random 进程（3 个预检查、60 个正式进程）的原最大 error/tolerance 不超过 0.6214159601，均通过原阈值；这是本形状、seed17 和固定 4096 点抽检的结果，不改变旧 K=20480、K=65536 random 失败，也不推广为任意 FP16 输入的精度保证。

- [738195 失败包](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-wide-input-job738195-failed/)只产生 cfg_a 三个预检查；GPU 均返回成功，随后 random 独立回放在 `import numpy` 处停止。三个记录已补做本地 CPU 回放并通过，没有正式性能采样。
- [738196 失败包](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-wide-input-job738196-failed/)在 Spack 加载 NumPy 时遇到 `Errno 122: Disk quota exceeded`，进入 setup 前退出，没有测量。738197 将 Spack 用户缓存放到计算节点 `/tmp`，在 ARM 分配内加载已安装 NumPy；没有改 HOME、CUDA 源码或容差。

复核结果在 [C-20261009-wide-input-v1](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-wide-input-job738197/reanalysis/C-20261009-wide-input-v1/)：`summary.json` 保留逐进程 ends 值与预检查回放数量，`metrics.csv` 保留九条件汇总，`input_pairs.csv` 保存 60 对逐次比较与进程间隔。原分析器只增加该配对表与已成功预检查的回放，不新增测量入口或验证框架。

```bash
ROOT=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
python3 microbench/gh200_resource_campaign/access_rules/r09_analyze.py --shared \
  --input "$ROOT/20261009-R09-wide-input-job738197" \
  --output "$ROOT/20261009-R09-wide-input-job738197/reanalysis/C-wide-replay-<新后缀>"
```

### Tensor 计算活动代理与 V09 输入校准边界

2026-10-09，离线。原 μ 使用拟合的聚合主循环周期/Cmax，会把已包含的供给等待、补齐 tile 额外时间计入“计算活动”。本次只在已有有界时长式 `f=a−bφg(W/τ)−cD−dμ` 中比较 μ 定义，不扩函数族；仍用原 V08/099dda56 的 64 点训练，21 个 geometry 组、三个 K 段留出。C、φ、F、κ 固定；自由预测自洽求 W/D，不使用观测时间或实测 C 作预测输入。

令 `q_cfg=2·tile_M·tile_N·64/4096`，cfg_a/b/c 分别为 **512/512/1024 cycle/Ktile**；Kt=ceil(K/64)。两种软件代理为：

- **同作用域替换**：`μ_TC,crit=N_crit·Kt·q_cfg/Cmax`，N_crit 是周期模型所选关键 CTA 的软件 tile 数。它用名义 Tensor 计算需求替代聚合主循环时长，避免把已建模的供给等待直接算成计算。
- **全卡作用域对照**：`μ_TC,device=(ΣN_i)·Kt·q_cfg/(132·Cmax)`。它额外改变作用域，不能将与原 μ 的全部差异解释成去掉等待。

tile 数包含软件实际调度的补齐/OOB tile，不按有效输出面积打折。二者都是理想计算需求占预测窗口的代理，不是实测 Tensor 活跃计数器，也不包含输入位模式的直接影响。原聚合主循环来自经验拟合，局部可以略低于该计算参考，因此替换不等于从观测中精确扣除一段等待。

| 有界函数的 μ | geometry：自由频率 / 完整时间 RMS | K分组：自由频率 / 完整时间 RMS |
|---|---:|---:|
| 原聚合 μ | 2.187% / 4.185% | 2.733% / 4.615% |
| μ_TC,crit | **2.007% / 4.099%** | **2.679% / 4.529%** |
| μ_TC,device | 2.134% / 4.116% | 3.237% / 4.942% |

同作用域的计算代理有小幅一致改善，暂不采用 K 分组退化的全卡版本。全部校准拟合得到 `a=1.928433, b=0.178588, c=0.028976, d=0.182815, τ=31.62 µs`；这是 **099dda56、dyadic、固定旧 C 下的候选定值**，不是 a057 的参数。τ 在 geometry 折中为28.18–44.67 µs、K折中为15.85–63.10 µs；尚无精确物理时间常数的识别依据。

已知 V08 heldout 的自由频率/时间 RMS 从原有界模型 1.810%/8.478% 变为 1.777%/8.409%；V08F 为 1.926%/8.062%→1.530%/7.955%。这些仅是开发诊断。尤其 cfg_b h06，μ 仅从 **0.987539→0.985701**，冻结 C 仍低估 **18.287%**，时间误差仍为 −16.400%。原周期预测已经漏掉其真实供给等待，换 μ 无法恢复这部分信息；自由频率还会受到错误 C 经 W/D 的传播。原 F/κ 与 C 使用过全部校准，故分组结果仍是条件式频率比较，不是全管线交叉验证。

**可交给完整模型的定义**是 μ_TC,crit，而非现成的跨卡系数。在完整预测中，分母应来自该输入、该条件的周期模型；若供给/输出改为自然纳秒单位并参与自洽换算，应同步更新 C、关键 CTA 和 μ，不能把本次固定 C 下的 μ 永久缓存。软件工作量相同而输入改变时，若 C 模型不感知输入/有效 f，μ 也完全相同，仍无法区分 R09 的三档输入。

#### a057/43269fbc 现有证据能约束什么

三批 GPU UUID 已核对相同，但不同作业、不同打点的频率不混入一个拟合；本次没有对它们拟合系数。

| 数据 | 可用窗口与输入 | 全 CTA 有效 cycle/ns | 不能混入全程拟合的量 |
|---|---|---|---|
| R09 job738197 | 三配置、三输入；每配置只有一个长窗口几何，ends中位1.365–2.568 ms | dyadic 0.966–1.301；zero 1.740–1.779；random 0.682–1.010 GHz | 只有这一长窗口的输入差，不能外推成短/中窗口固定偏移 |
| R10 job738203 | cfg_b、dyadic、五种B pitch；stamped/dual包络25.856–33.808 µs | stamped/dual全CTA约1.698–1.723 GHz | dual后续主循环约1.659–1.739 GHz是另一端点；wide是无时钟记录的scratch匹配plain对照 |
| R15 job738100 | cfg_c、dyadic；ends包络41.408/47.200/185.936 µs | ends全CTA 1.81527/1.65029/1.47135 GHz | global输出阶段1.89183/1.91343/1.59063 GHz，不能替代全CTA值 |

R10 的逐 tile 打点 event 比 plain 高约1.7%–6.3%，不直接充当 ends 频率校准；其局部主循环率只约束局部换算。R15 的 global 全CTA率与输出局部率也分别保存，不能取一个局部值填入整调用模型。现有数据只能定九个长条件的观测输入差，**不足以分别确定每种输入的时长响应/稳态幅值/τ**；三个配置的重复进程不等于新增时长条件。也不把旧 099dda56 或 201f9d2a 的系数搬到43269fbc。

已接受的最小缺口为 **M=N=2048、8192，K=1024，cfg_a/b/c×三输入，共18条**，仍用plain/ends、seed17及原阈值。分别覆盖短、中窗口；几何也改变流量、工作分配，不能称纯时长干预。新分配另加cfg_b的20480²×1024三输入作为旧长窗口桥接，共21条；桥接只判断旧长数据是否可复用，不静默校正不一致，不预先重复其余六个长条件。填补这些条件也不保证所有参数唯一可辨或V09通过。

结果：[Tensor活动比较](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/C-20261009-tensor-activity-v2/candidates.json)与逐点`scores.csv`；[同卡窗口证据](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-wide-input-job738197/reanalysis/C-20261009-input-clock-coverage-v1/input-clock-evidence.json)。后者保留25个独立协议条目，没有合并阶段频率。

```bash
ROOT=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
python3 microbench/gh200_resource_campaign/access_rules/r09_v08_clock.py --activity \
  --run "$ROOT/20261008-V08-job737322-v1" --followup "$ROOT/20261008-V08F-job737322-v1" \
  --trend-run "$ROOT/20261009-R09-wide-input-job738197" \
  --output "$ROOT/20261008-V08-job737322-v1/reanalysis/C-activity-replay-<新后缀>"
python3 microbench/gh200_resource_campaign/access_rules/r09_v08_clock.py --input-evidence \
  --run "$ROOT/20261009-R09-wide-input-job738197" \
  --r10-run "$ROOT/20261009-R10-b-coverage-job738203" --r15-run "$ROOT/20261009-R15-output-ns-job738100" \
  --output "$ROOT/20261009-R09-wide-input-job738197/reanalysis/C-input-coverage-replay-<新后缀>"
```

#### 21 条件同卡准备包

[20261009-R09-input-clock-calibration-prepared-v1](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-input-clock-calibration-prepared-v1/) 本节保留采样前准备；后续实测见 [job738376](#input-clock-job738376)。`cases.json` 明确区分以下用途：

| 用途 | M=N | K | 配置和输入 | 条件数 |
|---|---:|---:|---|---:|
| calibration | 2048、8192 | 1024 | cfg_a/b/c × dyadic/zero/random | 18 |
| bridge | 20480 | 1024 | cfg_b × dyadic/zero/random | 3 |

全部 seed=17、sm_count=0、swizzle=1、evict=0，沿用原暖机、数值阈值和随机化顺序；只用 plain/ends。2048²、8192² 是预期短/中窗口的取样几何，窗口分类以实际 ends 包络为准。K=1024 避开已有长 K random 失败条件，但仍逐例检查，不据旧结果预先认定成功。每条先独立检查，全部通过时为 **21 个预检查 + 420 个正式进程**。数值失败只停止对应条件的后续性能采样，原矩阵和失败记录保留，其余条件继续。

包完整复用 **6338653** 的公共 source/build，不需要 R18 后续 `input_map_m/n` 改动或重编。运行仅选择六个 plain/ends 程序；cfg_b 长桥接单 role 可达 97 tile，仍不得调用 stamped。`run_config.json` 分别保存公共来源与 R09 包装入口/分析器哈希，并要求 **GPU-43269fbc-449d-3e0f-908a-9c81229546d3**。与 job738197 是同卡新分配，不是同一次采样。

`bridge_reference.json` 保存 job738197 的原汇总哈希、环境和 cfg_b 三条结果，且准备时核对两种 cfg_b 二进制一致。结束后 `analysis/bridge_comparison.json` 单独给出 plain/ends 时间、ends C/f/包络的新旧变化和两批 CV；**不自动作偏移校正、合并或拟合**。是否复用旧长数据仍须结合变化幅度、离散程度和运行状态判断。cfg_b 一致也不能独立证明 cfg_a/c 的跨作业稳定；若后续仍需重测其他配置，应另行指出具体缺口，不在本包预先重复九条长条件。

CPU 核对已验证 21 个预检查与最多 420 个正式调用、预检查/正式调用失败的条件隔离、按 config/M/N/K 匹配 dyadic 基线、140 对同 trial 输入比较，以及桥接报告不改原始数据且拒绝异卡。另复用原 `test_r09_ends.py` 核对超过 64 tile 的 ends 头部解析。合成数据仅验证控制流和汇总口径，没有形成新的 GPU 性能证据；验证脚本和结果保存在包的 `validation/`。

```bash
ROOT=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
python3 microbench/gh200_resource_campaign/access_rules/r09_run.py shared-prepare \
  --batch clock-calibration \
  --shared-run "$ROOT/20261009-R09-R13-shared-smoke-job738097" \
  --bridge-run "$ROOT/20261009-R09-wide-input-job738197" \
  --output "$ROOT/<新目录>"

# 由管理者另行分配上述 GPU，将完整包放到计算节点 /tmp。
# 在 ARM 分配内加载 CUDA 12.9、Python/NumPy，并绑定获配 UUID 后执行：
bash /tmp/<准备包>/run.sh
```

`run.sh` 不负责加载 Spack。运行包装应在 setup 前确认 NumPy 可导入并记录 `python-runtime.json`；可沿用 job738197 的 NumPy 1.26.4 安装 `/c5oek34`。将 `SPACK_USER_CACHE_PATH` 放在计算节点 `/tmp`，保留 HOME，避免再次触发共享目录配额失败。


<a id="input-clock-job738376"></a>

### job738376：短/中输入窗口与长桥接实测

2026-10-09，独占 romeo-a057，**GPU-43269fbc-449d-3e0f-908a-9c81229546d3**。公共 source/build 仍为6338653；R09入口/分析器为准备包记录的版本，Python 3.11.9、NumPy 1.26.4。21个预检查、420个正式进程全部通过，作业退出0；回传传输SHA为 `9fd92d75322c54c6a8454a1a2c3332609ff8a9b5072c70da5c49e867e6e05c43`。本次离线核对source/build身份、441个原始记录SHA与检查状态；147个random进程的最大error/tolerance为 **0.6214159601**。原数值检查和独立回放的通过范围仍为固定条件、seed17、4096点，不扩大精度保证。

下表只列18条校准条件，K=1024。时间、C、f的统计口径与job738197一致；C列为ends最大CTA周期的千cycle，f为最多tile CTA的cycle/ns统计。它们不是同一个跨SM包络的简单相除关系。

| M=N | 配置 | 输入 | plain event µs | ends包络 µs | 有效f GHz | C，千cycle |
|---:|---|---|---:|---:|---:|---:|
| 2048 | a | dyadic | 19.392 | 15.776 | 1.718783 | 27.018 |
| 2048 | a | zero | 18.928 | 15.200 | 1.793209 | 27.053 |
| 2048 | a | random | 20.096 | 16.528 | 1.638806 | 26.939 |
| 8192 | a | dyadic | 247.184 | 242.704 | 1.475726 | 358.882 |
| 8192 | a | zero | 214.000 | 210.288 | 1.715817 | 360.956 |
| 8192 | a | random | 308.000 | 303.536 | 1.178033 | 358.096 |
| 2048 | b | dyadic | 17.664 | 14.288 | 1.707121 | 24.326 |
| 2048 | b | zero | 17.280 | 13.600 | 1.795991 | 24.402 |
| 2048 | b | random | 18.560 | 15.040 | 1.617361 | 24.214 |
| 8192 | b | dyadic | 225.408 | 222.864 | 1.254722 | 279.075 |
| 8192 | b | zero | 178.576 | 177.120 | 1.684170 | 297.056 |
| 8192 | b | random | 300.352 | 297.184 | 0.934590 | 278.072 |
| 2048 | c | dyadic | 19.296 | 15.536 | 1.757239 | 27.225 |
| 2048 | c | zero | 18.736 | 14.944 | 1.820873 | 27.198 |
| 2048 | c | random | 19.968 | 16.256 | 1.676568 | 27.224 |
| 8192 | c | dyadic | 241.264 | 236.576 | 1.516086 | 359.829 |
| 8192 | c | zero | 217.760 | 213.376 | 1.719023 | 367.616 |
| 8192 | c | random | 294.944 | 291.968 | 1.223635 | 354.481 |

2048²的九条件包络中位数为 **13.600–16.528 µs**，90次逐进程范围13.504–16.640 µs；8192²为 **177.120–303.536 µs**，逐进程174.912–311.200 µs。短条件plain event CV最大4.836%、ends最大3.223%，ends/plain中位数变化绝对值最大3.623%；中条件分别为1.153%、1.192%、1.344%。短窗口的完整event时间更易受固定项和进程状态影响，不能把几个百分点的plain变化全部归为计算频率。

按同trial、同variant逐次比较，短窗口zero的plain变化中位数约−2.55%至−3.65%，random约+3.22%至+6.09%；两者的plain方向都只有23/30对一致，不能称每次都更快/更慢。相应有效f方向均为30/30对一致：zero约+3.61%至+5.17%，random约−4.56%至−4.83%。中窗口plain方向均30/30一致，zero约−9.59%至−20.47%，random约+22.38%至+33.52%；有效f分别约+13.47%至+34.28%、−19.35%至−25.76%。这里是逐次比值的中位数，不是上表两个中位数之比。

输入也影响周期统计：8192² cfg_b zero的ends C配对变化中位数 **+6.320%**，其random为−0.372%；另外两配置zero约+0.654%/+2.056%。这些差异继续反对“输入只改f、C保持不变”的假设。相同trial中的进程仍是先后运行，不能解释为同时A/B或功率因果实验。

**cfg_b长桥接单列。** 下表的变化均为新旧条件中位数之比减一，不与本批不同M/N的dyadic基线混用。

| 输入 | 新plain µs | 新ends包络 µs | 新有效f GHz | plain变化 | f变化 | ends C变化 |
|---|---:|---:|---:|---:|---:|---:|
| dyadic | 1851.680 | 1843.840 | 0.968796 | -0.197% | +0.305% | -0.085% |
| zero | 1400.048 | 1400.928 | 1.744627 | -0.169% | +0.291% | +0.263% |
| random | 2589.136 | 2546.928 | 0.688303 | -0.333% | +0.952% | +0.120% |

三条新旧plain变化为−0.169%至−0.333%，f变化+0.291%至+0.952%，C变化绝对值小于0.263%；两批对应plain/ends event CV均低于1.055%。这些结果支持这三条cfg_b条件的跨作业一致性，仍不自动认证cfg_a/c旧长数据，也不作偏移修正。本次桥接三条保持独立用途；以下拟合没有使用它们。

原始实测见 [job738376](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-input-clock-calibration-job738376/) 的 `analysis/summary.json`、`metrics.csv`、140对 `input_pairs.csv` 和独立 `bridge_comparison.json`。

<a id="input-clock-mode-fit"></a>

#### 输入作用于计算活动项：无约束诊断

只在已有有界时长式中将一个d替换为三个输入系数，共用a/b/c/τ：

\[
f_i(W)=a-b\phi\,g(W/\tau)-c\frac{B}{10^6W}-d_i A(W),\qquad
g(x)=1-\frac{1-e^{-x}}{x},\quad i\in\{\mathrm{dyadic,zero,random}\}.
\]

W单位为µs，f为GHz，B为原LRU流量估计的byte，`B/(10^6 W)`为估计TB/s。未增加输入截距、每输入τ或配置偏移。仅比较以下两个活动项，函数族相同：

- **μ_TC**：`A=Qcrit/Cmodel`，无量纲，d_i单位GHz。Qcrit为周期模型所选关键CTA的软件tile数×ceil(K/64)×q_cfg；包含调度的补齐工作，q_cfg沿用512/512/1024 cycle/Ktile。
- **计算需求率**：`A=Qcrit/(1000 W)`，单位为名义SM计算cycle/ns，即GHz；d_i无量纲。这是已知Tensor工作量除以时间的代理，不用目标实测周期。4096 FLOP/SM-cycle只定义名义换算，不能把该值解释成实测Tensor活跃率或功率。

只用job738376的 **18条短/中条件**拟合，按条件中位数各计一点；180个ends重复进程不增加设计条件数。新cfg_b长桥接3条和job738197的旧cfg_a/c长条件6条只做开发诊断，不进入系数、τ或模型选择。C/φ/F/κ仍来自原099dda56 V08校准；未用目标ends C作输入，也未把旧卡的时钟系数直接移植到本卡。

校准允许用观测W构建设计矩阵；τ沿用0.1–100000 µs、121点对数网格，最小化训练GHz均方残差。以下最佳数值仅用于复核失败方式，**两组均未接受为完整模型参数**：

| 参数/诊断 | μ_TC | 计算需求率 |
|---|---:|---:|
| a | 1.728702 | 2.626561 |
| b | 167.533948 | 797.507187 |
| c | −0.857016 | −8.143029 |
| d_dyadic | 1.588866 | 9.126369 |
| d_zero | 1.498990 | 8.992763 |
| d_random | 1.634121 | 9.249207 |
| τ µs | 100000，上界 | 100000，上界 |
| 观测W下频率相对RMS，18点 | 1.226% | 7.162% |
| 固定τ线性设计秩 | 6/6 | 6/6 |
| 列归一化设计条件数 | 38.33 | 638.90 |
| 增加logτ后的Jacobian秩 / 条件数 | 7/7 / 6.96×10⁴ | 7/7 / 1.25×10⁵ |

数学满秩没有给出稳定的时间常数。以“网格MSE不超过最小值的1.10倍”为**敏感性范围，而非置信区间**，μ_TC的τ同时允许 **0.447–0.562 µs** 和 **199.5–100000 µs** 两段；a为1.689–9.641、b为0.505–167.534、c为−1.106至−0.857。三个d范围依次为1.589–1.672、1.499–1.619、1.634–1.691；zero−dyadic为−0.0899至−0.0504，random−dyadic为+0.0152至+0.0457 GHz。τ与共同项能互相补偿，输入差的方向比共同参数稳，但不能据此定出物理幅值。

需求率版本τ允许 **89.13–100000 µs**；a为2.627–2.789、b为2.227–797.507、c为−13.123至−8.143，三个d依次为9.126–14.260、8.993–14.138、9.249–14.400。其zero−dyadic为−0.1336至−0.1216，random−dyadic为+0.1228至+0.1398，均无量纲。两版本近优范围内c始终为负，不能把它解释成可信的流量降频系数；最佳参数的W→∞频率也为负，已有界函数并未因此获得可用的正频率稳态。

需求率与流量项的混淆可直接从软件工作量看出。18条中，2048²的 `Qcrit=16384 cycle`、`B=16777216 byte`，8192²为 `262144 cycle`、`285212672 byte`，各尺寸三个配置相同。因此 `A/D=1000Qcrit/B` 仅由尺寸取 **0.9765625、0.9191176** 两值。在单尺寸内，三个输入活动列之和与D严格成比例，c和共同d不能分别识别，设计秩为 **5/6**；两种尺寸合在一起才由这6%的比值差得到6/6。两次留出整个尺寸的需求率拟合均保留“rank deficient”，未用任意最小范数解生成预测。μ_TC两次尺寸留出虽满秩，观测W下的留出频率RMS却分别约15086%和208%，也不具备跨尺寸迁移能力。配置留出结果同样保存在产物中；本批K全为1024，无法识别K依赖。

#### 无约束自由预测与可组合边界

自由预测只解 `W=κ Cmodel/(1000 f_i(W))`。分析器将等式化为W的一元方程，利用其至多一个驻点检查全部正根；不让迭代初值替代分支选择。原始目标观测W/f/C/plain时间都只用于事后误差，已对两模型、27个条件做54次替换检查，正根完全不变。

| 18个校准条件的自由解 | μ_TC | 计算需求率 |
|---|---:|---:|
| 全正W范围：无正根 / 两个正根 / 唯一正根 | 6 / 12 / 0 | 4 / 14 / 0 |
| 限于训练W外包络13.6–303.536 µs：唯一根 / 缺根 | 12 / 6 | 14 / 4 |
| 仅域内有根子集：频率相对RMS | 12.276%，12点 | 12.464%，14点 |
| 同一子集：完整时间相对RMS | 19.205%，12点 | 10.753%，14点 |

域内统计同时保留缺根数，不能与18点观测W拟合RMS直接等同。第二个正根多在训练外包络之外；限制模型域可以移除这些域外分支，但仍有6/4个校准条件缺根，而且两档窗口之间的未采样区间没有被外包络验证。域边界只来自训练数据，未读取目标观测W来挑根。

旧周期模型经κ转换后，相对本批ends C的18点RMS为 **6.334%**，逐条件约+2.048%至+9.106%；cfg_b长桥接zero则为 **−26.294%**。新三条长桥接的自由解均落在训练窗口外，μ_TC完整时间RMS约65.875%、需求率约131.302%；旧cfg_a/c六条已知长条件分别约28.323%、61.718%，仅作跨作业开发诊断。这些结果没有形成新的时间预测通过结论。

可交给完整模型的是上述函数接口、Qcrit定义、单位及失败约束。以上无约束搜索不能交付可信的共同τ、独立流量系数或正稳态范围，也不能把任何一组最佳系数作为a057正式定值。更直接的后续是先用本批已归档C/f/窗口与供给/输出自然ns模型修正输入相关C，再重新计算关键CTA、φ和μ；若保留需求率对照，c与共同d还需要独立约束或不同Q/B的校准信息。现有数据可以重算，不据此新增函数族、自动混入旧长数据或立即重复GPU。

结果在 [C-20261009-input-clock-v3](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-input-clock-calibration-job738376/reanalysis/C-20261009-input-clock-v3/)：`input-clock-candidates.json`保存域、完整τ剖面、秩、参数范围、分组折和开发诊断；`input-clock-scores.csv`保存全部正根与域内根；`input-clock-profile.csv`保存两活动项的242个网格拟合；`cpu-verification.json`保存目标观测独立性、正根闭合、单根/双根/无根/切根及秩缺失检查。v1/v2为收尾过程中保留的初稿，引用v3。

```bash
ROOT=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
python3 microbench/gh200_resource_campaign/access_rules/r09_v08_clock.py --input-candidates \
  --run "$ROOT/20261009-R09-input-clock-calibration-job738376" \
  --calibration-run "$ROOT/20261008-V08-job737322-v1" \
  --trend-run "$ROOT/20261009-R09-wide-input-job738197" \
  --output "$ROOT/20261009-R09-input-clock-calibration-job738376/reanalysis/C-input-clock-replay-<新后缀>"
```


#### 同一函数的约束拟合与唯一正根

无约束最优解的不合法只说明该参数解不适用，不能据此否定有界函数形式。此次继续使用同一a/b/c/d_i/τ形式、同一18点训练和两个活动项，将参数约束直接加入固定τ的凸最小二乘；没有测后夹紧f，也没有扩函数族。

job738376归档的 `nvidia-smi-before.txt` 和 `nvidia-smi-after-ctrl.txt` 的 **Max Clocks / Graphics、SM均为1980 MHz**，故取 `a≤1.98 GHz` 的保守上界，记录两份文件的SHA。该值只约束零惩罚截距，不用NVML瞬时时钟替代clock64换算。其余约束为 `b,c,d_i≥0`，并以 `ε=10⁻⁶ GHz` 明确严格正余量：

- 需求率版本：`a−b≥ε`，适用 `0≤φ≤1`。
- μ_TC版本：每种输入都要求 `a−b−d_i≥ε`，适用 `0≤φ,μ≤1`。这比只约束a>b多限制了常数活动项。

18条校准、3条新桥接和6条旧cfg_a/c诊断的φ/μ均在该域内；其中校准μ为0.571822–0.858811。运行入口和自由求解拒绝域外代理，不静默clip。ε只保证数学严格不等式，不代表实测最低SM频率。

对需求率版本，将自洽式写成

\[
H_i(W)=(a-b\phi)W+b\phi\tau(1-e^{-W/\tau})
-\frac{\kappa C}{1000}-\frac{cB}{10^6}-\frac{d_iQ}{1000}=0.
\]

`H_i(0)<0`，且

\[
H_i'(W)=a-b\phi+b\phi e^{-W/\tau}\ge a-b>0,\qquad H_i(W)\to+\infty.
\]

因此在正C、非负B/Q及声明的φ域内，**存在且只有一个正根**。μ_TC版本的导数为 `a−d_iμ−bφ+bφ exp(−W/τ)≥a−b−d_i>0`，同样成立。所得根的 `0<f≤a≤1.98` 来自参数和非负惩罚，不是求解后裁剪。

固定τ用SLSQP求凸二次问题，校验可行性及KKT残差；仍扫描同一121点τ网格、按训练GHz MSE选值。两个完整18点设计仍为6/6秩；最佳Jacobian为7/7，列归一化条件数由无约束的约6.96×10⁴/1.25×10⁵降为 **324.95/1771.29**。最优解如下，数值只作为当前C下的条件式候选：

| 参数/边界 | 约束μ_TC | 约束计算需求率 |
|---|---:|---:|
| a GHz | **1.980000，上界** | 1.921456 |
| b GHz | 1.278776 | 1.856766 |
| c | **0，下界** | **0，下界** |
| d_dyadic | 0.401461 GHz | 0.142737 |
| d_zero | 0.134011 GHz | **0，下界** |
| d_random | 0.654768 GHz | 0.315288 |
| τ µs | 446.684 | 501.187 |
| 单位域的统一稳态下界 | a−b−max(d)=0.046455 GHz | a−b=0.064690 GHz |
| 最优点KKT最大绝对残差 | 8.88×10⁻¹² | 7.49×10⁻⁸ |

同样以网格MSE≤最小值1.10倍作敏感性范围：μ_TC的τ为 **158.49–562.34 µs**，a始终触顶1.98，b为0.6119–1.3118，c始终在0边界，三个d依次为0.3900–0.4425、0.1167–0.1707、0.6548–0.6900 GHz。需求率τ为 **158.49–630.96 µs**，a为1.8978–1.9357，b为0.8067–1.9119，c和d_zero始终在0边界，d_dyadic为0.1427–0.1509、d_random为0.3153–0.3310。τ不再落在网格上界，但仍有较宽的可替代范围；这些范围不是统计置信区间。c或d_zero触零也不证明实际流量或zero输入的Tensor指令没有代价。

| 18个校准条件，均值之外仍保留逐点误差 | 约束μ_TC | 约束计算需求率 |
|---|---:|---:|
| 观测W下频率相对RMS / 最大绝对误差 | 5.563% / 15.878% | 8.582% / 28.476% |
| 全正W范围：唯一根 / 无根 / 多根 | **18 / 0 / 0** | **18 / 0 / 0** |
| 自由频率相对RMS，18点 | 6.418% | 9.247% |
| 条件式完整时间相对RMS，18点 | 8.713% | 10.487% |
| 预测W落在训练外包络之外 | 3/18 | 3/18 |

最大观测W频率残差都出现在8192² cfg_b random。这是训练条件本身的失配，不能用唯一正根当作精度通过。μ_TC两次整个尺寸留出的观测W频率RMS约14.981%/11.580%，自由频率14.981%/13.894%；它们又分别超出训练所含的一档窗口范围。需求率的两次单尺寸训练仍为5/6秩，并存在**满足约束的连续等价解**；分析器检查设计零空间与活动约束后保留 `nonunique_constrained_solution`，不拿优化器碰巧选出的一个系数组合作迁移预测。非负约束保证解域，并未自动补出缺失的可辨识信息。

长条件未进入参数选择。3条新cfg_b桥接全部有唯一正根，但都在训练窗口外；其观测W频率RMS为 **42.501%/60.865%**，自由频率58.224%/89.597%，条件式完整时间168.895%/790.824%（μ_TC/需求率）。旧cfg_a/c六条仅作跨作业开发诊断，条件式时间约155.187%/1336.315%。约束后的稳态虽为正，当前短/中拟合仍明显不能外推到这些已知长条件。

可组合的数学域现已明确：GPU-43269fbc、cfg_a/b/c、seed17三种输入，经验校准仅K=1024、M=N=2048/8192；自由求解使用预测W/C，且要求相应φ/μ域和上述参数不等式。**当前没有新的完整时间预测通过，也未将系数接入正式模型**：18点时间仍含旧099dda56 C/F/κ，周期误差与长zero的−26.294%缺口保持原记录。约束μ_TC在本批误差较小，可保留为完整模型修正C后的首个重算候选；需求率对照保留其唯一根证明、较大残差和c/d混淆。下一步应使用已归档证据修正C及必要的独立项约束，再重评现有候选，而不是根据无约束多根断言函数族不可用。

约束结果见 [C-20261009-input-clock-constrained-v2](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-input-clock-calibration-job738376/reanalysis/C-20261009-input-clock-constrained-v2/)。它与前面的无约束v3分别保留；`input-clock-candidates.json`含Max Clocks来源SHA、参数不等式、活动边界、KKT、零空间非唯一性、τ范围及导数证明。CPU检查覆盖242个完整校准网格解的约束/KKT、54次目标观测替换、72个单位域角点/接近约束边界的唯一根和越域拒绝，均通过；全部产物仍为离线分析，没有新GPU作业。

```bash
ROOT=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
python3 microbench/gh200_resource_campaign/access_rules/r09_v08_clock.py --input-constrained \
  --run "$ROOT/20261009-R09-input-clock-calibration-job738376" \
  --calibration-run "$ROOT/20261008-V08-job737322-v1" \
  --trend-run "$ROOT/20261009-R09-wide-input-job738197" \
  --output "$ROOT/20261009-R09-input-clock-calibration-job738376/reanalysis/C-input-constrained-replay-<新后缀>"
```


<a id="source-clock-dev-v1"></a>

### 21条件开发版本：Tensor工作率与逻辑源请求率

2026-10-09，启动独立的 `R09-input-source-clock-dev-v1`。**job738376同一分配的全部21条件进入训练，包括三条原cfg_b长桥接**；原 `cases.json` 的18条calibration/3条bridge用途及测量归档不改。三条原桥接从本版本起不再是独立检验，其历史跨作业比较仍保留。job738197的旧cfg_a/c六条长条件继续只作已知跨作业开发诊断，不参与系数或τ选择。以下结果不能与前一个“仅18点训练、长桥接留外”的版本混称。

仅检验一个新候选：

\[
f_i(W)=a-d_i\frac{Q}{1000W}-e_i\frac{S}{W}g(W/\tau),
\qquad g(x)=1-\frac{1-e^{-x}}{x}.
\]

W、τ用µs，f、a用GHz；Q为**均摊到全部132个SM的名义Tensor cycle**，S为**均摊到全部132个SM的多播折算逻辑源KiB**。因此d_i无量纲，e_i单位为GHz·µs/KiB。Q/(1000W)是cycle/ns，S/W是KiB/µs；两项是瞬时计算需求率和经过共同平均响应g过滤的源请求需求率。它们是条件代理，不解释为真实功率、物理L2/HBM字节或实测Tensor活跃率。没有另加D、输入截距、配置偏移或每输入τ。

按本批无尺寸尾部的调度，对每个CTA的软件输出tile数求和，并包括未工作的SM在内平均：

\[
Q=\frac{K_t\sum_j N_jq_{cfg}}{132},\qquad
S=\frac{K_t\sum_j N_js_{cfg}}{132},\qquad K_t=16.
\]

| 配置 | q_cfg，cycle/Ktile | s_cfg，KiB/Ktile | 多播折算来源 |
|---|---:|---:|---|
| cfg_a | 512 | 24 | A 16 + B 16/2 |
| cfg_b | 512 | 32 | A 16 + B 16 |
| cfg_c | 1024 | 32 | A 32/2 + B 16 |

源请求与[R13已用定义](R13-async-retirement.md#v08-supply-fit)一致，公式为 `(2·tileM·tileK/clusterN + 2·tileN·tileK/clusterM)/1024`。2048²/8192²的Q在三配置间相同，分别为15887.515/254200.242 cycle/SM；相同几何的S却为cfg_b:cfg_a:cfg_c=**4:3:2**，故可以区分相同Tensor工作率下的不同源需求。cfg_b 20480²的Q为1588751.515、S为99296.970 KiB/SM。条件时钟特征Q/S不依赖旧周期模型或目标实测C；目标f仍是最多tile CTA的cycle/ns聚合，W仍为ends跨CTA包络，活动代理与目标的作用域差异保持显式。

固定τ只有a和六个非负d/e系数。沿用归档Max Clocks上界 `10⁻⁶≤a≤1.98 GHz`，其余系数≥0，直接用有界线性最小二乘求解并检查KKT；没有a>b约束或测后夹紧f。先扫描0.1–100000 µs的121点τ网格，再仅在最佳训练网格点的相邻对数区间细化81点；合并后完整训练剖面为200点，每个留出折独立选τ。

| 参数 | dyadic | zero | random |
|---|---:|---:|---:|
| d_i，无量纲 | 0.202810 | 0.178842 | 0.294641 |
| e_i，GHz·µs/KiB | 0.0171000 | 0.000905783 | 0.0311549 |

共同 **a=1.980000 GHz，触上界；τ=200.678 µs**。固定τ的设计为 **7/7秩，列归一化条件数30.34**；加logτ后的Jacobian为 **8/8秩，条件数38.55**。最优KKT残差为6.19×10⁻¹⁴。以MSE≤最小值1.10倍作敏感性范围，τ为 **179.89–222.59 µs**；a为1.9364–1.9800，三个d依次为0.1548–0.2108、0.1401–0.1798、0.2404–0.3064，三个e依次为0.01660–0.01746、0.0008941–0.0009787、0.02997–0.03189。这些范围不是置信区间，a触顶也保留为参数边界，不用它宣称真实稳态最高频率。

#### cfg_b额外降频的解释范围

21点观测W下频率相对RMS为 **2.385%**，按cfg_a/b/c分别为2.310%/1.194%/3.527%；最大误差为8192² cfg_c random的 **+8.075%**。下表只看8192² random，以同一组系数分解拟合频率：

| 配置 | 观测f GHz | 拟合f GHz | Tensor项GHz | 源请求项GHz | 相对误差 |
|---|---:|---:|---:|---:|---:|
| cfg_a | 1.178033 | 1.140641 | 0.246751 | 0.592608 | −3.174% |
| cfg_b | 0.934590 | 0.931322 | 0.252025 | 0.796653 | −0.350% |
| cfg_c | 1.223635 | 1.322441 | 0.256528 | 0.401032 | +8.075% |

三者Tensor惩罚接近，源请求项将cfg_b分到更大的条件惩罚，确实解释了本批中cfg_b random的额外降频；cfg_c残差同时说明这一代理仍有缺项。表中分解由拟合函数定义，是条件关联，不是独立功率分解。与前一版相比训练集也增加了长条件、Q作用域改为全SM平均，不能把误差改善全部归因于单独增加S。

整组留出cfg_b、只用同分配cfg_a/c的12条短/中条件拟合后，cfg_b九条观测W频率RMS为 **7.013%**，设计条件数升至163.18。8192² random误差为 **−2.342%**，dyadic为+6.204%；三个20480²输入的误差则约+11.035%/−11.754%/−11.476%。源请求特征对cfg_b中窗口random有条件迁移能力，尚不能据此认证cfg_b的所有输入和时长。

整个尺寸留出的开发诊断如下。全部折的固定τ设计仍为7/7，但系数和τ有明显变化；这些是已知数据上的条件式分组检查，没有新增独立验证条件。

| 留出M=N | 训练/留出条件数 | 训练选τ µs | 观测W频率RMS | 自由频率RMS，借旧C |
|---:|---:|---:|---:|---:|
| 2048 | 12 / 9 | 251.189 | 6.418% | 4.518% |
| 8192 | 12 / 9 | 107.771 | 10.710% | 6.071% |
| 20480 | 18 / 3 | 140.848 | 14.106% | 9.801% |

留出中窗口时，训练仅有短窗口三配置和长窗口cfg_b，即便目标W落在两端的外包络内，也不能把未采样区间当作验证过。留出长窗口三条时，它们才是该折的内部留出条件；在本版本全部21点最终拟合中仍是训练条件。本批K固定1024，不能验证K迁移。

#### 唯一根与旧模型时间迁移诊断

自由预测使用旧V08的 `Cconv=κ_old Cmodel_old`，解

\[
H_i(W)=aW-e_iSg(W/\tau)-\frac{C_{conv}+d_iQ}{1000}=0.
\]

`g(0)=0`，且g递增、凹；例如 `g(x)=∫₀¹(1−exp(−xt))dt` 可直接给出这两条性质。因此 `H_i(0)<0`、H为凸函数，并趋于正无穷，在正Cconv、非负Q/S/d/e、正a/τ域内恰有一个正根。H在零附近可以先下降，不要求全域单调。根位于明确区间

\[
\frac{C_{conv}+d_iQ}{1000a}\le W\le
\frac{(C_{conv}+d_iQ)/1000+e_iS}{a},
\]

分析器直接在此区间二分，不靠初值挑根。全部21条训练和6条旧cfg_a/c诊断均闭合且 `0<f≤a`，没有频率夹紧。该证明针对固定正Cconv；若完整模型组合后的C随f变化，须重新核对联合方程，不能直接沿用这一证明。

21点自由频率相对RMS为 **2.573%**；借旧C/F/κ的完整时间RMS为 **6.454%**，最大误差仍是原长桥接cfg_b zero的 **−21.986%**。其观测W频率拟合误差仅−1.287%，但旧转换C低估26.294%，自由频率误差变为−5.334%，说明好看的条件频率拟合不能替代周期模型修正。21点旧转换C误差RMS为8.225%，没有重新标定a057的C/F/κ，**本节时间一律为旧模型迁移诊断，不是同卡新定值或正式通过成绩**。

旧job738197的cfg_a/c六条长条件只作已知跨作业开发诊断：观测W频率RMS **6.561%**、最大11.561%（cfg_a random），自由频率4.515%、旧模型时间迁移4.276%。没有把这些已知结果当独立验证，也没有据原cfg_b桥接一致性自动认证它们。

本版本保留这个单一候选：源请求特征有助于解释cfg_b中窗口random，设计与τ剖面也比前一缺项形式稳定；但cfg_c残差、整尺寸留出和长cfg_b留出仍限制泛化。可交给完整模型组合的是Q/S定义、正根域、上述系数及明确的经验域（GPU-43269fbc、三输入seed17、K1024、cfg_a/c两尺寸和cfg_b三尺寸），而非已经认证的全模型参数。下一步应修正旧C/F/κ迁移并复核这一个候选；本轮未增加其他函数族、未混入旧长数据训练、未追加GPU。

最终产物：[C-20261009-source-clock-dev-v2](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-input-clock-calibration-job738376/reanalysis/C-20261009-source-clock-dev-v2/)。`source-clock.json`明确记录全部21个训练ID、原bridge从本版本进入训练、保持归档用途不变、单位、τ剖面、所有分组和旧长开发诊断；`source-clock-scores.csv`保存Tensor/源请求分项和自由预测；`source-clock-profile.csv`保存完整训练剖面。CPU验证核对1397个全拟合及分组剖面点的约束/KKT、27次目标观测替换、27次条件特征对旧C的独立性、162组根区间与闭合（含48组零附近先下降的H）、负工作量拒绝及训练/留出分离，均通过。浮点约束残差最大2.22×10⁻¹⁶，保留原求解值，未裁剪f。

```bash
ROOT=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
python3 microbench/gh200_resource_campaign/access_rules/r09_input_source_clock.py \
  --run "$ROOT/20261009-R09-input-clock-calibration-job738376" \
  --calibration-run "$ROOT/20261008-V08-job737322-v1" \
  --old-long-run "$ROOT/20261009-R09-wide-input-job738197" \
  --output "$ROOT/20261009-R09-input-clock-calibration-job738376/reanalysis/C-source-clock-replay-<新后缀>"
```


<a id="equal-work-job738459"></a>

### job738459：等Tensor/源请求工作量的跨K检验

2026-10-09，独占a057/GPU-43269fbc，Slurm COMPLETED/0。归档SHA为 `b5514903da686bbd6bb49545219db8bec5172a7b536610d121e1a9da92cb8d3c`，复用6338653 plain/ends。18个预检查和360个正式进程全部成功；本次独立CPU回放全部条件与原汇总一致，378个原始记录SHA和数值检查均通过。126个random进程中，K1024最大error/tolerance为0.5359343039，K4096为 **0.8684002002**；三配置K4096 random均通过原阈值，仍限seed17与4096点抽检。

**先核对九条跨作业桥接。** 本批8192² K1024的三配置×三输入，对照job738376相同九条件：plain中位数漂移−0.688%至+0.725%，ends event为−0.679%至+0.835%，有效f为 **−1.096%至+0.595%**，C为−0.301%至+0.181%，ends包络为−0.641%至+1.201%。逐条件和两批CV保存在 `bridge.csv`。原包 `bridge_comparison.json` 的“Only cfg_b bridges”是旧包装文案；其实际reference及数值比较包含cfg_a/b/c全部九条。本次注明有效范围，不追改原包或作漂移校正。

**全部九组等工作配对。** 每组比较4096² K4096 / 8192² K1024：名义Q/S完全相同，均摊Q=254200.242 cycle/SM，cfg_a/b/c的S为11915.636/15887.515/7943.758 KiB/SM。输出从256降至64 MiB，唯一输入足迹从32增至64 MiB，同时K、CTA输出tile数和重访改变，因此不是纯输出流量干预。

表中plain/C/f变化为十个同trial比值减一的中位数；**频率响应误差**先逐trial计算 `(fhat4096/fhat1024)/(f4096/f1024)−1` 再取中位数。最后一列另为条件汇总口径的K4096 `fhat/f−1`，不能与逐次比值中位数混用。所有九组、90对保留；同trial是随机化轮次，plain/ends配对进程相隔约0.676–22.531 s，不是同时调用。

| 配置 / 输入 | 配对plain变化 | 配对C变化 | 配对f变化 | 冻结clock的频率响应误差 | K4096条件频率残差 |
|---|---:|---:|---:|---:|---:|
| a / dyadic | -9.834% | -18.998% | -10.144% | +8.634% | +4.488% |
| a / zero | -15.002% | -17.000% | -1.113% | -1.367% | +0.495% |
| a / random | -4.997% | -18.936% | -13.540% | +12.775% | +9.694% |
| b / dyadic | -1.853% | -2.737% | +0.532% | -1.355% | -0.047% |
| b / zero | +1.557% | +1.397% | -0.235% | +0.397% | +0.965% |
| b / random | -0.873% | -2.497% | +0.238% | -1.342% | +0.261% |
| c / dyadic | -16.820% | -19.305% | -2.156% | -1.834% | -0.666% |
| c / zero | -20.313% | -20.573% | +0.230% | -3.453% | -1.657% |
| c / random | -14.892% | -18.795% | -4.682% | -0.536% | +6.727% |

冻结模型为原21点source-clock参数，文件SHA **`470bd395129abeec0f32aea62c87c8c1d78af2ea4a613b77fe67ca26a78101db`**，与采样前 `paired-work.json` 指纹一致。仅在各条件/进程的观测ends W下计算fhat，未重拟合任何系数/τ，未使用目标C作预测输入，也没有自由时间预测。九个K1024条件的频率RMS为 **3.099%**，九个新K4096条件为 **4.266%**；完整18条件残差及180个逐进程残差分别保存在 `clock-cases.csv`、`clock-processes.csv`。

**证据判断。** 三个全零K4096条件的频率残差只有+0.495%/+0.965%/−1.657%，没有重现R18长零区约−12%的低估；其逐trial频率响应误差为−1.367%/+0.397%/−3.453%。本批削弱了“K4096本身或统一缺少输出字节项导致长零区偏差”的解释，暂不支持据此添加统一D项；R18特有的部分零值/边界活动更值得优先核对，但本批没有直接识别其机制，也不能排除输出请求的贡献。

原式也没有解释全部配对：cfg_a dyadic/random在10/10对中均有正频率响应误差，中位 **+8.634%/+12.775%**，范围分别+6.369%至+10.430%、+9.041%至+18.291%。其余七组中位误差为−3.453%至+0.397%；cfg_c random的K4096绝对残差+6.727%主要延续K1024的+7.222%配置偏差，配对响应误差仅−0.536%。因此保留cfg_a非零输入的跨K/几何/输入足迹迁移缺口，不将它归成单一D功耗。C和f统计作用域不同，也不强求二者比值重构完整event时间。

结果见 [C-20261009-equal-work-clock-v1](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-equal-work-job738459/reanalysis/C-20261009-equal-work-clock-v1/)：`replay/`保留独立回放，`equal-work-clock.json`保存九条桥接、九组完整分布/方向计数、Q/S核对和冻结参数身份，`paired-trials.csv`保存全部90对的时间/C/f及进程间隔。仅新增离线分析与本节记录，没有新函数族、自由参数或补测。

```bash
# 在仓库根目录运行；OUT必须是新的C目录。
ROOT=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
RUN="$ROOT/20261009-R09-equal-work-job738459"
OUT="$RUN/reanalysis/C-equal-work-replay-<新后缀>"
python3 microbench/gh200_resource_campaign/access_rules/r09_analyze.py --shared --input "$RUN" --output "$OUT/replay"
python3 microbench/gh200_resource_campaign/access_rules/r09_equal_work_clock.py --run "$RUN" --reference-run "$ROOT/20261009-R09-input-clock-calibration-job738376" --clock "$ROOT/20261009-R09-input-clock-calibration-job738376/reanalysis/C-20261009-source-clock-dev-v2/source-clock.json" --output "$OUT"
```


<a id="r18-zero-activity-mixture"></a>

### R18八条件：固定零乘积比例的活动混合诊断

2026-10-09，仅离线，复用a057/GPU-43269fbc的M/N输入图批次job738296/738307。软件swizzle=8调度的八例均为 **768个名义输出tile，其中192个乘积必零，z=0.25**。按每tile×Kt×512 cycle加权，K1024/K4096的名义总周期分别为6291456/25165824，必零周期为1572864/6291456；z来自 `zero_m/zero_n` 整tile边界的OR判定，不由时间或频率拟合。这里计入调度的OOB补齐工作，不是只统计有效输出面积。双路径实际dual work与软件逐CTA序列一致，OOB/address_zero的z、Q、S严格相同。

冻结原source-clock的a=1.98、τ=200.678及Q/S定义，仅在dyadic评价槽使用 `d_mix=(1−z)d_dyadic+z d_zero`、`e_mix=(1−z)e_dyadic+z e_zero`。所得 **d_mix=0.1968177802、e_mix=0.01305147281 GHz·µs/KiB**，原模型和端点系数不改。K1024的Q/S为47662.545 cycle/SM、2978.909 KiB/SM，K4096为190650.182、11915.636；没有随z缩减名义Q/S。所有预测只在观测dual W下计算，未做自由组合或时间预测。基线也是观测W下重算，不能与根侧联合自由预测约12%的频率误差直接混用。

| 方向 / K / 输入路径 | 观测W µs | 观测f GHz | 原dyadic残差 | 固定z混合残差 |
|---|---:|---:|---:|---:|
| M / 1024 / oob | 46.720 | 1.703707 | -2.831% | -0.837% |
| M / 1024 / address_zero | 37.888 | 1.671817 | -3.962% | -1.822% |
| M / 4096 / oob | 168.544 | 1.616803 | -15.909% | -9.764% |
| M / 4096 / address_zero | 138.704 | 1.506955 | -14.210% | -7.247% |
| N / 1024 / oob | 46.624 | 1.715002 | -3.496% | -1.515% |
| N / 1024 / address_zero | 37.776 | 1.672525 | -4.050% | -1.908% |
| N / 4096 / oob | 168.720 | 1.618168 | -15.959% | -9.821% |
| N / 4096 / address_zero | 137.328 | 1.525064 | -15.466% | -8.568% |

八例全体保留，条件频率RMS从 **11.187%降至6.400%**；K1024四例为3.617%→1.578%，K4096四例为 **15.402%→8.912%**。八例绝对残差均缩小，但K4096仍低估7.247%–9.821%，故这一零新参数假设**只能解释部分偏差，尚不足以解决K4096低估**。

同z双路径仍有不同W：address_zero相对OOB，M/N的K1024窗口分别缩短18.904%/18.977%，K4096缩短17.705%/18.606%。混合式没有路径参数，频率差仅来自W；其address_zero/OOB频率变化与实测并列如下。

| 方向 / K | 实测频率变化 | 原dyadic预测变化 | 固定z混合预测变化 |
|---|---:|---:|---:|
| M / 1024 | -1.872% | -3.015% | -2.846% |
| M / 4096 | -6.794% | -4.911% | -4.194% |
| N / 1024 | -2.477% | -3.036% | -2.866% |
| N / 4096 | -5.754% | -5.201% | -4.444% |

K4096中，混合使共同频率上移，却没有完整解释路径对比：M方向实测下降6.794%，混合只下降4.194%；N方向为5.754%与4.444%。R18必零乘积区只保证一个操作数面板为零，而R09 zero端点将两输入全部置零；把e也按乘积零比例混合是未验证的活动假设，不能解释成物理源功耗。全卡平均z还没有表示逐CTA零工作分布。当前保留剩余误差，不拟合z、不增加系数/候选或补测，也不将这次改善写成新的完整模型通过。

[C-20261009-zero-activity-clock-v1](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R18-input-map-job738296/reanalysis/C-20261009-zero-activity-clock-v1/)保存全部八例、80个逐dual进程和四组路径响应。320个plain/wide/stamped/dual进程共1310720个抽检值独立回放通过，原始SHA及软件work核对通过；观测W/f与根侧 `manager-joint-clock-supply-v4/composition.json` 一致。`zero-activity-clock.json`记录原模型指纹、完整逐CTA零工作数和未验证假设，`cases.csv/processes.csv/paths.csv`并列保留原dyadic与混合结果，原数据和模型文件不改。

```bash
ROOT=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
python3 microbench/gh200_resource_campaign/access_rules/r09_zero_activity_clock.py --m-run "$ROOT/20261009-R18-input-map-job738296" --n-run "$ROOT/20261009-R18-input-map-n-job738307" --clock "$ROOT/20261009-R09-input-clock-calibration-job738376/reanalysis/C-20261009-source-clock-dev-v2/source-clock.json" --composition "$ROOT/20261009-R10-b-coverage-job738203/reanalysis/manager-joint-clock-supply-v4/composition.json" --output "$ROOT/20261009-R18-input-map-job738296/reanalysis/C-zero-activity-replay-<新后缀>"
```


### V09 的长 K 迁移（2026-10-09）

[V09](V09-component-validation.md) 已按获批r2预测在同卡完成新留出。cfg_b的6144×8192×32768自由有效频率为0.9000 GHz，同调用观测约1.2000 GHz，低估25.00%；总时间高估35.46%。仅作测后诊断，把观测频率代入其余原参数后，误差降至1.71%。这定位到时钟候选的跨K迁移，不证明某个物理功耗来源，也不允许改写冻结成绩。全组预填误差仍大，下一步复用现有长窗口、等工作和V09事件记录，分别检查频率代理与首段端点；当前不新增GPU任务。


<a id="v09-frequency-domain"></a>

### V09复核后的拟合口径与外推范围

[误差抵消复核](V09-component-validation.md#cycle-frequency-cancellation)提示应分别报告局部cycle/ns、有效频率和完整时间误差；换算周期与频率共享数学项，其相关性不能证明误差在训练时被吸收。当前source-clock已经拟合观测f，Q/S不依赖旧周期模型；下一步对照观测窗口下的频率拟合与自由组合结果，再检查局部服务模型，首段仍按原事件端点单独分析。

冻结前同卡[21点训练数据](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-input-clock-calibration-job738376/reanalysis/C-20261009-source-clock-dev-v2/source-clock.json)包含cfg_b 20480²×1024：dyadic为0.968796 GHz、1843.840 µs；random为0.688303 GHz、2546.928 µs。0.90 GHz低于dyadic训练范围，却不低于所有历史实测；5–7 ms超过训练窗口，但相对完整时钟训练集不是长十几倍。

后续按配置、输入模式、窗口和工作量说明内插/外推，范围只由测前可用证据确定。超范围不等于物理不可能，不把观测最小值当硬件下限，也不夹取频率改善分数。参数非唯一时可研究有条件区间，但须传播到频率联立与全时间；不改变V09的冻结拒绝项和评分。
