# V04：规则模型迁移到其他 CUTLASS 配置（先冻结、后测量）

[总计划](README.md)。2026-10-07，Slurm 735773，romeo-a054，GPU-572de9c0…（与 V02 的 GPU-3953fe72、R09 的 GPU-e403ae62 都不是同一块卡），驱动 590.48.01，CUDA 12.9.41 / sm_90a，CUTLASS v3.9.2，`-DNDEBUG`。同节点另一块 GPU 上同时运行着另一个作业（735775），本组未与其交互。

## 问题

[V02](V02-kernel-prediction.md) 的模型只针对 128×256×64 cooperative、cluster 2×1 一个配置。按明确的迁移规则，它能否在测量前预测另外三个配置的单次调用时间？如果预测失败，是哪一条假设出了问题？

## 配置与资源

三个配置都是 FP16→FP32、A 行主序（K-major）、B 行主序（MN-major）、`ElementC=void`、默认持久调度器（raster Heuristic、swizzle 1）、`-DNDEBUG`。资源数据取自 ptxas 日志、`cuobjdump` SASS 和探针的 `setup` 模式（`cudaOccupancyMaxActiveBlocksPerMultiprocessor`）。

| 名称 | tile | kernel 调度 | cluster | epilogue tile | stage | SMEM B | 线程 | 寄存器 | CTA/SM | HGMMA（形状） |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 基准（V02） | 128×256×64 | cooperative | 2×1 | 128×32 | 4 | 231424 | 384 | 168 | 1 | 8（64×256×16） |
| cfg_a | 128×128×64 | cooperative | 2×1 | 128×32 | 6（auto） | 231424 | 384 | 168 | 1 | 8（64×128×16） |
| cfg_b | 128×128×64 | pingpong | 1×1 | 64×32 | 6（auto） | 215040 | 384 | 168 | 1 | 16（64×128×16） |
| cfg_c | 256×128×64 | cooperative | 1×2 | 128×32 | 4 | 231424 | 384 | 168 | 1 | 16（64×128×16） |

- 6 个构建（3 个配置 × 无打点/打点）都没有 C7510，都没有 spill，每个都有 2 条 `WARPGROUP.DEPBAR`；打点版与无打点版的 HGMMA 数相同。
- pingpong 中每个消费者 warpgroup 独立计算一整个 128×128 tile。128×256 的 pingpong 每线程需要 256 个累加寄存器，不可行，因此 cfg_b 选用 128×128。这样 cfg_a 与 cfg_b 的 tile 相同，只有调度和 cluster 不同。
- 持久 grid 固定为 132 个 CTA（CUTLASS 默认的 `sm_count`）。实测每次调用中 CTA 所在的 SM 都互不相同。

## 迁移假设（冻结时）

基准常数全部取自 V02 冻结文件 `configs/v02-predictions.json` 的 `rules`：\(c_k=1024.03\)，\(c_0=186.3\)，\(P=4172.5\)，\(E=5448\)，\(h,s,p,\tau\)，以及频率规则 \(f=2.0087-0.08496\ln W\)。冻结时 V03 文档还不存在，因此沿用 V02 的频率规则。

| 编号 | 项 | 规则 | 来源 |
|---|---|---|---|
| T1 | 主循环计算 | \(c_k'=c_k\cdot F'/F_0\)，\(F=2\,T_MT_N\cdot64\)。cfg_a/b 为 512.0，cfg_c 为 1024.0 cycle/Ktile | R00-B：`m64n{64,128,256}k16`，1 或 2 个 warpgroup 都达到 4096 FLOP/cycle；R09 测得基准为 1024.03 |
| T2 | 供给 | 每个 SM 每 Ktile 写入 SMEM 的字节 \(B=(T_M+T_N)\cdot128\)（多播不减少每个 SM 收到的量）。若 \(B/c_k'>55.45\) B/cycle，主循环取 \(B/55.45\)。cfg_a/b 需要 64 B/cycle，判为**供给受限，590.9 cycle/Ktile**；cfg_c 需要 48，判为计算受限 | R05-D：1 CTA/SM、共享小源，55.45 B/cycle |
| T2′（不计分） | 供给变体 | 改用多播后的 L2 请求字节：cfg_a 为 24 KiB/Ktile，即 48 B/cycle，判为计算受限 | 同上 |
| T3 | 每 tile 主循环常数 | \(c_0'=c_0-0.62\,(256-N)\)，N=128 时为 106.9 | R01：WGMMA+wait0 ≈ 36+0.62N |
| T4 | 每 tile 主循环以外的成本 | \(E'=E\cdot\) 每 tile 输出字节 / 128 KiB。cfg_a/b 为 2724，cfg_c 为 5448 | R09：E 与 epilogue 段相当（128 KiB 时 5470–5850 cycle）；R05-E：TMA store 按速率计 |
| T5 | 预填 | \(P'=P\,(0.567+0.433\cdot\) 一个 stage 的字节 / 48 KiB\()\)，其中生产者准备部分固定、首个 stage 到达部分按字节缩放。cfg_a/b 为 3570，cfg_c 为 4172 | R09 warm M=2048：`producer_setup_med/prefill_med` 的中位数为 0.567 |
| T6 | 轮数 | \(U\)：先把 \(t_m,t_n\) 分别补齐到 cluster 形状的倍数，再取 \(R=\lceil U/132\rceil\)；1 CTA/SM | R08；本组 `setup` 的占用为 1 |
| T7 | pingpong | 两个 warpgroup 交替处理 tile，主循环和 epilogue 各自由顺序屏障串行；第 i 个 tile 的 epilogue 与第 i+1 个 tile 的主循环重叠：\(C=P+R\max(ML,E')+\min(ML,E')\)，\(ML=c_k'K_t+c_0'\) | CUTLASS `sm90_gemm_tma_warpspecialized_pingpong.hpp`（`OrderedSequenceBarrier` 深度 2） |
| T8 | 频率与固定项 | \(f(W)\)、\(h,s,p,\tau\) 不变 | V02 |

## 冻结

- 先编译，再用 `setup` 模式读取静态资源（只调用占用率 API，不运行 GEMM）。之后生成 `configs/v04-predictions.json`：SHA256 `dffb69a4986c7e6cc1ce8d26c5877ce9fe365c1f7574624f0b04bca090f70d5f`，生成时间 **2026-10-07T08:08:46Z**，随即 `chmod 444`。`v04_predict.py`（SHA256 `572c653a…2466`）同时改为只读。
- 第一个计时样本（pilot）开始于 08:09:31Z，晚于冻结时间。节点上的副本与 `source/v04-predictions.json` 哈希一致。
- 22 个 (配置, 尺寸) 组合都没有测过；它们的尺寸见下表。

## 测量

- 计时协议与 V02 相同：同步预热到末 5 次 CV≤2%，再用 event 包围一次调用。每点无打点版与打点版各 10 个进程，打点版与无打点版相邻运行，每轮随机打乱顺序。测量值取无打点版 10 个进程的中位数。
- 打点版：7 类位置的 `clock64`/`globaltimer`，以及每个消费者 warpgroup 各自的 tile 计数（pingpong 的 CTA tile 数为两者之和）。调用内频率取关键 CTA（tile 数最多的 CTA）“入口→退出”的 cycle/ns 中位数。
- 446 个进程（pilot 6 个、正式 440 个）都做了 FP64 抽样检查（4096 点），误差全部为 0。打点版与无打点版相差 −1.3%～+3.1%。

## 结果

e = 预测/实测 − 1。“cycle e”是冻结的 C 与关键 CTA 实测 cycle 中位数的相对差；“频率 e”是预测频率与实测频率的相对差；“e（实测频率）”保留冻结的 C 和各固定项，只把频率换成实测值。“Ktile 实测”由末个 tile 的主循环扣除 \(c_0'\) 得到；“E 实测”为每 tile 周期减去末个 tile 的主循环。

| 配置 | 名称 | M×N×K | R | 预测 µs | 实测 µs | CV | e | cycle e | 频率 预测/实测 GHz | e（实测频率） | Ktile 预测/实测 | E 预测/实测 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| a | pad_fills_wave | 1408×1408×4096 | 1 | 30.2 | 33.7 | 1.2% | −10.5% | −4.3% | 1.734/1.716 | −9.7% | 591/609 | 2724/3610 |
| a | pad_crosses_wave | 1400×1536×4096 | 2 | 55.3 | 48.9 | 1.1% | +13.1% | +7.6% | 1.675/1.760 | +8.1% | 591/510 | 2724/4886 |
| a | just_over_wave | 2304×1920×3072 | 3 | 63.1 | 55.5 | 0.8% | +13.7% | +9.4% | 1.663/1.729 | +9.7% | 591/510 | 2724/3856 |
| a | long_k | 1024×2048×16384 | 1 | 102.0 | 98.0 | 0.6% | +4.1% | +14.0% | 1.620/1.497 | +12.3% | 591/512 | 2724/3536 |
| a | short_k | 4096×3584×512 | 7 | 37.7 | 38.0 | 1.1% | −0.9% | −2.6% | 1.712/1.699 | −0.2% | 591/502 | 2724/3653 |
| a | multi_wave | 6144²×2048 | 18 | 261.6 | 256.5 | 1.1% | +2.0% | +9.8% | 1.537/1.455 | +7.6% | 591/509 | 2724/3370 |
| a | large | 7168²×4096 | 24 | 677.4 | 719.5 | 0.5% | −5.9% | +11.6% | 1.456/1.247 | +9.8% | 591/511 | 2724/3636 |
| b | partial_wave | 1280×1536×4096 | 1 | 30.2 | 27.5 | 1.0% | +9.7% | +9.2% | 1.734/1.693 | +11.9% | 591/514 | — |
| b | just_over_wave | 1792×1280×3072 | 2 | 41.8 | 36.0 | 1.6% | +16.2% | +11.1% | 1.702/1.750 | +13.3% | 591/520 | — |
| b | long_k | 1024×2048×16384 | 1 | 102.0 | 95.0 | 0.6% | +7.4% | +13.2% | 1.620/1.547 | +12.2% | 591/513 | — |
| b | short_k | 4096×3584×512 | 7 | 27.7 | 32.5 | 1.1% | −14.8% | −17.9% | 1.742/1.677 | −12.1% | 591/573 | — |
| b | very_short_k | 4096²×128 | 8 | 19.7 | 26.7 | 1.5% | −26.4% | −30.3% | 1.779/1.736 | −25.0% | 591/581 | — |
| b | edges_odd | 2600×3000×2000 | 4 | 53.8 | 75.9 | 1.5% | −29.2% | −30.2% | 1.678/1.733 | −31.2% | 591/788 | — |
| b | multi_wave | 6144²×2048 | 18 | 229.8 | 253.1 | 1.1% | −9.2% | +11.0% | 1.549/1.291 | +8.5% | 591/527 | — |
| c | partial_wave | 1792×2304×4096 | 1 | 49.3 | 50.3 | 1.3% | −1.9% | −3.0% | 1.686/1.709 | −3.1% | 1024/1027 | 5448/7536 |
| c | pad_crosses_wave | 2560×1664×4096 | 2 | 94.7 | 90.3 | 0.7% | +4.8% | −0.8% | 1.626/1.709 | −0.0% | 1024/1027 | 5448/5898 |
| c | just_over_wave | 2560×1792×3072 | 2 | 73.6 | 70.3 | 0.6% | +4.7% | −2.2% | 1.649/1.737 | −0.3% | 1024/1027 | 5448/6552 |
| c | long_k | 1024×3840×16384 | 1 | 177.8 | 179.9 | 0.5% | −1.2% | −0.4% | 1.571/1.567 | −1.0% | 1024/1025 | 5448/6212 |
| c | short_k | 4096×3584×512 | 4 | 39.3 | 40.7 | 1.1% | −3.5% | −8.5% | 1.708/1.759 | −6.0% | 1024/1050 | 5448/6621 |
| c | edges_odd | 2600×3000×2000 | 2 | 52.8 | 54.0 | 1.2% | −2.2% | −3.5% | 1.680/1.694 | −2.9% | 1024/1028 | 5448/6794 |
| c | multi_wave | 6144²×2048 | 9 | 230.1 | 242.2 | 0.3% | −5.0% | −1.4% | 1.548/1.505 | −2.3% | 1024/1028 | 5448/5839 |
| c | large | 7168²×4096 | 12 | 588.9 | 622.6 | 0.5% | −5.4% | −0.5% | 1.468/1.413 | −1.8% | 1024/1027 | 5448/5643 |

| 配置 | \|e\| 中位 / 最大 | \|cycle e\| 中位 / 最大 | \|频率 e\| 中位 / 最大 | \|e（实测频率）\| 中位 / 最大 |
|---|---|---|---|---|
| cfg_a | 5.9% / 13.7% | 9.4% / 14.0% | 4.8% / 16.7% | 9.7% / 12.3% |
| cfg_b | 14.8% / 29.2% | 13.2% / 30.3% | 3.2% / 20.0% | 12.2% / 31.2% |
| cfg_c | 4.1% / 5.4% | 1.8% / 8.5% | 2.9% / 5.1% | 2.0% / 6.0% |
| 全部 22 点 | 5.6% / 29.2% | | | |

结构项全部预测正确：22 个点的轮数、每个 CTA 的 tile 数分布、关键 CTA 个数，以及 cfg_a/cfg_c 中由 cluster 补齐造成的多一轮（a pad_crosses_wave 实际 132 个 tile、补齐后 144；c pad_crosses_wave 实际 130、补齐后 140）都与预测一致，10 个进程之间也相同。

![预测、误差拆分与频率](../../../../../../results/gh200_resource_campaign/access_rules/20261007-v04-transfer/plots/v04_transfer.png)

## 误差归因

**cfg_c（256×128，cluster 1×2）：迁移成立。** 各项假设与实测一致：主循环 1025–1028 cycle/Ktile（Kt=8 时 1050）；E 不变（R≥9 时实测 5640–5840；R=1–4 时为 5900–7540，与 V02 一样，轮数少时 store 后等项摊得不够薄）；cluster 1×2 的 \(t_n\) 补齐规则正确。cycle 误差主要落在 −3.5%～−0.4%，只有 short_k 为 −8.5%，原因与 V02 short_k 相同：Kt 小时每 tile 实测额外约 1200 cycle。其余误差来自频率，模式与 V02 相同：最后一轮未满时频率比规则高（+5%），W 长时比规则低（−3～−4%）。

**cfg_a（128×128 cooperative，cluster 2×1）：时间误差中位数尚可，但这是两项相反的错误相互抵消的结果。**

1. **供给假设 T2 错误。** 主循环实测 502–512 cycle/Ktile，也就是计算下界，而不是供给受限时的 591。每个 SM 实际写入 SMEM 的速率为 32768/510 ≈ 64 B/cycle，高于 R05-D 的 55.45。不计分的变体 T2′（按多播后的 L2 请求量 48 B/cycle 计）在主循环上判断正确。但 cfg_b 没有多播，同样以约 64 B/cycle 跑满（见下），所以按 L2 请求量计也不是正确的规则。正确的结论是：R05-D 的 55 B/cycle 不是目标 GEMM 的供给上限。推断：R05-D 用 48 KiB stage 加不做计算的消费者，测到的是那个协议下的观测值，不是每个 SM 的端口上限。
2. **E 的缩放 T4 偏小。** R≥3 的尺寸每 tile 实测 3370–3860 cycle（R=1–2 时为 3540–4890），冻结值为 2724；末 tile 的 epilogue 段只有 1980–2220 cycle，大约是基准的一半，符合“按字节缩放”。多出的约 1500 cycle 不随输出量缩放，属于 tile 切换的固定成本。事后按两点（64 KiB 约 3640、128 KiB 约 5450）拆分：E ≈ 1830 + 1810 × (输出字节/64 KiB)。这只是事后拟合，没有计入预测。
3. **预填 T5 方向错误。** 单 tile CTA 实测预填为 4030–4480 cycle，不比 48 KiB stage 的基准短，不应按 stage 字节缩小。对总时间的影响 ≤1.5%。
4. **频率。** 长调用比规则低得多：7168²（704 µs）为 1.247 GHz，规则给 1.456（−14%）。同一张卡上 cfg_c 的 7168²（610 µs）为 1.413。推断：128×128 tile 每 FLOP 的 L2/片外流量更多，功率封顶下频率更低。这与 V02 推断的“频率与片外流量有关”同向。短调用中最后一轮未满时，频率比规则高 3～5%，与 V02 相同。
5. pad_fills_wave（1 轮，11 个 cluster 对处理的是整块越界的补齐 tile）主循环中位数为 609 cycle/Ktile，同一次调用内各 CTA 在 528–696 之间分散；long_k 全部为 512。原因未查明。

**cfg_b（128×128 pingpong，cluster 1×1）：迁移失败。** 失败的有三条，另有一个尺寸出现异常：

1. **T2 同样错误。** Kt≥32 时，主循环为 513–527 cycle/Ktile，计算受限。这里没有多播，每个 SM 从 L2 读取约 64 B/cycle，直接否定了 55.45 的上限。这部分使 partial_wave、just_over_wave、long_k 的 cycle 预测偏长 9–13%。
2. **pingpong 的重叠假设 T7 只在长 K 时成立。** Kt=32（multi_wave）时，每 tile 周期为 17103 cycle，只比 ML 多约 130，epilogue 完全被隐藏。Kt=8 时，周期为 6083，比 max(ML 实测约 4690, E) 多约 1390；Kt=2 时，周期为 4171，比 max(ML 约 1270, epilogue 实测约 3250) 多约 920。短 K 时，每个 warpgroup 自己的 “ML→EP→下一个 ML” 链和顺序屏障的交接无法完全并行，模型中没有这一项，因此 short_k、very_short_k 的 cycle 误差为 −18%、−30%。
3. **E 的缩放对 pingpong 只在不受干扰时成立。** 长 K 单轮时 epilogue 实测 2683–2780 cycle，与 2724 吻合；与另一个 warpgroup 的主循环并行时（short_k、multi_wave、edges_odd），升到 3250–5390。
4. **edges_odd 异常（推断，未隔离）。** 主循环为 788 cycle/Ktile（CTA 间 667–967）；cfg_c 在同一尺寸上为 1028（即计算下界），V02 基准在 N=3496 上也正常。这里 N=3000，B 为行主序，行距 6000 B，只按 16 B 对齐。cfg_b 每个 SM 需要从 B 读取 32 B/cycle（无多播），实际只拿到 16384/788 ≈ 21；cfg_c 与基准对 B 的需求只有约 16 B/cycle，所以没有受影响。这与 R05-E 中“行距只按 16 B 对齐时 TMA 变慢”的现象同向，但本组没有用对照实验隔离。
5. **频率。** multi_wave（244 µs）为 1.291 GHz，规则给 1.549（+20%）；这是 22 个点里偏差最大的一个。同一尺寸下 cfg_a 为 1.455、cfg_c 为 1.505。推断：pingpong 的 Tensor Core 占空比更高，加上无多播时 L2 请求更多，功率更高。

**汇总：哪些假设能迁移。**

| 假设 | cfg_a | cfg_b | cfg_c |
|---|---|---|---|
| T1 计算速率（4096 FLOP/cycle） | 成立（502–512 cycle/Ktile） | 成立（513–527） | 成立（1025–1028） |
| T2 供给上限 55 B/cycle | **失败**：实测达到 64 B/cycle | **失败**：64 B/cycle，且无多播；16 B 对齐的 B 在 edges_odd 上例外 | 未触发（48） |
| T4 E 按输出字节缩放 | **偏小**：有约 1.8k cycle 的固定部分 | 无干扰时成立，重叠时变大 | 成立（字节不变） |
| T5 预填按 stage 字节缩放 | **方向错**，实测不变 | 同左 | 成立 |
| T6 轮数与 cluster 补齐 | 成立 | 成立 | 成立 |
| T7 pingpong 重叠 | — | Kt≥32 成立，短 K **失败** | — |
| T8 频率规则 | 长调用偏高 9–17% | 长调用偏高 5–20% | 与 V02 相同（±5%） |

## 结论

- 结构部分（轮数、cluster 补齐、持久调度的 tile 分配）和 WGMMA 计算速率可以直接迁移，三个配置的 22 个点全部正确。
- 与基准每 tile 输入量、输出量、调度方式都相同、只换了 tile 方向和 cluster 方向的配置（cfg_c），模型可以原样迁移：|e| 中位数 4.1%，最大 5.4%；换用实测频率后误差在 ±6% 以内。
- 改变 tile 面积或调度方式后，有三条迁移规则不成立：R05-D 的 55 B/cycle 不能用作供给上限（实测达到 64）；每 tile 固定成本中有一部分不随输出量缩放；pingpong 在短 K 下只能部分重叠。cfg_a 的总误差（中位数 5.9%）看起来不大，是因为主循环偏长 16% 被 E 偏短大致抵消；cfg_b 的误差最大达 29%。
- 频率规则不能跨配置使用：对 128×128 tile 的长调用，实测频率比规则低 9–20%。按 V02 的推断，频率与片外/L2 流量有关；128×128 tile 的流量更大，偏差也更大。
- 要让模型跨配置可用，至少需要：在真实 GEMM 的 TMA 输入条件下（含多播与行距对齐）重新测量供给上限；把 E 拆成固定段与按字节计的 epilogue 段；为 pingpong 增加短 K 时的交接成本；在频率规则中加入流量或功率相关的项。

## 未解决

- 供给上限：R05-D 的 55 B/cycle 与本组的 64 B/cycle 之间的差异来自协议（stage 大小、消费者是否计算）还是来自源数据的复用，未隔离。
- cfg_b edges_odd 的 788 cycle/Ktile 是否来自 B 行距只按 16 B 对齐，需要用同尺寸、只改行距（pitch padding）的对照来确认。
- cfg_a pad_fills_wave 的主循环分散（528–696），以及越界补齐 tile 是否参与了 B 多播，未查。
- E 的固定段（约 1.8k cycle）由哪些部分组成（主循环→epilogue 切换、调度器取下一个 tile、生产者重新填充），本组的打点粒度不够拆分。
- pingpong 短 K 时的交接成本（约 900–1400 cycle/tile）的来源未拆分。
- 频率差异只是推断，本组没有记录调用内的功率或 DRAM 流量。另外，测量期间同节点另一块 GPU 上有作业（735775）在运行，模块之间是否相互影响未知。

## 数据与复现

- 结果目录：[20261007-v04-transfer](../../../../../../results/gh200_resource_campaign/access_rules/20261007-v04-transfer/)，包括 [cases.csv](../../../../../../results/gh200_resource_campaign/access_rules/20261007-v04-transfer/cases.csv)、[summary.json](../../../../../../results/gh200_resource_campaign/access_rules/20261007-v04-transfer/summary.json)、[图](../../../../../../results/gh200_resource_campaign/access_rules/20261007-v04-transfer/plots/v04_transfer.png)、`samples/<case>/main-NN/`（原始 stdout，含每个 CTA 的打点，gzip 压缩）、`samples.jsonl`、`build/`（编译命令、ptxas 日志、SASS、`sass_check.json`、`resources.json`、二进制哈希）、`source/`（探针、覆盖头文件、冻结预测副本、预测器、分析器）、`environment.json`、`nvidia-smi-*.txt`、`build.sh`、`sample.sh`，以及作业输出 `build.out`、`sample.out`。
- 代码：[v04_predict.py](../../../../../../microbench/gh200_resource_campaign/access_rules/v04_predict.py)、[v04_run.py](../../../../../../microbench/gh200_resource_campaign/access_rules/v04_run.py)、[v04_analyze.py](../../../../../../microbench/gh200_resource_campaign/access_rules/v04_analyze.py)、[probes/v04_probe.cu](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/v04_probe.cu)（由 v02_probe.cu 改写，用 `-DV04_CFG=0/1/2` 选择配置）。冻结预测：[configs/v04-predictions.json](../../../../../../microbench/gh200_resource_campaign/access_rules/configs/v04-predictions.json)。

```bash
python3 microbench/gh200_resource_campaign/access_rules/v04_run.py build --cutlass-root <CUTLASS> --output <RUN>
python3 microbench/gh200_resource_campaign/access_rules/v04_run.py setup --output <RUN>
python3 microbench/gh200_resource_campaign/access_rules/v04_predict.py --resources <RUN>/build/resources.json --output <预测.json>
chmod 444 <预测.json>
python3 microbench/gh200_resource_campaign/access_rules/v04_run.py sample --output <RUN> --predictions <预测.json> --set pilot   # 然后 --set main
python3 microbench/gh200_resource_campaign/access_rules/v04_analyze.py --input <RUN>
```
