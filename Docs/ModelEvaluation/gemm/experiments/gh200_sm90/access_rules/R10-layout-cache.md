# R10：A/B 行距、地址覆盖与执行规模

**行距响应依赖配置、几何和输出位置，现有结果不支持通用的 32 B 或 128 B 对齐罚项。** 旧 job738203 的 B32/B64 虽满足 32 B 对齐仍变慢；V09 另一几何下 B32 未见明显惩罚，二者同时保留。地址覆盖只是软件需求，不是物理流量。有效供给与 stage 关系见 [R13](R13-async-retirement.md)，OOB/显式零的边界路径见 [R18](R18-cluster-boundary.md)；现行公式由 [RULES](RULES.md#v09-model)维护。

## 作业与证据索引

| 作业 | 固定对象 | 结果入口 |
|---|---|---|
| 735876 | 三配置，2600×3000×2000，B 行距 6000/6016/6144 B | [v2 完整时间](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v2/analysis-formal/summary.json)、[v4 cfg_b 局部周期](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/analysis-formal/summary.json)、[独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reviews/ROOT-independent.md) |
| 737322，V08F | A/B/D 行距、K/N 尾部；84 个解释性条件 | [原汇总](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/followup-v1/followup.json)、[供给残差](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/A-20261009-supply-v3/summary.json) |
| 738101 | cfg_a、2304×3072×4096，grid 32/64/96/132 × 三种行距 | [独立回放](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-sm-job738101/reanalysis/manager-formal-replay/summary.json) |
| 738169 | cfg_b h03 四角，2400×3000×1000，同容量 | [逐 trial 联合项与扰动](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-joint-pitch-job738169/reanalysis/B-20261009-matched-joint-final/joint-pitch.json) |
| 738203 | cfg_b B 行距五点，2304×3072×1024，局部冻结检验 | [冻结](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/frozen/r10-b-pitch.json)、[原评分](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/b-pitch-score-v1/b-pitch-score.json)、[位置与扰动](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/B-20261009-frozen-negative/localization.json) |
| 739011，V09 | 新几何 A16/B32/B80/B112 与 stage4 | [完整留出](V09-component-validation.md)、[逐例复核](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/manager-results-v1/cases.csv) |

## 原 B 行距对照：配置之间不能共用一个罚项

固定 M×N×K=2600×3000×2000，FP16 输入、FP32 累加/输出，α=1、β=0；A 的 lda=2000、D 的 ldd=3000，只改 B 的 ldb。job735876，a053/GPU-54896349，CUDA12.9、sm_90a、CUTLASS3.9.2。每点十个独立 plain 进程。

| 配置 | tile / schedule / cluster / stage | ldb3000 µs | ldb3008 µs | ldb3072 µs |
|---|---|---:|---:|---:|
| cfg_a | 128×128×64 / cooperative / 2×1 / 6 | 66.400 | 62.944 | 63.328 |
| cfg_b | 128×128×64 / pingpong / 1×1 / 6 | 76.272 | 58.272 | 57.136 |
| cfg_c | 256×128×64 / cooperative / 1×2 / 4 | 54.592 | 54.544 | 54.016 |

实际字节行距为 6000/6016/6144 B。cfg_b 从 ldb3000 到3008 减少 18.000 µs，cfg_a 较小，cfg_c 几乎不变；不能把差值单独命名为 TMA 等待，也不支持所有配置统一需要 128 B 行距对齐。

v2 全网格 trace 的 cfg_b3008/3072 有负向扰动超过 5%，不采用其周期；cfg_a/c 七点合格。最终 cfg_b 用 v4 对称准备、CTA0–3/consumer0 的 last-tile 局部 trace：

| ldb | v4 plain µs | 最大绝对配对扰动 | 周期/Ktile：局部池化中位 | 周期/Ktile：先按进程中位再汇总 |
|---:|---:|---:|---:|---:|
| 3000 | 76.160 | 2.068% | 854.531 | 869.906 |
| 3008 | 57.904 | 2.913% | 671.359 | 671.492 |
| 3072 | 57.456 | 3.775% | 659.047 | 662.812 |

局部周期与同档完整时间方向一致，但不替代全 grid 分布，也不与 v2 池化。v3 的3072仍有6.202%扰动，仅作诊断。完整时间包含 warm GEMM 和输出；trace 从首个 full-barrier 返回后至 mma_tail 完成，除以32个 Ktile，包含等待/循环/排空，首次输入等待在窗外。

实际 stride 同时用于初始化和 tensor map，B padding 填65504且计时外完整检查。v2/v3/v4 共300进程、1228800个保存值经未参与实现者逐K整数参考复算，误差0；数值见证不声明任意输入误差标准。[独立证据清单](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reviews/ROOT-independent.json)。历史准备、v1–v4检查及命令见[固定旧正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R10-layout-cache.md#时间抽样与数值边界)。

<a id="v08-stride"></a>

## V08F：分开行距和尺寸尾部

job737322，a043/GPU-099dda56，CUDA12.9.41、CUTLASS3.9.2、sm_90a、NDEBUG。V08 留出后追加84条件，plain/stamped/ends各十进程，属于开发解释，不回填 V08 判定；不与上文不同卡数据池化。参照 M=2304、N=3072，每次只改一项。

| 改动 | cfg_a 完整时间变化 | cfg_b | cfg_c |
|---|---:|---:|---:|
| K=1000，A 行距仍2048 B | +0.3% | −0.6% | −0.7% |
| N=3000，B/D 行距仍对齐 | −0.1% | +4.7% | −1.2% |
| D 行距12320 B | +0.1% | +0.2% | +1.9% |
| A +16 B，K1024 / K4096 | +7.7% / +8.7% | +15.0% / +16.4% | +0.2% / +0.3% |
| B +16 B，K1024 / K4096 | +4.7% / +6.2% | +28.2% / +33.7% | −0.4% / +0.8% |
| A 行距2000 B＋K尾部 | +7.3% | +17.1% | +0.1% |
| B/D行距6000/12000 B＋N尾部 | +2.7% | +31.5% | −0.2% |

这组数据里尾部本身和 D 行距影响较小，A/B 行距显著影响 cfg_b、其次 cfg_a，cfg_c 普通 tile 接近计算下界。这里的尾部是 M/N/K 不整除 tile，不是 epilogue 或最后完成 CTA。

同 cfg/同 K 对齐参照下，后续主循环 L 的进程内均值再跨进程中位变化如下；first 是每 CTA 首输出 tile，另列：

| 单轴条件 | cfg_a | cfg_b | cfg_c |
|---|---:|---:|---:|
| A16，K1024 | +2.00% | +12.88% | −0.03% |
| A16，K4096 | +6.00% | +13.99% | −0.004% |
| B16，K1024 | +0.74% | +36.11% | +0.01% |
| B16，K4096 | +6.12% | +33.74% | +0.001% |

微小正负值只表示当前精度内未分辨。cfg_a 的 K 依赖同时改变输入工作集，不能归为固定请求放大；cfg_b 的 B16 代价比 A16 大，不能共用操作数单价。

<a id="v08-stride-next"></a>

### 配置与位置反例

K4096 对齐/A16/B16 的 L/Ktile 为 cfg_a **548.933/581.891/582.535**，cfg_b **546.456/622.912/730.853**，cfg_c **1029.457/1029.420/1029.472**。cfg_a/b 计算项相同、逻辑源需求却为24/32 KiB；把 cfg_a 超额全解释成同一供给单价会把 cfg_b 推到约724.49 cycle/Ktile，与观测不符。共享单价缺少条件，不能从该矛盾选择某个 bank 机制。

cfg_a K4096 的位置分组更直接：

| 行距 | j1，132 CTA | j2，132 CTA | j3，36 CTA |
|---|---:|---:|---:|
| 对齐 | 579.27 | 526.39 | 515.77 |
| A16 | 629.96 | 553.29 | 515.78 |
| B16 | 627.00 | 557.61 | 515.77 |

单位 cycle/Ktile，逐进程对该位置求均值再取中位。最后一组已接近计算下界；j3 的 CTA 集合也不同，表中 CTA 数不是实际并发 SM 数。旧拟议 A/B64 等单轴点没有据此全部执行；详细候选与失败见 [R13](R13-async-retirement.md#v08-supply-fit)和[旧供给分析](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/A-20261009-supply-v3/summary.json)。

<a id="requested-sm-pitch"></a>

## 固定形状改变持久 grid：行距效应随工作位置变化

738101，a058/GPU-ef8692f3；cfg_a、2304×3072×4096、6 stages、sw1、dyadic seed17、重复同一 A/B。12条件为请求 SM 数32/64/96/132 × aligned/A16/B16，共240成功进程，983040个保存值通过回放。最大进程CV1.328%，完整时间打点变化最大2.343%；它不证明各局部区间无扰动。输入逻辑42 MiB、输出27 MiB不是 L2 驻留量。

| 请求/峰值 SM | CTA 总输出数 T | 末轮 CTA 数 | CTA 窗口平均重叠 | 包络 cycle/ns |
|---:|---|---:|---:|---:|
| 32 | 16 CTA×13；16 CTA×14 | 16 | 30.64–30.69 | 1.8038–1.8055 |
| 64 | 16 CTA×6；48 CTA×7 | 48 | 58.98–60.95 | 1.7193–1.7262 |
| 96 | 48 CTA×4；48 CTA×5 | 48 | 81.74–84.95 | 1.6717–1.6802 |
| 132 | 96 CTA×3；36 CTA×4 | 36 | 104.43–105.13 | 1.6647–1.6699 |

窗口重叠由同 SM CTA 生命周期求并集后计算，不是并发 TMA 数；包络 cycle/ns 也不是逐 tile 频率。grid 同时改变工作轮次、访问历史和频率，因此主比较为同档内按 CTA/j/T/坐标匹配的行距差。

| 请求 SM | j1 的两个 T | A16−aligned，cycle/Ktile | B16−aligned，cycle/Ktile |
|---:|---|---:|---:|
| 32 | 13 / 14 | −1.93 / −1.95 | −0.03 / −0.17 |
| 64 | 6 / 7 | +12.48 / +15.04 | +9.10 / +12.31 |
| 96 | 4 / 5 | +1.48 / +13.29 | +4.04 / +16.12 |
| 132 | 3 / 4 | +64.22 / +70.13 | +49.20 / +52.30 |

132档、T4 的 A16 在 j1/j2/j3 为 **70.13/17.43/0.009**，B16 为 **52.30/24.14/0.019 cycle/Ktile**。相同行距、K和总输出数仍非固定罚项。

96/132档在 T4、j1和j2各有24个相同当前 M/N 坐标。差分 `(variant−aligned)_132−(variant−aligned)_96` 的 A16 j1/j2中位为 **73.85/18.91**，B16为 **48.19/23.99 cycle/Ktile**；十次范围分别48.30～80.19、13.46～22.02、40.75～62.02、16.28～30.43。此前坐标和其他 CTA 访问历史仍不同，不能唯一定位 HBM/L2/SM 本地路径。候选拟合与低压力训练秩亏见 [R13](R13-async-retirement.md#requested-sm-pitch)。

## h03 联合行距：正交互不等于物理请求相互放大

738169，a043/GPU-099；cfg_b、2400×3000×1000、sw1，D ldd=3000固定，A/B容量固定为2400×1024和1000×3072 FP16元素；四角的实际 lda/ldb 为1024或1000、3072或3000，数值、尾部和grid一致。四条件×plain/wide/stamped/dual×十进程，655360个保存值通过参考回放。前一版738162按实际stride分配，容量不匹配，只保留[诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-joint-pitch-job738162/reanalysis/allocation-diagnostic-v2/joint-pitch.json)。

| A/B 条件 | plain µs | wide µs | dual µs | dual later cycle/Ktile | dual later ns/Ktile |
|---|---:|---:|---:|---:|---:|
| 对齐/对齐 | 29.888 | 29.840 | 30.960 | 566.379 | 334.935 |
| 未对齐/对齐 | 36.160 | 36.128 | 36.496 | 717.251 | 415.515 |
| 对齐/未对齐 | 37.920 | 38.048 | 38.736 | 779.464 | 447.485 |
| 未对齐/未对齐 | 46.672 | 46.832 | 47.216 | 998.478 | 561.861 |

后续窗口每进程先均值并除16，再跨十进程中位。wide/plain中位为−0.161%～+0.343%；dual相对同缓冲wide中位为+0.820%～+3.753%，但对齐最大单对+8.073%、仅B未对齐+5.378%，不能称所有trace通过5%。分项只属于dual观察窗口。

逐trial先算 `both−A_only−B_only+aligned`，十次均为正：plain中位 **2.528 µs**（范围1.344～3.008），dual later中位 **69.1169 cycle/Ktile、32.5586 ns/Ktile**。不能对四条件各取中位后再把差称为联合项中位。

trial0可由没有 A×B 交互项的计算/供给 max 解释：对齐被335.0556 ns计算锚点截住，隐含线性供给基线299.1235 ns，加A/B分别115.6296/146.7778 ns即可重现四点，正联合项恰为335.0556−299.1235=35.9321 ns。它是四点的可行事后解释，不是物理请求放大或新预测验证。全trial及实际坐标见[配对结果](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-joint-pitch-job738169/reanalysis/B-20261009-matched-joint-final/joint-pitch.json)。

<a id="bcurve-negative"></a>
<a id="局部冻结负结果job738203"></a>

## B 覆盖的冻结检验：两候选均失败

738203，a057/GPU-43269fbc；cfg_b、2304×3072×1024、A/D对齐、6 stages、sw1。五点共同分配 `alloc_lda=1024, alloc_ldb=3136`，仅B行距改变；每点四变体各十进程，共200成功进程、819200个保存值通过回放。plain/wide/stamped/dual均168寄存器、16条静态HGMMA、无spill/C7510。

| B增量 | ldb | 用途 | 额外32 B覆盖行数 | 额外128 B覆盖行数 | dual later ns/Ktile |
|---|---:|---|---:|---:|---:|
| 0 B | 3072 | 校准 | 0 | 0 | 329.9967 |
| 16 B | 3080 | 校准 | 32 | 56 | 439.1767 |
| 32 B | 3088 | 校准 | 0 | 48 | 385.9533 |
| 64 B | 3104 | 留出 | 0 | 32 | 365.7233 |
| 128 B | 3136 | 留出 | 0 | 0 | 331.7933 |

覆盖按64行×256 B B-box的有效地址跨度计算，不是物理事务数。目标为每trial后续 j≥1 主循环直接ns均值/16，再取十trial中位；首tile、完整时间和实测频率不进入目标。两候选分别使用32/128 B覆盖，形式为历史冻结的 `max(C,q0+q1*x)`，C只由对齐校准给出。只读p0/p16/p32后冻结，再采p64/p128；要求每候选的三个校准和两个留出最大绝对误差均≤5%。参数不唯一时按完整预测区间最差端点评分。

| 候选 | 校准最大误差 | p64预测/实测 ns | p64误差 | p128预测/实测 ns | p128误差 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 32 B覆盖 | 14.4983% | 329.9967 / 365.7233 | −9.7688% | 329.9967 / 331.7933 | −0.5415% | 失败 |
| 128 B覆盖 | 0% | 329.9967 / 365.7233 | −9.7688% | 329.9967 / 331.7933 | −0.5415% | 失败 |

32 B参数非唯一，但两候选留出预测均唯一，不能测后选择有利解。p128接近锚点不改变整组失败；被否定的是该冻结形式，不是识别出物理cache line。冻结SHA256 `6fd226ef90f1a9d9224d1d6db769744eaece3befd69e98abf4352c4676d072c5`保持原样；时序、身份和原只读位见[独立评分](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/B-20261009-frozen-negative/b-pitch-score.json)。

dual/wide完整时间中位扰动p0/p16/p32/p64/p128为+4.860%/+2.381%/+1.885%/+3.006%/+4.997%，单对仍有超5%，p0/p128最大+8.037%/+8.945%。所有有效trial保留，局部冻结只验证约定dual窗口。

### p64 失败位于多个后续输出位置

工作列表、CTA/j/T/坐标匹配后，每trial先取组均值再作p64−p0差；两个采样阶段不是同时调用。

| j/T | 每进程窗口数 | cycle/Ktile增量 | ns/Ktile增量 | 十trial ns范围 |
|---|---:|---:|---:|---:|
| 1/3 | 96 | 66.3281 | 37.2396 | 32.8333～42.1458 |
| 1/4 | 36 | 64.0026 | 35.8611 | 26.1111～40.3333 |
| 2/3 | 96 | 70.1533 | 37.7396 | 33.7708～40.8333 |
| 2/4 | 36 | 88.5694 | 47.2222 | 39.6667～61.6667 |
| 3/4 | 36 | 17.6771 | 9.4167 | 4.0556～12.6111 |

五组每trial的cycle/ns差均为正；j1/j2约贡献后续总增量的96.8%，不是只有最后tile变慢。p128同阶段接近p0；位置分组不足以命名物理请求放大、带宽下降或某个等待机制。

<a id="coverage-identifiability"></a>

## 覆盖组合的可辨识性与失败

以下均为测后开发，未改变上述冻结失败。[组合报告](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-composed-coverage-v5/summary.json)分别定值每张卡、每种观察协议，保存参数范围和逐窗口残差。

同时使用32/128 B覆盖时，原p0/p16/p32只提供两个独立服务等式，三个非负服务参数不能唯一确定；p64预测范围为 **329.9967～367.3011 ns/Ktile**，实测365.7233在其中。揭晓p64后能选定系数，只是已有可行集选点，不追认旧预测通过。已见0/+16 B的完整box恒有extra128=7×extra32；小拟合误差不能据此支持新覆盖比例。

| 数据 | 后续目标 | A/B共用覆盖单价最大误差 | A/B分开最大误差 | 活跃秩/参数数：共用；分开 |
|---|---|---:|---:|---|
| 432卡 B曲线 dual | 25组 | 16.33% | 16.33% | 4/4；4/6 |
| ef卡 cfg_a SM扫描 stamped | 144组 | 4.80% | 3.78% | 3/4；4/6 |
| 099卡 h03 dual | 20组 | 30.63% | 19.22% | 4/4；5/6 |
| 099卡 V08F stamped | 51个聚合窗口 | 18.14% | 8.66% | 5/6；6/8 |

h03 j1/T3 的 B-only 覆盖少于 A-only，实测却为833.682>700.956 cycle/Ktile，共用非负单价模型至少有8.65%最大相对误差；V08F对应下界9.33%。分开A/B后仍未消除位置差异。h03 aligned j3/T4比j2/T3更慢的反例还混入B部分OOB：前者60个窗口中19个含N尾部，后者没有；几何分组和填零解释见 [R18](R18-cluster-boundary.md#h03-partial-fill)，不能只据有效地址少而推定服务更快。

<a id="v09-pitch-comparison"></a>

## 旧 B32/B64 与 V09 B32：先比较几何，再谈对齐

两批都在a057/GPU-432，cfg_b、六stage、sw1、dyadic seed17，且各自组内固定容量、输入值与D行距；**跨批 M/N/K、总输出tile和波次数并不相同**：

| 条件 | 738203 B曲线 | V09 G5 |
|---|---|---|
| M×N×K | 2304×3072×1024 | 2304×2304×4096 |
| Ktile数 | 16 | 64 |
| 输出tile/grid | 432 / 132 CTA | 324 / 132 CTA |
| 软件工作轮次 | 96 CTA×3；36 CTA×4 | 72 CTA×2；60 CTA×3 |
| 对齐B行距 | 6144 B | 4608 B |
| B32实际行距 | 6176 B | 4640 B |
| B预留ldb | 3136 FP16元素 | 2360 FP16元素 |
| plain基线→B32 | 28.704→33.008 µs，+14.99% | 69.536→68.992 µs，−0.78% |
| B64 | 6208 B；31.440 µs，+9.53% | 未测 |

波次数由实际tile/grid分配计数，是软件工作组织，不是瞬时并发。旧B32/B64的dual later窗口也分别慢16.96%/10.83%；V09 B32的−0.78%只表示未见明显惩罚，不判为显著提速。这些结果否定“输入只要32 B对齐就不会慢”，却不能单凭跨几何对照认定32 B因素发生因果翻转。

V09 G3的cfg_b A16在2816×1536×4096下实测慢28.94%，模型只预计1.40%；G5 B32预计慢6.53%，实测未出现。两项说明当前条件供给迁移失败，不是改一个布尔对齐门槛就已解决。[EXP-15/R05-E](../EXP-15-tma-2d.md#r05-e)的32 B结果针对TMA输出，不直接迁入A/B输入。stage4响应归 [R13](R13-async-retirement.md)。

下一步先复用两批逐CTA/轮次记录比较请求组织与窗口位置；只有现有记录仍不能区分32/128 B候选，才在V09几何内选B64/B96或A32的最少配对。原冻结与V09判定保持，不能用测后重拟合消除负结果。

## 历史推导与复核

行距入口为 `run_r10.py`，joint/coverage重算为 `analyze_r10.py --joint-pitch`、`--score-b-pitch`，grid扫描为 `run_r13_sm.py` / `analyze_r13_sm.py`。已有reanalysis优先使用上表链接；采样前冻结算法、完整命令与准备过程见[固定 cee978a 旧正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R10-layout-cache.md)。历史参数只用于复核旧判定，现行规则不在本页复制。
