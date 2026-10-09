# R18：cluster 边界、有效零数据与影响范围

24个plain条件均完成10进程采样并独立复算，1,966,080个抽样输出值正确。完整trace有22个条件合格；另外两个短K条件原扰动5.34%/5.47%，经减少重复坐标写出的轻量补查后通过。轻量组含8个条件（两组边界及普通对照、两个K），再检查655,360个值。24个代表条件现均有可用的plain和合格范围内的事件证据。

job737122、romeo-a043、GPU-099dda56-d7af-f60e-c285-aa2dc7ddfcfe，CUDA12.9.41、sm_90a、CUTLASS3.9.2、NDEBUG。该设备身份与本轮R16配额、R19相同。

## 对照与不变条件

cfg_a沿M测试：普通M=1536、整块越界M=1408、偶数部分边界M=1504、奇数部分边界M=1376；N=2048。cfg_c沿N使用同样四个边界长度，M=3072。K=1024/8192。显式零组保留1536的有效descriptor范围，最后128行A（cfg_a）或128列B（cfg_c）置零。

整块组的普通、越界、显式零数据具有相同物理tile列表、grid、stride和分配容量；cfg_a保持6 stage、cfg_c保持4 stage，编译无spill、无C7510。两种输入仍为FP16，FP32累加/输出、α=1、β=0。输入和输出准备在计时外。

- 越界与有效零：二者都能让参与乘法的数据为零，但前者由descriptor边界导致TMA零填充，后者从有效全局地址读取显式零。能区分“值为零”与“边界/运输路径变化”，不能仅凭时间认定某个物理缓存机制。
- 部分边界：分别保留偶数或奇数个逻辑tile，区分部分有效tile与额外整块补齐。
- cluster1×1：只在长K增加两个条件。它同时改变多播、补齐和工作分配，不能用总时间差单独测多播成本。

## 完整GEMM时间

| 配置/条件 | K=1024 μs | K=8192 μs |
|---|---:|---:|
| cfg_a / ordinary | 19.456 | 90.080 |
| cfg_a / oob | 20.256 | 108.192 |
| cfg_a / explicit_zero | 19.008 | 89.776 |
| cfg_a / partial_even | 19.184 | 92.016 |
| cfg_a / partial_odd | 20.448 | 108.560 |
| cfg_a1 / oob | — | 89.456 |
| cfg_a1 / partial_odd | — | 88.928 |
| cfg_c / ordinary | 31.104 | 172.240 |
| cfg_c / oob | 30.816 | 171.440 |
| cfg_c / explicit_zero | 31.120 | 171.856 |
| cfg_c / partial_even | 31.136 | 172.256 |
| cfg_c / partial_odd | 30.688 | 170.960 |
| cfg_c1 / oob | — | 90.096 |
| cfg_c1 / partial_odd | — | 90.032 |

cfg_a长K整块越界比普通数据慢`108.192/90.080−1=20.1%`，有效零组89.776 μs与普通组接近。该条件不支持“只要数据为零就变慢”的解释。cfg_c的普通/越界/有效零分别172.240/171.440/171.856 μs，未呈现cfg_a的正惩罚，因此不能把cfg_a系数移植给cfg_c。

cfg_c切为cluster1后约90 μs，不能解释成多播翻倍成本：cluster1×2把12×11个逻辑tile补成12×12=144个物理tile，关键CTA执行两轮；cluster1×1恰好12×11=132个tile，一轮即可覆盖整卡。必须先算实际工作分配。

![边界完整服务](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/reanalysis/formal-v1/r18-service.png)

误差条为10个独立进程的标准差；橙色表示完整trace扰动超过5%，对应plain时间本身仍合格。

## 同物理tile的窗口增量

打点记录CTA入口、SM、producer首次工作、每consumer每tile的FIRST_MMA/MAIN_END/EPI_PERMIT/EPI_DONE和最终退出；首末SM相同。MAIN_END在mma_tail之后。它是包含等待的主循环窗口，不是单条MMA延迟。

```text
CTA entry → producer first work → FIRST_MMA → MAIN_END → EPI_PERMIT → EPI_DONE → next tile
                                  └── 主循环窗口 ──┘     └── 输出窗口 ──┘
```

先比较相同CTA编号、相同轮次和相同物理(mi,ni)的窗口差，再按边界双方、同列/同行其他CTA、其余CTA归组；分析器逐进程检查整份物理工作列表相同。两个不同进程的绝对clock64不能直接相减，必须各自先取窗口长度。

真实手算：cfg_a长K、trial0、CTA11、tile(11,0)，越界窗口为`4477056817549299−4477056817451969=97,330 cycle`；匹配普通组为`4477062071866019−4477062071800069=65,950 cycle`，增量31,380 cycle。该单点不是组中位数，也不代表最慢CTA。

10进程的首tile配对增量中位数如下，括号为进程标准差：

| 长K对照 | 边界tile cycle | cluster伙伴 cycle | 同列/同行其他CTA cycle |
|---|---:|---:|---:|
| cfg_a 整块越界−普通 | 21668 (1005) | 21776 (975) | 8091.5 (504) |
| cfg_a 有效零−普通 | −90 (261) | −86 (278) | −128.25 (207) |
| cfg_c 整块越界−普通 | 6.5 (131) | 1 (128) | −2.75 (136) |

cfg_a影响扩展到同列其他CTA，和共享B运输/服务竞争的解释相容；本组没有计数器与能唯一定位物理共享路径的第二干预，因此仍标为机制推断。cfg_c的微小正负差落在波动量级内，保留零差范围，不能解释成加速或端口规律。

每CTA每Ktile的工作量按真实tile计算：cfg_a为`2×128×128×64=2,097,152 FLOP`；cfg_c为`2×256×128×64=4,194,304 FLOP`。K=8192有128个Ktile。边界物理tile也执行指令，但完整GEMM的有效数学工作仍为2MNK，二者分开记。

主循环窗口差可以作为对应边界条件的候选增量。两个K点只能给出插值假设，不能凭零残差宣称可迁移斜率；V07之前冻结候选并用新形状检验，不把平均cycle/Ktile再叠一次截距。

## 两个超限条件的有界补查

轻量版本仅去掉consumer每tile的两个坐标写出，保留双方相同四事件与首末SM。cfg_a/c的plain GEMM SASS与完整版本逐字一致，[核对结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-lite-job737122-v1/reanalysis/local-independent/plain-sass-comparison.json)保存摘要。坐标由已在完整组逐进程验证的静态调度映射推得，并再次检查每CTA的tile计数；本组不声称重新直接记录了坐标。

| 轻量条件 | plain μs | 扰动 |
|---|---:|---:|
| cfg_a普通，K1024 / K8192 | 18.960 / 89.552 | 3.63% / 0.82% |
| cfg_a偶数部分边界，K1024 / K8192 | 19.168 / 92.352 | 3.09% / 0.97% |
| cfg_c普通，K1024 / K8192 | 31.136 / 171.664 | 1.34% / 0.69% |
| cfg_c奇数部分边界，K1024 / K8192 | 30.512 / 170.224 | 2.78% / 0.63% |

全部8条件各10组plain/trace通过，完整版本的两条超限记录仍不参与拟合。需要这两类的K插值时，使用轻量组自己的短/长K配对增量，不混搭完整组基线。[轻量归档](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-lite-job737122-v1/)、[独立检查结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-lite-job737122-v1/reanalysis/local-independent/rules.json)、[配对增量](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-lite-job737122-v1/reanalysis/local-independent/paired-tile-increments.json)。复现入口为`run_r18_lite.py`，prepare/build/setup/sample参数与完整组一致。

## 证据与复现

[归档](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/)、[条件与检查结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/reanalysis/formal-v1/rules.json)、[逐组窗口增量](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/reanalysis/formal-v1/paired-tile-increments.json)、[关键CTA和分布](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/reanalysis/formal-v1/tails.json)。所有原始坐标、时间戳和4096个检查位置保存在每进程压缩stdout中。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r18.py prepare --output <新目录> --cutlass-root <CUTLASS3.9.2>
# 将目录搬到单GPU GH200分配内；以下三步顺序执行。
python3 <目录>/source/run_r18.py build --output <目录>
python3 <目录>/source/run_r18.py setup --output <目录>
python3 <目录>/source/run_r18.py sample --procs 10 --output <目录>
python3 microbench/gh200_resource_campaign/access_rules/analyze_r18.py --input <归档> --output <新分析目录>
```

<a id="r17"></a>

## 前序：R17 cfg_a 边界 cluster 的主循环变慢（2026-10-08 迁入）

R17 是本页的前序实验，2026-10-08 并入本页；原文按原样保留，只把小节降一级。其中“原始经验修正”已被“离线修订”取代，二者都保留以便追溯。

**问题**：V06 中 cfg_a（128×128×64 cooperative，cluster 2×1，6 stage）在 tile 行数为奇数时，含补齐越界行的 cluster 主循环约 690–800 cycle/Ktile（其余约 512）。慢在哪些 CTA，由什么引起？

### 设计

作业 736958（romeo-a048，GPU-009a8880）。19 点，每点 plain、stamped 各 10 进程，交错随机，380 个全部成功；FP64 抽样核对误差均为 0。stamped 沿用 V06 打点，主循环/Ktile = (主循环结束 − 首 MMA)/Kt（clock64）。

cfg_a1 = cfg_a 改为 cluster 1×1：无 B 多播、无补齐行。SASS：4 个二进制 HGMMA 均为 8，168 寄存器，无 C7510、无 spill。

A 的 TMA descriptor 按逻辑 (M,K,L) 构造（`sm90_mma_tma_gmma_ss_warpspecialized.hpp` L215–218），m ≥ M 的部分由 TMA 填零。

tile 分类（cluster 沿 M 配对 mi 与 mi^1）：oob = 整行越界的补齐 tile；partner = 同 cluster 的另一 tile；partial = M%128≠0 的末行；in = 其余。

对照：M=1152 对 1280、1408 对 1536，tile 列表与 grid 完全相同，只差补齐行的 A 是越界填零还是真实数据；同尺寸 cfg_a 对 cfg_a1（cluster on/off）；M=1100、1400（部分越界末行 + 补齐行）与 1200、1500（偶数行，只有部分越界）；N=2304 时 K=1024/4096/10240；另一形状 1664/1792×2048×4096。

### 结果

第 0/1 轮 = 每个 CTA 的第 1/2 个 tile。cycle/Ktile，进程内 CTA 中位再取进程间中位。

| 点 | 行数 | 第 0 轮 边界 cluster | 第 0 轮 其余 | 第 1 轮 | plain µs |
|---|---:|---:|---:|---:|---:|
| a 1152×1664×2560 | 9 | oob 671 / partner 670 | 571 | — | 25.42 |
| a 1280×1664×2560 | 10 | — | 518 | — | 20.53 |
| a 1100×1664×2560 | 9 | oob 712 / partner 713 | 577 | — | 26.24 |
| a 1200×1664×2560 | 10 | partial 603 | 544 | — | 24.02 |
| a1 1152 / 1280 / 1100 | 9/10/9 | — | 519 / 519 / 519 | — | 19.33 / 19.70 / 19.25 |
| a 1408×2304×10240 | 11 | 701 / 701 | 592 | 514（oob 514） | 138.43 |
| a 1536×2304×10240 | 12 | — | 529 | 514 | 117.55 |
| a 1400×2304×10240 | 11 | 703 / 702 | 592 | 514 | 138.96 |
| a 1500×2304×10240 | 12 | partial 589 | 568 | 514 | 123.89 |
| a1 1408 / 1536（K=10240） | 11/12 | — | 514 / 514 | 514 | 113.18 / 114.43 |
| a 1408 / 1536×2304×1024 | 11/12 | 697 / — | 575 / 535 | 528 / 528 | 20.86 / 19.47 |
| a 1408 / 1536×2304×4096 | 11/12 | 664 / — | 555 / 517 | 516 / 516 | 56.69 / 49.89 |
| a 1664 / 1792×2048×4096 | 13/14 | 600 / — | 531 / 517 | 516 / 516 | 51.97 / 50.11 |

![主循环 cycle/Ktile](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R17-job736958/plots/r17_mainloop.png)

1. **哪些 CTA 变慢**：边界 cluster 的两个 CTA（oob 与 partner）同样慢，差 ≤5 cycle/Ktile，比偶数行对照多 +150～+190 cycle/Ktile（1664×2048 为 +83）。其余 CTA 也慢 +38～+63，并且按 N 列成片：M=1152 时第 3、5、7、11、12 列整列约 660，其他列约 570；偶数行对照各列 518–523。
2. **只在第 0 轮**：1408×2304 的第 1 轮里 oob tile 同样是越界填零，但为 514，与偶数行对照相同。
3. **越界填零 + cluster**：tile 列表完全相同、只把补齐行换成真实数据（1152→1280、1408→1536），变慢消失（518、529）。去掉 cluster（cfg_a1）时，同样的部分越界末行（M=1100）也是 519，不变慢。两个条件都需要。
4. **整行与部分越界**：不需要整行越界。偶数行、末行部分越界（1200、1500）同样变慢（603、589，略轻）；部分越界加补齐行（1100、1400）最重（712、703）。
5. **随 Kt 变化**：Kt=16/64/160 时边界 cluster 为 697/664/701，基本不随 Kt 变化，是每 Ktile 成本而非固定开销。K=10240 时边界 cluster 第一个 tile 多出约 2.7 万 cycle。

推断：第 0 轮同时启动时，越界 box 使边界 cluster 的节奏与同列 cluster 错开，后者经共享的 B panel（L2）被拖慢；第 1 轮起点已错开，故无此效应。具体机制未隔离（无 NCU）。

### 原始经验修正（后续修订，不能直接当斜率）

cfg_a：若 ceil(M/128) 为奇数或 M%128≠0，含最后真实行的 cluster（两 CTA）第一个 tile 的主循环斜率由 l1 改为 R，其余 tile 和其余 CTA 不变：

| 情形 | R（cycle/Ktile） | 依据点 |
|---|---:|---|
| 奇数行，M%128=0（整行越界） | 671 | 671、701、697、664、600 |
| 偶数行，M%128≠0（只有部分越界） | 596 | 603、589 |
| 奇数行，M%128≠0 | 707 | 712、703 |

实现（`r17_rule.py`）：在冻结 V06 参数上把边界 CTA 的 dL0 加 (R−l1)·Kt，时钟与 F 不变。

这是提交 `14eafd4` 时的历史代入方式。下述离线修订发现了参数所用数据的打点扰动与截距重复计算问题；原始 `rule_check.json` 保留用于追溯，此方式不再作为当前规则。

### 在 V06 两个边界例上的核对（非留出）

**这不是留出验证**：R17 测了这两个尺寸，R 也来自 R17 数据。要声称模型达标，需要新的留出测试。

| 例 | 实测 µs（V06） | V06 预测 / e | 加规则 / e | cycle e：V06 → 规则 |
|---|---:|---|---|---|
| 1152×1664×2560 | 26.50 | 21.47 / −19.0% | 25.24 / **−4.7%** | −22.5% → −5.5% |
| 1408×2304×10240 | 138.90 | 114.48 / −17.6% | 129.55 / **−6.7%** | −18.9% → −7.1% |

推断剩余偏差主要来自卡间差异：long_k 边界 cluster 在 V06 卡上 745、R17 卡上 701。

stamped 对 plain 扰动中位约 2%，最大 5.44%（a 1408×2304×1024），超过 5%，该点 cycle 仅作方向参考。

### 离线修订（2026-10-08）

原始 `(MAIN_END−FIRST_MMA)/Kt` 包含固定项，不能当新斜率后再加旧 `l0+dL0`（V06 为 490.77 cycle）。现改为相同物理边界位置的越界窗口减去有效窗口，再把额外耗时加到原模型。比较的是各自进程中位数之差，不是同一时刻的时间配对。两边的 grid 和首 tile 坐标集合经过核对。

扰动 5.44% 的 `1408×2304×1024` 退出定量拟合。仅作描述的合格平均值为 667.22 / 595.98 / 707.41 cycle/Ktile；这些不是导出的硬件斜率。

| 越界尺寸 | 有效对照的 M | 越界首 tile cycle | 同位置有效窗口 cycle | 额外 cycle |
|---|---:|---:|---:|---:|
| 1152×1664×2560 | 1280 | 26821.75 | 20741.75 | 6080.00 |
| 1408×2304×4096 | 1536 | 42475.75 | 33118.75 | 9357.00 |
| 1408×2304×10240 | 1536 | 112127.75 | 84109.50 | 28018.25 |

手算第一行：两边均 40 个 Ktile，`26821.75−20741.75=6080 cycle`，故本尺寸的额外窗口平均为 `6080/40=152 cycle/Ktile`；它不能自动解释成所有尺寸的斜率。

多数分类在固定 M/N 下只有一个合格 K，固定项和斜率不可辨识。1408×2304 只有两个合格 K，二点描述为 `ΔT=−3083.83+194.388·Kt`，没有独立检查点，残差为零也不足以支持外推。当时提出的有效地址显式零数据、cfg_c 的 N 边界以及新 K 对照，已在本页上文完成。

把两个尺寸各自的边界增量代入 V06，完整时间误差分别为 −5.38%、−5.64%，关键 CTA 周期误差为 −6.30%、−5.92%。这仍是事后点诊断；未建模同列扩散，也不是新留出验证。

新输出：[qualified-increments-v2](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R17-job736958/reanalysis/qualified-increments-v2/qualified-increments.json)。旧 CSV、JSON、预测、样本和源码未改。

```bash
python3 microbench/gh200_resource_campaign/access_rules/r17_rule.py \
  --run results/gh200_resource_campaign/access_rules/20261008-R17-job736958 \
  --v06 results/gh200_resource_campaign/access_rules/20261008-V06-job736868 \
  --output <新分析目录>
```

### 未解决

- 机制：越界 box 如何在多播 cluster 中拖慢主循环，同列 CTA 如何被带慢。
- R 随形状和卡变化（600～745），取中位会留下约 ±10% 的边界 cluster 误差；其余 CTA 第 0 轮的 +40～+60 未建模。
- cfg_c（cluster 1×2，N 方向补齐）是否有同类效应，当时未测；本页上文已测，cfg_c 的整块越界增量在波动范围内。

### 数据与复现

[20261008-R17-job736958](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R17-job736958/)：`cases.csv`、`classes.csv`（点×轮×类）、`summary.json`、`rule_check.json`、`plots/`、`samples/`、`build/`、`source/`。代码：`r17_{run,analyze,rule}.py`、`probes/r17.cu`（复用 v06_run、v06_model、v06_trace.hpp）。

<a id="v08-padding"></a>

## V08 后续对照：swizzle=8 的整 cluster 补齐 tile（2026-10-08 迁入）

job737322，romeo-a043，GPU-099dda56，CUDA 12.9.41、CUTLASS 3.9.2、`sm_90a`、NDEBUG；cfg_a/b/c 与 V07/V08 相同。V08 留出评分后在同一张卡上加测 84 个条件，每个 plain/stamped/ends 各 10 进程，只用于解释，不回填 V08 判定。上文固定物理 tile 列表比较越界与有效零；这里比较 swizzle=8 与 1，补齐同时改变工作量与轮数，两者分开解释。

**swizzle=8 的补齐 tile。** 三个形状（条件标签 padM/padN/padMN）× K 1024/4096。补齐使 T 增加 0–3，总时间比 swizzle=1 多 25%–80%（cfg_b 的 3072×3072 是 8 的整数倍 tile，不需要补齐，+0.6%/−2.4%）。旧汇总中，补齐 tile 相对有效 tile 的额外 cycle/Ktile 为 cfg_a 47–384、cfg_b 399–443、cfg_c 78–631；这些值跨轮次池化，不能直接解释为补齐路径自己的 K 依赖，见下节修订。沿 M 补齐的 tile 整块 A 在矩阵外，沿 N 补齐的整块 B 在矩阵外，但仍完整执行主循环（越界部分由 TMA 填零）。V08 的固定 ρ 没有获得迁移支持。

[结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/followup-v1/followup.json)。

### 按轮次与 CTA 工作量重算（2026-10-09）

**旧“随 K 增大”的描述不适用于所有分组。** `v08_model.summarize_case()` 的 `tile_L` 把十个进程中全部 `round>0` 的 tile 按 in/pad 类别合池；其差值同时包含轮次、每 CTA 总 tile 数和同期服务变化。本次重读 18 个 swizzle=8 补齐条件、每条件十次 stamped 原始记录，均通过既有数值/坐标复核，条件级扰动和 CV 均在原 5% 范围内。按 `(round, 每CTA总tile数, 类别)` 分组，每进程先取 pad 与 in 的中位差，再对十个进程取中位；round 从 0 起。

| 形状标签与分组 | K=1024，额外 cycle/Ktile | K=4096，额外 cycle/Ktile |
|---|---:|---:|
| cfg_a padN，旧池化 | 190.25 | 383.55 |
| cfg_a padN，round 2、5 tiles/CTA | 138.00 ± 25.97 | 159.47 ± 14.49 |
| cfg_a padN，round 2、6 tiles/CTA | 48.92 ± 20.61 | 31.98 ± 10.93 |
| cfg_a padN，round 5、6 tiles/CTA | 3.38 ± 11.96 | −9.27 ± 16.08 |
| cfg_c padM，旧池化 | 380.25 | 631.12 |
| cfg_c padM，round 1、3 tiles/CTA | 516.27 ± 28.95 | 593.14 ± 11.62 |
| cfg_c padM，round 1、4 tiles/CTA | 310.48 ± 29.58 | 312.52 ± 11.68 |
| cfg_c padM，round 3、4 tiles/CTA | 35.00 ± 25.31 | 50.34 ± 8.63 |

± 为十个进程差值的标准差，不是置信区间。原池化数字已从原始事件重现；cfg_b 的原池化 padM/padN 差也分别从 412.28/443.06 降到 398.51/436.08。形状标签 padMN 不等于实际同时存在 padM、padN：cfg_a 此标签下只有 padM，cfg_b 此标签没有补齐，分析以逐 tile 坐标分类。

还有不能省略的有效 tile 变化。cfg_a padN、K=4096、每 CTA 六个 tile 时，有效 tile 在 round 0/1/3/4 为约 520–533 cycle/Ktile，round 2/5 则为 904.23/834.82；同期 padN 为 940.93/828.62。cfg_c padM、K=4096、每 CTA 四个 tile 时，有效 tile 在 round 0/2 为 1030.68/1029.66，round 1/3 升至 1623.85/1494.84。只对 pad tile 乘 ρ 会漏掉这些有效 tile 窗口。注意表中的“差的中位数”不必等于此处“两个中位数的差”。

本次不能把这些差值直接接成 [R13](R13-async-retirement.md#v08-supply) 的统一有效供给函数：同一配置、K、pitch 和足迹下，不同轮次的有效 tile 已有明显差异；只依赖这些静态变量的服务率不能同时解释它们。额外引入执行相位或在途状态是候选，但现有窗口不能把供给等待、输出/邻居竞争和起始错位分开。`round` 相同也不等于实际时间重叠；新分组仍是不同物理坐标之间的观察比较，不是隔离路径的因果配对。两个 K 点也不足以确定固定项与斜率。

R18 原同物理工作列表配对可独立约束边界路径：cfg_a 首轮 oob−ordinary 在 K=1024/8192 为 1954/21668 cycle，partner 为 1936/21776，同列其他 CTA 为 598.25/8091.5；次轮 oob 为 −4/−1.5，首轮 explicit_zero 为 +1/−90。cfg_c 首轮 oob 为 +3/+6.5；其长 K 次轮为 +434.5 cycle（进程标准差 145.81），不能写成所有轮次严格无差异。这些复用[原合格配对](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/reanalysis/formal-v1/paired-tile-increments.json)，属于含有效伙伴的边界 cluster；V08F 的整 cluster OOB 不能直接借用该系数。

真正缺少的是同一物理 tile 列表上的整 cluster OOB/有效地址显式零配对。沿用已确认的 cfg_b、2304×4096、swizzle=8 条件，先在 K=1024/4096 比较两条输入路径；只扩大输入 tensor-map 的有效范围并填零，固定逻辑输出 mask、A/B/D stride、grid、分配容量及实际 CTA 工作序列。修改逻辑 GEMM 的 M/N 会连同输出一起改变，不能充当这一配对。若这四个条件已显示路径差，再决定是否增加 N 向或第三个 K；此处没有新增采样。

[按轮次的结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/B-20261009-padding-rounds/padding.json)保存每进程分组值、差值、旧池化结果及扰动。重算命令：

```bash
python3 microbench/gh200_resource_campaign/access_rules/analyze_r18.py --v08-padding \
  --input /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1 \
  --output <该run下新的reanalysis目录>
```


### 整 cluster 补齐：只改变输入 tensor map 的准备

按上节缺口，在公共 r18.cu 增加 `--input-map-m/n`：先按逻辑 GEMM 构造完整 CUTLASS Params，再只替换 mainloop 的源描述符。输出 descriptor、问题形状、scheduler、grid 和输出检查范围保持逻辑 M/N。默认不传时沿用原输入初始化范围；显式使用该开关的配对均初始化完整预留输入区，再把逻辑范围外的对应输入置零。每次测量后检查扩展输入区确为零，并检查逻辑输出范围外仍为原哨兵。

本批仅用 cfg_b、逻辑 2304×4096、swizzle=8、K=1024/4096。软件补齐至 3072×4096，四条件均按该容量预留 A/D，A 的第 2304 行起显式填零；对照的输入 map M=2304，变体为 M=3072。B/D 行距、输入 seed、逻辑输出 mask 和实际工作列表必须相同。源 map N 均为4096。本批不通过扩大逻辑 GEMM 来代替输入路径对照，也不同时改变 cluster 或 stage。

条件见 [r18-input-map-padding.json](../../../../../../microbench/gh200_resource_campaign/access_rules/configs/r18-input-map-padding.json)。沿用 plain/wide/stamped/dual 四变体，每条件十进程；wide 与 dual 共用相同记录容量，以直接纳秒窗口辅助周期比较。配对比较先核对完整工作坐标列表，再按 `(输出序号, CTA总tile数, 逻辑in/padM)` 分组。取每进程组内均值、再比较进程对和跨进程中位数；逻辑 padM 标签不会随源 map 扩大而消失。这是补齐路径的机制对照，不是新的完整预测留出。

代码与条件已准备；尚未将本节记为GPU通过。实际源码、SASS、数值与结果随新run保存，不改旧V08或R18归档。


### M 向配对结果：job738296

2026-10-09，独占 a057、GPU-43269fbc-449d-3e0f-908a-9c81229546d3，源码 **1f175c2**。四个二进制均为 168 registers、16 条静态 HGMMA，无 spill/C7510；160 个正式进程全部成功。655360 个保存输出值通过原数值回放；另逐次检查共 314572800 个扩展输入 FP16 值全为零，逻辑输出外保持 NaN 哨兵。配对的物理 CTA 工作坐标列表一致。

| K | map M=2304：plain µs | map M=3072：plain µs | 同 trial 的变体/对照−1：中位［范围］ |
|---:|---:|---:|---:|
| 1024 | 49.744 | 39.920 | **−19.98%［−21.51%, −17.20%］** |
| 4096 | 174.304 | 141.808 | **−18.96%［−20.39%, −16.92%］** |

每个方向均为十次进程配对，绝对时间列是各自中位数，不能将其比值混作配对比值中位数。wide/plain 中位变化为 −0.55%～+0.12%；dual/wide 为 −0.52%～+4.80%，部分单对超过5%（最大7.37%），不删除这些记录或宣称每次打点扰动都小于5%。完整时间最大 CV 为1.87%；分项只解释为当前双时钟观察协议下的窗口。

逻辑 padM 的主循环确实变快。K=4096 时，越界路径五个 `(j,T)` 组为约 747～1005 cycle/Ktile，有效地址零为约 521～528；直接纳秒也同向降低约118～251 ns/Ktile。K=1024 的 padM 增量同样显著，但不同输出位置并不相等。例如 j=4/T=6 的配对变化为 −484.56 cycle/Ktile，j=5/T=6 为 −290.16。

这份对照把逻辑工作量、输出mask及输入数值固定，证明**输入描述符越界路径本身会增加本批补齐成本**。它没有把零填充内部服务、供给/计算重叠和全卡竞争分别定值。有效tile也有位置相关变化，且首轮可能周期接近、纳秒不同，不能把一条全局频率或统一rho当作所有位置的解释。按原计划，下一步可复用同一二进制做 N 向四条件配对，检验 A/B 输入路径能否共用该规则；当前结果不外推给 cfg_a/c。

[原始归档](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R18-input-map-job738296/)、[逐进程和位置配对](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R18-input-map-job738296/reanalysis/input-map-pairs-v1/input-map-pairs.json)。复核沿用 `analyze_r18.py --input-map-pairs --input RUN --output 新目录`，不修改历史归档或冻结预测。


### N 向配对结果：job738307

同日仍在 a057/GPU-43269fbc，完整复用上批四个二进制，逻辑形状改为4096×2304、swizzle8，B/D 行距与预留 N 均为3072。两个路径的 B 第2304列起均为零，只把输入 map N 从2304扩大到3072；输出descriptor、mask和CTA坐标列表仍相同。四条件160个进程全部成功，另检查314572800个扩展FP16值为零以及逻辑输出外哨兵。

| K | map N=2304：plain µs | map N=3072：plain µs | 同 trial 变体/对照−1：中位［范围］ |
|---:|---:|---:|---:|
| 1024 | 49.200 | 39.952 | **−18.87%［−20.26%, −16.57%］** |
| 4096 | 170.400 | 140.352 | **−17.62%［−19.21%, −16.25%］** |

因此 A 与 B 的整 tile 越界路径在这组 cfg_b 条件下都具有额外代价；不能把有效地址覆盖减少理解为需求必然减少。K=4096 的 padN 越界窗口约793～1005 cycle/Ktile，而有效地址零为约521～529；与 M 向一样，位置差异和纳秒/周期差异仍需保留。两方向的形状、复用历史不同，不能由接近的百分比宣称它们具有同一个物理填零速率。

[条件与原始记录](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R18-input-map-n-job738307/)、[逐位置配对](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R18-input-map-n-job738307/reanalysis/input-map-pairs-v1/input-map-pairs.json)。这八个 M/N 向条件用于下一版模型区分有效地址与越界填零的需求；已有冻结分数保持原判定。
