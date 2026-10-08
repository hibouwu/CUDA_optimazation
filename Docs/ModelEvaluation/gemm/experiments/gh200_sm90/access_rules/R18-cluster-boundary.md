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

**swizzle=8 的补齐 tile。** 三个形状（只沿 M、只沿 N、两向补齐）× K 1024/4096。补齐使 T 增加 0–3，总时间比 swizzle=1 多 25%–80%（cfg_b 的 3072×3072 是 8 的整数倍 tile，不需要补齐，+0.6%/−2.4%）。补齐 tile 相对有效 tile 的额外 cycle/Ktile：cfg_a 47–384，cfg_b 399–443，cfg_c 78–631，且随 K 增大（cfg_a 沿 N：K=1024 时 190，K=4096 时 384）。沿 M 补齐的 tile 整块 A 在矩阵外，沿 N 补齐的整块 B 在矩阵外，但仍完整执行主循环（越界部分由 TMA 填零）；它的代价不是常数比值，V08 的 ρ 不能迁移。

[结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/followup-v1/followup.json)。
