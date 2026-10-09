# R10：A/B 行距与尺寸尾部

9点完整GEMM时间已测得；cfg_a/c使用v2全网格trace，cfg_b使用v4局部trace。本轮条件观测已完成核对；当时计划的预测迁移检验已记录在[V05](V05-rule-transfer.md)。本页上半部分是2600×3000×2000的B行距对照；V08后续对照的A/B/D行距与K/N尺寸尾部见[文末](#v08-stride)。

## 结果

固定M×N×K=2600×3000×2000，FP16输入、FP32累加与输出，α=1、β=0。只改变B的ldb；A的lda=2000，D的ldd=3000。设备为romeo-a053的GPU-54896349-d69d-9358-b526-433454c04733，job735876，CUDA12.9、sm_90a、CUTLASS3.9.2。

| 配置 | tile | schedule | cluster | stage |
|---|---|---|---|---:|
| cfg_a | 128×128×64 | cooperative | 2×1 | 6（沿用V04自动选择） |
| cfg_b | 128×128×64 | pingpong | 1×1 | 6（沿用V04自动选择） |
| cfg_c | 256×128×64 | cooperative | 1×2 | 4 |

下表完整时间均来自[v2全9点plain](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v2/analysis-formal/summary.json)，每点10个独立进程；三个行距分别为6000/6016/6144B。

| 配置 | ldb | 完整时间中位µs | CV | 同档全网格trace周期/Ktile | trace判定 |
|---|---:|---:|---:|---:|---|
| cfg_a | 3000 | 66.400 | 0.588% | 630.234 | 通过 |
| cfg_a | 3008 | 62.944 | 0.559% | 606.500 | 通过 |
| cfg_a | 3072 | 63.328 | 0.777% | 615.906 | 通过 |
| cfg_b | 3000 | 76.272 | 0.601% | 816.562 | 通过 |
| cfg_b | 3008 | 58.272 | 1.223% | 不采用 | 超5% |
| cfg_b | 3072 | 57.136 | 1.748% | 不采用 | 超5% |
| cfg_c | 3000 | 54.592 | 0.820% | 1031.594 | 通过 |
| cfg_c | 3008 | 54.544 | 0.888% | 1031.500 | 通过 |
| cfg_c | 3072 | 54.016 | 1.049% | 1031.500 | 通过 |

cfg_b的6000B行距明显较慢，6016/6144B恢复大部分差异；cfg_a差异较小，cfg_c主循环周期基本相同。因此规则应按配置分别使用：这组结果支持cfg_b的行起点对齐解释，不能推广为所有配置统一需要128B行距对齐，也不提供物理cache/TMA/HBM流量归因。

cfg_b的最终合格周期来自[v4局部补测](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/analysis-formal/summary.json)及[CSV](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/analysis-formal/cases.csv)。只采线性CTA0–3、consumer0的last tile，每行距10对进程；下表两个周期列分别保留池化与按进程聚合口径。

| ldb | plain中位µs | plain CV | trace中位µs | 最大绝对配对扰动 | 局部池化中位cycle/Ktile | 先按进程中位再汇总cycle/Ktile |
|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 76.160 | 0.401% | 76.448 | 2.068% | 854.531 | 869.906 |
| 3008 | 57.904 | 0.977% | 58.160 | 2.913% | 671.359 | 671.492 |
| 3072 | 57.456 | 0.895% | 57.568 | 3.775% | 659.047 | 662.812 |

v4三行距均通过每对绝对5%门槛，局部主循环和同档完整时间方向一致。v2完整时间与v4完整时间分别比较，局部周期不与旧全网格周期池化；v4不能替代整个grid的分布。

## 最小算例

B的物理地址为`B[k*ldb+n]`；ldb3008增加8个FP16 padding元素，使行距从6000B变为6016B。需要检查的B padding为`2000×8=16000`个元素；ldb3072为`2000×72=144000`，ldb3000明确为0。

例如v2 cfg_b在同一测量协议内从ldb3000的76.272µs变为ldb3008的58.272µs，完整时间减少18.000µs；这个差值不单独命名为TMA等待。Ktile数为ceil(2000/64)=32，末个Ktile只有16个有效K元素；局部周期指标将实际trace窗口除以32，保留循环与排空成本。单一K长度不能识别固定开销与每Ktile斜率。

## 时间、抽样与数值边界

完整服务时间取plain CUDA event，覆盖一次warm GEMM及输出完成。trace起点位于首个full-barrier等待返回后、首个MMA前，终点为mma_tail返回后，最终WGMMA等待和输入stage释放已执行。它包含此后输入等待、循环和排空，排除了起点之前的首次输入等待；不减旧模型c0，不作为裸MMA周期。clock64只在同CTA/同SM内相减。

v2 trace覆盖全grid两消费者的last tile；cooperative两组共同处理同tile，tile计数只用group0，pingpong两组交替处理不同tile。v4仅保留CTA0–3、consumer0，每进程4个last-tile观测，word4是该消费者实际完成tile数。未采CTA/WG为缺观测，不能当作0cycle或推断整个grid完成tile数。

v2 plain没有trace scratch准备；v3/v4的plain/trace都分配65536B scratch，初始化及预热后按相同顺序memset、device同步，再执行计时调用。三个版本分别归档和统计。每进程预热8–30次、末5次CV≤2%，trace/plain相邻配对、顺序随机；负向扰动超过5%同样不合格。

初始化与CUTLASS descriptor均使用真实stride，B视图为(N,K,L)，StrideB=(1,ldb,0)。所有B padding填65504（FP16位模式0x7bff），计时外检查完整padding；本矩阵A/D无padding。每进程保存4096个period17 dyadic见证的数值坐标与实际FP32值，覆盖首尾和64/128/256边界。v2/v3/v4共300进程、1228800个保存值通过独立逐K整数点积重算，误差0；每个GEMM工作量31200000000 FLOP，预热、padding和统计也已复核。该见证不声明任意输入误差标准。

## 来源与检查索引

测量源码、二进制和原始数值均保留。v3/v4 CUTLASS依赖按各自冻结SHA256匹配后由v2副本补齐，各838份；不是替换测量源码。资源与SASS保存在各build中。C为本组实现者，以下公式复算使用作者的另一种参考实现，不等同于由未参与实现者完成的复核；作者复算及对应hash见[v4 reviews](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reviews/independent-C.md)与[hash清单](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reviews/independent-C-hashes.json)。

未实施R10的ROOT已另行完成[独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reviews/ROOT-independent.md)与[核对证据清单](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reviews/ROOT-independent.json)：300个正式进程、1228800个保存值直接逐K参考通过，FLOP、padding、预热、HGMMA计数及零spill核对通过。本文的结果判定依据这份未参与实现者的复核。

| 来源 | 范围与检查结果 | 保留原因 |
|---|---|---|
| [v1代表](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v1/analysis-representatives/summary.json) | cfg_b的ldb3000/3072，40进程 | 行距差可辨，决定扩大9点；cfg_b整段16个静态HGMMA为prologue8+steady8，与V04一致 |
| [v2正式](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v2/analysis-formal/summary.json) | 9点180进程，plain全部保留；7点trace满足扰动要求 | cfg_b3008的两对−5.927%/−6.108%、3072的一对−5.341%超门槛，不采用这两点trace |
| [v3对称准备](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v3/analysis-formal/summary.json) | cfg_b三行距60进程 | plain中位76.256/57.840/57.584µs；最大trace扰动1.607%/4.538%/6.202%，3072仍失败 |
| [v4局部trace](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/analysis-formal/summary.json) | cfg_b三行距60进程，三组通过 | 对称准备，原事件位置不变，仅限制CTA/WG；最终周期只在此抽样域使用 |
| 本地编译诊断`/tmp/gh200-gaps-local/R10/` | 未运行GPU | GCC16超出CUDA13支持范围，override标准库语法失败，GCC15缺cc1plus；正式测量由CUDA12.9完成 |
| 首tile/不同起点候选 | 未进入GPU构建，不作为规则 | 后续恢复原事件边界并先修正准备对称性；最终采用v4局部last-tile方案 |

对齐结论限于本页配置、2600×3000×2000和三种实际行距。未扩做缓存准备、其他补齐矩阵；v4不提供全卡周期分布。这些条件观测在[V05](V05-rule-transfer.md)中接受了迁移检验；分项误差独立评分，不能从完整时间调整。

## 复现

在主对话调度的GH200作业内加载CUDA12.9，新RUN目录必须不存在。当前入口对cfg_b使用v4局部trace；旧档必须使用各自source中的冻结分析脚本。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r10.py cpu-check
python3 microbench/gh200_resource_campaign/access_rules/run_r10.py list
python3 microbench/gh200_resource_campaign/access_rules/run_r10.py build \
  --cutlass-root /PATH/TO/CUTLASS3.9.2 --output RUN
python3 microbench/gh200_resource_campaign/access_rules/run_r10.py sample --output RUN
```

sample结束自动写新`analysis-formal/`；不要在同一已有分析目录重复写入。若只复现cfg_b三行距，build加`--config cfg_b`后仍用sample，不加仅含两个行距的`--representatives`。CPU准备检查不运行GPU；已保存测量的独立复核证据见上表reviews。

<a id="v08-stride"></a>

## V08 后续对照：A/B/D 行距与 K/N 尺寸尾部（2026-10-08 迁入）

job737322，romeo-a043，GPU-099dda56，CUDA 12.9.41、CUTLASS 3.9.2、`sm_90a`、NDEBUG；cfg_a/b/c 与 V07/V08 相同。V08 留出评分后在同一张卡上加测 84 个条件，每个 plain/stamped/ends 各 10 进程，只用于解释，不回填 V08 判定。这里的“尾部”指 M、N、K 不整除 tile，不是 epilogue 尾部或最后完成的 CTA。与上文 2600×3000×2000 的对照不在同一张卡，不池化。

M=2304、N=3072，以对齐形状为参照，每次只改一项。数值为总时间变化（括号为后续 tile 主循环变化）：

| 改动 | cfg_a | cfg_b | cfg_c |
|---|---|---|---|
| K=1000（K 尾部，A 行距仍 2048 B） | +0.3% | −0.6% | −0.7% |
| N=3000（N 尾部，B/D 行距仍对齐） | −0.1% | +4.7%（+2%） | −1.2% |
| D 行距 12320 B | +0.1% | +0.2% | +1.9% |
| A 行距 2064 B，K=1024 / 8208 B，K=4096 | +7.7% / +8.7% | +15.0% / +16.4% | +0.2% / +0.3% |
| B 行距 6160 B，K=1024 / K=4096 | +4.7% / +6.2% | +28.2% / +33.7% | −0.4% / +0.8% |
| A 行距 2000 B + K 尾部（同 h03 的 A） | +7.3% | +17.1% | +0.1% |
| B/D 行距 6000/12000 B + N 尾部（同 h03 的 B、D） | +2.7% | +31.5%（+39%） | −0.2% |

尺寸尾部本身和 D 行距几乎无影响；A、B 的行距不是 128 B 倍数时 cfg_b 明显变慢，cfg_a 次之，cfg_c 不变。按每 SM 供给的解释见 [R13](R13-async-retirement.md#v08-supply)。[结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/followup-v1/followup.json)。

<a id="v08-stride-next"></a>

## 行距效应的供给候选与剩余条件（2026-10-09，A）

现有计时能辨别 A/B 行距对主循环的不同影响，不能判定物理请求放大或某个 bank 冲突。R13 的[离线 max 候选检验](R13-async-retirement.md#v08-supply-fit)已复用 84 条中的可比记录：51 条拟合、14 条尾部/D 行距诊断、1 条原打点扰动超限，另将 18 条 swizzle=8 分开。其普通输出 tile 的逻辑源请求为 cfg_a/b/c 每 SM 每 Ktile 24/32/32 KiB，按多播摊分，不是物理 L2 字节。

仅改变 A/B +16 B 行距时，后续输出 tile 的主循环增加量如下。分母是同 cfg、同 K 的对齐参照，使用每进程均值再跨进程中位数的 L；与首输出 tile 的 L0 分开。

| 条件 | cfg_a | cfg_b | cfg_c |
|---|---:|---:|---:|
| A +16 B，K=1024 | +2.00% | +12.88% | −0.03% |
| A +16 B，K=4096 | +6.00% | +13.99% | −0.004% |
| B +16 B，K=1024 | +0.74% | +36.11% | +0.01% |
| B +16 B，K=4096 | +6.12% | +33.74% | +0.001% |

cfg_c 的微小差异只表示本精度内未分辨，不作为正负物理效应。cfg_a 的 K 依赖也不能直接解释为固定请求放大：K 与输入工作集同时变化。共享 A/B 行距供给项加输入足迹后，cfg_a 的 A 行距长 K 点仍低估 11.05%；增加 cfg_a 专属项后，已有 K 组之间的最大迁移误差仍为 11.16%。所有条件都使用过，属于开发诊断，不是新的留出成绩。

单轴余数曲线原先列了以下 14 个候选，不必重跑 84 条。它们尚未采样；当前优先处理下文 h03 的联合行距缺口，不把整张单轴曲线列入这次提交批次：

| 新条件 | 固定条件 | 新增点数 | 要区分的现象 |
|---|---|---:|---|
| 仅 A 行距 `2K+64 B`、仅 B 行距 `6144+64 B` | M×N=2304×3072，cfg_a/b/c，K=1024/4096，swizzle=1 | 12 | 16 B 余数的效应能否迁移到 64 B；配置与 K 是否仍有交互 |
| 仅 B 行距 `6144+32 B`、`6144+128 B` | 同形状，cfg_b、K=1024 | 2 | 与已有 0/16 B 和上述 64 B 组成余数曲线；128 B 检查增加 padding 后回到同一余数 |

A/B +16 B 和对齐旧观测已存在，继续离线复用。新批次要得到成对变化，只需让六个 cfg×K 的对齐参照由本批新条件共用；同批参照不是补齐旧档或重做旧留出。按最大 stride 预留相同容量、按实际 stride 初始化，保持另一操作数、D 行距、grid、输入及缓存准备一致。旧档与新批次分别统计；仅凭它们的绝对周期相近，不能默认物理驻留或流量相同。

本次仅增加专用 CPU 分析器并修订 R10/R13 文档，尚未修改或构建公共 cfg_a/b/c 框架，也未采样上述新条件。原 R10 独立探针的不同卡、不同布局观测继续单列。

## h03 的 A/B 联合行距对照（2026-10-09）

V08 最大时间误差来自 cfg_b h03（2400×3000×1000）。现有 V08F 分别改变 A 或 B 时，也使用了不同于 h03 的尺寸；不能将这些单轴增量直接相加，便认为已经识别了 h03 的联合代价。`max(计算,供给)` 还可能把对齐时较小的供给需求隐藏在计算分支下，单轴观测未必足以确定联合供给。

本批只用原 h03 形状，cfg_b、swizzle=1、dyadic/seed17，**固定 D 行距为 3000 个 FP32 元素**，配四种 A/B 物理行距。数值、K/M/N 尾部、CTA grid 和准备协议在四条件间相同：

| 条件 | lda，FP16 元素 | ldb，FP16 元素 |
|---|---:|---:|
| A/B 对齐 | 1024 | 3072 |
| 仅 A 未对齐 | 1000 | 3072 |
| 仅 B 未对齐 | 1024 | 3000 |
| A/B 均未对齐 | 1000 | 3000 |

四条件按共同上限分配并初始化：A 为 2400×1024、B 为 1000×3072 个 FP16 元素，分别通过 `--alloc-lda 1024 --alloc-ldb 3072` 指定；tensor map 仍使用表中的实际 lda/ldb。未使用的容量写入同一哨兵，不改变逻辑 A/B 数值。默认未指定上限时保留原分配和初始化方式。

矩阵在 [r10-h03-source-pitch.json](../../../../../../microbench/gh200_resource_campaign/access_rules/configs/r10-h03-source-pitch.json)。每条件保留十个独立进程，四变体为原布局 plain、扩大 trace 缓冲区但无打点的 wide、同缓冲区的 clock64 stamped，以及同缓冲区的 dual。plain/wide 区分缓冲区清零的准备差异，stamped/dual 与 wide 配对评估打点影响。

公共 R18 探针的可选 dual 模式在原 FIRST_MMA、MAIN_END、EPI_PERMIT、EPI_DONE 四个位置同时保存 clock64 和 globaltimer，并记录 producer-first-work 的 ns；原事件含义不变。默认六字布局保持，dual 与其匹配基线采用十字布局，原周期和坐标字段位置保留。这样可以直接比较同次调用的主循环纳秒、周期及其比值，避免用整调用平均频率重建内部时间线。

首要输出是逐 trial 的 `both−A_only−B_only+aligned`，分别报告完整时间、后续主循环周期和直接纳秒；再按 j/T/坐标定位差异。主循环窗口重叠不等于 TMA 物理占用，联合效应也不直接命名为 bank 冲突或流量放大。本批用于机制识别和模型开发，不是 V09 留出。

沿用公共入口准备本批，不增加新实验编号：

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r18.py prepare \
  --output <新运行目录> --cutlass-root <已有CUTLASS_3.9.2目录> \
  --cases-file microbench/gh200_resource_campaign/access_rules/configs/r10-h03-source-pitch.json \
  --dual-clock
```

准备阶段的 CPU 检查已核对四条构建命令、同尺寸基线、直接 ns 解码和九份旧 V08 记录的兼容回放；新 dual 宏的实际结果见下文。

首批 job 738162 已在原参考卡 GPU-099dda56 完成，160 个进程数值和时间线检查通过；四个变体均为 168 个寄存器、16 条静态 HGMMA、无 spill。不过该版仍按实际行距分配 A/B，四组合容量不同，未满足上面的共同容量控制。其[原始记录](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-joint-pitch-job738162/cases.json)和[容量诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-joint-pitch-job738162/reanalysis/allocation-diagnostic-v2/joint-pitch.json)只保留作诊断，不替换或混入下面的匹配容量结果。

### 匹配容量结果：job738169

job738169，romeo-a043，GPU-099dda56-d7af-f60e-c285-aa2dc7ddfcfe。四角均分配 A=2,457,600、B=3,072,000 个 FP16 元素，初始化按相同容量上限执行，逻辑值、M/N/K、D行距、grid及dyadic/seed17不变。四条件×四变体×十进程共160个成功进程，655,360个保存输出值通过回放，无失败尝试。四变体均为168寄存器、16条静态HGMMA、无spill/C7510。

plain trace缓冲为827904 B，wide/stamped/dual均为1368576 B；打点扰动以同缓冲区的wide为基线。下表完整时间取进程中位；后续主循环先对每进程所有 `j>=1` tile 窗口取均值、除以 `ceil(1000/64)=16`，再取十进程中位。窗口仍包含等待和排空，不减预填常数。

| A/B行距条件 | plain μs | wide μs | dual μs | dual后续cycle/Ktile | dual后续ns/Ktile | dual/wide−1 |
|---|---:|---:|---:|---:|---:|---:|
| 对齐/对齐 | 29.888 | 29.840 | 30.960 | 566.379 | 334.935 | +3.753% |
| 未对齐/对齐 | 36.160 | 36.128 | 36.496 | 717.251 | 415.515 | +1.019% |
| 对齐/未对齐 | 37.920 | 38.048 | 38.736 | 779.464 | 447.485 | +1.808% |
| 未对齐/未对齐 | 46.672 | 46.832 | 47.216 | 998.478 | 561.861 | +0.820% |

四变体各条件CV最大1.583%，wide/plain的中位变化范围−0.161%～+0.343%。中位打点扰动小不表示每一对都合格：对齐条件dual/wide逐trial为+1.257%～+8.073%，仅B未对齐为−0.742%～+5.378%；stamped在对齐及仅A未对齐条件也出现超过5%的配对。因此本批满足容量控制与数值检查，但不能写成“所有单对trace扰动均通过5%”。所有有效trial保留，下面的局部量明确属于dual观察窗口，未证明等于未打点内核的阶段服务。

每trial先计算 `I=both−A_only−B_only+aligned`，不能对四个条件各取中位后再把差称为“联合项中位”。同trial是相邻随机化进程，不是同时执行的四次调用：

| trial | plain完整时间 I，μs | dual后续 I，cycle/Ktile | dual后续 I，ns/Ktile |
|---:|---:|---:|---:|
| 0 | 2.9440 | 86.6755 | 35.9321 |
| 1 | 2.0800 | 69.9936 | 32.4877 |
| 2 | 2.8160 | 85.9115 | 36.2593 |
| 3 | 2.5600 | 59.8418 | 32.6296 |
| 4 | 2.5280 | 68.2402 | 39.7407 |
| 5 | 1.3440 | 64.4850 | 28.5000 |
| 6 | 2.5280 | 71.4593 | 30.0062 |
| 7 | 1.5040 | 61.2602 | 30.3333 |
| 8 | 1.8560 | 76.5241 | 36.9383 |
| 9 | 3.0080 | 63.6582 | 32.3889 |
| 中位 | 2.5280 | 69.1169 | 32.5586 |

正联合项存在于全部十个trial，但**不等于物理请求互相放大**。例如trial0的四个直接ns/Ktile为335.0556、414.7531、445.9012、561.5309；令

```text
L(xA,xB) = max(C, S + a*xA + b*xB)
C=335.0556, S=299.1235, a=115.6296, b=146.7778  （ns/Ktile）
```

即可重现四点，正联合项为 `C−S=35.9321`，模型中并没有 `xA*xB` 请求交互项。C只是对齐窗口锚点，S/a/b也不是独立测得的物理供给时间。这是四点确定四参数的一个可行解释，不能称预测验证；主循环等待、地址覆盖、内部服务及观察扰动仍未分开。

[原joint-v1结果](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-joint-pitch-job738169/reanalysis/joint-v1/joint-pitch.json)不改动；[逐trial复核与配对扰动](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-joint-pitch-job738169/reanalysis/B-20261009-matched-joint-final/joint-pitch.json)保存全部原联合项，并新增逐trial stamped/dual相对wide的扰动。复核入口为 `analyze_r10.py --joint-pitch --input <run> --output <新reanalysis目录>`。

## B行距地址覆盖的局部预测入口（采样前固定）

固定 cfg_b、2304×3072×1024，A/D行距对齐，`lda=1024, ldd=3072`；仅令 B 行距为6144+{0,16,32,64,128} B。五点均为 `alloc_lda=1024, alloc_ldb=3136, storage_m=2304, storage_n=3072`，dyadic/seed17、swizzle1、sm_count0、evict0；公共dual-clock协议不变。配置及采样由管理对话维护，分析器不创建第二套准备或调度框架。

| ID | 用途 | ldb，FP16元素 | 额外32 B覆盖行数 | 额外128 B覆盖行数 |
|---|---|---:|---:|---:|
| cfg_b_bcurve_p0 | 校准 | 3072 | 0 | 0 |
| cfg_b_bcurve_p16 | 校准 | 3080 | 32 | 56 |
| cfg_b_bcurve_p32 | 校准 | 3088 | 0 | 48 |
| cfg_b_bcurve_p64 | 局部留出 | 3104 | 0 | 32 |
| cfg_b_bcurve_p128 | 局部留出 | 3136 | 0 | 0 |

覆盖量按每次TMA的64行×256 B B-box计算。对覆盖行宽W=32或128 B，逐行起点余数为 `r_i=(i*B_pitch_bytes)%W`，额外覆盖为 `sum_i(ceil((r_i+256)/W)−256/W)`。box的N起点是256 B倍数，K起点按64行移动，不改变本表余数。这只是地址覆盖计数，不命名为物理请求数、L2流量或HBM流量。

冻结目标只取 **dual后续主循环ns/Ktile**：每trial对全部CTA的 `j>=1` 窗口 `(MAIN_END_ns−FIRST_MMA_ns)` 取均值，除以16，再对十个成功trial取中位。首tile、完整GEMM时间、周期与实测频率不进入目标或预测特征，观察到的重叠也不作为输入。

两候选均为 `max(C, q0+q1*x)`，x为各自额外覆盖数，C固定为+0校准目标，`0≤q0≤C, q1≥0`。只使用p0/p16/p32，按三个条件等权最小化平方误差；代码分区枚举平坦段、一个激活点和两个激活点及边界，以有理数计算最优解集合，不靠优化器初值挑选参数。

参数唯一与留出预测唯一分别记录。例如32 B候选只有p16的x非零，p16>C时可有 `q0+32q1=y16` 的整条最优线段，参数并不唯一；但p64/p128的x均为0，预测都唯一等于C。128 B候选也可能存在被max平台隐藏的参数集合；冻结文件保存每个最优集合的顶点、参数范围以及留出预测区间，而不是任意选一个点后宣称参数已定。没有新增校准数据时，本节不填写真实q0/q1或留出预测值。

### freeze接口与最小依赖

入口沿用 [analyze_r10.py](../../../../../../microbench/gh200_resource_campaign/access_rules/analyze_r10.py)，仅依赖Python标准库和同目录的 `analyze_r18.py`、`analyze_r13_sm.py`、`v06_model.py`、`v06_run.py`、`v08_model.py`；本批dyadic路径不需要NumPy或SciPy。打包时这些依赖和本分析器一起纳入source哈希清单。

```bash
python3 analyze_r10.py --freeze-b-pitch --input RUN --output RUN/frozen/r10-b-pitch.json
# 管理者随后用原runner采样两个heldout；本命令不提交GPU。
python3 analyze_r10.py --score-b-pitch --input RUN \
  --predictions RUN/frozen/r10-b-pitch.json --output RUN/reanalysis/b-pitch-score-v1
```

freeze需要五个已知条件的 `cases.json`、`static_setup.json`、`environment.json`、`run_config.json`，源码/二进制清单与其文件，以及SASS哈希清单；样本**只读取p0/p16/p32三个目录**的plain/wide/stamped/dual记录，各十次成功trial，复用原数值、坐标、设备与容量检查。留出目录在freeze前不得存在，不读取留出测量。aligned C是校准参数，不是采样后的预测特征。

输出含 `status=frozen`、`frozen_unix_ns`、原environment的完整`gpu`字符串，以及以两个留出ID为键的`predictions`，兼容现有 `run_v08` 的heldout检查。两候选、校准误差/失败、全部最优参数集合、预测区间、目标和评分规则均保留；绑定模型、输入清单、校准样本、实际分析代码/依赖及二进制清单哈希，文件设为0444。已有冻结文件不覆盖。R10这是局部组件验证，不增加用户确认步骤，也不改变V09的人工作业边界。

### 预声明评分

同一目标、同一聚合顺序分别计算两个留出值。**每个候选独立要求三个校准点及两个留出点的最大绝对相对误差均≤5%**；两留出逐点报告有符号误差，不用它们重拟合。若最优参数不唯一而预测区间也不唯一，按区间两个端点中较大的绝对误差评分，不能看到留出后再选有利参数。校准失败候选仍输出预测和留出分数，但其局部判定保持失败；始终并列报告两候选，不事后挑一个改称整组通过。

评分只针对该dual观察协议下的后续主循环ns/Ktile。完整时间与逐trial dual/wide扰动作为诊断保留，不通过删去有效慢trial改变目标，也不据局部分数发布新的GEMM通过成绩。score核对冻结输入/代码身份，要求留出进程起始时间晚于冻结时间；生成新的评分目录，不改冻结文件和原测量。

CPU检查已覆盖给定覆盖数、参数不唯一而留出预测唯一、校准失败保留；临时合成样例另检查了仅三校准目录读取、两留出ID、只读freeze、评分及采样后拒绝freeze。它们不是新的GPU结果。`python3 analyze_r10.py --cpu-check` 可重复运行几何和参数检查。

### 局部冻结负结果：job738203

**两候选均未通过预声明的局部目标，保留失败，不回调参数。** job738203在独占的romeo-a057、GPU-43269fbc-449d-3e0f-908a-9c81229546d3上完成，同卡先校准再冻结、再测留出；没有拼接a043的旧常数。五条件×四变体×十进程共200个成功进程，819,200个保存输出值通过重放；source/bin/SASS/raw身份匹配。四个cfg_b二进制沿用eb49b87，均为168寄存器、16条静态HGMMA、无spill/C7510。

[冻结文件](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/frozen/r10-b-pitch.json)的SHA256为 `6fd226ef90f1a9d9224d1d6db769744eaece3befd69e98abf4352c4676d072c5`，与 `prediction_binding.json` 一致。冻结时刻为巴黎时间2026-10-09 04:04:50.203333（Unix ns `1791511490203333470`），晚于最后一个校准进程结束1.755716004秒；全部80个留出进程均在冻结后启动，最早晚0.243562114秒。回传解包后按tar原成员的0444恢复只读位，预测字节和SHA未变，没有重新生成冻结文件。

三校准目标p0/p16/p32分别为329.9967、439.1767、385.9533 ns/Ktile，锚点 `C=329.9967`。32 B候选参数不唯一，q0范围0～329.9967、q1范围3.411875～13.724271；它对p32仍预测C，校准最大误差14.4983%，已失败。128 B候选有唯一参数 `q0=66.613333, q1=6.652917`，校准误差为零；但在p64的线性分支只有279.5067，仍被max截到C。两候选的两条留出预测均唯一，不存在测后选择有利参数的余地。

| 候选 | 校准判定 | p64预测 / 实测，ns/Ktile | p64误差 | p128预测 / 实测，ns/Ktile | p128误差 | 局部判定 |
|---|---|---:|---:|---:|---:|---|
| 32 B覆盖 | 失败 | 329.9967 / 365.7233 | −9.7688% | 329.9967 / 331.7933 | −0.5415% | 失败 |
| 128 B覆盖 | 通过 | 329.9967 / 365.7233 | −9.7688% | 329.9967 / 331.7933 | −0.5415% | 失败 |

误差为预测/实测−1。p128单点接近锚点不改变整组失败；128 B校准拟合精确也没有保证p64迁移。被否定的是本组冻结的覆盖计数与max参数形式，不是对物理cache line或TMA流量作出的识别结论，也不是完整GEMM的通过或失败成绩。

dual/wide完整时间中位扰动按p0/p16/p32/p64/p128分别为+4.860%、+2.381%、+1.885%、+3.006%、+4.997%，四变体各条件CV最大2.610%。单trial配对仍有超5%者，p0与p128最大为+8.037%/+8.945%；p64配对范围为−0.606%～+4.974%。所有有效trial都保留，冻结评分继续针对既定dual观察窗口，不能把这些结果改称无打点阶段服务。

### p64的最小离线定位

以下只定位失败，不改变评分。比较p0与p64的实际CTA工作列表，十个trial的CTA编号、坐标、j及每CTA总tile数T均匹配。每进程按(j,T)对窗口取均值并除以16，再求相同trial编号的p64−p0差，最后对十个差取中位。它们属于前后两个采样阶段的不同调用；trial编号相同不表示同时发生或相同的硬件驻留状态。j从0起，表中只列冻结目标包含的后续j>=1。

| j / T | 每进程窗口数 | p64−p0，cycle/Ktile | p64−p0，ns/Ktile | ns增量的十trial范围 |
|---|---:|---:|---:|---:|
| 1 / 3 | 96 | +66.3281 | +37.2396 | +32.8333～+42.1458 |
| 1 / 4 | 36 | +64.0026 | +35.8611 | +26.1111～+40.3333 |
| 2 / 3 | 96 | +70.1533 | +37.7396 | +33.7708～+40.8333 |
| 2 / 4 | 36 | +88.5694 | +47.2222 | +39.6667～+61.6667 |
| 3 / 4 | 36 | +17.6771 | +9.4167 | +4.0556～+12.6111 |

五个分组、每个trial的cycle与直接ns增量都为正，差异不局限于最后一个tile。T=4的j=2每窗口增量最大，而最后j=3最小；按窗口数加权的trial均值计算，j=1/2贡献约96.8%的后续总增量。此处的分组中位不与冻结目标的全窗口中位强行相加。p128在对应位置接近p0，作为同一留出阶段的参照保留；不能据j/T分布单独命名物理请求放大、带宽下降或某个流水等待机制。本轮止于负结果，不扩矩阵。

[原评分](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/b-pitch-score-v1/b-pitch-score.json)与冻结文件保持原样。[本地独立重放评分](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/B-20261009-frozen-negative/b-pitch-score.json)完整重现两个候选；[位置分组、扰动和时间顺序](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/B-20261009-frozen-negative/localization.json)保存各trial分组值与全部样本哈希。另直接按原始dual数组的两个consumer交替次序解码，每调用恰有300个后续窗口，独立复现五个目标值及两候选误差；没有重拟合。

```bash
python3 microbench/gh200_resource_campaign/access_rules/analyze_r10.py --score-b-pitch \
  --input /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203 \
  --predictions /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/frozen/r10-b-pitch.json \
  --output <该run下新的reanalysis目录>
```
