# R18：OOB、有效地址零值与填零服务

**输入数值为零与 descriptor 越界填零具有不同的条件代价。** 同物理工作列表配对已在 cfg_a 的边界 cluster、cfg_b 的整 tile A/B 越界中观察到差异；cfg_c 早期首轮边界对照没有同类正惩罚。差异还依赖输出轮次，不能只给补齐 tile 乘一个常数。A/C 首轮补测改善了条件范围，但 cfg_c first-fill 在部分预测频率下仍不唯一。当前公式见 [RULES](RULES.md#v09-model)，可辨识性与阶段关系见 [R13](R13-async-retirement.md)。

## 作业与证据索引

| 作业 | 对照与作用 | 结果入口 |
|---|---|---|
| 736958，原 R17 | cfg_a 边界 cluster 的首轮窗口；历史修正已取代 | [匹配窗口增量修订](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R17-job736958/reanalysis/qualified-increments-v2/qualified-increments.json) |
| 737122，R18 | cfg_a/c 普通、OOB、有效零、部分边界、cluster1；24条件 | [规则与检查](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/reanalysis/formal-v1/rules.json)、[匹配增量](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/reanalysis/formal-v1/paired-tile-increments.json)、[轻量补查](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-lite-job737122-v1/reanalysis/local-independent/rules.json) |
| 737322，V08F | swizzle8 整 cluster 补齐，按 j/T 重分组 | [轮次与工作量重算](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/B-20261009-padding-rounds/padding.json) |
| 738296 / 738307 | cfg_b 固定逻辑输出与工作，只扩大 M/N 输入 map | [M 向配对](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R18-input-map-job738296/reanalysis/input-map-pairs-v1/input-map-pairs.json)、[N 向配对](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R18-input-map-n-job738307/reanalysis/input-map-pairs-v1/input-map-pairs.json) |
| 738203＋上两批 | 同卡有效地址＋whole-fill候选，保留有效tile慢窗口 | [填零候选与分项误差](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-valid-fill-v4/summary.json) |
| 738707 / 738865 | cfg_a/c partial/whole首轮填零与有效地址零配对 | [合并报告](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/summary.json)、[独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/verification.json) |

## 同物理工作列表：边界路径与数值零分开

737122，a043/GPU-099dda56，CUDA12.9.41、sm_90a、CUTLASS3.9.2、NDEBUG。cfg_a沿M测试普通1536、整块越界1408、偶数部分1504、奇数部分1376，N=2048；cfg_c沿N用同四个长度，M=3072。K=1024/8192；显式零保留1536的有效descriptor，末128行A或128列B写零。普通/整块越界/显式零具有相同物理tile列表、grid、stride和分配容量；cfg_a六stage、cfg_c四stage，FP16输入、FP32累加/输出。

| 配置/条件 | K1024 plain µs | K8192 plain µs |
|---|---:|---:|
| cfg_a / ordinary | 19.456 | 90.080 |
| cfg_a / oob | 20.256 | 108.192 |
| cfg_a / explicit_zero | 19.008 | 89.776 |
| cfg_a / partial_even | 19.184 | 92.016 |
| cfg_a / partial_odd | 20.448 | 108.560 |
| cfg_a1 / oob；partial_odd | — | 89.456；88.928 |
| cfg_c / ordinary | 31.104 | 172.240 |
| cfg_c / oob | 30.816 | 171.440 |
| cfg_c / explicit_zero | 31.120 | 171.856 |
| cfg_c / partial_even | 31.136 | 172.256 |
| cfg_c / partial_odd | 30.688 | 170.960 |
| cfg_c1 / oob；partial_odd | — | 90.096；90.032 |

cfg_a长K的OOB比普通慢20.1%，有效零与普通接近，不支持“值为零本身造成变慢”。cfg_c没有同类完整时间正惩罚，不能移植cfg_a系数。cfg_c切cluster1约90µs也不等于测出多播翻倍成本：cluster1×2将12×11逻辑tile补成12×12=144，关键CTA两轮；cluster1×1为132tile，只需一轮。

L窗口从FIRST_MMA到mma_tail后MAIN_END，包含等待与排空。先在每个进程内作clock64差，再匹配CTA/j/物理坐标求OOB−ordinary；不同进程的绝对clock64不相减。下表为十进程匹配增量的中位，括号为进程标准差。

| 长K首输出tile | 边界tile cycle | cluster伙伴 cycle | 同列/同行其他CTA cycle |
|---|---:|---:|---:|
| cfg_a OOB−ordinary | 21668 (1005) | 21776 (975) | 8091.5 (504) |
| cfg_a explicit_zero−ordinary | −90 (261) | −86 (278) | −128.25 (207) |
| cfg_c OOB−ordinary | 6.5 (131) | 1 (128) | −2.75 (136) |

cfg_a短K对应OOB/伙伴/同列其他为1954/1936/598.25 cycle，显式零边界为+1。次输出tile的cfg_a OOB差为短/长K **−4/−1.5 cycle**；同样零填充并非每轮都有惩罚。cfg_c长K次轮仍有 **+434.5 cycle**（标准差145.81），不能写成所有轮次严格无差异。cfg_a同列扩散与共享B竞争相容，但没有第二干预或计数器唯一定位物理路径。

24个plain条件各十进程、1966080个保存值通过独立复算；完整trace22条件合格。两个短K条件原扰动5.34%/5.47%不参与定量拟合，轻量版本仅去坐标写出、保留同四事件，八条件共655360个值通过，plain SASS逐字一致。轻量短/长K配对必须用自身基线，不与完整组混搭。[逐组增量](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-lite-job737122-v1/reanalysis/local-independent/paired-tile-increments.json)、[SASS核对](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-lite-job737122-v1/reanalysis/local-independent/plain-sass-comparison.json)。

两个K点只支持该范围的描述或插值假设，不足以定稳态硬件斜率；物理tile执行指令量与有效数学工作2MNK也须分开。

<a id="r17"></a>

## 前序 R17 的发现与已撤回推导

736958，a048/GPU-009a8880，19点、380进程。cfg_a边界cluster的OOB与伙伴首输出tile同样变慢，通常比偶数行对照多150～190 cycle/Ktile，同列其他CTA也慢；后续输出tile回到约514。只有部分越界也能出现较小惩罚；去掉cluster后这些例子不慢，但同时改变了多播和工作分配，不能单独归因。

旧推导把有限窗口平均cycle/Ktile当斜率，再叠旧截距，重复计入固定项；其中1408×2304×1024打点扰动5.44%也不合格。修订后仅用匹配物理位置的窗口差：

| OOB尺寸 | 普通对照M | OOB首窗 cycle | 同位置普通 cycle | 增量 cycle |
|---|---:|---:|---:|---:|
| 1152×1664×2560 | 1280 | 26821.75 | 20741.75 | 6080.00 |
| 1408×2304×4096 | 1536 | 42475.75 | 33118.75 | 9357.00 |
| 1408×2304×10240 | 1536 | 112127.75 | 84109.50 | 28018.25 |

第一行40个Ktile，额外窗口平均152 cycle/Ktile；它不自动成为跨尺寸斜率。多数类别只有一个合格K，1408×2304也只有两个，没有独立检查点。旧V06两例代入修订增量后的总时间误差为−5.38%/−5.64%，仍是测后点诊断，同列扩散未建模。被取代的671/596/707经验斜率和旧−4.7%/−6.7%成绩只供追溯，当前不用。

[合格增量修订](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R17-job736958/reanalysis/qualified-increments-v2/qualified-increments.json)保留完整对照与误差；原表和历史修正见[固定 cee978a 正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R18-cluster-boundary.md#r17)。

<a id="v08-padding"></a>
<a id="按轮次与-cta-工作量重算2026-10-09"></a>

## V08F 整 cluster 补齐：轮次与有效tile一起变化

737322、a043/GPU-099，swizzle8与1的比较同时改变物理工作量和每CTA轮数，不能视为单独OOB路径干预。三配置×padM/padN/padMN×K1024/4096共18条件；补齐增加0～3轮，完整时间多25%～80%，无需补齐的cfg_b 3072²仅+0.6%/−2.4%。V08固定ρ没有迁移支持。

旧汇总把全部round>0的pad/in合池，所得47～631 cycle/Ktile额外量混入轮次和工作量。重读每条件十份stamped记录，按 `(round,每CTA总tile数T,几何类别)` 分组，先逐进程取pad−in中位差，再跨十进程中位：

| 条件/分组 | K1024额外cycle/Ktile | K4096 |
|---|---:|---:|
| cfg_a padN，旧池化 | 190.25 | 383.55 |
| cfg_a padN，round2/T5 | 138.00 ±25.97 | 159.47 ±14.49 |
| cfg_a padN，round2/T6 | 48.92 ±20.61 | 31.98 ±10.93 |
| cfg_a padN，round5/T6 | 3.38 ±11.96 | −9.27 ±16.08 |
| cfg_c padM，旧池化 | 380.25 | 631.12 |
| cfg_c padM，round1/T3 | 516.27 ±28.95 | 593.14 ±11.62 |
| cfg_c padM，round1/T4 | 310.48 ±29.58 | 312.52 ±11.68 |
| cfg_c padM，round3/T4 | 35.00 ±25.31 | 50.34 ±8.63 |

±为进程差值标准差，不是置信区间。旧“随K增大”不适用于全部分组；padMN名称也不保证同时有两方向OOB，必须按坐标分类。有效tile同样变化：cfg_a padN/K4096/T6在round0/1/3/4为约520～533，round2/5为904.23/834.82 cycle/Ktile；同期pad为940.93/828.62。cfg_c padM/K4096/T4有效tile在round0/2约1030，round1/3达1623.85/1494.84。只乘pad罚项会漏掉有效tile慢窗。

相同round不等于实际同时执行；上述还是不同物理位置的观察比较，未固定供给、输出竞争和起点。原R18的边界cluster含有效伙伴，不能给V08F整cluster OOB直接套系数。[重算结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/B-20261009-padding-rounds/padding.json)保留逐进程值、旧池化重现和原扰动判定。

## 固定逻辑输出，只改变 source map

新配对先按逻辑GEMM构造完整参数，仅扩大mainloop输入tensor map的有效范围；输出descriptor/mask、scheduler、grid、容量和实际CTA工作列表保持不变。预留输入区在两路径都写显式零，区别是TMA从有效地址读取零，还是按OOB路径填零。逻辑pad标签不随map扩大消失，不能用扩大逻辑M/N代替此对照。

738296/738307均在a057/GPU-43269fbc，cfg_b、六stage、sw8、K1024/4096；每方向四条件×四变体×十进程。两个方向各655360个输出值通过回放，另检查314572800个扩展FP16值全零、逻辑输出外保持NaN哨兵，物理工作坐标列表一致。两批复用相同四个二进制，168寄存器、16静态HGMMA、无spill/C7510。

| 方向 | 逻辑M×N | 预留/map扩大 | K | OOB plain µs | 有效地址零 plain µs | 逐trial比值中位［范围］ |
|---|---|---|---:|---:|---:|---|
| A/M | 2304×4096 | M2304→3072 | 1024 | 49.744 | 39.920 | −19.98%［−21.51%,−17.20%］ |
| A/M | 同上 | 同上 | 4096 | 174.304 | 141.808 | −18.96%［−20.39%,−16.92%］ |
| B/N | 4096×2304 | N2304→3072 | 1024 | 49.200 | 39.952 | −18.87%［−20.26%,−16.57%］ |
| B/N | 同上 | 同上 | 4096 | 170.400 | 140.352 | −17.62%［−19.21%,−16.25%］ |

比值先逐trial计算再取中位，不能用两个时间中位的比值替代。M向wide/plain中位−0.55%～+0.12%、dual/wide−0.52%～+4.80%，仍有单对超5%（最大7.37%），不删记录；分项属于dual观察协议。

K4096的padM OOB五个j/T组约747～1005 cycle/Ktile，有效地址零约521～528，直接ns降低约118～251 ns/Ktile；padN对应约793～1005与521～529。K1024的位置变化也不同，例如padM j4/T6与j5/T6分别减少484.56与290.16 cycle/Ktile。

这组干预表明**输入descriptor的OOB路径会增加本批补齐成本**，并不分解内部零填充、供给/计算重叠和全卡竞争。A/B两个方向都受影响，接近的完整时间比例也不能证明同一物理填零速率。有效tile仍有位置变化、周期与直接ns可能不同，统一ρ和全局频率不足以解释所有窗口。

<a id="h03-partial-fill"></a>

## h03 的部分越界不能混入纯行距反例

[R10 四角](R10-layout-cache.md)的aligned j3/T4总体窗口为623.499 cycle/Ktile，高于j2/T3的540.905，尽管软件有效地址需求更少。按实际坐标重读[dual tiles.csv](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-joint-pitch-job738169/reanalysis/B-20261009-matched-joint-final/tiles.csv)后，j3/T4每进程60个窗口中19个含B的N向部分越界：

| j3/T4几何 | 窗口数 | 主循环cycle/Ktile |
|---|---:|---:|
| M/N完整 | 38 | 587.145 |
| 仅M尾部 | 3 | 599.521 |
| 仅N尾部 | 18 | 702.625 |
| M/N都有尾部 | 1 | 706.375 |

j2/T3没有B部分越界，完整/仅M尾部约540.969/541.354。各组中位不能加权重构总体中位；分开几何后仍有位置差异，先前访问与并发未固定。另一张卡的whole A/B map配对支持“未读源字节不等于零服务成本”，但不直接确定h03部分B越界价格，也不证明partial/whole可按字节线性外推。

<a id="valid-fill-candidate"></a>

## 有效地址＋目的填零候选：改善与残差同时保留

同a057卡的B曲线和M/N map共13条件用于测后开发：有效地址payload包含显式零，源覆盖只数有效地址；目的ZA/ZB数由OOB路径提供的nominal tile量。本批每操作数只有0或整块16 KiB，尚未证明partial-fill耗时与填零字节成正比。扩大map后有效源需求增加、目的fill归零，模型必须保留这个区别。完整定义与现行式只在 [RULES](RULES.md#v09-model)维护。

[旧有效地址＋填零报告](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-valid-fill-v4/summary.json)分别拟合cycle与直接ns；ns主表显式假设1.6 GHz，没有目标窗口实测频率回填。每case等权，各j/T/几何分组平分权重；全部是开发诊断。

| 目标组 | cycle 中位/最大/RMS误差 | ns 中位/最大/RMS误差 |
|---|---|---|
| 全部137个later组 | 2.18% / 24.22% / 6.03% | 4.30% / 25.41% / 7.13% |
| 20个当前whole-OOB组 | 5.54% / 14.10% / 7.18% | 4.73% / 13.21% / 6.01% |
| 25个B行距组 | 4.23% / 16.38% / 7.78% | 3.02% / 16.59% / 7.73% |
| 117个当前有效地址组 | 1.68% / 24.22% / 5.80% | 4.21% / 25.41% / 7.30% |

两形式活跃分支秩均6/6，不代表物理服务唯一；26个未参与拟合first组最大转移误差为13.70%/9.79%。最大later失败是padN OOB run、K4096、j2/T5的**有效地址**组：ZA=ZB=0，实测695.743 cycle/Ktile、432.792 ns/Ktile，预测527.258、322.826，低估24.22%/25.41%。加当前fill价格解释不了这个慢窗；padM也有类似现象。

同调用OOB/T6的j4相对j1，padM K1024/4096的周期变化+15.38%/−8.81%，直接ns仅+3.45%/+1.31%；padN相应+16.60%/−11.52%与+6.59%/−2.78%。部分ns窗口更稳定，不代表频率模型完成；改假设1.8 GHz后ns整体RMS/最大误差为8.90%/31.91%。只用K1024训练、测试K4096仍有约24%～25%最大误差；反向训练缺非零B覆盖，15组缺参数不评分。

后来cfg_b旧13＋新六条件联合时，这些失败继续存在：padN K4096 j2/T5有效组仍低估23.32%，单CTA whole B-fill最大高估76.07%。增加软件波次global-fill项只把ns最大误差从25.41%降到21.48%，未解释慢in窗口，未接入候选。[联合报告](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-b-joint-supply-v1/summary.json)、[未采用的global-fill诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/manager-shared-fill-v1/summary.json)。

<a id="first-fill"></a>

## A/C 首输出tile的 partial/whole-fill 配对

738707/738865均在a057/GPU-432，每批四条件×四变体×十进程，直接复用738496的12份binary/SASS且SHA相同。A为logical832×4096、storage1024×4096、mapM832/1024；C为logical1536×2624、storage1536×2816、mapN2624/2816；均sw1、K1024/4096，各配对保持输入数值、工作列表、容量和输出相同。

每个OOB条件的首轮，A有16个partial A-fill 8 KiB和16个whole A-fill 16 KiB；C有6个partial B-fill和6个whole B-fill。扩大map后目的fill归零。仅将新四点加入对应A/C first，B19与全部later参数原样保留，避免用first数据默默重定later。

| 对象 | 显式1.6 GHz下first中位/最大/RMS误差 | 仍存在的问题 |
|---|---|---|
| cfg_a新四点 | 3.89% / 29.15% / 8.56% | partial-fill组中位绝对误差K1024/4096为20.34%/16.49% |
| cfg_c新四点 | 3.36% / 5.76% / 3.50% | first为4/5秩，B-fill价格仍非唯一 |

A first满秩5/5，但有效源价格落在非负边界不能解释为物理读取免费。C first的B-fill代表价格32.2988 ns/KiB只是等价解之一，条件范围6.3092～33.4899。A新later仅诊断，逐case中位误差的中位/最大8.01%/9.64%，单窗口最大40.30%；C新四点全为单tile，没有later证据。

固定1.6 GHz下的局部成绩不能覆盖任意频率：cfg_c已见whole B-fill首窗和V09 G2在更高预测频率下转入未唯一识别的供给方向，仍有拒绝项，详见 [R13 可辨识范围](R13-async-retirement.md)。不能用这两批补测宣称first-fill全部闭合。

[模型与范围](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/models.json)、[分项残差](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/first-fill-residuals.csv)、[完整来源](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/provenance.json)保留精确训练行和原样本绑定。历史比例、拟合参数和命令由这些报告承担，本页不复制成另一套现行规则。

## 适用边界与复核

已隔离的是本批descriptor输入路径，未分别定值物理填零速率、L2/HBM服务和内部队列。边界cluster、整cluster OOB、partial-fill、first/later和有效tile受邻近OOB影响须分开；只有静态有效源量与填零量还不足以解释全部窗口。

复核入口为 `analyze_r18.py --input RUN --output 新目录`；V08F重分组加 `--v08-padding`，source-map配对加 `--input-map-pairs`。原轻量组使用 `run_r18_lite.py` 冻结协议。历史完整设计与命令见[固定 cee978a 旧正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R18-cluster-boundary.md)。本次仅整理论证归属，旧results、预测与评分保持原样。
