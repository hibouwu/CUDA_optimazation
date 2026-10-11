# R18：OOB、有效地址零值与填零服务

**输入数值为零与 descriptor 越界填零具有不同的条件代价。** 同物理工作列表配对已在 cfg_a 的边界 cluster、cfg_b 的整 tile A/B 越界中观察到差异；cfg_c 早期首轮边界对照没有同类正惩罚。差异还依赖输出轮次，不能只给补齐 tile 乘一个常数。A/C 首轮补测改善了条件范围，但 cfg_c first-fill 在部分预测频率下仍不唯一。V09 的 partial 失败主要落在主循环，且影响传播到完整 tile；不能增加一个只收 partial tile 的常数。固定输出四点不支持单一精确增量规律；cluster-max在观测baseline上有受限近似价值，但只替换fill输入的完整供给重拟合未改善V09，关键CTA定位尚未解决。当前公式见 [RULES](RULES.md#v09-model)，可辨识性与阶段关系见 [R13](R13-async-retirement.md)。

## 作业与证据索引

| 作业 | 对照与作用 | 结果入口 |
|---|---|---|
| 736958，原 R17 | cfg_a 边界 cluster 的首轮窗口；历史修正已取代 | [匹配窗口增量修订](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R17-job736958/reanalysis/qualified-increments-v2/qualified-increments.json) |
| 737122，R18 | cfg_a/c 普通、OOB、有效零、部分边界、cluster1；24条件 | [规则与检查](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/reanalysis/formal-v1/rules.json)、[匹配增量](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/reanalysis/formal-v1/paired-tile-increments.json)、[轻量补查](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-lite-job737122-v1/reanalysis/local-independent/rules.json) |
| 737322，V08F | swizzle8 整 cluster 补齐，按 j/T 重分组 | [轮次与工作量重算](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/B-20261009-padding-rounds/padding.json) |
| 738296 / 738307 | cfg_b 固定逻辑输出与工作，只扩大 M/N 输入 map | [M 向配对](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R18-input-map-job738296/reanalysis/input-map-pairs-v1/input-map-pairs.json)、[N 向配对](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R18-input-map-n-job738307/reanalysis/input-map-pairs-v1/input-map-pairs.json) |
| 738203＋上两批 | 同卡有效地址＋whole-fill候选，保留有效tile慢窗口 | [填零候选与分项误差](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-valid-fill-v4/summary.json) |
| 738707 / 738865 | cfg_a/c partial/whole首轮填零与有效地址零配对 | [合并报告](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/summary.json)、[独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/verification.json) |
| 739095 | P1同输出map配对：首轮partial及其完整cluster伙伴同步缩短，later近零响应 | [当前分析](#v09-boundary-failure)、[配对结果](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/C-controls.json) |
| 739529 | 同输出四map：单一增量不精确，max保留近似候选，later近零 | [当前辨识](#哪个静态特征能够表达这个作用)、[四点结果](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/C-fill-discrimination.json) |

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


<a id="v09-boundary-failure"></a>

## V09 partial：先区分输入路径与输出窗口

V09 的 P1（cfg_a，960×2816×4096）和 P2（cfg_c，4096×2240×4096、swizzle8）各复核十个 dual 进程。下表先在每进程内按逻辑输出几何分组，再取进程中位；模型误差是配对窗口差的中位，不能用表中两个独立中位数相减替代。L 为 FIRST_MMA→MAIN_END，E 为 merged PERMIT→DONE，不代表物理写完成。

| 条件 / 轮次 / 几何 | 实测 L ns | 模型 L 残差 ns | 实测 E ns | 模型 E 残差 ns |
|---|---:|---:|---:|---:|
| P1 / first / partial | 23712 | −4233 | 880 | +639 |
| P1 / first / real | 20648 | −1169 | 1120 | +399 |
| P1 / later / partial | 18304 | +1045 | 864 | +171 |
| P2 / later / OOB | 46168 | −3562 | 2000 | −86 |
| P2 / later / partial | 43248 | −1330 | 2352 | −36 |

P1 首轮 partial 的长窗口出现在主循环，输出项反而高估；later 的 L 残差方向相反。实际最后完成者还可能完全处理 real tile：trial0 的工作坐标为 `(6,4),(2,21)`。因此边界影响不能只贴在边界 CTA 上。P2 的 OOB 主循环也比模型慢，最大时长与最后完成者的关系见 [R19](R19-critical-cta-tail.md#v09-critical-failure)。

原 738707/738865 的配对保持逻辑输出、分配、初始化和工作列表不变，只扩大输入 descriptor。本次复用同角色 cycle/ns 端点：cfg_a K4096 的 first partial、role2 窗口从 OOB 的约25.040 μs降到有效地址零的约21.280 μs，完整 plain 时间下降12.78%；cfg_c 对应完整时间只变化+0.32%。这支持**输入路径有条件代价**，不支持统一填零系数。旧 cfg_a 的 M832 同时有 partial 与 whole OOB，不能把它的幅度移植到只有 partial 的 P1。

P1 的固定输出 map 配对已在 job739095、a058/GPU-ef8692f3 完成：logical M960、storage M1024、zero M960，只改变 input_map_m=960/1024。两点各十个 plain 与 dual 进程，plain 中位 **51.824→47.744 μs（−7.87%）**，dual **53.072→48.752 μs（−8.14%）**；dual 相对 plain 扰动2.41%/2.11%，plain CV0.81%/1.28%。新卡同期比较，不把这些绝对值代入旧卡 V09。

逐 CTA、同角色、同坐标配对说明差异集中在首轮边界 cluster。下表为 role2 的直接 L ns，差值先在匹配窗口相减、再按进程取中位；两个条件的独立中位数之差不必等于配对差。

| 轮次 / 几何 / CTA工作量 | OOB输入 L ns | 有效地址零 L ns | 配对差 ns |
|---|---:|---:|---:|
| first / partial / T2 | 24160 | 20080 | −4128 |
| first / real，与partial同cluster / T2 | 24128 | 20080 | −4144 |
| first / real，内部cluster / T2 | 20168 | 20112 | −80 |
| later / partial / T2 | 18240 | 18256 | 0 |
| later / real，与partial同cluster / T2 | 18240 | 18280 | +16 |
| later / real，内部cluster / T2 | 18232 | 18272 | +32 |

T1 的 first partial 与完整伙伴也分别缩短2.768/2.800 μs；role1 同方向。T2 first partial/伙伴的配对差在十个进程中均小于−3.58/−3.64 μs，内部real的差则跨过零。各组 E 的配对中位变化最多几十ns，不能解释约4 μs的总收益；later L接近globaltimer量化尺度。**这支持首轮 descriptor 输入边界 cluster 的条件代价，也排除了“只因输出面积较少而变慢”的解释。** 同cluster完整伙伴受到同量级影响，不能只按partial自身字节收费；静态工作坐标单独也不能解释map改变后的缩短。

观察仍未区分填零硬件、cluster同步和共享供给的具体物理成本，也不能把首轮差值复制到later。[同卡配对、关键CTA与原样本绑定](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/C-controls.json)。

<a id="boundary-candidate-features"></a>

### 哪个静态特征能够表达这个作用？

特征只从输入形状、descriptor范围、静态scheduler和multicast mask构造。cfg_a的A mask仅含本CTA，cfg_c的B mask也仅含本CTA；对应的mask最大值/总和退化为自身fill，不能表达完整伙伴的额外代价。完整cluster聚合能覆盖伙伴。当前两rank配置确有下面源码所示的共享full/empty依赖，但聚合特征本身不能证明实测增量来自其中哪一个等待。

| 每Ktile的cluster fill模式 | cluster最大 KiB | cluster总和 KiB | any_fill指示 |
|---|---:|---:|---:|
| 旧A/C首轮校准：partial 8、whole 16 | 16 | 24 | 1 |
| V09 P1 / 新P1：完整伙伴0、partial 8 | 8 | 8 | 1 |
| V09 P2的整cluster越界：16、16 | 16 | 32 | 1 |
| map全部覆盖 | 0 | 0 | 0 |

原卡738707/738865的三个cluster列只有秩1。把十进程内的配对L差减去同j/T内部real参照，再分别拟合三种单变量形式，三者的拟合值完全相同，不能从较好的残差选出max、sum或any。cfg_a first自身/mask的加权RMS为35.70 ns/Ktile，cluster三种形式均为29.41 ns/Ktile；这支持扩大作用范围，却没有确定聚合方式。

量纲也不能跳过：若采用any_fill，原卡K1024/4096的描述性价分别为98.875/60.938 ns/Ktile，换成每窗口为1582/3900 ns；两者都不是稳定常数。cfg_c first的差很小且随K变号，later则没有这类固定输出的A/C校准。V09及新卡只作开发诊断，不参与这些系数拟合。因此目前交付静态作用范围和可辨识限制，不给出新的通用填零价格。

四点辨识已在job739529完成，与P1配对同a058/GPU-ef8692f3。固定logical M832/N2816/K4096、storage M1024/zero M832，只改变map M；两项参照先采，四个目标各十次plain/dual，总计120个成功进程。受影响集合始终由静态map832确定，按相同CTA/角色/j/T与map1024配对，再扣除同期内部real的变化。下表选role2、first/T2中原本自身填零8KiB的成员；其逻辑输出始终partial，map896以后只是**输入不再填零**，不能改称完整输出tile。

| map M | cluster A fill KiB | 本成员输入状态 | plain μs | first L额外量 ns | later L额外量 ns |
|---|---|---|---:|---:|---:|
| 832 | (8,16) | 自身填零 | 57.968 | 8752 | 16 |
| 896 | (0,16) | 无填零伙伴 | 54.880 | 6664 | 64 |
| 960 | (0,8) | 无填零伙伴 | 51.840 | 4064 | 16 |
| 1024 | (0,0) | 已清除原边界 | 47.680 | 0 | 0 |

**max、sum和any三个单一价格都不是这里的精确增量规律；这不等于它们不能作为有误差的条件近似。** 832→896的max和any均不变，first/T2主循环却缩短约2.50μs；同期内部real也缩短0.624μs，扣除后原8KiB成员仍缩短1.944μs。896→960与960→1024对应扣除后的配对缩短为2.784/4.064μs。三次总fill都少8KiB，变化却不相等；第一步的十进程范围为1.504–3.056μs，最后一步为3.392–4.432μs，不支持统一线性sum价格。内部real随map变化，不能把原始总差全部收费给边界。

两个合作角色同方向：对同CTA与同端点计算相对map1024的L差，两角色差的进程中位绝对值为48–88ns，远小于数微秒响应；E额外量只有−32至+128ns。dual相对plain扰动1.49%–2.44%，各类进程CV最大1.85%。later主循环只剩几十ns差，first额外量又随T1/T2分组显著不同；这些分组同时改变工作位置与并发条件，尚不能把T单独命名为原因。

role2 first/T2的max/sum/any描述性增量拟合RMS分别为0.844/0.593/1.764μs，三项混合降至0.289μs。但四点仅提供三个非零特征模式，混合模型能插值三个均值；秩3与更小训练残差均不证明通用规则，也没有理由立即增加混合参数。

为了判断单一特征是否仍有近似价值，将739529拟合的价加到**同trial、同CTA的map1024观测first L**上，再加该条件观测到的内部real偏移，比较完整L；739095的P1只作同卡、不同logical M的开发迁移，不重拟合。下表是role2/T2逐进程中位绝对相对误差再取十进程中位，不是最坏误差。

| 诊断条件 / 受影响成员 | cluster-max | cluster-sum | any_fill |
|---|---:|---:|---:|
| 四点map832 / 自身fill | 5.65% | 2.09% | 9.86% |
| 四点map896 / 自身fill | 3.70% | 3.44% | 2.63% |
| 四点map960 / 自身fill | 2.05% | 4.49% | 10.12% |
| 739095 P1 / 自身fill | 2.56% | 5.65% | 10.21% |
| 739095 P1 / 无fill完整伙伴 | 2.37% | 5.45% | 10.19% |

max对P1典型窗口保有近似价值；若不加入目标条件的观测内部real偏移，两类P1误差为3.37%/2.92%，sum为6.23%/5.98%。不过P1单窗口最大误差仍达20.96%（max）和17.18%（sum），不能用典型值代替误差界。

条件排序再保留目标观测entry，以map1024的实际CTA内部时长加候选first增量，其余阶段沿用baseline。三个候选对四点map832/896/960各命中4/2/3次，对P1均只命中2/10；P1所选者距实际最晚退出的中位缺口3.664μs。它们都能指出受影响cluster，单一价格仍不能区分cluster类别内部的时长排序。

这些步骤明确使用观测baseline及部分目标观测量，只回答条件近似的价值，**不是可部署预测或新留出成绩**。下面将cluster-max放回完整供给模型，检验不依赖这些观测baseline时是否仍有收益。

### 只改静态fill输入的完整first重拟合

严格复用冻结r2的cfg_a first **2028行**，仅把X的A/B自身fill两列换为完整cluster最大fill；其余X、L_ns、校准计算下界与数据集合保持不变。冻结基线、同CPU重拟合自身fill基线、cluster-max候选并列；cfg_b/c、later、时钟、输出和plain转换参数全部保持原样。V09、739095与739529目标均未加入拟合。

| 比较项 | 冻结r2 / 自身fill重拟合 | cluster-max重拟合 |
|---|---:|---:|
| first校准窗口RMS | 5.620% | 5.134% |
| cfg_a全部9条件总时间中位 / 最大 | 3.006% / 10.285% | 3.167% / 10.236% |
| cfg_a全部9条件总时间RMS | 5.562% | 5.565% |
| P1自由总时间有符号误差 | −9.375% | −9.412% |
| P1 first L中位绝对误差 | 6.168% | 6.268% |
| P1冻结精确候选数 | 38 | 38 |

自身fill重拟合重现冻结结果，排除了不同CPU重拟合导致的主要变化。所有9条件的自由预测，以及逐进程代入观测频率后的预测，都通过基于**相应变换后校准行**构造的非负等价参数LP支持检查；通过支持检查没有带来预测改善。代入观测频率后，P1总时间误差仍由−10.207%变为−10.247%，first L误差由7.319%变为7.423%，不是简单换频率就能挽回。

原因在组合形式里可以直接计算：P1的cluster最大fill为8KiB，候选给出的服务项约`8×23.547=188.4 ns/Ktile`，低于`512/f≈299.7 ns/Ktile`的计算下界。当前`max(计算,供给)`把完整伙伴新增的fill输入完全屏蔽，32个受影响CTA与内部CTA仍取同一计算分支。重拟合主要把共同window项从298.816改为277.086ns，不能表达实测数微秒的边界差。因此**仅替换静态聚合特征、保持原max递推结构，并不能修复P1**；不能把前一节观测baseline上的近似收益当成完整模型已改进。

现行时间模型不启用这个候选。保留静态cluster作用范围与完整重拟合反例；模型选择计算下界分支不证明硬件没有等待，不能通过调大fill系数强行越过下界。[完整组合、逐进程分项、关键CTA类别与LP区间](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/C-first-cluster-model.json)；这些均为V09开发诊断，不是新留出成绩。

### 已能定位到哪个事件区间？

继续使用739529/739095原始端点，不重新拟合。下表均为role2/T2、相对同trial map1024并扣除同期内部real的进程中位差。S是producer首工作→该consumer的FIRST_MMA，属于同CTA跨角色准备窗口；L和E则分别由同一consumer的两个端点相减。

| 对象 | S额外 ns | L额外 ns | E额外 ns |
|---|---:|---:|---:|
| 739095 P1 / partial自身 | 16 | 4000 | 24 |
| 739095 P1 / 完整伙伴 | 16 | 4000 | −8 |
| 739529 map832 / 原8KiB成员 | 96 | 8752 | 48 |
| 739529 map896 / 同成员，输入已无fill | 16 | 6664 | 64 |
| 739529 map960 / 同成员，输入已无fill | 8 | 4064 | 32 |

MAIN_END→PERMIT的配对中位变化为0，最终consumer退出相对末次EPI_DONE的变化为0–16ns。两个伙伴共有的数微秒增量落在L，不能移到首段S或输出E。以上中位数来自分别分组，不要求相加等于首tile完整跨度；每个原始角色窗口的加法恒等式已核对。

归档CUTLASS源码给出了L的真实含义：`FIRST_MMA`在**初始consumer_wait返回后**、`warpgroup_arrive`和首批GMMA之前；`MAIN_END`在`mma()`及`mma_tail()`都返回后。L包含后续Ktile的full等待、GMMA发射/commit、`warpgroup_wait<K_PIPE_MMAS>`、buffer release以及末尾`warpgroup_wait<0>`和剩余release。现有最终exit_tail是输出之后的退出窗口，**不是GMMA尾排空**。缺少mma返回端点，所以现有数据还不能把K循环等待与尾排空分开。共同增量支持cluster相关的延迟暴露，不证明某一条硬件等待是原因。

这正是当前max形式的缺口：它把整个L视为稳定供给与计算中较慢者的K倍，而FIRST_MMA之后可能仍有启动过渡、阶段依赖或尾段暴露。观察到later输出tile代价近零，并不证明first输出tile内部只有一次固定延迟；这两批都为K4096，没有提供K内持续性的直接分解。

若后续把稀疏观测推广到cfg_a边界，按其合作角色语义最少需要五个新端点：一处放在`mma()`返回、进入`mma_tail()`之前，以现有MAIN_END分出尾段；另四处只围住一个选定稳态K迭代的`consumer_try_wait`之前、`consumer_wait`之后、`warpgroup_commit_batch`之后和`warpgroup_wait`之后。这样分别得到full等待、发射加commit、组等待，以及最终tail段，不把发射窗口命名为纯commit指令成本。可用同样的五点布局分别取稳态首迭代和中点，保留两consumer及受影响/内部cluster对照；不逐K打点。这只是所缺端点建议，当前先由cfg_b stage稀疏观测验证观测方法，本页不启动新的cfg_a采样。若额外时间集中在早段而中点无差，支持启动过渡；若中点等待也持续增加，再用同几何的K变化检查累积；若集中在mma返回→MAIN_END，则优先检查尾段。单个中点样本仍不能证明全K循环都遵循同一服务率。

[匹配端点分布、源码SHA与插入位置](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/C-endpoint-localization.json)。本次只用已有事件定位，没有新加时间项或GPU任务。

[四点配对、直接变化、两角色一致性与拟合残差](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/C-fill-discrimination.json)、[旧校准候选与静态范围](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/C-candidate.json)、[逐CTA静态特征](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/C-static-features.json)。

[逐窗口与逐进程结果](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/C-summary.json)、[原同输出配对复核](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/C-old-input-pairs.json)。同角色窗口与 merged 模型残差分开保存，前者用于周期/时间关系，后者用于比较递推模型。


### cluster伙伴是否在同一调用里共同延长？

按静态scheduler的逻辑cluster配对两个rank，比较**同一进程、同一consumer角色、同一j/T**的FIRST_MMA和MAIN_END。对照把rank1固定轮换到下一个同j/T、同逻辑几何与fill类别的cluster，没有按时间挑选“最近邻”。下表为role2、first/T2的进程中位；“增量差”来自分别与同trial map1024配对的两个rank，基线仍是另一个独立进程，不能称为同调用处理效应。

| 条件 / 配对 | FIRST_MMA差 ns | MAIN_END差 ns | 两rank的L增量差 ns |
|---|---:|---:|---:|
| P1 map960 / 实际完整＋partial伙伴 | 0 | 32 | 48 |
| P1 map960 / 轮换非伙伴 | 128 | 3200 | 2864 |
| 四点map832 / 实际partial＋whole-OOB伙伴 | 0 | 32 | 48 |
| 四点map832 / 轮换非伙伴 | 384 | 6448 | 6352 |
| P1 map1024 / 原边界伙伴 | 0 | 48 | 0（基线） |

P1两个rank的L增量分别约4.144/4.128μs，而在50个受影响cluster/进程样本中，FIRST_MMA差最大仅32ns。因此这批数微秒增量不是一个伙伴在FIRST_MMA之前晚到数微秒造成的；延长在实际cluster内共同变化，不能只描述为不同位置恰好具有相同类别均价。

不过map1024伙伴本来也紧密对齐，端点接近只证明轨迹相关。共享供给、相同执行路径、K内full等待、组等待或尾排空均可能造成相同起止；现有端点不能证明一次共同stall，不能判断哪个rank等待另一个。`L差=结束差−开始差`是恒等式，其相关性不能再算独立因果证据。

静态源码还证明了具体共享依赖，而不只是一种统计关联。cfg_a的`PipelineTmaAsync`有6个stage，每CTA有256个consumer线程，即两个warpgroup；empty到达数为`(cluster_M+cluster_N−1)×2=(2+1−1)×2=4`。`consumer_release`向同row **或** 同column的rank发送arrival，所以这个2×1配置的每个empty barrier必须等两rank、每rank两组consumer共同释放。它不是只等待本CTA。

producer acquire先等empty，再由本地leader对full设置期望字节。A在cfg_a不多播，B沿cluster-M分片并多播给两个rank：每rank本地full要收齐A的16KiB，以及两个rank分别发起的8KiB B片，总计32KiB。逻辑越界提供的填零仍属于目的数据完成协议，不能因为有效源量减少就删掉这部分full依赖。只有对应full代次完成，consumer才能继续；consumer的safe release又在`warpgroup_wait<1>`之后，尾段另有`wait<0>`。producer/consumer的pipeline状态跨输出tile继续推进，不能在每个输出tile把所有stage重置为空。

下一候选最少需要保留以下**事件约束**，不增加一个cluster罚项：

- 复用某stage的新代次，须等前一代次由两rank、两consumer组安全释放；起始6个空stage单独初始化。
- rank的full完成依赖本地A、两个B分片的事务完成及本地producer到达，不能由本CTA的平均源需求独立决定。
- GMMA发射同时受full就绪和本地GMMA在途限制；stage释放跟随安全消费，允许一条rank的延迟通过empty回压影响另一条rank的后续供给。
- 最后GMMA排空与producer退出前等待全部empty各自保留，不能合并成一项任意tail常数。

因此“A本地变慢→本rank释放变晚→两个producer可复用stage变晚→后续B/full一起变晚”在源码依赖图上可行，与两rank共同延长一致。它尚未证明实测4μs沿这条链传播，也没有给链中每段定值；单CTA平均供给取max会丢掉有限stage和跨rank代次关系，仅增加cluster字节价格不能恢复这些约束。

后续若使用前述稀疏布局，关键是两个rank保存同一K迭代的wait前/后和commit/group-wait后时刻，才能看出是否一方较晚到达、另一方在相应区间等待。若差异确实集中在full等待，仍需该pipeline阶段的producer acquire/issue及consumer release端点，才能区分未供给与晚伙伴释放导致的回压；当前不提前增加这些打点，也不拟合cluster罚项。[同调用配对、静态非伙伴对照与逐进程来源](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/C-cluster-coupling.json)。

### 两 rank 的显式依赖与剩余可辨识范围（2026-10-11）

`output_model.cfg_a_rank_dependencies` 将当前 cfg_a 源码写成纯依赖图：全局 K-group `g` 跨输出连续，slot 为 `g % S`，generation 为 `g // S`。本 rank 的 full 需要 producer 的一次 `arrive_and_expect_tx`、本地 A 的16KiB完成及两个 rank 各8KiB B分片到本目的地的完成，合计32KiB。B分片每个由两个4KiB atom构成。slot重用需要上一代两个rank、每rank两个consumer WG共四次empty arrival；release发出与远端可见是不同节点。

group g 的稳态安全释放需要 commit(g+1) 和 g 已退休；每个输出最后一组单列 `wait<0>`。MAIN_END在最后release发出之后，不保证远端empty已可见。图中没有填入供给、GMMA、仲裁或观察器服务价，也不复用到cfg_b/c。

离线入口 `analyze_r18.py --rank-dependencies` 读取739095、739529及四个V09条件，共100个dual进程，逐值重放4096个保存输出并核对原始SHA。静态选择每种完整工作/fill序列的首个cluster，对370个“进程内cluster轨迹”施加已有FIRST/MAIN/PERMIT/DONE/final端点，零最小延迟约束均可行。这仅证明粗端点没有与图冲突。以P1的静态CTA6/7为例，g6 full、mma返回和末组排空端点的区间宽度中位均为20224ns；producer尾部与D全局可见仍无有限上界，不能由可行性判定回压原因。

同样的静态边界类别还具有不同幅度。下表固定739095的logical M960/storage M1024，列出role2、first/T2的 map960 L减map1024 L；每个CTA先逐trial配对，再取十进程中位，未扣内部CTA偏移。

| 完整伙伴 / 自身partial CTA | 完整伙伴差 ns | partial差 ns |
|---|---:|---:|
| 6 / 7 | 96 | 80 |
| 14 / 15 | 4528 | 4576 |
| 22 / 23 | 4240 | 4208 |
| 30 / 31 | 1600 | 1616 |
| 38 / 39 | 4256 | 4192 |

因此，单个cluster类别的固定罚项不能决定类别内部的最后完成者；仅测类别中第一对也可能漏掉强响应。原逐进程配对与全部CTA残差保存在[新独立分析](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R18-cluster-fill-discrimination/reanalysis/C-20261011-rank-dependencies-v5/README.md)，未修改V09候选集合。

一个非拟合的构造性反例进一步限定了现有证据：固定同一套示意服务价，将同一B源对两目的地的额外延迟放在g6或g63，得到完全相同的两轮FIRST/MAIN端点、首轮两个rank的L共同增加2520ns、后轮L恢复；但commit16的增量分别为2520ns和0。这不能说明真实瓶颈就是B，也不能把示意价格用于预测；它证明现有粗端点不能唯一定位延迟位置。单点局部A延迟的另一个示例虽产生首轮共慢，却留下1480ns的rank间MAIN差，未复现已有近对齐现象；该失败只适用于这套固定价格和干预，不排除其它A路径解释。

已向主集成交付[最小稀疏观察请求](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R18-cluster-fill-discrimination/reanalysis/C-20261011-rank-dependencies-v5/minimum-observation.md)。第一步覆盖全部五个静态受影响T2 cluster及一个内部对照，记录四个固定commit切点和mma返回；若区间指向循环内等待，再申请对应代次的full/empty/release观察。该阶段用于辨识，不计作新的局部留出或模型通过。

### empty arrival 的实际发送线程

cluster2×1的有效发送者由`spread_arrivals_to_warpgroup`决定。其`Swizzle<2,0,-2>`在4×4线程布局上分散arrival，再按cluster大小及同row/column裁剪；不能把cfg_b cluster1的128线程arrival模式复制过来。将冻结函数原body抽出、保留原CUTE布局在CPU执行，与纯模型枚举一致：

| 每个源rank中的consumer WG | 发往rank0的physical thread | 发往rank1的physical thread |
|---|---:|---:|
| WG0 | 128 | 168 |
| WG1 | 256 | 296 |

两rank均采用此表，因此每个目的barrier仍需四次arrival。WG内分别是warp0/lane0与warp1/lane8；只记录WG leader128/256会漏掉发往rank1的路径。归档dual SASS在`1fe0`的wait1后执行`1ff0`稳态arrival，`2050`的wait0后执行`2120`末组arrival；它们是predicated指令，不能据leader的后clock认定另一个warp已emit或远端已可见。

图新增`empty_emit_to_r`作为实际发送事件，原`empty_to_r`保留为目的可见事件；两者之间没有定价。producer的`acquire`是empty等待返回，`arrive_expect_tx`随后发生，新增`acquire_return`表示完整调用返回；调用后clock还包含expect_tx路径，不能精确锚定empty等待返回。

第二阶段请求据此修订为两rank各四个实际发送线程的自身arrival前后界，绑定同一全局group/slot/generation/phase及目标producer下一代acquire。跨rank只比较globaltimer ns，clock64保留在各CTA本地。若四个emit分别在`[b_i,e_i]`、完整acquire调用后戳为A，则最后emit保守位于`[max(b_i), min(max(e_i), A)]`；某个后clock晚于A不构成失败，实际emit仍可能更早。剩余区间仍包含远端可见、处理、轮询、调度、expect_tx与打点，不作为裸延迟。

`cfg_a_empty_reuse_interval`拒绝缺失/重复发送者、错误physical thread、跨代或相位错配、非整数/非法globaltimer、未观测的零戳及不可行因果关系。新增节点只细分旧依赖边，T1/T2旧节点投影关系全部保持；没有改动冻结CTA或服务参数。CPU布局程序、源码/SASS SHA、投影证明与[条件观察接口](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R18-cluster-fill-discrimination/reanalysis/C-20261011-empty-arrival-routes-v6/observation-contract.json)见[本轮报告](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R18-cluster-fill-discrimination/reanalysis/C-20261011-empty-arrival-routes-v6/README.md)。第一阶段五切点请求不变，第二阶段仍待第一阶段定位，未新增GPU任务或冻结数值门槛。
