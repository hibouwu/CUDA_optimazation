# R09：调用内时钟、输入与时间换算

当前时钟候选以名义 Tensor/源请求工作和输入模式预测有效 cycle/ns，参数及组合见 [V09 r2](RULES.md#v09-model)。**V09 整体未通过**：cfg_b K32768 的频率低估 25.00%，完整时间高估 35.46%；训练窗口内的拟合不能代替跨 K 验证。
调用内也不能直接统一换算：同卡 stage4/6 的首输出与次输出周期率明显不同，且小幅跨 CTA 比率差足以影响数百 ns 的空间分散；这还不是物理 DVFS 归因，见[分阶段窗口](#stage-phase-clock)。
固定Q/S/D后，sw1→sw8的服务收益仍存在；把两组前序改成共同16次sw1后，周期减少24.60%，频率差却从同期adaptive的14.12%降到0.33%。前序协议参与了频率响应。连续调用进一步表明：共同sw1前序的最大CTA周期也随历史增加；切sw8后周期立即下降，频率却未立即改变。仅增加一个频率时间常数尚不能解释轨迹。镜像源前序后，目标窗口仍比早段慢约11%，表明大幅变慢不要求反复访问目标那份A/B。
同参数量的正值倒数响应已做离线比较：它缓解G6低估，却使独立于训练的旧wide六点频率RMS从6.56%升至10.74%；加入既有跨K数据仍未修复，因此没有替换现行组件。
逐进程分解进一步定位：cfg_b 长K在**观测包络下**的频率误差为−53.39%，自由联立缓和到−25.00%；代入同次观测频率后，包络误差仅−0.47%。直接删除源请求项使原训练频率RMS从2.39%恶化到30.76%，因此保留现行候选。新卡观察协议对照没有出现同量级的系统频率差，但长窗口plain仍比两种记录版本慢，后续须分开检查源工作代理与完整时间映射。同卡zero/random也会同时改变周期窗口与频率，不能只换一个GHz常数。完整旧推导见 [cee978a 固定正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md)。

## 用哪个窗口换算周期与时间？

| 量 | 实际边界 | 使用限制 |
|---|---|---|
| 有效频率 | 同 CTA 的 `Δclock64/Δglobaltimer` | 是配对窗口平均率，不能用调用后探针或 NVML 替换 |
| 包络 W | 同次调用 `max(final_ns)−min(entry_ns)` | 跨 SM 只相减 globaltimer；不是最大 CTA cycle 除中位 f |
| 完整时间 T | plain CUDA event 包围一次调用 | 初始化、驱逐准备与检查在外；包含最终写完成 |
| event−包络 | 同次带记录调用内相减 | 不能用不同进程群中位数之差命名物理固定开销 |
| 输出/最终记录 | cooperative `store()`/`store_tail()`，后者为 `.read0` | 源 SMEM 可复用不等于全局写完成；plain 映射仍需独立校准 |

旧 R09 的 globaltimer 实测格点为 32 ns，≥3.2 µs 窗口才用于其频率统计；格点不是端点误差上界。公共 cfg_a/b/c 与旧 128×256×64、cluster2×1、4 stage 探针的阶段、资源和端点不同，旧常数不直接迁入当前模型。

<a id="stage-phase-clock"></a>

### 首输出和后续输出能共用换算率吗？

job739095，a058/GPU-ef8692f3，cfg_b 2816×1536×4096、stage4/6 各10个 dual 进程。每 CTA 两个输出，分别由 thread128/256 记录；每个 L=`FIRST→MAIN`、对应输出 E=`EPI_PERMIT→EPI_DONE` 均使用同 CTA、同角色、同 tile 的整数 cycle/ns 差。先取每进程132个 CTA 的中位，再汇总10进程：

| stage / 输出 | L 周期率，GHz | 对应输出 E 周期率，GHz | 同调用 L 周期率跨度，GHz |
|---|---:|---:|---:|
| 4 / 首 | 1.56457 | 1.75419 | 0.03935 |
| 4 / 次 | 1.75875 | 1.82642 | 0.03431 |
| 6 / 首 | 1.52241 | 1.69049 | 0.03889 |
| 6 / 次 | 1.72620 | 1.85053 | 0.04383 |

次输出的 L 周期率在2640/2640个同 CTA 配对中升高；同一 thread128 的 E0 也全部高于自己的 L，所以固定角色差不能解释全部变化。stage4 的 L cycle/ns 空间相关为0.939/0.952，差异主要随 cycle；stage6 的 L cycle 跨度仅314.5/707，而 L ns 跨度为656/640，起点与比率残差不可忽略。不能把全部完成分散归入供给，也不能把 later 的共同 `c_ns` 未经检查用于首输出。

归档 SASS 确认两端都是相邻、同序的 CLOCK→GLOBALTIMER，未发现角色错配；相邻并不等于同时读取或固定延迟。原始端点还有“较短包含区间反而累计更多 cycle”的见证，说明精确共同非负 `f(t)` 不能复现全部读数，但不唯一指向物理时钟差。[官方 PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer-globaltimer-lo-globaltimer-hi) 定义了全局纳秒计时器，同时说明行为依赖目标架构，未给这些端点或跨 SM 偏差的误差上界。这里保留周期率与计时残差，不据此拟合新频率项。[逐进程统计、空间跨度与包含裕量](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-stage-clock.md)。

<a id="stage-order-decomposition"></a>

### 阶段逆序保留在哪些观测分量中？

同一CTA有`MAIN1−FIRST0=L0+G+L1`，其中`G=FIRST1−MAIN0`可为负，只表示跨consumer的相位关系。对739095每个进程的全部原逆序对，固定实际FIRST0，用每输出的进程中位周期率f或周期数C替换另一部分；下表再将G固定为进程中位。比例先逐进程计算，再取十进程中位[范围]：

| stage | 实际C、共同f/G后仍逆序 | 实际f、共同C/G后仍逆序 |
|---|---:|---:|
| 4 | 71.0% [55.6%,80.6%] | 49.3% [32.7%,60.1%] |
| 6 | 12.5% [1.6%,45.0%] | 77.9% [55.1%,89.8%] |

stage4的局部周期差在统一换算后仍保留多数逆序；stage6的名义ns排序更依赖观测周期率。但C与`f=C/L`共用端点，替换没有重算请求、屏障、消费和输出，因此比例不是因果贡献或预测成绩。此前stage6的CTA84/36见证也不能由其中一项独立解释：实际末端逆序768ns，两种单项替换分别只剩+18.84ns和−50.59ns，都不满足>32ns逆序。总体比例不能代替单个反例的联合约束。[逐进程比例与对称闭合分解](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-stage-clock.md)。

<a id="s-prefill-alignment"></a>

### 首段 S 与预填如何对齐？

公共事件为 `P0=producer_first_work−entry`、`S=first_MMA(tile0)−producer_first_work`。只有同 CTA、同首 tile、同首 MMA 端点下才有 `prefill=P0+S`。旧 R09 的 producer_setup 终点更晚、固定取消费者0，公共 cooperative 则取两消费者较早的首 MMA；不能把各自汇总常数相加。

V08 descriptor prefetch 的发射位于 P0；尚未完成的影响可以延续到 S。2×L2 写驱逐配对中，g1 的 ΔP0/ΔS 为 cfg_a **+353.75/+897.25**、cfg_b **+410.75/+1278**、cfg_c **+337.75/+1220.25 cycle**。变化不全在 S 内，也不足以区分描述符、数据缓存和前端。当前先保留原端点，不增加混合“冷启动常数”。[端点与驱逐配对](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md#s-prefill-alignment)。

<a id="r07"></a>
<a id="other-clock-evidence"></a>
<a id="empty-window"></a>

### 调用后频率曾怎样误导主循环解释？

旧 R07 在 GPU-009a8880 上用调用后约1.95 GHz换算，曾把约0.637 µs/Ktile解释成约1220 cycle及20%组合开销。R09 job735669 在另一卡 GPU-e403ae62 直接测得主循环 **1024 cycle/Ktile+约185 cycle**，调用内约1.64–1.70 GHz；8192³ 热调用平均仅1.38 GHz，调用后探针仍约1.85 GHz。原“20%组合开销”因此撤回。

旧128-CTA配置的打点截距约9.67 µs，主要为间隙3.9、epilogue3.0、预填2.0、尾部0.46、store后0.29 µs。打点使截距增加约0.46 µs、主循环斜率基本不变，未定位到具体阶段；不能把它分配成各阶段通用扣除项。[R09原报告与手算](../../../../../../results/gh200_resource_campaign/access_rules/20261007-r09-clock-stages/report.md)、[R07修正统计](../../../../../../results/gh200_resource_campaign/access_rules/20261007-ndebug-clock-fixedcost/analysis-r2/summary.json)。

持续负载的 NVML 证据指向680 W模组预算下的 SW Power Cap，但未解释全部窗口差：[V03](V03-clock-rule.md#功率nvml诊断) 的32-SM末1 s模组功率约472 W、未见封顶，调用内仍约1.82 GHz，NVML约1.98 GHz。二者非同一窗口，不能命名为硬件时钟上限。原EXP-04空窗口为34 cycle/128 ns，只是该探针参考，不等于 CUTLASS 打点扰动。[旧空窗口来源](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/memory-baseline-formal-v3-a/samples.csv)。

<a id="v08-clock-reanalysis"></a>

## 总时间接近时，周期和频率是否都准确？

V08 h06 为3584²×20480。保持冻结周期、F、κ，只换观测频率：

| 配置 | 冻结/观测 f，GHz | 原时间误差 | 换观测 f 后 |
|---|---:|---:|---:|
| cfg_a | 1.485835 / 1.584461 | +5.271% | −1.249% |
| cfg_b | 1.480212 / 1.587056 | −13.570% | −19.358% |
| cfg_c | 1.504003 / 1.620511 | +5.247% | −2.277% |

a/c 的偏长主要伴随频率低估；b 的周期低估被偏低频率部分掩盖。把旧 log 时长项截到校准最长期限仅提高15.62–23.28 MHz，仍低5.17%–6.20%，故“只是 log 外推”不足以解释误差。中位数代数分解严格闭合，但不是同次调用因果分解。[V08离线诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/C-20261009-clock-diagnostic-v2/diagnostic.json)。

V03 的大形状换观测频率后仍少60.513 µs，固定间隙增量仅2.479 µs；周期、关键 CTA 代理与协议差都还在。`median(C)/median(f)` 与 `median(C/f)`、跨 SM 包络是不同对象，报告中必须分开。[V03余差](../../../../../../results/gh200_resource_campaign/access_rules/20261007-v03-clock-rule/reanalysis/C-20261009-v03-residual-v1/diagnostic.json)。

<a id="input-modes"></a>

## 改变输入值会只改变频率吗？

公共输入为 dyadic（17级二进分数）、zero（A/B均为+0）、random（固定uint32坐标哈希、RN存入FP16的离散均匀样本）。行距、线程顺序不改变逻辑坐标值；默认seed17，生成定义见[固定版本](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md#input-modes)。

random抽检使用实际存储FP16输入的CPU double点积，阈值保持 `|D−ref|≤2^-20+2^-21 Σ|A·B|`，同时检查非有限值与padding。4096点通过只说明指定条件与抽样，dyadic的精确格点性质不推广到任意FP16。

<a id="r09-clock-input-job738110"></a>

### 长 K：zero 的频率上升为什么没有统一加速？

job738110，a043/GPU-201f9d2a，3584²×20480，dyadic→zero；环境、静态setup与raw均确认这张卡，不能并入后续GPU-43269fbc的训练：

| 配置 | plain时间变化 | 最大CTA周期变化 | 有效f变化 | ends包络变化 |
|---|---:|---:|---:|---:|
| cfg_a | −3.450% | +5.817% | +9.339% | −3.498% |
| cfg_b | +2.145% | +11.539% | +9.641% | +0.818% |
| cfg_c | −7.854% | +0.122% | +8.289% | −7.519% |

cfg_b 的最大stamped CTA平均周期增量中 **99.06%** 位于首/后续主循环；L含等待，不是纯Tensor指令服务。plain较慢仅8/10对，CV约1.4%–2.1%，适合结论是“本批未随频率上升而下降”，不能迁移为固定2.145%惩罚。同次event−包络仅增加约0.096 µs，跨进程中位差却可增加10.112 µs，后者不能命名新增launch成本。[比较与逐进程数据](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-clock-input-job738110/reanalysis/C-20261009-clock-input-v2/comparison.json)。

**长K random数值失败保留。** K20480三个配置各158/4096点超原阈值，最大error/tolerance=1.78958；先前K65536为958/4096、3.06834。未放宽阈值，失败条件不产生性能结论；长短组同时改变几何/seed，不能唯一归因为K或某种舍入机制。[K20480误差](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-clock-input-job738110/reanalysis/C-20261009-random-error-v1/diagnostic.json)、[K65536误差](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-R13-shared-smoke-job738097/reanalysis/C-20261009-random-error-v1/diagnostic.json)。

<a id="wide-input-preparation"></a>
<a id="wide-input-job738197"></a>

### 短 K、大几何：输入差会扩大吗？

job738197，a057/GPU-43269fbc，20480²×1024、三配置×三输入，9预检查与180正式进程全部通过；ends记录193/194或96/97 tile，未用容量64的逐tile stamped记录。全部90个ends窗口超过600 µs，条件中位为1.365–2.568 ms。

| 配置 | dyadic plain / f | zero plain / f | random plain / f |
|---|---:|---:|---:|
| cfg_a | 1816.784 µs / 1.248908 GHz | 1381.856 / 1.778852 | 2322.208 / 0.977298 |
| cfg_b | 1855.328 / 0.965850 | 1402.416 / 1.739571 | 2597.776 / 0.681815 |
| cfg_c | 1673.152 / 1.300832 | 1372.272 / 1.766605 | 2133.568 / 1.010465 |

同trial比值中位：random时间增加27%–39%、f降低22%–29%、C略降；zero时间降低18%–24%，C反而增加8%–36%。三配置两种variant方向均10/10一致；trial仍是先后运行，不是同时调用。这里只确认输入相关计时，不从cycle/ns推断功率、压缩或缓存机制。

所有random检查通过原阈值、最大error/tolerance≤0.62142，不改变长K失败。cfg_b random的跨进程`median(plain)−median(ends window)=30.144 µs`，同次event−window却为4.176 µs，再次说明统计边界的重要性。[九条件与全部配对](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-wide-input-job738197/reanalysis/C-20261009-wide-input-v1/summary.json)；[旧准备说明](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md#wide-input-preparation)。

<a id="input-clock-job738376"></a>

### 短、中、长窗口的输入差是固定偏移吗？

job738376 同卡、同公共6338653二进制，18条2048²/8192²×1024校准加cfg_b三条20480²长桥接；21预检查、420正式进程全部通过。

| 窗口 | ends包络范围 | zero有效f变化 | random有效f变化 | plain方向 |
|---|---:|---:|---:|---|
| 2048²短窗口 | 13.600–16.528 µs | +3.61%–5.17% | −4.56%–4.83% | 两种输入各23/30对一致 |
| 8192²中窗口 | 177.120–303.536 µs | +13.47%–34.28% | −19.35%–25.76% | 两种输入各30/30对一致 |

短窗口更受固定项和状态影响，频率方向比plain时间稳定；中窗口输入差显著放大，cfg_b zero的C也增加约6.32%。长桥接plain变化−0.169%至−0.333%、f变化+0.291%至+0.952%，支持这三条cfg_b跨作业一致性，不自动认证cfg_a/c长条件。

源工作时钟最终把本批全部21条件用于训练，因此原bridge已不再是该版本独立检验。旧job738197的cfg_a/c六条长条件仍仅作已知开发诊断。[训练数据与拟合来源](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-input-clock-calibration-job738376/reanalysis/C-20261009-source-clock-dev-v2/source-clock.json)。

<a id="source-clock-dev-v1"></a>

## 当前输入时钟候选能解释到哪里？

公式与接线统一见 [RULES](RULES.md#v09-model)，这里保留识别依据。名义Tensor工作Q按所有132个SM平均；源请求S按cluster多播折算，cfg_a/b/c每Ktile为24/32/32 KiB。同几何的Q相同，S比例b:a:c=4:3:2，提供了区分配置响应的静态特征。二者不依赖旧周期模型或目标实测C。

21条训练全部K1024、seed17；固定τ的七列设计满秩，允许输入各自d/e、共享a/τ。拟合直接面向观测有效f，约束d/e≥0、a≤归档Max Clocks 1.98 GHz。最优a触上界、τ=200.678 µs；τ的训练MSE≤最优1.10倍范围179.89–222.59 µs是敏感性范围，不是硬件响应时间或置信区间。

| 检验 | 观测W下频率RMS | 说明 |
|---|---:|---|
| 21条训练 | 2.385% | 最大8.075%，cfg_c中窗口random仍有缺项 |
| 整组留出cfg_b | 7.013% | 中窗口random可迁移，三个长条件约±11%，设计条件数升高 |
| 留出2048² | 6.418% | 仅已知数据分组诊断 |
| 留出8192² | 10.710% | W在训练端点之间仍不等于该组织已验证 |
| 留出20480² | 14.106% | K仍固定1024，不能验证K迁移 |

cfg_b中窗口random的拟合中，Tensor项与a/c近似，源请求项较大，能描述其额外降频；这是拟合分解，不是独立功率分解。借旧C/F/κ做自由时间诊断时，完整时间RMS仍6.454%、长cfg_b zero误差−21.986%；频率拟合好不能修复周期低估。

固定正C模型可证明唯一正根；组合ns递推后，需检查包络随f的性质，不能照搬旧证明。观测W条件拟合、自由频率与自由时间分别报告，任何一项都不代替另外两项。[参数、分组、τ剖面与旧C诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-input-clock-calibration-job738376/reanalysis/C-20261009-source-clock-dev-v2/source-clock.json)。

<a id="clock-form-candidates"></a>
<a id="input-clock-mode-fit"></a>

此前候选已停止并列维护：旧log四项全拟合/无时长项没有稳定跨K优势；饱和时长与Tensor活动代理只作来源；仅输入计算项的无约束/约束形式被源工作版本替代。[旧候选比较](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/C-20261009-clock-candidates-v1/candidates.json)与[未归档正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md#input-clock-mode-fit)保留推导，不作为第二份当前规则。

<a id="equal-work-job738459"></a>

## Q/S相同，跨K响应是否也能预测？

job738459固定原clock参数，对照4096²×4096与8192²×1024，九组配置/输入配对的Q/S完全相同。输出256→64 MiB、唯一输入足迹32→64 MiB、K和每CTA tile数同时改变，所以不是纯输出字节干预。18预检查与360正式进程通过，K4096 random最大error/tolerance=0.86840；九条跨作业桥接f变化−1.096%至+0.595%。

以下频率响应误差先逐trial计算预测比值/观测比值−1，再取中位；条件残差另算，不能混用。

| 配置/输入 | 实测f配对变化 | 频率响应误差 | K4096条件频率残差 |
|---|---:|---:|---:|
| a/dyadic | −10.144% | +8.634% | +4.488% |
| a/zero | −1.113% | −1.367% | +0.495% |
| a/random | −13.540% | +12.775% | +9.694% |
| b/dyadic | +0.532% | −1.355% | −0.047% |
| b/zero | −0.235% | +0.397% | +0.965% |
| b/random | +0.238% | −1.342% | +0.261% |
| c/dyadic | −2.156% | −1.834% | −0.666% |
| c/zero | +0.230% | −3.453% | −1.657% |
| c/random | −4.682% | −0.536% | +6.727% |

三个全零K4096残差≤1.66%，削弱“单是K4096或统一缺输出字节项造成R18长零区偏差”的解释，不支持据此添加统一D项。cfg_a非零输入响应却10/10对都偏高，保留跨K/几何/足迹缺口；c/random主要延续配置偏差，不能把全部残差叫跨K失效。只在观测W下计算f，没有自由时间预测。[全部配对与冻结身份](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-equal-work-job738459/reanalysis/C-20261009-equal-work-clock-v1/equal-work-clock.json)。

<a id="r18-zero-activity-mixture"></a>

## 必零乘积比例能解释边界频率吗？

R18 job738296/738307 的八个cfg_b条件均有768个调度tile、192个必零，静态z=0.25；OOB与有效地址零的Q/S/z一致。固定原参数，仅用 `(1−z)d_dyadic+zd_zero` 与同样的e混合，不减少名义Q/S。

| 观测W下的诊断 | 原dyadic频率RMS | 固定z混合RMS |
|---|---:|---:|
| 全八条件 | 11.187% | 6.400% |
| K1024四条件 | 3.617% | 1.578% |
| K4096四条件 | 15.402% | 8.912% |

八例绝对残差均缩小，长K仍低估7.247%–9.821%。相同z的有效地址零比OOB窗口短约18%–19%，混合模型对频率路径响应仍不完整；一侧面板零与R09两输入全零不同，e按z混合尚无物理依据，全卡z也没有表示逐CTA分布。这一假设进入V09但未成为通过规则。[八例与路径配对](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R18-input-map-job738296/reanalysis/C-20261009-zero-activity-clock-v1/zero-activity-clock.json)。

<a id="v09-frequency-domain"></a>

## V09长K失败约束了什么？

cfg_b 6144×8192×32768的自由f为0.9000 GHz、观测约1.2000 GHz；时间误差+35.46%，仅代入观测f后降至+1.71%。这定位时钟跨K迁移为主要误差，不能由此识别物理功耗来源或改写冻结成绩。首段/预填仍有独立失配，继续按原事件端点分析。[V09完整结果](V09-component-validation.md)。

测前训练已包含cfg_b dyadic **0.968796 GHz/1843.840 µs**、random **0.688303 GHz/2546.928 µs**。0.90低于同类dyadic训练范围，却不低于所有历史实测；5–7 ms超出训练窗口，但相对完整训练集不是长十几倍。范围按配置、输入、窗口和工作量说明，不把历史最小值当物理下限，也不夹紧频率改善分数。

V09的换算周期误差与频率误差相关，但换算量本身包含f，不能据此认定“训练时吸收了周期误差”。当前clock已直接拟合观测f；要分别比较局部cycle/ns、观测W条件频率和自由组合残差。[误差抵消复核](V09-component-validation.md#cycle-frequency-cancellation)。

<a id="v09-three-layer-diagnostic"></a>

### 观测包络能否消除时钟误差？

固定r2参数，对30条件的300个成功dual进程分别计算：原自由频率、代入该次观测包络W的条件频率、代入该次观测频率的事件递推。下表先逐进程算相对误差，再取10次中位，和先汇总中位数再相除不完全相同。拒绝的cfg_c G2保留诊断，但不补入V09成绩。

| G6配置 | 自由f误差 | 观测W下f误差 | 代入观测f后的包络误差 |
|---|---:|---:|---:|
| cfg_a | −4.63% | −0.17% | −8.19% |
| cfg_b | −25.00% | −53.39% | −0.47% |
| cfg_c | −8.05% | −12.93% | −0.73% |

cfg_b/c的条件频率10/10次偏低，不能把失败只归于周期预测通过联立污染频率；cfg_b的联立实际上缩小了频率误差。cfg_a则不同：条件频率中位接近观测，换入观测f仍少8.19%的包络，服务和事件组织仍有缺口。短G1三个配置的条件频率误差分别为+6.85%、+6.68%、+5.97%，也是公式在观测W下的偏差，不能只调整F掩盖。

**一个调用平均f何时够用？** 按消费者原始角色分别取相同端点的cycle/ns，first/later按全CTA的tile顺序区分，避免cooperative合并时cycle与ns各选不同角色。300个调用中，301440个合并端点有54处选择不同角色，局部频率统计不使用这些合并比值。所有tile时间戳落在32 ns格点；频率只汇总≥3.2 µs窗口，短窗口仍保留周期和纳秒。cfg_b G6的later主循环频率中位与同次CTA平均f相差最多0.47%；代入观测f后，later主循环误差由+33.12%降至−0.21%。单一f足以解释该条件的后续长循环，首轮却有一次相差26.80%，不能把同一结论推广到首段。cfg_b G6末输出仍低估24.26%，说明修时钟不等于所有组件准确。

**删掉源请求项是否更好？** 仅用原21个K1024条件，保留共享a与每输入Tensor项，重新拟合观测GHz；没有用V09时间、周期或频率拟合。训练频率RMS由2.39%升至30.76%，既有等Q/S的9个K4096条件由4.27%升至19.80%。虽然部分G6残差缩小，整体证据不支持删除源项。需要检查的是源工作代理的条件范围，不能把它称为已识别的物理功率分解。

上述[逐调用诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/A-clock-failure.json)核对了旧作业，没有同时匹配卡、几何、行距、grid/stage、输入和分配容量的成功ends/dual配对。因此补了下面两档，不重跑输入全矩阵。

### 观察协议是否造成长窗口频率偏差？

job739095，a058/GPU-ef8692f3，cfg_b、6144×8192、K4096/32768、dyadic/seed17、swizzle8、132个CTA、stage6；两档都用lda=32768、ldb=8192，同样分配384 MiB A和512 MiB B，但活动输入足迹不同。每档plain/ends/dual各10个进程，60次全部通过原数值检查。ends每tile预留6个word、dual为10，以下比较包含完整观察协议差异，不能把变化唯一归因于时间戳指令。

| K | ends / dual频率中位，GHz | dual/ends频率变化 | dual/ends包络变化 | ends/plain完整时间变化 | dual/plain完整时间变化 |
|---|---:|---:|---:|---:|---:|
| 4096 | 1.39393 / 1.39982 | +0.24% | +0.17% | −0.01% | +0.09% |
| 32768 | 1.26526 / 1.28042 | +0.42% | −0.44% | −3.79% | −3.84% |

变化列先算同trial先后进程的比值，再取中位，不能由前一列的两个频率中位直接相除。dual/ends完整时间变化分别为+0.12%和−0.43%；最大CTA周期则10/10次增加，中位为+0.47%和+0.06%。本卡没有观察到能解释旧25%频率低估的系统协议偏移，但长K的频率配对范围为−9.62%至+4.54%，ends/dual频率CV为1.03%/3.34%，不能声称每对都相差不到1%。

长K的plain/ends/dual完整时间中位为5191.70/5008.10/4954.74 µs，ends在10/10对中更快、dual在7/10对中更快。它说明当前状态下带记录包络到plain的映射仍有独立差异；跨进程比较不能证明记录本身加速，也不能用一个固定项消去。此结果来自新卡，不回溯认定a057无观察扰动，不把旧时钟公式在这里的绝对数值当验证成绩。下一步优先解释源工作代理的跨K残差与长窗口plain映射，暂不增加阶段时钟分支。[全部进程、配对分散及条件](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/A-controls.json)。

### 只改频率响应的形式，能修复跨K迁移吗？

在原线性响应之外，只比较一个同参数量的正值候选：`f=1/[a+d_mode·Q/(1000W)+e_mode·S·g(W/τ)/W]`，仍为共享a/τ、三种输入各自d/e，共8个参数。它保留原工作代理，不增加频率下限；约束`a≥1/1.98`、d/e非负，使`0<f≤1.98 GHz`。拟合目标仍是观测GHz的平方残差，不能因公式取倒数就改成拟合1/f。

先固定原21个K1024训练点，再检查既有数据。倒数21版在等Q/S的9个K4096条件上频率RMS为4.23%（线性为4.27%），但大几何与更长K仍失败，因此另列加入这9点的30点开发版，并同时重拟合线性形式以区分数据量与响应形式的影响。所有系数都来自GPU-43269fbc；V09没有进入拟合。

| 形式与训练集 | 原21点频率RMS | 同卡旧wide六点RMS | V09 cfg_b G6条件频率误差 |
|---|---:|---:|---:|
| 线性，21点 | 2.39% | 6.56% | −53.39% |
| 倒数，21点 | 2.46% | 10.74% | −36.28% |
| 线性，30点 | 2.55% | 6.54% | −52.57% |
| 倒数，30点 | 2.60% | 10.65% | −35.39% |

同卡旧wide六点为cfg_a/c的20480²×1024，不在这两版训练中。倒数响应减轻G6的极端低估，却使这组误差扩大；30点版本也没有解决问题，因此**不替换当前clock组件**。新卡739095只作条件趋势比较：K32768/K4096的实测频率比为ends 0.907、dual 0.916，倒数30版在观测窗口下仍给出0.786/0.780，仍夸大随K的下降。这不是新卡绝对频率或完整时间验证；旧738110同样只比较输入响应方向。

数学上，若事件包络`E(f)>0`连续、随f非增，且`f→0`时`E(f)→∞`，则候选`F(W)`随W非减，`F(E(f))−f`严格递减，并在`(0,1/a]`恰有一个零点。这只给出联立所需条件，不证明现有事件递推全部满足，也不证明预测准确。此次负结果把后续问题收紧到工作代理与状态：**保证正值并保持相同参数量，仍不足以解释迁移**，暂不继续扫描更多响应函数。[参数、分组残差、卡身份与趋势](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-response-candidates.json)。

### 源项是否只是在代替输出活动？

每个cfg内，原Q/S严格固定为a的512/24、b的512/32、c的1024/32 cycle/KiB；Q与S不是独立工作轴。原21点又全为K1024，有效FP32输出与名义计算满足`D_bytes/Q=8 B/cycle`。不带g的七列Q/D设计仅rank4；带g以后可满秩，但分离信息来自窗口变化，不能称为独立识别输出能耗。加入同卡九个K4096条件后，D/Q有8和2两档，不带g也达到rank7。

因此只试一个活动替代：保持线性频率响应、非负约束、输入模式分类和8个参数，用静态`Dmean_KiB=4MN/(132×1024)`替换S，仍乘g；D不取自观测输出窗口或elapsed。原21加跨K九点训练的结果为：

| 30点训练形式 | 原21子集频率RMS | K4096九点RMS | 未训练wide六点RMS | V09 G6频率偏差范围 |
|---|---:|---:|---:|---:|
| 原Q+S | 2.55% | 3.40% | 6.54% | −52.57%～+0.48% |
| 替代Q+D | 8.55% | 22.64% | 31.89% | +20.54%～+52.14% |

Q+D连训练内的跨K响应都无法解释，分输入模式的非负输出价也未解决问题；不能因G6低估变高估而采用。原Q+S包含的配置/访问组织信息不能由输出字节完全替代，具体物理来源仍未识别。结果与分组秩保存在同一份候选输出中，未修改现行时钟组件。

### 工作量不变，只改swizzle，频率与服务会变化吗？

job739539仍在a058/GPU-ef8692f3，固定cfg_b、6144×8192×32768、dyadic/seed17、行距、分配容量、stage6与132个CTA，只改sw1/sw8。两者均3072个输出tile、每CTA23或24个tile，名义Q/S和有效输出D相同。两目标各plain/ends×10个进程全部通过检查；先运行的40个参照进程不计入配对。

| sw1→sw8，同trial变化的中位 | 有效频率 | ends包络 | 最大CTA周期 | plain完整时间 |
|---|---:|---:|---:|---:|
| 实测 | +9.86% | −24.93% | −18.50% | −27.27% |

上述四项方向均10/10一致。频率配对范围+3.00%～+19.13%；sw1 ends的进程CV为6.22%，一次明显更快的成功记录仍保留。它表明次序会改变这组工作量下的服务与时钟，不能只换一个频率常数。更直接的反例是：固定Q/S时，现有非负价格形式的F(W)随W非减；这里每对都是W缩短而f上升，单一时长项无法描述这个方向。

静态上，按软件输出序号j对A/B源面板去重，面板槽数从1222降到644；每SM的`U=Kt×Σ_j(有效不同A/B源box KiB)/132`由75838.06降到39967.03 KiB。U只是重访机会代理，软件轮次不是实际同时发射，不能称为物理流量。另一个解释尚未隔离：ends自适应暖机总时间的中位从86.40 ms降到44.65 ms，暖机次数也不同。下面检验U是否足以替代原S，但不能将约10%的频率响应直接归为缓存功耗，更不能认为它已解释旧卡25%的频率误差；服务改善与暖机历史需要分别检查。[40目标进程、分散与静态需求](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-swizzle.json)。

**U替代也未成立。** 仍用原线性响应、原8个参数和观测GHz目标，分别在旧GPU-432的21/30条件拟合Q+U；不加入a058结果。训练中逐软件轮核对A按cluster-N、B按cluster-M切片的实际源盒去重，多播目的副本不再重复计源；全部21+9条件均支持，没有删除训练点。

| 形式 | 原21点频率RMS | K4096九点RMS | 未训练wide六点RMS |
|---|---:|---:|---:|
| Q+S，21点训练 | 2.39% | 4.27% | 6.56% |
| Q+U，21点训练 | 4.86% | 20.15% | 8.11% |
| Q+S，30点训练 | 2.55% | 3.40% | 6.54% |
| Q+U，30点训练 | 6.38% | 14.60% | 5.13% |

U30虽改善wide组，却让V09 G6频率高估18.17%～45.91%。a058的sw8/sw1频率比实测1.099，U21/U30条件预测为1.026/1.014：方向正确，幅度不足。当前U只定义完整in-bounds tile；V09的cfg_c G2和两个partial条件明确列为不支持。在共同支持的27条件上，S30/U30频率RMS为11.34%/11.41%，不能借少算三例宣称提升。Q+U尚不能替换现行时钟，软件轮的完全去重与真实重访、以及自适应暖机历史仍需区别；此处停止增加混合价格。[各版本、支持边界与次序趋势](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-response-candidates.json)。

### 共同前序以后，次序收益还在吗？

job739552在同a058/GPU-ef8692f3保留两条adaptive同期参照，再比较共同16次sw1后的一次sw1/sw8目标；仍是同一G6、输入与分配条件，四条件各10个ends进程。两套launch参数均在输入准备前构造，共同前序使用相同kernel、指针、输入和trace清零/同步操作。两种policy均实测确认sw1/sw8各先运行五次。

| sw1→sw8，同trial变化中位 | 有效频率 | 最大CTA周期 | 目标包络 | event完整时间 |
|---|---:|---:|---:|---:|
| adaptive同期参照 | +14.12% | −19.29% | −25.63% | −25.62% |
| 共同16次sw1前序 | +0.33% | −24.60% | −24.55% | −24.54% |

adaptive四项方向均10/10复现；与739539的+9.86%频率响应只能作分批参考。共同前序下，周期和包络仍10/10减少，但频率配对范围为−2.70%～+4.00%，只有7/10上升。结果支持**前序协议参与升频、次序服务收益仍在**，不将协议差单归因于暖机次数、功耗或缓存。

频率观察窗口是目标调用内、同CTA入口到最终记录的cycle/ns，先取最大工作量CTA的中位；不是连续时钟轨迹，也不同于跨CTA包络。下表观察窗口先取每进程这些CTA的窗口中位，再列10个进程的分布。前序时间是CUDA event记录的暖机kernel时间之和，不包含其间trace清零、同步和主机间隙；本探针没有记录完整前序的墙钟包络。

| policy / 目标 | 前序次数范围 | 前序kernel总和中位〔范围〕，ms | 目标频率观察窗口中位〔范围〕，ms |
|---|---:|---:|---:|
| adaptive / sw1 | 8–22 | 89.02〔48.22–147.94〕 | 6.620〔6.069–7.057〕 |
| adaptive / sw8 | 8–11 | 39.21〔39.02–55.28〕 | 5.021〔4.804–5.281〕 |
| common / sw1 | 16 | 105.75〔101.58–107.96〕 | 6.839〔6.644–7.101〕 |
| common / sw8 | 16 | 105.17〔100.88–107.69〕 | 5.465〔5.110–5.787〕 |

共同组的前序kernel总和也非数值相等：sw8/sw1配对变化中位−0.43%，范围−2.78%～+1.74%。相同声明前序没有消除随机卡状态，更不证明物理缓存内容相同。共同sw1/sw8分别有4/7个进程的前序末五次CV超过2%，全部保留；固定前序不能事后再套adaptive过滤。40进程的原始数据、源码及二进制SHA均复核通过。[完整四条件与协议间配对](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-common-history.json)。

后续频率规则的最小输入契约应包含前序协议：例如原8–30次收敛算法，或固定16次sw1后首个目标；前序kernel、次数/停止规则、输入初始化与驱逐准备需有定义。当前调用的W不能隐式代替这些历史。未知前序时应明确不支持无条件的频率与完整时间点预测，可保留静态工作量和指定f下的服务诊断；没有相应校准时也不能凭本批两种协议构造通用误差区间。这是后续模型契约建议，V09冻结公式和成绩保持不变，本节未增加拟合。

### 连续调用能定出一个频率时间常数吗？

此前739552只有每进程一个目标均值。按`df/dt=(f_eq(load)−f)/τ`及真实区间均值拟合，假设40进程共享初态时τ约11.06 ms；允许各进程初态未知后，45.3 ms至扫描上界10 s均能复现这些末端观测。原H还遗漏调用间间隙，因此11.06 ms不能命名为硬件时间常数。[单次目标的不可辨识反例](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-history-state.json)。

job739579在同a058/GPU-ef8692f3记录完整历史：16次sw1前序后，连续19次sw1或sw8。同期single对照只执行首个目标，两种模式均分配相同约0.56 MiB头记录容量和event对象，只有series逐次D2D保存头、批末一次读回。四条件各10进程，single/series先后各五次；700次series调用都核对工作数和端点，40进程数值检查全部通过，数值检查只覆盖各进程最后目标。19次目标约覆盖此前条件敏感性上界29.43 ms的三倍，不保证慢状态已达到平衡。

**周期响应与频率响应分开出现。** 下表先逐进程取比值，再取进程中位；前后段分别取三次调用中位。

| 同进程比较 | 有效f | 最大CTA周期 | 同CTA C/f窗口 | event时间 |
|---|---:|---:|---:|---:|
| 共同sw1前序末3次 / 前3次，20进程 | +4.83% | +24.07% | +12.80% | +17.49% |
| sw8首目标 / 紧前sw1末调用，10进程 | +0.018% | −20.32% | −18.02% | −20.51% |
| sw8目标末3次 / 前3次，10进程 | +3.12% | −0.009% | −3.05% | −3.38% |

同CTA C/f先用每个最大工作量CTA自身的周期与频率相除，再取CTA中位，严格等于对应globaltimer窗口；它与最大C/进程中位f、跨CTA包络分开保存。前序这四项均20/20增加，因而周期增加不只是随时钟抬升：纳秒窗口和event时间也变长。没有D2D复制的single，其前序末3/前3次event同样20/20变慢，中位+17.34%、范围+10.88%～+25.26%，排除了前序变慢完全由新记录操作产生的解释。

切sw8的周期和时间减少均10/10，首调用频率没有一致方向；随后19次sw8最大CTA周期的进程内CV中位仅0.0126%、最大0.0246%，sw1对照为4.10%和5.22%。频率仍可先下降再上升，末段较初段升频只有7/10；服务改善先出现，频率历史更复杂，不能把全部轨迹压成一个单调“暖机升频”。这些观测尚未把缓存、功率控制或其他状态唯一分开。

**记录扰动的中位小，配对离散并不小。** series首目标相对同期single的频率变化中位为sw1 −0.019%、sw8 −0.395%，各自范围−14.34%～+8.16%、−7.99%～+12.87%；两组前序event跨度变化中位+0.34%/+0.51%。因此没有看到同方向的大中位偏移，但不能据此证明D2D记录无扰动。调用间event间隔占全过程中位为single 0.257%、series 0.339%；series的globaltimer末端到下次入口间隔占0.404%。二者边界不同，均不能直接叫纯空闲时间，且小间隔占比不证明没有状态影响。

同CTA、同消费者角色的入口→最终端点进一步显示，变慢并非只发生在关键CTA。以连续两次比前三次配对基准高5%作为离线定位规则，ns起点在19/20进程出现，中位为第8次调用；累计kernel时间中位42.06 ms、范围29.72～61.60 ms，换时间轴后的离散程度并未减小。第8次受影响角色占比中位84.09%，第11次为100%；无D2D的single也在相近调用出现event变慢。调用次数、累计负载和同一源地址的重复仍共线，约42 ms不能当硬件阈值。相同负载访问目标源T或镜像源P后的对照已完成，见下文；原起点定位本身仍不能区分两者。[逐角色配对定位](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-history-localization.md)。

**同一单极模型未通过留段检查。** 用每进程初态、共享sw1/sw8平衡值与τ，直接拟合局部CTA窗口的平均GHz；前16次prefix和前10次target训练，末9次target留作时间外推。间隔内沿用上一次负载平衡值，这是未单独识别idle响应的近似。训练选择τ=3.04 s，sw1平衡值触1.98 GHz上界；末9次频率RMS为3.24%，差于保持最后训练频率的1.95%。把末9次也纳入描述拟合后τ变为25.12 ms；两次MSE≤最优1.10倍的敏感范围都延伸到10 s扫描边界。拟合参数不稳定，末段不支持新增这个动态频率组件。

下一步的缺口已收窄为**同一前序中周期成本为什么变化，以及局部频率状态为何出现平台和转向**，而非继续搜索响应函数。所有实测时轴、每进程初态与两个局部平衡值都只用于机制诊断，不接入静态预测，不改变V09冻结成绩。[全部轨迹、扰动配对、τ剖面与末段检查](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-history-series.json)。

### 镜像源前序：目标地址重复不是必要条件

739590在同a058/GPU-ef8692f3比较16次sw1(T/P)之后的19次sw1(T)。两组都先分配、初始化双份相同A/B，D不变；目标在两份分配中的slot0/1分别成组，每组same/mirror各十进程。全部40进程、1400次调用保留，源码/二进制/指针身份及端点通过回放；数值检查覆盖每进程末次目标。编译仍为168寄存器、16条HGMMA、零spill。

| mirror相对same，同CTA/消费者角色的ns变化 | 两slot共20对 | slot0 | slot1 |
|---|---:|---:|---:|
| 前序末3次 | −0.14% | +0.88% | −0.70% |
| 目标前3次 | −0.57% | +0.62% | −1.78% |
| 目标第9–11次 | −0.45% | −1.39% | +0.52% |
| 目标末3次 | +0.98% | +2.00% | +0.77% |

两组前序的整体服务接近，但逐对末段ns差为−5.54%～+7.62%、频率差为−9.54%～+11.23%，不是完全相同的初态；没有按差异或CV筛进程。目标差距从前3到末3缩小仅9/20，未见共同的大幅恢复再逐渐追上轨迹。更直接的是，mirror目标前3次相对其本进程前序最初3次的局部CTA窗口仍增加11.05%，20/20为正、范围6.28%～16.01%；same为12.74%。镜像前序没有通过GEMM重复读取T，T却已表现出晚段代价，因此不能把变慢归结为必须重复目标地址。slot早段方向不同，也不能把小总体中位数当成逐对等价。

该GPU批次前后NVML的SW Power Capping累计增加24.407 s；热降频、HW Power Braking和Sync Boost计数未增加。两端均在0% GPU利用率下读到SM1980 MHz、显存2619 MHz。这只说明整批期间发生过软件功率封顶，计数包含初始化、前序、目标和检查，不能定位第8次调用，也不能由端点时钟相同证明调用内显存时钟恒定。当前证据约束到跨这两份源数据延续的状态，没有唯一确定功率或其它共享资源机制。[完整分层轨迹与限制](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-mirror-history.md)、[NVML摘录](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-mirror-nvml-snapshots.json)。

<a id="time-input-contract"></a>

## 时间预测必须声明哪些输入？

2026-10-11离线复核固定r2的Q/S及参数，没有新增时钟价格。时间输入至少包含下面三类；只给M/N/K和输入模式，尚不能给出通用完整时间点预测。

| 输入 | 必须声明的内容 | 当前证据边界 |
|---|---|---|
| 静态执行与数值 | cfg、tile/cluster、stage、grid、软件调度、行距/分配/输入map、dyadic/zero/random与seed | 保留名义补齐工作；Q/S采用冻结版本，不能把源量换成valid/distinct后沿用旧系数 |
| 调用历史 | 前序kernel与二进制身份、输入及源/目标缓冲关系、固定次数或adaptive停止规则、目标序号、准备/清零/同步/复制顺序与调用间隔 | 固定16次sw1、adaptive、single/series和same/mirror均有独立身份；相同声明前序仍有进程分散，不等于初态相同 |
| 观测与完成边界 | plain/ends/dual、原消费者角色、FIRST/MAIN/PERMIT/DONE/final、clock/globaltimer读取顺序、完整event | 同角色整数差先相减；跨SM只用globaltimer。调用平均率不自动适用于首输出；final的源释放也不等于完整写完成 |

源工作口径由B维护。r2的`source_mean_kib`对应名义源请求payload（按理想cluster多播折算）除以132，包含OOB/填零工作；它不是有效源字节、去重源字节或实测物理流量。本轮只检验该冻结定义，不重新定价。正式自由预测只能读取声明输入与冻结参数；目标观测W/f只允许进入条件诊断。未知前序/初态、未经校准的新卡、未验证的首段换算或观察版本到plain映射，应明确返回不可支持，不能以测得的目标时间补齐。

### 忽略历史，固定点至少错多少？

对同一静态目标，若各历史组的时间中心为正数y，任何一个固定预测p都满足

\[
\min_p\max_y|p/y-1|=(y_{max}-y_{min})/(y_{max}+y_{min}),
\quad p^*=2y_{min}y_{max}/(y_{min}+y_{max}).
\]

这是本批观测的经验最优下界，不是总体置信界或硬件噪声下限。先按相同声明历史分组，组内进程分散单列；组内分散仍可能含隐藏状态与端点效应，不能全部称为重复测量噪声。

| 固定目标与分组 | 周期中心的最小最大误差 | 时间中心的最小最大误差 | 全部时间观测的最小最大误差 |
|---|---:|---:|---:|
| 739552 sw8，adaptive/共同16次sw1，各10进程 | 最大CTA周期0.0149% | event 4.01% | event 9.08% |
| 739579 sw1，35个调用位置，各10进程 | 同CTA周期9.20% | event 9.47% | event 15.29% |
| 739579 sw8，19个目标位置，各10进程 | 同CTA周期0.00446% | 同CTA时间2.00% | 同CTA时间6.87% |

sw1的35次调用使用同一形状、输入、地址与工作量，前16次和后19次只处于不同历史位置。时间中心的固定点RMS下界为5.98%，周期中心为6.31%；不能用一次“代表暖态”的观测代替整条历史。sw8目标周期中心稳定，时间中心仍变化；时间状态不能由周期稳定自动推出。739552 sw8的组内event CV为adaptive 3.39%、共同前序4.26%，两种历史中心差异和组内分散都须保留。

本轮复放共同前序40进程、长K观察对照60进程、连续/镜像80进程，共700/1400次历史调用；每个进程末次目标通过原4096值检查，逐次历史只有工作/端点检查。原V09的300个dual诊断逐一核对raw SHA后复用，stage4/6另复放20进程。历史分解对每个CTA/原消费者角色分别取前后各三次的cycle/ns中位，使用对称恒等式分开周期项与逆周期率项；逐进程均值闭合最大残差为2.7×10⁻¹⁰ ns。各项再跨进程取中位时不保证相加闭合，不能据此声称因果贡献比例。

### 条件换算与自由组合分别错多少？

条件时间误差定义为`f_obs/f(W_obs)−1`：固定对应观测周期，仅检查换算；它不是自由时间预测。代入观测f的事件递推还含服务/结构与端点差，不能命名为纯硬件服务周期误差。以下先逐进程计算，再取10进程中位：

| V09 G6配置 | 条件时间误差 | 自由包络误差 | 观测f下递推包络误差 | 自由plain时间误差 |
|---|---:|---:|---:|---:|
| cfg_a | +0.17% | −4.48% | −8.19% | −3.70% |
| cfg_b | +114.54% | +32.61% | −0.47% | +35.46% |
| cfg_c | +14.86% | +7.93% | −0.73% | +7.63% |

cfg_b的自由联立缓解了条件换算偏差，不能只看总时间后认定两部分都准确。全30条件与逐进程分项保存在本轮输出；cfg_c G2的冻结拒绝继续保留。a058的739095仅列固定r2 Q/S的跨卡条件换算诊断，完整供给递推的`card mismatch`拒绝保持有效；未通过改写GPU身份制造自由预测。plain与观察版本来自不同进程，trial配对不是同调用的因果分解。

### 单一调用换算率需要多大的端点解释？

对739095的同CTA首/后L，求一个任意正数f，使`N_i=C_i/f`的最大原生ns残差最小。若假定cycle差精确、每个globaltimer端点误差至多ε，则窗口残差至多2ε。两个窗口给出必要界

\[
\epsilon\ge\frac{|N_0C_1-N_1C_0|}{2(C_0+C_1)}.
\]

stage4/6各进程先取132个CTA必要界的中位，再取10进程中位，分别为**658.42/671.22 ns**；若要求整调用全部CTA共用一个常数f，则为**776.81/795.09 ns**。这些界不是timer精度测量，也不排除真实的调用内周期率变化；它们只排除“一个常数换算率加32 ns量级端点误差就够了”的解释。已有跨SM包含见证仍说明精确共同非负f(t)也不能复现全部端点；不能据这两类反例直接增加phase专属频率价格。

当前保留结构反例，没有采用纯时间新候选。单极状态末段RMS 3.24%劣于末值基线1.95%的失败继续有效。下一步需要区分同角色真实首/后换算变化与读寄存器配对延迟/端点误差；最小观察接口请求在本轮目录，交主集成决定公共probe和GPU调度。它是观察有效性实验，尚不是新的局部留出通过结果。

请求r2的纯CPU判定在[r09_clock_brackets.py](../../../../../../microbench/gh200_resource_campaign/access_rules/r09_clock_brackets.py)。每个戳记录`CLOCK_before/GT/CLOCK_after`；用整数差得到窗口cycle区间[Clo,Chi]。**128 ns是整个窗口起止差的总容差**，正f可行区间为`[Clo/(N+128), Chi/(N−128)]`，不是每端点各128 ns；计算用精确有理数。最小总区间容差δ再减一次128，剩余单端点等价量为`max(0,δ−128)/2`。首/后L的端点不共享；这不是任意共享事件图的联合可行性证明。

观察扰动的**进程中心屏**限定为：bracket-dual相对匹配dual的每phase配对绝对L差中位≤384 ns、9/10对≤1024 ns，cycle差中位≤512 cycle，完整event差中位≤512 ns。phase中心明确定义为`median(同CTA/原角色的signed差)`，不能先分别取两变体的CTA中位再相减；函数必须收到全部132×2原生配对值，保留逐CTA绝对差及去中心分布。384/1024 ns依据旧stage6的L进程MAD 192/128 ns和跨度896/528 ns，不能直接迁移成空间容差。半数CTA变快/半数变慢即使中心为零，也不通过局部观察稳定性判断。

[旧stage6空间重复参考](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/A-time-contract-20261011-v3/spatial-reference/repeat-spatial-reference.json)仅有预列相邻五对同观察版本进程；first/later去中心ns最大值分别达到656/512 ns，p95达到496/448 ns，状态与CTA到SM映射变化仍混在其中。它不是可靠空间界。本轮不由这五对拟合容差：中心屏通过后，非零空间变化仍标未决、原现象归因不可判，不能自动执行30进程方案并在采后补阈值。原dual还须在共同plain前序下复现必要端点界>256 ns；现有`common-sw1-16`使用当前编译变体，不能自动宣称plain前序相同。[r4的配对、聚合及完整边界](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/A-time-contract-20261011-v3/minimum-observer-request-r4.json)待主集成评估；原目标没有因此缩小。

r4纠正提案的grid维度：该cfg_b 2816×1536×4096、stage6、heuristic AlongN的原V09与739095 setup均为`[1,132,1]`，此前误写为`[132,1,1]`。总CTA数仍为132；prefix与target须逐维匹配，不能只检查乘积。[原setup哈希与校验](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/A-time-contract-20261011-v3/grid-correction-verification.json)保留修订来源，r3及主集成记录的740046/740047在GEMM前拒绝不追改。

读取顺序还需要机器码证据。2026-10-11的独立CPU汇编样件先检验“CLOCK值非零时读取GT、GT值非零时读取后CLOCK”的PTX谓词链；CUDA13.0的ptxas将它降为无条件`CS2R`后接`SEL`，预期的数据依赖没有保留，故拒绝该版本。随后[r09_clock_stamp.cuh](../../../../../../microbench/gh200_resource_campaign/access_rules/r09_clock_stamp.cuh)用最多两次零值重读及真实退出分支，得到`CLOCK→64位非零比较→BRA→GT→64位非零比较→BRA→CLOCK`。单处与双处展开都保留控制链，双处标签作用域也通过汇编；零值重读耗尽时返回全零，未来reader必须拒绝任何零字段及`CLOCK_after<CLOCK_before`，不能退回普通dual解析。

[静态样件与失败版本](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/A-clock-stamp-static-v1/README.md)保留源码、机械提取的PTX、cubin、SASS和工具版本。[专用检查器](../../../../../../microbench/gh200_resource_campaign/access_rules/r09_clock_stamp_static.py)仅检查该隔离样件的64位比较、退出分支、两次重读上限和全零失效路径，不是通用CUTLASS审计器。当地CUDA C++编译在主机GNU版本检查处失败，因此只声称同一inline PTX的sm_90a汇编证据；没有运行设备、没有得到实际CUTLASS帧或采样误差界。控制依赖比相邻读数提供更强的编译证据，仍不能单独证明硬件采样包围关系。r4继续未冻结/延后，公共集成、目标工具链复核与局部扰动资格由主集成决定，本样件不触发GPU请求。

随后用本地NVRTC13.0直接编译**同一未改写的设备C++ header**，显式配置CUDA提供的NVRTC适配`cstdint`与CCCL include路径，未注入typedef或覆盖主机编译器限制。[专属设备编译入口](../../../../../../microbench/gh200_resource_campaign/access_rules/r09_clock_stamp_device_cpp.py)检查64位类型、24字节结构和8字节对齐；单处/双处调用均生成PTX，汇编后的SASS与前述机械提取样件逐字节一致。增加一个word的布局反例被静态断言拒绝，前CLOCK改成32位的反例被inline asm的`l`约束拒绝，未产生PTX。[设备C++前端证据](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/A-clock-stamp-device-cpp-v1/README.md)保留四条件、SDK身份上集与两次回放；它补上本地设备C++层，不追改先前NVCC失败，也不证明目标CUDA12.9、实际CUTLASS帧、运行时观察或时间规则资格。

下一项有界需求是[实际CUTLASS编译接口](../../../../../../microbench/gh200_resource_campaign/access_rules/r09_clock_stamp_integration.json)，仅交主集成作目标CUDA12.9的CPU构建审查。保留原16个head word及每tile字段0–9，在原FIRST/MAIN各增加一个after word，形成12word/tile、1552word/CTA。plain、matched-dual与guarded-bracket都分配并在每次前序/目标前清零**1638912B**，避免740048已见的准备字节差；matched-dual在新增字段写零，bracket只在原thread128/256的FIRST/MAIN调用候选，工作和同步调用不改。新记录须有显式协议与arm身份，不沿用旧dual10解析；任何零stamp、类型/宽度错误或after<before均保留为无效，不能回退。

接口同时消除r4的same-process歧义：每臂进程内plain-prefix与target共享A/B/D；不同臂进程间绑定相同内容、布局、初始化、模块与准备协议，不宣称物理页或绝对指针相同。新matched基准因增加记录写入而不同于旧dual，后续仍须复现原现象，不能预先认定等价。[接口依据与编译审查边界](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/A-clock-stamp-integration-request-v1/README.md)不修改r4的数学门槛，也不解决空间扰动资格；此时不请求GPU、正确性调用、pilot或30进程采样。完整时间规则的新局部留出与跨卡要求保持不变。

### 同plain前序接口的两进程冒烟证明了什么？

主集成740048在GPU-43269fbc上完成plain-host/dual-host各一进程，二者都通过共享库调用16次同plain kernel，再在同A/B/D上执行目标。实际grid均为`[1,132,1]`、stage6；各4096值按原标准精确正确、padding/非有限为零。本组独立重放原gzip及检查值。设备SASS与冻结plain/dual各自相同的证明来自主集成，本组绑定其报告与源包哈希，不把两进程冒烟算正式计时、观察扰动资格或局部留出。

前序执行器相同还不代表准备过程相同。归档`timed()`在每个prefix/target前清零trace并同步；本批记录为：

| host目标 | 每调用清零scratch | 16次prefix的kernel event总和 | 首15个gap占prefix首入口到末出口 | 全16个gap中位 |
|---|---:|---:|---:|---:|
| plain | 827904 B | 814.080 µs | 29.59% | 22.368 µs |
| dual | 1368576 B | 816.864 µs | 27.92% | 20.784 µs |

gap由相邻CUDA event相减，包含准备、主机与设备路径，不能命名为纯空闲时间。当前`warmup_policy=common-sw1-16`仍是旧标签，plain模块身份只在call的`plain_prefix_abi/calls`中表达；旧reader仅检查该标签会接受这种不同执行器。因此正式清单/解析须同时绑定模块/二进制、准备字节与顺序、间隔规则；比较dual/bracket时先验证这些历史输入匹配。观测到的prefix间隔可作历史诊断，不能隐式成为自由预测的目标W/f。

这一个dual调用的原角色首/后L周期率中位为1.58318/1.76108 GHz，同CTA单常数换算的必要单端点容差中位为529.73 ns。它只表明新接口下仍有需要解释的端点算术；没有重复分布、夹取read跨度或误差界，不能据此判定物理频率或认证旧观察版本。最小下一步仍是审查采样包围关系，而不是增加phase时钟价格。

[独立审计结果](../../../../../../results/gh200_resource_campaign/access_rules/20261011-R09-prefix-host-check-r5/reanalysis/A-history-contract-audit-v1/prefix-history-audit.json)绑定原始样本、源包与有效trace include路径；[本组入口](../../../../../../microbench/gh200_resource_campaign/access_rules/r09_plain_prefix_audit.py)独立验证plain模块身份，不修改公共reader。源包中overlay trace有两个同名成员，最后的stub引用`source/probes/v06_trace.hpp`；审计保留两成员哈希并读取该有效实现，避免将较早成员误当实际编译源。

公共reader增加显式prefix协议/准备身份后，原740048 r5应以`undeclared plain-prefix protocol`拒绝，不能补写原metadata追认为新协议。本组入口改用仅接受这两份原raw SHA的历史诊断解码，重新核对各4096值、dual的132个CTA/264个输出和原端点；明确`current_prefix_protocol_qualified=false`，不调用或放宽公共身份验证。[当前公共reader下的复放](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-R13-R18-failure-controls/reanalysis/A-local-clock-consistency-current-reader-v1/current-reader-verification.json)保留两份拒绝，21进程全部旧数值记录逐项相同；原失败目录与旧输出不追改。

### 大的常数率残差一定要求大的局部计时误差吗？

不一定。对一个CTA的entry、producer、两原角色FIRST/MAIN/PERMIT/DONE/final，共12个配对端点，允许该SM的有效周期率随时间变化，但约束`0≤f(t)≤1.98 GHz`。1.98是r2声明的上限，本检验不新增物理硬件上限断言。若每个globaltimer端点允许误差ε，则对`C_i<C_j`必须有

\[
t_j-t_i\ge(C_j-C_i)/f_{cap},\qquad
\epsilon_{min}=\max\left(0,\max_{C_i<C_j}
\frac{(C_j-C_i)/f_{cap}-(N_j-N_i)}2\right).
\]

按C排序（同C按N排序），对`N−C/f_cap−ε`取前缀最大值，再加回`C/f_cap`，可构造满足所有端点与速率上限的见证；所有差先以整数相减，约束使用精确有理数。本轮逐CTA验证该构造，没有从见证读取预测参数。

| 数据 | 首/后L常数率：必要端点容差的进程中位再中位 | 可变局部率：必要端点容差的进程中位再中位 | 可变局部率：全部CTA最大 |
|---|---:|---:|---:|
| 739095 stage4，10进程 | 658.42 ns | 8.40 ns | 14.65 ns |
| 739095 stage6，10进程 | 671.22 ns | 4.78 ns | 13.55 ns |
| 740048 dual，单次冒烟 | 529.73 ns | 4.78 ns | 11.53 ns |

这说明**常数换算失败，不迫使局部端点存在数百ns误差**；允许局部率变化是与这些读数相容的另一解释。可变见证自由度更高、由目标端点构造，不能把表中差异叫预测精度提升，不能接入自由模型，也不能认定为已识别的DVFS或时钟门控。各CTA使用独立见证，它不修复跨SM共同f(t)的包含反例，也不消除读取配对延迟的可能性。下一步观察设计必须保留这两个解释，不能先把常数率残差全部归给观察器。

[2772个CTA的逐点见证与输入哈希](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-R13-R18-failure-controls/reanalysis/A-local-clock-consistency-v1/local-clock-consistency.json)由[专用入口](../../../../../../microbench/gh200_resource_campaign/access_rules/r09_local_clock_consistency.py)生成；纯构造函数在`r09_clock_brackets.local_clock_witness`，不解析数据、不拟合参数。

### 跨SM共同的可变周期率允许多少端点误差才相容？

进一步允许每个SM有独立未知计数原点，检验`C_sm(t)=offset_sm+F(t)`、`0≤F′(t)≤1.98 GHz`；所有SM共享任意连续可变的F，仍不假定物理DVFS响应形式。每个原端点的真实采样时刻允许位于`[N−ε,N+ε]`。同SM先去整数原点，不跨SM相减clock64。这个共同轨迹假设强于上节各CTA独立轨迹，弱于全卡一个常数率。

以`Y_sm=−offset_sm`表示未知原点，端点约束等价于`F(N−ε)−Y_sm≤C`、`Y_sm−F(N+ε)≤−C`，相邻时间节点另满足`0≤ΔF≤f_cap·Δt`。乘以1.98的分母50后全部成为整数差约束：可行势定义连续分段线性F，不可行由闭合负环的严格负整数和证明。对ε按整数ns二分，保留最小可行ε及ε−1的不可行证书；真实最小值在`(ε−1,ε]`，不把1ns扫描分辨率当计时器精度。

再单独移除速率上限，避免把假定上限的影响归给跨SM共享本身。下表共同轨迹列为最小整数单端点容差的进程中位〔范围〕。

| 数据 | 共同可变率，≤1.98 GHz | 共同非负率，无上限 | 各CTA独立可变率≤1.98：全部CTA最大 |
|---|---:|---:|---:|
| 739095 stage4，10进程 | 228.5 ns〔183–249〕 | 177 ns〔161–225〕 | 14.65 ns |
| 739095 stage6，10进程 | 247 ns〔218–262〕 | 233 ns〔161–257〕 | 13.55 ns |
| 740048旧dual冒烟，单进程 | 231 ns | 209 ns | 11.53 ns |

即使无上限，全部21进程的32ns单端点预算仍不足；32ns在此只是检验假设，不是timer精度断言，也不是前述128ns整个窗口差的容差。两个共同轨迹分支各自逐一反演验证33264个原端点，未使用跨SM计数差。因而“各CTA在小端点容差下相容”不能推出“全卡共同f(t)也在同一容差下相容”。该结果不证明SM时钟物理独立，也不把表中数值命名为实测timer误差：SM局部计数行为、读取配对及端点效应仍未区分。缺少实际读配对误差界时，既不能认证全卡共同周期率，也不能据此采用SM专属价格；应先完成观察资格，再决定时间组件的共享范围。

[逐进程上下界、负环及共同曲线证书](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-R13-R18-failure-controls/reanalysis/A-shared-clock-consistency-v1/README.md)由[r09_shared_clock.py](../../../../../../microbench/gh200_resource_campaign/access_rules/r09_shared_clock.py)生成，复用已核验的整数端点并重新核对21份raw及冻结上限来源的SHA。全部仍为开发重复或旧协议冒烟，没有新增时钟拟合、独立留出或公共模型资格。

### 只使用先前调用，能预测下一次时间吗？

2026-10-11对上述80进程检验两个零参数基线：固定保持第16次prefix的观测，或每次使用紧前已完成调用的观测。前者预测整条目标轨迹时不读任何目标观测；后者是逐次预测，只在前一目标完成后更新。两者均不使用当前或未来目标，不拟合参数；这是已知开发数据的诊断，尚不是预先冻结的新留出。

归档`timed()`逐次返回CUDA event结果，主机在下次调用前已可使用；逐角色cycle/ns则逐次D2D保存、整批后才D2H读回。因此event基线不需新增设备读取，角色分项仍是离线诊断，不能声称当前协议已经提供在线周期率状态。以下误差为`预测/实测−1`，表中为全部逐调用的RMS / 最大绝对误差，单位%。

| 目标条件 | 目标调用数 | 保持prefix末值 | 紧前调用末值 |
|---|---:|---:|---:|
| series sw1 | 190 | 5.97 / 17.51 | 3.34 / 10.04 |
| series sw8，含首个切换 | 190 | 26.51 / 43.31 | 6.50 / 40.62 |
| same源，两slot合并 | 380 | 4.42 / 16.91 | 3.82 / 15.38 |
| mirror源，两slot合并 | 380 | 5.33 / 23.17 | 3.52 / 17.62 |
| single sw1 | 10 | 2.87 / 5.62 | 2.87 / 5.62 |
| single sw8 | 10 | 28.20 / 39.09 | 28.20 / 39.09 |

series sw8排除首个切换后的180次，紧前event基线RMS为1.23%、最大7.76%；首个切换10次的误差中位+25.81%、最大40.62%仍在完整结果中。该预定义分段说明连续同负载状态可提供信息，不支持把首个切换删掉后宣称整族通过，也不排除使用静态负载信息的更完整历史规则。

对同CTA、同原消费者角色，以紧前窗口`(C_p,N_p)`预测当前`(C_t,N_t)`，保留`e_C=C_p/C_t−1`、条件换算项`e_f=(C_t/N_t)/(C_p/N_p)−1`和自由窗口项`e_N=N_p/N_t−1`。逐角色满足`(1+e_C)(1+e_f)=1+e_N`，全部601920条的最大闭合残差4.5×10⁻¹⁶；各项取角色中位后不保证闭合，也不重建完整event。series sw1的逐调用角色中位误差RMS为周期3.41%、条件换算1.13%；sw8首个切换之后为0.015%/1.25%。镜像四个源/slot分层的相应范围为周期3.14%–3.70%、换算0.93%–1.08%。这些分项保留观测端点效应，不能直接命名为物理服务价格或DVFS误差；但不能只增加频率状态，把sw1的原生周期变化全部吸收进去。

[完整记录、分层结果与可用时点证据](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/A-history-forecast-v1/README.md)由[r09_history_forecast.py](../../../../../../microbench/gh200_resource_campaign/access_rules/r09_history_forecast.py)生成：1160个目标、两个基线共2320个event预测，全部single/series、首个/连续目标、same/mirror及slot均保留。Q/S和校准系数未变，未增加纯时间候选；周期率的观测资格、独立局部留出和跨卡组合边界仍未完成。

复核入口：[r09_time_contract.py](../../../../../../microbench/gh200_resource_campaign/access_rules/r09_time_contract.py)。[本轮结果与契约](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/A-time-contract-20261011-v3/README.md)包含逐进程CSV、源哈希、经验下界、支持/排除/不可区分及独立复放命令。持续目标仍未完成，后续规则须另经冻结的新局部留出与组合检查。

## 作业与复核索引

| 实验问题 / 作业 | 一句结果 | 证据入口 |
|---|---|---|
| 调用内和调用后时钟，735634/735669 | 调用后探针导致旧20%开销误判 | [R07修正](../../../../../../results/gh200_resource_campaign/access_rules/20261007-ndebug-clock-fixedcost/analysis-r2/summary.json)、[R09报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-r09-clock-stages/report.md) |
| V03/V08频率和周期抵消 | 只换频率使cfg_b周期低估更明显 | [V08诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/C-20261009-clock-diagnostic-v2/diagnostic.json) |
| 公共数值检查，738097 | K65536 random失败保留 | [逐点误差](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-R13-shared-smoke-job738097/reanalysis/C-20261009-random-error-v1/diagnostic.json) |
| 长K输入，738110 | zero可升频而不加速，K20480 random失败 | [比较](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-clock-input-job738110/reanalysis/C-20261009-clock-input-v2/comparison.json) |
| 宽几何输入，738197 | 同形状random慢27%–39%，周期并非不变 | [九条件](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-wide-input-job738197/reanalysis/C-20261009-wide-input-v1/summary.json) |
| 短中窗口及长桥接，738376 | 21条训练支持输入相关候选，跨几何仍有误差 | [source-clock](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-input-clock-calibration-job738376/reanalysis/C-20261009-source-clock-dev-v2/source-clock.json) |
| 等Q/S跨K，738459 | cfg_a非零输入响应失败，zero不支持统一D项 | [完整配对](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R09-equal-work-job738459/reanalysis/C-20261009-equal-work-clock-v1/equal-work-clock.json) |
| R18部分零活动，738296/738307 | 固定z只能解释部分频率低估 | [八条件](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R18-input-map-job738296/reanalysis/C-20261009-zero-activity-clock-v1/zero-activity-clock.json) |
| V09迁移，739011 | cfg_b长K失败，拒绝项与分项独立保留 | [逐例诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/manager-results-v1/summary.json) |
| V09三层分解，复用739011 | cfg_b条件时钟失配，删源项无益，缺同条件ends/dual对照 | [300调用与删项结果](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/A-clock-failure.json) |
| 观察协议，739095 | 新卡长K无同量级系统频率差，plain映射仍有偏差 | [两档K、60进程](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/A-controls.json) |
| 次序配对，739539 | 固定Q/S/D后sw8仍升频并减少周期，时长项不充分 | [sw1/sw8](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-swizzle.json) |
| 共同前序，739552 | 共同前序下频率差缩小，周期收益保留 | [四条件、40进程](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-common-history.json) |
| 连续历史，739579 | 周期即时响应与频率历史分离，单极留段无优势 | [700次调用与同期对照](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-history-series.json) |
| 镜像源前序，739590 | 目标源未被前序重复计算仍表现晚段成本；未出现共同重置/追上轨迹 | [40进程、1400调用](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-mirror-history.json) |

离线复核使用各结果中的原分析器或现有 `r09_analyze.py --shared`、`r09_input_source_clock.py`、`r09_equal_work_clock.py`、`r09_zero_activity_clock.py`；参数和命令见对应归档及[固定旧正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md)。新输出写新目录，旧summary/frozen只读；本页不复制准备包启动步骤。
