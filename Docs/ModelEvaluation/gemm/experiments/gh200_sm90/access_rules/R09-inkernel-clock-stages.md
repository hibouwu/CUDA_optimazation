# R09：调用内时钟、输入与时间换算

当前时钟候选以名义 Tensor/源请求工作和输入模式预测有效 cycle/ns，参数及组合见 [V09 r2](RULES.md#v09-model)。**V09 整体未通过**：cfg_b K32768 的频率低估 25.00%，完整时间高估 35.46%；训练窗口内的拟合不能代替跨 K 验证。
固定Q/S/D的sw1→sw8新配对中，频率提高9.86%、最大CTA周期减少18.50%，十对方向一致；请求次序是现行静态代理缺少的条件。源面板重访机会与暖机历史仍未分离，不先命名为缓存功耗规则。
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

旧 R09 的 globaltimer 实测步长为 32 ns，≥3.2 µs 窗口才用于其频率统计。公共 cfg_a/b/c 与旧 128×256×64、cluster2×1、4 stage 探针的阶段、资源和端点不同，旧常数不直接迁入当前模型。

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

离线复核使用各结果中的原分析器或现有 `r09_analyze.py --shared`、`r09_input_source_clock.py`、`r09_equal_work_clock.py`、`r09_zero_activity_clock.py`；参数和命令见对应归档及[固定旧正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md)。新输出写新目录，旧summary/frozen只读；本页不复制准备包启动步骤。
