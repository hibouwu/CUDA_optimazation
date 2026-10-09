# R13：有效供给、阶段与可辨识性

当前供给关系能在已见条件内用于组合试算，但 **V09 的行距与 stage 响应未迁移成立**，还不是通用供给规则。源请求、地址覆盖和 OOB 目的填零是软件需求，不能解释成物理 L2/HBM 流量。first/later 分别定值，有限预填量仍是假设；部分新频率会激活未唯一识别的服务方向。现行公式只在 [RULES](RULES.md#v09-model) 维护。本页保留供给候选的依据与失败，行距实验见 [R10](R10-layout-cache.md)，边界/显式零/填零实验见 [R18](R18-cluster-boundary.md)。

## 作业与分析索引

| 证据 | 条件与作用 | 可复核入口 |
|---|---|---|
| R05-D，735059 | 立即释放消费者，stage 2/4；共享小源与独立大源 | [正式报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-formal-v2/report.md) |
| R13，735876/735985/736169 | 18 条件的 plain 服务率、ready/retire/reuse 事件对 | [ready](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job735876-v8-formal-ready-review-B/report.md)、[ready 补查](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job735985-v9-ready-recheck-review-B/report.md)、[retire](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job735985-v9-formal-retire-review-B/report.md)、[reuse](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job736169-v11-formal-reuse-review-B/report.md) |
| V08F，737322 | 51 条后续窗口拟合；14 条尾部/D 行距诊断 | [候选与留组结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/A-20261009-supply-v3/summary.json) |
| SM 扫描，738101 | cfg_a 四档 grid、三个 pitch，固定 K4096 | [软件波次](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-sm-job738101/reanalysis/shared-cap-candidates-v3/summary.json)、[有限预填](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-sm-job738101/reanalysis/prefill-cap-candidates-v1/summary.json) |
| 同卡 B13，738203/738296/738307 | B 行距和 A/B source-map 共 13 条件，供给与填零组合开发 | [需求/填零候选](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-valid-fill-v4/summary.json)、[CTA 组合](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-cta-composition-v3/summary.json) |
| 三配置，738496 | 29 条件的 first/later 定值；cfg_b 后与旧 B13 有条件联合 | [phase 模型](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-phase-supply-v1/models.json)、[cfg_b 联合报告](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-b-joint-supply-v1/summary.json) |
| 首轮填零，738707/738865 | 只补 A/C first；六组训练行与拒绝范围 | [模型](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/models.json)、[验证](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/verification.json)、[支持定位](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/support-diagnosis.json) |
| V09，739011 | 冻结后的新几何、行距和 stage 迁移 | [完整留出与判定](V09-component-validation.md) |

## 序列供给：消费者与在途组织会改变结果

R13 的 18 条件为输入 32/48 KiB × stage 1/2/4 × 直接退休/64 依赖 FFMA/真实 WGMMA。每序列 32 Ktile，固定四个输入槽、384 线程、132 CTA，每 SM 一个 CTA；所有 CTA 复用同一份小源。GMEM A 为 tile-major，B 为 row-major，不能直接代表普通 row-major GEMM。两个 consumer WG 均退休后 producer 才能覆盖槽；WGMMA stage2/4 采用 steady wait1、尾 wait0。

设备为 a053/GPU-54896349，CUDA12.9、sm_90a、NDEBUG。v8 ready-run 的 plain 结果如下，单位 **B/SM-covered-cycle**：每进程先在各 SM 取最早起点至最晚终点跨度，再将分母求和，最后跨十进程统计。它覆盖供给、消费、退休和发布，不是物理带宽或无限长流水上限。

| 输入 KiB | stage | 直接退休 | 64 依赖 FFMA | 真实 WGMMA |
|---:|---:|---:|---:|---:|
| 32 | 1 | 35.677 | 26.681 | 22.552 |
| 32 | 2 | 56.435 | 49.526 | 28.743 |
| 32 | 4 | 83.077 | 65.847 | 58.046 |
| 48 | 1 | 43.719 | 35.280 | 22.849 |
| 48 | 2 | 64.836 | 58.816 | 35.433 |
| 48 | 4 | 85.031 | 83.385 | 45.337 |

stage4 的真实 WGMMA 在 32/48 KiB 输入下分别为 58.046/45.337，而直接退休为 83.077/85.031。改变消费者就改变序列率，因此 55、64 或任何单点都不能作为无条件每 SM 上限。v8 plain 最大时间 CV 为 3.669%；不据小差异判物理因果。

<a id="r05-d"></a>

R05-D 使用不同协议：160 线程、48 KiB/Ktile、动态 SMEM 至少 192 KiB、每 SM 一个 CTA，consumer 在 mbarrier 完成后立即释放槽。共享小源实际访问 15 MiB，独立大源 241 MiB；stage2/4 的 132 CTA 中位供给率分别为 **55.45/55.48** 和 **15.26/15.26 B/cycle/CTA**。共享源条件满足约 48 B/cycle 的计算需求，独立源不满足；stage2 与 stage4 没有响应。此表与上述 SM-covered-cycle 的消费者、源布局及分母不同，不合并定值。完整设计见 [R05-D 报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-formal-v2/report.md)。

### ready、retire 与槽复用的观测边界

ready 是 issue→wait_return；retire 是 consume→empty.arrive 前；reuse 是 producer 成功 acquire 后→下一次 A TMA 发起前。reuse 的间隔只含成功 acquire 之后的准备/取时，不含先前 consumer 工作或 empty 等待。三个事件对来自独立调用，不能拼成一次完整绝对时间线，也不直接给出硬件内部完成瞬间或队列容量。

| 事件 | 定量有效性与保留的失败 |
|---|---|
| ready | v8 18 条件原有 5 项均值扰动超限；v9 CTA0 补查只有 n128_s2_c1 以 4.703% 通过，仍有 5/10 单对超过 5%。n128_s1_c0、n128_s2_c2、n128_s4_c1、n256_s1_c2 只保留定性顺序；前者另有两次 plain 预热未收敛。 |
| retire | n128_s2_c2、n128_s4_c0/c1/c2、n256_s1_c2、n256_s2_c2 均值超限，只保留定性顺序。 |
| reuse | v11 完整 180 对仅 CTA0/tile16 的两个字段非零且正序；16/18 条件通过均值门槛。n128_s1_c0 为 5.750%，n128_s4_c2 为 5.903%，仅定性。后者代表批 4.781% 通过不替代正式失败。 |

均值门槛通过不表示每对、每个 SM 均无扰动。事件失败的条件仍保留数值正确的 plain；v10 reuse 的零字段缺陷已由 v11 修正，但不采用 v10 的事件间隔。完整 reuse 的 397393920 个保存输出值与最后输入槽通过参考复算，续采按冻结来源核对。[独立 retire 复核](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job735985-v9-formal-retire/reviews/independent-C.md)、[独立 reuse 复核](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job736169-v11-formal-reuse/reviews/independent-C.md)保留源码、SASS、逐对扰动与字段证据。早期实现和续采流水见[固定旧正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R13-async-retirement.md#槽可复用与补发直接观测)。

<a id="v08-supply"></a>
<a id="v08-supply-fit"></a>

## 有效供给函数受到哪些约束

普通完整 Ktile 按理想多播摊分的逻辑源请求为 cfg_a/b/c 每 SM **24/32/32 KiB**；计算候选为 **512/512/1024 cycle/Ktile**，满速需求相应为 **48/64/32 B/cycle**。这些是软件需求，未测出物理流量；计算受限窗口只能给供给能力下界，不能把“逻辑请求量/窗口周期”当作唯一供给标签。

完整 GEMM 的 L 起于首个 full-barrier 等待返回后、首 MMA 前，止于 mma_tail 等待和最后 stage 释放后，包含后续等待、循环与排空，排除首次输入等待。**first/later 指每 CTA 的首个/后续输出 tile，不是 Ktile**。不得再向 L 叠加一次最终 WGMMA 排空；首段输入供给和输出窗口另列。

V08F 在 a043/GPU-099 上的 84 条件中，51 条普通后续窗口用于拟合、14 条尾部/D 行距诊断、1 条原扰动 5.34% 不作定量诊断，18 条 swizzle8 分开。L 的目标是每进程后续窗口均值，再跨十进程取中位；不是跨进程池化中位，也未直接观察逐 Ktile 稳态。

| V08F 候选 | 51 条最大误差 | 关键失败 |
|---|---:|---|
| 仅计算 | 28.61% | 不能解释行距与部分长 K 窗口 |
| 统一供给项 | 23.52% | 三配置不能共享一个条件常数 |
| 共享 A/B 行距项 | 11.05% | cfg_a、K4096、A16 仍被低估 |
| 再加输入足迹 | 11.05% | K 与足迹共同变化，无法选择因果解释 |
| 再加 cfg_a 专属项 | 5.71% | 留 K 组最大 11.16%；首段转移最大 15.27% |

这些都属于已见数据开发诊断。留出某 M×N 后训练缺非零行距、留配置后系数不可识别的折不评分，不用零系数补缺。有限窗口截距可能吸收供给代价，不能命名为纯启动/排空；改截距后上述失败仍在。[完整拟合、参数与敏感性](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/A-20261009-supply-v3/summary.json)。逐项行距响应和配置反例集中在 [R10](R10-layout-cache.md#v08-stride)。

<a id="requested-sm-pitch"></a>

### 软件波次和有限预填能解释多少

[738101 的 grid×pitch 观测](R10-layout-cache.md#requested-sm-pitch)表明同一 cfg/K/行距在不同输出位置仍有不同 L。固定 K4096 的后续 144 组中，每 SM 固定供给、整个 grid 共用供给、软件波次共用供给的最大拟合误差依次为 **14.70%、9.76%、7.55%**。软件波次取仍拥有第 j 个输出 tile 的 CTA 数，是可预测工作量，不是实际同时发 TMA 的数量。

只依赖波次、行距和 T 的单值模型已存在信息下界：132 SM、A16、T4 的 j1/j2 同为 132 个软件请求 CTA，实测却为 620.33/539.23 cycle/Ktile，任何此类模型的最大相对误差至少 6.99%。

加入固定 stage 数作为预填信用、同时考虑下一输出需求后，s=5/6 的最大误差降至 **4.41%/3.78%**；s=6 首段转移仍为 5.90%。移除 132 SM 档训练后，基准服务项秩亏，向该档外推最大误差仍为 24.02%。这是增加了需求状态信息，未测出 FIRST_MMA 时实际可消费的 Ktile 数。

原卡 V08F 重拟合中，原形式/波次/六 stage 预填最大误差为 **5.71%/5.80%/6.69%**；留 K1024 有改善、留 K4096 略变差，未得到一致迁移。[原卡转移重放](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/manager-wave-transfer-v1/result.txt)。stage、在途 SMEM 量和物理队列深度不能等同；有限预填还缺同一调用内的可消费量、当前/下一输出补发边界约束。

<a id="composed-coverage"></a>

### 地址覆盖与填零需求的边界

32/128 B 覆盖量只数实际有效地址，按真实发起 box 与 multicast ownership 聚合；相邻 box 可重复覆盖 granule，不能称物理事务数。cfg_a/c 的多播源 box 与目的 tile 不相同，不能机械平分每个目的 CTA 的有效量。逐 box ownership、25 个旧条件和拟议条件的静态检查见[报告](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-static-requests-v1/summary.json)。当前公式和输入定义见 [RULES](RULES.md#v09-model)。

共同覆盖单价、分开 A/B 单价都留下失败；覆盖的秩、非唯一参数范围、反例与原冻结判定见 [R10](R10-layout-cache.md#coverage-identifiability)。只有 p0/p16 时，extra32/extra128 为 1:7，能定的是组合价，不能拆成两项物理代价。有效地址需求下降也不意味着服务减少：OOB 填零有独立条件代价，证据集中在 [R18](R18-cluster-boundary.md#valid-fill-candidate)。

<a id="valid-fill-candidate"></a>
<a id="fill-cta-composition"></a>

同卡旧 B13 的有效地址＋填零候选能改善 CTA 组合，但局部有效地址慢窗口仍有约 24%～25% 误差；填零分项见 [R18](R18-cluster-boundary.md#valid-fill-candidate)。在相同 non-L 参数下，周期/ns 候选把 13 条件 dual event 的中位/最大误差从 **9.56%/13.76%、6.87%/15.50%** 降至 **3.76%/7.14%、3.46%/5.95%**。这是同数据开发组合，显式频率假设为 1.6 GHz，未回填目标例实测频率。

诊断性地接入同调用实测全部 L 后，ns event 最大误差为 1.60%，周期换算仍为 8.26%；后者说明周期路径和固定频率可能抵消误差。只换实测首 L 也未消除最大残差。组合报告见[CTA composition](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-cta-composition-v3/summary.json)，本页不再维护旧递推或参数副本。

## first/later 定值与非唯一预测

738496 在 a057/GPU-432 上提供 29 条件、128520 个原始 L 窗口，逐 CTA/j 跨进程中位后为 12852 个。first 只用 j0，later 只用 j≥1，保持同类静态特征和条件供给形式，不按 case ID 查表。训练可使用校准窗口下界；预测必须显式给频率。下表是显式 1.6 GHz 下的同批开发误差：

| 配置 | later 中位 / 最大 / RMS | first 中位 / 最大 / RMS | later / first 秩 |
|---|---|---|---|
| cfg_a | 6.62% / 44.01% / 9.57% | 2.80% / 16.16% / 4.81% | 5/5、3/4 |
| cfg_b | 5.83% / 35.05% / 9.58% | 3.57% / 31.32% / 7.11% | 3/4、3/4 |
| cfg_c | 6.64% / 32.05% / 8.37% | 4.97% / 13.33% / 5.49% | 4/5、3/4 |

未分 first 时 cfg_c B16/K4096 首 tile case 中位曾高估 75.20%；独立 first 定值后 K1024/4096 B16 的窗口平均有符号误差为 +0.26%/+5.61%。这只证明训练状态需要分开，不是新留出成功；single48 首段仍约 +12.61%。dual/plain 的 case 中位扰动范围为 +0.54%～+6.22%，不能忽略 observer 差异。

当时的非负条件 LP 中，A/B first/later 和 C first 在 1.6 GHz 下的训练条件预测唯一；C later 的 84 个 K4096/A16 窗口仍有至多 388.70 ns 范围，其余 1940 个唯一。范围不是置信区间；首次训练没看到目的 fill，不能从 later 补 first 参数。[phase 模型和范围](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-phase-supply-v1/models.json)。

### cfg_b 19 条件的有条件联合

旧 B13 与 738496 的六个 cfg_b 条件按 case 等权联合，first/later 分开、不加 run 偏置。四 run 的 dual SASS 相同、均六 stage，但 host probe 和容量准备不同；aligned/B16 首窗桥接中位绝对差为 0.286%/0.597%，最大 0.867%/3.659%。这支持本次有条件联合，不证明跨批所有边界等价。

| run | first 中位 / RMS | later 中位 / RMS |
|---|---|---|
| 新 738496 六点 | 3.77% / 7.08% | 5.45% / 9.40% |
| 旧 738203 B 曲线 | 2.11% / 6.65% | 6.47% / 9.59% |
| 旧 738296 M-map | 6.11% / 6.72% | 6.99% / 11.36% |
| 旧 738307 N-map | 5.98% / 6.32% | 6.24% / 11.24% |

联合 later 为 7/7 秩、8388 窗口；first 为 4/5、2508 窗口。显式 1.6 GHz 下 later 条件预测唯一；first 有 2376 个唯一，132 个范围不超过 24.59 ns。B32/B64 获得参考几何下的代数支持，A32/A64/A96 和未见的 first 目的 fill 仍不支持。B64 first/T4 组中位仍低估 9.56%，原 B 曲线冻结失败不改判。OOB/有效地址慢窗口与单 CTA 最大残差见 [R18](R18-cluster-boundary.md#valid-fill-candidate)。

### A/C 首轮填零后仍有拒绝范围

[738707/738865 的 first 配对](R18-cluster-boundary.md#first-fill)只加入 A/C first，B19 和全部 later payload 不变。A first 为 5/5 秩，源有效量系数停在非负边界不等于物理供给免费；C first 为 4/5，B-fill 条件价格范围仍为 **6.3092～33.4899 ns/KiB**。具体实验误差归 R18。

最终[精确训练行](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/training-rows.json.gz)按 first/later→cfg→有序行保存，重拟合同一实现的参数与预测逐值相同；[来源清单](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/provenance.json)绑定 50 个 run/case 的原样本、源码与哈希。

| 配置 | first 行数 | later 行数 |
|---|---:|---:|
| cfg_a | 2028 | 5368 |
| cfg_b | 2508 | 8388 |
| cfg_c | 1896 | 2024 |

**固定 1.6 GHz 支持不等于任意频率支持。** 已见 50 条件的自由组合仍拒绝 `cfg_c_first_fill_oob_k1024`：训练时六个 whole B-fill 首窗处于计算分支，自由频率 1.75195 GHz 使其转入尚未唯一识别的供给方向。C 新四点没有确定所有频率下的 first-fill 服务。

V09 冻结前 30 条件为 29 数值＋1 unsupported。cfg_c G2 的 CTA62/63 特征虽在请求张成空间内，在约 1.86 GHz 下却激活未识别方向；最终 r2 条件 LP 给首窗约 **9.066～9.409 µs**。它们可能决定包络，不能忽略或用中点替代；这是固定预测频率下的范围，未重新联立时钟。原诊断与 r2 数值口径分别见[支持定位](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/support-diagnosis.json)、[r2 复核](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/reanalysis/support-check/report.json)。

## V09 后当前缺口

cfg_b G3 stage4 相对同形状 stage6 实测慢 **10.47%**，冻结模型响应为零。单点总时间误差 −6.88% 不能掩盖这一响应缺失：stage 已出现在有限预填近似中，却还没有解释实际可重叠工作或槽释放约束。先从现有 first/later、当前/下一输出窗口找约束，不能将 stage4 差直接命名为物理队列深度。

行距方面，cfg_b A16 的实测惩罚 28.94% 只被预测为 1.40%；旧几何 B32/B64 变慢与 V09 新几何 B32 未见明显惩罚同时成立。几何、K 和软件波次数不同，完整比较由 [R10](R10-layout-cache.md#v09-pitch-comparison)维护，不据两批结果宣布 32 B 因果翻转。

当前可继续复用条件关系作开发组合，但需要分别验证：有限 stage 响应、first/later 转移，以及改变频率后计算分支是否暴露未识别服务。不能用代数支持、同批拟合或完整时间误差抵消替代这些检验。[V09 原评分](V09-component-validation.md)保持整体未通过。

## 复核入口

原 18 条件用 `analyze_r13.py --cpu-check` 和 `--input 冻结样本 --output 新目录`；供给开发用 `analyze_r13_supply.py` 的 `--sm-summary`、`--coverage-suite`、`--fill-suite`、`--compose-fill`。各选项的输入和旧命令见上表对应 reanalysis；历史实现细节见[固定 cee978a 正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R13-async-retirement.md)。本次整理不重跑 GPU、不改旧 results，不重新评分。

[组件消融](V01-validation.md#v09-ablation)中，仅退回聚合周期主循环使总误差从4.11%/35.46%变为4.36%/45.26%，first/later窗口中位误差也更大；但A+16B与部分边界反而改善。因此保留条件供给的同时，优先解释这些反例，不据整体均值给所有分支背书。
