# R13：有效供给、阶段与可辨识性

当前条件供给尚未通过V09的行距与stage迁移。入口条件重放后，旧ready延迟2257.5仍违反stage4后段的缓冲复用必要界；这次反证不依赖交接或第一输出初态。stage6接近计算饱和，不能独自辨识供给价格。下一候选用已公开的两stage开发数据约束共同复用代价，保留有限窗口与协议差异，不加stage专属常数，也不将观察值作为正式预测输入。

首次供给的释放前端点使新批40/40次ready观察能绑定首代事务，排除了本协议下“只是consumer晚返回”；请求排队、数据服务与就绪可见性仍未分开。逻辑源请求与填零量不等于物理流量。现行公式只在[RULES](RULES.md#v09-model)维护；行距证据见[R10](R10-layout-cache.md)，边界/填零证据见[R18](R18-cluster-boundary.md)。

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
| stage×K，739095 | 同卡四条件；固定启动差和恒定每Ktile罚项均不充分 | [成对窗口、协议扰动与缩放残差](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/B-controls.json) |
| 同次stage事件，739556 | CTA0第二输出Ktile16；额外L在mma返回前，局部等待与发射区间互相转移 | [窗口与累计进度](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-events.md) |
| consumer累计进度，739578 | 同几何stage4/6；相邻commit分解与早/后段累计差，区分循环推进和最终drain | [同次端点与递推对照](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-progress.md) |
| 组内服务递推，离线 | 仅stage6三个局部量校准；独立局部约束的q剖面仍高估stage4，推导有限窗口下界 | [参数族与逐分项比较](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-admission.md) |
| 交接不可辨识性，离线 | 不同ready/交接组合给出相同stage6局部量及L，却产生不同stage4响应 | [同调用下界与参数反例](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-handoff.md) |
| 同次计算组交接，739686 | 原两stage×plain/progress/handoff；交接非零，stage差主要在首full等待，局部观察扰动明显 | [三段闭合与累计推进](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-handoff-samples.md) |
| 固定入口，离线 | 旧延迟违反后段必要界；stage4必须有内部停顿，不能只移动入口或末端领先量 | [条件重放与参数无关界](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-conditioned.md) |
| 首次供给，739629 | 同binary新增打点开关 × 起步0/512 ns × B0/B32；定位S增长 | [同批观察扰动与端点](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R13-first-supply.md) |
| 独立ready，739645 | 只读phase0观察开关 × 起步0/512 ns × B0/B32；10/40可绑定、30/40边界跨FIRST | [观察扰动与区间](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R13-ready-observer.md) |
| 释放界ready，739653 | 同四条件、同binary联合观察开关；40/40可绑定，ready前增长不能由consumer返回单独解释 | [逐进程区间与模型必要界](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R13-ready-release.md) |

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

<a id="v09-stage-failure"></a>

## V09 后当前缺口

cfg_b G3 stage4 相对同形状 stage6 实测慢 **10.47%**，冻结模型响应为零。单点总时间误差 −6.88% 不能掩盖这一响应缺失。对原dual记录按同trial、CTA与工作坐标配对，先取窗口变化率中位，再取十trial中位：

| stage4相对stage6 | first L | later L | 首段入口→主循环 | 预填 | 末次输出 |
|---|---:|---:|---:|---:|---:|
| 直接ns变化 | +5.48% | +15.91% | −6.01% | −2.25% | −7.47% |
| 主循环cycle变化 | +8.49% | +17.89% | — | — | — |

first/later的ns变化在十trial中分别为+3.29%～+8.52%、+13.92%～+19.25%。差异集中在主循环，later更强；仅修频率、首段或输出都不能解释。L仍包含等待、循环和排空，这些端点没有分别测出TMA等待和槽释放。

### stage参数为什么没有改变预测？

该几何是132个CTA各做两个输出。现行近似以`stages/Ktile`将当前软件波次替换为下一波次需求；没有producer/consumer槽状态。在冻结频率1.54235 GHz下，计算下界为331.96 ns/Ktile：

| 输出位置 | stage6 / stage4有效源需求KiB | stage6 / stage4拟合供给ns/Ktile | max选中项 |
|---|---:|---:|---|
| 首轮j0 | 4224 / 4224 | 177.57 / 177.57 | 两者均为计算下界 |
| 末轮j1 | 3828 / 3960 | 227.04 / 234.87 | 两者均为计算下界 |

首轮的当前/下一波次都包含132个CTA，特征不变；末轮虽然变了，仍被计算下界截住。于是264个L窗口、整条事件递推、频率联立和plain预测完全相同。问题是模型缺少实际流水重叠的响应，不是stage开关没有传入。

静态资源排除了驻留CTA数和spill解释：stage6/4的SMEM为215040/149504 B，均1 CTA/SM、168寄存器、16条静态HGMMA、无spill。但SASS并不等价：plain的GEMM非NOP静态指令数1266/1254，dual为1525/1535，整数与uniform指令的组织也改变。静态条数不是动态工作量，不能把总差直接解释成物理队列容量。

[原卡成对分析](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/B-analysis.md)、[max分支、资源与SASS差别](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/B-summary.json)保留上述开发诊断，未改变V09评分。

### 同卡K对照：固定启动差和恒定每Ktile罚项均不充分

job739095在a058/GPU-ef8692f3上测cfg_b、2816×1536、`K1024/4096 × stage4/6`四条件，每条件plain/dual各十进程。固定132 CTA、sw1、dyadic seed17、实际lda=4096与ldb=1536；A/B分配量固定为11556864/6291456个FP16元素，有效源量仍从8.5增至34 MiB。本节只作这张卡内部对照，不迁入a057的绝对参数。下表的Δ均为stage4−stage6：先在同trial、同CTA/j/T/坐标作差，取CTA中位，再取十trial中位。

| K | first ΔL，ns | later ΔL，ns | first ΔL，cycle | later ΔL，cycle |
|---:|---:|---:|---:|---:|
| 1024 | −80 | +608 | −116.50 | +1106.00 |
| 4096 | +968 | +3384 | +2368.75 | +6457.25 |

**只给L加一个固定启动/排空项无法解释。** later的差从608增至3384 ns，十trial范围分别480～672、2976～3968 ns；first甚至由所有trial为负转为全部为正。cycle与直接ns的符号和缩放一致，不能靠频率换算解释。

**跨K共用一个恒定每Ktile罚项也不够。** later的ΔL/Ktile从38.00增至52.875 ns，十trial范围30～42与46.5～62不重叠；cycle/Ktile从69.125增至100.895，范围61.97～78与85.71～120.35也不重叠。这是重复范围而非置信区间。持续供给或指令组织限制仍与结果相容，但尚未从有效源足迹和工作历史中分离；没有识别物理槽容量，也不据此新增经验罚项。

plain中同trial的stage4相对stage6变化中位为K1024的+6.79%、K4096的+11.92%；dual为+5.48%/+11.20%，完整时间同样显示持续差。观察协议的影响单列：K1024的dual/plain中位约+4.15%，单对最大+8.74%；K4096的stage4/6分别为+1.91%/+3.29%，单对最大+4.84%。局部定量结论因此只属于dual窗口，不能用完整时间方向一致证明局部无扰动。首输出窗口也随stage改变（K1024 +14.81%、K4096 −20.92%），不能把总差全部归给L，或把不同中位直接相加。[四点对照与逐trial残差](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/B-controls.json)、[原始记录](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R09-R13-R18-failure-controls/)。

### 有限缓冲递推：拟合略好，仍不能迁移

实际CUTLASS程序的slot从producer成功acquire到consumer release期间不可覆盖。full等待返回后才提交当前WGMMA；`wait_group<1>`后释放前一组slot，尾部`wait_group<0>`释放最后slot。pingpong在`mma`返回后、`mma_tail`完成前向下一math WG交棒，producer与缓冲状态跨输出继续。旧`stages/K`折扣没有这些状态；把每个输出重新当成空流水也不符合程序。

据此试算一个三参数stationary递推：发起时刻`I_i=max(I_(i−1)+p,U_(i−S))`，ready时刻`A_i=I_i+ℓ`；consumer等待ready与上一轮wait1后提交，Tensor完成后按上述顺序释放slot。p/ℓ/c分别是有效发起间隔、ready时延和消费间隔，c≥512 cycle；观察量仍为首个full等待返回到最后WGMMA完成，不把初次ready时延再加进L。它们没有被解释为独立物理服务常数。

只用739095同卡的八个first/later周期目标，重拟合旧需求折扣形式与该递推；不用a057参数，不设stage4常数或自由K插值：

| 形式 | 参数数 | 同批中位/最大误差 | 留K1024 / K4096最大误差 | 留stage4 / stage6最大误差 |
|---|---:|---:|---:|---:|
| `stages/K`需求折扣 | 4 | 4.55% / 8.10% | 16.80% / 36.31% | 14.04% / 16.44% |
| 有限buffer递推 | 3 | 4.34% / 6.98% | 7.09% / 12.39% | 37.77% / 10.79% |

这是多起点最小二乘与测后留组开发预测，不是新的冻结验证。有限递推略降同批残差，却不能解释短K的first略变快、later明显变慢，并在只见stage6时严重误判stage4。完整八点的局部参数秩为3，留组后为1～3；数值局部可解不等于机制已识别。旧R13的ready/retire/reuse来自另一卡、不同源布局及独立调用，reuse又从acquire成功后开始，没有empty等待，不能补成当前程序的同次生命周期。

因此不将该递推接入现行预测，也不增加参数修补。随后用同一对stage4/6、K4096条件的稀疏事件区分输入等待与消费侧进度，结果见下节。详细公式、八点残差和参数解见[有限stage分析](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-finite-stage.md)、[可重放数据](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-finite-stage.json)。

### 同次稀疏事件：差异不在最终drain，单点等待也不能代表整L

job739556在同一a058卡复测cfg_b 2816×1536×4096的stage4/6，只记录CTA0第二输出Ktile16及同slot补发；两角色的SM、工作坐标、slot和phase全部匹配。四个plain/sparse二进制均168寄存器、16 HGMMA、无spill。下表为十进程周期中位；差列先按trial配对，不能直接用前两列相减。`mma_tail`是wait0与最终slot释放，不是输出完成后的退出段。

| 同调用窗口 | stage4 | stage6 | stage4−stage6 |
|---|---:|---:|---:|
| Ktile16：consumer full等待 | 234.5 | 65.5 | +168.0 |
| full返回→commit | 161.0 | 331.0 | −172.0 |
| commit→wait1返回 | 6.0 | 6.0 | 0.0 |
| **full等待前→wait1返回（直接合并）** | **404.0** | **404.0** | **0.0** |
| wait1(Ktile16)→目标slot release前（Ktile17） | 635.5 | 533.0 | +103.0 |
| FIRST_MMA→Ktile16 full等待前 | 11024.5 | 8111.0 | +2869.5 |
| Ktile16 wait1返回→mma返回 | 28427.5 | 24169.0 | +4259.5 |
| FIRST_MMA→mma返回 | 40038.0 | 32712.5 | +7183.0 |
| mma返回→mma_tail开始 | 10.0 | 13.0 | −0.5 |
| mma_tail | 412.0 | 418.0 | −5.5 |
| 完整L | 40461.0 | 33141.0 | +7177.0 |

额外L在mma返回前积累，最终drain没有变长。Ktile16之前已有+2600～3577 cycle差，之后到mma返回又增加+3718～4883；不是只有首段或最后排空变慢。但所选Ktile的full等待增加，被随后到commit的区间缩短抵消；同调用合并窗口的成对差中位为0，范围−5～153 cycle。因此不能拿full等待的+168乘64，也不能将wait1的6 cycle当作WGMMA完整延迟。producer在所选Ktile上的empty等待416.5/410、同slot补发empty等待422.5/360.5 cycle也不能单独区分输入饥饿与消费者较慢。

每次调用的`mma返回前 + handoff + mma_tail = L`精确闭合；各列中位数不要求加和闭合。稀疏/plain完整时间扰动stage4/6中位+3.49%/+1.69%，单对最大+4.77%/+4.44%；plain的stage差+10.28%，稀疏为+12.16%。与739095旧dual按同CTA第二输出整L分批比较，新值分别+2.84%/−0.93%，没有明显整L失真，但不证明每个局部窗口无扰动，更不能拼接两批时间线或直接采用事件价格。

紧接着的Ktile17并不相同：从Ktile16的wait1返回，到Ktile17中释放目标slot之前，成对差中位+103 cycle，范围−25～358；不能因Ktile16 combined相同就忽略下一步推进。两stage全部十次producer refill都在consumer release前已开始等待。把release发出时刻夹在调用前后，至producer acquire返回的可观察间隔在stage4为141～273、stage6为126～249 cycle；它包含轮询、调度、expect_tx和取时开销，不是裸barrier可见性延迟，也不是新的硬件常数。[逐进程窗口与来源](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-events.json)、[CSV](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-events.csv)、[原始记录](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R13-stage-events/)。

### 局部约束进一步否定了哪些递推简化？

原三参数递推在该位置预测stage6的wait1约523.65 cycle，实测只有6；它把整个Ktile计算放到一次瞬时commit之后，无法表达已有的full返回→commit区间及其与执行的重叠。原模型还将已等待producer的release→复用视为零成本，与上述观测接口不符；不能因此把约200 cycle直接改称硬件延迟。

又检验了一次最小修订：仍只拟合producer间隔p、ready时延ℓ和剩余尾延迟τ三个共享参数，把cfg_b的512 cycle吞吐下界放到commit推进，`C_i=max(B_i,C_(i−1)+512)`、`D_i=C_i+τ`，保留原slot释放语义。739095 dual八个whole-L目标的拟合中位/最大误差为3.32%/8.22%，但739556 sparse的局部约束仍不满足；两批分开评估，未混合拟合。该解局部秩为2，p增加1 cycle不改变任何训练L，不能声称三个服务参数均已识别。

这个失败不只是一套参数不好：在修订的零循环推进开销假设下，full-wait开始为`H_i=W_(i−1)`，wait1返回为`W_i=max(C_i,C_(i−1)+τ)`。若相邻commit都至少间隔512，便必有`W_i−H_i≥512`，而stage6全部实测只有399～407。任何p/ℓ/τ都不能同时满足这条局部区间。在“每次commit≥512”的假设下，下一full-wait前至少遗漏105 cycle的release/循环推进；也可能512只应约束服务或平均发射率，而不应限制每一步commit。739556的端点尚不能区分两者；下节用新增相邻端点直接检查，不用自由常数消差。

[局部约束与修订结果](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-local-constraints.md)保留两递推的局部残差与参数敏感性；后续consumer进度观测删除producer字段，只保留同一对stage条件，不增加几何。

### 累计进度把额外时间定位到哪里？

job739578在同一a058卡用相同几何记录CTA0第二输出的commit1/16/17/63、Ktile16的wait1返回和Ktile17的full-wait开始。四个二进制均168寄存器、16 HGMMA、无spill；plain使用相同scratch。以下周期先逐调用求差，stage差再按trial配对；两个累计段分别包含15和46个间隔，不是用单点乘K估算。

| 同调用窗口 | stage4中位 | stage6中位 | stage4−6配对中位 |
|---|---:|---:|---:|
| commit1→16，每间隔平均 | 718.13 | 522.27 | +196.10 |
| commit16→wait1返回16 | 24 | 27 | −3 |
| wait1返回16→full-wait开始17 | 85 | 89.5 | −5.5 |
| full-wait开始17→commit17 | 442.5 | 396.5 | +41 |
| commit16→17（三段合计） | 551.5 | 514 | +36 |
| commit17→63，每间隔平均 | 589.70 | 512.80 | +76.88 |
| commit63→mma返回 | 40 | 40 | 0 |
| mma返回→MAIN_END（handoff与drain合计） | 455 | 460 | −5 |
| 完整L | 39612.5 | 33231.5 | +6388.5 |

每个调用的相邻三段及完整L的不重叠分段精确闭合。stage6相邻commit的十次范围为514～517、后段平均512.783～512.826 cycle；本位置支持接近512 cycle的持续推进，未测出小于512的间隔，但不能据此限制未观测的每次commit。两个旧递推令下一次full-wait在上次wait1返回后立即开始，与实测stage4的84～85、stage6的86～95 cycle直接矛盾。stage4的循环推进反而略短，额外时间主要在full-wait开始到commit这一合并段；该段仍同时包含等输入和发射指令，不能仅据它判为TMA裸延迟。

stage4早段平均差的十次范围为+151.53～+229.80、后段为+60.57～+94.26 cycle/间隔，说明额外时间跨早段与后段累积，且早段更强；尾段不能解释它。原递推/commit推进修订对stage4早段平均仅预测588.67/583.22，后段为587.26/575.86；后段接近不等于早段机制成立。修订还把最后commit→MAIN_END预测为785.72，而实测两stage为495/500，剩余尾延迟仍吸收了别处误差。只增加共享循环项h并继续仅用739095八个whole-L目标训练，中位/最大误差仅降到3.01%/8.13%，局部秩3/4；拟合h=264.43、尾延迟τ=769.17。独立progress对照中，stage6的full-wait开始17→commit17预测为0、wait1为247.57，stage4早段平均也仍只有583.12。这一修订仍分错执行重叠，不继续叠加参数，不将h或τ接入现行规则。

progress/plain完整时间扰动stage4/6中位为+1.01%/+0.72%，十次范围为−1.43%～+1.76%/−1.93%～+3.02%；plain与progress的stage4减速分别为+11.37%/+11.18%。局部观察协议仍有可见影响：739556→739578的stage6 commit→wait1中位由6变27，full-wait开始→wait1由404变447。源码与SASS给出具体原因：旧stage6的commit16 clock（0x21d0）后紧接wait1；新clock（0x1f60）后还执行commit17/63的观测控制，包括predicated CS2R、IMAD和ISETP，才到0x1fb0的wait1。因此6→27包含观察器指令，不能定成裸wait开销，约90 cycle循环段同样不是硬件价格。两批只比较同名端点，不把旧combined窗口与新循环推进拼接；本批没有tail-begin，返回后只报告合并段。[逐调用数据与静态身份](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-progress.json)、[原始记录](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R13-stage-progress/)。

行距方面，cfg_b A16 的实测惩罚 28.94% 只被预测为 1.40%；旧几何 B32/B64 变慢与 V09 新几何 B32 未见明显惩罚同时成立。几何、K 和软件波次数不同，完整比较由 [R10](R10-layout-cache.md#v09-pitch-comparison)维护，不据两批结果宣布 32 B 因果翻转。

当前可继续复用条件关系作开发组合，但需要分别验证：有限 stage 响应、first/later 转移，以及改变频率后计算分支是否暴露未识别服务。不能用代数支持、同批拟合或完整时间误差抵消替代这些检验。[V09 原评分](V09-component-validation.md)保持整体未通过。

<a id="stage-admission"></a>

### 组内发射重叠与共享ready延迟的迁移

cfg_b每Ktile包含8条HGMMA；commit对先前已发起的异步MMA分组，不能把它当作整组计算才开始的时刻，参见[PTX8.8的commit语义](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group)。新增纯诊断递推将累计服务进度E、commit C、退休完成D分开：E_i=max(B_i,E_(i−1))+512，C_i=max(B_i+g,E_i−q)，D_i=E_i+λ；保留wait1释放前一slot、末slot在tail释放，以及跨输出的E和slot。q是有效提交领先周期，不是物理队列容量，512也是本候选的服务假设。

只用739578 stage6的三个局部中位量标定：FIRST→commit1、commit1→16平均间隔、commit17→63平均间隔；**whole-L不参与拟合**。同协议w/h/返回收尾分别取27/89.5/13 cycle，普通release与producer复用先取乐观下界。初始解q≈236.5虽使stage6总L接近，但不满足其他局部端点：FIRST→commit1的最短值要求progress stage6 q≥264、stage4 q≥389；739556的full返回→commit要求stage6 q≥368、stage4 q≥407、g≤105。各界独立由本调用导出，不拼批次时间线。

据这些端点检查q=407～500的五个位置、g=0/105两个边界；每个位置仅重解相同stage6目标的producer间隔与ready延迟。此条件剖面受stage4的另一观察批次约束，**不是未使用过stage4证据的独立验证**。十组均得到ℓ=2257.5 cycle，p存在相同损失参数族；末commit/完成分支允许λ=500−q，不能将这个拆分写成唯一物理延迟。

| 周期量 | stage6实测 | 条件剖面stage6 | stage4实测 | 条件剖面stage4 |
|---|---:|---:|---:|---:|
| FIRST→commit1 | 787.5 | 787.5 | 649.5 | 524～617 |
| commit1→16平均间隔 | 522.27 | 512 | 718.13 | 765.5～796.5 |
| commit17→63平均间隔 | 512.80 | 512 | 589.70 | 759.99～790.32 |
| L | 33231.5 | 33031.5 | 39612.5 | 47978～49931 |

对表列参数还有不依赖优化器的下界：slot复用给C_i≥C_(i−S+1)+ℓ+512−q+w；stage4的后46间隔包含15次跨3间隔的跳跃，其余只用非负界。因此平均间隔至少748.86～779.18 cycle，高于实测十进程范围573.37～607.09。即使立即释放slot，表列参数也太慢。这否定本轮参数与生命周期组合，未穷尽所有常量延迟模型；模型不进入正式预测。

供给状态的作用域仍有独立限制。若只给本CTA添加固定初态的串行服务Q_i=max(I_i,Q_(i−1))+s，首请求响应s+ℓ+max(Q_initial−I_0,0)随I_0延后只能不增；739653却有20/20次更晚issue与严格更长响应。因此孤立版本不能解释完整首次供给，必须明确共享请求前驱或另一可预测状态，不能按delay条件另拟初态。739556 stage6的issue_end→full返回为1858～2091 cycle，只给未排队底值的上界；不能把2257.5原样作为底值再叠排队。

<a id="stage-handoff"></a>

### ready与计算组交接不能由现有校准分开

当前递推令后一输出H=M_prev。实际路径是前WG的mma返回、128线程到达math-order barrier、后WG从等待恢复、首full等待返回。739578 SASS中前WG取mma_return在0x20e0、arrival在0x2110；后WG的math等待在0x1800，首full等待在0x1850/0x1870，FIRST在0x1890。大部分初始化已排在math等待之前，不能用源码行数估算交接价格。该`OrderedSequenceBarrier<2,2>`用独立depth顺序化math与epilogue，EPI_PERMIT也不能证明后一输出的math已经返回。

在连续TC、commit领先分支和首数据已齐的条件下，若前输出也满足M=C_last+40，则后一输出FIRST→commit1=1024−40−δ，δ为省略的交接段。δ=196.5可解释实测787.5，但40来自第二输出、196.5是条件解释量，二者不能当作前输出的直接测量。

| ready延迟ℓ / 交接δ，cycle | stage6 FIRST→C1 / C1→16均值 / C17→63均值 / L | stage4 L |
|---|---|---:|
| 2257.5 / 0 | 787.5 / 512 / 512 / 33031.5 | 49931 |
| 700 / 196.5 | 同上 | 33031.5 |
| 1858 / 196.5 | 同上 | 41941 |

这里固定p=50、q=407、λ=93及原w/h/r；700只作反例，1858来自独立供给上界。**三组没有按stage4挑选，也没有通过新的验证。** 相同stage6观测对应不同stage4响应，足以说明旧ℓ不是已辨识的ready价格。早段512仍低于实测522.27，旧参数的有限窗口反证也仍成立。

739095同次CTA0记录只能给FIRST1−MAIN0：K4096时stage6为−319 cycle、stage4为+535 cycle（十进程中位）。MAIN0晚于MMA0，所以它只是FIRST1−MMA0的下界；负值表示两输出重叠。739556/739578没有前输出MMA0，不能分离交接和full等待。[逐进程来源与范围](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-handoff.json)。

<a id="stage-handoff-measured"></a>

### 同调用补测：交接非零，stage差主要在首full等待

739686沿用cfg_b 2816×1536×4096、stage4/6，在a058/GPU-ef8692f3完成80个有效独立进程：20个plain参照、两stage各10个plain/原progress/新handoff，无失败尝试。CTA0位于SM124；sender thread128处理(0,0)，receiver thread256处理(11,0)。六个binary均168寄存器、16条静态HGMMA、零spill；原progress的GEMM指令行与739578逐行相同。

| 同调用窗口，cycle中位 | stage4 | stage6 | 逐trial stage4−stage6中位 [范围] |
|---|---:|---:|---:|
| sender返回标记→math等待返回 | 148.5 | 133 | 13.5 [0,69] |
| math等待返回→首full等待前 | 3 | 2 | 0.5 [−1,2] |
| 首full等待前→FIRST | 717 | 101.5 | 618 [104,1067] |
| sender返回标记→FIRST | 871.5 | 239 | 631.5 [121,1076] |

全部receiver在sender返回前已经进入math等待，排除本批“接收组较晚才到门口”。三段逐调用精确闭合，各列中位不要求相加。stage差主要落在首full等待，而非交接恢复段；首full仍包含轮询和观察指令，不等于裸TMA服务。以同调用sender标记为起点，到下一输出commit1、commit16、commit63和MAIN的stage4−6中位差依次为631.5、3721.5、7899、7890 cycle，均非仅FIRST边界移动，亦非固定启动差能解释。

**该协议有明显局部扰动。** handoff相对原progress的整调用中位只变+0.38%/−0.63%（stage4/6），但stage6 FIRST→commit1逐trial差为−142 cycle，10/10为负。lastcommit→mma返回多106/103 cycle，随后到MAIN少126/119。SASS显示新协议在循环退出后、返回取时前增加了CTA坐标读取和两个线程谓词判断；这解释了部分窗口的重新划分，不能给每条指令单独定价，也不能将239直接作为旧协议缺失的δ。

两plain参照相对739578中位变化−2.59%/−0.23%，只作跨批背景。各协议保留自己的w/h/r，不能混用旧27/40/500与新43/143/484。条件入口重放见下一节，本批不直接产生硬件定值。[同批扰动、原端点与来源](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-handoff-samples.json)。

<a id="stage-conditioned"></a>

### 固定入口后仍须解释第二输出内部的停顿

条件重放把第二输出H/B分别对齐本调用的首full等待前/FIRST，相对前输出模拟M0定位；保留producer、E前沿和slot状态，A>B即报告不相容。保留旧ℓ=2257.5的见证点在stage6有10/10次入口可行，在stage4只有1/10，最严重晚766.5 cycle。第一输出仍用了第二输出的软件映射，所以单凭入口冲突不能定位失败原因；冲突进程不计算预测误差，也不把观测入口放入正式预测。

更稳的检验只用第二输出17→63的46个commit间隔。由slot复用可得C_i≥C_(i−S+1)+ℓ+max(g,c−q)+w+a；所有代价非负，因此**ℓ+a≤(C63−C17)/floor(46/(S−1))**。stage6仅给2620.67 cycle上界，stage4给1786.73。旧ℓ=2257.5违反后者，不论如何调整交接、q或第一输出初态都不能修复这个见证；这没有否定所有较短的常量ready模型。

后段还必须有模型内部停顿。在当前递推的0≤g≤c、q/λ/w/h非负且g/q在窗口内固定条件下，普通位置c≤E_i−B_i≤2c，端点E−C的变化至多c，所以46步的停顿至少为max(0,C63−C17−47c)。无需指定q上限或λ拆分。即使不固定c=512，仅用stage6得到共同c≤524.1333，stage4仍需至少2166.73～4048.73 cycle内部停顿，中位3130.23。**这是模型解释局部端点所需的停顿，不是TC空闲实测，也没有唯一归因于TMA。**

逐调用波动不另立机制任务：stage6 trial1/9相同(w,h,r)、入口B只差1 cycle，但C1→16差637，共同常量递推至少留下约318 cycle的单调用最大残差，仅约L中位的0.96%。它不能单独证明中位预测失败，不据此扩大打点。

原“stage6定值→stage4诊断”在这里结束，失败与不可辨识结论保留。下一候选将已公开的两stage同列**开发校准**，优先约束共同的复用组合代价及其区间，而非逐个裸延迟；保留有限窗口递推，不把平均间隔直接当稳态等式。新候选另看未参与拟合的K/几何作开发检查，最终仍需新冻结验证，不能把训练数据上的吻合称为迁移通过。[逐调用结果与推导](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-conditioned.json)。

<a id="first-supply"></a>

## 首次供给增长发生在copy发出标记之后

739629沿用R10的同卡、同工作与SM55、0/512 ns起步等待及B0/B32，只增加首输出/Ktile0的两个软件端点：producer完成首对A/B copy后的`issue_end`，consumer thread128在prologue首个`consumer_try_wait`前的`wait_begin`。原FIRST位于`try_wait`与`consumer_wait`之后。producer立即写回新戳；consumer先在寄存器保存wait_begin，待原FIRST之后再写回。它们不是物理发射或硬件ready时刻。

先看同binary的开关对照。events0的S+L行距×延迟交互为−496 ns〔−704,−64〕，events1为−608〔−736,−288〕；两者十次均负，原效应保留。但新增打点使delay0/B32的S相对变化中位+5.49%、范围−9.68%～+35.71%；e1−e0的S+L交互变化为−128 ns〔−640,+416〕，不能称幅度不变。events0保留主要因果对照，events1用于定位数百ns增长，不能把新S绝对值当精确服务价。

以下在events1内先逐进程求直接延迟512−0，范围为十个独立进程配对；cycle来自同CTA/同SM，不经统一GHz换算。

| 增加起步延迟后的区间变化 | B0 | B32 |
|---|---:|---:|
| prod→issue_end，cycle | 0〔−4,+1〕 | −1.5〔−6,+2〕 |
| issue_end→FIRST，cycle | +807〔686,879〕 | +1013〔432,1245〕 |
| issue_end→FIRST，ns | +672〔640,736〕 | +800〔512,960〕 |
| entry→wait_begin，ns | +32〔−32,64〕 | 0〔−64,64〕 |

四条件的prod→issue_end中位均192 ns（约358～366 cycle），新增时间不在这段首次发射前的工作。wait_begin−issue_end在零档约−160 ns、延迟档约−672 ns：consumer已经在等，并非更晚走到等待点。因而现有证据不支持用“prod之后的首次准备变慢”或“consumer更晚开始等待”解释主要增量。异步请求排队、翻译/缓存与数据服务、完成通知的可见性、consumer恢复，以及producer后续stage推进仍未分开。

观察保持相同64 B侧缓冲，关闭时plain/dual、开启时dual，各十进程，加两个plain参照，共140成功进程、无失败。10次独立全输出比较均7077888元素逐位相同。两个二进制均168寄存器、215040 B SMEM、16 HGMMA、零spill；新增打点的整调用变化中位−0.17%～+0.76%，不能替代上述局部扰动检查。两参照相对739619变化+0.85%/+0.07%，只作跨批背景，不估计本批观察误差。80个dual进程的实际工作/SM保持一致，40个延迟进程的producer/FIRST均在同源组第18位；跨进程VA不当同PA。

<a id="ready-observer"></a>

## 首个ready随起步状态变晚，consumer返回不能单独解释

739645的只读stage0 `test_wait(phase0)`保留了主要效应，但按事先规定的`exit≤FIRST`，只有10/40次可绑定第一代。另30次的成功检测区间跨FIRST：观察早于issue，成功退出仅晚2～55 cycle。该批判定保留，不能从后续数据回填。

739653仍固定cfg_b、2304×3072×1024、stage6、AlongM、swizzle1、132 CTA、dyadic/seed17，只比较CTA55 producer起步0/512 ns与B行距0/32 B。观察开启时额外记录thread128、首输出、`release.count()==0`的stage0释放前cycle/ns。cluster1的empty barrier需要128次consumer到达；thread128尚未释放时producer不能再次取得slot0。因此本批仅成功且`exit_c≤release_pre_c`才绑定首代事务。这里约束的是槽复用，不是首次TMA完成时已经发生的full-barrier phase转换。语义依据为[CUDA12.9 PTX8.8](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)；实际SASS中clock/globaltimer读取位于`0x24e0/0x24f0`，早于`0x25e0`的empty arrival。

**本批40/40次都可绑定。** release前戳在FIRST后827～1258 cycle，observer成功退出比release至少早782 cycle，余量明显大于118～134 cycle的检测区间。29次成功退出仍略晚FIRST，按本批真实复用边界均合法；它们不是旧批30次的重分类。最后失败前戳与成功退出戳给出ready区间，不取中点，`ready→FIRST`保留有符号上下界。

同binary开关衡量**轮询与release前标记的总扰动**。四条件首输出完整窗口变化中位为−64、+16、−32、+32 ns，各十进程范围均跨零；整调用中位−0.66%～+0.59%，单对最大绝对4.90%。关闭/开启的延迟×行距交互在L中为−624〔−736,−320〕/−576〔−800,−160〕ns，在S+L中为−496〔−672,−320〕/−560〔−640,−160〕ns，全部十次为负。尤其delay512/B0中S中位增加112 ns（+7.33%，进程范围−11.76%～+24.44%），L减少160 ns，完整窗口的小变化掩盖了局部重分配。方向保留，不能称局部无扰动；以下结论限定本观察协议。

先逐进程求延迟512−0，再对区间端点取中位；cycle来自同CTA/SM，ns直接读取，二者不经统一GHz换算：

| 增量的中位界 | B0 | B32 |
|---|---:|---:|
| issue→ready，cycle | [845.5,1094] | [972.5,1218.5] |
| issue→ready，ns | [704,832] | [752,912] |
| ready→FIRST，cycle | [−133,120.5] | [−125,125.5] |
| ready→FIRST，ns | [−64,64] | [−80,64] |

两种行距的十次配对中，issue→ready增量下界全部为正，最小分别606/848 cycle，直接ns中分别544/672 ns；ready→FIRST的20个增量区间全部包含零。因而“ready不变、只是consumer晚返回”与这些端点不相容。consumer尾项不等于精确零；所有40个绝对ready→FIRST区间还允许一个共同的[42,74]cycle尾项，这只是可行交集，不能命名为唯一硬件延迟。此处ready是该线程能观察到barrier完成的时刻；仍无法把额外时间分给TMA排队、翻译/缓存、实际搬运、完成通知可见性或后续stage请求竞争。

### L缩短不等于持续供给变快

B32中，推迟起步后S增长、L缩短，S+L仍增加。考虑一个最小解释：后续ready根随producer共同平移，边成本不变，仅首consumer开始相对producer多延迟ΔS。固定max-plus完成时间对这个额外起点时移至多增长ΔS，因此必须有 **−ΔS≤ΔL≤0**。这是必要界，不是对硬件的归因或新的拟合。

B32开启观察的十次配对在cycles/ns均满足；关闭时ns十次满足，cycles有一例越下界48 cycle。现有证据不能排除其中大部分L缩短只是等待移到S。B0则在cycles中关闭0/10、开启2/10满足，ns为4/10、6/10；共同平移且固定边成本的单一解释不充分。两个时钟域的差异保留频率、服务和进程离散的影响，不能把单次小幅越界当稳健机制识别。

下一候选应把状态相关的ready服务与consumer返回尾项分开，再用已有stage4/6累计进度、循环推进和first/later窗口约束事件递推；不能由L变短直接提高稳态带宽，不能为延迟或行距条件各加常数。先做这些离线约束，不再追加ready观察矩阵。

本批140进程全部成功、无失败，10次7077888元素全输出比较逐位相同。两binary均168寄存器、215040 B SMEM、16 HGMMA、零spill，144 B侧缓冲容量相同。两plain参照相对739645为+0.06%/+0.38%，只作跨批背景。80个dual的工作/SM映射一致，没有八条件VA原点全相同的trial，不推定PA一致。

## 复核入口

原 18 条件用 `analyze_r13.py --cpu-check` 和 `--input 冻结样本 --output 新目录`；供给开发用 `analyze_r13_supply.py` 的 `--sm-summary`、`--coverage-suite`、`--fill-suite`、`--compose-fill`。各选项的输入和旧命令见上表对应 reanalysis；历史实现细节见[固定 cee978a 正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R13-async-retirement.md)。离线重放不修改原始样本和冻结评分。

[组件消融](V01-validation.md#v09-ablation)中，仅退回聚合周期主循环使总误差从4.11%/35.46%变为4.36%/45.26%，first/later窗口中位误差也更大；但A+16B与部分边界反而改善。因此保留条件供给的同时，优先解释这些反例，不据整体均值给所有分支背书。
