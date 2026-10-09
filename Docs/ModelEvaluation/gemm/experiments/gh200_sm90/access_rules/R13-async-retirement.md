# R13：供给与退役

实现与本文作者：B；GPU采样由主对话串行调度；未实施R13的C负责独立源码、SASS及原始工件复核，各冻结报告见下方证据链接。

## 结论

- 原问题是单 SM 供给上限是 55 还是 64 B/cycle。本组在真实 WGMMA 消费、4 槽、每 SM 一个 CTA、共享小源条件下测得：32 KiB tile、4 stage 为 58 B/cycle，48 KiB tile 为 45 B/cycle。这是条件化的序列速率，不是通用上限。
- 2026-10-07 依据 V05 判断这个问题对预测已不关键：V05 中 cfg_a/b 的主循环误差中位数 0.9%，主循环按计算下界即可。这个判断只适用于 V05 的条件：6 个尺寸的 A/B 行距都是 128 B 倍数，没有 swizzle 补齐，K 最长 12288。
- V08 超出了这个范围，上一条不再代表当前结论。A/B 行距变化和部分大足迹长 K 条件暴露了主循环额外代价，cfg_b_h03 的总时间预测偏短 35%。2026-10-09 的[离线候选检验](#v08-supply-fit)表明：共享 A/B 行距供给项能降低 cfg_b 残差，但不能同时解释 cfg_a/b/c；增加 cfg_a 参数后，已有 K 组之间的最大迁移误差仍达 11.16%。计算受限点只约束供给能力下界，当前不交付通用带宽或请求放大常数。每 SM 有效供给仍为 [PLAN 主线](PLAN.md#mainline)的首要扩展。

## 问题、矩阵与协议

SMEM中的A为K-major、B为MN-major，FP16输入/FP32累加；输入32/48KiB（128×128×64、128×256×64）×stage1/2/4×直接退休/64依赖FFMA/真实WGMMA，共18条件。每序列32Ktile，同形状固定4槽、384线程，整卡132CTA。所有CTA复用同一份A/B小源；4槽固定预留128/192KiB输入SMEM，另有barrier及publication存储，实测132CTA覆盖132SM、每SM一个CTA。

GMEM中的A按tile-major连续存放32个128×64块，half元素地址为`tile×128×64+r×64+k_local`，不是普通A[M,2048]的row-major/lda=2048布局。B为真正的全局B[K,N]行主序，ldb=N；3d TMA视图[64,2048,N/64]、strides[N×2,128]B、box[64,64,N/64]。两个consumer WG各自等待full、消费后组会合并向计数2的empty arrive；producer等两组退休才覆盖。c2每组每tile四条K16 MMA：stage1 wait0退当前，stage2/4首tile剥离、steady wait1退前一、consumer分支内尾wait0退最后。发布独立序号，途中不读异步累加器；完整输出计时后保存。c1读取当前A值后执行64依赖FFMA，发布真实结果。

## 计量与适用范围

plain窗口覆盖供给、消费、退休和最终CTA发布。每个进程在各SM内计算max(end_cycle)−min(begin_cycle)，再求和；B/SM-covered-cycle跨10plain进程给均值/范围/CV，pooled另列。跨度包含间隙，不跨SM减时钟；不是HBM/L2物理带宽或无限长流水上限。

ready事件为issue→wait_return；retire为consume→retire_arrive_start（empty.arrive前）。返回/发起时间均不是硬件内部完成瞬间。v8 ready观察全部CTA选定tile，v9只观察CTA0选定tile，其余CTA仍执行全部计算搬运且trace为0。两个事件对分run，不拼接绝对时间线。5%门槛保持冻结的trace/plain均值定义，逐pair范围及超限对数另列；均值满足扰动要求不等于每pair无扰动，近plain波动的小差异不判因果。

## 正式结果（2026-10-07）

job735876、735985的精确environment均为romeo-a053、GPU-54896349-d69d-9358-b526-433454c04733、CUDA12.9/sm90a/NDEBUG。v8与v9 plain的18个完整GPU函数SASS逐字一致；各18specialization，无C75xx/非零spill，c2异步wait1及尾wait0经独立核对。各run单独统计，不拼接样本。

v8 ready18×10对=360进程、397393920累加器元素；v9 ready补查5×10对=100进程、131788800元素；v9 retire18×10对=360进程、397393920元素，三个归档均在本地完整重算通过。每个正式plain进程实际132SM/132CTA，最后完整输入槽也核对。ready正式与retire正式预热均收敛；ready补查n128_s1_c0的plain trial1/8未收敛，原样保留。原v8所有18plain继续有效。

下表以v8 ready-run的plain均值为主服务表，单位B/SM-covered-cycle；retire-run自己的plain统计在独立报告中，不合并。

|输入KiB|stage|直接退休|64依赖FFMA|真实WGMMA|
|---:|---:|---:|---:|---:|
|32|1|35.677|26.681|22.552|
|32|2|56.435|49.526|28.743|
|32|4|83.077|65.847|58.046|
|48|1|43.719|35.280|22.849|
|48|2|64.836|58.816|35.433|
|48|4|85.031|83.385|45.337|

32KiB/WGMMA/stage4实测58.046，48KiB为45.337，不能据此把旧55、当前64或需求48替换成无条件上限。工作与退休组织影响窗口；这里只交付共享小源、GMEM A tile-major/B row-major、固定4槽及每SM一个CTA的条件序列率，不能作为普通row-major GEMM的普适供给上限。v8 plain最大时间CV3.669%；v9 retire最大2.752%，不能用小差异证明物理因果。

ready原5个均值超限条件经CTA0有界补查后，仅n128_s2_c1均值4.703%通过（5/10单pair仍超5）。n128_s1_c0、n128_s2_c2、n128_s4_c1、n256_s1_c2仍超限，只保留定性顺序，不作阶段时长的定量判定；n128_s1_c0还有上述预热限制。retire均值超限6项：n128_s2_c2、n128_s4_c0、n128_s4_c1、n128_s4_c2、n256_s1_c2、n256_s2_c2，同样仅定性。其余事件条件仅在各自观察范围内通过均值扰动检查，不能推到所有pair/所有SM，也不能据此解释硬件内部完成时间。打点失效的条件仍保留全部plain样本。

## 证据与复现

- [v8 ready正式报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job735876-v8-formal-ready-review-B/report.md)：parameters.json、逐plain/SM覆盖、逐pair扰动、图与分析源码身份同目录。该报告依据的原始运行 `20261007-R13-job735985-v9-ready-recheck` 已压缩归档到 `CUDA_optimazation_archive/gh200_access_rules/`（见其 README 与 SHA256SUMS）。
- [v9 ready补查报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job735985-v9-ready-recheck-review-B/report.md)：parameters.json、逐plain/SM覆盖、逐pair扰动、图与分析源码身份同目录。
- [v9 retire正式报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job735985-v9-formal-retire-review-B/report.md)：parameters.json、逐plain/SM覆盖、逐pair扰动、图与分析源码身份同目录。
- [retire独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job735985-v9-formal-retire/reviews/independent-C.md)。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r13.py --list
python3 microbench/gh200_resource_campaign/access_rules/analyze_r13.py --cpu-check
python3 microbench/gh200_resource_campaign/access_rules/analyze_r13.py \
  --input /冻结样本归档 --output /新分析目录
```

复现使用准确CUTLASS3.9.2、CUDA12.9及冻结source/build；新GPU运行由主对话调度。v1错配K/K、v2额外全fragment归约、v3/v4逐tilewait0和早期高扰动、v5–7实际串行化等检查失败记录均作为历史诊断保留，不覆盖。v8流水与v9 CTA0 observer分别冻结。R13的18条件已取得数值与plain服务观测；原ready/retire尚不直接观察槽可复用与下次补发，不能据此宣称PLAN事件链完整。上述超限项不作事件时长的定量判定，5%门槛不放宽。

## 槽可复用与补发直接观测

新增`--trace-pair reuse`，不增加原18条件。仅CTA0 producer在处理
`trace_tile+Stages`（默认16+Stages）时，等待两consumer的empty代际完成；
成功acquire后在同一PTX块记录`slot_reusable_observed`。这是
producer确认可复用后的观测边界，不是物理barrier完成瞬间。完成坐标与expect_tx
准备后，下一次A TMA请求前记录`refill_issue_before_A`；两次clock来自同一次
调用、同一SM，trace store在A/B发出后进行，不将两类历史run拼接。

packed数据只在选定tile的第一个记录word3/4保存这对producer事件；第二个记录、
其他tile、其他CTA均0。它不伪造consumer时间戳，原ready/retire仍沿原语义解释。
plain数学、32tile、4槽和资源不改；v11原生SASS、完整数值和字段检查已通过，可用于定量比较的范围如下。

reuse沿用原18条件。扰动超限的条件只保存定性顺序，不作事件时长的定量判定；各冻结版本和原始档案分别保留。

v10代表检查共40进程，保存数值核对通过，但N128的10个reuse记录word3均为0，该事件打点检查未通过。实际SASS的fast-success路径用P0分支，取时后却以只在retry路径定义的P2执行SEL，导致取时结果被清零；N256此次没有零字段，不能据此证明该路径可靠。v10归档原样保留，不采用该事件对。

v11只修复`reuse_wait_stamp`：输出初始化为0、约束改为`+l`，acquire循环成功退出后无条件读取clock64。冻结probe SHA256为`fc125fae9ebc15b6013488dac00394132f245c15240774cbe611b92c8b153660`。源码及B对原生SASS的检查通过：18个plain完整Function与v10逐字一致，四份编译日志无C75xx或非零spill。N128/stage4/c2的fast/retry成功路径都到1300的无谓词CS2R R12，word3在1570直接保存；N256对应14d0的CS2R R10及1780保存。两次TMA均先于trace stores，旧P2清零路径已消失。代表与完整18条件reuse均已完成采样及数值/字段复核；是否满足扰动要求按正式矩阵逐条件判定。

### v11 reuse代表结果

同job735985、a053及上述UUID，两点各10对，共40进程、129761280个FP32输出与最后完整输入槽本地重算通过；预热均收敛。20条trace只有CTA0、退休tile16的第一个packed记录word3/4非零，补发tile20/slot0；其余CTA/tile/第二记录均0。全部满足CTA0窗口内`slot_reusable_observed>0`且不晚于`refill_issue_before_A`。

|条件|trace/plain均值扰动|逐pair范围|单pair超5%数|两观测gap范围/cycles|
|---|---:|---:|---:|---:|
|n128_s4_c2|4.781%|3.429%–5.764%|5/10|67–82|
|n256_s4_c2|0.466%|−0.463%–1.094%|0/10|59–70|

两点通过冻结均值门槛，N128不能称每pair均通过。gap仅覆盖成功acquire后的观测到下次A请求前的准备/取时，不含此前consumer工作或empty等待，不能解释为总消费、释放等待或硬件内部完成延迟。plain序列率分别58.017、45.298 B/SM-covered-cycle，仅属本run，不替换或拼入历史18点表。

[代表本地重算报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job735985-v11-rep-reuse-review-B/report.md)保存独立分析身份、逐pair和每SM分母；`reuse_event_boundaries.json`保存20条同调用事件及scope。[独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job735985-v11-rep-reuse/reviews/independent-C.md)绑定冻结原始档案。代表批不替代完整正式批，不增加条件或放宽门槛。

### v11 reuse完整正式结果

job735985到期后的原部分归档326个result组成163个完整配对，无已保存半配对，缺17对；原目录保持partial且无summary。job736169在同一a053/GPU UUID上补17个完整配对，新归档共18×10对、360进程。所有163旧完整配对都带原result/environment SHA；2445个复制文件逐字节哈希一致。17新配对与独立缺失清单一致，每对两边来自同作业，source/binary manifests一致。逐case分段统计保留；新段每case至多1对，不能视为独立校准。

完整397393920个FP32输出及最后TMA输入槽本地参考复算通过；全部预热收敛，180个plain均132SM/132CTA。每SM覆盖分母核对，plain elapsed最大CV3.048%，B/SM-covered-cycle最大CV2.282%。180条reuse trace均只有CTA0、退休tile16的第一个记录word3/4非零；补发tile16+Stages/slot16%Stages，全部在同CTA窗口内非零且正序，其他packed记录均0。

冻结均值扰动门槛16/18通过，以下两项仅保留定性顺序，不作事件时长的定量判定：

|条件|正式均值扰动|逐pair范围|单pair超5%数|
|---|---:|---:|---:|
|n128_s1_c0|5.750%|3.017%–11.349%|4/10|
|n128_s4_c2|5.903%|2.841%–8.357%|8/10|

n128_s4_c2代表批4.781%通过与正式批5.903%失败分别保留，正式判定以完整批为准，不挑样替换。其余16项通过的是均值门槛，单pair超限及负扰动仍按原值保留，不推断打点改善硬件服务；打点失败的条件仍保留所有数值正确的plain样本。

[完整复核报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job736169-v11-formal-reuse-review-B/report.md)及其`analysis/report.md`给出18点服务/扰动统计；`continuation_review.json`保存来源、复制哈希和分段；`reuse_event_boundaries.json`保存180条事件边界及检查结果，`qualified_reuse_intervals.json`只含16个满足扰动要求的条件。[独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R13-job736169-v11-formal-reuse/reviews/independent-C.md)绑定完整冻结工件。

R13的18条件已完成plain及ready/retire/reuse三个事件对的采样与检查；原缺少的槽可复用/补发字段已直接补齐。事件对来自独立调用，不能拼接同一次完整绝对时间线。字段完整、数值正确和扰动是否满足要求分别判定：历史ready/retire失败及本reuse两项失败均保留，不能宣称18条件所有阶段时长全部取得。

2026-10-08：被取代的 R13 运行（v8-rep-ready、v8-rep-retire、v9-ready-recheck、v10-rep-reuse 与 job735985 的旧 v11-formal-reuse）已压缩归档到仓库外的 `CUDA_optimazation_archive/gh200_access_rules/`。最终目录 `20261007-R13-job736169-v11-formal-reuse` 已包含续采所需的全部原始样本（326 个结果文件哈希一致），可独立重算；`v8-formal-ready` 仍保留在 results。

<a id="r05-d"></a>

## 立即释放消费者的每 SM 供给（原 R05-D，2026-10-06）

2026-10-08 由 [R05](R05-async-lifecycle.md) 迁入，R05 的其余子集仍在原页。这里的消费者在 mbarrier 完成后立即释放槽、不做计算；上文正式表的消费者是直接退休、64 依赖 FFMA 和真实 WGMMA。几种消费者组织的数值分列，不合并成一个上限。

问题：V01 目标组合（128×256×64 tile、4 stage）每个 SM 只驻留 1 个 CTA，在途输入最多 192 KiB。满速 WGMMA 每 Ktile 约 1024 cycle、需输入 48 KiB，即约 48 B/cycle/SM；这样的单 CTA 能否从 L2 拿到这个速率？

- **配置**：grid = SM 数；动态 SMEM 预留不小于目标的 192 KiB 加 barrier，使每 SM 只能驻留 1 个 CTA（用 API 核对，并记录每个 CTA 的 SM ID）。每 stage 用 2D TMA 读一个 A tile（box (64,128) FP16，16 KiB）和一个 B tile（box (64,256) FP16，32 KiB），SW128，与目标组合相同。160 线程：warp 0 发 TMA，其余 128 线程为消费者；10 个进程都覆盖了全部 132 个 SM。
- **流水**：producer warp 发 TMA；一个 consumer warpgroup 等对应 mbarrier 完成后立即释放槽，不做计算。这样测的是无计算消费者条件的供给观测，不表示所有组织策略的严格供给上限。
- **源**：共享小源让所有 CTA 轮流读取同一个 L2/4 大小的 A/B 区域，模拟多个 CTA 复用同一 K 面板，实际访问 15 MiB，每个 Ktile 面板被复访 320 次。独立大源让各 CTA 读互不重叠的区域，合计 4×L2，实际 241 MiB，复访 39 次。
- **指标**：运输量为 `CTA数×Ktile数×48 KiB`。每个 CTA 用自身 `clock64` 窗口算 B/cycle，即该 SM 的供给率，报告 132 个值的中位数；整卡另报 GB/s。

| 源 | stage 2 | stage 4 |
|---|---:|---:|
| 共享小源 | 55.45 | 55.48 |
| 独立大源 | 15.26 | 15.26 |

单位 B/cycle/CTA，132 个 CTA 的中位数，job735059。有 L2 复用时单个目标 CTA 的供给高于 48 B/cycle 的满速需求，没有复用时只有约 1/3；stage 2 与 4 相同。

[R05 B/C/D/E 报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-formal-v2/report.md)，`cases.csv`、`rules.json`、图在同目录；入口 `run_r05.py`、`analyze_r05.py`。

<a id="v08-supply"></a>

## V08 后续对照中的每 SM 供给（2026-10-08 迁入）

job737322，romeo-a043，GPU-099dda56，CUDA 12.9.41、CUTLASS 3.9.2、`sm_90a`、NDEBUG；cfg_a/b/c 与 V07/V08 相同。V08 留出评分后在同一张卡上加测 84 个条件，每个 plain/stamped/ends 各 10 进程，只用于解释，不回填 V08 判定。这些是完整 CUTLASS GEMM 主循环的条件观测，行距与尺寸逐项表见 [R10](R10-layout-cache.md#v08-stride)。它们与上文探针的序列率协议不同，不合成一个上限。

**按每 SM 的逻辑源请求解释。** 每个 Ktile 的计算候选为 512 cycle（cfg_c 为 1024）。按 cluster 多播摊分源请求，每 SM 每 Ktile 为 cfg_a 24 KiB（A 16 KiB + B 8 KiB）、cfg_b 32 KiB、cfg_c 32 KiB。对应计算满速需求为 48、64、32 B/cycle/SM。这不是测到的 L2 字节。下表保留原分析的“逻辑请求量 / 有限主循环周期”描述值，不能作为供给能力标签；2026-10-09 分析采用明确的进程聚合口径，见后文。

| 条件 | cfg_a | cfg_b | cfg_c |
|---|---|---|---|
| 原分析的对齐参照 | 48（接近计算限） | 64（接近计算限） | 32 |
| 大足迹（105–260 MiB）长 K | 47.0–47.9 | 57.3–63.7 | 31.8–32.0 |
| 一个操作数行距未对齐 | 45.2–47.6 | 46.1–56.7 | 32.0 |

这些普通 tile 条件中的 cfg_c 接近计算下界，因而没有暴露供给上限。`max(计算, 条件供给)` 是待检验的候选；行距变化是否增加物理传输、工作集变化是否降低 L2 命中，均未由这些计时确定。可观测的是条件主循环代价，不是唯一的“放大倍数 × 物理供给上限”分解。

[结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/followup-v1/followup.json)。

<a id="v08-supply-fit"></a>

## V08 供给候选的离线检验（2026-10-09，A）

**当前可辨识的结论。** A/B 行距效应需要分开；cfg_b 对 B 行距的额外代价大于 A，cfg_c 的这些普通 tile 记录只给供给下界。统一有效供给常数不成立；共享 A/B 行距项显著改善拟合，但仍将 cfg_a、K=4096、A 行距变化低估 11.05%。为 cfg_a 增加两个行距参数后，拟合最大误差降到 5.71%，留 K 组的最大误差却为 11.16%。本次没有获得可直接接入完整预测的通用供给函数，也没有否定允许配置、并发和流水状态变化的 max 形式。

拟合只读复用 job737322 的 V08 校准和 V08F 后续摘要；另针对 cfg_a 的残差重读六条件的 60 份 stamped trace，没有新 GPU 数据、全档重放或 V08 重新评分。[分析源码](../../../../../../microbench/gh200_resource_campaign/access_rules/analyze_r13_supply.py)、[数值与留组结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/A-20261009-supply-v3/summary.json)、[逐条件残差](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/A-20261009-supply-v3/residuals.csv)。

### 数据与窗口

84 条件中，54 条满足 swizzle=1、完整 M/N cluster、K 为 64 倍数并有后续输出 tile。它们均满足原有 plain/stamped CV 与打点扰动绝对值不超过 5% 的条件；将 3 个 D 行距对照另列，主拟合为 **51 条，每配置 17 条**。另保留 14 条尾部/D 行距迁移诊断；`cfg_b_p_ktail` 的原打点扰动为 5.34%，不作定量迁移评分。18 条 swizzle=8 不参与本供给拟合，其中即使没有整个 cluster 越界，也不与 swizzle=1 合并。

`L0` 指每 CTA 首个**输出 tile**的完整 K 主循环，`L` 指后续输出 tile，不是首个/后续 Ktile。起点为首次 full-barrier 等待返回后、首 MMA 前；终点在 `mma_tail` 的 `warpgroup_wait<0>()` 和末尾 stage 释放之后。因而窗口含后续输入等待、循环与最终排空，首次输入等待在窗口之外。不能再把最终 WGMMA 排空加到 `w`。

本次目标量为 `intervals.L`：每进程对后续输出 tile 求均值，再跨 10 进程取中位数。`tile_L.in.median` 是跨进程池化中位数，另存作描述，不混入拟合。后续集合仍含每 CTA 最后输出 tile 以及 cfg_b 的不同输出重叠状态，尚未直接观察内部逐 Ktile 稳态。摘要不含逐进程 L 分布，完整 event 的 CV 不能代替 L 的置信区间。

### 最小候选与残差

对所有配置同时拟合：

\[
\widehat L=b_{cfg}+K_t\max(C_{cfg},D_{cfg}q),\qquad
C=(512,512,1024),\quad D=(24,32,32)\ \mathrm{KiB}.
\]

计算项依据既有 4096 FLOP/SM-cycle 指令服务，`q` 的单位为 cycle/逻辑源 KiB。它合并请求组织、等待与缓存等条件影响，不命名为物理带宽倒数。拟合直接最小化各条件相对周期残差的平方；计算分支不以 `D/L` 提供供给标签。在该分支下，只要求候选供给能力至少满足 48/64/32 B/cycle 的相应需求。

`b_cfg` 从 V08 合格、swizzle=1、无 cluster/K 尾部的校准记录取 `median(L−Kt·C)`，分别为 **356.070、623.314、430.471 cycle**。这是有限窗口修正，可能吸收校准集内的供给代价，不能解释成已测出的纯启动或排空。下表的参数数目只计供给项，三项 b 均固定。

| 供给候选 | 供给参数数 | 51 条中位绝对误差 | 最大绝对误差 | RMS |
|---|---:|---:|---:|---:|
| 仅计算，`q=0` | 0 | 0.44% | 28.61% | 6.88% |
| 统一 q | 1 | 0.53% | 23.52% | 5.75% |
| `q0+qA·IA+qB·IB` | 3 | 0.44% | 11.05% | 2.56% |
| 上式加输入足迹项 | 4 | 0.44% | 11.05% | 2.48% |
| 行距项加 K 项 | 4 | 0.44% | 11.05% | 2.48% |
| 行距、输入足迹加 cfg_a 专属 A/B 项 | 6 | 0.42% | 5.71% | 2.11% |

`IA/IB` 表示相应字节行距模 128 非零。纯行距对照只覆盖余数 16 B，不能从二值项预测完整余数曲线。输入项为 `log2(1+W_AB/32 MiB)`，K 项为 `log2(Kt/16)`；32 MiB 仅用于特征缩放，不是拟合或测定的缓存容量。`W_AB=2K(M+N)` 不含输出；原摘要 `fp` 包含 D 且忽略行距 padding，二者与实际分配容量在结果中分别保留。它们都不是 L2 驻留量。

四参数输入足迹候选的 `q0/qA/qB/qW` 为 **16.175895 / 2.275609 / 6.097226 / 0.258635 cycle/KiB**，只供复算本次诊断。其 cfg_a/b/c 最大误差分别为 **11.05% / 5.18% / 0.47%**。cfg_c 全部位于计算分支，不能由小误差认定其供给参数已识别。六参数结果只是检查配置项是否能吸收残差，不作为规则选择。

将 b 改为各配置校准残差最小值后重新拟合，四参数候选最大误差为 11.37%；令 b=0 后为 12.01%。六参数对应 6.05% / 6.73%。负结果不依赖所选中位截距，但这些敏感性比较没有把窗口代价分解成物理机制。

### 配置、K、工作集与首段的限制

同为 M×N=2304×3072、K=4096，后续主循环观测如下，单位 cycle/Ktile，未扣 b：

| 条件 | cfg_a | cfg_b | cfg_c |
|---|---:|---:|---:|
| 对齐 | 548.933 | 546.456 | 1029.457 |
| 仅 A 行距增加 16 B | 581.891 | 622.912 | 1029.420 |
| 仅 B 行距增加 16 B | 582.535 | 730.853 | 1029.472 |

在选定 b 下，对齐条件的 cfg_a 剩余约 543.37 cycle/Ktile，cfg_b 约 536.72。二者 C 相同而 D 相差 4/3：若 cfg_a 的这部分超额全由一个共享 q 解释，便会把 cfg_b 的供给项推到约 724.49 cycle/Ktile。这是共享供给项的直接矛盾；可能缺的是配置相关服务或窗口中的其他代价，不能从矛盾选择某个 bank 解释。

同一形状从 K=1024 到 4096，`ΔL/ΔKt` 的对齐/A/B 条件分别为 cfg_a **554.58/594.97/598.06**、cfg_b **545.47/623.82/725.20**、cfg_c **1023.86/1023.91/1023.83**。K 与输入足迹同时增大，这些是两点割线，不是已隔离的稳态斜率。足迹候选与 K 候选的 RMS 几乎相同，现有矩阵不能在两者之间作因果选择。

cfg_b 大足迹对照的 `L/Kt` 在原 `fp=185/217/260 MiB` 下为 **514.782/571.442/546.577**。其中 185 与 260 MiB 两点的逻辑输入容量同为 160 MiB，但 M/N、K 和输出 tile 数不同；仅用输入容量或原总足迹都不足以解释它们。

后续 L 拟合不等于首段解释：用相同供给系数和单独的校准首段截距预测 L0，四参数候选最大误差 **20.98%**，六参数仍为 **15.27%**。启动供给 S、首输出 tile 与后续 tile 需要分开，不能让供给常数包办这些差异。

### cfg_a 残差的输出 tile 位置

只对上述对齐/A/B 行距 × K=1024/4096 六条件，沿用原 R18 读取器重读 60 份 stamped trace；原始文件哈希和 245760 个已保存数值通过原参考检查，重算的后续 L 与摘要一致。这不是对全档正确性的新复核，也未增加 GPU 测量。

K=4096 的逐位置主循环如下，单位 cycle/Ktile；每进程先对该位置 CTA 求均值，再跨进程取中位数。j 从 0 开始，j=0 是首输出 tile。

| 行距条件 | j=1（132 CTA） | j=2（132 CTA） | j=3（36 CTA） |
|---|---:|---:|---:|
| 对齐 | 579.27 | 526.39 | 515.77 |
| A +16 B | 629.96 | 553.29 | 515.78 |
| B +16 B | 627.00 | 557.61 | 515.77 |

相同行距/K 的后续位置仍有明显差异，最后一组输出 tile 已接近计算下界。单个“行距→供给”常数无法描述这一变化；汇总 L 的残差并不全来自一个稳态服务率。j=3 的 CTA 集合也与 j=1 不同，序号下的 CTA 数不是实际同时活动 SM 数，不能把差异直接归因于全卡带宽竞争。结果 JSON 保留逐进程位置均值及 first/middle/last 分组；first 的位置均值与原 L0 的进程内中位数不混用。

### 留组诊断

以下均为已看过数据上的开发诊断，不是新留出。每次从 51 条中移除某配置、某 Ktile 数或某 M×N，重拟合供给参数；V08 的 b 保持固定，因此“留配置”只检验供给关系迁移，不验证该配置的窗口校准。

| 候选 | 留配置最大误差 | 留 K 组最大误差 | 留 M×N 最大误差 |
|---|---:|---:|---:|
| 共享 A/B 行距项 | 34.37% | 11.05% | 7.23%（39/51 条可预测） |
| 加输入足迹 | 33.47% | 11.05% | 6.89%（39/51） |
| 加 K | 25.35% | 11.05% | 9.70%（39/51） |
| 加 cfg_a 专属 A/B 项 | 33.47%（43/51） | 11.16% | 7.33%（39/51） |

移除 2304×3072 后，训练集中没有非零 A/B 行距余数，12 条行距测试缺参数，明确不评分；不能把默认零系数当成功迁移。六参数模型移除 cfg_a 后，4 条 cfg_a 行距条件缺专属参数；移除 cfg_b 后，cfg_c 仍在计算分支，训练只能约束共享项与 cfg_a 专属项之和，因此另 4 条 cfg_b 行距预测非唯一，也不评分。逐折条件、系数缺口和有符号残差均在 JSON 中。

未参与拟合的 14 条尾部/D 行距诊断，四参数候选最大误差为 3.14%。这说明已见余数的部分变化能在本批局部迁移，并不抵消配置、K 和首段的失败。

### 最小后续工作

1. **先补独立行距余数，见 [R10 条件](R10-layout-cache.md#v08-stride-next)。** 现有 A/B +16 B、K=1024/4096 已保留，不重测整套 84 条。补 A/B +64 B 的三配置两 K 共 12 个新条件，再补 cfg_b、K=1024 的 B +32/+128 B 两点；新的同批对齐参照按六个 cfg×K 共用。它们检验余数形状与配置迁移，不能单凭时延命名请求放大。
2. **若要分离工作集与 K，沿用已定的六条件 R13 配对。** cfg_b、1536×1280×4096、swizzle=1，stage=2/4/6 × 重复同一 A/B 或轮换独立 A/B；池大于该卡 L2 两倍、D 地址和写历史一致，按六 stage 预留相同 SMEM。固定 K 后检查缓存准备与有限 stage 的响应，仍只得到条件服务。现有 V08F 没有这组对照。
3. **额外端点只补 cfg_a 已暴露的位置变化。** 现有 trace 已定位第二至第四输出 tile 的差异，但每个 L 仍覆盖整段 K，无法分开内部稳态、等待与排空；先在 K=4096 的对齐/A/B 三个代表条件各取相同 CTA、相同输出 tile 位置，增加少量起段/中段/排空端点并比较原打点扰动。不要把不同位置或不同 CTA 集合当成同一个稳态样本，也不据此添加 bank 常数或硬件队列容量。

本次没有修改公共 runner、探针或模型接口。后续公共框架需要保留实际 A/B/D pitch、cfg、K、输入/缓存准备与原 first/rest 端点；内部打点和 stage 对照的实现由管理对话协调，不能从本次 CPU 分析推定已经可采样。

复核命令（输出目录必须尚不存在）：

```bash
python3 microbench/gh200_resource_campaign/access_rules/analyze_r13_supply.py \
  --run /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1 \
  --calibration-run /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1 \
  --output /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/A-20261009-supply-recheck
```

<a id="requested-sm-pitch"></a>

## 请求 SM 数与行距的最小对照（2026-10-09）

本批回答：**固定 GEMM 形状时，A/B 行距惩罚如何随持久 grid 规模变化？** 四档会改变每 CTA 的工作轮次、缓存访问历史和调用内频率，不能从一条扫描曲线唯一定位 SM 本地或共享供给瓶颈。主比较是同一档内的行距配对；跨档另取严格匹配当前输出工作的子集，不增加矩阵条件。

旧数据确认采用 cfg_a、**M×N×K=2304×3072×4096**，swizzle=1、重复同一 A/B、dyadic 输入、seed=17。共有 432 个完整输出 tile，无补齐，沿用 6 stages。三个布局为 aligned（lda/ldb/ldd=4096/3072/3072）、A+16 B（仅 lda=4104）、B+16 B（仅 ldb=3080）；请求 SM 数为 32/64/96/132，共 **12 条件**。不展开其他行距或 stage。输入逻辑容量均为 42 MiB，另有 27 MiB 输出；padding 分配变化在实际 stride 中保留，不将逻辑容量称为 L2 驻留。

由已核对的调度计算得到以下预期，运行时仍用 `setup.grid` 和实际 trace 重建：

| 请求 SM 数 | 预期 grid | 每 CTA 总输出 tile 数 T | 末轮 CTA 数 |
|---:|---|---|---:|
| 32 | [32,1,1] | 16 CTA×13；16 CTA×14 | 16 |
| 64 | [64,1,1] | 16 CTA×6；48 CTA×7 | 48 |
| 96 | [96,1,1] | 48 CTA×4；48 CTA×5 | 48 |
| 132 | [132,1,1] | 96 CTA×3；36 CTA×4 | 36 |

**三个比较口径。** 同一请求档内，按 CTA ID、输出序号 j、CTA 总 tile 数 T 和实际 M/N 坐标逐一匹配 A/B 与 aligned，先求进程内平均差，再跨进程汇总；每 Ktile 周期除以 64。first/middle/last 由 j 与 T 决定，不混成一个均值。跨档的所有 j/T 分组分别保留，不把不同 T 合并成稳态。

96 与 132 档还有一个更直接的现有子配对：T=4、j=1 和 j=2 各有 **24 个相同 M/N 输出坐标**。按 `(j,T,M_idx,N_idx)` 合并后比较 `(A−aligned)_132−(A−aligned)_96`，B 同理。j=0/3 没有这种同坐标交集。它固定当前工作和总轮次，但此前 tile 坐标与其他 CTA 的访问历史仍不同，因此结果仍称 grid 规模响应，不称物理共享带宽。

**实际并发的可观察范围。** 保存每个 CTA 的 SMID、入口/最终端点的 globaltimer 和 clock64；先对同 SM 的 CTA 观测窗口求并集，再扫不同 SM 的窗口重叠数，报告峰值、时间加权值和原窗口。该数衡量 CTA 观测生命周期的重叠，不表示它们同时执行 TMA 或 MMA。调用内频率只从同 CTA 的两种时钟跨度计算，按周期比较 L，同时保留频率。每输出 tile 当前只有 clock64，不跨 SM 相减，也不用插值假造逐 tile 并发。

公共端点的最小需求是给 `FIRST_MMA` 和 `MAIN_END` 各补 globaltimer；cfg_a 两个 consumer 分别保存，共每 CTA 每输出 tile **4 个 uint64 字段**，保留原周期字段及坐标。这样才可直接排列主循环窗口，并须与旧 stamped 核对扰动。本批现有二进制能先完成 CTA 窗口与配对分析，不声称已经提供逐 tile 的全局时间。

### 运行包与 CPU 检查

[专用入口](../../../../../../microbench/gh200_resource_campaign/access_rules/run_r13_sm.py)复用原 `run_r18.py` 的 setup/sample、配对顺序、锁和检查；[专用分析器](../../../../../../microbench/gh200_resource_campaign/access_rules/analyze_r13_sm.py)复用原 trace/数值读取器。公共源码未改。

二进制来源为主数据目录下 `20261009-R09-R13-shared-smoke-job738097`，运行源码固定为归档 **6338653b5a4127a7cb947b6355c38642310f3dcc**，分析器为本次提交。prepare 核对归档自身的源码/二进制清单及 SASS 哈希，保留归档 source，只取 cfg_a 的 plain/stamped 二进制和对应构建证据，并加入本批专用 Python 入口；不要求当前 checkout 的公共头与归档相同。不复制 smoke 样本，不重新编译。新 setup 的实际 CUDA GPU UUID 写入记录并用于样本核对，旧卡数据不替代新批次的 aligned 参照。

正式为每条件 10 对 plain/stamped，共 **240 进程**。每个请求档内三种 pitch 相邻，顺序随 trial 随机；plain/stamped 也相邻并随机换序。可先用 `R13_PROCS=1` 做 24 进程 pilot，在同一分配、同一源码下扩到 10 时复用 trial0；单进程结果只作诊断。沿用原预热和 5% CV/打点扰动口径，逐 trial 差分与失败记录保留，不由完整时间接近证明各主循环窗口没有扰动。

CPU 检查已确认 12 条件的参数传递、432 个完整坐标、每 CTA 总 tile 数以及 96/132 的 24+24 坐标交集；[四项测试](../../../../../../microbench/gh200_resource_campaign/access_rules/test_r13_sm.py)覆盖同 SM 窗口并集、重试去重、UUID 边界和跨档差分。另只回放旧三布局 K=4096 的 60 个 plain/stamped 进程，245760 个已保存数值通过原参考检查，21 个 `(case,j,T)` 组从 CSV 独立复算一致；30 个 ends 样本明确未分析。

[旧数据的分组复核](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/A-20261009-sm-legacy-v2/summary.json)中，同为 T=4 的 A−aligned 在 j=1/2/3 分别为 **53.248/26.523/0.014 cycle/Ktile**，B 为 **50.616/32.220/0.002**。旧三条件峰值 CTA 窗口重叠都是 132 个 SM，时间加权平均分别为 **105.22/105.61/106.28**；峰值、平均与请求数并非同一个量。旧样本缺少 setup GPU UUID，分析保留 `unknown`，没有从其他记录补填。

[本批 cases](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/A-20261009-sm-preparation-v2/cases.json)及同目录 `run.sh`、`source/`、`build/` 已在本地准备，约 32.1 MB，源码/二进制清单和 shell 语法检查通过。这个目录尚无新 GPU 样本。

以下准备只在本地执行；RUN 必须新建。复制完整运行包到获配计算节点的本地 `/tmp` 后，由管理对话选择已分配的 GPU UUID 并运行包内 `run.sh`，结束后回传整个包到本地主数据目录。不向超配额的 home/scratch/项目共享目录写入结果。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r13_sm.py cpu-check
python3 microbench/gh200_resource_campaign/access_rules/run_r13_sm.py prepare \
  --harness-run /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261009-R09-R13-shared-smoke-job738097 \
  --output RUN
```

节点上的入口如下；`V08_GPU` 沿用共享身份函数的变量名，只能填写本作业已分配的 UUID。目标目录 `reanalysis/sm-1procs` 或 `sm-10procs` 不覆盖已有分析。

```bash
cd /tmp/本批新运行目录
export V08_GPU=本作业已分配的完整GPU_UUID
export CUDA_VISIBLE_DEVICES="$V08_GPU"
R13_PROCS=1 bash run.sh
R13_PROCS=10 bash run.sh
```

以上为准备阶段记录；同日实测结果如下。

### 实测：行距罚项取决于 grid 规模和输出位置

运行于 **job 738101、romeo-a058、GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114**，独占节点、单 GPU 计时流。12 条件的 240 个进程全部成功，983040 个已保存输出值通过独立回放。最大进程 CV 为 1.328%，最大完整时间打点变化为 2.343%；这不证明各主循环区间完全无扰动。归档见 [本批原始结果](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-sm-job738101/cases.json)，管理者的[独立重放](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-sm-job738101/reanalysis/manager-formal-replay/summary.json)与原汇总一致，另从 pairs.csv 重新分组核对了全部 112 个行距差分组。

每个 stamped 进程的不同 SMID 数、CTA 窗口峰值重叠都等于本档请求值。时间加权平均与周期/纳秒比则随规模改变；下表范围覆盖 aligned/A+16 B/B+16 B 三布局，先在每个进程统计，再跨十个进程取中位：

| 请求 SM / 实际峰值 | CTA 窗口平均重叠 | CTA 包络有效 cycle/ns |
|---:|---:|---:|
| 32 | 30.64–30.69 | 1.8038–1.8055 |
| 64 | 58.98–60.95 | 1.7193–1.7262 |
| 96 | 81.74–84.95 | 1.6717–1.6802 |
| 132 | 104.43–105.13 | 1.6647–1.6699 |

这仍是 CTA 生命周期重叠；当前记录不能直接给出某个 Ktile 的并发 TMA 数。包络比值只表示该窗口的有效 cycle/ns，不据此归因功率或时钟控制机制。

固定为**第二个输出 tile（j=1）**，按 CTA 的总 tile 数 T 分组后，每 Ktile 的周期差如下。每格按该行两个 T 的顺序排列，不把两个组混为一个稳态样本：

| 请求 SM | 两个 T | A+16 B − aligned，cycle/Ktile | B+16 B − aligned，cycle/Ktile |
|---:|---|---:|---:|
| 32 | 13 / 14 | −1.93 / −1.95 | −0.03 / −0.17 |
| 64 | 6 / 7 | +12.48 / +15.04 | +9.10 / +12.31 |
| 96 | 4 / 5 | +1.48 / +13.29 | +4.04 / +16.12 |
| 132 | 3 / 4 | +64.22 / +70.13 | +49.20 / +52.30 |

同为 132 SM、T=4，A 的 j=1/2/3 罚项为 **70.13/17.43/0.009 cycle/Ktile**，B 为 **52.30/24.14/0.019**。行距和 K 都不变，位置效应依然显著；不能将不同输出位置的主循环平均后解释为一个恒定供给速率。

在 96/132 两档的严格子配对中，T=4、j=1 或 j=2 各有 24 个相同当前坐标。下表是 `(variant−aligned)_132−(variant−aligned)_96`；区间是十个进程差分的最小–最大值，不是置信区间：

| 行距 / 输出序号 | 差分中位，cycle/Ktile | 十次范围 |
|---|---:|---:|
| A+16 B，j=1 | +73.85 | +48.30～+80.19 |
| A+16 B，j=2 | +18.91 | +13.46～+22.02 |
| B+16 B，j=1 | +48.19 | +40.75～+62.02 |
| B+16 B，j=2 | +23.99 | +16.28～+30.43 |

**本批否定的是仅依赖 cfg、K 和行距余数的固定罚项。** 单靠总时间的频率换算误差也不能解释这些直接测得的周期差。服务规则需要包含执行规模、阶段或此前访问状态；共享资源竞争是候选解释，但本批同时改变工作分配、缓存历史和有效 cycle/ns，不能唯一定位到 HBM、L2 或 SM 本地路径。新卡的数值不直接替代旧 V08 卡的常数，也没有据此拟合通用的每 SM 供给上限。

下一步先让候选供给关系同时解释本批的位置分组和原 V08F 结果；只有候选仍因缺少逐 tile 全局时间而无法区分时，才补上文的少量直接端点。此时不展开其他行距、stage 或整个机制矩阵。

### 固定形状上的共享服务候选

在上述同卡数据上，比较三个最小形式：

\[
\widehat L/K_t=b+\max\{512,\ 24n(q_0+q_A a+q_B b_B)\}.
\]

24 KiB 是 cfg_a 的逻辑多播源需求；a、b_B 分别表示本批 A/B+16 B 条件。三个形式分别取 n=1（每 SM 固定上限）、n=持久 grid 大小、n=软件工作列表中仍有第 j 个输出 tile 的 CTA 数。最后一种称软件波次，只用形状和调度计算，不使用实测时间、频率或重叠。系数均非负；每个 case 总拟合权重相同，其后续 `(j,T)` 分组平分权重。

训练为 j>0 的 144 组，首 tile 的 24 组只做转移诊断。目标仍是每进程组内平均 L/Kt 的跨进程中位，不是完整 GEMM 时间。结果在[共享上限候选](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-sm-job738101/reanalysis/shared-cap-candidates-v3/summary.json)：

| 形式 | 后续窗口拟合：绝对误差中位 / 最大 | 首 tile 转移：中位 / 最大 |
|---|---:|---:|
| 每 SM 固定上限 | 1.92% / 14.70% | 1.57% / 16.58% |
| 整个 grid 共用上限 | 0.46% / 9.76% | 0.96% / 11.75% |
| 软件波次共用上限 | **0.42% / 7.55%** | **0.54% / 9.60%** |

软件波次形式拟得 b=7.7543、q0=0.166511、qA=0.012058、qB=0.011129；这里 b 是固定 K=4096 下的每 Ktile 窗口余项，不能据此外推 K，也不把 q 的倒数命名为物理带宽。整档留出 32/64/96 SM 时，最大误差分别为 **0.91%/2.96%/3.29%**。移除 132 SM 后，剩余 aligned 数据全落在计算分支，四参数的解析活跃分支 Jacobian 秩降为 3，基准服务项仅受界约束；所选拟合向 132 外推的最大误差为 **32.37%**。这轮没有证明仅靠低压力校准就能预测饱和端。

该形式也接近自身的信息上限：132 SM、A+16 B、T=4 的 j=1/2 都有 n=132，实测却为 620.33/539.23 cycle/Ktile。任何只依赖 n、行距和 T 的单值函数，对这两个窗口的最大相对误差至少为 `(620.33−539.23)/(620.33+539.23)=6.99%`。所以共享上限可以作为下一候选的组成部分，但若要更准，仍需表达输出位置对应的实际执行进度或访问历史；不能用一个更复杂的 n 函数消掉这个信息缺口。

复核入口沿用 [analyze_r13_supply.py](../../../../../../microbench/gh200_resource_campaign/access_rules/analyze_r13_supply.py)，增加 `--sm-summary` 读取已完成的 SM 扫描汇总；原 V08F 分析入口保留：

```bash
python3 microbench/gh200_resource_campaign/access_rules/analyze_r13_supply.py \
  --run results/gh200_resource_campaign/access_rules/20261009-R13-sm-job738101 \
  --sm-summary results/gh200_resource_campaign/access_rules/20261009-R13-sm-job738101/reanalysis/manager-formal-replay/summary.json \
  --output <新的分析目录>
```


为区分上面的同 n、不同 j 窗口，又离线加入一个**有限预填候选**：令
`n_eff = n_j − (s/64)·(n_j−n_{j+1})`，仍代入同一个四参数 max 式。它假设当前输出开始前已有 s 个 Ktile 的源数据，当前窗口还可能为下一输出补充数据；s=6 取配置的 stage 数，s=5 只作固定敏感性对照，不额外拟合 s。这只是从软件流水出发的近似，没有测得实际可用 stage 数。

[预填候选重放](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-sm-job738101/reanalysis/prefill-cap-candidates-v1/summary.json)中，s=5/6 的后续窗口最大拟合误差为 **4.41%/3.78%**，首 tile 转移最大误差为 **6.52%/5.90%**；s=6 的整档 32/64/96 SM 留出最大误差为 **0.88%/2.96%/3.31%**。移除 132 SM 时仍秩亏，向该档外推最大误差仍有 **24.02%**。这个结果支持在候选关系中表达“下一输出是否仍有需求”，尚不证明具体预填量、其他 K 的可迁移性或完整时间预测已经改善。原软件波次的信息下界仍成立；新候选改变的是输入信息，而非给原形式增加拟合阶数。

将同样的可预测波次权重放进原 V08F 的 51 条可比记录，并在原参考卡上重新拟合六参数供给候选，优势没有一致保留：原形式/波次/六 stage 预填的窗口 RMS 为 **2.11%/2.16%/2.14%**，最大误差为 **5.71%/5.80%/6.69%**。整档留出 K=1024 时，预填候选将最大误差从 11.16% 降至 7.81%；留出 K=4096 时则从 9.05% 升至 9.51%。详见[原卡转移诊断与重放脚本](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/manager-wave-transfer-v1/result.txt)。这里沿用旧校准窗口余项，参数也在旧数据上重拟合，仍是开发诊断；当前不把固定 K 的改善直接写进完整预测模型。

<a id="composed-coverage"></a>

### 两种地址覆盖与有限流水的组合诊断（2026-10-09，A）

**结论：两个覆盖量能解释 B 余数曲线的一部分结构，但还不能交付跨配置、跨 K 的通用供给函数。** 原 p0/p16/p32 校准只给出两个独立服务等式；32 B、128 B 合用后有三个服务参数，p64 预测原本就是一个区间。揭晓 p64 后可以选定这部分系数，但不能据此改变 [R10 原冻结失败](R10-layout-cache.md#局部冻结负结果job738203)。把这两个覆盖量与软件波次、固定预填信用组合后，cfg_a 的 SM 扫描仍能在原范围内拟合；B 余数曲线及 h03 的位置分组却保留大残差。共用 A/B 覆盖单价还有不依赖优化器的反例。

本节全部为**测后开发诊断**。每张卡分别定值；同一 099 参考卡的 V08F stamped 和 h03 dual 也分别拟合，不因同卡就混合观察协议。没有新测量、跨卡系数混合或冻结改判。[完整结果及参数范围](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-composed-coverage-v5/summary.json)、[逐窗口残差](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-composed-coverage-v5/residuals.csv)；入口仍为本页 `analyze_r13_supply.py`。

**地址输入。** 对 A、B 分别逐行计算合法地址跨度覆盖的 32 B 和 128 B 块数，减去同长度对齐跨度所需的最少块数。额外块数乘对应块宽，得到 `A32/A128/B32/B128`，单位为覆盖 KiB；另保留逻辑源 payload `D`。按 cluster 多播摊分，每个 Ktile 求均值。M/N/K 尾部只数实际有效地址，计算项仍按实际执行的完整 MMA tile；不把零填充字节算作源请求。**当前 Q 没有 OOB 填零服务项，因此有效地址需求减少不意味着总服务成本减少。** 这里的覆盖不是 TMA 内部事务、物理 L2 流量或 HBM 流量。

对完整 B-box，p0/p16/p32/p64/p128 的额外覆盖行数仍为：32 B 宽下 **0/32/0/0/0**，128 B 宽下 **0/56/48/32/0**。因此 p16 同时激活两种覆盖，p32/p64 只激活 128 B 覆盖。这两个输入在完整五点上独立，在原三个校准点的活跃服务分支上却不足以确定三个参数。

固定压力的地址服务片段可写为 `max(C,q0+q32*x32+q128*x128)`。保持原 ns/Ktile 目标和 `C=329.9967`，原校准的全部零误差非负解满足：

\[
q_0\in[66.6133,329.9967],\quad
q_{128}=(385.9533-q_0)/48,\quad
q_{32}=(439.1767-q_0-56q_{128})/32.
\]

由此得到 **p64 预测范围 [329.9967,367.3011] ns/Ktile**，不是一个已定常数。实测 365.7233 位于其中。利用已揭晓 p64 可解出 `q0=325.2633, q32=1.347135, q128=1.264375`；这只是在已有可行集中选择一个点。p128 仍预测 329.9967，相对实测误差 −0.5415%。这些 q 的单位是 ns/额外覆盖行或 ns/Ktile，不能与下文周期/KiB 系数混用，也不把聚合窗口的解释冒充逐位置预测。

**仅比较以下两个组合形式。** 对软件工作列表中仍执行第 j 个输出 tile 的 CTA，求逻辑源请求向量之和 `Q_j=(D,A32,A128,B32,B128)`。令 `s=min(StageCount,Kt)`，取固定预填近似

\[
Q^{eff}_j=(1-s/K_t)Q_j+(s/K_t)Q_{j+1},\qquad
\widehat L_j=b_{cfg}+K_t\max(C_{cfg},G(Q^{eff}_j)),
\]

其中 `C=(512,512,1024)` cycle/Ktile，`b_cfg≥0` 为完整窗口的周期余项；s 不拟合，也不声称已测出实际预填数量。

- **共用单价**：`G=q0*D+q32*(A32+B32)+q128*(A128+B128)`。
- **分开 A/B 单价**：`G=q0*D+qA32*A32+qA128*A128+qB32*B32+qB128*B128`。

全部 q 非负，单位为 cycle/该向量中的 KiB。计算分支只要求候选服务项不超过计算时间，即有效能力满足需求下界；没有使用 `D/实测周期` 作为带宽标签。V08F 的确定性预测先对各位置做 max，再按实际窗口数加权，不把“平均请求的 max”替代“各窗口 max 的平均”。拟合只用后续输出窗口，每个 case 总权重相同。原始主循环含等待、循环与排空，b 不是已独立测定的纯开销。

| 卡 / 观察数据 | 后续目标数 | 共用单价最大误差 | 分开单价最大误差 | 活跃分支秩 / 参数数：共用；分开 |
|---|---:|---:|---:|---|
| 43269fbc，B 余数曲线 dual | 25 个 j/T 组 | 16.33% | 16.33% | 4/4；4/6 |
| ef8692f3，cfg_a SM 扫描 stamped | 144 个 j/T 组 | 4.80% | 3.78% | 3/4；4/6 |
| 099dda56，h03 四角 dual | 20 个 j/T 组 | 30.63% | 19.22% | 4/4；5/6 |
| 099dda56，V08F stamped | 51 个 case 聚合窗口 | 18.14% | 8.66% | 5/6；6/8 |

参数数目包含窗口余项；V08F 为三个配置各一项，其余各一项。秩由解析分支 Jacobian 按列归一后计算，不用优化器数值噪声补秩。结果另外保存保持当前拟合预测和分支的全部非负参数范围；非唯一参数在 `parameters` 中为 null，优化器任取的一组数只放在 `optimizer_representative`。这些范围不是噪声置信区间，也不排除其他等价活跃区域给出更宽范围。首 tile 没有参与拟合；分开单价形式的首 tile 最大误差分别为 13.66%、5.90%、27.42%、27.21%，不把后续窗口关系直接套到首段。

**失败并非都能靠重调系数消除。** 对非负单价模型，输入向量逐项更小的窗口不应更慢。099 卡 h03 的 j=1/T=3 中，B-only 的额外覆盖少于 A-only，实测却为 **833.682 > 700.956 cycle/Ktile**。共用单价形式对此必有至少 `(833.682−700.956)/(833.682+700.956)=8.65%` 的最大相对误差。原 V08F cfg_b、K=1024 的 B/A 单轴对应下界为 **9.33%**。这要求区分请求来源或加入其他已定义状态，不能把“覆盖 KiB”直接乘一个通用单价。

分开单价后，**有效地址与软件波次特征仍不充分**：099 卡 h03 的 aligned、j=3/T=4 有更少的有效波次需求，实测 **623.499 cycle/Ktile**，却比 j=2/T=3 的 **540.905** 更慢；两种模型对这对窗口至少有 **7.09%** 最大相对误差。B 余数曲线 p16 的 j=2/T=4 实测 782.898，分开形式仅预测 655.034。前一项是针对当前特征的参数无关失败下界，后一项是本次拟合残差；不能据此唯一断言缺的是流水状态，也不能把有效地址更少直接等同于硬件服务需求更少。

同日依据管理者补核，再从[原 dual tiles.csv](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-joint-pitch-job738169/reanalysis/B-20261009-matched-joint-final/tiles.csv)按实际 M/N 坐标复算：j=3/T=4 每进程的 60 个窗口中，**19 个含 B 的 N 向部分越界**，其中 1 个还含 M 尾部。按每进程组内均值、再跨十进程取中位，得到：

| j=3/T=4 的几何类别 | 每进程窗口数 | 主循环 cycle/Ktile |
|---|---:|---:|
| M/N 均完整 | 38 | 587.145 |
| 仅 M 尾部 | 3 | 599.521 |
| 仅 N 尾部，即 B 部分越界 | 18 | 702.625 |
| M/N 均有尾部 | 1 | 706.375 |

j=2/T=3 没有 B 部分越界，完整与仅 M 尾部窗口分别约 540.969/541.354 cycle/Ktile。各组中位数不能直接加权重构总体中位数。上述分组显示，原次序矛盾混入了 B 部分越界；分开后仍有位置差异，但先前访问与并发状态也未固定。

R18 的 [A source-map 配对](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R18-input-map-job738296/reanalysis/input-map-pairs-v1/input-map-pairs.json)已经显示，OOB 映射路径与从有效地址读入显式零值具有不同代价，不能把未读源字节视为零服务成本。该证据来自另一张卡的 A 向 whole-padding；它不直接确定 h03 的 B 部分越界罚项，也不证明 A/B、部分/整 tile 零路径具有相同成本。当前结论应保留 OOB 填零、流水等待、pingpong 输出重叠与请求历史等解释，先分开有效地址需求和 OOB 填零需求。本补注不重拟合、不扩大模拟器，原数值与冻结判定保持。

**可组合的最小条件关系及其边界。** ef 卡的完整 box、0/+16 B 余数恒有 `A128=7*A32`、`B128=7*B32`。本批只能识别 `βA=qA32+7qA128=0.1665748625` 和 `βB=qB32+7qB128=0.6049830055`，不能独立识别四个覆盖系数。其余可定项为 `b=491.856139 cycle/输出tile`、`q0=0.1721083258 cycle/逻辑源KiB`。在**已见的 cfg_a、2304×3072×4096、6 stages、四档 grid、后续输出 tile、A/B 单轴 0/+16 B**范围内，可以写成

\[
L_j=491.856139+64\max\{512,
n^{eff}_j[24(0.1721083258)+2\beta_A I_A+0.5\beta_B I_B]\},
\quad n^{eff}_j=n_j-6(n_j-n_{j+1})/64.
\]

这只是分开单价形式在可识别输入子空间上的化简，不是新增查表或第三个拟合形式。若用于 CTA 递推试算，接口为形状、实际软件工作列表、j、stage、stride → L_j，然后 `me_j=fm_j+L_j`；只替代原 L_j，不能再叠加旧 `l0+l1*Kt`。原首段供给 S 与输出 E 继续单列。最大拟合误差仍为 3.78%，首 tile 不在该关系的已拟合范围，也未做新的完整 GEMM 验证。

同样拟合等价的参数，对该卡 **132 档 j=1 的 A+32 B**给出 **[552.924,590.618]**、B+32 B 给出 **[552.924,587.149] cycle/Ktile**。所以已见数据上的小误差没有消除新余数预测的不唯一。432 卡没有 A 行距变化，其 A 系数范围仍为 `[0,+∞)`；不能借原参考卡或 ef 卡的系数补上。099 卡 h03 的 A+32 B 区间仍有 **[597.173,748.932]**，且模型本身已有上述失败，不据这个区间推荐预测常数。

**真正尚缺的同卡约束。** 不再增加本轮矩阵，缺口分别交给后续选择：

1. 若最终使用 ef/099 卡的完整 box 覆盖分解，已有 0/+16 B 只能确定每个操作数的一条组合线。每个操作数至少需要一个**独立覆盖比例且确实暴露供给分支**的同卡约束，例如 +32 B 的已知位置对照；A、B 各一条即可解除这两条代数退化，计算平台上的新点仍只给界。432 卡的 B 轴已有独立比例，缺的是 A 轴两个独立覆盖比例，而不是再次扩大 B 曲线。这里只列识别所需约束，没有提交补测。
2. s=StageCount 是假设。当前完成窗口没有记录 FIRST_MMA 时已有多少 Ktile 真正可消费，也未观察跨输出的 TMA 源请求/槽释放进度；单靠更多余数不能把预填信用、窗口余项和服务系数分开。若要定 s，最小缺口是同一代表调用中的预填完成与当前/下一输出补发边界，而非一整张 stage 网格。
3. h03 的单调性反例证明**当前特征不足**，不能唯一归为缺流水状态。先按现有坐标分开有效地址与 OOB 填零需求，区分 A/B、部分/整 tile 越界；再复用 dual 的 MMA/epilogue 时间线判断哪些阶段状态仍缺约束。不要为消除残差直接扩大模拟器或增加覆盖系数。当前不把请求放大、零路径成本、L2/HBM 带宽或内部队列深度分别定值。

本次只补分析器和本页。B 余数曲线为取得完整位置数据只重读 50 个 dual 记录，其余复用已有摘要/CSV；原冻结文件 SHA256 保持 `6fd226ef90f1a9d9224d1d6db769744eaece3befd69e98abf4352c4676d072c5`，失败判定不变。

```bash
python3 microbench/gh200_resource_campaign/access_rules/analyze_r13_supply.py \
  --coverage-suite \
  --run /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203 \
  --output <该run下新的reanalysis目录>
```

<a id="valid-fill-candidate"></a>

### 有效地址与 OOB 填零的同卡周期/ns 候选（2026-10-09，A）

**开发版可先用周期形式替代后续输出 tile 的 L 项；直接 ns 形式保留为频率假设下的服务候选。** 分开两类需求后，20 个整块 OOB 分组的中位绝对误差为周期 5.54%、直接 ns 4.73%。整个 137 个后续分组仍有约 24%～25% 的最大误差，集中在本身没有 OOB 的 j=2/T=5 慢窗口；因此这次交付包含可计算的主要填零代价，也明确保留未解释的分项。没有把物理参数不唯一视为所有预测都不可用。

数据只取 **a057、GPU-43269fbc-449d-3e0f-908a-9c81229546d3** 的 R10 B 曲线 job738203，以及 R18 M/N source-map 配对 job738296/738307。共 13 条件，cfg_b、6 stages、K=1024/4096；R10 的三个旧校准点和两个已揭晓留出都只作测后开发数据。R18 原报告的双时钟分组直接复用，R10 为补齐位置下的直接 ns 再读 50 份 dual 记录。原评分和冻结保持只读。

[本次完整报告](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-valid-fill-v4/summary.json)保存需求向量、周期/ns 参数、分项成绩、留 K 组诊断、频率敏感性与输入哈希；[逐分组残差](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/A-20261009-valid-fill-v4/residuals.csv)保留全部慢窗口。

**需求由 descriptor 边界决定。** 对当前 CTA 输出坐标，分别求 A/B 有效源跨度及所需的目标 SMEM tile：

- `V` 为有效地址 payload，包含从扩展 map 中读取的显式零值；`B32/B128` 为 B 的额外地址覆盖。此批 A 行距始终对齐，没有拟合 A 行距系数。
- `ZA/ZB` 为对应 nominal tile 中需由 OOB 路径提供的填零量，单位 KiB。此批每个操作数只观察到 0 或整块 16 KiB，尚不证明部分越界代价与填零字节成正比。
- 各输出位置的有效地址需求按实际软件工作列表求和；沿用固定预填近似 `Veff_j=(1−s/Kt)V_j+(s/Kt)V_{j+1}`，B 覆盖量同样处理，`s=min(StageCount,Kt)`。这是软件需求近似，不是实际 TMA 并发。

扩大 source map 后，原 logical pad 分类保持，但 `ZA/ZB` 降为零、有效源需求增加。这样保留了“读入零值”与“OOB 填零”的可计算区别，而不是按输出是否在逻辑边界外给固定罚项。

只比较同一服务式的两种时间基准，全部系数非负：

\[
S_j=q_VV^{eff}_j+q_{32}B32^{eff}_j+q_{128}B128^{eff}_j
       +q_AZA_{c,j}+q_BZB_{c,j}.
\]

周期形式为 `L_cycle=b_cycle+Kt*max(512,S_cycle)`；直接 ns 形式为 `L_ns=b_ns+Kt*max(512/f_assumed,S_ns)`。主表显式假设 **f_assumed=1.6 GHz**，另列 1.8 GHz 敏感性；没有从各窗口的实测 cycle/ns 比值回填预测输入。`b` 为完整输出 tile 的窗口余项，q 为条件服务系数，不命名为物理带宽或内部填零速率。额外的“当前填零且还有下一输出”单价在两形式中均落到零，已从交付式删除。

| 目标组 | 周期：中位 / 最大 / RMS | 直接 ns：中位 / 最大 / RMS |
|---|---|---|
| 全部 137 个后续分组 | 2.18% / 24.22% / 6.03% | 4.30% / 25.41% / 7.13% |
| 20 个当前整块 OOB 分组 | 5.54% / 14.10% / 7.18% | 4.73% / 13.21% / 6.01% |
| 25 个 R10 B 余数分组 | 4.23% / 16.38% / 7.78% | 3.02% / 16.59% / 7.73% |
| 117 个当前有效地址分组 | 1.68% / 24.22% / 5.80% | 4.21% / 25.41% / 7.30% |

每个 case 总拟合权重相同；其 j/T/几何分组等分权重。各组目标仍为进程内均值再跨十进程取中位。两形式解析活跃分支 Jacobian 均为 **6/6**，只是本条件式在这些输入上的秩，不代表内部物理机制已唯一识别。26 个首输出分组不参与拟合，周期/ns 转移最大误差为 13.70%/9.79%。

参数顺序为 `(b; qV,q32,q128,qA,qB)`：

```text
cycle: (976.498744; 0.095667085, 1.278291466, 0.263462552, 34.393673, 35.663945)
ns:    (180.878928; 0.079845306, 0.821144340, 0.067725970, 17.208983, 17.940819)
```

b 的单位分别为 cycle/输出 tile、ns/输出 tile；其余单位为 cycle/KiB 或 ns/KiB，且 V/B 覆盖为全软件波次有效需求，ZA/ZB 为当前 CTA 需求。该作用范围不能互换。A/B 整块填零系数接近，但本批不足以将它们推广为跨形状的同一硬件常数。

**直接 ns 较稳定的证据及限制。** 对同一 dual 进程中的 OOB、T=6 分组，逐 trial 求 j=4 与 j=1 的比值，再取中位：

| 对照 | 周期变化 | 直接 ns 变化 |
|---|---:|---:|
| padM，K=1024 | +15.38% | +3.45% |
| padM，K=4096 | −8.81% | +1.31% |
| padN，K=1024 | +16.60% | +6.59% |
| padN，K=4096 | −11.52% | −2.78% |

例如 padM、K=4096 的分组有效 cycle/ns 从约 1.750 变为 1.567，而直接窗口 ns 仅小幅变化。这里的比值是同次调用的分组周期均值/直接 ns 均值，用于诊断，不是独立时钟传感器，也不进入预测特征。采用固定 1.8 GHz 重拟合 ns 形式后，整体 RMS/最大误差变为 8.90%/31.91%；因此“部分 OOB 窗口在 ns 下更稳定”不等于已完成全流程频率预测。

**已有组的留 K 诊断。** 移除 K=4096、只用 K=1024 拟合时，两形式仍为 6/6；在 56 个 K=4096 分组上，周期中位/最大为 **0.75%/24.25%**，ns 为 **5.72%/25.37%**。移除 K=1024 后，训练只有对齐地址，B32/B128 两列均为零，秩降为 4/6；测试只评分对应两列也为零的 66 组，p16/p32/p64 共 15 组明确缺参数、不评分。其周期中位/最大为 **6.49%/19.69%**，ns 为 **19.47%/35.42%**。这些是已揭晓数据上的开发诊断，不是新留出。

**可用于组合试算的接口。** `fill_geometry(row, setup)` 根据 descriptor、stride、stage 和实际软件工作列表形成需求；`fill_window_cycles(features, Kt, parameters, unit, frequency_ghz)` 返回 CTA 的 L 周期。周期形式直接返回 `b_cycle+Kt*max(512,S_cycle)`；ns 形式要求调用者明确传入预测频率 f，返回

\[
L_{cycle}(f)=f b_{ns}+K_t\max(512,fS_{ns}).
\]

因此管理者可以把它接入开发版的 `me_j=fm_j+L_j`，保留首段 S、输出 E 和其他递推边界，只替换 L 项。校准适用范围为本卡、cfg_b、六 stage、已见两种 K、记录中的全 tile/whole-padding 配对与 B 行距；部分 OOB、同时 A/B OOB、A 未对齐行距以及其他卡仍需单列。物理代价不唯一不妨碍本范围内试算，但试算结果必须保留以下失败。

最大剩余误差出现在 **padN OOB run、K=4096、j=2/T=5 的 logical-in 窗口**：当前 `ZA=ZB=0`，实测为 **695.743 cycle/Ktile、432.792 ns/Ktile**，候选分别为 **527.258、322.826**，低估 24.22%/25.41%。padM 对应窗口也偏慢。给当前 tile 的填零量加成本无法解释这种有效地址窗口；其同批其他 j/T 窗口并不都慢。前序请求、局部/共享等待和输出重叠尚未分开，当前不追加位置查表、通用事件模拟器或物理路径归因。

复核命令如下，输出目录必须新建；原冻结 SHA256 继续保持 `6fd226ef90f1a9d9224d1d6db769744eaece3befd69e98abf4352c4676d072c5`。

```bash
python3 microbench/gh200_resource_campaign/access_rules/analyze_r13_supply.py \
  --fill-suite --assumed-ghz 1.6 \
  --run /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203 \
  --output <该run下新的reanalysis目录>
```
