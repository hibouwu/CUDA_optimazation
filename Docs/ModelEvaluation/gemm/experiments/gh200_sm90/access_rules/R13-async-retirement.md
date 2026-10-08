# R13：供给与退役

实现与本文作者：B；GPU采样由主对话串行调度；未实施R13的C负责独立源码、SASS及原始工件复核，各冻结报告见下方证据链接。

## 结论

- 原问题是单 SM 供给上限是 55 还是 64 B/cycle。本组在真实 WGMMA 消费、4 槽、每 SM 一个 CTA、共享小源条件下测得：32 KiB tile、4 stage 为 58 B/cycle，48 KiB tile 为 45 B/cycle。这是条件化的序列速率，不是通用上限。
- 2026-10-07 依据 V05 判断这个问题对预测已不关键：V05 中 cfg_a/b 的主循环误差中位数 0.9%，主循环按计算下界即可。这个判断只适用于 V05 的条件：6 个尺寸的 A/B 行距都是 128 B 倍数，没有 swizzle 补齐，K 最长 12288。
- V08 超出了这个范围，上一条不再代表当前结论。A/B 行距未按 128 B 对齐时，cfg_b 只交付 46–57 B/cycle/SM，需求为 64；大足迹长 K 时为 57–64。cfg_b_h03 的总时间预测偏短 35%。每 SM 有效供给现为 [PLAN 主线](PLAN.md#mainline)的首要扩展，本页维护其证据：[立即释放消费者的整卡供给（原 R05-D）](#r05-d)、[V08 后续对照的供给解释](#v08-supply)。

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

**按每 SM 的操作数供给解释。** 每个 Ktile 的计算为 512 cycle（cfg_c 为 1024）。按 cluster 多播只读一次计算，每 SM 每 Ktile 从 L2 取的字节：cfg_a 24 KB（A 16 KB + B 16 KB/2），cfg_b 32 KB，cfg_c 32 KB/1024 cycle。对应需求 48、64、32 B/cycle/SM。实际交付的“对齐字节”：

| 条件 | cfg_a | cfg_b | cfg_c |
|---|---|---|---|
| 对齐、L2 内 | 48（计算限） | 64（计算限） | 32 |
| 大足迹（105–260 MiB）长 K | 47.0–47.9 | 57.3–63.7 | 31.8–32.0 |
| 一个操作数行距未对齐 | 45.2–47.6 | 46.1–56.7 | 32.0 |

cfg_c 需求最低，从不受限；cfg_b 需求最高，最先受限。推断：主循环受“计算”与“每 SM 操作数供给”两者中较慢者限制；未对齐行距让同样的有效字节需要更多传输，大足迹让 L2 命中下降，都降低可交付的供给。本组数据还不能把“未对齐的放大倍数”与“供给上限”分开定值（无 NCU 计数）。

[结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/followup-v1/followup.json)。
