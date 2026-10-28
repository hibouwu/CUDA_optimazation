# GH200 资源微基准：结果与结论

设备：ROMEO GH200（CC 9.0，132 SM，L2 60 MiB，CUDA 可见 95 GiB HBM3），CUDA 12.9，`sm_90a`。下文资源基准采样于 2026-10-01 至 10-05，suite `20261001-resource-suite-v2`，包含单 GPU 资源与两套固定 tile 组合。后续完整 GEMM 预测验证见下方 access_rules 入口。

<a id="research-questions"></a>

## 按研究问题阅读

当前进行：[a044 锁频完整重测与逐阶段说明](Q10-time-mapping/A044-REMEASUREMENT.md)。旧卡结果保持原样，新轮次单独比较。

实验代码按探针集中在四个阅读目录中（单 CTA TMA、驻留链、多 CTA tensor、插桩 GEMM），请求级组合模型另有一个，入口见 [microbench 的“按探针阅读代码”](../../../../../microbench/gh200_resource_campaign/access_rules/README.md#按探针阅读代码)。

问题 1–10 沿用[机制分解](../../model/mechanism/README.md#3-机制分解)的 ①–⑫，④⑥、⑦⑧ 各合为一组。当前规则、支持与排除、反例和下一步由路径页维护；实验可被多个问题引用，原记录只保留一份；原始轮次按“问题／合并实验”存放，[结果目录索引](../../../../../results/gh200_resource_campaign/access_rules/README.md)列出每组全部轮次，[非实验清单](access_rules/README.md#non-experiment-results)单列。验收类型按 [GOALS](../../GOALS.md) 区分 C2 机制主张、C3 完整时间、C1 配置选择。

| 问题 | 定价的事件 | 当前假设与规则入口 | 各轮实验 | 验收类型 | 状态 |
|---|---|---|---|---|---|
| <a id="question-01"></a>1. ① 调度与启动 | 工作列表、CTA→SM 放置、`entry → producer_first_work` | [程序图](../../model/mechanism/program/cutlass_program_graph.md)、[组合](../../model/mechanism/composition.md) | [R08](Q01-scheduling/R08-waves-l2-reuse.md)、[R18](Q01-scheduling/R18-cluster-boundary.md)（含 [R17](Q01-scheduling/R18-cluster-boundary.md#r17)）、[R19](Q01-scheduling/R19-critical-cta-tail.md)；[启动前缀](Q01-scheduling/startup-prefix.md)；[工作放置](Q01-scheduling/work-placement.md) | C2、C3 | 工作映射已核对，入口前缀仍用经验替代。 |
| <a id="question-02"></a>2. ② TMA 请求：发出、在途与目的投递 | 请求发出 → 目的 SMEM 投递 → full 事务完成 | [共享服务](../../model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md)、[候选与反例](../../model/mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md)、[单 CTA 控制](../../model/mechanism/hardware_rules/TMA-S16-STAGE-COST.md) | [EXP-12](Q02-tma-request/EXP-12-cp-async.md)、[EXP-14](Q02-tma-request/EXP-14-tma-1d.md)、[EXP-15](Q02-tma-request/EXP-15-tma-2d.md)、[EXP-16](Q02-tma-request/EXP-16-tma-pipeline.md)、[R05](Q02-tma-request/R05-async-lifecycle.md)、[R13](Q02-tma-request/R13-async-retirement.md)、[TRANSFER-01](Q02-tma-request/TRANSFER-01-replay.md)、[COMPUTE-01 D](Q14-composition/COMPUTE-01-resident-chain.md#d-measured)；联合服务引用 [R12](Q07-smem/R12-smem-path-contention.md)；[请求形状](Q02-tma-request/request-shape-response.md)、[预填与等待入口](Q02-tma-request/prefill-wait-entry.md)、[服务规则检验](Q02-tma-request/request-service-rules.md)、[准备与返回](Q02-tma-request/request-preparation-return.md)、[完成观测](Q02-tma-request/completion-observation.md)、[探针控制](Q02-tma-request/probe-control.md)、[原生D链](Q02-tma-request/native-d-chain.md)、[共享源固定形式](Q02-tma-request/shared-source-forms.md)、[源组织](Q02-tma-request/source-organization.md)、[GPC共享](Q02-tma-request/multi-cta-gpc-sharing.md)、[发出位置干扰](Q02-tma-request/issue-placement-interference.md)、[GEMM输入网络](Q02-tma-request/gemm-input-network.md)、[源复用](Q02-tma-request/gemm-source-reuse.md)、[S16控制](Q02-tma-request/S16-single-cta-control.md)、[首填](Q02-tma-request/first-fill.md) | C2 | 重复窗口：EXP-14/15/16 与 R05-A 的单 CTA 16 KiB 单请求为 766–1142 cycle，只有 EXP-15 是 tensor 路径，见[零负载服务点](../../hardware/gh200_sm90/L0.md#零负载服务点)。 |
| <a id="question-03"></a>3. ③ 源服务：L2 与 HBM | L2 命中或 HBM miss 后源就绪、输出写回 | [L2 服务](../../model/mechanism/hardware_rules/L2-SERVICE.md)、[HBM 服务](../../model/mechanism/hardware_rules/HBM-SERVICE.md)、[源状态候选](../../model/mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md) | [EXP-04](Q03-source-service/EXP-04-memory-baseline.md)、[EXP-13](Q03-source-service/EXP-13-global-rw.md)、[B01](Q03-source-service/B01-bandwidth-cache.md)、[R10](Q03-source-service/R10-layout-cache.md)、[L2-01](Q03-source-service/L2-01-saturation.md)、[HBM-01](Q03-source-service/HBM-01-replay.md)；[源状态](Q03-source-service/source-state-candidates.md)、[缓存组织](Q03-source-service/cache-organization.md)、[GEMM源工作集](Q03-source-service/gemm-source-workset.md)、[存储两域](Q03-source-service/storage-domain-numa.md) | C2 | 已有结构约束，cfg_b 的量化服务规则仍未通过。 |
| <a id="question-04"></a>4. ④ 输入同步：full 完成 → wait 返回；⑥ 槽释放与再取得 | full 完成 → consumer wait 返回；consumer release → 下一代 producer acquire | [同步服务](../../model/mechanism/hardware_rules/SYNC-SERVICE.md)、[输入事件模型](../../model/mechanism/hardware_rules/TMA-INPUT-EVENT-MODEL.md) | [EXP-11](Q04-sync-slot/EXP-11-sync.md)、[SYNC-01](Q04-sync-slot/SYNC-01-replay.md)、[SYNC-02](Q04-sync-slot/SYNC-01-replay.md#3-首复用稀疏补测)；[COMPUTE-01 B/C](Q14-composition/COMPUTE-01-resident-chain.md#bc-measured)、[D](Q14-composition/COMPUTE-01-resident-chain.md#d-measured)、[E](Q14-composition/COMPUTE-01-resident-chain.md#e-measured)；[活动槽干预](../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/20261015-g3-active-stages-development-001/README.md)、[计算量干预](../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/20261015-g3-compute-credit-discrimination-003/RESULTS.md)；[局部同步链](Q04-sync-slot/local-sync-chain.md)、[full等待方式](Q04-sync-slot/full-wait-policy.md)、[full就绪与返回](Q04-sync-slot/full-ready-return.md)、[G3 stage信用](Q04-sync-slot/g3-stage-credit.md)、[释放观察](Q04-sync-slot/release-feedback.md)、[barrier返回](Q04-sync-slot/barrier-return.md) | C2 | 重复观测：COMPUTE-01 B/C、D、E 与 SYNC-02 的短链循环内打点均未通过 5% 扰动门槛，未得到可迁移返回价格，该观测方法不再重复。 |
| <a id="question-05"></a>5. ⑤ WGMMA 计算 | fence、HGMMA、commit → compute complete | [计算服务](../../model/mechanism/hardware_rules/COMPUTE-SERVICE.md) | [EXP-07](Q05-wgmma/EXP-07-wgmma.md)、[R00](Q05-wgmma/R00-anchor-target.md)、[COMPUTE-01 A](Q14-composition/COMPUTE-01-resident-chain.md#a实测与长度留出)；联合服务引用 [R12](Q07-smem/R12-smem-path-contention.md) ；[计算完成与提交](Q05-wgmma/local-compute-chain.md) | C2 | A 的有限长度留出通过，完整短链价格仍待组合验证。 |
| <a id="question-06"></a>6. ⑦ 数学 WG 交接；⑧ 输出 | order barrier 交接、mma tail；STS → TMA store → 源复用与排空 | [程序图](../../model/mechanism/program/cutlass_program_graph.md)、[SMEM 服务](../../model/mechanism/hardware_rules/SMEM-SERVICE.md)、[组合](../../model/mechanism/composition.md) | [R14](Q06-handoff-output/R14-stage-handoff.md)、[R15](Q06-handoff-output/R15-output-service.md)、[COMPUTE-01 E](Q14-composition/COMPUTE-01-resident-chain.md#e-measured)；联合服务引用 [R12](Q07-smem/R12-smem-path-contention.md)；[输入／输出联合窗口](Q07-smem/input-output-joint.md) | C2、C3 | 程序顺序已核对，交接与输出仍无可迁移的独立价格。 |
| <a id="question-07"></a>7. ⑨ SMEM 多路共享 | TMA 写入、WGMMA 操作数引用、STS 写块与 TMA store 源读取 | [SMEM 服务](../../model/mechanism/hardware_rules/SMEM-SERVICE.md)、[联合服务](../../model/mechanism/hardware_rules/joint_service.md)、[m11](../../model/mechanism/hardware_rules/M11-SMEM.md) | [EXP-09](Q07-smem/EXP-09-smem.md)、[EXP-10](Q07-smem/EXP-10-ldmatrix-shfl.md)、[R03](Q07-smem/R03-access-demand.md)、[R11](Q07-smem/R11-mixed-issue.md)（含 [R04](Q07-smem/R11-mixed-issue.md#r04)）、[R12](Q07-smem/R12-smem-path-contention.md)（主归此项）、[SMEM-01](Q07-smem/SMEM-01-replay.md)、[输入／输出联合窗口](Q07-smem/input-output-joint.md) | C2 | 需求已核对，m11 两次冻结留出均未通过。 |
| <a id="question-08"></a>8. ⑩ 寄存器配额与驻留 | producer 释放、consumer 申请配额、CTA 驻留准入 | [寄存器服务](../../model/mechanism/hardware_rules/register_service.md)、[驻留与配额](../../model/mechanism/hardware_rules/residency.md) | [R01](Q08-register-residency/R01-readiness.md)、[R02](Q08-register-residency/R02-register-service.md)、[R06](Q08-register-residency/R06-issue-residency.md)、[R16](Q08-register-residency/R16-residency-quota.md)、[EXP-18](Q08-register-residency/EXP-18-occupancy-aux.md)、[REG-01](Q08-register-residency/REG-01-replay.md)、[SM-01](Q08-register-residency/SM-01-replay.md) | C2 | 实际配额与容量已核对，申请时长仍待定价。 |
| <a id="question-09"></a>9. ⑪ 有效频率 | 各服务的 cycle → ns | [频率主线](../../model/mechanism/hardware_rules/FREQUENCY-MECHANISM.md) | [R09](Q09-frequency/R09-inkernel-clock-stages.md)（含 [R07](Q09-frequency/R09-inkernel-clock-stages.md#r07)）、[V03](Q15-full-time/V03-clock-rule.md)；[频率机制](Q09-frequency/frequency-mechanism.md)、[运行前频率预测](Q09-frequency/frequency-prediction.md)、[plain前序模块](Q09-frequency/R09-plain-prefix.md)、[cfg_c双时钟](Q09-frequency/cfg-c-clock-bracket.md) | C2、C3 | 有限范围的完整时间留出通过，阶段状态规则仍未通过。 |
| <a id="question-10"></a>10. ⑫ 包络到 plain 时间 | dual 包络 → plain 完整时间 | [时间映射](../../model/semi_empirical/MODEL-CANDIDATES.md#v09-model)、[测量协议](access_rules/RESEARCH-GUIDE.md) | [INTERFACE 历史合同](HISTORICAL-PATHS.md)、[V09](Q15-full-time/V09-component-validation.md)、[C1-01 计时对照](Q16-config-selection/C1-01-selection.md#恢复前要定的事)；[相同plain重复](Q10-time-mapping/plain-repeat.md)、[C1计时协议](Q10-time-mapping/c1-timing-protocol.md)、[cfg_c producer-tail](Q10-time-mapping/cfg-c-producer-tail.md)；[a044重测与参照](Q10-time-mapping/A044-REMEASUREMENT.md) | C3、测量资格 | 当前沿用经验映射，作业间差异保留在原记录。 |
| <a id="question-11"></a>11. 其他计算路径 | FMA、mma.sync、FP8/INT8 的完成与持续服务 | [微架构证据审阅](../../model/mechanism/hardware_rules/MICROARCH-REVIEW.md) | [EXP-01](EXP-01-compute-audit.md)、[EXP-03](EXP-03-compute-baseline.md)、[EXP-05](Q11-other-compute/EXP-05-fma.md)、[EXP-06](Q11-other-compute/EXP-06-mma.md)、[EXP-08](Q11-other-compute/EXP-08-fp8-int8.md) | C2 | 资源实测保留，当前作为扩展路径储备。 |
| <a id="question-12"></a>12. cluster 多播与 DSM | 多播投递、远端 SMEM 请求与完成 | [微架构证据审阅](../../model/mechanism/hardware_rules/MICROARCH-REVIEW.md) | [EXP-17](Q12-cluster-dsm/EXP-17-cluster.md) | C2 | 已有资源实测，cfg_b 的 cluster 为 1×1。 |
| <a id="question-13"></a>13. 普通 L1/LSU | 普通 global/local 请求 → cache 服务 → 寄存器返回 | [L1/LSU 服务](../../model/mechanism/hardware_rules/L1-LSU-SERVICE.md) | [L1-01](Q13-l1-lsu/L1-01-replay.md)、[EXP-13](Q03-source-service/EXP-13-global-rw.md)、[EXP-18](Q08-register-residency/EXP-18-occupancy-aux.md) | C2 | 开发实测与评分完成，原留出尚未采样。 |
| <a id="question-14"></a>14. 程序与服务的组合、交互及端点验证 | 搬运 → 就绪 → 计算 → 释放 → 输出、关键 CTA 与阶段端点 | [程序图](../../model/mechanism/program/cutlass_program_graph.md)、[组合](../../model/mechanism/composition.md) | [EXP-19](Q14-composition/EXP-19-fp32-tile.md)、[EXP-20](Q14-composition/EXP-20-bf16-wgmma-tile.md)、[COMPUTE-01 A–E 完整链条](Q14-composition/COMPUTE-01-resident-chain.md)（[B/C](Q14-composition/COMPUTE-01-resident-chain.md#bc-measured)、[D](Q14-composition/COMPUTE-01-resident-chain.md#d-measured)、[E](Q14-composition/COMPUTE-01-resident-chain.md#e-measured)）、[COMPOSITION-01](Q14-composition/COMPOSITION-01-replay.md)；[G3 stage 复测](../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-g3-phase-reproduction-001/README.md)、[八小时窗口的 27 轮 CPU 迭代](../../../../../results/gh200_resource_campaign/access_rules/non-experiment/campaigns/20261016-cfg-b-window-001/README.md)；[cfg_b组合](Q14-composition/cfg-b-composition.md)、[局部规则接回GEMM](Q14-composition/gemm-connection.md) | C2、C3 | 请求级组合可重放，原 12 条件的完整验收仍未通过。 |
| <a id="question-15"></a>15. 完整时间预测与模型对照 | 运行前输入 → kernel 完整时间与误差 | [规格基线](../../model/spec/spec_baseline.md)、[半经验模型](../../model/semi_empirical/MODEL-CANDIDATES.md)、[机制组合](../../model/mechanism/composition.md) | [V01](Q15-full-time/V01-validation.md)、[V02](Q15-full-time/V02-kernel-prediction.md)、[V03](Q15-full-time/V03-clock-rule.md)、[V04](Q15-full-time/V04-config-transfer.md)、[V05](Q15-full-time/V05-rule-transfer.md)、[V06](Q15-full-time/V06-revised-transfer.md)、[V07](Q15-full-time/V07-rule-validation.md)、[V08](Q15-full-time/V08-wider-validation.md)、[V09](Q15-full-time/V09-component-validation.md)、[COMPOSITION-01](Q14-composition/COMPOSITION-01-replay.md)、[C1-01 模型对照](Q16-config-selection/C1-01-selection.md) | C3 | 三类模型入口齐备，各版范围与成绩由原评分记录维护。 |
| <a id="question-16"></a>16. 配置选择 | 候选预测时间与排序 → 所选配置 | [C1 目标](../../GOALS.md#52-c1-选配置)、[候选模型](../../model/semi_empirical/MODEL-CANDIDATES.md) | [C1-01](Q16-config-selection/C1-01-selection.md) | C1 | 开发比较已留档，正式独立留出未完成。 |

实验页按主归属问题存放在 `Qnn-*/` 目录，一个合并实验一页，其他问题以链接引用；问题 10 的观测协议另有 `Q10-time-mapping/`，时间映射的证据仍在 INTERFACE 兼容副本、V09 与 C1-01。原始轮次存放在 `results/gh200_resource_campaign/access_rules/<问题>/<合并实验>/<YYYYMMDD-Qnn-名称-NNN>/`，新轮次放进对应合并实验目录，并在实验页的轮次表登记；新脚本以 `qnn_` 为前缀放在现有脚本目录。2026-10-20 的结果目录重排与旧页面路径对照见 [MOVED-PATHS](MOVED-PATHS.md)。

## 测量口径

| 项目 | 约定 |
|---|---|
| 单 CTA | 同一 CTA 的 `clock64` 差，单位为 X/cycle/CTA |
| 整卡 | 所有 CTA 的 `globaltimer` 最早起点到最晚终点，单位为 GFLOP/s 或 GB/s；不乘 SM 数换算 |
| 工作量 | FLOP、OP、逻辑字节分别计；逻辑字节不等于 HBM 物理流量（无 NCU 权限，缓存命中与物理流量均未测） |
| 计时窗口 | 包含循环控制、规定的等待、同步与结果排空；不是裸指令延迟 |
| 采样 | 每配置 10 个独立进程，8–30 次预热；默认报告中位数（例外在实验页注明），CV 普遍 <1% |
| 正确性 | 每次启动做完整输出校验，另有非均匀短输入检查；SASS 确认目标指令在循环内 |

例：FP32 FMA，1 CTA、32 线程、1 条链、每轮 16 个 FMA、128 轮，工作量 \(2\times32\times16\times128=131072\) FLOP，用时 9247 cycle，得 14.17 FLOP/cycle/CTA。

按三类模型阅读见[模型入口](../../model/README.md)，按实验编号查阅见下表。类别与模型使用方分开：本表所有 EXP 均是实测证据，不能因被模型引用而升级为硬件机制。

## 主要结论

规格计算另有[公式、来源与脚本入口](../../model/spec/spec_baseline.md)。下文"理论值"按 CC 9.0 每 SM 每周期：FP32 FMA 256 FLOP，FP16/BF16 Tensor Core 4096 FLOP，FP8/INT8 8192，SMEM 128 B。整卡 WGMMA 980 TFLOP/s 对应平均 SM 时钟约 1.81 GHz。

| 资源 | 类别 | 供谁使用 | 结论 | 详见 |
|---|---|---|---|---|
| CUDA Core | 实测 | 半经验、机制-硬件 | FP32 整卡 65.7 TFLOP/s、FP64 33.5、FP16x2 117；单 SM 需 4 warp × 8 条独立链才接近 256 FLOP/cycle | [EXP-05](Q11-other-compute/EXP-05-fma.md) |
| mma.sync | 实测 | 半经验、机制-硬件 | FP16/BF16 约 2700 FLOP/cycle，TF32 约 1340，均约为理论值的 65%；**FP64 `m8n8k4` 只有 128 FLOP/cycle，整卡 33.5 TFLOP/s，与 FP64 FMA 相同，仅为数据表 67 TFLOP/s 的一半** | [EXP-06](Q11-other-compute/EXP-06-mma.md) |
| WGMMA | 实测 | 半经验、机制-硬件 | m64n64k16 单个 warpgroup 在 wait≥3 时达 4095.9 FLOP/cycle（满峰值），整卡 960–984 TFLOP/s；wait0 降到 3858；RS 与 SS 相差 <1% | [EXP-07](Q05-wgmma/EXP-07-wgmma.md) |
| FP8 / INT8 | 实测 | 半经验、机制-硬件 | WGMMA 两个 warpgroup 达 8180 FLOP(OP)/cycle，整卡约 1.96 PFLOP/s；**均匀输入下，FP8 WGMMA 的 FP32 累加器到 64 后不再增长**，单个乘积为 \(2^{-8}\)；结果与约 14 bit 对齐窗口的解释相符；FP8 `mma.sync` 被编译为转换加 FP16 MMA，不是原生路径 | [EXP-08](Q11-other-compute/EXP-08-fp8-int8.md) |
| SMEM | 实测 | 半经验、机制-硬件 | 8/16 B 访问的读、写、读写合计都约 128 B/cycle，建模时不能把读、写峰值相加；4 B 读只到 87；stride=k 时速率为 128/k 量级 | [EXP-09](Q07-smem/EXP-09-smem.md) |
| ldmatrix / shfl | 实测 | 半经验、机制-硬件 | 单 warp 下 `ldmatrix.x4` 35 B/cycle，shfl 依赖链约 26 cycle/条；[R03](Q07-smem/R03-access-demand.md) 测得 8 warp 时 `ldmatrix.x4` 127、`stmatrix.x4` 116 B/cycle | [EXP-10](Q07-smem/EXP-10-ldmatrix-shfl.md) |
| 同步 | 实测 | 半经验、机制-硬件 | CTA barrier 21–29 cycle/phase；mbarrier 218–226；GPU 范围 fence 267 cycle，CTA 范围 fence 与 `fence.proxy.async` 约 11 | [EXP-11](Q04-sync-slot/EXP-11-sync.md) |
| cp.async | 实测 | 半经验、机制-硬件 | 单 CTA 128 线程最多 7.5 B/cycle；整卡 16 B、4 stage 达 5.3 TB/s（源仅 8 MiB，小于 L2，速率高于 HBM 产品值） | [EXP-12](Q02-tma-request/EXP-12-cp-async.md) |
| global | 实测 | 半经验、机制-硬件 | 大工作集读 3.52 TB/s、写 3.79、1:1 读写 3.43（HBM3 产品值 4 TB/s）；小工作集 `.ca` 读 13.5 TB/s | [EXP-13](Q03-source-service/EXP-13-global-rw.md) |
| TMA 1D bulk | 实测 | 半经验、机制-硬件 | 整卡读 ≥8 KiB 时 3.8 TB/s；**bulk 写只有 2.6–2.7 TB/s**；单 CTA 单请求受往返时间限制 | [EXP-14](Q02-tma-request/EXP-14-tma-1d.md) |
| TMA 2D tensor | 实测 | 半经验、机制-硬件 | 连续布局读写整卡均 3.8–3.9 TB/s，SW128 无代价；行距 144 B（只按 16 B 对齐）时写降到 1.33–1.45 TB/s；[R05-E](Q02-tma-request/EXP-15-tma-2d.md#r05-e) 证实行距 160 B（32 B 对齐）和 256 B 都正常，原因是行起点对齐 | [EXP-15](Q02-tma-request/EXP-15-tma-2d.md) |
| TMA 流水 | 实测 | 半经验、机制-硬件 | 单 CTA 读速率随缓冲容量增加：16 KiB 时 14.7 B/cycle，64 KiB 时 28.5–39.8；S1R1 时每个 16 KiB 请求的完整周期约 1110 cycle，是单请求往返时间的上界，不是实测延迟；整卡与 S、R 无关，约 3.7 TB/s | [EXP-16](Q02-tma-request/EXP-16-tma-pipeline.md) |
| cluster / DSM | 实测 | 半经验、机制-硬件 | 整卡 DSM 读约 2.2–2.5 TB/s，本地 SMEM 读 10–12 TB/s；TMA 多播在 C=2/4/8 时接收量均约 5 TB/s，逻辑源请求按 1/C 下降 | [EXP-17](Q12-cluster-dsm/EXP-17-cluster.md) |
| 驻留与辅助 | 实测 | 半经验、机制-硬件 | local 访问每字约 59 cycle；同址原子约 73 cycle/轮，无争用 28；cvt、IMAD.WIDE 等依赖链 29–43 cycle/轮 | [EXP-18](Q08-register-residency/EXP-18-occupancy-aux.md) |
| FP32 tile 组合 | 实测 | 半经验、机制-硬件 | 32×32×32、128 线程只有 18 FLOP/cycle（峰值的 7%）；搬运与计算重叠后每 Ktile 从 4022 降到 3683 cycle | [EXP-19](Q14-composition/EXP-19-fp32-tile.md) |
| BF16 WGMMA 组合 | 实测 | 半经验、机制-硬件 | 64×64×64、1 warpgroup、每 Ktile wait0：纯计算 830 FLOP/cycle（20%），平均每条 WGMMA 约 150 cycle（含 fence、commit、wait0 与循环，不是裸延迟）；加入 TMA 供给后每 Ktile 约 1000–1120 cycle，搬运未被隐藏 | [EXP-20](Q14-composition/EXP-20-bf16-wgmma-tile.md) |

## 其他入口与历史合同

| 文档 | 类别 | 供谁使用与状态 |
|---|---|---|
| [规格基线](../../model/spec/spec_baseline.md) | 规格 | 三类模型的理想参照；硬件参数仍在 hardware 维护 |
| [程序图](../../model/mechanism/program/cutlass_program_graph.md) | 机制-程序 | CUTLASS 依赖与槽代次；供机制组合 |
| [硬件路径研究](../../model/mechanism/hardware_rules/README.md) | 机制-硬件 | 汇总 TMA、频率、m11 与其他路径；引用 EXP/R 证据 |
| [模块级微基准地图](../../model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md) | 机制-硬件 | 九个服务方向与组合层的补测设计、既有证据和模型接入；分阶段交付及有限实测，当前测量状态见各实验页 |
| [EXP-01](EXP-01-compute-audit.md)、[EXP-03](EXP-03-compute-baseline.md) | 实测 | 早期计算审查/基线，只读历史，不替代当前 EXP-05/06/07 |
| [EXP-04](Q03-source-service/EXP-04-memory-baseline.md) | 实测 | 内存基线已合并，保留原路径导航 |
| [INTERFACE 合同及旧 EXP 副本](HISTORICAL-PATHS.md) | 实测 | 旧 suite 的采样合同、设计和审查绑定；31 个只读文件逐项列于该页，不作为当前模型或执行要求 |

## 锚点与访问规则实验（access_rules，2026-10-06 起）

FP16 Tensor Core GEMM 的当前结论、适用范围和完整实验索引统一见 [access_rules](access_rules/README.md)。

半经验公式见[MODEL-CANDIDATES](../../model/semi_empirical/MODEL-CANDIDATES.md)，采用资格见[RULES](access_rules/RULES.md)，测量细则见[RESEARCH-GUIDE](access_rules/RESEARCH-GUIDE.md)。

## 资源基准未覆盖的量与路径

- TF32 WGMMA、FP16 累加形式；FP64 `mma` 的 sm_90 新形状。
- 多 CTA 竞争下的片上服务。
- 物理 HBM 流量、缓存命中、动态指令计数（需 NCU 权限）。
- 物理在途队列深度（软件 stage 扫描不能给出）。

## 实验记录约定

各实验页说明配置、工作量、计时起止事件、样本波动、适用条件以及数据与脚本位置。输出校验与 SASS 检查说明测到了什么；物理机制的解释单独标明推断。历史审查与过程记录保存在归档中。

## 数据与代码

- 原始数据：`results/gh200_resource_campaign/20261001-resource-suite-v2/`（不入 git），各实验文档给出具体目录。
- 全部 754 条条件观测的索引：[index.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s21-seventeen-family-index-v1/index.json)。
- 探针与运行脚本：[microbench/gh200_resource_campaign](../../../../../microbench/gh200_resource_campaign/README.md)。
- 实验设计、审查流程、旧版结果和 v1 历史记录：[gh200_sm90_archive](../gh200_sm90_archive/ARCHIVE.md)。
