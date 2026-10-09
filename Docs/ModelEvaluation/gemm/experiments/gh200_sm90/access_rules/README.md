# GH200 访问规则与 GEMM 时间预测

## 当前结论

**V08 整体未通过。** 同一张 GH200、固定 CUTLASS cfg_a/b/c 配置的 36 个新留出条件，总时间绝对相对误差中位数 **2.61%**、最大 **35.34%**，未达到 5% / 10% 目标。失败主要涉及 A/B 行距未按 128 B 对齐、swizzle=8 的整 cluster 补齐 tile，以及 cfg_b 大足迹长 K。跨卡尚未验证。

测后按 swizzle=1、A/B 行距为 128 B 倍数、K ≤ 16384 限定出的 21 个条件为 **1.90% / 6.04%**；这是完整留出集的子集分析，不是独立验证，也不代表整轮通过。逐例结果及后续对照见 [V08](V08-wider-validation.md)。

[V07](V07-rule-validation.md) 在同卡、cfg_a/b/c、swizzle=1、K 位于校准范围内的 24 个条件（12 个不同 M×N）上为 **3.06% / 8.71%**，总时间通过；供给、末次输出和关键 CTA 定位仍失配，20/24 预测偏长。它使用 V06 事件递推、同卡重新校准与 R18 边界修正；R19 只作诊断，R02/R11/R12/R16/B01 未用于 V07。

规则推导与计量约定见 [RULES](RULES.md)。[微架构机制审阅](MICROARCH-REVIEW.md)区分已有证据与模型缺口；[PLAN](PLAN.md)先恢复 V08 失败项主线（每 SM 有效供给、补齐 tile、输出窗口、时间换算、新完整验证），再把 25 类机制问题归入原 EXP/R/B 实验。实施已启动，当前批次与分工见 [EXECUTION](EXECUTION.md#status)；V09 尚未冻结。

<a id="当前状态2026-10-07"></a>

## 实验索引

“已测”表示有测量记录；预测是否通过以及可用范围另列。误差均为绝对相对误差中位数 / 最大值；各轮目标和测试集不同，不直接据此比较模型优劣。

| 实验 | 状态 | 主要结果与限制 |
|---|---|---|
| [R00](R00-anchor-target.md) GEMM 锚点 | 已测 | 所测 WGMMA 形状达到指令峰值；建立完整 GEMM 基线 |
| [R01](R01-readiness.md) 依赖与就绪 | 已测 | 依赖链与独立流服务；窗口含地址、消费和循环成本 |
| [R02](R02-register-service.md) 寄存器服务 | 已测，未用于 V07 | 固定 40 寄存器后轮换源慢 5.26%；RF 端口未归因 |
| [R03](R03-access-demand.md) 片上访问 | 已测 | 8 warp 的所测 16 B SMEM/ldmatrix 服务约 124–127 B/cycle |
| [R04](R11-mixed-issue.md#r04) 联合服务 | 已并入 R11 | WGMMA/FFMA 接近重叠；LDS 配对只作诊断 |
| [R05](R05-async-lifecycle.md) 异步生命周期 | 已测，范围受限 | 等待、stage 与复用服务；简单探针的流水重叠受限 |
| [R06](R06-issue-residency.md) 发射与驻留 | 已测 | FFMA 接近发射上限；未得到统一驻留惩罚 |
| R07 频率与固定项 | 已并入 [R00](R00-anchor-target.md#ndebug)、[R09](R09-inkernel-clock-stages.md#r07) | NDEBUG 锚点；调用后探针不能代替调用内频率 |
| [R08](R08-waves-l2-reuse.md) 波次与遍历 | 已测 | 所列形状按离散波次计费；遍历收益依条件变化 |
| [R09](R09-inkernel-clock-stages.md) 调用内分段 | 已测，输入扩展已完成 | 周期与时间须分开；三档输入已完成长窗口同卡对照，频率依赖输入数值 |
| [R10](R10-layout-cache.md) 行距与尺寸尾部 | 已测，局部覆盖候选未通过 | 匹配容量后 A/B 联合行距代价仍存在；五档 B 行距的冻结留出否定了本批两个简单覆盖候选 |
| [R11](R11-mixed-issue.md) 混合发射 | 已测，未用于 V07 | 消费者匹配后，各配对的重叠程度不同 |
| [R12](R12-smem-path-contention.md) SMEM 多路竞争 | 已测，未用于 V07 | TMA/WGMMA 接近重叠；与 STS 并发有额外代价 |
| [R13](R13-async-retirement.md) 供给与退役 | 已测，条件候选待转移 | 行距代价随活动规模和输出位置变化；有效地址/填零候选已进入开发递推；尚无新的完整留出成绩 |
| [R14](R14-stage-handoff.md) 阶段交接 | 已测，迁移受限 | 四 CTA 校准未能直接迁移供给与交接到整卡 |
| [R15](R15-output-service.md) 输出服务 | 已测，条件对照 | 标量探针不能直接作为 CUTLASS 输出常数 |
| [R16](R16-residency-quota.md) 驻留与配额 | 已测，未用于 V07 | 驻留收益依供给变化；寄存器申请等待取决于释放时序 |
| [R17](R18-cluster-boundary.md#r17) 越界 tile | 已并入 R18 | 改用匹配位置的窗口增量；旧两例修正只作诊断 |
| [R18](R18-cluster-boundary.md) cluster 边界 | 旧规则用于V07；新配对未接入 | cfg_b 同逻辑输出下，有效地址零比 M/N 向 TMA 越界填零快约17%–20%；分项仍随位置变化 |
| [R19](R19-critical-cta-tail.md) 关键 CTA 与尾部 | 已测，只作诊断 | swizzle 改变工作与尾部；V07 尾差项选择 none |
| [B01](B01-bandwidth-cache.md) 带宽与缓存 | 已测，未用于 V07 | 条件请求速率；不等同物理 HBM 流量或缓存收益 |
| [V01](V01-validation.md) 组合验证 | 部分通过 | 单 CTA 通过；整卡与完整 kernel 未通过 |
| [V02](V02-kernel-prediction.md) 完整 kernel | 初版目标通过 | 11 个新尺寸为 6.0% / 10.3%，按当时 10% / 20% 目标 |
| [V03](V03-clock-rule.md) 频率规则 | 同卡预测通过 | 11 个新尺寸为 2.6% / 5.0%；跨卡常数有差异 |
| [V04](V04-config-transfer.md) 配置迁移 | 部分通过 | cfg_c 迁移通过；cfg_a/b 仍失配 |
| [V05](V05-rule-transfer.md) 规则迁移 | 未通过 | 18 例为 10.7% / 31.9%；部分打点不作定量判定 |
| [V06](V06-revised-transfer.md) 递推修正 | 未通过 | 18 个留出条件为 5.25% / 19.0%；边界与尾部未解决 |
| [V07](V07-rule-validation.md) 边界迁移 | 总时间通过，分项受限 | 24 条件为 3.06% / 8.71%；范围与分项限制见上 |
| [V08](V08-wider-validation.md) 更宽留出 | 整体未通过 | 36 条件为 2.61% / 35.34%；21 条件子集见上 |

## 数据与代码

- [运行、拟合与分析代码](../../../../../../microbench/gh200_resource_campaign/access_rules/)；各实验页给出具体脚本、参数与复现命令。
- [原始结果归档](../../../../../../results/gh200_resource_campaign/access_rules/)按 run-id 保存源码、构建、环境、样本与分析；冻结预测及绑定记录见各 V 页。
- [规则说明](RULES.md)保留计量约定、手算例子与 2026-10-08 一次性离线复现记录。
- [历史过程文件](../../gh200_sm90_archive/access_rules/)保留原样；不作为当前执行要求。
