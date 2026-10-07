# V01：留出组合验证

[总计划](README.md)。状态：异步目标组合单 CTA 预测通过；整卡与完整 kernel 的规则预测未通过，原因已定位（见“结论”）。代码：`run_v01.py`、`v01_predict.py`、`probes/v01.cu`。

## 问题

R00–R06 的规则能否预测未参与拟合的组合？只有组合检查能发现重复计账、错误的完成事件和不匹配的服务条件。S19/S20 分别只到峰值的 7% 和 20%，在这类组合上预测准不代表在 GEMM 工作点上也准，所以本组加入接近实际的 Tensor Core 组合和完整 kernel。

## 默认矩阵（20 点）

| 组合 | 执行内容 | 长度 | 范围 |
|---|---|---|---|
| LDS→FFMA | 每线程 4×4 FP32 输出块；每 K 步读 4 个 A、4 个 B，做 16 次 FFMA；128 线程；输入预先在 SMEM | K 步 7/19/47 | 单 CTA / 整卡 |
| 小 TMA→WGMMA→输出 | S20 的 64×64×64 组织，改为 FP16 输入、FP32 累加；stage 2，合法等待与最终输出 | Ktile 7/19/47 | 同上 |
| 目标 TC 组合 | 见下文 | Ktile 7/19/47 | 同上 |
| 完整 kernel | R00-A 的 CUTLASS 固定配置 | 2048³、2048×2048×8192 | 整卡 |

三组合 × 三长度 × 两范围 = 18 点，加 2 个完整 kernel 点。整卡 grid 为 4×SM 数，资源拒绝记录在案。首次采样前，7/19/47与两个完整kernel尺寸不参与拟合或系数估计。观测后可以转为校准点，后续验证须换未观测的新点；完整kernel点在保存预测之后才运行。

### 目标 TC 组合

- FP16 输入、FP32 累加、FP32 输出；tile 128×256×64；cluster 1×1×1（不含多播）。
- 384 线程：线程 0–127 为 producer 区，其中 warp 0 发输入 TMA，其余线程参与 CTA 同步；线程 128–255 与 256–383 是两个 consumer warpgroup，各发 `m64n256k16`、各算 64×256 输出。
- 输入：每 stage 一个 A tile（16 KiB）+ 一个 B tile（32 KiB）= 48 KiB，4 stage 共 192 KiB，另加 barrier。
- 等待：稳态最多留 1 个最近的 group 未完成（`wait_group 1`）；输入槽在覆盖其最后一次消费的 group 完成后释放；最后 wait0 再读累加器。
- 输出：128×256 FP32 共 128 KiB，不另开缓冲（否则 192+128=320 KiB，超过 227 KiB 上限）。全部输入完成、两个 consumer 最终 wait0 后，复用输入区：分两块 64 KiB，依次做寄存器→SMEM、proxy/CTA 发布、TMA 写回、完整 wait0。两块都在停止计时前完成。
- 记录实际 SMEM、寄存器、occupancy 与 mbarrier/group/槽的对应关系；确认资源满足后再扩大测量。

CUTLASS 完整 kernel 使用 cluster 2×1×1，其多播与 cluster 调度在整卡预测中单独计入。

输入major也分别记录：本组受控TC组合的A/B均Major::K，而行主序完整CUTLASS为A Major::K、B Major::MN（`.tnspB`）；后者还有producer/consumer寄存器重分配。R00-B只提供相同形状的孤立服务，不能自动替代完整kernel的布局与资源条件。首版20项数值预测均为空，仅接受测量观测；后来已测长度和尺寸不再作为新预测的未知留出点。

## 预测

每个组合保存一张可读的计算表：工作与访问需求、事件依赖、引用的规则及条件、并发与资源、预测的起止时间。直接递推即可，不必先写通用模拟器。

- R00-B：目标形状的 WGMMA 服务；R01：依赖时间（R02 若启用则补充）。
- R03：多 warp 下的 SMEM/`ldmatrix`/`stmatrix` 服务。
- R04：哪些服务能同时兑现。
- R05：输入、消费、释放与完整输出事件；目标在途量下的每 SM 供给。
- R06：寄存器与并发条件。

已包含取数、等待的整段服务不再重复加。整卡预测需要整卡共享服务、grid 与波次条件；完整 kernel 还需要 L3 的任务分配与 L2 复用。缺输入时写"当前无法预测"并列出缺项，实测照常运行和保存。预测在正式采样前保存，之后不改。

## 指标

\[
e=\frac{\widehat T-T_{\mathrm{measured}}}{T_{\mathrm{measured}}}
\]

单 CTA 用 cycle，整卡用 ns。初版目标：可预测点的绝对误差中位数 ≤10%、最大 ≤20%。结果表给出预测、实测中位数、样本范围/CV、误差与引用规则。未达标时，先定位是需求映射、依赖/完成、共享服务、缓存状态还是调度，再补最小的区分实验。

## 工作量

- LDS→FFMA：每线程每 K 步读 8 个 FP32（32 B）、16 FFMA（32 FLOP）；128 线程、K=7 时为 28672 B 与 28672 FLOP。
- 小 TMA→WGMMA：`524288K` FLOP、输入 `16384K` B、输出 16384 B（只输出累加结果，不执行S20中额外8192 FLOP的epilogue）。
- 目标 TC 组合：每 Ktile \(2\times128\times256\times64=4194304\) FLOP、输入 \((128+256)\times64\times2=49152\) B；最终输出 \(128\times256\times4=131072\) B，只写累加结果。

完成标准：20 点都有正确结果或明确的资源终态；预测在采样前保存；误差可重算；未闭合的关系逐项列出。进入具体方案后，把方案的实际 kernel 加入留出点。

## 结论（2026-10-06）

| 对象 | 预测方式 | 误差 | 判定 |
|---|---|---|---|
| 异步目标组合，单 CTA，Ktile 7/19/47 | 规则组合，采样前冻结 | −1.7% / −0.9% / −0.4% | 通过 |
| 异步目标组合，整卡，Ktile 7/19/47 | 同上，按 1.83 GHz 换算 | −6.6% / −7.0% / −10.0% | 未达 ±5%；窗口比值频率 1.79/1.75/1.66 GHz，低于假设 |
| 固定 CUTLASS，2048³、2048×2048×8192 | 规则组合 | −28.3% / −22.7% | 未通过 |
| 固定 CUTLASS，新 K（4096、5120、6144） | 用同一 kernel 拟合后内插 | +0.5% / −0.7% / −0.6% | 拟合有效，但不是规则验证 |

1. **规则组合在单 CTA 层面成立。** 异步目标组合（2 个 consumer warpgroup、每 Ktile 4 条 `m64n256k16`、`wait_group 1`、4 stage）主循环约 1026 cycle/Ktile，理想值 1024，WGMMA 已满速。
2. **整卡误差与频率假设一致。** K=47 时窗口比值频率比假设低 10.3%，误差 −9.95%；这是优先解释，尚未做固定频率对照。整卡预测需要负载下的频率，见 [R07](R07-anchor-clock-fixedcost.md)。
3. **完整 kernel 的规则缺少固定开销与组合成本。** 同一 kernel 的经验关系（[R07](R07-anchor-clock-fixedcost.md)，`NDEBUG` 构建）为 \(T\approx9.45+0.637\,\text{Ktile}\ \mu s\)（M=N=2048）；截距在 2048³ 约占 1/3，与规则预测的 −28% 量级一致。[R09](R09-inkernel-clock-stages.md) 拆分后，该关系可由各段重建：主循环 1024 cycle/Ktile（已达计算下界），时间斜率来自调用内频率约 1.64 GHz；截距约 9.2 µs = 主机间隙 3.7–3.9 + 预填约 2.2 + epilogue 约 2.9 + store 后与尾部约 0.8 µs。规则模型漏掉的是这些固定段与调用内频率。
后续：补入 R07–R09 的输入后，[V02](V02-kernel-prediction.md) 对 11 个未测尺寸的完整 kernel 预测误差中位数 6.0%、最大 10.3%。

4. **R00 的 CUTLASS 没开 `NDEBUG`**，编译器把每条 MMA 都改成提交后 wait0；加 `NDEBUG` 后快 5–6%。重测见 R07。

预测文件冻结后计时探针改过一次；单 CTA 各段误差互相抵消（预填预测 1481、实测 1148 cycle；输出预测 6044、实测 6629）。

## 实现要点

- 目标组合：384 线程（线程 0–127 为 producer 区，warp 0 发 TMA；两个 consumer warpgroup），154 寄存器/线程、无 spill，SMEM 197888 B，1 CTA/SM。每 Ktile 每个 consumer 发 4 条 WGMMA 后 commit、`wait_group 1`；上一 tile 的 group 完成后释放其输入槽。最后 wait0，复用输入区分两块 64 KiB 经 TMA 写回并完整 wait0，窗口在第二块完成时结束。SASS 无 C7520/C7517/C7519，主循环 8 条 HGMMA、2 次 wait1。
- 预测由 `v01_predict.py` 生成：WGMMA 服务（R00-B）、每 SM 供给（R05-D）、事件与等待（R05、R01）、TMA 写回（R05-E）。预测文件 SHA256 `17026427…`，冻结于测量前。
- 整卡 grid 为 4×132，所有 CTA 读同一输入；完整 CUTLASS 复用 R00 二进制，CUDA event 计时。
- 首版（作业 735062/735138）目标组合被编译器逐条串行（C7520），且没有基于规则的预测，已被本版取代；LDS→FFMA 辅助组合重测了完成边界，预测仍为空。

## 数据

- 异步版：[V01 报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-async-v01-r04/v01-offline-review/report.md)（作业 735203，GPU-7c184a2e…）。
- 拟合与内插：[新长度报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-v01-followup/report.md)、[固定 CUTLASS 分阶段模型](../../../../../../results/gh200_resource_campaign/access_rules/20261006-cutlass-wait-matching/published/report.md)。
- 辅助 LDS 完成边界：[报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-lds-completion/review/report.md)。
- 旧版：`access_rules/20261006-A-job735062/`、`20261006-A-fix-job735138/`。

```bash
python3 microbench/gh200_resource_campaign/access_rules/v01_predict.py --output <预测.json>
python3 microbench/gh200_resource_campaign/access_rules/run_v01.py --cutlass-root <R00 source>/cutlass \
  --r00-archive <R00 归档> --predictions <预测.json> --prediction-sha256 <SHA> --output <新目录>
python3 <新目录>/source/analyze_v01.py --input <新目录>
```
