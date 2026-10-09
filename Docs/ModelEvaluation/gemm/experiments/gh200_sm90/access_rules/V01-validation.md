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
2. **整卡误差与频率假设一致。** K=47 时窗口比值频率比假设低 10.3%，误差 −9.95%；这是优先解释，尚未做固定频率对照。整卡预测需要负载下的频率，见 [R07](R09-inkernel-clock-stages.md#r07)。
3. **完整 kernel 的规则缺少固定开销与组合成本。** 同一 kernel 的经验关系（[R07](R09-inkernel-clock-stages.md#r07)，`NDEBUG` 构建）为 \(T\approx9.45+0.637\,\text{Ktile}\ \mu s\)（M=N=2048）；截距在 2048³ 约占 1/3，与规则预测的 −28% 量级一致。[R09](R09-inkernel-clock-stages.md) 拆分后，该关系可由各段重建：主循环 1024 cycle/Ktile（已达计算下界），时间斜率来自调用内频率约 1.64 GHz；截距约 9.2 µs = 主机间隙 3.7–3.9 + 预填约 2.2 + epilogue 约 2.9 + store 后与尾部约 0.8 µs。规则模型漏掉的是这些固定段与调用内频率。
后续：补入 R07–R09 的输入后，[V02](V02-kernel-prediction.md) 对 11 个未测尺寸的完整 kernel 预测误差中位数 6.0%、最大 10.3%。

4. **R00 的 CUTLASS 没开 `NDEBUG`**，编译器把每条 MMA 都改成提交后 wait0；加 `NDEBUG` 后快 5–6%。重测见 [R07，现并入 R00](R00-anchor-target.md#ndebug)。

预测文件冻结后计时探针改过一次；单 CTA 各段误差互相抵消（预填预测 1481、实测 1148 cycle；输出预测 6044、实测 6629）。

## 实现要点

- 目标组合：384 线程（线程 0–127 为 producer 区，warp 0 发 TMA；两个 consumer warpgroup），154 寄存器/线程、无 spill，SMEM 197888 B，1 CTA/SM。每 Ktile 每个 consumer 发 4 条 WGMMA 后 commit、`wait_group 1`；上一 tile 的 group 完成后释放其输入槽。最后 wait0，复用输入区分两块 64 KiB 经 TMA 写回并完整 wait0，窗口在第二块完成时结束。SASS 无 C7520/C7517/C7519，主循环 8 条 HGMMA、2 次 wait1。
- 预测由 `v01_predict.py` 生成：WGMMA 服务（R00-B）、每 SM 供给（[R05-D](R13-async-retirement.md#r05-d)）、事件与等待（R05、R01）、TMA 写回（[R05-E](../EXP-15-tma-2d.md#r05-e)）。预测文件 SHA256 `17026427…`，冻结于测量前。
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

<a id="joint-supply-clock"></a>

## 2026-10-09：供给、事件与频率的开发组合

这次复用 GPU-43269fbc 上 R10 行距和 R18 两方向输入 map 的 13 个条件，检查新组件能否组合；**这些条件已经用于供给与事件参数开发，不是新的冻结验证**。原 V01 冻结记录不变。

[v09_model.py](../../../../../../microbench/gh200_resource_campaign/access_rules/v09_model.py) 从软件调度计算逐 CTA 工作与请求，把 `max(512/f, 条件供给ns)` 接入原 pingpong 递推；P0/S/输出等使用同卡纳秒参数，F 使用同调用 CUDA event 减 CTA 包络后汇总的常数。R09 的计算/源请求频率规则与事件时间联立求解，不读取目标时间、实测频率或实测 CTA 列表。

| 形式 | dual CUDA event 误差中位 / 最大 | 含义 |
|---|---:|---|
| 条件供给＋固定 1.6 GHz | 3.46% / 5.95% | 前次开发基线 |
| 条件供给＋R09 频率联立 | 4.14% / 11.74% | 组合后退步，尚不能冻结为已验证规则 |
| 改用目标实测频率 | 4.08% / 7.06% | 只作归因诊断，不是可用预测 |

自由组合的 plain 时间误差为 3.97% / 12.31%；plain/dual 的观测换算尚未重新定值。最差的是 K=4096 的有效地址零填充：预测频率约 1.329 GHz，比观测低约 12%，使时间偏长约 10–12%。长 K 的 OOB 条件总时间虽接近，所选 CTA 仍比实际最后退出者早约 28.6–28.7 µs，关键 CTA 定位没有解决。不能以总时间接近替代分项验证。

当前组件只定值到 cfg_b、132 SM、6 stage、完整 Ktile、A 行距对齐。R09 的 21 个训练条件全部 K=1024，计算工作与输出字节共线；这次迁移失败尚不能归因于某一个物理功耗来源。下一步先在 R09 做等名义计算/源请求量、不同 K 与输出次数的配对，分离这个缺口；不在这 13 个目标上追加自由修正项。

[逐例结果](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/manager-joint-clock-supply-v4/composition.json)与[CPU 检查](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R10-b-coverage-job738203/reanalysis/manager-joint-clock-supply-v4/cpu-checks.json)保留全部条件、实测频率诊断和 CTA 定位差距。13 条件的目标观测污染检查、输入不变检查、频率网格上的回调单调性与联立闭合检查通过；网格检查不替代对所有未来组织的证明。


### 三配置与输出分类接入（2026-10-09）

job738496 在同一 GPU-43269fbc 上完成 29 个数值预检查和 1160 个正式进程；三配置共用这些数据定值。供给按软件可判定的 first/later 两个阶段使用同一 max 形式，避免把后续 B 行距代价直接搬到首段。cfg_b 经实际 SASS、端点与桥接核对后联合旧 13 条件，分配容量差异仍保留说明。以下 **42 条件均已用于开发，不是新留出**：

| 组合版本 | plain 时间误差中位 / 最大 |
|---|---:|
| phase 供给＋零工作比例混合时钟＋常量输出 | 4.35% / 8.21% |
| 再按 single/multi、real/whole-OOB 使用 R15 输出规则 | 4.57% / 7.88% |

后一版保留了分项更明确的输出关系，未因前一版中位误差略小就忽略输出类别。E0 只计 J+R；middle/last 已是 merged 窗口，不再加 R；整 case 的 T_max=1 才选 single 规则。whole-OOB 的输出和尾段非零；partial 按有效输出比例插值只是下一批待验证的假设。cfg_c 普通 T≥3 的 middle 仍借用 padding 中 real 输出的代理。零工作混合也只是固定端点参数的插值，不是物理功耗定律。

[完整组合与参数](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/manager-three-config-composition-v3/composition.json)保留全部条件及支持标记。plain/dual 协议换算只用实测 dual 包络与 plain event 拟合，不用不准确的组件预测吸收误差；三配置的校准最大换算残差为 2.13%/1.84%/1.22%。

[CTA 定位诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/manager-critical-cta-v1/summary.json)检查的是此前 29 条件 phase/mix 版本：不少 CTA 在模型中完全并列，不能把第一个索引称作唯一最慢者；两例 cfg_a 长 K 行距条件的这个代表 CTA 仍比真正最后完成者短约 10%，虽然后者都在并列集合内。两个 cfg_a OOB 条件的真正最后完成者还会落在该集合外。关键 CTA 唯一定位尚未解决，局部慢窗口残差也不因总时间接近而消失。

新 stage 组织的准备沿用公共框架：`stages=4` 选独立 cfg_b_s4 二进制，默认构建命令不变。job738599 在另一张 GH200 上完成默认/4-stage 的八个正确性检查；均 168 registers、16 HGMMA、无 spill，4-stage SMEM 为149504 B，仍1 CTA/SM。默认四变体 SASS 与 job738496 完全相同。这只证明合法构建与数值正确，检查中的时间不进入拟合；新组织的性能迁移留给 V09。


首轮填零补测后，A/C first 各合并4个新条件，原 later 与 B19 保持不变；全部50个已见开发条件的[组合重放](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/manager-three-config-composition-v4/report.json)中有49个支持，支持子集 plain 误差为3.35%/7.98%。剩余 cfg_c_first_fill_oob_k1024 在自由预测频率下离开已识别供给分支；其数值只作诊断，不能把49个子集成绩称为50个全部通过。新 cfg_a 的后续窗口也仍有局部最大约40%的误差。下一步冻结现有模型并保留拒绝项，不为了凑齐支持继续增加模型项或补测。


<a id="v09-ablation"></a>

## V09测后消融：同卡聚合基线与组件贡献

本次只读旧记录，V09是开发数据，不重新判定通过。基线只借V08的结构选择，用同卡51个既有校准条件重新定值；不借旧卡参数。四组共享软件工作、V09时钟和plain/dual映射，固定比较原有受支持预测的29个条件：

| 形式 | 总时间中位 / 最大 | RMS |
|---|---:|---:|
| V09完整候选 | 4.11% / 35.46% | 9.06% |
| 仅去掉输出分类 | 4.33% / 35.46% | 9.16% |
| 仅将主循环换成V08聚合周期形式 | 4.36% / 45.26% | 10.98% |
| 同卡重拟合V08形式＋新时钟 | 5.56% / 48.68% | 12.26% |

当前证据不支持整体退回聚合模型，但也不支持继续堆分支。供给条件化使first/later主循环中位误差由8.91%/8.92%降到4.97%/7.62%，却在cfg_b A+16B和cfg_a partial上更差；这些相反响应仍需从原对照解释。输出分类的总误差收益小，末次输出中位误差则由6.19%降到3.73%，小规模single也有改善。

cfg_b G6用观测频率作诊断时，完整候选误差为+1.71%，聚合供给为+17.60%，完整V08形式为+23.16%；因此不能把全部差异归到共享时钟。另一方面，四组自由频率下都未解决长K失败。现阶段保留三个组件，先处理长K频率、行距/填零跨几何响应和首段，不增加新模型分支。

完整条件、局部窗口、同卡拟合依据和复现命令见 [消融分析](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/model-simplification-v1/ablation/README.md)。V08结构与V09使用各自既定拟合策略，因此完整两模型的差距不能全部归因于供给；V08的X/边界总量修正也不被强造为局部事件。原冻结、拒绝项和评分不变。
