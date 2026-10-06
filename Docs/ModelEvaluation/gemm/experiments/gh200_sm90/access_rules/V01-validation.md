# V01：留出组合验证

[总计划](README.md)。状态：已实现并通过短检查；正式进度见[STATUS](STATUS.md)。代码：`run_v01.py`、`probes/v01.cu`。

## 问题

R00–R06 的规则能否预测未参与拟合的组合？只有组合检查能发现重复计账、错误的完成事件和不匹配的服务条件。S19/S20 分别只到峰值的 7% 和 20%，在这类组合上预测准不代表在 GEMM 工作点上也准，所以本组加入接近实际的 Tensor Core 组合和完整 kernel。

## 默认矩阵（20 点）

| 组合 | 执行内容 | 长度 | 范围 |
|---|---|---|---|
| LDS→FFMA | 每线程 4×4 FP32 输出块；每 K 步读 4 个 A、4 个 B，做 16 次 FFMA；128 线程；输入预先在 SMEM | K 步 7/19/47 | 单 CTA / 整卡 |
| 小 TMA→WGMMA→输出 | S20 的 64×64×64 组织，改为 FP16 输入、FP32 累加；stage 2，合法等待与最终输出 | Ktile 7/19/47 | 同上 |
| 目标 TC 组合 | 见下文 | Ktile 7/19/47 | 同上 |
| 完整 kernel | R00-A 的 CUTLASS 固定配置 | 2048³、2048×2048×8192 | 整卡 |

三组合 × 三长度 × 两范围 = 18 点，加 2 个完整 kernel 点。整卡 grid 为 4×SM 数，资源拒绝记录在案。7/19/47 与两个完整 kernel 尺寸都不参与任何拟合、长度选择或系数估计；完整 kernel 点在保存预测之后才运行。

### 目标 TC 组合

- FP16 输入、FP32 累加、FP32 输出；tile 128×256×64；cluster 1×1×1（不含多播）。
- 384 线程：线程 0–127 为 producer 区，其中 warp 0 发输入 TMA，其余线程参与 CTA 同步；线程 128–255 与 256–383 是两个 consumer warpgroup，各发 `m64n256k16`、各算 64×256 输出。
- 输入：每 stage 一个 A tile（16 KiB）+ 一个 B tile（32 KiB）= 48 KiB，4 stage 共 192 KiB，另加 barrier。
- 等待：稳态最多留 1 个最近的 group 未完成（`wait_group 1`）；输入槽在覆盖其最后一次消费的 group 完成后释放；最后 wait0 再读累加器。
- 输出：128×256 FP32 共 128 KiB，不另开缓冲（否则 192+128=320 KiB，超过 227 KiB 上限）。全部输入完成、两个 consumer 最终 wait0 后，复用输入区：分两块 64 KiB，依次做寄存器→SMEM、proxy/CTA 发布、TMA 写回、完整 wait0。两块都在停止计时前完成。
- 记录实际 SMEM、寄存器、occupancy 与 mbarrier/group/槽的对应关系；确认资源满足后再扩大测量。

CUTLASS 完整 kernel 使用 cluster 2×1×1，其多播与 cluster 调度在整卡预测中单独计入。

输入major也分别记录：本组受控TC组合的A/B均Major::K，而行主序完整CUTLASS为A Major::K、B Major::MN（`.tnspB`）；后者还有producer/consumer寄存器重分配。R00-B只提供相同形状的孤立服务，不能自动替代完整kernel的布局与资源条件。当前20项数值预测均为空，只接受测量观测，预测精度验收未完成；原已测长度和尺寸不再作为新预测的未知留出点。

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

## 当前实现与复核入口

`probes/v01.cu`包含三个组合，短检查为K=2/3，目标组合另以K=5检查四stage槽的复用。输入是R00同公式的坐标dyadic FP16值，CPU按精确周期17重算完整输出；这项有限输入检查不替代未定义的任务误差容差。

目标TC实际资源：154 registers/thread、无spill，动态SMEM196864 B（192KiB输入和256B barrier余量）+静态1024 B=197888 B，384线程，occupancy上限1 CTA/SM。每tile两个consumer分别发4条K16 WGMMA，commit/wait1使上一tile完成；CTA同步后producer仅覆盖上一tile的槽，填入上一tile+4。所有输入最后一次消费完成、两个consumer wait0并CTA同步后，输入区才复用为输出。每块64KiB的输出经过寄存器→SMEM、proxy fence、CTA发布、TMA写回和完整bulk wait0。

整卡组合grid为4×SM，所有CTA读取同一组输入，repeat-cache条件记录在案；计时是各CTA的globaltimer包络。LDS输入在计时前初始化，窗口包括输出store与消费者同步。完整CUTLASS复用R00归档固定二进制并核对SHA256，使用CUDA event计时，单位单独保留。

预测文件在采样前保存工作量、事件序列、资源条件、引用规则和具体缺项，绑定SHA256；没有完整参数时保存`predicted_time=null`。未取得匹配条件的服务与cluster任务分配前，本组只能交付留出观测，不能声称预测误差验收通过。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_v01.py \
  --prediction-template > /待保存预测.json
python3 microbench/gh200_resource_campaign/access_rules/run_v01.py \
  --cutlass-root /R00准确归档/source/cutlass \
  --output /新运行目录/v01-smoke --smoke
python3 microbench/gh200_resource_campaign/access_rules/run_v01.py \
  --cutlass-root /R00准确归档/source/cutlass \
  --r00-archive /R00准确归档 \
  --predictions /待保存预测.json \
  --output /新运行目录/v01-formal
python3 /新运行目录/v01-formal/source/analyze_v01.py \
  --input /新运行目录/v01-formal
```

三个组合18点和完整CUTLASS2点均只在预测文件保存后运行。7/19/47不作校准或拟合；没有新增长度校准。单CTA默认3进程，CV>1%补至10；整卡10进程。

## 实测结果（2026-10-06）

[V01留出观测](../../../../../../results/gh200_resource_campaign/access_rules/20261006-A-job735062/v01-offline-review/report.md)。设备romeo-a043，GPU-099dda56-d7af-f60e-c285-aa2dc7ddfcfe，CUDA12.9。

20点/137个正式进程，CPU完整输出和CUTLASS保存的4096个坐标样本最大误差0；所有预热窗口收敛，正式窗口最大CV约4.07%。18个组合保存完整输出，2个CUTLASS点只对保存的有限坐标样本重算。

完整CUTLASS留出2048³为31.248 µs、549.791 TFLOP/s，2048×2048×8192为97.952 µs、701.563 TFLOP/s，均为10进程中位数。预测文件在运行前保存且SHA256一致；20项预测均为空并有具体缺项，所以这是留出观测交付，预测误差验收尚未完成。

目标核存在C7520逐条串行化和C7517等待，小TC存在C7517等待，源级wait1不能证明实际保留一个group。机器码条件、工作量、资源与时间边界写入离线validation.json；后续规则必须匹配这些条件，或修正探针后仅重测受影响点。

## 串行生命周期修订与新留出

受控TC保留真实串行消费：每条K16 WGMMA显式fence/commit/wait0，块退出没有在途group；目标154 registers/thread、小TC58 registers/thread，无spill，C7520/C7517均消除。预填、主循环最终排空、两块完整TMA输出用同CTA clock64连续分段，分段之和等于完整窗口。它不是恢复后的理想异步wait1实现。

在GPU-7c184a2e-41ea-3d2b-df5b-1699c95fc1fe、CUDA12.9/sm90a上，以目标128×256×64、384线程、4输入stage、K/K、repeat-prepared输入，校准6/14/30，每点3独立进程。原7/19/47不参与系数估计。条件规则为：

\[
\widehat C(K)=979+9052+6270.5\frac{K-6}{4}+8959,
\qquad K\equiv2\pmod4.
\]

979为预填段中位数，9052为K=6主循环段，6270.5为每额外4 tile主循环完成增量，8959为完整输出段。主循环增量已包含输入ready、WGMMA、槽与CTA控制，不能再叠加R00/R03/R05成本。校准残差最大约0.105%。本规则只检验该整段生命周期的新长度外推，不证明独立原子服务能任意组合。

预测K=38为69154 cycle/CTA，在38测量前以只创建文件保存，后续核对GPU UUID、probe源码、目标kernel完整SASS与预测SHA256。3进程实测69119–69228，中位69122；预测误差+0.0463%，完整输出CPU误差0。38不是原已知检查点，未参与校准。

原12个TC的7/19/47单CTA/整卡点仅按修订kernel作已知检查复测，不再称为留出。原LDS组合及完整CUTLASS未重跑。受控K/K、154统一寄存器资源不能迁移到固定CUTLASS的B Major::MN/tnspB、producer40/consumer232动态资源；完整kernel与整卡L3关系继续列为缺项。
