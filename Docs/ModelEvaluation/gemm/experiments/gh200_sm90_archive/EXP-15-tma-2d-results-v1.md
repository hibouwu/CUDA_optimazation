# GH200 TMA 2D：第一组 56 配置的完成服务

本组 56 个自身 pilot 和 560 个正式进程已通过独立数据 C 与发布复审，56 条条件参数已发布。剩余 12 个全 GPU padding 配置未纳入本组，整个 S15 家族尚未完成。

## 测量什么

给定二维 UINT16 payload、GMEM 行跨度、SMEM 布局和 CTA 数，反复执行 TMA 并等待完整完成，测量搬运服务。GMEM→SMEM 包含 thread0 发起、mbarrier 完成等待及 CTA 会合；SMEM→GMEM 包含 store、commit、完整 wait0 及 CTA 会合。

本组覆盖两个方向全部 34 个单 CTA 配置，以及两个方向 22 个全 GPU continuous-none/SW128 配置。payload 是 1/4/8/16/32/64 KiB；SW128 只使用 1/4/8/16/32 KiB 合法子集。输入为非均匀 uint16 模式，32 个全局槽专属各 CTA。完整槽、最终 logical/physical tile、padding、guard 和完成记录均保留。

## 计量和真实手算

有效 payload Q=2WH；计时内完成量 X=BNQ。padding 占用地址空间，但不计入 Q。单 CTA 的分母是同一 CTA 的 clock64 差；全 GPU 的分母是所有 CTA 的 globaltimer 包络，不跨 SM 相减 clock64。

单 CTA 的真实第 0 次进程：B=1，N=13614，Q=1024 B，所以 X=13940736 B；Δcycle=7923050，得到 1.759516348 B_transport/clock64_cycle/CTA。该点十次有效进程的中位数为 1.742041382。最后导出两份 tile 的 2048 B 位于停止时间戳之后，不加入 X。raw SHA：`57c0b407e5427f7ca0df32cf9e41083214110c9ad317fed8d0f75a9d42935d55`。

全 GPU 同类样本：B=528，N=17985，Q=1024 B，X=9723985920 B；包络为 5559488 ns，得到 1749.079397239 GB_transport/s/GPU。B/ns 的数值与十进制 GB/s 一致。raw SHA：`b31f3d1c73bbf35993895ea4abcaf66bff8abab9a523ca2282d5d9e56ba50956`。

CUDA event 覆盖整个 kernel，包括初始化和计时后导出；它用于原校准及一致性核对，不替代主循环指标。图中误差条为十个有效独立进程的样本标准差。

## 实测

所有 56 点在第一批十进程后稳定，最大 CV 约 1.58%。原协议为 8–30 个预热窗口、末五窗口 CV≤2%，正式 CV≤5%，最多三批、保留全部样本。本组没有追加第二、第三批。

![单 CTA 完成搬运服务](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/published-first56-v1/one_cta.png)

![全 GPU 完成搬运服务](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/published-first56-v1/all_gpu.png)

单 CTA G2S 在大 payload 下，continuous-none 与 SW128 接近；padding-none 的完成速率较低。全 GPU 的非 padding 对照在本条件下从小 payload 增长后趋于平缓。它们是这些布局和循环的联合服务观察，不能单凭曲线判断物理端口、缓存命中或 HBM 流量。

17 个 S2G 单 CTA 配置各自十条 raw 的 clock64 差完全一致，因而对应 CV 为零；globaltimer 仍可变化。这里的零离散是实际记录的结果，不能推断重复测量没有误差。

## 条件与资格

实际设备为 GH200/SM90a，UUID `GPU-7c184a2e-41ea-3d2b-df5b-1699c95fc1fe`，CUDA 12.9。第一组由 734171 和恢复作业 734213 执行，恢复复用原 23 个 pilot 包，未重新采样；734213 最终 COMPLETED 0:0。全 GPU 本组各配置实际 CTA 数为 396、528，occupancy 上限不当作实际驻留证明。

完整归档保全 616 个 point 包及约 66.98 GB 未压缩数组。普通 ARM CPU 734264 全包重放成功，独立数据 C 核了全部 raw、工作量、预热、进程顺序和统计，另独立逐值复核四个正式代表包。734264 未记录实际 Python executable 身份，因此不称为严格同解释器证明；本机两项派生 CV 与原生结果相差 1 ULP，保留原值和原统计，不增加容差。

结果仅适用于固定 32 槽、一次请求/迭代、完整等待和 CTA 会合。参数含各点 N、B、资源、布局、输入与设备条件；不作为物理 HBM 带宽、TMA 队列深度、整卡峰值或完整 GEMM 性能。

## 数据与复现

- [逐进程 CSV](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/published-first56-v1/samples.csv)、[56 点结果表](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/published-first56-v1/results.csv)、[真实算例](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/published-first56-v1/worked-example.json)、[参数候选](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/published-first56-v1/parameter-candidates.json)。原候选文件保持 qualified=false；另行发布的[合格条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/published-first56-v1/qualified-parameters.json)和[发布依据](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/published-first56-v1/publication-bridge.json)保留原值、条件及审查来源，不回写 raw/summary。
- [完整中文表](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/published-first56-v1/RESULTS.zh.md)、[来源身份](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/published-first56-v1/sources.json)、[报告 manifest](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/published-first56-v1/report-manifest.json)。

完整本地归档是 [actual.tar](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/collection-job734213/actual.tar)，准确成员与哈希见[收集索引](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/collection-job734213/collection-index.json)。已经展开的 [actual 目录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/collection-job734213/actual)中，`repo/` 与 `formal-oneCTA_and_nonpadding/` 同级；换目录时保留该结构。已封存的普通 ARM 重放输出为 [replay.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/native-replay-first-cohort/actual/replay.json)，真实调用见 [driver.sh](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/native-replay-first-cohort/actual/driver.sh)。

在本地使用已封存重放输出生成新报告，可直接执行：

```sh
ARCHIVE_ROOT="/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/collection-job734213/actual"
REPLAY_JSON="/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/native-replay-first-cohort/actual/replay.json"
python3 -B /home/jianyeshi/Note/CUDA/CUDA_optimazation/microbench/gh200_resource_campaign/report_tma_tensor_2d.py \
  --run "$ARCHIVE_ROOT/formal-oneCTA_and_nonpadding" \
  --replay "$REPLAY_JSON" \
  --output /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/NEW_REPORT
```

输出目录必须尚不存在。生成图需要 Matplotlib；追加 `--tables-only` 可只用标准库生成表、手算与候选参数。此命令读取真实 raw 与封存统计，不执行 GPU；它不会授予发布资格。

重新完整审查数组和统计应在 ARM Python 3.9 环境执行；本组真实 CPU 作业使用 Python 3.9，但没有记录解释器 executable SHA，不能据此保证任意 Python 版本逐位一致。以下是原 ROMEO 保存目录上的普通入口；在有效 ARM CPU Slurm 分配中执行，不请求 GPU：

```sh
ARCHIVE_ROOT="/gpfs/projet/r260073/hibouwu/gh200_resource_campaign/compute_onchip/s15-formal-oneCTA-and-nonpadding-v1"
romeo_load_armgpu_env
spack load cuda@12.9.0
python3 -B "$ARCHIVE_ROOT/repo/audit_formal.py" \
  --run "$ARCHIVE_ROOT/formal-oneCTA_and_nonpadding"
```

归档迁移后将 ARCHIVE_ROOT 改为新位置。完整普通审查需要读取约 66.98 GB 未压缩数组，逐包暂时展开；本地 Python 3.14 的两项 CV 末位差异已记录，不通过改容差或改旧统计规避。日常重新生成图表优先使用前一条封存 replay 命令。

重新测量原 68 点与接续剩余 12 点分别使用 [run_s15_once.sh](../../../../../microbench/gh200_resource_campaign/runners/run_s15_once.sh) 的 all/remaining；用 `bash microbench/gh200_resource_campaign/runners/run_s15_once.sh --help` 查看准确参数。第一组结果审查不触发 GPU 重测。
