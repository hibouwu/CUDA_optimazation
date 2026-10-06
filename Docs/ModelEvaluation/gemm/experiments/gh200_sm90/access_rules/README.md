# GH200 访问与供给规则：实验计划

状态：R00、R01、R03、R05、R06 已完成；V01、R04 跨 warpgroup 部分改用异步 WGMMA 重做中（见下方“当前状态”）。设备 GH200 / `sm_90a`，CUDA 12.9。问题结构参考 Thor 的 [RF Write 分析](../../../../Thor_RF_Write_v6_Analysis_20260928.html)，数值与机器码不移植。

## 目标

主目标是 [workloads.yaml](../../../workloads.yaml) 的 GEMM：FP16 输入、FP32 累加、FP32 输出，Tensor Core 路径。R00-B、R01、R04、R05 与 V01 都按这条精度路径实例化；BF16 只做两点迁移对照，FP8 为扩展精度，FP32 SIMT 为辅助对照。

先用 R00 取得完整 GEMM 基线和目标 WGMMA 形状，再按选定形状测规则，最后用 V01 检查规则能否预测未参与拟合的组合和完整 kernel。

## 范围与顺序

| 顺序 | 实验 | 回答的问题 | 默认配置数 |
|---|---|---|---:|
| 1 | [R00：完整 GEMM 基线与目标 WGMMA 形状](R00-anchor-target.md) | 实际 GEMM 能到多少？`m64n256`、2 个 warpgroup 能否达到峰值？ | 30 |
| 2 | [R01：依赖与结果可用](R01-readiness.md) | 依赖链每步多长？WGMMA 依赖时间随 N 怎样变化？ | 37 |
| 3 | [R05：异步完成与缓冲复用](R05-async-lifecycle.md) | 等待、消费、复用怎样组合成完成时间？目标在途量下每 SM 能拿到多少输入？ | 40 |
| 4 | [R04：跨路径联合服务](R04-joint-service.md) | WGMMA 与 FFMA、LDS 与 FFMA/CVT 同时执行能否各自保持速率？ | 32 |
| 5 | [R03：片上访问并发](R03-access-demand.md) | SMEM、`ldmatrix`、`stmatrix` 在几个 warp 时饱和？ | 18 |
| 6 | [R06：资源与并发](R06-issue-residency.md) | 寄存器分配与多 CTA 驻留怎样改变 FFMA 服务？ | 6 |
| 穿插 | [V01：留出组合验证](V01-validation.md) | 规则能否预测新序列和完整 kernel？ | 20 |
| 条件 | [R02：RF 操作数与结果供给](R02-register-service.md) | 出现现有规则解释不了的差别时启用 | 0 |

默认 163 点（规则 133、R00 30），V01 另 20 点。计时控制、校准和短正确性检查另计。

不必等 163 点全部完成：先做 R00，再做 R01 并配一项 V01 检查，之后逐组推进。每组扩大矩阵前先跑一对基线/对照，确认能分辨目标差异。R00-B 的结果决定 R01-C、R04、R05 使用的 WGMMA 形状；R04 需要的 R03 代表点可以提前测。

## 当前状态（2026-10-07）

| 组 | 状态 | 主要结果 | 数据 |
|---|---|---|---|
| R00 | 完成，30 点 | WGMMA `m64n{64,128,256}k16`、1/2 个 warpgroup、SS/RS 均 4095 FLOP/cycle，整卡 FP16 989 TFLOP/s、FP8 1978；cuBLASLt FP16 8192³ 715 TFLOP/s、2048³ 555；固定 CUTLASS 8192³ 643 | [R00](R00-anchor-target.md) |
| R03 | 完成，18 点 | 8 warp 时 LDS.128 124、STS.128 125、`ldmatrix.x4` 127、`stmatrix.x4` 116 B/cycle/CTA | [R03](R03-access-demand.md) |
| R05 | 完成，40 点 | 目标在途量 1 CTA/SM：共享小源 55.5 B/cycle、独立大源 15.3（满速需 48）；TMA 写行距 144 B 慢 38%，160/256 B 正常 | [R05](R05-async-lifecycle.md) |
| R01 | 完成，37 点（每迭代展开 64 步重测） | 依赖每步：FFMA 4.27（扣循环 4.00）、add 4.98、LEA+LDS 29.0、global L2/HBM 288/630、`ldmatrix.x4` 34、WGMMA+wait0 N=64/128/256 为 76/115/195（≈36+0.62N）；首版 29 cycle 为循环开销，已作废 | [R01](R01-readiness.md) |
| R06 | 完成，6 点（重测） | 63/95/128 寄存器：单 CTA 241 FLOP/cycle，整卡 65.1/64.7/64.1 TFLOP/s；窗口内 SM 频率约 1.97–1.98 GHz。寄存器在此范围几乎不影响 FFMA 服务 | [R06](R06-issue-residency.md) |
| R04 | 部分重做 | 同 warpgroup WGMMA+FFMA 16 点可用；LDS/CVT 修订版完成；跨 warpgroup 7 点因编译器逐条串行（C7520），改用异步 WGMMA 重做 | [R04](R04-joint-service.md) |
| V01 | 重做 | 旧目标组合被逐条 wait0 串行，且没有基于规则的预测；改用异步流水，先保存预测再测 | [V01](V01-validation.md) |
| R02 | 未触发 | — | [R02](R02-register-service.md) |

验收看四项：SASS 中目标指令序列正确、输出经 CPU 校验、样本稳定、数值与已知物理量级相符（例如 FFMA 依赖约 4 cycle、128 线程 FFMA 约 230 FLOP/cycle、WGMMA 峰值 4096）。2026-10-06 的协调记录（分工、STATUS、ACCEPTANCE）移到 [归档](../../gh200_sm90_archive/access_rules/)。

## 文件结构

```text
Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/
  README.md、R00–R06、V01            # 各组一页：问题、矩阵、工作量与边界、正确性、输出

microbench/gh200_resource_campaign/access_rules/
  run_r00.py ... run_r06.py、run_v01.py              # 每组一个入口：配置、短检查、校准、预热、采样
  analyze.py（R00）、analyze_r01.py ... analyze_v01.py  # CPU 重算、报告、图
  probes/r00_*.cu/.hpp、r01.cu ... r06.cu、v01.cu      # r00_common.hpp 为公共头文件
  configs/、.clang-format

results/gh200_resource_campaign/access_rules/<run-id>/<组>/
  source/、build/（编译命令、SASS、二进制）、environment.json
  samples.jsonl、cases.csv、rules.json、report.md、plots/
```

每组结果取得后，在对应计划文件末尾补“实测结果”或链接 `report.md`。

## 统一计量与运行

- **计时**：单 CTA 用同一 CTA 的 `clock64` 差；整卡用所有 CTA 的 `globaltimer` 包络，不跨 SM 相减 `clock64`。记录实际 SM ID。
- **工作量**：thread 操作数、warp 指令数、FLOP/OP、逻辑读写字节分别记录；按实际 SASS 解释宽指令。
- **结束事件**：每个样本写明窗口在哪个事件结束（序列推进、消费者完成、数据可用、源缓冲可复用、完整输出完成）。
- **采样**：单 CTA `clock64` 配置 3 个独立进程；CV>1%，或配对差接近样本波动或计时分辨率时，补到 10 个再解释。整卡与 R00-A 10 个进程。预热 8–30 次窗口，末 5 次 CV≤2%。固定种子打乱顺序，配对点相邻运行。
- **长度**：同一对照内的配置用共同循环长度，由短校准选定并记录。
- **流程**：先对代表点做 SASS 与短正确性检查，再跑默认矩阵；完整采样后直接出报告。CV>5% 时至多补两批，全部保留。
- **分配**：一份单 GPU Slurm 分配内串行运行，时长由短校准估计。NCU 单独运行，权限被拒只记录一次。
- **命名**：结果按测到的量命名（每步增量、工作率、配对差、完成窗口）。物理部件归因（bank、端口、cache 层级、HBM 流量、裸延迟）需要额外证据，没有时写在报告的“解释”里并标为推断。

拟实现用法：

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r00.py \
  --output results/gh200_resource_campaign/access_rules/NEW/r00
python3 microbench/gh200_resource_campaign/access_rules/analyze.py \
  --input results/gh200_resource_campaign/access_rules/NEW/r00
```

新采样用新输出目录；续跑同一组时核对设备、源码、配置与已有样本身份即可。

## 规则输出

| 字段 | 内容 |
|---|---|
| 条件 | 设备、工具链、实际指令、参与者、输入/地址、资源、缓存准备与背景流 |
| 工作与边界 | 计数与单位；起点、结束事件，循环与同步是否在窗口内 |
| 结果 | 原始样本、统计、拟合残差或配对差 |
| 规则形式 | 条件查表、依赖链增量、联合服务曲线、完成/释放关系或负结果 |
| 适用域 | 已测坐标与验证点 |

规则状态：观测 / 预测检查通过 / 不稳定 / 未取得该能力。不同 GPU UUID、工具链或源码的样本不共同拟合。接入模型时沿用[整段或分段计账](../../../model/interfaces.md#5-计时与层间边界)，窗口已含的成本不再叠加。

## 默认范围之外

| 项目 | 启用条件 |
|---|---|
| R02 源供给 / 结果流 | mainloop、epilogue、RS 取数或 FP8 提升出现现有规则解释不了的稳定差别；两部分可分别启用 |
| R03 普通 global 地址需求 | 方案在 mainloop 或 epilogue 使用普通 global 访问，且 V01 相应路径失配 |
| R04 背景位置、LDG+FFMA、真实 FP8 分段累加 | 默认配对出现可重复差异；或 FP8 方案需要分段提升 |
| R06-A 循环展开与代码组织 | kernel 出现与展开、代码大小相关的未解释差别 |
| FP64 新 MMA 形状（R00 条件扩展，4 点） | 需要解释 EXP-06 FP64 只到产品值一半 |
| BF16 依赖与联合服务 | R00-B 的 BF16 迁移对照与 FP16 不一致 |
| 方案的实际 WGMMA/tile 组合 | 选定方案与 R00-B、V01 的形状不同 |
| 缓存替换、TLB/页、地址分区 | 工作集或地址变化出现未解释拐点 |
| DSM/cluster/多播竞争 | 方案使用 cluster；复用 EXP-17 并补对应组合 |
| `.param/.const`、原子、特殊函数、`setmaxnreg` 成本 | 方案使用且成为瓶颈 |
| Thor TMEM/TCGen05；稀疏、Batched/Grouped、多 GPU | 另立计划 / 问题范围改变时 |

## 来源

- [Thor RF Write v6.2](../../../../Thor_RF_Write_v6_Analysis_20260928.html)：只采用问题结构。
- [PTX ISA 8.8 / CUDA 12.9.1](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)：指令、等待、代理与复用合法性。
- [Nsight Compute：Quantities](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#quantities)：instruction、request、sector、wavefront 的区别。
- [已测 GH200 结果](../README.md)：条件匹配时优先复用。
