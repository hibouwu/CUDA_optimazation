# GH200 访问与供给规则：实验与结果

最新进展：[PLAN](PLAN.md) 的 R10/R13/R14/R15 与 V05 已完成（2026-10-07）。V05 的周期模型大体成立（关键 CTA 周期误差中位数 2.9%），但微秒预测误差中位数 10.7%、最大 31.9%，比 V04 差；事后诊断显示主要原因是换算用了固定校准频率而没用 V03 的频率规则（只换成实测频率后为 4.1%/12.0%）。初始供给和 pingpong 交接另有确定失配。[V06](V06-revised-transfer.md) 修正这三处后重新冻结预测：18 个新尺寸微秒误差中位数 5.25%、最大 19.0%（目标 5%/10%，未完全达标，但优于 V04 的 5.6%/29% 与 V05 的 10.7%/31.9%）；频率误差 ≤4.6%。剩余大误差来自 cfg_a 在 tile 行数为奇数时被 2×1 cluster 补齐的整行越界 tile，使该 cluster 主循环变慢到约 690–800 cycle/Ktile（校准尺寸未覆盖），去掉这两例为 4.5%/7.3%。[R17](R17-oob-tile.md) 定位：越界的 A TMA box 加 2×1 cluster 同时出现时，边界 cluster 两个 CTA 的第一轮主循环变慢到 600–745 cycle/Ktile（cluster 1×1 或去掉越界即消失）；加入该规则后两例误差降到 −4.7%、−6.7%（非留出检查，模型达标仍需新的留出验证）。

状态：R00–R06 默认点完成；V01 单 CTA 规则预测通过，整卡与完整 kernel 未通过，缺项为负载下频率与每 kernel 固定开销，R07–R09 补齐输入后，V02 对 11 个未测尺寸预测通过（6.0%/10.3%）；V03 的频率规则把误差降到 2.6%/5.0%；V04 迁移到同输入/输出字节的配置成功，迁移到小 tile 与 pingpong 未通过。设备 GH200 / `sm_90a`，CUDA 12.9。问题结构参考 Thor 的 [RF Write 分析](../../../../Thor_RF_Write_v6_Analysis_20260928.html)，数值与机器码不移植。

## 目标

主目标是 [workloads.yaml](../../../workloads.yaml) 的 GEMM：FP16 输入、FP32 累加、FP32 输出，Tensor Core 路径。R00-B、R01、R04、R05 与 V01 都按这条精度路径实例化；BF16 只做两点迁移对照，FP8 为扩展精度，FP32 SIMT 为辅助对照。

先用 R00 取得完整 GEMM 基线和目标 WGMMA 形状，再按选定形状测规则，最后用 V01 检查规则能否预测未参与拟合的组合和完整 kernel。

## 首轮范围与顺序

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
| R01 | 完成，37 点（每迭代展开 64 步重测） | 依赖每步：FFMA 4.27（扣循环约 4.00）、add 4.98、LEA+LDS 29.0、global 小/大工作集 288/630、`ldmatrix.x4` 34、WGMMA+wait0 N=64/128/256 为 76/115/195（≈36+0.62N）；首版 29 cycle 为循环开销，已作废 | [R01](R01-readiness.md) |
| R06 | 完成，6 点（重测） | 63/95/128 寄存器：单 CTA 241 FLOP/cycle，整卡 65.1/64.7/64.1 TFLOP/s；窗口内 clock64/globaltimer 比值约 1.97–1.98 GHz。寄存器在此范围几乎不影响 FFMA 服务 | [R06](R06-issue-residency.md) |
| R04 | WGMMA 部分完成；LDS 部分不可用 | WGMMA 与另一 warpgroup 的 FFMA 交错，只比较大一方多 0.3–3.8%；LDS+FFMA/CVT 修订版串行与交错代码组织不同，不作规则 | [R04](R04-joint-service.md) |
| V01 | 部分通过 | 异步目标组合主循环 1026 cycle/Ktile（理想 1024），单 CTA 预测误差 −0.4%~−1.7%；整卡 −7%~−10%（频率 1.66–1.79 GHz，假设 1.83）；完整 CUTLASS −23%~−28%（规则未含每 kernel 固定开销与完整 kernel 的组合成本） | [V01](V01-validation.md) |
| R07 | 完成（另一张卡 GPU-009a8880） | `NDEBUG` 使 CUTLASS 快 5.5–8%（8192³ 621→657 TFLOP/s，同次 cuBLASLt 695）；持续负载频率 2048³ 约 1.75、8192³ 1.40–1.45 GHz，98–99% 样本只有 SW Power Cap，模块功率上限 680 W（证据指向功率预算，未独立证明唯一原因）；短调用的调用后探针 1.93–1.98 GHz；M=N=2048 单调用 T≈9.45+0.637·Ktile µs（截距拆分只是协议/形状差值） | [R07](R07-anchor-clock-fixedcost.md) |
| R08 | 完成（GPU-3953fe72…） | 波次离散计费：K=4096 时 T≈47.1 µs×⌈tile/132⌉（残差 2 µs；小数波模型 16 µs），超出整波 1 个 cluster 对即多一整轮；奇数 tiles_m 被 2×1 cluster 补齐；8192²×4096 时 swizzle 8 比 1 快 6–8%，2×1 比 1×1 快 1–5%，cycle 换算依赖调用后频率，需按 R09 重核 | [R08](R08-waves-l2-reuse.md) |
| R09 | 完成（GPU-e403ae62…） | 调用内平均频率：2048²×512 约 1.82、×8192 约 1.65、8192³ 约 1.40 GHz，明显低于调用后探针（1.84–1.99）；主循环 1024 cycle/Ktile+约 185（计算下界）；M=N=2048 截距：主机间隙 3.9、预填 2.2、epilogue 2.9、store 后+尾部 0.8 µs，各段之和与 R07 拟合差 −0.2 µs、斜率 −0.6%；插桩使总时间慢 0.7–4.1% | [R09](R09-inkernel-clock-stages.md) |
| V02 | 通过 | 按 R08/R09 规则预测 11 个未测尺寸的完整 CUTLASS 时间，先冻结（SHA f37a5255…）后测量：误差中位数 6.0%、最大 10.3%（目标 ≤10%/≤20%）。轮数、每 CTA tile 数全部预测正确；把频率换成实测值后误差在 ±2% 内，剩余误差几乎全来自调用内频率规则 | [V02](V02-kernel-prediction.md) |
| V03 | 通过 | 新频率规则 f=2.229−0.0353·φ·ln(W/µs)−0.0859·D−0.387·μ GHz（φ 活跃 SM 比例、D 估计 DRAM 流量 TB/s、μ 主循环占比），26 个校准点拟合；11 个新尺寸先冻结后测量：误差中位数 2.6%、最大 5.0%（目标 ≤5%/≤10%）。常数只对校准用的卡有效，另一张卡上最大误差 7.7%；剩余误差主要是主机间隙（本卡 5.3–6.3 µs，模型 3.9） | [V03](V03-clock-rule.md) |
| V04 | 部分通过 | 迁移到 3 个配置（先冻结后测量，22 点，轮数全部正确）：256×128 cooperative 误差中位数 4.1%/最大 5.4%；128×128 cooperative 5.9%/13.7%；128×128 pingpong 14.8%/29.2%。失效假设：R05-D 的 55 B/cycle 不是供给上限（实测约 64）；每 tile 固定段约 1500 cycle 不随输出缩小；pingpong 的 epilogue 只在长 K 被隐藏 | [V04](V04-config-transfer.md) |
| R02 | 条件扩展，未触发 | RF 供给与结果消费 | [R02](R02-register-service.md) |

### PLAN 补测结果（2026-10-07）

| 组 | 针对的 V04 失效 | 结论 |
|---|---|---|
| [R10](R10-layout-cache.md) | cfg_b 行距 6000 B 时主循环变慢 | 确认：cfg_b 行距改为 6016/6144 B 后完整时间 76.3 → 57.1–58.3 µs（快约 24%）；cfg_a 约 5%，cfg_c 基本不变。按配置使用，不能推广为统一规则 |
| [R13](R13-async-retirement.md) | 单 SM 供给上限 55 vs 64 B/cycle | 未回答原问题：真实 WGMMA 消费下 32 KiB tile、4 stage 为 58 B/cycle，48 KiB tile 为 45，只在本探针条件下成立。V05 的主循环误差中位数 0.9%，说明 cfg_a/b 主循环不受供给限制，这个问题对预测已不关键 |
| [R15](R15-output-service.md) | 小 tile 每 tile 约 1500 cycle 固定段 | 未解释：探针用标量 STS，CUTLASS epilogue 用 STSM、向量化与寄存器重分配，结果不能直接作为 CUTLASS 输出常数 |
| [R14](R14-stage-handoff.md) | pingpong 短 K 交接 | 提供 4 CTA 条件下逐 tile 的供给、主循环、交接、输出事件，供 V05 校准；4 CTA 的初始供给不能迁移到整卡（见 V05） |
| [V05](V05-rule-transfer.md) | 迁移验证 | 周期：主循环 0.9%/5.5%、关键 CTA 2.9%/12.7%；初始供给 14 例全部偏差 49–61%；pingpong 交接把重叠预测成间隙。微秒：10.7%/31.9%（固定频率换算所致，见上） |
| [V06](V06-revised-transfer.md) | V05 的三处失配 | 频率用 V03 规则（本卡只重拟合常数项与 D 系数）、初始供给与输出段在整卡 CUTLASS 上校准、逐 CTA 事件递推：微秒 5.25%/19.0%，周期 3.8%/22.5%，频率 2.3%/4.6%；pingpong 交接实测 −271～+137 cycle（跨零），递推能区分重叠与正间隙。未解决：cfg_a 奇数 tile 行的越界补齐使主循环变慢，cfg_b 最慢 CTA 的离散（独立问题）尚无规则 |
| [R17](R17-oob-tile.md) | cfg_a 边界 tile 变慢 | 需同时满足：A 的 TMA box 越过 M 边界、cluster 2×1；只影响第一轮。边界 cluster 两个 CTA 同样变慢（671–707 cycle/Ktile，偶数行部分越界 596），同列其他 CTA 也慢 40–60。cluster 1×1 或改为偶数行无越界时消失。机制未分离（推断与同列共享 B 面板的步调有关） |

R02、R11、R12、R16 为条件扩展，启用条件见 [PLAN](PLAN.md#5-条件扩展默认不做)。

验收看四项：SASS 中目标指令序列与动态次数正确、输出经 CPU 校验、样本稳定、数值与已知物理量级相符（如 FFMA 依赖约 4 cycle、WGMMA 峰值 4096 FLOP/cycle）。前三项保证测得对，第四项保证测的是想测的量：R01 首版正是三项都通过、只在第四项暴露循环开销问题。2026-10-06 的协调记录移到[归档](../../gh200_sm90_archive/access_rules/)。

## 文件结构

```text
Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/
  README.md、PLAN.md                 # 导航/状态与本轮唯一实施计划
  R00–R09、V01–V04                  # 已有实验说明（R02 为条件扩展）
  R10、R13、R14、R15、V05            # 本轮结果与复现

microbench/gh200_resource_campaign/access_rules/
  run_r00.py ... run_r06.py、run_v01.py              # 每组一个入口：配置、短检查、校准、预热、采样
  analyze.py（R00）、analyze_r01.py ... analyze_v01.py  # CPU 重算、报告、图
  probes/r00_*.cu/.hpp、r01.cu ... r06.cu、v01.cu      # r00_common.hpp 为公共头文件
  run_r10.py、run_r13.py、run_r14.py、run_r15.py       # 本轮有限矩阵
  run_v05.py、v05_calibrate.py、v05_predict.py、analyze_v05.py
  probes/gaps_common.hpp、r10/r13/r14/r15/v05 探针
  configs/、.clang-format

results/gh200_resource_campaign/access_rules/<本轮run-id>/
  source/、build/（编译命令、SASS、二进制）、environment.json
  samples/、cases.csv、summary.json、analysis-*/、reviews/
  source/保留编译时版本，旧归档路径继续保留
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
| R02 物理寄存器布局与控制字段诊断 | 合法序列的源供给/结果消费已纳入本轮；机器码诊断在出现未解释的稳定差异时另行启动 |
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
