# GH200 L0：FMA / MMA

新一轮独立复审见 [EXP-01](../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/EXP-01-compute-audit.md)；后续测试由 [gh200_resource_campaign](../gh200_resource_campaign/README.md)管理，旧源码和目录保留。

当前审查结论见 [测试过程审查](AUDIT.md)，代表配置的最新证据见 [审查复测报告](results/20260930-audit/REPORT.md)。下述首批结果保留为探索性记录，不能将固定参考周期或最优测试点提升为无条件硬件上限。

在 SM90a 上测量指定指令形式、依赖条件和操作数供给条件下的单 CTA 计算服务。首批实测见 [报告](results/20260930-initial/REPORT.md) 和 [结构化结果](results/20260930-initial/summary.json)。

持续发射与整卡并发测试见 [压测报告](results/20260930-sustained/REPORT.md) 和 [完整统计](results/20260930-sustained/summary.json)。这批新增多 warp/warpgroup、全 GPU 配置，以及 WGMMA `wait_group 0/3/7` 对照。

报告统一使用周期：单 CTA 为实测 `clock64` 周期；整卡为固定 1.98 GHz 下的参考周期，单位 `FLOP/reference cycle/GPU`。参考频率不表示设备已锁频，原始时间戳保留供审计。

## 计算路径与数量口径

当前稠密浮点 GEMM 主要区分三条路径：逐线程 `fma`、warp 级 `mma.sync`、warpgroup 级 `wgmma.mma_async`。这是建模路径分类，不是 GH200 全部算术指令的数量；PTX 指令也不必与 SASS 一一对应。`wmma.mma` 是另一种 PTX 矩阵接口，不能另算一套 Tensor Core 计算资源。

| 路径 | GH200 上相关浮点类型 | 首批测试 |
|---|---|---|
| FMA | FP32、FP64、FP16、FP16×2、BF16、BF16×2 六种基础类型形式 | 六种均测，`.rn`，不含 sat/relu/ftz/oob 等修饰符扫描 |
| `mma.sync` | FP16、BF16、TF32、FP64、FP8 E4M3/E5M2，合法形状和累加类型依形式而定 | FP16/BF16 M16N8K16→FP32；TF32 M16N8K8→FP32；FP64 M8N8K4→FP64 |
| `wgmma.mma_async` | FP16、BF16、TF32、FP8 E4M3/E5M2；没有 FP64 WGMMA | FP16/BF16 M64N64K16→FP32，A/B 均来自 SMEM |

例如，dense FP16/BF16 WGMMA 的 M=64、K=16，N 可为 8 到 256 的 8 的倍数，单看形状就有 32 种。再区分累加类型、A 的来源、布局及其他修饰符，数量会继续增加。因此“支持多少种”应先指定计数维度；首批 12 个指令配置不是全部合法形式。

整数/位运算 MMA、稀疏 MMA、普通 add/mul/div、特殊函数、DPX 等不属于本次测试范围。GH200 不具有 Thor 的 `tcgen05` 指令路径；FP6/FP4 不能从新版本 PTX 文档的通用列表推定为 GH200 原生计算能力。

依据：[FMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fma)、[半精度 FMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-fma)、[mma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)、[WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions)。使用各节 Target ISA Notes 限定 SM90/SM90a，不能把整个最新 PTX 指令集当作 Hopper 指令集。

## 实验配置

- 6 个 FMA 配置、4 个 `mma.sync` 配置、2 个 WGMMA 配置。
- FMA 和 `mma.sync`：一个 warp，一个 CTA；1 或 8 条独立累加链，循环内每链展开 16 次。
- WGMMA：一个 warpgroup，一个 CTA；1 或 2 条独立累加链，每链每批 1/4/16 条指令。每批 commit 后 wait_group 0；不是跨 batch 多 group 流水。
- 32 个编译配置，循环次数 128/512/2048；每点 2 次预热、7 次记录。
- FMA 使用 `d = 0.5*d + 0.25`；MMA 使用均为 1/16 的有限操作数，检查所有输出。输入较简单，仅验证本探针的数值执行，不代表任意数据和布局的完整验证。
- 计时排除初始化与结果 GMEM 写出，包含循环控制，以及结束处的结果归约、SMEM 捕获和 CTA barrier。WGMMA 包含 batch 提交、等待和编译器必要的同步。
- 用循环长度拟合周期斜率，保存截距、样本内 bootstrap 区间、SASS 和源码 hash。当前没有空循环扣除或独立硬件发射间隔辨识。

这些结果支持 L0 的条件化经验服务模型；不代表 SM/整卡饱和能力，也不直接构成严格时间下界。单 CTA、单 warp/warpgroup 的结果不能乘以 SM 数作为整卡预测。

## 生成、运行与分析

```bash
python3 microbench/gh200_l0/generate.py
```

把生成的 `probe.cu` 与 `run_remote.sh` 复制到 ROMEO 的新实验目录。在分配的 GH200 节点上，通过登录 shell 执行：

```bash
# 替换为新实验目录；不要覆盖已归档运行。
source /path/to/new-run/run_remote.sh /path/to/new-run
```

runner 使用 CUDA 12.9 和 `-gencode arch=compute_90a,code=sm_90a`，保留编译、SASS、设备和原始输出。现有 `raw.jsonl` 会使 runner 拒绝覆盖。分配单 GPU 的 Slurm 方法见 [访问说明](../../docs/romeo_gh200_access.md)。

将远程目录取回后分析：

```bash
python3 microbench/gh200_l0/analyze.py microbench/gh200_l0/results/20260930-initial
```

分析器要求完整的 672 条记录、每个配置各循环长度的 7 个不同 repeat、数值检查成功，以及各 kernel 对应的目标 SASS 指令。当前 `cases.json` 必须与被分析的生成源码对应；本次源码与远程归档已比较 SHA-256。

首批结果来自 2026-09-30 的 `romeo-a057`，132 SM、CUDA 12.9.41、驱动 590.48.01。未修改时钟；原运行只有一条遥测样本，不能证明计时期间锁频或频率稳定。当前 runner 已把后续遥测间隔改为 100 ms；归档中的旧 runner 保留原状。

## 持续发射压测

```bash
python3 microbench/gh200_l0/generate_stress.py
```

生成器读取当前 `probe.cu` 与 `cases.json`，在 `stress_cases.json` 记录原源码 hash；不覆盖首批源码与归档。生成 42 个 kernel，分别在单 CTA 和全 GPU 范围测量，合计 84 组、每组 5 次正式运行。

- FMA / `mma.sync`：8 条独立累加链，每链每轮 16 条，32/128/256 线程。
- WGMMA：2 条独立链，每链每组 16 条，128/256 线程；每组 commit 后分别 wait 0、3 或 7，最终 wait 0 后才读取结果。
- 每次 kernel 先用短探针估算循环长度，目标约 100 ms。矩阵循环最多 65536 次，以保证当前精确测试数据不会因长期 FP32 累加失去精度；实际时长由结果给出。
- 全 GPU grid 为 SM 数 × min(4, occupancy API 的 CTA/SM 上限)。逐 CTA 记录 SM ID、周期和全局时间戳，按整张 grid 的时间窗口计算吞吐。
- 单 GPU Slurm 作业限额 5 分钟，测试程序限额 220 秒；采集 100 ms 间隔遥测，不修改频率或功率。

将 `stress.cu`、`stress_cases.json`、`run_stress_remote.sh` 放进新的远程实验目录，在分配的 GH200 节点登录 shell 中执行：

```bash
source /path/to/new-run/run_stress_remote.sh /path/to/new-run
```

取回结果后分析：

```bash
python3 microbench/gh200_l0/analyze_stress.py microbench/gh200_l0/results/20260930-sustained
```

统计脚本读取结果目录中归档的 manifest，重算每次工作量、全局时间窗口和 FLOP/s，核验 SM 覆盖、重复次数与目标 SASS 指令。报告中的最佳配置只是已测配置内的最高中位数；不将其写成物理峰值或通用模型的无条件上限。

`analyze_stress.py` 使用 `--reference-clock-hz 1980000000` 作为默认整卡周期归一化频率。`summary.json` 的 `cycle_basis` 区分实际局部周期与参考周期，`flop_per_cycle_median` 和 `timed_cycles_median` 为周期口径字段。
