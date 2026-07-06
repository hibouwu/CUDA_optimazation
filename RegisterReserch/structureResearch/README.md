# Thor（SM110）寄存器操作数路径研究

本目录用 PTX 基线和 patched SASS 微基准研究 Thor 上标量指令读取物理寄存器时的可见行为。当前最强结论是：

> 对已测试的 `LOP3`、寄存器范围和单 warp 条件，寄存器读取路径表现为两个按物理寄存器编号奇偶划分的**有效服务组**。三个同奇偶源操作数比混合奇偶源操作数多约一个周期。

这不等价于“已经证明寄存器文件由两个物理 SRAM bank 构成”。计时无法区分寄存器阵列、读端口、operand collector 或它们组合造成的限制。完整证据和各实验是否必要见 [EVIDENCE_SUMMARY.md](EVIDENCE_SUMMARY.md)。

## 当前可复现的主实验

正式链路是 `LOP3` patched-SASS bank scan：

```bash
cd CUDA_optimazation/RegisterReserch/structureResearch
CUDA_ARCH=110 ./scripts/build.sh
ITERS=20000 WARMUPS=3 REPEATS=10 ./scripts/run_bank_scan.sh
```

`run_bank_scan.sh` 会：

1. 编译 `sass_lop3_template.cu` 和 CUDA Driver API 测量程序。
2. 把定时区内 128 条 `LOP3` 的物理寄存器字段改成目标 tuple。
3. 清除 `.reuse` 位，并用 `nvdisasm` 逐条回读验证。
4. 轮换执行全部 cubin，输出 CSV。
5. 生成汇总图。

输出位于：

```text
results/bank_scan/manifest.csv
results/bank_scan/results.csv
assets/register_bank_stride_scan.png
```

`build/` 和 `results/bank_scan/` 默认被忽略，不会提交生成的 cubin 和原始 CSV。若结果用于报告，应同时保存运行环境、CSV 和 SASS 验证信息。

## LOP3 用例组成

每个被 patch 的定时指令可表示为：

```text
LOP3 Rbase, R(base + stride), R(base + 2 * stride), Rbase
```

共有 87 个 cubin：

| 组别 | 数量 | 作用 |
|---|---:|---|
| `B0`–`B3` source-count controls | 4 | 区分 1、2、3 次 RF 读取，并比较混合/同奇偶源 |
| `M0`–`M2` slot permutations | 3 | 排除某个固定 source slot 导致的差异 |
| `L_bXX_sYY` single-chain scan | 64 | `R4`–`R7` 四个 base × stride 1–16，暴露依赖延迟 |
| `T_sYY` four-chain scan | 16 | 检查独立指令是否能隐藏同一延迟 |

模板初始化并使用的物理寄存器范围是 `R4`–`R39`。因此当前结果不能直接外推到全部 255 个线程寄存器编号。

## 已保存的结果

已保存的 Thor 数据显示：

| RF 源操作数模式 | Median cycles/LOP3 |
|---|---:|
| 1 次 RF 读取 | 2.086031 |
| 2 次读取、同奇偶 | 2.086031 |
| 3 次读取、混合奇偶 | 2.086031 |
| 3 次读取、全部同奇偶 | 3.070406 |

在 `R4`–`R7` 四个 base 上，奇数 stride 都约为 `2.086` cycles/op，偶数 stride 都约为 `3.070` cycles/op。可见增量约为 `0.984` cycles/op。

![Thor LOP3 physical-register stride scan](assets/register_bank_stride_scan.png)

正确表述是：

```text
tested effective group = physical_register_id % 2
```

不应写成：

```text
physical SRAM bank = physical_register_id % 2
```

## 其他实验和文件状态

### PTX IMAD 基线

`src/register_bench.cu` 包含五个 PTX 层用例：

| Case | 用途 |
|---|---|
| `R0_imad_chain` | 依赖链延迟 |
| `R1_imad_independent_x4` | 四条独立链吞吐 |
| `R2_reuse_hot_x4` | 重复源和 `.reuse` 行为 |
| `R3_bank_dense_x4` | 密集虚拟源布局 |
| `R4_bank_sparse_x4` | 稀疏虚拟源布局 |

可直接运行：

```bash
./scripts/build.sh
./build/register_bench --case all --iters 100000 --warmups 5 --repeats 20
```

这些用例适合作为延迟、吞吐和编译器行为基线，但 PTX 寄存器是虚拟寄存器，`ptxas` 可以重新分配物理编号，所以它们不是 bank 映射的核心证据。

### 其他 SASS 模板

`sass_template.cu`、`sass_imad_template.cu` 和 `sass_fma_template.cu` 当前会被 CMake 编译，但 `run_bank_scan.sh` 只 patch 和运行 `sass_lop3_template.cu`。因此 IMAD/FMA 文件目前是后续跨指令验证的脚手架，不属于自动化结果链路。

`assets/register_experiment_results.png` 是早期 IMAD 试验的保留图。当前目录缺少生成该图所对应的 patched-IMAD 脚本和原始 CSV，所以它只能作为历史记录，不能作为当前可复现实验的主证据。

### NCU/CUPTI 探索代码

`run_ncu_analysis.py`、`test_profiler.cu` 和 `src/cupti_rf_bank_profiler.cu` 是探索性原型，未接入 CMake 主实验和 `run_bank_scan.sh`。当前也没有已验证的、可直接报告寄存器 bank 冲突数的 Thor 计数器，因此 NCU/CUPTI 最多提供旁证。

### 分析脚本

`scripts/plot_bank_scan.py` 是当前结果图的正式生成脚本。其余 `analyze_*`、`sass_register_port_analysis.py` 和 `plot_stride_periodicity.py` 是早期离线分析；其中部分把有效奇偶行为过度解释成物理 bank 数，不能作为最终结论来源。

## 结果解释边界

- patched cubin 使用未公开且可能变化的指令编码；换 CUDA Toolkit 或架构后必须重新执行反汇编验证。
- `.reuse` 已从被测 `LOP3` 中清除，但仍不能单独定位限制发生在 RF、端口还是 collector。
- 64 个 base × stride 点是系统化扫描点，不是 64 个相互独立的架构证明，不能据此直接给出“99.99% 物理 2-bank 置信度”。
- 单链延迟结果和四链吞吐结果回答不同问题；后者可能通过指令级并行隐藏前者。
- 结论只覆盖当前指令、寄存器范围、单 warp 和当前工具链。

## 目录中的已知冗余

以下文件没有进入当前正式链路，后续可以单独归档或删除，但本次文档合并不修改实验代码：

- `check_cap`：已提交的 ARM 可执行文件。
- `bank_scan_extended.log`：一次因访问未初始化 `R40` 而失败的旧日志。
- `check_capability.cu`：包含“typical”硬编码值，不能用于推断 Thor bank 结构。
- `run_ncu_analysis.py`、`test_profiler.cu`、`src/cupti_rf_bank_profiler.cu`：未集成的 profiler 原型。
- 多个早期 `analyze_*` 脚本：与正式 plot 脚本重复，且结论口径不一致。

## 参考资料

- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
- [CUDA Programming Guide：Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [CUDA Programming Guide：On-chip Register File](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
