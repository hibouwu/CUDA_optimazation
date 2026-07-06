# Thor SM110 Register Operand Path

本目录用 PTX baseline 和 patched SASS 微基准研究 Thor `sm_110` 上寄存器源操作数读取路径的可见分组行为。

当前结论：

```text
tested visible service group = physical_register_id % 2
```

这表示已测试的 `LOP3`、`FFMA` 和 `IMAD` 对照用例都只暴露出按物理寄存器编号奇偶划分的两个有效服务组。它不等价于“物理 SRAM register file 已证明只有 2 个 bank”。完整证据边界见 [EVIDENCE_SUMMARY.md](EVIDENCE_SUMMARY.md)。

## 运行入口

统一入口：

```bash
cd CUDA_optimazation/RegisterReserch/structureResearch
STAGE=all OPCODES=all ./scripts/run_opcode_suite.sh
```

常用子集：

```bash
STAGE=main OPCODES=lop3,ffma ./scripts/run_opcode_suite.sh
STAGE=tuple OPCODES=lop3,imad ./scripts/run_opcode_suite.sh
STAGE=physical OPCODES=lop3,imad ./scripts/run_opcode_suite.sh
```

默认参数可用环境变量覆盖：

```bash
ITERS=5000 WARMUPS=2 REPEATS=5 STAGE=tuple OPCODES=lop3,imad ./scripts/run_opcode_suite.sh
```

## 实验组成

| Stage | 覆盖 | 作用 |
|---|---|---|
| `main` | `LOP3`、`FFMA` | 完整 source-count、slot permutation、base × stride、multi-chain scan |
| `tuple` | `LOP3`、`IMAD` | 任意三源 tuple，比较 `mod2/mod4/mod8/mod16` 解释力 |
| `physical` | `LOP3`、`IMAD` | source-slot permutation、多独立链 pressure、NCU metric 候选查询 |

`IMAD` 目前只进入 `tuple` 和 `physical` 阶段；还没有与 `LOP3/FFMA` 完全同构的完整 stride 主扫描。

## 关键脚本

| 脚本 | 作用 |
|---|---|
| `scripts/run_opcode_suite.sh` | 统一实验入口 |
| `scripts/patch_main_scan.py` | 生成并验证 `LOP3/FFMA` 主扫描 cubin |
| `scripts/plot_main_scan.py` | 生成 `LOP3/FFMA` 主扫描图 |
| `scripts/patch_tuple_scan.py` | 生成并验证 `LOP3/IMAD` tuple scan cubin |
| `scripts/analyze_tuple_scan.py` | 汇总 tuple scan 的模数模型命中率 |
| `scripts/physical_probe.py` | 生成 physical probe cubin、分析结果、查询 NCU 指标 |
| `scripts/run_ncu.sh` | 探索性 NCU profiling 入口 |

## 输出文件

主扫描：

```text
results/bank_scan/manifest.csv
results/bank_scan/results.csv
assets/register_bank_stride_scan.png
results/bank_scan_ffma/manifest.csv
results/bank_scan_ffma/results.csv
assets/register_bank_stride_scan_ffma.png
```

扩展验证：

```text
results/tuple_scan_lop3/
results/tuple_scan_imad/
results/physical_probe/
```

`build/` 和 `results/` 下的生成物默认不提交。正式报告应同时保存运行环境、CSV、manifest 和 patch 后反汇编验证信息。

## 已保存结果摘要

`LOP3` source-count controls：

| RF 源模式 | Median cycles/op |
|---|---:|
| 1 RF source | 2.086031 |
| 2 same-parity sources | 2.086031 |
| 3 mixed-parity sources | 2.086031 |
| 3 same-parity sources | 3.070406 |

`FFMA` source-count controls：

| RF 源模式 | Median cycles/op |
|---|---:|
| 1 RF source | 1.113438 |
| 2 same-parity sources | 2.072172 |
| 3 mixed-parity sources | 2.072164 |
| 3 same-parity sources | 3.064367 |

Tuple scan 结果中，`LOP3` 和 `IMAD` 均为：

```text
mod2  = 40/40 = 100.0%
mod4  = 30/40 = 75.0%
mod8  = 26/40 = 65.0%
mod16 = 18/40 = 45.0%
```

Physical probe 未发现稳定的 `mod4/mod8` 分层；NCU metric 查询只发现 L1TEX/LSU data-bank conflict 类指标，未发现可直接报告 RF/register/operand-collector bank conflict 的计数器。

## 结论边界

- 正确表述：`tested visible service group = physical_register_id % 2`。
- 不应表述：`physical SRAM bank = physical_register_id % 2`。
- 计时结果无法单独区分限制发生在 register SRAM、read port、operand collector，还是它们的组合。
- 当前完整 stride 主扫描只覆盖 `R4`–`R39` 相关物理寄存器编号、单 warp、当前 CUDA 工具链和 `LOP3/FFMA`。
- patched cubin 依赖未公开指令编码；更换 toolkit 或架构后必须重新反汇编验证。
