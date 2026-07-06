# SM110 寄存器结构证据与实验取舍

## 结论分级

### 直接观测

在 Thor、CUDA 13、`sm_110`、单 warp 和已测试的 `LOP3` 路径上：

- 三个同奇偶物理寄存器源约为 `3.070` cycles/op。
- 两个同奇偶源或三个混合奇偶源约为 `2.086` cycles/op。
- `R4`–`R7` × stride 1–16 的单链扫描中，奇数 stride 全部快，偶数 stride 全部慢。
- 同奇偶 source pair 放到不同 source slot 后，快慢关系不变。
- 四条独立 accumulator chain 可以隐藏单链中约一个周期的额外延迟。

### 数据支持的推断

这些观测支持如下有效模型：

```text
effective_group(register) = physical_register_id % 2

每个有效组在该 LOP3 路径上可服务至少两个源读取；
同一条指令的第三个同组源会暴露额外的读取/收集步骤。
```

这里的“有效组”描述的是被指令计时看见的服务能力，不指定它具体位于寄存器阵列、端口还是 operand collector。

### 当前没有证明的内容

现有数据不能证明：

- 寄存器文件恰好只有两个物理 SRAM bank。
- 每个物理 bank 的容量和读写端口数。
- 所有物理寄存器编号都遵循同一映射。
- IMAD、FFMA、Tensor Core 或 uniform register 使用相同路径。
- 多个物理 bank 不可能在当前路径上汇聚为两个奇偶服务组。

因此，文档中不再使用“物理 2-bank 已确认”或“99.99% 置信度”的表述。

## 核心数据如何形成

被测三源 tuple 是：

```text
(Rbase + stride, Rbase + 2 * stride, Rbase)
```

按奇偶分组：

```text
stride 为奇数：opposite, same, same  -> 最多两个同组源 -> 快
stride 为偶数：same,     same, same  -> 三个同组源     -> 慢
```

source-count controls 给出了关键对照：

| 对照 | 结果 | 能说明什么 |
|---|---:|---|
| 1 RF source | 2.086031 | 指令和依赖链基线 |
| 2 same-parity sources | 2.086031 | 两个同组源本身不产生可见额外延迟 |
| 3 mixed-parity sources | 2.086031 | 三源读取本身不是慢的原因 |
| 3 same-parity sources | 3.070406 | 慢点与第三个同组源相关 |

这组对照比单独观察奇偶 stride 更重要。只看 stride 周期，无法区分源操作数数量、依赖结构和执行管线造成的变化。

## 为什么不能据此确定物理 bank 数

时间测量识别的是“哪些 operand tuple 竞争同一服务资源”。下面多种实现都可能产生相同结果：

1. 两个单独的物理 bank，按寄存器奇偶映射。
2. 四个或更多物理 bank，在当前 collector/端口处合并成两个奇偶服务组。
3. 更多物理 bank，但某个三源指令读取网络只有两组可见带宽。

简单的 `physical_bank = register_id % 4` 独立端口模型不能直接解释所有结果，但这只排除了该简单模型，不能排除所有多-bank 微架构。

把 4 个 base × 16 个 stride 称为 64 个“独立证明”也不准确：它们来自同一 tuple 生成规则、同一指令和同一测量系统。它们证明模式在测试范围内稳定，却不能转换成物理 bank 数的常规 p-value。

## 每个实验是否必要

“必要”取决于目标。下表以两个目标区分：

- 目标 A：证明当前 `LOP3` 路径存在稳定的奇偶服务差异。
- 目标 B：把结论推广为整个 SM110 寄存器文件的物理组织。

| 实验 | 对目标 A | 对目标 B | 建议 |
|---|---|---|---|
| PTX `R0`–`R4` 基线 | 非必要 | 不充分 | 保留作延迟、吞吐、spill 和 `.reuse` sanity check |
| `B0`–`B3` source-count controls | **必要** | 必要但不充分 | 必须保留；它把“第三个同组源”与“三源指令本身”分开 |
| `M0`–`M2` source-slot permutations | 强烈建议 | 必要但不充分 | 保留；排除固定 operand slot 的特殊路由 |
| `L_bXX_sYY` 单链 base × stride | **必要** | 必要但范围不足 | 保留；这是奇偶模式和依赖延迟的主体证据 |
| `T_sYY` 四链吞吐扫描 | 非必要 | 非必要 | 保留作解释性对照，说明 ILP 可以隐藏单链延迟 |
| 早期 patched-IMAD 结果 | 非必要 | 当前不充分 | 当前缺少生成脚本和原始 CSV，只能视为历史负结果 |
| IMAD/FMA 跨指令 patched scan | 非必要 | **若要跨指令推广则必要** | 先补 patch/verify/run 链路，再纳入证据 |
| 扩大寄存器编号范围 | 非必要 | **若要全寄存器推广则必要** | 覆盖更多 base 和任意 tuple，而不只是 `R4`–`R39` |
| NCU 通用 stall 指标 | 非必要 | 旁证 | 可用于关联，但不能替代专用 RF bank counter |
| 直接 RF bank/port counter | 非必要 | 最有价值 | 若 Thor/工具暴露，可显著加强物理解释 |
| Tensor Core/uniform 路径 | 非必要 | 仅在声称覆盖这些路径时必要 | 应作为独立研究，不与标量 `LOP3` 混为一谈 |

## 最小实验集

如果文档目标只是可靠说明 `LOP3` 的有效奇偶行为，最小集合应包括：

1. `B0`–`B3`：源读取数和奇偶压力对照。
2. `M0`–`M2`：source slot 置换。
3. 至少两个不同奇偶 base 的单链 stride 扫描。
4. cubin patch 后逐条 SASS 回读、确认无 `.reuse`。
5. 多次重复、轮换 case 顺序、确认 `local_bytes == 0`。

当前完整的四 base × 16 stride 扫描适合保留在正式报告中，因为额外成本不高，并能显示范围内的一致性。`T_sYY` 不是核心证明，但有助于解释为什么其他吞吐实验看不到差异。

## 下一步实验优先级

### P0：让现有结论完全可复现

- 提交一次正式运行的 `manifest.csv`、`results.csv`、GPU/driver/toolkit 信息和验证后的 SASS 摘要。
- 不再引用缺失的 `run_sass_patched.sh`、`run_ncu.sh` 等旧脚本。
- 只使用 `scripts/plot_bank_scan.py` 生成正式结论图。

### P1：判断结论是否只属于 LOP3

- 为 IMAD 和 FFMA 实现与 LOP3 相同的 patch、清除 `.reuse`、反汇编回读和 source-count controls。
- 保持 opcode 数、依赖链、物理寄存器 tuple 和计时方法可比。

如果 IMAD/FFMA 不显示差异，只能说明它们的执行或 operand path 隐藏了差异，不能反向否定 LOP3 数据。

### P2：扩大寄存器空间与 tuple 形状

- 增加远离 `R4`–`R39` 的 base。
- 测试不等距任意三元组，而不只使用 `(base, base+s, base+2s)`。
- 分别控制“两个同奇偶”和“三个同奇偶”，并构造能区分简单 mod-4/mod-8 模型的 tuple。

仅把 stride 从 16 扩到 64 不能自动排除“多物理 bank 汇聚成两个有效组”；它只能检查当前奇偶模式是否在更大编号范围继续成立。

### P3：寻找直接硬件证据

- 查询 Thor 上是否存在公开且经验证的 RF read/port/bank counter。
- 若没有专用 counter，NCU stall 指标只作为时序结果的相关性旁证。
- 要确定 SRAM 宏数量，最终仍可能需要 NVIDIA 文档或更低层硬件信息。

## 可用于报告的结论

建议写成：

> 在 Thor SM110 上，patched-SASS `LOP3` 微基准显示：测试范围内的物理寄存器操作数按编号奇偶形成两个有效读取服务组。每组至少可在无额外可见延迟的情况下服务两个源；第三个同组源使依赖链增加约 0.984 cycles/op。该结果描述的是 `LOP3` operand path 的有效行为，不足以单独确定底层物理寄存器文件的 SRAM bank 数。

不建议写成：

> Thor 的寄存器文件已被证明是两个物理 bank，且 `register_id % 2` 就是物理 bank 映射。
