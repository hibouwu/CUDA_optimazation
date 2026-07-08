# Thor TCGen05 SS/TS MMA 与 tcgen05.cp 微基准设计

## 测试目的与边界

本文定义 `microbench/mma_with_cp` 的 TCGen05 微基准实验设计、结果口径、指标公式、表格模板和作图规划。当前目录下只有 `README.md` 与本文档，尚未包含 `benchmark_src/`、运行脚本、结果 CSV 或 NCU 报告；因此本文只给出待实现和待填充的实验定义，不写入尚未运行得到的性能数值。

本实验不包含 TMA。实验只研究 `tcgen05` 内部的以下路径：

1. SS MMA-only
2. TS MMA-only
3. `tcgen05.cp`-only
4. TS `tcgen05.cp` + `tcgen05.mma` 串行路径
5. TS `tcgen05.cp` + `tcgen05.mma` 双缓冲重叠路径
6. D-ring 对 MMA 和 cp+mma 路径的影响

SS 表示 A/B 均来自 SMEM descriptor；TS 表示 A 来自 TMEM、B 来自 SMEM descriptor。TS 路径中的 A tile 由 `tcgen05.cp` 从 SMEM 写入 TMEM。本文不测 GMEM、TMA、epilogue、TMEM readback、global store、sparse MMA、2CTA/cluster 路径。

![SS 路径流水线：tcgen05.mma 直接从 SMEM 读取 A/B tile](图片和附件/img_v3_0213d_d8abef8b-af78-405e-8b4e-ccb3557bcaag.jpg)

![TS 路径流水线：tcgen05.cp 先把 A tile 写入 TMEM，再执行 tcgen05.mma](图片和附件/img_v3_0213d_ab5d3d01-d9b6-4678-828b-01ec28efde9g.jpg)

测试精度为 BF16、FP8、FP4。测试 MMA shape 为 `M128N64`、`M128N128`、`M128N256`，因此共有 9 个 precision-shape 组合。K 由精度对应的 TCGen05 指令路径决定：BF16 使用 `K_inst=16`，FP8 使用 `K_inst=32`，FP4 使用 `K_inst=64`。

## 与已有 compute-only 实验的关系

`../mma_compute_only` 已有 dense SS MMA completion throughput 微基准、`current_document/Thor MMA instruction throughput microbenchmark.md`、`benchmark_src/`、`build_and_run.sh`、`分析报告.txt` 和 plot 输出。该目录的结果可作为理解 TCGen05 dense MMA 指令吞吐的背景材料，但不能直接替代本文的新实验结果。

需要特别区分旧 shared-D multi-issuer stress test 与本文新基线：

|项目|旧 shared-D multi-issuer stress test|本文新实验|
|---|---|---|
|D tile 归属|多个 issuer warp 可能写同一个 D tile 起点|每个 MMA issuer warp 至少拥有自己的独立 D tile|
|用途|压力测试 shared-D 写入和多 issuer 行为|拆分 SS/TS、cp-only、cp+mma overlap、D-ring 的可比路径|
|图 5 baseline|不能作为图 5 统一 baseline|`SS MMA D1` 是图 5 统一 baseline|
|结论迁移|只可作为背景，不直接写入新结果表|必须由 `mma_with_cp` 新 case 实测填充|

因此，本文后续所有 speedup 和 heatmap 都以同一 precision-shape 下的 `SS MMA D1` 为归一化基线，而不是以旧 compute-only shared-D case 为基线。

## TCGen05 指令口径

`tcgen05.mma` 执行的数学动作是：

```text
C[M,N] += A[M,K_inst] * B[K_inst,N]
```

SS MMA 使用 SMEM A/B descriptor：

```ptx
// SS: A from SMEM, B from SMEM, D/C in TMEM.
tcgen05.mma.cta_group::1.kind::<dtype>
  [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
```

TS MMA 使用 TMEM A 和 SMEM B descriptor：

```ptx
// TS: A from TMEM, B from SMEM, D/C in TMEM.
tcgen05.mma.cta_group::1.kind::<dtype>
  [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
```

FP4/block-scale 路径还需要 scale operand。FP4 SS/TS case 必须在 setup 阶段完成 scale TMEM 初始化，计时区间不能包含 scale 初始化。

`tcgen05.cp` 在 TS 路径中把 A tile 从 SMEM 搬到 TMEM：

```ptx
// [taddr]: TMEM destination; s_desc: SMEM source descriptor.
tcgen05.cp.cta_group::1.<cp-shape> [taddr], s_desc;
```

`<cp-shape>` 描述单条 copy 指令写入 TMEM 的范围，不等同于 MMA 的 `M128N64/M128N128/M128N256` shape。cp-only 的 `effective_bytes_per_cp` 必须根据实际使用的 cp 后缀、数据类型和 A tile 有效搬运范围计算，不用 cp 后缀字符串做近似替代。

## 测试结果定义

### 统一命名

`A1/A2` 表示 TMEM 中 A operand slot 数量。`D1/D2` 表示每个 warp 使用的 accumulator D tile 数量。

A double buffering 用于 cp/mma overlap：当 A slot 数量为 2 时，`cp(A_next)` 可以与 `mma(A_current)` 在不同 TMEM A slot 上交错执行。

D-ring 用于隐藏同一 D 的 accumulator dependency：当 D tile 数量为 2 时，同一个 warp 在 D0、D1 之间轮换累加，避免连续写同一个 accumulator D tile。

A double buffering 和 D-ring 不是同一个概念。A1/A2 控制 A operand slot；D1/D2 控制 accumulator D tile。

### Case 定义

|Case|路径|A slot|D tile/warp|计时区间|用途|
|---|---|---:|---:|---|---|
|SS MMA D1|SS MMA-only|-|1|SS `tcgen05.mma` 批量循环 + 最终 completion wait|每个 MMA issuer warp 使用一个独立 D tile；每个 warp 在循环中连续累加自己的同一个 D tile；作为图 5 的统一 baseline|
|SS MMA D2|SS MMA-only|-|2|SS `tcgen05.mma` 批量循环 + D0/D1 轮换 + 最终 completion wait|每个 warp 使用两个独立 D tile，并在 D0、D1 间轮换；用于观察 D-ring 是否隐藏 accumulator dependency|
|TS MMA D1|TS MMA-only|预驻留 A|1|TS `tcgen05.mma` 批量循环 + 最终 completion wait|A 在计时前预驻留于 TMEM，B 在 SMEM；计时区间只包含 TS MMA 和最终 wait；用于和 SS MMA D1 做纯 MMA source-mode 对比|
|tcgen05.cp-only|cp-only|目标 A slot|无|`tcgen05.cp` 批量循环 + 最终确认 copy 完成所需的 commit/barrier wait|源 A tile 已在 SMEM，目标 A slot 位于 TMEM；测 SMEM→TMEM copy 性能，输出 bytes/cycle 和 cycles/cp|
|TS CP+MMA Serial A1D1|TS cp+mma serial|1|1|每轮 cp 完成后执行 MMA，MMA 不再使用该 A slot 后进入下一轮；包含最终 wait|一个 TMEM A slot、一个 D tile；无 A double buffering 的串行基线|
|TS CP+MMA Overlap A2D1|TS cp+mma overlap|2|1|`cp(A_next)` 与 `mma(A_current)` 在不同 A slot 上交错执行；包含最终 wait|两个 TMEM A slot、一个 D tile；测量 A double buffering 带来的 cp/mma overlap|
|TS CP+MMA Overlap A2D2|TS cp+mma overlap + D-ring|2|2|A2 overlap 基础上，D0/D1 轮换；包含最终 wait|两个 TMEM A slot、两个 D tile；判断 overlap 后 accumulator dependency 是否仍限制性能|

上述 case 应覆盖 9 个 precision-shape 组合：

|Precision|MMA shapes|
|---|---|
|BF16|M128N64K16, M128N128K16, M128N256K16|
|FP8|M128N64K32, M128N128K32, M128N256K32|
|FP4|M128N64K64, M128N128K64, M128N256K64|

## 计时规则

setup 不计时，包括：

- `tcgen05.alloc`
- `mbarrier.init`
- descriptor 构造
- SMEM 初始化
- MMA-only 中的 A 预填充
- FP4 scale 初始化

计时必须覆盖待测指令序列，以及最后用于确认待测异步操作完成的 commit/barrier wait。`tcgen05.ld`、结果验证、`tcgen05.dealloc` 和 `tcgen05.relinquish_alloc_permit` 不计入纯吞吐时间。

本文不区分 startup 和 steady，不做拟合；所有 case 使用足够大的固定迭代次数测批量完成吞吐。所有对比 case 应尽量保持相同 launch、warp 角色、迭代次数、B descriptor、D 累加规则和计时方法。

推荐实现约束：

- 同一 precision-shape 下，SS/TS/cp+mma case 使用相同 grid、block、MMA issuer warp 数和 `iters`。
- 每个 MMA issuer warp 只由固定 lane 发射对应 inline PTX，避免不同 case 的 warp 角色变化影响结果。
- `mma_instruction_count`、`cp_instruction_count` 必须来自 launch 配置、issuer warp 数和循环次数的显式计算；后续可用 NCU counter 交叉验证。
- FP4 case 的 A/B scale 初始化必须在计时前完成，且所有相关 case 使用一致 scale 布局。

## 指标公式

### MMA TFLOP/s

```text
P = 2 * M * N * K_inst * mma_instruction_count / elapsed_seconds
MMA TFLOP/s = P / 1e12
```

`mma_instruction_count` 是计时区间内实际发射并完成的 MMA 指令总数。对于 cp+mma case，TFLOP/s 仍只统计 MMA 的数学计算量，elapsed time 使用 cp+mma 路径的计时区间。

### Peak Ratio

```text
Peak Ratio = measured_TFLOPS / theoretical_peak_for_precision
```

`theoretical_peak_for_precision` 按当前设备和 precision 的 dense 理论峰值填写。若 launch 只占用部分 SM，应在表格中明确理论峰值是否按 active SM 缩放；同一张图内必须使用一致口径。

### cp bytes/cycle

```text
bytes_per_cycle = cp_instruction_count * effective_bytes_per_cp / elapsed_cycles
```

`effective_bytes_per_cp` 必须使用实际每条 `tcgen05.cp` 搬运的有效字节数计算。低精度 packed case 需要按有效 A 数据字节数和实际 cp 形态确认，不得简单用 MMA FLOP 或 TMEM address stride 代替。

### cp cycles/cp

```text
cycles_per_cp = elapsed_cycles / cp_instruction_count
```

### 图 5 统一归一化收益

```text
Normalized Speedup(case) = Throughput(case) / Throughput(SS MMA D1)
```

对于同一 precision-shape，`SS MMA D1` 行恒为 `1.00x`。大于 1 表示快于 `SS MMA D1`，小于 1 表示慢于 `SS MMA D1`。图 5 只放归一化 speedup，不混入 TFLOP/s、cycles/cp 或 bytes/cycle。

### 附加 speedup

以下指标在结果表中额外给出，但不强制放入主 heatmap：

```text
A-double-buffer speedup =
  Throughput(TS Overlap A2D1) / Throughput(TS Serial A1D1)

SS D-ring speedup =
  Throughput(SS MMA D2) / Throughput(SS MMA D1)

TS pipeline D-ring speedup =
  Throughput(TS Overlap A2D2) / Throughput(TS Overlap A2D1)
```

## 结果表模板

尚未运行的 case 使用 `TBD` 或空单元格，不提前写性能结论。

### MMA-only TFLOP/s 与 Peak Ratio

|Precision|Shape|K_inst|SS MMA D1 TFLOP/s|TS MMA D1 TFLOP/s|SS MMA D1 Peak Ratio|TS MMA D1 Peak Ratio|备注|
|---|---|---:|---:|---:|---:|---:|---|
|BF16|M128N64|16|TBD|TBD|TBD|TBD||
|BF16|M128N128|16|TBD|TBD|TBD|TBD||
|BF16|M128N256|16|TBD|TBD|TBD|TBD||
|FP8|M128N64|32|TBD|TBD|TBD|TBD||
|FP8|M128N128|32|TBD|TBD|TBD|TBD||
|FP8|M128N256|32|TBD|TBD|TBD|TBD||
|FP4|M128N64|64|TBD|TBD|TBD|TBD||
|FP4|M128N128|64|TBD|TBD|TBD|TBD||
|FP4|M128N256|64|TBD|TBD|TBD|TBD||

### tcgen05.cp-only

|Precision|Shape|cp shape / suffix|effective bytes/cp|cp instruction count|elapsed cycles|bytes/cycle|cycles/cp|备注|
|---|---|---|---:|---:|---:|---:|---:|---|
|BF16|M128N64|TBD|TBD|TBD|TBD|TBD|TBD||
|BF16|M128N128|TBD|TBD|TBD|TBD|TBD|TBD||
|BF16|M128N256|TBD|TBD|TBD|TBD|TBD|TBD||
|FP8|M128N64|TBD|TBD|TBD|TBD|TBD|TBD||
|FP8|M128N128|TBD|TBD|TBD|TBD|TBD|TBD||
|FP8|M128N256|TBD|TBD|TBD|TBD|TBD|TBD||
|FP4|M128N64|TBD|TBD|TBD|TBD|TBD|TBD||
|FP4|M128N128|TBD|TBD|TBD|TBD|TBD|TBD||
|FP4|M128N256|TBD|TBD|TBD|TBD|TBD|TBD||

### 图 5 speedup 数据源

|Case|BF16-N64|BF16-N128|BF16-N256|FP8-N64|FP8-N128|FP8-N256|FP4-N64|FP4-N128|FP4-N256|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|SS MMA D1|1.00x|1.00x|1.00x|1.00x|1.00x|1.00x|1.00x|1.00x|1.00x|
|SS MMA D2|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|TS CP+MMA Serial A1D1|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|TS CP+MMA Overlap A2D1|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|TS CP+MMA Overlap A2D2|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|

### 附加 speedup 表

|Metric|BF16-N64|BF16-N128|BF16-N256|FP8-N64|FP8-N128|FP8-N256|FP4-N64|FP4-N128|FP4-N256|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|A-double-buffer speedup|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|SS D-ring speedup|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|TS pipeline D-ring speedup|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|

## 作图规划

最终报告规划五张图。

图 1：SS/TS MMA-only TFLOP/s 分组柱状图

- 横轴：9 个 precision-shape 组合
- 系列：`SS MMA D1`、`TS MMA D1`
- 纵轴：TFLOP/s

图 2：SS/TS MMA-only Peak Ratio 分组柱状图

- 横轴：9 个 precision-shape 组合
- 系列：`SS MMA D1`、`TS MMA D1`
- 纵轴：Peak Ratio (%)

图 3：`tcgen05.cp`-only bytes/cycle 柱状图

- 横轴：9 个 precision-shape 组合
- 纵轴：bytes/cycle

图 4：`tcgen05.cp`-only cycles/cp 柱状图

- 横轴：9 个 precision-shape 组合
- 纵轴：cycles/cp

图 5：相对 `SS MMA D1` 的收益 heatmap

- 列：`BF16-N64`、`BF16-N128`、`BF16-N256`、`FP8-N64`、`FP8-N128`、`FP8-N256`、`FP4-N64`、`FP4-N128`、`FP4-N256`
- 行：`SS MMA D1`、`SS MMA D2`、`TS CP+MMA Serial A1D1`、`TS CP+MMA Overlap A2D1`、`TS CP+MMA Overlap A2D2`
- 单元格显示归一化 speedup，例如 `0.96x`、`1.00x`、`1.18x`
- 色阶以 `1.00` 为中心
- 不把 TFLOP/s、cycles/cp、bytes/cycle 混入这张 heatmap

## 实现和验证待办

`mma_with_cp` 目前没有源码、脚本和结果文件。后续实现时应补齐：

- `benchmark_src/`：按 case、precision、shape 生成或维护 CUDA benchmark 源码。
- `build_and_run.sh` 或等价入口：统一 build、run、ncu、plot 命令。
- 结果 CSV：至少包含 case、precision、shape、K_inst、iters、launch、instruction counts、elapsed cycles、elapsed seconds、TFLOP/s、Peak Ratio、bytes/cycle、cycles/cp。
- 作图脚本：从结果 CSV 生成图 1 到图 5。
- NCU 验证：核对目标 SASS、MMA/cp 指令计数、completion wait、tensor/copy pipe counter 与源码循环一致。

在这些文件和结果出现前，本文所有数据表保持 `TBD`，不写性能结论。
