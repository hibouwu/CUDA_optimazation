# Thor TCGen05 cp + MMA pipeline microbenchmark

## **测试目的与背景**

本文用 Thor/SM110 上更贴近真实 GEMM 主循环的 SS/TS 流水线实验，测量 A 操作数输入准备对 `tcgen05.mma` 完成吞吐的影响。这里的输入准备指 `tcgen05.cp` 把 A tile 从 shared memory（SMEM）搬到 Tensor Memory（TMEM），让后续 TS `tcgen05.mma` 可以从 TMEM 读取 A；B 操作数仍然通过 SMEM descriptor 读取。

真实 GEMM 主循环通常先把 A/B tile 搬到 SMEM，再让 Tensor Core 消费这些 tile。上一篇 compute-only MMA 报告已经给出不含 A tile SMEM->TMEM copy 的 dense SS 基线，例如 `FullSM4WarpBlock M128N256K64 FP4` 达到 `1032.111 TFLOP/s`；本文继续拆开五组动作：SS MMA-only、TS MMA-only、`tcgen05.cp`-only、TS CP+MMA Serial A1、TS CP+MMA Overlap A2。

当前实验矩阵覆盖 45 个基础组合：5 类 case、3 种精度、3 种矩阵形状。矩阵形状按 `M*N*K` 顺序记录为 `M128N64`、`M128N128`、`M128N256`；K 随精度配置为 BF16 K=16、FP8 K=32、FP4 K=64。本版先固定实验动作、指标公式、结果表和图表规划，实测数据在 benchmark 实现后填入。

本文关注 SMEM/TMEM 输入路径对 `tcgen05.mma` 的影响。GMEM、TMA、epilogue、TMEM 读回、全局写回、sparse MMA、2CTA/cluster 路径留给后续实验；TMEM 分配、释放和 relinquish 也不放入本轮纯吞吐计时窗口。

![SS 路径流水线：tcgen05.mma 直接从 SMEM 读取 A/B tile](图片和附件/img_v3_0213d_d8abef8b-af78-405e-8b4e-ccb3557bcaag.jpg)

![TS 路径流水线：tcgen05.cp 先把 A tile 写入 TMEM，再执行 tcgen05.mma](图片和附件/img_v3_0213d_ab5d3d01-d9b6-4678-828b-01ec28efde9g.jpg)

SS 图展示真实 GEMM 主循环里最直接的 Tensor Core 消费路径。黑色 TMA load 阶段把连续 K 阶段的 A/B tile 放进 SMEM，红色 `tcgen05.mma` 随后从 SMEM descriptor 读取 A/B tile，并把累加结果写入 TMEM accumulator。图中的 GMEM、TMA 和后续写回用于说明真实 kernel 背景，本轮计时窗口只覆盖 `tcgen05` 指令序列和等待完成。

TS 图展示带 A 输入准备的主循环路径。黑色 TMA load 仍然先把 tile 放进 SMEM，蓝色 `tcgen05.cp` 把 A tile 从 SMEM 搬到 TMEM，红色 TS `tcgen05.mma` 再从 TMEM 读取 A、从 SMEM descriptor 读取 B，并更新 TMEM accumulator。蓝色 copy 阶段是本文相对上一篇 compute-only MMA 报告新增的核心实验对象。

## **tcgen05.mma / tcgen05.cp 指令解析**

本文用 SS/TS 描述 `tcgen05.mma` 的两个操作数来源组合。第一个字母描述 A 操作数的位置，第二个字母描述 B 操作数的位置：`S` 表示 shared memory descriptor，`T` 表示 Tensor Memory address。当前 dense GEMM 路径里 B 操作数仍来自 SMEM descriptor，所以本文只比较 `SS` 和 `TS` 两种形态。

SS 表示 A/B 都从 SMEM descriptor 读取，指令操作数是 `[d_tmem], a_desc, b_desc, idesc...`。TS 表示 A 从 TMEM 读取、B 从 SMEM descriptor 读取，指令操作数变成 `[d_tmem], [a_tmem], b_desc, idesc...`。TS 路径中的 A tile 由 `tcgen05.cp` 从 SMEM 写入 TMEM，再由后续 TS `tcgen05.mma` 作为 A 操作数读取；因此 TS case 需要额外定义 A slot 数量和 cp/mma 的先后关系。

`tcgen05.mma` 的单线程发射语义指的是：CTA 内一个发射线程执行一条 `tcgen05.mma` PTX，就足以描述并启动一个完整的 Tensor Core tile 操作。这个 tile 的数学范围由 `idesc` 中的 `M/N` 和 precision 对应的 `K` 决定，例如 `M128N256K64 FP4` 对应一条完整的 `128 * 256 * 64` MAC 指令，而不是 32 个 lane 各自提交一小片 MMA。

benchmark 中一个 CTA 共享同一组 SMEM tile、TMEM allocation 和用于确认完成的 mbarrier。当前真实路径口径只让 `threadIdx.x == 0` 作为发射线程提交 inline PTX，其余线程负责初始化、同步和等待；这样每个 CTA 每轮只提交一条完整的 cp 或 MMA 指令。

同一个发射线程循环中的 `tcgen05.mma` 和 `tcgen05.cp` 按程序顺序发射。也就是说，代码里写成 `mma; mma; mma` 或 `cp; wait; mma` 时，发射线程不会在同一线程内重排这些 inline PTX；异步操作的完成边界由后续 `tcgen05.commit`、mbarrier 等待或等待 copy 完成来确认。本文的计时窗口覆盖待测指令序列和最后的等待完成，所以表格中的 cycles 反映批量发射后的完成吞吐，而不是只记录提交成本。

本文固定 accumulator 目标策略，把变量集中在 A 操作数来源、`tcgen05.cp` copy 和 A double buffering。多个 accumulator tile 属于另一类 dependency/issue 压测，不作为本轮实验变量。

矩阵乘法的基本动作是把 A 和 B 的一小块矩阵相乘并累加到 C。硬件执行的数学形式可以写成 `C[M,N] += A[M,K] * B[K,N]`；这里 A 提供左侧输入，B 提供右侧输入，C/D 保存在 Tensor Memory 中。

本文的 shape 始终按 `M*N*K` 解释。`M` 是 C 矩阵的行数，`N` 是 C 矩阵的列数，`K` 是 A/B 之间相乘并规约的维度；例如 `M128N256K64` 表示 `C[128,256] += A[128,64] * B[64,256]`。

单条 MMA 指令的计算量来自 `M * N * K`。对 `M128N256K64 FP4` 来说，C 有 `128 * 256` 个元素，每个元素沿 K 方向做 `64` 次乘加，所以一条 MMA 指令包含 `128 * 256 * 64 = 2097152 MAC`；对 `M128N256K16 BF16` 来说，单条 MMA 指令包含 `128 * 256 * 16 = 524288 MAC`。

### **tcgen05 语法和操作数**

一条 `tcgen05.mma` 指令首先要把四类信息交给硬件：A 从哪里读、B 从哪里读、D/C 累加器写到哪里、这条 MMA 按什么 shape 和数据类型执行。SS 和 TS 的差异只发生在 A operand：SS 把 A 描述成 SMEM 中的一块矩阵，TS 把 A 描述成 TMEM 中的一块 A tile。B 在本文所有 dense case 中都来自 SMEM descriptor，D/C 都写入 TMEM accumulator。

传给 `tcgen05.mma` 和 `tcgen05.cp` 的操作数本身都是寄存器里的标量值。`a_desc`、`b_desc`、`s_desc` 是 64-bit descriptor，描述 SMEM 中某块矩阵的起始地址和布局；`d_tmem`、`a_tmem`、`taddr` 是 32-bit TMEM 地址，指向 TMEM 中的 D accumulator 或 A slot；`idesc` 描述本条 MMA 的 M/N shape 和数据类型。硬件拿到这些寄存器值后，才知道要从哪块 SMEM/TMEM 读数据、把结果写到哪块 TMEM。

SS `tcgen05.mma` 使用 SMEM A/B descriptor。硬件根据 `a_desc` 读取 A tile，根据 `b_desc` 读取 B tile，把结果累加到 `[d_tmem]` 指向的 TMEM 区域。对初学者来说，可以把 `a_desc` 和 `b_desc` 理解成“SMEM 矩阵的地址 + stride + layout”的打包描述，而不是普通指针。

```ptx
// SS: A from SMEM, B from SMEM, D/C in TMEM.
tcgen05.mma.cta_group::1.kind::<dtype>
  [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
```

TS `tcgen05.mma` 使用 TMEM A 和 SMEM B descriptor。硬件通过 `[a_tmem]` 读取已经写入 TMEM 的 A tile，通过 `b_desc` 继续从 SMEM 读取 B tile。这里 `[a_tmem]` 不是 SMEM descriptor，而是 TMEM 地址；这也是 TS 路径必须先安排 `tcgen05.cp` 的原因。

```ptx
// TS: A from TMEM, B from SMEM, D/C in TMEM.
tcgen05.mma.cta_group::1.kind::<dtype>
  [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
```

`tcgen05.cp` 把 A tile 从 SMEM 搬到 TMEM。`s_desc` 描述源 A tile 在 SMEM 中的位置和布局，`[taddr]` 指向目标 TMEM A slot。TS MMA-only 的 setup 会先用这条指令把 A 准备好；TS cp+mma pipeline 则把这条 copy 指令放进计时窗口，观察它和后续 TS MMA 的串行或重叠成本。

```ptx
// [taddr]: TMEM 目标地址；s_desc: SMEM 源 descriptor。
tcgen05.cp.cta_group::1.<cp-shape> [taddr], s_desc;
```

`<cp-shape>` 描述单条 copy 指令写入 TMEM 的范围，MMA shape 描述 `M128N64/M128N128/M128N256` 的矩阵乘加范围。cp-only 的 copied bytes 根据实际 cp 后缀、数据类型和 A tile 搬运范围计算。

下面的简化代码块展示这些寄存器操作数如何进入 `tcgen05.cp`、SS `tcgen05.mma` 和 TS `tcgen05.mma`。inline PTX 约束中，`"l"` 表示 64-bit register operand，常用于 SMEM descriptor；`"r"` 表示 32-bit register operand，常用于 TMEM 地址和 `idesc` 高 32 位字段。

```C++
uint64_t a_desc = make_smem_desc(smem_a, desc_leading, desc_stride);
uint64_t b_desc = make_smem_desc(smem_b, desc_leading, desc_stride);
uint64_t idesc = make_idesc_for_shape_and_precision();

uint32_t d_tmem = tmem_base + d_offset;
uint32_t a_tmem = tmem_base + a_slot_offset;

// 1. SS MMA: A/B 都从 SMEM descriptor 读取。
asm volatile(
  "tcgen05.mma.cta_group::1.kind::<dtype> "
  "[%0], %1, %2, %3, 0, 1;"
  :: "r"(d_tmem), "l"(a_desc), "l"(b_desc),
     "r"(uint32_t(idesc >> 32)));

// 2. cp: 把 a_desc 描述的 SMEM A tile 写到 a_tmem 指向的 TMEM A slot。
asm volatile(
  "tcgen05.cp.cta_group::1.<cp-shape> [%0], %1;"
  :: "r"(a_tmem), "l"(a_desc) : "memory");

// 3. TS MMA: A 从 TMEM A slot 读取，B 仍从 SMEM descriptor 读取。
asm volatile(
  "tcgen05.mma.cta_group::1.kind::<dtype> "
  "[%0], [%1], %2, %3, 0, 1;"
  :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc),
     "r"(uint32_t(idesc >> 32)));
```

FP4/block-scale 路径额外传入 scale operand。FP4 SS/TS case 在 setup 阶段完成 scale TMEM 初始化，计时区间从待测 MMA 或 cp 指令序列开始。也就是说，FP4 的 scale 数据是 MMA 指令需要的输入解释信息，但 scale 初始化本身不是本文要测的吞吐对象。

PTX 是源码里写的汇编形式，SASS 是 GPU 最终执行的机器指令。本文后续实现需要用 `cuobjdump --dump-sass` 检查 BF16、FP8、FP4 的目标 MMA SASS 和 `tcgen05.cp` 对应的 `UTCCP` SASS，并用 NCU 计数器核对实际指令计数。注意源码中的 `tcgen05.cp...` 不会在 SASS 中按原字符串出现；例如 `tcgen05.cp.cta_group::1.128x128b...` 会编译成 `UTCCP.T.S.128dp128bit...` 形式，所以检查应匹配 SASS opcode 和 decoded shape token，而不是匹配源码 PTX 字符串。完整 `tcgen05.mma` 语法以 NVIDIA PTX ISA 文档中的 [`tcgen05.mma` 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-instructions-mma) 章节为准。

## **实验环境**

实验设备为 NVIDIA Jetson AGX Thor Developer Kit。官方 dense 理论峰值沿用上一篇 MMA throughput 报告中的 Thor dense 指标：FP4 1035 TFLOP/s、FP8 517 TFLOP/s、BF16/FP16 258.5 TFLOP/s。脚本读取 GPU GPC 当前频率作为实测频率；Peak Ratio 使用当前 precision 对应的理论峰值。

硬件、软件和 benchmark 固定参数如下。

|硬件平台|参数|
|---|---|
|GPU|NVIDIA Thor|
|Compute capability|`11.0`|
|SM count|`20`|
|Warp size|`32`|
|L2 cache|`32768.0 KiB`|
|GPU GPC frequency|`1.575 GHz`|

|软件环境|参数|
|---|---|
|OS / kernel|Linux `6.8.12-tegra`，`aarch64`|
|CUDA runtime / driver|`13000` / `13000`|
|CUDA Toolkit / nvcc|`13.0` / `V13.0.88`|
|cuobjdump|CUDA `13.0` / `V13.0.85`|
|Python|`3.12.3`|
|NVCC target|`arch=compute_110a,code=sm_110a`|

|设备资源上限|参数|
|---|---|
|Max threads/block|`1024`|
|Max threads/SM|`1536`|
|Max blocks/SM|`24`|
|Registers/block|`65536`|
|Registers/SM|`65536`|
|Shared memory/block|`48.0 KiB` (`49152 bytes`)|
|Shared memory/block opt-in|`227.0 KiB` (`232448 bytes`)|
|Shared memory/SM|`228.0 KiB` (`233472 bytes`)|
|Reserved shared memory/block|`1.0 KiB` (`1024 bytes`)|

|Benchmark 固定配置|参数|
|---|---|
|Grid size|`20` CTAs|
|Block size|`128` threads|
|Warps/CTA|`4`|
|MMA / CP issuer|仅 `threadIdx.x == 0`|
|TMEM allocation|`512 columns = 256 KiB`|
|D accumulator|`d_tmem = tmem_base`|
|A slot 0|`a_tmem0 = tmem_base + 256`|
|A slot 1|`a_tmem1 = tmem_base + 320`|
|FP4 scale A|`tsfa = tmem_base + 384`|
|FP4 scale B|`tsfb = tmem_base + 448`|

|生成 kernel 资源用量|REG|SHARED|
|---|---:|---:|
|MMA-only / TS MMA-only / TS CP+MMA|`21`|`66572 bytes`|
|`tcgen05.cp-only`|`14`|`33804 bytes`|

## **测试动作**

脚本需要生成 5 类 benchmark case。每个 case 覆盖 BF16、FP8、FP4 与 `M128N64`、`M128N128`、`M128N256` 的 9 个 precision-shape 组合，总计 45 个基础测试组合。

五组计时动作和上面的流水线图直接对应。`SS MMA-only` 对应 SS 图中红色 `tcgen05.mma` 消费 SMEM A/B 的成本；`TS MMA-only` 对应 TS 图中 A 已经在 TMEM 后红色 TS `tcgen05.mma` 的使用成本；`tcgen05.cp-only` 对应 TS 图里的蓝色 A tile 输入准备；`TS CP+MMA Serial A1` 测单 A slot 下 copy 和 MMA 依次暴露的成本；`TS CP+MMA Overlap A2` 测两个 A slot 下 copy 和 MMA 交错后仍暴露的 cycles/tile。

|Case|A 来源|B 来源|A slots|计时窗口|输出指标|
|---|---|---|---:|---|---|
|SS MMA-only|SMEM|SMEM|-|SS `tcgen05.mma` 循环 + 等待完成|TFLOP/s, Peak Ratio|
|TS MMA-only|TMEM|SMEM|预填 A|TS `tcgen05.mma` 循环 + 等待完成|TFLOP/s, Peak Ratio|
|tcgen05.cp-only|SMEM|-|1|`tcgen05.cp` 循环 + 等待 copy 完成|bytes/cycle, cycles/cp|
|TS CP+MMA Serial A1|SMEM->TMEM|SMEM|1|`cp -> wait -> mma -> wait` 循环|TFLOP/s, cycles/tile|
|TS CP+MMA Overlap A2|SMEM->TMEM|SMEM|2|`cp(A_next)` overlaps `mma(A_current)`|TFLOP/s, cycles/tile, Overlap Gain|

SS MMA-only 在 A/B 都来自 SMEM descriptor、D 位于 TMEM 的配置下循环发射 SS `tcgen05.mma`。这个 case 给出每个 precision-shape 的基线，图 5 的归一化 speedup 都以同 shape、同 precision 的 SS MMA-only 吞吐为分母。

TS MMA-only 先用 `tcgen05.cp` 把 A tile 准备到 TMEM，再只对 TS `tcgen05.mma` 循环和等待完成计时。它和 SS MMA-only 使用同一 `M128N*` shape、`K_inst`、B descriptor 和计时方式，用来比较 A 操作数来源从 SMEM 改为 TMEM 后的 MMA 完成吞吐。

tcgen05.cp-only 把源 A tile 放在 SMEM，循环发射 `tcgen05.cp` 写入 TMEM A slot，并等待 copy 完成。这个 case 输出 `bytes/cycle = total copied bytes / elapsed cycles` 和 `cycles/cp = elapsed cycles / cp instruction count`。

TS CP+MMA Serial A1 使用一个 TMEM A slot，按 `cp -> wait -> mma -> wait` 顺序串行处理 A tile。

```text
SMEM A
  |
tcgen05.cp
  |
TMEM A
  |
tcgen05.mma
```

TS CP+MMA Overlap A2 使用两个 TMEM A slot，在一个 slot 上执行 `mma(A_current)`，同时向另一个 slot 发射 `cp(A_next)`。

```text
cp(A1) overlaps mma(A0)
cp(A0) overlaps mma(A1)
```

A double buffering 的收益用 `Overlap Gain` 单独记录：

```text
Overlap Gain =
  Throughput(TS CP+MMA Overlap A2) /
  Throughput(TS CP+MMA Serial A1)
```

基础测试组合如下。

|Case|BF16|FP8|FP4|合计|
|---|---:|---:|---:|---:|
|SS MMA-only|3 shapes|3 shapes|3 shapes|9|
|TS MMA-only|3 shapes|3 shapes|3 shapes|9|
|tcgen05.cp-only|3 shapes|3 shapes|3 shapes|9|
|TS CP+MMA Serial A1|3 shapes|3 shapes|3 shapes|9|
|TS CP+MMA Overlap A2|3 shapes|3 shapes|3 shapes|9|
|Total||||45|

## **实现方案**

实现入口保持和上一篇 MMA throughput benchmark 相同的自动化流程：读取 GPU 信息和频率，生成 CUDA benchmark，编译二进制，运行全部 45 个基础组合，并写出结构化结果和中文报告。

生成的 benchmark 按 case、shape、precision 组织，例如：

```text
tcgen05_ss_mma_only_m128n256_fp4
tcgen05_ts_mma_only_m128n128_fp8
tcgen05_cp_only_m128n64_bf16
tcgen05_ts_cp_mma_serial_a1_m128n256_fp4
tcgen05_ts_cp_mma_overlap_a2_m128n128_fp8
```

每份 CUDA 源码固定一个 `kMacPerInst`、`kInstK` 和 shape 标签。以 `M128N256K64 FP4` 为例，`kMacPerInst = 128 * 256 * 64 = 2097152`。

```C++
static constexpr long long kMacPerInst = 2097152LL;
static constexpr int kInstK = 64;
static constexpr char kPrecision[] = "FP4";
static constexpr char kShape[] = "M128N256K64";
```

### **CUDA 源码执行流程**

每个 benchmark kernel 在计时前完成 SMEM 初始化、mbarrier 初始化、descriptor 创建、TMEM 分配、TMEM A 初始化和 FP4 scale 初始化。计时窗口覆盖待测 `tcgen05` 指令序列和对应的等待完成；`tcgen05.ld` 读回、全局写回和 TMEM 释放放在计时窗口外。

```C++
__shared__ alignas(16) uint8_t smem_a[/* TBD */];
__shared__ alignas(16) uint8_t smem_b[/* TBD */];
__shared__ alignas(8) uint64_t done_barrier;
__shared__ uint32_t tmem_base;
```

descriptor 规则沿用 compute-only 的 shape/K 口径。BF16、FP8 和 FP4 的 K 分别为 16、32、64；`M128N64`、`M128N128`、`M128N256` 通过 `idesc` 的 M/N 字段选择。

|Precision|K|A bytes/element|说明|
|---|---:|---:|---|
|BF16|16|2|dense f16/bf16 路径|
|FP8|32|1|dense f8/f6/f4 路径|
|FP4|64|0.5|mxf4/nvf4 block-scale 路径|

```C++
uint64_t a_desc = make_smem_desc(smem_a, desc_leading, desc_stride);
uint64_t b_desc = make_smem_desc(smem_b, desc_leading, desc_stride);
uint64_t idesc = make_idesc_for_shape_and_precision();
```

计时窗口使用 `clock64()` 包住目标循环和等待完成。多 block 测试建议保存每个 block 的 cycles，并用 `max_cycles` 作为整卡完成时间口径。

```C++
unsigned long long start = clock64();
// 目标 tcgen05 循环：mma-only、cp-only、serial cp+mma 或 overlap cp+mma
// commit / mbarrier 等待，确认操作完成
unsigned long long stop = clock64();
```

吞吐和 copy 指标使用以下公式：

```text
MMA TFLOP/s =
  2 * M * N * K_inst * mma_instruction_count / elapsed_seconds / 1e12

Peak Ratio =
  measured_TFLOP/s / theoretical_peak_for_precision

bytes/cycle =
  total copied bytes / elapsed cycles

cycles/cp =
  elapsed cycles / cp instruction count

cycles/tile =
  elapsed cycles / processed tile count
```

图 5 的归一化使用同一 precision-shape 下的 SS MMA-only：

```text
Normalized Speedup =
  Throughput(case) / Throughput(SS MMA-only)
```

## **反汇编验证**

脚本需要用 `cuobjdump --dump-sass` 检查目标 MMA 和 copy 指令是否出现在二进制中。BF16 dense MMA 应命中 BF16 对应 SASS，FP8 dense MMA 应命中 FP8 对应 SASS，FP4 dense MMA 应命中 FP4/block-scale 对应 SASS；`tcgen05.cp` case 应命中 `UTCCP`，并进一步核对 PTX suffix 对应的 SASS shape/decode token。不能用“源码里包含 `tcgen05.cp`”作为反汇编检查通过条件。

当前脚本的 dense MMA SASS 检查口径如下。

|Precision|PTX MMA instruction|期望 SASS token|
|---|---|---|
|BF16|`tcgen05.mma.cta_group::1.kind::f16`|`UTCHMMA`|
|FP8|`tcgen05.mma.cta_group::1.kind::f8f6f4`|`UTCQMMA`|
|FP4|`tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16`|`UTCOMMA.4X`|

当前脚本的 `tcgen05.cp` SASS 检查口径按 precision 区分如下。BF16 和 FP8 当前都使用 plain `128x128b` copy；FP4 使用 packed/decode 形式，把 packed FP4 A tile 写入 TMEM A slot。

|Precision|PTX cp instruction|PTX cp suffix|期望 SASS token|
|---|---|---|---|
|BF16|`tcgen05.cp.cta_group::1.128x128b`|`128x128b`|`UTCCP` + `128DP128BIT`|
|FP8|`tcgen05.cp.cta_group::1.128x128b`|`128x128b`|`UTCCP` + `128DP128BIT`|
|FP4|`tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64`|`128x128b.b8x16.b4x16_p64`|`UTCCP` + `128DP128BIT` + `U4X16P64`|

代表性 SASS 形态如下。

```sass
UTCCP.T.S.128dp128bit tmem[...], gdesc[...];
UTCCP.T.S.128dp128bit.U4x16P64 tmem[...], gdesc[...];
```

NCU 计数器用来核对硬件实际执行的 MMA/cp 指令数量。当前真实路径口径下，如果 `grid=SM_count`、`block=128`、每 CTA 1 个发射线程、`iters=10000`，目标 MMA 指令数应为 `SM_count * 1 * 10000`；cp-only 和 cp+mma case 按各自循环中的 cp 指令数计算。

后续 NCU 小指标集应至少覆盖：

- cycles
- 目标 MMA 指令数
- 目标 cp 指令数
- tensor pipe 计数器
- copy / non-MMA pipe 计数器
- warp 发射 / stall 计数器
- launch grid、block、active warps

## **实验结果**

### MMA-only TFLOP/s 与 Peak Ratio

当前还没有 `mma_with_cp` benchmark 结果。下表固定 SS/TS MMA-only 的填数口径，结果生成前使用 `TBD`。

|Precision|Shape|K|SS MMA-only TFLOP/s|TS MMA-only TFLOP/s|SS Peak Ratio|TS Peak Ratio|
|---|---|---:|---:|---:|---:|---:|
|BF16|M128N64|16|TBD|TBD|TBD|TBD|
|BF16|M128N128|16|TBD|TBD|TBD|TBD|
|BF16|M128N256|16|TBD|TBD|TBD|TBD|
|FP8|M128N64|32|TBD|TBD|TBD|TBD|
|FP8|M128N128|32|TBD|TBD|TBD|TBD|
|FP8|M128N256|32|TBD|TBD|TBD|TBD|
|FP4|M128N64|64|TBD|TBD|TBD|TBD|
|FP4|M128N128|64|TBD|TBD|TBD|TBD|
|FP4|M128N256|64|TBD|TBD|TBD|TBD|

图 1 规划为 SS/TS MMA-only TFLOP/s 分组柱状图。横轴为 9 个 precision-shape 组合，系列为 `SS MMA-only` 和 `TS MMA-only`。

图 2 规划为 SS/TS MMA-only Peak Ratio 分组柱状图。Peak Ratio 使用 `Measured TFLOP/s / Theoretical Peak`，不同 precision 使用对应理论峰值。

### tcgen05.cp-only 结果

cp-only 表格记录 SMEM->TMEM copy 的 bytes/cycle 和 cycles/cp。`effective bytes/cp` 使用每条 `tcgen05.cp` 的实际搬运字节数。

|Precision|Shape|cp shape / suffix|effective bytes/cp|cp instruction count|elapsed cycles|bytes/cycle|cycles/cp|
|---|---|---|---:|---:|---:|---:|---:|
|BF16|M128N64|TBD|TBD|TBD|TBD|TBD|TBD|
|BF16|M128N128|TBD|TBD|TBD|TBD|TBD|TBD|
|BF16|M128N256|TBD|TBD|TBD|TBD|TBD|TBD|
|FP8|M128N64|TBD|TBD|TBD|TBD|TBD|TBD|
|FP8|M128N128|TBD|TBD|TBD|TBD|TBD|TBD|
|FP8|M128N256|TBD|TBD|TBD|TBD|TBD|TBD|
|FP4|M128N64|TBD|TBD|TBD|TBD|TBD|TBD|
|FP4|M128N128|TBD|TBD|TBD|TBD|TBD|TBD|
|FP4|M128N256|TBD|TBD|TBD|TBD|TBD|TBD|

图 3 规划为 `tcgen05.cp`-only bytes/cycle 柱状图。图 4 规划为 `tcgen05.cp`-only cycles/cp 柱状图。

### CP+MMA pipeline 结果

Serial A1 和 Overlap A2 使用同一 tile 计数口径。Overlap Gain 使用 `Throughput(TS CP+MMA Overlap A2) / Throughput(TS CP+MMA Serial A1)`。

|Precision|Shape|Serial A1 TFLOP/s|Serial A1 cycles/tile|Overlap A2 TFLOP/s|Overlap A2 cycles/tile|Overlap Gain|
|---|---|---:|---:|---:|---:|---:|
|BF16|M128N64|TBD|TBD|TBD|TBD|TBD|
|BF16|M128N128|TBD|TBD|TBD|TBD|TBD|
|BF16|M128N256|TBD|TBD|TBD|TBD|TBD|
|FP8|M128N64|TBD|TBD|TBD|TBD|TBD|
|FP8|M128N128|TBD|TBD|TBD|TBD|TBD|
|FP8|M128N256|TBD|TBD|TBD|TBD|TBD|
|FP4|M128N64|TBD|TBD|TBD|TBD|TBD|
|FP4|M128N128|TBD|TBD|TBD|TBD|TBD|
|FP4|M128N256|TBD|TBD|TBD|TBD|TBD|

图 5 规划为 Pipeline performance heatmap。列为 `BF16-N64`、`BF16-N128`、`BF16-N256`、`FP8-N64`、`FP8-N128`、`FP8-N256`、`FP4-N64`、`FP4-N128`、`FP4-N256`；行为 `SS MMA-only`、`TS MMA-only`、`TS CP+MMA Serial A1`、`TS CP+MMA Overlap A2`。单元格只显示相对 `SS MMA-only` 的 normalized speedup。

|Case|BF16-N64|BF16-N128|BF16-N256|FP8-N64|FP8-N128|FP8-N256|FP4-N64|FP4-N128|FP4-N256|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|SS MMA-only|1.00x|1.00x|1.00x|1.00x|1.00x|1.00x|1.00x|1.00x|1.00x|
|TS MMA-only|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|TS CP+MMA Serial A1|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|TS CP+MMA Overlap A2|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|

### NCU 抓取结果

NCU 抓取结果用于确认 launch 规模、MMA/cp 指令计数和关键计数器。后续报告保留两个到三个代表 case 的截图和文字说明，例如 `M128N256K64 FP4`、`M128N64K16 BF16`、以及一个 cp+mma overlap case。

## **总结**

本文把 `mma_with_cp` 的实验范围收敛为 5 类 TCGen05 内部路径：SS MMA-only、TS MMA-only、tcgen05.cp-only、TS CP+MMA Serial A1 和 TS CP+MMA Overlap A2。三种精度和三种 shape 组合后共有 45 个基础 benchmark。

SS/TS MMA-only 对比回答 A 操作数来源从 SMEM 切到 TMEM 后，MMA 完成吞吐是否变化。cp-only 表格回答 SMEM->TMEM 的 copy bytes/cycle 和 cycles/cp。Serial A1 与 Overlap A2 的对比回答 A double buffering 能否降低 cp+mma pipeline 的 cycles/tile。

当前需要实现的 benchmark case 如下：

|Case|Case ID|
|---|---|
|SS MMA-only|`ss_mma_only`|
|TS MMA-only|`ts_mma_only`|
|tcgen05.cp-only|`tcgen05_cp_only`|
|TS CP+MMA Serial A1|`ts_cp_mma_serial_a1`|
|TS CP+MMA Overlap A2|`ts_cp_mma_overlap_a2`|

## **附录：环境参数测量方法**

GPU、thread/block limit、register file、SMEM limit 和 L2 cache 来自 CUDA runtime 查询；OS/kernel 来自系统内核版本查询；CUDA/nvcc/cuobjdump/Python 版本来自对应工具的版本输出；GPU GPC frequency 来自系统 devfreq 接口的当前频率读数。

本实验 kernel 资源用量来自当前脚本生成的 45 个 CUDA benchmark，经 `nvcc -O3 -gencode arch=compute_110a,code=sm_110a` 编译后用 `cuobjdump --dump-resource-usage` 检查。结果按 case 收敛为：MMA/TS CP+MMA case 使用 `REG:21`、`SHARED:66572 bytes`；`tcgen05.cp-only` 使用 `REG:14`、`SHARED:33804 bytes`。

TMEM size 来自独立 TMEM sweep probe。该 probe 对每个 column request 单独启动进程，执行 `tcgen05.alloc` 后用 `tcgen05.st/ld.sync.aligned.32x32b.x1.b32` 读写第 0 列和 `columns - 1` 最后一列；首尾都读回正确值才记为 OK。这个口径下 `512 columns = 256 KiB` 是可首尾读写的最大 allocation。

## **边界**

本文覆盖 `tcgen05` 内部 A 操作数来源、SMEM->TMEM copy 和 cp-mma overlap。GMEM、TMA、epilogue、TMEM 读回、全局写回、sparse MMA、2CTA/cluster 路径后续单独测；结果生成前，本文所有性能表保持 `TBD`。
