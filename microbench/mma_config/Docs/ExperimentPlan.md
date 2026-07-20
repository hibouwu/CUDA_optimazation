# tcgen05 MMA 硬件路径标定与配置敏感性实验计划

## 1. 目标与结论边界

本组实验分为两层：

1. 先标定 `tcgen05.mma` 的有效硬件行为；
2. 再研究 shape、SMEM layout、TMEM columns 和 D 地址策略对性能的影响。

优先回答以下问题：

1. SMEM 到 Tensor Core 的最大有效 operand 供数率是多少；
2. `tcgen05.mma` 对 SMEM bank pattern 的敏感性，以及可观察到的有效 bank 并行度；
3. Tensor Core operand fetch 是否与普通 LSU `ld.shared` 竞争同一关键资源；
4. 可以同时隐藏多少条独立 MMA，其有效在途窗口有多大；
5. TMEM accumulator 地址、column 和 `input_d` 是否引入额外限制。

这些实验不能直接观察物理连线和内部队列，因此报告中应使用以下名称：

| 希望了解的硬件量 | 微基准实际报告的量 | 结论边界 |
| --- | --- | --- |
| SMEM→Tensor Core 端口宽度 | 最大有效 operand bytes/cycle | 物理宽度的下界或性能等效值 |
| 每周期读取的 SMEM bank 数 | 有效 bank 并行度与 bank-pattern 敏感性 | 不等同于内部物理读端口数 |
| 每周期写入的 TMEM bank 数 | accumulator update 吞吐与地址周期性 | 不等同于物理 TMEM bank 数 |
| 是否与 LSU 完全共享端口 | `ld.shared` 与 MMA 的干扰曲线 | 可证明共享瓶颈，通常不能证明物理端口完全相同 |
| operand collector 深度 | 有效独立在途 MMA 窗口 | 可能同时包含 collector、scoreboard 和异步队列限制 |

后续文档不得仅凭 `logical bytes / cycles`，直接把结果命名为物理端口宽度、物理 bank 数或 collector entry 数。

## 2. 建议目录结构

```text
microbench/mma_config/
  Docs/
    MicroBench.md
    ExperimentPlan.md

  common/
    tcgen05_helpers.cuh
    timing.cuh
    result_writer.cuh

  00_instruction_baseline/
    README.md
    benchmark_src/
    scripts/
    plots/

  01_effective_smem_ingress/
    README.md
    benchmark_src/
    scripts/
    plots/

  02_smem_bank_pattern/
    README.md
    benchmark_src/
    scripts/
    plots/

  03_ldshared_contention/
    README.md
    benchmark_src/
    scripts/
    plots/

  04_inflight_window/
    README.md
    benchmark_src/
    scripts/
    plots/

  05_tmem_address_pattern/
    README.md
    benchmark_src/
    scripts/
    plots/

  06_shape_layout_sensitivity/
    README.md
    benchmark_src/
    scripts/
    plots/
```

`common/` 只保存共享的 PTX wrapper、计时、CSV 输出和参数解析。每个子目录保留独立 kernel，使指令序列和控制变量能够直接检查。

## 3. 统一实验口径

所有子实验固定并记录：

- GPU 型号、compute capability、driver 和 CUDA toolkit 版本；
- power mode、SM clock 和 memory clock；
- kernel launch 配置、动态 SMEM、TMEM columns 和实际 occupancy；
- CUDA event 计时方式、warmup、repeat 和迭代次数；
- grid 至少覆盖全部 SM，同时保留单 SM 或低 grid 的诊断结果；
- CTA thread 数默认与目标 GEMM 一致，即 128 threads；
- BF16 与 FP16 分开记录，不混在同一条曲线中；
- SASS/PTX 检查结果，确认编译器没有删除或移动干扰指令。

默认 shape：

```text
m128n64k16
m128n128k16
m128n256k16
```

至少提供两种计时口径：

1. 单 CTA/低 grid：观察指令延迟和资源竞争；
2. 全 SM 满载：得到目标 GEMM 可用的稳态吞吐。

统一 CSV 字段建议：

```text
experiment, gpu, cuda_version, clock_mhz,
dtype, m, n, k, iterations,
batch_size, outstanding_groups, wait_mode,
smem_layout, swizzle, alignment_bytes, lda, ldb, smem_base_offset,
tmem_columns, d_base_column, d_column_stride, d_mode, input_d,
ldshared_mode, ldshared_bytes_per_iter, ldshared_bank_relation,
active_warps, resident_ctas, elapsed_us,
issue_cycles_per_mma, completion_cycles, steady_cycles_per_mma,
logical_smem_bytes_per_mma, effective_smem_bytes_per_cycle,
tflops, notes
```

对于 A/B 都来自 SMEM 的 BF16/FP16 MMA：

```text
logical_smem_bytes_per_mma = sizeof(dtype) * k * (m + n)
effective_smem_bytes_per_cycle =
    logical_smem_bytes_per_mma / steady_cycles_per_mma
```

例如 `m128n128k16` 的 FP16/BF16 逻辑 operand 流量为：

```text
A = 128 * 16 * 2 = 4096 B
B = 128 * 16 * 2 = 4096 B
A + B = 8192 B/MMA
```

`effective_smem_bytes_per_cycle` 是有效供数率，不是未经验证的物理端口宽度。

## 4. 第一层：硬件路径标定

## 4.1 `00_instruction_baseline`

### 要回答的问题

- 一条独立 MMA 的异步完成延迟是多少；
- 连续发射时的稳态 issue/execute 吞吐是多少；
- `commit/wait` 本身给测量引入多少固定成本。

### 必要 case

```text
empty_loop
commit_wait_only
mma_then_immediate_commit_wait
batched_independent_mma_then_commit_wait
```

MMA case 同时比较：

```text
d_mode = same_d, alternating_d, independent_d
input_d = 0, 1
```

### 判读

- `mma_then_immediate_commit_wait` 测到的是完成延迟、同步开销和等待空泡之和；
- batched case 收敛后的 cycles/MMA 才用于估计稳态吞吐；
- forced-wait 下 same-D 与 independent-D 的差距用于识别 D 地址依赖污染；
- 后续实验必须使用本节确认过的独立 D 策略和低开销 wait 方式。

## 4.2 `01_effective_smem_ingress`

### 要回答的问题

- SMEM operand path 能达到的最大有效 bytes/cycle；
- 不同 shape 是否落在同一供数 roofline 上；
- 小 N shape 是否更容易受到 operand feed 限制。

### 方法

对每个 shape 连续发射足够多的 independent-D MMA，再统一完成等待：

```text
batch_size = 1, 2, 4, 8, 16, 32, 64
```

先使用推荐 swizzle、alignment 和 leading dimension。每个 case 同时报告：

```text
steady_cycles_per_mma
logical_smem_bytes_per_mma
effective_smem_bytes_per_cycle
TFLOP/s
```

### 判读

- 多个 shape 在相近的 effective bytes/cycle 处饱和，支持 SMEM operand path roofline 假设；
- 多个 shape在相近 FLOP/cycle 处饱和，更像 Tensor Core compute limit；
- 最大 effective bytes/cycle 只能作为物理端口宽度下界或等效上限；
- 如果 shape、layout 或 D 策略改变后平台明显移动，不应提取单一端口常数。

## 4.3 `02_smem_bank_pattern`

### 要回答的问题

- `tcgen05.mma` operand fetch 是否对 bank pattern、swizzle 和地址对齐敏感；
- 是否存在稳定的地址模周期和 2×、4× 等吞吐退化；
- 可以观察到多大的有效 SMEM bank 并行度。

### 方法

固定 shape、batch size、D 地址和 `input_d`，改变：

```text
swizzle = 128B, 64B, 32B, none
alignment = 16B, 32B, 64B, 128B
lda/ldb = recommended, recommended + padding
smem_base_offset = 0, 16B, 32B, ... 至一个完整候选周期
```

优先构造逻辑矩阵内容和总字节数相同、仅物理映射不同的 pairwise case。

### 判读

- 有规律的整数倍退化和地址周期性支持 bank/fabric 冲突解释；
- 只有某种 swizzle 下出现退化，说明 descriptor layout 与 operand fetch 组织耦合；
- 无明显变化只能说明测试范围内 bank pattern 不是瓶颈，不能证明内部没有 bank；
- 结果命名为 effective bank parallelism，不直接写成“每周期读取 X 个物理 bank”。

## 4.4 `03_ldshared_contention`

### 要回答的问题

- 普通 LSU `ld.shared` 是否与 MMA operand fetch 共享关键资源；
- 竞争发生在 bank array、数据 fabric、仲裁还是更上层的 warp issue；
- 两条路径能否同时达到各自单独运行时的吞吐。

### 必要控制组

```text
MMA only
ld.shared only
MMA + register-only interference
MMA + ld.shared interference
```

`register-only interference` 使用与 `ld.shared` 干扰 warp 接近的指令数量和 active warp 数，用于排除 scheduler 与 occupancy 影响。

`ld.shared` 干扰分为：

```text
ldshared_bank_relation = same_pattern, disjoint_pattern, shifted_pattern
ldshared_bytes_per_iter = 0, 128, 256, 512, ...
active_interference_warps = 1, 2, 3
```

干扰 load 的结果必须进入不可删除的寄存器归约，最终写出一个校验值，防止编译器消除。

### 判读

- 若增加 `ld.shared` 流量后 MMA 吞吐单调下降，说明两者共享某个关键资源；
- same-pattern 比 disjoint-pattern 干扰更强，支持 bank-level 竞争；
- same 和 disjoint 同样干扰，可能是公共 fabric、仲裁或总端口竞争；
- register-only 也造成相同退化时，应先归因于 warp scheduling/issue，而不是 SMEM 端口；
- 若归一化吞吐近似满足

```text
B_mma / B_mma_only + B_ld / B_ld_only ~= 1
```

  则支持共享饱和资源模型，但不能据此宣称物理端口完全相同；
- 没有干扰也不能立即证明端口独立，应先确认 MMA 或 `ld.shared` 单独运行时已经饱和目标资源。

## 4.5 `04_inflight_window`

### 要回答的问题

- 独立 MMA 数量增加到多少后 cycles/MMA 不再下降；
- 同时保留多少个未完成 group 后不再提高吞吐；
- completion latency 需要多大的独立工作窗口才能隐藏。

### 方法

分别扫描两个维度：

```text
batch_size = 1, 2, 4, 8, 16, 32, 64
outstanding_groups = 1, 2, 3, ... 到 ISA/实现允许的范围
```

使用 independent D；若 TMEM 容量不足，使用 alternating D，但必须单独标记。尽可能轮换 A/B descriptor，另外保留相同 A/B descriptor 的对照组。

需要区分：

```text
多条 MMA -> 单次 commit -> wait
多组 MMA -> 每组 commit -> 延后 wait
```

### 判读

- batch-size 拐点给出隐藏固定完成延迟所需的有效窗口；
- outstanding-groups 拐点给出异步 group/scoreboard 路径的有效容量；
- 两个拐点都可能同时受 Tensor Core pipeline、collector、scoreboard、TMEM D 依赖和 commit queue 影响；
- 最终报告 `effective independent in-flight MMA window`，不直接命名为 operand collector depth。

## 4.6 `05_tmem_address_pattern`

### 要回答的问题

- TMEM D 地址复用是否造成 RAW/WAW 类限制；
- D base column 和 stride 是否表现出周期性冲突；
- 128、256、512 columns 配置的吞吐与 occupancy 是否不同；
- `input_d=1` 的 accumulator 读改写是否增加限制。

### 方法

固定 shape、SMEM layout 和 batch size，扫描：

```text
d_mode = same_d, alternating_d, independent_d
input_d = 0, 1
d_base_column = 0, 1, 2, 4, 8, ...
d_column_stride = 1, 2, 4, 8, ...
tmem_columns = 128, 256, 512
d_tiles_per_cta = 1, 2, 4
```

同时记录 resident CTA 数。资源限制导致 launch 失败时保留失败配置和 CUDA error，不静默跳过。

### 判读

- same-D 慢于 independent-D，说明 accumulator 地址依赖参与限制；
- `input_d=1` 明显慢于 `input_d=0`，说明读旧 D 或 RAW 链存在实际成本；
- D base/stride 出现稳定周期性，支持 TMEM 地址映射或内部并行度限制；
- columns 增加后性能下降时，必须先区分 occupancy 变化与单 CTA accumulator path 变化；
- 只能报告有效 accumulator update 吞吐与地址周期性，不能用完整 D tile 逻辑字节数直接推导物理 TMEM bank 写带宽。

## 5. 第二层：`06_shape_layout_sensitivity`

第一层确定可靠的 batch size、D 策略和 wait 方式后，再系统比较配置：

```text
shape = m128n64k16, m128n128k16, m128n256k16
dtype = fp16, bf16
swizzle = 128B, 64B, 32B, none
alignment = 16B, 32B, 64B, 128B
lda/ldb = recommended, recommended + padding
tmem_columns = 128, 256, 512
```

### 要回答的问题

- 哪些配置处于 compute limit；
- 哪些配置处于 SMEM operand supply limit；
- layout 敏感性是否能由第一层发现的 bank/fabric 规律解释；
- 现有 512-column 微基准能否代表目标 GEMM 的 128-column 配置。

如果第一层没有支持某个物理结构假设，本节不得仅凭 shape 性能差异重新宣称该结构存在。

## 6. 与 GEMM stage model 的关系

`thor_sm110_gemm_stage_model.md` 中的计算吞吐参数 \(P_C\) 应优先由同构配置给出：

```text
FP16/BF16
M128N128K16 atom
每 K-stage 4 条 MMA
128-column TMEM
64 KiB SMEM
grid 压力匹配 tc3
```

模型输入应使用：

- `00_instruction_baseline` 得到的延迟与同步固定成本；
- `01_effective_smem_ingress` 得到的稳态 MMA 吞吐与有效 operand 供数率；
- `03_ldshared_contention` 判断 epilogue/shared load 是否可能和 MMA 竞争；
- `04_inflight_window` 判断实际 mainloop 是否有足够在途工作隐藏完成延迟；
- `05_tmem_address_pattern` 校准 128-column 与现有 512-column 测试的差异。

只有当 \(P_C\) 随 layout、TMEM columns、在途窗口或 `ld.shared` 干扰系统变化时，才把对应因素展开成 stage model 的二级项。

## 7. 最小执行顺序

建议按以下顺序实现和运行：

1. `00_instruction_baseline`：分开 completion latency、同步成本和稳态吞吐；
2. `01_effective_smem_ingress`：取得最大有效 SMEM operand bytes/cycle；
3. `02_smem_bank_pattern`：检查 swizzle、alignment 和地址模周期；
4. `03_ldshared_contention`：判断普通 LSU 与 MMA operand path 的竞争关系；
5. `04_inflight_window`：估计有效独立在途 MMA 窗口；
6. `05_tmem_address_pattern`：研究 accumulator 地址周期性和 TMEM column 压力；
7. `06_shape_layout_sensitivity`：用前述标定结果解释完整配置矩阵。

前五项是硬件路径标定。第六项中 TMEM 物理 bank 数最难从软件侧唯一反演，因此只在已有稳定基线后进行。最后一项用于形成对 GEMM 配置选择真正有用的结论。
