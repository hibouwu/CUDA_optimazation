# tcgen05 MMA 硬件路径标定与配置敏感性实验计划

## 1. 目标

本组实验分为三层：

1. 先验证 PTX、descriptor、TMEM 地址和同步序列正确；
2. 再标定 `tcgen05.mma` 的 ISA 可见行为与有效硬件参数；
3. 最后研究 shape、SMEM layout、TMEM columns 和并发访存对 GEMM 的影响。

优先回答：

1. `tcgen05.mma` 的隔离完成延迟和稳态边际成本分别是多少；
2. ISA-visible collector 能减少多少 SMEM operand fetch，其复用边界是什么；
3. collector discard 条件下，SMEM 到 Tensor Core 的最大有效供数率是多少；
4. `tcgen05.mma` 对 SMEM layout/address pattern 的敏感性如何；
5. 普通 LSU `ld.shared` 是否与 MMA operand fetch 竞争关键资源；
6. TMEM D 地址复用、alias、column 占用和 `input_d` 如何限制流水线；
7. 在合法 TMEM 容量内，需要多大的 D reuse distance 才能隐藏完成延迟。

## 2. 结论边界

| 希望了解的量 | 微基准实际报告的量 | 不允许直接宣称 |
| --- | --- | --- |
| SMEM→Tensor Core 端口宽度 | collector-discard 下最大有效 operand bytes/cycle | 物理端口宽度 |
| 每周期读取多少 SMEM bank | layout/address-pattern 敏感性和有效 bank 并行度 | 内部物理读 bank 数 |
| 每周期写入多少 TMEM bank | accumulator update 吞吐、alias 和地址周期性 | 物理 TMEM bank 数 |
| 是否与 LSU 完全共享端口 | `ld.shared` 与 MMA 的干扰及归一化吞吐曲线 | 两者物理连线完全相同 |
| operand collector 深度 | ISA-visible collector 行为和有效在途窗口 | 隐藏 collector/scoreboard entry 数 |

PTX 已暴露部分 collector 控制：默认 activation-stationary `tcgen05.mma` 可使用 A collector 的 `fill/use/lastuse/discard`；`tcgen05.mma.ws` 可使用 B collector `b0`–`b3`。这些 ISA-visible buffer 应直接测试，不通过 batch-size 拐点猜测。

`tcgen05.commit` 使 mbarrier 跟踪执行线程发出的所有先前 async-tcgen05 操作。连续 commit 对应累计 completion prefix，不等同于 `wgmma.commit_group` 的独立 group。因此本计划不再使用 `outstanding_groups` 推断 group queue 深度。

## 3. 建议目录结构

```text
microbench/mma_config/
  Docs/
    MicroBench.md
    ExperimentPlan.md

  common/
    tcgen05_helpers.cuh
    descriptor_builder.cuh
    validation.cuh
    timing.cuh
    result_writer.cuh

  00_validation/
  01_collector_protocol/
  02_latency_throughput/
  03_effective_smem_ingress/
  04_smem_layout_address/
  05_ldshared_contention/
  06_tmem_dependency/
  07_config_matrix/
```

每个实验目录包含：

```text
README.md
benchmark_src/
scripts/
plots/
```

`common/` 只保存不会隐藏实验控制变量的 PTX wrapper、descriptor builder、正确性检查、计时和 CSV 输出。每个实验保留独立 kernel，使实际指令序列可以直接审计。

## 4. 统一实验口径

### 4.1 环境与计时

所有结果记录：

- GPU 型号、compute capability、driver、CUDA toolkit 和 PTX ISA 版本；
- power mode、实际 SM clock、memory clock、温度和功耗；
- kernel launch 配置、静态/动态 SMEM、TMEM columns 和 resident CTA 数；
- warmup、repeat、迭代次数和 case 执行顺序；
- 单 CTA/低 grid 与全 SM 满载两种口径；
- PTX 和 SASS hash，确保比较时使用预期指令序列。

case 顺序必须随机化，并周期性插入固定 reference kernel。报告 median、p10 和 p90，避免热状态或 DVFS 漂移伪装成性能拐点。

默认 CTA 为 128 threads，默认 shape 为：

```text
m128n64k16
m128n128k16
m128n256k16
```

FP16 与 BF16 分开运行和绘图。

### 4.2 正确性门槛

任何性能数据进入图表前必须满足：

1. descriptor 组合符合当前 PTX ISA 的 swizzle、alignment 和 stride 限制；
2. A/B 使用非对称且可复现的数据，避免全零或全一掩盖 layout 错误；
3. 等待 MMA 完成后通过 `tcgen05.ld` 读回 D；
4. 与 CPU reference 比较，误差标准按 dtype 和累计 K 次数固定；
5. D tile 周围的 TMEM guard 区域保持不变；
6. kernel launch、执行和 deallocation 均无 CUDA error；
7. SASS 中目标 MMA、commit、wait 和干扰指令数量符合预期。

不满足上述任意条件的 case 只进入 `invalid_cases.csv`，不能进入性能统计。

### 4.3 统一变量

```text
Q                       timed region 内发射的 MMA 总数
collector_mode          discard, fill_use_lastuse
operand_address_mode    same, pingpong, rotating
independent_d_count     不重叠 D tile 数
d_reuse_distance        同一 D 两次使用之间的 MMA 数
d_tile_base_delta       相邻 D tile base 的 column 差
d_alias_class           none, partial, full
commit_interval         相邻 commit 之间的 MMA 数
pending_mbarriers       已 commit 但尚未 wait 的 mbarrier 数
```

注意：`pending_mbarriers` 表示累计 completion prefix 的数量，不表示独立硬件 group 数。

### 4.4 统一 CSV 字段

```text
experiment, case_id, valid, invalid_reason,
gpu, compute_capability, driver, cuda_version, ptx_version,
sm_clock_mhz, mem_clock_mhz, temperature_c, power_w,
dtype, m, n, k, Q, iterations, repeat,
collector_mode, operand_address_mode,
independent_d_count, d_reuse_distance,
commit_interval, pending_mbarriers, wait_polling_mode,
smem_layout, swizzle, alignment_bytes, lda, ldb, smem_base_offset,
tmem_columns, d_base_column, d_tile_base_delta, d_alias_class, input_d,
interference_mode, interference_ops_per_iter, interference_warps,
resident_ctas, elapsed_cycles, elapsed_us,
alpha_cycles, beta_cycles_per_mma,
logical_smem_bytes_per_mma, effective_smem_bytes_per_cycle,
tflops, poll_count, max_abs_error, guard_ok, sass_hash, notes
```

## 5. `00_validation`

### 5.1 目的

建立后续实验共用的合法 descriptor、TMEM 地址和同步模板，防止“跑得快但算错了”。

### 5.2 必测项

```text
每个 shape × dtype
每个计划使用的 swizzle/layout
每个合法 TMEM allocation size
input_d = 0, 1
collector = discard, fill/use/lastuse
```

每个 case 做单条 MMA 和短累计链，读回完整 D 并检查 guard。

### 5.3 输出

- `valid_descriptor_cases.csv`；
- `invalid_cases.csv`；
- 每种 wrapper 的 PTX/SASS 指令摘要；
- 每个 shape 的 TMEM footprint 和可容纳的最大不重叠 D tile 数。

对 M128、FP32 accumulator 和 512-column allocation，预期独立 D 上限为：

| Shape | D footprint | 最大不重叠 D 数 |
| --- | ---: | ---: |
| M128N64 | 64 columns | 8 |
| M128N128 | 128 columns | 4 |
| M128N256 | 256 columns | 2 |

若实际 dtype/layout 的 TMEM packing 不同，以 PTX layout 和验证结果为准，不硬编码上表。

## 6. `01_collector_protocol`

### 6.1 要回答的问题

- ISA-visible collector 的 fill/use/lastuse 成本；
- collector reuse 能减少多少 SMEM operand fetch 成本；
- collector 生命周期结束后，下一次 fill 是否出现额外 stall；
- activation-stationary A collector 与 weights-stationary B collector 行为是否不同。

### 6.2 基本序列

```text
discard × Q
fill -> lastuse
fill -> use × R -> lastuse
fill -> use × R -> discard
```

若实现 `tcgen05.mma.ws`，分别测试：

```text
b0 only
b0/b1 pingpong
b0/b1/b2/b3 rotating
```

所有序列必须遵守 PTX collector 生命周期。非法 use、重复 fill 或 use-after-lastuse 不作为性能 case。

### 6.3 控制变量

- 固定 shape、D reuse distance、SMEM layout 和 commit interval；
- 同时记录 same 与 pingpong operand address；
- `discard × Q` 作为后续 SMEM ingress 的默认基线；
- collector reuse case 不计算“每 MMA 完整读取 A/B”的物理带宽。

### 6.4 判读

- fill/use 明显快于 discard：collector 正在减少 operand fetch 或 descriptor 路径成本；
- b0–b3 rotating 优于单 buffer：多个 ISA-visible B collector 能提高 weights-stationary 重用窗口；
- collector 模式不变但地址模式影响吞吐：仍可能有 collector 之外的缓存、descriptor 或 SMEM 路径效应。

## 7. `02_latency_throughput`

### 7.1 要回答的问题

- 单条 MMA 到完成通知的隔离延迟；
- 长 batch 的稳态边际 cycles/MMA；
- `commit`、最终 drain 和 wait polling 对截距的贡献；
- D reuse distance 多大后吞吐收敛。

### 7.2 方法：批长度回归

对每个 shape 测：

```text
Q = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64
```

拟合：

```text
T(Q) = alpha + beta * Q
```

其中：

- `beta` 是长 batch 的边际稳态成本；
- `alpha` 包含 loop、commit、最终 drain 和 wait 的综合固定项；
- `Q=1` 单独报告为 forced-completion diagnostic，不用空 commit 结果简单相减。

### 7.3 D 地址设计

分别测试：

```text
same D, input_d=0
same D, input_d=1
legal independent D ring, input_d=0
legal independent D ring, input_d=1
```

独立 D ring 大小不得超过 `00_validation` 得到的上限。超过上限时继续增加 Q，但报告真实 `independent_d_count` 和 `d_reuse_distance`，不得称为 independent-D batch。

### 7.4 Completion tracking

扫描：

```text
commit_interval = 1, 2, 4, 8, ...
pending_mbarriers = 1, 2, 3, ... 在资源允许范围内
```

每个 commit 都标注其 prefix length。该实验只研究累计 completion prefix 和 mbarrier tracking 成本，不推断独立 async group queue 深度。

instrumented 版本可记录 `mbarrier.try_wait` poll count，但性能主结果必须来自不计数版本。

## 8. `03_effective_smem_ingress`

### 8.1 要回答的问题

- collector-discard 条件下最大有效 SMEM operand bytes/cycle；
- 不同 shape 是否更接近相同 bytes/cycle roofline 或 FLOP/cycle roofline；
- same/pingpong/rotating operand 地址是否改变结果。

### 8.2 固定条件

```text
collector_mode = discard
Q 使用 02 中已经进入线性稳态的范围
D reuse distance 使用该 shape 可实现的最大合法值
SMEM layout 使用 00 验证过的推荐配置
```

operand 地址模式：

```text
same
pingpong
rotating，在 SMEM 容量允许范围内
```

### 8.3 指标

对于 A/B 都来自 SMEM 的 BF16/FP16 MMA：

```text
logical_smem_bytes_per_mma = sizeof(dtype) * k * (m + n)
effective_smem_bytes_per_cycle = logical_smem_bytes_per_mma / beta
```

例如 M128N128K16 FP16/BF16：

```text
A = 4096 B
B = 4096 B
A+B = 8192 B/MMA
```

### 8.4 判读

- 多 shape 在相近 bytes/cycle 饱和：支持 operand path roofline；
- 多 shape 在相近 FLOP/cycle 饱和：更像 Tensor Core compute limit；
- same 比 pingpong/rotating 更快：存在地址复用效应，same-address 结果不能用于端口下界；
- 最大值只报告为有效供数率，不命名为物理端口宽度。

## 9. `04_smem_layout_address`

### 9.1 方法

只从 `00_validation` 的合法 descriptor 集合取 case，做成逻辑矩阵内容、总字节数和 MMA 序列相同的 pairwise 比较：

```text
swizzle = legal set
alignment = legal set
lda/ldb = legal set
smem_base_offset = 合法地址范围内的完整候选周期
```

所有 case 固定 collector discard、Q、D reuse distance 和 commit interval。

### 9.2 判读

- 只有整数倍退化加上稳定地址周期时，才提出 bank/fabric 冲突假设；
- 单纯 base-offset 敏感性报告为 address/partition sensitivity，不直接解释为 bank-conflict degree；
- 无变化只说明测试范围内 layout 不是当前瓶颈；
- 最终报告 effective bank parallelism 或相对退化率，不声明物理 bank/cycle。

## 10. `05_ldshared_contention`

### 10.1 要回答的问题

- `ld.shared` 与 MMA 是否竞争共享的 bank array、fabric、仲裁或 LSU/issue 资源；
- 干扰是否依赖 shared address pattern；
- 两条路径能否同时接近各自单独运行的吞吐。

### 10.2 控制组

固定 active warp 数，比较：

```text
MMA only
interference only
MMA + register ALU
MMA + predicated-off load wrapper
MMA + L1-hit ld.global
MMA + ld.shared
```

主 sweep 不改变 interference warp 数，通过每轮 load 数或 load/ALU 比例改变流量。active warp 数变化放入次级实验。

`ld.shared` 地址模式定义为软件可见的候选关系：

```text
same_candidate_bank_subset
shifted_candidate_bank_subset
full_32_bank_pattern
single/few_bank_hotspot
```

不得把这些名称解释为与 async-proxy MMA 逐周期真实 bank schedule 完全对齐。

### 10.3 实现要求

- 使用同一 CTA 中固定 warp 分工，并在 timed region 前同步开始；
- 干扰 load 进入不可删除的寄存器归约并最终写出校验值；
- generic proxy 写 SMEM 后，在 MMA async proxy 使用前执行正确的 proxy fence；
- 分别测单 CTA 和可控 resident CTA 数；
- 报告 MMA 与干扰路径各自完成的实际工作量。

### 10.4 判读

- `ld.shared` 特有退化大于 register 和其他 LSU 对照：支持 shared-memory 关键资源竞争；
- bank hotspot 比 full/distributed pattern 干扰更强：支持 bank-level 竞争；
- 所有 LSU load 都同样干扰：先考虑 LSU issue/dispatch；
- register ALU 也同样干扰：先考虑 warp scheduler/occupancy；
- 若两条路径已单独饱和且近似满足

```text
B_mma / B_mma_only + B_ld / B_ld_only ~= 1
```

  则支持共享饱和资源模型，但不证明物理端口完全相同。

## 11. `06_tmem_dependency`

### 11.1 要回答的问题

- D 的 same/partial/non-overlap 地址关系如何影响 MMA；
- `input_d=1` 的读改写链成本；
- D tile base 是否存在可重复的地址周期性；
- TMEM columns 对单 CTA 吞吐和 resident CTA 数的影响。

### 11.2 变量

```text
d_alias_class = full, partial, none
d_tile_base_delta = 0, 合法部分重叠值, N, 2N, ...
input_d = 0, 1
tmem_columns = 128, 256, 512 中能容纳目标 tile 的合法值
independent_d_count = 1 ... floor(tmem_columns / D_footprint)
d_reuse_distance = 1 ... independent_d_count
```

`d_tile_base_delta` 表示不同 MMA 的 D base column 差，不是 MMA tile 内部 stride。

### 11.3 判读

- `input_d=1` 与 same-D 组合变慢：支持 accumulator RAW 链限制；
- `input_d=0` 的 full/partial alias 仍变慢：支持 WAW/alias tracking 限制；
- non-overlap base 仍出现周期性：可以提出 TMEM partition/address mapping 假设；
- columns 增加后性能变化必须分解为单 CTA 变化和 resident CTA 变化；
- 不使用完整 D tile 逻辑字节数推导物理 TMEM bank 写带宽。

## 12. `07_config_matrix`

前述实验确定合法 descriptor、collector mode、稳态 Q 和 D reuse distance 后，再扫描：

```text
shape = m128n64k16, m128n128k16, m128n256k16
dtype = fp16, bf16
swizzle/alignment/lda/ldb = validated legal set
tmem_columns = legal set
collector_mode = discard, reuse protocol
operand_address_mode = same, pingpong
```

本节回答：

- 哪些配置处于 compute limit；
- 哪些配置处于 SMEM operand supply limit；
- layout 敏感性是否符合第 9 节观察到的模式；
- collector reuse 对真实 GEMM mainloop 有多少收益；
- 512-column 微基准能否代表目标 GEMM 的 128-column 配置。

第一层实验没有支持的物理结构假设，本节不得仅凭 shape 性能差异重新宣称其存在。

## 13. 与 GEMM stage model 的关系

`thor_sm110_gemm_stage_model.md` 中的计算吞吐参数 \(P_C\) 优先使用同构配置：

```text
FP16/BF16
M128N128K16 atom
每 K-stage 4 条 MMA
128-column TMEM
64 KiB SMEM
grid 压力匹配 tc3
```

模型分别使用：

- `02_latency_throughput` 的 `alpha`、`beta` 和 D reuse distance 曲线；
- `03_effective_smem_ingress` 的 collector-discard 有效供数率；
- `01_collector_protocol` 的 operand reuse 收益；
- `05_ldshared_contention` 的共享路径竞争证据；
- `06_tmem_dependency` 的 128/256/512-column 差异。

只有参数随这些变量系统变化时，才将对应因素展开为 stage model 的二级项。

## 14. 最小执行顺序与停止条件

1. `00_validation`：任何核心 shape 无法正确读回时停止，不运行性能实验；
2. `01_collector_protocol`：确认 ISA-visible collector 序列和复用收益；
3. `02_latency_throughput`：用回归分开固定项与稳态边际成本；
4. `03_effective_smem_ingress`：只在 collector discard 和合法 D ring 下计算有效供数率；
5. `04_smem_layout_address`：只扫描已验证的 descriptor；
6. `05_ldshared_contention`：加入 scheduler、LSU 和 shared-memory 对照；
7. `06_tmem_dependency`：显式标记 D alias 和 reuse distance；
8. `07_config_matrix`：汇总为可用于 GEMM 选型的结论。

如果某项实验无法同时控制 collector、D alias、descriptor legality 和 resident CTA 数，则只报告现象，不反演端口宽度、bank 数或隐藏队列深度。
