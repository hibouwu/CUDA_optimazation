# Thor/SM110 性能与带宽资源总表（最终版）

> 报告版本：2026-08-21  
> 硬件作用域：`thor_t5000_sm110_20sm`，20 SM，MAXN  
> 每周期归一化时钟：1.575 GHz GPC/SM-core clock snapshot  
> 当前模型代码：`f06f2cd917a4cb23806b5e1be06120be9152ed7b`  
> 当前模型完成状态：`false`

## 1. 报告目的

本报告把 Thor/SM110 GEMM 建模中已经采集的 compute、memory、TMA、TMEM、
epilogue 与 duplex 资源统一整理为一张五列表。表中保留完整 `Resource` ID、
实测中位 rate、实测每周期 rate，以及能够与该资源保持单位和作用域一致的理论或
条件上界。

本表是 **resource capacity ledger**，不是完整 GEMM 性能排行榜，也不是当前模型
已经闭合的 integrated envelope。完整 GEMM observation、经验理想包络与条件上界
仍按三层模型分别解释。

## 2. 列定义与选择规则

- **实测 Median rate**：对应 case 的 10-trial 中位数。2026-08-14 composite
  closure 行来自结果提交 `ba651f0ebddd0983ceca5b352e65aa7ed5b7f32c`；
  2026-08-17 parameter supplement 行来自 GPU 数据提交
  `aa845dd9e70e2c541ae3a7d5293bf8de4bd55092`。
- **实测 Per cycle**：按 `实测 Median rate / 1.575 GHz` 归一化。`/GPU` 表示
  整卡聚合或 GPU-wide shared resource；只有 Resource 明确包含 `.per_sm` 时使用
  `/SM`。
- **理论 Median rate**：为满足用户指定表头而保留的名称。理论值没有 trial
  分布，因此这里的 “Median” 实际指模型选定的 `specified_upper`、
  `derived_upper` 或 `profiler_model_peak`，不是统计中位数。
- **理论 Per cycle**：按同一 1.575 GHz clock snapshot 对理论 rate 归一化。
- 用户指定的五列表没有 `Case` 列；仅对三个共享同一 Resource ID 的 NVFP4
  epilogue case，在 Resource 单元格后附加 `(constant)`、`(normal)`、
  `(outlier)` qualifier，以避免丢失独立 case。
- 理论列中的 `—`：当前没有与该 Resource 在资源语义、精度、shape、作用域和
  单位上可直接比较的可信理论 capacity。不得用邻近精度、不同 topology 或实测
  最快值补齐。实测列中的 `—` 表示该行是理论-only Resource。
- GB/s 使用十进制 `10^9 B/s`；TFLOP/s、TOP/s 分别使用 `10^12 FLOP/s`、
  `10^12 OP/s`。

理论值标记：

- `*`：产品级 aggregate dense FP8/FP4 数字到具体 PTX 精度合同的**条件映射**。
  产品资料不拆分 E4M3/E5M2，也没有明确 FP4 encoding；FP4 数值仅映射到当前
  `NVFP4 block16` 合同，不推广到 raw E2M1 或 MXFP4。
- `†`：从 sparse 产品数字按 2:1 稀疏倍率推导的 dense 条件上界。FP16/BF16
  来自 sparse FP16；S8/U8 来自未区分 signed/unsigned 的 sparse INT8。
- `‡`：273 GB/s 是 GPU-wide shared LPDDR5X **总带宽**，因此只列为
  `hbm.total`。它不是两个可同时达到的独立 read/write roof，也不能直接赋给
  `hbm.read`、`hbm.write`、TMA engine 或 `hbm.duplex.proxy`。
- `§`：Nsight Compute profiler model peak，不是实测 sustained rate，也不是
  已证明的 read/write joint capacity。

## 3. 性能与带宽总表

| **Resource** | **实测 Median rate** | **实测 Per cycle** | **理论 Median rate** | **理论 Per cycle** |
| --- | ---: | ---: | ---: | ---: |
| `tensor.bf16.m128n128` | 256.984 TFLOP/s | 163.164 kFLOP/cycle/GPU | 258.500 TFLOP/s† | 164.127 kFLOP/cycle/GPU† |
| `tensor.bf16.m128n256` | 229.422 TFLOP/s | 145.665 kFLOP/cycle/GPU | 258.500 TFLOP/s† | 164.127 kFLOP/cycle/GPU† |
| `tensor.bf16.m128n64` | 171.052 TFLOP/s | 108.604 kFLOP/cycle/GPU | 258.500 TFLOP/s† | 164.127 kFLOP/cycle/GPU† |
| `tensor.e2m1.m128n128` | 513.988 TFLOP/s | 326.341 kFLOP/cycle/GPU | — | — |
| `tensor.e2m1.m128n256` | 516.044 TFLOP/s | 327.647 kFLOP/cycle/GPU | — | — |
| `tensor.e2m1.m128n64` | 342.059 TFLOP/s | 217.181 kFLOP/cycle/GPU | — | — |
| `tensor.e2m3.m128n128` | 411.739 TFLOP/s | 261.422 kFLOP/cycle/GPU | — | — |
| `tensor.e2m3.m128n256` | 413.059 TFLOP/s | 262.260 kFLOP/cycle/GPU | — | — |
| `tensor.e2m3.m128n64` | 342.041 TFLOP/s | 217.169 kFLOP/cycle/GPU | — | — |
| `tensor.e3m2.m128n128` | 411.739 TFLOP/s | 261.422 kFLOP/cycle/GPU | — | — |
| `tensor.e3m2.m128n256` | 413.059 TFLOP/s | 262.260 kFLOP/cycle/GPU | — | — |
| `tensor.e3m2.m128n64` | 342.019 TFLOP/s | 217.155 kFLOP/cycle/GPU | — | — |
| `tensor.e4m3.m128n128` | 411.733 TFLOP/s | 261.418 kFLOP/cycle/GPU | 517.000 TFLOP/s* | 328.254 kFLOP/cycle/GPU* |
| `tensor.e4m3.m128n256` | 413.056 TFLOP/s | 262.258 kFLOP/cycle/GPU | 517.000 TFLOP/s* | 328.254 kFLOP/cycle/GPU* |
| `tensor.e4m3.m128n64` | 342.126 TFLOP/s | 217.223 kFLOP/cycle/GPU | 517.000 TFLOP/s* | 328.254 kFLOP/cycle/GPU* |
| `tensor.e5m2.m128n128` | 464.766 TFLOP/s | 295.090 kFLOP/cycle/GPU | 517.000 TFLOP/s* | 328.254 kFLOP/cycle/GPU* |
| `tensor.e5m2.m128n256` | 458.843 TFLOP/s | 291.329 kFLOP/cycle/GPU | 517.000 TFLOP/s* | 328.254 kFLOP/cycle/GPU* |
| `tensor.e5m2.m128n64` | 342.068 TFLOP/s | 217.186 kFLOP/cycle/GPU | 517.000 TFLOP/s* | 328.254 kFLOP/cycle/GPU* |
| `tensor.fp16.m128n128` | 205.871 TFLOP/s | 130.712 kFLOP/cycle/GPU | 258.500 TFLOP/s† | 164.127 kFLOP/cycle/GPU† |
| `tensor.fp16.m128n256` | 206.530 TFLOP/s | 131.130 kFLOP/cycle/GPU | 258.500 TFLOP/s† | 164.127 kFLOP/cycle/GPU† |
| `tensor.fp16.m128n64` | 171.030 TFLOP/s | 108.590 kFLOP/cycle/GPU | 258.500 TFLOP/s† | 164.127 kFLOP/cycle/GPU† |
| `tensor.mxfp4.m128n128` | 1027.975 TFLOP/s | 652.683 kFLOP/cycle/GPU | — | — |
| `tensor.mxfp4.m128n256` | 1032.093 TFLOP/s | 655.297 kFLOP/cycle/GPU | — | — |
| `tensor.mxfp4.m128n64` | 684.217 TFLOP/s | 434.423 kFLOP/cycle/GPU | — | — |
| `tensor.nvfp4.m128n128` | 1027.975 TFLOP/s | 652.683 kFLOP/cycle/GPU | 1035.000 TFLOP/s* | 657.143 kFLOP/cycle/GPU* |
| `tensor.nvfp4.m128n256` | 1032.093 TFLOP/s | 655.297 kFLOP/cycle/GPU | 1035.000 TFLOP/s* | 657.143 kFLOP/cycle/GPU* |
| `tensor.nvfp4.m128n64` | 684.342 TFLOP/s | 434.503 kFLOP/cycle/GPU | 1035.000 TFLOP/s* | 657.143 kFLOP/cycle/GPU* |
| `tensor.s8.m128n128` | 513.983 TOP/s | 326.338 kOP/cycle/GPU | 517.500 TOP/s† | 328.571 kOP/cycle/GPU† |
| `tensor.s8.m128n256` | 516.047 TOP/s | 327.649 kOP/cycle/GPU | 517.500 TOP/s† | 328.571 kOP/cycle/GPU† |
| `tensor.s8.m128n64` | 342.184 TOP/s | 217.260 kOP/cycle/GPU | 517.500 TOP/s† | 328.571 kOP/cycle/GPU† |
| `tensor.tf32.m128n128` | 128.494 TFLOP/s | 81.584 kFLOP/cycle/GPU | — | — |
| `tensor.tf32.m128n256` | 114.711 TFLOP/s | 72.833 kFLOP/cycle/GPU | — | — |
| `tensor.tf32.m128n64` | 85.525 TFLOP/s | 54.302 kFLOP/cycle/GPU | — | — |
| `tensor.u8.m128n128` | 513.988 TOP/s | 326.341 kOP/cycle/GPU | 517.500 TOP/s† | 328.571 kOP/cycle/GPU† |
| `tensor.u8.m128n256` | 516.047 TOP/s | 327.649 kOP/cycle/GPU | 517.500 TOP/s† | 328.571 kOP/cycle/GPU† |
| `tensor.u8.m128n64` | 342.175 TOP/s | 217.254 kOP/cycle/GPU | 517.500 TOP/s† | 328.571 kOP/cycle/GPU† |
| `hbm.read` | 253.588 GB/s | 161.008 B/cycle/GPU | — | — |
| `hbm.total` | — | — | 273.000 GB/s‡ | 173.333 B/cycle/GPU‡ |
| `hbm.write` | 201.158 GB/s | 127.719 B/cycle/GPU | — | — |
| `l2.read` | 1505.112 GB/s | 955.626 B/cycle/GPU | 1612.800 GB/s§ | 1024.000 B/cycle/GPU§ |
| `l2.write` | 545.416 GB/s | 346.296 B/cycle/GPU | 806.400 GB/s§ | 512.000 B/cycle/GPU§ |
| `tma.hbm` | 185.509 GB/s | 117.784 B/cycle/GPU | — | — |
| `tma.hbm.diagnostic.serial32k` | 261.556 GB/s | 166.067 B/cycle/GPU | — | — |
| `tma.hbm.inflight4` | 259.193 GB/s | 164.567 B/cycle/GPU | — | — |
| `tma.hbm.payload_4k` | 97.886 GB/s | 62.150 B/cycle/GPU | — | — |
| `tma.hbm.payload_8k` | 160.436 GB/s | 101.864 B/cycle/GPU | — | — |
| `tma.hbm.payload_16k` | 250.959 GB/s | 159.339 B/cycle/GPU | — | — |
| `tma.hbm.payload_32k` | 261.746 GB/s | 166.188 B/cycle/GPU | — | — |
| `tma.hbm.payload_64k` | 263.974 GB/s | 167.602 B/cycle/GPU | — | — |
| `tma.smem_ingress.diagnostic.serial32k.per_sm` | 68.615 GB/s/SM | 43.565 B/cycle/SM | — | — |
| `tma.smem_ingress.per_sm` | 193.366 GB/s/SM | 122.772 B/cycle/SM | — | — |
| `tma.smem_ingress.per_sm.inflight4` | 129.398 GB/s/SM | 82.157 B/cycle/SM | — | — |
| `tma.smem_ingress.per_sm.payload_4k` | 12.421 GB/s/SM | 7.886 B/cycle/SM | — | — |
| `tma.smem_ingress.per_sm.payload_8k` | 23.328 GB/s/SM | 14.812 B/cycle/SM | — | — |
| `tma.smem_ingress.per_sm.payload_16k` | 41.595 GB/s/SM | 26.410 B/cycle/SM | — | — |
| `tma.smem_ingress.per_sm.payload_32k` | 68.506 GB/s/SM | 43.496 B/cycle/SM | — | — |
| `tma.smem_ingress.per_sm.payload_64k` | 101.502 GB/s/SM | 64.446 B/cycle/SM | — | — |
| `l2.duplex.r3_w1` | 1672.986 GB/s | 1062.213 B/cycle/GPU | — | — |
| `l2.duplex.r4_w1` | 1413.188 GB/s | 897.262 B/cycle/GPU | — | — |
| `l2.duplex.r6_w1` | 1763.075 GB/s | 1119.413 B/cycle/GPU | — | — |
| `l2.duplex.r8_w1` | 1700.164 GB/s | 1079.470 B/cycle/GPU | — | — |
| `l2.duplex.r12_w1` | 1640.688 GB/s | 1041.706 B/cycle/GPU | — | — |
| `l2.duplex.r16_w1` | 1610.920 GB/s | 1022.806 B/cycle/GPU | — | — |
| `l2.duplex.r24_w1` | 1580.044 GB/s | 1003.202 B/cycle/GPU | — | — |
| `l2.duplex.r27_w4` | 1667.613 GB/s | 1058.802 B/cycle/GPU | — | — |
| `l2.duplex.r27_w8` | 1476.860 GB/s | 937.689 B/cycle/GPU | — | — |
| `l2.duplex.r27_w16` | 1140.100 GB/s | 723.873 B/cycle/GPU | — | — |
| `l2.duplex.r32_w1` | 1564.444 GB/s | 993.298 B/cycle/GPU | — | — |
| `l2.duplex.r48_w1` | 1551.395 GB/s | 985.013 B/cycle/GPU | — | — |
| `l2.duplex.r64_w1` | 1543.253 GB/s | 979.843 B/cycle/GPU | — | — |
| `l2.duplex.r96_w1` | 1539.025 GB/s | 977.158 B/cycle/GPU | — | — |
| `hbm.duplex.proxy.r1_w1` | 192.480 GB/s | 122.210 B/cycle/GPU | — | — |
| `hbm.duplex.proxy.r1_w2` | 182.179 GB/s | 115.669 B/cycle/GPU | — | — |
| `hbm.duplex.proxy.r1_w4` | 162.902 GB/s | 103.430 B/cycle/GPU | — | — |
| `hbm.duplex.proxy.r2_w1` | 201.940 GB/s | 128.216 B/cycle/GPU | — | — |
| `hbm.duplex.proxy.r3_w8` | 176.530 GB/s | 112.083 B/cycle/GPU | — | — |
| `hbm.duplex.proxy.r9_w32` | 167.695 GB/s | 106.473 B/cycle/GPU | — | — |
| `hbm.duplex.proxy.r17_w64` | 168.156 GB/s | 106.766 B/cycle/GPU | — | — |
| `tmem.readback` | 34270.415 GB/s | 21,758.994 B/cycle/GPU | — | — |
| `tmem.readback.x16.warps1` | 1343.479 GB/s | 853.003 B/cycle/GPU | — | — |
| `tmem.readback.x8.warps1` | 686.033 GB/s | 435.576 B/cycle/GPU | — | — |
| `tmem.readback.x8.warps4` | 19768.340 GB/s | 12,551.327 B/cycle/GPU | — | — |
| `tmem.scale_ingress` | 239.259 GB/s | 151.910 B/cycle/GPU | — | — |
| `epilogue.nvfp4_requant (constant)` | 2.460 Gelement/s | 1.562 element/cycle/GPU | — | — |
| `epilogue.nvfp4_requant (normal)` | 2.460 Gelement/s | 1.562 element/cycle/GPU | — | — |
| `epilogue.nvfp4_requant (outlier)` | 2.460 Gelement/s | 1.562 element/cycle/GPU | — | — |

## 4. 当前证据状态

总表共有 86 个 Resource/case 行，其中 85 行有实测值，另有一行理论-only
`hbm.total`：

- 2026-08-14 composite closure 的 54 个 case 全部保留；三个
  `epilogue.nvfp4_requant` distribution case 使用 Resource 后缀 qualifier 区分。
- 2026-08-17 parameter supplement 增加 10 个 TMA payload Resource、14 个
  ratio-qualified `l2.duplex` Resource 和 7 个 `hbm.duplex.proxy` Resource，
  共 31 行。

需要分别理解两批证据：

1. 2026-08-14 行保留为通过当时逐 case 门禁的历史实测；旧 composite report
   同时记录了基础 campaign 的 `oc3_event_cnt +179` warning，且用 current
   语义重审时整体 `audit pass=false`。因此这些行不能整体升级成 current
   integrated closure。
2. 2026-08-17 的 TMA payload 和 hot-L2 duplex 行可以被 current importer
   重放。本报告生成时已用 current `evidence_import.py` 对两棵结果树及 canonical
   auditor 做实际重放：TMA 10/10、duplex 21/21，得到
   `measured_sustained=10`、`measured_joint=21`、
   `closure_qualified=31`。cold 行的资源名明确包含 `.proxy`，只证明
   external-read/L2-write-path proxy，不证明 physical external write bytes。
3. 当前仍缺 physical `hbm.duplex`、部分 block-scale 小 payload、26 个 exact
   TMA schedule/precision pair 和 closure-qualified Thor causal profile，故当前
   integrated empirical envelope 的合法输出仍是 `insufficient_evidence`。

## 5. 为什么部分理论列必须为空

- raw E2M1、E2M3、E3M2、MXFP4 和 TF32 当前没有可进入相同实现域的独立 strict
  compute upper；不能把 NVFP4 或 aggregate FP8 产品数字平移过去。
- `hbm.read` 与 `hbm.write` 有独立实测，但当前指定上界是共享的 `hbm.total`；
  为防止形成两个独立方向 roof 的错误语义，本表不把 273 GB/s 复制到两行。
- TMA 表中既有 `/GPU` 的 cold aggregate，也有 `/SM` 的 hot ingress；目前没有
  TMA engine 的同作用域理论 capacity。共享 L2/HBM roof 不是 per-SM TMA roof。
- `l2.duplex.*` 统计 read+write total issued bytes。独立的 L2 read/write model
  peak 不能被复制成一个已证明的 joint capacity，因此理论列为空。
- `hbm.duplex.proxy.*` 混合了 external-read proxy 和 L2 write issue，不能与
  273 GB/s physical LPDDR 总带宽直接作同量比较。
- TMEM readback、scale ingress 和 fused epilogue 只有特定指令/工作量合同下的
  measured service rate；没有公开或已证明的同口径理论峰值。

## 6. 数据来源与复核入口

- 历史完整数值表：
  [`../thor_sm110_current_model_replay.md`](../thor_sm110_current_model_replay.md)
- current 模型重放：
  [`appendices/current_model_replay.md`](appendices/current_model_replay.md)
- current coverage：
  [`model/08_current_coverage_and_gaps.md`](model/08_current_coverage_and_gaps.md)
- 基础理论与历史 profile：
  [`../../../scripts/sm110_gemm_model/profiles/capacities.json`](../../../scripts/sm110_gemm_model/profiles/capacities.json)
- 硬件/时钟作用域：
  [`../../../scripts/sm110_gemm_model/profiles/thor_sm110.json`](../../../scripts/sm110_gemm_model/profiles/thor_sm110.json)
- 实验到 runner/auditor/result 的映射：
  [`appendices/microbenchmark_sources.md`](appendices/microbenchmark_sources.md)

2026-08-17 parameter supplement 的 Git object 路径：

```text
aa845dd9e70e2c541ae3a7d5293bf8de4bd55092:
  results/sm110_tma_payload_campaign/
    thor-t5000-parameter-plots-maxn-20260817-i-tma-payload/summary.json
  results/sm110_memory_duplex_campaign/
    thor-t5000-parameter-plots-maxn-20260817-i-memory-duplex/summary.json
```

两个 summary 的 SHA-256 分别为：

```text
TMA payload    2feba5979d623f44bf27da943a0d51e8b36f4506468af10723de9ee953e57604
Memory duplex ac8e24e9d7c0585c6f732559e98f638d3c96c0b460a06e6bddafd4b5dcd4310d
```

结果分支最终 HEAD 为 `78e09488c51b3d81ac2ec9596630f238af11ad91`；它相对
数值提交 `aa845dd9...` 只增加 orchestrator completion log，不改变表中数值。

每周期值只做单位换算：

```text
rate_per_cycle = rate_per_second / 1.575e9
```

该换算不会提升证据等级，也不能把 `measured_sustained`、proxy 或 profiler model
peak 改写为物理结构证明。

## 7. 明确不包含的对象

- 完整 GEMM candidate/cuBLAS observation：它属于模型第三层，不是 Resource
  capacity；历史 FP16 N=2048 结果见
  [`appendices/historical_results.md`](appendices/historical_results.md)。
- CP/MMA overlap、DSMEM topology 等 diagnostics：它们未进入 current capacity
  selector，不能与正式 capacity 行混排。
- 从图像像素反推的数据、缺少 raw bundle/hash/auditor 的 README 快照，以及仅有
  static SASS presence 的 runtime 未测 case。
