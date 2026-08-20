# EXP-02：GPU-wide L2 物理边界

## 1. 研究问题

Thor/SM110 共享 L2 fabric 的 read/write rate upper、L2 capacity 和独立 sustained 点分别是什么作用域？

## 2. 对应模型参数

- strict `l2.read`；
- strict `l2.write`；
- `l2_capacity_bytes`；
- historical/diagnostic `l2.read` 与 `l2.write` sustained points。

## 3. 当前条件上界

当前模型使用：

| 参数 | 值 | scope | evidence |
| --- | ---: | --- | --- |
| L2 read model peak | 1024 B/cycle/GPU | GPU-wide shared | `profiler_model_peak` |
| L2 write model peak | 512 B/cycle/GPU | GPU-wide shared | `profiler_model_peak` |
| L2 capacity | 32 MiB | GPU-wide | `device_record` |

在 1.575 GHz snapshot 下：

\[
U_{\mathrm{L2,R}}=1.6128\ \mathrm{TB/s},
\qquad
U_{\mathrm{L2,W}}=0.8064\ \mathrm{TB/s}.
\]

二者绝不乘 `sm_count`。

## 4. 实验面

历史 L2 microbenchmark 包含 unique read、write/store path 和 working-set sweep；统一 component campaign 还包含 16 MiB hot working set 的全 GPU read/write case。

这些 sustained point 用于诊断和旧 empirical calibration；迁移后的 current empirical memory layer 使用 EXP-04 的 ratio-qualified `l2.duplex`。

容量阶跃图不是一组独立的新 capacity。同一 microbenchmark 家族的 16 MiB
baseline `read-unique`/`write-unique` 代表点已经分别由
`l2_read_unique_measured=946.701 B/cycle/GPU` 和
`l2_write_unique_measured=299.373 B/cycle/GPU` 进入 historical/base
capacity 表；validation SVG 是重复采样中位曲线，额外支持约 32 MiB capacity
transition，不应假装 baseline 数字就是 SVG 某一个点的精确 raw value。不得从
SVG 像素反推或把同一实验家族的曲线点再次导入并与代表点重复计数。

## 5. 接受门禁

- working set 与 residency 合同明确；
- function-scoped `LDG.E.128` / `STG.E.128` SASS；
- `%globaltimer` 全网格区间；
- 20-SM coverage；
- NCU model peak 与 clock snapshot 的换算来源可定位；
- L2 capacity 的 source row/locator 可机械验证。

## 6. 进入模型

strict 层对最低 read/write 工作分别使用：

\[
T_{\mathrm{L2,R}}^{\mathrm{LB}}
=Q_{\mathrm{L2,R}}^{\mathrm{LB}}/U_{\mathrm{L2,R}},
\]

\[
T_{\mathrm{L2,W}}^{\mathrm{LB}}
=Q_{\mathrm{L2,W}}^{\mathrm{LB}}/U_{\mathrm{L2,W}}.
\]

当前没有可引用的 joint outer region 证明 \(R/1024+W/512\le1\)，所以严格层允许二者理想重叠；EXP-04 measured duplex 仍只是经验内点。

`hot_l2` workload 还要求逻辑输入工作集不超过 `l2_capacity_bytes`。

## 7. 不能证明什么

- 1024/512 B/cycle 不是 per-SM rate；
- unique read/store sustained point 不是 physical upper；
- 能放入 32 MiB 不证明具体 schedule 的所有请求命中 L2；
- 独立 read/write peak 不证明同时可达；
- L2 peak 不替代 per-SM TMA→SMEM ingress。

## 8. 源码与工件

- L2 说明：[L2throughtput/README.md](../../../../microbench/L2throughtput/README.md)
- 历史结果：[l2_throughput.csv](../../../../microbench/L2throughtput/results/l2_throughput.csv)
- 容量阶跃图：[l2_capacity_staircase.svg](../../../../microbench/L2throughtput/plots/l2_capacity_staircase.svg)
- 并发饱和图：[l2_concurrency_saturation.svg](../../../../microbench/L2throughtput/plots/l2_concurrency_saturation.svg)
- memory path source：[memory_path_bandwidth.cu](../../../../microbench/14_memory_path_bandwidth/memory_path_bandwidth.cu)
- component runner：[run_component_campaign.py](../../../../microbench/sm110_gemm_component_campaign/run_component_campaign.py)
- profile：[thor_sm110.json](../../../../scripts/sm110_gemm_model/profiles/thor_sm110.json)
- capacity declarations：[capacities.json](../../../../scripts/sm110_gemm_model/profiles/capacities.json)
