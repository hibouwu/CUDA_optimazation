# 02 符号、单位、workload 与硬件作用域

## 1. GEMM 数学语义

定义 \(M\) 为输出行数，定义 \(N\) 为输出列数，定义 \(K\) 为 reduction 维度，单位均为 element。定义 \(A,B\) 为输入矩阵，定义 \(C\) 为可选累加输入，定义 \(D\) 为输出：

\[
D=\alpha\operatorname{op}(A)\operatorname{op}(B)+\beta C.
\]

定义 \(\alpha\) 和 \(\beta\) 为无量纲标量。当前可执行 schedule 只覆盖 NN；当 \(\beta=0\) 时，最低工作量不读取 \(C\)。

## 2. 性能与频率单位

定义 \(W\) 为算术工作量，定义 \(T\) 为设备端计时区间，主性能变量为：

\[
P=\frac{W}{T}.
\]

浮点 workload 的单位为 FLOP/s；S8/U8 的单位为 OP/s。一次 multiply-add 计作一次乘法和一次加法，即 2 FLOP 或 2 OP。

定义 \(f_g\) 为与该证据同一运行区间、同一 clock domain 的 GPU 时钟；只为架构归一化展示定义：

\[
\Pi=\frac{P}{f_g}
\]

单位为 FLOP/cycle/GPU 或 OP/cycle/GPU。若证据明确为 per-SM，再定义：

\[
\Pi_{\mathrm{SM}}=\frac{P}{S f_g},
\]

其中 \(S\) 是参与服务的 SM 数。不同 clock domain、GPU-wide bus 和 per-SM 出口不能只凭每周期数值直接相加或比较。

## 3. Workload 合同

一个 `Workload` 至少冻结：

| 字段 | 定义 |
| --- | --- |
| `workload_id` | 问题和验证场景的稳定 ID |
| `m`, `n`, `k` | 对应 \(M,N,K\) |
| `precision_id` | 输入、累加器、输出、scale 和算术单位合同 |
| `transpose_a`, `transpose_b` | A/B 转置标志；当前 v1 只接受 false/false |
| `alpha`, `beta` | GEMM 标量 |
| `epilogue` | 当前只实现 `none` |
| `output_mode` | 当前只实现 `accumulator` |
| `residency` | `cold_hbm`、`hot_l2` 或 `compute_oracle` |
| `validation_split` | `exploratory`、`calibration` 或 `holdout` |
| `implementation_domain` | `tensor_core_classical` 或 `all_classical` |
| `timed_scope` | `device_kernel` 或 `device_kernel_plus_launch` |

`cold_hbm` 是通用外部 DRAM 冷入口名字；Thor T5000 的物理器件是 LPDDR5X，不表示使用 HBM 器件。

`hot_l2` 不是一个自由标签。模型要求已证明的 `l2_capacity_bytes`，并检查逻辑输入工作集不超过该容量；能放入 L2 是必要条件，但仍不自动证明具体 schedule 的所有请求命中 L2。

`compute_oracle` 表示输入已经处在 Tensor Core 可消费位置，只用于隔离 compute service，不代表完整 GEMM。

## 4. 精度合同

每个 `PrecisionSpec` 定义：

| 参数 | 单位 | 含义 |
| --- | --- | --- |
| `input_bytes` | B/element | 逻辑输入存储宽度 |
| `accumulator_bytes` | B/element | 累加器宽度 |
| `output_bytes` | B/element | 输出宽度 |
| `mma_k` | element/instruction | MMA atom 的 K 深度 |
| `compute_resource` | ID | 严格层算术资源 |
| `compute_work_unit` | `flop` 或 `operation` | 算术工作单位 |
| `input_scale_block` | element/scale | block scale 覆盖元素数 |
| `input_scale_bytes` | B/scale | scale 存储宽度 |

逻辑位宽、可执行 transport container 和 block-scale 物理布局是三个不同概念。FP6 可以具有 0.75 B/element 的逻辑值，但 direct-SMEM path 使用 b8 container；block-scaled FP4 还包含独立 SFA/SFB request。

## 5. 硬件合同

一个 `Hardware` 记录：

- `hardware_id`；
- `sm_count`；
- `clock_hz`；
- `operating_mode`；
- `l2_capacity_bytes`；
- L2 capacity 的 evidence kind、source path 和 locator。

当前 Thor profile 是 `thor_t5000_sm110_20sm`、20 SM、MAXN、1.575 GHz snapshot。只要 capacity/profile 声明了硬件作用域，选择器必须逐字段匹配；相同 SM 数不代表相同产品。

## 6. Rate 作用域

每条容量必须区分：

| scope | 示例 |
| --- | --- |
| GPU-wide shared | `hbm.total`、`l2.read`、`l2.write`、`l2.duplex` |
| per-SM | `tma.smem_ingress.*.per_sm` |
| full-grid aggregate | cold TMA/DRAM path |
| per-CTA/profile | single-worker latency 或 causal profile |

GPU-wide rate 不乘 SM 数；per-SM rate只有在任务分解和 SM service-unit 数明确时才能合成整卡 makespan。

下一章从这些 workload/precision/schedule 参数推导 compute、memory、TMA、TMEM 和 task 工作量。
