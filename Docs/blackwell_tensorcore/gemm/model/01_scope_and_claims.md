# 01 范围、主张与证据边界

## 1. 研究问题

目标不是预测仓库中某一个 kernel，而是回答：在冻结的 GEMM 数学语义、实现域和 Thor/SM110 物理约束下，一个没有可避免性能浪费的经典稠密 GEMM 最快可以到哪里。

“最快”必须拆成三个输出：

1. 条件可证明性能上界 \(P_{\mathrm{ub}}\)；
2. microbenchmark 驱动的经验理想包络 \(\widehat P_{\mathrm{env}}\)；
3. 完整 GEMM 已观测最好值 \(P_{\mathrm{obs}}\)。

定义 \(P^\star\) 为声明实现域中真实但未知的最优性能。在 workload、实现域和上界假设一致时：

\[
P_{\mathrm{obs}}\le P^\star\le P_{\mathrm{ub}}.
\]

\(\widehat P_{\mathrm{env}}\) 不自动进入该不等式，因为 measured sustained rate 只证明硬件已经达到某个内点，不证明真实容量绝不更高。

## 2. 两种实现域

定义 `tensor_core_classical` 为使用经典 \(2MNK\) 算术、允许 Tensor Core schedule 的实现域；定义 `all_classical` 为所有不改变经典 GEMM 算术复杂度的实现，包括 Tensor Core、CUDA core 或 mixed path。

产品级 Tensor Core peak 只能约束 `tensor_core_classical`。要约束 `all_classical`，必须提供 `compute.total.<precision_id>` 形式的 aggregate compute upper；不能假设 Tensor Core peak 同时覆盖其它算术管线，也不能把多个管线 peak 直接相加。

当前默认 workload 使用 `tensor_core_classical`。`all_classical` schema 已实现，但 aggregate compute upper 证据尚不完整。

## 3. 第一版包含范围

第一版模型覆盖：

- 单次稠密 GEMM；
- 经典 \(2MNK\) 工作量；
- 单 GPU；
- NN 数据布局的可执行 schedule；
- `epilogue=none`；
- accumulator output；
- CTA-group-1；
- `split_k=1`；
- `cold_hbm`、`hot_l2` 和 `compute_oracle` 三种入口条件；
- 浮点 FLOP/s 与整数 OP/s 两类性能单位。

当前不覆盖：

- 稀疏、Strassen 或近似矩阵乘法；
- batched/grouped GEMM；
- 多 GPU；
- host-device copy；
- 跨算子 persistent reuse；
- 未实现工作量合同的 bias/ReLU/GELU/residual/requant；
- 任意转置布局的完整 data-movement schedule；
- 非整除 shape 的任意 exact tail kernel。

排除项不代表硬件做不到，只表示当前上界和经验搜索不声称覆盖它们。

## 4. 五类主张

| 主张 | 必须具备的证据 | 缺失时的合法输出 |
| --- | --- | --- |
| 数学工作量主张 | workload/schedule 的可复算计账 | 拒绝 workload 或 schedule |
| 物理 rate upper | 官方规格、架构推导或明确工具模型外边界 | 更松的 `partial` upper 或 `insufficient_evidence` |
| 经验容量 | 同硬件、同 topology、同 residency、同计时域的 microbenchmark | `insufficient_evidence`，不能借邻近 rate |
| causal/joint 可达性 | exact joint profile 或联合 runner | 资源层可以单独报告，集成包络 fail closed |
| 完整 GEMM 结果 | 独立 correctness、same-contract reference、trial、SASS、环境和审计工件 | 不进入 observation closure |

## 5. “没有性能浪费”的精确定义

严格上界允许所有可避免开销消失，包括完美资源重叠和零可避免 fixed cost；因此它是乐观时间下界，不是某个现有 kernel 的预测。

经验理想包络只允许消除当前模型明确识别的 schedule 损失。它仍受：

- issued work；
- task waves；
- per-SM ingress；
- GPU-wide duplex service；
- TMEM readback；
- exact causal DAG；
- 已证明 hard upper；

约束。缺少 joint evidence 时，不能把独立 component peak 自动称为同时可达。

## 6. 正确性与紧致性分开

一个上界可以方向正确但很松。缺少某个 rate upper 通常只会减少时间下界约束，使 \(P_{\mathrm{ub}}\) 变大；这不会制造假上界，但会降低实用性。

以下错误会破坏方向正确性：

- 用 measured sustained rate 冒充 rate upper；
- 把 GPU-wide L2 B/cycle 乘以 SM 数；
- 把 per-SM ingress 当作整卡唯一共享出口；
- 把 cold proxy 当作 physical HBM duplex；
- 忽略一个合法但无数值的 schedule，再把剩余 schedule 最大值称为 manifest 上界；
- 把 FP16 denominator 当作 FP4 同精度性能对照。

## 7. 当前完成边界

当前模型代码和 fail-closed 门禁可执行，但最终 `complete=false`。原因包括 physical HBM duplex、26/28 exact TMA topology、causal profiles、全精度完整 GEMM 和部分 strict compute upper 缺失。详见 [08 当前覆盖与缺口](08_current_coverage_and_gaps.md)。

下一章定义全部数学符号、单位、workload 与硬件作用域。
