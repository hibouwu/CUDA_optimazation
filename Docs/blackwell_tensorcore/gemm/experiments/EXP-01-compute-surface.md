# EXP-01：12 精度 Tensor Core compute surface

## 1. 研究问题

在排除 GMEM/TMA、TMEM readback、epilogue 和 launch 后，Thor/SM110 对各 precision/MMA shape 能持续提供多少 full-GPU compute service？

## 2. 对应模型参数

```text
tensor.<format>.m128n64
tensor.<format>.m128n128
tensor.<format>.m128n256
```

这些是 `measured_sustained` empirical capacities，不是物理 compute upper。

## 3. Case matrix

- 12 个 precision contract；
- MMA M=128；
- MMA N∈{64,128,256}；
- full-SM 4-warp launch 共 36 个模型容量点；
- runner 同时保留 single-warp/single-block 对照，但不把它们导入 full-GPU service capacity；
- 每 case 10 个外部 trial。

precision 覆盖 FP16、BF16、TF32、E4M3、E5M2、E3M2、E2M3、raw E2M1、MXFP4、NVFP4、S8 和 U8。

## 4. 工作量与计时

定义一条 MMA 的标量工作量：

\[
W_{\mathrm{inst}}=2M_{\mathrm{MMA}}N_{\mathrm{MMA}}K_{\mathrm{MMA}}.
\]

runner 使用设备 `%globaltimer` 的 MMA issue-to-completion barrier 区间，按实际迭代和活跃 SM 数重算：

\[
\widehat C_{\mathrm{compute}}
=\frac{W_{\mathrm{issued}}}{T_{\mathrm{globaltimer}}}.
\]

整数精度单位为 OP/s，其余为 FLOP/s。

## 5. 接受门禁

- SM110a descriptor 与 precision contract 一致；
- 目标函数块出现预期 TCGen05/HMMA/IMMA SASS；
- full-GPU case 覆盖预期 20 SM；
- 10 trial 全部完成且 rate 可由 raw 字段重算；
- 选中的 case 保留 NCU report；
- source、compile command、binary hash、SASS、environment、run spec 和 COMPLETE 可审计；
- NVFP4 历史错误 descriptor 数据保持 quarantined。

## 6. 当前状态

- runner precision surface：12/12；
- full-SM shape capacity：36；
- 公共 compute campaign 定义完整；
- 历史 Thor closure 已取得 12 精度 compute-only 数据；
- strict compute upper 仍只有约 7/12 precision 有独立证据。

## 7. 进入模型

importer 只导入 `launch=full_sm_4warp_block` 且 M128/N64/128/256 的点，并绑定：

- precision ID；
- MMA shape；
- CTA group 1；
- 20 SM；
- Thor T5000；
- MAXN；
- 128 threads/CTA；
- one resident CTA/SM；
- SMEM operand residency；
- timed scope。

## 8. 不能证明什么

- measured TFLOP/s/TOP/s 不是任何实现都不能突破的 rate upper；
- compute-only 不包含完整 GEMM 数据搬运和输出；
- 一种 MMA shape 的 rate 不能借给另一 shape；
- static SASS presence 不证明 runtime numerical correctness；
- full-GPU aggregate rate 不单独证明 per-CTA latency lower bound。

## 9. 源码与工件

- runner：[run_compute_campaign.py](../../../../microbench/sm110_gemm_campaign/run_compute_campaign.py)
- independent auditor：[audit_campaign.py](../../../../microbench/sm110_gemm_campaign/audit_campaign.py)
- descriptor encoder：[tcgen05_descriptors.py](../../../../scripts/sm110_gemm_model/tcgen05_descriptors.py)
- importer：[evidence_import.py](../../../../scripts/sm110_gemm_model/evidence_import.py)
- 历史结果入口见 [microbenchmark_sources](../appendices/microbenchmark_sources.md)。
