# EXP-06：TMEM readback、scale ingress 与 epilogue diagnostics

## 1. 研究问题

不同 LDTM width/readback warp 合同的 accumulator readback service 是多少；block-scale SFA/SFB 从 SMEM 进入 TMEM 的 source payload service 是多少；epilogue/store 是否存在独立诊断瓶颈？

## 2. 对应模型参数

- `tmem.readback`；
- `tmem.readback.x8.warps1`；
- `tmem.readback.x8.warps4`；
- `tmem.readback.x16.warps1`；
- `tmem.scale_ingress`；
- NVFP4 epilogue diagnostics；
- causal profile 中的 `epilogue_latency_seconds`。

## 3. Readback case matrix

TMEM accumulator readback 组合：

- registers∈{8,16}；
- warps∈{1,4}；
- 128 threads/CTA；
- one CTA/SM；
- 20-SM aggregate timing；
- 每 case 10 trials。

模型根据 schedule 的 `tmem_load_registers` 和 `readback_warps` 精确选择资源。tc5a 有 192 threads/6 CTA warps，但只有 4 个 epilogue warps；不能默认使用 `threads/32=6`。

## 4. Scale ingress

block-scale source 使用同构 `tcgen05.cp` S2T atom。每条 source atom 读取 512 B，并 multicast 到 TMEM partitions；模型按唯一 512 B source payload 计费，不按 destination footprint 重复乘四。

定义：

\[
\widehat C_{\mathrm{TMEM,scale}}
=\frac{Q_{\mathrm{source}}}{T_{\mathrm{globaltimer}}}.
\]

## 5. Epilogue diagnostics

NVFP4 requant case 覆盖 normal、outlier、constant 三种输入分布，用于检查 value/scale correctness 和 fused output path。它是特定 epilogue diagnostic，不自动适用于所有 accumulator-output GEMM。

当前最终 causal envelope 主要使用 EXP-07 joint profile 的单 task epilogue latency；独立 epilogue capacity 不应与 profile drain 重复计费。

## 6. 接受门禁

- function-scoped `LDTM.x8` / `LDTM.x16` / `UTCCP.T.S.4x32dp128bit`；
- 精确 warp/register/threads/residency；
- 10 external trials；
- `%globaltimer` 或明确 CUDA-event timed scope；
- source/binary/SASS/environment/hash；
- value/scale mismatch 为 0；
- independent component auditor。

## 7. 当前状态

统一 component campaign 已返回并通过审计：

- 4 个 TMEM readback contract；
- 1 个 scale ingress contract；
- 3 个 NVFP4 requant diagnostic case。

这些点关闭公共 component surface，但不关闭所有 schedule 的 causal profile。

## 8. 不能证明什么

- readback peak 不证明 TMA/MMA/readback 同时可达；
- scale ingress rate 不替代 exact TMA scale payload；
- multicast destination bytes 不能重复计入 source numerator；
- NVFP4 epilogue 不适用于其它 precision/output semantic；
- component case complete 不等于 full-GEMM correctness。

## 9. 源码与工件

- TMEM source：[tmem_readback_bandwidth.cu](../../../../microbench/12_tmem_readback_bandwidth/tmem_readback_bandwidth.cu)
- scale source：[tmem_scale_ingress_bandwidth.cu](../../../../microbench/13_tmem_scale_ingress_bandwidth/tmem_scale_ingress_bandwidth.cu)
- epilogue source：[requant_epilogue_benchmark.cu](../../../../GEMMsm110/tests/requant_epilogue_benchmark.cu)
- component runner：[run_component_campaign.py](../../../../microbench/sm110_gemm_component_campaign/run_component_campaign.py)
- component auditor：[audit_campaign.py](../../../../microbench/sm110_gemm_component_campaign/audit_campaign.py)
