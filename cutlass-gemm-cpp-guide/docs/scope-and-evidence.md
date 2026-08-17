# 范围与证据分层

## 产品边界

本仓库只验证 NVIDIA Thor / CC 11.0 / `sm_110a`。CUTLASS C++ builder 中的
`cutlass::arch::Sm100` 是共享的 TCGen05 编程模型 tag，不代表 binary 编译为 SM100，
也不允许据此宣称 B200/B300/RTX 50 支持。

首版覆盖：

- FP16/BF16/FP8 dense；
- 1SM 与 2SM；
- MXFP8、MXFP4、NVFP4 block scaling；
- structured sparse candidate；
- fused bias+ReLU；
- logical tail、padded stride和 alignment 负例。

首版不覆盖：TF32、INT8、grouped/batched/pointer-array、mixed-input 大矩阵、
blockwise/groupwise scaling、Stream-K、性能与 NCU。

## 七个独立状态

每个 case 只能分别报告：

```text
documented
source_present
compile_passed
ptx_verified
sass_verified
runtime_correct
performance_measured
```

含义不能互相替代：

- `compile_passed` 不证明生成了预期指令；
- PTX mnemonic 不证明对应 SASS 在目标函数中；
- SASS token 不证明输出正确；
- 数值 PASS 不证明没有未覆盖的 legal shape；
- 小矩阵 correctness 不是性能证据；
- self-hosted runner 未执行时必须保留 `NOT_RUN`。

## 当前状态

固定容器已完成 10 个 case 的 `sm_110a` compile、PTX 和 SASS function-block audit。
由于当前主机没有可用 NVIDIA driver，本仓库没有在本轮产生新鲜 Thor runtime 结果。
因此当前仓库是 pre-release static closure，不是 v0.1 runtime closure。

Sparse case 额外保持严格边界：builder 和 codegen 已闭合，独立 host 2:4 primitive 已测试，
但 CUTLASS compressor metadata 到 full-GEMM output 的独立数值桥仍为 `NOT_RUN`。
