# GEMM Shape Optimization Cases

这些 case 用来先定位瓶颈，再决定是否写专用 kernel。不要把 square GEMM
的优化结论直接套到 skinny 或 GEMV-like shape。

| Case | Shape examples | 主要问题 | 优先比较对象 | 可能优化方向 |
| --- | --- | --- | --- | --- |
| SQ | `1024x1024x1024`, `2048x2048x2048`, `4096x4096x4096` | 标准吞吐和稳定性 | `cublas_tc`, `cutlass`, `tc5a`, `tc5b` | 维持 tc5a/tc5b 主线，观察 cuBLASLt 新 reference 后 ratio 变化 |
| RG | `260x132x256`, `384x520x300`, `1536x1792x1152` | M/N/K 都可能有 tail，cleanup 成本可能主导 | `cublas_tc`, `cutlass`, `tc5a`, `tc5b` | fast tile 与 tail tile 分开，避免大 kernel 为少量 tail 付出过高同步成本 |
| TK | `1024x1024x1000`, `2048x2048x2016` | 只有 K tail，主 tile 足够多 | `cublas_tc`, `tc5a`, `tc5b` | K tail micro-kernel 或最后 K-slice scalar/vector cleanup |
| TMN | `1152x768x1024`, `2304x1792x2048` | K regular，但 M/N tail 影响 store/readback | `cublas_tc`, `cutlass`, `tc5a` | split fast rectangle + boundary epilogue，避免完整 tile 路径写越界判断 |
| SN | `4096x32x4096`, `4096x64x4096`, `8192x128x4096` | N 很小，TC tile 的 N 方向利用率低 | `cublas_tc`, `cutlass` | skinny-N 专用 tile，减少 TMEM column 浪费，考虑多 row block 合并 |
| SM | `32x4096x4096`, `64x4096x4096`, `128x8192x4096` | M 很小，CTA/warp 利用率低 | `cublas_tc`, `cutlass` | skinny-M 专用 tile，减少 epilogue warp 空转，考虑 row batch 合并 |
| GV | `1x4096x4096`, `8x4096x4096`, `4096x1x4096`, `4096x8x4096` | GEMV-like，launch latency 和 memory reuse 比 GFLOP/s 更重要 | `cublas_tc`, `cutlass` | vector/micro-batch kernel，必要时从 GEMM 路径分流 |
| LLM | `1x11008x4096`, `128x11008x4096`, `1024x4096x11008` | decode/prefill 的真实非方阵 workload | `cublas_tc`, `cutlass` | runtime router：decode 走 GEMV-like，prefill 走 skinny/square-like |

推荐执行层级：

1. `smoke_shapes.csv`：确认新 binary、cuBLASLt reference 和脚本可跑。
2. `core_shapes.csv`：做优化决策和写报告。
3. `extended_shapes.csv`：确认 LLM/vision 风格 shape 是否需要单独路径。
