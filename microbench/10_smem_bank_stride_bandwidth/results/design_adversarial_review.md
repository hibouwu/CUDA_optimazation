# SMEM bank/stride 设计对抗式审查

设计风险和控制：

- 风险：编译器把 shared access 优化掉或改成非预期访问。控制：使用 inline
  `ld.shared.u32` / `st.shared.u32`，SASS 检查 `LDS`/`STS`。
- 风险：stride sweep 没有真实 bank conflict。控制：按 32-bit word stride 访问，
  stride 1/2/4/8/16/32 覆盖从无冲突到单 bank 多地址冲突；NCU bank-conflict
  counter 必须随高 stride 上升。
- 风险：把 wavefront proxy 当 direct byte counter。控制：报告中明确没有 local
  shared direct byte counter，rough peak 是 payload-normalized wavefront `%peak` 估计。
- 风险：多 wave launch 污染 per-CTA clock。控制：默认一 CTA/SM，并检查 occupancy。

设计审查结论：可以运行；运行后必须同时满足 SASS、吞吐下降和 NCU bank-conflict 上升。
