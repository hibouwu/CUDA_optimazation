# L2 throughput 设计对抗式审查

结论：设计可运行，但结论必须区分 measured sustained 与 NCU 反推 model peak。

## 设计目标

测 L2-resident global load/store path 的请求吞吐，重点是：

- `read-unique`：L2-sized working set 的 128-bit `ld.global.cg` unique load。
- `write-unique`：L2-sized working set 的 128-bit `st.global.cg` store path。
- `read-same`：同地址请求/广播压力诊断，不作为物理 L2 带宽结论。

## 风险和控制

- 风险：源码 `uint4` 被 ptxas 标量化，导致请求字节统计高于真实 load 宽度。
  控制：使用 volatile inline PTX `ld.global.cg.v4.u32` 并消费四个 lane；运行后检查
  SASS 必须包含 `LDG.E.128`。
- 风险：power-of-two ring aliasing 只触碰工作集子集，伪造 L2 capacity sweep。
  控制：index stride 使用与 2 的幂工作集互质的 odd stride，并在 CSV 记录
  `index_stride_elements` / `stream_period_iters`。
- 风险：per-CTA `clock64()` 在多 wave launch 下高估吞吐。
  控制：检查 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`，拒绝超过 resident CTA
  上限的 `blocks-per-sm`。
- 风险：把 store-path 结果当成纯 L2 write-port peak。
  控制：write 结论只写端到端 global-store path throughput；L2 write peak 只能写成
  NCU utilization 反推的 model peak。
- 风险：NCU direct DRAM metric 缺失。
  控制：L2 实验主要用 LTS sector/throughput/hit-rate 证明 L2-resident；缺失的
  `dram__bytes*` 必须列入 missing metrics，不能当作 0。

运行后判据：

- `read-unique` SASS 是 128-bit global load，且 LTS bytes 与请求字节接近。
- L2-sized read capacity sweep 稳定，超过 L2 容量后出现 capacity cliff。
- NCU LTS throughput utilization 可反推 read model peak，且 hit rate 高。
- 写路径必须保留 store/fence/drain 边界，不宣称 raw L2 write-port 峰值。
