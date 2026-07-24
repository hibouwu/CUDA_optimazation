# DSMEM topology/contention 设计对抗式审查

设计风险和控制：

- 风险：`cluster_size=4` 不稳定。控制：默认只启动 `SM_count / 4` 个 cluster，
  避免多 wave；若运行失败则降级记录为不支持，不能伪造结果。
- 风险：fan-in 模式把 inactive rank 也计入请求字节。控制：app 输出
  `active_remote_blocks`，请求字节只按 active remote CTA 计。
- 风险：fan-in write 全部 source 写同一地址导致测的是同地址冲突。控制：
  source rank 加 offset，尽量分散到 target rank0 的 DSMEM working set。
- 风险：SASS 看不出 DSMEM。控制：运行后检查 mapped shared 的 `LD.E.128` /
  `ST.E.128`，并以 NCU `mem_dshared` byte counter 为主证据。
- 风险：把结果解读为物理拓扑。控制：只报告 rank-distance/fan-in pattern 的
  end-to-end remote DSMEM throughput，不推断物理 fabric 拓扑。

设计审查结论：可以运行；运行后必须通过 NCU dshared traffic 校验。
