# DSMEM topology/contention 运行对抗式审查

通过：all remote topology modes have NCU dshared bytes matching requested bytes within 25%.

## 已检查的问题

- SASS：read modes 包含 mapped remote shared `LD.E.128`，write modes 包含 `ST.E.128`；初始化仍有 local `STS.128`，不计入结论证据。
- 统计修复：初版 host vector 初始化写错，导致 fan-in cycles 读回越界、结果为 0；已改成显式 `resize()` 后重跑。
- active bytes：fan-in 模式只按 rank 1..3 三个 active remote CTA 计请求字节；rank 0 只参与 barrier。
- NCU traffic：所有模式 `dshared/expected` 为 `1.011-1.032`，说明请求字节与 DSMEM counter 匹配。
- NCU utilization：dshared throughput 为 `53.31-146.41 B/cycle`，`0.52-1.43% peak`；反推 dshared 模型上限约 `10.2 KiB/cycle/GPU`。

## 保留边界

这些结果是 rank-distance/fan-in pattern 的 remote DSMEM end-to-end throughput，不证明物理 fabric 拓扑。fan-in write 是多 source 写 rank0 的压力测试，不能当作无冲突单端口写峰值。
