# GMEM/DRAM 运行对抗式审查

通过：修复 read-stream 标量化后已重编译重跑，LTS miss-sector proxy 显示大部分流量超过 L2，符合 GMEM/DRAM streaming 目标。

## 已检查的问题

- SASS：read-stream 现在是 `LDG.E.128.STRONG.GPU`；write-stream 是 `STG.E.128.STRONG.GPU`；copy-stream 同时包含 128-bit load/store。旧版 read-stream 曾被编译器标量化为 `LDG.E`/`LDG.E.64`，该问题已通过消费 `uint4` 四个 lane 修复并重跑。
- working set：默认 256 MiB，大于 32 MiB L2；capacity sweep 中 64 MiB 明显更快，256/512 MiB 更接近 DRAM streaming，说明默认数据不是小容量 L2 resident 测量。
- NCU traffic：`lts__t_bytes.sum / requested_bytes` 约为 read 1.031、write 1.031、copy 1.292；LTS miss-sector proxy / requested_bytes 约为 read 0.949、write 0.860、copy 0.970。

## 保留结论边界

此 Thor/NCU 组合缺失 `dram__bytes*`，所以本实验不声称有直接 DRAM byte counter，只报告 LTS miss-sector proxy 和 app throughput。copy-stream 的 `lts__t_bytes.sum` 高于请求量，最终 DRAM-path 判断以 miss-sector proxy 为主。
