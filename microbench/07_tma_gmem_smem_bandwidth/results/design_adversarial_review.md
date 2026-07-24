# TMA GMEM-to-SMEM 设计对抗式审查

初始设计风险和对应控制：

- 风险：普通 global load/store 被误当成 TMA。控制：device code 只用
  `cp.async.bulk.tensor.3d.shared::cta.global` 发起数据搬运，run 后用 SASS
  检查 `CP_ASYNC`/TMA 指令。
- 风险：只测 L2 resident 数据，误称 DRAM。控制：分 `l2-hit` 16 MiB 和
  `dram-stream` 256 MiB；NCU 用 LTS miss-sector proxy 判定 DRAM path。
- 风险：只测 mbarrier 轮询开销。控制：统计 payload bytes，并在每次 TMA 完成后
  读取 shared destination 的少量 word 形成 checksum，避免 destination 完全死代码。
- 风险：多 CTA wave 让 per-CTA max clock 高估/低估。控制：默认一 CTA/SM，
  并记录 occupancy limit。
- 风险：把结果说成纯硬件峰值。控制：报告为 end-to-end TMA ingress throughput，
  只用 NCU `%peak` 反推 rough LTS/TMA 模型上限。

设计审查结论：可以运行；结果审查必须检查 SASS 和 NCU traffic 后才能通过。
