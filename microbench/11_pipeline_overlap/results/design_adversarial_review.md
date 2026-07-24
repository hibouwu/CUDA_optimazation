# Pipeline overlap 设计对抗式审查

结论：可以运行。

## 设计目标

测 `tcgen05.cp` SMEM-to-TMEM ingress 与 TS MMA consuming TMEM operand 之间的
组合效率，而不是再测单独 cp 或单独 MMA 峰值。实验用 FP4 M128N256，因为已有
component 数据、吞吐最高、NCU 代表性最好。

## 风险和控制

- 风险：把 overlap case 的 `bytes_per_cycle` 误读成总 TMEM 带宽。控制：
  明确二进制输出的 `bytes_per_cycle` 是 cp payload；TMEM consume demand 单独按
  `mma_instruction_count * 2048B / cycles` 估计。
- 风险：只测一个 overlap case，无法证明重叠。控制：同时跑 `cp-only`、
  `ts-mma-only`、`serial-a1`、`overlap-a2`、`warp-split-a2` 和
  `mainloop-a2-k16`。
- 风险：SASS 与 case 名不匹配。控制：每个 case 采集 SASS；cp-only 必须有
  `UTCCP`，TS/combined case 必须有 `UTCOMMA`，combined case 必须同时有
  `UTCCP` 和 `UTCOMMA`。
- 风险：把 NCU 的缺失 UTCCP pipe counter 当成没有 cp。控制：NCU 不作为 cp
  指令存在性的唯一证据；SASS 和 app `cp_instruction_count` 是 cp 存在性证据。
- 风险：mainloop 结果只反映短 microbenchmark。控制：短 tile overlap 与 K16
  mainloop 同时测；通过 `cycles_per_cp`/`cycles_per_tile` 比较稳态一致性。

运行后必须满足：

- `overlap-a2` 比 `serial-a1` 至少快 20%。
- `mainloop-a2-k16` 的 per-cp cycle 与 `overlap-a2` 单 tile cycle 接近。
- combined cases 的 SASS 同时含 `UTCCP` 和 `UTCOMMA`。
- NCU 成功采集 SM/tensor/stall 指标，并给出 rough bandwidth upper estimate。
