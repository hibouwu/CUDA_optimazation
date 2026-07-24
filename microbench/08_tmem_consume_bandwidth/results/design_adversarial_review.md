# TMEM consume 设计对抗式审查

设计结论：可以运行，但必须限制解释范围。

风险和控制：

- 风险：把 TS MMA operand demand 误称为 raw TMEM read bandwidth。控制：
  结果字段命名为 `estimated_tmem_consume_bytes_per_cycle`，报告说明它是
  2048 B/TS-MMA 的需求率估计。
- 风险：TS CP+MMA 混入 `tcgen05.cp` 写入。控制：同时测 `ts-mma-only`
  和 `ts-cp-mma-a2-k16`，并在 SASS/报告中区分。
- 风险：没有非 TMEM baseline。控制：加入 `ss-mma-mainloop-k16` 作为
  SMEM descriptor path baseline。
- 风险：NCU tmem pipe counter 不覆盖 UTCOMMA operand read。控制：报告
  tensor/TC pipe utilization、SM throughput，并把 tmem pipe counter 作为
  coverage observation 而不是硬性证明。
- 风险：SASS 不是真的 consume TMEM。控制：要求 TS SASS 包含
  `UTCOMMA... tmem[...]`。
