# TMEM consume 运行对抗式审查

通过：app、SASS 和 NCU 均已覆盖代表性的 TS TMEM-consume path。

已检查/需保留的边界：

- TS cases 的 SASS 必须包含 `UTCOMMA... tmem[...]`，说明 MMA 从 TMEM operand 消费。
- `ts-cp-mma-a2-k16` 同时包含 `UTCCP`，所以它是 pipeline consume+cp，不是纯 consume。
- `estimated_tmem_consume_bytes_per_cycle` 是按 2048 B/TS-MMA 推导的 operand demand rate，不是 raw TMEM read-port peak。
- `rough_consume_upper_from_sm_peak_bytes_per_cycle` 只是按 NCU SM throughput 归一化的需求率上限估计，不是物理 TMEM 端口峰值。
- NCU 的 `sm__inst_executed_pipe_tmem.*` 可能不覆盖 UTCOMMA 的 TMEM operand 读取；最终利用率以 tensor/TC pipe 和 SM throughput 为主，tmem pipe counter 只作 coverage 观察。
