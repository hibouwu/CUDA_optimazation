# TMEM cp ingress 设计对抗式审查

结论：设计可运行，但 scope 必须限定为 `tcgen05.cp` SMEM-to-TMEM ingress。

## 设计目标

本目录不重新发明 TMEM cp microbenchmark，而是把 `../mma_with_cp` 已生成并已运行的
代表性 `tcgen05.cp` 数据整理成独立带宽实验：

- cp-only app timing：测 `tcgen05.cp` 指令流的 payload B/cycle 与 cycles/cp。
- cp interference：观察额外 cp traffic 对 SS/TS MMA mainloop 的性能影响。
- representative NCU：导出已有 key NCU report 的 SM/memory utilization 指标。

## 风险和控制

- 风险：把 `tcgen05.cp` ingress 当成 raw TMEM write-port peak。
  控制：README 和运行审查都明确只报告 SMEM-to-TMEM cp ingress，不报告 raw TMEM
  bank/read/write bandwidth。
- 风险：wrapper 只复制旧 CSV，缺少指令路径证明。
  控制：保存 representative SASS summary，必须包含 `UTCCP.T.S.128dp128bit`。
- 风险：NCU `sm__inst_executed_pipe_tmem.*` 为 0 被误读为 cp 没执行。
  控制：UTCCP 存在性以 SASS 和 app `cp_instruction_count` 为证据；NCU tmem pipe
  counter 只作为工具覆盖边界记录。
- 风险：跨 precision/shape 结果相同被误解为数据重复错误。
  控制：该 cp-only suffix/effective bytes 相同，结果相同是预期；审查中保留说明。
- 风险：单独 cp 峰值不能解释 pipeline 里的可重叠程度。
  控制：本目录只给 component ingress；组合 overlap 另由 `../11_pipeline_overlap`
  测量。

运行后判据：

- cp-only row 报告稳定的 payload throughput 与 cycles/cp。
- SASS summary 存在 UTCCP。
- NCU summary 存在，并明确当前工具链对 UTCCP/tmem pipe metric 的限制。
- 运行审查必须拒绝 raw TMEM bank count、bank width、raw read/write bandwidth 结论。
