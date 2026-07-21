# 静态校准分析

## 观察

- 静态 calibration aggregate cases: 576 valid, 0 invalid.
- 静态 collector quick cases: 32 valid, 0 invalid.
- 静态 ingress quick cases: 12 valid, 0 invalid.
- 静态 ld.shared quick cases: 11 valid, 0 invalid.
- 静态 TMEM quick cases: 8 valid, 0 invalid.
- BF16 N128 Q4 same-D `input_d=0` full-grid median cycles/MMA: `145.581`；可信 BF16 K4 reference 约为 `146.132 cycles/MMA`。
- 同一 row 记录 `113.44 clock64 full-grid TFLOP/s` 和 `66.78 TFLOP/s` event-wall throughput。event wall time 包含 setup/readback。
- SASS dump 位于 `plots/static_sass/`；single-case binary 中目标 `UTCHMMA` 数量与配置 Q 对齐。
- 静态 CSV telemetry status: `mem_clock_mhz:unavailable;temperature_c:unavailable;power_w:unavailable;sm_clock_mhz:sysfs_or_default`.

## 推断

- 静态 single-case binary 将 runtime collector/shape/D dispatch、descriptor construction 和 operand address selection 移出 timed loop。
- Q4 gate 匹配已知 K4 mainloop baseline，因此静态测量是硬件路径标定的主证据。
- CUDA event wall time 只作为端到端 launch/setup/readback 交叉检查；clock64 cycles 是校准用 timed-window 指标。
- logical effective bytes/cycle 只有在结合 address mode 对比和 shape scaling 并使用静态 beta 时才有解释意义。

## 不支持的说法

- 这些数据本身不能识别物理 SMEM port width、SMEM bank count、TMEM bank width/count、hidden collector depth 或 hidden async group queue depth。
- 旧 runtime-dispatch 矩阵是负控制，不用于物理硬件推断。
