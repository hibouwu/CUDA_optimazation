# 静态 collector 分析

## 观察

- aggregate cases: 32
- valid aggregate cases: 32
- invalid aggregate cases: 0
- telemetry status: `mem_clock_mhz:unavailable;temperature_c:unavailable;power_w:unavailable;sm_clock_mhz:sysfs_or_default`.

## 推断

- 静态 single-case binary 将 runtime collector/shape/D dispatch 和 descriptor construction 移出 timed loop。
- CUDA event wall time 只作为端到端 launch/setup/readback 交叉检查；clock64 cycles 是校准用 timed-window 指标。
- logical effective bytes/cycle 只有在比较 same/pingpong/rotating address mode、shape scaling 和 control-subtracted cycles 后才可解释。

## 不支持的说法

- 这些数据本身不能识别物理 SMEM port width、SMEM bank count、TMEM bank width 或 hidden collector depth。
