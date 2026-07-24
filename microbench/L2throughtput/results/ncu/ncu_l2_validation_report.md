# NCU L2 验证报告

生成时间：2026-07-23 08:47:18 UTC

本次运行由 exclusive monitor 启动；启动前已观察到 `0.0s` GPU idle，
期间没有 profiler 进程，也没有 `nvidia-smi` 可见的其他 compute 进程。
配置的 idle-confirm 阈值是 `0.0s`。

启动 benchmark 前已用 `ncu --query-metrics --query-metrics-mode all` 检查 metric 支持情况；
不支持的候选 metric 会明确列出，不会被解释成 0。

|模式|预期请求字节|app clock B/cycle|LTS throughput avg %peak elapsed|LTS throughput max %peak elapsed|LTS bytes %peak elapsed|LTS bytes/cycle elapsed|LTS read sectors %peak elapsed|LTS write sectors %peak elapsed|LTS hit rate|缺失 metric 数|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|read-unique|5368709120|947.207|93.220|93.250|93.220|954.620|93.220|0.010|99.74%|3|
|write-unique|5368709120|234.741|48.290|48.300|22.840|233.860|0.000|45.730|99.72%|3|

`app clock B/cycle` 来自 NCU 启动前同参数 app probe 的 `clock64()` 计时；NCU counter 用来验证 L1TEX/LTS 流量和 hit/miss 情况。
`%peak elapsed` 是 NCU 按 elapsed cycles 计算的 pct_of_peak_sustained_elapsed；它是 NCU 内部 peak 定义下的利用率，不等同于本 microbench 的理论硬件峰值。

## L1/LTS/DRAM 流量归一化

|模式|L1TEX LD/预期|L1TEX ST/预期|LTS bytes/预期|LTS read sector B/预期|LTS write sector B/预期|DRAM read B/预期|DRAM write B/预期|
|---|---:|---:|---:|---:|---:|---:|---:|
|read-unique|1.031|0.000|1.031|1.031|0.000|||
|write-unique|0.000|1.031|1.028|0.000|1.029|||

## peak model 与 measured sustained

|项目|数值|含义|
|---|---:|---|
|L2 read peak model|约 1024 B/cycle/GPU|由 NCU peak 利用率口径反推的模型峰值|
|measured L2-hit read sustained|947.207 B/cycle/GPU|本次 app clock 实测持续值|
|L2 write peak model|约 512 B/cycle/GPU|由 NCU write-sector peak 口径反推的模型峰值|
|measured global-store sustained|234.741 B/cycle/GPU|本次 app clock 实测端到端 store-path 值|

`1024/512 B/cycle` 只能作为 model peak 写入结论；当前 microbench 没有实测打满这两个数。

## 产物

- `read_unique_ncu_raw.csv`
- `write_unique_ncu_raw.csv`
- `read_unique_validation.ncu-rep`
- `write_unique_validation.ncu-rep`
- `query_metrics.log`
- `ncu_l2_validation_summary.csv`
