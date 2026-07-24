# L2 吞吐验证报告

## 方法

- 容量 sweep 将工作集从 64 KiB 扫到 128 MiB。
- 并发 sweep 通过改变 threads/block 和 blocks/SM 来改变每 SM active warps。
- 每个数据点使用多次 kernel 内 `clock64()` 计时的中位数。
- benchmark 会拒绝超过 CUDA occupancy 上限的 `blocks_per_sm`。
- `read-same` 不进入证明链，因为同地址 load 更多是在测广播/请求压力，不代表物理 L2 数据搬运。

## 并发饱和峰值


| 模式         | 峰值 B/cycle | 每 SM B/cycle | threads/block | blocks/SM | active warps/SM |
| ------------ | -----------: | ------------: | ------------: | --------: | --------------: |
| read-unique  |      951.905 |        47.595 |           256 |         3 |            24.0 |
| write-unique |      384.288 |        19.214 |           256 |         5 |            40.0 |

## 容量阶跃解释

- `read-unique` 在 1 MiB 到 32 MiB 之间基本持平：中位数 943.425 B/cycle，范围 938.657-946.310 B/cycle。这是非 NCU 证据中最强的 L2-resident 读平台。
- 当读工作集超过 32 MiB L2 后，吞吐在 64 MiB 降到 374.167 B/cycle，在 128 MiB 降到 169.327 B/cycle。
- `write-unique` 在很小工作集下最高；1-8 MiB 大约为 399.698-405.586 B/cycle，到 16 MiB 降到 288.703 B/cycle。这个结果应解释为端到端 store path，而不是纯 L2 write-port 上限。

## 容量 sweep 最优点


| 模式         | 最佳 B/cycle | 工作集 MiB |
| ------------ | -----------: | ---------: |
| read-unique  |      953.999 |      0.062 |
| write-unique |      655.278 |      0.062 |

## 产物

- `results/l2_concurrency_sweep.csv`
- `results/l2_capacity_sweep.csv`
- `results/ncu/ncu_l2_validation_summary.csv`
- `results/ncu/ncu_l2_validation_report.md`
- `results/ncu/read_unique_ncu_raw.csv`
- `results/ncu/write_unique_ncu_raw.csv`
- `plots/l2_concurrency_saturation.svg`
- `plots/l2_capacity_staircase.svg`

## NCU counter 验证

强制 NCU pass 完成于 `2026-07-23 08:47:18 UTC`。这次按要求直接跑，启动前做了瞬时 exclusive pre-check；当时没有 profiler 进程，也没有 `nvidia-smi` 可见的其他 compute 进程。`idle-confirm` 阈值为 `0.0s`。

|模式|app clock B/cycle|LTS throughput avg %peak elapsed|LTS throughput max %peak elapsed|LTS bytes %peak elapsed|LTS bytes/cycle elapsed|LTS read sectors %peak elapsed|LTS write sectors %peak elapsed|LTS hit rate|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|read-unique|947.207|93.220|93.250|93.220|954.620|93.220|0.010|99.74%|
|write-unique|234.741|48.290|48.300|22.840|233.860|0.000|45.730|99.72%|

这里的 `app clock B/cycle` 来自 NCU 启动前同参数 app probe 的 `clock64()` 计时，不是由 NCU counter 反推出的硬件峰值。`%peak elapsed` 是 NCU 的 `pct_of_peak_sustained_elapsed`，也就是按 elapsed cycles 和 NCU 内部 sustained-peak 定义算出的利用率。

流量归一化结果：

|模式|L1TEX LD/预期|L1TEX ST/预期|LTS bytes/预期|LTS read sector B/预期|LTS write sector B/预期|DRAM read B/预期|DRAM write B/预期|
|---|---:|---:|---:|---:|---:|---:|---:|
|read-unique|1.031|0.000|1.031|1.031|0.000|||
|write-unique|0.000|1.031|1.028|0.000|1.029|||

解释：

- `read-unique` 的 L1TEX global-load bytes 和 LTS read-sector bytes 都在逻辑请求字节数的约 3.1% 以内，说明主要流量确实到达 L2，而不是被 L1 吸收；NCU 给出的 LTS throughput 利用率约为 93.2% peak。
- `write-unique` 的 L1TEX global-store bytes 和 LTS write-sector bytes 分别在逻辑请求字节数的约 3.1% 和 2.9% 以内，并且 LTS write miss-sector bytes 很小；NCU 给出的 LTS throughput 利用率约为 48.3% peak。它仍然是端到端 store-path 数字，不是纯 L2 write-port 峰值。
- 本机 Thor/NCU query 不支持 `dram__bytes*` metrics，summary 中已明确列为 missing。因此 DRAM-path 检查使用 LTS miss-sector counters，而不是直接 DRAM byte counters。

## 最终结论口径

|项目|数值|含义|
|---|---:|---|
|L2 read peak model|约 1024 B/cycle/GPU|由 NCU peak 利用率口径反推的模型峰值|
|measured L2-hit read sustained|约 943-947 B/cycle/GPU|本 microbench 实测的持续读吞吐|
|L2 write peak model|约 512 B/cycle/GPU|由 NCU write-sector peak 口径反推的模型峰值|
|measured global-store sustained|约 235-300 B/cycle/GPU|本 microbench 在 L2-sized 工作集下测到的端到端 store-path 持续吞吐|

可以在报告正文中写“读 peak model 约 1024 B/cycle、写 peak model 约 512 B/cycle”，但必须同时写清楚：当前实测 sustained 读吞吐是约 947 B/cycle，当前实测 store-path sustained 是约 235-300 B/cycle。64 KiB 工作集的 655.278 B/cycle 是 tiny working set fast case，不作为 L2 写带宽上限。
