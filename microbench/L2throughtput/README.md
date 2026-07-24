# L2 吞吐微基准

本目录用于测量 L2 读/写路径的请求吞吐，单位是 byte/cycle。

```bash
./build_and_run.sh run
```

默认 sweep 会生成 `results/l2_throughput.csv`，包含三个测试模式。编译时使用
`-Xptxas -dlcm=cg`，尽量让普通 global load 使用 global/L2 cache policy。

- `read-same`：所有线程反复从同一个 16B 地址执行 `ld.global.cg.v4.u32`。这是广播/请求发射压力测试，统计的是请求载荷字节，不代表真实物理 L2 数据搬运量。
- `read-unique`：每个线程从连续 ring buffer 读取对齐的 `uint4`，默认工作集放在 L2 容量内。这是主要的 L2 读带宽测试。生成的 SASS 记录在 `results/sass_summary.txt`。
- `write-unique`：每个线程对同类 ring buffer 执行 `st.global.cg.v4.u32`，计时结束前执行 device-wide store fence。这是主要的 L2 写路径测试。

每条 load/store 指令按 16 个请求字节统计。报告中的 `bytes_per_cycle` 等于全 GPU 请求字节数除以所有 CTA 中最大的 `clock64()` 计时周期。除以 `sm_count` 可得到每 SM byte/cycle。程序会用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 检查 occupancy；如果 `--blocks-per-sm` 超过可驻留 CTA 数，会直接拒绝运行，因为多 CTA wave 会让 per-CTA 计时高估吞吐。

当前 Thor baseline 来自 `./build_and_run.sh run`：

|模式|bytes/cycle|每 SM bytes/cycle|工作集|
|---|---:|---:|---:|
|`read-same` 请求/广播|31903.430|1595.171|实际触碰 16 B|
|`read-unique`|946.701|47.335|16 MiB|
|`write-unique`|299.373|14.969|16 MiB|

L2 读带宽结论使用 `read-unique`。`write-unique` 只报告为端到端 store-path throughput；不要把它写成纯 L2 write-port 峰值。`read-same` 只保留为同地址请求/广播压力测试。

## 结论口径

- **L2 read peak model**：约 `1024 B/cycle/GPU`。这是由 NCU `pct_of_peak_sustained_elapsed` 口径反推的模型峰值，不是本 microbench 直接测到的 sustained 值。
- **measured L2-hit read sustained**：约 `943-947 B/cycle/GPU`；最新 NCU run 为 `947.207 B/cycle`，LTS throughput 约 `93.2% peak`。
- **L2 write peak model**：约 `512 B/cycle/GPU`。这是由 NCU write-sector peak 口径反推的模型峰值，不是当前 store microbench 打满的实测值。
- **measured global-store sustained**：L2-sized 工作集下约 `235-300 B/cycle/GPU`；最新 NCU run 为 `234.741 B/cycle`，LTS throughput 约 `48.3% peak`。64 KiB 工作集的 `655.278 B/cycle` 是 tiny working set fast case，不作为 L2 写带宽上限。

报告里应把 `1024/512 B/cycle` 写成 **model peak**，把 `947 B/cycle` 和 `235-300 B/cycle` 写成 **measured sustained**，不要混用。

常用控制参数：

```bash
ITERS=8192 BLOCKS_PER_SM=4 THREADS_PER_BLOCK=256 ./build_and_run.sh run
BYTES=$((4 * 1024 * 1024)) ./build_and_run.sh run
./build_and_run.sh run-one --mode read-unique --iters 4096 --bytes 8388608
```

如果目标是 L2-hit 带宽，`BYTES` 应低于 L2 容量；只有在故意引入 DRAM 压力时才把工作集放到 L2 容量以上。

## 验证 sweep

运行容量阶跃和并发饱和验证：

```bash
./build_and_run.sh validate
```

该命令会生成：

- `results/l2_concurrency_sweep.csv`
- `results/l2_capacity_sweep.csv`
- `results/validation_report.md`
- `plots/l2_concurrency_saturation.svg`
- `plots/l2_capacity_staircase.svg`

验证脚本会重复采样并报告中位数。强制 Nsight Compute 验证由 `scripts/ncu_exclusive_monitor.py` 负责，因为 NCU 必须在没有其他 profiler 进程、且 GPU 上没有其他可见 compute 进程时运行。

共享机器上做强制 NCU counter 证明时，用 exclusive monitor，不要直接手动启动 NCU：

```bash
setsid -f bash -c 'echo "$$" > results/ncu/ncu_monitor.pid; exec scripts/ncu_exclusive_monitor.py --poll-seconds 30 --idle-confirm-seconds 300 > results/ncu/ncu_monitor_stdout.log 2>&1'
```

monitor 发现 profiler 进程或 `nvidia-smi` 中存在其他 compute app 时会拒绝启动。设置 `--idle-confirm-seconds 300` 后，它还要求连续 5 分钟空闲后才启动本测试，避免抢到别人任务之间的短暂空窗。

进入 idle window 后，monitor 会先查询本机支持的 NCU metric，再记录 raw NCU CSV 和 `results/ncu/ncu_l2_validation_summary.csv`。候选 metric 包含 L1TEX global load/store bytes、LTS request/sector counters、DRAM read/write bytes。不支持的 metric 会明确列入 summary，不会被当成 0 来解释。

当前 NCU pass 完成于 `2026-07-23 08:47:18 UTC`。这次按要求直接跑，启动前做了瞬时 exclusive pre-check，`idle-confirm` 阈值为 `0.0s`。这个 Thor/NCU 组合不支持直接的 `dram__bytes*` metric，但已采集 read/write 两个 kernel 的 LTS utilization、LTS sector 和 miss-sector counters。

当前对抗式审查记录在 `results/adversarial_review.md`。最重要的修复是：`read-unique` 现在强制生成 `LDG.E.128` SASS，并消费 `uint4` 的所有四个 lane；旧的 scalarized read kernel 结果应丢弃。
