# 对抗式审查

结论：通过。L2 read 结论可报告为 measured sustained `943-947 B/cycle/GPU`
和 NCU 反推 read model peak `~1024 B/cycle/GPU`；write 只可报告为
end-to-end store-path throughput，不可报告为纯 L2 write-port sustained peak。

这份审查把 L2 throughput microbenchmark 当成一个需要被反驳的实验来检查，记录已经修复的问题和仍然存在的限制。

## 已修复的问题

1. **读字节数最初没有绑定到真实 load 宽度。**

   早期 `read-unique` kernel 把每个源码层面的 `uint4` load 计为 16 B，但代码只消费了部分 `uint4` 字段。`ptxas` 因此可能把 load stream scalarize，导致报告的 B/cycle 不是干净的 128-bit load 结果。

   修复：`read-unique` 现在使用 volatile inline PTX `ld.global.cg.v4.u32`，并消费每个 loaded `uint4` 的四个字段。当前 SASS 中 `read-unique` kernel 包含 `LDG.E.128.STRONG.GPU`。旧的约 470 B/cycle 读结果应丢弃。

2. **power-of-two aliasing 可能伪造容量 sweep。**

   早期 index 更新使用 `round_stride = total_threads * unroll`。工作集大小是 2 的幂，因此 stride 可能和 ring size 共享较大公因子，只反复访问名义工作集的一个子集。

   修复：unique-address 模式现在使用 `round_stride = total_threads * unroll + 1`。该 stride 是奇数，与 2 的幂工作集大小互质。CSV 输出也记录 `index_stride_elements`、`stream_period_iters` 和 `requested_to_working_set_ratio`。

3. **计时 bookkeeping 产生了 local-memory 流量。**

   早期每线程 `clock64()` start value 让 SASS 出现 `LDL/STL`。这部分流量相比主循环很小，但属于不必要的 memory noise。

   修复：只有 `threadIdx.x == 0` 记录 CTA 时间，start timestamp 保存在 shared memory。当前 SASS 检查显示主 benchmark kernel 中没有 `LDL/STL`。

4. **超过 occupancy 的配置可能高估 B/cycle。**

   per-CTA timer 只有在 grid 能放进单个 resident CTA wave 时才有效。

   修复：binary 会调用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`，并拒绝超过该 mode/thread occupancy 上限的 `--blocks-per-sm`。

5. **NCU 验证最初失败，并且 parser 假设了错误的 CSV 形状。**

   第一次 NCU 尝试失败，因为另一个 profiler 占用了 driver profiling resource。后来等到独占窗口后 NCU 成功，但本机 `--page raw --csv` 输出的是宽表 CSV，而不是 `Metric Name` / `Metric Value` 长表。

   修复：exclusive monitor 会等待没有其他 profiler 进程、且 `nvidia-smi` 没有 compute 进程后才 profile benchmark。parser 同时支持 NCU 长表和宽表 CSV。

## 修复后的当前结果

容量 sweep 现在给出干净的 `read-unique` L2-resident 平台：

|工作集|read-unique B/cycle|
|---:|---:|
|1 MiB|940.947|
|4 MiB|943.245|
|8 MiB|945.408|
|16 MiB|946.310|
|32 MiB|938.657|
|64 MiB|374.167|
|128 MiB|169.327|

解释：读路径在 1 MiB 到 32 MiB 之间稳定在约 **943 B/cycle/GPU**，一旦工作集超过 32 MiB L2 就明显下降。并发 sweep 在约 24 active warps/SM 时达到 **951.905 B/cycle/GPU**。

写路径不如读路径干净：

|工作集|write-unique B/cycle|
|---:|---:|
|1 MiB|405.586|
|4 MiB|403.865|
|8 MiB|399.698|
|16 MiB|288.703|
|32 MiB|171.246|
|64 MiB|110.897|
|128 MiB|95.885|

解释：这些数字应视为端到端 store-path throughput，不是纯 L2 write-port limit。store buffering、write combining、eviction、fence/drain 行为仍然都包含在被测路径里。

特别注意：655.278 B/cycle 的 write 原始最高点来自 64 KiB 工作集，这是 tiny working set fast case。它不能和 16 MiB NCU validation 的 265.901 B/cycle 直接比较；同工作集更接近的 capacity sweep 结果是 16 MiB 下 288.703 B/cycle。

## NCU counter 证据

独占 NCU pass 完成于 `2026-07-23 08:47:18 UTC`。这次按要求直接跑，启动前做了瞬时 exclusive pre-check，`idle-confirm` 阈值为 `0.0s`。该 pass 捕获了 read 和 write 两个 validation kernel：

|模式|app clock B/cycle|LTS throughput avg %peak elapsed|LTS throughput max %peak elapsed|LTS bytes/cycle elapsed|LTS read sectors %peak elapsed|LTS write sectors %peak elapsed|LTS hit rate|
|---|---:|---:|---:|---:|---:|---:|---:|
|read-unique|947.207|93.220|93.250|954.620|93.220|0.010|99.74%|
|write-unique|234.741|48.290|48.300|233.860|0.000|45.730|99.72%|

解释：

- 对 `read-unique`，L1TEX load bytes 和 LTS read-sector bytes 都在逻辑请求字节数的约 3.1% 以内，因此 L1 没有吸收主要流量；NCU LTS throughput 利用率约为 93.2% peak。
- 对 `write-unique`，LTS write-sector bytes 与逻辑 store bytes 的差距约 2.9%，write miss-sector bytes 很小；NCU LTS throughput 利用率约为 48.3% peak。这证明 store path 是 L2-resident，但仍不能隔离出纯 L2 write-port limit。

## 剩余限制

1. **本机没有直接 DRAM byte counters。**

   `dram__bytes.sum`、`dram__bytes_read.sum` 和 `dram__bytes_write.sum` 没有出现在本机 Thor/NCU metric query 中。它们已在 `results/ncu/ncu_l2_validation_summary.csv` 里列为 missing。

   缓解方式是使用 LTS miss-sector counters。它们直接显示只有极小一部分请求流量在 L2 miss。

2. **系统级 L2 pollution 被最小化，但无法完全排除。**

   L2 被所有 GPU client 共享。benchmark 使用重复采样和中位数；最新 NCU pass 是按要求直接跑，只做了启动前瞬时 pre-check，没有再等待 300 秒 idle-confirm。它仍然无法排除 firmware、driver、copy engine 或不显示为普通 compute 进程的系统 client。

3. **同地址 read 仍然不是物理带宽结论。**

   `read-same` 只作为请求/广播压力 case 保留，不用于 L2 bandwidth 结论。

## 最强可辩护表述

有了 NCU counters 之后，最强可辩护表述是：

> 在强制 128-bit `ld.global.cg` SASS、odd-stride unique addressing、L2-sized 工作集、occupancy 检查过的高并发，以及 NCU 确认 LTS sector traffic、LTS miss-sector bytes 极低、LTS throughput 利用率约 93% peak 的条件下，Thor 的 L2-hit global-load path 持续吞吐约为 943-947 B/cycle/GPU；当工作集超过 32 MiB L2 后出现明显容量 cliff。

峰值模型和实测持续值要分开写：

|项目|数值|说明|
|---|---:|---|
|L2 read peak model|约 1024 B/cycle/GPU|由 NCU peak 利用率口径反推|
|measured L2-hit read sustained|约 943-947 B/cycle/GPU|本 microbench 实测|
|L2 write peak model|约 512 B/cycle/GPU|由 NCU write-sector peak 口径反推|
|measured global-store sustained|约 235-300 B/cycle/GPU|本 microbench 实测端到端 store path|

不要把 `1024/512 B/cycle` 写成当前实测 sustained。它们是 model peak；当前 write microbench 没有打满 512 B/cycle。

不要把它写成理论 L2 硬件峰值。这个微基准测到的是端到端 global-load throughput，包含 LSU issue、L1/TEX front-end 行为、L2、片上返回网络和 register writeback。
