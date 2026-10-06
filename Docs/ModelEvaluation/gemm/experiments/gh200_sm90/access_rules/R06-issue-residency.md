# R06：资源与并发

[总计划](README.md)。状态：6 点 / 39 正式进程完成（循环展开版），CPU 与 SASS 复核通过。代码：`run_r06.py`、`probes/r06.cu`。

## 问题

工作量相同时，每线程寄存器分配和多 CTA 驻留怎样改变 FFMA 服务？用于判断前面的规则是在发射、依赖还是并发不足的条件下取得的。

## 默认矩阵（6 点）

| 子集 | 坐标 | 数量 |
|---|---|---:|
| R06-B，资源与并发 | FP32 FFMA，128 线程、8 条独立链、每 lane 16384 FFMA；每线程约 64/96/128 个寄存器三档；单 CTA / 整卡 grid | 6 |

- 寄存器档位以 ptxas、SASS 与 API 返回的实际值为准。附加的活跃值在计时前准备、计时后消费，确认计时内没有新增更新或 spill。
- 整卡 grid 为 4×SM 数；记录每个 CTA 的首末 SM ID 与 `globaltimer` 区间，按 SM 汇总区间重叠与波次。
- 这 6 点是 128 线程 FFMA 的资源对照；384 线程目标组合的驻留由 R00/V01 给出。

## 时间与单位

- 单 CTA：`clock64` 窗口，报告 warp 指令/cycle 与 FLOP/cycle。
- 整卡：`globaltimer` 包络，报告 GFLOP/s；各 SM 区间单列。
- 各寄存器档位每 lane FFMA 数相同；同时报告 occupancy 上限与整卡完成时间。

手算模板：128 线程 × 16384 FFMA = 4194304 FLOP；单 CTA 20000 cycle 时为 209.7 FLOP/cycle。整卡 528 个 CTA 时工作量乘 528，分母为 `globaltimer` 包络。

## 输出

按参与 warp、实际寄存器分配与并发条件索引的 FFMA 服务，用于 L1 的共享执行与 L3 的任务安排。目标 GEMM 采用 warp specialization 且寄存器重分配成本影响预测时，再补 `setmaxnreg` 的服务测量。

## 条件扩展：R06-A 代码组织（16 点）

kernel 出现与展开、代码大小相关的未解释差别时启用：FP32 FFMA 与整数 add；1/4 warp；循环展开 8/32/128/512 条，每 lane 共 16384 操作，链数固定 8。

- 代码大小以 SASS 字节、动态分支与控制指令数计量。每个展开点核对 16384 次操作、8 条链、消费者、寄存器与 spill。
- 选一个稳定点，把同一序列放在两个不同的代码位置作控制；先核对 SASS 再解释。
- 若大展开出现稳定下降，先区分寄存器、调度、动态控制与代码组织；需要归因到指令 cache 时，再设计动态工作匹配的布局对照。

## 实现与复核

`probes/r06.cu`：128 线程，每 lane 8 条独立 FFMA 链、每链 2048 步（每 lane 16384 FFMA）。计时循环每次迭代每链 32 步，共 8×32=256 条 FFMA、64 次迭代；SASS 核对每次迭代 256 条 FFMA、1 条比较、1 条回跳。三档额外活跃值 40/72/104，CUDA 12.9 原生 sm_90a 下实际每线程 63/95/128 个寄存器，均无 spill；额外值计时前准备、计时后消费，窗口内不更新。ptxas 日志、SASS 与 API 字段按实际档位保留，不用“64/96/128”标签替代实值。

短检查用 32/64 步（1/2 次迭代），全部主累加器和额外值 CPU 复核。单 CTA 与 4×SM 数 CTA 两个 scope 共 6 点；每 CTA 首末 SM、clock64 与 globaltimer 区间原样保存。每 SM 计时区间最大重叠只是窗口并发的观测下界，不等于完整 CTA 驻留；API 给资源上限。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r06.py \
  --output results/gh200_resource_campaign/access_rules/NEW/r06
python3 microbench/gh200_resource_campaign/access_rules/analyze_r06.py \
  --input results/gh200_resource_campaign/access_rules/NEW/r06
```

`--list` 列 6 点；`--smoke` 只做编译和 12 个短进程。输出 `cases.csv`、`rules.json`、`cpu_check.json`、`sass_check.json`、`sm_intervals.json`、`report.md` 与 `plots/`。128 线程 FFMA 资源不代表 384 线程目标 Tensor Core 组合。

## 实测结果（2026-10-06，循环展开版）

[完整报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-r01r06-unrolled/r06/report.md)、[每 SM 计时区间](../../../../../../results/gh200_resource_campaign/access_rules/20261006-r01r06-unrolled/r06/sm_intervals.json)。作业 735191，romeo-a041，GPU-7c184a2e-41ea-3d2b-df5b-1699c95fc1fe，与 R01 同 UUID 串行。6 点 39 正式进程，加 12 短检查共 51 进程，21914496 个结果 CPU 重算通过；最大正式 CV 0.06%，预热全部收敛。

|实际寄存器/thread|API 上限 CTA/SM|单 CTA 窗口 cycle|单 CTA FLOP/cycle|整卡包络 ns|整卡 TFLOP/s|
|---:|---:|---:|---:|---:|---:|
|63|8|17386|241.2|34028.8|65.08|
|95|5|17401|241.0|34224.0|64.71|
|128|4|17401|241.0|34553.6|64.09|

单 CTA 工作量 128×16384×2=4194304 FLOP；整卡 132 SM×4=528 CTA，2214592512 FLOP，TFLOP/s=FLOP/ns/1000。窗口内 SM 时钟按每个 CTA 的 clock64/globaltimer 计算，单 CTA 约 1980 MHz、整卡约 1970 MHz。

单 CTA 每个 SMSP 只有 1 个 warp，8 条独立链足以每 cycle 发射一条 FFMA，达到 256 FLOP/cycle/SM 上限的 94%，剩余约为每 256 条 FFMA 一次的循环成本；三档寄存器差 <0.1%。整卡为 132×256 FLOP/cycle×窗口时钟的 96–98%，与 [EXP-05](../EXP-05-fma.md) 的 65.7 TFLOP/s 一致；寄存器从 63 增到 128，整卡速率降 1.5%（CV 0.06%，10 进程）。每 SM 的 4 个 CTA 计时窗口分两批：第 3 个窗口在前两个之一结束时开始（中位差 −0.4 至 +0.1 µs），所以观测重叠多为 2–3；探针只记录窗口起止，不能区分后两个 CTA 是未驻留还是已驻留但未获发射。FFMA 发射在每 SMSP 1 个 warp 时已接近饱和，这 6 点仍不能给出驻留减少造成的速率惩罚。

2026-10-06 早先的作业 735060 结果（[20261006-b-job735060](../../../../../../results/gh200_resource_campaign/access_rules/20261006-b-job735060/)）每次迭代只有 8 条 FFMA，被约 29 cycle/迭代的循环成本限制（单 CTA 约 70 FLOP/cycle、整卡约 42 TFLOP/s），已被本节取代。
