# Thor/SM110 GEMM 性能上限模型：Microbenchmark 实验设计、采集合同与证据状态

> **文档目的**：说明 Thor/SM110 GEMM 性能上限模型为什么需要这些
> microbenchmark、每个实验具体测什么、如何控制变量、怎样进入模型，以及哪些
> 数字仍不能称为物理上限。
>
> **状态日期**：2026-08-17。
>
> **硬件目标**：单颗 NVIDIA Thor/T5000、compute capability 11.0、20 个 SM、
> MAXN 功耗模式、`sm_110a` 代码路径。
>
> **证据纪律**：runner 存在不等于硬件测量完成；静态编译和 SASS 存在不等于
> runtime correctness；microbenchmark sustained rate 不等于 physical upper；
> 完整 GEMM 不能反向替代独立 component capacity。

配套运行和恢复指令见
[`microbench/README.md`](../../microbench/README.md)，模型公式和最终证据分层见
[`thor_sm110_gemm_performance_bounds.md`](./thor_sm110_gemm_performance_bounds.md)。

## 1. 研究问题与模型接口

定义 \(M\) 为输出矩阵行数，单位 element；定义 \(N\) 为输出矩阵列数，单位
element；定义 \(K\) 为 reduction 长度，单位 element。经典稠密 GEMM 的有用浮点
工作量定义为：

\[
W_{\mathrm{GEMM}}=2MNK,
\]

单位为 FLOP。整数 Tensor Core 路径使用相同计数，但单位写作 OP，不把整数操作
误报为 FLOP。

定义 \(x\) 为一个 GEMM workload；定义 \(w\) 为一个合法 schedule；定义 \(r\)
为被约束的硬件资源；定义 \(Q_r(x,w)\) 为 workload \(x\) 使用 schedule \(w\)
时向资源 \(r\) 提交的工作量，单位可能为 byte、element、FLOP 或 OP；定义 \(c\)
为 microbenchmark 测量合同；定义 \(\widehat C_r(c)\) 为资源 \(r\) 在合同 \(c\)
下的 microbenchmark sustained
rate，单位为对应 work-unit/s。该资源给出的经验时间为：

\[
\widehat T_r(x,w;c)=\frac{Q_r(x,w)}{\widehat C_r(c)}.
\]

只有当合同 \(c\) 与 schedule \(w\) 的 precision、payload、request topology、
thread count、resident CTA、cache residency、SM scope、row stride 和 timed scope
全部匹配时，\(\widehat C_r(c)\) 才能进入经验理想包络。否则模型必须返回
`insufficient_evidence`。

定义 \(P_{\mathrm{obs}}\) 为完整 GEMM 已观测性能，单位 FLOP/s 或 OP/s；定义
\(P_{\mathrm{ub}}\) 为条件物理上界；定义 \(\widehat P_{\mathrm{env}}\) 为
microbenchmark 驱动的经验理想包络。三者不能合并：

\[
P_{\mathrm{obs}}\le P^\star\le P_{\mathrm{ub}},
\]

而 \(\widehat P_{\mathrm{env}}\) 只是可重校准的经验预测，不自动进入上述严格
不等式。

## 2. 当前实验覆盖总览

以下“完整”只表示 runner case matrix 已定义并可审计，不表示本轮 Thor 数据已经
回传。

| 实验面 | runner 合同 | 当前定义覆盖 | 当前硬件证据状态 | 进入模型的作用 |
| --- | --- | ---: | --- | --- |
| Tensor Core compute | 12 precision × 3 full-SM MMA atom | 12/12 precision，36 个 full-SM 点 | 历史 closure 已有；新参数补测不重复 | shape-qualified compute sustained rate |
| 公共 component | TMA、HBM/L2、TMEM、epilogue | 必需公共资源完整 | 历史 closure 已有 | 经验资源 rate |
| TMA payload surface | 5 payload × 2 residency | 10/10 case | 2026-08-17 `-e`：10/10 通过 | payload-indexed per-SM/full-GPU TMA rate |
| L2 duplex + cold read/write-path proxy | 7 cold ratio + 14 L2 ratio | 21/21 case | `-e`：前 20 个通过；最后 `96:1` 暴露旧 binary `ops<=64` 矛盾，整批 fail-closed | read/write 联合服务曲线；不关闭 external write bytes |
| exact TMA topology | schedule × precision | 2/28 pair | 仅 tc5a FP16/BF16 | 禁止把 generic payload curve 冒充具体流水线 |
| independent joint pipeline | TMA/MMA/TMEM 因果组合 | 0 | 未定义独立 runner | overlap、startup、drain、backpressure |
| full GEMM validation | candidate/reference，3 个 N | 当前绘图分支 5/12 precision | 历史 15 case | 数值正确性与端到端反证 |
| strict compute upper | 规格或严格推导 | 7/12 precision 有来源 | 不由 runner 自动补齐 | 条件物理上界 |
| fixed launch cost | startup/launch/fence | runner=0 | 未测 | 小 GEMM wall-time；零浪费上界可显式放松为 0 |

当前机器审计必须保持：

```text
all_performance_parameter_runner_definition_complete = false
physical_memory_duplex_closed = false
cold_external_write_bytes_closed = false
empirical_parameter_runner_definition_complete = false
all_precisions_covered = false
joint_pipeline_surface.complete = false
exact_tma_topology_surface.complete = false
```

## 3. 公共平台与采集合同

定义 \(S=20\ \mathrm{SM/GPU}\) 为目标 Thor 的可用 SM 数；定义
\(n_{\mathrm{trial}}=10\) 为每个 runtime case 的外部独立 trial 数。所有正式采集
必须满足：

- 恰好一个 GPU identity，并证明是 Thor/SM110；
- `nvpmodel -q` 明确包含 `MAXN`；
- Git HEAD 等于 run spec 中的 40-hex `expected_commit`；
- `git status --short --untracked-files=no` 为空；
- source、compile command、binary hash、function-scoped SASS、环境快照和原始
  stdout/CSV 全部进入结果目录；
- 所有 GPU runner 共用 `results/.sm110_gpu_campaign.lock`，禁止并发采集；
- trial timeout 为 120 s，NCU timeout 为 300 s；
- timeout 先对完整进程组发送 `SIGTERM`，有界等待后再发送 `SIGKILL`；
- `termination_failed=true` 时必须重启 Thor，不能继续其他 GPU 任务；
- 每个新代码提交使用新 `RUN_ID`，不能跨 commit 复用旧 `run_spec.json`。

定义 `static_only` 为只执行编译、descriptor、SASS 和 host self-test 的布尔合同。
`static_only=true` 时不得生成 runtime performance line，也不得写成 Thor 测量完成。

## 4. 实验 A：12 精度 Tensor Core compute surface

### 4.1 目的

隔离 `tcgen05.mma` completion throughput，回答不同 precision 和 MMA atom N shape
在 full-SM 条件下能达到的 sustained compute rate。该实验排除 GMEM、TMA、
epilogue、TMEM accumulator readback 和用户可见 global store。

### 4.2 自变量与 case matrix

定义 `precision_id` 为输入编码、accumulator 类型和输出合同的稳定标识。当前 12 个
标识为：

```text
fp16_f32, bf16_f32, tf32_f32, e4m3_f32, e5m2_f32,
e3m2_f32, e2m3_f32, e2m1_f32, mxfp4_f32, nvfp4_f32,
s8_s32, u8_s32
```

定义 \(N_{\mathrm{MMA}}\in\{64,128,256\}\) 为 `M128N*` MMA atom 的 N 维，
单位 element/instruction。每个 precision 有三个 full-SM case：

```text
M128N64, M128N128, M128N256
```

full-SM launch 使用 4 warp/CTA、20 个 CTA 覆盖 20 个不同 SM；single-warp case
只作局部协议诊断，不能和整 GPU full-SM rate 混成一条曲线。

### 4.3 计时和工作量

定义 `iters` 为每个 kernel 中重复发出的 MMA 指令批次数。计时窗口从第一个
`tcgen05.mma` issue 前到 commit barrier 完成后，使用 `%globaltimer` 形成整 GPU
interval，并同时保留 `clock64()` 诊断。rate 由 issued work 除以该 interval 重算，
而不是信任程序打印的标签。

### 4.4 接受条件

- 每个 full-SM case 覆盖 20 个唯一 SM ID；
- descriptor storage bits 与 precision 合同一致；
- source 中没有把 dense case 偷换为 sparse MMA；
- function SASS 包含对应 `UTCHMMA`、`UTCQMMA`、`UTCOMMA` 或 `UTCIMMA`；
- 10 个 trial 全部成功且 timeout contract 一致；
- 可选 NCU 只采每 precision 的 `M128N256` 代表点，不能替代其他 shape 的 timing。

### 4.5 图表

自动生成 single-warp 和 full-SM 两张 SVG。每个 precision 是独立小面板，横轴为
\(N_{\mathrm{MMA}}\)，纵轴为 TFLOP/s 或 TOP/s，点为 median，whisker 为
min–max。浮点和整数单位不混用。

## 5. 实验 B：公共 component surface

### 5.1 目的

为完整 GEMM 的数据搬运、accumulator 回读和 epilogue 提供独立 rate，包括：

- serialized 32 KiB TMA；
- uniform inflight=4 TMA；
- tc5a 四 stage、八 request、A16 KiB+B32 KiB 精确 TMA；
- HBM/L2 单向 read/write；
- block-scale SMEM→TMEM scale ingress；
- `tcgen05.ld` accumulator readback；
- NVFP4 requant epilogue。

定义 `inflight` 为同一 CTA 在等待完成前允许同时在途的 TMA request 数；它不是
pipeline stage 数。定义 `stage_count` 为 schedule 中逻辑双缓冲/多缓冲 stage 数；
同一个 stage 可以包含多个 request，二者不能互换。

### 5.2 精确 tc5a 合同

tc5a 的已测合同为：

```text
stage_count = 4
A payload/stage = 16 KiB
B payload/stage = 32 KiB
requests/stage = 2
total inflight requests = 8
threads/CTA = 192
SMEM = 192 KiB
```

hot-L2 使用单 CTA、单观测 SM，得到 per-SM TMA→SMEM 出口；cold-DRAM 使用
20 CTA 覆盖 20 SM，得到 full-GPU ingress。共享 `l2.read` 仍由另一条整 GPU
资源约束，不能把 per-SM rate 乘 20 后当作可同时实现的 L2 throughput。

### 5.3 TMEM 与 epilogue

定义 `tmem_load_registers` 为一次 `tcgen05.ld.32x32b.xN` 向每线程写入的
32-bit register 数，当前取 8 或 16；定义 `readback_warps` 为同时消费 TMEM 的
warp 数，当前取 1 或 4。两者必须进入 capacity ID，不能只写成一个通用
`tmem.readback` 数字。

NVFP4 epilogue 使用 normal、outlier、constant 三种输入分布，要求 packed E2M1
value 和 scale 与 host reference bit-exact，并保存 function-scoped `LDTM`/`STTM`
证据。

## 6. 实验 C：TMA payload/residency surface

### 6.1 研究假设

32 KiB TMA rate 不能无条件迁移到 4/8/16/64 KiB payload。定义
\(p\in\{4,8,16,32,64\}\ \mathrm{KiB/request}\) 为单条 TMA request payload；
本实验测量 \(\widehat C_{\mathrm{TMA}}(p,\rho)\)，其中 \(\rho\) 为 residency
合同。

### 6.2 Case matrix

| residency \(\rho\) | backing working set | launch scope | payload | case 数 |
| --- | ---: | --- | --- | ---: |
| `hot_l2` | 16 MiB | 1 CTA、1 observed SM | 4/8/16/32/64 KiB | 5 |
| `cold_hbm` | 256 MiB | 20 CTA、20 observed SM | 4/8/16/32/64 KiB | 5 |

定义 `destination_slots=2` 为两个轮换 SMEM destination；定义 `inflight=1` 为每次
issue 后等待完成的 serialized payload curve；定义 `threads_per_cta=128`；定义
`resident_ctas_per_sm=1`。该实验不证明多 request topology，也不替代 tc5a
stage4/inflight8 精确点。

每个 normal trial 的 target issued payload 为约 512 MiB；每个 NCU case 的 target
issued payload 为约 64 MiB。`iterations` 由 target bytes、payload 和 block 数向上
取整，避免大 payload case 因工作量更少而获得不公平优势。

### 6.3 计时公式

定义 \(B_{\mathrm{req}}\) 为 timed launch 的 requested TMA bytes；定义
\(t_{\min}\) 为最早 CTA start globaltimer；定义 \(t_{\max}\) 为最晚 CTA stop
globaltimer。吞吐重算为：

\[
\widehat C_{\mathrm{TMA}}=\frac{B_{\mathrm{req}}}{t_{\max}-t_{\min}}.
\]

hot-L2 只有一个 CTA，因此直接隔离一个 SM 出口；cold-DRAM 使用 full-grid
interval，不采用 `max(clock64 per CTA)` 代替整 GPU makespan。

### 6.4 NCU 接受门禁

定义 \(B_{\mathrm{TMA,NCU}}\) 为
`l1tex__...tma_ld.sum`；定义 \(B_{\mathrm{LTS,NCU}}\) 为 `lts__t_bytes.sum`；
定义 \(H\) 和 \(L\) 分别为 L2 read hit/miss sector 数。必须满足：

\[
B_{\mathrm{TMA,NCU}}\ge0.98B_{\mathrm{req}},\qquad
B_{\mathrm{LTS,NCU}}\ge0.90B_{\mathrm{req}}.
\]

hot-L2 还要求 \(H>L\)；cold-DRAM 使用 \(32L\) 作为 miss-byte proxy，并要求：

\[
32L\ge0.70B_{\mathrm{req}}.
\]

### 6.5 NCU 2025.3.1 CSV 合同

Thor 回传证明 NCU 2025.3.1 raw page 会生成带十进制千分位的基础单位数字，例如：

```text
"8,299,136" ns
"83,886,080" byte
"2,098,918" sector
```

runner 与独立 auditor 均使用严格分组语法解析，并强制 unit row 为 `ns`、`byte`、
`sector`。离线 `ncu --import` 可能把同一 report 缩放为 `ms`/`Mbyte`；该文件只作
诊断，不能覆盖原始 `raw.csv`，也不能在未换算单位时进入证据。

### 6.6 当前失败记录

`thor-t5000-parameter-plots-maxn-20260817-a-tma-payload` 在第一个
`tma_l2_hit_4k_slots2_single_sm` case 已生成合法 `profile.ncu-rep` 和 raw kernel
row，但旧 host parser 对 `float("8,299,136")` 抛出异常。该目录必须保留为失败
诊断，不能补写成成功结果。`-b` 又因后台 shell 缺少 `nvcc` PATH 在 GPU 采集前
失败；`-c` 已完成 10/10 TMA payload，但 duplex 在 metric preflight fail-closed。
修改 cold duplex 证据合同后的新提交必须使用新的 `-d` run ID，并在同一提交上
重采 TMA 与 duplex，不能把 `-c` TMA 与 `-d` duplex 拼成一次 composite run。
随后 `-e` 已证明新 proxy metric 合同可执行，但最后一个不可约 `96:1` case 超过
旧 binary 的 host 参数上限 64。修复把显式上限提高并冻结为 128；下一次完整重跑
必须使用新的 `-f` run ID，不能复用 `-e` 的前 20 个 duplex case。

## 7. 实验 D：hot-L2 duplex 与 cold-DRAM-read/write-path proxy surface

### 7.1 研究假设

独立的 read peak 和 write peak 只能给出两个轴向约束，不能证明二者能够同时满速。
定义 \(r\) 为每 iteration 的逻辑 read operation 数；定义 \(w\) 为逻辑 write
operation 数；定义约化比 \(r:w\) 为该 case 的 issued byte ratio。每个逻辑
operation 由 8 条独立 128-bit transaction 组成，因此工作量为 128 B。

定义 \(B_R\) 和 \(B_W\) 为 timed launch 的 read/write bytes；定义 read share：

\[
x_R=\frac{B_R}{B_R+B_W}.
\]

实验输出 total/read/write GB/s 随 \(x_R\) 的联合服务曲线，而不是把两个单向峰值
机械相加。

### 7.2 cold proxy ratio matrix

HBM ratio 从 12 precision、`N=1024/2048/4096` 的 unique input/output byte
accounting 自动推导并约分：

```text
1:4, 17:64, 9:32, 3:8, 1:2, 1:1, 2:1
```

working set 为每方向 256 MiB；它定量证明 cold-DRAM reads 与 write-path issue，
不声称 physical external write-byte closure。

### 7.3 L2 ratio matrix

L2 ratio 从 schedule-level repeated TMA input bytes 与 output bytes 自动推导：

```text
27:16, 3:1, 27:8, 4:1, 6:1, 27:4, 8:1,
12:1, 16:1, 24:1, 32:1, 48:1, 64:1, 96:1
```

`96:1` 与 1 互质，不能在 `ops<=64` 内等价缩放。binary、runner manifest 和 auditor
共同冻结 `max_operation_groups=128`；任何超过 128 的派生 ratio 在启动前失败。
device kernel 使用 runtime `operation_groups` 循环，每个 group 重用固定的 8 个
`uint4` 局部 load 值，因此该修改放宽的是 host legality，不是静态展开 96 份寄存器。

working set 为每方向 16 MiB，目标是 hot-L2 traffic。`27:16`、`27:8`、`27:4`
显式覆盖 block-scaled value+scale transport，不得被近似成整数 ratio。

### 7.4 Launch 和工作量

每个 case 使用 4 CTA/SM、256 thread/CTA、20 SM，共 80 CTA。定义
`target_bytes=512 MiB` 为 normal trial 的目标 total issued bytes；定义
`ncu_target_bytes=64 MiB` 为 profiler case 的目标。`iterations` 由 \(r+w\) 自动
缩放。

kernel 同时包含 `LDG.E.128` 和 `STG.E.128`。store 不依赖 loaded value，避免把
数据依赖延迟误当成共享总线容量；所有 load lane 保持 live，避免 compiler 删除
或缩窄 transaction。

### 7.5 NCU 接受门禁

定义 \(S_R\) 和 \(S_W\) 为 L2 read/write sector 数，每 sector 32 B。要求：

\[
32S_R\ge0.90B_R,\qquad 32S_W\ge0.90B_W.
\]

hot-L2 要求 read hit sectors 大于 miss sectors。cold-DRAM 定义
\(L_R\) 为 L2 read lookup miss sector 数，并要求：

\[
32L_R\ge0.60B_R.
\]

Thor 不暴露 `dram__bytes_op_read/write.sum`。因此 cold case 只能定量证明 read
离开 L2；\(32S_W\ge0.90B_W\) 证明 write 进入 L2 write path，但不证明等量
physical DRAM write bytes。结果必须记录
`qualification=cold_dram_read_plus_write_path_proxy` 和
`external_write_bytes_proven=false`。

`mcc__dram_throughput_op_read/write...pct_of_peak_sustained_*` 表示达到 sustained
peak 的百分比，不是 byte counter；在没有 peak-rate、MCC instance 聚合和 timed
duration 换算合同前，只能作诊断，不能替代上述字节门禁。

独立 auditor 重新解析带千分位的 raw CSV、检查 base unit row，并逐 metric 比较
summary 与 raw value；只验证文件 hash 不足以通过。

## 8. 实验 E：exact TMA topology 缺口

TMA payload surface 的 10 个点只覆盖 `inflight=1`、两个 destination slots 的
uniform request。模型实际需要的键是：

```text
(schedule_id, precision_id, payload pattern, request count,
 stage count, threads, SM scope, residency, row stride)
```

当前 schedule manifest 导出 28 个 schedule/precision pair，只有以下两个有精确
独立 TMA 证据：

```text
tc5a_m128n256k64_stage4 × fp16_f32
tc5a_m128n256k64_stage4 × bf16_f32
```

其余 26 个 pair 必须保留 missing。下一轮 exact-topology runner 至少需要分组覆盖：

- generic stage2 `M128N64/N128/N256`；
- direct-SMEM FP6/raw E2M1 byte-container；
- MXFP4/NVFP4 block-scaled value+scale；
- `N=1024/2048/4096` 对应共同 A/B row stride；
- per-SM hot-L2 和 full-GPU cold-DRAM 两个 scope。

不能按 precision 数机械复制同一个 payload case；只有合同相同的实现才能共享一个
capacity ID。

## 9. 实验 F：independent joint pipeline 缺口

定义 \(L_r\) 为资源 stage (r) 的 latency，单位 cycle；定义 \(I_r\) 为 initiation
interval，单位 cycle/request；定义 startup 和 drain 为流水启动/排空周期。独立
component peak 取最大值只在理想完全重叠假设下成立。以下效应需要 joint runner：

- TMA 与 MMA 的 producer/consumer backpressure；
- MMA completion 与 TMEM readback dependency；
- TMEM readback 与 global store overlap；
- stage barrier、startup、drain；
- CTA wave 和 tail-wave；
- cache residency 与并发流量相互改变。

当前 `joint_pipeline_surface.independent_runner_defined=false`。完整 GEMM 可以反证一组
component prediction，但不能唯一反演每个 \(L_r\) 或 \(I_r\)，因此不把 full-GEMM
time 自动拟合成独立 joint capacity。

## 10. 实验 G：完整 GEMM candidate/reference validation

定义 `candidate` 为被评估的自研完整 GEMM backend；定义 `reference` 为相同输入
precision、相同 accumulator/output 合同的数值和性能 denominator。每个 precision
冻结 `N=1024/2048` calibration 和 `N=4096` holdout，各 10 个外部 trial。

两级 correctness 门禁为：

1. reference 使用实际量化后的输入，在 host 上检查 64 个确定性 output sample；
2. candidate 完整输出矩阵与已检查 reference 比较。

S8→S32 要求 bit-exact；浮点 accumulator 使用显式 `atol`/`rtol`。性能图同时输出
candidate/reference 绝对曲线和百分比曲线。

当前 `codex/sm110-closure-plots` 线冻结 5 个 precision、15 个 case。另一条
`codex/sm110-all-precision-closure` 分支上的 E5M2 runner 不能在未合并代码和重新
冻结 run spec 前被写成本分支的硬件证据。

## 11. 自动图表与可视化边界

每个 runner 在写 `summary.json` 后、写 `COMPLETE` 前生成 SVG：

| campaign | 自动图 |
| --- | --- |
| compute | single-warp/full-SM throughput vs MMA N |
| TMA payload | GB/s vs payload KiB，按 residency 分面 |
| duplex | total/read/write GB/s vs read share，HBM/L2 分面 |
| component | 按兼容单位和精确合同分组的 bar/point 图 |
| full GEMM | candidate/reference throughput 与 ratio vs N |
| closure report | observed/reference/empirical/upper 绝对图与 ratio 图 |

`plots/manifest.json` 保存 source JSON 相对路径、source SHA-256、plot generator
SHA-256 和每张 SVG 的 SHA-256。图只是派生展示：

- missing evidence 保持断点或不出现，不做插值；
- `fixed_seconds=0` 不画成实测值；
- per-SM 和 GPU-shared 不用同一个 scope 标签；
- FLOP/s 与 OP/s 不共用同一轴；
- byte/s 与 element/s 不共用同一轴；
- component 中语义不同的资源不用误导性折线连接。

## 12. 原始工件、独立审计与状态机

正式结果目录必须至少包含：

```text
run_spec.json
environment.json
environment_snapshots.jsonl
campaign_status.json
progress.jsonl
cases/<case-id>/result.json
cases/<case-id>/trials.jsonl
cases/<case-id>/ncu/profile.ncu-rep
cases/<case-id>/ncu/raw.csv
cases/<case-id>/ncu/summary.json
summary.json
plots/manifest.json
plots/*.svg
COMPLETE
```

状态语义：

| 状态 | 含义 |
| --- | --- |
| `running` | runner 活跃，结果可不完整 |
| `failed` | 已保留失败位置和错误；不得写 `COMPLETE` |
| `static_complete` | 仅静态合同完成；不是 runtime evidence |
| `complete` | runner case 完整；仍需 independent auditor |
| `PARAMETER_SUPPLEMENT_COMPLETE` | TMA 和 duplex 均 complete 且两个 auditor 通过 |

独立 auditor 不信任 summary 中的派生数字；它重新解析 raw stdout/CSV，重算 byte
work、rate、trial statistics、NCU traffic ratio、residency 和 hash。任何 commit、
source、SASS、unit、timeout 或 cardinality 不一致都 fail closed。

## 13. 推荐采集顺序

1. host regression、manifest audit 和 static preflight；
2. TMA payload 10 case，全部强制 NCU；
3. TMA independent audit；
4. duplex 21 case，全部强制 NCU；
5. duplex independent audit；
6. 自动图表和 manifest 检查；
7. 独立结果分支提交；
8. 模型 import，但仍只关闭 payload/duplex 对应缺口；
9. 设计 exact-topology 和 independent joint-pipeline 下一轮 runner；
10. 最后才重新评估 \(\widehat P_{\mathrm{env}}\) 与完整 GEMM observation。

如果任一阶段失败，后续阶段不得启动。新的修复 commit 使用新的 run ID；旧失败目录
保留，不复制 trial 或 `.ncu-rep` 到新目录。

## 14. Microbenchmark 源码与 runner 来源

| 研究对象 | CUDA/核心源码 | runner | 独立 auditor | 说明文档 |
| --- | --- | --- | --- | --- |
| Tensor Core compute | runtime 生成的 per-case CUDA source；descriptor 逻辑在 `tcgen05_descriptors.py` | [`run_compute_campaign.py`](../../microbench/sm110_gemm_campaign/run_compute_campaign.py) | [`audit_campaign.py`](../../microbench/sm110_gemm_campaign/audit_campaign.py) | [`README.md`](../../microbench/sm110_gemm_campaign/README.md) |
| TMA payload | [`tma_gmem_smem_bandwidth.cu`](../../microbench/07_tma_gmem_smem_bandwidth/tma_gmem_smem_bandwidth.cu) | [`run_tma_payload_campaign.py`](../../microbench/sm110_tma_payload_campaign/run_tma_payload_campaign.py) | [`audit_campaign.py`](../../microbench/sm110_tma_payload_campaign/audit_campaign.py) | [`README.md`](../../microbench/sm110_tma_payload_campaign/README.md) |
| memory duplex | [`memory_path_bandwidth.cu`](../../microbench/14_memory_path_bandwidth/memory_path_bandwidth.cu) | [`run_memory_duplex_campaign.py`](../../microbench/sm110_memory_duplex_campaign/run_memory_duplex_campaign.py) | [`audit_campaign.py`](../../microbench/sm110_memory_duplex_campaign/audit_campaign.py) | [`README.md`](../../microbench/sm110_memory_duplex_campaign/README.md) |
| public component | TMA、memory、TMEM、epilogue 多源码，由 runner dependency manifest 冻结 | [`run_component_campaign.py`](../../microbench/sm110_gemm_component_campaign/run_component_campaign.py) | [`audit_campaign.py`](../../microbench/sm110_gemm_component_campaign/audit_campaign.py) | [`README.md`](../../microbench/sm110_gemm_component_campaign/README.md) |
| TMEM readback | [`tmem_readback_bandwidth.cu`](../../microbench/12_tmem_readback_bandwidth/tmem_readback_bandwidth.cu) | component runner | component auditor | component README |
| scale ingress | [`tmem_scale_ingress_bandwidth.cu`](../../microbench/13_tmem_scale_ingress_bandwidth/tmem_scale_ingress_bandwidth.cu) | component runner | component auditor | component README |
| HBM/L2 single direction | [`memory_path_bandwidth.cu`](../../microbench/14_memory_path_bandwidth/memory_path_bandwidth.cu) | component runner | component auditor | [`README.md`](../../microbench/14_memory_path_bandwidth/README.md) |
| full GEMM | FP16、quant、extended benchmark 源码由 case manifest 指定 | [`run_full_gemm_campaign.py`](../../microbench/sm110_full_gemm_campaign/run_full_gemm_campaign.py) | [`audit_campaign.py`](../../microbench/sm110_full_gemm_campaign/audit_campaign.py) | [`README.md`](../../microbench/sm110_full_gemm_campaign/README.md) |
| automatic plots | [`campaign_plots.py`](../../scripts/sm110_gemm_model/campaign_plots.py) | [`plot_campaign_results.py`](../../scripts/sm110_gemm_model/plot_campaign_results.py) | source/plot SHA manifest | [`README.md`](../../scripts/sm110_gemm_model/README.md) |
| coverage audit | model manifests 与上述 runner cases | [`runner_coverage.py`](../../scripts/sm110_gemm_model/runner_coverage.py) | `test_runner_coverage.py` | [`sm110_gemm_runner_adversarial_audit.md`](./sm110_gemm_runner_adversarial_audit.md) |

总协调器为
[`run_sm110_parameter_supplement.sh`](../../microbench/run_sm110_parameter_supplement.sh)。
它只顺序运行 TMA payload 和 memory duplex；不能被描述为 exact-topology 或 joint
pipeline runner。

## 15. 完成判据

本轮 parameter supplement 只有在以下条件全部为真时完成：

```text
TMA case_count = 10
TMA trial_count/case = 10
TMA NCU/case = complete and independently audited
duplex case_count = 21
duplex trial_count/case = 10
duplex NCU/case = complete and independently audited
all source/binary/SASS/environment hashes match
all base units and numeric grammar pass
plots are regenerated from the final summary SHA
orchestrator log contains PARAMETER_SUPPLEMENT_COMPLETE
result branch and exact result commit are returned
```

即使以上全部完成，仍不能把以下字段改成 true：

```text
exact_tma_topology_surface.complete
joint_pipeline_surface.complete
all_precisions_covered
all_performance_parameter_runner_definition_complete
```

这些状态只能由各自缺失的独立实验合同关闭，不能由“已有数字很多”推断完成。
