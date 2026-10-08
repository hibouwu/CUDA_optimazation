# R15：输出服务

v4的18个plain条件观测已完成并独立重算；16个六事件trace条件满足5%扰动门槛。两个TMA-only的v4 trace超标，v5同次配对补测的末块read/full两事件已通过，缺失的事件仍未验证，不能据此判定完整输出阶段。[当前计划](PLAN.md)。

## 结论

- 原问题是小 tile 每个 tile 约 1500 cycle 不随输出缩小的固定段。本组探针用标量 STS，CUTLASS 的 epilogue 用 STSM、向量化与寄存器重分配，所以结果只说明本探针的整理、TMA 写回与并发关系，不能作为 CUTLASS 输出常数，那 1500 cycle 仍未解释。
- 要解释它，应直接在 CUTLASS epilogue 内打点（V06 按此从实际 kernel 校准输出段）。

## 问题与矩阵

固定M128，FP32输出N128/256（64/128KiB）。两个输出warpgroup按M64N128/M64N256累加器片段映射整理输出，第三个warpgroup独立执行FP16 SS `m64n128k16`，每16KiB输出块16条MMA。

| 输出缓冲 | 模式 | 点数 |
|---|---|---:|
| 单缓冲 | reg→SMEM、TMA-only、完整输出、background-only、串行、并发 | 2×6=12 |
| 双缓冲 | 完整输出、串行、并发 | 2×3=6 |

统一384线程、72KiB动态SMEM预留：两个16KiB输出槽及40KiB背景输入区；背景实际使用24KiB K-major SW128输入。v4全部48个编译specialization均0 stack/0 spill，最大150 registers/thread。各模式实际寄存器与API occupancy另行记录，统一预留不等于相同驻留。

## 组织与工作量

每块一条`box={128,32}`的FP32 TMA请求，共16KiB；64/128KiB分别4/8块。SMEM按32行×128列row-major打包。`chunk_m=chunk%4`、`chunk_n=chunk/4`，目标`(x,y)=(128*chunk_n,32*chunk_m)`，输出WG为`chunk_m/2`。

每块64个活动线程，各执行64条`st.volatile.shared.b32`，合计4096个FP32写。片段共用一份按角色分配的寄存器数组，整理仅保留一个源基址，偏移为立即数。CPU检查确认lane/register映射完整、无重复，每个16KiB块无缺口。

输出`ldd=N+32`，padding保留公共sentinel。TMA-only源在窗口外预填；各模式使用相同的行mod16、列mod128非均匀dyadic见证。寄存器片段按HGMMA映射初始化，该见证不代表任意GEMM数值分布。reg→SMEM和background-only不写D，逻辑D和padding都保持sentinel。

背景每条MMA为262144 FLOP，每块16条即4194304 FLOP。每lane64个累加结果全部保存，共8192个FP32，参考为`repeats×chunks`。正式每进程32个输出序列；输出大小、模式及背景工作量在配对内保持一致。

## 计量边界

输入、TMA-only源及角色寄存器初始化在窗口外。两端都经过共享发布词、CTA barrier、共享读及消费该读的ISETP，随后才读取clock64/globaltimer。背景最终wait0后消费checksum，输出在非`.read` bulk wait0后会合取终点；完整BG和SMEM见证的global保存位于服务窗口之外。

单缓冲覆盖前read0；双缓冲覆盖前read1，最多两个source group在途。末尾read0后再执行完整wait0。串行每块完整输出完成后启动背景；并发让输出整理/TMA与独立背景同时推进，块末会合。

v4仅观察最后repeat的最后chunk，其余trace记录保持零：

| 字段 | 实际含义 |
|---|---|
| `prepare` | 整理或预填源经过proxy fence与256输出线程会合 |
| `issued` | 一条16KiB请求发出并commit后的推进点 |
| `read_done` | `wait_group.read`返回，源读完 |
| `released` | 源读完后的发布会合与guard读取，源槽可复用 |
| `full_done` | 非`.read`的`wait_group 0`返回，完整输出完成 |
| `background_done` | 第三WG的16条MMA与最终wait0返回 |

源释放与完整输出是不同事件，末块的release会合可能晚于full_done。`.read`不能当作写完成。`read_done-issued`、`full_done-issued`包含实际等待、会合和排空，不能重命名为裸TMA延迟或物理带宽。reg→SMEM模式的总窗口也包含循环、发布、同步和末尾消费，不由缺失的整理起点推算裸STS延迟。

## 正式结果

设备romeo-a053，GPU-54896349-d69d-9358-b526-433454c04733，作业735876，CUDA12.9.41、sm_90a、CUTLASS3.9.2、`-DNDEBUG`。

正式18条件×3个plain/trace配对=108进程；代表12进程单独统计。4866048个保存FP32位置（含padding sentinel）全部核对，逻辑D/BG/SMEM误差0，末块选择器与事件顺序通过。所有正式预热收敛，最大plain进程CV约0.2194%。代表full/concurrent的trace扰动分别为3.70774%/2.81227%，因此进入正式矩阵。

| 输出KiB | buffer | 模式 | plain cycle/序列 | 最大trace扰动 | trace判定 |
|---:|---:|---|---:|---:|---|
| 64 | 1 | reg_smem | 4923.750 | 4.6425% | 通过 |
| 64 | 1 | tma_only | 3037.281 | 10.0954% | 扰动超限，不作定量判定 |
| 64 | 1 | full_output | 7144.500 | 2.8243% | 通过 |
| 64 | 1 | background | 4680.406 | 0.4714% | 通过 |
| 64 | 1 | serial | 11669.125 | 2.1783% | 通过 |
| 64 | 1 | concurrent | 10167.969 | 1.9160% | 通过 |
| 64 | 2 | full_output | 5897.875 | 3.0557% | 通过 |
| 64 | 2 | serial | 11687.000 | 2.0990% | 通过 |
| 64 | 2 | concurrent | 8838.969 | 3.1586% | 通过 |
| 128 | 1 | reg_smem | 9864.094 | 3.2511% | 通过 |
| 128 | 1 | tma_only | 5920.750 | 9.3017% | 扰动超限，不作定量判定 |
| 128 | 1 | full_output | 14288.438 | 2.6324% | 通过 |
| 128 | 1 | background | 9256.594 | 0.3109% | 通过 |
| 128 | 1 | serial | 23402.750 | 3.0150% | 通过 |
| 128 | 1 | concurrent | 20036.469 | 1.8755% | 通过 |
| 128 | 2 | full_output | 11303.406 | 3.5540% | 通过 |
| 128 | 2 | serial | 23423.875 | 3.0528% | 通过 |
| 128 | 2 | concurrent | 17061.188 | 2.4326% | 通过 |

上表plain时间为每进程32序列总窗口的中位数除32，没有减除控制成本。TMA-only64KiB trace扰动10.0933–10.0954%，128KiB为9.2968–9.3017%；这两个plain观测保留，v4六事件trace扰动超限，不作定量判定，v5另列如下。

同工作量serial/concurrent完成窗口缩短如下：

| 输出KiB | buffer | 窗口缩短 |
|---:|---:|---:|
| 64 | 1 | 12.864% |
| 64 | 2 | 24.369% |
| 128 | 1 | 14.384% |
| 128 | 2 | 27.163% |

完整结果和CPU入口：[独立报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R15-job735876-v4/final-analysis-A/report.md)、[条件/资源表](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R15-job735876-v4/final-analysis-A/cases.csv)、[末块事件](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R15-job735876-v4/final-analysis-A/selected-events.csv)、[独立重算脚本](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R15-job735876-v4/final-analysis-A/replay.py)。冻结source/build/samples及哈希已回收。GPU程序保存elapsed标量而未保存原始begin/end数值，独立复核可重算统计，不能声称再次相减了原始时钟；边界依据为已审查的guard和等待SASS。

## 可用规则与限制

本组可提供上述scalar STS、16KiB请求、独立K-major背景、固定线程/SMEM条件下的完成窗口查表，以及同工作量serial/concurrent差值。它没有复现CUTLASS epilogue的STSM/向量化/fusion、producer40/consumer232动态配额、B Major::MN或cluster多播；未经指令、资源和生命周期映射，不能直接作为CUTLASS输出常数，也不能从配对差识别物理bank/端口争用。

v5只复核两个TMA-only条件：末块仅记录read/full事件，同次plain/trace相邻、随机顺序，各3进程，必要时补到10。本次12进程（两点各3对）通过。v5 plain仅作本次扰动对照，不与v4合并或择优；其余16点不重跑。未观察的prepare/issued/released保持零，不补造时间，5%门槛不变。

v5结果（固定32repeat，仅本次相邻随机配对，不与v4混统计）：

| TMA-only输出KiB | pairs | plain中位cycle | trace中位cycle | 配对扰动范围 |
|---:|---:|---:|---:|---:|
| 64 | 3 | 97193 | 99035 | 1.8931%–1.8952% |
| 128 | 3 | 189464 | 191102 | 0.5678%–0.8661% |

12个进程、540672个保存FP32位置独立核对，D/BG/SMEM误差0、padding正确，预热全部收敛。
每个trace仅末块read_done/full_done非零，其余事件为零；两点read→full均48 cycle，此差含
记录写入、等待和控制成本，不能称裸TMA延迟。plain与trace实际寄存器分别同为76/140，均无spill。
完整原始归档为`20261007-R15-job735876-v5/`；[独立报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R15-job735876-v5/independent-review-A/report.md)与同目录replay.py、pairs.csv、summary.json、identity.json均已保存。

v1/v2的spill构建日志保留在对应原始运行目录，本地工具链诊断保留在`/tmp/gh200-gaps-local/R15/`；v3逐块trace超标证据保留，当前交付采用v4原生无spill与正式结果。

## 复现

CPU与矩阵：

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r15.py cpu-check
python3 microbench/gh200_resource_campaign/access_rules/run_r15.py list
```

CUDA12.9构建与代表点，由主对话统一GPU调度：

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r15.py build \
  --cutlass-root results/gh200_resource_campaign/access_rules/20261006-r00-job734996/formal-v1/source/cutlass \
  --output /新运行目录/r15
python3 /新运行目录/r15/source/run_r15.py sample \
  --output /新运行目录/r15 --set representative
```

代表点通过后采正式矩阵；复现v4必须使用归档中的v4源码版本：

```bash
python3 /新运行目录/r15/source/run_r15.py sample \
  --output /新运行目录/r15 --set formal --repeats 32
python3 microbench/gh200_resource_campaign/access_rules/analyze_r15.py \
  --input /新运行目录/r15 --output /新独立复核目录
```

v5两点扰动复核使用新运行目录：

```bash
python3 /新运行目录/r15-v5/source/run_r15.py sample \
  --output /新运行目录/r15-v5 --set tma-trace-recheck
```

原source/build/samples不替换；分析修订写新子目录，不同UUID不共同拟合。
