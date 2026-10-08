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

<a id="cutlass-epilogue"></a>

## 实际 CUTLASS 输出窗口的已有观测

本页探针不能直接给出 CUTLASS 输出常数。实际 epilogue 的观测分散在下列页面，在此汇总以便维护输出规则；数值以原页为准，不与本页探针合并拟合。

| 来源 | 观测 | 窗口终点 |
|---|---|---|
| [R09](R09-inkernel-clock-stages.md) | 128×256 配置 M=N=2048、128 个 CTA 同时输出时 epilogue 5470–5850 cycle；M=N=256、2 个 CTA 时约 4130 | `store_tail` 返回，即 `.read` 等待返回 |
| [V08 校准](V08-wider-validation.md#校准中得到的事实) | cfg_a 最后一个 tile 的 epilogue 几乎都是约 1922 cycle；V07 的 2161 是全部后续 tile 的均值 | cooperative 为 `store()` 返回，pingpong 为 `store_tail()` 返回 |
| [V08 失败原因](V08-wider-validation.md#失败原因测后诊断不改判定) | cfg_c 单 tile epilogue 在整卡 CTA 条件约 5800 cycle，部分 CTA 条件约 4040；这是旧诊断的静态规模描述，实际重叠见下节 | 同上 |

这些观测提示输出窗口与参与规模有关，但静态 CTA 数不等于实际同时输出者。各终点只说明源缓冲已读完或 store 调用已返回，不证明全局写入已完成。

## V08 输出窗口离线复核（2026-10-09）

**约 4040/5800 cycle 的差异与输出错峰相容，现有记录不能直接证明重叠是原因。** 本次读取 V08 的 cfg_a/c 共 78 个条件、780 次 stamped 和 780 次 ends 调用；两种调用分别分析，不拼接时间线。原始压缩记录 SHA、数值检查和工作坐标沿用原分析器复核；新结果为[输出与候选](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/B-20261009-output-issuer-v2/output.json)。这是事后诊断，不改 V08 冻结参数或成绩。

先区分两个输出口径。历史模型合并 cooperative 两个 consumer 的事件，令 `E = max(EPI_DONE) − max(EPI_PERMIT)`。实际发出 TMA 的 warp 位于 role2；其入口线程 thread256 的 `store()` 窗口是 `E_issue = EPI_DONE_role2 − EPI_PERMIT_role2`。这里的 issue 下标只标识 issuer，不表示第一条 TMA 指令的发出时刻：窗口还包含准备、协作同步、SMEM 搬运与等待。旧源码的 cooperative kernel 在 `store()` 前后打点，epilogue 内由 `thread_idx / 32 == 0` 的 warp 发出 TMA。role1 的 permit 通常更晚，因此合并窗口比 issuer 窗口短约 500–600 cycle，不能静默互换。

逐 tile 只有 `clock64`；`globaltimer` 只有 CTA 入口和最终 `store_tail()` 后的记录。对同一次 stamped 调用、同一 CTA，使用下面的**仿射估计**定位 role2 输出端点：

```text
t_hat(c) = entry_ns + (c − entry_cycle) × (final_ns − entry_ns) / (final_cycle − entry_cycle)
Nbar_i = integral_[start_i,end_i] N(t) dt / (end_i − start_i)
```

`N(t)` 计数至少有一个有效输出元素、尚处于 role2 `store()` 窗口的 CTA，包含自身；它不是物理 store 队列占用。`Nbar` 是每个窗口的平均重叠数，再对 CTA 取中位、对十次进程取中位。假设是该 CTA 全窗口的 cycle/ns 可用于内部端点；没有内部 globaltimer，无法验证。另用本次调用各 CTA 完整窗口 cycle/ns 的中位值，分别从入口、最终退出反推端点，检验对换算方式的敏感性。这三种方式都只使用同次 stamped 调用。

| cfg_c 条件 | K | 静态 CTA 数 | 历史 E / issuer E，cycle | 仿射 Nbar | 入口锚点 / 退出锚点 Nbar |
|---|---:|---:|---:|---:|---:|
| g1 | 1024 | 70 | 4043.0 / 4626.0 | 65.3 | 66.0 / 65.2 |
| g3 | 4096 | 48 | 4042.5 / 4629.8 | 43.9 | 45.9 / 43.8 |
| c1_r_k512 | 512 | 132 | 5865.8 / 6270.3 | 121.8 | 121.0 / 121.8 |
| c2_k4096 | 4096 | 132 | 5626.5 / 6118.5 | 118.9 | 121.8 / 118.7 |
| c6_longk | 16384 | 132 | 4885.5 / 5397.5 | 94.6 | 120.5 / 94.4 |
| h08 | 24576 | 128 | 4213.0 / 4785.5 | 77.8 | 118.2 / 77.5 |
| h09 | 768 | 60 | 4043.0 / 4625.0 | 56.2 | 56.7 / 56.1 |

这些都是每 CTA 只做一个 tile 的条件。E 也先取进程内中位再取进程间中位，与旧 `E_last` 池化所有 CTA/进程的数值略有不同，未改写旧 summary。g2 的 trace 扰动为 5.0167%，驱逐版 g1 为 7.8624%，保留结果但不参与本次候选拟合。多 tile 数据在 JSON 中按每 CTA 的首、中、末和单 tile 分开，不能把单 tile 常数直接用于中间输出。

候选形式为 `E_issue = max(E_floor, 131072 × Nbar / B_eff)`；131072 B 是 cfg_c 完整 FP32 输出 tile 的逻辑字节。用 g1/g3 的平台值固定 `E_floor=4627.875 cycle`，这不是已测得的孤立单 CTA 成本。仅对表中五个合格校准点 g1/g3/c1/c2/c6 最小化周期平方误差，仿射重叠给出 `B_eff=2482.81 B/cycle`，拟合残差为约 −7.46%～+2.59%；它只是条件比例参数，不能解释为物理 HBM 带宽。

对已经看过的 h08 做事后代入：静态 CTA 候选误差 +20.14%，仿射重叠候选 −3.29%，入口锚点候选 +20.95%。历史 E 口径也保存在 JSON 中，对应 +25.65%、−4.04%、+26.51%。短窗口的高低两档在三种换算下方向一致；长 K 的结论明显依赖端点换算。此外 Nbar 用已测输出窗口计算，包含待解释的 E 本身，不能作为独立的测前预测输入。由此不能宣布输出公式已经确定。

最小补测是给同次调用的 role2 `EPI_PERMIT/EPI_DONE` 增加 globaltimer，保留 clock64 和最终 store_tail 端点；本次实际准备的三条件见下节。若要归因于并发而非规模、K 或前序供给，固定一个整卡单 tile 条件及全部有效输出字节，仅改变主循环后输出启动的错峰，并直接量出重叠。现有证据不需要计数器就能完成这一步；全局写完成另需对应完成端点。

```bash
python3 microbench/gh200_resource_campaign/access_rules/analyze_r15.py --v08-output \
  --input /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1 \
  --output <该run下新的reanalysis目录>
```

## 直接输出 globaltimer 的最小补测准备（2026-10-09）

本节只完成专用配置与 CPU 分析入口，未编译或采样新增打点。基于公共框架 `6338653`，沿用 cfg_c 的 tile256×128×64、cluster1×2、四 stage、dyadic/seed17、swizzle1、无驱逐、默认全部 SM；三条全部改列为 `ctrl`，不是新留出。

| 条件 | M×N×K | 预期 CTA 数 / 每 CTA tile 数 | 对照用途 |
|---|---:|---:|---|
| cfg_c_g3 | 1024×1536×4096 | 48 / 1 | 与 c2 同 K，比较规模相关的输出重叠 |
| cfg_c_c2_k4096 | 1536×2816×4096 | 132 / 1 | 两组对照共用基准 |
| cfg_c_c6_longk | 1536×2816×16384 | 132 / 1 | 与 c2 同 M/N、grid 和输出字节，改变 K |

配置为 [configs/r15-output-ns.json](../../../../../../microbench/gh200_resource_campaign/access_rules/configs/r15-output-ns.json)，`lda=K, ldb=ldd=N`。实际 grid、坐标、SMID 和 tile_count 仍由新调用确认。c6 的旧平均重叠估计为仿射94.61、入口锚点120.51，已足以检验换算歧义；h08 同时改变几何和 K，本轮不需要加它。g3/c2 仍改变总字节、足迹和前序供给，c2/c6 仍改变输入量、执行时长与频率状态，三点都不是纯并发因果干预。

### 给公共框架维护者的两字段修改

复用每 CTA 16-word 头部的空闲槽位，不改变784-word布局或每tile六字记录；只由 thread256、tile0 写入：

| 事件 | 原记录及位置 | 本批新增记录 |
|---|---|---|
| FIRST_MMA | 原 clock64，在首次 MMA 前；位置不动 | 无 |
| MAIN_END | 原 clock64，在 mma_tail 后；包含等待的主循环窗口终点 | 无 |
| EPI_PERMIT | 原 role2 clock64，在 cooperative `store()` 前；不是首条 TMA issue | header[13]：`issuer_store_enter_ns` |
| EPI_DONE | 原 role2 clock64，在 cooperative `store()` 返回后 | header[14]：`issuer_store_return_ns` |
| 最终 final | 原 header[7]/[8]：role2 的 clock64/globaltimer，在 post-loop `store_tail()` 返回后 | 原样保留 |

具体公共建议：`r18_trace.hpp::v06_stamp()` 在 `-DR15_OUTPUT_NS` 下，只对 `threadIdx.x==256 && tile==0` 的上述两个输出事件采样；globaltimer 紧随原 clock64 读取，两个时钟都先读取，再保存 trace。`r18.cu` 对这个模式写 `trace_version="r15-first-output-ns"`，header[15] 保留。既有 cooperative overlay 已有正确的两处 `v06_stamp()`，无需再插新调用，也不增加等待。源中的 FIRST_MMA/MAIN_END/EPI_PERMIT/EPI_DONE 和 final 原语义均不变。

新二进制命名 `cfg_c_global`，编译在原 cfg_c stamped 参数上增加 `-DR15_OUTPUT_NS`；plain/stamped/ends 使用原参数。同卡、同批交错执行四个变体，报告 global 相对三者的时间扰动与各自离散程度。公共 runner 若复用 `run_v08.run_one()`，global 必须走完整坐标 trace 的 `analyze_r18.replay()`；当前 `v08_model.observe()` 会把未知 variant 当 ends，不能直接将 global 传给它。无需为此改模型公共接口。构建、SASS/寄存器和新打点扰动仍待该版本的实际 GPU 检查，旧版本的编译结果不覆盖新增 profile。

`store_tail()` 使用的 `tma_store_wait<0>()` 在该 CUTLASS 版本实际发出 `cp.async.bulk.wait_group.read 0`，只保证源 SMEM 已被读完、可复用；`store()` 返回和最终 final 都不叫全局目标写完成。这次不增加全写排空等待，也不把已有端点重新命名为 write_complete。

### 专用分析入口

[analyze_r15_output_ns.py](../../../../../../microbench/gh200_resource_campaign/access_rules/analyze_r15_output_ns.py) 读取每个 global 原始调用自己的 setup、header[13]/[14] 和 role2 clock64，要求每 CTA 两个 consumer 都恰好一个完整有效 tile；不会把旧 trace 的零槽位或多 tile trace 当作直接测量。`static_setup.json` 来自 plain，其版本字段不能用于识别 global profile。

分析输出 `output-ns.json`：逐 CTA 的直接输出 ns/cycle、输出启动分散、到 final 的剩余窗口、直接平均/峰值重叠，以及同次调用仿射/入口/退出锚点估计用于比较。所有绝对 globaltimer 先用整数减去本次最早入口，再做积分，不拼接其他调用或进程中位时间线。三个条件分别保留进程中位数、进程范围及 plain/stamped/ends/global 的时间与 CV；不在三点上重新拟合或宣布带宽规律。

```bash
python3 microbench/gh200_resource_campaign/access_rules/test_r15_output_ns.py
python3 microbench/gh200_resource_campaign/access_rules/analyze_r15_output_ns.py \
  --input <本地接收的新R15运行目录> --output <该run下新的reanalysis目录>
```

CPU 测试使用临时合成记录，检查直接重叠与错误的周期换算分离、并列端点、整数时间精度、旧profile/多tile拒绝，以及从 raw setup 识别 global；这些测试不是 GPU 数据。后续准备与运行目录使用节点本地 `/tmp/gh200-r15-output-ns-<run-id>`，完整批次回传本地 results 后分析；不向已超配额的共享存储追加。本轮只交回上述配置、分析器和公共修改需求，不提交作业。
