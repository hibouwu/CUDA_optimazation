# R11：混合发射与联合服务

当前以v6为准：消费者带不同阶段初值和非线性摘要，避免不变结果被跨阶段CSE复用；旧v4/v5的共同成本解释退出当前规则。

本轮明确区分了三种组合：WGMMA与独立IMAD在测试帧中能接近完全重叠；FFMA与IMAD.WIDE交错反而慢于串行；LDS与转换动作只有部分重叠。不是所有独立操作都适用max(A,B)。

针对[R04](#r04)的不可比基线，本组每路径固定128动作/单元、256单元，不按160次单独基线缩放。两个scalar配对的六种组织共用一份机器码；WGMMA配对使用角色隔离的编译组织，以保留批量异步提交，各点寄存器差异明确记录。

## 对照

| 配对 | A 的一个动作 | B 的一个动作 | 参与者 |
|---|---|---|---|
| FFMA + IMAD.WIDE | FP32 FFMA，八条累加链 | 32-bit乘数、64-bit累加结果的MAD，八条链 | 每路径32 lane；同warp或两个warp |
| LDS.128 + CVT | 16 B/lane的volatile SMEM读取，8个保留向量 | 更新FP32输入、转换成FP16、累加转换位模式校验 | 每路径32 lane；同warp或两个warp |
| WGMMA + IMAD | m64n128k16、FP16输入、FP32累加、SS | 独立32-bit MAD，八条链 | 每路径128 thread；同warpgroup或两个warpgroup |

每配对六种组织：A-only、B-only、A后B、B后A、同组交错、跨组并发，共18条件。CVT动作包含更新与校验，不能作为裸转换吞吐。整数回归明确按unsigned模32算术或64-bit累加定义，不把OP与FLOP混用。

## 边界和控制

所有初值在计时前准备并实际消费到共享就绪数据，首时钟依赖同步后读取的数据。每单元两次消费/阶段同步；先完成的角色也必须消费另一角色的就绪数据，避免只依赖barrier后的寄存器时钟。

WGMMA每8条提交、wait1，阶段末wait0并消费全部保留结果；输入缓冲在整个计时期间不改写。所有组织使用相同消费者和资源预留。额外测空单元、B-only在另一个角色上的基线；每轮保留完整结果而非只做错误计数。

交错与并发需要报告：实际T_AB、T_A+T_B、max(T_A,T_B)，以及扣除共同空单元后的T_A+T_B−T_0。不能要求所有组合必然处在未经扣除公共成本的max/sum之间；控制与完成服务本身可被不同组织隐藏。

短检查采用1/3个单元，非均匀输入、奇数次正负交替WGMMA，使全部累加器可核对；正式256单元避免累加数值不断增长。逐指令数、循环、异步提交/排空和资源经SASS核对，无spill才采样。WGMMA为每批8条、部分等待1、阶段末等待0；SASS静态8个HGMMA和各一个wait1/wait0，没有逐条完整等待。

## 计量例子

WGMMA A-only：128条/单元×256单元=32768条；每条2×64×128×16=262144 FLOP，总8589934592 FLOP。B-only为128 lane×32768次32-bit MAD，每次乘加2 OP，总8388608 OP。两个量共用同一完整窗口，不能相加为一个FLOP数。

## 实测：消费者匹配后的条件服务

job736989、romeo-a048、GPU-009a8880，v6含18条件、54正式进程、36个奇数短检查及18个空单元/匹配B角色控制，共108进程。全部1783296个见证字独立核对；正式最大CV0.026%，无spill。单独基线、串行、交错与并发保持每路128动作/单元、256单元。

| 配对 | A-only | B-only | A后B | B后A | 同组交错 | 跨组并发 |
|---|---:|---:|---:|---:|---:|---:|
| FFMA+IMAD.WIDE | 232375 | 390073 | 431031 | 426424 | 478321 | 478323 |
| LDS.128+CVT动作 | 822547 | 935337 | 1216219 | 1216423 | 1187172 | 1153894 |
| WGMMA+IMAD | 2337092 | 344405 | 2454497 | 2460036 | 2338677 | 2333566 |

单位是同SM的clock64完整窗口cycle。Scalar使用同一机器码及相同资源（58/226 reg/thread）；WGMMA采用角色隔离的编译组织（152–160 reg/thread），逐点记录差异，不将它们视作只改指令次序的物理端口实验。

WGMMA交错与跨组并发仍接近A-only，差约+0.07%/−0.15%；这个量级结合不同编译帧解释，不称“整数使TC变快”。FFMA交错比A后B慢11%；LDS/CVT跨warp比串行快约5%，仍不接近完全覆盖。服务的组合方式依赖配对与消费者，不能统一取max。

每条m64n128k16为262144 FLOP，32768条为8589934592 FLOP；交错完整窗口2338677，则有效矩阵生产率约3672.99 FLOP/cycle/CTA，B另计8388608整数OP。归约、同步与排空在分母中，不能当裸TC服务。

### 为什么修订消费者

R12揭示相同结果的两次归约可能被复用。R11存在同类情况，旧版的“两个同型消费者”是否真正执行了相同数量的工作，不能仅凭源码调用数确定。v6的浮点归约使用不同阶段初值，整数结果用阶段种子驱动非线性摘要；GPU结果与独立CPU摘要核对。SASS中TC A-only/交错/跨组的主单元均保留两条64 FADD归约，额外前缀归约在计时前。旧版结果保留，不混进新参数。

仍分别报告原始T_A+T_B、max及扣共同空单元的和；空单元只是该实现的控制对照，不是任意操作图的可加常数。精确数值见结构化规则，跨组使用匹配B角色的单独基线。

![匹配消费者后的服务窗口](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R11-job736989-v6/analysis/plots/r11-joint-service.png)

图为进程中位数，不绘误差条；全部CV和原始样本保留。WGMMA逐8条提交、部分等待1和阶段末等待0，已排除逐条完整等待；每个TC组织的静态服务体8个HGMMA，按16批/单元动态计量。

## 可用于模型的范围

结构化接口为`joint_service(pair, organization, work, consumers, barriers, SASS)`条件表。在本TC帧中，独立IMAD可几乎完全覆盖；scalar两组应引用联合实测。不能由此推出物理执行管线或返回端口完全独立，也不能把动态组织差值除以动作数，变成每opcode固定惩罚。

CVT动作包含输入更新、转换和数值摘要，它的工作量和消费者与各基线一致，但不是裸CVT率。单CTA与实际GEMM的整卡竞争仍分开。资源数量、循环长度、源模式或同步变化后，需要验证迁移，而非默认沿用本倍率。

## 数据与脚本

代码：`probes/r11.cu`、`run_r11.py`、`analyze_r11.py`；矩阵：`configs/r11.json`。本地用`--prepare-only`冻结完整CUTLASS3.9.2依赖，节点有效Slurm分配内一次调用完成构建、检查、采样及重算：

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r11.py \
  --cutlass-root <CUTLASS3.9.2> --output <新目录>
```

`--plan-only`列18条件，`--smoke-only`只完成编译/SASS/数值检查，`--resume`在同一分配和UUID恢复。[正式归档](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R11-job736989-v6/)含冻结源码、CUTLASS、二进制、SASS和全部样本；[条件规则](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R11-job736989-v6/analysis/rules.json)、[CSV](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R11-job736989-v6/analysis/cases.csv)、[独立重算](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R11-job736989-v6/analysis/checks.json)。CPU重放：`python3 <归档>/source/analyze_r11.py --input <归档> --output <新分析目录>`。

<a id="r04"></a>

## 前序：R04 跨路径联合服务（2026-10-06，2026-10-08 迁入）

R04 是本页的前序实验，2026-10-08 并入本页；原文保留，只把小节降一级。WGMMA+FFMA 两种配对仍可作条件参照；LDS+FFMA/CVT 的串行与交错代码组织不同，不能作联合服务结论，本页 v6 用匹配消费者重做。R04 列出的条件扩展未执行。

代码：`run_r04.py`、`probes/r04.cu`。

### 问题

两类操作单独执行都很快，同时执行时能否各自保持速率？对固定工作量 A、B，把实测时间与三种预期比较：完全重叠 \(\max(T_A,T_B)\)、完全串行 \(T_A+T_B\)、依赖图加共享服务的推算。

### 默认矩阵（32 点）

| 配对 | 内容 | 参与范围 | 对应场景 |
|---|---|---|---|
| WGMMA + FFMA，同一 warpgroup | FP16 SS WGMMA（R00-B 选定形状，默认 `m64n256k16`）；FFMA 使用另一套寄存器 | 1 CTA、128 线程 | 同一 warpgroup 在 MMA 间隙做其他计算 |
| WGMMA + FFMA，跨 warpgroup | warpgroup 0 只发 WGMMA，warpgroup 1 只做 FFMA | 1 CTA、256 线程 | warp-specialized 中 mainloop 与其他计算并行 |
| LDS + FFMA | 连续 16 B SMEM 读；独立 FP32 累加器 | 1 CTA、128 线程 | SIMT mainloop |
| LDS + CVT | 连续 16 B 读；FP32→FP16 转换，非均匀输入 | 同上 | epilogue 转换 |

每对 8 点：A-only、B-only；A:B 发出次数比 1:1、1:4、4:1 的串行与交错各一点。跨 warpgroup 的"串行"指两组按阶段轮流（named barrier 分隔），"交错"指两组同时执行。

- 比例按 warp 指令或 warpgroup 协同发出次数计；一条 `m64n256k16` 为 524288 FLOP。
- 每个对照单元固定 A+B 总次数（如 \(n_A+n_B=160\)：80/80、32/128、128/32），所有点用同一单元重复长度。A-only/B-only 若随次数不线性，补对应次数的单独基线。

### 时间与正确性

- 串行点在 A 完成其等待与消费后再执行 B；交错点工作量相同，各自合法完成与排空。
- 默认 A、B 数据互不依赖。WGMMA 使用合法 fence/commit/wait；FFMA 不读写 WGMMA 正在使用的累加器或输入寄存器。
- 窗口在全部必要结果完成后结束；原始记录分别给出 A 工作、B 工作、控制与完成窗口。两类工作率（如 FLOP/cycle 与 B/cycle）按同一窗口分别报告。
- 所有计算结果、访存输出与累加器分别校验；SASS 核对两路径的动态指令数、交错顺序、寄存器分配与 spill，按实际顺序记录。

手算模板：32 条 warp FFMA 与 128 条 warp 级 16 B/lane LDS（32 活跃 lane），分别为 2048 FLOP 与 65536 B，两个工作率使用同一窗口。

### 输出

`joint_service(A, B, ratio, order, resources)` 的查表，用于 L0/L1 时间递推。若各配对都无额外差异，记录无变化的范围。

### 条件扩展

默认点出现可重复差异时，再加：1:1 交错点的两种背景相对位置（16/64 条填充指令）、LDG + FFMA（SIMT 方案用普通 global 读时）、真实依赖的消费者、不同 warp 分工、不同缓存准备。每次写明两个竞争解释及各自的预期结果。

**真实 FP8 分段累加**（FP8 方案需要提升时启用）：使用 FP8 形状，测合法等待 → 读取部分和 → 加入长期 FP32 累加状态 → 清零复用的完整序列，记录分段 K、转换与缩放、实际指令、寄存器占用与 spill。两套累加状态（WGMMA 累加器与长期 FP32 状态）同时占用寄存器，先确认无 spill。

### 实现要点

- 每单元 A+B=160：A 为一条 warpgroup WGMMA 或每 warp 一条 LDS.128，B 为每 warp 一条 FFMA 或 CVT（四个独立状态轮换）。交错序列 1:1 为 AB、1:4 为 ABBBB、4:1 为 AAAAB。
- WGMMA 每 8 条 commit/`wait_group 1`，单元末尾 wait0；跨 warpgroup 用 warp 一致的角色分派，SASS 无 C7520/C7517。
- 单 CTA clock64，窗口含循环、消费与 CTA 同步；串行点在 A 排空后执行 B。CPU 重算全部输出。

### 结果（2026-10-06，cycle，3 进程中位数）

**WGMMA + FFMA**

| 配对 | A:B | 串行 | 交错 | 缩放 max | 交错/max |
|---|---:|---:|---:|---:|---:|
| 跨 warpgroup（异步版） | 80:80 | 1921179 | 1814864 | 1796965 | +1.0% |
| 跨 warpgroup（异步版） | 32:128 | 912675 | 745808 | 718786 | +3.8% |
| 跨 warpgroup（异步版） | 128:32 | 2929683 | 2883918 | 2875143 | +0.3% |
| 同 warpgroup | 80:80 | 1934784 | 1831485 | — | — |
| 同 warpgroup | 32:128 | 910464 | 749072 | — | — |
| 同 warpgroup | 128:32 | 2959104 | 2916523 | — | — |

- WGMMA 与另一 warpgroup 的 FFMA 几乎完全重叠，交错时间只比较大一方多 0.3–3.8%。同一 warpgroup 内交错的时间与跨 warpgroup 相近。
- 缩放 max/sum 由 160 次 A-only/B-only 按次数缩放，线性未单独验证。

**LDS + FFMA / LDS + CVT（修订版）**：交错时间低于按单独基线缩放的 max（如 LDS+CVT 80:80 交错 1113895，缩放 max 1981190），这在纯资源竞争模型下不可能。原因是串行/单独版本每次读都做 volatile 消费与同步（24 寄存器），交错版本是另一种展开组织（70–72 寄存器），两者测的是不同代码。这组数据只说明两种实现各自的时间，不作联合服务规则；需要同一代码组织下只改比例的对照才能回答原问题。首版 LDS/CVT 因编译器合并 LDS、把 CVT 提到循环外已作废。

### 数据

- 跨 warpgroup 异步版：[报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-async-v01-r04/r04-offline-review/report.md)（作业 735203）。
- 同 warpgroup 与首版：[报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-A-job735062/r04-final-review/combined_report.md)；LDS/CVT 修订版：`20261006-A-fix-job735138/r04-volatile-review/`。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r04.py --cutlass-root <R00 source>/cutlass --output <新目录>/r04
python3 <新目录>/r04/source/analyze_r04.py --input <新目录>/r04
```
