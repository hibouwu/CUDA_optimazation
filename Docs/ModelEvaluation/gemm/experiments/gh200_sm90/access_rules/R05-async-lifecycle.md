# R05：异步完成与缓冲复用

[总计划](README.md)。状态：40 点完成（2026-10-06）。代码：`run_r05.py`、`probes/r05.cu`。

## 问题

发出、等待、消费、复用这几个事件怎样组合成完成时间？目标 GEMM 的在途输入量下，一个 SM 能从 L2 拿到多少输入？EXP-12/14/15/16 与 S19/S20 已有合法协议，本组复用其布局和完成规则，只补事件位置、目标在途量和输出行距三类问题。

需要区分的事件：发出后可以继续做独立工作；对应等待之后数据可用；消费者结束后输入槽可复用；输出源读完后源槽可复用；输出写入全部完成。

## 默认矩阵（40 点）

| 子集 | 协议与坐标 | 数量 |
|---|---|---:|
| R05-A，等待位置 | cp.async 输入、TMA 输入、TMA 输出、FP16 SS WGMMA（R00-B 选定形状）共 4 种；发出后做 0/32/128/512 条独立整数操作再等待 | 16 |
| R05-B，流水与消费 | cp.async 与 TMA 输入；stage 1/2/4；轻/重消费者 | 12 |
| R05-C，输出源复用 | TMA 输出；1/4 个请求；源读完等待（`.read`）后复用 / 完整等待后复用 | 4 |
| R05-D，目标在途量供给 | 1 CTA/SM；每 stage 一个 A tile + 一个 B tile 共 48 KiB；stage 2/4 × 共享小源 / 独立大源 | 4 |
| R05-E，输出行距 | TMA 2D 输出，box (64,128) 16-bit 共 16 KiB；GMEM 行距 128/144/160/256 B；整卡 | 4 |

R05-A/B/C 为 1 CTA、128 线程；cp.async 每线程 16 B（2 KiB/槽），TMA 每请求 16 KiB。R05-A 另有 16 个不发异步请求的填充/消费控制。

**R05-B**：每个完整序列 8 次请求，固定预留 4 个物理槽，stage 只改变活动槽数。轻/重消费者读取同样的输入，每线程分别做 16/128 个依赖 FFMA，结果依赖输入并校验。各模式的寄存器与 SMEM 分配保持一致。

## R05-D：目标在途量下的每 SM 供给

问题：V01 目标组合（128×256×64 tile、4 stage）每个 SM 只驻留 1 个 CTA，在途输入最多 192 KiB。满速 WGMMA 每 Ktile 约 1024 cycle、需输入 48 KiB，即约 48 B/cycle/SM；这样的单 CTA 能否从 L2 拿到这个速率？

- **配置**：grid = SM 数；动态 SMEM 预留不小于目标的 192 KiB 加 barrier，使每 SM 只能驻留 1 个 CTA（用 API 核对，并记录每个 CTA 的 SM ID）。每 stage 用 2D TMA 读一个 A tile（box (64,128) FP16，16 KiB）和一个 B tile（box (64,256) FP16，32 KiB），SW128，与目标组合相同。
- **流水**：producer warp 发 TMA；一个 consumer warpgroup 等对应 mbarrier 完成后立即释放槽，不做计算。这样测的是无计算消费者条件的供给观测，不表示所有组织策略的严格供给上限。
- **源**：共享小源——所有 CTA 轮流读取同一个 L2/4 大小的 A/B 区域，模拟多个 CTA 复用同一 K 面板；独立大源——各 CTA 读互不重叠的区域，合计 4×L2。记录实际访问的地址范围与复访间隔。
- **指标**：每个 CTA 用自身 `clock64` 窗口算 B/cycle（即该 SM 的供给率），报告 132 个值的中位数与范围；整卡另报 GB/s。与 48 B/cycle 的需求直接比较。

## R05-E：输出行距

区分 EXP-15 写回下降的原因：128 B 连续；144 B 只按 16 B 对齐（EXP-15 的条件）；160 B 按 32 B 对齐；256 B 是 128 B 倍数的 padding。基地址 128 B 对齐，非均匀数据，只计有效 16 KiB；完成边界同 EXP-15（commit 后 `wait_group 0`，再 CTA 会合）。预期判读：256 B 正常而 144/160 B 下降，指向行起点对齐；160 B 也正常，则 32 B 对齐已足够。

## 合法事件

| 路径 | 保留的事件 |
|---|---|
| cp.async 输入 | commit、对应 group 等待、跨线程同步、消费完成、槽释放 |
| TMA 输入 | expect-tx/arrive、mbarrier 完成与 acquire、CTA/proxy 发布、消费、槽释放 |
| WGMMA | 输入与寄存器访问顺序、发出与 commit、覆盖该操作的 wait、累加器消费、存储复用 |
| TMA 输出 | 源发布、bulk 发出与 commit、源读完（`.read`）、写入完成、最终排空 |

WGMMA 的累加器只在覆盖该操作的 wait 之后读取。R05-C 在源读完后把 SMEM 改写为下一代数据并按 proxy/线程同步规则发布，输出应保持旧一代的数据；两种模式最后都执行完整 wait 再读回输出。

## 计量与正确性

- 主窗口按子集在输入就绪、累加器就绪或输出完成时结束；等待所在位置与等待返回时间分别记录。填充按实际指令数命名。
- R05-B 每序列运输量 `8×Q`，不乘 stage；消费 FFMA 另计 FLOP。R05-D 运输量为 `CTA数×Ktile数×48 KiB`。
- 每个槽、每一代用不同位模式，检查所有输出与最终状态。完整 trace 只在短检查中做，正式窗口只保留约定的消费者。

手算模板：TMA 输入 8×16 KiB = 131072 B，stage 4 时仍为 131072 B（4×16 KiB 物理预留另记）；轻消费者每线程每请求 16 FFMA，共 `2×128×8×16=32768` FLOP。

## 输出

等待位置—完成时间曲线；给定消费者下的 stage 曲线；源复用与完整输出的事件关系；目标在途量下的每 SM 供给率；TMA 写回随行距的查表。已有 S16/S19/S20 条件匹配时直接引用。完成后用 V01 预测一个未参与拟合的 K 序列，终点包含最终输出完成。

## 实现要点

- B：固定 4 个物理槽、每序列 8 次请求；先填 stage 个槽，再逐个等待、消费、补发。消费者从输入构造 dyadic 数做 16/128 条依赖 FFMA。
- C：每序列 1/4 个请求为一组，分别在 `.read` 或完整等待后把源改写成旧数据的按位反值；输出必须保持旧代数据，最后都做完整等待。
- D：160 线程（warp 0 发 TMA，其余 128 线程为消费者）；SMEM 预留 192 KiB 使 API 驻留上限为 1；grid = SM 数，10 个进程都覆盖了全部 132 个 SM。共享源实际访问 15 MiB（每个 Ktile 面板被复访 320 次），独立源 241 MiB（复访 39 次）。
- A：WGMMA 的发出、commit、填充（依赖 LCG IMAD）、等待和时间戳放在同一段内联 PTX，SASS 逐条核对填充数量与位置。
- 共同长度 2048；每点 3 或 10 个进程；全部输出 CPU 精确重算通过。

## 结果（2026-10-06，job735059，单 CTA 用 clock64，D/E 为整卡 globaltimer）

**A：等待位置**（cycle/步，发出后插入 0/32/128/512 条依赖 IMAD 再等待）

| 协议 | 0 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|
| WGMMA FP16 SS `m64n256k16` | 271 | 272 | 631 | 2182 |
| TMA 输入 16 KiB | 1142 | 1292 | 1724 | 3452 |
| TMA 输出 16 KiB | 1777 | 1777 | 2085 | 3789 |
| cp.async 输入 | 799 | 1072 | 1498 | 3202 |

由 128→512 条的增量，依赖 IMAD 约 4.4 cycle/条，32 条约 140 cycle。WGMMA 与 TMA 输出在发出后插入 32 条填充，总时间不变，说明这段独立工作被异步操作覆盖；TMA 输入与 cp.async 插入 32 条后总时间增加 150/273 cycle，不少于填充本身，没有被覆盖。

**B：stage 与消费者**（cycle/请求，每序列 8 次请求）

| 协议 | 消费者 | stage 1 | stage 2 | stage 4 |
|---|---|---:|---:|---:|
| TMA 16 KiB | 16 FFMA | 1310 | 1179 | 1169 |
| TMA 16 KiB | 128 FFMA | 1823 | 1690 | 1680 |
| cp.async 2 KiB | 16 FFMA | 966 | 982 | 975 |
| cp.async 2 KiB | 128 FFMA | 1463 | 1479 | 1472 |

- 重消费者比轻消费者每请求多约 500 cycle，即多出的 112 条依赖 FFMA 约 4.5 cycle/条，与 FFMA 依赖延迟一致。
- stage 从 1 增到 4，TMA 只快约 11%，cp.async 不变。按 EXP-16 的往返时间（16 KiB 约 1100 cycle），4 个槽在途时本应明显重叠；本协议中补发可能被等待或同步串行化，需要核对 SASS 与事件顺序后再作为流水规则使用。

**C：源复用**：`.read` 后复用源与完整等待后复用的完整时间相同（1 请求 1776 vs 1785 cycle/步，4 请求 3389 vs 3398）；本条件下提前释放源没有收益。

**D：目标在途量下的每 SM 供给**（B/cycle/CTA，132 个 CTA 的中位数；满速 WGMMA 需约 48）

| 源 | stage 2 | stage 4 |
|---|---:|---:|
| 共享小源 | 55.45 | 55.48 |
| 独立大源 | 15.26 | 15.26 |

有 L2 复用时单个目标 CTA 的供给高于满速需求，没有复用时只有约 1/3；stage 2 与 4 相同。

**E：TMA 2D 写回行距**（整卡完整窗口，ns）：128 B 950714、144 B 1314624（慢 38%）、160 B 966899、256 B 898909。行起点按 32 B 对齐即可避免下降；EXP-15 的写回下降来自 144 B 行距只按 16 B 对齐。

## 数据与复现

- [B/C/D/E 报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-formal-v2/report.md)、[A 修正版报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-a-formal-v5/report.md)；`cases.csv`、`rules.json`、图在同目录。
- 首版 A（96 进程）因编译器合并整数链、把等待提前，计量无效，保留在 `r05-formal-v2/` 并标为 `invalid_sass_schedule`。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r05.py --smoke --cutlass-root <R00 source>/cutlass --output <新目录>/r05-short
python3 microbench/gh200_resource_campaign/access_rules/run_r05.py --cutlass-root <R00 source>/cutlass --output <新目录>/r05
python3 microbench/gh200_resource_campaign/access_rules/analyze_r05.py --input <结果目录>
```
