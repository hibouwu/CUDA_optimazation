# R04：跨路径联合服务

[总计划](README.md)。状态：WGMMA+FFMA 两种配对完成；LDS+FFMA/CVT 修订版的串行与交错代码组织不同，不能作联合服务结论。代码：`run_r04.py`、`probes/r04.cu`。

## 问题

两类操作单独执行都很快，同时执行时能否各自保持速率？对固定工作量 A、B，把实测时间与三种预期比较：完全重叠 \(\max(T_A,T_B)\)、完全串行 \(T_A+T_B\)、依赖图加共享服务的推算。

## 默认矩阵（32 点）

| 配对 | 内容 | 参与范围 | 对应场景 |
|---|---|---|---|
| WGMMA + FFMA，同一 warpgroup | FP16 SS WGMMA（R00-B 选定形状，默认 `m64n256k16`）；FFMA 使用另一套寄存器 | 1 CTA、128 线程 | 同一 warpgroup 在 MMA 间隙做其他计算 |
| WGMMA + FFMA，跨 warpgroup | warpgroup 0 只发 WGMMA，warpgroup 1 只做 FFMA | 1 CTA、256 线程 | warp-specialized 中 mainloop 与其他计算并行 |
| LDS + FFMA | 连续 16 B SMEM 读；独立 FP32 累加器 | 1 CTA、128 线程 | SIMT mainloop |
| LDS + CVT | 连续 16 B 读；FP32→FP16 转换，非均匀输入 | 同上 | epilogue 转换 |

每对 8 点：A-only、B-only；A:B 发出次数比 1:1、1:4、4:1 的串行与交错各一点。跨 warpgroup 的"串行"指两组按阶段轮流（named barrier 分隔），"交错"指两组同时执行。

- 比例按 warp 指令或 warpgroup 协同发出次数计；一条 `m64n256k16` 为 524288 FLOP。
- 每个对照单元固定 A+B 总次数（如 \(n_A+n_B=160\)：80/80、32/128、128/32），所有点用同一单元重复长度。A-only/B-only 若随次数不线性，补对应次数的单独基线。

## 时间与正确性

- 串行点在 A 完成其等待与消费后再执行 B；交错点工作量相同，各自合法完成与排空。
- 默认 A、B 数据互不依赖。WGMMA 使用合法 fence/commit/wait；FFMA 不读写 WGMMA 正在使用的累加器或输入寄存器。
- 窗口在全部必要结果完成后结束；原始记录分别给出 A 工作、B 工作、控制与完成窗口。两类工作率（如 FLOP/cycle 与 B/cycle）按同一窗口分别报告。
- 所有计算结果、访存输出与累加器分别校验；SASS 核对两路径的动态指令数、交错顺序、寄存器分配与 spill，按实际顺序记录。

手算模板：32 条 warp FFMA 与 128 条 warp 级 16 B/lane LDS（32 活跃 lane），分别为 2048 FLOP 与 65536 B，两个工作率使用同一窗口。

## 输出

`joint_service(A, B, ratio, order, resources)` 的查表，用于 L0/L1 时间递推。若各配对都无额外差异，记录无变化的范围。

## 条件扩展

默认点出现可重复差异时，再加：1:1 交错点的两种背景相对位置（16/64 条填充指令）、LDG + FFMA（SIMT 方案用普通 global 读时）、真实依赖的消费者、不同 warp 分工、不同缓存准备。每次写明两个竞争解释及各自的预期结果。

**真实 FP8 分段累加**（FP8 方案需要提升时启用）：使用 FP8 形状，测合法等待 → 读取部分和 → 加入长期 FP32 累加状态 → 清零复用的完整序列，记录分段 K、转换与缩放、实际指令、寄存器占用与 spill。两套累加状态（WGMMA 累加器与长期 FP32 状态）同时占用寄存器，先确认无 spill。

## 实现要点

- 每单元 A+B=160：A 为一条 warpgroup WGMMA 或每 warp 一条 LDS.128，B 为每 warp 一条 FFMA 或 CVT（四个独立状态轮换）。交错序列 1:1 为 AB、1:4 为 ABBBB、4:1 为 AAAAB。
- WGMMA 每 8 条 commit/`wait_group 1`，单元末尾 wait0；跨 warpgroup 用 warp 一致的角色分派，SASS 无 C7520/C7517。
- 单 CTA clock64，窗口含循环、消费与 CTA 同步；串行点在 A 排空后执行 B。CPU 重算全部输出。

## 结果（2026-10-06，cycle，3 进程中位数）

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

## 数据

- 跨 warpgroup 异步版：[报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-async-v01-r04/r04-offline-review/report.md)（作业 735203）。
- 同 warpgroup 与首版：[报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-A-job735062/r04-final-review/combined_report.md)；LDS/CVT 修订版：`20261006-A-fix-job735138/r04-volatile-review/`。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r04.py --cutlass-root <R00 source>/cutlass --output <新目录>/r04
python3 <新目录>/r04/source/analyze_r04.py --input <新目录>/r04
```
