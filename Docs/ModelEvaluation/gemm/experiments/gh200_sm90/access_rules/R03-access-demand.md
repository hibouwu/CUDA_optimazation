# R03：片上访问并发

<a id="m11-current"></a>

## m11 当前结论

- **开发条件关系：** job740961 的兼容窗口复现 4 B stride 2–32 的约128/stride B/cycle；新窗口的特定独立访问组织下，stride 响应与候选 bank 冲突计数相容。这是开发数据上的时间关系，未测内部 wavefront，也没有独立留出资格。
- **已经证伪：** 原需求按128 B lane 子组累加，令16 B连续与宽广播拥有相同输入；相同内核/组织/轮数的实测窗口分别2097378、1135930 cycle，任何共同预测的最大相对误差至少29.7357%，因此原需求与时间描述不能覆盖全范围。
- **尚缺：** 宽广播下的时间下限不能由非广播训练唯一确定；依赖读三点还违反后续两形式共有的凸性约束。全 warp 需求接原包络虽有0.49%/9.62%的开发相对误差，仍未满足绝对周期要求，留布局最大84.65%。缺少宽广播和依赖执行的可迁移描述，不能冻结或接入主线。

原作业、第一轮反证及后续离线推导分列如下。两轮研究均已停止；旧66条件留出没有执行。新的48条件对照及独立留出已获用户批准在整理后推进，见 [后续测量合同](PLAN.md#measurement-next)；原准备包的未授权文字是历史状态，原包不回写。


[总计划](README.md)。状态：18 点完成（2026-10-06）。代码：`run_r03.py`、`probes/r03.cu`。

## 问题

参与 warp 增加时，SMEM 线程访问和 `ldmatrix`/`stmatrix` 在几个 warp 时达到上限、上限是多少？EXP-09 只测了单 CTA 256 线程的 SMEM；EXP-10 的 `ldmatrix` 只有 1 个 warp（35 B/cycle）。SIMT 与 `mma.sync` 方案的片上供给、Tensor Core 方案 epilogue 的 `stmatrix` 都需要多 warp 下的数值。WGMMA 自身的 SMEM 取数已包含在 [R00-B](R00-anchor-target.md) 中。

## 默认矩阵（18 点）

1 CTA；连续、合法对齐的地址；各 warp 访问互不重叠的区域。

| 子集 | 配置 | 数量 |
|---|---|---:|
| SMEM 线程访问 | load/store × 每 lane 4/16 B × 1/4/8 warp | 12 |
| 矩阵搬运 | `ldmatrix.x4`、`stmatrix.x4`（`m8n8.b16`）× 1/4/8 warp | 6 |

`.trans` 在 EXP-10 中与普通形式相同，不再单列。

## 时间与工作量

窗口包含访存、消费（load 做 checksum，`ldmatrix` 结果参与依赖归约）、地址更新、循环和 CTA 同步，报告 B/cycle/CTA 与完整时间。

- SMEM：`B_requested=宽度×32×warp数×每轮访问数×轮数`。
- `ldmatrix.x4`/`stmatrix.x4`：每条 warp 指令 512 B，`B_requested=512×warp数×每轮指令数×轮数`。

手算模板：8 warp × 8192 轮 × 每轮 8 条 `ldmatrix.x4`，请求 \(512\times8\times8\times8192=268435456\) B；若用时 \(2.1\times10^6\) cycle，则为 127.8 B/cycle。

## 正确性

非均匀输入覆盖 lane、warp、矩阵片段与轮次；load/`ldmatrix` 校验 checksum 或归约结果，store/`stmatrix` 计时后校验完整区域。SASS 确认 LDS/STS/LDSM/STSM 的宽度、数量与循环位置。

## 输出

`service(op, width, warps)` 的查表与饱和点，用于 L1 的片上供给与 epilogue。

## 条件扩展：普通 global 地址需求

默认只提供离线工具：按每条 warp 访存的 lane 地址算出覆盖的 32B sector 集合（例如 32 lane 各读 4 B，offset 0 覆盖 4 个 sector，offset 4 覆盖 5 个）。

启用条件：方案在 mainloop 或 epilogue 使用普通 global load/store（如 FP32 SIMT、读 C 的 epilogue），且 V01 中相应路径失配。启用时用整卡 grid（单 CTA 的 global 访问受延迟限制），一次只改一个轴：

| 子集 | 配置 | 数量 |
|---|---|---:|
| 基线 | load/store × 4/16 B × 小/大工作集（L2/4、4×L2），连续、offset 0 | 8 |
| 4 B 对齐 | offset 4/16/32 B | 12 |
| 16 B 对齐 | offset 16/32 B | 8 |
| 4 B 跨步 | lane stride 2/8 | 8 |
| 活跃 lane | 每 warp 连续活跃 16/8 lane | 8 |
| 广播 | 4 B load，warp 内同址 | 2 |

工作集按实际访问的地址集合计算，另列分配大小与复访间隔。global store 在停止计时前执行 device 范围 fence。出现容量或页尺度拐点时，再扩展工作集 {L2/2, L2, 2L2, 8L2}。

## 实现要点

- `probes/r03.cu` 6 个 kernel 覆盖 18 点。每个 warp 有 8 个互不重叠的槽，每轮 8 条访问，按 `(轮次+访问号)%8` 轮转；矩阵搬运每槽 512 B。
- load 与 `ldmatrix` 的返回值逐 lane 归约，store 与 `stmatrix` 写入随轮次变化的值；1/2/9 轮非均匀检查覆盖全部输出。
- 窗口从初始化后的时间戳与 CTA 会合开始，到 8 次/轮循环完成、checksum 写入 volatile sink 并 CTA 会合结束；store 区域的读回在窗口外。
- SASS 核对：每轮 8 条 LDS/LDS.128、STS/STS.128、LDSM.16.M88.4 或 STSM.16.M88.4。
- job735059，共同 44693 轮，每点 3 个进程，最大 CV 0.25%，全部输出 CPU 精确重算通过。

## 结果（2026-10-06，B/cycle/CTA）

| 访问 | 1 warp | 4 warp | 8 warp |
|---|---:|---:|---:|
| LDS 4 B | 10.1 | 40.2 | 62.1 |
| LDS.128 16 B | 29.9 | 101.1 | 124.4 |
| STS 4 B | 11.8 | 47.1 | 55.8 |
| STS.128 16 B | 25.9 | 92.6 | 125.3 |
| `ldmatrix.x4` | 31.8 | 115.1 | 126.9 |
| `stmatrix.x4` | 22.9 | 87.6 | 115.8 |

- 16 B 访问和 `ldmatrix.x4` 在 8 warp 时达到 124–127 B/cycle，接近每 SM 128 B/cycle；`stmatrix.x4` 为 116。4 warp 时约为上限的 70–90%。
- 4 B 访问在 8 warp 时只有 56–62 B/cycle，低于 EXP-09 中同为 8 warp 的 4 B 写（128）。两者访问组织不同（这里每次访问后都参与 checksum 并轮转槽位），4 B 访问的服务依赖访问组织，应按实际方案的访问方式引用。
- 1 warp 的值与 EXP-10 一致（`ldmatrix.x4` 约 32–35），属于单 warp 的延迟受限情形。

[报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r03-formal-v1/report.md)、`cases.csv`、`rules.json`、`plots/service.svg` 在同目录。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r03.py --output <新目录>/r03
python3 microbench/gh200_resource_campaign/access_rules/analyze_r03.py --input <结果目录>
python3 microbench/gh200_resource_campaign/access_rules/analyze_r03.py --sectors '[0,4,8,...]' --access-width 4   # 地址 → 32B sector
```


## m11 普通 LDS/STS 有界扩展（2026-10-11）

范围为单 CTA、每 warp 全部 32 lane、1/4/8 warp、每 lane 4/8/16 B 的普通 LDS/STS；矩阵、TMA/WGMMA 和多 CTA 不在范围。专属入口为 `prepare_r03_lds_sts.py`、`run_r03_lds_sts.py`、`analyze_r03_lds_sts.py`，默认 18 点入口保持原样。实际开始为 Paris 2026-10-11 23:07:45，原截止为次日 07:07:45、原提交预算为 2；2026-10-12 人类明确修订为累计 3 次提交、截止当天 Paris 18:00，先 CPU CUDA 12.9/SASS 预检，再第二提交开发校准，冻结后第三提交独立留出。原失败计入预算，历史快照保留。

开发预列 540 条条件与 12 条计时对照，按同一映射比较八条独立读流和实际地址依赖读链；store 的每个字在完整序列中只有一个线程所有者。留出预列 66 条，包括不只是 lane 重排的随机新映射，以及 CUTLASS 3.9.2 `TileIteratorTensorOp` 的普通 64 bit load/store 地址序列和真实 iterator 重放。该序列不代表完整 epilogue。

候选时间形式为 `fixed_g + N × max(floor_g, offset_g + price_g × batches)`，各参数只表示指定执行组织下的窗口周期。`batches` 按连续 128 B lane 组内每 bank 的不同 32 bit 字推导，是待检验假设，没有内部事务计数器证明；固定项、循环、消费与同步不被当作裸硬件服务剔除。

740708 在 CUDA 12.9.41 下完成构建，但自动 SASS 门槛失败，计时样本为 0，未定价、未冻结、未执行留出。实际分配为 romeo-a055、4 GPU、288 CPU、16 GiB；只准备一个计时流，失败是累计第一次提交。最终 SASS 中三个宽度的独立读与真实 iterator 读均位于重复循环外，功能 checksum 无法证明预期动态访存工作量。

修正版普通访问显式使用 PTX volatile，每轮加入一个 warp 同步边界，并保留真实 iterator 的 C++ load/store。初始 CUDA 13.0.88 device-only 检查只覆盖九个普通内核。2026-10-12 在 ROMEO 登录节点 CPU 上使用官方 SHA256 核对的 CUDA 12.9.41 和 GCC 11.5，完成全部三程序编译、链接与反汇编；九个普通内核及两个真实 CUTLASS iterator 内核均在循环内保留八条正确宽度 LDS/STS，目标内核无 spill，没有在登录节点执行 GPU 程序。第二提交 740961 于 romeo-a048 完成全部开发采样，ARM CUDA 12.9 构建也通过全部十一内核的 SASS 检查。

准备、失败构建、逐 kernel 动态指令审查与修正版源码保存在 `/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-m11-lds-sts-bounded/run-20261011T210745Z`；续做授权、CUDA 12.9 CPU 证据及新数据在其 `continuation-20261012` 子目录。原准备、旧 R03/EXP-09 和失败包保持只读。

### 开发反证（2026-10-12）

740961 正常结束，退出码 `0:0`、18 分 18 秒；540 条普通访问条件和 12 条计时控制各三个独立进程，共 1656 个新窗口样本，失败 0。实际 CUDA UUID 为 `GPU-009a8880-f309-a55d-d256-748fb9adad74`，全部开发样本 SMID 为 124。540 条条件的最大跨进程 CV 为 1.41651%；旧 EXP-09 兼容门槛均通过。

27 组参数可拟合，但候选的开发绝对相对误差中位数为 0.1621176%、最大为 83.83293%；绝对周期误差中位数为 949.908、最大为 952283.390。75 条条件超过 5%，24 条超过 10%，全部保留。以下反例不依赖拟合器：

| 同一 LDS.128 内核、8 warp、独立读、8192 轮 | 候选批次/轮 | 三次窗口周期 |
|---|---:|---|
| 连续 stride 1 | 256 | 2097378 / 2097378 / 2097378 |
| 同址广播 | 256 | 1135930 / 1135930 / 1135930 |

模型输入 `g,N,batches` 完全相同，因此必须给出同一个预测值。对这两点，任意共同预测的最大相对误差下界为 `(2097378-1135930)/(2097378+1135930)=29.735738%`，高于 10%。这否定了当前需求聚合与时间形式在完整范围内的资格；不能仅重新定价修复，也不能据此声称测得了物理事务数或唯一端口结构。

同 SM 计数器读差值的 cycle GCD 与最小间距都是 2；正 globaltimer 差值 GCD 为 32 ns，640 对中 608 对 ns 差为零。0 轮窗口在 1/4/8 warp 都为 112 cycle，拟合固定项为 54.126–1020.763 cycle；这些是声明协议内的观测与聚合项，不是固有计时分辨率或裸硬件服务开销。

本轮达到反证终点：未冻结，第三提交保留，66 条预列留出未执行，真实 CUTLASS 仅取得静态 SASS 资格，未执行其数值重放，不向主线接入参数。[完整报告](../../../../../../results/gh200_resource_campaign/access_rules/20261011-m11-lds-sts-bounded/run-20261011T210745Z/continuation-20261012/README.md)、`development-analysis/counterexample.json` 和全部原始条件可离线复核。


## m11 后续离线需求诊断（2026-10-12）

第一轮普通 LDS/STS 开发作业740961及反证见[原交付](../../../../../../results/gh200_resource_campaign/access_rules/20261011-m11-lds-sts-bounded/run-20261011T210745Z/continuation-20261012/README.md)。本次用户批准GPU预算0，复用全部540开发条件，终点为没有满足原联合要求的可冻结候选；没有冻结或执行66条留出。

全warp内按bank去重不同32bit地址字后，连续16B与广播不再具有相同需求。保留原时间包络、改用最大相对误差拟合，开发中位/最大误差为0.49113%/9.61986%；加性发射形式为6.27053%/25.03275%。前者仍有20/27组不存在同时满足原绝对周期容差及最大10%误差的参数，不能用相对误差覆盖局部失败。全部旧条件保留，需求批次仍非实测。

三个直接结论：原非广播连续读的一部分误差来自共同拟合；移除宽广播后，其他布局不能唯一确定广播的时间下限；已有依赖读三点违反两个公式共有的凸性约束，即使各取原±64cycle边界，仍差63293.94cycle。因此不继续第三种形式，不以当前拟合宣布bank机制通过。推导、训练参数见证、完整误差与留布局诊断见[本轮报告](../../../../../../results/gh200_resource_campaign/access_rules/20261012-m11-demand-offline-v1/README.md)。

后续48条件/144进程提案只用于区分完整warp地址集合与lane子组排列效应；同址复制的向量连续或交错排列，保持完整bank地址集合和逻辑返回字节不变。该提案在离线交付时未获授权、没有提交。用户现已批准整理后补开发对照并追加独立留出预算；授权变化见 [后续测量合同](PLAN.md#measurement-next)，原提案与旧预算记录保持只读。
