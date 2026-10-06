# R03：片上访问并发

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
