# R01：依赖与结果可用

[总计划](README.md)。状态：37 点 / 118 正式进程完成（循环展开版），CPU 与 SASS 复核通过。代码：`run_r01.py`、`probes/r01.cu`。

## 问题

依赖链每步多长，独立流能把工作率提高多少？对 Tensor Core GEMM 最关键的是 WGMMA 随 N 的依赖时间：它决定需要几个 stage、`wait_group` 能留几组。FFMA、LDS、LDG 的依赖延迟已有公开 Hopper 微基准，本组只用少量点核对。

只测合法依赖：消费者确实读取生产者的值，不删除硬件依赖检查或异步等待来制造提前读取。

## 默认矩阵（37 点）

| 子集 | 操作 | 坐标 | 数量 |
|---|---|---|---:|
| R01-A，依赖链 | FP32 FFMA、32-bit 整数 add、SMEM `ld.u32`、global `.cg ld.u32` 小/大工作集，共 5 类 | 1 warp、1 链；128/512/2048 步 | 15 |
| R01-C，WGMMA 依赖 | FP16 SS `m64nNk16`，N=64/128/256（按 [R00-B](R00-anchor-target.md) 调整）；1 个 warpgroup、1 条累加链；每步 1 条 WGMMA + commit + wait0 | 128/512/2048 步 | 9 |
| R01-D，`ldmatrix` 依赖 | `ldmatrix.x4`，结果中的一个元素决定下一次行地址 | 1 warp；128/512/2048 步 | 3 |
| R01-B，独立流 | R01-A 的 5 类；访存用多条独立地址链，计算用独立累加器 | 4 warp × 每线程 1/4 流；2048 步 | 10 |

- R01-A 的访存只有 1 个活跃 lane 走地址链；计算用全 warp。R01-B 用全 warp。
- 地址链是固定种子生成的完整随机环，每个节点恰好访问一次；SMEM 存索引，global 存索引或指针，地址转换计入链时间。
- SMEM 工作集 16 KiB；global 小档 L2/4、大档 4×L2。两档每个窗口前用并行 `.cg` 线性读取整个集合。准备不计入窗口；小档准备不证明固定驻留，大档准备不证明后续每次来自 HBM。记录工作集与准备方式。
- 另有 1/4 warp 空计时与消费控制，不计入配置数。

## 时间与工作量

计时前初始化并预热；窗口内执行链、循环控制和最终消费者，消费者完成后停止。对三种长度拟合 \(T(N)=a+bN\)，报告原时间、每步增量 \(b\) 与残差；三点不呈线性时改为查表。V01 的不同长度由 A 执行对话留出；本组不提前采样该点。

- 计算：`FLOP=2×有效lane×warp数×流数×步数`；整数 add 每次 1 OP。
- 访存：`B_requested=4×有效lane×warp数×流数×步数`，地址转换不计字节。
- WGMMA：`FLOP=2×64×N×16×步数`（warpgroup 协同，不乘线程数）。`ldmatrix.x4`：`B_requested=512×步数`。

WGMMA 每步只有一条指令并以 wait0 结束，所以每步增量是一条 WGMMA 从发出到完成的上界（含 commit 与 wait；循环控制每 64 步一次）。

手算模板：一个活跃 lane 1024 次 32-bit load，请求 4096 B；若窗口 81920 cycle，则每步 80 cycle、请求率 0.05 B/cycle。

## 正确性

- 保存每个 lane、每条链的最终值，CPU 按相同步数计算参考；输入避免溢出或快速收敛为常量。
- 与预装结果的消费者控制比较，确认窗口没有在值可用前结束。
- SASS 确认依赖、load、消费者都在，记录地址指令与循环结构。

## 输出

5 类依赖链的每步增量与独立流工作率；WGMMA 每步增量随 N 的变化；`ldmatrix.x4` 每步增量。用于 L0/L1 的就绪时间与吞吐。若 R00 的 BF16 迁移对照与 FP16 不一致，补 BF16 目标形状的依赖点。

## 实现与复核

`probes/r01.cu` 使用 CUDA 12.9 与 CuTe v3.9.2，原生 `-gencode=arch=compute_90a,code=sm_90a`。37 点用同一结构：计时循环每次迭代含 64 个依赖步（R01-B 为每条流 64 步），循环计数、比较和回跳每 64 步一次；N=128/512/2048 对应 2/8/32 次迭代。SASS 核对每次迭代恰有 64×流数条目标指令、1 条比较和 1 条回跳；单链 kernel 逐条核对每条链指令读取上一条的结果寄存器。

- FFMA：`fma.rn.f32 x,x,1,2^-10`，SASS 为 64 条同寄存器 FFMA。
- 整数 add：\(x_{n+1}=x_n+x_{n-1}\)（mod 2^32）。加常数或循环不变量时，ptxas 把两次 add 合成一条 IMAD/LEA，链长减半；此式每个中间值被读两次，不能合并。SASS 为 IADD3 与 IMAD.IADD 混合（单链 29/35 条），不归因到单一执行部件。
- 访存：地址链是固定种子的完整随机环，存索引；SMEM 每步 LEA+LDS，global 每步 IMAD.WIDE+LDG.STRONG.GPU（`.cg`）。R01-A 只有 lane0 走链，R01-B 全 lane 各走 1/4 条链。
- `ldmatrix.x4`：三个 512 B 块成环，块 k 的偶数 b16 元素存块 (k+1)%3 的字节偏移、奇数元素为 0，各 lane 的 d0 直接是下一块偏移，每步 IADD+LDSM；N 步后偏移为 512·(N%3)。
- WGMMA：128B swizzle 行，描述符覆盖指令 K=16；每步 1 条 SS `m64nNk16`、commit、wait0，SASS 为 HGMMA+WARPGROUP.DEPBAR，每次迭代 1 条 WARPGROUP.ARRIVE。输入按行列变化且可精确表示，CPU 按 fragment 行列映射复核全部累加器。

计时内最终结果写入 volatile shared 消费者，CTA barrier 后停止；全局输出拷出在窗口外。空控制和预装结果消费者控制保留原始窗口，不扣减。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r01.py \
  --output results/gh200_resource_campaign/access_rules/NEW/r01 \
  --cutlass-root /path/to/cutlass-v3.9.2
python3 microbench/gh200_resource_campaign/access_rules/analyze_r01.py \
  --input results/gh200_resource_campaign/access_rules/NEW/r01
```

`--list` 只列 37 点；`--smoke` 编译后对每个坐标做 64/128 步（1/2 次迭代）短检查，共 38 进程。正式运行先做同样短检查和 12 个控制，再按固定种子打乱、成对相邻采样。每进程重新初始化、8–30 个预热窗口（末 5 个 CV≤2%）、1 个正式窗口；3 进程起步，CV>1% 补到 10；独立流配对差 ≤max(64 cycle, 3σ) 时两点共同补 10；CV>5% 最多补两批。三点线性判断用残差 ≤max(64 cycle, 3σ, 窗口 1%)，是预先定义的经验检查。输出 `cases.csv`、`rules.json`、`cpu_check.json`、`sass_check.json`、`report.md` 和 `plots/dependency_windows.svg`。

## 实测结果（2026-10-06，循环展开版）

[完整报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-r01r06-unrolled/r01/report.md)、[规则与原始拟合](../../../../../../results/gh200_resource_campaign/access_rules/20261006-r01r06-unrolled/r01/rules.json)、[SASS 核对](../../../../../../results/gh200_resource_campaign/access_rules/20261006-r01r06-unrolled/r01/sass_check.json)。作业 735191，romeo-a041，GPU-7c184a2e-41ea-3d2b-df5b-1699c95fc1fe。37 点 118 正式进程，加 38 短检查、12 控制共 168 进程，393120 个输出 CPU 逐值重算通过；最大正式 CV 1.12%（global 大工作集 128 步，已补到 10 进程），预热全部收敛。

|路径|每步 SASS|每步增量 b，clock64 cycle|
|---|---|---:|
|FFMA|FFMA|4.27|
|32-bit add|IADD3 / IMAD.IADD|4.98|
|SMEM 地址链（单 lane）|LEA+LDS|29.00|
|global 小 / 大工作集（单 lane）|IMAD.WIDE+LDG|287.7 / 629.6|
|`ldmatrix.x4` 结果到下一地址|IADD+LDSM|34.02|
|WGMMA N=64 / 128 / 256|HGMMA+wait0|76.3 / 115.1 / 194.9|

R01-B（4 warp、2048 步）4 流相对 1 流的请求工作率提升：FFMA 4.07×、add 4.66×、SMEM 2.43×、global 小 2.74×、大 3.48×。

b 是 T(N)=a+bN 的斜率；三点残差除 global 外为 0，global 在预设阈值 max(64 cycle, 3σ, 窗口 1%) 内。b 仍含每 64 步一次的循环成本：同一源码 U=32 的对照（[unroll32_reference](../../../../../../results/gh200_resource_campaign/access_rules/20261006-r01r06-unrolled/r01/unroll32_reference/)）给出 FFMA 每次迭代约 17 cycle，即 U=64 时每步残余 0.27 cycle，扣除后 FFMA 为 4.00 cycle，与已知 Hopper FFMA 依赖延迟 4 cycle 及 [EXP-05](../EXP-05-fma.md) 的 4.4 cycle/FMA（每迭代 16 条）一致；add 对 U 无可见依赖，5.0 cycle 是 IADD3/IMAD.IADD 混合链的平均。SMEM 与 `ldmatrix` 每步各含一条整数地址指令。global 两档与旧结果基本相同，长延迟原本就遮住了循环。WGMMA 每步约 36+0.62·N cycle；N=256 为 2690 FLOP/cycle，是单 SM 峰值 4096 FLOP/cycle（[EXP-07](../EXP-07-wgmma.md)）的 66%，逐步 wait0 的单链填不满 Tensor Core。R01-B 中 FFMA 1 流与 4 流窗口几乎相同（9015 / 8855 cycle）：4 条独立链按每 cycle 一条发射，正好覆盖 4 cycle 依赖延迟。

2026-10-06 早先的作业 735060 结果（[20261006-b-job735060](../../../../../../results/gh200_resource_campaign/access_rules/20261006-b-job735060/)）每次迭代只有 1 步，被约 29 cycle/迭代的循环成本限制（FFMA/add 均为 29，`ldmatrix` 58，WGMMA 105/137/201），已被本节取代。
