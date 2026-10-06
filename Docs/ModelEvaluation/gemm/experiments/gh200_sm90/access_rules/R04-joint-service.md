# R04：跨路径联合服务

[总计划](README.md)。状态：旧LDS/CVT计量无效，修订版选择性复核中；正式进度见[STATUS](STATUS.md)。代码：`run_r04.py`、`probes/r04.cu`。使用 R00、R01、R03 的基线点。

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

## 当前实现与复核入口

每个单元的 A+B=160。A 计为一条 warpgroup WGMMA 或每个参与 warp 一条 LDS.128；B 计为每个参与 warp 一条 FFMA 或 CVT，四个独立状态轮换。128线程共有4个warp，聚合B指令数为 `4×n_B×repeats`。例如 LDS+FFMA 的 A:B=128:32，每单元是262144 B逻辑读和8192 FLOP。

WGMMA每8次commit/wait1，单元末尾wait0；串行在A排空后执行B。交错使用固定指令序列，1:1为AB，1:4为ABBBB，4:1为AAAAB。跨warpgroup的串行点含阶段CTA同步，所有跨组点在单元末尾同步；控制成本包含在窗口中。WGMMA和FFMA的输入在相邻单元正负配对，使长循环结果有界，短检查使用坐标相关输入。两条路径无数据依赖。

160次A-only/B-only按次数缩放的max/sum只作比较，未独立证明线性；它们不是已验收预测。匹配R03的LDS地址/宽度/warp数基线仍为依赖。资源、完整输出、短检查、预热和原始计时逐进程保存；机器码中的控制分支成本不能归因于物理端口。

在单GPU GH200 Slurm分配内运行（CUTLASS使用R00归档的准确头文件）：

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r04.py \
  --cutlass-root /准确归档/source/cutlass \
  --output /新运行目录/r04 --smoke
python3 microbench/gh200_resource_campaign/access_rules/run_r04.py \
  --cutlass-root /准确归档/source/cutlass \
  --output /新运行目录/r04-formal
python3 /新运行目录/r04-formal/source/analyze_r04.py \
  --input /新运行目录/r04-formal
```

正式入口先检查4个8197单元长循环，按家族最慢pilot选共同长度，再完成32点3进程；CV>1%或串行/交错差不足3倍进程波动时，整个家族补至10。CV>5%时最多额外两批，每批3进程，旧样本全部保留。

## 实测结果（2026-10-06）

[R04正式结果](../../../../../../results/gh200_resource_campaign/access_rules/20261006-A-job735062/r04-final-review/combined_report.md)。设备romeo-a043，GPU-099dda56-d7af-f60e-c285-aa2dc7ddfcfe，CUDA12.9。

32点/96个有效正式进程，CPU完整输出最大误差0。WGMMA16点采用formal-v4；LDS16点复用formal-v3，八个相关kernel的完整SASS含编码逐字相同，UUID/驱动/CUDA一致。早期formal-v1/v2数值失败与formal-v3被修订替代的WGMMA数据全部保留。

同warpgroup八条成组发出的修订消除了逐条串行化警告。跨warpgroup的7个有WGMMA点仍存在C7520与逐条wait0，适用域是实际生成序列；不能按源码wait1描述有效在途量。原LDS+CVT的5.36%/9.82%结论已撤回：转换被外提，动态工作量不成立。旧数据标invalid_dynamic_work保留。

R03代表点已取得，但在另一UUID上、使用8槽轮转和每次checksum消费；本组是固定地址重读、末尾消费，不能直接拼接101.132 B/cycle。需要条件匹配的最小点，不需要重测整组R03。

一次CPU复核入口（输出到新目录、保留原始报告）：

```bash
python3 microbench/gh200_resource_campaign/access_rules/analyze_r04.py \
  --input results/gh200_resource_campaign/access_rules/20261006-A-job735062/r04-formal-v4 \
  --reuse-lds results/gh200_resource_campaign/access_rules/20261006-A-job735062/r04-formal-v3 \
  --output /新目录/r04-review
```

## 动态工作量修订

主对话CFG审查发现旧16个LDS家族条件表不成立：固定地址LDS的na内层被消掉，恒定输入CVT被外提；CPU结果和静态时钟窗口内出现次数不能证明动态工作。对应原始数据保留，停止用于工作率、缩放或并发结论。旧FFMA-only原点单独保留，不作无效LDS表的补位。

修订LDS为`ld.volatile.shared.v4.b32`。串行/A-only单元在末次四寄存器读的真实消费与CTA同步后结束A，再进入B；这些控制和消费成本包含在窗口。CVT每个action输入精确增加1/256，RNE FP16位模式逐次进入16bit校验和；B包含FADD准备和checksum消费，不是裸CVT速率。

`build/dynamic-work-cfg.json`逐kernel保存内层回边、counter更新/比较、目标PC和实际序列。基线/串行：A loop每次1 LDS，trip=n_A；B loop每次4 FFMA/CVT，trip=n_B/4。交错1:4为1A+4B×32，1:1为4A+4B×20，4:1为16A+4B×8；每个目标同时处在action和repeat循环。编译器可能重排展开体，按证据中的实际顺序解释。

跨WG保留真实串行条件，源级显式每条`fence→MMA→commit/wait0`；内联块未能恢复异步UR描述符路径，不再以wait1解释。新编译无C7520，C7519只留在未改同WG的初始化路径。新数据按7c184…UUID保存，不与旧099dda…或R00 ec947…拟合。
