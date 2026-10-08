# B01：带宽与缓存准备覆盖

24个代表坐标已有完整覆盖：global 6个、独立SMEM读/写4个、TMA tensor 6个复用合格证据；另外2个大工作集TMA、2个SMEM独立联合读写和4个CUTLASS缓存准备条件已在job737122正式采样并独立复算。新增8条件均每点10进程；旧数据未重新采样或与新卡拼接拟合。

## TMA tensor：按不同地址量匹配

复用S15的rank-2 UINT16 tensor copy、连续布局、每CTA32个独立全局槽，单请求在途。输入完成边界为mbarrier acquire后CTA会合；输出为commit与wait_group 0后CTA会合。计量分子为`CTA数×迭代数×payload`，不把不同地址量当作重复请求的分子。

| 方向 | 范围 | payload | CTA数 | 不同地址量 | 条件服务 | 单位 |
|---|---|---:|---:|---:|---:|---|
| input | one_cta, below_l2 | 16384 B | 1 | 524288 B | 21.377 | B_transport/clock64_cycle/CTA |
| input | whole_gpu, below_l2 | 1024 B | 528 | 17301504 B | 1754.746 | GB_transport/s/GPU |
| input | whole_gpu, above_l2 | 8192 B | 528 | 138412032 B | 3815.629 | GB_transport/s/GPU |
| output | one_cta, below_l2 | 16384 B | 1 | 524288 B | 22.200 | B_transport/clock64_cycle/CTA |
| output | whole_gpu, below_l2 | 1024 B | 528 | 17301504 B | 2885.323 | GB_transport/s/GPU |
| output | whole_gpu, above_l2 | 8192 B | 528 | 138412032 B | 3810.237 | GB_transport/s/GPU |

设备L2实查为62,914,560 B。例：整卡1 KiB payload的不同地址量为`528×32×1024=17,301,504 B`，小于L2；8 KiB payload为`528×32×8192=138,412,032 B`，大于L2。两者改变payload，并非只改工作集大小的因果A/B；用于填补各自条件坐标，不能把速率之比解释为纯缓存收益。单CTA既有最大工作集只有`32×64KiB=2MiB`，因此另测大于L2的单CTA条件，见下文。

[原实验解释](../EXP-15-tma-2d.md)、[合格参数](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/published-all68-v1/qualified-parameters.json)。本轮核对了每条所指summary/device/protocol/source-manifest哈希与既有独立数值、发布审阅的通过状态；复用旧审阅，不声称本轮重放了旧组全部大数组。机器覆盖清单保存精确case、SHA256、工作集、完成语义和单位。

## 其余路径

global六坐标沿用[全局读写实验](../EXP-13-global-rw.md)，请求速率按整卡GB/s。独立LDS.128/STS.128四坐标沿用[R03](R03-access-demand.md)，完成边界包含消费者及CTA会合，按B/clock64-cycle/CTA。旧SMEM拷贝存在读到写的数据依赖，不能代替独立读写联合服务。

[R08](R08-waves-l2-reuse.md)有swizzle对照，但没有匹配的驱逐准备条件；[R00](R00-anchor-target.md)的驱逐对照使用cuBLASLt，旧CUTLASS重复条件又含debug断言。这些都不能拼成当前要求的四条件CUTLASS比较，因此四点使用同一cfg_a可执行文件补测。驱逐准备只定义操作协议，没有计数器时不称保证冷HBM。

以上条件来自各自设备和时钟域；不能把B/cycle/CTA与GB/s/GPU直接取最大值或相加，也不能由大工作集请求速率推物理HBM流量。完整GEMM参数迁移另由V07判定。

## 新增四个微基准坐标

同一GPU-099dda56、CUDA12.9.41、sm_90a、NDEBUG。每点10正式进程，另有4个短检查，共44进程、671,307,872个保存值独立重算；所有首末SM一致、无spill。TMA使用rank-2 UINT16位模式，不更改swizzle、interleave或L2 promotion。

| 条件 | 请求字节 | 不同地址量 | cycle中位数 | B/cycle/CTA | CV |
|---|---:|---:|---:|---:|---:|
| LDS.128＋STS.128，1 warp | 2097152 | 8192 | 56573 | 37.070 | 0 |
| LDS.128＋STS.128，8 warp | 16777216 | 65536 | 131389 | 127.691 | 0 |
| TMA输入，单CTA大工作集 | 134217728 | 134217728 | 10323295 | 13.001 | 0.051% |
| TMA输出，单CTA大工作集 | 134217728 | 134217728 | 5480674 | 24.489 | 0 |

SMEM读区和写区独立，写入值不依赖读值；每lane每轮8次16 B读与8次16 B写，共256轮。真实手算：`32×256×8×(16+16)=2,097,152 B`，除以56573为37.070 B/cycle/CTA。读取有独立整数sink，所有warp消费后会合再取末时钟；全部保存写值和sink核对。该分母也包含地址、整数消费、循环与会合，不能将旧单独读/写率直接相加来替代它。

大工作集TMA每CTA128线程、thread0发起，8192个不同16 KiB槽，一轮遍历`8192×16384=134,217,728 B`，超过62,914,560 B的L2。每次仅一个请求在途；输入等mbarrier acquire，输出commit后wait_group0，两者再CTA会合。输入保存最后代际全部8192半字；输出保存全部67,108,864半字，每次启动改变输出代际，避免旧暖机写入冒充正式完成。短检查用3槽识别代际，正式工作量经SASS循环/请求核对。

这些是新编译帧的条件服务；不能和另一张卡的旧S15小源相除，导出纯缓存惩罚或物理HBM带宽。[微基准归档](../../../../../../results/gh200_resource_campaign/access_rules/20261008-B01-job737122-v4/)、[独立规则](../../../../../../results/gh200_resource_campaign/access_rules/20261008-B01-job737122-v4/reanalysis/local-independent/rules.json)。v2的SASS检查误把UTMACMDFLUSH计成tensor请求；v3的8-warp启动缺少64 KiB SMEM opt-in；均保留为诊断，不进入正式表。

## CUTLASS缓存准备的四条件

cfg_a、4096×4096×2048、同一二进制、cluster2×1、6 stage；1024个物理tile，swizzle1/8都不额外补齐。输入不同地址量33,554,432 B，输出67,108,864 B。驱逐组每次调用前写125,829,120 B独立缓冲，即2×实际L2，同步完成后才开始CUDA event；重复组不执行该准备。分配、准备与检查均在窗口外。

| swizzle | 重复输入 μs | 驱逐准备 μs | 驱逐相对重复 |
|---|---:|---:|---:|
| 1 | 103.008 | 104.384 | +1.34% |
| 8 | 104.368 | 103.888 | −0.46% |

40个进程的163,840个抽样输出值全部正确，CV为0.40–0.98%。两种swizzle没有呈现统一的大幅“冷缓存惩罚”，负差也保留。有效工作为`2×4096×4096×2048=68,719,476,736 FLOP`，swizzle1重复输入对应约667.13 TFLOP/s；未把驱逐写入计为GEMM FLOP或放进GEMM计时分母。

[缓存准备归档](../../../../../../results/gh200_resource_campaign/access_rules/20261008-B01-cache-job737122-v1/)、[独立核对](../../../../../../results/gh200_resource_campaign/access_rules/20261008-B01-cache-job737122-v1/reanalysis/local-independent/rules.json)。工作集大小和请求量不证明实际缓存命中或物理流量。

![新增带宽与缓存条件](../../../../../../results/gh200_resource_campaign/access_rules/20261008-B01-job737122-v4/reanalysis/local-independent/b01-services.png)

误差条为进程标准差；前两图为单CTA请求B/SM-cycle，右图为完整GEMM微秒，不互相换算。

## 复现

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_b01.py prepare --output <新微基准目录>
python3 <微基准目录>/source/run_b01.py build --output <微基准目录>
python3 <微基准目录>/source/run_b01.py sample --output <微基准目录>
python3 microbench/gh200_resource_campaign/access_rules/run_b01_cache.py prepare --output <新缓存目录> --cutlass-root <CUTLASS3.9.2>
python3 <缓存目录>/source/run_b01_cache.py build --output <缓存目录>
python3 <缓存目录>/source/run_b01_cache.py setup --output <缓存目录>
python3 <缓存目录>/source/run_b01_cache.py sample --procs 10 --output <缓存目录>
python3 microbench/gh200_resource_campaign/access_rules/analyze_b01.py --input <微基准归档> --output <新分析目录>
python3 microbench/gh200_resource_campaign/access_rules/analyze_b01_cache.py --input <缓存归档> --micro <同卡微基准归档> --output <新分析目录>
```
