# EXP-09：SMEM 访问宽度、广播与联合读写

## 实验问题与参数

在固定单 CTA、256 线程、64 KiB SMEM 分配下，普通线程的访问宽度、地址映射和同时读写怎样影响请求服务？结果是给定访问组织的 B/clock64 cycle/CTA，不是整卡或物理端口峰值。

这是S09-A实验约定，尚无新实测。32 KiB分配的S04基线继续独立保存；这里固定两个32 KiB数组，保证本组的read-only、write-only和duplex分配条件相同。

## 配置与最小例子

| 访问 | 配置 | 数量 |
|---|---|---:|
| scalar read | 4 B，stride 1/2/4/8/16/32 | 6 |
| vector read | 8/16 B，stride 1 | 2 |
| write | 4/8/16 B，stride 1 | 3 |
| broadcast read | 4 B，同warp同一地址 | 1 |
| independent duplex | 4/8/16 B，各8次读及8次写/线程/轮 | 3 |

共15点，每点8192轮。每线程每轮每个方向8次访问。16 B的1:1独立读写一次测量请求量为 `256×8×16×8192×2=536,870,912 B`；读写各268,435,456 B。分子是请求量，读写结果不能各自称为总速率。

普通地址以32-bit word计为 `((tid+q×256)×stride×W) & 8191`，W是一次访问含几个word。广播仅scalar read，地址为 `(tid/32)×8+q`，一个warp内32线程消费同一word；广播的逻辑请求量与唯一地址字节数分列。stride和广播不会自动降低分子，更不能把请求量当物理事务量。

## 计时与正确性

入口准备同步后取时，再经CTA barrier确保所有被计访问晚于已记录起点；结束同步后取时。循环、checksum累加和同步在窗口内，初始化/host回读/输出核验排除。

每个加载lane都参与checksum，host独立枚举地址和uint32回绕。写入先初始化不同poison，计时后逐word对照地址/seed值。duplex输入输出为不同数组，store值只由地址/seed计算，与load值无依赖；不与S04的copy-like duplex混同。广播不建立同址并发写案例。

## 接受条件与使用边界

遵守公共10外部进程、有界预热/三批和不可择优协议。B阶段必须定位实际计时循环中的LDS/STS、访问宽度和次数，检查vector对齐及所有输出。A约定通过不表示B已实现或GPU通过。

机器矩阵见 [shared_memory.json](../../../../../microbench/gh200_resource_campaign/contracts/shared_memory.json)。源码、复现命令、原始结果和图表在B/C完成后补入；不填虚构结果。该服务不替代ldmatrix、Tensor Core取数、TMA写SMEM或多CTA并发能力。
