# EXP-04：GH200 访问路径基线与稳定性协议

## 测什么

在统一 v2 运行协议下重新测当前 12 个 SMEM/global 配置，检查 stride=1 的样本波动；增加两个空计时窗口。它们是给定访问组织的经验服务，不是物理端口上限。当前文件先冻结 S04-A 约定，未产生新的 GPU 结果。

## 参数与例子

单 CTA SMEM 读：256 个线程，每轮每线程 8 次 4 B 访问，8192 轮，计请求字节为 `256×8×4×8192=67,108,864 B`。除以该 CTA 的 clock64 差，得到 B/cycle/CTA；不把分配的 32 KiB 再乘入分子。

全局分配按 `blocks×256×16 B` 的整轮向上对齐，分子使用实际分配字节和迭代数；读写双向分别计数。`global_duplex` 是加载值再写出的依赖复制，不是独立读写竞争实验，后者由 S13 测量。

## 配置矩阵

| 家族 | 配置 | 范围与计量 |
|---|---|---|
| SMEM read | stride=1/2/4/8/16/32 | 一CTA、256线程、32KiB分配，8192轮，B/clock64 cycle/CTA |
| SMEM write | stride=1 | 同上，互不重叠写入；初值为poison |
| global read | 8MiB请求，ca/cg；256MiB请求，cg | 4×实际SM数CTA、256线程；8MiB为64轮，256MiB为16轮 |
| global write | 256MiB请求，16轮 | 同上，结束包含device fence |
| dependent copy | 输入输出各128MiB请求，16轮 | 同上，分子为读加写 |
| empty window | 一CTA、4×SM数CTA | 无访存工作量，报告ns/window；短grid不强制覆盖所有SM |

机器配置见 [memory_baseline.json](../../../../../microbench/gh200_resource_campaign/contracts/memory_baseline.json)，共14点。正式全局访存配置要求实测SM覆盖完整；空grid记录真实覆盖，不能据此导出整卡吞吐。

## 计时、正确性与稳定性

CTA在入口同步后取globaltimer/clock64，结束前同步；global写额外包含device fence。初始化、host传输和结果验证不在设备计时窗口内。空窗口不从实际窗口中扣除。

SMEM使用PTX volatile访问并检查实际循环中的LDS/STS；global读取全部uint4 lane，host按地址公式独立核对逐线程checksum。global写用另一个kernel逐元素验证；SMEM写核对所有已写区域的逐线程checksum。每个进程复用自己的分配进行8–30次预热，保留每次窗口，末5次CV≤2%记录为收敛；未收敛不能伪称稳定。

每点每批10次独立进程，正式CV>5%最多触发3批同协议采样，全部保留并合并统计。仍不稳定的点保留分布与说明，不作为合格标量参数导出。源码/协议变化建立新run，不跨协议挑选最好样本。

## 审查和使用边界

A审查实验约定；B审查探针、CPU负例、编译/SASS与最小GPU正确性；C在完整采样后独立重算与检查说明。不能用S04-A通过代替执行通过。

计数器只在独立运行中尝试；权限拒绝不影响已经验证的请求量计时，但缓存命中率与物理HBM流量保持未证明。global请求GB/s、单CTA B/cycle与空窗口ns分别展示。设备UUID、编译器、SM数与遥测绑定归档，旧memory-c保留不改写。

## 复现与结果

入口接到新 `run_suite.py` 的 S04 family；在 S02 接口与 S04-B 审查通过后给出实际命令、报告和图表。当前无新实测值，不能把v1结果填成v2结果。
