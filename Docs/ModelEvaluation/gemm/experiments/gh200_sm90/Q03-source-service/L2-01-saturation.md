# L2-01：固定工作集的并发服务曲线

更新：2026-10-17。阶段1，两个先导与开发48进程已完成并合格；六个预置读形式按稳定单调反例停止，未拟合参数、未冻结、未采留出。候选与模型使用位置唯一维护在[L2服务](../../../model/mechanism/hardware_rules/L2-SERVICE.md)，总体顺序见[模块地图](../../../model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)。

## 1. 对象与计时

GPU本地`cudaMalloc`分配，普通global地址空间，读用`.cg.v4.u32`，独立写用`.wb.v4.u32`。每数组17,301,504 B（16.5 MiB），混合两数组共33 MiB；实际L2容量由设备查询。输入和输出分别用地址相关uint32模式，读结果逐线程checksum、写结果逐word核对。

固定128线程、16轮完整遍历、相同总逻辑字节。J为1／4／8个软件独立加载／累加流，源码按组加载再消费；实际发出与消费次序逐构建核对。改变J只改变组织。16.5 MiB可被本轮所有活动CTA数×128线程×J×16B整除，所有条件完整覆盖同一大小的地址集合。

采用合作式启动，grid等于活动CTA数，动态SMEM预留使查询驻留上限为1 CTA/SM；各CTA在窗口前会合后访存。记录实际SMID、各活动CTA的起止和共同重叠区间，按观测报告活动规模。诊断写入在窗口后grid会合完成时执行。

主窗口为活动CTA的`max(end_globaltimer)-min(begin_globaltimer)`，另存每CTA的`clock64`差与CUDA event。读窗口包含结果消费与CTA排空；写／混合包含device fence和CTA排空。分配、初始化、poison、回读及host校验在窗口外，实际顺序进入结果。缓存准备为同一目标自身暖机，末5次CV≤2%，最少8次、最多60次；未收敛保留记录并停止该条件的性能拟合。

## 2. 有限条件与预算

| 用途 | 条件 | 数量 |
|---|---|---|
| 开发读曲线 | 活动CTA={1,8,32,132} × J={1,4,8} | 12 |
| 开发写／混合选点 | mode={write,mixed} × 活动CTA={32,132}，J=4 | 4 |
| 冻结留出 | mode=read，活动CTA={16,66} × J={4,8} | 4 |

每条件3个独立进程，每进程1个正式窗口，种子和执行顺序在prepare清单中固定；开发48进程、留出12进程。正式测量前先用最大活动规模的read/J8、mixed/J4各做一次正确性与计时先导，总上限62进程。每进程先执行1轮／2轮遍历的正确性检查各一次，再做最多60次暖机和1次正式测量。开发和留出分开目录；先保存开发候选及参数身份，再显式执行留出。

本轮最多六个读模型形式，定义见模型页。首轮GPU分配上限10分钟；先导不能在该预算内完成，或出现共同正确性／同步故障，就停止该批，保留失败。修复最多两次，只复跑实际受影响的条件；没有平台或候选均失败时交付未解决结果，不自动扩扫J、数组或SM集合。正式GPU执行许可按[GOALS当前范围](../../../GOALS.md#current-scope)处理，不占GPU分配等待实现或人工答复。

## 3. 执行与复现

源码：[l2_saturation.cu](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/l2_saturation.cu)。入口：[l2_saturation.py](../../../../../../microbench/gh200_resource_campaign/access_rules/l2_saturation.py)。`prepare`生成条件、总工作量及种子；`compile`保存构建和SASS；`run`才执行GPU测量。

准备阶段已生成[准备包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-preparation-001/plan.json)，包含probe、runner、候选数值内核及其SHA。原编译与开发入口如下，实际实机结果见本节后续执行目录：

```bash
python results/gh200_resource_campaign/access_rules/20261017-L2-01-preparation-001/source/l2_saturation.py compile \
  --output results/gh200_resource_campaign/access_rules/20261017-L2-01-preparation-001 --nvcc nvcc
python results/gh200_resource_campaign/access_rules/20261017-L2-01-preparation-001/source/l2_saturation.py run \
  --output results/gh200_resource_campaign/access_rules/20261017-L2-01-preparation-001 --split dev
```

两项先导在开发批之前用相同probe单独保存命令与输出；留出另需`--split heldout --dev-freeze <开发冻结身份>`。本地CUDA13与GCC16／glibc有兼容冲突，本地临时头覆盖只用于[静态审查](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-preparation-001/local-static-review/gh200-l2-saturation-audit.json)；正式目标使用受支持工具链重新构建并核对自己的SASS。

旧S13的源码、18条件、审核器及冻结结果保持原样。新批次保存自身源码／二进制／SASS身份、设备与资源、逐进程命令和原始JSON，性能资格由本轮检查决定。

本次[执行目录](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-execution-002/dev/summary.json)与[判定结果](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-fit-001/service.json)分别保存GPU观测和CPU分析；实际分析[命令](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-fit-001/execution/command.json)及[收据](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-fit-001/execution/receipt.json)已归档。用保存的入口复核开发证据时，输出到新目录：

```bash
python results/gh200_resource_campaign/access_rules/20261017-L2-01-fit-001/l2_fit.py fit \
  --input results/gh200_resource_campaign/access_rules/20261017-L2-01-execution-002 \
  --sass-review results/gh200_resource_campaign/access_rules/20261017-L2-01-fit-001/target-review.json \
  --output /tmp/l2-01-fit-replay
```

此入口只做CPU证据回放与六形式判定，不执行GPU；输出目录须尚不存在。准备阶段11项CPU检查保留，新增有限拟合／冻结／评分入口后共[20项CPU检查](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-fit-001/cpu-validation/stderr)通过，[检查命令](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-fit-001/cpu-validation/command.json)已保存。

## 4. 当前结果与下一步

两个先导通过；开发16条件／48独立进程完整，全部暖机收敛、实际活动窗口重叠、数值与性能资格通过。目标CUDA12.9.41的九个模板均通过SASS宽度／数量及零local审核。J仍按软件独立流解释，实际生成代码与资源保存在本批构建身份中。

[无损回传核对](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L2-01-execution-002/transport-verified.json)验证318个文件、231098176 B原始内容的全部SHA；GPU执行返回码0。原始计时、逐线程checksum及完整写出均保留，内存路径与第1节计时边界不变。

读窗口的三个进程中位数如下，单位µs；它们包含该访问体的加载、地址计算、消费和排空。

| 实际活动SM数S | J1 | J4 | J8 |
|---|---:|---:|---:|
| 1 | 22612.384 | 10776.608 | 10732.896 |
| 8 | 2865.056 | 1363.776 | 1359.712 |
| 32 | 742.496 | 354.528 | 355.424 |
| 132 | 180.864 | 87.904 | 87.392 |

J1→4的逻辑吞吐改善约2.1倍，J4→8近平台；固定J时逻辑吞吐随S近似线性增加，共享服务平台尚未单独辨识。S32／J4三个窗口为353856–354592 ns，S32／J8为355328–355552 ns；J8的中位窗口增加约0.253%，且两组三进程范围不重叠，触发预置的稳定单调反例规则。

六形式全部记为`refuted_stable_monotonic_reversal`，`selected=null`。参数与误差未拟合，未生成`dev-freeze.json`，四个留出条件／12进程未采。本轮按原候选与停止规则收口，不扩点或开展新模型搜索；具体单调供给函数未通过，物理原因仍未区分。写／混合四个选点与原窗口单列保存；没有从读结果推算其服务参数，也没有将整窗率接入cfg_b源价格。后续按已授权顺序推进计算A，独立C2/C3验收未完成。
