740160 在 a058 排他分配（288 CPU、4 GPU）完成固定 cfg_c 集成诊断，实际 CUDA UUID 为 `GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114`。使用 740156 的同一 `c5476d9b...` binary；命令、严格 reader、分析入口和全部依赖在采样前冻结于 86fd7a8。前次 740159 的 setup 误带 full-D 导出参数，在 Args 检查阶段退出；该失败单独保留，实际三个采集命令与分析规则没有调整。

全 D 7,077,888 元素 off/on 逐位相同，全部与独立 period-17 精确参考一致。随后各一个新进程的 off/on warm 记录均完整解码 132 CTA、工作坐标、两个 WG、所有原端点；每个记录 4096 个边界检查值通过。off shadow 全零；on shadow 没有零读数、逆序、越界或残留未用槽。

on 的 432 个 WG 窗口提供 864 个 GT-C-GT 三元组。GT 读数域宽度分布为 0:58、32:729、64:75、96:1、128:1；中位32、最大128。2 个宽度超过64，没有超过256。这里的0是两次GT返回相同数值，不是零物理采样误差或已标定分辨率。它们是单次记录的描述统计，不能把旧10-pair64/256门槛用于这批数据后授资格。

A的精确区间模块与reader对全部432窗口重算一致，显式额外GT坐标误差预算设为0仅用于检查读数域算术，不声称连续物理ns误差为0。FIRST/MAIN的条件elapsed为[MAIN_pre−FIRST_post, MAIN_post−FIRST_pre]。二进制身份变化，旧prefix模型及旧校准不能直接用于本数据。

本批不检验observer/off-off分布，不重新评分预测，不授读取硬件包围、物理精度、局部观测或组合预测资格。一次off事件93.12000126us、on事件94.24000233us仅保存原记录，不能估计总体扰动。下一步在同binary、完整固定几何与历史下独立冻结扰动和null控制。

`python verify.py` 核对冻结输入/返回hash、排他分配和卡身份、全部D、数值样本、完整trace与shadow，输出verification.json和stamps.json.gz；`python crosscheck_intervals.py` 复核与A模块的432条件区间一致性。完整原始数据和full-D输出在output.tar.gz；全部采集进程已结束。
