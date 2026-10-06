# S08：FP8 / INT8 批量测量方案

测量 E4M3、E5M2、S8、U8 在 `mma.sync` 与 WGMMA 路径上的计算服务。浮点用 FLOP，整数用 OP；单 CTA 与全 GPU 分别报告。

## 固定配置

| 路径 | 形状 | 参与组 | 类型 | 范围 | 配置数 |
|---|---|---|---|---|---:|
| FP8 WGMMA bounded | M64N64K32 | 1/2 warpgroup | E4M3/E5M2→FP32 | 单 CTA / 全 GPU | 8 |
| FP8 MMA lowering | M16N8K32 | 1/2 warp | E4M3/E5M2→FP32 | 单 CTA / 全 GPU | 8 |
| INT8 MMA、WGMMA | 上述对应形状 | 1/2 参与组 | S8/U8→S32 | 单 CTA / 全 GPU | 16 |

保留原两条独立链、每链每批 16 条指令、8192 轮、WGMMA wait0 及结束排空。FP8 WGMMA 使用已约定 bounded 输入，保留原正输入长累加失败记录，不混合两者。

FP8 MMA 的当前生成代码是 compiled lowering。其逻辑工作量和实际指令路径分别说明，保留 `exportable=false`；不能发布为原生 FP8 Tensor Core 吞吐。INT8 长循环仍须核对不溢出。

## 一次运行

提交前把原低精度与 bounded 两个探针、完整输出保存、独立参考和报告入口准备好。一次作业内编译结果按路径复用，连续完成所有未覆盖短检查；优先检查 WGMMA INT8、bounded 长循环及布局边界。

短验证通过适用的冻结规则后连续正式采样，不为每种精度另申请 GPU 或人工签一次。原 24 点与 bounded 8 点分别保留准确版本及工作量，最终在同一实验报告汇总。完成路径合法性和短正确性不等于性能合格。

逻辑工作量为 `2MNK×参与组G×累加链C×每链每迭代指令U×循环I×CTA数B`；单 CTA 除以自己的 `Δclock64`，全 GPU 除以 `globaltimer` 包络。S8/U8 将相同数量称为 OP。描述符准备、初始化与主计时循环分别保存。

## 已完成的入口与输出

本轮32配置在GPU734492的一份short分配内完成全部128次短kernel检查和320个正式进程。原生ARM CPU734502完整重算及独立实际B/C、发布核对均通过，24条条件参数已发布；结果见[正式说明](EXP-08-low-precision-results-v2.md)。

本次完整保存版本的探针、参考、运行器和收集器位于[冻结源码](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/collection-job734492/actual/repo/)。`run_node_s08.py`在单GPU分配内连续完成全家族，`collect_node_s08.py`从本地并行收集，`audit_s08.py`只读重算。完整二进制、源码gate、协议、进程与原始值按同一归档保留。复现使用新的输出身份和本次冻结版本，原32点不因文档或统计重生而重测。

## 完成后的结果

一张原 32 点的终态表，按路径、类型和范围分别画图；原生整数/浮点服务与 lowering 观察分开。每个合格参数附线程、链数、batch、wait、实际资源、输入和设备条件。流程与采样规则见[批量执行约定](RUN-REMAINING.md)。
