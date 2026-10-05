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

单 CTA 指标为 `2MNK×实际参与组×链数×指令数 / Δclock64`；全 GPU 再乘实际 CTA 数，除以 `globaltimer` 包络。S8/U8 将相同数量称为 OP。描述符准备、初始化与主计时循环分别保存。

## 当前需要补的代码

现有 [low_precision.cu](../../../../../microbench/gh200_resource_campaign/probes/low_precision.cu) 与 [bounded 探针](../../../../../microbench/gh200_resource_campaign/probes/low_precision_bounded_fp8_v1.cu)已经存在。需补必要完整输出归档，将两条路径接入同一个正式执行入口，并在提交前验证该入口。现有 error_count-only 输出不能补称为可独立逐值重算。

## 完成后的结果

一张原 32 点的终态表，按路径、类型和范围分别画图；原生整数/浮点服务与 lowering 观察分开。每个合格参数附线程、链数、batch、wait、实际资源、输入和设备条件。流程与采样规则见[批量执行约定](RUN-REMAINING.md)。
