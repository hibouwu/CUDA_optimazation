# Evidence Summary

本文是当前唯一推荐引用的证据说明。更早的 bank/stride/physical-2bank 分析文档已删除，因为它们与当前口径重复，且部分表述过度外推。

## 直接观测

在 Thor `sm_110`、CUDA 13、单 warp、patched SASS 条件下：

- `LOP3`：三个同奇偶源约 `3.070` c/op；混合奇偶三源约 `2.086` c/op。
- `FFMA`：三个同奇偶源约 `3.064` c/op；混合奇偶三源约 `2.072` c/op。
- `LOP3/FFMA` 的 stride 1-16 扫描均表现为奇数 stride 快、偶数 stride 慢。
- `LOP3/IMAD` tuple scan 中，`mod2` 模型 40/40 命中，`mod4/mod8/mod16` 命中率更低。
- source-slot permutation 不改变快慢关系。
- multi-chain pressure 未暴露稳定的 `mod4/mod8` 物理分层。
- NCU 离线 metric 查询未找到直接 RF/register/operand-collector bank conflict counter。

## 支持的模型

这些结果支持如下可见行为模型：

```text
visible_group(register) = physical_register_id % 2
```

更具体地说：

- 每个可见组至少能在无额外可见延迟下服务两个源读取。
- 单条指令第三个同组源会暴露约一个周期的额外读取或收集步骤。
- 该可见分组跨 `LOP3`、`FFMA`，并在 `IMAD` tuple/physical 对照中复现。

## 没有证明的内容

当前数据不能证明：

- 寄存器文件物理 SRAM bank 数正好是 2。
- `physical_register_id % 2` 就是底层 SRAM bank 映射。
- 限制一定发生在 SRAM array，而不是 read port、collector 或仲裁逻辑。
- 全部寄存器编号、全部 opcode、Tensor Core、uniform register 都使用相同路径。

因此，报告中应避免“物理 2-bank 已证明”“99.99% 置信度”这类表述。

## 为什么 timing 不足以推出物理 bank 数

时间测量识别的是 operand tuple 是否竞争同一服务资源。以下微架构都可能产生相同 timing：

1. 两个物理 bank，按奇偶映射。
2. 四个或更多物理 bank，在 read/collector 层汇聚成两个奇偶可见组。
3. 更细的 SRAM 分组存在，但当前指令路径没有把它暴露成可测瓶颈。

所以当前实验能排除的是简单的独立 `mod4/mod8` 可见服务模型，而不是所有多物理-bank 设计。

## 实验必要性

| 实验 | 作用 | 状态 |
|---|---|---|
| source-count controls | 区分“三源本身慢”和“第三个同组源慢” | 必要，已完成 |
| source-slot permutation | 排除固定 operand slot 特殊路由 | 必要，已完成 |
| base × stride scan | 暴露奇偶周期和单链延迟 | 必要，`LOP3/FFMA` 已完成 |
| tuple scan | 区分 `mod2` 与简单 `mod4/mod8/mod16` 模型 | 已完成，`LOP3/IMAD` |
| multi-chain pressure | 检查更细分层是否在吞吐压力下暴露 | 已完成，未见稳定分层 |
| NCU metric query | 寻找直接 counter 或旁证 | 已完成，未找到 RF bank counter |

## 推荐报告表述

推荐：

> Thor SM110 上的 patched-SASS 微基准显示：已测试的标量寄存器操作数读取路径暴露出按物理寄存器编号奇偶划分的两个有效服务组。当单条指令三个 RF 源都落在同一可见组时，会出现约一个周期的额外延迟。该结果描述的是已测试 operand path 的可见行为，不足以单独确定底层物理 SRAM bank 数。

不推荐：

> Thor 的 register file 已被证明是两个物理 SRAM bank，且 `register_id % 2` 就是物理 bank 映射。

## 下一步

若要继续逼近物理 SRAM bank 数，优先做：

1. 补一个与 `LOP3/FFMA` 同构的干净 `IMAD/IADD3` 主扫描。
2. 扩大物理寄存器编号和任意 tuple 覆盖，而不只是增加等距 stride。
3. 寻找经验证的 RF read/port/bank counter；若没有，NCU stall 指标只能作为旁证。
