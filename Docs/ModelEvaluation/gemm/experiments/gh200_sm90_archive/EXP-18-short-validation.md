# S18：驻留条件、local/spill 与辅助操作的短验证

作业 `731482` 在同一次 `short` 单 GPU 分配中完成 28 个配置、84 次短启动，作业状态为 `COMPLETED 0:0`，耗时 3 分 59 秒。28 个配置的完整输出、实际编译与资源、计时域及归档已通过独立短正确性 B3 审查。本记录不提供正式延迟、吞吐或 FLOP/cycle 参数。

## 验证范围

| 配置类别 | 配置数 | 检查对象 |
|---|---:|---|
| 线程、活跃值、SMEM 与 carveout 条件 | 11 | 资源 API、完整递推输出及 SMEM 内容 |
| 显式 local 数组 | 2 | 32/128 个 u32 的逐线程数组 |
| 寄存器压力与 spill | 2 | 每 kernel 请求 REG32 后的实际代码和结果 |
| 地址计算与类型转换 | 8 | 各操作的一条或四条数据流 |
| 原子更新 | 4 | global/shared，同址或各线程独立地址 |
| `setmaxnreg` | 1 | 初始 64 寄存器条件下 dec32/inc64 的两阶段完整输出 |

每配置启动一 CTA。入口 seed 为 3，实际三次启动的 `(iterations, internal_seed)` 是 `(1, 0)`、`(2, 3)`、`(5, 4294967295)`，没有执行校准或预热。`setmaxnreg` 只验证约定执行的合法性和结果，不进入正式服务采样。

## 实际设备与资源观察

本次节点为 `romeo-a047`，GPU UUID 为 `GPU-a97903c9-11d5-330a-0635-ec3931cbae6e`。程序查询到 `NVIDIA GH200 120GB`、132 SM、60 MiB L2、每 SM 65,536 个寄存器与 233,472 B SMEM；后续使用这些记录时必须保留设备和配置条件。

### SMEM 条件与 occupancy API

下面两点均使用 128 线程、每线程 48 个寄存器、静态 SMEM 为 0：

| 配置 | 动态 SMEM/CTA | occupancy API 上限 |
|---|---:|---:|
| `resource_threads128` | 0 B | 10 CTA/SM |
| `resource_smem128k` | 131,072 B | 1 CTA/SM |

这些值是函数资源条件下的 API 上限。本实验只启动一个 CTA，没有测出实际同时驻留 10 个 CTA。carveout 配置保存了请求值，实际分区仍记为 `null`，不能由请求值推定。

### local 与 spill

| 配置 | 实际寄存器/线程 | 函数 API 的 local 字节/线程 |
|---|---:|---:|
| `local_explicit_words32` | 14 | 128 B |
| `spill_pressure_words32` | 32 | 64 B |
| `spill_pressure_words128` | 32 | 576 B |

源码中的显式数组、编译器 spill 计数和实际 local 属性是不同记录。两个压力目标使用新的完整机器码，未沿用旧版本的数值资格；19 个 C++ 目标中其余 17 个机器码与之前审查的版本相同。

这 28 个配置的实际 API 静态 SMEM 中，两个 shared 原子配置为 512 B，其余为 0。编译阶段保留的 ptxas 与链接资源 dump 存在 1,024 B 差值；实际 API 是独立观察，没有把其中一份报告改写成另一份。

## 用原始字段手算两条记录

`local_explicit_words32` 的第三次启动使用 128 线程、每线程 32 个 u32、5 次迭代。按计时循环中每个元素一次 4 B 读和一次 4 B 写，其逻辑 local 请求量是：

\[
128\times32\times5\times(4+4)=163\,840\ \mathrm{B}.
\]

原始字段 `resource_identity.extensions.explicit_local_request_bytes[2]` 为 163,840。它不包含循环外的初始化和回读，也不是物理 HBM 流量。

`atomic_add_u32_global_same_word` 的第三次启动有 `128×5=640` 次线程级逻辑更新。每线程每轮增加 `1+(thread_id mod 7)`，每轮增量之和为 `18×28+1+2=507`，五轮合计增加 2,535。该次目标初值为 65,793，因此最终值应为 `65,793+2,535=68,328`；实际 `auxiliary_2_values.u32le` 中的单个 u32 为 68,328。

本路径的已审 SASS 存在 warp 聚合，原记录另外给出 20 次预期原生原子更新，即 `4 warp×5`。这是固定执行路径上的静态推导，不是硬件计数器观测。逻辑更新次数、数值增量和原生指令数量不能互换。

## 完整输出与计时边界

归档保存了 1,088,874 个 u32 元素，包括结果、输入、保护区、SMEM 和时间字段。27 个普通配置各有三组 12-word 时间记录，共 972 个 u32；其中包含 `clock64` 起止值和起止 SMID，以检查同 SM 时间域。`setmaxnreg` 保存各次迭代的两阶段输出，不从其事件时间推导通用寄存器重分配成本。

短测试的 CUDA event、globaltimer 和 `clock64` 记录用于检查执行与计时接口；尚无正式预热、独立进程重复统计或稳定性结论，因此此处不绘制性能图，也不导出标量服务速率。

## 证据与复现

- [本次来源包与 28 配置初始化记录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s18-short-deployment-a/package.json)
- [完整收集记录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s18-short-deployment-a/collection.json)
- [完整 28 配置的独立 B3 审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S18-early-validation-B3-review.json)
- [冻结源码审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S18-short-source-B-review-r2.json)
- [独立完整数值检查程序](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/evidence/S18-short-source-B/full_values.py)

完整归档 `job731482-complete.tar.gz` 为 94,626,688 B，SHA256 为 `40ad5ef79479d3dc05c4ce849e604af7556b293d3960a5bd211910d854395b3f`。各运行的冻结源码、编译记录、设备属性、原始数组与 manifest 位于收集目录的 `repo/results/gh200_resource_campaign/20261001-resource-suite-v2/auxiliary/`。

离线复核使用冻结记录和全部数组；重新运行应使用来源包中的命令、配置与工具链，在新的部署和运行目录执行。原始结果保存在被 Git 忽略的本地 `results/`，仅取得文档不等于取得原始归档。正式服务采样及相应图表、参数和稳定性结论仍待后续完成。
