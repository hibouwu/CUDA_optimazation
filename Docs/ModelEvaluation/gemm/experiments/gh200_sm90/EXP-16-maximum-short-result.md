# S16：TMA stage/request 最大点短验证

`gmem_to_smem_16kib_s1_r4_all_gpu` 已通过独立数值审查。该点用于先检查较大的数据与缓冲配置，尚未测量正式服务参数：没有校准、预热或性能采样，不能从进程耗时推导 TMA 带宽。

## 配置与实际资源

| 项目 | 原始记录 |
|---|---:|
| 每请求 payload | 16,384 B |
| 软件 stage 数 | 1 |
| 每次迭代的请求数 | 4 |
| 每 CTA 线程数 | 128 |
| CTA 数 | 396 |
| 实际设备 SM 数 | 132 |
| 四次独立 kernel 启动的迭代数 | 1、2、5、33 |
| 输入 seed | 3 |
| 编译及运行记录的每线程寄存器数 | 40 |
| 每 CTA 动态 SMEM | 65,576 B |
| 静态 SMEM / local 字节 | 0 / 0 |
| occupancy API 上限 | 3 CTA/SM |

396 个 CTA 来自本配置的网格规则：`132 × min(4, 3)`。occupancy API 的 3 是资源条件下的上限，不能据此声称 396 个 CTA 同时驻留。

## 从原始字段计算一次工作量

最后一次启动的 `iterations=33`，其有效 GMEM→SMEM 运输请求量为：

\[
396\times33\times4\times16384=856\,424\,448\ \text{B}.
\]

四次启动合计 41 次迭代，因此逻辑有效运输请求为 1,064,042,496 B。该量不包括验证用输出、保护区、计数和文件归档，也不等于物理 HBM 流量。完整原始证据较大，是因为实验还保存了用于检查输入、输出和缓冲复用的数组。

## 检查了什么

独立审查对 24 个数组、共 1,122,748,736 个 u32 元素进行完整重放，其中：

- 1,122,716,264 个元素参与 payload、保护区、计数、生命周期或时间整数条件检查。
- 32,472 个 opaque token 字只核完整保存和取值域，未猜测内部编码，也未计入精确数值匹配。
- 实际 SASS、18 个指令目标的完整编码、资源记录、进程清理状态和原始文件身份一并核对。

原目标进程耗时 29.857989885 秒，未超过该短验证的 30 秒上限；进程返回 0，清理状态确认。这个耗时包含 host 侧工作和完整输出处理，不能作为 kernel 服务时间。它接近上限，也不能证明后续配置一定有足够运行余量。

## 原始证据与复现入口

GPU 作业为 `731220`，CPU 无损封包作业为 `731228`。两地主机对同一包做了完整 EOF 和逐成员 SHA 检查：486 个成员、4,515,271,988 B，压缩包 146,656,164 B。

- [独立数值审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S16-max0-B3-review.json)
- [实际双主机收集记录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-max0-731220-collection-a/implementation-r2-actual-collection.json)
- [完整包及成员索引](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-max0-731220-collection-a/offhost-deployment/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/tma_stage_request_packed/v1/index0-job731220-v1/entry.json)
- [独立完整数值重放程序](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/evidence/s16_max0_full_values.py)

归档 SHA256 为 `846eb803602259cb8850ac9c45220f4f68b44993cf717fbb9527a48720d811dd`。原始结果遵守项目忽略规则，保存在本地 `results/`；仅取得 Git 文档不会自动取得这些原始文件。复现应沿用收集记录中的冻结源码、参数和验证命令，不以当前可变源码冒充原运行版本。

## 当前覆盖边界

本记录只授予上述一个配置的短正确性结论。其余 31 个合法点、4 个容量拒绝点、全家族 B、正式 C 和性能参数仍未完成。

后续单 CTA 批次 `731315` 在存储准入处形成检查点，四个候选点均未启动目标 kernel，不能计为新完成配置。此后 CPU 作业 `731400` 已回收完整保全后的 4,515,271,988 B 展开副本，双地压缩归档、来源文件和账本保留；[实际回收结果独立审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S16-retirement-v2-result-B-review.json)已通过。该操作解决的是存储阻碍，没有增加数值覆盖或性能参数；后续续跑前重新查询实际配额。
