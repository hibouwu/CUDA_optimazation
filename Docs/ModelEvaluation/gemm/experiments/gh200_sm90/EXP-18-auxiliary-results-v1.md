# GH200 驻留与辅助资源：27 个条件化测量结果

S18 在 job733338 的同一 GPU 分配内完成 27 个配置、每配置 10 个独立进程的正式采样。独立 C、原 ARM 解释器严格封存及换目录重放均通过；结果与四组图见[完整实验说明](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/job733338-results/published-v1/EXPERIMENT.md)、[27 点统计表](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/job733338-results/published-v1/TABLE.md)和[复现入口](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/job733338-results/published-v1/REPRODUCE.md)。

## 测量对象

每次启动一个 CTA，在同一物理 SM 上用 `clock64` 测完整循环。线程数、live 变量、SMEM 预留、carveout、local/spill、地址计算、转换和原子更新分别采用有限代表配置。`setmaxnreg` 只保留原合法性检查，不进入本性能表。

| 字段 | 定义 | 单位 |
|---|---|---|
| C | 约定开始至完成事件的同 SM 周期差 | clock64 cycle/CTA |
| N | 本分配校准并冻结的循环次数 | iteration |
| Q | 依配置线程和流数计算的逻辑操作请求量 | logical request |
| C/N | 完整循环的周期成本 | clock64 cycle/iteration/CTA |
| Q/C | 同一窗口内的逻辑请求服务率 | logical request/clock64 cycle/CTA |

这些指标包含合同规定的循环、同步和排空。occupancy 字段为 API 容量上限；carveout 是请求提示，实际分配仍未知。SMEM 预留点没有逐轮搬运同等字节，不能用其 Q/C 推导 SMEM 带宽。local 与 spill pressure 的循环不同，比较时应匹配实际资源及工作量。

## 从原始记录计算一个结果

`resource_threads64` 的 batch0/trial0：N=65536，C=5507377，Q=134217728。

- C/N = 5507377 / 65536 = 84.0359039307 cycle/iteration/CTA。
- Q/C = 134217728 / 5507377 = 24.3705357378 logical request/cycle/CTA。

这一窗口为 2782784 ns。周期与纳秒都来自原始记录，没有用固定频率互换。对应 raw 的 SHA 为 `5289bc401d168ba6da27313df13ec1a24185d25d8c2353a66406e669ccd429be`，准确路径和其余九次样本都在[条件参数文件](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/job733338-results/published-v1/qualified-parameters.json)的 `source_samples` 中。

## 接受条件与适用范围

27 点均第一批稳定；C/N 最大 CV 为 0.255918%，Q/C 最大 CV 为 0.255276%，都小于原 5% 判据。计时窗口为 0.927104–20.143168 ms，30/270 个样本达到 20 ms 软目标；软目标未达到的样本按原硬窗口和校准上限审查，没有删掉后只留较长样本。低 CV 只说明这批采样的重复性。

设备为 GH200/CC9.0、132 SM，job733338/romeo-a041，UUID `GPU-565e9222-0d4c-d7b1-4908-73a9d422a257`，CUDA12.9。参数用于这些输入、二进制、资源和单 CTA 完整循环条件，不外推整卡吞吐或完整 GEMM。NCU 为 `permission_denied`，因此 Q 仍按合同解释为逻辑请求。

## 结果与证据

- [四组图与观察](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/job733338-results/published-v1/EXPERIMENT.md)：线程与资源、local/spill、辅助操作、原子争用；每组提供 PNG 和 SVG。
- [全精度条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/job733338-results/published-v1/qualified-parameters.json)：27 点及270条 raw/receipt 来源，保留两个指标和资源条件。
- [最终发布独审](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/reviews/published-v1-independent-review.json)：绑定最终图文和参数。
- [最终交接索引](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/S18-FINAL-HANDOFF.md)：独立 B/C、COMPLETE、原 ARM 严格复算、封存及迁移重放的准确位置。

原发布 manifest 的待独审标签是签署前保全状态；当前资格由之后的发布独审与 C/COMPLETE 共同确认。原发布目录、summary 和 measurement manifest 保持不变。
