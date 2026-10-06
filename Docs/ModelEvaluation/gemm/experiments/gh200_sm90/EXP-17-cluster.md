# EXP-17：cluster、DSM 与 TMA 多播

[结果总览](README.md)

## 结论

- 当前设备支持 cluster 大小 2/4/8，occupancy 查询给出的整卡 cluster 数分别为 528、248、124（CTA 数 1056、992、992）。
- **DSM 比本地 SMEM 慢得多**：整卡 DSM 读 2.2–2.5 TB/s，本地 SMEM 读 10–12 TB/s，约为 1/5；DSM 写 2.3–2.6 TB/s，本地写 5.3–6.1 TB/s。单 cluster 下 DSM 读约为本地的一半。本地和 DSM 的窗口都含消费、读回或循环控制，不是端口带宽。
- **TMA 多播**：整卡全目标多播的逻辑源请求速率为 2528 / 1193 / 631 GB/s（C=2/4/8），乘以接收 CTA 数后接收量为 5.06 / 4.77 / 5.05 TB/s，高于单播读的 3.8 TB/s。已测的是源请求与接收 payload 的关系；L2/HBM 的物理读取量没有计数器证据。
- cluster 同步：单 cluster 每个完整同步阶段约 190–195 ns（按 1.81 GHz 约 350 cycle），与 cluster 大小基本无关；这是阶段吞吐的倒数，不是裸 barrier 延迟。

## 配置

每 CTA 128 线程；DSM/本地读写每次 4096 B；TMA bulk 每次 16 KiB 源请求，单目标或 cluster 内全部 CTA 为目标。单 cluster 与整卡 cluster grid 各测，计时用 globaltimer。

## 结果（中位数）

整卡 cluster grid，GB/s：

| 形式 | C=2 | C=4 | C=8 |
|---|---:|---:|---:|
| 本地 SMEM 读 | 12015 | 10630 | 10151 |
| DSM 读 | 2524 | 2170 | 2200 |
| 本地 SMEM 写（含读回） | 6131 | 5567 | 5299 |
| DSM 写（含读回） | 2631 | 2308 | 2277 |
| TMA 单目标（源请求） | 2931 | 1566 | 825 |
| TMA 全目标多播（源请求） | 2528 | 1193 | 631 |
| 多播送达量（源 × C） | 5056 | 4771 | 5050 |

单 cluster，GB/s：

| 形式 | C=2 | C=4 | C=8 |
|---|---:|---:|---:|
| 本地读 / DSM 读 | 29.8 / 15.4 | 59.0 / 30.1 | 117.1 / 60.3 |
| 本地写 / DSM 写 | 17.9 / 14.1 | 35.5 / 27.2 | 70.2 / 54.0 |
| TMA 单目标 / 全目标（源请求） | 15.4 / 13.7 | 15.4 / 13.7 | 15.5 / 13.7 |
| cluster 同步（phase/ns） | 0.00522 | 0.00516 | 0.00513 |

![cluster grid](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s17-main-execution/actual-report-v1/cluster_grid.png)

## 限制

单目标 TMA 每 cluster 只有一个源请求，整卡速率随 cluster 数下降，不代表多播本身变慢。写实验窗口内含读回校验，不能与纯 store 比较。

## 数据

[results.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s17-main-execution/actual-report-v1/results.csv)、[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s17-main-execution/actual-report-v1/samples.csv)、[单 cluster 图](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s17-main-execution/actual-report-v1/one_cluster.png)。脚本：[families/s17](../../../../../microbench/gh200_resource_campaign/families/s17/README.md)。
