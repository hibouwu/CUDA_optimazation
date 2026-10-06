# EXP-16：TMA 多请求与多 stage

[结果总览](README.md)

## 结论

- 单 CTA 读速率随缓冲容量 \(S\times R\times16\) KiB 增加：16 KiB 时 14.7 B/cycle，32 KiB 约 21，64 KiB 为 28.5–39.8，128 KiB 为 37.4–42.3。
- S1R1 每轮只有一个 16 KiB 请求，完整周期为 \(16384/14.75\approx1110\) cycle，包含发出、完成等待和 CTA 同步，是单请求往返时间的**上界**。更大的 S、R 只给出缓冲容量，没有测平均在途量，不能用 Little 定律直接换算成延迟。
- 推断：按 128×256×64 的 BF16 tile 估算，满速 WGMMA 每 SM 每 Ktile 约 1024 cycle、需输入 48 KiB，即约 48 B/cycle，高于本实验单 CTA 的最高 42 B/cycle；单个 CTA 的 TMA 供给可能不足，需要多 CTA/SM、多播或更深流水。
- 同样 64 KiB 缓冲容量下，组织方式影响速率：S1R4 为 28.5 B/cycle，S2R2 为 39.8。
- 单 CTA 写最高约 18.2 B/cycle（S2R2、S2R4、S4R2 相同），不到读的一半。
- 整卡读约 3.5–3.8 TB/s，写约 2.6–2.8 TB/s，与 S、R 基本无关，说明整卡时瓶颈已不在单 CTA 的缓冲组织上。写的上限与 1D bulk 写（[EXP-14](EXP-14-tma-1d.md)）相同。
- S4R4 需 256 KiB SMEM，超出单 CTA 上限，启动前被拒绝。

## 配置

1D bulk，每请求 16 KiB；S 为 stage 数，R 为每 stage 的请求数，各取 1/2/4。每 CTA 128 线程，单 CTA 与整卡各测。

- 运输量：\(B\times N\times R\times16384\) B（B 为 CTA 数，N 为循环次数），**不乘 S**；S 只改变缓冲与并发组织。
- 读（GMEM→SMEM）：issuer 等对应 mbarrier 完成，再经 CTA 同步发布；槽在该 item 退役、消费者离开后复用。正式采样不做逐 item 数据读回，只保留完成与槽释放同步。
- 写（SMEM→GMEM）：退役使用完整 bulk-group 等待（不只等源侧读取），结束前 `cp.async.bulk.wait_group 0` 排空全部输出，排空在停止计时之前。
- 计时后的 SMEM 导出不计入运输量和窗口。

## 结果（中位数）

单 CTA，B/cycle（读 / 写）：

| S \ R | 1 | 2 | 4 |
|---:|---|---|---|
| 1 | 14.7 / 9.2 | 21.4 / 12.5 | 28.5 / 14.7 |
| 2 | 21.1 / 14.8 | 39.8 / 18.2 | 42.3 / 18.2 |
| 4 | 19.7 / 14.9 | 37.4 / 18.2 | 超出 SMEM |

整卡，GB/s（读 / 写）：

| S \ R | 1 | 2 | 4 |
|---:|---|---|---|
| 1 | 3835 / 2629 | 3706 / 2685 | 3695 / 2705 |
| 2 | 3687 / 2635 | 3664 / 2639 | 3765 / 2765 |
| 4 | 3601 / 2636 | 3513 / 2709 | 超出 SMEM |

![单 CTA](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/actual-report-v2/one_cta.png)

## 限制

没有单独测 TMA 往返延迟；软件 stage 数和缓冲容量都不等于硬件队列深度。

## 数据

[results.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/actual-report-v2/results.csv)、[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/actual-report-v2/samples.csv)、[整卡图](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/actual-report-v2/all_gpu.png)。脚本：[families/s16](../../../../../microbench/gh200_resource_campaign/families/s16/README.md)。
