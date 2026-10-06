# EXP-14：TMA 1D bulk

[结果总览](README.md)

## 结论

- 单 CTA、一次只有一个请求在途时，速率由往返时间决定：GMEM→SMEM 从 1 KiB 的 1.8 B/cycle 增到 64 KiB 的 31.6 B/cycle。按 \(q/(\text{速率})\) 估算单请求完成时间，1 KiB 约 565 cycle，64 KiB 约 2080 cycle，两点连线近似为"约 540 cycle 固定项 + 约 43 B/cycle"。这是包含 mbarrier 等待和 CTA 同步的整轮周期，只能作为单请求往返时间的上界，不是 TMA 的裸延迟或带宽。
- 整卡 GMEM→SMEM 在 payload ≥8 KiB 时达 3.77–3.83 TB/s，与 global 读相当；1 KiB 只有 1.77 TB/s，4 KiB 为 2.94 TB/s。
- **整卡 SMEM→GMEM（bulk 写）只有 2.54–2.74 TB/s**，比 `st.global` 写（3.79 TB/s）和 2D tensor 写（[EXP-15](EXP-15-tma-2d.md)，3.8 TB/s）低约 30%。epilogue 若用 1D bulk store 需单独计。
- 单 CTA 写在小 payload 时快于读（1 KiB：3.8 vs 1.8 B/cycle），大 payload 时慢于读（64 KiB：17.3 vs 31.6）。

## 配置

无 tensor-map 的 1D bulk；每 CTA 128 线程、一个槽，每轮发出一个请求并等完成（读：mbarrier；写：`wait_group 0`），再 CTA 同步。payload 1–64 KiB；整卡 528 CTA（64 KiB 时 396）。

## 结果（中位数）

| payload KiB | 读 单 CTA B/cycle | 读 整卡 GB/s | 写 单 CTA B/cycle | 写 整卡 GB/s |
|---:|---:|---:|---:|---:|
| 1 | 1.81 | 1770 | 3.78 | 2536 |
| 4 | 6.27 | 2941 | 9.33 | 2607 |
| 8 | 10.77 | 3766 | 12.36 | 2708 |
| 16 | 17.39 | 3831 | 14.75 | 2674 |
| 32 | 25.50 | 3797 | 16.33 | 2690 |
| 64 | 31.56 | 3771 | 17.25 | 2739 |

最大 CV 1.2%。

![整卡读](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/analysis/s14-formal-733392-qualified/all_gpu-gmem_to_smem.png)

## 数据

[cases.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/analysis/s14-formal-733392-qualified/cases.csv)、[参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/analysis/s14-formal-733392-qualified/parameter-candidates.json)、[复现说明](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/analysis/s14-formal-733392-qualified/REPRODUCE.md)。
