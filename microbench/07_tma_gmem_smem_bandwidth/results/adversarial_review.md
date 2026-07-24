# TMA GMEM-to-SMEM 运行对抗式审查

通过：LTS bytes match requested TMA payload and DRAM mode misses L2 as expected.

## 已检查的问题

- SASS：`results/sass_summary.txt` 中 `tma_kernel` 包含 `UTMALDG.3D`，确认不是普通 `LDG/STG` 伪装成 TMA。
- descriptor：初版把 32 KiB tile 放进 `boxDim[0]`，违反 `boxDim[0] <= 256 elements`；已改成 `256 x rows x tile` 的 3D tensor map 后重跑。
- working set：初版 `dram-stream` 只有 256 MiB backing storage，20 CTA 的 timed 请求共 2.68 GiB，导致大量 L2 复用；已改成 dram mode 至少分配 `blocks * (warmup + timed iters)` 个唯一 tile 后重跑。
- NCU row：大 working set 下 `init_kernel` 比 `tma_kernel` 更久，初版 NCU parser 按最长 kernel 选错 row；已改为显式选择 `tma_kernel` 后重跑。
- NCU traffic：`TMA bytes / requested` 为 l2-hit `1.008`、dram-stream `1.008`；`LTS bytes / requested` 为 l2-hit `1.008`、dram-stream `1.008`；dram-stream 的 LTS miss-sector proxy / requested 为 `1.008`。
- NCU utilization：TMA read bytes 为 l2-hit `753.86 B/cycle`、`29.45% peak`，dram-stream `157.32 B/cycle`、`6.15% peak`；反推 TMA 模型上限约 `2.56 KiB/cycle/GPU`。LTS 反推模型上限约 `1.024 KiB/cycle/GPU`。

保留边界：这是 TMA ingress end-to-end throughput，不是纯 DRAM pin peak 或纯 shared write-port peak。
