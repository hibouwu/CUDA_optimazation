# SMEM bank/stride 运行对抗式审查

结论：通过。

本实验可以作为 local scalar SMEM bank-conflict bandwidth 数据使用；它不能被外推为
vectorized shared path、DSMEM 或 TMEM 的原始端口峰值。

## 证据

- SASS 检查通过：read kernel 包含 unrolled `LDS`，write kernel 包含 unrolled
  `STS`；初始化和计时辅助指令不作为带宽结论证据。
- app 端 3 次重复稳定：read 从 stride 1 的 `1056.451 B/cycle/GPU`
  下降到 stride 32 的 `79.994 B/cycle/GPU`；write 从 stride 1 的
  `821.227 B/cycle/GPU` 下降到 stride 32 的 `79.999 B/cycle/GPU`。
- NCU shared wavefront/bank-conflict 证据匹配预期：stride 1 bank conflicts 为
  `0`；stride 32 bank conflicts 为 `660275200`。高冲突 stride 的 shared
  wavefront `%peak` 接近 `100%`，说明瓶颈是 bank conflict 展开的 shared LSU
  wavefront，而不是 app 计时或 dead-code。
- conflict-free scalar local SMEM 的 payload-normalized rough peak 约为
  `2.57 KiB/cycle/GPU`：read stride 1 为 `2572.3 B/cycle/GPU`，write
  stride 1 为 `2567.1 B/cycle/GPU`。这是用 app payload B/cycle 除以 NCU
  shared wavefront `%peak` 得到的估计，不是 direct byte-counter peak。

## 对抗性检查

- 如果编译器消除了 shared access，NCU wavefront 和 SASS `LDS/STS` 不会同时出现；
  当前二者都存在。
- 如果 stride sweep 没有制造 bank conflict，NCU bank conflicts 不应随 stride
  放大；当前从 stride 1 的 `0` 增至 stride 32 的 `660275200`。
- 如果只是 clock/occupancy artifact，stride 4/8/16/32 不应在 read/write 两个方向
  都按冲突阶数接近成比例下降；当前 read/write 高 stride 都约为
  `640/320/160/80 B/cycle/GPU`。
- stride 2 的 app 吞吐没有明显下降，尤其 write 与 stride 1 基本相同；这不推翻
  实验结论，因为 NCU 已显示 wavefront 翻倍，但该冲突程度还没有成为 app 端主要瓶颈。

## 保留边界

- 本机 NCU 没有 local shared direct byte counter；报告使用
  `l1tex__data_pipe_lsu_wavefronts_mem_shared*` 和
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared*`。
- 结果代表 32-bit scalar `ld.shared.u32` / `st.shared.u32` 的 bank/stride 行为；
  不能直接当作 128-bit vector shared、TMA shared ingress、DSMEM 或 tensor-core
  operand path 的带宽。
