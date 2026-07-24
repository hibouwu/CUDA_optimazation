# DSMEM 设计对抗式审查

## 目标

测 local shared 与同 cluster 内 remote DSMEM 的 read/write 请求吞吐。每个线程每次发 16 B `uint4` 请求，报告全 GPU 请求字节 / 最大 CTA `clock64()` 周期。

## 主要攻击点与设计响应

1. **多 wave launch 会高估吞吐。**

   设计响应：默认 `--clusters 0` 使用保守的一 CTA/SM 规则，即 `SM_count / cluster_size`；超过该值会被拒绝，除非显式 `--allow-waves`。CUDA 返回的 `cudaOccupancyMaxActiveClusters` 只记录，不作为默认，因为 remote DSMEM 在多 CTA/SM 下会进入多 wave 或失败区。

2. **remote 访问可能退化成本地 shared 访问。**

   设计响应：remote 模式强制 `cluster_size >= 2`，并用 `cluster.map_shared_rank(smem, (rank + 1) % cluster_size)` 映射邻居 CTA 的 dynamic shared memory。运行后必须用 NCU `mem_dshared` byte counters 验证。

3. **shared load/store 可能被 scalarize，导致 16 B/op 统计不干净。**

   设计响应：第一版使用 volatile `uint4` 访问并保留 SASS 摘要；运行审查必须确认是否存在宽 LDS/STS 或解释 scalarization。若 SASS 不满足 128-bit 请求假设，需要改 PTX 或重定义 bytes/op 后重跑。

4. **写入路径没有等完成就停表。**

   设计响应：`local-write` 在 stop 前 `__syncthreads()`；`remote-write` 在 stop 前 `cluster.sync()`，因此报告的是包含 completion boundary 的端到端 store-path throughput。

5. **local shared 和 remote DSMEM 的 NCU 计数能力不同。**

   设计响应：remote DSMEM 使用 `l1tex__t_bytes_pipe_lsu_mem_dshared*` 计算利用率和上限；local shared 若本机没有 byte counter，用 `l1tex__data_pipe_lsu_wavefronts_mem_shared*` 和 bank conflict 作为利用率 proxy，不写成 NCU byte-verified bandwidth。

## 设计审查结论

设计可以进入第一轮运行，但运行审查必须检查 SASS 宽度、NCU dshared bytes/expected、occupancy、bank conflicts 和 write completion 边界。
