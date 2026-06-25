1.2.2 persistent kernels 静态调度

也叫 Persistent clusters 或 Workers，是一种软件调度方式。

基本做法：划分好 work tile 后，只启动固定数量的 CTA。常见配置是 CTA 数量接近 SM 数量，每个 CTA 作为一个 worker，在 kernel 生命周期内连续处理多个 work tile。

历史上 `tc2p2/tc2p3` 做过 persistent WMMA 对照实验：只改变 C tile 的分发方式，从普通 2D grid 改成 CTA grid-stride 静态调度。它没有做到完整 warp specialization，CTA 内 8 个 warp 仍然都参与 WMMA 计算。

这组实验已经从当前 benchmark 删除。原因是它只是 `tc2` 的调度变体，不是新的 Tensor Core 指令路线；继续保留会干扰 `tc3/tc4` 的 SM120/SM110 bring-up 边界。

缺点：静态分配容易造成负载不均衡。某些 SM 可能因为 IO 阻塞、共享资源竞争、被其他任务干扰或局部调度变慢而拖尾，但它仍然要完成预先分配的 work tile；其他提前完成的 SM 无法动态偷取剩余工作。

1.2.3 Tensor Core 版本边界

当前版本划分如下：

| 版本 | 目标 | 当前定位 |
| --- | --- | --- |
| `tc1` | WMMA FP16 baseline | 最小 Tensor Core 正确性基线。 |
| `tc2` | TMA + WMMA staged mainloop | 当前性能基线，使用 TMA 把 A/B 从 GMEM 搬到 SMEM，再用 WMMA 消费 SMEM。 |
| `tc3` | SM120a FP8 MMA GEMM | 真实 FP8 e4m3 GEMM bring-up：CTA-level 128x64x32 tile，A/B 使用 TMA 2-stage pipeline 搬到 SMEM，输出 FP32，并输出 GFLOPS。 |
| `tc4` | SM120 mainloop 重构 | producer / consumer warp-specialized pipeline，整合 TMA pipeline、SM120 narrow/block-scaled MMA 和 epilogue。 |

`tc3` 和 `tc4` 的关系：

- `tc3` 先证明 SM120a narrow MMA 的真实 GEMM 路径：FP8 e4m3 A/B -> `mma.sync.aligned.kind::f8f6f4` -> FP32 C。
- `tc3` 不做 TMA、不做完整 warp specialization、不做 Scheduler/CLC、不做 Epilogue Load warp，也不做 TMA store D。
- `tc4` 在 `tc3` 的 GEMM 路径确认后，再重构完整 SM120 mainloop，把 producer/consumer warp、TMA pipeline、narrow/block-scaled MMA 和 epilogue 组织起来。

注意：`tcgen05/TMEM` 属于 SM100/SM110 这类 datacenter Blackwell family-specific 目标；RTX 5070 Laptop GPU 报告为 `sm_120`，对应 CUTLASS 文档里的 SM120 GEMM 路线，应该使用 `mma.sync.aligned.kind::f8f6f4`、`mma.sync.aligned.kind::mxf8f6f4.block_scale`、`mma.sync.aligned.kind::mxf4.block_scale` 等 PTX MMA 指令，而不是 `tcgen05.mma`。

1.3 warp 类别 & pipeline

下表是 `tc4` 的目标 warp-specialized 设计。它把一个 CTA 内的 warp 划分为不同角色，让不同 warp 通过 pipeline/barrier 协作处理同一个 work tile，并让 mainloop 与 epilogue 尽量重叠。

| Warp 编号 | 类别 | 线程数 | 说明 |
| --- | --- | ---: | --- |
| 0 | MMA | 32 | 执行 SM120 narrow/block-scaled `mma.sync`，消费 SMEM 中已经 ready 的 A/B tile，并把 accumulator 保存在寄存器。 |
| 1 | Scheduler | 32 | 发起 CLC 查询或软件 work queue 查询，获取下一个 work tile 坐标，并把结果广播给 CTA 内其他 warp。 |
| 2 | Mainloop Load | 32 | 负责 mainloop 的 TMA load，把 A/B/SFA/SFB 从 GMEM 搬到 SMEM，对应 `MainloopPipeline` 的 producer。 |
| 3 | Epilogue Load | 32 | 负责 epilogue 相关的 TMA load，例如 beta*C 或 scale/metadata 数据，为后处理准备输入。 |
| 4-7 | Epilogue | 128 | 消费 MMA warp/group 交接的寄存器 accumulator，执行后处理、量化或写回前的数据整理，并配合 TMA store 写回 GMEM。 |

当前 `tc2` 的实际 warp 分工更简单：

| Warp 编号 | 类别 | 线程数 | 当前实现 |
| --- | --- | ---: | --- |
| 0-7 | WMMA compute | 256 | 每个 warp 计算一个 32x32 C 子块。8 个 warp 合作完成一个 128x64 CTA tile。 |
| thread 0 | TMA issuer | 1 | 在每个 CTA 内由 thread 0 发起 A/B 的 TMA copy，并通过 `cuda::barrier` 等待 TMA 完成。 |
| 全 CTA | CTA tile worker | 256 | 普通 2D grid，一个 CTA 处理一个 128x64 output tile。 |

pipeline 关系如下：

| name | producer | consumer | 说明 |
| --- | --- | --- | --- |
| MainloopPipeline | Mainloop Load | MMA | 管理 A/B/SFA/SFB 从 GMEM 到 SMEM 的 TMA load 与 MMA 消费之间的同步。producer 在 TMA load 完成后 arrive；consumer 等待数据 ready 后执行 MMA。当前 `tc2` 代码中对应 `tc2_launch_a_tile_tma`、`tc2_launch_b_tile_tma` 和 `tc2_wait_tma_stage`。 |
| CLCPipeline | Scheduler | All warp（包括 scheduler） | 管理 work tile 查询响应。Scheduler 发起查询，硬件或软件队列返回下一个 work tile 坐标，所有 consumer warp 独立调用 `fetch_next_work` 或读取广播结果。当前代码没有 CLC 硬件查询。 |
| CLCThrottlePipeline | Mainloop 启动阶段 | CLCPipeline | 速度控制。防止 Scheduler 查询 work tile 的速度超过 Mainloop Load 消费 work tile 的速度。处理每个 work tile 时先 wait，完成后 commit，避免工作负载过度倾斜。当前代码没有单独 throttle pipeline。 |
| AccumulatorPipeline | MMA | Epilogue | 管理 accumulator 从 MMA 到 epilogue 的交接。SM120 路线没有 `tcgen05` 的 TMEM accumulator，`tc4` 应围绕寄存器 accumulator、warp group 调度和 epilogue store 组织。当前 WMMA 版本 accumulator 保存在每个 warp 的寄存器里，MMA 与 store 没有拆成不同 warp。 |
| EpiStorePipeline | Epilogue | TMA store | 管理 D 矩阵从 SMEM 到 GMEM 的 TMA store。Epilogue warp 把后处理结果写入 SMEM 后，由 TMA 异步写回 GMEM，并通过 scoreboarding/barrier 同步。当前 `tc2` 使用 `wmma::store_matrix_sync` 直接写 GMEM，没有 TMA store。 |
| LoadOrderPipeline | Mainloop Load | Epilogue Load | 确保 mainloop 的前几个 prologue stage 完成后，Epilogue Load 才开始工作。目的是让 A/B/SFA/SFB 先占用 TMA 带宽喂饱 MMA，再让 epilogue load 介入，避免 load 顺序破坏 mainloop 吞吐。当前代码没有 epilogue load。 |

1.4 `tc3` SM120a FP8 GEMM

`tc3` 现在是一个真实的 SM120a FP8 GEMM correctness/perf bring-up：

1. A/B 输入使用 `__nv_fp8_e4m3`，输出 C 使用 FP32。
2. 一个 CTA 计算一个 128x64 C tile，K 维按 32 分块累加。
3. 每个 CTA 使用 8 个 warp；每个 warp 负责一个 16-row band，并顺序覆盖 8 个 8-column subtile。
4. 每个 K tile 用 `CUtensorMap + cp_async_bulk_tensor_2d_global_to_shared` 把 A[128x32] 和 B[32x64] 搬到 shared memory。
5. mainloop 使用 2-stage ping-pong buffer：计算当前 stage 时，下一 K tile 预取到另一 stage。
6. MMA 指令使用 `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32`。
7. GFLOPS 按完整 GEMM workload `2*M*N*K / time` 计算。
8. correctness 使用 sampled CPU FP8 reference：抽样若干 C 元素，用同样的 FP8 e4m3 输入反量化后做 CPU dot product 对比。

它仍然不是最终优化版：当前 `tc3` 已有 CTA tiling、TMA、2-stage buffering 和 SMEM staging，但还没有 SMEM swizzle、warp-specialized producer/consumer 组织或更完整的 epilogue pipeline。后续 `tc4` 再把这些部分系统化。

构建限制：CUDA 13.0 中，ptxas 对 plain `-arch=sm_120` 会拒绝 `.kind::f8f6f4`；同一条指令在 `compute_120a/sm_120a` family-specific 目标下可以通过。因此当前脚本需要显式指定：

```bash
CUDA_ARCH=120a ./scripts/run_gemm_backend.sh tc3
```

如果用默认 `sm_120` 构建，`tc3` 会明确 skip，避免把不被 ptxas 接受的 PTX 编进默认 benchmark。

1.5 `tc4` Blackwell mainloop 重构

`tc4` 是完整 Blackwell 组织方式：

- Scheduler warp 负责 work tile 获取和广播。
- Mainloop Load warp 负责 TMA load A/B/SFA/SFB。
- MMA warp 负责 SM120 narrow/block-scaled `mma.sync`，把 accumulator 保存在寄存器。
- Epilogue Load warp 负责 TMA load C 或后处理输入。
- Epilogue warps 消费寄存器 accumulator，执行 `alpha/beta`、后处理或量化。
- TMA store 把 D 从 SMEM 异步写回 GMEM。

`tc4` 的关键不是简单增加 stage 数，而是把“加载、计算、后处理、写回”拆成不同 warp 角色，用 pipeline/barrier 管理依赖，让 mainloop 与 epilogue 的延迟重叠起来。
