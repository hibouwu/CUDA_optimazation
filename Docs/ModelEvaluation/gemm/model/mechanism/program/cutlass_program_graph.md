# CUTLASS cooperative / pingpong 程序图

范围：现有 cfg_a/b/c、FP16→FP32、α=1、β=0、固定 persistent 调度。cfg_a 为 128×128×64 cooperative、cluster 2×1、S6；cfg_b 同 tile、pingpong、cluster 1×1、S6；cfg_c 为 256×128×64 cooperative、cluster 1×2、S4。构建查询值优先于模板名。

本页维护软件操作和依赖，已核对下列归档源码与现有 Python 递推。**尚未形成独立图导出器，也未完成每个配置的整图 SASS/trace 对照验收。** 源码模板中的其他调度分支不自动纳入支持范围。

## 1. 来源与核对边界

| 源码快照 | SHA256 |
|---|---|
| [cooperative kernel](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-formal-v2/source/cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp) | `8f09c79f5e52fffd3a2fb0e7810727dc9480d336452f69dec2d11cce3ca0fe62` |
| [pingpong kernel](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-formal-v2/source/cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp) | `83a8e63dd9e7f70a735c17fbb779782fc31ce5215509ed644d13e68428438dee` |
| [TMA/WGMMA collective](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-formal-v2/source/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp) | `b78d838244bbc71e915cf0ec7efb353d0cdda2826de945a344c558923c90f375` |
| [PipelineState](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-formal-v2/source/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp) | `7c61d847933cb12a14fcdc68cec902b44c802c9eb34273218c29e601f9c53dee` |

上述快照用于说明控制与状态语义；目标模型冻结时还须绑定其实际编译源码/二进制，不能由快照相同推定所有 V 轮二进制相同。现有数值实现为[v06_model](../../../../../../microbench/gh200_resource_campaign/access_rules/v06_model.py)、[v08_model](../../../../../../microbench/gh200_resource_campaign/access_rules/v08_model.py)和[v09_model](../../../../../../microbench/gh200_resource_campaign/access_rules/v09_model.py)。

## 2. 对象、操作与依赖

图索引保留 CTA/cluster rank、warpgroup、输出 tile j、Ktile、操作数 A/B、槽 index、代次 generation 和请求身份。实际 tile 顺序由 `v08_model.scheduled_work` 给出，包含 swizzle、raster、整 cluster 补齐及 OOB；数学输出面积不替代执行列表。

| 操作/事件 | 源与目的 | 必要软件前驱或后继 | 来源 |
|---|---|---|---|
| `producer_acquire` | 等待输入 SMEM 槽可重用 | 同槽上一代的消费者释放；首代按 empty 初态 | collective `load` / pipeline |
| 发 A/B TMA 请求 | 本 GPU global 分配 → 目标 CTA 本地 SMEM；cluster 按 multicast mask 投递 | acquire 完成、descriptor 与事务计数就绪 | collective `load` |
| full / `consumer_wait` 返回 | 事务完成使目标输入满足消费条件 | 对应槽、相位与事务计数满足；等待返回另有控制执行 | pipeline / collective `mma` |
| WGMMA 发出、commit、wait | A/B 在 SMEM（本研究的 SS 路径）→ 寄存器累加器 | 对应输入 ready、proxy/fence 与 group 依赖 | collective `mma` |
| `consumer_release` | 槽可供后续 producer 再用 | 使用该槽的异步计算满足源码规定的完成等待 | `warpgroup_wait<K_PIPE_MMAS>` 后释放；尾部 `mma_tail` 排空 |
| 输出准备与 TMA store | 累加器寄存器 → epilogue SMEM → global D | 本输出计算完成、epilogue 顺序/槽条件满足 | kernel 调用 collective epilogue |
| 输出源复用/末尾排空 | epilogue SMEM 可复用；目标完成由实际等待语义定义 | store pipeline 的对应等待 | `store` / `store_tail`；完整 D 完成仍由完整调用验证 |

TMA 完成如何耗时、并发请求如何仲裁、WGMMA 的持续服务率不是上述边的内容，接口留给[硬件规则](../hardware_rules/README.md)。cluster full/empty 跨 rank 依赖还需按实际 mask 和 barrier 到达者展开，已有核对证据见[R18](../../../experiments/gh200_sm90/access_rules/R18-cluster-boundary.md)。

## 3. 槽代次与两种调度

令 g 为同一输入 pipeline 的逻辑 Ktile 序号，S 为其 stage 数。初始 index 为 0 时：

```text
slot(g) = g % S
generation(g) = floor(g / S)
consumer_phase(g) = generation(g) % 2
producer_phase(g) = 1 xor (generation(g) % 2)
release(g) -> acquire(g + S)
```

这里 phase 是 `PipelineState` 传给等待操作的状态值；producer 初相位为 1、consumer 为 0。两者用于不同 full/empty barrier，不能把这个位当成统一的“数据已就绪位”。相同槽/相位还会再次出现，所以事件身份保留完整代次。producer/consumer 本地状态可能通过 `advance` 跳过逻辑工作，不能用线程执行了几次循环直接代替 g。

- **cooperative：** 两个数学 warpgroup 协作同一输出；消费者状态随已处理 Ktile 推进。该输出的观察边界常需合并两个角色。下一输出的数学工作与前一输出 epilogue 有软件顺序约束；producer 可受空槽规则提前供给，不能跟着合并的主循环窗口一并串行化。
- **pingpong：** 两个数学 warpgroup 交替处理输出 j；一组计算下一输出时，另一组可以执行 epilogue。第二组起始 consumer 状态前进一个输出的 Ktile 数，每组完成后 `advance(k_tile_count * NumMmaWarpGroups)`；不是每个输出都把输入槽归零。
- **输入与输出有不同 pipeline：** epilogue 的 load/store 状态、槽数和每输出推进次数独立；不能套用 mainloop 的 g、S。β=0 不需要读 C 的数学需求，但仍以实际实例化分支核对 epilogue 控制。

pingpong kernel 的明确顺序为：`math_wg_order_barrier.wait → mma → arrive → mma_tail → wait → store → store_tail → arrive`。第一次 arrive 允许下一组进入 MMA，早于当前组 `mma_tail`；输入槽释放仍由 collective 的异步 wait 控制。不能用“上一输出完全结束”替代这个先行关系。

<a id="recurrence"></a>

## 4. 现有半经验递推与图的对应

令 `fm/me/ep/ed` 分别为模型的主循环起点/终点、epilogue 起点/终点，j 为同 CTA 的输出序号。它们是现有观察协议的聚合事件，不等于一条指令的 issue/complete。

| 调度 | `v08_model.cta_cycles` 的现有关系 | 依赖所表达的内容 |
|---|---|---|
| 两者首输出 | `fm[0]=P0+S_wait`，`me[j]=fm[j]+L[j]` | 首段准备和主循环；S_wait 是拟合等待，不是 stage 数 |
| cooperative | `fm[j]=ed[j−1]+h`；`ep[j]=me[j]+w`；`ed[j]=ep[j]+E[j]` | 上一输出后本组继续下一输出；两个角色边界已经合并 |
| pingpong 主循环 | `fm[j]=max(me[j−1]+gm, ed[j−2]+h)` | 前一数学组交接，以及同组前一输出结束；第二项从 j=2 起 |
| pingpong 输出 | `ep[j]=max(me[j]+w, ed[j−1]+we)` | 本输出主循环完成及前一输出的 epilogue 顺序；第二项从 j=1 起 |
| pingpong 输出推进 | `ed[j]=epilogue_end(ep[j], next_mainloop)` | 当前采用经验重叠服务，不能由 DAG 边推导比例 r |
| 完成 | `max(ed)+Etail`，实现还支持 `x0+x1*ntiles+xk*base` | 排空和尾差都是该版本的时间假设 |

V09 以 ns 供给 L/E 及事件间隔；函数名 `cta_cycles` 不保证所有调用单位为 cycle。V06/V08 的原调用按各自冻结版本读取。频率求根、E 的 overlap 比例、供给价格、入口分散近似和 event 映射统一归[半经验模型](../../semi_empirical/MODEL-CANDIDATES.md)，不进入无时间价格的程序图。

R14 的合并事件和 V06/V08/V09 的 trace 对照用于核对聚合边界；未来细图不能直接把 `me` 当成 kernel 源码的第一次 order-barrier arrive，必须绑定实际打点位置。当前已有[TMA 探针图](../hardware_rules/TMA-INPUT-EVENT-MODEL.md)使用另一套消费控制，只能复用槽身份接口，不能替代 CUTLASS 图。

## 5. 给机制组合的交付与缺口

交付字段为静态工作列表、操作节点、必要前驱、请求的源/目的/字节/描述符、槽代次、参与角色和观察端点。硬件规则以版本号提供 eligible→issue→complete→visible/release 的服务；程序图不夹带实测 L/E 或目标频率。

下一步把上述映射输出为可检查的静态图，对 cooperative/pingpong 各一个实际配置核对请求数、代次、order barrier、尾部释放和 SASS/trace 端点；再扩展 cluster/padding 和 stage 干预。此后才能声明程序图完成验收。机制组合尚未实现，不以本页完成替代实机验证。
