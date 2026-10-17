# CUTLASS cooperative / pingpong 程序图

范围：现有 cfg_a/b/c、FP16→FP32、α=1、β=0、固定 persistent 调度。cfg_a 为 128×128×64 cooperative、cluster 2×1、S4/S6；cfg_b 同 tile、pingpong、cluster 1×1、S6；cfg_c 为 256×128×64 cooperative、cluster 1×2、S4。构建查询值优先于模板名。

本页维护软件操作和依赖，已核对下列归档源码与现有 Python 递推。**cfg_b S6/S4 已有 CPU 程序实例和独立图导出，完成冻结请求核对与历史工作映射对照；cfg_a S4/S6和cfg_c S4的cluster整图已导出，并与指定历史条件的工作和角色端点核对；其余配置的逐指令对照尚未完成。** 源码模板中的其他调度分支不自动纳入支持范围。

## 1. 来源与核对边界

| 源码快照 | SHA256 |
|---|---|
| [cooperative kernel](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-formal-v2/source/cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp) | `8f09c79f5e52fffd3a2fb0e7810727dc9480d336452f69dec2d11cce3ca0fe62` |
| [pingpong kernel](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-formal-v2/source/cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp) | `83a8e63dd9e7f70a735c17fbb779782fc31ce5215509ed644d13e68428438dee` |
| [TMA/WGMMA collective](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-formal-v2/source/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp) | `b78d838244bbc71e915cf0ec7efb353d0cdda2826de945a344c558923c90f375` |
| [PipelineState](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-formal-v2/source/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp) | `7c61d847933cb12a14fcdc68cec902b44c802c9eb34273218c29e601f9c53dee` |

上述快照用于说明控制与状态语义；目标模型冻结时还须绑定其实际编译源码/二进制，不能由快照相同推定所有 V 轮二进制相同。现有数值实现为[v06_model](../../../../../../microbench/gh200_resource_campaign/access_rules/v06_model.py)、[v08_model](../../../../../../microbench/gh200_resource_campaign/access_rules/v08_model.py)和[v09_model](../../../../../../microbench/gh200_resource_campaign/access_rules/v09_model.py)。

当前 cfg_b 实现另绑定 [739011 冻结目录](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011)：CPU 入口核对预测/manifest 绑定、关键源码、S6/S4 SASS 与构建记录的 SHA256，报告保留二进制登记身份和 setup 哈希。实例来源以该报告为准。

## 2. 对象、操作与依赖

图索引保留 CTA/cluster rank、warpgroup、输出 tile j、Ktile、操作数 A/B、槽 index、代次 generation 和请求身份。实际 tile 顺序由 `v08_model.scheduled_work` 给出，包含 swizzle、raster、整 cluster 补齐及 OOB；数学输出面积不替代执行列表。

| 操作/事件 | 源与目的 | 必要软件前驱或后继 | 来源 |
|---|---|---|---|
| `entry` / `producer_first_work` | thread 0 的入口与首个生产者工作循环端点 | producer 首路径由 entry 进入；PROD 在坐标生成、acquire 和首条 A 请求之前 | 739011 `v06_entry` / `v06_begin`；[端点与SASS核对](../../../../../../results/gh200_resource_campaign/access_rules/20261016-cfg-b-prefix-001/README.md#2-假设与接线) |
| `producer_acquire` | 等待输入 SMEM 槽可重用 | 同槽上一代的消费者释放；首代按 empty 初态 | collective `load` / pipeline |
| 发 A/B TMA 请求 | 本 GPU global 分配 → 目标 CTA 本地 SMEM；cluster 按 multicast mask 投递 | acquire 完成、descriptor 与事务计数就绪 | collective `load` |
| full / `consumer_wait` 返回 | 事务完成使目标输入满足消费条件 | 对应槽、相位与事务计数满足；等待返回另有控制执行 | pipeline / collective `mma` |
| WGMMA 发出、commit、wait | A/B 在 SMEM（本研究的 SS 路径）→ 寄存器累加器 | 对应输入 ready、proxy/fence 与 group 依赖 | collective `mma` |
| `consumer_release` | 槽可供后续 producer 再用 | 使用该槽的异步计算满足源码规定的完成等待 | `warpgroup_wait<K_PIPE_MMAS>` 后释放；尾部 `mma_tail` 排空 |
| 输出准备与 TMA store | 累加器寄存器 → epilogue SMEM → global D | 本输出计算完成、epilogue 顺序/槽条件满足 | kernel 调用 collective epilogue |
| 输出源复用/末尾排空 | epilogue SMEM 可复用；目标完成由实际等待语义定义 | store pipeline 的对应等待 | `store` / `store_tail`；完整 D 完成仍由完整调用验证 |

TMA 完成如何耗时、并发请求如何仲裁、WGMMA 的持续服务率不是上述边的内容，接口留给[硬件规则](../hardware_rules/README.md)。cluster full/empty 跨 rank 依赖还需按实际 mask 和 barrier 到达者展开，已有核对证据见[R18](../../../experiments/gh200_sm90/Q01-scheduling/R18-cluster-boundary.md)。

cfg_b 每个 Ktile 的实际输入为 A 一次 16 KiB、B0/B1 各一次 8 KiB，共同登记 32768 B 的 full barrier。数学 A 为 128×64、B 为 64×128；B 沿 N 分为两个 64×64 请求。S6/S4 的 plain SASS 每轮各有三个 `UTMALDG.3D`；构建日志的 A atom 为 131072 bit、B atom 为 65536 bit。来源为冻结目录的 `build/cfg_b_plain.{log,sass}` 和 `build/cfg_b_s4_plain.{log,sass}`。

源请求使用 A 的 M×K、B 的 N×K 坐标；每个 Ktile 将第二维增加 64。目的位置记录操作数缓冲区内的 stage 偏移与槽代次，B1 相对 B0 偏移 8192 B。TMA 的 SMEM swizzle 为 128B，独立于 case 的调度 swizzle。[746953 实例查询](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-descriptor-query-001/RESULTS.md)已核对八候选、两种形状的 32 个 A/B 描述符及其实际 launch params 绑定：搬运类型为 UINT16、rank 3、interleave NONE、L2 promotion 128B、OOB fill NONE、element stride `[1,1,1]`；完整编码参数、原始 128 字节和各次分配地址已归档。

cfg_b的FP32输出实际采用普通`STS.64`。冻结S6/S4 SASS均为每输出8个64×32子块，每子块128线程各8次8 B写入，共8 KiB；两个源槽相距8192 B。`SM90_U32x4_STSM_N`用于构造复制布局，实际`CopyOpR2S`在[sm90_common.inl](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/source/cutlass/include/cutlass/epilogue/collective/builders/sm90_common.inl)中为FP32选择普通向量复制。程序接口`smem_demands`列出输入投递、WGMMA操作数引用、输出STS与TMA源读取；`epilogue_smem_addresses`恢复完整lane地址，已与冻结CuTe的两槽4096个word逐项核对，见[SMEM-01](../../../experiments/gh200_sm90/Q07-smem/SMEM-01-replay.md)。首子块的8条STS之间穿插后续FMUL，后七子块多为连续STS；共同写块价格及准备／写入的可加性仍由[SMEM服务](../hardware_rules/SMEM-SERVICE.md)声明为组合假设。

cfg_b S4/S6的CTA均为384线程，ptxas报告168 reg/thread、stack与spill均为0。实际[pingpong源码](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/source/cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp)由128×128 FP32累加器推得每consumer线程128个累加寄存器，选择producer40／consumer232配额；SASS保留`DEALLOC.CTAPOOL 0x28`和`TRY_ALLOC.CTAPOOL 0xe8`失败重试。初始与稳态账目同为`384×168=128×40+256×232=64512`。`register_demands`保存需求与覆盖关系，可选图展开4个producer warp释放和8个consumer warp申请成功；各warp只保留本地软件先后，CTA池资源依赖见[寄存器服务](../hardware_rules/register_service.md)。来源、独立重放与prefix边界见[REG-01](../../../experiments/gh200_sm90/Q08-register-residency/REG-01-replay.md)。


cfg_b冻结r18传入α=1、β=0及空alpha指针，alpha batch stride为0。主kernel两构建各有两处32-bit `LDG.E`静态scalar分支：启动期空指针走直接常量读取，输出期零stride复用已有scalar；由源码与SASS分支推导的普通global动态加载需求为0。普通STG、LDL／STL均无，A/B与D矩阵沿上述TMA路径。两构建均有79处静态LDC／ULDC（57处32-bit、22处64-bit），用于启动、坐标及barrier／循环控制，时间继续由现有复合服务覆盖。`ordinary_lsu_demands`保存这些需求与条件；逐PC、分支、源码行和构建身份见[L1-01](../../../experiments/gh200_sm90/Q13-l1-lsu/L1-01-replay.md)及其交付包。


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

cfg_b 的 `K_PIPE_MMAS=1`：提交当前 Ktile 后，`wait<1>` 保证较早 Ktile 已完成，再释放较早槽；`mma_tail` 的 `wait<0>` 后释放最后槽。导出器分别保留读取组、计算完成和释放节点，末尾等待单列。每个 Ktile 的 `compute_batch` 暂保留 fence/计算发出这一复合操作，`compute_complete` 是计算服务的完成接口。

观察映射为 `ENTRY → thread 0 入口`、`PRODUCER_FIRST_WORK → 首个生产者工作循环顶部`、`FIRST_MMA → 首次 consumer_wait 返回及 accumulator operand fence 后、WARPGROUP.ARRIVE 前`、`MAIN_END → mma_tail 后`、`EPI_PERMIT → 输出许可`、`EPI_DONE → store_tail 返回`。`store_tail` 的 `.read 0` 等待提供输出源复用边界。trace 的退出戳来自消费者，导出图另保留 `producer_tail`，程序结束连接二者。

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

[cfg_b_program.py](../../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_program.py) 提供 `build_program(row, setup, source_identity)` 和 `expand_cta(program, cta)`：前者复用现有请求账本生成工作、源请求模板和输出，后者按需展开操作节点、前驱、拓扑顺序、槽代次和观察映射。程序中没有时间价格；长 K 保留模板，选择 CTA 后再展开。

CPU 检查覆盖 G3 S6/S4 的全部 132 个 CTA，每 CTA 128 个输入组、384 个输入请求；另检查跨输出多次环绕、单 Ktile 尾部释放和部分输出的有效/填零字节。12 个 cfg_b 历史条件的工作坐标与消费者归属已逐进程对照，运行命令及计数统一见[组合 §4](../composition.md#4-输出)。

TMA 请求完成和计算完成/槽释放的候选已接到这些端点，CPU 入口与检查结果见[组合 §4](../composition.md#4-输出)。已有 trace 只观察输出 tile 级端点，逐请求完成和逐槽释放时刻仍需对应实验。

[cooperative_program.py](../../../../../../microbench/gh200_resource_campaign/access_rules/cooperative_program.py)提供cfg_c S4的`build_program`、`expand_cluster`和`bind_trace`。已核对2304×3072×1024条件：66个cluster、132个CTA，展开131850个节点、10368个源请求、13824次投递，包含跨rank multicast、四路empty到达、持久槽代次和两个计算WG的观察映射。10个dual进程的1320个CTA、19920个角色端点均匹配；完整来源、范围与重算命令见[cfg_c程序交付](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-c-program-001/README.md)。

同一入口已接入cfg_a S6，复用现有跨rank依赖并展开六槽producer-tail等待。739011的K1024/K4096两条件各20个plain/dual记录通过核对，共74400个trace端点、163840个输出校验值；22项归档哈希及9个重放产物逐字一致。每rank每Ktile为本地A 16KiB、向两个rank多播的B0/B1各4KiB；B逻辑形状为[N,K]=[64,32]，rank1从K+32起。构建trait的B尺寸经descriptor构造按num_multicast=2截半，源码依据、支持范围与复现命令见[cfg_a程序交付](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-a-program-001/README.md)。

cfg_a S4 已接入同一入口，2048×8064×640 条件的 528 个 trace CTA、33312 个角色端点和 57344 个输出校验值通过核对；S6 的两份完整实例保持一致。构建、槽代次、原始记录和复现命令见 [S4 程序交付](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-a-stage4-program-001/README.md)。

cfg_a S4/S6 的程序入口现覆盖 M/N 为64倍数的边界，保留越界rank的multicast、槽协议及名义输出子块。16个非零输入case覆盖两stage和全部M/N边界类别，144个历史进程的2032个CTA、22112个角色端点、589824个输出校验值通过核对；见[边界程序交付](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-a-boundary-program-001/README.md)。

[长工作列表核对](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-a-long-trace-program-001/README.md)复用四个非零进程，覆盖S4/S6、每CTA最多119个输出及Ktiles≤stages的边界；528个CTA、427296个角色端点与既有程序依赖相容，16384个输出校验值通过，未新增服务价格。

[cfg_c1程序入口](../../../../../../microbench/gh200_resource_campaign/access_rules/cfg_c1_program.py)已接入单CTA的A32KiB、B0/B1各8KiB与256次线程到达计数。图保留八个warp各32线程的释放可见集合，MAIN观察点与其他warp到达分别表示。五条件的45个进程、620个trace CTA、8056个角色端点和184320个数值样本通过核对，覆盖d332及四种M/N边界组合；[程序交付](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-c1-program-001/README.md)保留来源、字节守恒、源码/SASS及旧调度工作列表回归，尚未新增服务价格。
