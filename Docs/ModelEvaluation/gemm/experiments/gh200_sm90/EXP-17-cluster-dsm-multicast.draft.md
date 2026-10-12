# S17：DSM、cluster 同步与普通 TMA 多播

状态：有限范围草案，等待独立 A 审查及代表形式的目标编译。没有探针、GPU 正确性或性能资格。

## 实验问题与配置

固定 128 线程／CTA、cluster 大小 C∈{2,4,8}，测量跨 CTA shared 访问、cluster 同步和一个源请求向不同数量目标复制的完成服务。使用单 GPU 内部的 cluster，不涉及多 GPU。

每个 C 有七种形式、两个执行范围，共 42 个名义坐标，不增加维度的组合扫描：

| 形式 | 固定工作 | 本轮回答的问题 |
|---|---|---|
| local_read | 每 CTA 每轮读自己的 4 KiB shared 数据 | 相同布局与同步条件的本地读对照 |
| dsm_read | 每 CTA 读 `(rank+1)%C` 的 4 KiB 数据 | 远端映射读与本地读的服务差异 |
| local_write | 每 CTA 每轮写自己的 4 KiB shared 数据 | 本地写对照 |
| dsm_write | 每 CTA 写 `(rank+1)%C` 的 4 KiB 数据 | 一对一远端写服务，无同址竞争 |
| cluster_sync | 每轮一次 `cluster.sync()` | 齐步参与条件下的同步阶段服务 |
| bulk_single_target | 每 cluster 每轮发起一次 16 KiB bulk，目标为 rank C−1 | 单个远端接收者的完成服务 |
| bulk_all_targets | 每 cluster 每轮发起一次 16 KiB bulk，目标为全部 C 个 CTA | 多播接收量增加时的服务 |

执行范围为 `one_cluster` 与 `cluster_grid`。后者的 cluster 数由当前 kernel 的设备查询、合法 cluster 配置和 `cudaOccupancyMaxActiveClusters` 决定并记录，不从 SM 数推算。API 给出的容量上限不能当作实际同时驻留或全部 SM 已覆盖的证明。

实际能力查询先于启动。记录 cluster 支持、`cudaOccupancyMaxPotentialClusterSize`、所选 C 的 active-cluster 容量、每 CTA 寄存器和 shared 大小，以及最终 grid 的 cluster 整除关系。不支持的 C 保留为明确能力终态，不替换成别的 C，也不使用其他 Hopper 产品的非可移植上限。

## DSM 的生命周期与数值

每 CTA 的 4 KiB 数据分成 128×8 个 uint32，线程 t 访问 t+128j，j=0…7。各 CTA 的初值包含 rank 和 word；写入值另包含轮数和 seed，CPU 参考按接收者对应的唯一发送者重算。远端写采用环形一对一映射，每个目标只有一个写 CTA，避免把争用或原子服务混入本实验。

代表可行性探针的 local 和 remote 都通过 `cluster.map_shared_rank` 映射地址，local 映射自身 rank，remote 映射另一 rank；实际 SASS 读写分别使用相同的 `LD.E.STRONG.SYS` / `ST.E.STRONG.SYS` 路径。因此这是映射到自身与远端的匹配对照，仅改变目标 rank，不是普通 `LDS` / `STS` 的裸 SMEM 基线。

初始化之后全部 CTA 先 `cluster.sync()`，保证目标 CTA 已存在。每轮 read 为八次访问／线程、结果消费、CTA 同步和一次 cluster 同步；每轮 write 为八次写入／线程、CTA 同步、一次 cluster 同步、目标 CTA 消费自己的完整 tile、CTA 同步和第二次 cluster 同步。前一次保证可消费，后一次保证消费者结束后才允许下一轮覆盖。local 对照分别保留同样的一次或两次 cluster 同步，不能给 local 去掉同步再把差值全部归因于 DSM。

循环后再做一次独立于计时循环的 cluster 退出同步，确保最后的远端访问和消费完成后才能退出。CUDA 指南明确要求 DSM 访问期间所有相关 CTA 存活，并在退出前完成远端访问。[CUDA 12.9.1 DSM 说明](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#distributed-shared-memory)

读结果须参与逐线程累加／检查，写结果须由目标 CTA 消费并转存；短验证保存所有必要值和各 CTA guard。实际 SASS 必须证明访问仍在计时循环中。具体消费者发布顺序和指令在 B1/B2 固定，不以 `volatile` 关键字代替机器码检查。

## 多播的生命周期与计量

采用无 tensor-map 的 1D `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster`，与 S14 的普通 bulk 对接，避免依赖二维描述符。每 cluster 仅 rank0 的 thread0 发起一个请求；单目标 mask 为 `1<<(C−1)`，全目标 mask 为 `(1<<C)−1`。

所有接收 CTA 使用相同 shared 相对偏移的 payload 和 mbarrier。最小事件序列固定为：

1. 仅选中接收者的 thread0 初始化本地 mbarrier，arrival count=1；发布初始化及短验证 poison reset，经 CTA 同步和 cluster 同步确认各 CTA 存活与布局就绪。
2. 每轮每个选中接收者的 thread0 用一次 `mbarrier.arrive.expect_tx` 设置 Q 字节并完成其唯一 arrival，保存 opaque token／phase 条件。完成所需 proxy 发布、CTA 同步和一次 cluster 同步之后，rank0 才能发起请求。未选中 CTA 不 init、expect、arrive 或 wait；它们仍参与所有 cluster 同步。
3. rank0 的 thread0 发起一次带固定 mask 的复制。选中接收者等待各自本地 mbarrier 的本轮完成，经 acquire 和 CTA 发布后完整消费 payload。
4. 消费者完成后做 CTA 同步和一次 cluster 同步，释放 slot；下一轮才允许设置新完成条件并覆盖。每轮一共两次 cluster 同步，分别是 armed 和 consumer-done 门槛。
5. 尾部再做退出 cluster 同步，所有异步完成和消费均结束后，选中接收者才 invalidate 自己的 mbarrier，CTA 同步后退出；禁止失效尚被 engine／线程使用的对象。

opaque token 只保存并检查合法域；是否实际写入由独立完成记录证明，不能把某个 uint64 值当作必然非法的 token。不猜其编码，不计入 exact 数值已核元素。B1 必须证明等待确实关联本轮 phase，B2 核对实际指令顺序。PTX 说明数据和 mbarrier 信号按 mask 发送到目标 CTA 的相同相对偏移。[PTX ISA 8.8 `cp.async.bulk`](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)

令 G 为实际 cluster 数、I 为轮数、Q=16384 B、R=popcount(mask)。必须分别记录：

| 数量 | 公式 | 含义 |
|---|---|---|
| 源请求数 | GI | 指令语义中的发起请求 |
| 源请求字节 | GIQ | 请求的源 payload，不能据此宣称物理 HBM 流量 |
| 接收字节 | GIRQ | 全部选中目标的有效接收量 |
| 每目标写入量 | GIQ | 对每个选中 rank 单独核算 |

例如 G=1、C=4、I=2：两种 mask 都发起 2 次、请求 32768 B；单目标接收 32768 B，全目标合计接收 131072 B。不能把四倍接收量写成四倍源带宽。

DSM read/write 的请求字节分别为 GCI×4096；同步只记 G×I 个集体阶段，并另记 C 个参与者。不能把一次集体同步算成 C 次独立同步。

## 计时、短验证与接受条件

保留每 CTA 的 `clock64` 区间，仅用于同一 CTA 的周期差；cluster 或网格完成时间用 globaltimer 包络，不跨 SM 相减 `clock64`。访问服务包括约定的同步、地址计算和结果消费。local/remote 差值是该对照条件下的整体变化，不作为裸指令延迟；也不从联合时间中减一个独立同步速率后宣称得到纯访存时间。

全坐标短验证拟固定 `(I,seed)=(1,0)、(2,3)、(5,UINT32_MAX)`，覆盖第一次、复用和尾部排空，保存完整参考所需输出、guard、完成计数、rank 和时间戳。支持的有限矩阵全部通过独立短 B3 后，才做校准、原有界预热及十进程采样。

B1 必须包含 wrong-rank、错误 mask、漏消费完成、提前退出、错误字节计数的可识别负例；B2 核对实际 local／remote shared 操作、cluster 同步及 multicast 指令、循环和资源。发生实现或计量错误先修复，不写成设备不支持。

## 待完成

代表指令目标编译、设备能力查询探针、机器合同、独立 CPU 参考、实际短验证、正式测量及结果图表。本文的 API、生命周期和工作量是待审约定，不能作为已测设备参数导出。
