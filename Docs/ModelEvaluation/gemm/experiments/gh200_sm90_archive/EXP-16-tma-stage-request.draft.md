# S16：16 KiB TMA 的软件 stage 与请求数

状态：A 草案，只做有限范围与 CPU 生命周期初检；没有探针、GPU 或性能资格。

每个请求固定 Q=16384 B，采用 S14 的无 tensor-map 1D bulk 双向形式、128线程、thread0统一发起。软件stage数 S 与每stage item请求数 R分别取1/2/4，范围为one_cta/all_gpu，共36名义坐标。stage item是一次逻辑运输批次，具有R个不重叠请求；每个软件slot保存该item的R块payload。stage、request与commit group分别记录，不能把SR当成物理队列深度。

| S\R | 1 | 2 | 4 |
|---|---:|---:|---:|
| 1 | 16 KiB | 32 KiB | 64 KiB |
| 2 | 32 KiB | 64 KiB | 128 KiB |
| 4 | 64 KiB | 128 KiB | 256 KiB |

表中是payload共享容量，实际另分配8S+32 B控制/对齐空间。两方向采用同一分配规则；G2S使用S个8B mbarrier，S2G这部分保留但不执行mbarrier操作。actor只允许最多S个“尚未取得完成证明”的item，这个软件上限并不等于某时刻真正未完成的物理请求数。S4/R4的256KiB payload已超过当前GH200实际232448B opt-in上限，双向/双scope四个点保留为启动前资源拒绝，其余32点只是必要容量满足，仍须真实编译/attribute/occupancy检查。static/shared/local与寄存器限制不能用此表替代。

## 完成与复用

item i使用slot `i mod S`。slot内第r个SMEM块为 `(slot×R+r)×Q`，global第b个CTA第i个item第r请求为 `((b×32+i mod32)×R+r)×Q`。global环32item，所有请求相互不重叠、16B对齐；分配另外有两侧16B guard。正式S2G各slot/request的源pattern在计时前准备，循环中不重新生产payload。G2S来源按global slot/request/word区分，源环保持不变。

G2S：每软件slot一个mbarrier，init expected-arrival=1。每item先expect RQ bytes，再发R条各Q字节copy，release arrive一次取得opaque64bit token。复用slot前等待其上一item token的acquire完成，随后CTA发布只授予消费者读取/capture。所有消费者完成全部R块后，再经过独立consumer-done CTA门槛才能释放slot；短模式此后才反相reset该slot，由执行reset的各生产者完成shared generic→async proxy fence并CTA会合，下一item才可issue。不能把运输完成或第一道CTA发布当成消费者完成。逻辑generation由i/S记录，不解码或猜测token位。初始化thread0执行mbarrier.init后，初始化者执行shared generic→async proxy fence，再通过CTA会合发布，其他发起/消费逻辑才允许开始。末尾排空所有已发item、消费者完成后统一会合并invalidate所有初始化barrier。

S2G：每item的R条bulk请求由同一thread0提交到一个bulk async-group；每item恰好一次commit。S=软件slot数，group=提交批次，R=批次请求数，三者仍是不同对象。prefill最多S个item；当i>=S时，先执行full `wait_group S-1`，确保item i-S及更老提交都完成，再通过CTA门槛释放该slot。末尾full `wait_group0`保证所有尾部输出完成。此处采用完整输出等待来定义复用，不额外扫描`.read`。fullwait包含写目标和对发起线程的可见性，`.read`只保证源读取，不足以证明最终输出。[PTX ISA8.8的commit/wait语义](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)

同一group内部请求没有次序保证，因此R个目标必须不重叠；不能通过提交顺序假定结果顺序。short消费者在issuer fullwait后的CTA发布与global async proxy fence后读取。源shared普通生产者在首个请求前各自执行shared async proxy fence并CTA会合。

## 计量与短验证范围

I表示logical item数：运输字节 `B×I×R×Q`，一份payload只计一次。G2S arrival=Nitems、expected-total=Nitems×RQ；S2G commits=Nitems、submitted requests=Nitems×R。稳态fullwait调用数max(0,I-S)，尾部fullwait0一次；调用数与完成item数不能混为一谈。单CTA为clock64同CTA差，全GPU为全grid globaltimer包络。起点在准备之后、首次prefill之前；止点在tail全部完整完成且CTA门槛之后，包含prefill/steady/drain，不能只取最优稳态片段。CUDA event只作辅助包围。

候选短profile固定seed3、I=1/2/5/33，单进程4个target、一个非均匀input profile，满足现有core上限。1/2覆盖partial fill，5触发S4的第一次复用，33覆盖global32环回绕。不能因capacity拒绝而缩小S/R/Q后继续用原case ID。短按实际stage/request映射重置每份必写输出为其期望反相，捕获每个retired item的R请求全部值；完整global环/guard、used与unused共享slot、完成/提交/到达/复用generation及token身份均保存。循环内reset/capture只属于短验证，正式循环不执行；正式路径仅保留运输完成、必要CTA门槛和缓冲生命周期，不加入调试消费者的reset/capture工作。短模式初始化共享和global必写区域的反相poison后，所有执行generic写入的生产者分别执行对应shared/global generic→async proxy fence；共享reset还须CTA会合发布，然后才允许覆盖该区域的异步运输。global reset发生在主机时须完成其受控CUDA初始化操作；发生在device时依上述生产者proxy fence发布，二者不得混称为由CTA本身刷新proxy。

G2S正式尾部保留每个used软件slot最后一个item的R块数据；其item为 `s+floor((I-1-s)/S)×S`，s<I。post-export字节另外记录为min(S,I)×R×Q，不加到运输分子。S2G核对完整32R目标环及guard。所有case记录实际global/SMEM分配、grid、各软件stage与请求量；whole-GPU结果还受occupancy/CTA数及工作集变化影响，不能单凭S变化归因于TMA队列。

当前CPU检查只证明固定矩阵、容量必要条件、地址互斥、FIFO retirement/slot generation以及上述逻辑计量，不证明硬件完成时刻、descriptor、队列深度或实际吞吐。独立A通过后才实施，B1/B2与全合法短GPU通过后才能校准、预热和正式采样。

## 依赖与后续接口

本草案只新增有限范围约定和 CPU 生命周期账本，不改 S14 原合同、原探针、任何受审归档或共享 core。未来探针沿用 `probe_runtime.cuh` 的设备查询与计时基础，但需要单独实现 S/R 生命周期和数值参考；不能用 S14 的单请求完成证明替代本阶段多请求的完成证明。当前 JSON 的 cases 是规划坐标表，不是已注册运行合同；探针实现阶段才补齐固定 launch、正式 iterations、校准政策和 adapter 绑定，经独立接口审查后接入现有总控。

whole-GPU 候选网格为实际 SMS×min(4,该 kernel 的实际 occupancy API 上限)，one_cta 固定1；实际 kernel attribute、static+dynamic SMEM、寄存器与 API 上限必须随每个坐标保存。32 个必要容量满足坐标若出现真实编译或启动限制，必须单列终态，不得自动删点或改变分配规则。正式循环次数继续采用现有128–65536有界校准与预热协议；这些行为需在后续实现接口中写成可核对字段，当前 CPU 账本不声称已经验证它们。

A r2 CPU账本把运输完成、acquire后的CTA发布、消费者读取、consumer-done门槛和slot释放分开记录，复用前另记短reset/fence/CTA。删除consumer-done的负例必须在slot释放前拒绝。此账本描述短验证的逻辑协议，正式无循环reset/capture；不模拟实际硬件完成时刻。

短循环reset按运输方向限定：G2S重置即将被覆盖的共享payload；S2G重置即将被覆盖的global目标，保持计时前准备的共享源pattern。两者均在旧使用者完成且区域可复用之后执行，再完成对应proxy fence和CTA发布。不能把S2G共享源反相清空后当作合法运输输入。
