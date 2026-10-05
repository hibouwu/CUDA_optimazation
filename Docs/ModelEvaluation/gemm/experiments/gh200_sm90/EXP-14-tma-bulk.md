# EXP-14：普通 1D bulk 双向复制

A 修订草案，等待 `S14-A-review-r2.json` 独立复审。合同见 [tma_bulk.json](../../../../../microbench/gh200_resource_campaign/contracts/tma_bulk.json)，代表指令源见 [tma_bulk_forms.cu](../../../../../microbench/gh200_resource_campaign/probes/feasibility/tma_bulk_forms.cu)。正式探针和 adapter 尚未实现；本文没有 GPU 正确性或性能结果。

## 有限配置与指标

测量一段连续数据在 GMEM→SMEM、SMEM→GMEM 两个方向的完整复制服务。每个方向使用 1/4/8/16/32/64 KiB 六档 payload，单 CTA 与全 GPU 两个范围，共 24 个正式配置。每 CTA 128 线程，只有 thread0 发起复制，每轮一个请求、一个 shared 缓冲，不设置多个在途 stage。没有 tensor-map、二维寻址、swizzle、多播或跨 CTA shared 访问。

单 CTA 报告 `B_transport/clock64_cycle/CTA`；全 GPU 报告 `GB_transport/s/GPU`，由运输字节除以全网格 globaltimer 纳秒包络得到。每次运输 Q 字节只计 Q，不因一读一写计成 2Q，也不乘 128 个参与线程。设 B 为实际 CTA 数、I 为正式轮数：

`work_count = B×I×Q`。

接口中的读写 payload 仅统计 global 一侧：GMEM→SMEM 为读 `BIQ`、写 0；SMEM→GMEM 为读 0、写 `BIQ`。控制对象、输入初始化、结果检查和回读不加入分子。例如 1 CTA、65,536 轮、64 KiB 对应 4,294,967,296 B，已经超出 uint32；所有总量和地址偏移用 uint64/size_t 计算并先检查溢出。

全 GPU 使用 `sms×min(4, occupancy_limit_ctas_per_sm)` 个 CTA，并要求实际覆盖全部设备 SM。该 occupancy 是实际特化、动态 shared 大小下的资源上限。SMEM 用 Q 字节 payload 和另外 32 字节控制区；64 KiB 档需要显式动态 shared opt-in，并在启动前核对静态加动态容量和可驻留 CTA 数。

## 地址和输入

每个 CTA 独占连续的 32 个 global 槽，每槽 Q 字节。第 i 轮地址偏移是 `(uint64(block_id)×32+i%32)×Q`；同一方向仅分配一个活动 global 数组，聚合大小为 `B×32×Q`，另加数组两侧各 16 B guard。不同 CTA 不共享写区。保留实际分配大小和 L2 容量，不把重复访问的运输量当作物理 HBM 流量。

源、目的地址至少 16 B 对齐，Q 是 16 B 的倍数；范围不能越过实际分配。mbarrier 为独立、8 B 对齐的对象，不与 payload 重叠。PTX 的 size 和 shared 地址操作数是 32 位，global 指针及 mbarrier 返回状态是 64 位；Q 最大 65,536 B，不累计成跨阶段 tx-count。[PTX 8.8 的 bulk 定义](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)

输入以 uint32 word 区分地址：GMEM→SMEM 为 `uint32(17×global_word_index+seed)`；SMEM→GMEM 为 `uint32(29×(block_id×(Q/4)+word_in_tile)+seed)`。仅 word 值按模 2^32 截断，地址、次数和总量不截断。正式 shared 源在计时前填好，在各轮间保持不变；global 源也保持不变。每次启动前把目的和 guard 初始化为已知 poison，完成后逐 word 校验。

## GMEM→SMEM：到达数和完成字节分开

使用本 CTA 目的形式 `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes`。PTX 8.8 包含这个形式；`.shared::cta` 目的限定符从 PTX 8.6 引入，bulk 基本形式要求 `sm_90` 或更高。本项目仍须用 CUDA 12.9、`sm_90a` 实际编译核实，不能仅凭文本语法认定编译通过。

thread0 在计时前把 mbarrier 初始化为 **1 次线程到达**，通过 `fence.proxy.async.shared::cta` 发布初始化，再经过 CTA barrier。每轮由同一个线程执行：

1. `mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [bar], Q`，为本阶段登记 Q 个待完成字节。
2. 发起一次 bulk copy，绑定该 mbarrier。
3. `mbarrier.arrive.release.cta.shared::cta.b64 state, [bar]`，完成唯一一次线程到达，取得本阶段的 opaque 64 位 state。
4. 用此 state 反复执行 `mbarrier.try_wait.acquire.cta.shared::cta.b64`，直到成功；再让全 CTA 经过 barrier，才读取或复用 shared 目的。

Q 是事务字节数，不是到达线程数。提前登记事务、复制后再 release arrive，使本阶段必须同时满足线程到达和异步事务完成；成功 acquire wait 是输入可用的依据。下一轮复用已经完成的对象，不重复 init；最后一次复制完成、所有参与者会合后，才在计时外 invalidate。[expect_tx](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx)、[arrive](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive)、[wait](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)

## SMEM→GMEM：源复用和完整写出

所有 shared 生产者写好各自 word 后，各自执行 `fence.proxy.async.shared::cta`，再经过 CTA barrier，由 thread0 发起 `cp.async.bulk.global.shared::cta.bulk_group`。generic 与 async proxy 的交接不能由普通 CTA barrier 单独替代。[Async proxy](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#async-proxy)

同一发起线程接着 `cp.async.bulk.commit_group`，正式循环使用 `cp.async.bulk.wait_group 0`。成功后再经过 CTA barrier，才能复用目的槽、开始下一次完整复制或让其他线程读取输出。bulk group 属于发起线程；另一个线程执行 wait 不能代替它。wait 的写入可见性不能被扩写成无同步的跨 CTA 或主机可见性，主机读取仍在 CUDA 完成同步之后。[commit_group](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group)

每个 SMEM→GMEM 配置另有一次独立单请求诊断，不增加正式吞吐坐标：thread0 记录起始 clock64，发起并提交一次复制，执行 `wait_group.read 0` 后记录源释放时刻。全 CTA 会合后，把全部 shared 源 word 改成原值的按位取反；再次会合，由原发起线程执行完整 `wait_group 0` 并记录完整完成时刻。最后逐 word 检查 global 目的仍为原值。

`.read` 只保证源读取结束，可以释放 shared 源；无 `.read` 的 wait 才等待包括目的写入的完整操作。两者不保证存在可观察的时间差。诊断保留每 CTA 的三个本地时间戳，报告 cycle/request；全 GPU 情况也不跨 CTA 相减 clock64，不生成源释放 GB/s。完整诊断终点还包含源覆盖和诊断 barrier，不能减去源释放时间就宣称测得纯写出尾延迟。[wait_group](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)

## 计时、长度和正确性

正式起点在操作数准备、shared 生产者发布和 mbarrier 初始化之后，thread0 依次读 globaltimer、clock64，然后经过 CTA 入口 gate。每轮只有一次 Q 字节复制，包含发起、完成等待及 CTA gate；最后一个请求完整结束并会合后，thread0 依次读 clock64、globaltimer。窗口内不放逐 word 消费循环或逐请求时间戳数组。最终输出写回和主机验证位于终点之后，因此该指标是完整复制及控制服务，不是裸 bulk 引擎吞吐。

合同中的 32 轮仅为校准 pilot。完成全部 24 个配置的短正确性筛查后，才按 pilot 事件时间估计 20 ms 目标长度，截断并限制到 128–65,536 轮，保存带二进制、合同、设备和 pilot 哈希的 `resolved_cases.json`。随后预热和正式十进程采样固定使用这一长度；resume 不重新校准。

短检查冻结为下列三组配对，每个正式配置分别运行三个独立 profile/进程，每个进程只启动一次 target kernel，合计 3 次 target；不展开轮数与 seed 的笛卡尔积。

| 独立短 profile | target 轮数 | 顶层 seed | target 启动数 |
|---|---:|---:|---:|
| `bulk_short_1_seed0_v1` | 1 | 0 | 1 |
| `bulk_short_2_seed3_v1` | 2 | 3 | 1 |
| `bulk_short_33_seed4294967295_v1` | 33 | 4294967295 | 1 |

seed 使用既有命令参数和 validation 记录的顶层字段，不向固定 `launch_fields` 添加 seed。三个 profile 的正确配对、完整输出和独立收据全部收齐并通过，才满足该 case 的短检查要求；24 个 case 全部满足后才能推进 pilot/预热。复用实际复制原语、线程数、Q 和网格；33 轮覆盖 32 槽回绕，满足 `early_validation_v1` 最多 64 轮的限制。检查每次请求的全部 Q/4 个 word，检查后才复用；短检查不作为性能样本。

SMEM→GMEM 的源释放诊断使用另一个独立 profile `bulk_source_release_one_request_v1`，顶层 seed 固定为 3，独立进程/收据，只发起一次 target 内的一次复制请求；不隐含在上表任何 profile 中，也不加入 24 个正式吞吐坐标。该诊断同样在预热前完成。正式 GMEM→SMEM 检查最后一轮 shared 中全部 word；正式 SMEM→GMEM 检查全部 32 个 global 槽及 guard。32 轮 pilot 恰好写满 32 槽，检查全部目的 word 和 guard；1/2 轮短检查的未访问槽必须仍为 poison，33 轮则额外检查回绕后第 0 槽的覆盖。

保存完整检查输出或带哈希的逐 word 工件，不能只保存 `errors=0` 或一个 checksum。记录实际完成请求数、mbarrier 等待次数、超时标记、资源和短检查绑定。启动前拒绝错误到达数、Q、对齐、越界、重叠及资源条件；负例还应覆盖单 word 损坏、32 位地址截断、漏 full wait 和先预热后全家族短检查。

mbarrier 轮询每次请求最多等待 1 秒，内部 globaltimer 读取保留为超时开销，不算额外测量窗口。超时后不能正常返回、释放或 invalidate 仍有异步请求访问的 shared 对象；正式实现必须失败并保留证据，可采用 fatal trap 后受控进程清理。`bulk.wait_group` 没有软件轮询超时，受控主机运行器必须提供有界进程终止和确认清理。任何未知 CUDA 失败、未完成复制或超时都不是合格样本，也不标为 unsupported。

## 目标编译入口

代表源包含 GMEM→SMEM 完成、SMEM→GMEM 完整 wait、源释放后覆盖再完整 wait 三个 kernel，Q 是受限的运行时 size。它没有正式计时循环、CLI 或 GPU 启动，`main` 只返回 0；不把可编译源当成可直接运行的正确性程序。

```bash
nvcc -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_90a,code=sm_90a -Xptxas=-v \
  microbench/gh200_resource_campaign/probes/feasibility/tma_bulk_forms.cu \
  -o /tmp/tma-bulk-forms
cuobjdump --dump-sass /tmp/tma-bulk-forms > /tmp/tma-bulk-forms.sass
```

后续实际编译需归档命令、工具链、source/binary/PTX/SASS 哈希与资源记录，并由未实施者核对两方向的地址空间、操作数位宽和完成指令。正式各特化、回边、计时位置和诊断覆盖还须进入 B 审查；普通 bulk 的通过不覆盖 tensor-map、TMA 2D 或并发 stage 实验。
