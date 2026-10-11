# 原 R10 cfg_b：D 生命周期源码接口

交付对象是 `M2600 N3000 K2000`、原 pingpong 调度、`C=void`、float D、`64×32` epilogue。三个原 B pitch 为 3000/3008/3072。源文件逐一核对原 `source_hashes.json`；本接口提供必要先后约束、坐标和端口，不提供时间价或通过资格。正式数据在 `data-final-v3/`。

## 接线与端点

`j` 是同 CTA 的输出序号，所属 WG 为 `j mod 2`。所有 `*_all128` 都是对应 128 个线程事件的最大值或全部所需到达完成界。它们不是新增 barrier，也不能用单线程采样替代。

| 接口 | 原源码语义 | 组合时可保留的约束 |
|---|---|---|
| `math_order.arrive_all128[j]` | `mma()` 返回后，128 个线程通知另一 WG 的 math stage | 下一 WG 可开始 MMA；不要求本 WG 的 `mma_tail` 或 D tail 已结束 |
| `D.entry_all128[j]` | 本轮各线程先返回 `mma_tail`，再返回 epilogue order wait，进入 `store` | 同时受自身 tail 与前一输出全部 epilogue 到达约束；`j=0` 的 WG0 等待由初始相位满足 |
| `store.return_all128[j]` | 8 次子块 store 序列及最后一次 acquire 后的全 WG 同步已执行 | `read1` 允许最新提交的一组仍在读源；不是全部 D 源完成或全局可见 |
| `store_tail.return[t,j]` | 各线程执行自己的 `read0` 并重置计数 | 只等待该线程所属 bulk groups。非 issuing 线程可以先返回并到达 |
| `epilogue_order.arrive[t,j]` | 线程自身 tail 返回、更新状态后，向另一 WG 的 epilogue stage 到达 | 逐线程成立，不增加“所有线程先等 issuing warp”的同步 |
| `epilogue_order.arrive_all128[j]` / `D.exit_all128[j]` | 全部 128 次 epilogue 到达完成 | 包含 issuing warp 的源读取排空；下一 WG 的 epilogue 可安全复用共享 D 源缓冲 |
| `math_order.wait.return[t,j+2]` | 本线程自身上一输出的 epilogue 到达后，再等待另一 WG 的 math stage | 依赖 `arrive[t,j]` 和 `math_order.arrive_all128[j+1]`；不依赖 `epilogue_order.arrive_all128[j+1]` |
| `same_wg_next_mma_permission_all128[j+2]` | 上一行的全 128 线程聚合界 | 可作为全 WG 的必要顺序端口，不能作为最早线程或 warp 的实际起点 |
| `input.producer_tail_return` | 原输入 producer 遍历最终 S6 ring 的 empty 等待 | 关闭输入 slot 生命周期；不表示 D 源排空或全局 D 可见 |
| `source_lifetimes_closed` | 输入 producer tail，加两个 WG 各自最后输出的 D source/order 闭合 | CTA 源生命周期的必要完成界；不等于 384 线程实际退出或 kernel 完成 |
| `cuda_kernel_completion_requirement` | 全部 384 线程返回与异步 D 操作完成的外部完成义务 | 全局 D 可见与源读取分别保留为未观测的义务；不能由 read0 推出此端点时刻 |

`store_tail.return_all128 → epilogue_order.arrive_all128` 是各线程自身先后关系取 max 后的必要不等式。不存在 `store_tail.return_all128 → 每一个线程的 arrive` 这条源码约束。

原 order barrier 是 `OrderedSequenceBarrier<2,2>`，每个目标 barrier 初始化计数 128。stage0 是 MMA，stage1 是 epilogue；两者分别排序。`j=0,1,2,3` 的等待相位依次为 `1,0,0,1`；每次 `arrive` 后本线程状态推进。不能合成一条把前轮 D tail 置于另一 WG MMA 之前的串行链。

## 八个子块与源复用

原 epilogue 是 `epi_n` 外层、`epi_m` 内层。相对 tile 的 D 原点依次为：

```text
q:       0       1       2        3        4        5        6        7
(m,n): (0,0)  (64,0)  (0,32)  (64,32)  (0,64)  (64,64)  (0,96)  (64,96)
slot:    0       1       0        1        0        1        0        1
```

B 的原 `D_boxes` 是坐标排序。本交付按源码顺序重新绑定，逐个检查原点、形状、有效形状和字节，不改变 B ledger。

每个 q 的源路径为 R2S、proxy fence、128 线程 named barrier、issuing warp 的 TMA store/commit、`read1`、128 线程 named barrier。`DelayTmaStore=true` 在准备当前子块时先提交上一子块；当前寄存器计算可早于该提交，接口未把它强制串行化。q 后的 `read1` 等待本线程本输出所有 `0..q-1` 组源读取完成，允许 q 仍未完成。因此 q=1 后的 acquire/sync 保护 q=2 对 slot0 的写入。源读取可以早于 commit 完成，图中没有 `commit → source_read_complete` 的虚构约束。

`store()` 的最后 `read1` 仍允许 q=7 pending。紧接着每个输出都调用 `store_tail` 的 `.read0`，然后才发出自己的 epilogue order arrival。原 pingpong 的这一步不能复用 cooperative 中间输出的 pending-source 图。

输出 j 的 D state 初值 count=8j；8 次子块后为 8j+8，kernel 再 advance 一整 tile，得到本 WG 的下一输出 j+2 的 8j+16。slot=q mod2；逻辑 generation=floor((8j+q)/2)。state phase 被保留，但 `PipelineTmaStore` 的 wait 决策使用 count/always_wait，不使用 phase。kernel 明确设置 `always_wait=true`，包括第一次 acquire。

源码 first-warp 路径为 WG 局部线程 0..31，所有 128 线程参与 R2S、同步和输出 order arrival。图展开的是这些 C++/PTX 线程路径；没有审计本二进制 SASS 的 lane lowering，也不据此收费或推断实际硬件 TMA 指令数量。

## 有效、部分有效与全越界

每个 pitch 都有 132 CTA、504 输出和 4032 子块：3720 full、134 partial、178 zero-valid。CTA0..107 各 T4，CTA108..131 各 T3。三组 D 输出列表完全一致，仅输入 B pitch 条件不同。

完整 tile 名义 D 源为 64KiB，每个子块 8KiB；三个条件各累计名义源 33,030,144B，有效 D 全局区域 31,200,000B。名义源不是实测物理读取流量。

有效面积按原 descriptor 的 M/N 边界裁剪，最右下有效子块包含 `40×24`。全越界子块的有效 D 写入义务为空，但源码不按有效面积跳过 R2S、fence、sync、store、commit、acquire 或 tail/order。partial/zero-valid 的 SMEM 实际读取量仍未知；保留源读完义务，不设服务价为零。

`.read0` 等到已提交 bulk groups 的源和 tensormap 读取完成，足以释放共享源，不保证目标写入对执行线程可见。每个非零有效子块另保留 `global_valid_writes_visible`；源码没有提供这个内部时间点。整个 async completion 同时要求源读取完成和有效写入可见；全越界没有有效写入节点。参见 [PTX bulk wait 语义](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)、[CUDA TMA 线程等待与边界](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma)。裁剪后的有效区域是归档 descriptor 与 B 几何的源码推论，不证明物理请求量。

这里的 CUDA 完成端口要求匹配原 kernel 调用以及流顺序。成功等待记录在调用后的 event，才能把捕获的工作视为完成；不能把某个 CTA 的 read0 或 producer_tail 当成该完成事件。[CUDA event record/synchronize](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__EVENT.html)

## 交付与复核

- `analyze.py`：核对 7 个原 manifest 源文件、3 个 B ledger、B request；重算所有 D 绑定，逐 CTA 检查展开图无环。
- `data-final-v3/C-legacy-D-lifecycle.json`：`C_boundary_response`、源码 SHA/行号、15 项输入 SHA、各条件计数和资格边界。
- `data-final-v3/cfg_b_ldb*-D-lifecycle-ledger.json.gz`：每输出 8 个子块、128 组线程 tail/order 端口与聚合端口，deterministic gzip。
- `make_cta_graph(bound_outputs_for_one_CTA)`：需要时展开一个 CTA 的必要无价约束；不必常驻展开全卡图。各节点 `time_price=null`；这些边不构成数值时间预测或 GT 认证。
- `test_lifecycle.py` / `tests-final-v3.txt`：18 项 CPU 检查，包括逐线程 own mma_tail 先于 epilogue wait 返回、非 issuer 提前到达、all128 聚合、同槽复用、两阶段 order、first/next MMA 与 D 解耦、D 可见与 producer_tail 分离、T3/T4 与错误绑定拒绝。
- `source/R15-output-service-at-e6eda283.md`：只复用已有几何和 `.read0` 语义；cooperative 图、scalar STS 价格与 R14 不同几何未迁移。

使用新的输出目录复算，旧报告不会被覆盖：

```sh
C_D_DIR=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/C-20261011-legacy-D-lifecycle-v1
python3 "$C_D_DIR/analyze.py" \
  --run /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4 \
  --out /tmp/C-R10-D-independent-replay
python3 -m unittest discover -s "$C_D_DIR" -p test_lifecycle.py -v
```

初次运行因源码别名匹配漏写 `cutlass::gemm::` 停止，未生成 payload；记录在 `initial-source-site-failure.txt`。`data-v1/` 是通过后的中间版。v2 补齐初始两轮 math wait 端口、有效 D 写入的 issuing-path 前置约束，并澄清线程路径计数。主线预读又指出逐线程 own mma_tail → epilogue wait 返回的必要边；正式 v3 已补齐并增加针对检查。data-final-v2、tests-final.txt、README-v2.md 与原 delivery.json 保留为中间版，不用于正式接线。

本次只写 C reanalysis。没有 GPU 执行、时间价、物理 D 可见观测、观察器资格或新的独立局部留出。旧 V09 候选不变，`qualified=false`；C 持续目标未完成。
