# 原 R10 生命周期组合：C 接口审查

审查主线 `analyze_r10_lifecycle_composition.py` 的 `compose_cta`，目标 SHA 为 `bd4f2b75901d4e15222ac3e8859df688aa4ebf60a737b3335642b80b44350363`。只核对六个 all128 alias、聚合与个体线程边界、输入 producer 退出，以及独立 D 可见义务。未发现需修改的接线问题。

| C D 源图端口 | 主线输入图端口 | 对应意义 |
|---|---|---|
| `math_order.wait.return_all128` | `mma_order_wait_return_all` | 同一 math stage 的 128 线程 wait 返回最大值 |
| `math_order.arrive_all128` | `mma_order_arrive_all` | `mma()` 返回后、own tail 前的 128 次 math order 到达完成 |
| `mma_tail.return_all128` | `mma_tail_return_all` | 本 WG wait0 与末组输入释放之后的 128 线程 tail 返回最大值 |
| `epilogue_order.wait.return_all128` | `epilogue_order_wait_return_all` | 各线程 own tail 后、前 WG 全部 epilogue 到达满足后的 wait 返回最大值 |
| `store_tail.return_all128` | `store_tail_return_all` | 每输出的 128 线程 store_tail 返回最大值，包含 issuing-path read0 |
| `epilogue_order.arrive_all128` | `epilogue_order_arrive_all` | 各线程自身 tail 与状态推进之后的 128 次 epilogue 到达完成 |

`input.producer_tail_return → input_producer_exit` 对应输入 producer_tail 完成义务。此 alias 不证明两个物理函数返回时间相等，也不代表 D 可见、全部 384 线程返回或 CUDA kernel 完成。

审查读取主线保存的 CTA0、107、108 图，覆盖 T4/T3 接口，使用正式 C v3 D 源图及已冻结输入图逐一重建。三个保存图与该函数重建结果一致；边集恰为原输入边集与 alias 重命名后的 D 边集的并集。没有额外制造 aggregate→individual 边。六个 alias 之外的个体线程输入端口保留在 `unresolved_thread_input_ports`，`event_times` 和 `service_durations` 均为 None。

逐输出、逐线程检查保留：own mma_tail 返回先于自身 epilogue wait 返回；自身 store_tail 返回先于自身 order arrival；全部 128 次到达才形成聚合完成界。所有非零有效 D 的 `global_valid_writes_visible` 仍是独立外部义务，不在 store_tail 或 source_lifetimes_closed 的前置依赖中，并作为 CUDA completion 的前置依赖；zero-valid 未凭空添加有效全局写节点。

每个受查 CTA 另保存四个无环的额外先后约束见证：

1. 非 issuing 线程 t96 的 epilogue arrival 可先于本输出 q7/t0 源读完。
2. 同 WG j2 的 t96 math wait 返回可先于自己 j0 的 q7/t0 源读完；all128 math wait 与后续整体依赖仍保留。
3. 下一 WG j1 的 MMA order wait 聚合返回可先于 j0 的 D store_tail 聚合返回。
4. source_lifetimes_closed 可先于一个有效 D 子块的 global visibility。

这些见证只说明源码必要依赖图没有增加更强的先后关系。它们不是物理时序、采样认证、GPU 可行性或性能预测；不能据此估计差值、上界或最后完成 CTA。

`review.py` 可重放以上定向检查；`review.json` 保存目标 SHA、六端口定义、三个受查 CTA 的检查和见证。程序只固定本次审查所需脚本/图输入，不重复整个 source manifest、全卡计数或 GPU 检查。原源码、主线图、公共组合脚本均未修改。观察器资格、联合物理执行、时间预测和独立局部留出资格均未取得。
