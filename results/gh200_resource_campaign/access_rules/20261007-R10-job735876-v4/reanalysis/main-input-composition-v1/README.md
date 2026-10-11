原始 R10 的有效源/fill/full 端口与已验收的消费者释放生命周期已接成必要依赖图。输入绑定 B completion-v2 与 lifecycle-v2；27 项输入 SHA 已核对。三种行距均生成相同的 361746 节点、478668 条边，拓扑检查无环。相同必要顺序不表示时间相同，图没有选择跨 CTA 源共享、队列、仲裁或服务价格。

图补齐以下源码依赖：

- 每个生产者按持续组序推进，前组的 A/B 提交完成后才进入后组 acquire；g≥6 的 acquire 还需要 g−6 的 empty 完成。
- full 合流保留完整 A/B0/B1 目的义务。消费者依序等待、commit；q>0 经 wait1 释放上一组，末组由 wait0 释放。
- MMA 顺序通知位于 mma 返回与 mma_tail 之间。下一 WG 的 MMA 顺序等待不依赖前输出 tail，同 WG 的 j+2 仍需要自己的前输出 epilogue 顺序到达。
- producer_tail 从最终持续 ring 状态依次等待六槽，与较早的后继 acquire 一起覆盖全部输入组。

`*_all` 消费者端口表示参与的 128 个线程对应事件的最大时刻。它们是聚合必要边，不能用单线程 trace 替代；尤其不意味着非 TMA issuing 线程必须在 issuing warp 的 read0 返回后才到达 epilogue 顺序屏障。`mma_retired` 表示已经 commit 的组完成这一逻辑义务，不作为裸 MMA 执行终点或硬件价格。store_tail 内部暂未展开，只保留返回与顺序到达的接口。

`release_emit_all` 指 128 次实际 arrival 发出的最大时刻，`empty_complete` 指全部到达在目标 barrier 生效。前者不能替换为 `consumer_release` 返回后的打点：目标 barrier 可以在最后发出线程读返回后时间戳之前已完成。当前图没有赋予这些端口观测时间。

`verify.py` 直接查询完整图的祖先关系：

| 源码约束的结果 | 核对数 |
|---|---:|
| producer 退出需要每个已用组的 empty 完成 | 16128 |
| 各输出 tail 返回需要自己的最后 commit 组完成 | 504 |
| 各输出 tail 返回需要自身全部 commit 组完成 | 16128 |
| 下一 WG 的 MMA 顺序等待不含前输出 tail 前提 | 372 |
| 输出切换后的前五个 acquire 不含前输出 tail 前提 | 1860 |
| 第六个 acquire 保留前输出 tail 前提 | 372 |
| 同 WG 的 j+2 保留自己的前输出 epilogue 前提 | 240 |
| 零源 B1 仍通过 fill 完成进入 full | 672 |

所有服务时间为 None；D 全局可见、kernel 完成也为 None。没有运行零 lag 时间求解，没有预测关键 CTA、输出时间或完整时间。C 的旧 pingpong D 源码接口仍需绑定，之后才能进一步细化输出内部；有效观测与服务识别要求也尚未满足。

B 已完成限定源码边审查，未发现不成立的必要边或遗漏的输入 producer 退出前提。其 16 项定向父边/祖先检查全部通过，覆盖普通释放、末组 wait0、下一输出首组、同 WG j+2 和 T3/T4 最后六槽。主线已核对审查脚本 SHA 与保存图身份，见 `B-review-receipt.json`；完整审查在 `../B-20261011-input-edge-review-v1/B-input-edge-review.json`。该审查不授予时间或输出完成资格。

```sh
python microbench/gh200_resource_campaign/access_rules/analyze_r10_input_dependency_composition.py \
  --completion results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/B-20261011-boundary-completion-v2 \
  --lifecycle results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/B-20261011-boundary-lifecycle-v2 \
  --output NEW_OUTPUT_DIRECTORY
python results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-input-composition-v1/verify.py
```
