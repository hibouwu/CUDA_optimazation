旧 R10 的输入必要图与 C v3 输出必要图已按对应端点组合。每输出匹配六个全 128 线程聚合端点：MMA order wait、MMA order arrival、mma_tail 返回、epilogue order wait、store_tail 返回和 epilogue order arrival；每 CTA 另接输入 producer tail 退出条件。共 3156 个端点映射，132 个 CTA 的组合图均无环。

这些映射保留同一源码事件的聚合含义，没有把聚合 max 接到每个线程上形成新 barrier。输入侧的 `input_producer_exit` 用作输入 load-tail 完成的必要接口，不表示所有线程已返回。129024 个线程级输入端口仍为符号端口；不能由聚合事件生成它们各自的时间。全部事件时间与服务时间为 None。

| 全量组合核对 | 数量 |
|---|---:|
| 自身输入 tail 先于输出等待与 store_tail | 504 输出 |
| 源生命周期完成义务包含各 issuing-thread 源读路径 | 129024 条 C++ 路径 |
| 有效 D 全局可见属于 CUDA 完成要求，但不属于源释放要求 | 3854 子块 |
| 全越界子块不虚构有效 D 全局写节点 | 178 子块 |
| 下一 WG 的 MMA 未被串接到前输出 D tail 之后 | 372 次切换 |
| 下一 WG 的 epilogue 保留前输出全部 128 次到达 | 372 次切换 |
| 同 WG 的下一输出保留自身 epilogue 许可 | 240 次复用 |

129024 是展开的源码线程路径数，不是硬件 TMA 请求数或物理读取次数。source_lifetimes_closed、threads_return_all384 和 CUDA completion requirement 仍为不同端点。图只能表达必要不等式，不能证明存在完整物理执行、给出有限退出上界、预测最后 CTA 或形成零延迟时间解。

三个原 B pitch 的 D 输出列表逐项相同，输入必要图也已核对相同，因此只重放一次共同必要顺序；地址与服务并未被判定相同。`summary.json` 保留所有 132 CTA 的组合图 SHA 与全量检查，CTA0、107、108 的压缩图分别覆盖 T4 内部、T4 双边界和 T3 路径。其它图可由固定输入与程序完整重建，不常驻展开所有图。C 的限定端点映射审查另行交回。

```sh
python microbench/gh200_resource_campaign/access_rules/analyze_r10_lifecycle_composition.py \
  --input results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-input-composition-v1 \
  --D-data results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-legacy-D-lifecycle-v1/data \
  --D-module results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-legacy-D-lifecycle-v1/analyze.py \
  --output NEW_OUTPUT_DIRECTORY
```
