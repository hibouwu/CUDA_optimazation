# 整卡读写共享外存服务

更新：2026-10-17。本页定义阶段5的资源排队工作假设及状态化源接法；旧证据与外推检验见[HBM-01](../../../experiments/gh200_sm90/Q03-source-service/HBM-01-replay.md)，有限登记见[PLAN](MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)。

## 1. 工作假设与接口

[SharedExternalMemory](../../../../../../microbench/gh200_resource_campaign/access_rules/hbm_service.py)表示单GPU上共同竞争的外存资源。所有CTA的读miss和调用方声明的写请求共享一个FIFO；读写工作串行支付，各方向服务率由调用方显式提供，单位为B/ns，没有默认硬件价格。

```python
memory = SharedExternalMemory(read_bytes_per_ns, write_bytes_per_ns)
request = memory.submit(arrival_ns, read_bytes=0, write_bytes=0)
```

对请求的读／写字节数\(Q_r,Q_w\)及方向服务率\(B_r,B_w\)：

\[
s=\max(t_{arrival},f),\qquad
e=s+\frac{Q_r}{B_r}+\frac{Q_w}{B_w},\qquad f\leftarrow e.
\]

返回`request_id`、`arrival_ns`、`service_start_ns`、`service_end_ns`与方向字节数。`service_end_ns`是该资源服务完成，不含后续home返回、跨域传送或目的SMEM完成。调用方按事件引擎给出的非递减arrival提交，同刻按调用顺序预约；时间回退会被拒绝。服务的`frontier_ns`是共享权威前沿，源对象的`hbm_frontier`只镜像其最近一次读完成，两者在其他调用方提交写请求后可以不同。

## 2. 状态化源与写请求接入

[StatefulTensorSource](../../../../../../microbench/gh200_resource_campaign/access_rules/tma_stateful_source.py)新增可选`external_memory_service`。缓存查找与在途合并决定出现新miss后，每条128 B线调用同一对象的`submit(t, read_bytes=128)`，用返回完成时间替换原`hbm_frontier`增量，再加原`home_ns`；远端域继续支付原`link_ns`。同一个A／B atom的全部线在logical issue同刻按行序预约，未新增逐线发出节拍。缓存命中和仍在途的同线填充不重复提交读字节。

未提供此对象时保留原实现。缓存映射、替换、store更新、源前端、TMA目的服务、同步与计算继续由各自模块维护，输入数据就绪仍与已有源服务取最大值。后续TMA边界见[TMA共享服务](TMA-SHARED-SERVICE.md#stage-4-service)。

写竞争由调用方在确定可发出时刻后，向同一个`memory`显式提交`write_bytes`。当前`store`只更新缓存状态，没有dirty追踪或实际写回适配器；`output_source_observer`登记的输出缓存变化也不会自动产生外存写请求。因此本轮接法能表达给定读写事件的竞争，尚不能从GEMM输出自动推导HBM写流量。

## 3. 当前证据与最小缺口

[EXP-13](../../../experiments/gh200_sm90/Q03-source-service/EXP-13-global-rw.md)给出global读写复合窗口的逻辑请求速率。固定纯端点向混合完整窗口外推时，完全重叠的`max`形式与串行`sum`形式均有反例；这没有辨识裸HBM的方向服务率。本服务使用调用方显式输入的价格，不使用旧窗口率校准，也不额外叠加旧窗口中的控制开销。

CPU检查覆盖共享读写排队、128 B计账、在途合并／命中与原默认回归；4／13／32 Ktile×S4／S6六个请求图接线结果逐对象保持原默认输出。真实miss字节、写回可发出时点及并发服务平台仍需匹配程序的观测；本轮交付为可执行工作假设与旧证据回放，不新增独立C2成绩。

[固定1:1组织配对](../../../experiments/gh200_sm90/Q03-source-service/HBM-01-replay.md)完成6进程；CTA分阶段相对交错+4.144951%，在冻结5%判据内。当前不加入单独组织修正，组织相关服务的量化规则仍为信息不足；原纯端点外推失败及TMA迁移资格保持。

[后端共享范围局部筛查](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/HBM-01-backend/20261019-Q03-backend-scope-001/README.md)固定原总能力与所有其他服务，按home／请求域各设两个半速FIFO；两固定接法在S6窗口及结束跨度上均失败，S4未运行。每新miss一笔128B计账和同条件／异目录复核通过，66次局部调用、GPU／拟合0；两接法停止，当前共享FIFO仍为原工作假设。
