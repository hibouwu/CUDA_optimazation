# EXP-12：cp.async 复制与有限缓冲

状态：A 草案，尚未独立审查或实施；没有实际性能结果。

## 实验问题与最小例子

测量给定请求宽度、软件 stage 数和参与范围下，GMEM→SMEM 复制、等待、跨线程消费及缓冲复用的组合服务。

例如一个 128 线程 CTA，每线程复制 16 B，每轮八次，共 8192 轮。运输 payload 是 `128×16×8×8192=134,217,728 B`。原始记录分别保存相同数量的 GMEM 读取和 SMEM 写入，但主指标只将运输的数据计一次。消费者再次读取 SMEM 的字节另列，不把它加入运输 payload。

## 有限配置矩阵

| 坐标 | 取值 |
|---|---|
| 请求形式 | `.ca` 的 4/8/16 B，`.cg` 的 16 B |
| 软件 stage | 1、2、4 |
| 范围 | 一个 CTA、全 GPU |
| 线程与长度 | 每 CTA 128 线程，8192 轮，每轮每线程八次复制 |

共 24 点，每线程每个 commit group 只有一个请求。stage 数与请求宽度共同决定 payload SMEM 分配，最多 8192 B。所有 CTA 读取同一块 8 MiB GMEM 数组；其缓存状态保持未证明。完整约定见 [async_copy.json](../../../../../microbench/gh200_resource_campaign/contracts/async_copy.json)。

## 计时、等待与复用

初始化和 poison 在窗口前；保存起点并完成 CTA 入口同步后，预填充、循环、尾部和排空都在计时范围中。

先发出 stage 数量的请求组，再逐组等待与消费。线程完成自己的 `wait_group` 后，经过 CTA barrier，才读取相邻线程复制的数据；消费后再用一次 CTA barrier 释放槽位，才能写入下一轮。

尾部停止发出新组后，等待参数必须随剩余组数下降。例如四 stage 只剩两个组时，`wait_group 3` 并不能证明下一组已完成；合同要求使用 `min(stages−1, remaining−1)` 对应的合法常量形式。最后 `wait_group 0`、CTA 同步和结果消费完成后才取终点。

单 CTA 用本地周期，全 GPU 用 globaltimer 网格包络并验证 SM 覆盖。软件 stage、每线程 commit group 与物理队列分别解释，不从本实验推断硬件队列深度。

## 正确性与审查

先对全部 24 点做短 GPU 检查，再进行正式预热。短检查用 1/2 轮及最终槽内容核对；源数据按 word 地址与 seed 构造，消费者检查邻居请求的全部 32 位字。CPU 模拟源地址、缓冲生命周期、尾部等待和无符号模运算，不能只验证某一个 lane 或请求的第一个字。

SASS 检查核对实际复制宽度、cache hint、commit/wait 的降低、循环位置及 CTA 同步；不能只因源码写了异步复制就接受测量。请求地址范围、对齐、独立目的槽及资源上界也必须通过检查。

本组测的是包含消费者和同步的完整服务。`.ca/.cg` 是请求策略，不能单凭它们或工作集大小宣称缓存命中、HBM 流量或裸异步复制延迟。

## 实测与复现

探针、适配器和受审运行入口尚未实现。C 阶段保留公共十进程、有界预热与最多三批规则，图表按 scope 和请求形式分开，注明误差条与原始记录，并补一条真实字节数和时间手算。

语义依据 [PTX ISA 8.8 cp.async](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) 及其 commit/wait 条目；跨线程消费同时受显式 CTA 同步约束。
