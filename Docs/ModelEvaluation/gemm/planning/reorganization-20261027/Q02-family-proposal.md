# Q02 分族方案（含 Q14 归入的 45 个目录）

本方案只给出归属，不移动目录。逐目录的对应关系见 [Q02-family-proposal.tsv](Q02-family-proposal.tsv)，共 258 行：
- 每行给出当前路径、目标路径、原族和目标族；
- 每个目录恰好归入一个族，目标路径互不冲突；
- 目录深度保持 `Q02-tma-request/<族>/<原轮次名>`，轮次名不变。

分族依据是各轮 README 里写的问题陈述，不看名字。族按主题划分。17 个族的 README 草稿见 [Q02-family-readmes/](Q02-family-readmes/)，每轮一行，搬迁前用它审阅同族各轮是否适合放在一起。

## 结果

Q02 最后有 17 个族、252 个目录；外移 6 个，其中 5 个去 Q03，1 个去 Q01。需要移动的目录共 130 个。`row-stride` 和 `gemm-source-candidates` 两个族的目录全部分走，原族名不再保留。

| 族 | 目录 | 主题 | 来源 |
|---|---:|---|---|
| `probe-control` | 29 | 探针控制路径与软件预算 | 原族（移出 3 个） |
| `request-service-rules` | 25 | 单 CTA 请求服务的规则检验：覆盖计费、准入、返回及各类界 | 从 row-stride 拆出 |
| `R13-async-retirement` | 22 | R13 异步退役（早期轮次） | 原族不变 |
| `gemm-input-network` | 22 | GEMM 负载下输入请求的共享、路由、信用与投递容量候选 | Q14：cfg-b-composition 10 个，gemm-connection 12 个 |
| `request-shape-response` | 20 | 单 CTA 请求的形状、行距、长度、并发与次序的实测响应 | 从 row-stride 拆出 19 个，加 Q14 的描述符查询 1 个 |
| `source-organization` | 20 | D 链多 CTA 的源 batch 组织、服务准入与 full 完成跨度 | 原族（移出 6 个） |
| `native-d-chain` | 18 | 原生 D 局部链：启动、首次等待与长度外推 | 原族（移出 6 个） |
| `shared-source-forms` | 18 | 共享请求服务的候选与判别 | 原族（移出 3 个） |
| `completion-observation` | 16 | 请求完成状态与时刻的观测：full 位图、发布计数、发布后年龄、空间跟随 | shared-source-forms 3 个，Q14 13 个 |
| `request-preparation-return` | 14 | 请求准备、发出受理与返回反馈 | native-d-chain 6 个，probe-control 3 个，source-organization 1 个，Q14 4 个 |
| `gemm-source-reuse` | 10 | GEMM 中跨 CTA 源复用的干预与候选，含等值源 001–004 | gemm-source-candidates 5 个，Q14 5 个 |
| `prefill-wait-entry` | 10 | 首次预填、在途组数与等待入口 | 从 row-stride 拆出 |
| `issue-placement-interference` | 7 | TMA 发出位置与本地计算背景的干扰 | multi-cta-gpc-sharing 6 个，gemm-source-candidates 1 个 |
| `multi-cta-gpc-sharing` | 7 | 同 GPC 共享服务与 SM 分组 | 原族（移出 6 个） |
| `S16-single-cta-control` | 7 | S16 单 CTA 控制（早期轮次） | 原族不变 |
| `TRANSFER-01-tensor-2d` | 5 | TRANSFER-01 tensor 2D 传输 | 原族不变 |
| `R18-first-fill` | 2 | R18 首次填充（早期轮次） | 原族不变 |

## 外移

**移到 Q03 `source-state-candidates` 的 5 个目录，都来自 source-organization**

| 轮次 | 改动的对象 |
|---|---|
| `fill-completion-replay` | 填充完成时缓存的插入、占位与替换 |
| `fill-recency-replay` | 填充完成时的 LRU/MRU 更新 |
| `source-base-replay` | 实际源基址到两个域的映射 |
| `output-state-replay` | 输出写入对源缓存状态的改变（没有新增输出吞吐价格） |
| `admitted-data-replay` | 缓存查询和 miss 提交挪到 GPC 受理之后，即源数据分支的访问时点；在 Q02 交叉引用 |

`data-gated-service` 留在 `source-organization`。它的缓存仍在 issue 时查询和更新，改动的是“数据就绪如何约束进入 GPC 请求服务队列”，属于请求服务的准入。它和 `admitted-data-replay` 的依赖方向相反，两轮互相引用。

**移到 Q01 `work-placement` 的 1 个目录**

`d226-fixed-history-discrimination` 发现了两套 CTA→SM 放置，按目标调用的奇偶完全分组。它的直接后续实验 `d226-work-placement-discrimination-001` 已经在 `Q01/work-placement` 里，归到一起后，“调用历史 → 放置干预 → A 源交换”这条链就连上了。
- Q02 的 `gemm-source-reuse` 引用后续的 A 源交换结果；Q10 引用它的固定调用次数协议。
- 放置与其他调用状态尚未完全区分，原记录的这一结论保留。

## 记录性质不作为分族边界

同一族内可以同时有新 GPU 测量、已有数据分析、CPU 模型筛查和实现核对。同一假设的预测、实测、修复和评分保持在一起，不按执行设备拆开，这与 COMPUTE 系列整组保留的原则一致。

族 README 的逐轮表标出每轮的记录性质，判断依据是目录内有没有 Slurm 日志、receipt、submission、allocation 或回传目录。这个判断是自动的，定稿时按原记录核对。例如 `shared-source-forms` 里的 `source-rotation-001` 执行过作业 753347（12 个进程），属于直接判别实验。

`request-service-rules` 和 `gemm-input-network` 的主题仍然较宽。族 README 内分小节，分别列出覆盖计费、准入、信用、路由、投递等假设，不再增加目录层级。

## 早期族

R13、R18、S16 三族保持不变。它们的轮次没有 README 正文，是早期版本的记录。

## 下一步

1. 审阅 17 份 README 草稿：确认同族各轮放在一起是否合适，并补齐 `request-service-rules` 和 `gemm-input-network` 的小节划分。
2. 把 Q14 外移（[Q14-question-ownership.tsv](Q14-question-ownership.tsv)）和本表合成一张联合对应表。
3. 按已定流程执行：
   - 先做 reflink 备份；
   - 同盘 `mv`；
   - 更新 `MOVED-PATHS`；
   - 替换 Docs 中的链接；
   - 验收。
