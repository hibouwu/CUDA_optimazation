# Q14 gemm-connection 分族方案（主题即族）

依据 [逐轮候选](Q14-move-proposal.tsv) 中的对象、变量、端点。目录只到族：`Q14-composition/<族>/<原轮次名>`，深度与轮次名不变；原 29 个候选降为族 README 内的子组。本表为方案，未移动任何轮次。

规则沿用结果首页：轮次原件逐字节不变，不修改轮次内部链接；历史入口通过 `flat_layout_view.py` 运行。Q14 其余族（cfg-b-composition、COMPUTE-01-04-resident-chain、program-graph-export）不动。

## 总览

| 族（目录） | 主题 | 轮次 | 子组 | 内容 |
|---|---|---:|---:|---|
| `source-sharing` | 源共享 | 14 | 3 | 跨 CTA 源重用、源缓存组织、源行合并 |
| `read-request-service` | 读取与请求服务 | 13 | 5 | 输入读取与应答网络、GPC 投递、TMA 请求形成与返回、读取服务下界、输入计数 |
| `publication-timing` | 发布与时序 | 10 | 3 | producer 发布进度、请求年龄与查询时刻、年龄的空间（raster）分布 |
| `wait-completion` | 等待与完成 | 9 | 4 | consumer full 等待策略、末尾 wait0、进入等待前的 full 状态、操作数完成集合 |
| `compute-program` | 计算与程序 | 11 | 5 | WGMMA 等待、主循环重复计算与活动额度、计算完成滞后、同行 CTA 活动 |
| `output-path` | 输出路径 | 4 | 3 | epilogue 交接顺序、输出后台与输入共享、同行输出 TMA |
| `local-rule-transfer` | 局部规则迁移 | 8 | 4 | 把局部服务规则接回完整 GEMM 的 CPU 迁移与回归（不新增实机测量） |
| 待定（见末节） | 重复性与契约 | 2 | 2 | 两轮更像其他问题的内容 |

合计 71 轮：7 个族 69 轮，待定 2 轮。

## `source-sharing`（源共享）

| 子组 | 轮次 | 对象 | 变量 | 端点 | 备注 |
|---|---|---|---|---|---|
| input-cache-organization | `20261022-Q14-banked-source-cache-001` | 输入服务中的源缓存状态 | 分区组相联/端口受理后查询 | G3 两段 L、plain、full 状态 | 缓存组织候选；不与实机等值源改址直接混合 |
| input-cache-organization | `20261023-Q14-port-ordered-cache-001` | 输入服务中的源缓存状态 | 分区组相联/端口受理后查询 | G3 两段 L、plain、full 状态 | 缓存组织候选；不与实机等值源改址直接混合 |
| input-line-merge | `20261022-Q14-bounded-sector-merge-001` | 输入源行发出与订阅 | 32/128B 合并粒度、域范围、合并截止 | G3 两段 L、plain、请求工作计数 | 共同主端点为 G3 行距响应；每轮具体合并规则必须保留 |
| input-line-merge | `20261022-Q14-domain-line-pool-001` | 输入源行发出与订阅 | 32/128B 合并粒度、域范围、合并截止 | G3 两段 L、plain、请求工作计数 | 共同主端点为 G3 行距响应；每轮具体合并规则必须保留 |
| input-line-merge | `20261022-Q14-line-subscription-screen-001` | 输入源行发出与订阅 | 32/128B 合并粒度、域范围、合并截止 | G3 两段 L、plain、请求工作计数 | 共同主端点为 G3 行距响应；每轮具体合并规则必须保留 |
| input-line-merge | `20261022-Q14-source-line-pool-001` | 输入源行发出与订阅 | 32/128B 合并粒度、域范围、合并截止 | G3 两段 L、plain、请求工作计数 | 共同主端点为 G3 行距响应；每轮具体合并规则必须保留 |
| gemm-source-sharing | `20261022-Q14-equal-value-source-001` | 真实 GEMM 的跨 CTA 源重用 | 等值地址折叠/逐 CTA 等值副本 | CTA0 两段 L、plain 及对应预测 | 001–004 换卡、换序与修复保持同族；44/50 是该干预的程序接入 |
| gemm-source-sharing | `20261022-Q14-equal-value-source-002` | 真实 GEMM 的跨 CTA 源重用 | 等值地址折叠/逐 CTA 等值副本 | CTA0 两段 L、plain 及对应预测 | 001–004 换卡、换序与修复保持同族；44/50 是该干预的程序接入 |
| gemm-source-sharing | `20261022-Q14-equal-value-source-003` | 真实 GEMM 的跨 CTA 源重用 | 等值地址折叠/逐 CTA 等值副本 | CTA0 两段 L、plain 及对应预测 | 001–004 换卡、换序与修复保持同族；44/50 是该干预的程序接入 |
| gemm-source-sharing | `20261022-Q14-equal-value-source-004` | 真实 GEMM 的跨 CTA 源重用 | 等值地址折叠/逐 CTA 等值副本 | CTA0 两段 L、plain 及对应预测 | 001–004 换卡、换序与修复保持同族；44/50 是该干预的程序接入 |
| gemm-source-sharing | `20261022-Q14-program-intervention-001` | 真实 GEMM 的跨 CTA 源重用 | 等值地址折叠/逐 CTA 等值副本 | CTA0 两段 L、plain 及对应预测 | 001–004 换卡、换序与修复保持同族；44/50 是该干预的程序接入 |
| gemm-source-sharing | `20261022-Q14-source-fold-replay-001` | 真实 GEMM 的跨 CTA 源重用 | 等值地址折叠/逐 CTA 等值副本 | CTA0 两段 L、plain 及对应预测 | 001–004 换卡、换序与修复保持同族；44/50 是该干预的程序接入 |
| gemm-source-sharing | `20261023-Q14-private-source-001` | 真实 GEMM 的跨 CTA 源重用 | 等值地址折叠/逐 CTA 等值副本 | CTA0 两段 L、plain 及对应预测 | 001–004 换卡、换序与修复保持同族；44/50 是该干预的程序接入 |
| gemm-source-sharing | `20261023-Q14-private-source-transfer-001` | 真实 GEMM 的跨 CTA 源重用 | 等值地址折叠/逐 CTA 等值副本 | CTA0 两段 L、plain 及对应预测 | 001–004 换卡、换序与修复保持同族；44/50 是该干预的程序接入 |

## `read-request-service`（读取与请求服务）

| 子组 | 轮次 | 对象 | 变量 | 端点 | 备注 |
|---|---|---|---|---|---|
| input-read-reply-network | `20261022-Q14-address-routed-input-001` | 输入读取与逐接收者应答 | 分区路由、在途共享、订阅上限与服务价格 | G3 L/plain，部分轮次另读纯 TMA release 窗口 | 较宽候选：69 是参数标定，纯 TMA 与 G3 双端点需审阅是否再拆 |
| input-read-reply-network | `20261022-Q14-request-reply-network-001` | 输入读取与逐接收者应答 | 分区路由、在途共享、订阅上限与服务价格 | G3 L/plain，部分轮次另读纯 TMA release 窗口 | 较宽候选：69 是参数标定，纯 TMA 与 G3 双端点需审阅是否再拆 |
| input-read-reply-network | `20261023-Q14-bounded-read-fanout-001` | 输入读取与逐接收者应答 | 分区路由、在途共享、订阅上限与服务价格 | G3 L/plain，部分轮次另读纯 TMA release 窗口 | 较宽候选：69 是参数标定，纯 TMA 与 G3 双端点需审阅是否再拆 |
| input-read-reply-network | `20261023-Q14-public-fold-correction-001` | 输入读取与逐接收者应答 | 分区路由、在途共享、订阅上限与服务价格 | G3 L/plain，部分轮次另读纯 TMA release 窗口 | 较宽候选：69 是参数标定，纯 TMA 与 G3 双端点需审阅是否再拆 |
| input-read-reply-network | `20261023-Q14-read-sharing-network-001` | 输入读取与逐接收者应答 | 分区路由、在途共享、订阅上限与服务价格 | G3 L/plain，部分轮次另读纯 TMA release 窗口 | 较宽候选：69 是参数标定，纯 TMA 与 G3 双端点需审阅是否再拆 |
| input-read-reply-network | `20261023-Q14-source-only-calibration-001` | 输入读取与逐接收者应答 | 分区路由、在途共享、订阅上限与服务价格 | G3 L/plain，部分轮次另读纯 TMA release 窗口 | 较宽候选：69 是参数标定，纯 TMA 与 G3 双端点需审阅是否再拆 |
| gpc-copy-service | `20261022-Q14-copy-work-bound-001` | 逐 CTA 副本的 GPC 投递 | 固定价格的 FIFO/工作守恒约束 | 完整时间预测/必要下界、L 响应 | 29 是 37 的必要下界诊断；若只收测量轮次则作为关联分析保留 |
| gpc-copy-service | `20261022-Q14-gpc-delivery-bound-001` | 逐 CTA 副本的 GPC 投递 | 固定价格的 FIFO/工作守恒约束 | 完整时间预测/必要下界、L 响应 | 29 是 37 的必要下界诊断；若只收测量轮次则作为关联分析保留 |
| tma-formation-return | `20261023-Q14-address-matched-return-001` | 请求形成与 producer 返回 | 形成服务位置、返回反馈及地址绑定修复 | 纯 TMA release/早晚窗口；55 另做 GEMM 迁移 | 53 为同址修正，保留同一次模型谱系；不能归成实机发布时戳实验 |
| tma-formation-return | `20261023-Q14-formation-before-cache-001` | 请求形成与 producer 返回 | 形成服务位置、返回反馈及地址绑定修复 | 纯 TMA release/早晚窗口；55 另做 GEMM 迁移 | 53 为同址修正，保留同一次模型谱系；不能归成实机发布时戳实验 |
| tma-formation-return | `20261023-Q14-formation-return-001` | 请求形成与 producer 返回 | 形成服务位置、返回反馈及地址绑定修复 | 纯 TMA release/早晚窗口；55 另做 GEMM 迁移 | 53 为同址修正，保留同一次模型谱系；不能归成实机发布时戳实验 |
| tma-read-service-bound | `20261023-Q14-independent-service-bounds-001` | 独立 TMA 读取/应答容量 | 按实际地址与资源归属统计工作 | 每片段/每16KiB服务时间上界 | 与 copy 下界对象不同；不能归入 input-counter |
| gemm-input-sector-counter | `20261023-Q14-input-counter-localization-001` | 真实 GEMM 输入路径工作量 | base/A16 行距 | NCU L2 sector 比值及采集资格 | 实际计数未取得；plain 有效不等于 counter 实验成功 |

## `publication-timing`（发布与时序）

| 子组 | 轮次 | 对象 | 变量 | 端点 | 备注 |
|---|---|---|---|---|---|
| request-age-raster | `20261022-Q14-completion-spatial-screen-001` | 请求年龄的空间分布 | 源组/逻辑组与 raster 对应 | 中心化年龄预测/空间投影 | 28 为干预前诊断，48 为预注册可分性，47 为实机尝试 |
| request-age-raster | `20261022-Q14-raster-age-discrimination-001` | 请求年龄的空间分布 | 源组/逻辑组与 raster 对应 | 中心化年龄预测/空间投影 | 28 为干预前诊断，48 为预注册可分性，47 为实机尝试 |
| request-age-raster | `20261022-Q14-raster-contrast-prediction-001` | 请求年龄的空间分布 | 源组/逻辑组与 raster 对应 | 中心化年龄预测/空间投影 | 28 为干预前诊断，48 为预注册可分性，47 为实机尝试 |
| producer-publication-progress | `20261022-Q14-producer-publication-001` | g43 查询处 producer 发布进度 | 发布标记的全局/CTA 本地实现 | B1 后发布计数、full 未完成状态 | 45 是对应计数和时间线的 CPU 核对 |
| producer-publication-progress | `20261022-Q14-producer-publication-002` | g43 查询处 producer 发布进度 | 发布标记的全局/CTA 本地实现 | B1 后发布计数、full 未完成状态 | 45 是对应计数和时间线的 CPU 核对 |
| producer-publication-progress | `20261022-Q14-producer-publication-003` | g43 查询处 producer 发布进度 | 发布标记的全局/CTA 本地实现 | B1 后发布计数、full 未完成状态 | 45 是对应计数和时间线的 CPU 核对 |
| producer-publication-progress | `20261022-Q14-publication-time-alignment-001` | g43 查询处 producer 发布进度 | 发布标记的全局/CTA 本地实现 | B1 后发布计数、full 未完成状态 | 45 是对应计数和时间线的 CPU 核对 |
| request-query-age | `20261022-Q14-published-request-age-001` | B1 后发布至 full 查询/完成边界 | 软件计时原点及返回规则对齐 | 年龄区间、query−FIRST、MAIN−query、full 状态 | 57 修正上界原点；不得沿用已撤销的 issue 年龄推断 |
| request-query-age | `20261023-Q14-full-boundary-screen-001` | B1 后发布至 full 查询/完成边界 | 软件计时原点及返回规则对齐 | 年龄区间、query−FIRST、MAIN−query、full 状态 | 57 修正上界原点；不得沿用已撤销的 issue 年龄推断 |
| request-query-age | `20261023-Q14-query-return-connection-001` | B1 后发布至 full 查询/完成边界 | 软件计时原点及返回规则对齐 | 年龄区间、query−FIRST、MAIN−query、full 状态 | 57 修正上界原点；不得沿用已撤销的 issue 年龄推断 |

## `wait-completion`（等待与完成）

| 子组 | 轮次 | 对象 | 变量 | 端点 | 备注 |
|---|---|---|---|---|---|
| full-wait-policy | `20261018-Q14-full-wait-policy-001` | GEMM consumer full 等待 | blocking/polling，含行距对照 | plain 响应；具备 dual 的轮次补充 L | 14 只有 plain；若要求所有辅助端点也完全相同，应继续拆分 |
| full-wait-policy | `20261018-Q14-runtime-full-wait-001` | GEMM consumer full 等待 | blocking/polling，含行距对照 | plain 响应；具备 dual 的轮次补充 L | 14 只有 plain；若要求所有辅助端点也完全相同，应继续拆分 |
| full-wait-policy | `20261020-Q14-full-wait-plain-002` | GEMM consumer full 等待 | blocking/polling，含行距对照 | plain 响应；具备 dual 的轮次补充 L | 14 只有 plain；若要求所有辅助端点也完全相同，应继续拆分 |
| full-wait-policy | `20261023-Q14-stride-wait-policy-001` | GEMM consumer full 等待 | blocking/polling，含行距对照 | plain 响应；具备 dual 的轮次补充 L | 14 只有 plain；若要求所有辅助端点也完全相同，应继续拆分 |
| wait0-tail-observer | `20261019-Q14-runtime-tail-001` | 末尾 wait0 返回至结束 | 循环外时戳观察实现 | 边界时戳与观察资格 | 包含失败实现，不能按 runtime 字样另分 |
| wait0-tail-observer | `20261019-Q14-wait0-boundary-001` | 末尾 wait0 返回至结束 | 循环外时戳观察实现 | 边界时戳与观察资格 | 包含失败实现，不能按 runtime 字样另分 |
| full-entry-state | `20261022-Q14-full-entry-mask-001` | consumer 进入等待前的 full 状态 | 位图观察实现/overlay 修复 | 各 Ktile ready/unready 位图及扰动资格 | 实现1未采到机制状态，失败轮次仍保留 |
| full-entry-state | `20261022-Q14-full-entry-mask-002` | consumer 进入等待前的 full 状态 | 位图观察实现/overlay 修复 | 各 Ktile ready/unready 位图及扰动资格 | 实现1未采到机制状态，失败轮次仍保留 |
| operand-completion-set | `20261023-Q14-operand-completion-001` | g43 分操作数完成集合 | A/B 请求分组的观察 | A-only/B-only/A+B 未完成集合 | 端点区别于整组 full 位图，不强并单轮 |

## `compute-program`（计算与程序）

| 子组 | 轮次 | 对象 | 变量 | 端点 | 备注 |
|---|---|---|---|---|---|
| compute-completion-lag | `20261018-Q14-compute-completion-transfer-001` | 组合中的计算完成 | 完成滞后参数 | G3 S6/S4 的 L 与 plain 预测 | CPU 局部完成规则迁移；与 wait0 实机干预分开 |
| wgmma-wait-policy | `20261018-Q14-gemm-wait-001` | GEMM 正常计算等待 | 正常 wait1/wait0 | S6/S4 的 L、plain 与对应预测 | 预测和实测围绕同一等待干预 |
| wgmma-wait-policy | `20261018-Q14-gemm-wait-prediction-001` | GEMM 正常计算等待 | 正常 wait1/wait0 | S6/S4 的 L、plain 与对应预测 | 预测和实测围绕同一等待干预 |
| mainloop-active-repeat | `20261018-Q14-program-budget-scope-001` | 固定物理 S6 的主循环 | 活动槽额度及已测重复计算 | FIRST_MMA→MAIN_END、阶段响应 | 6/12 为已有数据诊断，8 为程序接入；是否并入原 G3 族待核对 |
| mainloop-active-repeat | `20261018-Q14-runtime-intervention-001` | 固定物理 S6 的主循环 | 活动槽额度及已测重复计算 | FIRST_MMA→MAIN_END、阶段响应 | 6/12 为已有数据诊断，8 为程序接入；是否并入原 G3 族待核对 |
| mainloop-active-repeat | `20261019-Q14-stage-scope-screen-001` | 固定物理 S6 的主循环 | 活动槽额度及已测重复计算 | FIRST_MMA→MAIN_END、阶段响应 | 6/12 为已有数据诊断，8 为程序接入；是否并入原 G3 族待核对 |
| mainloop-compute-repeat | `20261021-Q14-direct-repeat-path-001` | GEMM 行距代价的隐藏 | 单遍/三遍及其实现修复 | CTA 两段 L 的行距增量、plain 桥接 | 恢复执行路径和静态 dispatch 属于同一干预的实现演进 |
| mainloop-compute-repeat | `20261021-Q14-static-stage-repeat-001` | GEMM 行距代价的隐藏 | 单遍/三遍及其实现修复 | CTA 两段 L 的行距增量、plain 桥接 | 恢复执行路径和静态 dispatch 属于同一干预的实现演进 |
| mainloop-compute-repeat | `20261021-Q14-stride-compute-discrimination-001` | GEMM 行距代价的隐藏 | 单遍/三遍及其实现修复 | CTA 两段 L 的行距增量、plain 桥接 | 恢复执行路径和静态 dispatch 属于同一干预的实现演进 |
| mainloop-compute-repeat | `20261022-Q14-static-repeat-dispatch-001` | GEMM 行距代价的隐藏 | 单遍/三遍及其实现修复 | CTA 两段 L 的行距增量、plain 桥接 | 恢复执行路径和静态 dispatch 属于同一干预的实现演进 |
| gemm-active-peers | `20261022-Q14-active-peer-001` | 同行 CTA 对目标 CTA 的影响 | 132/1 活动 CTA | CTA0 两段 L 与 plain | 不能并入仅修改源地址或输出 copy 的族 |

## `output-path`（输出路径）

| 子组 | 轮次 | 对象 | 变量 | 端点 | 备注 |
|---|---|---|---|---|---|
| epilogue-defer-order | `20261018-Q14-epilogue-defer-001` | 输出与下一数学 WG 交接 | E0 permit 延后依赖 | C16−C1、M−C16、E0、plain | 计划和 EXECUTION 代替缺失的 README；保留独立 |
| output-shared-service | `20261018-Q14-output-write-gemm-001` | 输出后台工作与输入共享 | 输出源读取后的共享占用规则 | 早段/后段响应及 plain 回归 | 5 是 9 的同版完整时间回归，标为关联回归而非新增物理测量 |
| output-shared-service | `20261018-Q14-shared-output-service-001` | 输出后台工作与输入共享 | 输出源读取后的共享占用规则 | 早段/后段响应及 plain 回归 | 5 是 9 的同版完整时间回归，标为关联回归而非新增物理测量 |
| gemm-peer-output | `20261022-Q14-peer-output-001` | 同行 SMEM→global 输出 TMA | 同行输出 copy 开关 | CTA0 行距周期增量与 plain | 输出保留/屏蔽与 epilogue 顺序不是同一种操纵 |

## `local-rule-transfer`（局部规则迁移）

| 子组 | 轮次 | 对象 | 变量 | 端点 | 备注 |
|---|---|---|---|---|---|
| gemm-local-completion-release | `20261020-Q14-local-bundle-G6-001` | 局部完成/继续/释放规则接回组合 | 固定联合规则替换旧 reuse 项 | G3及开发条件的 L/plain、关键 CTA | G6 补测不另分族；18 为同版时间线分析 |
| gemm-local-completion-release | `20261020-Q14-local-bundle-regression-001` | 局部完成/继续/释放规则接回组合 | 固定联合规则替换旧 reuse 项 | G3及开发条件的 L/plain、关键 CTA | G6 补测不另分族；18 为同版时间线分析 |
| gemm-local-completion-release | `20261020-Q14-local-bundle-transfer-001` | 局部完成/继续/释放规则接回组合 | 固定联合规则替换旧 reuse 项 | G3及开发条件的 L/plain、关键 CTA | G6 补测不另分族；18 为同版时间线分析 |
| gemm-local-completion-release | `20261021-Q14-critical-path-current-001` | 局部完成/继续/释放规则接回组合 | 固定联合规则替换旧 reuse 项 | G3及开发条件的 L/plain、关键 CTA | G6 补测不另分族；18 为同版时间线分析 |
| producer-preparation-cost | `20261021-Q14-producer-preparation-transfer-001` | producer 逐轮请求准备 | 两档已测准备价格替换软件 gap | G3/A16 的 L、plain、CTA 结束跨度 | 准备成本与消费者完成/释放的对象不同，保留独立 |
| input-compute-service-bound | `20261022-Q14-joint-smem-bound-001` | TMA 与 WGMMA 共同服务必要容量 | 全逻辑引用/相邻 B 复用的假设 | 原生 D 窗口与服务下界 | 下界推导与 GEMM 时间预测不同，保留独立候选 |
| gemm-input-compute-sharing | `20261023-Q14-gather-joint-service-001` | GEMM 输入与计算共同服务 | 逻辑字节/源片段引用计费 | G3 两段 L、plain、后续 full 未完成数 | 同一共同资源模型的需求计数变化 |
| gemm-input-compute-sharing | `20261023-Q14-local-joint-service-001` | GEMM 输入与计算共同服务 | 逻辑字节/源片段引用计费 | G3 两段 L、plain、后续 full 未完成数 | 同一共同资源模型的需求计数变化 |

## 待定：重复性与契约（2 轮）

| 轮次 | 对象 | 端点 | 建议 |
|---|---|---|---|
| `20261019-Q14-compute-handoff-contract-001` | 跨输出计算模块状态 | CPU 逐节点一致性和默认回归 | 归 Q06/R14-stage-handoff，或作为程序模块契约留在 Q14/local-rule-transfer |
| `20261022-Q14-bridge-null-001` | 同二进制桥接的可重复性 | plain、L cycle/ns 配对偏差 | 归 Q10/plain-repeat（同二进制重复性控制） |

## 需要确认

1. 7 个族的粒度是否合适（每族 4–14 轮）。
2. `active-peer-001`（同行 CTA 活动对目标 CTA 的影响）现归 `compute-program`；它测的是 CTA 间相互影响，也可归 `read-request-service`。
3. `local-rule-transfer` 收录 CPU 迁移与回归，不是新的实机测量；同意保留为单独的族，还是把各轮放回其规则来源所在的族。
4. 待定两轮跨问题移动与否。

## 执行前提

移动前先建立可恢复的副本并核对条目数；移动后用 `flat_layout_view.py` 抽查几个历史入口能否运行，并更新 Docs 中指向这些轮次的链接。
