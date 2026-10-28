# 驻留链探针（COMPUTE A→E）：阅读目录

这里只放指向上级真实文件的相对软链接，把这套探针的代码集中到一处，方便连续阅读。修改代码请改原文件，运行命令也照旧。新增或改名模块后补上链接；`find . -xtype l` 可以列出失效的链接。测试文件不放进这里。

理解某个具体实验时，先读该族 README 的逐轮表，再进入对应的实验模块和探针源码；底座工具按调用处查阅。

## 测什么

单 CTA 上把 cfg_b 的主循环拆成逐步加入计算、交接、TMA 与实际输出程序的短链。各级之间除了新增组件，线程参与、程序组织和首次供给的计时边界也有变化：

- **A** `cfg_b_compute_chain.cu`：一个数学 WG、两份 M64 累加、每 Ktile 8 条 WGMMA；六个 SMEM tile 在计时前准备，没有 producer、输入 mbarrier 和 epilogue。
- **B/C** `cfg_b_handoff_chain.cu`：加入 producer WG 与 consumer WG，六个物理槽、4 或 6 个活动 full(1)/empty(128) barrier，用普通 arrival 交接，没有 TMA。`cfg_b_handoff_sparse.cu` 保留原算术与协议，只取稀疏端点（SYNC-02）。
- **D** `cfg_b_tma_chain.cu`：每 Ktile 三个真实 TMA 请求加 8 条 WGMMA，所有 TMA（包括初始供给）都在计时内。`q02_tma_compute_ctas.cu` 在它基础上扩到多 CTA，可选运行时源 batch；`q02_fixed_source_pair.cu` 在同一分配上配对 stage 与源到 SM 的绑定。两者都直接 include `cfg_b_tma_chain.cu`。
- **E** `cfg_b_pingpong_chain.cu`：实际 S6 CUTLASS pingpong kernel、三个持久输出 tile；trace 版用 V06 overlay 加四个调用边界戳（`cfg_b_pingpong_trace.hpp`）。

相邻两级的差是同条件下的完整程序响应，不能直接当作单个组件的耗时；各级的对照关系见 [COMPUTE 页](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md)。

## 阅读顺序

1. `mbarrier_service.py`：输入槽的 mbarrier 生命周期与条件返回假设。
2. 四个探针与对应的准备模块，按 A→E：`cfg_b_compute_chain`（含构建与采样的通用流程）→ `cfg_b_handoff_chain` → `cfg_b_tma_chain` → `cfg_b_pingpong_chain`。
3. 各实验入口：`q05_*` 为计算等待（Q05），`q04_*` 为同步与槽复用的干预（Q04），`q02_*` 为 D 链输入（Q02）。

D 链多 CTA 源 batch 的 CPU 回放（source-organization、cache-organization 的 replay 类轮次）在 `cfg_b_tma_chain` 的多 CTA 入口上叠加 `tma_stateful_source` 等服务候选，后者见 [read-multicta-tensor](../read-multicta-tensor/README.md)。共用的模型模块 `supply_model.py` 不在这里。

## 模块

| 模块 | 行数 | 做什么 | 主要使用示例 |
|---|---:|---|---|
| `mbarrier_service.py` | 164 | cfg_b 输入槽的 mbarrier 生命周期与显式条件返回假设。 | [native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md)、[source-organization](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/source-organization.md)、[COMPUTE-01-04-resident-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md)、[request-preparation-return](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-preparation-return.md) |
| `cfg_b_compute_chain.py` | 337 | cfg_b 驻留计算短链的有限准备、构建与采样入口。 | [COMPUTE-01-04-resident-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md)、[local-compute-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q05-wgmma/local-compute-chain.md)、[native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md)、[source-organization](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/source-organization.md) |
| `cfg_b_handoff_chain.py` | 386 | cfg_b 实际计算与普通 full/empty 交接短链的 CPU 准备和离线核对。 | [COMPUTE-01-04-resident-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md)、[local-sync-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q04-sync-slot/local-sync-chain.md)、[SYNC-01-02-control](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q04-sync-slot/SYNC-01-replay.md)、[native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md) |
| `cfg_b_tma_chain.py` | 513 | cfg_b 三笔TMA、真实计算与槽释放短链的有限CPU准备。 | [COMPUTE-01-04-resident-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md)、[native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md)、[source-organization](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/source-organization.md)、[request-preparation-return](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-preparation-return.md) |
| `cfg_b_pingpong_chain.py` | 264 | cfg_b 三输出真实pingpong程序的有限CPU准备。 | [COMPUTE-01-04-resident-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md)、[source-organization](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/source-organization.md) |
| `q05_compute_wait.py` | 227 | 驻留计算wait0／wait1配对的准备、构建与全输出评分。 | [local-compute-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q05-wgmma/local-compute-chain.md) |
| `q04_index_control.py` | 195 | 四条件plain索引干预：准备、目标执行与独立重算。 | [native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md)、[SYNC-01-02-control](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q04-sync-slot/SYNC-01-replay.md)、[source-organization](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/source-organization.md)、[request-preparation-return](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-preparation-return.md) |
| `q04_handoff_wait.py` | 169 | 驻留B/C短链的同binary full等待干预；复用原数值及计时检查。 | [full-wait-policy](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q04-sync-slot/full-wait-policy.md)、[native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md)、[source-organization](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/source-organization.md)、[request-preparation-return](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-preparation-return.md) |
| `q04_handoff_model.py` | 107 | 普通arrival短链的局部时间线与已有plain响应；CPU，无参数拟合。 | [local-sync-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q04-sync-slot/local-sync-chain.md) |
| `q04_native_compute_wait.py` | 248 | 原生驻留短链正常wait0的固定预测、构建和有限验证。 | [native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md)、[source-organization](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/source-organization.md)、[request-preparation-return](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-preparation-return.md)、[local-sync-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q04-sync-slot/local-sync-chain.md) |
| `q04_native_feedback.py` | 178 | 原生驻留短链的固定反馈假设、冻结外推和同批对照。 | [local-sync-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q04-sync-slot/local-sync-chain.md) |
| `q04_first_entry_control.py` | 171 | 驻留B与原生D的首次等待延迟配对。 | [native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md)、[full-ready-return](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q04-sync-slot/full-ready-return.md) |
| `q02_local_input_chain.py` | 101 | D三请求到full就绪的局部接线与既有输入边界；CPU，无硬件拟合。 | [native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md) |
| `q02_native_tma_chain.py` | 307 | 固定源、原生索引的D短链构建与有限采样。 | [native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md)、[source-organization](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/source-organization.md)、[request-preparation-return](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-preparation-return.md)、[full-ready-return](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q04-sync-slot/full-ready-return.md) |
| `q02_native_input_growth.py` | 171 | 原生B/D短链的固定控制、长度增长配对。 | [native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md) |
| `q02_native_input_validation.py` | 122 | 原生D的冻结stage／长度局部外推；采样不调用模型。 | [native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md) |
| `q02_first_wait_entry.py` | 138 | 原生D首次full等待前的软件延迟；固定两种窗口放置。 | [native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md) |
| `q02_startup_prediction_validation.py` | 114 | B/D首次等待局部预测类的固定延迟外推验证。 | [native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md) |

## 现行代码与历史快照

各轮次目录的 `source/` 保存当轮的源码版本（有些轮次只做了准备，没有执行），本目录链接的是现行版本，二者不一定相同。本目录用于阅读现行实现。重建或重跑历史测量时使用当轮快照，旧路径入口需要时使用兼容视图（`flat_layout_view.py`）。已有数据的 CPU 重评分按各族 README 中经过验证的入口执行。

表中“主要使用示例”先列直接研究该模块的实验族，再按各轮记录（plan、README、脚本等，不含回传数据）提到该模块的轮次数补足，最多列四个族，用来提示依赖关系，不等于实验归属的完整判断；`cfg-b-composition` 的组合交付包引用了大量模块，没有列出。
