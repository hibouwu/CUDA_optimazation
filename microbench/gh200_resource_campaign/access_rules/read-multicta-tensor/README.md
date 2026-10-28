# 多 CTA tensor TMA 探针：阅读目录

这里只放指向上级真实文件的相对软链接，把这套探针的代码集中到一处，方便连续阅读。修改代码请改原文件，运行命令也照旧。新增或改名模块后补上链接；`find . -xtype l` 可以列出失效的链接。测试文件不放进这里。

理解某个具体实验时，先读该族 README 的逐轮表，再进入对应的实验模块和探针源码；底座工具按调用处查阅。

## 测什么

多个 CTA 同时发出 cfg_b 的三请求 tensor TMA，观察共享源、SM 分组与存储域对搬运时间的影响：

- `tma_tensor_multicta.cu`：cfg_b tensor 请求加一个独立的探针控制程序（源环、在途组数、起步偏移等）。
- `tma_tensor_background.cu`：另一个 WG 发出受控的 WGMMA 背景，测本地与远端计算对 TMA 的干扰（issue-placement-interference）。
- `gpc_topology.cu`：用 cluster 共驻留识别同 GPC 的 SM 边，不拟合延迟价格；`gpc_l2_contention.cu`：未解析 SM 对的同／跨 GPC 小对照。
- `numa_latency.cu`：在专用 scratch 行上 discard 后做 CG 冷／热加载，识别存储局部域（storage-domain-numa）。

## 阅读顺序

1. `tma_tensor_multicta.py`：tensor 探针的 CPU 程序与候选比较（不调用 GPU），其余模块都以它为程序模型。
2. 服务候选：`tma_gpc_service.py`（每个声明共享组的有限并行服务）、`tma_stateful_source.py`（两域缓存与在途填充，叠加每 GPC 请求服务）；`tma_gemm_sources.py` 把 cfg_b 的源坐标接到 tensor 程序。
3. 冻结、拟合与评分入口：`prepare_tma_tensor_discrimination`、`analyze_tma_tensor_candidates`、`fit_tma_tensor_*`、`score_tma_gpc`、`analyze_numa_affinity`／`score_numa_affinity`、`q02_source_isolation`、`replay_transfer_evidence`。

GPC 拓扑、GPC 竞争、NUMA 和 `tma_tensor_background` 的构建与运行脚本在各自的轮次目录中，`microbench` 里没有对应的 Python 入口。请求级组合模型 `cfg_b_system`、`cfg_b_request_model`、`cfg_b_input_network` 使用这里的服务候选，属于模型代码，不在本目录。

## 模块

| 模块 | 行数 | 做什么 | 主要使用示例 |
|---|---:|---|---|
| `tma_tensor_multicta.py` | 325 | cfg_b 三请求组探针的 CPU 程序与候选比较；无 GPU 调用。 | [shared-source-forms](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/shared-source-forms.md)、[gemm-input-network](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/gemm-input-network.md)、[request-service-rules](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-service-rules.md)、[source-organization](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/source-organization.md) |
| `tma_gpc_service.py` | 28 | 每个声明共享组的有限并行服务候选；组到 GPC 的映射由证据文件给出。 | [shared-source-forms](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/shared-source-forms.md)、[gemm-input-network](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/gemm-input-network.md)、[request-service-rules](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-service-rules.md)、[completion-observation](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/completion-observation.md) |
| `tma_stateful_source.py` | 246 | cfg_b G3 开发候选：两域缓存与在途填充，叠加每 GPC 请求服务。 | [cache-organization](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q03-source-service/cache-organization.md)、[source-organization](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/source-organization.md)、[gemm-input-network](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/gemm-input-network.md)、[completion-observation](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/completion-observation.md) |
| `tma_gemm_sources.py` | 46 | 把 cfg_b 的源坐标接到 tensor 探针程序，比较源订阅范围。 | [request-shape-response](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-shape-response.md)、[probe-control](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/probe-control.md)、[request-service-rules](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-service-rules.md) |
| `prepare_tma_tensor_discrimination.py` | 78 | 冻结 12 个条件、价格组、预测、判据和已核对的原生二进制。 | [TRANSFER-01-tensor-2d](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/TRANSFER-01-replay.md) |
| `analyze_tma_tensor_candidates.py` | 113 | 用 Q2 校准筛选价格组，计算源环相位干预的候选预测。 | [TRANSFER-01-tensor-2d](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/TRANSFER-01-replay.md) |
| `fit_tma_tensor_two_tier.py` | 95 | 两种源服务率、两种完成延迟与在途共享的开发拟合；不改冻结成绩。 | — |
| `fit_tma_tensor_workingset.py` | 95 | 源工作集状态候选：用同一请求服务引擎拟合开发端点。 | [source-state-candidates](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q03-source-service/source-state-candidates.md)、[TRANSFER-01-tensor-2d](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/TRANSFER-01-replay.md) |
| `score_tma_gpc.py` | 79 | 按预冻结的共享组窗口检验有限并行源服务候选。 | [shared-source-forms](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/shared-source-forms.md)、[multi-cta-gpc-sharing](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/multi-cta-gpc-sharing.md)、[gemm-input-network](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/gemm-input-network.md) |
| `analyze_numa_affinity.py` | 34 | 读取页对的冷/热加载窗口，检验共享存储域的同向与反向响应。 | [storage-domain-numa](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q03-source-service/storage-domain-numa.md) |
| `score_numa_affinity.py` | 47 | 按冻结的相关性、页对同号率与冷/热窗口门槛核对局部域。 | [storage-domain-numa](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q03-source-service/storage-domain-numa.md) |
| `q02_source_isolation.py` | 330 | Q02 三种行距的现有源服务隔离回放；CPU、固定参数、最多 33 次调用。 | [request-service-rules](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-service-rules.md)、[request-shape-response](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-shape-response.md) |
| `replay_transfer_evidence.py` | 352 | CPU 回放有限异步搬运旧证据；运输率保持其原协议与完成边界。 | [TRANSFER-01-tensor-2d](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/TRANSFER-01-replay.md) |

## 现行代码与历史快照

各轮次目录的 `source/` 保存当轮的源码版本（有些轮次只做了准备，没有执行），本目录链接的是现行版本，二者不一定相同。本目录用于阅读现行实现。重建或重跑历史测量时使用当轮快照，旧路径入口需要时使用兼容视图（`flat_layout_view.py`）。已有数据的 CPU 重评分按各族 README 中经过验证的入口执行。

表中“主要使用示例”先列直接研究该模块的实验族，再按各轮记录（plan、README、脚本等，不含回传数据）提到该模块的轮次数补足，最多列四个族，用来提示依赖关系，不等于实验归属的完整判断；`cfg-b-composition` 的组合交付包引用了大量模块，没有列出。
