# cfg_b 请求级组合模型：阅读目录

这里只放指向上级真实文件的相对软链接，把请求级组合模型的核心代码集中到一处，方便连续阅读。修改代码请改原文件，运行命令也照旧。新增或改名模块后补上链接；`find . -xtype l` 可以列出失效的链接。测试文件不放进这里。

理解某次模型回放时，先读该族 README 的逐轮表，再进入对应的程序定义、服务候选和事件推进代码；依赖按调用处查阅。

## 是什么

把真实 cfg_b 程序展开到每个 CTA 的输入槽与依赖，接上各项服务候选（输入请求、同步、计算、输出、频率），推进事件得到时间线和 plain 预测。Q14 的 G3 回放、Q02/Q03 的 GEMM 负载实验，以及 C1/V09 的组合预测都用这套代码。它是模型代码，不是探针；只收五个核心入口。

## 阅读顺序

1. `cfg_b_system.py`：总入口。读入运行前输入与服务参数（`load_profile`、`prepare`），调用 `predict` 得到时间线、请求观测与 plain。
2. `cfg_b_program.py`：无时间价格的程序实例。`build_program` 由 case 与 setup 生成程序，`expand_cta` 展开每个 CTA 的输入槽与 pingpong 依赖。
3. `cfg_b_request_model.py`：`simulate` 先调用 `supply_model.shared_stage_events` 推进事件，再由 `expand_cta`／`bind_cta` 将已求解的请求、同步与计算时间线绑定到程序图并核验依赖；`critical_path_analysis` 做关键路径分析。全部价格由调用方以 ns 给出。
4. `cfg_b_input_network.py`：输入服务候选。`CfgBInputNetwork` → `AddressRoutedInputNetwork` → `ReadSharedInputNetwork`，再由读共享版本分别派生本地联合、端口排序和形成服务前置版本，对应 [gemm-input-network](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/gemm-input-network.md) 等页的轮次。
5. `supply_model.py`：静态 TMA 请求与供给账本（`static_tma_requests`、`supply_work_ledger`），以及多 CTA 共享的阶段事件推进（`shared_stage_events`）。全仓库 73 个模块使用它。

## 不在本目录的依赖

按阅读需要查看上级目录中的：同步服务 `mbarrier_service.py`；源状态与共享服务 `tma_stateful_source.py`、`tma_gpc_service.py`、`tma_tensor_multicta.py`（见 [read-multicta-tensor](../read-multicta-tensor/README.md)）；频率 `cfg_b_frequency.py`、`cfg_b_phase_frequency.py`；旧聚合组合 `cfg_b_composition.py`；以及 `register_quota.py`、`sm_activity.py`、`case_inputs.py`。

## 模块

| 模块 | 行数 | 做什么 | 主要使用示例 |
|---|---:|---|---|
| `cfg_b_system.py` | 227 | cfg_b同版请求级组合：运行前输入、状态化服务、时间线与plain。 | [gemm-input-network](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/gemm-input-network.md)、[gemm-connection](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q14-composition/gemm-connection.md)、[request-preparation-return](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-preparation-return.md)、[local-compute-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q05-wgmma/local-compute-chain.md) |
| `cfg_b_program.py` | 309 | 无时间价格的 cfg_b 程序实例；按 CTA 展开输入槽与 pingpong 依赖。 | [completion-observation](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/completion-observation.md)、[gemm-input-network](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/gemm-input-network.md)、[gemm-connection](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q14-composition/gemm-connection.md)、[probe-control](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/probe-control.md) |
| `cfg_b_request_model.py` | 723 | 将已有请求服务候选绑定到共同 cfg_b 程序；全部服务价格由调用方以 ns 给出。 | [completion-observation](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/completion-observation.md)、[probe-control](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/probe-control.md)、[request-preparation-return](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-preparation-return.md)、[gemm-input-network](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/gemm-input-network.md) |
| `cfg_b_input_network.py` | 466 | cfg_b输入候选：共享读取、GPC应答处理和逐CTA投递。 | [gemm-input-network](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/gemm-input-network.md)、[request-preparation-return](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-preparation-return.md)、[input-output-joint](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q07-smem/input-output-joint.md)、[cache-organization](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q03-source-service/cache-organization.md) |
| `supply_model.py` | 1595 | Static TMA requests and calibrated supply prediction, without fitting or IO. | [native-d-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/native-d-chain.md)、[COMPUTE-01-04-resident-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md)、[request-preparation-return](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-preparation-return.md)、[local-compute-chain](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q05-wgmma/local-compute-chain.md) |

## 现行代码与历史快照

各轮次目录的 `source/` 保存当轮的源码版本（有些轮次只做了准备，没有执行），本目录链接的是现行版本，二者不一定相同。本目录用于阅读现行实现。重建或重跑历史测量时使用当轮快照，旧路径入口需要时使用兼容视图（`flat_layout_view.py`）。已有数据的 CPU 重评分按各族 README 中经过验证的入口执行。

表中“主要使用示例”先列直接研究该模块的实验族，再按各轮记录（plan、README、脚本等，不含回传数据）提到该模块的轮次数补足，最多列四个族，用来提示依赖关系，不等于实验归属的完整判断；`cfg-b-composition` 的组合交付包引用了大量模块，没有列出。
