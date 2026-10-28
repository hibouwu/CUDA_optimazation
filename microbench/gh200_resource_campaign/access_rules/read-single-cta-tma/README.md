# 单 CTA TMA 探针：阅读目录

这里只放指向上级真实文件的相对软链接，把这套探针的代码集中到一处，方便连续阅读。修改代码请改原文件，运行命令也照旧。新增或改名模块后补上链接；`find . -xtype l` 可以列出失效的链接。测试文件不放进这里，以免被重复发现。

## 测什么

以单个 CTA（SM124）为主运行 cfg_b 的 TMA 输入路径；最早的 `q02_tma_stride.py` 定位用 132 个 CTA、Q4。每组三个请求（A 16 KiB、B0／B1 各 8 KiB，SW128），四个物理槽，源环 64。用 clock64 和 globaltimer 记录完整窗口，可选加一个单点。同一进程内把不同行距或干预成对比较，取同 seed 的增量。

## 阅读顺序

理解某个具体实验时，先读该族 README 的逐轮表，再进入对应的实验模块（评分函数）和 CUDA 代码；底座工具按调用处查阅。例如预填间隔：逐轮表 → `q02_prefill_observer.py` 的 `summarize_prefill_gap` → `tma_gemm_sources.cu` 的首次预填段。通读整套探针时按下面的顺序：


1. **探针程序**：`tma_gemm_sources.cu`（kernel 与 host 参数）和 `r00_common.hpp`。
   - `tensor_probe` 的主要参数：
     - `active`：初始在途组数，Q1/Q2/Q4；
     - `iterations`：正式组数 G；
     - `trace`：0 为 plain，−1 为稀疏单点；
     - `wait_delay_cycles`：每组 full 等待前的自旋；
     - `request_mask`：7 为三请求，1 为只发 A，6 为只发双 B；
     - `source_k_step`：相邻 Ktile 的源步长；
     - `prefill_gap_cycles`：首次预填各组之间的间隔。
   - 成对比较由 `--paired-strides`、`--paired-request-masks`、`--paired-wait-delays`、`--paired-source-steps`、`--paired-prefill-gaps`、`--paired-traces` 给出。
   - 首次预填按四组显式展开（`issue_group(0..3)`），间隔插在组与组之间。
2. **底座**
   - `l2_saturation.py`：命令记录、GPU 独占与分配检查、JSON 与摘要工具，全仓库 42 个模块共用。
   - `q02_tma_stride.py`：准备、CPU 构建与目标审核（`build_review`）、执行、单进程校验、汇总的通用流程。其他实验模块直接复用它，或经 `q02_prefill_observer.py` 间接复用它。
3. **二级底座**
   - `q02_tma_single_cta.py`：复用原构建的单 CTA 定位，`q02_tma_active.py` 和 `q02_tma_length.py` 在它之上。
   - `q02_prefill_observer.py`：首次预填的单点观测，含预填间隔（`prepare_prefill_gap`）和源步长（`prepare_source_step`）两个变体的准备与评分；`q02_descriptor_factor.py`、`q02_request_order.py`、`q02_prefill_active.py`、`q02_wait_entry.py` 复用它，`q02_prefill_shared_prefix.py` 又在 `q02_prefill_active.py` 之上。
4. **各实验的入口**（见下表）。

## 模块与结果族

| 模块 | 做什么 | 结果族（Docs） |
|---|---|---|
| `q02_tma_stride.py` | 固定分配内的 B 行距定位（132 CTA，Q4）；通用流程 | [request-shape-response](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-shape-response.md) |
| `q02_tma_single_cta.py` | 单 CTA 行距定位 | 同上 |
| `q02_tma_active.py` | 固定 Q4 暖机，比较正式 Q1／Q4 | 同上；也被 prefill-wait-entry 的 wait-hint 轮次引用 |
| `q02_tma_length.py` | 单 CTA 行距增量的长度规则 | request-shape-response |
| `q02_a_stride.py` | A 行距的分支判别 | request-shape-response |
| `q02_descriptor_factor.py` | A tensor-map 外层维度重分解 | request-shape-response |
| `q02_request_order.py` | A／B 请求次序 | request-shape-response |
| `q02_source_permutation.py` | 同源集合下的 Ktile 排列 | request-shape-response |
| `q02_single_b.py` | 真实单 B 与双 B 的配对 | request-shape-response、[request-service-rules](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/request-service-rules.md) |
| `q02_prefill_observer.py` | 首次预填单点观测；预填间隔、源步长变体 | [prefill-wait-entry](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/prefill-wait-entry.md)、request-shape-response |
| `q02_prefill_active.py` | 初始在途组数 Q2／Q4 的预填响应 | prefill-wait-entry |
| `q02_prefill_shared_prefix.py` | Q2／Q4 共用首次预填代码的对照 | prefill-wait-entry |
| `q02_wait_entry.py` | 等待入口推迟与 A／B 操作数分解 | prefill-wait-entry、request-shape-response |
| `q02_control_floor.py` | 同一 mask 程序的零搬运控制窗口 | [probe-control](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/Q02-tma-request/probe-control.md) |

`prefill-wait-entry` 的正式等待 hint 两轮用到的 `q02_wait_hint.py` 不在这里，只保存在这两个轮次的 `source/` 中。10-22 的两轮预填间隔使用轮次目录里的 `measure.py`／`capture.py` 执行和评分。

## 现行代码与历史快照

每个轮次目录的 `source/` 保存当轮的源码版本（其中有准备后未执行的轮次）。本目录链接的是现行版本，二者不一定相同：

- 10 个轮次中的 `q02_tma_stride.py` 快照与现行版本完全相同；
- `q02_prefill_observer.py` 有 3 个快照版本，`q02_wait_entry.py` 有 2 个，都早于现行版本；
- 现行的 `tma_gemm_sources.cu` 已包含预填间隔的展开版；循环版保存在 10-18 预填间隔轮的 `build-attempt-001/source/` 中，并由 10-22 的循环执行轮（`prefill-gap-execution-001/source/`，同一 SHA `96b6915e82bb`）复用。

各族的逐轮代码对应、快照差异与重评分命令写在 results 的族 README 中，例如 [prefill-wait-entry](../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/prefill-wait-entry/README.md#代码)。

本目录用于阅读现行实现。重建或重跑历史测量时使用当轮快照，旧路径入口需要时使用兼容视图（`flat_layout_view.py`）。已有数据的 CPU 重评分，按族 README 中经过验证的入口执行。
