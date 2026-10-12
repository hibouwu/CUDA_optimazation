# GPU 性能建模参考资料

本目录服务于 ModelEvaluation 的 GPU 性能建模，[GEMM](../gemm/README.md) 是当前主线。资料按操作服务、执行依赖、流水线、完整 kernel 和算子图组织；硬件常数、测量窗口和模型假设需要与本项目的目标条件对应。

`papers/` 保存论文与学位论文 PDF；`repositories/` 保存代码；`research/` 保存官方文档、会议 slides 和作者 poster。[sources.json](sources.json) 统一记录原始链接、版本、下载状态、文件校验值及代码 commit。下载文件保留在本项目内，Git 跟踪索引和清单。

当前保存 17 份论文、学位论文或技术报告 PDF（16 项研究，GPU cores 一项有两个版本）、2 份 slides/poster、4 篇网页和 11 个代码仓库快照，共约 540 MiB。

## 优先阅读

| 顺序 | 资料 | 对当前模型最有用的内容与边界 |
|---|---|---|
| 1 | [TileSight，2026](papers/2026-tilesight.pdf) · [作者来源](https://arxiv.org/abs/2607.22432) | 先读 §3.2–3.5、§5.3、§7：静态 tile DAG、资源成本、流水重叠与缓存流量，主要对应 L2–L3。预测在 CPU 进行，硬件参数需要校准；不显式建模频率和未公开的 warp/CTA 调度，不能视为已解决长 K 与关键 CTA。当前为预印本。 |
| 2 | [Understanding Latency Hiding on GPUs，2016](papers/2016-volkov-latency-hiding.pdf) · [大学来源](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.html) | 先读第 III–IV 章：区分延迟、吞吐、独立指令并发和 occupancy；用来检查 L0 服务到 L1 时间的组合是否符合极限情况。经典 GPU 的硬件价格不能直接移植到 GH200。 |
| 3 | [Dissecting and Modeling the Architecture of Modern GPU Cores，MICRO 2025](papers/2025-modern-gpu-cores-final.pdf) · [机构来源](https://upcommons.upc.edu/entities/publication/e31cdd46-ce96-41ba-8ba0-1919bb7adeb4) | 先读 §5–7：发射、依赖控制、RF、访存 pipeline 与 SASS 控制字段，直接服务 L0–L1。使用增强的动态 trace，正式稿验证 Turing、Ampere 和 Blackwell，未验证 Hopper/GH200，也没有 WGMMA/TMA 模型。 |
| 4 | [Warpspeed，JPDC 2023](papers/2023-warpspeed.pdf) · [作者来源](https://arxiv.org/abs/2204.14242) | 从静态地址表达式估计各级缓存与内存流量，分开逻辑请求和物理流量。主要评价 V100/A100 的访存型 kernel，不能直接替代 TMA/WGMMA GEMM 模型。 |
| 5 | [Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs，2025](papers/2025-twill.pdf) · [作者来源](https://arxiv.org/abs/2512.18134) | 先读 §3–4：Twill 的依赖约束、资源预约和流水调度可用于 L2。“最优”相对于给定成本和约束；可变延迟 streaming 操作按零延迟处理，流水深度交给外部调优，不能当作完整时间预测器。 |

## 其余论文

| 资料与版本 | 用途与适用边界 |
|---|---|
| [Roofline，Berkeley 技术报告 2008](papers/2008-roofline-report.pdf) · [来源](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html) | 计算与访存的吞吐上界、时间下界及瓶颈基线；不描述依赖、启动、排空或关键 CTA。下载的是技术报告，与 CACM 2009 的短标题正式论文区分。 |
| [Hong/Kim MWP/CWP，ISCA 2009](papers/2009-hong-kim-mwp-cwp.pdf) · [来源](https://hparch.gatech.edu/papers/hong_isca09.pdf) | 访存并行度、warp 并行度与计算重叠；早期架构的假设与常数需要重新检查。 |
| [GPUMech，MICRO 2014](papers/2014-gpumech.pdf) · [来源](https://hsienhsinlee.github.io/MARS/pub/micro14.pdf) | 按依赖和停顿区间组合执行时间；需要指令 trace 与功能仿真。 |
| [PPT-GPU: Scalable GPU Performance Modeling，CAL 2019](papers/2019-ppt-gpu.pdf) · [LANL 来源](https://laro.lanl.gov/view/pdfCoverPage?download=true&filePid=13158082150003761&instCode=01LANL_INST) | warp/CTA 到完整 kernel 的时间组合；早期模型含 profiler 提供的缓存命中率输入。下载的是 CAL 2019，当前仓库所引用的 SC 2021 全文尚未取得。 |
| [Accel-Sim，ISCA 2020](papers/2020-accelsim.pdf) · [来源](https://people.ece.ubc.ca/~aamodt/publications/papers/accelsim.isca2020.pdf) | trace 驱动的逐周期模拟与验证框架；适合做结构和机制对照。 |
| [Shared/Local Memories of Ampere，MEMSYS ’23 作者稿](papers/2023-ampere-shared-local.pdf) · [会议来源](https://www.memsys.io/wp-content/uploads/2023/09/21.pdf) | A100 的 LDGSTS 路径、shared/local memory 与 spill；正式出版记录为 2024。未给出本项目普通 LDS/STS 宽广播的完整 lane 排列模型。 |
| [Real GPU NoC Characteristics，MICRO 2024](papers/2024-real-gpu-noc.pdf) · [来源](https://people.ece.ubc.ca/~aamodt/publications/papers/realgpu-noc.micro2024.pdf) | SM、L2 和互联中的源端、目标端、地址位置与互扰，帮助分开延迟与带宽。 |
| [Forecasting GPU Performance for Deep Learning Training and Inference，ASPLOS 2025](papers/2025-neusight.pdf) · [作者来源](https://arxiv.org/abs/2407.13853) | NeuSight 的 tile 学习模型与算子图时间组合，服务 L3–L4；需要训练数据，不能解释 bank 内部机制。 |
| [Hopper 微基准，2025，arXiv v2](papers/2025-hopper-microbenchmark.pdf) · [作者来源](https://arxiv.org/abs/2501.12084) | TMA、WGMMA、DSM 等多层测量；主要硬件为 H800，迁移前需核对执行组织和计时窗口。 |
| [Analyzing Modern NVIDIA GPU cores，2025，arXiv v1](papers/2025-modern-gpu-cores-preprint.pdf) · [作者来源](https://arxiv.org/abs/2503.20481) | 上方 MICRO 2025 正式稿的早期版本；保留用于比较模型演变，计数时属于同一研究。 |
| [Microbenchmark-Driven Analytical Performance Modeling，2026](papers/2026-microbenchmark-driven.pdf) · [作者来源](https://arxiv.org/abs/2605.04178) | 微基准定价与跨架构迁移的补充对照；部分工作量来自 profiler，不能用摘要中的低误差代表 H200 静态预测效果。 |
| [Architecting the Next Generation of Asynchronous, Distributed GPUs for the AI Era，2026](papers/2026-accelsim2.pdf) · [作者来源](https://arxiv.org/abs/2608.22602) | Accel-Sim 2.0 的 TMA、WGMMA、mbarrier、cluster 及 H100/H200 模拟；输入依然包含目标程序的动态 trace。 |

## 代码入口

每个仓库的 `source.tar.gz` 是固定 commit 的源码包；`source/` 展开主要模型、探针、配置和文档。大批数据、权重、trace、第三方依赖与二进制留在压缩包中，未安装或运行这些项目。子模块不随 GitHub 源码包展开。

| 作者仓库与本地源码 | 首先查看什么 |
|---|---|
| [TileSight](https://github.com/tile-ai/TileSight) · [本地](repositories/tilesight/source/README.md) | `tilesight/modeling/program/` 的 `contract.py`、`analysis.py`、`results.py`，再读成本、缓存与调度；硬件 profile 混有实测、推导与继承参数，迁移前需检查。 |
| [Warpspeed](https://github.com/te42kyfo/warpspeed) · [本地](repositories/warpspeed/source/README.md) | `warpspeedkernel.py` 的地址描述与 `predict_metrics.py` 的流量估计。 |
| [Modern GPU Simulator](https://github.com/upc-arco/modern-gpu-simulator-micro-2025) · [本地源码](repositories/modern-gpu-simulator/source/) | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/` 的 `subcore.cc`、`warp_dependency_state.cc`、`register_file.cc`、`ldst_unit_sm.cc`。 |
| [PPT-GPU](https://github.com/NMSU-PEARL/PPT-GPU) · [本地源码](repositories/ppt-gpu/source/) | `src/warps.py`、`warp_scheduler.py`、`blocks.py`、`kernels.py`、`memory_model.py`，观察局部服务如何组合。需要指令与访存 trace，现成 ISA/配置主要到 Volta、Turing，没有 Hopper 模型声明。 |
| [GCoM](https://github.com/yonsei-hpcp/gcom) · [本地源码](repositories/gcom/source/) | `src/` 的详细 core 模型与 `configs/`；主要验证 Turing，输入为 SASS trace。 |
| [AMALi](https://github.com/ShihengCao/AMALi) · [本地源码](repositories/amali/source/) | `hardware/H100_PCIe.py`、`hardware/ISA/Hopper.py`，以及 `src/kernels.py`、`src/warps.py` 的发射、等待与 scoreboard；需要动态 trace。有 H100 配置，完整异步协议覆盖仍需逐项检查。 |
| [NeuSight](https://github.com/scai-tech/NeuSight) · [本地源码](repositories/neusight/source/) | `neusight/Model/mlp_wave_mm.py`、`mlp_wave_vec.py`、`Model/other/` 的基线，以及 `neusight/Prediction/` 的时间组合。 |
| [Accel-Sim framework](https://github.com/accel-sim/accel-sim-framework) · [本地源码](repositories/accel-sim/source/) | `util/tracer_nvbit/`、`util/tuner/`、`util/tracer_nvbit/others/spinlock_tool/`，分别用于 trace、定价和异步同步处理。 |
| [Accel-Sim 的 GPGPU-Sim 后端](https://github.com/accel-sim/gpgpu-sim_distribution) · [本地源码](repositories/accel-sim-backend/source/) | `src/abstract_hardware_model.cc` 的逐 lane 请求形成；`src/gpgpu-sim/shader.cc`、`src/gpgpu-sim/scoreboard.cc` 的发射和依赖。归档的是独立源码版本，未验证与 framework 的构建配套。 |
| [GPUNetBench](https://github.com/chrirocca/GPUNetBench) · [本地源码](repositories/gpunetbench/source/) | `SMEM_LAT-BW/`、`SM2SM/`、`MP/`、`PER_SLICE/` 的最小对照；属于硬件探针。仓库与 NoC 论文作者和主题相关，未逐项核实为论文的完整 artifact。 |
| [Hopper Benchmark](https://github.com/HPMLL/NVIDIA-Hopper-Benchmark) · [本地源码](repositories/hopper-benchmark/source/) | TMA、WGMMA、DSM 的代码与测量边界；属于硬件探针。 |

## 官方文档与研究材料

| 本地资料与原始来源 | 用途 |
|---|---|
| [CUDA Best Practices 12.9.1](research/cuda-best-practices-12.9.1.html) · [NVIDIA](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html) | bank、broadcast、coalescing 的公开规则。 |
| [Nsight Compute Profiling Guide](research/nsight-compute-profiling-guide.html) · [NVIDIA](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | metric、wavefront、replay 和计量作用域。 |
| [64 bit shared-memory wavefront 讨论](research/nvidia-wide-shared-wavefront.html) · [NVIDIA 论坛](https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-with-64-bit-data-type/293904) | 连续宽访问、128 B/cycle 与广播例外；没有提供完整 SM90 阶段映射。 |
| [Hopper TMA tutorial](research/colfax-hopper-tma.html) · [Colfax Research](https://research.colfax-intl.com/tutorial-hopper-tma/) | TMA 生产、消费和 barrier 生命周期。 |
| [GCoM slides，ISCA 2022](research/2022-gcom-slides.pdf) · [会议来源](https://www.iscaconf.org/isca2022/slides/isca22-lee-gcom.pdf) | 已取得官方 slides；[论文正式记录](https://doi.org/10.1145/3470496.3527384)，尚未取得全文。 |
| [AMALI poster，ISCA 2025](research/2025-amali-poster.pdf) · [作者来源](https://raw.githubusercontent.com/ShihengCao/AMALi/main/docs/ISCA25_poster3.pdf) | 已取得作者 poster；[论文正式记录](https://doi.org/10.1145/3695053.3731064)，尚未取得全文。 |

HTML 保存原始页面正文，图片、样式等外部资源仍可能需要联网。静态输入模型、学习模型、trace 驱动模拟器和实测探针在上表分别注明；它们的精度不能直接互相替代。模拟器的 bank/子组计数公式也只是待检验假设，不能用来覆盖本项目已经保留的反例。
