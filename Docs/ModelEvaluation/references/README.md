# GPU 性能建模参考资料

本目录服务于 ModelEvaluation 的 GPU 性能建模，[GEMM](../gemm/README.md) 是当前主线。资料按操作服务、执行依赖、流水线、完整 kernel 和算子图组织；硬件常数、测量窗口和模型假设需要与本项目的目标条件对应。

`papers/<论文名>/` 含原 PDF、Markdown 阅读版和图片；`repositories/` 是固定 commit 的源码；`research/` 是官方文档、slides 和 poster。[sources.json](sources.json) 记录来源、版本和 commit。项目对本仓库模型的对照见 [literature_comparison](../gemm/literature_comparison.md)。

## 优先阅读

| 顺序 | 资料 | 对当前模型最有用的内容与边界 |
|---|---|---|
| 1 | [TileSight，2026](papers/2026-tilesight/2026-tilesight.pdf) · [作者来源](https://arxiv.org/abs/2607.22432) | 先读 §3.2–3.5、§5.3、§7：静态 tile DAG、资源成本、流水重叠与缓存流量，主要对应 L2–L3。预测在 CPU 进行，硬件参数需要校准；不显式建模频率和未公开的 warp/CTA 调度，不能视为已解决长 K 与关键 CTA。当前为预印本。 |
| 2 | [Understanding Latency Hiding on GPUs，2016](papers/2016-volkov-latency-hiding/2016-volkov-latency-hiding.pdf) · [大学来源](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.html) | 先读第 III–IV 章：区分延迟、吞吐、独立指令并发和 occupancy；用来检查 L0 服务到 L1 时间的组合是否符合极限情况。经典 GPU 的硬件价格不能直接移植到 GH200。 |
| 3 | [Dissecting and Modeling the Architecture of Modern GPU Cores，MICRO 2025](papers/2025-modern-gpu-cores-final/2025-modern-gpu-cores-final.pdf) · [机构来源](https://upcommons.upc.edu/entities/publication/e31cdd46-ce96-41ba-8ba0-1919bb7adeb4) | 先读 §5–7：发射、依赖控制、RF、访存 pipeline 与 SASS 控制字段，直接服务 L0–L1。使用增强的动态 trace，正式稿验证 Turing、Ampere 和 Blackwell，未验证 Hopper/GH200，也没有 WGMMA/TMA 模型。 |
| 4 | [Warpspeed，JPDC 2023](papers/2023-warpspeed/2023-warpspeed.pdf) · [作者来源](https://arxiv.org/abs/2204.14242) | 从静态地址表达式估计各级缓存与内存流量，分开逻辑请求和物理流量。主要评价 V100/A100 的访存型 kernel，不能直接替代 TMA/WGMMA GEMM 模型。 |
| 5 | [Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs，2025](papers/2025-twill/2025-twill.pdf) · [作者来源](https://arxiv.org/abs/2512.18134) | 先读 §3–4：Twill 的依赖约束、资源预约和流水调度可用于 L2。“最优”相对于给定成本和约束；可变延迟 streaming 操作按零延迟处理，流水深度交给外部调优，不能当作完整时间预测器。 |

## 其余论文

| 资料与版本 | 用途与适用边界 |
|---|---|
| [Roofline，Berkeley 技术报告 2008](papers/2008-roofline-report/2008-roofline-report.pdf) · [来源](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html) | 计算与访存的吞吐上界、时间下界及瓶颈基线；不描述依赖、启动、排空或关键 CTA。下载的是技术报告，与 CACM 2009 的短标题正式论文区分。 |
| [Hong/Kim MWP/CWP，ISCA 2009](papers/2009-hong-kim-mwp-cwp/2009-hong-kim-mwp-cwp.pdf) · [来源](https://hparch.gatech.edu/papers/hong_isca09.pdf) | 访存并行度、warp 并行度与计算重叠；早期架构的假设与常数需要重新检查。 |
| [GPUMech，MICRO 2014](papers/2014-gpumech/2014-gpumech.pdf) · [来源](https://hsienhsinlee.github.io/MARS/pub/micro14.pdf) | 按依赖和停顿区间组合执行时间；需要指令 trace 与功能仿真。 |
| [PPT-GPU: Scalable GPU Performance Modeling，CAL 2019](papers/2019-ppt-gpu/2019-ppt-gpu.pdf) · [LANL 来源](https://laro.lanl.gov/view/pdfCoverPage?download=true&filePid=13158082150003761&instCode=01LANL_INST) | warp/CTA 到完整 kernel 的时间组合；早期模型含 profiler 提供的缓存命中率输入。下载的是 CAL 2019，当前仓库所引用的 SC 2021 全文尚未取得。 |
| [Accel-Sim，ISCA 2020](papers/2020-accelsim/2020-accelsim.pdf) · [来源](https://people.ece.ubc.ca/~aamodt/publications/papers/accelsim.isca2020.pdf) | trace 驱动的逐周期模拟与验证框架；适合做结构和机制对照。 |
| [Shared/Local Memories of Ampere，MEMSYS ’23 作者稿](papers/2023-ampere-shared-local/2023-ampere-shared-local.pdf) · [会议来源](https://www.memsys.io/wp-content/uploads/2023/09/21.pdf) | A100 的 LDGSTS 路径、shared/local memory 与 spill；正式出版记录为 2024。未给出本项目普通 LDS/STS 宽广播的完整 lane 排列模型。 |
| [Real GPU NoC Characteristics，MICRO 2024](papers/2024-real-gpu-noc/2024-real-gpu-noc.pdf) · [来源](https://people.ece.ubc.ca/~aamodt/publications/papers/realgpu-noc.micro2024.pdf) | SM、L2 和互联中的源端、目标端、地址位置与互扰，帮助分开延迟与带宽。 |
| [Forecasting GPU Performance for Deep Learning Training and Inference，ASPLOS 2025](papers/2025-neusight/2025-neusight.pdf) · [作者来源](https://arxiv.org/abs/2407.13853) | NeuSight 的 tile 学习模型与算子图时间组合，服务 L3–L4；需要训练数据，不能解释 bank 内部机制。 |
| [Hopper 微基准，2025，arXiv v2](papers/2025-hopper-microbenchmark/2025-hopper-microbenchmark.pdf) · [作者来源](https://arxiv.org/abs/2501.12084) | TMA、WGMMA、DSM 等多层测量；主要硬件为 H800，迁移前需核对执行组织和计时窗口。 |
| [Analyzing Modern NVIDIA GPU cores，2025，arXiv v1](papers/2025-modern-gpu-cores-preprint/2025-modern-gpu-cores-preprint.pdf) · [作者来源](https://arxiv.org/abs/2503.20481) | 上方 MICRO 2025 正式稿的早期版本；保留用于比较模型演变，计数时属于同一研究。 |
| [Microbenchmark-Driven Analytical Performance Modeling，2026](papers/2026-microbenchmark-driven/2026-microbenchmark-driven.pdf) · [作者来源](https://arxiv.org/abs/2605.04178) | 微基准定价与跨架构迁移的补充对照；部分工作量来自 profiler，不能用摘要中的低误差代表 H200 静态预测效果。 |
| [Architecting the Next Generation of Asynchronous, Distributed GPUs for the AI Era，2026](papers/2026-accelsim2/2026-accelsim2.pdf) · [作者来源](https://arxiv.org/abs/2608.22602) | Accel-Sim 2.0 的 TMA、WGMMA、mbarrier、cluster 及 H100/H200 模拟；输入依然包含目标程序的动态 trace。 |

## 同类时间预测与 GEMM 结构

| 资料与版本 | 用途与适用边界 |
|---|---|
| [A Performance Model for Warp Specialization Kernels，2025，arXiv v2](papers/2025-warp-specialization-model/2025-warp-specialization-model.pdf) · [作者来源](https://arxiv.org/abs/2506.11209) | 生产者/消费者 warp 的逐 stage 开始时刻递推，与 L2 流水最接近。在 A6000 的 2048 个配置上平均误差 4.5%、最大 21.5%。DMA/MATH 成本由两组 tile 实测拟合，没有 TMA、cluster 和频率。 |
| [Sim-FA，2026，arXiv v4](papers/2026-sim-fa/2026-sim-fa.pdf) · [作者来源](https://arxiv.org/abs/2605.00555) | warpgroup 级 TMA/WGMMA/mbarrier 模拟：23 个 GEMM 形状在 H800 上 MAPE 5.49%，FlashAttention-3 为 5.7%。输入是 Triton TTGIR 插桩 trace，不是静态预测；可作 DAG 阶段 2 的对照。 |
| [LLMCompass，ISCA 2024（arXiv v1）](papers/2024-llmcompass/2024-llmcompass.pdf) · [作者来源](https://arxiv.org/abs/2312.03134) | 算子 tile-by-tile 模拟加 mapper，再组合到 LLM 推理，对应 L3–L4。硬件用参数化计算单元和存储层级描述，不涉及 bank、异步协议和频率。下载版报告算子平均误差 10.4%，与正式稿数字不同。 |
| [Habitat，USENIX ATC 2021](papers/2021-habitat/2021-habitat.pdf) · [会议来源](https://www.usenix.org/conference/atc21/presentation/yu) | wave scaling：把一张 GPU 上的实测 kernel 时间按 SM 数和带宽比例迁移到另一张。必须先在源 GPU 上实测，可作"wave 数×单 wave 时间"基线。 |
| [tritonBLAS，2025](papers/2025-tritonblas/2025-tritonblas.pdf) · [作者来源](https://arxiv.org/abs/2512.04226) | 层级 tile 划分，时间取 wave 数×单 tile 延迟。在 MI300X 上用于选择配置，达到自动调优的 95% 以上性能；不报告绝对时间误差。 |
| [WaveTune，2026](papers/2026-wavetune/2026-wavetune.pdf) · [作者来源](https://arxiv.org/abs/2604.10187) | H100 上 GEMM 延迟随 wave 数和 K 循环次数分段双线性。每个配置需要采样锚点，可用来对照长 K 关系是否线性。 |
| [Stream-K，2023](papers/2023-stream-k/2023-stream-k.pdf) · [作者来源](https://arxiv.org/abs/2301.03598) | wave 量化，以及按 K 迭代均分工作；改变 CTA 到 SM 的映射，属于 L3。是调度方法，不是时间模型。 |
| [Hierarchical Roofline for Deep Learning，2020，arXiv v4](papers/2020-hierarchical-roofline-dl/2020-hierarchical-roofline-dl.pdf) · [作者来源](https://arxiv.org/abs/2009.05257) | L1、L2、HBM 多层 roofline 与 Tensor Core 上界，V100。应用流量依赖 Nsight Compute；本项目没有计数器，只能代入逻辑流量。 |
| [TileLang，2025，arXiv v2](papers/2025-tilelang/2025-tilelang.pdf) · [作者来源](https://arxiv.org/abs/2504.17577) | `T.copy`、`T.gemm`、`T.Pipelined` 等 tile 算子及降级流程，是 DAG 前端的语义来源；不含性能模型。 |

## 微基准与片上服务

| 资料与版本 | 用途与适用边界 |
|---|---|
| [Dissecting Volta，2018 技术报告](papers/2018-volta-microbenchmark/2018-volta-microbenchmark.pdf) · [作者来源](https://arxiv.org/abs/1804.06826) | SASS 控制字、两组 64 bit 寄存器 bank 的冲突、SMEM 冲突延迟；探针写法可借用于 L0 的 RF 与 SMEM。仅覆盖 V100。 |
| [Dissecting Turing T4，2019 技术报告](papers/2019-turing-microbenchmark/2019-turing-microbenchmark.pdf) · [作者来源](https://arxiv.org/abs/1903.07486) | Volta 报告的续篇，涉及寄存器 reuse、指令延迟与内存层级；T4。 |
| [Dissecting GPU Memory Hierarchy，TPDS 2017（arXiv v2）](papers/2017-gpu-memory-hierarchy/2017-gpu-memory-hierarchy.pdf) · [作者来源](https://arxiv.org/abs/1509.02308) | pointer-chasing 缓存结构探针、SMEM bank 冲突对延迟的影响；Fermi 到 Maxwell，硬件常数不能迁移。 |
| [Demystifying Ampere，HPEC 2022（arXiv v1）](papers/2022-ampere-microbenchmark/2022-ampere-microbenchmark.pdf) · [作者来源](https://arxiv.org/abs/2208.11174) | A100 上 PTX 与对应 SASS 的指令周期及各级内存延迟，短文。 |
| [Dissecting Tensor Cores，TPDS 2023（arXiv v3）](papers/2023-tensor-core-microbenchmark/2023-tensor-core-microbenchmark.pdf) · [作者来源](https://arxiv.org/abs/2206.02874) | `ldmatrix`、`mma`、`mma.sp` 的延迟与吞吐；止于 Ampere，没有 WGMMA。 |

## 频率与功率

| 资料与版本 | 用途与适用边界 |
|---|---|
| [Input-Dependent Power Usage in GPUs，2024](papers/2024-input-dependent-power/2024-input-dependent-power.pdf) · [作者来源](https://arxiv.org/abs/2409.18324) | GEMM 输入的数值分布、位相似度、位置和稀疏性可使功率变化近 40%。以 A100 为主，H100、V100 部分复现；特意选择不降频的尺寸，所以没有给出对时间的影响。 |
| [Not All GPUs Are Created Equal，SC22](papers/2022-gpu-variability/2022-gpu-variability.pdf) · [作者来源](https://arxiv.org/abs/2208.11035) | 同型号 GPU 之间 SGEMM 时间、频率、功率与温度的差异，归因于功耗管理加制造差异；V100/MI60 集群。可用于确定同卡配对的必要性和跨卡误差下限。 |
| [Power-capping H100 and H200，2026，arXiv v2](papers/2026-h100-h200-power-capping/2026-h100-h200-power-capping.pdf) · [作者来源](https://arxiv.org/abs/2604.11391) | 功率上限下的 SM 降频、显存功率份额与 DGEMM/Triad 表现。对象是 H100/H200，不是 GH200 的 CPU–GPU 共享功率。 |
| [AccelWattch，MICRO 2021](papers/2021-accelwattch/2021-accelwattch.pdf) · [作者来源](https://paragon.cs.northwestern.edu/papers/2021-MICRO-AccelWattch-Kandiah.pdf) | 逐周期功率模型，包含常数与静态功率、DVFS，接入 Accel-Sim；在 Volta 上验证。 |
| [GPGPU Power Modeling for Multi-Domain DVFS，HPCA 2018](papers/2018-gpgpu-power-dvfs/2018-gpgpu-power-dvfs.pdf) · [作者来源](https://www.inesc-id.pt/wp-content/uploads/2024/03/HPCA2018_Ilic-1.pdf) | 核心与显存双频率下的功率模型，考虑电压的非线性；Kepler 到 Pascal，需要性能计数器。 |

<a id="论文阅读版"></a>

## 论文阅读版

阅读版由 MinerU 4.0.12 在本地转换（`basic` 档位，未上传论文），并按原页做过抽查修正。独立公式、表格和代码默认显示原页裁剪，自动识别文本放在折叠区；行内公式未逐项校对，精确引用以原 PDF 为准。

## 代码入口

本地只保留主要源码、配置和文档，未安装或运行；数据、权重、trace 和子模块需按 sources.json 中的 commit 重新获取。

| 作者仓库与本地源码 | 首先查看什么 |
|---|---|
| [TileSight](https://github.com/tile-ai/TileSight) · [本地](repositories/tilesight/README.md) | `tilesight/modeling/program/` 的 `contract.py`、`analysis.py`、`results.py`，再读成本、缓存与调度；硬件 profile 混有实测、推导与继承参数，迁移前需检查。 |
| [Warpspeed](https://github.com/te42kyfo/warpspeed) · [本地](repositories/warpspeed/README.md) | `warpspeedkernel.py` 的地址描述与 `predict_metrics.py` 的流量估计。 |
| [Modern GPU Simulator](https://github.com/upc-arco/modern-gpu-simulator-micro-2025) · [本地源码](repositories/modern-gpu-simulator/) | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/` 的 `subcore.cc`、`warp_dependency_state.cc`、`register_file.cc`、`ldst_unit_sm.cc`。 |
| [PPT-GPU](https://github.com/NMSU-PEARL/PPT-GPU) · [本地源码](repositories/ppt-gpu/) | `src/warps.py`、`warp_scheduler.py`、`blocks.py`、`kernels.py`、`memory_model.py`，观察局部服务如何组合。需要指令与访存 trace，现成 ISA/配置主要到 Volta、Turing，没有 Hopper 模型声明。 |
| [GCoM](https://github.com/yonsei-hpcp/gcom) · [本地源码](repositories/gcom/) | `src/` 的详细 core 模型与 `configs/`；主要验证 Turing，输入为 SASS trace。 |
| [AMALi](https://github.com/ShihengCao/AMALi) · [本地源码](repositories/amali/) | `hardware/H100_PCIe.py`、`hardware/ISA/Hopper.py`，以及 `src/kernels.py`、`src/warps.py` 的发射、等待与 scoreboard；需要动态 trace。有 H100 配置，完整异步协议覆盖仍需逐项检查。 |
| [NeuSight](https://github.com/scai-tech/NeuSight) · [本地源码](repositories/neusight/) | `neusight/Model/mlp_wave_mm.py`、`mlp_wave_vec.py`、`Model/other/` 的基线，以及 `neusight/Prediction/` 的时间组合。 |
| [Accel-Sim framework](https://github.com/accel-sim/accel-sim-framework) · [本地源码](repositories/accel-sim/) | `util/tracer_nvbit/`、`util/tuner/`、`util/tracer_nvbit/others/spinlock_tool/`，分别用于 trace、定价和异步同步处理。 |
| [Accel-Sim 的 GPGPU-Sim 后端](https://github.com/accel-sim/gpgpu-sim_distribution) · [本地源码](repositories/accel-sim-backend/) | `src/abstract_hardware_model.cc` 的逐 lane 请求形成；`src/gpgpu-sim/shader.cc`、`src/gpgpu-sim/scoreboard.cc` 的发射和依赖。归档的是独立源码版本，未验证与 framework 的构建配套。 |
| [GPUNetBench](https://github.com/chrirocca/GPUNetBench) · [本地源码](repositories/gpunetbench/) | `SMEM_LAT-BW/`、`SM2SM/`、`MP/`、`PER_SLICE/` 的最小对照；属于硬件探针。仓库与 NoC 论文作者和主题相关，未逐项核实为论文的完整 artifact。 |
| [Hopper Benchmark](https://github.com/HPMLL/NVIDIA-Hopper-Benchmark) · [本地源码](repositories/hopper-benchmark/) | TMA、WGMMA、DSM 的代码与测量边界；属于硬件探针。 |

## 官方文档与研究材料

| 本地资料与原始来源 | 用途 |
|---|---|
| [CUDA Best Practices 12.9.1](research/cuda-best-practices-12.9.1.html) · [NVIDIA](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html) | bank、broadcast、coalescing 的公开规则。 |
| [Nsight Compute Profiling Guide](research/nsight-compute-profiling-guide.html) · [NVIDIA](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | metric、wavefront、replay 和计量作用域。 |
| [64 bit shared-memory wavefront 讨论](research/nvidia-wide-shared-wavefront.html) · [NVIDIA 论坛](https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-with-64-bit-data-type/293904) | 连续宽访问、128 B/cycle 与广播例外；没有提供完整 SM90 阶段映射。 |
| [Hopper TMA tutorial](research/colfax-hopper-tma.html) · [Colfax Research](https://research.colfax-intl.com/tutorial-hopper-tma/) | TMA 生产、消费和 barrier 生命周期。 |
| [GCoM slides，ISCA 2022](research/2022-gcom-slides.pdf) · [Markdown](research/2022-gcom-slides.md) · [会议来源](https://www.iscaconf.org/isca2022/slides/isca22-lee-gcom.pdf) | 已取得官方 slides；[论文正式记录](https://doi.org/10.1145/3470496.3527384)，尚未取得全文。 |
| [AMALI poster，ISCA 2025](research/2025-amali-poster.pdf) · [Markdown](research/2025-amali-poster.md) · [作者来源](https://raw.githubusercontent.com/ShihengCao/AMALi/main/docs/ISCA25_poster3.pdf) | 已取得作者 poster；[论文正式记录](https://doi.org/10.1145/3695053.3731064)，尚未取得全文。 |

HTML 保存原始页面正文，图片、样式等外部资源仍可能需要联网。静态输入模型、学习模型、trace 驱动模拟器和实测探针在上表分别注明；它们的精度不能直接互相替代。模拟器的 bank/子组计数公式也只是待检验假设，不能用来覆盖本项目已经保留的反例。
