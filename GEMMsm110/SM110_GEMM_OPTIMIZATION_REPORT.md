# SM110 GEMM Optimization Report

本文记录 Thor/SM110 上 FP16 x FP16 -> FP32 TCGen05 GEMM 的对抗式审查和优化过程。报告按阶段逐段写出测量口径、技术思路、收益来源和失败点；没有重新验证的历史结论不会混成最终结论。本轮已经闭环的是脚本清理、图表重跑、1024/2048 两个硬目标是否达到 cuBLAS Tensor Core 90%，以及主报告集合与不稳定实验集合的命名整理。

后续 reference 口径已经升级：`cublas_tc` 源码从 `cublasGemmEx` 改为 cuBLASLt Matmul heuristic，并通过 `cublasLtMatmulAlgoGetHeuristic` 选择算法。当前已归档 CSV 和图还没有在这个新口径下重跑，因此已有 0.927x/0.917x 数字仍按旧 `cublasGemmEx` reference 解释；重新编译并重跑后，新 CSV 的 `Reference` 字段会变为 `cuBLASLt Matmul heuristic`。

第一步审查的是数据本身，而不是 kernel。原来的 `results/gemm_sm110/figures/gemm_tensor_core_gflops.svg` 虽然标题是 Tensor Core GEMM sweep，但对应的 CSV 不是一次干净的全量运行：128/256/512 主要来自少量 trial，1024/2048/4096 混有多轮和手工追加行；更关键的是，脚本的 `all` 集合没有包含后续推荐路径 `tc5a/tc5b`，而 plotter 只是在图例里后来补了 `tc5b`。这会造成一个很危险的错觉：图看起来像在比较最终版本，其实最终版本并不是由同一个 sweep 机制稳定产出的。因此我先修改 `scripts/run_gemm_sm110_experiments.sh`，让它显式支持 `core`、`unstable` 和 `nvfp4` 三种集合；`core` 只放 FP32 主报告里要反复比较的版本，`unstable` 保留那些有教学价值但会 timeout 或 launch failure 的阶段，`nvfp4` 单独放 `tc6`，避免把 packed NVFP4 输出和 FP32 输出混在同一条验收线上。

第二步是反向验证哪些版本应该从主图里移走。原始 `all` 全量跑在 1024 上复现了 `tc4c` 的 `cudaErrorLaunchFailure`，也出现过 `tc5b` 的 2-SM cluster path 偶发 launch failure；2048/4096 上 `tc4b/tc4c` 多次进入 120 秒 timeout，`tc5c` 在 4096 也出现过 timeout。后来收窄后的 `core` 仍暴露了 `tc3` 在 4096 的偶发 timeout，`tc6` 在 4096 第 9 轮和 retry 中也 timeout。这个审查结论很重要：这些版本不是被删除源码，而是从主报告集合中移出。`tc3/tc4b/tc4c/tc5c` 仍然保留为阶段性实验和教学材料，`tc6` 保留为 NVFP4 fused epilogue 路线；但它们不再参与 FP32 主图的稳定性结论。对应代码也做了清理：批量脚本、C++ backend registry 和 `build_and_run.sh` 的默认入口都以 `core` 为主，`all` 只作为显式诊断入口保留；主图另加 `scripts/run_sm110_gemm_core_sweep.sh` 这个明确命名的包装脚本，避免后续复现时误选 suite。

第三步是重跑主图需要的完整数据。当前主 CSV 是 `results/gemm_sm110/gemm_sm110_sweep.csv`，由 `GEMM_SUITE=core` 的 128 到 4096 结果整理而来。SM110 性能主图现在只画 `cublas_tc`、`cutlass`、`tc2a`、`tc2b`、`tc4a`、`tc5a`、`tc5b` 七条线；`tc0` 保留为 WMMA correctness baseline，但不再作为性能基线进入主图。128 上 `cutlass/tc4a` 因 tile 约束报告 unavailable，这是显式不可用，不是缺测；plotter 会按 `Matched=1` 聚合，因此不会把 0 值画成性能点。对应图是 `results/gemm_sm110/figures/gemm_tensor_core_gflops.svg` 和 `results/gemm_sm110/figures/gemm_tensor_core_ratio_to_cublas_tc.svg`。我还修改了横轴标签策略，现在 128、256、512、1024、2048、4096 会全部以精确数字显示，避免 4096 只表现成一个无标签数据点。为了让归档名表达真实口径，脚本现在还会生成 `sm110_gemm_core_128_4096_10trials.csv`、`sm110_gemm_core_128_4096_10trials_gflops.svg` 和 `sm110_gemm_core_128_4096_10trials_ratio_to_cublas_tc.svg` 这组别名；旧名保留给已有引用，新名用于报告和长期比较。

第四步是检查 1024 和 2048 的硬门槛。按当前 CSV 聚合，1024 方阵中 `tc5b` 10/10 轮 `matched=1`，平均 95,183 GFLOP/s，平均 ratio 为 0.927，最小 ratio 为 0.925。这个尺寸应继续推荐 `tc5b`，因为它的 2-SM overlapped cluster path 把小尺寸 tile 数不足的问题转化成更粗的 M256N256K128 cluster tile，并把 epilogue 和下一 tile mainloop 重叠起来。2048 方阵中 `tc5a` 10/10 轮 `matched=1`，平均 119,550 GFLOP/s，平均 ratio 为 0.917，最小 ratio 为 0.898；`tc5b` 在 2048 只是同一个 `tc5a` fallback 的单独运行，平均 ratio 为 0.920，最小 ratio 为 0.909。因此 2048 的算法推荐仍写成 `tc5a`，但不能再把 `tc5a` 这一轮描述为“每轮都超过 0.90x”。这个结论比“挑最好看的线”更保守，也更符合对抗式审查的要求。

本轮留下的不足也要明说。第一，当前使用 `SKIP_BUILD=1` 复用了现有 `GEMMsm110/build/gemm_sm110_bench`，因为当前环境没有找到默认 `CUTLASS_ROOT=/xplorer/shijy/third_party/cutlass`，也没有用其它目录里的 CUTLASS 4.3.3 checkout 冒充 README 要求的 4.5.2；脚本已经支持这个入口，但后续如果要声明源码到二进制完全可复现，仍需要补齐 CUTLASS 4.5.2 checkout 并重新编译。第二，4096 上 `tc5a/tc5b` 的单轮波动比 1024/2048 大，虽然不是这次硬目标，但报告需要把它解释成大尺寸下 worker 数、tile 数、TMA/TMEM 固定成本和系统噪声共同作用的结果，而不是把 4096 的最好轮次当成稳定结论。第三，`unstable` 集合里的 `tc3/tc4b/tc4c/tc5c` 目前是隔离和记录，不是源码级修复；它们保留教学价值，但在重新证明稳定性之前不能进入主图。

从阶段路线看，`tc0` 的价值不是性能，而是建立一个完全自有的 Tensor Core correctness baseline。它只用传统 CUDA WMMA，不涉及 TMA、TCGen05、TMEM 或 cluster，所以当后续 raw PTX 路径出错时，`tc0` 可以把问题限定在新引入的搬运、descriptor、barrier 或 accumulator 读回逻辑里。阶段 1 的 `tc1a/tc1b` 则是一次必要但没有进入主结论的 bring-up：它们尝试用 linear/16B SMEM descriptor 接上 2D/3D TMA 和 TCGen05，但在修正输入生成的 unsigned wrap 问题后没有通过 finite-input correctness。因此这里不能再把 `tc1` 当作一个有效性能点，也不能用它和 `tc2` 的差异直接讲 bank conflict 收益。正确做法是把 `tc1` 留作 descriptor 映射问题的历史记录，主线从已经 matched 的 SW128 路径重新开始。

阶段 2 的 `tc2a/tc2b` 是第一个可以信任的 raw TCGen05/TMEM 路径。它们把共享内存布局切到 128B swizzle，使 TMA 写入、TCGen05 读取和 TMEM accumulator 生命周期在同一个最小内核里闭环。这个阶段的重点是“少做事”：不引入多 stage pipeline，不做 warp specialization，也不做 persistent scheduling。这样它的性能不高，但结论干净，能证明 CUtensorMap 坐标、mbarrier phase、`tcgen05.mma`、TMEM readback 和 FP32 store 的基础链路是对的。后续 `tc3/tc4/tc5` 的每一次性能变化，都应该回到这条正确性链路上解释，而不是把多个变量同时混在一起。

阶段 3 的 `tc3` 在 `tc2a` 之上加入 double buffer、多 stage TMA prefetch 和 load/compute overlap，目标是隐藏 TMA 和 barrier 等待延迟。这个方向在中等尺寸上能看到收益，但它仍然是非 persistent、非 warp-specialized 的单一 pipeline；当矩阵变大到 4096 时，本轮复测暴露了 timeout 风险。因此 `tc3` 的教学意义大于主报告意义：它说明 latency hiding 是必要的，但只靠多 stage pipeline 还不足以构成最终稳定路径。把 `tc3` 移到 `GEMM_SUITE=unstable`，不是否定这一步优化，而是避免主图把偶发 timeout 的版本包装成可推荐版本。

阶段 4 验证了两个容易混淆的问题：warp specialization 和 2-SM cluster MMA。`tc4a` 把 TMA producer、MMA consumer 和 epilogue/readback 拆开，形成当前主图里仍然保留的 warp-specialized 对照。`tc4b/tc4c` 则重新校正了 2-SM `cta_group::2` 的 accumulator ownership：两个 CTA 不是按 N 方向各算半个输出，而是各自负责自己的 M half，并通过 B 的 N half 分区共同形成完整 N 输出。修正后它们能够 matched，但 1024/2048 上 cluster tile 数少，且没有 overlapped epilogue；2048/4096 复测又出现 timeout。因此它们保留为 `unstable` 对照，用来说明 2-SM 路径的正确 ownership 和同步代价，不进入 FP32 主图的稳定推荐集合。

阶段 5 才是这轮接近 cuBLAS 的主线。`tc5c` 先把 `tc4a` 的计算路径改造成 resident static persistent scheduler，证明 persistent worker 能降低启动和调度层面的空转，但它的 epilogue 仍然拖后腿。`tc5a` 在这个基础上加入 6-warp 分工、4-stage mainloop、双 TMEM accumulator buffer 和 overlapped epilogue，让上一个 tile 的 readback/store 与下一个 tile 的 mainloop 重叠；这就是 2048 推荐 `tc5a` fast path 的原因。`tc5b` 再针对 1024 的 tile 数不足问题加入 2-SM overlapped cluster fast path，并在其他尺寸回退到 `tc5a`。当前 CSV 显示，1024 用 `tc5b` 能 10/10 matched 且最小 ratio 仍高于 0.90；2048 的 `tc5a` 平均 ratio 为 0.917，但最低单轮为 0.898，所以 2048 只能写成平均/中位数达到 90% 量级，不能写成每轮都严格过线。

阶段 6 的 `tc6` 必须单独讲，因为它不是 FP32 D 矩阵输出。它复用 resident persistent mainloop，但在 TMEM readback 阶段直接写 packed E2M1 value 和 E4M3 block scale，目标是验证 fused NVFP4 epilogue 的数据通路和量化 correctness。它的 GFLOP/s 仍按 GEMM 数学量打印，方便观察 fused epilogue 代价，但 ratio 不能和 FP32 cuBLAS 画在同一张主图上。本轮 `tc6` 在 4096 复测中出现 timeout，因此被放到 `GEMM_SUITE=nvfp4`，作为单独路线继续复测，而不是混入 `gemm_tensor_core_gflops.svg`。

最终对这张图的解释应该很明确：`gemm_tensor_core_gflops.svg` 现在不是“所有写过的 backend 大杂烩”，而是 FP32 主报告性能集合的 128、256、512、1024、2048、4096 完整比较。它保留 cuBLAS/CUTLASS reference、SW128 最小 TCGen05 路径、warp-specialized 对照和当前推荐路径；`tc0` 这类传统 WMMA correctness baseline、已知不稳定版本或输出语义不同的实验版都不进入主图。这样图上的每条线都能回答一个清楚的问题：它在同一套输入、同一套 correctness reference、同一套 trial 规则下，是否稳定地产出 FP32 GEMM 结果，以及距离同进程 cuBLAS Tensor Core 还有多少。

测量纪律是这条优化线最重要的第一课。GEMM 优化很容易出现“某一轮结果很好看”的错觉，尤其是在同一个 CSV 里混入不同 trial 数、不同 backend 集合、甚至不同输出语义时。这里要求每条 FP32 性能线都满足三个条件：同一组 finite 输入，同一个 cuBLAS Tensor Core reference，同样的 warmup/repeat 和独立 trial 规则。这样做的代价是一些有趣但不稳定的版本会被移出主图，看起来少了很多“探索故事”；收益是 1024 和 2048 的 90% 门槛可以被复查，而不是靠一次偶然的最快样本支撑结论。

`tc0` 到 `tc2` 这一段的教学重点是 correctness chain。`tc0` 用 WMMA 建立一个不依赖 TMA/TCGen05/TMEM 的自有基线，它慢但简单，能帮助判断问题是不是出在更底层的 SM110 数据通路。`tc1a/tc1b` 尝试 linear SMEM descriptor，是合理的 bring-up 顺序，因为如果 linear descriptor 能先跑通，后续再引入 swizzle 时变量更少；但修正输入生成后它们没有通过 finite-input correctness，所以不能继续把它们画成有效性能点。真正可用的最小 raw 路径是 `tc2a/tc2b`，它们把 SMEM descriptor 切到 SW128，让 TMA 写入和 TCGen05 读取的布局对齐。这里的不足也很明确：`tc2` 只有单 stage，不擅长隐藏 TMA 和 barrier 延迟，因此它是正确性基线，不是性能答案。

`tc3` 引入 pipeline 的思路很自然：当一个 K tile 的 MMA 在吃当前 SMEM stage 时，下一份 A/B 应该通过 TMA 装入另一块 SMEM buffer。这个设计的优点是把 load/compute 从串行相加改成稳态取最大值，能解释为什么 `tc3` 比 `tc2` 快；缺点是它还没有解决两个问题。第一，epilogue 仍然在 mainloop 之后串行执行，TMEM readback 和 FP32 store 还是尾部成本。第二，非 persistent grid 在不同尺寸下的尾波和调度开销仍然明显，本轮 4096 复测还暴露了 timeout 风险。所以 `tc3` 的正确位置是教学阶段：它说明 pipeline 是必要条件，但不是最终主线。

`tc4a` 的 warp specialization 是对 `tc3` 的下一次拆解。TMA producer、MMA consumer 和 epilogue/readback 如果都由同一组 warp 交替执行，就会把长延迟操作和计算发射纠缠在一起；拆开后，每个 warp 的职责更稳定，barrier 的意义也更清楚。它进入 `core` 不是因为性能已经足够，而是因为它是一个稳定的调度对照：它能告诉我们在不引入 2-SM cluster、不引入 persistent scheduling 的情况下，单纯分工能带来多少收益。对抗式审查的结论是，`tc4a` 稳定但不够快；2048 只有约 0.736x cuBLAS，说明 mainloop 分工之后，epilogue 和整体 tile 调度仍然是主要瓶颈。

`tc4b/tc4c` 的价值在于修正 2-SM cluster 的理解，而不是成为当前推荐路径。早期最容易犯的错误是把两个 CTA 想象成按 N 方向左右切输出；实际的 accumulator ownership 要按每个 CTA 的 M half 来理解，两个 CTA 共同形成完整 N 输出。修正 ownership 后，`cta_group::2` 路径可以通过 correctness，也能在一些尺寸上看到 cluster tile 的收益。但它的缺点同样尖锐：cluster tile 更大，1024/2048 上 tile 数更少，尾波更重；没有 overlapped epilogue 时，TMEM readback 和 store 仍然压在 mainloop 后面；更重要的是本轮复测出现 timeout。因此 `tc4b/tc4c` 被清理到 `unstable`，保留其教学价值，移出主报告性能线。

公开 tc5 版本按主图优先排序：`tc5a/tc5b` 是 `GEMM_SUITE=core` 主图中的推荐路径，放在列表最前；`tc5c/tc5d/tc5e/tc5f/tc5g/tc5h/tc5i/tc5j` 是不进主图的阶段或调参版本，放在后面。`tc5c` 是 persistent scheduling 的基线，它使用 resident worker 和 static grid stride，让 SM 上常驻的 CTA 持续处理固定间隔的 output tile，减少普通 grid 在小尺寸和尾波下的调度浪费。`tc5d` 到 `tc5h` 用来比较 TileN、TileK 和 stage 数，`tc5i/tc5j` 用来比较 overlapped epilogue 的 tile shape。

这次命名整理的对应关系是：旧 `tc5h` 改名为现在的 `tc5a`，旧 `tc5n` 改名为现在的 `tc5b`；原来的 static persistent `tc5a` 后移为 `tc5c`，其余不进主图的调参版本顺延。

`tc5a` 是 2048 推荐路径，关键不是又换了一个 tile 名字，而是把 epilogue 从纯尾部成本变成可重叠成本。它使用 6-warp 分工、4-stage mainloop 和双 TMEM accumulator buffer：当前 tile 完成后，epilogue warp 可以从一个 accumulator 读回并写 GMEM，同时 TMA/MMA warp 在另一个 accumulator 上推进下一个 tile。这样做的优点是直接攻击 `tc3/tc4/tc5c` 共同留下的尾部瓶颈；缺点是资源和同步复杂度上升，worker 数、TMEM buffer、mbarrier phase 和 store 指令都必须配合。当前 2048 的 10 轮平均 ratio 为 0.917、最低单轮为 0.898，说明这条复杂度是值得的，但单轮稳定性仍受运行波动影响。

`tc5b` 是 1024 推荐路径，它不是简单替代 `tc5a`，而是尺寸特化。1024 的问题不是单个 tile 算不动，而是 tile 数少，固定成本和尾波比例高；因此 `tc5b` 在精确 1024 方阵上走 2-SM overlapped cluster fast path，用更粗的 M256N256K128 cluster tile 和 tile0/tile1 overlap 去增加有效工作粒度。它在其它尺寸回退到 `tc5a`，所以 2048 上 `tc5b` 和 `tc5a` 的差异只是同一 fast path 的两组独立运行样本，不应解释成新的 2048 算法收益。这个结论体现了对抗式审查的原则：推荐 backend 要看路径本身和重复样本，而不是只看某一组均值。

最后，`tc6` 展示的是另一条产品路线：fused NVFP4 epilogue。它的优点是避免先写 FP32 D 再另起量化 kernel，能把 TMEM readback 和 packed E2M1/E4M3 scale 写出合在一起；缺点是输出语义已经变了，不能和 FP32 cuBLAS 放在同一张 ratio 图里。把 `tc6` 放进 `nvfp4` 集合，是为了避免报告读者误解：它不是“FP32 GEMM 比 cuBLAS 更快或更慢”的证据，而是“TCGen05 mainloop 后直接接量化 epilogue 是否可行”的证据。后续如果继续优化 `tc6`，也应该使用独立图和独立 correctness 指标。

本轮最终验收不是看某一张图是否“画出来了”，而是看证据链是否闭合。性能门槛的证据来自 `results/gemm_sm110/sm110_gemm_core_128_4096_10trials.csv`：1024 的推荐 backend 是 `tc5b`，10 个 trial 全部 `matched=1`，平均 ratio 0.927，最小 ratio 0.925；2048 的推荐 fast path 是 `tc5a`，10 个 trial 全部 `matched=1`，平均 ratio 0.917，最小 ratio 0.898。命名整理的证据来自三类入口同时一致：批量推荐入口是 `scripts/run_sm110_gemm_core_sweep.sh`，通用入口 `scripts/run_gemm_sm110_experiments.sh` 和单次入口 `GEMMsm110/build_and_run.sh` 默认都指向 `core`，C++ registry 也认识 `core/unstable/nvfp4` 三个分组。图表证据来自 `sm110_gemm_core_128_4096_10trials_gflops.svg` 和 `sm110_gemm_core_128_4096_10trials_ratio_to_cublas_tc.svg`；旧的 `gemm_tensor_core_gflops.svg` 保留，是为了兼容已有引用，不再作为唯一规范名。

最后一次对抗式审查还发现一个容易遗漏的工程问题：仓库顶层 `.gitignore` 原本忽略整个 `results/`，所以新命名 CSV/SVG 虽然已经生成，却不会出现在工作树状态里。现在 `.gitignore` 只精确放行 SM110 主报告集合的 CSV 和两张 SVG，raw run、timeout partial CSV 和其它实验目录仍然保持忽略。这一点很重要，因为“整理命名”如果不能留下可追踪产物，就只是本地文件整理；现在新旧文件名的关系、默认脚本入口和报告里的引用是一致的。

仍然不能把本轮说成完全源码级再编译复现。当前机器上默认 `CUTLASS_ROOT=/xplorer/shijy/third_party/cutlass` 不存在；虽然能找到 `/xplorer/liuss11/s2x/external/cutlass`，但它是 CUTLASS 4.3.3，不是本文档要求的 4.5.2，因此没有用它来覆盖现有二进制。全量 sweep 使用的是 `SKIP_BUILD=1` 和现有 `GEMMsm110/build/gemm_sm110_bench`。这不影响这次 CSV 对“现有二进制在当前机器上是否达到 90%”的性能结论，但影响“从干净 checkout 到二进制”的可复现声明。后续要把这个限制拿掉，需要补齐 CUTLASS 4.5.2 checkout，然后不带 `SKIP_BUILD=1` 重新编译并重跑同一条 `scripts/run_sm110_gemm_core_sweep.sh`。
