# 文档维护说明

## 写作原则

- README 面向首次接触 Blackwell tcgen05 和 CuTe layout 的读者，所有术语在首次出现时定义。
- 正文保留最终定义、坐标推导、数据流证据和结论；审校过程、生成命令和调整原因记录在本文件。
- 段首先给出正向结论，再说明适用边界。避免连续使用“不是 A，而是 B”的对照句式。
- 使用具体对象和动作，例如“`cute.copy` 将 A 从 GMEM 写入 SMEM”，避免使用“该机制”“进行处理”等泛化表达。
- 每个关键结论应由 shape、坐标、存储位置或具体操作支持。
- 对坐标、数据流或实验能够直接支持的结论使用正向描述，必要边界放在结论之后。
- API 名称可以保留英文；读者可能不熟悉的体系结构术语应在首次出现时说明中文物理含义。
- 区分软件可观察语义与微架构推断。概念图和 API 行为只用于说明其直接支持的关系，不据此推断硬件内部严格串行或固定端口结构。
- 彩色图表示逻辑 shape 和坐标分解。除明确标注 Thread/Register 的图外，不将概念图解释为真实 Thread-Value ownership。

## 图片生成

运行以下命令可重建 SVG，并刷新 README 中的生成图片区块：

```bash
python3 -m pip install -r scripts/requirements.txt
python3 scripts/generate_tiled_mma_diagrams.py
```

脚本使用固定版本 `tensor-layouts[viz]==0.3.2` 生成 layout 网格，再用 Matplotlib 组合存储层级、操作箭头和解释性标注。该版本的公开 `draw_*` 会关闭 Figure，无法继续组合多面板教学图，因此脚本有意调用 `_build_composite_figure` 等私有 builder；升级依赖时必须重新做视觉回归。

原始手绘总图 `Blackwell_tcgen05_GEMM_Dataflow_Overview.jpg` 不由脚本生成或替换。脚本只替换 `BEGIN/END GENERATED DIAGRAM` 标记之间的 Markdown 图片，不改写正文解释。图形约定如下：

- `M` 始终向下，`K/N` 始终向右。
- 蓝/绿/紫/金面板分别代表 GMEM/SMEM/TMEM/RMEM；橙色始终跟踪 `A[133,70]` 或它对输出的贡献。
- 实线箭头代表 copy/compute 等物理数据操作，虚线代表 view/layout/descriptor 关系。
- 完整 `128×256` atom 不逐元素绘制；大 shape 采用明确标注的 block compression，防止图片在 README 宽度下失去可读性。

## 调整记录

- 主线示例统一采用真实 shape：problem 为 `(512,768,384)`，A tile 为 `(128,64)`，instruction K 为 `16`，SMEM stage 数为 `3`。完整矩阵图使用真实 `A(512×384)`，并通过 `8×8` 放大窗口逐元素标出 `A[133,70]`；不再用 `(8,8,8)` 教学 problem 推导正文坐标。
- `tCgA → tAgA/tAsA → sA` 使用 `A[133,70]` 展开全局坐标、tile 局部坐标、`MMA_K`、TMA 组合传输坐标和 SMEM stage，并用并排图表示 `q=1 → stage=1` 的逻辑对应关系。
- 完整矩阵图中的整数是 row-major 元素编号，不作为 F16/BF16 输入值，避免将坐标标识与数值表示范围混为一谈。
- 早期图示曾用 `Element#`、`Thread#`、`Register#` 等无实际 layout 支持的规则网格，容易被误解为真实 Thread-Value ownership。当前只画可验证的坐标、atom coverage、descriptor slot 或代表性 token，并在图脚注明准确性边界。
- 目标架构是 SM110。`tensor-layouts` 尚无 SM110 atom；脚本只借用 SM100 UMMA atom 中相同的 `128×256×16` 指令 shape 做粗粒度逻辑投影，不据此推断 SM110 微架构、完整 layout、TMEM bank 或 epilogue lane ownership。
- 整数 shape 公式仅用于 trivial、`CtaGroup.ONE`、无 permutation 且整除的情况；其他配置以实际 `partition_*` layout 为准。
- `tAsA` 定义为 TMA 的 SMEM destination view，`tCrA` 定义为 MMA 使用的 descriptor tensor。
- Pipeline 说明分别处理 TMA-ready、stage-reuse 和 TMEM-ready，避免将 stage 释放等同于 MMA 完成。
