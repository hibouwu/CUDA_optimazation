# 文档维护说明

## 写作原则

- README 只服务于一个问题：同一份矩阵 A 为什么会出现许多 CuTe 对象，以及数据在哪一步真正移动。
- 全文采用 CUTLASS C++ CuTe API，固定追踪 `A[133,70]`，沿 `mA → gA → tCgA → tCsA → tCrA → tCtAcc → tDrAcc → mD` 展开。
- 每一节都按“操作前对象 → 操作 → 操作后对象”展开，先解释变化，再给代码。
- 只在首次出现时解释术语，不设置学习目标、自检题、阶段检查、术语表或重复总结表。
- 主文只讲单 CTA 主线。`CtaGroup.TWO`、atom repeat、permutation 和硬件验证不放进这篇入门教程。
- API 名称和代码标识符保留英文，其他说明使用自然中文。代码示例必须使用 C++ CuTe，不得混入 CuTe Python DSL 语法。
- C++ API 与变量命名以 `examples/cute/tutorial/blackwell/02_mma_tma_sm100.cu` 为主参考。

## 图片生成

运行以下命令可重建 SVG，并刷新 README 中的生成图片区块：

```bash
python3 -m pip install -r scripts/requirements.txt
python3 scripts/generate_tiled_mma_diagrams.py
```

脚本使用固定版本 `matplotlib==3.10.9` 生成六张过程图。每张图都有相同的六步进度条、颜色、箭头和橙色追踪标记。

脚本生成一张 C++ CuTe 总览图和六张分步过程图，并只替换 `BEGIN/END GENERATED DIAGRAM` 标记之间的 Markdown 图片，不改写正文解释。图形约定如下：

- 蓝色表示 GMEM，绿色表示 SMEM，紫色表示 descriptor/TMEM，金色表示 RMEM。
- 橙色始终跟踪 `A[133,70]`，进入 MMA 后改为表示它对输出的贡献。
- 粗实线箭头表示 copy 或 compute；虚线箭头表示 view、partition 或 descriptor 构造。
- `BK=64` 的四段 `MMA_K=16` 在相关图片中保持同一顺序和颜色。

## 当前图组

1. `overview_cpp_cute.svg`：C++ CuTe 完整数据流。
2. `step1_local_tile.svg`：`mA → gA`。
3. `step2_partition_a.svg`：`gA → tCgA`。
4. `step3_tma_copy.svg`：`tCgA → tAgA/tAsA → tCsA`。
5. `step4_descriptor.svg`：`tCsA → tCrA`。
6. `step5_mma_tmem.svg`：`tCrA/tCrB → tCtAcc`。
7. `step6_epilogue.svg`：`tCtAcc → tDrAcc → mD`。
