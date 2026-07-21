# 06_tmem_dependency

本目录是独立的 tcgen05 MMA 配置微基准 stage。

- `benchmark_src/`: 本 stage 的 CUDA 源码。
- `scripts/run.py`: 只构建并运行本 stage。
- `plots/`: raw CSV、aggregate CSV、SVG 图、SASS 摘要和分析文档。

从仓库根目录或本目录运行：

```bash
python3 microbench/mma_config/06_tmem_dependency/scripts/run.py --quick
```
