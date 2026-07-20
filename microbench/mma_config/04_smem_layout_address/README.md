# 04_smem_layout_address

This directory is an independent tcgen05 MMA configuration microbenchmark stage.

- `benchmark_src/`: CUDA source for this stage.
- `scripts/run.py`: builds and runs only this stage.
- `plots/`: raw CSV, aggregate CSV, SVG plots, SASS summary, and analysis.

Run from the repository root or this directory:

```bash
python3 microbench/mma_config/04_smem_layout_address/scripts/run.py --quick
```
