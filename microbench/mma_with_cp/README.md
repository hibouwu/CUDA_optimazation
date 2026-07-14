# Thor TCGen05 MMA With Copy Pipeline

This directory benchmarks TCGen05 SS/TS MMA and `tcgen05.cp` paths on Thor.
The main entry point generates CUDA sources, builds them, runs the selected
microbenchmarks, checks SASS, and writes `分析报告.txt`.

```bash
python3 run_thor_tcgen05_cp_mma_report.py --iters 10000
python3 plot_tcgen05_cp_mma_results.py
```

Plot output includes `speedup_heatmap.svg` for normalized speedup and
`tflops_heatmap.svg` for absolute TFLOP/s.

Default coverage is `M128N64`, `M128N128`, and `M128N256` across BF16,
FP8, and FP4. This is 14 cases x 3 precisions x 3 shapes:

- SS MMA-only and TS MMA-only forced-wait diagnostics: one complete MMA atom,
  then one completion wait. These cases intentionally break asynchronous
  pipelining and should not be interpreted as peak or mainloop throughput.
- SS MMA Mainloop K2/K4/K8/K16 sweep, the closest current cases to a
  CUTLASS-style dense GEMM mainloop boundary: multiple K-block SS MMAs, then
  one completion wait.
- TS CP+MMA Mainloop A2 K2/K4/K8/K16 sweep for the A-from-TMEM path: copy the
  next A panel into one TMEM slot, consume the current A panel from the other
  slot, and wait at each K-block dependency boundary.
- `tcgen05.cp`-only SMEM-to-TMEM copy throughput.
- TS CP+MMA Serial A1, Overlap A2, Warp Split A2.

Interpretation note: ordinary dense GEMM is primarily represented by the SS
SMEM-descriptor path. TS A-from-TMEM cases are kept as valid hardware-path
experiments for FMHA, mixed-input GEMM, or other kernels that intentionally
stage/reuse A in TMEM; they should not be read as the default dense GEMM path.
Grouped 4Warp / N-slice diagnostics were removed from the default benchmark
because they split a logical MMA tile into smaller atoms and mix atom-shape
effects into the pipeline comparison.

Use `--primary-shape-only` for a faster pass that only runs `M128N256`.
`--all-shapes` is kept as a compatibility no-op because all shapes are now the
default.

## CTA group 2 SS mainloop supplement

`run_thor_tcgen05_g2_report.py` is an independent BF16 supplement for
`tcgen05.mma.cta_group::2`. It generates `M256N128K16` and `M256N256K16`
SS mainloop cases, with 1, 4, 8, or 16 MMA instructions per completion wait
for each shape. The existing CTA group 1 runner and result files are not
modified.

Generate the CUDA sources without compiling or running them:

```bash
python3 run_thor_tcgen05_g2_report.py --generate-only
```

Generate and compile all group2 cases for `compute_110a/sm_110a`, run the SASS
checks, but do not launch the kernels:

```bash
python3 run_thor_tcgen05_g2_report.py --skip-run
```

If the host compiler selected by `nvcc` is not compatible, pass it explicitly:

```bash
python3 run_thor_tcgen05_g2_report.py --skip-run --ccbin /path/to/g++-13
```

On Thor, first run a short cluster-launch and completion-wait smoke test, then
run the full measurement:

```bash
python3 run_thor_tcgen05_g2_report.py --iters 10 --trials 1
python3 run_thor_tcgen05_g2_report.py --iters 10000 --trials 20
```

The runner reads the current GPC frequency automatically. Use `--freq-hz` to
record a known fixed frequency instead:

```bash
python3 run_thor_tcgen05_g2_report.py --iters 10000 --trials 20 --freq-hz 1575000000
```

Generated sources are written to
`benchmark_src/tcgen05_g2_ss_mainloop_k{1,4,8,16}_m256n{128,256}_bf16_benchmark.cu`,
and binaries use the separate `build_g2/` directory; the runner inspects their
SASS before any launch. A completed Thor run writes
`plots/g2_ss_mainloop_sweep_results.csv` and
`plots/g2_ss_mainloop_report.txt`. `plot_g2_ss_mainloop_comparison.py` writes
`plots/g2_vs_g1_tflops.svg` and `plots/g2_vs_g1_tflops_results.csv`.
Group 1 comparisons are loaded from
`plots/mma_only_results.csv` for K1 and
`plots/mma_mainloop_sweep_results.csv` for K4/K8/K16.

## NCU collection

The main benchmark script does not run Nsight Compute. Build the binaries first,
then collect counters with:

```bash
python3 run_thor_tcgen05_cp_mma_report.py --iters 10000
./run_ncu_reports.sh
```

By default the NCU script scans all default binaries and writes
`ncu_reports/*.ncu-rep` plus matching `.log` files. The default metric list is
`ncu_tcgen05_cp_mma_metrics.txt`, a small counter set for cycles, TCGen05 MMA,
TMEM/cp-side pipe activity, warp stalls, and launch limits.

Full collection is 14 cases x 3 precisions x 3 shapes, so a representative pass
is usually more practical:

```bash
ITERS=10000 OUT_DIR=./ncu_reports_key \
CASES="ss_mma_only ss_mma_mainloop_k16 ts_cp_mma_mainloop_a2_k16 tcgen05_cp_only ts_cp_mma_overlap_a2 ts_cp_mma_warp_split_a2" \
SHAPES=m128n256 PRECISIONS="fp4 bf16" \
./run_ncu_reports.sh
```

Useful controls:

```bash
DRY_RUN=1 MAX_REPORTS=4 ./run_ncu_reports.sh
SKIP_EXISTING=1 ./run_ncu_reports.sh
NCU_METRICS='regex:.*' OUT_DIR=./ncu_reports_full ./run_ncu_reports.sh
NCU_METRICS= NCU_METRICS_FILE= NCU_SET=full ./run_ncu_reports.sh
```

If GPU performance counters are not enabled for the current user, logs will show
`ERR_NVGPUCTRPERM`; the script keeps the log and continues unless
`FAIL_ON_NCU_ERROR=1` is set.

Boundary:

- `../mma_compute_only` measures dense MMA completion throughput without the
  measured copy path.
- GMEM/TMA staging, epilogue, TMEM readback, global stores, sparse MMA, and
  2CTA/cluster paths are out of scope for the default report. The CTA group 2
  runner above covers only the separate SS MMA mainloop supplement.
