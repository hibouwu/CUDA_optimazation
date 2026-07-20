# tcgen05 MMA hardware-path calibration report

This report is generated from the latest CSV artifacts under `microbench/mma_config/*/plots/`.

## Reproduction

```bash
python3 microbench/mma_config/scripts/run_all.py
python3 microbench/mma_config/scripts/run_all.py --quick --repeats 3
python3 microbench/mma_config/scripts/run_all.py --stage 02_latency_throughput --case-id lat_bf16_m128n128k16_legal_ring_in0_q16 --repeats 5
```

## Environment

- GPU: NVIDIA Thor
- Compute capability: 11.0
- Driver: 580.00
- CUDA toolkit/PTX: 13.0 / PTX ISA 9.3 / CUDA 13.0
- SM clock MHz: 1575.000
- Memory clock MHz: 
- Temperature C / power W:  / 

## Artifact Index

| Stage | Benchmark source | Raw CSV | Aggregate CSV | Invalid CSV | Plot | Analysis |
| --- | --- | --- | --- | --- | --- | --- |
| 00_validation | `microbench/mma_config/00_validation/benchmark_src/` | `microbench/mma_config/00_validation/plots/raw_results.csv` | `microbench/mma_config/00_validation/plots/benchmark_results.csv` | `microbench/mma_config/00_validation/plots/invalid_cases.csv` | `microbench/mma_config/00_validation/plots/00_validation_elapsed_cycles.svg` | `microbench/mma_config/00_validation/plots/analysis.md` |
| 01_collector_protocol | `microbench/mma_config/01_collector_protocol/benchmark_src/` | `microbench/mma_config/01_collector_protocol/plots/raw_results.csv` | `microbench/mma_config/01_collector_protocol/plots/benchmark_results.csv` | `microbench/mma_config/01_collector_protocol/plots/invalid_cases.csv` | `microbench/mma_config/01_collector_protocol/plots/01_collector_protocol_tflops.svg` | `microbench/mma_config/01_collector_protocol/plots/analysis.md` |
| 02_latency_throughput | `microbench/mma_config/02_latency_throughput/benchmark_src/` | `microbench/mma_config/02_latency_throughput/plots/raw_results.csv` | `microbench/mma_config/02_latency_throughput/plots/benchmark_results.csv` | `microbench/mma_config/02_latency_throughput/plots/invalid_cases.csv` | `microbench/mma_config/02_latency_throughput/plots/02_latency_throughput_elapsed_cycles.svg` | `microbench/mma_config/02_latency_throughput/plots/analysis.md` |
| 03_effective_smem_ingress | `microbench/mma_config/03_effective_smem_ingress/benchmark_src/` | `microbench/mma_config/03_effective_smem_ingress/plots/raw_results.csv` | `microbench/mma_config/03_effective_smem_ingress/plots/benchmark_results.csv` | `microbench/mma_config/03_effective_smem_ingress/plots/invalid_cases.csv` | `microbench/mma_config/03_effective_smem_ingress/plots/03_effective_smem_ingress_tflops.svg` | `microbench/mma_config/03_effective_smem_ingress/plots/analysis.md` |
| 04_smem_layout_address | `microbench/mma_config/04_smem_layout_address/benchmark_src/` | `microbench/mma_config/04_smem_layout_address/plots/raw_results.csv` | `microbench/mma_config/04_smem_layout_address/plots/benchmark_results.csv` | `microbench/mma_config/04_smem_layout_address/plots/invalid_cases.csv` | `microbench/mma_config/04_smem_layout_address/plots/04_smem_layout_address_tflops.svg` | `microbench/mma_config/04_smem_layout_address/plots/analysis.md` |
| 05_ldshared_contention | `microbench/mma_config/05_ldshared_contention/benchmark_src/` | `microbench/mma_config/05_ldshared_contention/plots/raw_results.csv` | `microbench/mma_config/05_ldshared_contention/plots/benchmark_results.csv` | `microbench/mma_config/05_ldshared_contention/plots/invalid_cases.csv` | `microbench/mma_config/05_ldshared_contention/plots/05_ldshared_contention_tflops.svg` | `microbench/mma_config/05_ldshared_contention/plots/analysis.md` |
| 06_tmem_dependency | `microbench/mma_config/06_tmem_dependency/benchmark_src/` | `microbench/mma_config/06_tmem_dependency/plots/raw_results.csv` | `microbench/mma_config/06_tmem_dependency/plots/benchmark_results.csv` | `microbench/mma_config/06_tmem_dependency/plots/invalid_cases.csv` | `microbench/mma_config/06_tmem_dependency/plots/06_tmem_dependency_tflops.svg` | `microbench/mma_config/06_tmem_dependency/plots/analysis.md` |
| 07_config_matrix | `microbench/mma_config/07_config_matrix/benchmark_src/` | `microbench/mma_config/07_config_matrix/plots/raw_results.csv` | `microbench/mma_config/07_config_matrix/plots/benchmark_results.csv` | `microbench/mma_config/07_config_matrix/plots/invalid_cases.csv` | `microbench/mma_config/07_config_matrix/plots/07_config_matrix_tflops.svg` | `microbench/mma_config/07_config_matrix/plots/analysis.md` |

## Observation

| Stage | Valid aggregate cases | Invalid aggregate cases | Key measured field |
| --- | ---: | ---: | --- |
| 00_validation | 294 | 96 | elapsed_cycles: 1812.000 to 5946.000 |
| 01_collector_protocol | 90 | 0 | tflops: 5.967 to 36.785 |
| 02_latency_throughput | 300 | 0 | elapsed_cycles: 369114.000 to 3298792.000 |
| 03_effective_smem_ingress | 18 | 0 | effective_smem_bytes_per_cycle: 6.935 to 13.875 |
| 04_smem_layout_address | 54 | 90 | effective_smem_bytes_per_cycle: 6.841 to 14.859 |
| 05_ldshared_contention | 42 | 0 | tflops: 0.913 to 18.387 |
| 06_tmem_dependency | 172 | 0 | elapsed_cycles: 3181149.000 to 3678996.000 |
| 07_config_matrix | 192 | 64 | tflops: 8.950 to 46.796 |

- `00_validation` core pass: yes
- `02_latency_throughput` fitted beta range: 752.564 to 874.469 cycles/MMA.
- `03_effective_smem_ingress` reports logical effective operand supply: 6.935 to 13.875 bytes/cycle.
- `05_ldshared_contention` includes controls: interference_only, l1_hit_global, ld_shared, none, predicated_off_load, register_alu.

### Invalid Case Summary

| Stage | Invalid reasons |
| --- | --- |
| 00_validation | max_abs_error>0.05:48, max_abs_error>0.25:48 |
| 01_collector_protocol | none |
| 02_latency_throughput | none |
| 03_effective_smem_ingress | none |
| 04_smem_layout_address | max_abs_error>0.05:9, max_abs_error>0.25:9, misaligned address:72 |
| 05_ldshared_contention | none |
| 06_tmem_dependency | none |
| 07_config_matrix | max_abs_error>0.05:32, max_abs_error>0.25:32 |

## Inference

- `tcgen05.commit` is analyzed as cumulative completion-prefix tracking for prior async tcgen05 operations. The CSV field `pending_mbarriers` is not interpreted as an independent async group queue.
- `Q`, `independent_d_count`, and `d_reuse_distance` are separate CSV fields. When D capacity is exhausted, the run records the clamped independent D count and the reuse distance instead of labeling the sequence as independent-D.
- Effective SMEM bytes/cycle is only reported as logical operand bytes divided by measured cycles under validated collector-discard cases. It is not a physical port-width measurement.
- ld.shared contention conclusions must be read relative to the register ALU, predicated-off load, L1-hit global-load, MMA-only, and interference-only controls.

## Unsupported Claim

- These microbenchmarks do not prove physical SMEM port width, physical bank count, physical TMEM bank width, hidden collector depth, or hidden async group queue depth.
- Shape or layout performance differences are reported as software-visible sensitivity unless corroborated by the controlled experiments listed above.

## Deliverable Audit

| Requirement | Status | Evidence |
| --- | --- | --- |
| One independent subfolder per experiment | done | `microbench/mma_config/00_validation` through `07_config_matrix` each contain `benchmark_src`, `scripts`, and `plots` |
| Validation before performance stages | done | top-level runner stops after `00_validation` if core sw128/512 rows are missing; current core pass is recorded above |
| Raw and aggregate CSV | done | each stage has `raw_results.csv`, `benchmark_results.csv`, and `invalid_cases.csv` |
| Numeric and legality checks | done | valid rows require CUDA success, status ok, guard_ok=1, and max_abs_error within dtype tolerance |
| PTX/SASS audit trail | done | each stage writes `sass_summary.txt` with SASS/PTX hashes and instruction counts |
| Randomized performance order | done | non-validation stages shuffle cases with a recorded `run_order` field |
| p10/p90/median timing | done | aggregate CSV records median `elapsed_cycles` plus `elapsed_cycles_p10` and `elapsed_cycles_p90`; raw CSV keeps all repeats |
| Plots and short analyses | done | each stage has an SVG plot and `analysis.md` |
| Final report with claim boundaries | done | this file separates observation, inference, and unsupported claims |
