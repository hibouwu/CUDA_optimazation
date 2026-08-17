# SM110 GEMM-shaped L2 duplex and cold-read/write-path proxy campaign

This campaign measures simultaneous read and write service, rather than
combining two independent one-direction peaks.  Every case uses one kernel
containing both `LDG.E.128` and `STG.E.128`, ten external trials, all 20 SMs,
earliest-CTA to latest-CTA `%globaltimer` timing, and mandatory NCU evidence.
One logical read or write operation is eight independent 128-bit transactions
(128 B).  Read and write groups are interleaved, stores do not depend on loaded
values, and all four lanes of every load remain live.  The auditor recomputes
the absolute byte count from this instruction contract; matching only the
read:write ratio is insufficient.

The cold-entry matrix covers the logical input:output byte ratios of the twelve
declared accumulator-output precision contracts.  The L2 matrix covers the
schedule-level repeated input request:output ratios mechanically derived from
the current executable workload, schedule, and precision manifests at the
frozen full-GEMM shapes `N=1024,2048,4096`.  This includes the block-scaled
transport points `27:16`, `27:8`, and `27:4`.  A rate is applicable only to the
exact reduced ratio in its resource ID.  The manifests and model source are
SHA-256-bound dependencies of the frozen run.

The largest derived L2 ratio is the irreducible `96:1` point. The CUDA binary
therefore freezes `max_operation_groups=128`; the runner rejects any manifest
ratio above that bound, and every runtime row must report the same limit. The
operation loop is runtime-controlled and retains a fixed eight-`uint4` local
load set, so raising the former host validation limit from 64 does not unroll
96 register groups. Removing or approximating `96:1` is not permitted.

NCU must confirm both requested L2 read/write sectors. `hot_l2` additionally
requires more read hits than misses. On Thor, direct `dram__bytes_op_read/write`
metrics do not exist. `cold_hbm` therefore remains the compatibility residency
name, but its qualification is explicitly
`cold_dram_read_plus_write_path_proxy`: read lookup misses must prove at least
60% of requested read bytes reached beyond L2, while write sectors prove only
that requested stores entered the L2 write path. The runner records
`external_write_bytes_proven=false`; this surface must not be described as
physical DRAM write-byte closure.

Available `mcc__dram_throughput_op_*...pct_of_peak_sustained_*` metrics are
percentages of sustained peak, not byte counters. NVIDIA documents these as
how close a unit came to peak throughput; without a sourced peak-rate and
instance/time aggregation contract they cannot replace an absolute-byte gate:
https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide

Nsight Compute 2025.3.1 raw CSV may quote base-unit metrics with decimal
grouping such as `"8,299,136"`. Both the runner and an independently
implemented auditor parser now require the exact base-unit row, accept valid
grouping, reject malformed or scaled values, and compare every stored NCU
metric back to the raw CSV.

The process entrypoint catches `Exception`, not `BaseException`. A normal
`return 0` or `SystemExit(0)` must preserve the final `complete` status;
`RuntimeError` and other real exceptions still call `mark_failed`. This state
transition is covered by unit tests because the suite orchestrator relies on
`campaign_status.json` before printing `PARAMETER_SUPPLEMENT_COMPLETE`.

Run on Thor from a clean checkout at the frozen commit:

```bash
RUN_ID=thor-t5000-memory-duplex-maxn-YYYYMMDD-a
EXPECTED_COMMIT=$(git rev-parse HEAD)
bash microbench/sm110_memory_duplex_campaign/launch_memory_duplex_campaign.sh \
  "$RUN_ID" "$EXPECTED_COMMIT" --nvcc-host-undef-gnu-source
```

The runner is deliberately separate from the historical 18-case component
suite: adding a new physical contract must not relabel old artifacts as if they
had measured simultaneous traffic.

After collection, run the independent auditor:

```bash
python3 microbench/sm110_memory_duplex_campaign/audit_campaign.py \
  "results/sm110_memory_duplex_campaign/$RUN_ID"
```

After `summary.json`, the runner automatically generates
`plots/memory-duplex-service-curves.svg`. Each cold-proxy/L2 panel plots total,
read, and write GB/s against the issued-byte read share. Min/max whiskers retain the
ten-trial spread. The SVG is a derived view; the raw summary and NCU evidence
remain authoritative and are SHA-256-bound by `plots/manifest.json`.
