# SM110 GEMM-shaped HBM/L2 duplex campaign

This campaign measures simultaneous read and write service, rather than
combining two independent one-direction peaks.  Every case uses one kernel
containing both `LDG.E.128` and `STG.E.128`, ten external trials, all 20 SMs,
earliest-CTA to latest-CTA `%globaltimer` timing, and mandatory NCU evidence.
One logical read or write operation is eight independent 128-bit transactions
(128 B).  Read and write groups are interleaved, stores do not depend on loaded
values, and all four lanes of every load remain live.  The auditor recomputes
the absolute byte count from this instruction contract; matching only the
read:write ratio is insufficient.

The HBM matrix covers the logical input:output byte ratios of the twelve
declared accumulator-output precision contracts.  The L2 matrix covers the
schedule-level repeated input request:output ratios mechanically derived from
the current executable workload, schedule, and precision manifests at the
frozen full-GEMM shapes `N=1024,2048,4096`.  This includes the block-scaled
transport points `27:16`, `27:8`, and `27:4`.  A rate is applicable only to the
exact reduced ratio in its resource ID.  The manifests and model source are
SHA-256-bound dependencies of the frozen run.

NCU must confirm both requested L2 read/write sectors.  `hot_l2` additionally
requires more read hits than misses.  `cold_hbm` requires direct DRAM read and
write byte counters, so a large-working-set label cannot masquerade as proof of
physical DRAM traffic.

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
`plots/memory-duplex-service-curves.svg`. Each HBM/L2 panel plots total, read,
and write GB/s against the issued-byte read share. Min/max whiskers retain the
ten-trial spread. The SVG is a derived view; the raw summary and NCU evidence
remain authoritative and are SHA-256-bound by `plots/manifest.json`.
