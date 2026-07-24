# DSMEM bandwidth microbenchmark

This directory measures local shared-memory and distributed shared-memory
request throughput on Thor.  Results are reported in request bytes per GPU
cycle, using 16-byte `uint4` accesses.

The benchmark has four modes:

- `local-read`: each CTA reads its own dynamic shared memory.
- `local-write`: each CTA writes its own dynamic shared memory.
- `remote-read`: each CTA reads the next CTA rank's shared memory in the same
  cluster through `cooperative_groups::cluster_group::map_shared_rank`.
- `remote-write`: each CTA writes the next CTA rank's shared memory in the same
  cluster.

Default launch uses a conservative one-CTA-per-SM cluster count:
`SM_count / cluster_size`.  The program still records
`cudaOccupancyMaxActiveClusters`, but does not use it as the default because
remote DSMEM can fail or enter multi-wave behavior when multiple CTAs per SM are
allowed.

```bash
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

Important interpretation rules:

- `remote-*` is DSMEM traffic.  `local-*` is a local SMEM baseline for contrast.
- Write modes are end-to-end store-path throughput, including the completion
  synchronization needed before the timing window closes.
- NCU exposes direct byte counters for `mem_dshared`, but not direct byte
  counters for local `mem_shared` on this machine.  Local shared utilization is
  therefore checked with shared LSU wavefront and bank-conflict counters.

Generated files:

- `results/dsmem_bandwidth.csv`
- `results/dsmem_cluster_sweep.csv`
- `results/design_adversarial_review.md`
- `results/adversarial_review.md`
- `results/ncu/ncu_dsmem_summary.csv`
- `results/ncu/ncu_dsmem_report.md`
