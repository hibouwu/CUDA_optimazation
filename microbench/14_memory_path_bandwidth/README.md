# SM110 HBM/L2 read/write component microbenchmark

This benchmark replaces the legacy max-CTA `clock64()` snapshots used by the
GEMM model with four closure-compatible whole-GPU measurements:

- `hbm.read`: 256-MiB streaming loads;
- `hbm.write`: 256-MiB streaming stores plus device-scope completion;
- `l2.read`: 16-MiB warmed working-set loads;
- `l2.write`: 16-MiB warmed working-set stores plus device-scope completion.

Every case launches four blocks per SM, records each CTA's `%globaltimer`
start/stop and SM ID, and reports requested bytes over the earliest-start to
latest-stop interval.  The unified component campaign requires all 20 SM IDs,
ten trials, exact rate arithmetic, matching case fields and read/write SASS.

The read helper requests cache-global loads and feeds all four 32-bit lanes of
every `uint4` into the final checksum.  The campaign requires `LDG.E.128` in
the built SASS, so the 16-B accounting cannot survive while ptxas silently
eliminates half of each vector.  The write path is likewise anchored by
`STG.E.128` and includes a device-scope fence before the stop timestamp.
These are measured sustained rates for the stated access contract, not
physical port-rate uppers.

Use the versioned component or closure commands in
[`THOR_CLOSURE_RUNBOOK.md`](../../Docs/blackwell_tensorcore/THOR_CLOSURE_RUNBOOK.md)
for qualified hardware collection.  A local CUDA 13.0 static build is:

```bash
nvcc -O3 -std=c++17 \
  -gencode arch=compute_110a,code=sm_110a \
  microbench/14_memory_path_bandwidth/memory_path_bandwidth.cu \
  -o /tmp/memory_path_bandwidth
cuobjdump --dump-sass /tmp/memory_path_bandwidth | grep -E 'LDG.E.128|STG.E.128'
```
