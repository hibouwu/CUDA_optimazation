# GMEM / DRAM streaming bandwidth microbenchmark

This directory measures large-working-set global memory streaming throughput on
Thor.  The default working set is 256 MiB, well above the 32 MiB L2 cache, and
global operations use `.cg` cache policy to avoid L1 hits.

Modes:

- `read-stream`: streaming `ld.global.cg.v4.u32`.
- `write-stream`: streaming `st.global.cg.v4.u32` plus a device fence before
  timing stops.
- `copy-stream`: streaming read from one buffer and write to another; reports
  read+write request bytes.

```bash
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

This Thor/NCU combination may not expose direct `dram__bytes*` metrics.  The NCU
script therefore records direct DRAM counters when present and otherwise uses
LTS read/write miss-sector bytes as the DRAM-path proxy.
