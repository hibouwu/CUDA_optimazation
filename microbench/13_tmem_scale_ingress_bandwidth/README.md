# SM110 block-scale TMEM ingress microbenchmark

This benchmark measures the SFA/SFB shared-memory-to-TMEM path used by
Blackwell block-scaled MMA.  It is intentionally narrower than a generic
`tcgen05.cp` benchmark: the issued instruction is
`tcgen05.cp.cta_group::1.32x128b.warpx4`, the same copy shape and multicast
contract used by CUTLASS `Cp4x32x128bOp` for scale factors.

## Rate contract

One instruction reads a unique `32 x 128-bit = 512 B` scale atom from shared
memory and multicasts it to four 32-lane TMEM partitions.  The reported
`bytes_per_second` is normalized by the unique 512-B source payload.  The
kernel separately reports the 2048-B multicast destination footprint and does
not use that larger number as GEMM scale work.

The timed interval is the earliest CTA `%globaltimer` start through the latest
CTA stop for one block on each of Thor's 20 SMs.  Each batch ends with
`tcgen05.commit` and an mbarrier wait, so the measurement includes completion,
not just instruction issue.  A 32-copy batch writes 32 distinct four-column
TMEM slots in columns 384--511; it does not create artificial write-after-write
hazards by repeatedly targeting the same asynchronous destination.  After the
timed region, the first warp reads the final slot in one
multicast partition with `tcgen05.ld ... x4` and requires zero value
mismatches against the initialized shared-memory atom.

The unified component campaign runs ten independent trials and independently
audits:

- exactly 20 distinct SM IDs;
- 512 source bytes per instruction and four multicast partitions;
- 32 non-overlapping destination slots of four TMEM columns each;
- zero value mismatches;
- aggregate-globaltimer rate arithmetic;
- `UTCCP.T.S.4x32dp128bit` and `LDTM.x4` in SASS;
- source, binary, SASS, run-contract and environment hashes.

## Sources

- PTX ISA 9.0 `tcgen05.cp`, issue granularity, shared-memory descriptors and
  block-scale TMEM layouts:
  <https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-cp>
- NVIDIA CUTLASS tcgen05 programming guide, block-scaled S2T copy using
  `Cp4x32x128bOp`:
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html#block-scaled-mma>
- NVIDIA CUTLASS raw instruction wrapper; `SM100_UTCCP_4x32dp128bit_1cta`
  lowers to `.32x128b.warpx4`:
  <https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm100.hpp>

## Local static check

CUDA 13.0 can compile the source without a GPU:

```bash
nvcc -O3 -std=c++17 \
  -gencode arch=compute_110a,code=sm_110a \
  microbench/13_tmem_scale_ingress_bandwidth/tmem_scale_ingress_bandwidth.cu \
  -o /tmp/tmem_scale_ingress
cuobjdump --dump-sass /tmp/tmem_scale_ingress | \
  grep -E 'UTCCP.T.S.4x32dp128bit|LDTM.x4'
```

On hosts whose system libc headers conflict with CUDA's default host compiler,
use the `--nvcc-host-undef-gnu-source` option supplied by the unified campaign
instead of changing this source.  Hardware evidence must be collected through
the versioned commands in
[`THOR_CLOSURE_RUNBOOK.md`](../../Docs/blackwell_tensorcore/THOR_CLOSURE_RUNBOOK.md),
not by treating a one-off binary run as closure-qualified data.
