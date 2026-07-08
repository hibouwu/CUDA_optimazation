# GEMMComponentsSM110Thor

This directory contains small, standalone component probes for NVIDIA Thor /
SM110 / `sm_110a`.  The goal is to isolate runtime setup, TCGen05/TMEM bring-up,
and CLC-style persistent work scheduling before turning them into a full GEMM.

This is intentionally **not** an SM120, Hopper, Ampere, CUTLASS, or
`mma.sync.aligned.kind::f8f6f4` path.

## Layout

| Folder | Focus |
| --- | --- |
| `00_runtime_sanity` | Minimal CUDA Runtime validation: version, device count, `cudaFree(0)`, properties, 4-byte `cudaMalloc`. |
| `01_tcgen05_tmem_probe` | Minimal TCGen05/TMEM path: allocate TMEM, store one FP32 bit pattern, load it back, write to global memory. |
| `02_clc_persistent_tmem_probe` | Persistent CTA worker probe with static and dynamic CLC-style work-tile assignment, reusing one TMEM allocation per worker. |
| `mma` | TCGen05 dense MMA throughput benchmark generation, SASS checks, NCU collection, and plotting. |
| `common` | Shared SM110-only helpers and kernels used by the demos. |

## Build And Run

Run each component from its own subdirectory entrypoint:

```bash
00_runtime_sanity/build_and_run.sh run
01_tcgen05_tmem_probe/build_and_run.sh run
02_clc_persistent_tmem_probe/build_and_run.sh run 128 1
mma/build_and_run.sh run --iters 10000
mma/build_and_run.sh ncu
mma/build_and_run.sh plot
```

Build only:

```bash
00_runtime_sanity/build_and_run.sh build-only
01_tcgen05_tmem_probe/build_and_run.sh build-only
02_clc_persistent_tmem_probe/build_and_run.sh build-only
```

Clean:

```bash
00_runtime_sanity/build_and_run.sh clean
01_tcgen05_tmem_probe/build_and_run.sh clean
02_clc_persistent_tmem_probe/build_and_run.sh clean
```

The build always uses:

```bash
-DTC3_SM110_HOST_HAS_TCGEN05=1
-gencode arch=compute_110a,code=sm_110a
```

No demo links cuBLAS.  No demo explicitly links the CUDA Driver API.

## Expected Behavior

On a non-SM110 GPU, the TCGen05/TMEM demos should print a clear skip message:

```text
Not SM110-class device. Skip TCGen05/TMEM probe.
```

On a broken CUDA runtime/container setup, `00_runtime_sanity` should fail before
any TCGen05 kernel launch and print the exact CUDA Runtime error code/name/msg.
If `cudaFree(0)` or `cudaMalloc(4)` returns `cudaErrorNotSupported`, the problem
is the CUDA Runtime / driver / container / `libcuda` resolution, not the
TCGen05/TMEM kernel.

## Component Boundary

These demos are probes, not full GEMM kernels:

- No A/B matrix allocation.
- No cuBLAS reference.
- No benchmark input generation.
- No SM120 FP8 MMA path.
- No fake fallback kernel that pretends TCGen05 passed.

`01_tcgen05_tmem_probe` and `mma` both use TCGen05/TMEM allocation, but their
roles are different:

- `01_tcgen05_tmem_probe` is a minimal bring-up probe. It validates that a
  TCGen05 TMEM allocation plus `tcgen05.st`/`tcgen05.ld` can round-trip one
  FP32 value.
- `mma` is a dense TCGen05 MMA throughput benchmark. It uses shared-memory
  descriptors, instruction descriptors, `tcgen05.mma`, commit/wait barriers,
  SASS checks, timing, reporting, and optional NCU/plot tooling.
