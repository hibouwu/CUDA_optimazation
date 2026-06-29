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
| `common` | Shared SM110-only helpers and kernels used by the demos. |

## Build And Run

From this directory:

```bash
./build_and_run.sh sanity
./build_and_run.sh tcgen05
./build_and_run.sh clc
./build_and_run.sh all
```

Build only:

```bash
./build_and_run.sh build-only
```

Clean:

```bash
./build_and_run.sh clean
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

