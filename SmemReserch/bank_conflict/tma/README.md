# TMA shared-memory layout benchmark

This benchmark uses `CUtensorMapEncodeTiled` and
`cp.async.bulk.tensor.2d.shared::cta.global` with an mbarrier. Each operation
moves 4096 bytes from global memory to a 1024-byte-aligned shared buffer.

| Case | Tensor-map swizzle | Legal 2D box (`x * y`, bytes) |
|---|---|---|
| `swizzle_none` | none | `128 * 32` |
| `swizzle_32b` | 32-byte | `32 * 128` |
| `swizzle_64b` | 64-byte | `64 * 64` |
| `swizzle_128b` | 128-byte | `128 * 32` |

The box geometry changes for 32B/64B because the tensor map requires the
innermost box span not to exceed the selected swizzle span. Total transferred
bytes remain equal, but this means runtime differences are not automatically
caused by swizzling alone.

TMA runs in the async proxy and does not issue one ordinary LSU
`ld.shared`/`st.shared` instruction per lane. Therefore the benchmark reports
TMA throughput and exposes candidate NCU metrics, but it does not label any
case as an N-way bank conflict.

```bash
CUDA_ARCH=110 ./scripts/build.sh
./scripts/run_basic.sh
./scripts/run_ncu.sh
```

References:

- [PTX `cp.async.bulk.tensor`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
- [CUDA TMA swizzle guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tma-swizzle)

