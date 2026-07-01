# `tcgen05.mma` shared-memory operand benchmark

This is an SM110a-only microbenchmark for the async-proxy shared-memory
operand path of `tcgen05.mma`. It does not use CUTLASS.

The kernel:

1. initializes constant FP16 A/B operands in shared memory;
2. allocates 64 TMEM columns with `tcgen05.alloc`;
3. creates K-major SMEM descriptors;
4. issues `tcgen05.mma.cta_group::1.kind::f16` from one CTA thread;
5. commits groups through an mbarrier;
6. loads one TMEM slice to registers and deallocates TMEM.

| Case | Descriptor mode | Descriptor stride |
|---|---|---|
| `swizzle_32b` | 32-byte K-major swizzle, code 6 | `8 * 32` bytes |
| `swizzle_64b` | 64-byte K-major swizzle, code 4 | `8 * 64` bytes |
| `swizzle_128b` | 128-byte K-major swizzle, code 2 | `8 * 128` bytes |

All operand values are one, so rearranging their physical positions does not
change the mathematical input. The benchmark compares descriptor/layout modes,
not scalar bank IDs.

`tcgen05.mma` shared-memory accesses execute in the async proxy. Ordinary
`pipe_lsu` shared bank-conflict counters are not assumed to cover them. Compare
runtime, tensor-pipe activity, MIO stalls, and queried SM110 metrics. Inspect
SASS before accepting a result.

```bash
./scripts/build.sh
./scripts/run_basic.sh
./scripts/run_ncu.sh
```

The build is intentionally fixed to
`-gencode arch=compute_110a,code=sm_110a`; architecture-specific `a` features
are required for these instructions.

References:

- [PTX shared-memory descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor)
- [PTX `tcgen05.mma`](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-mma)

