# `cp.async` shared-destination benchmark

This benchmark issues one 16-byte
`cp.async.ca.shared::cta.global` per lane, commits the async group, and waits
for completion. It studies the LDGSTS/async-copy path, not ordinary
`st.shared`.

| Case | Destination mapping | Intent |
|---|---|---|
| `contiguous` | lane `i` starts at word `4*i` | Non-overlapping contiguous 16-byte destinations |
| `stride` | lane `i` starts at `i * stride_words` | Destination-start spacing of 4/8/16/32 words |
| `source_broadcast` | contiguous destinations, one shared global source per warp | Separate source coalescing/reuse from destination layout |

Because every instruction writes 16 bytes and touches four banks, scalar
`stride -> N-way conflict` terminology does not transfer directly. Use runtime,
LDGSTS sector/conflict candidates, and SASS together.

```bash
CUDA_ARCH=110 ./scripts/build.sh
./scripts/run_basic.sh
./scripts/run_ncu.sh
```

The default loop waits after every async group. It measures serialized
issue-to-completion cost. A future pipelined benchmark should vary the number
of outstanding groups separately rather than mixing that question into this
destination-layout baseline.

Reference: [PTX `cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async).

