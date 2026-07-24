# SMEM bank/stride bandwidth microbenchmark

This directory measures local shared-memory throughput under controlled bank
stride patterns.

Modes:

- `read`: each thread issues scalar `ld.shared.u32`.
- `write`: each thread issues scalar `st.shared.u32`.

Stride is in 32-bit words.  For a warp, stride 1 is conflict-free, stride 2/4/8
create progressively wider bank conflicts, and stride 32 maps all lanes in a
warp to the same bank but different addresses.

```bash
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

Interpretation boundaries:

- This is a local SMEM scalar 32-bit bank-conflict test, not the same thing as
  the vectorized 128-bit shared path in `../03_dsmem_bandwidth`.
- This Thor/NCU combination does not expose a direct local shared byte counter.
  NCU validation therefore uses shared LSU wavefront and bank-conflict counters.
  The rough peak is payload-normalized from app B/cycle and NCU wavefront
  `%peak`, not a direct byte-counter peak.
