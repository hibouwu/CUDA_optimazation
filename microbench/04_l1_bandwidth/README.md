# L1 global-path bandwidth microbenchmark

This directory measures software-visible L1TEX global load/store throughput on
Thor.  The benchmark uses one CTA per SM, a small per-CTA global working set, and
explicit PTX cache operators.  The default per-CTA working set is rounded up to
32 KiB so the 256-thread, eight-way unrolled loop has a full unique first
iteration.

Modes:

- `read-ca`: timed `ld.global.ca.v4.u32` after an in-kernel `.ca` warmup.
- `read-cg`: timed `ld.global.cg.v4.u32` control path, expected to bypass L1
  caching and exercise L2.
- `write-wb`: timed `st.global.wb.v4.u32`, reported as L1TEX store-front-end /
  end-to-end store-path throughput.
- `write-cg`: timed `st.global.cg.v4.u32` control path.

```bash
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

Interpretation rules:

- `read-ca` is the only L1-hit read bandwidth conclusion.
- Global stores are not treated as a pure "L1 cache write bandwidth" limit;
  they are L1TEX LSU store path numbers with LTS traffic checked by NCU.
- NCU validation must show `read-ca` has high L1TEX lookup-hit bytes and much
  lower LTS bytes than the timed logical load stream.
