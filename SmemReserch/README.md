# Shared-memory research

This directory organizes shared-memory experiments by instruction path and
access pattern. Each benchmark owns its source, scripts, documentation, and
results so conclusions from one path are not implicitly applied to another.

Current benchmarks:

- [`bank_conflict/ld_shared_1d`](bank_conflict/ld_shared_1d/README.md):
  one-dimensional scalar/vector `ld.shared` mapping, stride conflicts,
  broadcast, and multicast.
- [`bank_conflict/st_shared_1d`](bank_conflict/st_shared_1d/README.md):
  one-dimensional scalar/vector `st.shared` mappings.
- [`bank_conflict/cp_async`](bank_conflict/cp_async/README.md):
  asynchronous copy destination-layout cases.
- [`bank_conflict/transpose_2d_case`](bank_conflict/transpose_2d_case/README.md):
  pitch-32/pitch-33 transpose load and store cases.
- [`bank_conflict/tma`](bank_conflict/tma/README.md):
  TMA tensor-map swizzle modes.
- [`bank_conflict/tcgen05_smem_operand`](bank_conflict/tcgen05_smem_operand/README.md):
  SM110a `tcgen05.mma` shared-operand descriptor modes.

The five newly added families are pending compilation and runtime validation
on the target Thor system.
