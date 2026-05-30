# Contributing

This repository is organized around measurable CUDA kernel optimization steps. Contributions should preserve that structure.

## Kernel Changes

When adding or changing a kernel:

1. Keep the previous version intact unless it is incorrect.
2. Add a new version name when the optimization changes the implementation strategy.
3. Explain the optimization intent in source comments.
4. Keep correctness checks against CPU, CUB, cuBLAS, or another trusted baseline.
5. Add benchmark output to CSV if the new version is timed.

## Benchmark Rules

- Use CUDA events for timing.
- Warm up before measuring.
- Do not include host/device copies in kernel time.
- Report both absolute performance and baseline ratio where possible.
- Keep generated CSV files out of git.

## Documentation

Update these files when behavior changes:

- `README.md` for user-facing features and quick start.
- `docs/benchmark.md` for benchmark methodology.
- `docs/toolchain.md` for build or environment notes.
- `WORKLOG.md` for implementation milestones.

## Style

- Prefer simple, inspectable CUDA C++ over overly abstract helper layers.
- Keep comments focused on optimization intent, indexing strategy, and hardware behavior.
- Avoid unrelated refactors in benchmark kernels.
