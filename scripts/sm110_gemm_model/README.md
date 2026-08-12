# Thor/SM110 three-layer GEMM reference model

This directory is the executable companion to
`Docs/blackwell_tensorcore/thor_sm110_gemm_performance_bounds.md`.

The model deliberately separates:

1. a conditional upper bound using only rate-cap evidence;
2. an empirical ideal envelope using measured sustained component rates;
3. observed full-GEMM results, which are imported by the validation workflow
   and are not treated as component capacities.

Run the current evidence audit:

```bash
python3 -m scripts.sm110_gemm_model.cli audit \
  --capacities scripts/sm110_gemm_model/profiles/capacities.json \
  --repo-root .
```

Evaluate all example precisions:

```bash
python3 -m scripts.sm110_gemm_model.cli evaluate \
  --hardware scripts/sm110_gemm_model/profiles/thor_sm110.json \
  --capacities scripts/sm110_gemm_model/profiles/capacities.json \
  --workloads scripts/sm110_gemm_model/examples/workloads.json \
  --schedules scripts/sm110_gemm_model/examples/schedules.json \
  --output /tmp/sm110-gemm-model-example.json
```

`partial` and `insufficient_evidence` are intentional states. A strict upper
may be `partial` and still be a valid loose bound from the proven subset of
resource constraints. The empirical envelope is fail-closed: if any demanded
component capacity is absent it emits no numeric performance. The repository
snapshot does not yet contain valid numeric capacity evidence for every
precision named by the project goal. The executable exposes that gap instead
of substituting another precision's rate.
