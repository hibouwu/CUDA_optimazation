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

## Importing a returned closure suite

Thor execution instructions are versioned with the runner in
`Docs/blackwell_tensorcore/THOR_CLOSURE_RUNBOOK.md`. The committed wrapper
derives the expected commit from the checked-out `HEAD`, captures platform and
overcurrent evidence, and launches the suite detached:

```bash
SUITE_ID=thor-t5000-closure-maxn-YYYYMMDD-a
bash microbench/sm110_closure_campaign.sh start "$SUITE_ID"
bash microbench/sm110_closure_campaign.sh status "$SUITE_ID"
```

After `SUITE_COMPLETE`, the same wrapper re-runs all three independent
auditors and creates the model input bundle:

```bash
bash microbench/sm110_closure_campaign.sh finish "$SUITE_ID"
```

The output is
`results/sm110_model_closure/<suite-id>/model_inputs.json`. It contains the
selected full-SM compute capacities, component capacities, paired full-GEMM
observations, exact artifact paths, platform qualification and independent
audit results. A positive overcurrent delta is preserved as a warning because
it describes the sustained MAXN platform condition; it does not by itself
invalidate otherwise complete measurements. A counter reset, which breaks the
continuity of the evidence interval, or missing evidence fails the import.

The lower-level import command is available for an already complete evidence
tree:

```bash
python3 -m scripts.sm110_gemm_model.cli import-closure \
  --repo-root . \
  --suite-id "$SUITE_ID" \
  --expected-commit "$(git rev-parse HEAD)" \
  --output "results/sm110_model_closure/$SUITE_ID/model_inputs.json"
```
