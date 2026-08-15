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
Measured sustained rates are not treated as physical upper bounds. Every
empirical schedule is additionally intersected with all conditional hard
ceilings applicable to the same workload and schedule. In particular,
direction-specific `hbm.read` and `hbm.write` points cannot bypass the shared
`hbm.total` LPDDR ceiling.
Likewise, the 1024-B/cycle/GPU L2 bus remains a shared `l2.read` constraint,
while a single-CTA L2-hit probe directly measures
`tma.smem_ingress.per_sm`, which is applied with a slowest-wave makespan. It is
not inferred by dividing a concurrent full-GPU TMA result by the SM count.
Schedules with fewer than four stages use the separately named `.inflight4`
component capacities. Four-stage schedules use the exact tc5a A/B mixed
contract, so a faster shallow or serial diagnostic cannot silently replace it.

The schedule manifest separates executable transport contracts. Standard
FP16/BF16/TF32/FP8/INT8 schedules use their native logical payload; raw
E3M2/E2M3/E2M1 direct-SMEM schedules use the byte containers encoded by the
closure compute campaign; MXFP4/NVFP4 schedules reserve the v1 512-column TMEM
layout for the accumulator and SFA/SFB operands. Fractional logical storage is
therefore never silently treated as an executable direct-SMEM TMA layout.

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
`results/sm110_model_closure/<suite-id>/model_inputs.json`. It contains 36
shape-qualified full-SM compute capacities (12 precisions times M128N64,
M128N128, and M128N256), 18 component capacities, paired full-GEMM
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

If compute/full-GEMM were already collected at an earlier immutable commit and
only the component contract changed, use the bounded component-supplement flow
in the runbook instead of rerunning unchanged GPU work. Its lower-level import
is deliberately multi-commit rather than a relaxed single-suite import:

```bash
python3 -m scripts.sm110_gemm_model.cli import-composite-closure \
  --repo-root . \
  --composite-id "$SUPPLEMENT_ID" \
  --base-suite-id "$BASE_SUITE_ID" \
  --base-expected-commit "$BASE_EXPECTED_COMMIT" \
  --component-expected-commit "$EXPECTED_COMMIT" \
  --output "results/sm110_model_closure/$SUPPLEMENT_ID/model_inputs.json"
```

The importer does not trust an old `model_inputs.json`. It re-runs the current
independent compute and full-GEMM auditors against the raw base artifacts,
audits the new component artifacts separately, validates each campaign's
environment against its own commit, and validates MAXN/1.575-GHz/overcurrent
evidence for both time intervals. The resulting `campaign_sources` field makes
the split explicit. Capacity and observation IDs belong to the composite ID,
while `source_id`, `run_id`, and artifact paths retain the actual producer.

Render the final numerical tables and model comparisons without copying rates
by hand:

```bash
MODEL_DIR="results/sm110_model_closure/$SUITE_ID"
python3 -m scripts.sm110_gemm_model.cli report-closure \
  --closure-import "$MODEL_DIR/model_inputs.json" \
  --repo-root . \
  --capacities scripts/sm110_gemm_model/profiles/capacities.json \
  --hardware scripts/sm110_gemm_model/profiles/thor_sm110.json \
  --schedules scripts/sm110_gemm_model/examples/schedules.json \
  --output-json "$MODEL_DIR/closure_analysis.json" \
  --output-markdown "$MODEL_DIR/closure_summary.md"
```

The report preserves N=1024/2048 as the predeclared calibration points and
N=4096 as holdout, but does not infer cache residency from that split. It
evaluates both hot-L2 and cold-HBM scenarios: the conditional check uses the
larger (safe) performance upper and the empirical result remains an interval.
Exceeding both empirical scenarios is a calibration warning because a measured
component rate is not a physical rate upper.

`closure_analysis.json` separately reports `pass`, bounded-campaign measurement
closure, all-precision numeric closure, and common-resource closure. A successful
five-precision campaign therefore does not silently claim that every one of the
twelve declared precision contracts has a proven compute upper and full GEMM.

Merge numeric evidence with the full-GEMM implementation/reference support map:

```bash
python3 -m scripts.sm110_gemm_model.cli report-precision-closure \
  --repo-root . \
  --capacities scripts/sm110_gemm_model/profiles/capacities.json \
  --closure-import "$MODEL_DIR/model_inputs.json" \
  --support-manifest microbench/sm110_full_gemm_campaign/support_manifest.json \
  --output-json Docs/blackwell_tensorcore/thor_sm110_all_precision_evidence_matrix.json \
  --output-markdown Docs/blackwell_tensorcore/thor_sm110_all_precision_evidence_matrix.md \
  --require-all-closed
```

Without `--require-all-closed`, the command writes an honest intermediate gap
matrix and exits successfully.  With it, any missing strict upper, compute
shape, full-GEMM shape, numerical validation, same-precision denominator, or
implementation source makes the command fail.  Structural validity of
`support_manifest.json` is not treated as evidence that its `partial` or
`missing` rows are complete.
