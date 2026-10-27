#!/usr/bin/env python3
"""Replay the parameter-independent Task A boundary rejection; CPU only."""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


BASE = Path("/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules")
P1 = "v09d_cfg_a_P1_partial_first_and_later"
P2 = "v09d_cfg_c_P2_partial_later_sw8"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write(output, name, value):
    (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def replay(base, bundle, output):
    # Import only the immutable task snapshot, never the live public modules.
    sys.path.insert(0, str(bundle))
    sys.path.insert(0, str(bundle / "model-snapshot"))
    import cluster_first_L_extra as residual
    import supply_model as supply
    import v09_model

    source_manifest = json.loads((bundle / "model-snapshot.json").read_text())
    for name, item in source_manifest.items():
        if digest(bundle / "model-snapshot" / name) != item["sha256"]:
            raise ValueError("snapshot hash mismatch: " + name)
    input_hashes = {}

    def load(path):
        input_hashes[str(path)] = digest(path)
        return json.loads(path.read_text())

    frozen = base / "Q15-full-time/V09-component-validation/20261009-V09-freeze-job738972-r2"
    heldout = base / "Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011"
    contract = load(bundle / "interface-contract.json")
    rows = [row for row in load(frozen / "cases.json") if row["set"] == "heldout"]
    setups = {item["case"]: item["setup"] for item in load(frozen / "static_setup.json")}
    calibration = load(frozen / "frozen/calibration.json")
    predictions = load(frozen / "frozen/predictions.json")["predictions"]
    scores = {item["case"]: item for item in load(heldout / "score.json")["cases"]}
    old = load(heldout / "reanalysis/candidate-development-v1/C-first-cluster-model.json")
    if old["calibration_hash"] != digest(frozen / "frozen/calibration.json"):
        raise ValueError("failed cluster-max report uses another calibration")
    old_cases = {item["case"]: item for item in old["cases"]}
    if len(rows) != 30 or set(predictions) != {row["id"] for row in rows}:
        raise ValueError("fixed V09 population changed")

    table = []
    static = {}
    for row in rows:
        case = row["id"]
        setup = setups[case]
        features = supply.boundary_fill_features(row, setup)
        weights = residual.first_loop_weights(row, setup)
        fill_counts = {}
        for operand in ("A", "B"):
            fill_counts[operand] = {
                phase: sum(f["phase"] == phase and f["operands"][operand]["cluster_max"] > 0
                           for f in features)
                for phase in ("first", "later")
            }
        active = {f["cta"] for f in features}
        if any(weight != 0 and cta not in active for cta, weight in enumerate(weights)):
            raise ValueError("inactive CTA acquired residual weight")
        static[case] = dict(config=row["config"], shape=[row["m"], row["n"], row["k"]],
            scheduled_ctas=len(weights), active_ctas=len(active),
            cluster_fill_output_counts=fill_counts,
            calibrated_operand=residual.CALIBRATED_OPERAND.get(row["config"]),
            first_loop_weights=weights, nonzero_first_weights=sum(w > 0 for w in weights))
        score = scores[case]
        prediction = predictions[case]
        supported = bool(score["prediction_supported"])
        if supported and score["predicted_plain_us"] != prediction["plain_us"]:
            raise ValueError("score/frozen baseline mismatch")
        range_out = []
        if row["lda"] * 2 % 128 or row["ldb"] * 2 % 128:
            range_out.append("non128B_input_pitch")
        if setup["stages"] != {"cfg_a": 6, "cfg_b": 6, "cfg_c": 4}[row["config"]]:
            range_out.append("nondefault_stage")
        zero_vector = not any(weights)
        old_versions = {}
        for version, result in old_cases.get(case, {}).get("versions", {}).items():
            old_versions[version] = dict(plain_us=result["free_prediction"]["plain_us"],
                signed_relative_error=result["free_plain_error"])
        table.append(dict(case=case, config=row["config"], fixed29_scoreable=supported,
            range_out_reasons=range_out, observed_plain_us=score["observed_plain_us"],
            frozen_r2_plain_us=score["predicted_plain_us"],
            frozen_r2_signed_relative_error=score["relative_error"],
            frozen_r2_absolute_relative_error=(abs(score["relative_error"]) if supported else None),
            failed_cluster_max_reference=old_versions or None,
            first_weight_nonzero_ctas=static[case]["nonzero_first_weights"],
            boundary_family_effect="identically_zero" if zero_vector else "depends_on_unpriced_beta",
            boundary_family_plain_us=(score["predicted_plain_us"] if supported and zero_vector else None),
            candidate_status="rejected_family_no_fit",
            missing_candidate_numeric_reason=(None if supported and zero_vector else
                "r2_already_rejected" if not supported else "not_priced_after_decisive_P2_rejection")))
    scoreable = [r for r in table if r["fixed29_scoreable"]]
    outside = [r for r in scoreable if r["range_out_reasons"]]
    if len(scoreable) != 29 or len(outside) != 7:
        raise ValueError("fixed comparison/support counts changed")
    if static[P1]["cluster_fill_output_counts"] != {"A": {"first": 32, "later": 12}, "B": {"first": 0, "later": 0}}:
        raise ValueError("P1 static premise changed")
    if static[P2]["cluster_fill_output_counts"] != {"A": {"first": 0, "later": 0}, "B": {"first": 0, "later": 256}}:
        raise ValueError("P2 static premise changed")
    if any(static[P2]["first_loop_weights"]):
        raise ValueError("P2 counterexample no longer holds")

    row = next(r for r in rows if r["id"] == P2)
    baseline = v09_model.predict_components(row, setups[P2], calibration)
    regressions = []
    for beta in (1.0, 1000000.0):
        extra = [beta * weight for weight in static[P2]["first_loop_weights"]]
        corrected = v09_model.predict_components(row, setups[P2], calibration, first_loop_extra_ns=extra)
        equal = canonical(baseline) == canonical(corrected)
        if not equal:
            raise ValueError("zero-vector full-prediction regression failed")
        regressions.append(dict(audit_only_beta=beta, fitted_price=False,
            extra_nonzero_ctas=sum(x != 0 for x in extra), full_serialized_prediction_equal=equal,
            frequency_equal=baseline["frequency_ghz"] == corrected["frequency_ghz"],
            all_cta_events_equal=baseline["cta_events"] == corrected["cta_events"],
            plain_us_equal=baseline["plain_us"] == corrected["plain_us"]))
    p2_score = scores[P2]
    proof = dict(status="rejected_structural", case=P2,
        argument="Every first cluster has zero A and B descriptor fill. The first-only additive vector is zero for every finite nonnegative coefficient. Frozen supply, free-clock closure, all CTA events and complete time are unchanged. Required strict P2 error reduction is impossible.",
        static_inputs_only=True, target_trace_used_for_scope=False, regressions=regressions,
        frozen_r2_plain_us=p2_score["predicted_plain_us"], observed_plain_us=p2_score["observed_plain_us"],
        frozen_r2_signed_relative_error=p2_score["relative_error"],
        exact_same_runtime_baseline_vs_zero=True,
        recomputed_minus_frozen_plain_us=baseline["plain_us"] - p2_score["predicted_plain_us"],
        baseline_prediction_sha256=hashlib.sha256(canonical(baseline).encode()).hexdigest(),
        acceptance=dict(target_P2_strict_improvement=False,
            target_P1="not_priced_after_decisive_rejection", non_target="not_claimed_after_rejection",
            no_frozen_candidate=True, GPU_jobs=0))

    folds = []
    for run in ("Q02-tma-request/R18-first-fill/20261009-R18-first-fill-job738707", "Q02-tma-request/R18-first-fill/20261009-R18-first-fill-job738865"):
        root = base / run
        reference = load(root / "cases.json")
        reference_setup = load(root / "static_setup.json")
        environment = load(root / "environment.json")
        if environment["host"] != "romeo-a057" or not environment["gpu"].startswith("GPU-43269fbc-"):
            raise ValueError("pricing reference card changed")
        cfgs = {r["config"] for r in reference}
        shapes = sorted({(r["m"], r["n"]) for r in reference})
        if len(cfgs) != 1 or len(shapes) != 1 or sorted({r["k"] for r in reference}) != [1024, 4096]:
            raise ValueError("reference shape/K premise changed")
        if {r["id"] for r in reference} != {r["case"] for r in reference_setup}:
            raise ValueError("reference setup identity changed")
        folds.append(dict(config=next(iter(cfgs)), job=int(environment["job"]), host=environment["host"],
            gpu=environment["gpu"], heldout_MN=list(shapes[0]), case_ids=[r["id"] for r in reference],
            K_values=[1024, 4096], remaining_same_config_shapes=0, status="not_estimable",
            reason="Leaving the sole M×N shape removes every same-config pricing condition.",
            borrowed_from_V09=False, borrowed_across_config=False))
    params = dict(schema="v10-taskA-cluster-residual/1", status="rejected_structural",
        beta_ns_per_ktile_per_fill_fraction={"cfg_a": None, "cfg_c": None},
        pricing="not_performed_after_parameter_independent_rejection",
        unit=contract["form"]["unit"], form=contract["form"]["correction"],
        placement="outside frozen r2 max; first L only; retain frozen own-fill term",
        intended_reference_K_span=[1024, 4096], calibrated_operand=residual.CALIBRATED_OPERAND,
        granted_support=[], rejection_case=P2,
        null_is_not_zero_price=True, not_frozen=True)
    errors = [abs(r["frozen_r2_signed_relative_error"]) for r in scoreable]
    comparison = dict(fixed_scoreable_conditions=29, all_prelisted_conditions=30,
        new_range_out_numeric_diagnostics=7, previous_cfg_c_G2_rejection=1,
        numeric_baseline_retained_for_all29=True, rows=table,
        frozen_r2_absolute_error_percent=dict(median=statistics.median(errors)*100, maximum=max(errors)*100),
        failed_cluster_max_metrics_preserved=old["metrics"],
        candidate_adopted=False, rejection="P2 error cannot strictly decrease for any beta",
        non_target_acceptance="not_evaluated_for_unpriced_rejected_family")
    # Fresh replay checks all recorded inputs before writing to an exclusive directory.
    prior = bundle / "provenance.json"
    if prior.exists():
        if json.loads(prior.read_text())["input_sha256"] != input_hashes:
            raise ValueError("recorded inputs changed")
    output.mkdir(parents=True, exist_ok=False)
    write(output, "cluster-residual.json", params)
    write(output, "static-scope.json", static)
    write(output, "structural-proof.json", proof)
    write(output, "comparison.json", comparison)
    write(output, "leave-MN-shape.json", dict(diagnostic="new component leave-M×N shape only",
        full_model_leaveout=False, folds=folds, prices_fitted=False))
    write(output, "provenance.json", dict(input_sha256=input_hashes,
        model_snapshot_sha256={name: item["sha256"] for name,item in source_manifest.items()},
        analyzer_sha256=digest(Path(__file__)), static_interface_sha256=digest(Path(residual.__file__)),
        GPU_jobs=0, raw_samples_read=False, target_trace_read=False,
        PLAN_sha256_at_declaration=contract["plan_sha256"]))
    print(json.dumps(dict(status=proof["status"], fixed29=len(scoreable), outside7=len(outside),
        P2_zero_vector=True, full_output_equal=True, leave_shape="not_estimable", GPU_jobs=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.base.resolve(), args.bundle.resolve(), args.output.resolve())
