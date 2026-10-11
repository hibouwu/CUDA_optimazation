#!/usr/bin/env python3
"""R17 qualified reanalysis; preserve the original run and frozen V06 predictions.

Compare first-tile windows at the same physical boundary coordinates with a fully
valid descriptor and identical padded work/grid. The measured difference is an
extra window, not an absolute slope. All checks here are post-hoc diagnostics.

  r17_rule.py --run R17RUN --v06 V06RUN [--output NEW_ANALYSIS_DIRECTORY]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import v06_model as model
import r17_run  # noqa: F401 (register cfg_a1)


def eligibility(run):
    """Recompute timing eligibility from process records, not summary labels."""
    records = [json.loads(s) for s in (run / "samples/r17.jsonl").read_text().splitlines() if s]
    result = {}
    for cid in sorted({r["case"] for r in records}):
        rows = [r for r in records if r["case"] == cid]
        plain = [r["elapsed_us"] for r in rows if r["variant"] == "plain"]
        traced = [r["elapsed_us"] for r in rows if r["variant"] == "stamped"]
        reasons = []
        if len(plain) != 10 or len(traced) != 10:
            reasons.append("expected_10_plain_and_10_stamped_processes")
        if any(r["returncode"] != 0 or r.get("check") != "ok" or
               r.get("padding_errors", 0) != 0 for r in rows):
            reasons.append("execution_or_correctness_failure")
        if any(r.get("last5_cv", 1.0) > 0.02 for r in rows):
            reasons.append("warmup_not_converged")
        perturbation = (statistics.median(traced) / statistics.median(plain) - 1
                        if plain and traced else None)
        cv = statistics.pstdev(plain) / statistics.mean(plain) if plain else None
        trace_cv = statistics.pstdev(traced) / statistics.mean(traced) if traced else None
        if perturbation is None or abs(perturbation) > 0.05:
            reasons.append("trace_perturbation_above_5_percent")
        if cv is None or cv > 0.05:
            reasons.append("plain_cv_above_5_percent")
        if trace_cv is None or trace_cv > 0.05:
            reasons.append("trace_cv_above_5_percent")
        result[cid] = dict(qualified=not reasons, reasons=reasons,
                           perturbation=perturbation, plain_cv=cv, trace_cv=trace_cv)
    return result, records


def rule_rates(run, qualified=None):
    """Descriptive average cycles/Ktile. These values are NOT slope parameters."""
    if qualified is None:
        qualified, _ = eligibility(run)
    kinds = {"oob": [], "partial": [], "both": []}
    with (run / "classes.csv").open() as stream:
        classes = list(csv.DictReader(stream))
    for r in classes:
        if r["config"] != "cfg_a" or r["round"] != "0":
            continue
        if not qualified[r["case"]]["qualified"]:
            continue
        m = int(r["m"])
        odd, part = model.cdiv(m, 128) % 2 == 1, m % 128 != 0
        kind = "both" if odd and part else "oob" if odd else "partial" if part else None
        if kind and r["cls"] in ("oob", "partial"):
            kinds[kind].append(float(r["per_ktile"]))
    return {k: statistics.median(v) for k, v in kinds.items() if v}, kinds


def first_tile_windows(run, records, row, boundary):
    """Median per process at identical physical first-tile row coordinates."""
    values = []
    coordinates = None
    grid = None
    for r in records:
        if r["case"] != row["id"] or r["variant"] != "stamped":
            continue
        rec = model.load_process(run / r["raw"])
        if rec["check"]["status"] != "ok":
            raise ValueError(f"raw correctness failed: {r['raw']}")
        work = model.scheduled_work(row["config"], row["m"], row["n"], rec["setup"]["grid"])
        selected = [(w[0], c["tiles"][0][1] - c["tiles"][0][0])
                    for c, w in zip(rec["ctas"], work) if w and w[0][0] in boundary]
        coords = sorted(t for t, _ in selected)
        if not selected:
            raise ValueError(f"no boundary first-tile observations: {row['id']}")
        if coordinates is not None and (coords != coordinates or grid != rec["setup"]["grid"]):
            raise ValueError("work coordinates changed between processes")
        coordinates, grid = coords, rec["setup"]["grid"]
        values.append(statistics.median(v for _, v in selected))
    return dict(window_cycles=statistics.median(values), process_windows=values,
                coordinates=coordinates, grid=grid)


def qualified_pairs(run, qualified, records):
    rows = json.loads((run / "cases.json").read_text())
    lookup = {(r["config"], r["m"], r["n"], r["k"]): r for r in rows}
    pairs, rejected = [], []
    for r in rows:
        if r["config"] != "cfg_a":
            continue
        odd, part = model.cdiv(r["m"], 128) % 2 == 1, r["m"] % 128 != 0
        if not odd and not part:
            continue
        valid_m = model.cdiv(model.cdiv(r["m"], 128), 2) * 256
        control = lookup.get(("cfg_a", valid_m, r["n"], r["k"]))
        if control is None:
            rejected.append(dict(case=r["id"], reason="no_same_padded_grid_control"))
            continue
        if not qualified[r["id"]]["qualified"] or not qualified[control["id"]]["qualified"]:
            rejected.append(dict(case=r["id"], control=control["id"],
                                 reason="case_or_control_timing_unqualified"))
            continue
        last = model.cdiv(r["m"], 128) - 1
        boundary = {last, last ^ 1}
        observed = first_tile_windows(run, records, r, boundary)
        baseline = first_tile_windows(run, records, control, boundary)
        if observed["coordinates"] != baseline["coordinates"] or observed["grid"] != baseline["grid"]:
            raise ValueError("case/control do not have identical physical boundary work")
        kt = model.cdiv(r["k"], 64)
        delta = observed["window_cycles"] - baseline["window_cycles"]
        pairs.append(dict(case=r["id"], control=control["id"], kind="both" if odd and part
                          else "oob" if odd else "partial", m=r["m"], n=r["n"], k=r["k"],
                          kt=kt, boundary_rows=sorted(boundary), grid=observed["grid"],
                          boundary_coordinate_count=len(observed["coordinates"]),
                          observed_cycles=observed["window_cycles"],
                          control_cycles=baseline["window_cycles"], extra_cycles=delta,
                          extra_per_ktile=delta / kt,
                          case_process_windows=observed["process_windows"],
                          control_process_windows=baseline["process_windows"],
                          comparison="difference_of_process_medians_not_temporally_paired"))
    return pairs, rejected


def fit_descriptions(pairs):
    """Do not pool different grids/shapes into a hardware slope."""
    groups = {}
    for p in pairs:
        groups.setdefault((p["kind"], p["m"], p["n"]), []).append(p)
    out = []
    for (kind, m, n), points in sorted(groups.items()):
        xs = [p["kt"] for p in points]
        entry = dict(kind=kind, m=m, n=n, cases=[p["case"] for p in points],
                     exportable=False, distinct_kt=len(set(xs)))
        if len(set(xs)) < 2:
            entry["status"] = "fixed_term_and_slope_not_identifiable"
        else:
            ys = [p["extra_cycles"] for p in points]
            xm, ym = statistics.mean(xs), statistics.mean(ys)
            b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm)**2 for x in xs)
            a = ym - b * xm
            entry.update(status="descriptive_fit_without_independent_check", a_cycles=a,
                         b_cycles_per_ktile=b,
                         residual_cycles=[a + b * x - y for x, y in zip(xs, ys)])
        out.append(entry)
    return out


def predict_with_increment(row, grid, params, rule, fixed_us, extra_cycles):
    """Add a measured extra window only to boundary CTAs whose first work tile is affected."""
    sched = model.CONFIGS[row["config"]]["schedule"]
    work = model.scheduled_work(row["config"], row["m"], row["n"], grid)
    kt = model.cdiv(row["k"], 64)
    last = model.cdiv(row["m"], 128) - 1
    boundary = {last, last ^ 1}
    slow = dict(params, dL0=params["dL0"] + extra_cycles)
    per = [slow if w and w[0][0] in boundary else params for w in work]
    cyc = [model.cta_cycles(p, sched, len(w), kt) for p, w in zip(per, work)]
    i = max(range(len(cyc)), key=cyc.__getitem__)
    c_max, stages = model.cta_cycles(per[i], sched, len(work[i]), kt, detail=True)
    phi = sum(cyc) / (model.SMS * c_max)
    mu = stages["mainloop"] / c_max
    dbytes = model.dram_bytes(row["config"], row["m"], row["n"], row["k"], work)
    w = c_max / 1.6e3
    for _ in range(200):
        f = model.clock_ghz(rule, phi, math.log(w), dbytes / (w * 1e-6) / 1e12, mu)
        w = 0.5 * w + 0.5 * c_max / (f * 1e3)
    return dict(critical_cycles=c_max, clock_ghz=f, predicted_us=fixed_us + w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--v06", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    run, v06 = args.run.resolve(), args.v06.resolve()
    output = args.output or run / "reanalysis/qualified-increments-v1"
    output.mkdir(parents=True, exist_ok=False)
    qualified, records = eligibility(run)
    rates, points = rule_rates(run, qualified)
    pairs, rejected = qualified_pairs(run, qualified, records)
    frozen = json.loads((v06 / "v06-predictions.json").read_text())
    with (v06 / "cases.csv").open() as stream:
        measured = {r["case"]: r for r in csv.DictReader(stream)}
    checks = []
    for cid in ("cfg_a_partial_wave", "cfg_a_long_k"):
        prediction = frozen["predictions"][cid]
        row = prediction["row"]
        pair = next(p for p in pairs if (p["m"], p["n"], p["k"]) == (row["m"], row["n"], row["k"]))
        q = predict_with_increment(row, prediction["grid"], frozen["params"]["cfg_a"],
                                   frozen["clock_rule"], frozen["fixed_us"]["cfg_a"],
                                   pair["extra_cycles"])
        checks.append(dict(case=cid, extra_cycles=pair["extra_cycles"], **q,
                           measured_us=float(measured[cid]["meas_us"]),
                           time_error=q["predicted_us"] / float(measured[cid]["meas_us"]) - 1,
                           cycle_error=q["critical_cycles"] / float(measured[cid]["meas_cycles"]) - 1))
    result = dict(schema_version=2, qualification="posthoc_diagnostic_not_heldout",
                  note="extra windows from matched physical coordinates; no transferable boundary slope yet",
                  eligibility=qualified, descriptive_average_cycles_per_ktile=rates,
                  descriptive_points=points, paired_increments=pairs, rejected_pairs=rejected,
                  fit_descriptions=fit_descriptions(pairs), v06_point_diagnostics=checks,
                  original_rule_check_sha256=model.sha(run / "rule_check.json"),
                  frozen_v06_prediction_sha256=model.sha(v06 / "v06-predictions.json"),
                  rule_code_sha256=model.sha(Path(__file__)))
    (output / "qualified-increments.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), descriptive_rates=rates, accepted_pairs=len(pairs),
                          rejected_pairs=rejected, fits=result["fit_descriptions"],
                          v06_point_diagnostics=checks), indent=2))


if __name__ == "__main__":
    main()
