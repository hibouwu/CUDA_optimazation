#!/usr/bin/env python3
"""R13: small max(compute, logical-source service) fits to existing V08 summaries.

Development diagnostics only. No GPU work or V08 refitting. Fits reuse summaries;
six cfg_a pitch cases receive a bounded trace replay to locate position effects.
L is a finite mainloop window including drain, not an internal steady-state timer.
Requires numpy/scipy; all input archives are read-only and output must be new.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import least_squares
from analyze_r18 import replay

# tm, tn, cluster_m, cluster_n; fixed from V08/CUTLASS configurations.
CONFIGS = {"cfg_a": (128, 128, 2, 1), "cfg_b": (128, 128, 1, 1),
           "cfg_c": (256, 128, 1, 2)}
MODELS = {
    "compute": [],
    "constant": ["q0"],
    "pitch": ["q0", "A", "B"],
    "pitch_input": ["q0", "A", "B", "input"],
    "pitch_K": ["q0", "A", "B", "K"],
    "cfg_a_pitch_input": ["q0", "A", "B", "input", "cfg_a_A", "cfg_a_B"],
}


def read_json(path):
    return json.loads(path.read_text())


def records(run, summary_path):
    cases = {c["id"]: c for c in read_json(run / "cases.json")}
    result = []
    for name, summary in read_json(summary_path).items():
        c = cases[name]
        tm, tn, cm, cn = CONFIGS[c["config"]]
        compute = 2 * tm * tn * 64 / 4096
        a_bytes, b_bytes = tm * 64 * 2 / cn, tn * 64 * 2 / cm
        lda, ldb, ldd = c.get("lda", c["k"]), c.get("ldb", c["n"]), c.get("ldd", c["n"])
        result.append(dict(
            case=name, config=c["config"], set=c["set"], group=c.get("group", "calib"),
            m=c["m"], n=c["n"], k=c["k"], kt=summary["kt"], swizzle=c["swizzle"],
            lda=lda, ldb=ldb, ldd=ldd, a_mod128=2 * lda % 128, b_mod128=2 * ldb % 128,
            d_mod128=4 * ldd % 128, full_cluster=c["m"] % (tm * cm) == 0 and c["n"] % (tn * cn) == 0,
            compute=compute, source_A_B=a_bytes, source_B_B=b_bytes, source_KiB=(a_bytes + b_bytes) / 1024,
            input_MiB=2 * c["k"] * (c["m"] + c["n"]) / 2**20,
            logical_ABD_MiB=summary["fp"], allocation_ABD_MiB=(2*c["m"]*lda + 2*c["k"]*ldb + 4*c["m"]*ldd) / 2**20,
            L=summary["intervals"].get("L"), L0=summary["intervals"]["L0"],
            in_tile_pooled_median=summary["tile_L"].get("in", {}).get("median"),
            tile_classes=list(summary["tile_L"]), boundary=summary["boundary"],
            perturbation=summary["perturbation"], plain_cv=summary["plain_cv"], stamped_cv=summary["stamped_cv"],
        ))
    return result


def exclusions(row):
    reasons = []
    if row["swizzle"] != 1:
        reasons.append("swizzle_not_1")
    if not row["full_cluster"] or row["boundary"]:
        reasons.append("M_N_cluster_tail")
    if row["k"] % 64:
        reasons.append("K_tail")
    if row["d_mod128"]:
        reasons.append("D_pitch_control")
    if row["L"] is None:
        reasons.append("no_later_tile")
    if abs(row["perturbation"]) > .05 or max(row["plain_cv"], row["stamped_cv"]) > .05:
        reasons.append("existing_5pct_interval_qualification")
    return reasons


def baseline(calibration):
    # These are window nuisance constants, not measured startup/drain constants.
    result = {}
    for cfg in CONFIGS:
        rows = [r for r in calibration if r["config"] == cfg and r["set"] == "calib" and not exclusions(r)]
        later = [r["L"] - r["kt"] * r["compute"] for r in rows]
        first = [r["L0"] - r["kt"] * r["compute"] for r in rows]
        result[cfg] = dict(cases=[r["case"] for r in rows], later=statistics.median(later),
                           first=statistics.median(first), low=max(0, min(later)),
                           later_range=[min(later), max(later)])
    return result


def features(row, model):
    a, b = float(row["a_mod128"] != 0), float(row["b_mod128"] != 0)
    values = dict(q0=1., A=a, B=b, input=np.log2(1 + row["input_MiB"] / 32),
                  K=np.log2(row["kt"] / 16), cfg_a_A=a * (row["config"] == "cfg_a"),
                  cfg_a_B=b * (row["config"] == "cfg_a"))
    return [values[key] for key in MODELS[model]]


def predict(rows, model, coefficients, offsets, first=False):
    compute = np.array([r["compute"] for r in rows])
    service = np.zeros(len(rows))
    if MODELS[model]:
        service = np.array([r["source_KiB"] for r in rows]) * (np.array([features(r, model) for r in rows]) @ coefficients)
    b = np.array([offsets[r["config"]]["first" if first else "later"] for r in rows])
    return b + np.array([r["kt"] for r in rows]) * np.maximum(compute, service), service


def fit(rows, model, offsets):
    if not MODELS[model]:
        return np.array([])
    y = np.array([r["L"] for r in rows])
    # Start above the compute plateaus as well as near them; an inactive max
    # branch has zero derivative and a single low start can miss the fit.
    solutions = []
    for start in (16., 24., 40.):
        initial = np.array([start] + [1.] * (len(MODELS[model]) - 1))
        solution = least_squares(lambda p: predict(rows, model, p, offsets)[0] / y - 1,
                                 initial, bounds=(0, np.inf), ftol=1e-11, xtol=1e-11, gtol=1e-11)
        solutions.append(solution)
    return min(solutions, key=lambda s: s.cost).x


def metrics(errors):
    errors = np.asarray(errors)
    return dict(n=len(errors), median_abs_pct=float(np.median(abs(errors)) * 100),
                max_abs_pct=float(max(abs(errors)) * 100), rms_pct=float(np.sqrt(np.mean(errors**2)) * 100),
                signed_mean_pct=float(np.mean(errors) * 100))


def score(rows, predictions, field="L"):
    errors = predictions / np.array([r[field] for r in rows]) - 1
    result = metrics(errors)
    result["per_config"] = {cfg: metrics([e for r, e in zip(rows, errors) if r["config"] == cfg])
                            for cfg in CONFIGS if any(r["config"] == cfg for r in rows)}
    return result


def leave_groups(rows, model, offsets, axis):
    def key(r):
        return f'{r["m"]}x{r["n"]}' if axis == "shape" else str(r[axis])
    folds = []
    for group in sorted({key(r) for r in rows}):
        train = [r for r in rows if key(r) != group]
        test = [r for r in rows if key(r) == group]
        coefficients = fit(train, model, offsets)
        # A removed group can be the sole source of a feature. Do not silently
        # treat an unobserved coefficient as an identified zero.
        xtrain = np.array([features(r, model) for r in train])
        missing = [j for j in range(len(MODELS[model])) if np.all(xtrain[:, j] == 0)]
        supported = [r for r in test if all(features(r, model)[j] == 0 for j in missing)]
        # Without cfg_b, cfg_c stays on the compute plateau: only cfg_a's
        # A+cfg_a_A and B+cfg_a_B sums are identified. Their split changes
        # cfg_b pitch predictions without changing any training prediction.
        confounded = model == "cfg_a_pitch_input" and axis == "config" and group == "cfg_b"
        if confounded:
            supported = [r for r in supported if not r["a_mod128"] and not r["b_mod128"]]
        fold = dict(group=group, train_n=len(train), test_n=len(test),
                    missing_features=[MODELS[model][j] for j in missing],
                    nonunique_shared_cfg_a_pitch_split=confounded,
                    unsupported=[r["case"] for r in test if r not in supported])
        if supported:
            p, _ = predict(supported, model, coefficients, offsets)
            fold["score"] = score(supported, p)
            fold["residuals"] = [{"case": r["case"], "predicted_L": float(v), "relative_error": float(v/r["L"]-1)}
                                 for r, v in zip(supported, p)]
        folds.append(fold)
    return folds


def position_diagnostic(run, selected):
    """Only the six cfg_a pitch controls; reuse R18's existing trace/reference reader."""
    cases = {c["id"]: c for c in read_json(run / "cases.json")}
    observations = {r["case"]: r for r in selected}
    result = {}
    for kind in ("ref", "apitch", "bpitch"):
        for k in (1024, 4096):
            name = f"cfg_a_p_{kind}_k{k}"
            groups, counts, identities, later = {}, {}, {}, []
            for path in sorted((run / "samples" / name).glob("stamped-*.json")):
                record = read_json(path)
                process = replay(run, record, cases[name])
                identities[record["raw"]] = record["raw_sha256"]
                values, rest = {}, []
                for cta in process["ctas"]:
                    for j, t in enumerate(cta["tiles"]):
                        duration = t[1] - t[0]
                        if j:
                            rest.append(duration)
                        position = "first" if j == 0 else "last" if j == len(cta["tiles"])-1 else "middle"
                        for key in (f"j{j}", position):
                            values.setdefault(key, []).append(duration)
                later.append(statistics.fmean(rest))
                for key, durations in values.items():
                    groups.setdefault(key, []).append(statistics.fmean(durations))
                    counts[key] = len(durations)
            reconstructed = statistics.median(later)
            if not np.isclose(reconstructed, observations[name]["L"], rtol=0, atol=1e-8):
                raise ValueError("position replay differs from summary L: " + name)
            result[name] = dict(processes=len(later), reconstructed_L=reconstructed,
                raw_sha256=identities, checked_values=4096*len(later),
                positions={key: dict(records_per_process=counts[key],
                    median_process_mean_cycle_per_Ktile=statistics.median(v)/(k//64),
                    process_means_cycles=v) for key, v in groups.items()})
    return result


def analyze(run, calibration_run, output):
    summary_path = run / "reanalysis/followup-v1/summary.json"
    followup_path = run / "reanalysis/followup-v1/followup.json"
    cal_path = calibration_run / "derived/summary.json"
    rows = records(run, summary_path)
    for r in rows:
        r["excluded"] = exclusions(r)
    selected = [r for r in rows if not r["excluded"]]
    offsets = baseline(records(calibration_run, cal_path))
    # Tail and D-pitch cases remain visible, but the failed original interval
    # qualification is not used quantitatively even in tail diagnostics.
    tail = [r for r in rows if r["swizzle"] == 1 and r["excluded"] and
            "existing_5pct_interval_qualification" not in r["excluded"]]
    results, residuals = {}, []
    for model in MODELS:
        coefficients = fit(selected, model, offsets)
        p, service = predict(selected, model, coefficients, offsets)
        pfirst, _ = predict(selected, model, coefficients, offsets, first=True)
        result = dict(coefficients_cycle_per_KiB=dict(zip(MODELS[model], coefficients.tolist())),
                      later=score(selected, p), first_transfer=score(selected, pfirst, "L0"),
                      compute_branch_cases=[r["case"] for r, s in zip(selected, service) if s <= r["compute"]])
        result["tail_D_transfer"] = score(tail, predict(tail, model, coefficients, offsets)[0])
        result["leave_groups"] = {axis: leave_groups(selected, model, offsets, axis) for axis in ("config", "kt", "shape")}
        result["offset_sensitivity"] = {}
        for kind in ("zero", "low"):
            changed = {cfg: dict(o, later=0. if kind == "zero" else o["low"]) for cfg, o in offsets.items()}
            coef = fit(selected, model, changed)
            result["offset_sensitivity"][kind] = score(selected, predict(selected, model, coef, changed)[0])
        results[model] = result
        for r, v, f, s in zip(selected, p, pfirst, service):
            residuals.append(dict(case=r["case"], config=r["config"], model=model,
                                  measured_L=r["L"], predicted_L=float(v), relative_error=float(v/r["L"]-1),
                                  measured_L0=r["L0"], predicted_L0=float(f), first_relative_error=float(f/r["L0"]-1),
                                  branch="compute_lower_bound_only" if s <= r["compute"] else "conditional_service_candidate"))
    pairs = []
    index = {r["case"]: r for r in selected}
    for cfg in CONFIGS:
        for kind in ("ref", "apitch", "bpitch"):
            lo, hi = [index[f"{cfg}_p_{kind}_k{k}"] for k in (1024, 4096)]
            pairs.append(dict(config=cfg, condition=kind,
                              later_secant=(hi["L"]-lo["L"])/(hi["kt"]-lo["kt"]),
                              first_secant=(hi["L0"]-lo["L0"])/(hi["kt"]-lo["kt"]),
                              note="K and input footprint change together; not a pure steady slope"))
    paths = [run / "cases.json", summary_path, followup_path, calibration_run / "cases.json", cal_path]
    report = dict(
        protocol="development diagnostic; not a new holdout or a physical bandwidth fit",
        formula="L = b_cfg + Kt * max(C_cfg, D_source_KiB * q); q is nonnegative cycle/KiB",
        features="A/B indicate nonzero byte pitch modulo 128; input=log2(1+logical_input_MiB/32); K=log2(Kt/16). 32MiB is feature scaling, not cache capacity.",
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_cases=len(rows), fit_cases=len(selected), tail_D_diagnostic_cases=len(tail),
        offsets=offsets, candidates=results, K_secants=pairs, observations=rows,
        cfg_a_positions=position_diagnostic(run, selected),
        limits=["L includes subsequent waits, loop work and final WGMMA drain; first/rest denote output tiles, not Ktiles.",
                "Compute-branch observations constrain effective capacity from below; no D/measured-cycle bandwidth labels are fitted.",
                "q combines requests, scheduling, stage reuse and cache state; no physical amplification or bandwidth separation.",
                "Input footprint is an allocation proxy, not measured L2 residency; K, geometry and reuse remain confounded.",
                "Only 16-byte pitch remainder is varied alone. Other remainders occur with K/N tails.",
                "Calibration offsets remain fixed across leave-group fits; these diagnostics do not independently validate those offsets.",
                "Case summaries lack per-process L distributions; full-event CV is not a confidence interval for L.",
                "Fits reuse existing summaries. Only six cfg_a pitch conditions are replayed (60 stamped processes, 245760 saved numeric values) with the existing R18 reader.",
                "Position means are per process then median; first position mean is not the summary L0 median. Ordinal CTA counts do not measure concurrent active SMs."])
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "residuals.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(residuals[0]))
        writer.writeheader()
        writer.writerows(residuals)
    print(json.dumps(dict(fit_cases=len(selected), offsets=offsets,
                         scores={k: v["later"] for k, v in results.items()}), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--calibration-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.run.resolve(), args.calibration_run.resolve(), args.output.resolve())
