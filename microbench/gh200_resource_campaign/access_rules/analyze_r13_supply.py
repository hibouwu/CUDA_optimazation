#!/usr/bin/env python3
"""R13 offline max(compute, logical-source service) development diagnostics.

No GPU work or modification of frozen predictions. Default V08F mode reuses
summaries plus six cfg_a trace controls; --sm-summary uses the archived SM scan;
--coverage-suite compares two address-coverage/prefill forms on separate cohorts.
--fill-suite separates valid-address and OOB-fill demands on one card in cycle/ns forms.
--compose-fill evaluates those windows in the existing CTA event recurrence.
--supply-calibration prepares per-CTA three-configuration dual-clock calibration.
L is a finite mainloop window including drain, not an internal steady-state timer.
Requires numpy/scipy; all input archives are read-only and output must be new.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import linprog
from analyze_r18 import replay
from supply_model import (
    CONFIGS, FILL_FIELDS, TMA_SOURCE_FIELDS, SUPPLY_FIELDS, address_box,
    static_tma_requests, supply_request_rows, supply_predict, supply_predict_bounds,
    supply_pitch_support, fill_cta_features, fill_window_cycles,
)

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
        supply_model_sha256=hashlib.sha256(Path(__file__).with_name('supply_model.py').read_bytes()).hexdigest(),
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


def sm_candidates(run, summary_path, output):
    """Compare per-SM and shared caps on one fixed-K card; no observed time input."""
    from collections import Counter, defaultdict
    from v08_model import scheduled_work

    source = read_json(summary_path)
    cases = {r["id"]: r for r in read_json(run / "cases.json")}
    setups = {r["case"]: r["setup"] for r in read_json(run / "static_setup.json")}
    if len({r["gpu_uuid"] for r in setups.values()}) != 1:
        raise ValueError("SM-scan fits require one GPU")
    rows = []
    for group in source["mainloop_groups"]:
        if not group["qualified"]:
            raise ValueError("this cap comparison requires qualified interval groups")
        r = cases[group["case"]]
        if r["config"] != "cfg_a" or r["k"] != 4096:
            raise ValueError("this comparison is the fixed-K cfg_a SM scan")
        work = scheduled_work(r["config"], r["m"], r["n"], setups[r["id"]]["grid"], r["swizzle"])
        rows.append(dict(case=r["id"], sm=r["sm_count"], pitch=r["pitch"],
            j=group["j"], T=group["T"], active=sum(len(w) > group["j"] for w in work),
            next_active=sum(len(w) > group["j"] + 1 for w in work),
            y=group["median_process_mean_L_per_kt"]))

    def inputs(observations, kind):
        n = np.array([1 if kind == "local" else r["sm"] if kind == "grid" else r["active"] for r in observations])
        if kind in ("prefill5", "prefill6"):
            credit = 5 if kind == "prefill5" else 6
            n = n - credit / 64 * (n - np.array([r["next_active"] for r in observations]))
        a = np.array([r["pitch"] == "a16" for r in observations])
        b = np.array([r["pitch"] == "b16" for r in observations])
        return 24 * n[:, None] * np.column_stack([np.ones(len(n)), a, b])

    def prediction(observations, kind, p):
        service = inputs(observations, kind) @ p[1:]
        return p[0] + np.maximum(512, service), service

    def fit_cap(observations, kind):
        counts = Counter(r["case"] for r in observations)
        weights = np.array([1 / counts[r["case"]] for r in observations])
        y = np.array([r["y"] for r in observations])
        scale = 132 if kind == "local" else 1
        solutions = []
        for start in (0.12, 0.17, 0.23):
            result = least_squares(lambda p: np.sqrt(weights) * (prediction(observations, kind, p)[0] / y - 1),
                [5, start * scale, .02 * scale, .02 * scale], bounds=(0, np.inf), max_nfev=2000)
            solutions.append(result)
        result = min(solutions, key=lambda r: r.cost)
        _, service = prediction(observations, kind, result.x)
        active = service > 512
        # Analytic derivatives avoid treating finite-difference noise as identified rank.
        jacobian = np.column_stack([np.ones(len(y)), inputs(observations, kind) * active[:, None]])
        rank = int(np.linalg.matrix_rank(jacobian))
        return result.x, dict(active_jacobian_rank=rank, parameters=4,
            aligned_service_groups=int(sum(on and r["pitch"] == "aligned" for on, r in zip(active, observations))),
            note="Rank describes this assumed piecewise model, not physical bandwidth identifiability.")

    def cap_score(observations, kind, p):
        values, _ = prediction(observations, kind, p)
        return metrics(values / np.array([r["y"] for r in observations]) - 1)

    train = [r for r in rows if r["j"] > 0]
    first = [r for r in rows if r["j"] == 0]
    fits, residuals = {}, []
    for kind in ("local", "grid", "wave", "prefill5", "prefill6"):
        p, identification = fit_cap(train, kind)
        folds = []
        for sm in sorted({r["sm"] for r in train}):
            fit_rows = [r for r in train if r["sm"] != sm]
            test_rows = [r for r in train if r["sm"] == sm]
            q, ident = fit_cap(fit_rows, kind)
            folds.append(dict(removed_sm=sm, parameters=q.tolist(), identification=ident,
                score=cap_score(test_rows, kind, q),
                limitation="Without an aligned service-limited training point, the base service coefficient can be only bounded; extrapolated aligned predictions may be nonunique." if not ident["aligned_service_groups"] else None))
        fits[kind] = dict(parameters=dict(zip(("window_floor_extra_cycle_per_Ktile", "q0", "qA", "qB"), p.tolist())),
            identification=identification, later_fit=cap_score(train, kind, p),
            first_transfer=cap_score(first, kind, p), leave_SM=folds)
        predictions, services = prediction(rows, kind, p)
        for r, value, service in zip(rows, predictions, services):
            residuals.append(dict(**r, model=kind, predicted=float(value),
                relative_error=float(value / r["y"] - 1), service_branch=bool(service > 512)))

    collision_groups = defaultdict(list)
    for r in train:
        collision_groups[r["active"], r["pitch"], r["T"]].append(r)
    collision_bounds = []
    for (active, pitch, total), group in collision_groups.items():
        lo, hi = min(group, key=lambda r: r["y"]), max(group, key=lambda r: r["y"])
        collision_bounds.append(dict(active=active, pitch=pitch, T=total, lower=lo, upper=hi,
            unavoidable_max_relative_error=(hi["y"] - lo["y"]) / (hi["y"] + lo["y"])))

    report = dict(protocol="Offline model development on one card and one M/N/K; not new frozen validation.",
        formula="L/Kt = b + max(512, 24*N*(q0+qA*A16+qB*B16)); b and q are nonnegative.",
        modes=dict(local="N=1, per-SM cap", grid="N=requested persistent grid", wave="N=count of software CTA work lists containing output index j",
            prefill5="N=Nj-5/64*(Nj-Nj+1), assumed five-stage source credit",
            prefill6="N=Nj-6/64*(Nj-Nj+1), assumed six-stage source credit"),
        weighting="Each case has equal total fit weight; its later (j,T) groups share that weight equally.",
        scope="cfg_a, M=2304, N=3072, K=4096, swizzle1, dyadic17; frame b is not transferable in K from this fit.",
        gpu=next(iter(setups.values()))["gpu_uuid"], fits=fits, collision_bounds=collision_bounds,
        limits=["24 KiB is logical multicast-adjusted source demand, not measured physical traffic.",
                "Software wave count is predictable but is not measured simultaneous mainloop activity.",
                "Prefill candidates assume five or six Ktiles already available and replacement demand for the next output; this is not a measurement of pipeline credits.",
                "Six is the configured StageCount; five is a fixed sensitivity check, not a fitted extra parameter.",
                "No measured frequency, time, output overlap or target L is used as a prediction feature.",
                "First-output windows are transfer diagnostics; only later windows train these parameters.",
                "The leave-132 test extrapolates beyond training pressure and may not identify the base service cap.",
                "A good in-range fit does not identify shared-resource location or separate cache history from concurrency."],
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [summary_path, run / "cases.json", run / "static_setup.json"]},
        supply_model_sha256=hashlib.sha256(Path(__file__).with_name('supply_model.py').read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "residuals.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(residuals[0])); writer.writeheader(); writer.writerows(residuals)
    print(json.dumps({k:dict(fit=v["later_fit"], first=v["first_transfer"], rank=v["identification"]["active_jacobian_rank"]) for k,v in fits.items()}, indent=2))


def source_request(row, mi, ni):
    """Valid source KiB/Ktile with tails/multicast; OOB zero-fill service is absent."""
    tm, tn, cm, cn = CONFIGS[row['config']]
    kt = (row['k'] + 63)//64
    values = np.zeros(5)
    for start in range(0, row['k'], 64):
        kk = min(64, row['k']-start)
        a = address_box(min(tm, row['m']-mi*tm), 2*kk, 2*row['lda'],
                        2*(mi*tm*row['lda']+start) % 128)
        b = address_box(kk, 2*min(tn, row['n']-ni*tn), 2*row['ldb'],
                        2*(start*row['ldb']+ni*tn) % 128)
        values += [a[0]/cn+b[0]/cm, a[1]/cn, a[2]/cn, b[1]/cm, b[2]/cm]
    return values/kt


def wave_requests(row, setup):
    from v08_model import scheduled_work
    work = scheduled_work(row['config'], row['m'], row['n'], setup['grid'], row['swizzle'])
    counts = Counter((j, len(w)) for w in work for j in range(len(w)))
    wave = [sum((source_request(row, *w[j]) for w in work if len(w) > j), np.zeros(5))
            for j in range(max(map(len, work)))]
    # Configured capacity is an assumption about prefill credit, not observed occupancy.
    fraction = min(setup['stages'], (row['k']+63)//64) / ((row['k']+63)//64)
    effective = [(1-fraction)*v + fraction*(wave[j+1] if j+1 < len(wave) else np.zeros(5))
                 for j, v in enumerate(wave)]
    return counts, effective


def coverage_rows(run, kind):
    """One card and one observer cohort per fit; never pool coefficients across runs."""
    cases = {r['id']: r for r in read_json(run/'cases.json')}
    setups = {r['case']: r['setup'] for r in read_json(run/'static_setup.json')}
    features_by_case = {k: wave_requests(r, setups[k]) for k, r in cases.items()
                        if kind != 'v08' or (r['swizzle']==1 and r['m']%256==0 and r['n']%256==0 and r['k']%64==0)}
    observations, paths = [], [run/'cases.json', run/'static_setup.json', run/'environment.json']

    def add(name, j, total, value, parts=None):
        r = cases[name]; tm, tn, _, _ = CONFIGS[r['config']]
        observations.append(dict(case=name, config=r['config'], kt=(r['k']+63)//64,
            C=2*tm*tn*64/4096, j=j, T=total, y=value,
            parts=parts or [(1., features_by_case[name][1][j].tolist())]))

    if kind == 'sm':
        path = run/'reanalysis/manager-formal-replay/summary.json'; paths.append(path)
        for g in read_json(path)['mainloop_groups']:
            if not g['qualified']: raise ValueError('unqualified SM interval')
            add(g['case'], g['j'], g['T'], g['median_process_mean_L_per_kt'])
    elif kind == 'v08':
        path = run/'reanalysis/followup-v1/summary.json'; paths.append(path)
        for r in records(run, path):
            if exclusions(r): continue
            counts, effective = features_by_case[r['case']]
            count = sum(n for (j, _), n in counts.items() if j)
            weights = Counter()
            for (j, _), n in counts.items():
                if j: weights[j] += n/count
            add(r['case'], -1, None, r['L']/r['kt'], [(w, effective[j].tolist()) for j,w in weights.items()])
            add(r['case'], 0, None, r['L0']/r['kt'])
    else:
        groups = defaultdict(list)
        if kind == 'joint':
            path = run/'reanalysis/B-20261009-matched-joint-final/tiles.csv'; paths.append(path)
            with path.open() as stream:
                for t in csv.DictReader(stream):
                    groups[t['case'], int(t['trial']), int(t['j']), int(t['T'])].append(float(t['L_cycles']))
        else:
            # Only the 50 dual records: existing summaries lack p16/p32 position cycles.
            for name, row in cases.items():
                for path in sorted((run/'samples'/name).glob('dual-*.json')):
                    rec = read_json(path)
                    if rec['returncode']: continue
                    paths += [path, run/rec['raw']]
                    for cta in replay(run, rec, row)['ctas']:
                        for j,t in enumerate(cta['tiles']):
                            groups[name, rec['trial'], j, len(cta['tiles'])].append(t[1]-t[0])
        across = defaultdict(list)
        for (name, trial, j, total), values in groups.items():
            across[name,j,total].append(statistics.fmean(values)/((cases[name]['k']+63)//64))
        for (name,j,total), values in across.items():
            if len(values)!=10: raise ValueError('missing dual process')
            add(name,j,total,statistics.median(values))
    return observations, paths, cases, setups


def coverage_design(rows, model):
    arrays = []
    for r in rows:
        x = np.array([p[1] for p in r['parts']])
        if model == 'pooled': x = np.column_stack([x[:,0], x[:,1]+x[:,3], x[:,2]+x[:,4]])
        arrays.append(x)
    return arrays


def coverage_fit(rows, model):
    configs = sorted({r['config'] for r in rows}); nb = len(configs)
    names = ['window_cycles_'+c for c in configs] + (['q0','q32','q128'] if model=='pooled' else ['q0','qA32','qA128','qB32','qB128'])
    arrays = coverage_design(rows, model)
    counts = Counter(r['case'] for r in rows)
    weights = np.sqrt([1/counts[r['case']] for r in rows]); y = np.array([r['y'] for r in rows])

    def predict(p):
        return np.array([p[configs.index(r['config'])]/r['kt'] +
            np.dot([v[0] for v in r['parts']], np.maximum(r['C'], x@p[nb:])) for r,x in zip(rows,arrays)])

    solutions = [least_squares(lambda p: weights*(predict(p)/y-1),
                  [300.]*nb+[start]+[.05]*(len(names)-nb-1), bounds=(0,np.inf), max_nfev=3000,
                  ftol=1e-11, xtol=1e-11, gtol=1e-11) for start in (.08,.17,.3)]
    solution = min(solutions, key=lambda z:z.cost); p = solution.x
    jacobian, rhs, ub, bounds = [], [], [], []
    for r,x in zip(rows,arrays):
        active = x@p[nb:] > r['C']
        derivative = np.zeros(len(p)); derivative[configs.index(r['config'])] = 1/r['kt']
        derivative[nb:] = np.array([v[0] for v in r['parts']]) @ (x*active[:,None])
        jacobian.append(derivative); rhs.append(float(derivative@p))
        for a,on in zip(x,active):
            constraint=np.r_[np.zeros(nb), -a if on else a]
            ub.append(constraint); bounds.append(-r['C'] if on else r['C'])
    jacobian=np.array(jacobian); norms=np.linalg.norm(jacobian,axis=0)
    scaled=jacobian/np.where(norms>0,norms,1)
    singular=np.linalg.svd(scaled,compute_uv=False)
    rank=int(np.linalg.matrix_rank(scaled, tol=1e-9))
    constraints=dict(A_eq=jacobian,b_eq=rhs,A_ub=np.array(ub),b_ub=bounds,bounds=(0,None),method='highs')

    def linear_range(vector):
        low=linprog(vector,**constraints); high=linprog(-vector,**constraints)
        if low.status not in (0,3) or high.status not in (0,3): raise ValueError('equivalent-fit LP failed')
        return [float(low.fun) if low.success else None, float(-high.fun) if high.success else None]

    ranges={name:linear_range(np.eye(len(p))[i]) for i,name in enumerate(names)}
    combinations={}
    for label,left,right in ([('shared','q32','q128')] if model=='pooled' else
                             [('A','qA32','qA128'),('B','qB32','qB128')]):
        vector=np.zeros(len(p)); vector[names.index(left)]=1; vector[names.index(right)]=7
        combinations[label+'_32_plus_7x128']=linear_range(vector)
    unique={name:(lo is not None and hi is not None and abs(hi-lo)<1e-7*max(1,abs(lo)))
            for name,(lo,hi) in ranges.items()}
    fitted=predict(p)
    return dict(parameters={name:float(v) if unique[name] else None for name,v in zip(names,p)},
        optimizer_representative=dict(zip(names,p.tolist())), parameter_ranges=ranges,
        full_box_16B_combinations=combinations,
        active_jacobian_rank=rank, parameter_count=len(p), scaled_singular_values=singular.tolist(),
        fit=metrics(fitted/y-1), residuals=[dict(case=r['case'],j=r['j'],T=r['T'],observed=r['y'],
            predicted=float(v), relative_error=float(v/r['y']-1)) for r,v in zip(rows,fitted)],
        range_scope='All nonnegative parameters retaining this fitted active-region prediction; null upper endpoint means unbounded. Not a noise confidence interval.'), p, configs, constraints


def coverage_predict(row, model, parameters, configs):
    x=coverage_design([row],model)[0]
    return parameters[configs.index(row['config'])]/row['kt'] + np.dot(
        [p[0] for p in row['parts']],np.maximum(row['C'],x@parameters[len(configs):]))


def coverage_probe_range(row, model, parameters, configs, constraints):
    """Exact equivalent-fit range for one proposed output window, not an aggregate."""
    x=coverage_design([row],model)[0]
    if len(x)!=1: raise ValueError('probe must describe one output window')
    nb=len(configs); floor=np.zeros(len(parameters)); floor[configs.index(row['config'])]=1/row['kt']
    service=floor.copy(); service[nb:]=x[0]
    # min max(floor+C, service) via one epigraph variable; max via two linear programs.
    n=len(parameters)
    minimum=linprog(np.r_[np.zeros(n),1.], A_eq=np.c_[constraints['A_eq'],np.zeros(len(constraints['b_eq']))],
        b_eq=constraints['b_eq'], A_ub=np.vstack([np.c_[constraints['A_ub'],np.zeros(len(constraints['b_ub']))],
                                               np.r_[floor,-1.],np.r_[service,-1.]]),
        b_ub=np.r_[constraints['b_ub'],-row['C'],0.],bounds=(0,None),method='highs')
    maxima=[linprog(-v,**constraints) for v in (floor,service)]
    if not minimum.success or any(z.status not in (0,3) for z in maxima): raise ValueError('probe LP failed')
    upper=None if any(z.status==3 for z in maxima) else max(-maxima[0].fun+row['C'],-maxima[1].fun)
    return [float(minimum.fun),float(upper) if upper is not None else None]


def monotone_conflicts(rows, model):
    """Certificate of insufficient features; source-only ordering omits OOB fill."""
    arrays=coverage_design(rows,model); conflicts=[]
    for i,a in enumerate(rows):
        for j,b in enumerate(rows):
            if a['config']!=b['config'] or a['kt']!=b['kt'] or a['y']<=b['y']: continue
            wa=[p[0] for p in a['parts']]; wb=[p[0] for p in b['parts']]
            if len(wa)!=len(wb) or not np.allclose(wa,wb,rtol=0,atol=1e-12): continue
            if np.all(arrays[i]<=arrays[j]+1e-10):
                conflicts.append(dict(smaller_request_case=a['case'],smaller_request_j=a['j'],smaller_request_T=a['T'],
                    larger_request_case=b['case'],larger_request_j=b['j'],larger_request_T=b['T'],
                    observed_smaller_request=a['y'],observed_larger_request=b['y'],
                    unavoidable_max_relative_error=(a['y']-b['y'])/(a['y']+b['y'])))
    return sorted(conflicts,key=lambda r:-r['unavoidable_max_relative_error'])[:3]


def dual_coverage_slice(run):
    """Address-only fixed-pressure slice; all five targets have already been revealed."""
    path=run/'reanalysis/B-20261009-frozen-negative/localization.json'
    values={int(k.rsplit('p',1)[1]):v['target_ns_per_kt'] for k,v in read_json(path)['observations'].items()}
    C=values[0]; lower=max(0,7*values[32]-6*values[16]); upper=C
    def coefficients(q0):
        q128=(values[32]-q0)/48
        return [q0,(values[16]-q0-56*q128)/32,q128]
    q0=3*values[64]-2*values[32]; q=coefficients(q0)
    return dict(unit='ns/Ktile, q32/q128 per extra covering line; fixed-pressure slice only',
        targets=values, calibration_rank=2, parameters=3,
        calibration_q0_interval=[lower,upper],
        calibration_p64_interval=[max(C,(2*values[32]+lower)/3),max(C,(2*values[32]+upper)/3)],
        revealed_p64_identified_parameters=dict(zip(['q0','q32','q128'],q)),
        p128_prediction=C,p128_error=C/values[128]-1,
        conclusion='p64 selects a previously unidentifiable coefficient split; this is post-reveal explanation, not a repaired frozen prediction.')


def coverage_suite(run, output):
    """Only two new forms: pooled or operand-specific 32B/128B costs, fixed prefill."""
    base=run.parent
    cohort_paths={
        '43269fbc_B_dual':(run,'coverage'),
        'ef8692f3_SM_stamped':(base/'20261009-R13-sm-job738101','sm'),
        '099dda56_joint_dual':(base/'20261009-R10-joint-pitch-job738169','joint'),
        '099dda56_V08F_stamped':(base/'20261008-V08F-job737322-v1','v08')}
    result={}; hashes={}; residuals=[]
    for label,(folder,kind) in cohort_paths.items():
        rows,paths,cases,setups=coverage_rows(folder,kind)
        train=[r for r in rows if r['j']!=0]; first=[r for r in rows if r['j']==0]
        result[label]=dict(gpu=read_json(folder/'environment.json').get('gpu'),kind=kind,
                          later_rows=len(train), first_rows=len(first),observations=rows,models={})
        for model in ('pooled','operand'):
            fit,p,configs,constraints=coverage_fit(train,model)
            fit['first_transfer']=metrics([coverage_predict(r,model,p,configs)/r['y']-1 for r in first])
            fit['monotonicity_failure_certificates']=monotone_conflicts(train,model)
            fit['probe_ranges']={}
            # Two analytic rank probes, not new GPU cases or an expanded experiment matrix.
            name=next(k for k,r in cases.items() if r['config']==configs[0] and r['lda']*2%128==r['ldb']*2%128==0 and r['swizzle']==1)
            if kind=='sm': name='cfg_a_sm132_aligned'
            if kind=='v08': name='cfg_a_p_ref_k4096'
            row=cases[name]
            for operand in ('A','B'):
                probe=dict(row);probe['lda' if operand=='A' else 'ldb']+=16
                _,vectors=wave_requests(probe,setups[name])
                tm,tn,_,_=CONFIGS[probe['config']]
                target=dict(config=probe['config'],kt=(probe['k']+63)//64,C=2*tm*tn*64/4096,parts=[(1.,vectors[1].tolist())])
                fit['probe_ranges'][operand+'_plus32B_j1']=dict(base_case=name,
                    cycle_per_Ktile_interval=coverage_probe_range(target,model,p,configs,constraints))
            for r in fit.pop('residuals'):
                residuals.append(dict(cohort=label,model=model,**r))
            result[label]['models'][model]=fit
        for path in paths: hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    frozen=run/'frozen/r10-b-pitch.json'
    for path in (frozen,run/'reanalysis/B-20261009-frozen-negative/localization.json'):
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(protocol='Post-reveal development diagnostics; no change to the R10 frozen failure.',
        formula='L_j=b_cfg+Kt*max(C_cfg, Qeff_j @ q); Qeff_j=(1-s/Kt)*Q_j+(s/Kt)*Q_{j+1}, s=min(stages,Kt).',
        request_fields=['logical_source_KiB','extra_A_32B_coverage_KiB','extra_A_128B_coverage_KiB','extra_B_32B_coverage_KiB','extra_B_128B_coverage_KiB'],
        models=dict(pooled='q0*D+q32*(A32+B32)+q128*(A128+B128)', operand='q0*D+qA32*A32+qA128*A128+qB32*B32+qB128*B128'),
        cohorts=result, address_only_slice=dual_coverage_slice(run),
        limits=['Separate fit per card AND observer cohort; no coefficient transfer across cards or silent pooling of stamped and dual.',
                'Q counts logical requests after multicast sharing and valid address spans, not physical L2/HBM transactions.',
                'Q omits OOB zero-fill service. Smaller valid-address demand does not imply smaller total service demand.',
                'Monotonicity conflicts identify insufficient current features, not uniquely missing pipeline state; separate A/B partial/whole OOB paths before attribution.',
                'Software waves and configured prefill credit are assumptions, not measured active TMA occupancy.',
                'All fitting targets are cycles/Ktile; direct-ns address-only slice is reported separately.',
                'Compute branch restricts service from above (capacity from below); D/measured cycles is never used as a bandwidth label.',
                'Ranges preserve fitted predictions in the selected piecewise region; they are not confidence intervals or proof of physical identifiability.',
                'First tile is diagnostic; b is outside max and does not replace startup supply S or epilogue E.'],
        frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),input_sha256=hashes,
        supply_model_sha256=hashlib.sha256(Path(__file__).with_name('supply_model.py').read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'residuals.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(residuals[0]));writer.writeheader();writer.writerows(residuals)
    print(json.dumps({c:{m:dict(rank=f['active_jacobian_rank'],nparam=f['parameter_count'],fit=f['fit']) for m,f in v['models'].items()} for c,v in result.items()},indent=2))


def supply_dual_rows(row, setup, observation):
    """Calibration observations per CTA/output; measured clocks are not features."""
    result=supply_request_rows(row,setup)
    if len(observation['ctas'])!=int(np.prod(setup['grid'])): raise ValueError('CTA count mismatch')
    counts=Counter(r['cta'] for r in result)
    for c,cta in enumerate(observation['ctas']):
        if 'tiles_ns' not in cta or any(len(cta[k])!=counts[c] for k in ('work','tiles','tiles_ns')):
            raise ValueError('complete dual tile intervals required')
    for r in result:
        cta=observation['ctas'][r['cta']];j=r['j']
        if tuple(cta['work'][j])!=tuple(r['coord']): raise ValueError('software work differs from recorded work')
        cycles,ns=cta['tiles'][j],cta['tiles_ns'][j];lc,ln=cycles[1]-cycles[0],ns[1]-ns[0]
        if lc<=0 or ln<=0: raise ValueError('positive dual L required')
        r.update(trial=observation['trial'],L_cycle=lc,L_ns=ln,calibration_floor_ns=r['compute_cycles']*ln/lc)
    return result


def read_supply_run(run):
    """Replay successful dual calls only; retain per-window data and provenance."""
    paths=[run/name for name in ('cases.json','static_setup.json','environment.json')]
    cases=read_json(paths[0]); setups={s['case']:s['setup'] for s in read_json(paths[1])}
    gpu=read_json(paths[2])['gpu'].split(',')[0]; rows=[]
    for row in cases:
        setup=setups[row['id']]
        if setup['gpu_uuid']!=gpu: raise ValueError('setup/environment card mismatch')
        records_=[]
        for path in sorted((run/'samples'/row['id']).glob('dual-*.json')):
            rec=read_json(path)
            if rec['returncode']: continue
            if rec['variant']!='dual': raise ValueError('not a dual record')
            records_.append(rec['trial']);paths += [path,run/rec['raw']]
            rows += [dict(r,run=str(run)) for r in supply_dual_rows(row,setup,replay(run,rec,row))]
        if not records_ or len(records_)!=len(set(records_)): raise ValueError('missing/duplicate dual trials: '+row['id'])
    return rows,dict(gpu_uuid=gpu,cases={r['id']:dict(row=r,setup=setups[r['id']]) for r in cases},
                     input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def supply_calibration_rows(rows):
    """Median across processes at the same CTA/j; never average boundary demands."""
    groups=defaultdict(list)
    for r in rows:groups[r['gpu_uuid'],r.get('run',''),r['case'],r['cta'],r['j']].append(r)
    result=[]
    for values in groups.values():
        first=values[0]
        if any(any(v[k]!=first[k] for k in ('config','coord','T','kt','compute_cycles','X')) for v in values):
            raise ValueError('static request changed across trials')
        result.append(dict({k:v for k,v in first.items() if k not in ('trial','L_cycle','L_ns','calibration_floor_ns')},
            trials=len(values),L_cycle=statistics.median(v['L_cycle'] for v in values),
            L_ns=statistics.median(v['L_ns'] for v in values),
            calibration_floor_ns=statistics.median(v['calibration_floor_ns'] for v in values)))
    return result


def supply_basis(rows):
    """Collapse only collinear coverage pairs; keep their combination explicit."""
    x=np.array([r['X'] for r in rows]); columns=[]; fields=[]; combinations=[]
    for indices in ((0,),(1,2),(3,4),(5,),(6,)):
        block=x[:,indices]
        if not np.any(block): continue
        if len(indices)==2 and np.linalg.matrix_rank(block)==1:
            col=np.zeros(7);col[list(indices)]=1;columns.append(col)
            fields.append(SUPPLY_FIELDS[indices[0]].split('_extra')[0]+'_coverage_combination_KiB')
            example=block[np.argmax(np.linalg.norm(block,axis=1))]
            combinations.append(dict(field=fields[-1],original_fields=[SUPPLY_FIELDS[i] for i in indices],
                fixed_fractions=(example/sum(example)).tolist(),meaning='coefficient prices the sum only at these fractions'))
        else:
            for i in indices:
                if np.any(x[:,i]):columns.append(np.eye(7)[i]);fields.append(SUPPLY_FIELDS[i])
    return np.array(columns).reshape(-1,7).T,fields,combinations


def fit_supply(rows, unit='ns', calibration_clock='observed', assumed_ghz=1.6, phase='later'):
    """One configuration/card/phase; observed f is legal only in calibration."""
    if phase not in ('later','first'):raise ValueError('phase must be later or first')
    rows=[r for r in rows if (r['j']==0)==(phase=='first')]
    if not rows: raise ValueError('no '+phase+' calibration windows')
    if len({r['config'] for r in rows})!=1 or len({r['gpu_uuid'] for r in rows})!=1:
        raise ValueError('fit one configuration and one card at a time')
    if unit not in ('cycle','ns') or calibration_clock not in ('observed','assumed') or assumed_ghz<=0:
        raise ValueError('invalid time basis')
    projection,fields,combinations=supply_basis(rows)
    x=np.array([r['X'] for r in rows]);z=x@projection;kt=np.array([r['kt'] for r in rows])
    target=np.array([r['L_'+unit] for r in rows]);floor=np.array([r['compute_cycles'] for r in rows])
    if unit=='ns':
        floor=(np.array([r.get('calibration_floor_ns',r['compute_cycles']*r['L_ns']/r['L_cycle']) for r in rows])
               if calibration_clock=='observed' else floor/assumed_ghz)
    counts=Counter((r.get('run',''),r['case']) for r in rows)
    weights=np.sqrt([1/counts[r.get('run',''),r['case']] for r in rows])
    def predict(p):return p[0]+kt*np.maximum(floor,z@p[1:])
    scale=1 if unit=='cycle' else assumed_ghz
    initial=np.r_[600,[30 if 'fill' in field else .1 if field=='valid_source_KiB' else .8 for field in fields]]/scale
    fits=[least_squares(lambda p:weights*(predict(p)/target-1),initial*factor,bounds=(0,np.inf),
        max_nfev=3000,ftol=1e-11,xtol=1e-11,gtol=1e-11) for factor in (1,2,4)]
    best=min(fits,key=lambda f:f.cost);p=best.x
    jac=np.column_stack([np.ones(len(rows)),kt[:,None]*z*(z@p[1:]>floor)[:,None]])
    norms=np.linalg.norm(jac,axis=0);js=np.where(norms>0,norms,1);_,singular,jb=np.linalg.svd(jac/js,full_matrices=False)
    rank=int(sum(singular>1e-9));norms=np.linalg.norm(x,axis=0);xs=np.where(norms>0,norms,1)
    _,s,xb=np.linalg.svd(x/xs,full_matrices=False)
    model=dict(config=rows[0]['config'],gpu_uuid=rows[0]['gpu_uuid'],unit=unit,
        phase=phase,
        calibration_clock=calibration_clock if unit=='ns' else 'cycle',assumed_ghz=assumed_ghz,
        fields=['window_'+unit]+fields,parameters=p.tolist(),projection=projection.tolist(),
        coverage_combinations=combinations,request_scale=xs.tolist(),request_basis=xb[s>1e-9].tolist(),
        jacobian_scale=js.tolist(),jacobian_basis=jb[:rank].tolist(),active_jacobian_rank=rank,
        parameter_count=len(p),conditionally_identified=bool(best.success and rank==len(p)),optimizer_success=bool(best.success),
        calibration_score=metrics(predict(p)/target-1))
    return model


def supply_score(rows, model, frequency_ghz):
    """Score with an explicit predicted/assumed clock; never replace it with targets."""
    predicted,supported=supply_predict(rows,model,frequency_ghz)
    result=dict(requested=len(rows),scored=int(sum(supported)),
                unsupported_cases=sorted({r['case'] for r,s in zip(rows,supported) if not s}))
    if np.any(supported):
        errors=predicted[supported]/np.array([r['L_'+model['unit']] for r,s in zip(rows,supported) if s])-1
        result['windows']=metrics(errors);by_case=defaultdict(list)
        for r,e in zip([r for r,s in zip(rows,supported) if s],errors):by_case[r.get('run',''),r['case']].append(e)
        result['case_medians']=metrics([statistics.median(v) for v in by_case.values()])
    return result


def supply_identification_bounds(rows, model):
    """Nonnegative parameter set preserving fitted training predictions/branches.

    These conditional LP bounds are not confidence intervals or physical prices.
    Calibration clocks are used only here; exported constraints contain no targets.
    """
    rows=[r for r in rows if (r['j']==0)==(model.get('phase','later')=='first')]
    if any(r['config']!=model['config'] or r['gpu_uuid']!=model['gpu_uuid'] for r in rows):
        raise ValueError('constraint calibration/configuration mismatch')
    z=np.array([r['X'] for r in rows])@np.array(model['projection']);p=np.array(model['parameters'])
    floor=np.array([r['compute_cycles'] for r in rows])
    if model['unit']=='ns':
        floor=(np.array([r['calibration_floor_ns'] for r in rows]) if model['calibration_clock']=='observed'
               else floor/model['assumed_ghz'])
    constraints={}
    for x,c in zip(z,floor):
        sign=-1 if x@p[1:]>c else 1
        key=tuple(np.r_[0,sign*x]);constraints[key]=min(constraints.get(key,np.inf),sign*c)
    basis=np.array(model['jacobian_basis']).reshape(-1,len(p));scale=np.array(model['jacobian_scale'])
    # Convert the scaled Jacobian's right basis back to parameter coordinates.
    # Zero inactive-column SVD noise in older entry models explicitly.
    basis[:,scale<1e-20]=0;scale=np.where(scale<1e-20,1,scale)
    eq=basis*scale
    eq=eq/np.maximum(np.max(abs(eq),axis=1,keepdims=True),1e-30)
    result=dict(A_ub=[list(k) for k in constraints],b_ub=list(constraints.values()),
                A_eq=eq.tolist(),b_eq=(eq@p).tolist())
    intervals=[]
    for i,field in enumerate(model['fields']):
        bounds=[]
        for sign in (1,-1):
            objective=np.zeros(len(p));objective[i]=sign
            fit=linprog(objective,**result,bounds=(0,None),method='highs',
                        options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            if fit.status not in (0,3):raise ValueError('conditional parameter set infeasible: '+fit.message)
            bounds.append(None if fit.status==3 else float(sign*fit.fun))
        intervals.append(dict(field=field,minimum=min(bounds[0],float(p[i])),
                              maximum=max(bounds[1],float(p[i])) if bounds[1] is not None else None))
    return dict(interpretation='Fixed fitted active pattern/predictions; nonnegative prices; null means unbounded, not a confidence interval.',
                parameters=intervals,constraints=result)


def supply_calibration(run, output, assumed_ghz):
    """Single-run development entry; no automatic pooling with older B batches."""
    raw,manifest=read_supply_run(run);rows=supply_calibration_rows(raw);fits={}
    for config in sorted({r['config'] for r in rows}):
        selected=[r for r in rows if r['config']==config];fits[config]={}
        if not any(r['j']>=1 for r in selected):
            fits[config]['status']='no later windows: first-output data only';continue
        for unit,clock in (('cycle','assumed'),('ns','observed'),('ns','assumed')):
            model=fit_supply(selected,unit,clock,assumed_ghz)
            fits[config][unit+'_'+clock]=dict(model=model,
                fixed_frequency_later=supply_score([r for r in selected if r['j']>=1],model,assumed_ghz),
                fixed_frequency_first=supply_score([r for r in selected if r['j']==0],model,assumed_ghz))
        reference=next(c for c in manifest['cases'].values() if c['row']['config']==config)
        fits[config]['pitch_support_at_reference_geometry']=dict(case=reference['row']['id'],
            interpretation='Static feature/active-branch span only; no timing qualification of unseen residues.',
            rows=supply_pitch_support(reference['row'],reference['setup'],fits[config]['ns_observed']['model'],assumed_ghz))
    report=dict(protocol='Single-run development calibration; observed clocks allowed in training only, not prediction/heldout scoring.',
        raw_windows=len(raw),calibration_windows=len(rows),fields=SUPPLY_FIELDS,assumed_ghz=assumed_ghz,
        manifest=manifest,fits=fits,
        limits=['Same max(compute, source + local-fill) form; b outside max; j0 transfer only.',
                'One shared valid A+B price per configuration/card; no cross-card parameter pooling.',
                'Coverage combinations are valid only at their recorded fractions; unsupported features are not scored.',
                'Conditional rank does not establish physical uniqueness or qualify a frozen predictor.',
                'Old B13 requires a separate bridge with allocation/preparation differences before any joint fit.'],
        supply_model_sha256=hashlib.sha256(Path(__file__).with_name('supply_model.py').read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'calibration-rows.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    print(json.dumps({c:{k:v.get('fixed_frequency_later') if isinstance(v,dict) else v for k,v in f.items() if k!='pitch_support_at_reference_geometry'} for c,f in fits.items()},indent=2))


def fill_geometry(row, setup):
    """Compatibility grouping of per-CTA descriptor and local zero-fill demands."""
    from v08_model import scheduled_work
    requests=fill_cta_features(row,setup)
    work=scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'])
    grouped=defaultdict(list)
    for coords,values in zip(work,requests):
        for j,((mi,ni),v) in enumerate(zip(coords,values)):
            cls='padM' if mi*128>=row['m'] else 'padN' if ni*128>=row['n'] else 'in'
            grouped[j,len(coords),cls].append(v)
    return {key:dict(X=np.mean(values,axis=0).tolist(),windows=len(values)) for key,values in grouped.items()}


def fill_observations(run):
    base=run.parent; rows=[]; inputs=[]; reports=[]; gpu=None
    for folder in (base/'20261009-R18-input-map-job738296',base/'20261009-R18-input-map-n-job738307'):
        path=folder/'reanalysis/input-map-pairs-v1/input-map-pairs.json'; data=read_json(path); reports.append(data)
        current=data['environment']['gpu']; gpu=current if gpu is None else gpu
        if current!=gpu: raise ValueError('fill fits must remain on one card')
        setups={s['case']:s['setup'] for s in read_json(folder/'static_setup.json')}
        cases={c['id']:c for c in read_json(folder/'cases.json')}
        inputs += [path,folder/'static_setup.json',folder/'cases.json']
        for pair in data['pairs']:
            for side,key in (('oob','control'),('address_zero','address_zero')):
                name=pair[key]; row=cases[name]; features=fill_geometry(row,setups[name])
                for g in pair['groups']:
                    feature=features[g['j'],g['T'],g['logical_class']]
                    if any(t['windows']!=feature['windows'] for t in g['processes']): raise ValueError('work-list/group mismatch')
                    rows.append(dict(case=name,kt=row['k']//64,j=g['j'],T=g['T'],cls=g['logical_class'],
                        X=feature['X'],windows=feature['windows'],
                        cycle=statistics.median(t[side+'_cycle_per_kt'] for t in g['processes']),
                        ns=statistics.median(t[side+'_ns_per_kt'] for t in g['processes'])))
    env=read_json(run/'environment.json')
    if env['gpu']!=gpu: raise ValueError('B curve and fill pairs use different cards')
    cases={c['id']:c for c in read_json(run/'cases.json')}; setups={s['case']:s['setup'] for s in read_json(run/'static_setup.json')}
    inputs += [run/'environment.json',run/'cases.json',run/'static_setup.json']
    for name,row in cases.items():
        geometry=fill_geometry(row,setups[name]); groups=defaultdict(list)
        for path in sorted((run/'samples'/name).glob('dual-*.json')):
            rec=read_json(path)
            if rec['returncode']: continue
            inputs += [path,run/rec['raw']]; process=defaultdict(list)
            for cta in replay(run,rec,row)['ctas']:
                for j,(c,t) in enumerate(zip(cta['tiles'],cta['tiles_ns'])):
                    process[j,len(cta['tiles'])].append([(c[1]-c[0])/16,(t[1]-t[0])/16])
            for key,values in process.items(): groups[key].append(np.mean(values,axis=0))
        for (j,total),values in groups.items():
            if len(values)!=10: raise ValueError('missing B-curve dual process')
            feature=geometry[j,total,'in']; target=np.median(values,axis=0)
            rows.append(dict(case=name,kt=16,j=j,T=total,cls='in',X=feature['X'],windows=feature['windows'],
                             cycle=float(target[0]),ns=float(target[1])))
    # Same dual group/process, compared across positions; these ratios are diagnostics only.
    phase=[]
    for data in reports:
        for pair in data['pairs']:
            pad='padM' if 'padM' in pair['group'] else 'padN'
            a=next(g for g in pair['groups'] if (g['j'],g['T'],g['logical_class'])==(1,6,pad))
            b=next(g for g in pair['groups'] if (g['j'],g['T'],g['logical_class'])==(4,6,pad))
            x={t['trial']:t for t in a['processes']}; y={t['trial']:t for t in b['processes']}
            samples=[dict(trial=t,cycle_ratio_j4_j1=y[t]['oob_cycle_per_kt']/x[t]['oob_cycle_per_kt'],
                ns_ratio_j4_j1=y[t]['oob_ns_per_kt']/x[t]['oob_ns_per_kt'],
                effective_cycle_per_ns_j1=x[t]['oob_cycle_per_kt']/x[t]['oob_ns_per_kt'],
                effective_cycle_per_ns_j4=y[t]['oob_cycle_per_kt']/y[t]['oob_ns_per_kt']) for t in sorted(x)]
            phase.append(dict(group=pair['group'],samples=samples,
                medians={k:statistics.median(s[k] for s in samples) for k in samples[0] if k!='trial'}))
    return rows,inputs,gpu,phase


def fill_predict(rows, parameters, unit, assumed_ghz):
    floor=512 if unit=='cycle' else 512/assumed_ghz
    return np.array([parameters[0]/r['kt']+max(floor,np.dot(r['X'],parameters[1:])) for r in rows])


def fit_fill(rows, unit, assumed_ghz):
    counts=Counter(r['case'] for r in rows); weights=np.sqrt([1/counts[r['case']] for r in rows])
    target=np.array([r[unit] for r in rows]); scale=1 if unit=='cycle' else assumed_ghz
    candidates=[]
    for q,zero in ((.08,30),(.17,60),(.3,90)):
        initial=np.array([600,q,.8,.2,zero,zero])/scale
        candidates.append(least_squares(lambda p:weights*(fill_predict(rows,p,unit,assumed_ghz)/target-1),
            initial,bounds=(0,np.inf),max_nfev=3000,ftol=1e-11,xtol=1e-11,gtol=1e-11))
    p=min(candidates,key=lambda c:c.cost).x
    x=np.array([r['X'] for r in rows]); floor=512/scale; active=x@p[1:]>floor
    jac=np.column_stack([[1/r['kt'] for r in rows],x*active[:,None]])
    norms=np.linalg.norm(jac,axis=0); singular=np.linalg.svd(jac/np.where(norms>0,norms,1),compute_uv=False)
    unused=[i for i in range(x.shape[1]) if np.all(x[:,i]==0)]
    return p,dict(active_jacobian_rank=int(sum(singular>1e-9)),parameters=6,
                  scaled_singular_values=singular.tolist(),unused_features=[FILL_FIELDS[i] for i in unused])


def fill_suite(run,output,assumed_ghz):
    rows,paths,gpu,phase=fill_observations(run)
    train=[r for r in rows if r['j']>0]; first=[r for r in rows if r['j']==0]
    fits={}; residuals=[]
    for unit in ('cycle','ns'):
        p,ident=fit_fill(train,unit,assumed_ghz); values=fill_predict(train,p,unit,assumed_ghz)
        errors=values/np.array([r[unit] for r in train])-1
        fit=dict(parameters=dict(zip(['window_'+unit]+FILL_FIELDS,p.tolist())),identification=ident,
                 later_fit=metrics(errors),first_transfer=metrics(fill_predict(first,p,unit,assumed_ghz)/np.array([r[unit] for r in first])-1))
        for subset,selected in {'R10_Bcurve':[i for i,r in enumerate(train) if 'bcurve' in r['case']],
             'whole_fill':[i for i,r in enumerate(train) if r['X'][3]+r['X'][4]>0],
             'valid_address':[i for i,r in enumerate(train) if r['X'][3]+r['X'][4]==0]}.items():
            fit[subset]=metrics(errors[selected])
        folds=[]
        for kt in (16,64):
            training=[r for r in train if r['kt']!=kt]; testing=[r for r in train if r['kt']==kt]
            q,info=fit_fill(training,unit,assumed_ghz)
            missing=[FILL_FIELDS.index(k) for k in info['unused_features']]
            supported=[r for r in testing if all(r['X'][i]==0 for i in missing)]
            fold=dict(removed_K=kt*64,identification=info,test_rows=len(testing),scored_rows=len(supported),
                      unsupported_cases=sorted({r['case'] for r in testing if r not in supported}))
            if supported: fold['score']=metrics(fill_predict(supported,q,unit,assumed_ghz)/np.array([r[unit] for r in supported])-1)
            folds.append(fold)
        fit['leave_K']=folds;fits[unit]=fit
        for r,v,e in zip(train,values,errors):residuals.append(dict(case=r['case'],j=r['j'],T=r['T'],cls=r['cls'],unit=unit,
            observed=r[unit],predicted=float(v),relative_error=float(e),fill_KiB=r['X'][3]+r['X'][4]))
    sensitivity={}
    for f in (1.6,1.8):
        if f==assumed_ghz: sensitivity[str(f)]=fits['ns']['later_fit']
        else:
            p,_=fit_fill(train,'ns',f)
            sensitivity[str(f)]=metrics(fill_predict(train,p,'ns',f)/np.array([r['ns'] for r in train])-1)
    frozen=run/'frozen/r10-b-pitch.json';paths.append(frozen)
    report=dict(protocol='Post-reveal development on GPU-43269fbc; no new heldout or physical zero-fill bandwidth claim.',
        gpu=gpu,features=FILL_FIELDS,
        formula='L=b+Kt*max(C, q_valid*Veff+q32*B32eff+q128*B128eff+qA*ZA+qB*ZB)',
        time_basis=dict(cycle='C=512 cycle/Ktile; coefficients in cycles',ns='C=512/assumed_ghz ns/Ktile; coefficients in ns'),
        assumed_ghz=assumed_ghz,ns_frequency_sensitivity=sensitivity,fits=fits,phase_diagnostics=phase,observations=rows,
        scope='cfg_b, stage6, recorded grids/strides and complete Ktiles; A pitch aligned; zero demand observed only at 0 or whole 16 KiB per operand.',
        limits=['Veff uses descriptor bounds, including valid reads of explicit zeros; ZA/ZB are separate destination fill demands.',
                'Software prefill credit is a finite-window approximation, not measured TMA occupancy.',
                'All measured cycle/ns ratios are diagnostics; predictions and leave-K tests use the explicit scalar frequency assumption only.',
                'Unused coverage coordinates are not scored in leave-K tests. Rank is conditional on the fitted piecewise model.',
                'Whole-fill calibration does not establish proportional costs for partial OOB; A stride effects are uncalibrated.',
                'The caller may supply a predicted frequency: NS-form cycles=f*(b_ns+Kt*max(512/f,service_ns)); first output remains diagnostic.',
                'Only L is replaced in CTA recursion; startup supply and epilogue remain separate. Remaining group failures are retained.'],
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),supply_model_sha256=hashlib.sha256(Path(__file__).with_name('supply_model.py').read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=False);(output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'residuals.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(residuals[0]));writer.writeheader();writer.writerows(residuals)
    print(json.dumps({u:{k:v for k,v in f.items() if k in ('later_fit','whole_fill','identification','leave_K')} for u,f in fits.items()},indent=2))


def compose_fill(run, report_path, output, assumed_ghz):
    """Use existing CTA recurrence; shared-card calibration and case-local diagnostics."""
    import v06_fit
    import v08_model
    source=read_json(report_path); base=run.parent
    folders=[run,base/'20261009-R18-input-map-job738296',base/'20261009-R18-input-map-n-job738307']
    processes=[];paths=[report_path]; raw_intervals=defaultdict(list); case_rows={}; setup_by_case={}
    for folder in folders:
        env=read_json(folder/'environment.json')
        if env['gpu']!=source['gpu']: raise ValueError('composition must use the fitted card')
        cases={c['id']:c for c in read_json(folder/'cases.json')}
        setups={s['case']:s['setup'] for s in read_json(folder/'static_setup.json')}
        paths += [folder/'cases.json',folder/'static_setup.json',folder/'environment.json']
        for name,row in cases.items():
            case_rows[name]=row;setup_by_case[name]=setups[name]
            plain_paths=sorted((folder/'samples'/name).glob('plain-*.json'));paths+=plain_paths
            plain=[read_json(p) for p in plain_paths if read_json(p)['returncode']==0]
            for path in sorted((folder/'samples'/name).glob('dual-*.json')):
                record=read_json(path)
                if record['returncode']: continue
                observed=replay(folder,record,row); ctas=observed['ctas']
                paths += [path,folder/record['raw']]
                ns_ctas=[dict(c,entry_c=c['entry_ns'],prod_c=c['prod_ns'],end_c=c['end_ns'],tiles=c['tiles_ns']) for c in ctas]
                interval={}
                for unit,seq in [('cycle',ctas),('ns',ns_ctas)]:
                    iv=v06_fit.intervals(dict(ctas=seq),'pingpong')[0]
                    raw_intervals[name,unit].append(iv);interval[unit]=iv
                envelope=max(c['end_ns'] for c in ctas)-min(c['entry_ns'] for c in ctas)
                max_local_ns=max(c['end_ns']-c['entry_ns'] for c in ctas)
                processes.append(dict(case=name,trial=record['trial'],ctas=ctas,intervals=interval,
                    dual_event_ns=observed['elapsed_us']*1000,envelope_ns=envelope,
                    event_extra_ns=observed['elapsed_us']*1000-envelope,
                    entry_skew_excess_ns=envelope-max_local_ns,
                    plain_event_ns=next(p['elapsed_us']*1000 for p in plain if p['trial']==record['trial'])))
    case_intervals={}
    for key,records_ in raw_intervals.items():
        names=set().union(*records_)
        case_intervals[key]={name:statistics.median(r[name] for r in records_ if name in r) for name in names}

    def parameters(values,unit):
        keys=['P0','S','w','we','E','h','gm','Etail','Efull']
        p={key:statistics.median(v[key] for v in values if key in v) if any(key in v for v in values) else 0. for key in keys}
        p['r']=p['E']/p['Efull'] if p['Efull']>0 else 1.
        # The aggregate baseline is calibrated on these same-card case medians.
        pairs=[(case_rows[name]['k']//64,iv['L']) for (name,u),iv in case_intervals.items() if u==unit]
        x=np.array([k for k,y in pairs]);y=np.array([y for k,y in pairs]);slope=float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2))
        p.update(l0=float(y.mean()-slope*x.mean()),l1=slope,
            dL0=statistics.median(iv['L0']-(float(y.mean()-slope*x.mean())+slope*(case_rows[name]['k']//64))
                                 for (name,u),iv in case_intervals.items() if u==unit),
            x0=0.,x1=0.,xk=0.,rho={})
        return p

    shared={}
    for unit in ('cycle','ns'):
        shared[unit]=parameters([iv for (name,u),iv in case_intervals.items() if u==unit],unit)
    shared_F=statistics.median(p['event_extra_ns'] for p in processes)
    case_F={name:statistics.median(p['event_extra_ns'] for p in processes if p['case']==name) for name in case_rows}
    features={name:fill_geometry(row,setup_by_case[name]) for name,row in case_rows.items()}
    predictions=[]
    for proc in processes:
        row=case_rows[proc['case']];kt=row['k']//64
        for parameter_scope in ('shared','case_diagnostic'):
            for unit in ('cycle','ns'):
                p=dict(shared[unit])
                if parameter_scope=='case_diagnostic':
                    iv=case_intervals[proc['case'],unit]
                    for key in ('P0','S','w','we','E','h','gm','Etail'):
                        if key in iv:p[key]=iv[key]
                    if iv.get('Efull',0)>0:p['r']=p['E']/iv['Efull']
                fit=source['fits'][unit]['parameters'];q=np.array(list(fit.values()))
                for mode in ('aggregate_baseline','valid_fill','valid_fill_measured_first','measured_L_diagnostic'):
                    ends=[];cta_errors=[];first_errors=[]
                    for cta in proc['ctas']:
                        seq=cta['tiles'] if unit=='cycle' else cta['tiles_ns'];T=len(seq)
                        actual=[t[1]-t[0] for t in seq]
                        if mode=='aggregate_baseline':ls=None
                        elif mode=='measured_L_diagnostic':ls=actual
                        else:
                            ls=[]
                            for j,(mi,ni) in enumerate(cta['work']):
                                cls='padM' if mi*128>=row['m'] else 'padN' if ni*128>=row['n'] else 'in'
                                x=features[proc['case']][j,T,cls]['X']
                                ls.append(q[0]+kt*max(512 if unit=='cycle' else 512/assumed_ghz,np.dot(x,q[1:])))
                            first_errors.append(ls[0]/actual[0]-1)
                            if mode=='valid_fill_measured_first':ls[0]=actual[0]
                        end=v08_model.cta_cycles(p,'pingpong',T,kt,mainloops=ls)
                        observed_end=(cta['end_c']-cta['entry_c']) if unit=='cycle' else (cta['end_ns']-cta['entry_ns'])
                        cta_errors.append(end/observed_end-1);ends.append(end)
                    # All predicted CTAs start together; no measured entry skew is an input.
                    env=max(ends)/(assumed_ghz if unit=='cycle' else 1.)
                    fixed=shared_F if parameter_scope=='shared' else case_F[proc['case']]
                    predictions.append(dict(case=proc['case'],trial=proc['trial'],parameter_scope=parameter_scope,
                        unit=unit,mode=mode,predicted_envelope_ns=env,observed_envelope_ns=proc['envelope_ns'],
                        envelope_error=env/proc['envelope_ns']-1,predicted_event_ns=env+fixed,
                        observed_dual_event_ns=proc['dual_event_ns'],dual_event_error=(env+fixed)/proc['dual_event_ns']-1,
                        plain_diagnostic_error=(env+fixed)/proc['plain_event_ns']-1,
                        cta_duration_median_abs_error=float(np.median(np.abs(cta_errors))),
                        first_transfer_median_abs_error=float(np.median(np.abs(first_errors))) if first_errors else None))
    scores={}
    for scope in ('shared','case_diagnostic'):
        scores[scope]={}
        for unit in ('cycle','ns'):
            scores[scope][unit]={}
            for mode in ('aggregate_baseline','valid_fill','valid_fill_measured_first','measured_L_diagnostic'):
                selected=[p for p in predictions if p['parameter_scope']==scope and p['unit']==unit and p['mode']==mode]
                by_case={}
                for name in case_rows:
                    records_=[p for p in selected if p['case']==name]
                    by_case[name]={field:statistics.median(p[field] for p in records_) for field in
                        ('predicted_envelope_ns','observed_envelope_ns','envelope_error','dual_event_error','plain_diagnostic_error','cta_duration_median_abs_error')}
                scores[scope][unit][mode]=dict(case_scores=by_case,
                    envelope=metrics([v['envelope_error'] for v in by_case.values()]),
                    dual_event=metrics([v['dual_event_error'] for v in by_case.values()]),
                    plain_diagnostic=metrics([v['plain_diagnostic_error'] for v in by_case.values()]))
    report=dict(protocol='All-data same-card development composition; no new heldout. Case-local parameters and measured L are diagnostics only.',
        gpu=source['gpu'],assumed_ghz=assumed_ghz,shared_parameters=shared,
        case_parameters={name:{u:iv for (c,u),iv in case_intervals.items() if c==name} for name in case_rows},
        fixed_event_extra_ns=shared_F,
        observed_event_extra_ns_range=[min(p['event_extra_ns'] for p in processes),max(p['event_extra_ns'] for p in processes)],
        entry_skew_excess_ns=dict(median=statistics.median(p['entry_skew_excess_ns'] for p in processes),
                                  maximum=max(p['entry_skew_excess_ns'] for p in processes)),
        scores=scores,process_predictions=predictions,
        limits=['P0/S/E/gm/w/we/h/Etail follow v06_fit.intervals; r=E_alone/E_full; x0/x1/xk are zero.',
                'Shared constants are medians of the 13 case medians. Baseline L0/L use the same-card linear-K rule.',
                'NS recurrence evaluates the same homogeneous event equations in ns, with explicit 512/f compute floor.',
                'Cycle-domain envelope conversion uses one assumed f; no per-case or per-window measured frequency is supplied.',
                'F is calibrated from each dual event minus that same call envelope, then pooled; it is not copied per prediction.',
                'Predicted entry skew is zero. Observed entry skew is diagnostic only.',
                'First-output valid-fill L is an unfitted transfer. Measured-first/measured-L controls isolate effects, not mathematical error bounds; errors can cancel.',
                'Case-local non-L calibration is a diagnostic, not prediction of those parameters for unseen cases.',
                'Missing case-local branch parameters retain the same-card shared value; no old-card constants are borrowed.',
                'Plain uses a different observer/preparation path and is scored separately, not used to fit F.'],
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        recurrence_sha256=hashlib.sha256(Path(v08_model.__file__).read_bytes()).hexdigest(),
        event_definitions_sha256=hashlib.sha256(Path(v06_fit.__file__).read_bytes()).hexdigest(),
        supply_model_sha256=hashlib.sha256(Path(__file__).with_name('supply_model.py').read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=False);(output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'process-predictions.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    print(json.dumps({s:{u:{m:v['dual_event'] for m,v in modes.items()} for u,modes in units.items()} for s,units in scores.items()},indent=2))


def _failure_pair_windows(control, variant):
    """Pair identical software work, never CTA order or physical SM identity."""
    def windows(ctas):
        result = {}
        for c in ctas:
            total = len(c['work'])
            for j, (coord, cycles, ns) in enumerate(zip(c['work'], c['tiles'], c['tiles_ns'])):
                key = (c['cta'], j, total, *coord)
                if key in result:
                    raise ValueError('duplicate CTA/work window')
                result[key] = dict(L_ns=ns[1]-ns[0], L_cycles=cycles[1]-cycles[0],
                    E_ns=ns[3]-ns[2], prefix_ns=ns[0]-c['entry_ns'] if j == 0 else None,
                    prefill_ns=ns[0]-c['prod_ns'] if j == 0 else None,
                    tail_ns=c['end_ns']-max(t[3] for t in c['tiles_ns']) if j == total-1 else None)
        return result
    a, b = windows(control), windows(variant)
    if a.keys() != b.keys():
        raise ValueError('paired cases have different CTA/j/T/work coordinates')
    result = []
    for key in sorted(a):
        item = dict(zip(('cta', 'j', 'T', 'mi', 'ni'), key))
        for field in a[key]:
            av, bv = a[key][field], b[key][field]
            if av is None:
                continue
            item[field+'_base'] = av
            item[field+'_variant'] = bv
            item[field+'_delta'] = bv-av
            if av > 0:
                item[field+'_ratio'] = bv/av-1
        fa = a[key]['L_cycles']/a[key]['L_ns']
        fb = b[key]['L_cycles']/b[key]['L_ns']
        item['L_rate_ratio'] = fb/fa-1
        result.append(item)
    return result


def _failure_pair_summary(cohort, control, variant):
    """Within-trial paired-window medians, then the median over ten trials."""
    left = {c['trial']: c for c in control}
    right = {c['trial']: c for c in variant}
    if left.keys() != right.keys() or len(left) != 10:
        raise ValueError('failure comparison requires the same ten successful trials')
    positions, phases = [], defaultdict(list)
    for trial in sorted(left):
        rows = _failure_pair_windows(left[trial]['observed']['ctas'], right[trial]['observed']['ctas'])
        for phase in ('first', 'later'):
            subset = [r for r in rows if (r['j'] == 0) == (phase == 'first')]
            if not subset:
                continue
            fields = sorted(set().union(*(r.keys() for r in subset)) - {'cta', 'j', 'T', 'mi', 'ni'})
            phases[phase].append({f: statistics.median(r[f] for r in subset if f in r) for f in fields})
        for j, total in sorted({(r['j'], r['T']) for r in rows}):
            subset = [r for r in rows if (r['j'], r['T']) == (j, total)]
            entry = dict(cohort=cohort, control=left[trial]['row']['id'], variant=right[trial]['row']['id'],
                trial=trial, j=j, T=total, windows=len(subset))
            for field in ('L_ns', 'L_cycles', 'E_ns', 'prefix_ns', 'prefill_ns', 'tail_ns'):
                for suffix in ('base', 'variant', 'delta', 'ratio'):
                    key = field+'_'+suffix
                    values = [r[key] for r in subset if key in r]
                    entry[key] = statistics.median(values) if values else None
            for key in ('L_ns_delta', 'L_cycles_delta', 'L_rate_ratio'):
                entry[key+'_p10'], entry[key+'_p90'] = map(float, np.quantile([r[key] for r in subset], [.1, .9]))
            entry['L_rate_ratio'] = statistics.median(r['L_rate_ratio'] for r in subset)
            positions.append(entry)
    phase_summary = {}
    for phase, trials in phases.items():
        fields = sorted(set().union(*(t.keys() for t in trials)))
        phase_summary[phase] = {f: dict(median=statistics.median(t[f] for t in trials if f in t),
            trial_min=min(t[f] for t in trials if f in t), trial_max=max(t[f] for t in trials if f in t)) for f in fields}
    position_summary = []
    for j, total in sorted({(r['j'], r['T']) for r in positions}):
        group = [r for r in positions if (r['j'], r['T']) == (j, total)]
        position_summary.append(dict(j=j, T=total, windows=group[0]['windows'],
            **{f: statistics.median(r[f] for r in group) for f in ('L_ns_base', 'L_ns_variant',
                'L_ns_delta', 'L_ns_ratio', 'L_cycles_ratio', 'L_rate_ratio')}))
    return dict(cohort=cohort, control=control[0]['row']['id'], variant=variant[0]['row']['id'],
        trials=len(left), phases=phase_summary, positions=position_summary), positions


def _failure_case_context(call, calibration):
    row, setup = call['row'], call['setup']
    requests = static_tma_requests(row, setup)
    groups = defaultdict(list)
    for cta, tiles in enumerate(requests['ctas']):
        for j, tile in enumerate(tiles):
            groups[j, len(tiles)].append(dict(cta=cta, coord=tile['coord'],
                preceding_coords=[t['coord'] for t in tiles[:j]],
                source=tile['A']['source_KiB']+tile['B']['source_KiB']))
    prices = defaultdict(list)
    for r in supply_request_rows(row, setup):
        model = calibration['first_supply' if r['j'] == 0 else 'supply'][row['config']]
        service = float(np.array(r['X']) @ np.array(model['projection']) @ np.array(model['parameters'][1:]))
        prices[r['j'],r['T']].append((r['X'],service))
    return dict(config=row['config'], m=row['m'], n=row['n'], k=row['k'], stages=setup['stages'],
        gpu=setup['gpu_uuid'], grid=setup['grid'],
        pitch_bytes=dict(A=2*row.get('lda', row['k']), B=2*row.get('ldb', row['n'])),
        allocated_elements={k: setup.get(k) for k in ('allocated_a_elements', 'allocated_b_elements')},
        logical_input_bytes=2*row['k']*(row['m']+row['n']),
        source_waves=requests['source_waves'], effective_source_waves=requests['effective_source_waves'],
        frozen_service_prices=[dict(j=j,T=t,windows=len(v),
            X_range=[dict(min=min(x[0][i] for x in v),max=max(x[0][i] for x in v)) for i in range(7)],
            service_ns_per_ktile=dict(min=min(x[1] for x in v),max=max(x[1] for x in v)))
            for (j,t),v in sorted(prices.items())],
        positions=[dict(j=j, T=t, windows=len(v), distinct_current_coords=len({tuple(x['coord']) for x in v}))
            for (j,t),v in sorted(groups.items())],
        work_sha256=hashlib.sha256(json.dumps([[t['coord'] for t in tiles] for tiles in requests['ctas']],
            separators=(',', ':')).encode()).hexdigest())


def _failure_stage_response(control, variant):
    """Show the exact max branches that erase the only implemented stage effect."""
    calibration = control['calibration']
    frequency = control['prediction']['frequency_ghz']
    records = {}
    for name, call in (('stage6', control), ('stage4', variant)):
        groups = defaultdict(list)
        for r in supply_request_rows(call['row'], call['setup']):
            model = calibration['first_supply' if r['j'] == 0 else 'supply'][r['config']]
            service = float(np.array(r['X']) @ np.array(model['projection']) @ np.array(model['parameters'][1:]))
            floor = r['compute_cycles']/frequency
            groups[r['j'], r['T']].append(dict(source_KiB=r['X'][0], service_ns=service,
                floor_ns=floor, supply_active=service > floor,
                L_ns=model['parameters'][0]+r['kt']*max(service, floor)))
        records[name] = dict(stages=call['setup']['stages'],
            smem=call['setup']['smem'], max_active_ctas_per_sm=call['setup']['max_active_ctas_per_sm'],
            positions=[dict(j=j,T=t,windows=len(v),supply_active_windows=sum(x['supply_active'] for x in v),
                **{k:dict(min=min(x[k] for x in v),max=max(x[k] for x in v))
                   for k in ('source_KiB','service_ns','floor_ns','L_ns')}) for (j,t),v in sorted(groups.items())])
    records['common_frequency_ghz'] = frequency
    records['frozen_outputs_equal'] = {key: control['prediction'][key] == variant['prediction'][key]
        for key in ('frequency_ghz','window_us','plain_us','cta_duration_ns','cta_events')}
    records['interpretation'] = ('Both 132-CTA software waves are equal, so first demand is stage-independent. '
        'Only last demand changes; both fitted supply prices remain below the compute floor. '
        'All L predictions, event timelines, clock closure and plain predictions are therefore equal. '
        'This model has no producer/consumer slot state or stage-dependent compute overlap.')
    return records


def failure_diagnostics(base, output, calls_factory):
    """V09 paired failure analysis; old cohorts stay separate and no fitting occurs."""
    from analyze_r13_sm import select_attempts
    import re

    base, output = Path(base), Path(output)
    root = base/'20261009-V09-heldout-job739011'
    new = defaultdict(list)
    needed = {f'v09d_cfg_{c}_G3_{v}' for c in 'abc' for v in ('base', 'A16')}
    needed |= {f'v09d_cfg_b_G5_{v}' for v in ('base', 'B32', 'B80', 'B112')}
    needed.add('v09d_cfg_b_G3_stage4')
    for call in calls_factory(base):
        if call['row']['id'] in needed:
            new[call['row']['id']].append(call)
    pairs = [('V09', new[f'v09d_cfg_{c}_G3_base'], new[f'v09d_cfg_{c}_G3_A16']) for c in 'abc']
    pairs += [('V09', new['v09d_cfg_b_G5_base'], new[f'v09d_cfg_b_G5_{v}']) for v in ('B32','B80','B112')]
    pairs += [('V09', new['v09d_cfg_b_G3_base'], new['v09d_cfg_b_G3_stage4'])]
    hashes, old = {}, {}
    for name in ('20261009-R10-b-coverage-job738203', '20261009-R13-R15-R18-composition-job738496'):
        run = base/name
        setups = {r['case']:r['setup'] for r in read_json(run/'static_setup.json')}
        for filename in ('cases.json', 'static_setup.json', 'environment.json'):
            p = run/filename; hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        cohort = 'R10-Bcurve' if '738203' in name else 'R13-calibration'
        old[cohort] = defaultdict(list)
        for row in read_json(run/'cases.json'):
            if cohort == 'R13-calibration' and ('r13_comp' not in row['id'] or row['k'] != 4096):
                continue
            selected, _, _ = select_attempts(run, row['id'], ('dual',))
            for item in selected:
                record = item['record']
                metadata = run/item['path']
                hashes[str(metadata)] = hashlib.sha256(metadata.read_bytes()).hexdigest()
                raw = run/record['raw']; digest = hashlib.sha256(raw.read_bytes()).hexdigest()
                if digest != record['raw_sha256']:
                    raise ValueError('raw sample hash changed: '+str(raw))
                hashes[str(raw)] = digest
                old[cohort][row['id']].append(dict(row=row, setup=setups[row['id']], trial=record['trial'],
                    record=record, observed=replay(run, record, row)))
        if cohort == 'R10-Bcurve':
            pairs += [(cohort, old[cohort]['cfg_b_bcurve_p0'], old[cohort][f'cfg_b_bcurve_p{v}']) for v in (16,32,64,128)]
        else:
            pairs += [(cohort, old[cohort][f'cfg_{c}_r13_comp_k4096_aligned'],
                       old[cohort][f'cfg_{c}_r13_comp_k4096_{v}']) for c in 'abc' for v in ('a16','b16')]
    comparisons, positions, contexts = [], [], {}
    for cohort, control, variant in pairs:
        summary, rows = _failure_pair_summary(cohort, control, variant)
        comparisons.append(summary); positions.extend(rows)
        for calls in (control, variant):
            call = calls[0]
            contexts[call['row']['id']] = _failure_case_context(call, new['v09d_cfg_b_G3_base'][0]['calibration'])
    stage = _failure_stage_response(new['v09d_cfg_b_G3_base'][0], new['v09d_cfg_b_G3_stage4'][0])
    resources = read_json(root/'build/resources.json')['facts']
    hashes[str(root/'build/resources.json')] = hashlib.sha256((root/'build/resources.json').read_bytes()).hexdigest()
    stage['resources'] = {}
    opcodes = {}
    for name in ('cfg_b_plain','cfg_b_s4_plain','cfg_b_dual','cfg_b_s4_dual'):
        path = root/'build'/f'{name}.sass'
        text = path.read_text()
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        functions = text.split('Function : ')
        gemm = [f for f in functions[1:] if f.startswith('_ZN7cutlass13device_kernel')]
        if len(gemm) != 1:
            raise ValueError('expected one GEMM SASS entry')
        instructions = re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Za-z0-9_.]*)(?=\s|;)', gemm[0])
        opcodes[name] = Counter(x.split('.')[0] for x in instructions)
        if opcodes[name]['HGMMA'] != resources[name]['hgmma']:
            raise ValueError('GEMM SASS instruction extraction missed HGMMA')
        stage['resources'][name] = dict(**resources[name], instruction_count=len(instructions),
            non_nop_instruction_count=len(instructions)-opcodes[name]['NOP'],
            opcode_counts=dict(sorted(opcodes[name].items())), sass_sha256=hashes[str(path)])
    stage['opcode_changes'] = {variant:{op:dict(stage6=opcodes[f'cfg_b_{variant}'][op],stage4=opcodes[f'cfg_b_s4_{variant}'][op])
        for op in sorted(opcodes[f'cfg_b_{variant}'].keys() | opcodes[f'cfg_b_s4_{variant}'].keys())
        if opcodes[f'cfg_b_{variant}'][op] != opcodes[f'cfg_b_s4_{variant}'][op]} for variant in ('plain','dual')}
    report = dict(protocol='Post-V09 development diagnosis only; same trial and CTA/j/T/coordinate within each cohort. '
        'Window ratios/differences are formed before within-trial medians; ten trial medians are then summarized. '
        'Trial indices pair repetitions, not simultaneous calls. Old and new geometry cohorts are not pooled.',
        comparisons=comparisons, case_context=contexts, stage_response=stage,
        next_measurement=dict(stage='Measure only cfg_b 2816x1536, K1024/4096 x stages4/6 in one allocation. '
            'A fixed per-output startup/drain difference predicts nearly constant delta-L; a sustained overlap limit predicts '
            'delta-L increasing with K. Their per-Ktile differences consequently decay or persist. '
            'This distinguishes window models, not physical queue depth.',
            pitch='Do not add B64/B96 by default: old B32/B64 already reject universal 32-byte no-penalty. '
            'Geometry, footprint and prior source reuse still co-vary; a new pair needs a specific competing prediction.'),
        limits=['No fitting, input-condition changes or new heldout score.',
            'Dual local windows can be perturbed even when full-call timing changes little.',
            'Software source waves are not observed concurrent TMA requests or physical memory traffic.',
            'Same registers/occupancy/HGMMA count do not prove identical instruction scheduling.',
            'A whole L window cannot separate TMA wait, WGMMA execution and slot release.'],
        additional_input_sha256=hashes,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True, exist_ok=True)
    (output/'B-summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'B-paired-windows.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(positions[0]))
        writer.writeheader(); writer.writerows(positions)
    lines = ['# 行距与 stage：成对主循环分析', '',
        '这是一轮V09测后开发分析，没有重拟合或重评旧冻结。每批内部按同trial、CTA/j/T/坐标配对，'
        '先算窗口差及比值，再取进程内中位和十进程中位；trial只是重复编号，不是同时调用。旧新几何不合并。', '',
        '| 批次 | 变化条件 | first L ns | later L ns | first L cycle | later L cycle |',
        '|---|---|---:|---:|---:|---:|']
    for pair in comparisons:
        vals = [pair['phases'][phase][field]['median']*100 if phase in pair['phases'] else None
            for field in ('L_ns_ratio','L_cycles_ratio') for phase in ('first','later')]
        lines.append('| '+pair['cohort']+' | '+pair['variant']+' | '+' | '.join(f'{v:+.2f}%' if v is not None else '—' for v in vals)+' |')
    lines += ['', '同 trial 的逐 CTA 比值先取中位，再跨十 trial 汇总；完整位置结果见 B-paired-windows.csv。', '',
        'G3 stage4/6 首轮软件需求相同，末轮供给变化仍落在计算下界之下，故全部预测窗口与完整时间相同。'
        '实测主循环 cycle/ns 都变慢，尤其 later；只修频率、首段或输出不能解释。两个版本驻留均为1 CTA/SM，'
        '寄存器及HGMMA条数相同，但SASS不是同一程序，不把差异直接命名为硬件队列容量。', '',
        '旧B32/B64与新B32的响应相反在主循环直接ns中仍成立；新A16的主循环惩罚更大。'
        '当前/下一软件波次、几何、K与源重访条件均保存在 B-summary.json；旧新批次不合并拟合。', '',
        '下一项最小补测为同一几何 K1024/4096 × stage4/6 四点。固定每输出tile成本预期ΔL近常数、ΔL/K下降；'
        '持续流水限制预期ΔL随K增长、ΔL/K保留。保持输入、容量、grid与观察协议一致，复测已有K4096对。'
        'B64/B96暂不追加，等行距候选具有不同可检验预期后再选。', '']
    (output/'B-analysis.md').write_text('\n'.join(lines))
    return report


def _failure_k_scaling(low_positions, high_positions, low_kt, high_kt):
    """Compare fixed-per-output and constant-per-Ktile stage penalties."""
    low = {(r['trial'],r['j'],r['T']):r for r in low_positions}
    high = {(r['trial'],r['j'],r['T']):r for r in high_positions}
    if low.keys() != high.keys():
        raise ValueError('K controls must preserve trial/j/T groups')
    trials = []
    for (trial,j,total), a in sorted(low.items()):
        b = high[trial,j,total]
        row = dict(trial=trial,j=j,T=total)
        for unit in ('ns','cycles'):
            lo, hi = a[f'L_{unit}_delta'], b[f'L_{unit}_delta']
            row.update({f'delta_low_{unit}':lo, f'delta_high_{unit}':hi,
                f'delta_low_{unit}_per_ktile':lo/low_kt,
                f'delta_high_{unit}_per_ktile':hi/high_kt,
                f'fixed_only_residual_{unit}':hi-lo,
                f'per_ktile_only_residual_{unit}':hi/high_kt-lo/low_kt})
        trials.append(row)
    groups = []
    for j,total in sorted({(r['j'],r['T']) for r in trials}):
        selected = [r for r in trials if (r['j'],r['T']) == (j,total)]
        fields = [k for k in selected[0] if k not in ('trial','j','T')]
        summary = {f:dict(median=statistics.median(r[f] for r in selected),
            trial_min=min(r[f] for r in selected),trial_max=max(r[f] for r in selected)) for f in fields}
        groups.append(dict(j=j,T=total,windows=low_positions[0]['windows'],statistics=summary))
    return dict(low_kt=low_kt,high_kt=high_kt,positions=groups,trials=trials,
        interpretation='A fixed per-output difference has zero high-minus-low residual; a constant per-Ktile '
        'difference has zero normalized residual. Trial ranges describe these ten repetitions, not confidence '
        'intervals. K also changes active source footprint; neither pattern uniquely identifies a hardware queue.')


def control_diagnostics(root, output):
    """Analyze the new-card four-condition stage control without old-card constants."""
    from analyze_r13_sm import select_attempts

    root, output = Path(root), Path(output)
    rows = {r['id']:r for r in read_json(root/'cases.json')}
    setups = {r['case']:r['setup'] for r in read_json(root/'static_setup.json')}
    calls, times, hashes, failures = {}, {}, {}, {}
    identities = []
    for k in (1024,4096):
        for stages in (4,6):
            name = f'r13_stage_k{k}_s{stages}'
            row, setup = rows[name], setups[name]
            selected, failed, _ = select_attempts(root,name,('plain','dual'))
            failures[name] = [dict(path=f['path'],trial=f['record']['trial'],
                variant=f['record']['variant'],returncode=f['record']['returncode']) for f in failed]
            calls[name], times[name] = [], defaultdict(dict)
            for item in selected:
                record = item['record'];raw = root/record['raw'];metadata = root/item['path']
                digest = hashlib.sha256(raw.read_bytes()).hexdigest()
                if digest != record['raw_sha256']:
                    raise ValueError('control raw sample hash changed: '+str(raw))
                hashes[str(raw)] = digest
                hashes[str(metadata)] = hashlib.sha256(metadata.read_bytes()).hexdigest()
                observed = replay(root,record,row)
                times[name][record['variant']][record['trial']] = observed['elapsed_us']
                if record['variant'] == 'dual':
                    calls[name].append(dict(row=row,setup=setup,trial=record['trial'],record=record,observed=observed))
            for variant in ('plain','dual'):
                if len(times[name][variant]) != 10:
                    raise ValueError(name+' requires ten successful '+variant+' processes')
            identities.append(dict(case=name,config=row['config'],m=row['m'],n=row['n'],k=k,stages=stages,
                **{key:setup[key] for key in ('gpu_uuid','grid','lda','ldb','alloc_lda','alloc_ldb',
                    'allocated_a_elements','allocated_b_elements','smem','max_active_ctas_per_sm')},
                active_source_bytes=2*k*(row['m']+row['n'])))
    gpu_ids = {i['gpu_uuid'] for i in identities}
    if len(gpu_ids) != 1:
        raise ValueError('four-condition stage comparison must use one GPU')
    def span(values):
        return dict(median=statistics.median(values),trial_min=min(values),trial_max=max(values))
    comparisons, positions, timing = [], {}, []
    for k in (1024,4096):
        control, variant = f'r13_stage_k{k}_s6', f'r13_stage_k{k}_s4'
        summary, positions[k] = _failure_pair_summary('same-job-stage-controls',calls[control],calls[variant])
        summary['k'] = k
        comparisons.append(summary)
        record = dict(k=k)
        for protocol in ('plain','dual'):
            a,b = times[control][protocol],times[variant][protocol]
            if a.keys() != b.keys():
                raise ValueError('stage timing trials differ')
            record[protocol] = dict(stage6_us=span(list(a.values())),stage4_us=span(list(b.values())),
                stage4_minus_stage6_us=span([b[t]-a[t] for t in sorted(a)]),
                stage4_relative_change=span([b[t]/a[t]-1 for t in sorted(a)]))
        for stages in (4,6):
            t = times[f'r13_stage_k{k}_s{stages}'];a,b=t['plain'],t['dual']
            if a.keys() != b.keys():
                raise ValueError('plain/dual timing trials differ')
            record[f'stage{stages}_dual_relative_to_plain'] = span([b[i]/a[i]-1 for i in sorted(a)])
        timing.append(record)
    scaling = _failure_k_scaling(positions[1024],positions[4096],16,64)
    for filename in ('cases.json','static_setup.json','environment.json'):
        path = root/filename;hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(protocol='New-card same-job stage4/stage6 controls only. No old-card model parameter is used. '
        'Local windows use matched trial/CTA/j/T/coordinates and paired-window medians; complete plain and dual '
        'times remain separate. Same trial indices do not imply simultaneous invocation.',
        gpu=next(iter(gpu_ids)),conditions=identities,comparisons=comparisons,
        timing=timing,scaling=scaling,failed_attempts=failures,
        limits=['Full-call observer changes do not bound every local window perturbation.',
            'Allocations are fixed by design, but K changes the active source footprint.',
            'Static stage4/stage6 instruction organization differs; a persistent per-Ktile penalty is not uniquely a slot limit.',
            'This measures a stage response on the new card; it does not transfer old absolute calibration.'],
        input_sha256=hashes,analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    (output/'B-controls.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def _finite_stage_cycles(kt, stages, issue_interval, ready_latency, consumer_interval, detail=False):
    """Two pingpong outputs, one producer, wait1 releases and persistent SMEM slots.

    These are stationary effective times, not a claim about physical TMA queues.
    A batch models one complete Ktile; consumer_interval includes issue/compute
    organization. FIRST_MMA follows the first full wait, so initial ready latency
    is outside L. The next math WG starts at mma return, before mma_tail completes.
    """
    if kt < 2 or stages < 2 or min(issue_interval, ready_latency, consumer_interval) < 0:
        raise ValueError('finite-stage diagnostic requires Ktile>=2, stages>=2 and nonnegative times')
    issued, ready, starts, done, available, retired = [], [], [], [], [], []
    released = [0.]*(2*kt)
    next_consumer = 0.
    for i in range(2*kt):
        issue = max(issued[-1]+issue_interval if issued else 0., released[i-stages] if i >= stages else 0.)
        issued.append(issue)
        ready.append(issue+ready_latency)
        available.append(next_consumer)
        commit = max(ready[-1],next_consumer)
        starts.append(commit)
        done.append(max(commit,done[-1] if done else 0.)+consumer_interval)
        if i % kt == 0:
            next_consumer = commit  # prologue commit has no wait1/release
        else:
            next_consumer = max(commit,done[i-1])
            released[i-1] = next_consumer  # wait1 has completed the previous group
        if i % kt == kt-1:
            released[i] = done[i]  # mma_tail wait0 releases the last stage
        retired.append(next_consumer)
    windows = [done[kt-1]-starts[0],done[-1]-starts[kt]]
    if detail:
        return dict(windows=windows,issued=issued,ready=ready,first_mma_or_commit=starts,
            completed=done,released=released,consumer_available=available,wait1_return=retired,
            commits=starts,slot=[i % stages for i in range(2*kt)])
    return windows


def _commit_paced_stage_cycles(kt, stages, issue_interval, ready_latency, residual_tail, detail=False, loop_advance=0.):
    """Three-parameter alternative: 512-cycle commit throughput, residual tail only.

    Producer issue and software slot lifetimes are unchanged. The 512-cycle floor
    comes from cfg_b's existing arithmetic demand, not the measured stage penalty.
    The default has no loop-bookkeeping cost; the diagnostic wrapper adds one
    shared loop parameter without changing slot or inter-output handoff semantics.
    """
    issued,ready,available,starts,commits,done,retired=[],[],[],[],[],[],[]
    released=[0.]*(2*kt);next_consumer=0.
    for i in range(2*kt):
        issue=max(issued[-1]+issue_interval if issued else 0.,released[i-stages] if i>=stages else 0.)
        issued.append(issue);ready.append(issue+ready_latency);available.append(next_consumer)
        start=max(ready[-1],next_consumer);starts.append(start)
        commit=max(start,commits[-1]+512 if commits else start)
        commits.append(commit);done.append(commit+residual_tail)
        if i%kt==0:next_consumer=commit
        else:
            next_consumer=max(commit,done[i-1]);released[i-1]=next_consumer
        if i%kt==kt-1:released[i]=done[i]
        retired.append(next_consumer)
        if i%kt!=kt-1:next_consumer+=loop_advance
    windows=[done[kt-1]-starts[0],done[-1]-starts[kt]]
    if detail:
        return dict(windows=windows,issued=issued,ready=ready,consumer_available=available,
            first_mma_or_commit=starts,commits=commits,completed=done,released=released,
            wait1_return=retired,slot=[i%stages for i in range(2*kt)])
    return windows


def _loop_paced_stage_cycles(kt, stages, issue_interval, ready_latency, residual_tail, loop_advance, detail=False):
    """One shared loop-progress parameter, excluding the inter-output handoff."""
    return _commit_paced_stage_cycles(kt,stages,issue_interval,ready_latency,residual_tail,
        detail=detail,loop_advance=loop_advance)


def _discount_stage_cycles(kt, stages, first_offset, first_price, later_offset, later_price):
    """Refit only the old stage/K demand-discount form on this card."""
    return [first_offset+kt*max(512.,first_price),
            later_offset+kt*max(512.,(1-stages/kt)*later_price)]


def _fit_stage_candidate(rows, kind):
    """Small deterministic multistart fit; equal optima expose missing information."""
    if kind == 'finite_buffer':
        evaluate = _finite_stage_cycles
        starts = [[p,latency,c] for p in (10.,300.,600.) for latency in (500.,1500.,2500.,4000.) for c in (512.,600.)]
        lower,upper = [0.,0.,512.],[2000.,20000.,1500.]
    elif kind == 'commit_paced':
        evaluate = _commit_paced_stage_cycles
        starts = [[p,latency,tail] for p in (10.,300.,600.) for latency in (500.,1500.,2500.,4000.) for tail in (64.,256.,512.,1024.)]
        lower,upper = [0.,0.,0.],[2000.,20000.,2000.]
    elif kind == 'loop_paced':
        evaluate = _loop_paced_stage_cycles
        starts = [[p,latency,tail,h] for p in (10.,500.) for latency in (500.,1500.,4000.)
            for tail in (256.,512.,1024.) for h in (0.,100.,300.)]
        lower,upper = [0.,0.,0.,0.],[2000.,20000.,2000.,1024.]
    else:
        evaluate = _discount_stage_cycles
        starts = [[b,c,b,c] for b in (0.,500.,2000.) for c in (512.,600.,900.)]
        lower,upper = [0.,0.,0.,0.],[20000.,5000.,20000.,5000.]
    def residual(parameters):
        return np.array([evaluate(r['kt'],r['stages'],*parameters)[r['j']]/r['cycles']-1 for r in rows])
    fitted = [least_squares(residual,start,bounds=(lower,upper),diff_step=1e-4,
        ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=400) for start in starts]
    best = min(fitted,key=lambda r:float(r.fun @ r.fun))
    loss = float(best.fun @ best.fun)
    equivalent = [r.x.tolist() for r in fitted if float(r.fun @ r.fun) <= loss+max(1e-10,loss*1e-7)]
    # Column scaling avoids interpreting the units of p/latency/c as rank loss.
    scaled_jacobian = best.jac*np.maximum(1.,np.abs(best.x))
    singular = np.linalg.svd(scaled_jacobian,compute_uv=False)
    rank = int(sum(singular > max(singular[0],1e-12)*1e-6))
    return dict(kind=kind,parameters=best.x.tolist(),sum_squared_relative_error=loss,
        calibration_metrics=metrics(best.fun),calibration_cases=[r['case']+':'+str(r['j']) for r in rows],
        local_jacobian_rank=rank,parameter_count=len(best.x),singular_values=singular.tolist(),
        equal_loss_multistart_parameters=equivalent,
        numerical_search_bounds=dict(lower=lower,upper=upper),
        limits='Equal-loss multistart solutions are examples, not exhaustive parameter bounds or confidence intervals.')


def finite_stage_diagnostics(base, output):
    """Test finite slots against old demand discount, without changing prediction code."""
    base,output = Path(base),Path(output)
    run = base/'20261010-R09-R13-R18-failure-controls'
    source = base/'20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/B-controls.json'
    controls = read_json(source)
    targets = []
    for pair in controls['comparisons']:
        for stages,suffix in ((6,'base'),(4,'variant')):
            for j,phase in enumerate(('first','later')):
                d = pair['phases'][phase]
                targets.append(dict(case=f'r13_stage_k{pair["k"]}_s{stages}',kt=pair['k']//64,
                    stages=stages,j=j,cycles=d['L_cycles_'+suffix]['median'],
                    trial_min=d['L_cycles_'+suffix]['trial_min'],trial_max=d['L_cycles_'+suffix]['trial_max']))
    results = {}
    for kind,evaluate in (('demand_discount',_discount_stage_cycles),('finite_buffer',_finite_stage_cycles)):
        fitted = _fit_stage_candidate(targets,kind)
        fitted['rows'] = [dict(**r,predicted_cycles=evaluate(r['kt'],r['stages'],*fitted['parameters'])[r['j']],
            relative_error=evaluate(r['kt'],r['stages'],*fitted['parameters'])[r['j']]/r['cycles']-1) for r in targets]
        folds = []
        for axis,values in (('kt',(16,64)),('stages',(4,6))):
            for held in values:
                train = [r for r in targets if r[axis] != held]
                test = [r for r in targets if r[axis] == held]
                fit = _fit_stage_candidate(train,kind)
                scored = []
                for row in test:
                    values_ = [evaluate(row['kt'],row['stages'],*p)[row['j']]
                        for p in fit['equal_loss_multistart_parameters']]
                    value = evaluate(row['kt'],row['stages'],*fit['parameters'])[row['j']]
                    scored.append(dict(**row,predicted_cycles=value,relative_error=value/row['cycles']-1,
                        equal_loss_observed_prediction_range=[min(values_),max(values_)]))
                folds.append(dict(held_axis=axis,held_value=held,fit=fit,rows=scored,
                    development_metrics=metrics([r['relative_error'] for r in scored])))
        fitted['leave_group_development'] = folds
        results[kind] = fitted
    headers = [run/'source/overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp',
        run/'source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp',
        run/'source/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp']
    report = dict(scope='Post-measurement development on GPU-ef8692f3 only. Eight whole-L cycle targets; '
        'no stage-specific constants, free K interpolation, old-card parameters or new validation score.',
        gpu=controls['gpu'],targets=targets,models=results,
        formulas=dict(issue='I_i=max(I_(i-1)+p,U_(i-S))',ready='A_i=I_i+latency',
            commit='Q_i=max(A_i,H_i)',complete='D_i=max(Q_i,D_(i-1))+c',
            release='After commit i, wait1 gives U_(i-1)=H_(i+1)=max(Q_i,D_(i-1)); prologue has no release. '
            'At each output tail, U_i=D_i after wait0. Next math WG starts at mma return, before the previous tail completes.',
            observable='L_j=D_(last j)-Q_(first j), excluding initial full-barrier wait.'),
        fitted_parameter_meanings=dict(p='effective producer issue interval, cycle',latency='effective issue-to-ready latency, cycle',
            c='effective consumer/Tensor interval including instruction organization, constrained >=512 cycles'),
        source_semantics=['The slot is owned from successful producer_acquire until its consumer_release.',
            'producer_acquire waits for empty and arms full transaction bytes; full becomes ready after TMA A/B complete_tx.',
            'consumer_wait observes full completion; wait_group<1> precedes release of the previous group slot.',
            'mma_tail uses wait_group<0>; producer and pipeline state continue over output boundaries.',
            'Software stages are SMEM buffers; they are not physical TMA queue depth.'],
        old_R13_limits='Old ready, retire and reuse pairs were measured in separate calls on GPU-54896349 with '
            'different consumers/source layout. Reuse begins after acquire succeeds and excludes empty wait. '
            'They constrain event semantics but cannot calibrate same-call latency/issue/release parameters on this card.',
        minimal_event_pair=dict(cases='cfg_b 2816x1536x4096, stages4/6, unchanged input and capacity protocol',
            selection='CTA0, second output, Ktile16; release of this slot occurs after committing Ktile17; refill is Ktile16+S.',
            producer_events=['before empty acquire','after acquire succeeds','after both A/B TMA issues',
                'same-slot refill acquire/issue'],
            consumer_events=['before full wait','after full wait','after commit','after wait1',
                'consumer_release for the selected slot'],
            clock='same-call same-SM clock64, plus selected-CTA FIRST_MMA/MAIN_END',
            contrasting_expectations=['Input starvation: additional stage4 mainloop time appears in consumer full-barrier waits.',
                'Consumer instruction/retirement limit: full is already ready; extra time lies in commit/wait1/release-side intervals.'],
            interpretation='Producer empty wait alone cannot separate the two, because a slow consumer also blocks producer reuse. '
                'Full-wait return is an observed availability bound, not an exact hardware completion timestamp.',
            protocol='Two conditions, ten processes each of plain and sparse events. Existing plain already '
                'allocates and clears the same scratch, so a separate matched-scratch variant is redundant. '
                'Record only selected events and defer trace stores; static resources and perturbation remain separate.'),
        limits=['This minimal stationary queue omits explicit producer instruction scheduling, data-dependent service and cache state.',
            'FIRST_MMA hides initial ready latency; active-branch fits may identify only combinations of p and latency.',
            'The two K values change effective source footprint; a fitted c is not an isolated WGMMA instruction service.',
            'Whole-L cycle targets do not distinguish memory waits, math issue organization and slot-release stalls.',
            'Leave-group numbers are development extrapolation after all four conditions are known, not a new heldout pass.'],
        input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [source,*headers]},
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    (output/'B-finite-stage.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines = ['# 有限stage递推能否解释四点对照？', '',
        '结论：三参数有限缓冲递推略改善同批拟合，但留K/留stage迁移不稳定，不能替换现行模型。'
        '本分析只使用GPU-ef8692f3上的八个first/later周期目标；没有stage4常数、自由K插值或跨卡参数。', '',
        '## 生命周期与最小模型', '',
        'slot从producer_acquire成功到consumer_release期间不可覆盖。acquire等待empty并设置full事务字节，'
        'A/B TMA的complete_tx使full可用；consumer_wait返回后发当前WGMMA，wait_group<1>后释放前一组slot，'
        '尾部wait_group<0>释放最后slot。pingpong的下一math WG在mma返回后、mma_tail前取得执行次序；'
        'producer和SMEM状态跨输出继续，不能每个输出重新预填。软件stage是SMEM缓冲数量，不是物理队列深度。', '',
        '`I_i=max(I_(i-1)+p,U_(i-S)); A_i=I_i+ℓ; Q_i=max(A_i,H_i); D_i=max(Q_i,D_(i-1))+c`。'
        '提交i后，wait1给出`U_(i-1)=H_(i+1)=max(Q_i,D_(i-1))`；prologue不释放，tail取`U_i=D_i`。'
        '观察量`L_j=D_last−Q_first`排除首次full等待。p/ℓ/c是stationary有效间隔；c≥512包含消费侧指令组织，'
        '不是对物理发起率、TMA延迟和Tensor服务分别完成了识别。', '',
        '## 同批拟合与留组开发预测', '',
        '| 形式 | 参数数 | 拟合中位/最大误差 | 局部Jacobian秩 |', '|---|---:|---:|---:|']
    for kind,fit in results.items():
        met=fit['calibration_metrics']
        lines.append(f'| {kind} | {fit["parameter_count"]} | {met["median_abs_pct"]:.2f}% / {met["max_abs_pct"]:.2f}% | {fit["local_jacobian_rank"]} |')
    lines += ['', '| 留出组（开发用） | 需求折扣最大误差 | 有限缓冲最大误差 |', '|---|---:|---:|']
    for a,b in zip(results['demand_discount']['leave_group_development'],results['finite_buffer']['leave_group_development']):
        label=f'K={a["held_value"]*64}' if a['held_axis']=='kt' else f'stage={a["held_value"]}'
        lines.append(f'| {label} | {a["development_metrics"]["max_abs_pct"]:.2f}% | {b["development_metrics"]["max_abs_pct"]:.2f}% |')
    lines += ['', '以上是多起点最小二乘得到的开发结果，不是新的留出通过或全局最优证明。参数局部满秩不等于机制识别；'
        '留组会丢失活跃约束，等损失解和预测示例范围在JSON中保留，不把多起点范围称为完整可辨识区间。'
        '第一输出的初始ready延迟不在L内，消费/访存与编译指令组织又共同作用，单靠八个整窗不能分别定值。', '',
        '## 还缺什么观测', '',
        '旧R13的ready、retire、reuse来自不同调用、另一张卡和不同源布局。reuse起点在acquire成功之后，'
        '没有记录empty等待，不能拼成当前程序的同次槽生命周期。这里只采用其语义证据。', '',
        '最小事件对照保持cfg_b的2816×1536×4096与stage4/6，仅记录CTA0第二输出的Ktile16：producer的acquire前/后及A/B发出后；'
        'consumer的full-wait前/后、commit后、wait1后、下一Ktile释放该slot的时刻；再记录i+S的同slot补发。'
        '全部采用同调用clock64，保留该CTA的FIRST_MMA/MAIN_END。输入饥饿预期差异主要落在full等待；'
        '若full已ready而差在commit/wait1/release侧，则支持消费/指令组织限制。单独测producer empty等待不能区分两者。'
        '两条件各用plain、匹配scratch无事件、稀疏事件十进程；是否执行由管理者决定，本分析没有启动GPU。', '']
    (output/'B-finite-stage.md').write_text('\n'.join(lines))
    return report


def _stage_event_intervals(sparse):
    """One selected position; drain is mma_tail, never post-output kernel exit."""
    intervals = dict(sparse['intervals'])
    closure = (intervals['mainloop_before_tail']+intervals['handoff_before_tail']+
               intervals['mma_tail']-intervals['L'])
    if closure:
        raise ValueError('FIRST_MMA -> mma return -> tail begin -> MAIN_END does not close')
    clocks = sparse['clocks']
    intervals['issue_to_observed_full'] = clocks['full_wait_end']-clocks['issue_end']
    intervals['slot_live_until_release_begin'] = clocks['release_begin']-clocks['acquire_end']
    intervals['refill_after_release_begin'] = clocks['refill_end']-clocks['release_begin']
    intervals['refill_after_release_end'] = clocks['refill_end']-clocks['release_end']
    intervals['release_begin_after_refill_wait_begin'] = clocks['release_begin']-clocks['refill_begin']
    intervals['wait1_to_next_release_begin'] = clocks['release_begin']-clocks['wait1_end']
    intervals['wait1_to_next_release_end'] = clocks['release_end']-clocks['wait1_end']
    intervals['consumer_wait_to_commit'] = clocks['commit_end']-clocks['full_wait_begin']
    intervals['consumer_wait_to_wait1'] = clocks['wait1_end']-clocks['full_wait_begin']
    intervals['before_selected_wait'] = clocks['full_wait_begin']-clocks['first_mma']
    intervals['through_selected_commit'] = clocks['commit_end']-clocks['first_mma']
    intervals['through_selected_wait1'] = clocks['wait1_end']-clocks['first_mma']
    intervals['through_selected_release'] = clocks['release_end']-clocks['first_mma']
    intervals['after_selected_wait1_to_mma_return'] = clocks['mma_return']-clocks['wait1_end']
    if (intervals['before_selected_wait']+intervals['consumer_wait_to_wait1']+
            intervals['after_selected_wait1_to_mma_return'] != intervals['mainloop_before_tail']):
        raise ValueError('selected-position cumulative intervals do not close')
    intervals['closure_cycles'] = closure
    return intervals


def stage_event_diagnostics(root, output):
    """Same-call sparse stage pair, with separate whole-L dual cohort context."""
    from analyze_r13_sm import select_attempts

    root,output = Path(root),Path(output)
    rows = {r['id']:r for r in read_json(root/'cases.json')}
    setups = {r['case']:r['setup'] for r in read_json(root/'static_setup.json')}
    processes,by_stage,timing,hashes,failures = [],{},[],{},{}
    def span(values):
        return dict(median=statistics.median(values),trial_min=min(values),trial_max=max(values))
    for stages in (4,6):
        name = f'r13_stage_k4096_s{stages}';row = rows[name]
        selected,failed,_ = select_attempts(root,name,('plain','stamped'))
        failures[name] = [dict(path=f['path'],variant=f['record']['variant'],trial=f['record']['trial'],
            returncode=f['record']['returncode']) for f in failed]
        times = defaultdict(dict);by_stage[stages] = {}
        for item in selected:
            record = item['record'];observed = replay(root,record,row)
            for path in (root/item['path'],root/record['raw']):
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            times[record['variant']][record['trial']] = observed['elapsed_us']
            if record['variant']=='stamped':
                sparse = observed['sparse']
                intervals = _stage_event_intervals(sparse)
                entry = dict(case=name,stages=stages,trial=record['trial'],elapsed_us=observed['elapsed_us'],
                    cta=sparse['cta'],sm=sparse['sm'],mi=sparse['work'][0],ni=sparse['work'][1],
                    output=sparse['output'],ktile=sparse['ktile'],slot=sparse['slot'],phase=sparse['phase'],**intervals)
                processes.append(entry);by_stage[stages][record['trial']] = entry
        if len(times['plain'])!=10 or times['plain'].keys()!=times['stamped'].keys():
            raise ValueError('sparse comparison requires ten matched plain/stamped trials')
        fields = [k for k in processes[-1] if k not in
            ('case','stages','trial','elapsed_us','cta','sm','mi','ni','output','ktile','slot','phase')]
        timing.append(dict(stages=stages,plain_us=span(list(times['plain'].values())),
            stamped_us=span(list(times['stamped'].values())),
            stamped_relative_to_plain=span([times['stamped'][t]/times['plain'][t]-1 for t in sorted(times['plain'])]),
            intervals={f:span([r[f] for r in by_stage[stages].values()]) for f in fields},
            plain_trials=times['plain'],stamped_trials=times['stamped']))
    if by_stage[4].keys()!=by_stage[6].keys():raise ValueError('stage repetition sets differ')
    paired = []
    for trial in sorted(by_stage[4]):
        a,b = by_stage[6][trial],by_stage[4][trial]
        if tuple(a[k] for k in ('cta','mi','ni','output','ktile')) != tuple(b[k] for k in ('cta','mi','ni','output','ktile')):
            raise ValueError('stage pair does not refer to the same logical CTA/Ktile')
        paired.append(dict(trial=trial,**{f:b[f]-a[f] for f in fields}))
    paired_summary = {f:span([r[f] for r in paired]) for f in fields}
    whole_time = {}
    for protocol in ('plain','stamped'):
        a,b=timing[1][protocol+'_trials'],timing[0][protocol+'_trials']
        whole_time[protocol] = dict(stage4_minus_stage6_us=span([b[t]-a[t] for t in sorted(a)]),
            stage4_relative_change=span([b[t]/a[t]-1 for t in sorted(a)]))
    old = root.parent/'20261010-R09-R13-R18-failure-controls'
    old_rows = {r['id']:r for r in read_json(old/'cases.json')}
    old_setups = {r['case']:r['setup'] for r in read_json(old/'static_setup.json')}
    cohort_comparison = []
    for stages in (4,6):
        name=f'r13_stage_k4096_s{stages}'
        if old_setups[name]['gpu_uuid']!=setups[name]['gpu_uuid']:
            raise ValueError('declared same-card whole-L context is on a different GPU')
        selected,_,_=select_attempts(old,name,('dual',));values=[]
        for item in selected:
            record=item['record'];observed=replay(old,record,old_rows[name])
            cta=next(c for c in observed['ctas'] if c['cta']==0)
            expected=by_stage[stages][record['trial']]
            if tuple(cta['work'][1])!=(expected['mi'],expected['ni']):
                raise ValueError('old dual and sparse software work differ')
            tile=cta['tiles'][1]
            values.append(dict(trial=record['trial'],L_cycles=tile[1]-tile[0],event_us=observed['elapsed_us']))
            for path in (old/item['path'],old/record['raw']):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        current=next(t for t in timing if t['stages']==stages)['intervals']['L']
        old_span=span([x['L_cycles'] for x in values])
        cohort_comparison.append(dict(stages=stages,old_dual_CTA0_second_L=old_span,
            sparse_CTA0_second_L=current,ratio_of_cohort_medians=current['median']/old_span['median']-1,
            old_processes=values,
            interpretation='Different jobs and observer programs; compare whole L only, never splice their event windows.'))
    source_files = ['environment.json','static_setup.json','cases.json','source_hashes.json',
        'build/resources.json','build/sass_hashes.json','source/probes/r18_trace.hpp','source/probes/r18.cu',
        'source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp',
        'source/overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp']
    for relative in source_files:
        path=root/relative;hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (root/'build').glob('*.sass'):
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(scope='CTA0 second output Ktile16 only, same-call SM clock64 endpoints. '
        'No per-K extrapolation or physical queue-depth attribution. Pair differences precede trial summary.',
        gpu=setups['r13_stage_k4096_s4']['gpu_uuid'],processes=processes,timing=timing,
        paired_stage4_minus_stage6=paired,paired_summary=paired_summary,
        whole_time_response=whole_time,old_dual_cohort=cohort_comparison,
        resources=read_json(root/'build/resources.json'),sass_hashes=read_json(root/'build/sass_hashes.json'),
        failed_attempts=failures,input_sha256=hashes,
        interval_limits=['consumer_full_wait includes try_wait plus consumer_wait and timestamp cost.',
            'consumer_issue_to_commit includes fences, WGMMA issue/commit and timestamp cost.',
            'consumer_wait1 is after commit through wait1 return; release for Ktile16 is measured in Ktile17.',
            'issue_to_observed_full is an observed availability interval, not bare TMA latency.',
            'mma_tail is wait0/final release; it is distinct from post-epilogue kernel exit.',
            'mainloop_before_tail + handoff_before_tail + mma_tail equals L for each call, before aggregation.',
            'Plain already allocates and clears identical scratch; it does not match sparse instructions or registers.',
            'Full-call perturbation does not bound every selected local interval; never multiply Ktile16 by all Ktiles.'],
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    (output/'B-stage-events.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'B-stage-events.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(processes[0]));writer.writeheader();writer.writerows(processes)
    lines=['# stage稀疏事件：额外时间在主循环，单点还不能解释全部差异','',
        '仅统计CTA0第二输出的Ktile16；绝对值为十个进程中位，差值先按trial配对再取中位。'
        '每个调用的mainloop_before_tail + handoff_before_tail + mma_tail严格等于L，各列中位不必相加闭合。','',
        '| 周期窗口 | stage4中位 | stage6中位 | stage4−stage6中位 | 差值十次范围 |',
        '|---|---:|---:|---:|---:|']
    labels={'producer_empty_wait':'producer empty等待','producer_issue':'A/B TMA发出',
        'refill_empty_wait':'同slot补发前empty等待','consumer_full_wait':'consumer full等待',
        'consumer_issue_to_commit':'full返回→commit','consumer_wait1':'commit→wait1返回',
        'consumer_wait_to_commit':'full等待前→commit','consumer_wait_to_wait1':'full等待前→wait1返回',
        'wait1_to_next_release_begin':'Ktile16 wait1返回→目标slot release前（Ktile17）',
        'wait1_to_next_release_end':'Ktile16 wait1返回→目标slot release后（Ktile17）',
        'release_begin_after_refill_wait_begin':'补发开始等待→consumer release前',
        'refill_after_release_end':'consumer release后→补发acquire返回',
        'before_selected_wait':'FIRST_MMA→Ktile16 full等待前',
        'through_selected_commit':'FIRST_MMA→Ktile16 commit',
        'after_selected_wait1_to_mma_return':'Ktile16 wait1返回→mma返回',
        'consumer_release':'目标slot release','mainloop_before_tail':'FIRST_MMA→mma返回',
        'handoff_before_tail':'mma返回→tail开始','mma_tail':'mma_tail','L':'完整L'}
    for field,label in labels.items():
        a,b=timing[0]['intervals'][field],timing[1]['intervals'][field];d=paired_summary[field]
        lines.append(f'| {label} | {a["median"]:.1f} | {b["median"]:.1f} | {d["median"]:+.1f} | {d["trial_min"]:+.1f}～{d["trial_max"]:+.1f} |')
    lines+=['','最终drain没有解释stage4增加的L。所选Ktile的full等待变长，但full返回到commit变短，'
        '应先检查同调用的合并窗口，不能把某个中位等待差乘64解释全部L。wait1返回间隔也不等于完整WGMMA延迟。',
        '','| 协议比较 | 中位变化 | 十次范围 |','|---|---:|---:|']
    for t in timing:
        d=t['stamped_relative_to_plain']
        lines.append(f'| stage{t["stages"]} sparse/plain | {100*d["median"]:+.2f}% | {100*d["trial_min"]:+.2f}%～{100*d["trial_max"]:+.2f}% |')
    for protocol,d in whole_time.items():
        x=d['stage4_relative_change']
        lines.append(f'| {protocol} stage4/stage6 | {100*x["median"]:+.2f}% | {100*x["trial_min"]:+.2f}%～{100*x["trial_max"]:+.2f}% |')
    lines+=['','plain分配与清零相同scratch，但不匹配sparse指令和寄存器生存期。完整时间扰动小不证明局部窗口无扰动。'
        '与739095旧dual只按相同CTA、相同输出比较整L；两批窗口不拼接，事件价格不直接接入模型。',
        '','JSON保存逐进程值、静态资源/SASS及归档源码身份；CSV保留全部选中窗口，失败编译记录仍在原run的attempts目录。','']
    (output/'B-stage-events.md').write_text('\n'.join(lines))
    return report


def _stage_candidate_local_events(evaluate, parameters, stages):
    """Expose what each simplified recurrence actually predicts at the sampled nodes."""
    trace=evaluate(64,stages,*parameters,detail=True);i=80
    I,A,H,B,C,D,U,W=(trace[k] for k in ('issued','ready','consumer_available','first_mma_or_commit',
        'commits','completed','released','wait1_return'))
    values=dict(consumer_full_wait=B[i]-H[i],consumer_issue_to_commit=C[i]-B[i],
        consumer_wait1=W[i]-C[i],consumer_wait_to_wait1=W[i]-H[i],
        wait1_to_next_release_begin=U[i]-W[i],
        refill_empty_wait=I[i+stages]-(I[i+stages-1]+parameters[0]),
        refill_after_release_begin=I[i+stages]-U[i],
        mainloop_before_tail=W[-1]-B[64],mma_tail=D[-1]-W[-1],L=trace['windows'][1])
    return dict(intervals=values,nodes_relative_to_FIRST_MMA=[dict(global_ktile=k,
        issue=I[k]-B[64],ready=A[k]-B[64],consumer_available=H[k]-B[64],
        full_return=B[k]-B[64],commit=C[k]-B[64],complete=D[k]-B[64],release=U[k]-B[64])
        for k in sorted({i-1,i,i+1,i+stages})])


def stage_event_model_diagnostics(base, output):
    """Whole-window fits versus independently reported sparse local constraints."""
    base,output=Path(base),Path(output)
    fit_path=output/'B-finite-stage.json';event_path=output/'B-stage-events.json'
    old=read_json(fit_path);events=read_json(event_path)
    if old['gpu']!=events['gpu']:raise ValueError('local model comparison must use the same GPU')
    revised=_fit_stage_candidate(old['targets'],'commit_paced')
    revised['rows']=[dict(**r,predicted_cycles=_commit_paced_stage_cycles(r['kt'],r['stages'],*revised['parameters'])[r['j']],
        relative_error=_commit_paced_stage_cycles(r['kt'],r['stages'],*revised['parameters'])[r['j']]/r['cycles']-1)
        for r in old['targets']]
    differences=[]
    for n,name in enumerate(('producer_interval','ready_latency','residual_tail')):
        p=revised['parameters'][:];p[n]+=1.
        differences.append(dict(parameter=name,max_whole_L_change_for_plus_one_cycle=max(abs(
            _commit_paced_stage_cycles(r['kt'],r['stages'],*p)[r['j']]-
            _commit_paced_stage_cycles(r['kt'],r['stages'],*revised['parameters'])[r['j']]) for r in old['targets'])))
    revised['local_sensitivity']=differences
    comparisons={}
    for name,evaluate,parameters in [('post_commit_compute',_finite_stage_cycles,old['models']['finite_buffer']['parameters']),
                                    ('commit_paced',_commit_paced_stage_cycles,revised['parameters'])]:
        cases=[]
        for s in (4,6):
            pred=_stage_candidate_local_events(evaluate,parameters,s)
            actual=next(t['intervals'] for t in events['timing'] if t['stages']==s)
            cases.append(dict(stages=s,predicted=pred,
                observed={key:actual[key] for key in pred['intervals']},
                median_residual_cycles={key:v-actual[key]['median'] for key,v in pred['intervals'].items()}))
        comparisons[name]=cases
    c6=next(t for t in events['timing'] if t['stages']==6)['intervals']['consumer_wait_to_wait1']
    waiting={s:[p for p in events['processes'] if p['stages']==s] for s in (4,6)}
    release_bounds={s:dict(already_waiting_before_release=sum(p['release_begin_after_refill_wait_begin']>0 for p in rows),
        processes=len(rows),return_after_release_call_min=min(p['refill_after_release_end'] for p in rows),
        return_after_release_call_max=max(p['refill_after_release_begin'] for p in rows)) for s,rows in waiting.items()}
    report=dict(scope='Post-measurement candidate diagnosis. Parameters use only 739095 dual whole-L windows; '
        '739556 sparse intervals remain a separate comparison, with no pooled likelihood or new validation score.',
        gpu=events['gpu'],old_fit_parameters=old['models']['finite_buffer']['parameters'],revised_fit=revised,
        revised_formula='I_i=max(I_(i-1)+p,U_(i-S)); A_i=I_i+latency; B_i=max(A_i,H_i); '
            'C_i=max(B_i,C_(i-1)+512); D_i=C_i+tau. Wait1/release uses max(C_i,D_(i-1)); same slot lifecycle.',
        local_comparisons=comparisons,
        structural_constraint=dict(model='commit_paced with zero loop-bookkeeping time',
            derivation='For ordinary iterations H_i=W_(i-1), W_i=max(C_i,C_(i-1)+tau). '
                'Both arguments advance at least 512, so W_i-H_i>=512 for every parameter choice.',
            observed_stage6_range=[c6['trial_min'],c6['trial_max']],
            minimum_missing_interval_under_this_assumption=512-c6['trial_max'],
            implication='No adjustment of the three parameters can place this combined interval inside the raw observed range. '
                'Either loop/release bookkeeping before the next full-wait must be represented, or the arithmetic '
                'throughput bound must constrain a service/average issue process rather than every commit spacing. '
                'This is a limitation of the simplified event mapping, not a measured hardware 512-cycle issue rule.'),
        release_to_acquire_bounds=release_bounds,
        interpretation=['Ktile16 combined equality does not mean the next iteration advances equally: the same-call '
            'wait1(K16)-to-release(K16, in K17) difference is reported independently.',
            'Old recurrence treats a whole Ktile as starting compute after instantaneous commit; observed issue-to-commit '
            'is nonzero and overlaps progress. Its fitted stage6 wait1 is therefore not a physical service estimate.',
            'The three-parameter revision moves throughput earlier but whole-L fitting drives residual tail into a '
            'different compensation; sparse commit/wait1/combined windows still reject that event representation.',
            'Release-call to producer-acquire return includes polling, scheduling, expect_tx and observer overhead. '
            'It is not a bare visibility latency; do not set a hardware constant from the roughly 200-cycle interval.'],
        next_minimal_observation=dict(conditions='Same cfg_b stage4/6 pair; no new geometry or stage penalty.',
            consumer_progress_commits=[1,16,17,63],
            additional_endpoints=['FIRST_MMA','wait1 return at Ktile16','full-wait begin at Ktile17',
                'mma return','mma_tail begin','MAIN_END'],
            purpose='Adjacent commits separate per-step pacing from bursts; wait1(K16) to full-wait(K17) measures '
                'the missing loop/slot-release progression; final commit separates residual completion from tail bookkeeping. '
                'Ktile1 retains early cumulative progress. Drop producer trace now that slot/phase semantics are confirmed.',
            status='Design only; no GPU or probe changes in this analysis.'),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (fit_path,event_path)},
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    (output/'B-stage-local-constraints.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 相邻Ktile约束与commit推进候选','',
        '同调用Ktile16 combined相同，不能推出Ktile17进度相同。完整进程与配对结果仍见B-stage-events.json；'
        '本页仅将739095 whole-L定值候选与739556局部观测分开比较，不拼接时间线。','',
        '| 窗口 | 实测stage4 / stage6中位 | 原递推预测 | commit推进预测 |','|---|---:|---:|---:|']
    for key in ('consumer_full_wait','consumer_issue_to_commit','consumer_wait1','consumer_wait_to_wait1',
                'wait1_to_next_release_begin','refill_after_release_begin','mma_tail','L'):
        measured=[next(t['intervals'][key]['median'] for t in events['timing'] if t['stages']==s) for s in (4,6)]
        original=[c['predicted']['intervals'][key] for c in comparisons['post_commit_compute']]
        changed=[c['predicted']['intervals'][key] for c in comparisons['commit_paced']]
        fmt=lambda a:' / '.join(f'{x:.2f}' for x in a)
        lines.append(f'| {key} | {fmt(measured)} | {fmt(original)} | {fmt(changed)} |')
    met=revised['calibration_metrics']
    lines+=['',f'修订仍只用三个共享参数p/ℓ/τ，commit最小间隔固定512。whole-L开发拟合中位/最大误差'
        f'为{met["median_abs_pct"]:.2f}%/{met["max_abs_pct"]:.2f}%，局部参数秩{revised["local_jacobian_rank"]}。'
        '拟合whole-L没有使局部关系成立。', '',
        '更强的限制不依赖最佳参数：该零循环开销递推满足H_i=W_(i−1)，而W_i=max(C_i,C_(i−1)+τ)。'
        '两个自变量都至少前进512，故full等待前→wait1返回至少512 cycle。stage6实测仅399～407。'
        '不能靠调这三个参数修复；在每commit≥512假设下至少缺少105 cycle的下一次full-wait前推进，'
        '或该吞吐下界只能限制服务/平均发射而不限制每一步commit。现有端点尚不能区分这两种情况。', '',
        'producer在两stage的全部十次refill中都先于release开始等待，release调用结束到acquire返回仍有正间隔。'
        '这个界限含轮询、软件调度、expect_tx和取时开销，不能当作约200 cycle的硬件裸可见性常数。', '',
        '若继续，最小观测改为同一对条件的consumer commit后Ktile1/16/17/63，加wait1(K16)和full-wait开始(K17)，'
        '保留FIRST_MMA与mma/tail端点。相邻commit直接检验单步吞吐；两个相邻循环边界给出遗漏推进；'
        '最后commit分开残余完成与tail前软件时间。删除producer字段，不扩展几何。本轮只给设计，不申请GPU。','']
    (output/'B-stage-local-constraints.md').write_text('\n'.join(lines))
    return report


def _stage_progress_intervals(progress):
    """Direct differences within one call, including non-overlapping L partitions."""
    c=progress['clocks']
    pairs=[('commit16_to17','commit16','commit17'),
        ('loop_advance_16_to17','wait1_end16','full_wait_begin17'),
        ('full_wait_to_commit16','full_wait_begin16','commit16'),
        ('full_wait_to_commit17','full_wait_begin17','commit17'),
        ('commit_to_wait1_16','commit16','wait1_end16'),
        ('first_to_commit1','first_mma','commit1'),
        ('first_to_commit16','first_mma','commit16'),
        ('first_to_commit17','first_mma','commit17'),
        ('first_to_commit63','first_mma','commit63'),
        ('commit1_to16','commit1','commit16'),
        ('commit17_to63','commit17','commit63'),
        ('last_commit_to_mma_return','commit63','mma_return'),
        ('mma_return_to_main_end','mma_return','main_end'),
        ('last_commit_to_main_end','commit63','main_end'),
        ('L','first_mma','main_end')]
    values={name:c[end]-c[start] for name,start,end in pairs}
    for name,value in progress.get('intervals',{}).items():
        if values[name]!=value:raise ValueError('progress decoded interval disagrees with clocks: '+name)
    values['average_commit1_to16']=values['commit1_to16']/15
    values['average_commit17_to63']=values['commit17_to63']/46
    values['full_wait_to_wait1_16']=c['wait1_end16']-c['full_wait_begin16']
    values['adjacent_closure_cycles']=(values['commit_to_wait1_16']+values['loop_advance_16_to17']+
        values['full_wait_to_commit17']-values['commit16_to17'])
    values['L_closure_cycles']=sum(values[k] for k in ('first_to_commit1','commit1_to16',
        'commit16_to17','commit17_to63','last_commit_to_mma_return','mma_return_to_main_end'))-values['L']
    if abs(values['adjacent_closure_cycles'])>1e-7 or abs(values['L_closure_cycles'])>1e-7:
        raise ValueError('progress same-call partitions do not close')
    return values


def _stage_candidate_progress(evaluate, parameters, stages):
    """Map the existing recurrence to progress endpoints, without refitting."""
    t=evaluate(64,stages,*parameters,detail=True)
    C,H,W,B,D=(t[k] for k in ('commits','consumer_available','wait1_return',
        'first_mma_or_commit','completed'))
    clocks=dict(first_mma=B[64],commit1=C[65],full_wait_begin16=H[80],commit16=C[80],
        wait1_end16=W[80],full_wait_begin17=H[81],commit17=C[81],commit63=C[127],
        mma_return=W[127],main_end=D[127])
    return _stage_progress_intervals(dict(clocks=clocks))


def stage_progress_diagnostics(root, output):
    """Consumer progress on the fixed stage pair; no instantaneous throughput gate."""
    from analyze_r13_sm import select_attempts

    root,output=Path(root),Path(output)
    rows={r['id']:r for r in read_json(root/'cases.json')}
    setups={r['case']:r['setup'] for r in read_json(root/'static_setup.json')}
    processes,by_stage,timing,hashes,failures=[],{},[],{},{}
    def span(values):
        return dict(median=statistics.median(values),trial_min=min(values),trial_max=max(values))
    for stages in (4,6):
        name=f'r13_stage_k4096_s{stages}';row=rows[name]
        selected,failed,_=select_attempts(root,name,('plain','stamped'))
        failures[name]=[dict(path=f['path'],variant=f['record']['variant'],trial=f['record']['trial'],
            returncode=f['record']['returncode']) for f in failed]
        times=defaultdict(dict);by_stage[stages]={}
        for item in selected:
            record=item['record'];observed=replay(root,record,row)
            for path in (root/item['path'],root/record['raw']):
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            times[record['variant']][record['trial']]=observed['elapsed_us']
            if record['variant']=='stamped':
                progress=observed['progress'];intervals=_stage_progress_intervals(progress)
                entry=dict(case=name,stages=stages,trial=record['trial'],elapsed_us=observed['elapsed_us'],
                    cta=progress['cta'],sm=progress['sm'],mi=progress['work'][0],ni=progress['work'][1],
                    output=progress['output'],**intervals)
                processes.append(entry);by_stage[stages][record['trial']]=entry
        if len(times['plain'])!=10 or times['plain'].keys()!=times['stamped'].keys():
            raise ValueError('progress comparison requires ten matched plain/stamped trials')
        fields=list(intervals)
        timing.append(dict(stages=stages,plain_us=span(list(times['plain'].values())),
            stamped_us=span(list(times['stamped'].values())),
            stamped_relative_to_plain=span([times['stamped'][t]/times['plain'][t]-1 for t in sorted(times['plain'])]),
            intervals={f:span([r[f] for r in by_stage[stages].values()]) for f in fields},
            sub512_adjacent_commits=sum(r['commit16_to17']<512 for r in by_stage[stages].values()),
            plain_trials=times['plain'],stamped_trials=times['stamped']))
    if by_stage[4].keys()!=by_stage[6].keys():raise ValueError('stage repetition sets differ')
    paired=[]
    for trial in sorted(by_stage[4]):
        a,b=by_stage[6][trial],by_stage[4][trial]
        if tuple(a[k] for k in ('cta','mi','ni','output'))!=tuple(b[k] for k in ('cta','mi','ni','output')):
            raise ValueError('stage progress pair refers to different logical work')
        paired.append(dict(trial=trial,**{f:b[f]-a[f] for f in fields}))
    paired_summary={f:span([r[f] for r in paired]) for f in fields}
    whole_time={}
    for protocol in ('plain','stamped'):
        a,b=timing[1][protocol+'_trials'],timing[0][protocol+'_trials']
        whole_time[protocol]=dict(stage4_minus_stage6_us=span([b[t]-a[t] for t in sorted(a)]),
            stage4_relative_change=span([b[t]/a[t]-1 for t in sorted(a)]))

    # Compare observer programs as separate jobs, never concatenate their clocks.
    old_root=root.parent/'20261010-R13-stage-events'
    old_rows={r['id']:r for r in read_json(old_root/'cases.json')}
    old_setups={r['case']:r['setup'] for r in read_json(old_root/'static_setup.json')}
    old_cohort=[]
    common_fields=('L','first_to_commit16','full_wait_to_commit16','commit_to_wait1_16',
        'full_wait_to_wait1_16','mma_return_to_main_end')
    for stages in (4,6):
        name=f'r13_stage_k4096_s{stages}'
        if old_setups[name]['gpu_uuid']!=setups[name]['gpu_uuid']:
            raise ValueError('sparse/progress observer comparison needs the same GPU')
        selected,_,_=select_attempts(old_root,name,('plain','stamped'))
        times=defaultdict(dict);old_processes=[]
        for item in selected:
            record=item['record'];observed=replay(old_root,record,old_rows[name])
            times[record['variant']][record['trial']]=observed['elapsed_us']
            for path in (old_root/item['path'],old_root/record['raw']):
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            if record['variant']=='stamped':
                sparse=observed['sparse'];c=sparse['clocks'];current=by_stage[stages][record['trial']]
                if (sparse['cta'],*sparse['work'],sparse['output'])!=tuple(current[k] for k in ('cta','mi','ni','output')):
                    raise ValueError('sparse/progress observer comparison changed selected work')
                old_processes.append(dict(trial=record['trial'],L=c['main_end']-c['first_mma'],
                    first_to_commit16=c['commit_end']-c['first_mma'],
                    full_wait_to_commit16=c['commit_end']-c['full_wait_begin'],
                    commit_to_wait1_16=c['wait1_end']-c['commit_end'],
                    full_wait_to_wait1_16=c['wait1_end']-c['full_wait_begin'],
                    mma_return_to_main_end=c['main_end']-c['mma_return']))
        if len(old_processes)!=10 or times['plain'].keys()!=times['stamped'].keys():
            raise ValueError('sparse observer cohort is incomplete')
        old_span={f:span([p[f] for p in old_processes]) for f in common_fields}
        current=next(t for t in timing if t['stages']==stages)
        old_cohort.append(dict(stages=stages,source=str(old_root),processes=old_processes,intervals=old_span,
            progress_minus_sparse_cohort_medians={f:current['intervals'][f]['median']-old_span[f]['median'] for f in common_fields},
            sparse_stamped_relative_to_plain=span([times['stamped'][t]/times['plain'][t]-1 for t in sorted(times['plain'])]),
            progress_stamped_relative_to_plain=current['stamped_relative_to_plain'],
            plain_us=span(list(times['plain'].values())),stamped_us=span(list(times['stamped'].values())),
            interpretation='Separate observer cohorts; only identical endpoints are compared. No paired cross-job timeline.'))

    # Reuse parameters fitted to 739095 whole L. Progress is a development check,
    # not another set of fit targets and not independent frozen validation.
    fit_path=output/'B-finite-stage.json';local_path=output/'B-stage-local-constraints.json'
    fit,local=read_json(fit_path),read_json(local_path)
    gpu=setups['r13_stage_k4096_s4']['gpu_uuid']
    if fit['gpu']!=gpu or local['gpu']!=gpu:raise ValueError('candidate parameters came from another GPU')
    loop_fit=_fit_stage_candidate(fit['targets'],'loop_paced')
    loop_fit['rows']=[dict(**r,predicted_cycles=_loop_paced_stage_cycles(r['kt'],r['stages'],*loop_fit['parameters'])[r['j']],
        relative_error=_loop_paced_stage_cycles(r['kt'],r['stages'],*loop_fit['parameters'])[r['j']]/r['cycles']-1)
        for r in fit['targets']]
    loop_fit['local_sensitivity']=[]
    for i,name in enumerate(('producer_interval','ready_latency','residual_tail','loop_advance')):
        changed=loop_fit['parameters'][:];changed[i]+=1
        loop_fit['local_sensitivity'].append(dict(parameter=name,max_whole_L_change_for_plus_one_cycle=max(abs(
            _loop_paced_stage_cycles(r['kt'],r['stages'],*changed)[r['j']]-
            _loop_paced_stage_cycles(r['kt'],r['stages'],*loop_fit['parameters'])[r['j']]) for r in fit['targets'])))
    loop_fit['equal_loss_parameter_example_ranges']=[dict(parameter=name,
        minimum=min(p[i] for p in loop_fit['equal_loss_multistart_parameters']),
        maximum=max(p[i] for p in loop_fit['equal_loss_multistart_parameters']))
        for i,name in enumerate(('producer_interval','ready_latency','residual_tail','loop_advance'))]
    comparisons={}
    for name,evaluate,parameters in [
        ('post_commit_compute',_finite_stage_cycles,fit['models']['finite_buffer']['parameters']),
        ('commit_paced',_commit_paced_stage_cycles,local['revised_fit']['parameters']),
        ('loop_paced',_loop_paced_stage_cycles,loop_fit['parameters'])]:
        comparisons[name]=[]
        for stages in (4,6):
            predicted=_stage_candidate_progress(evaluate,parameters,stages)
            observed=next(t['intervals'] for t in timing if t['stages']==stages)
            comparisons[name].append(dict(stages=stages,parameters=parameters,predicted=predicted,
                observed={k:observed[k] for k in predicted},
                median_residual_cycles={k:v-observed[k]['median'] for k,v in predicted.items()}))
    for relative in ('environment.json','static_setup.json','cases.json','source_hashes.json',
        'build/resources.json','build/sass_hashes.json','source/probes/r18_trace.hpp','source/probes/r18.cu',
        'source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp',
        'source/overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp'):
        path=root/relative;hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [fit_path,local_path,*sorted((root/'build').glob('*.sass')),
        old_root/'build/cfg_b_stamped.sass',
        old_root/'source/overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp']:
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(scope='Same-call CTA0 second-output consumer progression, fixed cfg_b stage4/6 pair. '
        'Endpoint partitions close before trial reduction; averages divide by 15 and 46 commit intervals. '
        'No instantaneous commit spacing threshold, stage-specific fit constant or physical queue-depth claim.',
        root=str(root),gpu=gpu,processes=processes,timing=timing,paired_stage4_minus_stage6=paired,
        paired_summary=paired_summary,whole_time_response=whole_time,sparse_observer_cohort=old_cohort,
        candidate_comparisons=comparisons,shared_loop_fit=loop_fit,
        shared_loop_formula='H_(i+1)=W_i+h inside each output, while the handoff remains at mma return. '
            'I,A,C,D and slot-release equations are unchanged; p,latency,tau,h are shared by all stages/K. '
            'Only 739095 whole-L windows train these parameters.',
        observer_instruction_evidence=dict(
            source='After the progress commit16 stamp, source checks commit17 and commit63 before wait1; '
                'the old sparse source checked only commit16.',
            stage6_sparse='0x21d0 commit16 CS2R; 0x21e0 WARPGROUP.DEPBAR.LE 1; 0x21f0 wait1 CS2R.',
            stage6_progress='0x1f60 commit16 CS2R; 0x1f70 predicated CS2R; 0x1f80 IMAD; '
                '0x1f90 ISETP K63; 0x1fa0 predicated CS2R; 0x1fb0 WARPGROUP.DEPBAR.LE 1; 0x1fc0 wait1 CS2R.',
            interpretation='The 6-to-27-cycle change includes observer/control instructions. '
                'Neither 27 nor the roughly 90-cycle loop window is a bare hardware price.'),
        resources=read_json(root/'build/resources.json'),
        sass_hashes=read_json(root/'build/sass_hashes.json'),failed_attempts=failures,input_sha256=hashes,
        interval_limits=['full-wait begin to commit includes readiness, instruction issue and observer effects.',
            'wait1(K16) to full-wait begin(K17) is loop progression, not isolated barrier release service.',
            'mma return to MAIN_END combines handoff and final drain; this probe omits tail-begin.',
            'Commit1-to16 and commit17-to63 are measured cumulative spans, not extrapolations of Ktile16.',
            'Candidate comparisons retain parameters fitted to earlier 739095 whole windows; cohorts are not pooled.',
            'Plain matches scratch allocation/clear, not progress instructions or registers.'],
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    (output/'B-stage-progress.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'B-stage-progress.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(processes[0]));writer.writeheader();writer.writerows(processes)
    labels={'first_to_commit1':'FIRST_MMA→commit1','average_commit1_to16':'commit1→16 每间隔平均',
        'full_wait_to_commit16':'full等待开始16→commit16','commit_to_wait1_16':'commit16→wait1返回16',
        'loop_advance_16_to17':'wait1返回16→full等待开始17','full_wait_to_commit17':'full等待开始17→commit17',
        'commit16_to17':'commit16→17','average_commit17_to63':'commit17→63 每间隔平均',
        'last_commit_to_mma_return':'commit63→mma返回','mma_return_to_main_end':'mma返回→MAIN_END',
        'L':'完整L'}
    lines=['# stage 累计进度与相邻迭代分解','',
        '仅分析同调用 CTA0 第二输出。差值先按 trial 配对再汇总；commit1→16 和 commit17→63 分别含 15、46 个间隔。'
        '相邻三段及完整 L 的不重叠分段逐调用闭合，中位数之间不要求相加闭合。','',
        '| 周期窗口 | stage4 中位 | stage6 中位 | stage4−6 中位 | 差值十次范围 |','|---|---:|---:|---:|---:|']
    for field,label in labels.items():
        a,b=timing[0]['intervals'][field],timing[1]['intervals'][field];d=paired_summary[field]
        lines.append(f'| {label} | {a["median"]:.2f} | {b["median"]:.2f} | {d["median"]:+.2f} | {d["trial_min"]:+.2f}～{d["trial_max"]:+.2f} |')
    lines+=['','| 完整调用比较 | 中位变化 | 十次范围 |','|---|---:|---:|']
    for t in timing:
        d=t['stamped_relative_to_plain']
        lines.append(f'| stage{t["stages"]} progress/plain | {100*d["median"]:+.2f}% | {100*d["trial_min"]:+.2f}%～{100*d["trial_max"]:+.2f}% |')
    for protocol,response in whole_time.items():
        d=response['stage4_relative_change']
        lines.append(f'| {protocol} stage4/stage6 | {100*d["median"]:+.2f}% | {100*d["trial_min"]:+.2f}%～{100*d["trial_max"]:+.2f}% |')
    lines+=['','| 递推检查（stage4 / stage6） | 当前实测中位 | 原递推预测 | commit 推进预测 | 加共享循环项 |','|---|---:|---:|---:|---:|']
    for field in ('commit16_to17','loop_advance_16_to17','full_wait_to_commit16','full_wait_to_commit17',
        'commit_to_wait1_16','average_commit1_to16','average_commit17_to63','last_commit_to_main_end','L'):
        groups=[[t['intervals'][field]['median'] for t in timing],
            [c['predicted'][field] for c in comparisons['post_commit_compute']],
            [c['predicted'][field] for c in comparisons['commit_paced']],
            [c['predicted'][field] for c in comparisons['loop_paced']]]
        cells=[' / '.join(f'{x:.2f}' for x in values) for values in groups]
        lines.append(f'| {field} | '+ ' | '.join(cells)+' |')
    met=loop_fit['calibration_metrics']
    lines+=['',f'共享循环项仅以 739095 八个 whole-L 目标拟合 p/ℓ/τ/h；中位/最大误差 '
        f'{met["median_abs_pct"]:.2f}%/{met["max_abs_pct"]:.2f}%，局部参数秩 {loop_fit["local_jacobian_rank"]}/4。'
        '多起点等损失参数示例范围与逐参数敏感性保存在 JSON，不把这些范围当完整置信区间。'
        '该候选的循环推进、wait1、尾段及stage4早段均与独立progress窗口不符；整体拟合改善不足以支持此修订。'
        '不继续叠加参数，也不把拟合h或τ作为硬件常数。',
        '','所有递推仅以 739095 whole-L 拟合参数；新进度没有参与拟合。'
        '前两模型的零循环推进可以直接与 wait1→下一次 full-wait 比较；512 仅是旧候选假设，不是数据筛选条件。',
        '','| 独立 observer 批次比较 | progress L 中位 | 739556 sparse L 中位 | 两批中位差 |',
        '|---|---:|---:|---:|']
    for old in old_cohort:
        current=next(t for t in timing if t['stages']==old['stages'])
        lines.append(f'| stage{old["stages"]} | {current["intervals"]["L"]["median"]:.1f} | '
            f'{old["intervals"]["L"]["median"]:.1f} | {old["progress_minus_sparse_cohort_medians"]["L"]:+.1f} |')
    lines+=['','739556 与本批只比较同名端点和各自 plain 扰动，不能拼成一条时间线。'
        'stage6旧SASS的commit16 clock（0x21d0）后紧接wait1；新SASS的commit16 clock（0x1f60）后，'
        '还执行predicated CS2R、IMAD、ISETP和另一次predicated CS2R，才到0x1fb0的wait1。'
        '因此6→27 cycle含新增观测控制指令，不是裸wait服务价。'
        '本 probe 没有 tail-begin，mma 返回后只报告 handoff 与 drain 的合并段。'
        'JSON/CSV 保留逐调用周期、配对差、源码/SASS 身份及资源，结论不直接生成硬件服务价格。','']
    (output/'B-stage-progress.md').write_text('\n'.join(lines))
    return report


def _first_supply_intervals(cta, events):
    """CTA55 first-output endpoints; producer/consumer markers have no total order."""
    values={}
    for unit,entry,prod,first,end in (
        ('cycles',cta['entry_c'],cta['prod_c'],*cta['tiles'][0][:2]),
        ('ns',cta['entry_ns'],cta['prod_ns'],*cta['tiles_ns'][0][:2])):
        if not entry<=prod<=first<=end:
            raise ValueError('first-supply ordinary endpoints are not ordered')
        intervals=dict(P0=prod-entry,S=first-prod,L=end-first,
            S_plus_L=end-prod,entry_to_FIRST=first-entry,entry_to_MAIN_END=end-entry)
        if events['enabled']:
            suffix='c' if unit=='cycles' else 'ns'
            issue,wait=events['issue_end_'+suffix],events['wait_begin_'+suffix]
            if issue<prod or not entry<=wait<=first:
                raise ValueError('first-supply role-local endpoints are not ordered')
            intervals.update(prod_to_issue=issue-prod,issue_to_FIRST=first-issue,
                wait_minus_issue=wait-issue,wait_to_FIRST=first-wait,
                entry_to_wait=wait-entry,prod_to_wait=wait-prod)
            if intervals['prod_to_issue']+intervals['issue_to_FIRST']!=intervals['S']:
                raise ValueError('first-supply producer partition does not close')
            if intervals['wait_minus_issue']+intervals['wait_to_FIRST']!=intervals['issue_to_FIRST']:
                raise ValueError('first-supply role partition does not close')
        if intervals['P0']+intervals['S']+intervals['L']!=intervals['entry_to_MAIN_END']:
            raise ValueError('first-supply complete window does not close')
        values.update({name+'_'+unit:value for name,value in intervals.items()})
    return values


def first_supply_diagnostics(root, output):
    """First-supply event decomposition, preceded by same-binary observer controls."""
    from analyze_r13_sm import select_attempts
    from analyze_r10 import _work_swap_pitch_windows, _producer_group_rank
    import v08_model
    import v06_run as common

    root,output=Path(root),Path(output);common.verify(root)
    rows={r['id']:r for r in read_json(root/'cases.json')}
    setups={r['case']:r['setup'] for r in read_json(root/'static_setup.json')}
    controls=[r for r in rows.values() if r['set']=='ctrl']
    ignored={'id','group','variants','first_supply_events','producer_delay_ns','ldb'}
    fixed={k:v for k,v in controls[0].items() if k not in ignored}
    if len(controls)!=8 or any({k:v for k,v in r.items() if k not in ignored}!=fixed for r in controls):
        raise ValueError('first-supply event controls changed another condition')
    if {(r['first_supply_events'],r['producer_delay_ns'],r['ldb']) for r in controls}!={
        (e,d,p) for e in (0,1) for d in (0,512) for p in (3072,3088)}:
        raise ValueError('first-supply factors incomplete')
    def span(values):
        values=list(values)
        return dict(median=statistics.median(values),process_min=min(values),process_max=max(values))
    def summary(records):
        return {k:span(r[k] for r in records) for k in records[0]}
    def name(e,d,p):return f'r13_first_e{e}_d{d}_p{p}'
    calls={};plain={};targets={};processes=[];hashes={};failures={};contexts={};work=None
    placements={}
    for row in rows.values():
        case=row['id'];calls[case]={};plain[case]={};targets[case]={}
        if row['set']=='ctrl':
            expected=['dual'] if row['first_supply_events'] else ['plain','dual']
            if row['variants']!=expected:raise ValueError('first-supply variants changed')
            setup=setups[case]
            current=v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'],row['raster_order'])
            if work is None:work=current
            if current!=work:raise ValueError('first-supply events changed software work')
            contexts[case]=dict(row=row,setup=setup)
        selected,failed,_=select_attempts(root,case,tuple(row['variants']))
        failures[case]=[r['path'] for r in failed]
        for item in selected:
            record=item['record'];obs=replay(root,record,row);trial=record['trial']
            for path in (root/item['path'],root/record['raw']):hashes[str(path)]=common.sha(path)
            if record['variant']=='plain':plain[case][trial]=obs['elapsed_us'];continue
            if [c['work'] for c in obs['ctas']]!=work:raise ValueError('actual first-supply work changed')
            events=obs['first_supply']
            if (events['cta'],int(events['enabled']),events['protocol'])!=(55,row['first_supply_events'],'r13-first-supply-v1'):
                raise ValueError('first-supply endpoint identity changed')
            c=next(c for c in obs['ctas'] if c['cta']==55)
            intervals=_first_supply_intervals(c,events);delay=obs['producer_delay']
            intervals.update(delay_ns=delay['duration_ns'],delay_cycles=delay['duration_cycles'],
                end_to_prod_ns=c['prod_ns']-delay['end_ns'],end_to_prod_cycles=c['prod_c']-delay['end_c'])
            placement=tuple((z['cta'],z['sm']) for z in obs['ctas'])
            placement_key=hashlib.sha256(json.dumps(placement).encode()).hexdigest()
            placements[placement_key]=placement
            target=dict(case=case,trial=trial,events=row['first_supply_events'],delay_ns=row['producer_delay_ns'],
                pitch_bytes=2*(row['ldb']-3072),sm=c['sm'],work=c['work'],intervals=intervals,
                first_supply=events,producer_delay=delay,elapsed_us=obs['elapsed_us'],
                source_group=_producer_group_rank(obs['ctas']),placement=placement_key,b_address=obs['b_address'])
            processes.append(target);targets[case][trial]=target;calls[case][trial]=obs
        for variant in row['variants']:
            if len(plain[case] if variant=='plain' else calls[case])!=10:
                raise ValueError('first-supply comparison requires ten successful processes per variant')
    if len({setups[r['id']]['gpu_uuid'] for r in controls})!=1:raise ValueError('first-supply controls span cards')

    paired=[];pair_summaries=[];background=[];background_summaries=[];selected_windows=[];selected_summaries=[]
    window_fields=('L_cycles_delta','L_ns_delta','entry_relative_ns_delta','producer_relative_ns_delta',
        'FIRST_relative_ns_delta','entry_to_producer_ns_delta','entry_to_FIRST_ns_delta','entry_to_MAIN_END_ns_delta',
        'entry_to_FIRST_cycles_delta','entry_to_MAIN_END_cycles_delta','prefill_ns_delta')
    def compare(kind,factors,left,right):
        pair_id=right+' minus '+left;first_B=work[55][0][1];target_rows=[];relative_rows=[];time_rows=[]
        selected_groups=defaultdict(list);background_groups=defaultdict(list)
        for trial in range(10):
            a,b=targets[left][trial],targets[right][trial]
            common_fields=a['intervals'].keys()&b['intervals'].keys()
            delta={k:b['intervals'][k]-a['intervals'][k] for k in sorted(common_fields)}
            target_rows.append(delta)
            relative={k:b['intervals'][k]/a['intervals'][k]-1 for k in
                ('S_ns','L_ns','entry_to_MAIN_END_ns','S_cycles','L_cycles')}
            relative_rows.append(relative)
            time=dict(elapsed_us=b['elapsed_us']-a['elapsed_us'],elapsed_ratio=b['elapsed_us']/a['elapsed_us']-1)
            time_rows.append(time)
            paired.append(dict(pair=pair_id,kind=kind,**factors,trial=trial,target_delta=delta,
                target_relative=relative,whole_call_delta=time))
            windows=_work_swap_pitch_windows(calls[left][trial],calls[right][trial])
            for w in windows:
                if w['cta'] not in (55,91):continue
                values={k:w[k] for k in window_fields if k in w}
                selected_groups[w['cta'],w['j']].append(values)
                selected_windows.append(dict(pair=pair_id,trial=trial,cta=w['cta'],j=w['j'],T=w['T'],
                    coord=[w['mi'],w['ni']],base_sm=w['base_sm'],variant_sm=w['variant_sm'],delta=values))
            first=[w for w in windows if w['j']==0]
            groups={'same_B_partners':[w for w in first if w['ni']==first_B and w['cta']!=55],
                'other130':[w for w in first if w['cta'] not in (55,91)]}
            for group,subset in groups.items():
                values={k:statistics.median(w[k] for w in subset) for k in window_fields}
                background_groups[group].append(values)
                background.append(dict(pair=pair_id,trial=trial,group=group,CTAs=len(subset),delta=values))
        pair_summaries.append(dict(pair=pair_id,kind=kind,**factors,target_delta=summary(target_rows),
            target_relative=summary(relative_rows),whole_call_delta=summary(time_rows)))
        for (cta,j),values in selected_groups.items():
            selected_summaries.append(dict(pair=pair_id,cta=cta,j=j,delta=summary(values)))
        for group,values in background_groups.items():
            background_summaries.append(dict(pair=pair_id,group=group,delta=summary(values)))
    for d in (0,512):
        for p in (0,32):compare('observer',dict(delay_ns=d,pitch_bytes=p),name(0,d,p),name(1,d,p))
    for e in (0,1):
        for d in (0,512):compare('pitch',dict(events=e,delay_ns=d),name(e,d,0),name(e,d,32))
        for p in (0,32):compare('delay',dict(events=e,pitch_bytes=p),name(e,0,p),name(e,512,p))
    interactions=[];interaction_summaries=[]
    for e in (0,1):
        group=[]
        for trial in range(10):
            a=next(r for r in paired if r['kind']=='pitch' and r['events']==e and r['delay_ns']==0 and r['trial']==trial)
            b=next(r for r in paired if r['kind']=='pitch' and r['events']==e and r['delay_ns']==512 and r['trial']==trial)
            delta={k:b['target_delta'][k]-a['target_delta'][k] for k in a['target_delta']}
            group.append(delta);interactions.append(dict(events=e,trial=trial,delta=delta))
        interaction_summaries.append(dict(events=e,delta=summary(group)))
    observer_interaction=[]
    for trial in range(10):
        a=next(r['delta'] for r in interactions if r['events']==0 and r['trial']==trial)
        b=next(r['delta'] for r in interactions if r['events']==1 and r['trial']==trial)
        observer_interaction.append(dict(trial=trial,delta={k:b[k]-a[k] for k in a}))
    observer_interaction_summary=summary([r['delta'] for r in observer_interaction])
    case_summaries=[]
    for case,values in targets.items():
        if not values:continue
        values=list(values.values());sample=values[0]
        case_summaries.append(dict(case=case,events=sample['events'],delay_ns=sample['delay_ns'],pitch_bytes=sample['pitch_bytes'],
            intervals=summary([r['intervals'] for r in values]),sms=sorted({r['sm'] for r in values}),
            ranks={event:summary([{k:r['source_group'][event][k] for k in
                ('rank_min','rank_max','target_minus_peer_median_ns')} for r in values]) for event in ('entry','producer','FIRST')}))
    timing={}
    for e in (0,1):
        for d in (0,512):
            for p in (0,32):
                case=name(e,d,p);base=name(0,d,p)
                timing[case]=dict(dual_us=span(r['elapsed_us'] for r in targets[case].values()),
                    events0_plain_us=span(plain[base].values()),
                    dual_relative_to_events0_plain=span(targets[case][t]['elapsed_us']/plain[base][t]-1 for t in range(10)))
    previous=root.parent/'20261010-R10-producer-delay';reference_comparison={}
    if read_json(previous/'environment.json')['gpu']!=read_json(root/'environment.json')['gpu']:
        raise ValueError('first-supply reference comparison changed card')
    for row in rows.values():
        if row['set']!='reference':continue
        case=row['id'];old=[]
        for item in select_attempts(previous,case,('plain',))[0]:
            old.append(replay(previous,item['record'],row)['elapsed_us'])
            for path in (previous/item['path'],previous/item['record']['raw']):hashes[str(path)]=common.sha(path)
        if len(old)!=10:raise ValueError('previous first-supply reference needs ten processes')
        reference_comparison[case]=dict(current=span(plain[case].values()),previous=span(old),
            median_change=statistics.median(plain[case].values())/statistics.median(old)-1)
    for filename in ('cases.json','static_setup.json','environment.json','source_hashes.json'):
        path=previous/filename;hashes[str(path)]=common.sha(path)
    for filename in ('cases.json','static_setup.json','environment.json','source_hashes.json','build/resources.json','build/sass_hashes.json'):
        path=root/filename;hashes[str(path)]=common.sha(path)
    for path in sorted(root.glob('correctness-*.jsonl')):hashes[str(path)]=common.sha(path)
    origins=[dict(trial=t,by_case={r['case']:r['b_address']['b_buffer_ptr'] for r in processes if r['trial']==t}) for t in range(10)]
    report=dict(scope='CTA55 first output/Ktile0, same-binary events0/1 observer controls precede first-supply decomposition. '
        'Process-level intervals close before independent column reductions; no fitted hardware prices.',
        root=str(root),environment=read_json(root/'environment.json'),contexts=contexts,processes=processes,
        case_summaries=case_summaries,paired_processes=paired,paired_summaries=pair_summaries,
        interaction_processes=interactions,interaction_summaries=interaction_summaries,
        observer_interaction_change_processes=observer_interaction,observer_interaction_change=observer_interaction_summary,
        selected_window_processes=selected_windows,selected_window_summaries=selected_summaries,
        background_processes=background,background_summaries=background_summaries,timing=timing,
        references=reference_comparison,
        placement_maps=placements,work_sha256=hashlib.sha256(json.dumps(work).encode()).hexdigest(),
        virtual_origins=dict(per_trial=origins,same_origin_trials=[r['trial'] for r in origins if len(set(r['by_case'].values()))==1]),
        resources=read_json(root/'build/resources.json'),sass_hashes=read_json(root/'build/sass_hashes.json'),
        failed_attempts=failures,input_sha256=hashes,analyzer_sha256=common.sha(Path(__file__)),
        interpretation=['issue_end follows the producer two-copy issue sequence; it does not timestamp physical TMA service start or completion.',
            'wait_begin precedes consumer128 first consumer_try_wait; FIRST follows the existing consumer wait/instruction path. '
            'wait_begin-to-FIRST is not pure wait latency and FIRST is not a measured hardware-ready event.',
            'Producer and consumer roles have no imposed total order: wait-minus-issue and issue-to-FIRST may be negative.',
            'Events0 and events1 use one binary and equal scratch capacity but different dynamic observer instructions. '
            'Report their changes in S/L/complete window before interpreting events1 local intervals.',
            'The two plain references compare the present batch with 739619 only for card/binary context; '
            'they do not estimate events1-minus-events0 observer error.',
            'Descriptor prefetch precedes the producer delay. Delay changes temporal state and prefetch completion opportunity, not uniquely source arrival order.',
            'Each column is independently reduced over ten processes; interval closure is checked per process, not by summing medians.',
            'The zero-delay path still reads timers. Short cycle/ns spans do not calibrate frequency.',
            'Trial pairs are independent processes. Common virtual origins do not establish common physical pages; backgrounds are not extra process replicas.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'R13-first-supply.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def cell(v):return f'{v["median"]:.1f} [{v["process_min"]:.1f},{v["process_max"]:.1f}]'
    lines=['# CTA55首次供给分解','',
        '先比较同一binary的events1−events0，随后解释新增端点。每列独立取十进程中位；闭合在逐进程检验。'
        '方括号为十进程范围。两个角色不预设总顺序，FIRST不等于硬件ready时刻。','',
        '| 新增打点扰动 | ΔS ns | ΔL ns | Δ(entry→MAIN_END) ns | 完整调用变化 |',
        '|---|---:|---:|---:|---:|']
    for r in pair_summaries:
        if r['kind']!='observer':continue
        s=r['target_delta'];v=r['whole_call_delta']['elapsed_ratio']
        lines.append(f'| delay{r["delay_ns"]}, B+{r["pitch_bytes"]} B | '+
            ' | '.join(cell(s[k]) for k in ('S_ns','L_ns','entry_to_MAIN_END_ns'))+f' | {100*v["median"]:+.2f}% |')
    observer=next(r for r in pair_summaries if r['kind']=='observer' and r['delay_ns']==0 and r['pitch_bytes']==32)
    s=observer['target_relative']['S_ns']
    lines+=['',f'delay0/B+32 B的S观察扰动中位 {100*s["median"]:+.2f}%，十进程范围 '
        f'{100*s["process_min"]:+.2f}%～{100*s["process_max"]:+.2f}%。'
        '整窗口接近不能证明局部S无扰动；下述分解用于定位增长区间，不将绝对区间作为裸服务价格。']
    lines+=['','| events1条件 | prod→issue_end ns | issue_end→FIRST ns | wait_begin−issue_end ns | wait_begin→FIRST ns |',
        '|---|---:|---:|---:|---:|']
    for r in case_summaries:
        if r['events']!=1:continue
        lines.append(f'| delay{r["delay_ns"]}, B+{r["pitch_bytes"]} B | '+' | '.join(cell(r['intervals'][k]) for k in
            ('prod_to_issue_ns','issue_to_FIRST_ns','wait_minus_issue_ns','wait_to_FIRST_ns'))+' |')
    lines+=['','| events1直接延迟512−0 | Δ(prod→issue_end) cycles | Δ(issue_end→FIRST) cycles | Δ(issue_end→FIRST) ns | Δ(entry→wait_begin) ns |',
        '|---|---:|---:|---:|---:|']
    for r in pair_summaries:
        if r['kind']!='delay' or r['events']!=1:continue
        lines.append(f'| B+{r["pitch_bytes"]} B | '+' | '.join(cell(r['target_delta'][k]) for k in
            ('prod_to_issue_cycles','issue_to_FIRST_cycles','issue_to_FIRST_ns','entry_to_wait_ns'))+' |')
    lines+=['','增长落在两次copy发出后的issue_end→FIRST区间；prod之后的发射准备没有相应增长，'
        '消费者进入首次等待也没有同量级推迟。它限制了“发射准备变慢”或“消费者更晚到达等待点”作为主要解释。'
        'TMA完成、屏障可见性、消费者从等待中恢复的调度，以及producer继续推进后续tile的相互影响仍未分开。']
    lines+=['','| 首输出直接延迟512−0 | ΔS ns | ΔL ns | Δ(S+L) ns | Δ(entry→MAIN_END) ns |',
        '|---|---:|---:|---:|---:|']
    for r in pair_summaries:
        if r['kind']!='delay':continue
        lines.append(f'| events{r["events"]}, B+{r["pitch_bytes"]} B | '+' | '.join(cell(r['target_delta'][k]) for k in
            ('S_ns','L_ns','S_plus_L_ns','entry_to_MAIN_END_ns'))+' |')
    lines+=['','| delay×pitch交互 | events0 ns | events1 ns |','|---|---:|---:|']
    for k in ('S_ns','L_ns','S_plus_L_ns','entry_to_MAIN_END_ns'):
        lines.append(f'| {k} | '+ ' | '.join(cell(r['delta'][k]) for r in interaction_summaries)+' |')
    lines+=['','| 交互的观察协议差（events1−events0） | ns中位与十进程范围 |','|---|---:|']
    for k in ('L_ns','S_plus_L_ns','entry_to_MAIN_END_ns'):
        lines.append(f'| {k} | {cell(observer_interaction_summary[k])} |')
    lines+=['','e0保留为主要因果对照；e1用于定位端点之间的增长。交互方向保留不等于幅度完全一致，'
        'e1的S交互范围跨零，不能单凭其中位值声称首段行距惩罚稳定增大。']
    lines+=['','| plain参照 | 本批中位 µs | 739619中位 µs | 中位变化 |','|---|---:|---:|---:|']
    for case,r in reference_comparison.items():
        lines.append(f'| {case} | {r["current"]["median"]:.3f} | {r["previous"]["median"]:.3f} | {100*r["median_change"]:+.2f}% |')
    lines+=['','跨批参照只提供卡状态与二进制背景，不作为本批events1−events0的观察误差。']
    lines+=['','JSON保留同端点cycles、ns、逐进程差、实际延迟和同B进度名次；'
        'CTA55/91各输出及17个同B伙伴、其余130CTA背景分列。'
        'issue_end与FIRST之间包含异步事务完成、消费者进度和观测指令，不能直接当成裸TMA延迟。'
        '描述符预取、请求服务与调度初态尚需根据这些窗口判断可区分范围，不拟合新常数。','']
    (output/'R13-first-supply.md').write_text('\n'.join(lines))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--calibration-run", type=Path)
    parser.add_argument("--sm-summary", type=Path, help="fit local/grid/wave caps to an existing R13 SM-scan summary")
    parser.add_argument("--coverage-suite", action='store_true', help="two post-reveal coverage/prefill forms; --run is R10 job738203")
    parser.add_argument("--fill-suite", action='store_true',help='same-card valid-address/OOB cycle versus direct-ns development; --run is R10 job738203')
    parser.add_argument("--compose-fill",type=Path,help='compose an existing fill-suite report with same-card dual event calibration')
    parser.add_argument("--supply-calibration",action='store_true',help='single-run three-config per-CTA supply calibration; j0 remains diagnostic')
    parser.add_argument("--assumed-ghz",type=float,default=1.6,help='explicit frequency assumption for supply NS predictions and comparisons')
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.supply_calibration:
        if args.assumed_ghz<=0:parser.error('--assumed-ghz must be positive')
        supply_calibration(args.run.resolve(),args.output.resolve(),args.assumed_ghz)
    elif args.compose_fill:
        if args.assumed_ghz<=0:parser.error('--assumed-ghz must be positive')
        compose_fill(args.run.resolve(),args.compose_fill.resolve(),args.output.resolve(),args.assumed_ghz)
    elif args.fill_suite:
        if args.assumed_ghz<=0:parser.error('--assumed-ghz must be positive')
        fill_suite(args.run.resolve(),args.output.resolve(),args.assumed_ghz)
    elif args.coverage_suite:
        coverage_suite(args.run.resolve(), args.output.resolve())
    elif args.sm_summary:
        sm_candidates(args.run.resolve(), args.sm_summary.resolve(), args.output.resolve())
    else:
        if args.calibration_run is None:parser.error("--calibration-run is required without --sm-summary")
        analyze(args.run.resolve(), args.calibration_run.resolve(), args.output.resolve())
