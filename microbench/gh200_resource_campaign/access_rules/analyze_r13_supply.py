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


def _ready_window_bounds(lower, upper, issue, first):
    """Conservative endpoint intervals; a first successful poll has no lower bound."""
    if lower is not None and upper is not None and lower>upper:
        raise ValueError('ready lower bound exceeds upper bound')
    return dict(ready=dict(lower=lower,upper=upper),
        issue_to_ready=dict(lower=None if lower is None else lower-issue,
            upper=None if upper is None else upper-issue),
        ready_to_FIRST=dict(lower=None if upper is None else first-upper,
            upper=None if lower is None else first-lower),
        width=None if lower is None or upper is None else upper-lower)


def _ready_bound_difference(left, right):
    """right-minus-left without choosing a readiness point inside either interval."""
    return dict(lower=None if right['lower'] is None or left['upper'] is None else right['lower']-left['upper'],
        upper=None if right['upper'] is None or left['lower'] is None else right['upper']-left['lower'])


def _ready_bound_summary(bounds):
    """Median bounds keep censored repetitions as unbounded, rather than dropping them."""
    import math
    lows=[-math.inf if b['lower'] is None else b['lower'] for b in bounds]
    highs=[math.inf if b['upper'] is None else b['upper'] for b in bounds]
    lo,hi=statistics.median(lows),statistics.median(highs)
    return dict(processes=len(bounds),both_finite=sum(b['lower'] is not None and b['upper'] is not None for b in bounds),
        median_lower=lo if math.isfinite(lo) else None,median_upper=hi if math.isfinite(hi) else None,
        process_lower=min(lows) if all(math.isfinite(v) for v in lows) else None,
        process_upper=max(highs) if all(math.isfinite(v) for v in highs) else None)


def _ready_call_bounds(cta, first_supply, observer):
    """Use the protocol's recorded generation boundary; never upgrade an old trace."""
    release_bound=observer.get('generation_bound')=='first_release'
    limit=observer['release_pre_c'] if release_bound else cta['tiles'][0][0]
    late_status='after_release' if release_bound else 'after_first'
    success=bool(observer['success']);bound=success and observer['exit_c']<=limit
    if observer['sm']!=cta['sm']:raise ValueError('ready observer is not on the target SM')
    status=('upper_only' if observer['tests']==1 else 'interval') if bound else (late_status if success else 'poll_limit')
    if observer['first_generation']!=bound or observer['status']!=status:
        raise ValueError('ready first-generation classification disagrees with clocks')
    windows={};widths={}
    for unit,suffix,first in (('cycles','c',cta['tiles'][0][0]),('ns','ns',cta['tiles_ns'][0][0])):
        lower=observer['last_false_pre_'+suffix] if status=='interval' else None
        upper=observer['exit_'+suffix] if bound else None
        result=_ready_window_bounds(lower,upper,first_supply['issue_end_'+suffix],first)
        windows['issue_to_ready_'+unit]=result['issue_to_ready']
        windows['ready_to_FIRST_'+unit]=result['ready_to_FIRST']
        wait=first_supply['wait_begin_'+suffix]
        windows['wait_begin_to_ready_'+unit]=dict(lower=None if lower is None else lower-wait,
            upper=None if upper is None else upper-wait)
        widths[unit]=result['width']
    return dict(status=status,first_generation=bound,tests=observer['tests'],bounds=windows,width=widths)


def _startup_shift_check(delta_s, delta_l):
    """Necessary max-plus bound for a delayed first consumer and fixed later roots."""
    return dict(delta_S=delta_s,delta_L=delta_l,delta_S_plus_L=delta_s+delta_l,
        premise_delta_S_nonnegative=delta_s>=0,satisfies=delta_s>=0 and -delta_s<=delta_l<=0,
        lower_violation=max(0,-delta_s-delta_l),upper_violation=max(0,delta_l))


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


def ready_observer_diagnostics(root, output):
    """Read-only first-generation readiness bounds, retaining every censored process."""
    from analyze_r13_sm import select_attempts
    from analyze_r10 import _work_swap_pitch_windows, _producer_group_rank
    import v08_model
    import v06_run as common

    root,output=Path(root),Path(output);common.verify(root)
    rows={r['id']:r for r in read_json(root/'cases.json')}
    setups={r['case']:r['setup'] for r in read_json(root/'static_setup.json')}
    controls=[r for r in rows.values() if r['set']=='ctrl']
    ignored={'id','group','variants','ready_observer','producer_delay_ns','ldb'}
    fixed={k:v for k,v in controls[0].items() if k not in ignored}
    if len(controls)!=8 or any({k:v for k,v in r.items() if k not in ignored}!=fixed for r in controls):
        raise ValueError('ready observer changed another factorial condition')
    if {(r['ready_observer'],r['producer_delay_ns'],r['ldb']) for r in controls}!={
        (o,d,p) for o in (0,1) for d in (0,512) for p in (3072,3088)}:
        raise ValueError('ready observer factors incomplete')
    release_bound=controls[0].get('ready_generation_bound')=='first_release'
    protocol='r13-ready-release-v1' if release_bound else 'r13-ready-observer-v1'
    artifact='R13-ready-release' if release_bound else 'R13-ready-observer'
    boundary='release_pre_c' if release_bound else 'FIRST_c'
    late_status='after_release' if release_bound else 'after_first'
    previous_name='20261010-R13-ready-observer' if release_bound else '20261010-R13-first-supply'
    previous_job='739645' if release_bound else '739629'
    def span(values):
        values=list(values)
        return dict(median=statistics.median(values),process_min=min(values),process_max=max(values))
    def summarize(values):return {k:span(v[k] for v in values) for k in values[0]}
    def name(o,d,p):return f'r13_ready_o{o}_d{d}_p{p}'
    calls={};plain={};targets={};processes=[];contexts={};hashes={};failures={};placements={};work=None
    for row in rows.values():
        case=row['id'];calls[case]={};plain[case]={};targets[case]={}
        if row['set']=='ctrl':
            if row['first_supply_events']!=1 or row['variants']!=(['dual'] if row['ready_observer'] else ['plain','dual']):
                raise ValueError('ready observer changed first-supply events or variants')
            setup=setups[case]
            current=v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'],row['raster_order'])
            if work is None:work=current
            if work!=current:raise ValueError('ready observer changed software work')
            contexts[case]=dict(row=row,setup=setup)
        selected,failed,_=select_attempts(root,case,tuple(row['variants']));failures[case]=[r['path'] for r in failed]
        for item in selected:
            record=item['record'];trial=record['trial'];obs=replay(root,record,row)
            for path in (root/item['path'],root/record['raw']):hashes[str(path)]=common.sha(path)
            if record['variant']=='plain':plain[case][trial]=obs['elapsed_us'];continue
            if [c['work'] for c in obs['ctas']]!=work:raise ValueError('ready observer actual work changed')
            c=next(c for c in obs['ctas'] if c['cta']==55);first=obs['first_supply'];ready=obs['ready_observer']
            if not first['enabled'] or (ready['cta'],int(ready['enabled']),ready['protocol'])!=(55,row['ready_observer'],protocol):
                raise ValueError('ready observation identity changed')
            if ready['enabled'] and release_bound and ready.get('generation_bound')!='first_release':
                raise ValueError('ready release boundary missing from new protocol')
            intervals=_first_supply_intervals(c,first);delay=obs['producer_delay']
            intervals.update(delay_cycles=delay['duration_cycles'],delay_ns=delay['duration_ns'],
                end_to_prod_cycles=c['prod_c']-delay['end_c'],end_to_prod_ns=c['prod_ns']-delay['end_ns'])
            mapping=tuple((q['cta'],q['sm']) for q in obs['ctas'])
            mapping_id=hashlib.sha256(json.dumps(mapping).encode()).hexdigest();placements[mapping_id]=mapping
            target=dict(case=case,trial=trial,observer=row['ready_observer'],delay_ns=row['producer_delay_ns'],
                pitch_bytes=2*(row['ldb']-3072),sm=c['sm'],work=c['work'],intervals=intervals,
                ready_observer=ready,first_supply=first,producer_delay=delay,elapsed_us=obs['elapsed_us'],
                source_group=_producer_group_rank(obs['ctas']),placement=mapping_id,b_address=obs['b_address'])
            if ready['enabled']:
                target['ready_bounds']=_ready_call_bounds(c,first,ready)
                first_c=c['tiles'][0][0]
                target['observer_position']=dict(start_minus_issue_cycles=ready['start_c']-first['issue_end_c'],
                    start_minus_wait_cycles=ready['start_c']-first['wait_begin_c'],
                    start_minus_FIRST_cycles=ready['start_c']-first_c,exit_minus_FIRST_cycles=ready['exit_c']-first_c,
                    last_false_pre_minus_FIRST_cycles=ready['last_false_pre_c']-first_c if ready['last_false_pre_c'] else None,
                    last_false_pre_to_exit_cycles=ready['exit_c']-ready['last_false_pre_c'] if ready['last_false_pre_c'] else None)
                if release_bound:
                    target['observer_position'].update(release_pre_minus_FIRST_cycles=ready['release_pre_c']-first_c,
                        exit_minus_release_pre_cycles=ready['exit_c']-ready['release_pre_c'])
            calls[case][trial]=obs;targets[case][trial]=target;processes.append(target)
        for variant in row['variants']:
            if len(plain[case] if variant=='plain' else calls[case])!=10:
                raise ValueError('ready observer requires ten processes without censor-based replacement')
    if len({setups[r['id']]['gpu_uuid'] for r in controls})!=1:raise ValueError('ready controls span cards')

    paired=[];paired_summaries=[];background=[];background_summaries=[]
    def compare(kind,factors,left,right):
        pair=right+' minus '+left;differences=[];ratios=[];elapsed=[];bound_rows=[];bg=defaultdict(list)
        for trial in range(10):
            a,b=targets[left][trial],targets[right][trial]
            delta={k:b['intervals'][k]-a['intervals'][k] for k in a['intervals']};differences.append(delta)
            relative={k:b['intervals'][k]/a['intervals'][k]-1 for k in ('S_cycles','L_cycles','S_ns','L_ns','entry_to_MAIN_END_ns')}
            ratios.append(relative);elapsed.append(dict(delta_us=b['elapsed_us']-a['elapsed_us'],ratio=b['elapsed_us']/a['elapsed_us']-1))
            entry=dict(kind=kind,pair=pair,**factors,trial=trial,delta=delta,relative=relative,elapsed=elapsed[-1])
            if a['observer'] and b['observer']:
                bounds={k:_ready_bound_difference(a['ready_bounds']['bounds'][k],b['ready_bounds']['bounds'][k])
                    for k in a['ready_bounds']['bounds']}
                entry['bounds']=bounds;bound_rows.append(bounds)
            paired.append(entry)
            first=[w for w in _work_swap_pitch_windows(calls[left][trial],calls[right][trial]) if w['j']==0]
            for label,group in (('same_B_partners',[w for w in first if w['ni']==work[55][0][1] and w['cta']!=55]),
                ('other130',[w for w in first if w['cta'] not in (55,91)]),('CTA91',[w for w in first if w['cta']==91])):
                values={k:statistics.median(w[k] for w in group) for k in ('L_cycles_delta','L_ns_delta',
                    'prefill_ns_delta','entry_relative_ns_delta','producer_relative_ns_delta','FIRST_relative_ns_delta','entry_to_MAIN_END_ns_delta')}
                bg[label].append(values);background.append(dict(pair=pair,trial=trial,group=label,CTAs=len(group),delta=values))
        result=dict(kind=kind,pair=pair,**factors,delta=summarize(differences),relative=summarize(ratios),elapsed=summarize(elapsed))
        if bound_rows:result['bounds']={k:_ready_bound_summary([r[k] for r in bound_rows]) for k in bound_rows[0]}
        paired_summaries.append(result)
        for label,values in bg.items():background_summaries.append(dict(pair=pair,group=label,delta=summarize(values)))
    for d in (0,512):
        for p in (0,32):compare('observer',dict(delay_ns=d,pitch_bytes=p),name(0,d,p),name(1,d,p))
    for o in (0,1):
        for d in (0,512):compare('pitch',dict(observer=o,delay_ns=d),name(o,d,0),name(o,d,32))
        for p in (0,32):compare('delay',dict(observer=o,pitch_bytes=p),name(o,0,p),name(o,512,p))
    interactions=[];interaction_summaries=[]
    for o in (0,1):
        group=[];bound_rows=[]
        for trial in range(10):
            a=next(r for r in paired if r['kind']=='pitch' and r['observer']==o and r['delay_ns']==0 and r['trial']==trial)
            b=next(r for r in paired if r['kind']=='pitch' and r['observer']==o and r['delay_ns']==512 and r['trial']==trial)
            delta={k:b['delta'][k]-a['delta'][k] for k in a['delta']};group.append(delta)
            entry=dict(observer=o,trial=trial,delta=delta)
            if o:
                entry['bounds']={k:_ready_bound_difference(a['bounds'][k],b['bounds'][k]) for k in a['bounds']}
                bound_rows.append(entry['bounds'])
            interactions.append(entry)
        result=dict(observer=o,delta=summarize(group))
        if bound_rows:result['bounds']={k:_ready_bound_summary([r[k] for r in bound_rows]) for k in bound_rows[0]}
        interaction_summaries.append(result)
    observer_interactions=[]
    for trial in range(10):
        a=next(r['delta'] for r in interactions if r['observer']==0 and r['trial']==trial)
        b=next(r['delta'] for r in interactions if r['observer']==1 and r['trial']==trial)
        observer_interactions.append(dict(trial=trial,delta={k:b[k]-a[k] for k in a}))
    observer_interaction_summary=summarize([r['delta'] for r in observer_interactions])
    shift_processes=[];shift_summaries=[];ready_change_checks=[]
    for r in paired:
        if r['kind']!='delay':continue
        for unit in ('cycles','ns'):
            shift_processes.append(dict(observer=r['observer'],pitch_bytes=r['pitch_bytes'],trial=r['trial'],unit=unit,
                **_startup_shift_check(r['delta']['S_'+unit],r['delta']['L_'+unit])))
    for observer in (0,1):
        for pitch in (0,32):
            for unit in ('cycles','ns'):
                group=[r for r in shift_processes if (r['observer'],r['pitch_bytes'],r['unit'])==(observer,pitch,unit)]
                shift_summaries.append(dict(observer=observer,pitch_bytes=pitch,unit=unit,processes=len(group),
                    premise_count=sum(r['premise_delta_S_nonnegative'] for r in group),satisfies_count=sum(r['satisfies'] for r in group),
                    **{k:span(r[k] for r in group) for k in ('delta_S','delta_L','delta_S_plus_L','lower_violation','upper_violation')}))
    for pitch in (0,32):
        group=[r for r in paired if r['kind']=='delay' and r['observer']==1 and r['pitch_bytes']==pitch]
        for unit in ('cycles','ns'):
            issued=[r['bounds']['issue_to_ready_'+unit] for r in group]
            tails=[r['bounds']['ready_to_FIRST_'+unit] for r in group]
            ready_change_checks.append(dict(pitch_bytes=pitch,unit=unit,processes=len(group),
                finite_ready_change=sum(b['lower'] is not None and b['upper'] is not None for b in issued),
                positive_ready_lower=sum(b['lower'] is not None and b['lower']>0 for b in issued),
                finite_tail_change=sum(b['lower'] is not None and b['upper'] is not None for b in tails),
                tail_change_contains_zero=sum(b['lower'] is not None and b['upper'] is not None and b['lower']<=0<=b['upper'] for b in tails)))
    cases=[];coverage=[];timing={}
    for case,records in targets.items():
        if not records:continue
        values=list(records.values());first=values[0]
        cases.append(dict(case=case,observer=first['observer'],delay_ns=first['delay_ns'],pitch_bytes=first['pitch_bytes'],
            intervals=summarize([r['intervals'] for r in values]),sms=sorted({r['sm'] for r in values}),
            ranks={event:summarize([{k:r['source_group'][event][k] for k in ('rank_min','rank_max','target_minus_peer_median_ns')}
                for r in values]) for event in ('entry','producer','FIRST')}))
        base=name(0,first['delay_ns'],first['pitch_bytes'])
        timing[case]=dict(dual_us=span(r['elapsed_us'] for r in values),observer0_plain_us=span(plain[base].values()),
            dual_relative_to_observer0_plain=span(records[t]['elapsed_us']/plain[base][t]-1 for t in range(10)))
        if first['observer']:
            bounded=[r['ready_bounds'] for r in values]
            widths={unit:[r['width'][unit] for r in bounded if r['width'][unit] is not None] for unit in ('cycles','ns')}
            coverage.append(dict(case=case,processes=len(values),status_counts=dict(Counter(r['status'] for r in bounded)),
                tests=span(r['tests'] for r in bounded),width={u:dict(finite_processes=len(w),values=w,summary=span(w) if w else None) for u,w in widths.items()},
                bounds={k:_ready_bound_summary([r['bounds'][k] for r in bounded]) for k in bounded[0]['bounds']}))
    enabled=[r for r in processes if r['observer']]
    total_coverage=dict(processes=len(enabled),status_counts=dict(Counter(r['ready_bounds']['status'] for r in enabled)))
    shared_tail={}
    for unit in ('cycles','ns'):
        bounds=[r['ready_bounds']['bounds']['ready_to_FIRST_'+unit] for r in enabled]
        lowers=[b['lower'] for b in bounds if b['lower'] is not None]
        uppers=[b['upper'] for b in bounds if b['upper'] is not None]
        lo=max(lowers) if lowers else None;hi=min(uppers) if uppers else None
        shared_tail[unit]=dict(processes=len(bounds),lower_constraints=len(lowers),upper_constraints=len(uppers),
            lower=lo,upper=hi,nonempty=lo<=hi if lo is not None and hi is not None else None)
    late=[r for r in enabled if r['ready_observer']['success'] and r['observer_position']['exit_minus_FIRST_cycles']>0]
    late_false=[r for r in late if r['observer_position']['last_false_pre_minus_FIRST_cycles'] is not None]
    observer_positions=dict(enabled_processes=len(enabled),start_before_issue=sum(r['observer_position']['start_minus_issue_cycles']<0 for r in enabled),
        tests=span(r['ready_observer']['tests'] for r in enabled),
        all_start_minus_issue_cycles=span(r['observer_position']['start_minus_issue_cycles'] for r in enabled),
        late_processes=len(late),late_with_false_endpoint=len(late_false),
        late_intervals_crossing_FIRST=sum(r['observer_position']['last_false_pre_minus_FIRST_cycles']<0<r['observer_position']['exit_minus_FIRST_cycles'] for r in late_false),
        late_exit_minus_FIRST_cycles=span(r['observer_position']['exit_minus_FIRST_cycles'] for r in late) if late else None,
        late_last_false_pre_minus_FIRST_cycles=span(r['observer_position']['last_false_pre_minus_FIRST_cycles'] for r in late_false) if late_false else None,
        late_last_false_pre_to_exit_cycles=span(r['observer_position']['last_false_pre_to_exit_cycles'] for r in late_false) if late_false else None,
        interpretation='Positions relative to FIRST do not change the protocol generation boundary: '+boundary+'.')
    if release_bound:
        observer_positions.update(release_pre_minus_FIRST_cycles=span(r['observer_position']['release_pre_minus_FIRST_cycles'] for r in enabled),
            exit_minus_release_pre_cycles=span(r['observer_position']['exit_minus_release_pre_cycles'] for r in enabled))
    previous=root.parent/previous_name;references={}
    if read_json(previous/'environment.json')['gpu']!=read_json(root/'environment.json')['gpu']:
        raise ValueError('ready reference comparison changed card')
    for row in rows.values():
        if row['set']!='reference':continue
        case=row['id'];old=[]
        for item in select_attempts(previous,case,('plain',))[0]:
            old.append(replay(previous,item['record'],row)['elapsed_us'])
            for path in (previous/item['path'],previous/item['record']['raw']):hashes[str(path)]=common.sha(path)
        if len(old)!=10:raise ValueError('ready historical reference incomplete')
        references[case]=dict(current=span(plain[case].values()),previous=span(old),median_change=statistics.median(plain[case].values())/statistics.median(old)-1)
    for run in (root,previous):
        for filename in ('cases.json','static_setup.json','environment.json','source_hashes.json'):
            path=run/filename;hashes[str(path)]=common.sha(path)
    for path in [root/'build/resources.json',root/'build/sass_hashes.json',*sorted(root.glob('correctness-*.jsonl'))]:hashes[str(path)]=common.sha(path)
    origins=[dict(trial=t,by_case={r['case']:r['b_address']['b_buffer_ptr'] for r in processes if r['trial']==t}) for t in range(10)]
    report=dict(scope='CTA55 first-output first-generation readiness bounds from an independent read-only stage0 observer. '
        'Every enabled process, including censored results, remains in the denominator and interval statistics.',
        root=str(root),protocol=protocol,generation_boundary=boundary,
        environment=read_json(root/'environment.json'),contexts=contexts,processes=processes,case_summaries=cases,
        total_coverage=total_coverage,coverage=coverage,observer_positions=observer_positions,
        shared_tail_feasible_intersection=shared_tail,
        paired_processes=paired,paired_summaries=paired_summaries,
        delay_bound_checks=ready_change_checks,startup_shift_processes=shift_processes,startup_shift_summaries=shift_summaries,
        interaction_processes=interactions,interaction_summaries=interaction_summaries,
        observer_interaction_change_processes=observer_interactions,observer_interaction_change=observer_interaction_summary,
        background_processes=background,background_summaries=background_summaries,timing=timing,references=references,
        placement_maps=placements,work_sha256=hashlib.sha256(json.dumps(work).encode()).hexdigest(),
        virtual_origins=dict(per_trial=origins,same_origin_trials=[r['trial'] for r in origins if len(set(r['by_case'].values()))==1]),
        resources=read_json(root/'build/resources.json'),sass_hashes=read_json(root/'build/sass_hashes.json'),
        failed_attempts=failures,input_sha256=hashes,analyzer_sha256=common.sha(Path(__file__)),
        interpretation=['Same-binary observer1-minus-observer0 S/L/complete-window and pitch interactions precede any readiness attribution.',
            f'Only success with exit_c<={boundary} binds the first stage generation. {late_status} and poll_limit are censored, not errors to retry. '
            'The old FIRST-bound protocol is unchanged and receives no release-bound backfill.',
            'Last-false pre-test timestamp is a lower bound; successful exit timestamp is an upper bound. First-test success supplies only an upper bound.',
            'Censored bounds are unbounded (JSON null), retained in median bounds. Finite widths are separately labeled by coverage.',
            'Clock cycles are primary. Adjacent per-role globaltimer reads are quantized at 32ns; negative ns residuals are retained, not rejected.',
            'The measured event is test_wait visibility to the observer thread, not a direct DRAM/TMA completion timestamp.',
            'Condition changes subtract intervals conservatively; no midpoint or per-process constant assigns uncertainty to ready versus consumer progress.',
            'issue-to-ready and ready-to-FIRST share a boundary and are complementary. Their independent extrema must not be added as independent errors.',
            'The intersection of ready-to-FIRST intervals is only a feasible common-tail set, not an identified hardware latency. '
            'A single quantized ns value is not used as a constant or frequency conversion.',
            'The startup-shift check is only a necessary bound: if all later ready roots translate with the producer and edge costs stay fixed, '
            'extra first-consumer delay delta_S>=0 requires -delta_S<=delta_L<=0. Satisfying it does not identify the cause or fit a model.',
            'Observer0 still includes the existing dual and first-supply instrumentation. Fixed edge costs must be assessed separately in cycles and ns; '
            'frequency/service-rate changes and independent-process variability can change which bound holds in each domain.',
            'Independent processes remain repetitions; CTA backgrounds and common virtual origins do not create additional samples or establish common physical pages.',
            f'{previous_job} plain references only contextualize cross-batch card/binary changes; they are not same-binary observer error.'])
    if release_bound:
        report['interpretation'].append('Observer1 enables both polling and the first-release marker; observer0 enables neither. '
            'Their same-binary contrast measures combined observer perturbation. ready-to-FIRST remains signed and is never clipped to zero.')
    output.mkdir(parents=True,exist_ok=True)
    (output/(artifact+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def cell(v):return '未知' if v is None else f'{v["median"]:.1f} [{v["process_min"]:.1f},{v["process_max"]:.1f}]'
    def bound_cell(v):
        a='未知' if v['median_lower'] is None else f'{v["median_lower"]:.1f}'
        b='未知' if v['median_upper'] is None else f'{v["median_upper"]:.1f}'
        return f'[{a}, {b}]'
    lines=['# CTA55独立ready观察','',
        f'先检查同一binary观察开关的扰动；所有40个启用进程保留。仅成功且exit_c≤{boundary}绑定第一代。'
        '区间按端点传播，不取中点；条件中位界包含censored的未知区间。','',
        '| 观察器1−0 | ΔS ns | ΔL ns | Δ(entry→MAIN_END) ns | 整调用变化 |','|---|---:|---:|---:|---:|']
    for r in paired_summaries:
        if r['kind']!='observer':continue
        lines.append(f'| delay{r["delay_ns"]}, B+{r["pitch_bytes"]} B | '+' | '.join(cell(r['delta'][k]) for k in
            ('S_ns','L_ns','entry_to_MAIN_END_ns'))+f' | {100*r["elapsed"]["ratio"]["median"]:+.2f}% |')
    lines+=['','| 条件 | 全部进程与状态 | 有限宽度 cycles | issue→ready 中位界 cycles | ready→FIRST 中位界 cycles |',
        '|---|---|---:|---:|---:|']
    for r in coverage:
        width=r['width']['cycles'];w=cell(width['summary']) if width['summary'] else '未知'
        lines.append(f'| {r["case"]} | {r["processes"]}: {r["status_counts"]} | {width["finite_processes"]}次: {w} | '
            f'{bound_cell(r["bounds"]["issue_to_ready_cycles"])} | {bound_cell(r["bounds"]["ready_to_FIRST_cycles"])} |')
    p=observer_positions
    lines+=['',f'{p["start_before_issue"]}/{p["enabled_processes"]}个观察器在issue_end前已启动，'
        f'start−issue_end为{cell(p["all_start_minus_issue_cycles"])} cycles；每进程测试次数为{cell(p["tests"])}。']
    if late:
        lines+=[f'{p["late_intervals_crossing_FIRST"]}/{p["late_processes"]}个退出晚于FIRST的最后失败到成功退出区间跨过FIRST。'
            f'last_false_pre−FIRST为{cell(p["late_last_false_pre_minus_FIRST_cycles"])} cycles，'
            f'exit−FIRST为{cell(p["late_exit_minus_FIRST_cycles"])} cycles，'
            f'区间宽度为{cell(p["late_last_false_pre_to_exit_cycles"])} cycles。'
            f'这里只描述相对FIRST的位置，第一代绑定仍以本协议的{boundary}为准；旧批次不回填。']
    if release_bound:
        lines+=[f'release_pre−FIRST为{cell(p["release_pre_minus_FIRST_cycles"])} cycles，'
            f'exit−release_pre为{cell(p["exit_minus_release_pre_cycles"])} cycles。'
            '本批开关比较轮询与释放前标记的总扰动；ready→FIRST保留有符号区间，不强裁为0。']
    lines+=['','| 观察器1中直接延迟512−0 | Δ(issue→ready) 中位界 cycles | Δ(ready→FIRST) 中位界 cycles | Δ(issue→FIRST) cycles |',
        '|---|---:|---:|---:|']
    for r in paired_summaries:
        if r['kind']!='delay' or not r['observer']:continue
        lines.append(f'| B+{r["pitch_bytes"]} B | {bound_cell(r["bounds"]["issue_to_ready_cycles"])} | '
            f'{bound_cell(r["bounds"]["ready_to_FIRST_cycles"])} | {cell(r["delta"]["issue_to_FIRST_cycles"])} |')
    lines+=['','| 观察器1中直接延迟512−0 | Δ(issue→ready) 中位界 ns | Δ(ready→FIRST) 中位界 ns |',
        '|---|---:|---:|']
    for r in paired_summaries:
        if r['kind']!='delay' or not r['observer']:continue
        lines.append(f'| B+{r["pitch_bytes"]} B | {bound_cell(r["bounds"]["issue_to_ready_ns"])} | '
            f'{bound_cell(r["bounds"]["ready_to_FIRST_ns"])} |')
    lines+=['','| 逐进程延迟配对 | ready变化有限 | ready增长下界>0 | 尾段变化有限且含0 |','|---|---:|---:|---:|']
    for r in ready_change_checks:
        lines.append(f'| B+{r["pitch_bytes"]} B, {r["unit"]} | {r["finite_ready_change"]}/{r["processes"]} | '
            f'{r["positive_ready_lower"]}/{r["processes"]} | {r["tail_change_contains_zero"]}/{r["processes"]} |')
    lines+=['','| 等待转移必要界：−ΔS≤ΔL≤0 | 满足/ΔS≥0配对数 | 最大下界违背 | 最大上界违背 | Δ(S+L) |',
        '|---|---:|---:|---:|---:|']
    for r in shift_summaries:
        lines.append(f'| o{r["observer"]}, B+{r["pitch_bytes"]} B, {r["unit"]} | {r["satisfies_count"]}/{r["premise_count"]} | '
            f'{r["lower_violation"]["process_max"]:.1f} | {r["upper_violation"]["process_max"]:.1f} | {cell(r["delta_S_plus_L"])} |')
    lines+=['','该必要界假设后续ready根仅随producer共同平移、边成本不变，只首consumer增加ΔS。'
        '满足不能证明等待转移，更不能说明稳态供给变快；违反也须结合域内固定成本、频率/服务率变化和进程波动。'
        'o0仍有原dual与first-supply打点；cycles与ns分列，不跨域套用同一组固定边成本。']
    tail=shared_tail['cycles'];tail_ns=shared_tail['ns']
    lines+=[f'ready→FIRST共有区间为[{tail["lower"]}, {tail["upper"]}] cycles，来自'
        f'{tail["lower_constraints"]}/{tail["processes"]}个下界与{tail["upper_constraints"]}/{tail["processes"]}个上界约束。'
        '它只是小尾项的可行交集，不是唯一物理延迟，也未被采用为模型常数。'
        f'ns交集[{tail_ns["lower"]}, {tail_ns["upper"]}]受32ns量化限制，不作定值。']
    lines+=['','| delay×pitch交互 | 观察器0 ns | 观察器1 ns |','|---|---:|---:|']
    for k in ('S_ns','L_ns','S_plus_L_ns','entry_to_MAIN_END_ns'):
        lines.append(f'| {k} | '+' | '.join(cell(r['delta'][k]) for r in interaction_summaries)+' |')
    lines+=['','| 交互的观察器1−0变化 | ns中位与十进程范围 |','|---|---:|']
    for k in ('S_ns','L_ns','S_plus_L_ns','entry_to_MAIN_END_ns'):
        lines.append(f'| {k} | {cell(observer_interaction_summary[k])} |')
    lines+=['',f'| plain参照 | 本批中位 µs | {previous_job}中位 µs | 变化 |','|---|---:|---:|---:|']
    for case,r in references.items():
        lines.append(f'| {case} | {r["current"]["median"]:.3f} | {r["previous"]["median"]:.3f} | {100*r["median_change"]:+.2f}% |')
    lines+=['','cycles为主要区间；ns受32ns量化与相邻读表影响，可出现有符号的边界差。'
        f'首测成功只有上界，poll_limit/{late_status}不生成ready上界，也不删除或重测。'
        'JSON保留所有逐进程端点、覆盖、区间传播、同B17伙伴/其他130CTA背景、SM/work/VA和原始文件SHA。','']
    (output/(artifact+'.md')).write_text('\n'.join(lines))
    return report


def _stage_handoff_windows(cta):
    """Same-CTA endpoints give bounds, not the unrecorded sender MMA return."""
    result={}
    for unit,key in (('cycles','tiles'),('ns','tiles_ns')):
        before,after=cta[key][:2]
        offset=after[0]-before[1]
        result[unit]=dict(first1_minus_main0=offset,
            mma0_to_first1_lower_bound=max(0,offset))
    return result


def stage_handoff_sample_diagnostics(root, output):
    """Replay the paired progress/handoff run without fitting any model."""
    from analyze_r13_sm import select_attempts
    import v06_run as common

    root,output=Path(root),Path(output)
    common.verify(root)
    rows={r['id']:r for r in read_json(root/'cases.json')}
    setups={r['case']:r['setup'] for r in read_json(root/'static_setup.json')}
    controls=[r for r in rows.values() if r.get('set')!='reference']
    if len(controls)!=2 or {setups[r['id']]['stages'] for r in controls}!={4,6}:
        raise ValueError('handoff samples require the two stage4/6 controls')
    gpu={setups[r['id']]['gpu_uuid'] for r in rows.values()}
    if len(gpu)!=1:raise ValueError('handoff run spans different GPUs')
    handoff_fields=('sender_to_math_return','math_wait_total','math_return_to_full_begin',
        'first_full_wait','sender_to_first','receiver_late_entry','after_sender_and_wait_entry',
        'math_wait_begin_minus_sender')
    processes=[];calls={};failures=[];hashes={};contexts={}
    def span(values):
        values=list(values)
        return dict(median=statistics.median(values),trial_min=min(values),trial_max=max(values))
    def remember(run,item):
        for path in (run/item['path'],run/item['record']['raw']):hashes[str(path)]=common.sha(path)
    for row in rows.values():
        name=row['id'];reference=row.get('set')=='reference';stage=setups[name]['stages']
        variants=('plain',) if reference else ('plain','progress','stamped')
        selected,failed,_=select_attempts(root,name,variants)
        contexts[name]=dict(row=row,setup=setups[name])
        for item in failed:
            failures.append(dict(case=name,set=row.get('set'),**item));remember(root,item)
        grouped={v:{} for v in variants}
        for item in selected:
            record=item['record'];variant=record['variant'];obs=replay(root,record,row)
            remember(root,item)
            entry=dict(case=name,set=row.get('set'),stages=stage,variant=variant,trial=record['trial'],
                elapsed_us=obs['elapsed_us'],source=record['raw'],record_path=item['path'],
                host_start_ns=record['host_start_ns'],host_stop_ns=record['host_stop_ns'])
            if variant!='plain':
                progress=obs['progress'];clocks=progress['clocks']
                intervals=_stage_progress_intervals(dict(clocks=clocks))
                entry.update(cta=progress['cta'],sm=progress['sm'],work=progress['work'],
                    output=progress['output'],clocks=clocks,intervals=intervals)
                if variant=='stamped':
                    if 'sender_work' not in progress:raise ValueError('stamped run lacks handoff endpoints')
                    intervals.update({k:progress['intervals'][k] for k in handoff_fields[:-1]})
                    intervals['math_wait_begin_minus_sender']=clocks['math_wait_begin']-clocks['sender_mma_return']
                    for endpoint in ('commit1','commit16','commit63','main_end'):
                        intervals['sender_to_'+endpoint]=clocks[endpoint]-clocks['sender_mma_return']
                    entry['sender_work']=progress['sender_work']
                    entry['handoff_closure_cycles']=(intervals['sender_to_math_return']+
                        intervals['math_return_to_full_begin']+intervals['first_full_wait']-intervals['sender_to_first'])
                    entry['arrival_split_closure_cycles']=(intervals['receiver_late_entry']+
                        intervals['after_sender_and_wait_entry']-intervals['sender_to_math_return'])
                    if entry['handoff_closure_cycles'] or entry['arrival_split_closure_cycles']:
                        raise ValueError('handoff same-call intervals do not close')
                elif 'sender_work' in progress:
                    raise ValueError('progress control unexpectedly contains handoff markers')
            grouped[variant][record['trial']]=entry;processes.append(entry)
        if any(set(v)!=set(range(10)) for v in grouped.values()):
            raise ValueError(name+': ten matched independent processes required')
        calls[name]=grouped
    control_by_stage={setups[r['id']]['stages']:r['id'] for r in controls}
    common_fields=tuple(calls[control_by_stage[4]]['progress'][0]['intervals'])
    timing=[];perturbation=[];stage_pairs=[]
    for stage in (4,6):
        name=control_by_stage[stage];group=calls[name]
        pairs=[]
        for trial in range(10):
            p,h=group['progress'][trial],group['stamped'][trial]
            if (p['cta'],p['work'],p['output'])!=(h['cta'],h['work'],h['output']):
                raise ValueError('handoff/progress pair changed selected work')
            pairs.append(dict(trial=trial,progress_sm=p['sm'],handoff_sm=h['sm'],
                delta_cycles={k:h['intervals'][k]-p['intervals'][k] for k in common_fields},
                whole_delta_us=h['elapsed_us']-p['elapsed_us'],
                whole_relative=h['elapsed_us']/p['elapsed_us']-1))
        perturbation.append(dict(stages=stage,processes=pairs,
            delta_cycles={k:span(p['delta_cycles'][k] for p in pairs) for k in common_fields},
            whole_delta_us=span(p['whole_delta_us'] for p in pairs),
            whole_relative=span(p['whole_relative'] for p in pairs)))
        timing.append(dict(stages=stage,
            elapsed_us={v:span(p['elapsed_us'] for p in records.values()) for v,records in group.items()},
            relative_to_plain={v:span(group[v][t]['elapsed_us']/group['plain'][t]['elapsed_us']-1
                for t in range(10)) for v in ('progress','stamped')},
            intervals={v:{k:span(p['intervals'][k] for p in group[v].values())
                for k in group[v][0]['intervals']} for v in ('progress','stamped')}))
    for variant in ('plain','progress','stamped'):
        a,b=(calls[control_by_stage[s]][variant] for s in (6,4));paired=[]
        for trial in range(10):
            x,y=a[trial],b[trial]
            entry=dict(trial=trial,whole_delta_us=y['elapsed_us']-x['elapsed_us'],
                whole_relative=y['elapsed_us']/x['elapsed_us']-1)
            if variant!='plain':
                if (x['cta'],x['work'],x['output'])!=(y['cta'],y['work'],y['output']):
                    raise ValueError('stage pair changed selected receiver work')
                if variant=='stamped' and x['sender_work']!=y['sender_work']:
                    raise ValueError('stage pair changed selected sender work')
                entry.update(stage6_sm=x['sm'],stage4_sm=y['sm'],
                    delta_cycles={k:y['intervals'][k]-x['intervals'][k] for k in x['intervals']})
            paired.append(entry)
        summary=dict(variant=variant,processes=paired,whole_delta_us=span(p['whole_delta_us'] for p in paired),
            whole_relative=span(p['whole_relative'] for p in paired))
        if variant!='plain':summary['delta_cycles']={k:span(p['delta_cycles'][k] for p in paired)
            for k in paired[0]['delta_cycles']}
        stage_pairs.append(summary)
    references=[];old=root.parent/'20261010-R13-stage-progress'
    old_rows={r['id']:r for r in read_json(old/'cases.json')}
    old_setups={r['case']:r['setup'] for r in read_json(old/'static_setup.json')}
    for row in rows.values():
        if row.get('set')!='reference':continue
        name=row['id'];selected,failed,_=select_attempts(old,name,('plain',));prior=[]
        for item in selected:
            record=item['record'];obs=replay(old,record,old_rows[name]);remember(old,item)
            prior.append(dict(trial=record['trial'],elapsed_us=obs['elapsed_us'],source=record['raw']))
        if len(prior)!=10:raise ValueError('old plain reference is incomplete: '+name)
        current=span(p['elapsed_us'] for p in calls[name]['plain'].values())
        previous=span(p['elapsed_us'] for p in prior)
        references.append(dict(case=name,current_us=current,previous_us=previous,previous_processes=prior,
            previous_root=str(old),previous_failed_attempts=failed,
            same_gpu=setups[name]['gpu_uuid']==old_setups[name]['gpu_uuid'],
            median_relative=current['median']/previous['median']-1))
    ref_end=max(p['host_stop_ns'] for p in processes if p['set']=='reference')
    ctrl_begin=min(p['host_start_ns'] for p in processes if p['set']!='reference')
    for path in (root/'cases.json',root/'static_setup.json',root/'source_hashes.json',
            root/'build/resources.json',root/'build/sass_hashes.json',root/'build/binary_hashes.json',
            old/'cases.json',old/'static_setup.json',Path(__file__),Path(__file__).with_name('analyze_r18.py')):
        hashes[str(path)]=common.sha(path)
    report=dict(scope='Same-call CTA0 sender output0 and receiver output1 on one SM; development measurement, no fit.',
        root=str(root),gpu=next(iter(gpu)),contexts=contexts,processes=processes,timing=timing,
        handoff_minus_progress=perturbation,stage4_minus_stage6=stage_pairs,references=references,
        reference_order=dict(last_reference_stop_ns=ref_end,first_control_start_ns=ctrl_begin,
            references_completed_first=ref_end<=ctrl_begin),failed_attempts=failures,
        resources=read_json(root/'build/resources.json'),input_sha256=hashes,
        limitations=['stamped is the new handoff observer; progress is the unchanged progress observer.',
            'sender_to_math_return combines remaining sender arrivals, receiver arrival and barrier recovery.',
            'math_wait_begin can precede or follow sender_mma_return; the signed difference is retained.',
            'first_full_wait includes try_wait, wait and observation instructions; it is not bare TMA latency.',
            'New timestamps and register lifetimes can change the observed path; local and whole-call perturbations are separate.',
            'Trial pairing compares independent processes. Cross-job reference medians are background, not observer error.',
            'No parameters are fitted and the prior B-stage-handoff counterexamples are not overwritten.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'B-stage-handoff-samples.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def cell(value):return f'{value["median"]:.2f} [{value["trial_min"]:.2f}, {value["trial_max"]:.2f}]'
    labels={'sender_to_math_return':'sender mma返回→math等待返回',
        'math_return_to_full_begin':'math等待返回→首full等待前','first_full_wait':'首full等待前→FIRST',
        'sender_to_first':'sender mma返回→FIRST','math_wait_total':'math等待前→返回',
        'math_wait_begin_minus_sender':'math等待前−sender返回（有符号）',
        'receiver_late_entry':'receiver较晚进入','after_sender_and_wait_entry':'两者均到达→math等待返回'}
    ts={x['stages']:x for x in timing};delta=next(p['delta_cycles'] for p in stage_pairs if p['variant']=='stamped')
    lines=['# 同调用 math交接与首full等待','',
        'CTA0第一输出的thread128记录mma返回，第二输出的thread256记录math等待前后、首full等待前及原progress端点。'
        '以下cycle差都在同调用、同SM内计算，再按trial比较两个stage；方括号为十个独立进程的范围。','',
        '| 新handoff窗口 | stage4 cycle | stage6 cycle | stage4−stage6 cycle |',
        '|---|---:|---:|---:|']
    for k,label in labels.items():
        lines.append(f'| {label} | {cell(ts[4]["intervals"]["stamped"][k])} | {cell(ts[6]["intervals"]["stamped"][k])} | {cell(delta[k])} |')
    lines+=['','三段sender→math返回、math返回→full等待前、full等待前→FIRST逐调用精确闭合；'
        'receiver较晚进入与双方均到达后的剩余窗口也逐调用闭合。各列独立取中位，不能要求中位数相加。','',
        '| 从sender mma返回起的累计推进 | stage4 cycle | stage6 cycle | stage4−stage6 cycle |',
        '|---|---:|---:|---:|']
    for endpoint in ('commit1','commit16','commit63','main_end'):
        k='sender_to_'+endpoint
        lines.append(f'| →{endpoint} | {cell(ts[4]["intervals"]["stamped"][k])} | {cell(ts[6]["intervals"]["stamped"][k])} | {cell(delta[k])} |')
    lines+=['','这些累计窗口直接相减同一次调用的原端点，用来检查stage差是否只来自FIRST边界移动；'
        '没有拼接旧progress的价格，也没有将独立中位数相加。','',
        '| handoff−progress同名局部窗口 | stage4 cycle | stage6 cycle |','|---|---:|---:|']
    ps={x['stages']:x for x in perturbation}
    for k in ('first_to_commit1','average_commit1_to16','average_commit17_to63','commit_to_wait1_16',
            'loop_advance_16_to17','last_commit_to_mma_return','mma_return_to_main_end','L'):
        lines.append(f'| {k} | {cell(ps[4]["delta_cycles"][k])} | {cell(ps[6]["delta_cycles"][k])} |')
    lines+=['','| 整调用对照 | stage4相对变化 | stage6相对变化 |','|---|---:|---:|']
    def percent(v):return cell({k:100*x for k,x in v.items()})+'%'
    for variant in ('progress','stamped'):
        lines.append(f'| {variant}/plain | {percent(ts[4]["relative_to_plain"][variant])} | {percent(ts[6]["relative_to_plain"][variant])} |')
    lines.append(f'| handoff/progress | {percent(ps[4]["whole_relative"])} | {percent(ps[6]["whole_relative"])} |')
    lines+=['','| stage4−stage6 | 局部L差 cycle | 整调用差µs | 整调用相对变化 |','|---|---:|---:|---:|']
    for pair in stage_pairs:
        local=cell(pair['delta_cycles']['L']) if 'delta_cycles' in pair else '—'
        lines.append(f'| {pair["variant"]} | {local} | {cell(pair["whole_delta_us"])} | {percent(pair["whole_relative"])} |')
    lines+=['','| plain参照 | 本批µs | 739578 µs | 中位变化 |','|---|---:|---:|---:|']
    for r in references:lines.append(f'| {r["case"]} | {r["current_us"]["median"]:.3f} | {r["previous_us"]["median"]:.3f} | {100*r["median_relative"]:+.2f}% |')
    lines+=['',f'两参照的成功进程先于控制采样完成：{report["reference_order"]["references_completed_first"]}。'
        '跨批参照只提供背景，不用于估计本批插桩误差。JSON保留原端点、SM、工作坐标、raw定位、失败尝试和源码身份。','',
        'sender到math返回包含发送方剩余arrival、接收方进入和恢复；没有记录128次arrival完成时刻。'
        '首full窗口包含等待及观察指令，不直接当作TMA服务。寄存器暂存避免即时全局写，仍可能改变等待路径；'
        '先看同批局部扰动，再解释窗口差。本报告不拟参数，不改变旧递推反例或冻结预测。','']
    (output/'B-stage-handoff-samples.md').write_text('\n'.join(lines))
    return report


def epilogue_order_diagnostics(root, output):
    """Same-binary E0 defer control; same-call windows, independent-process pairs."""
    import gzip
    from analyze_r13_sm import select_attempts, validate_setup
    import v06_run as common

    root,output=Path(root),Path(output)
    common.verify(root)
    rows={r['id']:r for r in read_json(root/'cases.json')}
    setups={r['case']:r['setup'] for r in read_json(root/'static_setup.json')}
    references=[r for r in rows.values() if r.get('set')=='reference']
    controls=[r for r in rows.values() if r.get('set')!='reference']
    keys={(setups[r['id']]['stages'],r.get('epilogue_defer')):r['id'] for r in controls}
    if len(references)!=2 or len(controls)!=4 or set(keys)!={(s,d) for s in (4,6) for d in (0,1)}:
        raise ValueError('epilogue-order run requires two references and four stage/defer controls')
    for row in controls:
        if (row['config'],row['m'],row['n'],row['k'])!=('cfg_b',2816,1536,4096):
            raise ValueError('epilogue-order geometry differs from declared pair')
    gpu={s['gpu_uuid'] for s in setups.values()}
    if len(gpu)!=1:raise ValueError('epilogue-order run spans GPUs')
    binaries=read_json(root/'build/binary_hashes.json')
    processes=[];failures=[];hashes={};calls={};contexts={}

    def span(values):
        values=list(values)
        return dict(median=statistics.median(values),trial_min=min(values),trial_max=max(values))

    def remember(run,item):
        raw=run/item['record']['raw']
        if common.sha(raw)!=item['record']['raw_sha256']:
            raise ValueError('epilogue raw identity changed: '+str(raw))
        for path in (run/item['path'],raw):hashes[str(path)]=common.sha(path)

    for row in rows.values():
        name=row['id'];reference=row.get('set')=='reference'
        variants=('plain',) if reference else ('plain','progress')
        selected,failed,_=select_attempts(root,name,variants)
        contexts[name]=dict(row=row,setup=setups[name])
        for item in failed:
            remember(root,item);failures.append(dict(case=name,**item))
        grouped={v:{} for v in variants}
        for item in selected:
            record=item['record'];variant=record['variant'];remember(root,item)
            obs=replay(root,record,row)
            with gzip.open(root/record['raw'],'rt') as stream:
                events={e['event']:e for line in stream if line.strip() for e in [json.loads(line)]}
            setup,call=events['setup'],events['call']
            validate_setup(row,setups[name],setup)
            binary='build/'+common.binary_prefix(row)+'_'+variant
            if Path(record['command'][0]).name!=Path(binary).name or binary not in binaries:
                raise ValueError('epilogue sample binary differs from its configuration')
            warm=call['warmup_us']
            if call['warmup_calls']!=len(warm) or not warm:
                raise ValueError('epilogue warmup count mismatch')
            addresses={key:setup[key] for key in ('target_a_ptr','target_b_ptr')}
            if any(not value or call.get(key)!=value for key,value in addresses.items()):
                raise ValueError('epilogue target address changed within a process')
            entry=dict(case=name,set=row.get('set'),stages=setup['stages'],variant=variant,
                epilogue_defer=row.get('epilogue_defer',0),trial=record['trial'],
                elapsed_us=obs['elapsed_us'],source=record['raw'],source_sha256=record['raw_sha256'],
                record_path=item['path'],binary=binary,binary_sha256=binaries[binary],
                gpu_uuid=setup['gpu_uuid'],host_start_ns=record['host_start_ns'],host_stop_ns=record['host_stop_ns'],
                warmup_calls=len(warm),warmup_last5_cv=call['warmup_last5_cv'],
                warmup_us=warm,target_addresses=addresses)
            if variant=='progress':
                if setup['trace_version']!='r13-epilogue-order-v1':
                    raise ValueError('epilogue observation lacks its explicit protocol')
                p=obs['progress'];c=p['clocks']
                intervals=_stage_progress_intervals(dict(clocks=c))
                intervals.update({key:p['intervals'][key] for key in
                    ('sender_epilogue','epilogue_overlap_L','epilogue_overlap_early')})
                intervals['sender_epi_begin_minus_mma_return']=c['sender_epi_begin']-c['mma_return']
                if p['sender_work']!=[0,0] or p['work']!=[11,0]:
                    raise ValueError('epilogue treatment changed selected output coordinates')
                if row['epilogue_defer'] and intervals['sender_epi_begin_minus_mma_return']<0:
                    raise ValueError('deferred output began before receiver mma returned')
                entry.update(cta=p['cta'],sm=p['sm'],work=p['work'],sender_work=p['sender_work'],
                    output=p['output'],clocks=c,intervals=intervals)
            grouped[variant][record['trial']]=entry;processes.append(entry)
        if any(set(v)!=set(range(10)) for v in grouped.values()):
            raise ValueError(name+': ten independent processes per variant required')
        calls[name]=grouped

    timing=[];treatment=[];observer=[]
    for stage in (4,6):
        for defer in (0,1):
            name=keys[stage,defer];group=calls[name]
            timing.append(dict(stages=stage,epilogue_defer=defer,
                elapsed_us={v:span(p['elapsed_us'] for p in group[v].values()) for v in ('plain','progress')},
                intervals={k:span(p['intervals'][k] for p in group['progress'].values())
                    for k in group['progress'][0]['intervals']}))
            pairs=[]
            for trial in range(10):
                plain,progress=(group[v][trial] for v in ('plain','progress'))
                pairs.append(dict(trial=trial,whole_delta_us=progress['elapsed_us']-plain['elapsed_us'],
                    whole_relative=progress['elapsed_us']/plain['elapsed_us']-1,
                    source_plain=plain['source'],source_progress=progress['source'],
                    same_target_addresses=plain['target_addresses']==progress['target_addresses']))
            observer.append(dict(stages=stage,epilogue_defer=defer,processes=pairs,
                whole_delta_us=span(p['whole_delta_us'] for p in pairs),
                whole_relative=span(p['whole_relative'] for p in pairs)))
        for variant in ('plain','progress'):
            a,b=(calls[keys[stage,d]][variant] for d in (0,1));pairs=[]
            for trial in range(10):
                normal,deferred=a[trial],b[trial]
                if normal['binary_sha256']!=deferred['binary_sha256']:
                    raise ValueError('normal/deferred control does not share a binary')
                pair=dict(trial=trial,whole_delta_us=deferred['elapsed_us']-normal['elapsed_us'],
                    whole_relative=deferred['elapsed_us']/normal['elapsed_us']-1,
                    source_normal=normal['source'],source_deferred=deferred['source'],
                    same_target_addresses=normal['target_addresses']==deferred['target_addresses'],
                    warmup_calls_normal=normal['warmup_calls'],warmup_calls_deferred=deferred['warmup_calls'])
                if variant=='progress':
                    pair.update(normal_sm=normal['sm'],deferred_sm=deferred['sm'],
                        delta_cycles={k:deferred['intervals'][k]-normal['intervals'][k] for k in normal['intervals']})
                pairs.append(pair)
            summary=dict(stages=stage,variant=variant,processes=pairs,
                whole_delta_us=span(p['whole_delta_us'] for p in pairs),
                whole_relative=span(p['whole_relative'] for p in pairs))
            if variant=='progress':summary['delta_cycles']={k:span(p['delta_cycles'][k] for p in pairs)
                for k in pairs[0]['delta_cycles']}
            treatment.append(summary)

    old=root.parent/'20261010-R13-stage-handoff';common.verify(old)
    old_rows={r['id']:r for r in read_json(old/'cases.json')}
    old_setups={r['case']:r['setup'] for r in read_json(old/'static_setup.json')}
    reference_results=[]
    for row in references:
        name=row['id'];selected,failed,_=select_attempts(old,name,('plain',));prior=[]
        for item in selected:
            remember(old,item);record=item['record'];obs=replay(old,record,old_rows[name])
            prior.append(dict(trial=record['trial'],elapsed_us=obs['elapsed_us'],source=record['raw'],
                source_sha256=record['raw_sha256']))
        for item in failed:remember(old,item)
        if {p['trial'] for p in prior}!=set(range(10)):raise ValueError('739686 reference incomplete')
        current=span(p['elapsed_us'] for p in calls[name]['plain'].values())
        previous=span(p['elapsed_us'] for p in prior)
        reference_results.append(dict(case=name,current_us=current,previous_us=previous,previous_root=str(old),
            previous_processes=prior,previous_failed_attempts=failed,
            same_gpu=setups[name]['gpu_uuid']==old_setups[name]['gpu_uuid'],
            median_relative=current['median']/previous['median']-1))
    early_bounds=[]
    for defer in (0,1):
        t4=next(t for t in timing if (t['stages'],t['epilogue_defer'])==(4,defer))
        t6=next(t for t in timing if (t['stages'],t['epilogue_defer'])==(6,defer))
        c=t6['intervals']['commit17_to63']['median']/46
        # Same reduced/default-inheritance subclass as stage_reuse_diagnostics.
        # D46 >= 15*(Z+w)+c; D15 <= 5*(Z+w) in the stage4 stalled branch.
        upper=max(15*c,(t4['intervals']['commit17_to63']['median']-c)/3)
        early_bounds.append(dict(epilogue_defer=defer,effective_c=c,
            stage4_early_upper=upper,stage4_early_observed=t4['intervals']['commit1_to16']['median'],
            stage4_early_shortfall=t4['intervals']['commit1_to16']['median']-upper))
    correctness=[]
    for path in sorted((root/'correctness').glob('*.jsonl')):
        check=next(json.loads(line) for line in path.read_text().splitlines() if '"epilogue_order_check"' in line)
        if check['different'] or check['nonfinite'] or check['elements']!=2816*1536:
            raise ValueError('full output comparison failed')
        correctness.append(dict(source=str(path),**check));hashes[str(path)]=common.sha(path)
    if len(correctness)!=4:raise ValueError('four full-output comparisons required')
    ref_end=max(p['host_stop_ns'] for p in processes if p['set']=='reference')
    ctrl_begin=min(p['host_start_ns'] for p in processes if p['set']!='reference')
    for run in (root,old):
        for relative in ('cases.json','static_setup.json','source_hashes.json','build/binary_hashes.json',
                'build/resources.json','build/sass_hashes.json'):
            path=run/relative;hashes[str(path)]=common.sha(path)
    for path in (Path(__file__),Path(__file__).with_name('analyze_r18.py')):hashes[str(path)]=common.sha(path)
    report=dict(scope='CTA0 E0 order intervention after its mma_tail; no model fit or full-validation score.',
        root=str(root),gpu=next(iter(gpu)),contexts=contexts,processes=processes,timing=timing,
        deferred_minus_normal=treatment,progress_minus_plain=observer,references=reference_results,
        reference_order=dict(last_reference_stop_ns=ref_end,first_control_start_ns=ctrl_begin,
            references_completed_first=ref_end<=ctrl_begin),failed_attempts=failures,
        resources=read_json(root/'build/resources.json'),input_sha256=hashes,
        full_output_checks=correctness,full_inheritance_early_bounds=early_bounds,
        limitations=['Differences are formed within each call before pairing independent processes by trial.',
            'Only CTA0 E0 is delayed; other CTAs retain their output organization.',
            'Sender trace writes occur after its epilogue, which can still overlap receiver math in the normal arm.',
            'Moving E0 also changes its phase relative to other CTAs; its shorter window does not uniquely identify local-SM contention.',
            'The added barrier wait can alter scheduling; removing overlap does not isolate a unique physical resource.',
            'E0 end is the source-reuse/store-tail boundary, not proof of global D visibility.',
            'Virtual addresses are recorded per process; equal virtual addresses do not prove identical physical mappings.',
            'Cross-job reference drift is separate from same-job treatment and observation perturbation.',
            'No timing-derived parameters are fitted and frozen V09 remains unchanged.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'B-epilogue-order.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def cell(value):return f'{value["median"]:.2f} [{value["trial_min"]:.2f}, {value["trial_max"]:.2f}]'
    def percent(value):return cell({k:v*100 for k,v in value.items()})+'%'
    lines=['# 前一输出与下一主循环的同CTA顺序对照','',
        'cfg_b 2816×1536×4096，stage4/6各比较同一二进制的正常输出与CTA0首输出延期。'
        '延期等待放在首输出mma_tail之后，等待第二消费者mma返回。全部差值先在同调用内形成；'
        '处理差按trial配对独立进程。方括号为十个进程或十个配对的最小、最大值，不是置信区间。','',
        '| stage | 输出顺序 | C1→C16 cycle | C16→C17 cycle | C17→C63 cycle | L cycle | E0与早段重叠 cycle |',
        '|---|---|---:|---:|---:|---:|---:|']
    for item in timing:
        v=item['intervals'];fields=('commit1_to16','commit16_to17','commit17_to63','L','epilogue_overlap_early')
        lines.append(f'| {item["stages"]} | {"延期" if item["epilogue_defer"] else "正常"} | '+' | '.join(cell(v[k]) for k in fields)+' |')
    lines+=['','| 延期−正常 | early cycle | late cycle | L cycle | 整调用µs | 整调用相对变化 |',
        '|---|---:|---:|---:|---:|---:|']
    for item in treatment:
        v=item.get('delta_cycles');fields=('commit1_to16','commit17_to63','L')
        local=' | '.join(cell(v[k]) if v else '—' for k in fields)
        lines.append(f'| stage{item["stages"]} {item["variant"]} | {local} | {cell(item["whole_delta_us"])} | {percent(item["whole_relative"])} |')
    lines+=['','两种顺序各用自身两stage后段作条件诊断，沿用[B-stage-reuse](B-stage-reuse.md)'
        '的固定复用/默认继承子类及其全部前提；stage6后段计算受限只是有效参数假设。'
        'stage4早段上界为max(15c,(D46−c)/3)，不是任意初态或统计置信界。','',
        '| 顺序 | stage4早段条件上界 | 实测早段中位 | 仍少算cycle |','|---|---:|---:|---:|']
    for b in early_bounds:
        lines.append(f'| {"延期" if b["epilogue_defer"] else "正常"} | {b["stage4_early_upper"]:.2f} | '
            f'{b["stage4_early_observed"]:.2f} | {b["stage4_early_shortfall"]:.2f} |')
    lines+=['','| stage | 正常E0窗口中位 | 延期E0窗口中位 | 延期−正常的M1返回→MAIN中位差 |',
        '|---|---:|---:|---:|']
    for stage in (4,6):
        a,b=[next(t for t in timing if (t['stages'],t['epilogue_defer'])==(stage,d)) for d in (0,1)]
        pair=next(t for t in treatment if (t['stages'],t['variant'])==(stage,'progress'))
        lines.append(f'| {stage} | {a["intervals"]["sender_epilogue"]["median"]:.2f} | '
            f'{b["intervals"]["sender_epilogue"]["median"]:.2f} | '
            f'{pair["delta_cycles"]["mma_return_to_main_end"]["median"]:+.2f} |')
    lines+=['','延期排除了E0与C1→C16的重叠，但E0仍可与接收组mma_tail重叠；'
        'M1返回→MAIN的增长不能直接命名为硬件drain变化。E0变短同时改变了它与其他CTA的时间位置。'
        'sender记录在自身epilogue结束后才写出，正常组中这些写仍可能落在receiver主循环内；'
        '整调用观察扰动较小不能排除局部观察影响。四次同分配完整D比较均逐位相同，每次4325376个有效元素。','']
    lines+=['','源码核对没有发现跨输出slot错位：每个WG prologue只提交，不释放前WG尾槽；'
        'local1起wait1后释放前一输入槽，tail只释放自身最后一个槽。'
        '第二输出local S−1引用前输出tail，local j≥S才恢复内部S−1复用关系；当前递推已分别处理。'
        '仍未定量的是128线程empty-arrive完成到producer重发的路径、两次producer load之间的控制间隙，'
        '以及把两WG折叠成同一固定TC服务frontier的假设；源码不能证明这些时间跨早/晚段恒定。','']
    lines+=['','| progress相对plain的观察扰动 | 整调用µs | 整调用相对变化 |','|---|---:|---:|']
    for item in observer:
        lines.append(f'| stage{item["stages"]} {"延期" if item["epilogue_defer"] else "正常"} | {cell(item["whole_delta_us"])} | {percent(item["whole_relative"])} |')
    lines+=['','| plain参照 | 本批µs | 739686 µs | 中位变化 | 同GPU |','|---|---:|---:|---:|---|']
    for item in reference_results:
        lines.append(f'| {item["case"]} | {item["current_us"]["median"]:.3f} | {item["previous_us"]["median"]:.3f} | {100*item["median_relative"]:+.2f}% | {item["same_gpu"]} |')
    lines+=['',f'参照成功进程先于处理采样完成：{ref_end<=ctrl_begin}。'
        '每个延期progress进程都通过E0开始不早于同调用M1返回的检查。JSON保留原端点、工作坐标、SM、'
        'A/B虚拟地址、预热次数与CV、二进制和raw哈希、失败尝试及逐trial差值。','',
        '局部early或L缩短，只能支持输出并发或新等待改变执行组织参与了原偏差；'
        '不能直接命名为某种硬件端口冲突。若early保持不变，也要结合等待扰动解释，不能仅凭阴性结果排除全部竞争。'
        '本报告不拟合参数，不把跨批参照漂移当作本批因果差，不更新冻结成绩。','']
    (output/'B-epilogue-order.md').write_text('\n'.join(lines))
    return report


def stage_boundary_diagnostics(base, output):
    """Analytic single-gap exclusion and existing same-call refill bounds."""
    from supply_model import admission_stage_events,stage_reuse_window
    from analyze_r13_sm import select_attempts
    import v06_run as common
    base,output=Path(base),Path(output)
    source=output/'B-epilogue-order.json';observed=read_json(source)
    parameters=dict(producer_interval=50.,ready_latency=1548.,commit_lead=400.,
        completion_lag=100.,service_cycles=512.,issue_floor=0.,wait_overhead=24.,
        loop_advance=90.,return_overhead=40.,reuse_delay=100.,handoff_delay=134.)
    witnesses=[]
    for stage in (4,6):
        original=admission_stage_events(64,stage,**parameters)
        c=parameters['service_cycles'];m=stage-1
        Z=parameters['ready_latency']+parameters['reuse_delay']+c-parameters['commit_lead']
        extra=max(0,Z+parameters['wait_overhead']-m*c)
        for gap in (0.,512.,2048.,8192.,65536.):
            t=admission_stage_events(64,stage,**parameters,producer_boundary_delay=gap)
            C=t['commits'];before=original['commits']
            error=max(abs(C[64+j]-max(before[64+j],C[64]+j*c+((j-1)//m)*extra))
                      for j in range(1,64))
            early=C[80]-C[65];bound=stage_reuse_window(15,stage,c,Z,parameters['wait_overhead'])[1]
            if error>1e-8 or early>bound+1e-8:raise ValueError('single-gap analytic identity failed')
            witnesses.append(dict(stages=stage,producer_boundary_delay=gap,
                first_commit_shift=C[64]-before[64],first_entry_after_previous_mma=
                t['first_mma_or_commit'][64]-t['mma_returns'][0],early=early,
                early_upper=bound,late=C[127]-C[81],L=t['windows'][1],identity_max_error=error))

    # A counterexample to extending the narrow theorem, not a fitted predictor.
    wide_parameters=dict(producer_interval=306.75038136816534,ready_latency=1473.0148793166884,
        commit_lead=1821.3211425857085,completion_lag=354.6860660718234,
        issue_floor=0.,reuse_delay=16.594099396544753,handoff_delay=987.6127317827884,
        producer_boundary_delay=0.,service_cycles=23587/46,wait_overhead=43.,
        loop_advance=89.,return_overhead=10.)
    wide=[]
    for stage in (4,6):
        trace=admission_stage_events(64,stage,**wide_parameters)
        C=trace['commits'];W=trace['wait1_return'];B=trace['first_mma_or_commit']
        wide.append(dict(stages=stage,early=C[80]-C[65],late=C[127]-C[81],
            first_to_commit1=C[65]-B[64],commit_to_wait1_16=W[80]-C[80],
            last_commit_to_mma_return=trace['mma_returns'][1]-C[127]))
    h=89.;c=23587/46
    wide_limits=dict(scope='Necessary bounds for a deterministic central target, not statistical parameter confidence.',
        g_upper=(652-h)/2,q_lower=2*c-652,w_upper=43.,w_plus_r_upper=53.,
        lambda_plus_g_plus_w_upper=23587/23-h)
    root=base/'20261010-R13-stage-events';common.verify(root)
    rows=read_json(root/'cases.json');refill=[];hashes={str(source):common.sha(source)}
    for row in rows:
        if row.get('set')=='reference':continue
        selected,_,_=select_attempts(root,row['id'],('stamped',))
        for item in selected:
            record=item['record'];obs=replay(root,record,row)['sparse'];c=obs['clocks']
            # All 128 arrivals include the selected lane: completion cannot
            # precede its pre-arrive marker. Acquire return is after completion.
            lower=c['refill_issue_end']-c['refill_end']
            upper=c['refill_issue_end']-c['release_begin']
            if not (c['refill_begin']<=c['release_begin']<=c['refill_end']<=c['refill_issue_end']):
                raise ValueError('refill markers do not bracket the same release completion')
            for path in (root/item['path'],root/record['raw']):hashes[str(path)]=common.sha(path)
            refill.append(dict(case=row['id'],stages=row.get('stages',6),trial=record['trial'],
                cta=obs['cta'],sm=obs['sm'],work=obs['work'],clocks=c,source=record['raw'],
                record_path=item['path'],empty_completion_to_issue_bounds=[lower,upper],
                release_marker_to_acquire_return=c['refill_end']-c['release_begin']))
    if len(refill)!=20:raise ValueError('expected twenty existing sparse processes')
    summary=[]
    for stage in (4,6):
        chosen=[r for r in refill if r['stages']==stage]
        lower=[r['empty_completion_to_issue_bounds'][0] for r in chosen]
        upper=[r['empty_completion_to_issue_bounds'][1] for r in chosen]
        summary.append(dict(stages=stage,processes=len(chosen),
            median_bounds=[statistics.median(lower),statistics.median(upper)],
            constant_interval=[max(lower),min(upper)],
            release_marker_to_acquire_return_median=statistics.median(r['release_marker_to_acquire_return'] for r in chosen)))
    interval=[max(r['empty_completion_to_issue_bounds'][0] for r in refill),
              min(r['empty_completion_to_issue_bounds'][1] for r in refill)]
    for path in (Path(__file__),Path(__file__).with_name('supply_model.py'),root/'source_hashes.json',root/'cases.json'):
        hashes[str(path)]=common.sha(path)
    report=dict(scope='No GPU sampling, no fitting, no formal-prediction change.',
        single_gap=dict(assumptions=['p<=c; 0<=g<=c-q; q+lambda<=c; q>=w+h; q>=w+r+delta; nonnegative costs.',
            'Default first output and kt=64; no observed_second_entry cut.',
            'Only the producer issue carry before the second load receives a nonnegative gap; slot state is preserved.'],
            equations=dict(internal='C_j=max(C_(j-1)+c,U_(64+j-S)+Z)',
                impulse='Cgap_j=max(Cbase_j,Cgap_0+j*c+floor((j-1)/(S-1))*R), j>=1',
                R='max(0,Z+w-(S-1)*c)',early='Cgap_16-Cgap_1 <= 15*c+15/(S-1)*R, S=4 or 6'),
            witness_parameters=parameters,witnesses=witnesses,
            observation_bounds=observed['full_inheritance_early_bounds'],
            decision='Any nonnegative single producer-boundary gap preserves the early upper bound in this subclass; do not fit it to the residual.'),
        wider_branch_counterexample=dict(parameters=wide_parameters,windows=wide,necessary_limits=wide_limits,
            decision='Matches the two late windows but fails measured first and retirement windows; not a candidate and not an infeasibility proof for the wider family.'),
        refill=dict(root=str(root),processes=refill,summary=summary,common_constant_interval=interval,
            common_constant_not_excluded=interval[0]<=interval[1],
            interpretation='The interval is empty-completion-to-pair-issue. It is not a timestamp for completion or the model wait1-to-issue reuse_delay.'),
        limitations=['These existing sparse markers are at local16 release/local16+S refill, not a time series over early and late iterations.',
            'A nonempty intersection only means these intervals do not reject a constant; it is not a measured common latency.',
            'The 739556 sparse protocol is kept separate from 739717 progress; their endpoints and costs are not spliced.',
            'The analytic exclusion does not cover time-varying readiness, release completion, service lead or arbitrary initial states.'],
        input_sha256=hashes)
    (output/'B-stage-boundary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 输出边界间隙与释放完成的可辨识范围','',
        '本轮只作离线推导，未拟合新参数。先排除一次性producer边界间隙能解释剩余早段的说法，再单独检查已有同调用释放端点。','',
        '## 任意单一producer间隙仍不能突破早段上界','',
        '沿用[B-stage-reuse](B-stage-reuse.md)的固定服务子类：p≤c、0≤g≤c−q、q+λ≤c、q≥w+h、q≥w+r+δ，'
        '默认第一输出初始化，无观测入口切割。令m=S−1、Z=ℓ+a+c−q、T=Z+w、R=max(0,T−mc)。','',
        '在第二输出首issue的carry上加任意Δ≥0，槽位释放约束不变。后续I[j−1]+p分支贡献最多C[j−1]+p，'
        '被C[j−1]+c支配，所以间隙只通过增大的C0传播。local m引用的是前WG tail释放，不引用本输出C0；'
        '第一条内部长边是local1→localS。故j≥1时的最大传播路径为v_j=jc+floor((j−1)/m)R，'
        '并有Cgap_j=max(Cbase_j,Cgap_0+v_j)。','',
        '原默认继承状态满足Cbase_16≤Cbase_1+B，B=15c+(15/m)R。'
        '因此Cgap_16≤max(Cbase_1+B,Cgap_0+c+B)=Cgap_1+B。'
        'stage4/6的m=3/5均整除15，原早段上界对任意Δ仍成立。'
        '前输出末槽tail不晚于虚拟普通释放、且最早只影响local m，因而不能在local1之前改变上述锚点。','',
        '| 演示stage | producer间隙cycle | 首commit平移 | C1→C16 | 早段上界 | C17→C63 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in witnesses:
        lines.append(f'| {r["stages"]} | {r["producer_boundary_delay"]:.0f} | {r["first_commit_shift"]:.0f} | '
            f'{r["early"]:.0f} | {r["early_upper"]:.0f} | {r["late"]:.0f} |')
    lines+=['','表中只是固定参数的数值见证，不是拟合或证明替代。解析界覆盖任意非负间隙。'
        '739717正常/延后两组自身后段给出的stage4早段缺口仍分别约2242/1511 cycle；'
        '不能靠给producer的输出边界添加一个常数消除。','',
        '## 旧端点只能约束释放完成后的重新发出请求','',
        '739556的同调用、同slot记录在producer已进入acquire后打到所选consumer的release前戳。'
        'cfg_b的empty barrier需128线程arrive，因此真正完成U不早于所选线程release前戳，'
        '又不晚于producer acquire返回。下一对TMA请求发出后的标记为I，故'
        'I−acquire_end ≤ I−U ≤ I−release_begin。','',
        '| stage | 十进程的中位区间cycle | 常数与十个区间的交集 | release前戳→acquire返回中位 |',
        '|---|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["stages"]} | {r["median_bounds"]} | {r["constant_interval"]} | '
            f'{r["release_marker_to_acquire_return_median"]:.1f} |')
    lines+=['',f'两stage的20个区间共同交集为{interval} cycle。'
        '非空只表示当前端点没有排除共同常数，不是测得该常数。'
        '它也不是模型a：模型的乐观起点是所选lane的wait1返回，实际empty完成还需其它线程到达；'
        '本记录没有同一次local17的wait1端点，不能拼接另一批progress来补。','',
        '这些点只覆盖local16释放及其重用，不能判断早段到后段是否改变。'
        '保留“实际ready演化、释放完成、组内服务/观察路径仍未分离”的结论；'
        '不把新的时间常数或状态项直接接入预测。','']
    lines+=['## 不能把子类反证扩大成整个固定参数递推失败','',
        '源码没有保证g≤c−q或q+λ≤c。以下宽分支参数点几乎匹配两stage的late，却超过原子类early界；'
        '它同时严重违反已观测首段与退休端点，只作限定定理范围的反例，不作为候选。JSON保存全部参数。','',
        '| 反例stage | C17→C63 | C1→C16 | FIRST→C1 | W16−C16 | M−C63 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in wide:
        lines.append(f'| {r["stages"]} | {r["late"]:.2f} | {r["early"]:.2f} | '
            f'{r["first_to_commit1"]:.2f} | {r["commit_to_wait1_16"]:.2f} | {r["last_commit_to_mma_return"]:.2f} |')
    lines+=['','联立端点时，还要保留实际等待与基础软件项的区别：W16−C16=max(0,D15−C16)+w，'
        '所以43只能给w≤43；M−C63给w+r≤53，不能先令w=43、r=10再声称排除宽族。'
        '首窗口还要求FIRST→C1≥2g+h和≥2c−q；普通二步推进给'
        'C_i−C_(i−2)≥λ+w+h+g。以h=89、c=23587/46及中央目标量计算，'
        f'g≤{wide_limits["g_upper"]:.3f}，q≥{wide_limits["q_lower"]:.3f}，'
        f'λ+g+w≤{wide_limits["lambda_plus_g_plus_w_upper"]:.3f}。这些是确定性中央候选约束，不是统计置信界。','',
        '现有端点排除了上表这类只匹配late的见证，却尚未给出覆盖所有宽分支的不可行性证明。'
        '下一步先联立这些约束检查宽递推，不因几次数值搜索没找到解就宣称必须新增状态或申请GPU。','']
    (output/'B-stage-boundary.md').write_text('\n'.join(lines))
    return report


def stage_conditioned_diagnostics(output):
    """Observed-entry cuts and internal necessary bounds; no parameter fitting."""
    from supply_model import admission_stage_events
    import v06_run as common
    output=Path(output);source=output/'B-stage-handoff-samples.json'
    data=read_json(source)
    rows=[r for r in data['processes'] if r['variant']=='stamped' and r['set']=='ctrl']
    stage6=[r for r in rows if r['stages']==6]
    q_min=max(1024-r['intervals']['first_to_commit1'] for r in stage6)
    tail=min(r['intervals']['last_commit_to_main_end'] for r in stage6)
    # The old latency and one formerly flat producer interval are witnesses,
    # not new calibration. q takes the stage6 necessary bound; tail-q is an
    # explicit active-branch subclass, checked in each replay below.
    parameters=dict(producer_interval=50.,ready_latency=2257.5,
        commit_lead=float(q_min),completion_lag=tail-q_min)
    c_upper=min(r['clocks']['commit63']-r['clocks']['commit17'] for r in stage6)/45
    records=[]
    for row in rows:
        c=row['clocks'];v=row['intervals'];stage=row['stages']
        w=v['commit_to_wait1_16'];h=v['loop_advance_16_to17'];r=v['last_commit_to_mma_return']-w
        H=c['first_wait_begin']-c['sender_mma_return'];B=c['first_mma']-c['sender_mma_return']
        protocol=dict(wait_overhead=w,loop_advance=h,return_overhead=r)
        free=admission_stage_events(64,stage,**parameters,**protocol)
        violation=free['ready'][64]-(free['mma_returns'][0]+B)
        diff=c['commit63']-c['commit17']
        entry=dict(case=row['case'],trial=row['trial'],stages=stage,source=row['source'],
            protocol=protocol,observed_entry=[H,B],entry_ready_minus_B=violation,
            entry_feasible=violation<=1e-8,observed=v,
            late_commit_difference=diff,
            ready_combination_upper=diff/(46//(stage-1))-w,
            ready_plus_reuse_upper=diff/(46//(stage-1)),
            late_stall_lower_512=max(0,diff-47*512),
            late_stall_lower_common_c=max(0,diff-47*c_upper))
        if entry['entry_feasible']:
            t=admission_stage_events(64,stage,**parameters,**protocol,observed_second_entry=(H,B))
            C,W,A,E,Bs,Hs=(t[k] for k in ('commits','wait1_return','ready','service_frontier',
                'first_mma_or_commit','consumer_available'))
            clocks=dict(first_mma=Bs[64],commit1=C[65],full_wait_begin16=Hs[80],commit16=C[80],
                wait1_end16=W[80],full_wait_begin17=Hs[81],commit17=C[81],commit63=C[127],
                mma_return=t['mma_returns'][1],main_end=t['main_ends'][1])
            entry.update(predicted=_stage_progress_intervals(dict(clocks=clocks)),
                entry_return_slack=Bs[64]-max(A[64],Hs[64]),
                last_commit_lead_branch=abs(C[127]-(E[127]-q_min))<1e-8,
                last_MAIN_completion_branch=abs(t['main_ends'][1]-t['completed'][127])<1e-8,
                extra_wait16=W[80]-C[80]-w)
        records.append(entry)
    def span(v):
        v=list(v);return dict(median=statistics.median(v),minimum=min(v),maximum=max(v))
    summaries={}
    for stage in (4,6):
        selected=[r for r in records if r['stages']==stage]
        summaries[stage]=dict(processes=len(selected),entry_feasible=sum(r['entry_feasible'] for r in selected),
            maximum_entry_violation=max(r['entry_ready_minus_B'] for r in selected),
            ready_combination_upper=min(r['ready_combination_upper'] for r in selected),
            ready_plus_reuse_upper=min(r['ready_plus_reuse_upper'] for r in selected),
            late_stall_lower_512=span(r['late_stall_lower_512'] for r in selected),
            late_stall_lower_common_c=span(r['late_stall_lower_common_c'] for r in selected))
    # This post-measurement pair is a deterministic variation diagnostic, not
    # a heldout score or a reason by itself to add state to a median predictor.
    pair=[next(r for r in records if r['stages']==6 and r['trial']==t) for t in (1,9)]
    if pair[0]['protocol']!=pair[1]['protocol']:raise ValueError('variation pair changed software protocol')
    input_delta=abs(pair[1]['observed_entry'][1]-pair[0]['observed_entry'][1])
    observed_delta=abs(pair[1]['observed']['commit1_to16']-pair[0]['observed']['commit1_to16'])
    residual=max(0,(observed_delta-input_delta)/2)
    variation=dict(trials=[1,9],protocol=pair[0]['protocol'],
        first_entry_difference=input_delta,early_interval_difference=observed_delta,
        minimum_maximum_per_process_residual=residual,
        residual_fraction_of_median_L=residual/statistics.median(r['intervals']['L'] for r in stage6))
    report=dict(scope='Development conditional replay. Observed H/B are diagnostic inputs only; no fit or new prediction.',
        witness_parameters=parameters,stage6_necessary_q_lower=q_min,
        common_service_upper=c_upper,records=records,summaries=summaries,variation_pair=variation,
        input_sha256={str(p):common.sha(p) for p in (source,Path(__file__),Path(__file__).with_name('supply_model.py'))},
        limitations=['The first output is simulated with the second-output w/h/r: an unmeasured cross-output hypothesis.',
            'The observed commit-to-wait16 interval is used as w conditionally; extra modeled completion waiting is reported.',
            'lambda=tail-q applies only to the checked last lead/completion branch subclass, not a universal equality.',
            'Entry conflicts reject a tuple and inherited-state hypothesis, not every constant-ready model.',
            'Internal slot/stall bounds do not use first-output state; they remain conditional on the recurrence structure.',
            'Model service-frontier stalls are not measured TC idle cycles and do not identify TMA as the sole cause.',
            'The process-pair variation bound does not establish failure of a median predictor or justify another GPU matrix.'])
    (output/'B-stage-conditioned.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 固定观测入口后的条件检验','',
        '只切第二输出入口：H=M0+(实测first_wait_begin−sender标记)，B=M0+(实测FIRST−sender标记)。'
        'producer、E和slot状态沿用递推；A>B时明确拒绝，不提前ready迁就观测。观测不进入正式预测。','',
        '第一输出仍使用第二输出的w/h/r模拟，这是未测得的跨输出假设；条件切割没有恢复真实初态。'
        '本次不做拟合，见证取p=50、ℓ=2257.5、q='+str(q_min)+f'、λ={tail-q_min}、g=a=0。'
        'q来自本批stage6局部必要界，λ拆分仅在末端分支成立时解释。','',
        '| stage | 入口因果可行 | 最大 A−B cycle | 后段Z上界 | 不定w时ℓ+a上界 |', '|---|---:|---:|---:|---:|']
    for stage in (6,4):
        r=summaries[stage]
        lines.append(f'| {stage} | {r["entry_feasible"]}/{r["processes"]} | {r["maximum_entry_violation"]:.2f} | {r["ready_combination_upper"]:.2f} | {r["ready_plus_reuse_upper"]:.2f} |')
    lines+=['','Z=ℓ+max(g,512−q)+a。后段slot复用给C_i≥C_(i−S+1)+Z+w，'
        '46间隔至少容纳floor(46/(S−1))次跳跃，其他间隔仅取非负。这个必要界只用第二输出内部，'
        '不依赖入口或第一输出初态；进一步只用w/g/a非负，可得不定软件价格的ℓ+a上界。'
        '因此旧ℓ=2257.5连stage4的这个宽松上界也违反，不能靠入口或继承初态修复；'
        'stage4上界只作诊断，不能用它反选stage6参数。','',
        '| stage6局部量 | 合法条件重放中位 | 观测中位 |','|---|---:|---:|']
    valid=[r for r in records if r['stages']==6 and r['entry_feasible']]
    for f in ('first_to_commit1','average_commit1_to16','average_commit17_to63','full_wait_to_commit16','last_commit_to_main_end','L'):
        lines.append(f'| {f} | {statistics.median(r["predicted"][f] for r in valid):.2f} | '
            f'{statistics.median(r["observed"][f] for r in valid):.2f} |')
    lines+=['','入口冲突的stage4进程不计算预测误差；JSON记录逐调用违反量、返回余量、末端分支及额外wait16，'
        '不能把强制入口后产生的非因果轨迹作为成绩。','',
        '## 后段必须出现内部停顿','',
        '沿用当前递推的0≤g≤c以及q/λ/w/h非负约束。普通位置有B_i≥max(B_(i−1),E_(i−2))=E_(i−1)−c，所以c≤E_i−B_i≤2c。'
        '当前递推中g/q固定，d_i=E_i−C_i=min(E_i−B_i−g,q)，两个后段端点的d差最多c。'
        '因此46步的模型内部停顿Σmax(0,B_i−E_(i−1))至少为max(0,C63−C17−47c)。'
        '这不需要q上限、λ=tail−q、第一输出初态或w/h的具体取值。','',
        f'同理C63−C17≥45c，用stage6自身得到共同c≤{c_upper:.4f}。'
        '允许c在这个范围内变化，stage4的正下界仍然存在：','',
        '| stage | c=512时下界，中位 [范围] | 仅用stage6约束c时下界，中位 [范围] |',
        '|---|---:|---:|']
    for stage in (4,6):
        a=summaries[stage]['late_stall_lower_512'];b=summaries[stage]['late_stall_lower_common_c']
        lines.append(f'| {stage} | {a["median"]:.2f} [{a["minimum"]:.2f},{a["maximum"]:.2f}] | '
            f'{b["median"]:.2f} [{b["minimum"]:.2f},{b["maximum"]:.2f}] |')
    lines+=['','这些是解释当前局部端点所需的模型内部停顿，不是硬件TC空闲实测，也不能单独归因于TMA带宽。'
        '需要继续检验ready、循环推进与有限slot的组合关系；不能只移动入口或末端领先量来消除后段差。','',
        '## 逐调用波动的有限结论','',
        f'stage6 trial1/9的(w,h,r)相同，强制B只相差{input_delta} cycle；对共同常量及相同第一输出状态，'
        f'max-plus递推使后续任意区间变化至多{input_delta}，实测C1→16却差{observed_delta}。'
        f'每调用至少需{residual:.1f} cycle量级的最大残差，约为本批stage6 L中位的'
        f'{100*variation["residual_fraction_of_median_L"]:.2f}%。'
        '它只拒绝逐调用的精确共同常量解释；不能据此宣称中位预测失败，或为这约1%的波动继续扩大状态模型/打点。','']
    (output/'B-stage-conditioned.md').write_text('\n'.join(lines))
    return report


def stage_reuse_diagnostics(output):
    """Joint median development of the identifiable internal reuse combination."""
    from supply_model import stage_reuse_window
    import v06_run as common
    output=Path(output);source=output/'B-stage-handoff-samples.json';data=read_json(source)
    summaries={}
    fields=('commit1_to16','commit16_to17','commit17_to63','commit_to_wait1_16','L')
    for variant in ('stamped','progress'):
        summaries[variant]={}
        for stage in (4,6):
            rows=[r for r in data['processes'] if r['variant']==variant and r['stages']==stage]
            summaries[variant][stage]={f:statistics.median(r['intervals'][f] for r in rows) for f in fields}
    calibration=summaries['stamped'];c=calibration[6]['commit17_to63']/46
    constraints=[]
    for stage in (4,6):
        r=calibration[stage];m=stage-1;w=r['commit_to_wait1_16']
        excess=r['commit17_to63']-46*c
        if abs(excess)<1e-8:
            lower,upper=0.,m*c-w
        elif excess>0:
            lower=m*c-w+excess/((46+m-1)//m)
            upper=m*c-w+excess/(46//m)
        else:raise ValueError('median late window is below the proposed compute floor')
        early_upper=r['commit1_to16']/(15//m)-w
        constraints.append(dict(stages=stage,late_Z_interval=[max(0.,lower),upper],
            early_Z_upper=early_upper,late_excess=excess))
    interval=[max(r['late_Z_interval'][0] for r in constraints),
        min(min(r['late_Z_interval'][1],r['early_Z_upper']) for r in constraints)]
    if interval[0]>interval[1]:raise ValueError('joint median reuse interval is empty')
    comparisons=[]
    def distance(value,limits):return max(limits[0]-value,value-limits[1],0.)
    for variant in ('stamped','progress'):
        for stage in (4,6):
            r=summaries[variant][stage];w=r['commit_to_wait1_16'];m=stage-1
            late=[stage_reuse_window(46,stage,c,interval[0],w)[0],
                  stage_reuse_window(46,stage,c,interval[1],w)[1]]
            adjacent=[c,c+max(0.,interval[1]+w-m*c)]
            # This bound uses the complete default initial/return state and
            # q>=w+r+handoff_delay; it is not valid for an arbitrary entry cut.
            early=stage_reuse_window(15,stage,c,interval[1],w)[1]
            comparisons.append(dict(variant=variant,stages=stage,late_envelope=late,
                late_observed=r['commit17_to63'],late_distance=distance(r['commit17_to63'],late),
                adjacent_envelope=adjacent,adjacent_observed=r['commit16_to17'],
                adjacent_distance=distance(r['commit16_to17'],adjacent),
                full_inheritance_early_upper=early,early_observed=r['commit1_to16'],
                early_shortfall=max(0.,r['commit1_to16']-early),L_observed=r['L']))
    # Recalibrate the older observer independently as a diagnostic. This is
    # not a cross-protocol price and is not used to select the primary interval.
    old=summaries['progress'];old_c=old[6]['commit17_to63']/46
    old_T_upper=(old[4]['commit17_to63']-old_c)/15
    old_check=dict(service_cycles=old_c,stage4_T_upper=old_T_upper,
        stage4_early_upper=5*old_T_upper,
        stage4_early_observed=old[4]['commit1_to16'],
        stage4_early_shortfall=old[4]['commit1_to16']-5*old_T_upper)
    report=dict(scope='Both stamped stages are development calibration medians; progress is an observer comparison. No validation score.',
        calibration_fields=['stage6 median C17-to-C63 / 46 sets an effective compute-limited c',
            'both-stage median C17-to-C63 finite windows constrain Z',
            'both-stage C1-to-C16 supply a one-sided path bound, not a steady-state equality'],
        service_cycles=c,Z_interval=interval,constraints=constraints,condition_medians=summaries,
        comparisons=comparisons,independent_progress_check=old_check,
        equations=dict(internal='C_j=max(C_(j-1)+c,C_(j-S+1)+Z+w), j>=S; Z=ell+a+c-q',
            period='m=S-1; R=max(0,Z+w-m*c); C_(j+m)-C_j=m*c+R for j>=S-1',
            finite_window='n*c+floor(n/m)*R <= C_(j+n)-C_j <= n*c+ceil(n/m)*R',
            full_inheritance_early='C16-C1 <= (15/m)*max(m*c,Z+w), stage4/6 only'),
        assumptions=['p<=c, 0<=g<=c-q, q+lambda<=c, q>=w+h, and nonnegative times.',
            'w/h and the reuse combination are constant in the internal window; sampled marker paths can violate this.',
            'The full-inheritance early bound additionally needs q>=w+r+handoff_delay, the default first output and no observed entry cut.',
            'The final previous-output slot is released by tail; the internal recurrence cannot be applied at local j=S-1.',
            'The c/Z interval describes representative condition medians, not bare hardware constants or a statistical confidence interval.',
            'Satisfying the internal late window does not reconstruct the first S steps or prove a complete-L predictor.'],
        decision='Do not adopt the complete constant-reuse subclass: its stage4 early upper bound is below the observed median. '
            'Keep the late-window combination as a bounded development constraint; do not fit separate early/stage prices.',
        input_sha256={str(p):common.sha(p) for p in (source,Path(__file__),Path(__file__).with_name('supply_model.py'))})
    (output/'B-stage-reuse.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 共同复用组合：中位开发区间与启动缺口','',
        '本轮明确把739686新stamped的两stage同列开发校准；原stage6单独标定的失败不改。'
        '使用各条件独立进程的窗口中位量，不将不同进程或不同窗口拼成一次实际调用。','',
        '在p≤c、g≤c−q、q+λ≤c、q≥w+h的子类中，普通位置C恒取E−q、W=C+w，'
        '递推化为C_j=max(C_(j−1)+c,C_(j−S+1)+Z+w)，Z=ℓ+a+c−q。'
        '只从输出内j≥S使用；j=S−1复用的是前输出末slot，不能按普通释放代换。','',
        '令m=S−1、R=max(0,Z+w−mc)，内部每m个间隔合计mc+R；未知初态只改变周期相位。'
        'n个间隔的有限界为nc+floor(n/m)R到nc+ceil(n/m)R，保留了余段，不用渐近平均替代。','',
        f'把stage6后段视为计算受限，名义c={c:.6f}。两个后段窗口及早段单侧路径界允许'
        f'共同Z∈[{interval[0]:.6f},{interval[1]:.6f}] cycle。'
        '这个区间不是逐调用置信域，也没有分别识别ℓ、a、q。','',
        '| 协议 / stage | 后段观测 / 候选包络 | 后段包络外距离 | 相邻C16→17观测 / 候选包络 |',
        '|---|---:|---:|---:|']
    for r in comparisons:
        lo,hi=r['late_envelope'];a,b=r['adjacent_envelope']
        lines.append(f'| {r["variant"]} / {r["stages"]} | {r["late_observed"]:.2f} / [{lo:.2f},{hi:.2f}] | '
            f'{r["late_distance"]:.2f} | {r["adjacent_observed"]:.2f} / [{a:.2f},{b:.2f}] |')
    lines+=['','stage6新观察器相邻窗口仍比c多约19cycle，原progress约1cycle；'
        '固定内部软件项不能精确代表被选中打点的每次迭代。旧progress只作协议比较，没有用于选择c/Z。','',
        '## 完整继承状态仍不足','',
        '再限制q≥w+r+δ且沿默认首输出初始化，首H不会阻塞继承E。前输出末slot的tail释放早于普通释放，'
        '只能提前后续提交；它最早影响第二输出local S−1，不影响stage4/6的local C1。'
        '所以C1与虚拟持续序列的锚点相同，C16不会更晚：C16−C1≤(15/m)max(mc,Z+w)。'
        '这不是给任意未知初态或observed_second_entry切割的上界。','',
        '| 新stamped | C1→16观测 | 完整子类上界 | 至少缺少 |','|---|---:|---:|---:|']
    for r in comparisons:
        if r['variant']=='stamped':lines.append(f'| stage{r["stages"]} | {r["early_observed"]:.2f} | '
            f'{r["full_inheritance_early_upper"]:.2f} | {r["early_shortfall"]:.2f} |')
    lines+=['',f'原progress另用自身两stage后段作独立诊断，stage4早段上界为{old_check["stage4_early_upper"]:.2f}，'
        f'对观测{old_check["stage4_early_observed"]:.2f}仍缺{old_check["stage4_early_shortfall"]:.2f} cycle。'
        '缺口不要求新增handoff打点存在；但原progress本身的非均匀观察路径仍未被排除，'
        '不能只因2079比19大就把所有观察扰动排除。',
        '完整子类不进入预测，也不为early另加常数；后段共同代价仅保留为受限的开发结果。'
        '下一步区分未建模的输入/继承状态与前一输出并发路径，先复用现有同调用输出对照；'
        '本轮没有新的GPU条件、完整L成绩或冻结验证。','']
    (output/'B-stage-reuse.md').write_text('\n'.join(lines))
    return report


def stage_handoff_diagnostics(base,output):
    """Expose ready/handoff ambiguity without choosing parameters on stage4."""
    from analyze_r13_sm import select_attempts
    from supply_model import admission_stage_events
    import v06_run as common
    output=Path(output);root=Path(base)/'20261010-R09-R13-R18-failure-controls'
    rows={r['id']:r for r in read_json(root/'cases.json')}
    progress=read_json(output/'B-stage-progress.json')
    cal=next(r['intervals'] for r in progress['timing'] if r['stages']==6)
    # Conditional algebra: continuous TC, lead branch, ready inputs, and the
    # measured second-output return interval also holding for the first output.
    delta=1024-cal['last_commit_to_mma_return']['median']-cal['first_to_commit1']['median']
    cases=[];hashes={};processes=[];failed_records=[]
    def span(v):return dict(median=statistics.median(v),trial_min=min(v),trial_max=max(v))
    for k in (1024,4096):
        for stage in (4,6):
            name=f'r13_stage_k{k}_s{stage}';row=rows[name];values=[]
            selected,failed,_=select_attempts(root,name,('dual',))
            failed_records.extend(dict(case=name,path=r['path']) for r in failed)
            for item in selected:
                record=item['record'];obs=replay(root,record,row)
                cta=next(c for c in obs['ctas'] if c['cta']==0)
                v=dict(case=name,trial=record['trial'],cta=0,sm=cta['sm'],work=cta['work'],
                    source=record['raw'],tiles=cta['tiles'],tiles_ns=cta['tiles_ns'],
                    **_stage_handoff_windows(cta))
                values.append(v);processes.append(v)
                for path in (root/item['path'],root/record['raw']):hashes[str(path)]=common.sha(path)
            if len(values)!=10:raise ValueError('handoff diagnosis requires ten valid independent processes')
            cases.append(dict(case=name,processes=len(values),**{unit:{field:span([v[unit][field] for v in values])
                for field in values[0][unit]} for unit in ('cycles','ns')}))
    protocol=dict(wait_overhead=cal['commit_to_wait1_16']['median'],
        loop_advance=cal['loop_advance_16_to17']['median'],
        return_overhead=cal['last_commit_to_mma_return']['median']-cal['commit_to_wait1_16']['median'])
    examples=[]
    # These are counterexamples to identification, not a fit or a selection
    # among candidates. 1858 is the independent source-response upper bound;
    # 700 is an illustrative feasible smaller value, not an estimated latency.
    for latency,handoff in ((2257.5,0.),(700.,delta),(1858.,delta)):
        comparisons={}
        for stage in (6,4):
            t=admission_stage_events(64,stage,50.,latency,407.,93.,handoff_delay=handoff,**protocol)
            c=t['commits'];b=t['first_mma_or_commit']
            comparisons[stage]=dict(first_to_commit1=c[65]-b[64],
                average_commit1_to16=(c[80]-c[65])/15,
                average_commit17_to63=(c[127]-c[81])/46,L=t['windows'][1],
                ready_minus_consumer_available=t['ready'][64]-t['consumer_available'][64],
                mma0_to_first1=b[64]-t['mma_returns'][0])
        examples.append(dict(ready_latency=latency,handoff_delay=handoff,comparisons=comparisons))
    for path in (output/'B-stage-progress.json',output/'B-stage-events.json',root/'cases.json',
            Path(__file__),Path(__file__).with_name('supply_model.py')):hashes[str(path)]=common.sha(path)
    source_root=Path(base)/'20261010-R13-stage-progress'
    for relative in ('source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp',
            'source/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp','build/cfg_b_stamped.sass'):
        path=source_root/relative;hashes[str(path)]=common.sha(path)
    report=dict(scope='CTA0, first-to-second output, same invocation and SM. Development diagnosis only.',
        processes=processes,cases=cases,failed_records=failed_records,input_sha256=hashes,
        conditional_handoff_delta=delta,examples=examples,
        conclusion='Stage6 local observables do not distinguish source readiness from consumer availability. '
            'The old zero-handoff fit does not identify ready latency. No example is adopted or selected on stage4.',
        missing_endpoints=['sender first-output mma return', 'receiver second-output math-order wait return',
            'receiver second-output first full-wait entry'],
        limitations=['FIRST1-MAIN0 is a lower bound on FIRST1-MMA0, not a measured handoff interval.',
            'The two order-barrier depths independently order math and epilogue; EPI_PERMIT0 does not imply MMA1 has returned.',
            'The conditional delta also assumes the second-output 40-cycle return interval transfers to the first output.',
            'The sensitivity input includes sender arrivals and receiver progress; it is not a bare barrier latency.',
            'Examples retain the stage6 early-interval residual; identical model observables are not a successful calibration.',
            'Finite-window rejection of the old tuples remains valid, but it cannot identify a queued-ready mechanism.'])
    (output/'B-stage-handoff.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 交接与ready延迟的不可辨识性','',
        '前一WG的mma返回、128线程完成order-barrier到达、后一WG恢复、首full等待返回是不同事件。'
        '现有H_next=M把后几段设为零，可能让ready延迟吸收缺失时间。math和epilogue使用两个独立depth；'
        'EPI_PERMIT不证明后一输出的math已完成，不能据此拼出非阻塞入口。','',
        '| cfg_b条件 | FIRST1−MAIN0 cycle，中位 [范围] | 同端点ns，中位 [范围] |',
        '|---|---:|---:|']
    for r in cases:
        a=r['cycles']['first1_minus_main0'];b=r['ns']['first1_minus_main0']
        lines.append(f'| {r["case"]} | {a["median"]} [{a["trial_min"]}, {a["trial_max"]}] | '
            f'{b["median"]} [{b["trial_min"]}, {b["trial_max"]}] |')
    lines+=['','每行先保留十个独立进程；这是CTA0同调用跨WG、同SM的时间差。'
        'MAIN0晚于MMA0，因此这些差只给FIRST1−MMA0下界，负值说明重叠，不能当作负交接价格。','',
        f'条件代数：若连续TC和commit领先分支有效、首数据已齐，且前一输出也满足M=C_last+40，'
        f'则FIRST→commit1=1024−40−δ。δ={delta}能解释787.5，但这是解释量而非实测；还要求δ≤q−40。','',
        '| ℓ / δ cycle | stage6 FIRST→C1 / C1→16均值 / C17→63均值 / L | stage4 L |',
        '|---|---|---:|']
    for r in examples:
        a=r['comparisons'][6];fields=('first_to_commit1','average_commit1_to16','average_commit17_to63','L')
        lines.append(f'| {r["ready_latency"]} / {r["handoff_delay"]} | '+
            ' / '.join(f'{a[k]:.2f}' for k in fields)+f' | {r["comparisons"][4]["L"]:.2f} |')
    lines+=['','三组均固定p=50、q=407、λ=93、g=0及原w/h/r。700是演示值，1858来自独立供给上界；'
        '不以stage4选择ℓ。它们给出相同stage6四个观测量，却给出不同stage4响应，证明当前校准不能分开ready与交接。'
        '三组对stage6早段的512预测仍低于522.27实测，不能称为已通过。','',
        '下一次最小补测应在同一次调用记录前输出mma返回、后输出math等待前后、首full等待前后，'
        '并保留局部commit进度。若增长主要在math恢复前，先解释交接段，不能直接计作ready等待；若math恢复后仍等首full，'
        '才继续检验ready服务。全部区间仍含软件与观察开销；需与原progress和平常计时比较扰动。','']
    (output/'B-stage-handoff.md').write_text('\n'.join(lines))
    return report


def admission_stage_diagnostics(output):
    """Local phase calibration and constrained transfer; never refit whole L."""
    from supply_model import admission_stage_events
    import v06_run as common
    output=Path(output)
    progress_path=output/'B-stage-progress.json';sparse_path=output/'B-stage-events.json'
    progress,sparse=read_json(progress_path),read_json(sparse_path)
    if progress['gpu']!=sparse['gpu']:raise ValueError('admission local cohorts changed GPU')
    observed={t['stages']:t['intervals'] for t in progress['timing']}
    cal=observed[6]
    protocol=dict(service_cycles=512.,wait_overhead=cal['commit_to_wait1_16']['median'],
        loop_advance=cal['loop_advance_16_to17']['median'],
        return_overhead=cal['last_commit_to_mma_return']['median']-cal['commit_to_wait1_16']['median'])
    tail=cal['last_commit_to_main_end']['median']
    fields=('first_to_commit1','average_commit1_to16','average_commit17_to63')
    targets=np.array([cal[k]['median'] for k in fields])
    def evaluate(parameters,stages,issue_floor=0.):
        p,latency,q=parameters
        trace=admission_stage_events(64,stages,p,latency,q,tail-q,
            issue_floor=issue_floor,**protocol)
        C,H,W,B=(trace[k] for k in ('commits','consumer_available','wait1_return','first_mma_or_commit'))
        clocks=dict(first_mma=B[64],commit1=C[65],full_wait_begin16=H[80],commit16=C[80],
            wait1_end16=W[80],full_wait_begin17=H[81],commit17=C[81],commit63=C[127],
            mma_return=trace['mma_returns'][1],main_end=trace['main_ends'][1])
        intervals=_stage_progress_intervals(dict(clocks=clocks))
        return dict(intervals=intervals,second_output_service_carry=max(0.,trace['service_frontier'][63]-B[64]),
            last_commit_uses_lead_branch=abs(C[127]-(trace['service_frontier'][127]-q))<1e-7,
            last_MAIN_uses_completion_branch=abs(trace['main_ends'][1]-trace['completed'][127])<1e-7,
            full_return_to_commit16=C[80]-B[80])
    def residual(z,g=0.):return np.array([evaluate(z,6,g)['intervals'][k] for k in fields])/targets-1
    fits=[]
    for p in (50.,400.):
        for latency in (1000.,2000.,3000.):
            for q in (280.,407.,490.):
                f=least_squares(residual,[p,latency,q],bounds=([0.,0.,0.],[1024.,8000.,tail]),
                    diff_step=1e-4,max_nfev=400,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                fits.append(dict(loss=float(f.fun@f.fun),parameters=f.x.tolist()))
    fits.sort(key=lambda x:x['loss']);best=fits[0]
    necessary=[]
    for stage in (4,6):
        values=[p['first_to_commit1'] for p in progress['processes'] if p['stages']==stage]
        necessary.append(dict(source='739578 progress',stages=stage,endpoint='FIRST to commit1',
            minimum_cycles=min(values),q_min=1024-min(values)))
        values=[p['consumer_issue_to_commit'] for p in sparse['processes'] if p['stages']==stage]
        necessary.append(dict(source='739556 sparse',stages=stage,endpoint='full return to commit',
            minimum_cycles=min(values),q_min=512-min(values),issue_floor_max=min(values)))
    q_min=max(r['q_min'] for r in necessary)
    g_max=min(r['issue_floor_max'] for r in necessary if 'issue_floor_max' in r)
    profiles=[]
    for g in (0.,float(g_max)):
        for q in np.linspace(q_min,tail,5):
            candidates=[]
            for p in (30.,400.,600.):
                for latency in (1000.,2200.,3000.):
                    f=least_squares(lambda z:residual([*z,q],g),[p,latency],
                        bounds=([0.,0.],[1024.,8000.]),diff_step=1e-4,max_nfev=400,
                        ftol=1e-10,xtol=1e-10,gtol=1e-10)
                    candidates.append(dict(loss=float(f.fun@f.fun),parameters=[*f.x.tolist(),float(q)]))
            candidates.sort(key=lambda x:x['loss']);fit=candidates[0];z=fit['parameters']
            # I_i >= U_(i-S)+a; U_(i-S)>=C_(i-S+1)+w;
            # C_i >= I_i+latency+c-q. This bounds a finite 46-interval window.
            jump=z[1]+512-q+protocol['wait_overhead']
            lower=(46//3)*max(0.,jump)/46
            profiles.append(dict(issue_floor=g,**fit,
                same_loss_examples=[r for r in candidates if r['loss']<=fit['loss']+1e-9],
                stage6=evaluate(z,6,g),stage4=evaluate(z,4,g),
                stage4_late_average_lower_bound=lower,
                stage4_late_observed_range=[observed[4]['average_commit17_to63']['trial_min'],
                                           observed[4]['average_commit17_to63']['trial_max']]))
    ready_path=output/'R13-ready-release.json';ready=read_json(ready_path)
    queue_pairs=[]
    for r in ready['paired_processes']:
        if r['kind']!='delay' or r['observer']!=1:continue
        queue_pairs.append(dict(trial=r['trial'],pitch_bytes=r['pitch_bytes'],
            issue_relative_to_entry_change_ns=r['delta']['P0_ns']+r['delta']['prod_to_issue_ns'],
            ready_response_lower_ns=r['bounds']['issue_to_ready_ns']['lower']))
    queue_scope=dict(first_request_formula='A0-I0=s+ell+max(Qinitial-I0,0)',
        implication='For a fixed initial state relative to CTA entry and fixed s/ell, later own issue cannot increase first response.',
        paired_processes=queue_pairs,
        later_issue_with_strictly_larger_response=sum(r['issue_relative_to_entry_change_ns']>0 and
            r['ready_response_lower_ns']>0 for r in queue_pairs),
        scope='Rejects only an isolated per-CTA fixed-initial-state server as a complete first-supply model. '
            'Shared external request predecessors or another explicit state remain possible; do not fit an initial state per condition.')
    sparse_source_bounds={}
    for stage in (4,6):
        values=[r['issue_to_observed_full'] for r in sparse['processes'] if r['stages']==stage]
        sparse_source_bounds[stage]=dict(minimum=min(values),maximum=max(values),
            meaning='Same-call issue_end to consumer full return is an upper bound on issue_end to readiness, not a measured bare latency.')
    hashes={str(p):common.sha(p) for p in (progress_path,sparse_path,ready_path,Path(__file__),Path(__file__).with_name('supply_model.py'))}
    report=dict(scope='Development diagnosis, not frozen validation. Only three stage6 progress medians fit p/latency/q; '
        'stage4 progress and all whole-L windows are comparisons. Independent sparse endpoints constrain the q profile, '
        'so that profile is conditional evidence, not an untouched stage4 validation.',gpu=progress['gpu'],
        equations='I_i=max(I_(i-1)+p,U_(i-S)); A_i=I_i+l; B_i=max(A_i,H_i); E_i=max(B_i,E_(i-1))+512; '
            'C_i=max(B_i+g,E_i-q); D_i=E_i+lambda; W_i=max(C_i,D_(i-1))+w except prologue; '
            'U_(i-1)=W_i; H_(i+1)=W_i+h; M_last=W_last+r; MAIN=max(M_last,D_last).',
        protocol_parameters=protocol,tail_constraint=tail,calibration_fields=list(fields),
        calibration_medians=dict(zip(fields,targets.tolist())),necessary_local_bounds=necessary,
        initial_fit=best,initial_fit_multistart=fits,
        initial_comparison={str(s):evaluate(best['parameters'],s) for s in (4,6)},
        observed_progress=observed,constrained_profiles=profiles,input_sha256=hashes,
        next_queue_scope_check=queue_scope,sparse_source_response_upper_bounds=sparse_source_bounds,
        limitations=['E is aggregate service progress. This does not model eight instruction arrival times or identify a physical queue.',
            'The initial g=0,q<=500 subclass still gives C_i>=C_(i-1)+512. It is not evidence that every real commit spacing obeys that floor.',
            'w,h,r and the 500-cycle tail come from the same stage6 progress observer and are effective software/measurement intervals.',
            'lambda=500-q is an active-last-commit/completion-branch restriction; both branches and the resulting tail are reported, not assumed universal.',
            'Ordinary release at W and zero reuse_delay are optimistic. The observed nonzero release/acquire path is not silently identified with zero hardware delay.',
            'g=105 is a sensitivity endpoint from another observer, not a fitted hardware instruction cost.',
            'Each profile fits only p/latency to the same stage6 targets. No profile is selected using stage4 error.',
            'Five q values and local multistart solutions are not exhaustive bounds on all continuous parameter space.',
            'The finite-window lower bound applies to each profiled parameter tuple and optimistic lifecycle, not every possible constant-latency architecture.',
            'No ready observation, measured frequency or per-case correction enters the formal V09 predictor.'])
    (output/'B-stage-admission.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 组内发射与TC服务重叠的递推检验','',
        '本次只用739578 stage6的三个局部中位量拟合：FIRST→commit1、commit1→16平均间隔、commit17→63平均间隔。'
        'whole-L和stage4进度不参与拟合；条件q剖面另受739556局部端点约束，属于开发诊断。','',
        'E表示累计服务进度，C=max(B+g,E−q)，D=E+λ；保留wait1释放前一slot、末slot的tail释放，以及跨输出的E和slot。'
        '512是此候选的服务假设。q不命名为物理队列容量。','',
        '| 局部必要界 | stage | 最短周期 | q下界 |','|---|---:|---:|---:|']
    for r in necessary:lines.append(f'| {r["source"]}，{r["endpoint"]} | {r["stages"]} | {r["minimum_cycles"]} | {r["q_min"]} |')
    lines+=['',f'初始最优p/ℓ/q={best["parameters"]}；p存在相同损失解。该q不满足独立局部下界，不能作为参数交付。',
        '条件剖面固定q于407～500的五个位置，只重解p/ℓ；g=0为乐观边界、g=105为敏感性端点。软件w/h/r固定为27/89.5/13，λ=500−q，逐解检查末端分支。','',
        '| g | q | ℓ | stage6 三局部量 | stage4 L | stage4 后46间隔均值 | 该均值的必要下界 |',
        '|---:|---:|---:|---|---:|---:|---:|']
    for r in profiles:
        v=r['stage4']['intervals'];a=r['stage6']['intervals']
        lines.append(f'| {r["issue_floor"]:.0f} | {r["parameters"][2]:.2f} | {r["parameters"][1]:.2f} | '
            +'/'.join(f'{a[k]:.2f}' for k in fields)+f' | {v["L"]:.2f} | {v["average_commit17_to63"]:.2f} | {r["stage4_late_average_lower_bound"]:.2f} |')
    actual=observed[4]
    lines+=['',f'stage4实测L中位{actual["L"]["median"]:.2f}，后46间隔均值中位{actual["average_commit17_to63"]["median"]:.2f}、'
        f'十进程范围[{actual["average_commit17_to63"]["trial_min"]:.2f},{actual["average_commit17_to63"]["trial_max"]:.2f}] cycle。',
        '必要界由 C_i≥C_(i−S+1)+ℓ+512−q+w 得到：46间隔至少包含floor(46/3)次这样的跳跃，其余间隔仅取非负。'
        '它证明表列参数即使乐观地立即释放slot也太慢；不把有限剖面冒充整个连续参数域的穷尽证明。','',
        '这版候选不进入预测。允许组内重叠修正了事件位置，但本轮所列共享ready延迟参数未能同时解释stage响应；未穷尽所有常量延迟模型。'
        '下一步先用[B-stage-handoff](B-stage-handoff.md)核查被省略的计算组交接，不能直接把本轮失败归因于ready队列。'
        '新候选须预先写清校准/诊断分工，保留本次失败，不能继续用stage4反选本次参数。','']
    lines+=['','新Q候选拟合前也有现成约束：若只排列本CTA请求且初态固定，'
        '首请求响应为s+ℓ+max(Qinitial−I0,0)，延后I0只能不增。739653的'
        f'{queue_scope["later_issue_with_strictly_larger_response"]}/{len(queue_pairs)}个配对却同时表现为更晚issue与严格更长响应。'
        '因此孤立、固定初态的版本不能覆盖首次供给；必须限定为输出内子模型，或明确外部请求/共享状态的可预测前驱。',
        f'739556 stage6的同调用issue_end→full返回范围为[{sparse_source_bounds[6]["minimum"]},'
        f'{sparse_source_bounds[6]["maximum"]}] cycle，对新s+ℓ只给上界。'
        '不能将2257.5原样作为无排队底值后再加排队；也不能把不同观察批次拼成绝对事件线。','']
    (output/'B-stage-admission.md').write_text('\n'.join(lines))
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
