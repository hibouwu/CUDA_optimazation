#!/usr/bin/env python3
"""R13 offline max(compute, logical-source service) development diagnostics.

No GPU work or modification of frozen predictions. Default V08F mode reuses
summaries plus six cfg_a trace controls; --sm-summary uses the archived SM scan;
--coverage-suite compares two address-coverage/prefill forms on separate cohorts.
--fill-suite separates valid-address and OOB-fill demands on one card in cycle/ns forms.
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
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import linprog
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
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "residuals.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(residuals[0])); writer.writeheader(); writer.writerows(residuals)
    print(json.dumps({k:dict(fit=v["later_fit"], first=v["first_transfer"], rank=v["identification"]["active_jacobian_rank"]) for k,v in fits.items()}, indent=2))


@lru_cache(None)
def address_box(rows, row_bytes, pitch_bytes, start_mod128):
    """Valid address spans, counted per row; extra coverage is not physical traffic."""
    extra = []
    for width in (32, 128):
        minimum = (row_bytes + width - 1) // width
        count = sum(((start_mod128 + i*pitch_bytes) % width + row_bytes + width-1)//width - minimum
                    for i in range(rows))
        extra.append(count * width / 1024)
    return rows * row_bytes / 1024, *extra


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
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'residuals.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(residuals[0]));writer.writeheader();writer.writerows(residuals)
    print(json.dumps({c:{m:dict(rank=f['active_jacobian_rank'],nparam=f['parameter_count'],fit=f['fit']) for m,f in v['models'].items()} for c,v in result.items()},indent=2))


FILL_FIELDS = ['valid_source_KiB', 'B_extra32_KiB', 'B_extra128_KiB',
               'A_fill_KiB', 'B_fill_KiB']


def fill_geometry(row, setup):
    """Descriptor-address demand and local zero-fill demand for this cfg_b cohort."""
    from v08_model import scheduled_work
    if row['config']!='cfg_b': raise ValueError('this fill component is calibrated for cfg_b only')
    work=scheduled_work('cfg_b',row['m'],row['n'],setup['grid'],row['swizzle'])
    kt=(row['k']+63)//64
    if row['k']%64: raise ValueError('this fill comparison uses complete Ktiles')
    map_m=row.get('input_map_m',row['m']); map_n=row.get('input_map_n',row['n'])
    requests=[]
    for coords in work:
        values=[]
        for mi,ni in coords:
            am=max(0,min(128,map_m-mi*128)); bn=max(0,min(128,map_n-ni*128))
            a=address_box(am,128,2*row['lda'],mi*256*row['lda']%128) if am else (0,0,0)
            b=address_box(64,2*bn,2*row['ldb'],ni*256%128) if bn else (0,0,0)
            if a[1] or a[2]: raise ValueError('A pitch is aligned in this same-card batch')
            values.append(np.array([a[0]+b[0],b[1],b[2],16-a[0],16-b[0]]))
        requests.append(values)
    waves=[sum((v[j][:3] for v in requests if len(v)>j),np.zeros(3))
           for j in range(max(map(len,requests)))]
    credit=min(setup['stages'],kt)/kt
    effective=[(1-credit)*v+credit*(waves[j+1] if j+1<len(waves) else np.zeros(3))
               for j,v in enumerate(waves)]
    grouped=defaultdict(list)
    for coords,values in zip(work,requests):
        for j,((mi,ni),v) in enumerate(zip(coords,values)):
            cls='padM' if mi*128>=row['m'] else 'padN' if ni*128>=row['n'] else 'in'
            grouped[j,len(coords),cls].append(np.r_[effective[j],v[3:]])
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


def fill_window_cycles(request_features, kt, parameters, unit='cycle', frequency_ghz=None):
    """Development CTA L replacement; NS service requires an explicit predicted f."""
    service=float(np.dot(request_features,parameters[1:]))
    if unit=='cycle': return parameters[0]+kt*max(512,service)
    if frequency_ghz is None or frequency_ghz<=0: raise ValueError('NS form requires an explicit positive frequency assumption')
    return frequency_ghz*parameters[0]+kt*max(512,frequency_ghz*service)


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
        frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=False);(output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'residuals.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(residuals[0]));writer.writeheader();writer.writerows(residuals)
    print(json.dumps({u:{k:v for k,v in f.items() if k in ('later_fit','whole_fill','identification','leave_K')} for u,f in fits.items()},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--calibration-run", type=Path)
    parser.add_argument("--sm-summary", type=Path, help="fit local/grid/wave caps to an existing R13 SM-scan summary")
    parser.add_argument("--coverage-suite", action='store_true', help="two post-reveal coverage/prefill forms; --run is R10 job738203")
    parser.add_argument("--fill-suite", action='store_true',help='same-card valid-address/OOB cycle versus direct-ns development; --run is R10 job738203')
    parser.add_argument("--assumed-ghz",type=float,default=1.6,help='explicit compute-frequency assumption for --fill-suite NS form')
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.fill_suite:
        if args.assumed_ghz<=0:parser.error('--assumed-ghz must be positive')
        fill_suite(args.run.resolve(),args.output.resolve(),args.assumed_ghz)
    elif args.coverage_suite:
        coverage_suite(args.run.resolve(), args.output.resolve())
    elif args.sm_summary:
        sm_candidates(args.run.resolve(), args.sm_summary.resolve(), args.output.resolve())
    else:
        if args.calibration_run is None:parser.error("--calibration-run is required without --sm-summary")
        analyze(args.run.resolve(), args.calibration_run.resolve(), args.output.resolve())
