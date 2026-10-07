#!/usr/bin/env python3
"""V06 calibration fit (calibration sizes only) and prediction freeze.

  v06_fit.py --run RUN                       fit + in-sample report -> RUN/calibration.json
  v06_fit.py --run RUN --freeze OUT.json     also predict held-out sizes, write OUT read-only

Interval definitions (per stamped process: mean over CTAs/tiles for per-tile intervals, median
for one-off intervals; then median over processes):
  P0 = producer first work - entry          S = tile-0 first MMA - producer first work
  L  = me - fm for tiles j>=1 (dL0 = tile-0 excess)
  cooperative: w = ep - me, E0 / E = ed - ep (tile 0 / later), h = fm[j+1] - ed[j],
               Etail = end - ed[last]
  pingpong:    gm = fm[j] - me[j-1] where the MMA chain binds, h = fm[j] - ed[j-2] where the
               own-epilogue chain binds; w / we likewise for the epilogue permit;
               E = ed - ep of the CTA's last tile (alone), r = E / (ed - ep) of tiles whose
               window is fully covered by the other WG's next mainloop.
Per-config constants are the median over calibration cases; L = l0 + l1*Kt by least squares;
max-CTA excess X (none / const / proportional / linear in T) chosen by leave-one-out.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import statistics
import time

import v06_model as model

med = statistics.median
ACCUMULATING = ("L", "E", "Efull", "h", "gm", "w", "we")


def load_records(run, set_name):
    rows = [json.loads(l) for l in (run / "samples" / f"{set_name}.jsonl").open() if l.strip()]
    return rows


def intervals(rec, schedule):
    """Per-process medians of every interval (cycles)."""
    keys = ("P0", "S", "L0", "L", "w", "we", "E0", "E", "Efull", "h", "gm", "Etail")
    acc = {k: [] for k in keys}
    for c in rec["ctas"]:
        t = c["tiles"]
        if not t:
            continue
        acc["P0"].append(c["prod_c"] - c["entry_c"])
        acc["S"].append(t[0][0] - c["prod_c"])
        acc["L0"].append(t[0][1] - t[0][0])
        acc["L"] += [x[1] - x[0] for x in t[1:]]
        acc["Etail"].append(c["end_c"] - max(x[3] for x in t))
        for j, (fm, me, ep, ed) in enumerate(t):
            if schedule == "cooperative":
                acc["w"].append(ep - me)
                acc["E0" if j == 0 else "E"].append(ed - ep)
                if j >= 1:
                    acc["h"].append(fm - t[j - 1][3])
                continue
            # Pingpong epilogue: alone on the CTA's last tile; "full" when the next tile's
            # mainloop (other WG) covers the whole epilogue window.
            if j == len(t) - 1:
                acc["E"].append(ed - ep)
            elif t[j + 1][0] <= ep and t[j + 1][1] >= ed:
                acc["Efull"].append(ed - ep)
            if j >= 1:
                mma_prev, epi_prev = t[j - 1][1], t[j - 1][3]
                own = t[j - 2][3] if j >= 2 else -1
                (acc["gm"] if mma_prev >= own else acc["h"]).append(
                    fm - (mma_prev if mma_prev >= own else own))
                (acc["w"] if me >= epi_prev else acc["we"]).append(
                    ep - (me if me >= epi_prev else epi_prev))
            else:
                acc["w"].append(ep - me)
    # Durations that accumulate along a CTA (per-tile) use the mean; one-off ones the median.
    agg = {k: (statistics.fmean(v) if k in ACCUMULATING else med(v)) for k, v in acc.items() if v}
    return agg, {k: len(v) for k, v in acc.items()}


def choose_excess(pts):
    """pts: (T, B, C_meas). Forms: none, const (x0), prop (xk*B), linT (x0 + x1*T)."""
    def solve(form, idx):
        if form == "none":
            return dict(x0=0.0, x1=0.0, xk=0.0)
        if form == "const":
            return dict(x0=med(pts[i][2] - pts[i][1] for i in idx), x1=0.0, xk=0.0)
        if form == "prop":
            return dict(x0=0.0, x1=0.0, xk=med(pts[i][2] / pts[i][1] - 1 for i in idx))
        xs, ys = [pts[i][0] for i in idx], [pts[i][2] - pts[i][1] for i in idx]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        x1 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        return dict(x0=my - x1 * mx, x1=x1, xk=0.0)

    def predict(c, T, B):
        return B + c["x0"] + c["x1"] * T + c["xk"] * B

    fits = {}
    allidx = list(range(len(pts)))
    for form, nparam in (("none", 0), ("const", 1), ("prop", 1), ("linT", 2)):
        coef = solve(form, allidx)
        loo = [predict(solve(form, [j for j in allidx if j != i]), pts[i][0], pts[i][1])
               / pts[i][2] - 1 for i in allidx]
        fits[form] = dict(coef=coef, nparam=nparam,
                          loo_rms=math.sqrt(sum(e * e for e in loo) / len(loo)),
                          loo_max=max(map(abs, loo)))
    best = min(f["loo_rms"] for f in fits.values())
    ok = [k for k, f in fits.items() if f["loo_rms"] <= best + 0.002]
    return min(ok, key=lambda k: (fits[k]["nparam"], fits[k]["loo_rms"])), fits


def fit(run):
    setups = {s["case_id"]: s for s in json.loads((run / "static_setup.json").read_text())}
    rows = {r["id"]: r for r in model.case_rows("calib")}
    records = load_records(run, "calib")
    bad = [r for r in records if r["returncode"] != 0 or r.get("check") != "ok"]
    by_case = {}
    for r in records:
        if r["returncode"] == 0 and r.get("check") == "ok":
            by_case.setdefault(r["case"], {"plain": [], "stamped": []})[r["variant"]].append(r)
    cases = {}
    for cid, groups in sorted(by_case.items()):
        cfg = rows[cid]["config"]
        sched = model.CONFIGS[cfg]["schedule"]
        per_proc, summaries = [], []
        for r in groups["stamped"]:
            rec = model.load_process(run / r["raw"])
            per_proc.append(intervals(rec, sched)[0])
            summaries.append(model.process_summary(rec))
        keys = set().union(*per_proc) if per_proc else set()
        plain = med(x["elapsed_us"] for x in groups["plain"])
        stamped = med(x["elapsed_us"] for x in groups["stamped"])
        cases[cid] = dict(
            config=cfg, ktiles=model.cdiv(rows[cid]["k"], 64), grid=setups[cid]["grid"],
            plain_us=plain, stamped_us=stamped, perturbation=stamped / plain - 1,
            plain_cv=statistics.pstdev([x["elapsed_us"] for x in groups["plain"]]) / plain,
            n_plain=len(groups["plain"]), n_stamped=len(groups["stamped"]),
            intervals={k: med(p[k] for p in per_proc if k in p) for k in keys},
            c_max=med(s["c_max"] for s in summaries), window_us=med(s["window_us"] for s in summaries),
            ghz=med(s["ghz"] for s in summaries), tiles_max=summaries[0]["tiles_max"],
            span_us=med(s["span_us"] for s in summaries))
    params = {}
    for cfg, spec in model.CONFIGS.items():
        cs = [c for c in cases.values() if c["config"] == cfg]
        p = {}
        names = ["P0", "S", "w", "E", "h", "Etail"]
        names += ["gm", "we", "Efull"] if spec["schedule"] == "pingpong" else ["E0"]
        for k in names:
            vals = [c["intervals"][k] for c in cs if k in c["intervals"]]
            p[k] = med(vals) if vals else 0.0
        if spec["schedule"] == "pingpong":
            p["r"] = p["E"] / p.pop("Efull")
        pts = [(c["ktiles"], c["intervals"]["L"]) for c in cs if "L" in c["intervals"]]
        n = len(pts)
        mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
        p["l1"] = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
        p["l0"] = my - p["l1"] * mx
        p["dL0"] = med(c["intervals"]["L0"] - (p["l0"] + p["l1"] * c["ktiles"]) for c in cs)
        # Max-over-CTAs excess X: candidate forms fitted to (measured C_max - B), B = recursion
        # with median intervals; chosen by leave-one-out RMS of the relative cycle error
        # (fixed rule: fewest parameters within 0.2 pp of the best).
        p["x0"] = p["x1"] = p["xk"] = 0.0
        pts = []
        for c in cs:
            row = rows[[k for k, v in cases.items() if v is c][0]]
            work = model.scheduled_work(cfg, row["m"], row["n"], c["grid"])
            base = max(model.cta_cycles(p, spec["schedule"], len(w), c["ktiles"]) for w in work)
            pts.append((c["tiles_max"], base, c["c_max"]))
        p["excess_form"], p["excess_fits"] = choose_excess(pts)
        p.update(p["excess_fits"][p["excess_form"]]["coef"])
        params[cfg] = p
    # In-sample cycle check and clock features at the measured window.
    for cid, c in cases.items():
        row = rows[cid]
        work = model.scheduled_work(c["config"], row["m"], row["n"], c["grid"])
        sched = model.CONFIGS[c["config"]]["schedule"]
        cyc = [model.cta_cycles(params[c["config"]], sched, len(w), c["ktiles"]) for w in work]
        cmax, stages = model.cta_cycles(params[c["config"]], sched, max(map(len, work)),
                                        c["ktiles"], detail=True)
        c["c_pred"] = max(cyc)
        c["cycle_err"] = c["c_pred"] / c["c_max"] - 1
        c["phi"] = sum(cyc) / (model.SMS * c["c_pred"])
        c["mu"] = stages["mainloop"] / c["c_pred"]
        c["dram_tbs"] = (model.dram_bytes(c["config"], row["m"], row["n"], row["k"], work)
                         / (c["window_us"] * 1e-6) / 1e12)
        c["lnw"] = math.log(c["window_us"])
        base = dict(model.V03_RULE)
        c["ghz_v03"] = model.clock_ghz(base, c["phi"], c["lnw"], c["dram_tbs"], c["mu"])
    rule, rule_fits = fit_clock(list(cases.values()))
    fixed = {}
    for cfg in model.CONFIGS:
        cs = [c for c in cases.values() if c["config"] == cfg]
        # Same-process quantity: stamped event time minus its critical-CTA window.
        fixed[cfg] = med(c["stamped_us"] - c["window_us"] for c in cs)
    for cid, c in cases.items():
        row = rows[cid]
        pred = model.predict_case(row, c["grid"], params[c["config"]], rule, fixed[c["config"]])
        c["insample_us"] = pred["predicted_us"]
        c["insample_us_err"] = pred["predicted_us"] / c["plain_us"] - 1
        c["insample_ghz"] = pred["clock_ghz"]
        c["insample_ghz_err"] = pred["clock_ghz"] / c["ghz"] - 1
    return dict(cases=cases, params=params, clock_rule=rule, clock_fits=rule_fits,
                fixed_us=fixed, failed_processes=len(bad), processes=len(records))


def fit_clock(cases):
    """Card correction to the V03 rule: offset da, optionally dc (D coefficient)."""
    resid = [c["ghz"] - c["ghz_v03"] for c in cases]
    fits = {}
    for name in ("offset", "offset+D"):
        def solve(idx):
            if name == "offset":
                return [sum(resid[i] for i in idx) / len(idx), 0.0]
            xs = [cases[i]["dram_tbs"] for i in idx]
            ys = [resid[i] for i in idx]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            return [my - slope * mx, -slope]  # residual = da - dc*D
        allidx = list(range(len(cases)))
        da, dc = solve(allidx)
        rel = [(cases[i]["ghz_v03"] + da - dc * cases[i]["dram_tbs"]) / cases[i]["ghz"] - 1
               for i in allidx]
        loo = []
        for i in allidx:
            a, b = solve([j for j in allidx if j != i])
            loo.append((cases[i]["ghz_v03"] + a - b * cases[i]["dram_tbs"]) / cases[i]["ghz"] - 1)
        fits[name] = dict(da=da, dc=dc, rms=math.sqrt(sum(r * r for r in rel) / len(rel)),
                          max_abs=max(map(abs, rel)),
                          loo_rms=math.sqrt(sum(r * r for r in loo) / len(loo)),
                          loo_max_abs=max(map(abs, loo)))
    # Fixed selection rule: dc only if LOO RMS improves by more than 0.2 percentage points.
    chosen = "offset+D" if fits["offset+D"]["loo_rms"] < fits["offset"]["loo_rms"] - 0.002 else "offset"
    base = model.V03_RULE
    rule = dict(a=base["a"] + fits[chosen]["da"], b=base["b"], c=base["c"] + fits[chosen]["dc"],
                d=base["d"], refit=chosen, da=fits[chosen]["da"], dc=fits[chosen]["dc"])
    return rule, fits


def report(cal):
    print("processes", cal["processes"], "failed", cal["failed_processes"])
    for cfg, p in cal["params"].items():
        print(cfg, {k: round(v, 3) for k, v in p.items() if isinstance(v, float)},
              "F_us", round(cal["fixed_us"][cfg], 3), "excess", p["excess_form"],
              {k: round(f["loo_rms"], 4) for k, f in p["excess_fits"].items()})
    print("clock", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in cal["clock_rule"].items()})
    for name, f in cal["clock_fits"].items():
        print("  ", name, {k: round(v, 5) for k, v in f.items()})
    print(f"{'case':22} {'kt':>4} {'T':>3} {'plain':>8} {'pert':>6} {'Cmeas':>9} {'Cerr':>6} "
          f"{'GHz':>6} {'v03':>6} {'fit':>6} {'us_err':>7}  intervals")
    for cid, c in cal["cases"].items():
        iv = " ".join(f"{k}={v:.0f}" for k, v in sorted(c["intervals"].items()))
        print(f"{cid:22} {c['ktiles']:4} {c['tiles_max']:3} {c['plain_us']:8.2f} "
              f"{100 * c['perturbation']:6.2f} {c['c_max']:9.0f} {100 * c['cycle_err']:6.2f} "
              f"{c['ghz']:6.3f} {c['ghz_v03']:6.3f} {c['insample_ghz']:6.3f} "
              f"{100 * c['insample_us_err']:7.2f}  {iv}")


def freeze(run, cal, out: Path):
    if out.exists():
        raise ValueError("prediction file exists")
    novelty = json.loads((run / "novelty.json").read_text())
    n_prior, clash = novelty["prior_shapes_scanned"], novelty["heldout_clashes"]
    if clash:
        raise ValueError(f"held-out sizes already measured: {clash}")
    setups = {s["case_id"]: s for s in json.loads((run / "static_setup.json").read_text())}
    preds = {}
    for row in model.case_rows("heldout"):
        s = setups[row["id"]]
        p = model.predict_case(row, s["grid"], cal["params"][row["config"]], cal["clock_rule"],
                               cal["fixed_us"][row["config"]])
        preds[row["id"]] = dict(row=row, stages=s["stages"], **p)
    env = json.loads((run / "environment.json").read_text())
    now = time.time_ns()
    doc = dict(status="frozen", frozen_unix_ns=now,
               frozen_utc=datetime.datetime.fromtimestamp(now / 1e9, datetime.timezone.utc)
               .isoformat(timespec="seconds"),
               gpu=env["gpu"], job=env["job"], host=env["host"],
               model=model.__doc__.strip(),
               code_sha256={name: model.sha(Path(__file__).resolve().parent / name)
                            for name in ("v06_model.py", "v06_fit.py")},
               calibration_jsonl_sha256=model.sha(run / "samples/calib.jsonl"),
               static_setup_sha256=model.sha(run / "static_setup.json"),
               binary_hashes_sha256=model.sha(run / "build/binary_hashes.json"),
               novelty_prior_shapes_scanned=n_prior,
               params=cal["params"], clock_rule=cal["clock_rule"], clock_fits=cal["clock_fits"],
               fixed_us=cal["fixed_us"], predictions=preds)
    with out.open("x") as stream:
        json.dump(doc, stream, indent=2)
        stream.write("\n")
    out.chmod(0o444)
    print("frozen", doc["frozen_utc"], model.sha(out))
    for cid, p in preds.items():
        print(f"{cid:26} tiles {p['tiles']:5} rounds {p['rounds']:3} C {p['critical_cycles']:9.0f} "
              f"GHz {p['clock_ghz']:.3f} W {p['window_us']:8.2f} T {p['predicted_us']:8.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--freeze", type=Path)
    a = ap.parse_args()
    run = a.run.resolve()
    cal = fit(run)
    (run / "calibration.json").write_text(json.dumps(cal, indent=2) + "\n")
    report(cal)
    if a.freeze:
        freeze(run, cal, a.freeze.resolve())


if __name__ == "__main__":
    main()
