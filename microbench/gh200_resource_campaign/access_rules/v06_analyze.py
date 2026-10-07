#!/usr/bin/env python3
"""V06 held-out scoring against the frozen prediction file (no refit).

  v06_analyze.py --run RUN [--predictions RUN/v06-predictions.json]
Writes RUN/cases.csv, RUN/summary.json, RUN/plots/v06_heldout.png.
Scored: plain CUDA-event microseconds (median of processes). Diagnostics from the stamped
build: critical-CTA cycles, in-call clock, measured intervals (post hoc, not used to refit).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import v06_fit
import v06_model as model

med = statistics.median
PRIOR = dict(V04=dict(points=22, median=0.056, max=0.292),
             V05=dict(points=18, median=0.1073, max=0.3186))


def analyze(run: Path, pred_path: Path):
    frozen = json.loads(pred_path.read_text())
    binding = json.loads((run / "prediction_binding.json").read_text())
    if binding["sha256"] != model.sha(pred_path):
        raise ValueError("prediction file differs from the one bound before sampling")
    records = v06_fit.load_records(run, "heldout")
    first_start = min(r["start_ns"] for r in records)
    rows, by_case = [], {}
    for r in records:
        by_case.setdefault(r["case"], {"plain": [], "stamped": []})[r["variant"]].append(r)
    for cid, p in frozen["predictions"].items():
        g = by_case[cid]
        ok = [x for x in g["plain"] + g["stamped"] if x["returncode"] == 0 and x.get("check") == "ok"]
        plain = [x["elapsed_us"] for x in g["plain"] if x in ok]
        stamped = [x["elapsed_us"] for x in g["stamped"] if x in ok]
        sched = model.CONFIGS[p["row"]["config"]]["schedule"]
        sums, ivs = [], []
        for x in g["stamped"]:
            if x in ok:
                rec = model.load_process(run / x["raw"])
                sums.append(model.process_summary(rec))
                ivs.append(v06_fit.intervals(rec, sched)[0])
        meas_us, c_meas = med(plain), med(s["c_max"] for s in sums)
        ghz_meas, w_meas = med(s["ghz"] for s in sums), med(s["window_us"] for s in sums)
        tiles_meas = {s["tiles_max"] for s in sums}
        row = dict(
            case=cid, config=p["row"]["config"], label=p["row"]["label"],
            m=p["row"]["m"], n=p["row"]["n"], k=p["row"]["k"], lda=p["row"]["lda"],
            ldb=p["row"]["ldb"], ldd=p["row"]["ldd"], tiles=p["tiles"], rounds=p["rounds"],
            rounds_meas=sorted(tiles_meas), ctas=p["ctas"],
            pred_us=p["predicted_us"], meas_us=meas_us, us_err=p["predicted_us"] / meas_us - 1,
            plain_cv=statistics.pstdev(plain) / statistics.mean(plain), n_plain=len(plain),
            n_stamped=len(stamped), failed=len(g["plain"]) + len(g["stamped"]) - len(ok),
            perturbation=med(stamped) / meas_us - 1,
            pred_cycles=p["critical_cycles"], meas_cycles=c_meas,
            cycle_err=p["critical_cycles"] / c_meas - 1,
            pred_ghz=p["clock_ghz"], meas_ghz=ghz_meas, ghz_err=p["clock_ghz"] / ghz_meas - 1,
            pred_window_us=p["window_us"], meas_window_us=w_meas,
            fixed_us_pred=p["fixed_us"], fixed_us_meas=meas_us - w_meas,
            # Post-hoc diagnostics only (measured quantity substituted, frozen rest kept).
            diag_us_err_meas_clock=(p["fixed_us"] + p["critical_cycles"] / (ghz_meas * 1e3))
            / meas_us - 1,
            diag_us_err_meas_cycles=(p["fixed_us"] + c_meas / (p["clock_ghz"] * 1e3)) / meas_us - 1,
            phi=p["phi"], mu=p["mu"], dram_tbs=p["dram_tbs"])
        for k in ("S", "L", "E", "h", "gm", "w", "we", "Etail", "P0", "L0"):
            vals = [iv[k] for iv in ivs if k in iv]
            row[f"meas_{k}"] = med(vals) if vals else None
        rows.append(row)
    absval = lambda key, rs: sorted(abs(r[key]) for r in rs)
    def stats(key, rs):
        v = absval(key, rs)
        return dict(median=med(v), max=max(v)) if v else None
    summary = dict(
        prediction_sha256=binding["sha256"], frozen_utc=frozen["frozen_utc"],
        frozen_unix_ns=frozen["frozen_unix_ns"], first_heldout_sample_unix_ns=first_start,
        freeze_before_measurement=frozen["frozen_unix_ns"] < first_start,
        gpu=frozen["gpu"], job=frozen["job"], host=frozen["host"],
        processes=len(records), failed=sum(r["failed"] for r in rows),
        checks_ok=sum(1 for r in records if r.get("check") == "ok"),
        max_check_error=max(r.get("max_err", 0) or 0 for r in records),
        us=stats("us_err", rows), cycles=stats("cycle_err", rows), clock=stats("ghz_err", rows),
        perturbation=stats("perturbation", rows),
        diag_meas_clock=stats("diag_us_err_meas_clock", rows),
        diag_meas_cycles=stats("diag_us_err_meas_cycles", rows),
        per_config={cfg: dict(us=stats("us_err", [r for r in rows if r["config"] == cfg]),
                              cycles=stats("cycle_err", [r for r in rows if r["config"] == cfg]),
                              clock=stats("ghz_err", [r for r in rows if r["config"] == cfg]))
                    for cfg in model.CONFIGS},
        prior=PRIOR, target=dict(median=0.05, max=0.10))
    summary["passed"] = summary["us"]["median"] <= 0.05 and summary["us"]["max"] <= 0.10
    with (run / "cases.csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(run, rows)
    return rows, summary


def plot(run, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (run / "plots").mkdir(exist_ok=True)
    colors = {"cfg_a": "#3b6ea8", "cfg_b": "#c8742c", "cfg_c": "#4c9a5b"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels = [r["case"].replace("cfg_", "") for r in rows]
    x = range(len(rows))
    for ax, key, title in ((axes[0], "us_err", "µs error (frozen)"),
                           (axes[1], "cycle_err", "critical-CTA cycle error"),
                           (axes[2], "ghz_err", "clock error")):
        ax.bar(x, [100 * r[key] for r in rows], color=[colors[r["config"]] for r in rows])
        ax.axhline(0, color="k", lw=0.6)
        for lim in (5, -5, 10, -10):
            ax.axhline(lim, color="grey", lw=0.5, ls=":" if abs(lim) == 5 else "--")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=75, fontsize=7)
        ax.set_ylabel("%")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(run / "plots/v06_heldout.png", dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--predictions", type=Path)
    a = ap.parse_args()
    run = a.run.resolve()
    rows, s = analyze(run, (a.predictions or run / "v06-predictions.json").resolve())
    print(f"{'case':26} {'pred':>8} {'meas':>8} {'e%':>6} {'cyc%':>6} {'clk%':>6} {'pert%':>6} "
          f"{'GHz p/m':>12} {'T':>3}")
    for r in rows:
        print(f"{r['case']:26} {r['pred_us']:8.2f} {r['meas_us']:8.2f} {100 * r['us_err']:6.2f} "
              f"{100 * r['cycle_err']:6.2f} {100 * r['ghz_err']:6.2f} {100 * r['perturbation']:6.2f} "
              f"{r['pred_ghz']:.3f}/{r['meas_ghz']:.3f} {r['rounds']:3}")
    print(json.dumps({k: s[k] for k in ("us", "cycles", "clock", "perturbation", "passed",
                                        "freeze_before_measurement", "failed")}, indent=1))


if __name__ == "__main__":
    main()
