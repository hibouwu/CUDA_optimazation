#!/usr/bin/env python3
"""R17 analysis: per-CTA mainloop cycles/Ktile by tile class and round.

  r17_analyze.py --run RUN        writes RUN/cases.csv, RUN/classes.csv, RUN/summary.json, plots/

Tile classes (cfg_a, cluster 2x1 along M; tile (mi, ni), 128x128):
  oob      mi*128 >= M              padded row, A box fully outside the tensor (TMA zero fill)
  partner  cluster mate of an oob tile (same cluster, mi^1)
  partial  last real row, M not a multiple of 128 (A box partly outside)
  in       everything else
Mainloop per Ktile = (MAIN_END - FIRST_MMA) / Kt of one tile of one CTA (clock64, same SM).
Per (case, round, class): median over CTAs of each stamped process, then median over processes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import v06_model as model
import r17_run  # registers cfg_a1 in model.CONFIGS

med = statistics.median


def tile_class(config, m, mi):
    if mi * 128 >= m:
        return "oob"
    if config == "cfg_a" and (mi ^ 1) * 128 >= m:
        return "partner"
    if m % 128 and mi == m // 128:
        return "partial"
    return "in"


def load_records(run):
    return [json.loads(l) for l in (run / "samples/r17.jsonl").read_text().splitlines() if l]


def per_process(rec, row):
    """{(round, class): [cycles/Ktile per CTA]} plus first-tile start skew of one process."""
    kt = model.cdiv(row["k"], 64)
    work = model.scheduled_work(row["config"], row["m"], row["n"], rec["setup"]["grid"])
    out = {}
    for cta, tiles in zip(rec["ctas"], work):
        for j, (t, (mi, _)) in enumerate(zip(cta["tiles"], tiles)):
            out.setdefault((j, tile_class(row["config"], row["m"], mi)), []).append(
                (t[1] - t[0]) / kt)
    return out


def analyze(run: Path):
    rows = {r["id"]: r for r in json.loads((run / "cases.json").read_text())}
    recs = load_records(run)
    case_out, class_out = [], []
    for cid, row in rows.items():
        mine = [r for r in recs if r["case"] == cid]
        ok = [r for r in mine if r["returncode"] == 0 and r.get("check") == "ok"]
        plain = [r["elapsed_us"] for r in ok if r["variant"] == "plain"]
        stamped = [r["elapsed_us"] for r in ok if r["variant"] == "stamped"]
        per_key, sums = {}, []
        for r in ok:
            if r["variant"] != "stamped":
                continue
            rec = model.load_process(run / r["raw"])
            sums.append(model.process_summary(rec))
            for key, vals in per_process(rec, row).items():
                per_key.setdefault(key, []).append(vals)
        kt = model.cdiv(row["k"], 64)
        case = dict(case=cid, group=row["group"], config=row["config"], m=row["m"], n=row["n"],
                    k=row["k"], kt=kt, rows=model.cdiv(row["m"], 128),
                    oob_row=row["config"] == "cfg_a" and model.cdiv(row["m"], 128) % 2 == 1,
                    n_plain=len(plain), n_stamped=len(stamped), failed=len(mine) - len(ok),
                    plain_us=med(plain), plain_cv=statistics.pstdev(plain) / statistics.mean(plain),
                    stamped_us=med(stamped), perturbation=med(stamped) / med(plain) - 1,
                    crit_cycles=med(s["c_max"] for s in sums), ghz=med(s["ghz"] for s in sums),
                    max_err=max(r.get("max_err") or 0 for r in ok))
        for (j, cls), procs in sorted(per_key.items()):
            meds = [med(v) for v in procs]
            pooled = [x for v in procs for x in v]
            class_out.append(dict(case=cid, config=row["config"], m=row["m"], n=row["n"],
                                  k=row["k"], kt=kt, round=j, cls=cls, ctas=len(procs[0]),
                                  per_ktile=med(meds), min=min(pooled), max=max(pooled)))
            case[f"r{j}_{cls}"] = med(meds)
        case_out.append(case)
    is_class = lambda k: k[0] == "r" and k[1].isdigit()
    keys = [k for k in case_out[0] if not is_class(k)]
    keys += sorted({k for c in case_out for k in c if is_class(k)})
    with (run / "cases.csv").open("w", newline="") as s:
        w = csv.DictWriter(s, fieldnames=keys)
        w.writeheader()
        w.writerows(case_out)
    with (run / "classes.csv").open("w", newline="") as s:
        w = csv.DictWriter(s, fieldnames=list(class_out[0]))
        w.writeheader()
        w.writerows(class_out)
    summary = dict(processes=len(recs), failed=sum(c["failed"] for c in case_out),
                   checks_ok=sum(1 for r in recs if r.get("check") == "ok"),
                   max_check_error=max(c["max_err"] for c in case_out),
                   perturbation_max_abs=max(abs(c["perturbation"]) for c in case_out),
                   plain_cv_max=max(c["plain_cv"] for c in case_out),
                   environment=json.loads((run / "environment.json").read_text()),
                   resources=json.loads((run / "build/resources.json").read_text()),
                   analyzer_sha256=model.sha(Path(__file__)))
    return case_out, class_out, summary


def plot(run, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (run / "plots").mkdir(exist_ok=True)
    series = (("r0_in", "round 0, other CTAs", "#8aa6c8"),
              ("r0_bnd", "round 0, boundary cluster (oob/partner/partial)", "#c8742c"),
              ("r1_in", "round 1, other CTAs", "#4c9a5b"))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    x = range(len(cases))
    for i, (key, label, color) in enumerate(series):
        vals = [c.get("r0_oob") or c.get("r0_partial") if key == "r0_bnd" else c.get(key)
                for c in cases]
        ax.bar([j + (i - 1) * 0.27 for j in x], [v or 0 for v in vals], 0.27, label=label,
               color=color)
    ax.axhline(512, color="k", lw=0.6, ls=":")
    ax.set_ylim(450, 760)
    ax.set_xticks(list(x))
    ax.set_xticklabels([c["case"].replace("cfg_", "") for c in cases], rotation=60, fontsize=7)
    ax.set_ylabel("mainloop cycles / Ktile")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "plots/r17_mainloop.png", dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    a = ap.parse_args()
    run = a.run.resolve()
    cases, classes, summary = analyze(run)
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(run, cases)
    print(f"{'case':24} {'plain':>8} {'cv%':>5} {'pert%':>6} {'GHz':>6}  classes r<round>_<cls>")
    for c in cases:
        cls = " ".join(f"{k}={v:.0f}" for k, v in c.items() if k.startswith("r") and k[1].isdigit())
        print(f"{c['case']:24} {c['plain_us']:8.2f} {100 * c['plain_cv']:5.2f} "
              f"{100 * c['perturbation']:6.2f} {c['ghz']:6.3f}  {cls}")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("resources",)}, indent=1))


if __name__ == "__main__":
    main()
