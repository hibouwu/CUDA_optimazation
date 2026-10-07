#!/usr/bin/env python3
"""R17 rule applied to the frozen V06 model (cfg_a only) and checked on the two V06 cases
with an odd tile-row count. NOT a held-out validation: R17 measured these sizes and the rule
values come from R17 data.

Rule (cfg_a, cluster 2x1 along M, tile 128x128x64):
  boundary cluster = the cluster that holds the last real tile row (and its padded partner row).
  If its A TMA box crosses the M bound (odd tile-row count -> padded fully-OOB row, or
  M % 128 != 0), the first tile (round 0) of both CTAs of every boundary cluster runs its
  mainloop at R per Ktile instead of l1:
      R = R_OOB      odd row count, M % 128 == 0  (padded row fully out of bounds)
      R = R_PARTIAL  even row count, M % 128 != 0 (last row partial, no padded row)
      R = R_BOTH     odd row count, M % 128 != 0
  Later tiles and all other CTAs keep the V06 parameters.
  R values = median over R17 points of each kind (round-0 boundary-cluster median per Ktile).

  r17_rule.py --run R17RUN --v06 V06RUN
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import v06_model as model
import r17_run  # noqa: F401  (registers cfg_a1)


def rule_rates(run: Path):
    """R per kind from R17 classes.csv (cfg_a, round 0, oob/partner/partial classes)."""
    kinds = {"oob": [], "partial": [], "both": []}
    for r in csv.DictReader((run / "classes.csv").open()):
        if r["config"] != "cfg_a" or r["round"] != "0":
            continue
        m = int(r["m"])
        odd, part = model.cdiv(m, 128) % 2 == 1, m % 128 != 0
        kind = "both" if odd and part else "oob" if odd else "partial" if part else None
        if kind and r["cls"] in ("oob", "partial"):  # partner equals oob within 5 cycles
            kinds[kind].append(float(r["per_ktile"]))
    return {k: statistics.median(v) for k, v in kinds.items() if v}, kinds


def boundary_rate(m, rates):
    odd, part = model.cdiv(m, 128) % 2 == 1, m % 128 != 0
    if odd:
        return rates["both" if part else "oob"]
    return rates["partial"] if part else None


def predict_with_rule(row, grid, params, rule, fixed_us, rate):
    """model.predict_case with per-CTA first-tile slope for the boundary cluster."""
    sched = model.CONFIGS[row["config"]]["schedule"]
    work = model.scheduled_work(row["config"], row["m"], row["n"], grid)
    kt = model.cdiv(row["k"], 64)
    last = model.cdiv(row["m"], 128) - 1
    bnd = {last, last ^ 1}
    slow = dict(params, dL0=params["dL0"] + ((rate - params["l1"]) * kt if rate else 0.0))
    per = [slow if w and w[0][0] in bnd else params for w in work]
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
    f = model.clock_ghz(rule, phi, math.log(w), dbytes / (w * 1e-6) / 1e12, mu)
    return dict(critical_cycles=c_max, clock_ghz=f, predicted_us=fixed_us + w)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--v06", required=True, type=Path)
    a = ap.parse_args()
    run, v06 = a.run.resolve(), a.v06.resolve()
    rates, kinds = rule_rates(run)
    frozen = json.loads((v06 / "v06-predictions.json").read_text())
    meas = {r["case"]: r for r in csv.DictReader((v06 / "cases.csv").open())}
    out = []
    for cid in ("cfg_a_partial_wave", "cfg_a_long_k"):
        p = frozen["predictions"][cid]
        row, mrow = p["row"], meas[cid]
        q = predict_with_rule(row, p["grid"], frozen["params"]["cfg_a"], frozen["clock_rule"],
                              frozen["fixed_us"]["cfg_a"], boundary_rate(row["m"], rates))
        m_us, m_cyc = float(mrow["meas_us"]), float(mrow["meas_cycles"])
        out.append(dict(case=cid, m=row["m"], n=row["n"], k=row["k"], meas_us=m_us,
                        v06_us=p["predicted_us"], v06_err=p["predicted_us"] / m_us - 1,
                        rule_us=q["predicted_us"], rule_err=q["predicted_us"] / m_us - 1,
                        v06_cycle_err=p["critical_cycles"] / m_cyc - 1,
                        rule_cycle_err=q["critical_cycles"] / m_cyc - 1,
                        rule_ghz=q["clock_ghz"], v06_ghz=p["clock_ghz"],
                        meas_ghz=float(mrow["meas_ghz"])))
    result = dict(note="not held-out: these sizes were measured in R17 and informed the rule",
                  rates=rates, rate_points=kinds, cases=out,
                  rule_code_sha256=model.sha(Path(__file__)))
    (run / "rule_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
