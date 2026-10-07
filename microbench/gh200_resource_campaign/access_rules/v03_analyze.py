#!/usr/bin/env python3
"""V03 analysis.

  v03_analyze.py calib --input RUN            -> RUN/calibration.csv, RUN/calibration_profiles.json
  v03_analyze.py power --input RUN            -> RUN/power.csv (NVML, sustained-load diagnostic)
  v03_analyze.py main  --input RUN
                                              -> RUN/cases.csv, RUN/summary.json, RUN/plots/*.png

Trace record per CTA (v03_run.py TRACE_HEADER): slots s -> words (2s, 2s+1) = (clock64,
globaltimer); entry slot 0, exits 9/10/11; word 24 smid, word 25 tile count; from word 32 one
(clock64, globaltimer) pair after mma_tail of each tile.
In-call clock of a config = median over processes of the median over critical CTAs (those with the
most tiles) of (exit - entry cycles) / (exit - entry ns), as in V02. Time-resolved clock: every
interval between consecutive stamps of one CTA (entry, tile ends, exit) gives cycles/ns at its
midpoint; intervals are pooled over CTAs into 10 equal bins of the call envelope (weight = ns).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

WORDS = 32 + 2 * 128
ENTRY, EXITS, SMID, TILES, TILE0 = 0, (9, 10, 11), 24, 25, 32
BINS = 10
REASONS = {0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal",
           0x80: "hw_power_brake", 0x1: "gpu_idle", 0x100: "display_clock"}


def median(values):
    values = [v for v in values if v == v]
    return statistics.median(values) if values else math.nan


def cv(values):
    return statistics.pstdev(values) / statistics.fmean(values) if len(values) > 1 else math.nan


def load_rows(run: Path, set_name):
    rows = [json.loads(x) for x in (run / "samples.jsonl").read_text().splitlines() if x]
    return [r for r in rows if r["set"] == set_name]


def read_stdout(run: Path, row):
    with gzip.open(run / row["folder"] / "stdout.txt.gz", "rt") as stream:
        return [json.loads(x) for x in stream if x.strip()]


def cta_records(trace):
    ctas = []
    for index in range(len(trace) // WORDS):
        w = trace[WORDS * index: WORDS * index + WORDS]
        exit_slot = max(EXITS, key=lambda s: w[2 * s])
        tiles = w[TILES]
        stamps = [(w[0], w[1])]
        stamps += [(w[TILE0 + 2 * t], w[TILE0 + 2 * t + 1]) for t in range(min(tiles, 128))]
        stamps.append((w[2 * exit_slot], w[2 * exit_slot + 1]))
        ctas.append(dict(tiles=tiles, smid=w[SMID], entry_ns=w[1], exit_ns=w[2 * exit_slot + 1],
                         cycles=w[2 * exit_slot] - w[0], ns=w[2 * exit_slot + 1] - w[1],
                         stamps=stamps))
    return ctas


def trace_metrics(event_us, ctas):
    rounds = max(c["tiles"] for c in ctas)
    critical = [c for c in ctas if c["tiles"] == rounds]
    start = min(c["entry_ns"] for c in ctas)
    end = max(c["exit_ns"] for c in ctas)
    span = end - start
    bins = [[0.0, 0.0] for _ in range(BINS)]  # (cycles, ns) per bin
    for c in ctas:
        for (c0, t0), (c1, t1) in zip(c["stamps"], c["stamps"][1:]):
            if t1 <= t0:
                continue
            b = min(BINS - 1, int(BINS * ((t0 + t1) / 2 - start) / span))
            bins[b][0] += c1 - c0
            bins[b][1] += t1 - t0
    active = [0.0] * BINS  # CTA-busy ns per bin / bin length -> active CTAs
    for c in ctas:
        for b in range(BINS):
            lo, hi = start + span * b / BINS, start + span * (b + 1) / BINS
            active[b] += max(0.0, min(hi, c["exit_ns"]) - max(lo, c["entry_ns"])) / (hi - lo)
    histogram = {}
    for c in ctas:
        histogram[c["tiles"]] = histogram.get(c["tiles"], 0) + 1
    return dict(
        event_us=event_us, grid_ctas=len(ctas), distinct_sms=len({c["smid"] for c in ctas}),
        rounds=rounds, tiles_total=sum(c["tiles"] for c in ctas),
        tile_histogram=json.dumps(dict(sorted(histogram.items()))), critical_ctas=len(critical),
        critical_cycles=median([c["cycles"] for c in critical]),
        window_us=median([c["ns"] for c in critical]) / 1e3,
        ghz=median([c["cycles"] / c["ns"] for c in critical]),
        all_cta_ghz=median([c["cycles"] / c["ns"] for c in ctas]),
        envelope_us=span / 1e3, host_gap_us=event_us - span / 1e3,
        bin_ghz=[cy / ns if ns else math.nan for cy, ns in bins], bin_active=active)


def process_metrics(run, row):
    lines = read_stdout(run, row)
    call = next(x for x in lines if x.get("event") == "call")
    m = trace_metrics(call["elapsed_us"], cta_records(call["trace"]))
    m["probe_ghz"] = call["probe_ghz_median"]
    m["setup"] = next(x for x in lines if x.get("event") == "setup")
    nvml = next((x for x in lines if x.get("event") == "nvml"), None)
    if nvml and nvml["ok"] and nvml["rows"]:
        rows = nvml["rows"]  # [t_ns, gpu_w, module_w, avg_w, sm_mhz, reasons]
        # NVML refreshes slowly: summarize the last second of the sustained sequence.
        inside = [r for r in rows if r[0] >= nvml["call_start_ns"] - 1e9] or rows
        reasons = 0
        for r in rows:
            reasons |= int(r[5])
        m["nvml"] = dict(
            samples=len(rows), samples_in_call=len(inside),
            warm_seq_ms=nvml["call_end_ns"] / 1e6,
            gpu_w_max=max(r[1] for r in rows), module_w_max=max(r[2] for r in rows),
            gpu_w_median=median([r[1] for r in rows]),
            module_w_median=median([r[2] for r in rows]),
            gpu_w_last_s=median([r[1] for r in inside]),
            module_w_last_s=median([r[2] for r in inside]),
            sm_mhz_min=min(r[4] for r in rows), sm_mhz_last_s=median([r[4] for r in inside]),
            sustained_call_us=median(call["warmup_us"][-50:]),
            reasons=[name for bit, name in REASONS.items() if reasons & bit],
            sw_power_cap_fraction=statistics.fmean([1.0 if int(r[5]) & 0x4 else 0.0
                                                    for r in inside]))
    return m


def calib(run: Path):
    """Per-config in-call clock of the calibration set -> calibration.csv (input of the fit)."""
    import v03_run
    rows = [r for r in load_rows(run, "calib") if r["status"] == "measured"]
    failed = [r["case_id"] for r in load_rows(run, "calib") if r["status"] != "measured"]
    out, profiles = [], {}
    for label, m, n, k, swizzle, sm_count in v03_run.CALIBRATION:
        per = [process_metrics(run, r) for r in rows if r["case_id"] == f"trace_{label}"]
        if not per:
            continue
        setup = per[0]["setup"]
        bins = [median([p["bin_ghz"][b] for p in per]) for b in range(BINS)]
        active = [median([p["bin_active"][b] for p in per]) for b in range(BINS)]
        out.append(dict(
            label=label, m=m, n=n, k=k, swizzle=swizzle, sm_count=sm_count, processes=len(per),
            raster=setup["raster_actual"], log_swizzle=setup["log_swizzle"],
            units=setup["blocks_per_problem"], grid_meas=per[0]["grid_ctas"],
            distinct_sms=per[0]["distinct_sms"], rounds_meas=per[0]["rounds"],
            tile_histogram=per[0]["tile_histogram"],
            event_us=median([p["event_us"] for p in per]),
            window_us_meas=median([p["window_us"] for p in per]),
            ghz_meas=median([p["ghz"] for p in per]), ghz_cv=cv([p["ghz"] for p in per]),
            ghz_min=min(p["ghz"] for p in per), ghz_max=max(p["ghz"] for p in per),
            all_cta_ghz=median([p["all_cta_ghz"] for p in per]),
            critical_cycles=median([p["critical_cycles"] for p in per]),
            probe_ghz=median([p["probe_ghz"] for p in per]),
            ghz_first_bin=bins[0], ghz_last_bin=bins[-1]))
        profiles[label] = dict(bin_ghz=bins, bin_active_ctas=active,
                               envelope_us=median([p["envelope_us"] for p in per]))
    write_csv(run / "calibration.csv", out)
    (run / "calibration_profiles.json").write_text(json.dumps(
        dict(failed=failed, profiles=profiles), indent=2) + "\n")
    for r in out:
        print(f"{r['label']:22} G {r['grid_meas']:3} R {r['rounds_meas']:3} "
              f"W {r['window_us_meas']:8.1f} GHz {r['ghz_meas']:.3f} (cv {100 * r['ghz_cv']:.1f}%)"
              f" first/last bin {r['ghz_first_bin']:.3f}/{r['ghz_last_bin']:.3f}")
    print("failed", failed)


def power(run: Path):
    """NVML diagnostic set (sustained load) and the warm-protocol clock of the same configs."""
    import v03_run
    by_label = {c[0]: c for c in v03_run.CALIBRATION}
    warm = [r for r in load_rows(run, "calib") if r["status"] == "measured"]
    rows = [r for r in load_rows(run, "nvml") if r["status"] == "measured"]
    out = []
    for label in v03_run.POWER:
        per = [process_metrics(run, r) for r in rows if r["case_id"] == f"nvml_{label}"]
        base = [process_metrics(run, r) for r in warm if r["case_id"] == f"trace_{label}"]
        if not per or "nvml" not in per[0]:
            continue
        row = dict(label=label, m=by_label[label][1], n=by_label[label][2],
                   k=by_label[label][3], swizzle=by_label[label][4], sm_count=by_label[label][5],
                   processes=len(per), ghz_warm=median([p["ghz"] for p in base]),
                   ghz_sustained=median([p["ghz"] for p in per]),
                   window_us_sustained=median([p["window_us"] for p in per]))
        for key in ("sustained_call_us", "module_w_last_s", "gpu_w_last_s", "module_w_max",
                    "gpu_w_max", "sm_mhz_last_s", "sm_mhz_min", "sw_power_cap_fraction",
                    "samples", "warm_seq_ms"):
            row[key] = median([p["nvml"][key] for p in per])
        row["reasons"] = "|".join(sorted({x for p in per for x in p["nvml"]["reasons"]}))
        out.append(row)
        print(row)
    if out:
        write_csv(run / "power.csv", out)


def write_csv(path: Path, rows):
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main_set(run: Path):
    frozen = json.loads((run / "source/v03-predictions.json").read_text())
    rows = load_rows(run, "main")
    failed = [r["case_id"] for r in rows if r["status"] != "measured"]
    cases = []
    for group, predictions in (("heldout", frozen["heldout"]),
                               ("v02_secondary", frozen["v02_secondary"])):
        for p in predictions:
            label = p["label"]
            plain = [r for r in rows if r["case_id"] == f"plain_{label}"
                     and r["status"] == "measured"]
            trace = [r for r in rows if r["case_id"] == f"trace_{label}"
                     and r["status"] == "measured"]
            plain_us = [r["calls"][0]["elapsed_us"] for r in plain]
            per = [process_metrics(run, r) for r in trace]
            measured = median(plain_us)
            t = {key: median([x[key] for x in per]) for key in
                 ("event_us", "critical_cycles", "window_us", "ghz", "host_gap_us", "rounds",
                  "grid_ctas", "critical_ctas")}
            checks = [r["check"]["max_storage_reference_error"] for r in plain + trace]
            case = dict(
                group=group, label=label, m=p["m"], n=p["n"], k=p["k"], swizzle=p["swizzle"],
                processes_plain=len(plain), processes_trace=len(trace),
                predicted_us=p["event_us"], measured_us=measured, measured_cv=cv(plain_us),
                measured_min=min(plain_us), measured_max=max(plain_us),
                error=p["event_us"] / measured - 1,
                v02_rule_predicted_us=p.get("v02_rule_event_us", math.nan),
                v02_rule_error=p.get("v02_rule_event_us", math.nan) / measured - 1,
                trace_over_plain=t["event_us"] / measured - 1,
                rounds_pred=p["rounds"], rounds_meas=t["rounds"],
                grid_pred=p["grid_ctas"], grid_meas=t["grid_ctas"],
                critical_ctas_pred=p["ctas_at_max_rounds"], critical_ctas_meas=t["critical_ctas"],
                tile_histogram=per[0]["tile_histogram"] if per else "",
                cycles_pred=p["cta_cycles"], cycles_meas=t["critical_cycles"],
                cycles_error=p["cta_cycles"] / t["critical_cycles"] - 1,
                ghz_pred=p["clock_ghz"], ghz_meas=t["ghz"],
                clock_error=p["clock_ghz"] / t["ghz"] - 1,
                window_us_pred=p["cta_window_us"], window_us_meas=t["window_us"],
                host_gap_meas=t["host_gap_us"],
                phi=p["phi"], dram_tbs=p["feat_dram"], mu=p["mu"],
                pred_with_measured_clock_us=p["event_us"] - p["cta_window_us"]
                + p["cta_cycles"] / (t["ghz"] * 1e3),
                max_check_error=max(checks) if checks else math.nan)
            case["error_with_measured_clock"] = case["pred_with_measured_clock_us"] / measured - 1
            cases.append(case)
    held = [c for c in cases if c["group"] == "heldout"]
    sec = [c for c in cases if c["group"] == "v02_secondary"]
    stats = lambda cs, key: dict(median_abs=median([abs(c[key]) for c in cs]),
                                 max_abs=max(abs(c[key]) for c in cs))
    summary = dict(
        environment=json.loads((run / "environment.json").read_text()), failed=failed,
        clock_rule=frozen["clock_rule"], frozen_utc=frozen["frozen_utc"],
        heldout=stats(held, "error"), heldout_clock=stats(held, "clock_error"),
        heldout_with_measured_clock=stats(held, "error_with_measured_clock"),
        target=dict(median_abs=0.05, max_abs=0.10),
        passed=stats(held, "error")["median_abs"] <= 0.05 and stats(held, "error")["max_abs"]
        <= 0.10,
        v02_secondary_new_rule=stats(sec, "error"), v02_secondary_v02_rule=stats(
            sec, "v02_rule_error"), cases=cases)
    with (run / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    (run / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    plot(run, cases, frozen)
    for c in cases:
        print(f"{c['group'][:4]} {c['label']:20} pred {c['predicted_us']:8.2f} meas "
              f"{c['measured_us']:8.2f} cv {100 * c['measured_cv']:4.1f} e {100 * c['error']:+6.1f}"
              f" R {c['rounds_pred']}/{c['rounds_meas']} GHz {c['ghz_pred']:.3f}/{c['ghz_meas']:.3f}"
              f" cyc {100 * c['cycles_error']:+5.1f} e|f {100 * c['error_with_measured_clock']:+5.1f}"
              f" v02 {100 * c['v02_rule_error']:+6.1f}")
    for key in ("heldout", "heldout_clock", "heldout_with_measured_clock",
                "v02_secondary_new_rule", "v02_secondary_v02_rule"):
        print(key, summary[key])
    print("passed", summary["passed"], "failed", failed)


def plot(run: Path, cases, frozen):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    (run / "plots").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for group, color in (("heldout", "#2a6fb0"), ("v02_secondary", "#c0792a")):
        cs = [c for c in cases if c["group"] == group]
        ax.loglog([c["measured_us"] for c in cs], [c["predicted_us"] for c in cs], "o",
                  color=color, label=group)
        for c in cs:
            ax.annotate(c["label"], (c["measured_us"], c["predicted_us"]), fontsize=6,
                        xytext=(3, -7), textcoords="offset points")
    xs = [c["measured_us"] for c in cases]
    lo, hi = min(xs) * 0.8, max(xs) * 1.25
    ax.plot([lo, hi], [lo, hi], color="#888", lw=1)
    ax.fill_between([lo, hi], [lo * 0.95, hi * 0.95], [lo * 1.05, hi * 1.05], color="#ddd",
                    alpha=0.6)
    ax.set_xlabel("measured event time (us)")
    ax.set_ylabel("frozen prediction (us)")
    ax.set_title("V03 prediction vs measurement (band +-5%)")
    ax.legend(fontsize=8)
    ax = axes[1]
    fit = frozen["candidate_fits"][frozen["clock_rule"]["name"]]
    cal = fit["per_config"]
    ax.semilogx([math.exp(c["feats"]["lnW"]) for c in cal], [c["ghz_meas"] for c in cal], "s",
                color="#999", label="calibration (fit)")
    ax.semilogx([c["window_us_meas"] for c in cases], [c["ghz_meas"] for c in cases], "o",
                color="#2a6fb0", label="main: measured")
    ax.semilogx([c["window_us_pred"] for c in cases], [c["ghz_pred"] for c in cases], "x",
                color="#c03030", label="main: rule")
    ax.set_xlabel("critical-CTA window (us)")
    ax.set_ylabel("in-call clock (GHz)")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.axhline(0, color="#888", lw=1)
    ax.semilogx([math.exp(c["feats"]["lnW"]) for c in cal], [100 * c["residual"] for c in cal],
                "s", color="#999", label="calibration residual")
    ax.semilogx([c["window_us_meas"] for c in cases], [100 * c["clock_error"] for c in cases],
                "o", color="#2a6fb0", label="main: clock error")
    ax.set_xlabel("critical-CTA window (us)")
    ax.set_ylabel("rule / measured - 1 (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "plots/v03_prediction.png", dpi=130)


def plot_profiles(run: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    data = json.loads((run / "calibration_profiles.json").read_text())["profiles"]
    (run / "plots").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for label, p in data.items():
        xs = [(b + 0.5) / BINS * p["envelope_us"] for b in range(BINS)]
        axes[0].semilogx(xs, p["bin_ghz"], "-", marker=".", lw=1, label=label)
        axes[1].plot([(b + 0.5) / BINS for b in range(BINS)], p["bin_ghz"], "-", lw=1)
    axes[0].set_xlabel("time in call (us)")
    axes[0].set_ylabel("clock in bin (GHz)")
    axes[0].legend(fontsize=5, ncol=2)
    axes[1].set_xlabel("fraction of call envelope")
    axes[1].set_ylabel("clock in bin (GHz)")
    fig.tight_layout()
    fig.savefig(run / "plots/v03_calibration_profiles.png", dpi=130)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("calib", "power", "main"))
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if args.step == "calib":
        calib(args.input)
        plot_profiles(args.input)
    elif args.step == "power":
        power(args.input)
    else:
        main_set(args.input)


if __name__ == "__main__":
    main()
