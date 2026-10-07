#!/usr/bin/env python3
"""V04 analysis: frozen transfer predictions vs measured warm single-call times, three configs.

  v04_analyze.py --input RUN [--output DIR]

Measured value = <cfg>_plain CUDA-event time (median over processes). The <cfg>_trace build gives
per CTA: tiles (words 27 + 28 for pingpong, 27 for cooperative), entry/exit clock64/globaltimer,
last-tile stage stamps (see v04_run.py TRACE_HEADER). Writes DIR/cases.csv, DIR/summary.json,
DIR/plots/v04_transfer.png (DIR defaults to RUN).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

WORDS = 32
ENTRY, FIRST_MMA, MAIN_END, EPI_START, STORE_DONE, EXITS = 0, 2, 4, 6, 8, (10, 11, 12)
SMID, TILES = 26, 27
PINGPONG = {"cfg_b"}


def median(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.median(values) if values else math.nan


def cv(values):
    return statistics.pstdev(values) / statistics.fmean(values) if len(values) > 1 else math.nan


def cta_records(run: Path, row, pingpong):
    """Per-CTA dicts from the raw trace of the timed warm call."""
    with gzip.open(run / row["folder"] / "stdout.txt.gz", "rt") as stream:
        call = next(json.loads(x) for x in stream if '"event":"call"' in x)
    trace = call["trace"]
    ctas = []
    for index in range(len(trace) // WORDS):
        w = trace[WORDS * index: WORDS * index + WORDS]
        cycle = lambda slot: w[2 * slot]
        nanos = lambda slot: w[2 * slot + 1]
        exit_slot = max(EXITS, key=cycle)
        groups = (0, 1) if pingpong else (0,)
        # Last-tile mainloop of each consumer warpgroup that ran at least one tile.
        mains = [cycle(MAIN_END + g) - cycle(FIRST_MMA + g) for g in groups if w[TILES + g] > 0]
        epis = [cycle(STORE_DONE + g) - cycle(EPI_START + g) for g in groups if w[TILES + g] > 0]
        ctas.append(dict(
            tiles=sum(w[TILES + g] for g in groups), smid=w[SMID], entry_ns=nanos(ENTRY),
            exit_ns=max(nanos(s) for s in EXITS),
            cycles=cycle(exit_slot) - cycle(ENTRY), ns=nanos(exit_slot) - nanos(ENTRY),
            last_main_cycles=median(mains), last_epi_cycles=median(epis),
            first_mma_cycles=cycle(FIRST_MMA) - cycle(ENTRY)))
    return call["elapsed_us"], ctas


def trace_metrics(event_us, ctas):
    rounds = max(c["tiles"] for c in ctas)
    critical = [c for c in ctas if c["tiles"] == rounds]
    exits = sorted(c["exit_ns"] for c in ctas)
    envelope_us = (exits[-1] - min(c["entry_ns"] for c in ctas)) / 1e3
    histogram = {}
    for c in ctas:
        histogram[c["tiles"]] = histogram.get(c["tiles"], 0) + 1
    return dict(
        event_us=event_us, grid_ctas=len(ctas), distinct_sms=len({c["smid"] for c in ctas}),
        rounds=rounds, tiles_total=sum(c["tiles"] for c in ctas),
        tile_histogram=json.dumps(dict(sorted(histogram.items()))),
        critical_ctas=len(critical),
        critical_cycles=median([c["cycles"] for c in critical]),
        critical_window_us=median([c["ns"] for c in critical]) / 1e3,
        critical_ghz=median([c["cycles"] / c["ns"] for c in critical]),
        envelope_us=envelope_us, host_gap_us=event_us - envelope_us,
        tail_us=(exits[-1] - median(exits)) / 1e3,
        last_main_cycles=median([c["last_main_cycles"] for c in critical]),
        last_epi_cycles=median([c["last_epi_cycles"] for c in critical]),
        first_mma_cycles_1tile=median([c["first_mma_cycles"] for c in ctas if c["tiles"] == 1]))


def analyze_case(run, rows, p):
    group = f"{p['config']}_{p['label']}"
    pingpong = p["config"] in PINGPONG
    plain = [r for r in rows if r["case_id"] == f"plain_{group}" and r["status"] == "measured"]
    trace = [r for r in rows if r["case_id"] == f"trace_{group}" and r["status"] == "measured"]
    plain_us = [r["calls"][0]["elapsed_us"] for r in plain]
    per_process = [trace_metrics(*cta_records(run, r, pingpong)) for r in trace]
    t = {key: median([m[key] for m in per_process]) for key in per_process[0]
         if not isinstance(per_process[0][key], str)}
    measured = median(plain_us)
    # Error attribution (us, sign = predicted - measured), using the trace build's terms.
    cycles_term = (p["cta_cycles"] - t["critical_cycles"]) / (t["critical_ghz"] * 1e3)
    clock_term = p["cta_cycles"] / 1e3 * (1 / p["clock_ghz"] - 1 / t["critical_ghz"])
    host_term = p["host_gap_us"] - t["host_gap_us"]
    rest_term = (p["event_us"] - measured) - cycles_term - clock_term - host_term
    # Cycle-model components (diagnostic): last-tile mainloop -> cycles/Ktile; per-tile rest.
    main_per_ktile = (t["last_main_cycles"] - p["c_0"]) / p["ktiles"]
    if pingpong:  # C = P + R*max(ML,E) + min(ML,E): per-tile period with frozen P and min term
        period = (t["critical_cycles"] - p["prefill_cycles"]
                  - min(p["mainloop_cycles_per_tile"], p["tile_extra_cycles"])) / t["rounds"]
        extra = period - t["last_main_cycles"]
    else:
        period = (t["critical_cycles"] - p["prefill_cycles"]) / t["rounds"]
        extra = period - t["last_main_cycles"]
    checks = [r["check"]["max_storage_reference_error"] for r in plain + trace]
    return dict(
        config=p["config"], label=p["label"], m=p["m"], n=p["n"], k=p["k"],
        processes_plain=len(plain), processes_trace=len(trace),
        predicted_us=p["event_us"], measured_us=measured, measured_cv=cv(plain_us),
        measured_min=min(plain_us), measured_max=max(plain_us),
        error=(p["event_us"] - measured) / measured,
        trace_us=t["event_us"], trace_over_plain=t["event_us"] / measured - 1,
        rounds_pred=p["rounds"], rounds_meas=t["rounds"], units_pred=p["units"],
        tiles_total_meas=t["tiles_total"], grid_ctas_meas=t["grid_ctas"],
        distinct_sms_meas=t["distinct_sms"],
        tile_histogram=per_process[0]["tile_histogram"],
        tile_histogram_same_all_processes=len({m["tile_histogram"] for m in per_process}) == 1,
        critical_ctas_pred=p["ctas_at_max_rounds"], critical_ctas_meas=t["critical_ctas"],
        cycles_pred=p["cta_cycles"], cycles_meas=t["critical_cycles"],
        cycles_error=p["cta_cycles"] / t["critical_cycles"] - 1,
        ghz_pred=p["clock_ghz"], ghz_meas=t["critical_ghz"],
        clock_error=p["clock_ghz"] / t["critical_ghz"] - 1,
        window_us_pred=p["cta_window_us"], window_us_meas=t["critical_window_us"],
        host_gap_pred=p["host_gap_us"], host_gap_meas=t["host_gap_us"],
        envelope_us_meas=t["envelope_us"], tail_us_meas=t["tail_us"], ktiles=p["ktiles"],
        ktile_cycles_pred=p["mainloop_cycles_per_ktile"], ktile_cycles_meas=main_per_ktile,
        tile_extra_pred=p["tile_extra_cycles"], tile_extra_meas=extra,
        tile_period_pred=(p["mainloop_cycles_per_tile"] + p["tile_extra_cycles"]
                          if not pingpong else max(p["mainloop_cycles_per_tile"],
                                                   p["tile_extra_cycles"])),
        tile_period_meas=period, last_epi_cycles=t["last_epi_cycles"],
        prefill_pred=p["prefill_cycles"], first_mma_cycles_1tile=t["first_mma_cycles_1tile"],
        pred_with_measured_clock_us=p["event_us"] - p["cta_window_us"]
        + p["cta_cycles"] / (t["critical_ghz"] * 1e3),
        attr_cycles_pct=100 * cycles_term / measured, attr_clock_pct=100 * clock_term / measured,
        attr_host_pct=100 * host_term / measured, attr_rest_pct=100 * rest_term / measured,
        max_check_error=max(checks) if checks else math.nan)


def analyze(run: Path, output: Path):
    frozen = json.loads((run / "source/v04-predictions.json").read_text())
    rows = [json.loads(x) for x in (run / "samples.jsonl").read_text().splitlines() if x]
    rows = [r for r in rows if r["set"] == "main"]
    cases = [analyze_case(run, rows, p) for p in frozen["predictions"]]
    for c in cases:
        c["error_with_measured_clock"] = c["pred_with_measured_clock_us"] / c["measured_us"] - 1
    per_config = {}
    for config in sorted({c["config"] for c in cases}):
        sub = [c for c in cases if c["config"] == config]
        stat = lambda key: dict(median_abs=median([abs(c[key]) for c in sub]),
                                max_abs=max(abs(c[key]) for c in sub))
        per_config[config] = dict(n=len(sub), error=stat("error"),
                                  cycles_error=stat("cycles_error"),
                                  clock_error=stat("clock_error"),
                                  error_with_measured_clock=stat("error_with_measured_clock"))
    variants = {(v["config"], v["label"]): v for v in frozen["variant_l2_request_bytes_not_scored"]}
    for c in cases:
        c["variant_l2req_error"] = variants[(c["config"], c["label"])]["event_us"] / c[
            "measured_us"] - 1
    environment = json.loads((run / "environment.json").read_text())
    summary = dict(
        predictions_sha256=environment.get("predictions_sha256"), environment=environment,
        failed=[r["case_id"] for r in rows if r["status"] != "measured"],
        all=dict(median_abs_error=median([abs(c["error"]) for c in cases]),
                 max_abs_error=max(abs(c["error"]) for c in cases)),
        per_config=per_config, cases=cases)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    plot(output, cases, frozen["base_rules"])
    print_table(summary)


def plot(output: Path, cases, rules):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"cfg_a": "#2a6fb0", "cfg_b": "#c0504d", "cfg_c": "#4f9a3a"}
    (output / "plots").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    xs = [c["measured_us"] for c in cases]
    lo, hi = min(xs) * 0.8, max(xs) * 1.25
    ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1], color="#ddd", alpha=0.5)
    ax.plot([lo, hi], [lo, hi], color="#888", lw=1)
    for config, color in colors.items():
        sub = [c for c in cases if c["config"] == config]
        ax.loglog([c["measured_us"] for c in sub], [c["predicted_us"] for c in sub], "o",
                  color=color, label=config)
    ax.set_xlabel("measured event time (us)")
    ax.set_ylabel("frozen prediction (us)")
    ax.set_title("prediction vs measurement (band +-10%)")
    ax.legend(fontsize=8)
    ax = axes[1]
    for config, color in colors.items():
        sub = [c for c in cases if c["config"] == config]
        ax.plot([100 * c["cycles_error"] for c in sub], [100 * c["clock_error"] for c in sub], "o",
                color=color, label=config)
    ax.axhline(0, color="#888", lw=1)
    ax.axvline(0, color="#888", lw=1)
    ax.set_xlabel("cycle-model error (%)")
    ax.set_ylabel("clock-rule error (%)")
    ax.set_title("error split (critical CTAs)")
    ax.legend(fontsize=8)
    ax = axes[2]
    grid = [10 * 1.1 ** i for i in range(60)]
    ax.semilogx(grid, [rules["clock"]["a"] - rules["clock"]["b"] * math.log(x) for x in grid],
                color="#888", lw=1, label="V02 rule")
    for config, color in colors.items():
        sub = [c for c in cases if c["config"] == config]
        ax.semilogx([c["window_us_meas"] for c in sub], [c["ghz_meas"] for c in sub], "o",
                    color=color, label=config)
    ax.set_xlabel("critical CTA window (us)")
    ax.set_ylabel("in-call clock (GHz)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "plots/v04_transfer.png", dpi=130)


def print_table(summary):
    print(f"{'config':6} {'label':17} {'pred':>8} {'meas':>8} {'cv%':>4} {'e%':>6} {'R p/m':>6} "
          f"{'cyc e%':>7} {'clk e%':>6} {'eMc%':>6} {'Kt p/m':>12} {'E p/m':>12} {'l2r e%':>6}")
    for c in summary["cases"]:
        print(f"{c['config']:6} {c['label']:17} {c['predicted_us']:8.2f} {c['measured_us']:8.2f} "
              f"{100 * c['measured_cv']:4.1f} {100 * c['error']:+6.1f} "
              f"{c['rounds_pred']:>3}/{c['rounds_meas']:<2} {100 * c['cycles_error']:+7.1f} "
              f"{100 * c['clock_error']:+6.1f} {100 * c['error_with_measured_clock']:+6.1f} "
              f"{c['ktile_cycles_pred']:5.0f}/{c['ktile_cycles_meas']:<6.0f} "
              f"{c['tile_extra_pred']:5.0f}/{c['tile_extra_meas']:<6.0f} "
              f"{100 * c['variant_l2req_error']:+6.1f}")
    for config, s in summary["per_config"].items():
        print(config, json.dumps(s))
    print("all", json.dumps(summary["all"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyze(args.input, args.output or args.input)


if __name__ == "__main__":
    main()
