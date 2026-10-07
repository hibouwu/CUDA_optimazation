#!/usr/bin/env python3
"""V02 analysis: frozen predictions vs measured warm single-call times of the fixed CUTLASS GEMM.

  v02_analyze.py --input RUN [--output DIR]

Measured value = v02_plain CUDA-event time (median over processes). The v02_trace build gives,
per CTA: tiles processed (word 25), entry/exit clock64 and globaltimer, last-tile stage stamps.
Writes DIR/cases.csv, DIR/summary.json, DIR/plots/*.png (DIR defaults to RUN).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

WORDS = 32  # per-CTA trace record, see v02_run.py TRACE_HEADER
ENTRY, FIRST_MMA, MAIN_END, EPI_START, STORE_DONE, EXITS = 0, 2, 4, 6, 8, (9, 10, 11)
SMID, TILES = 24, 25


def median(values):
    return statistics.median(values) if values else math.nan


def cv(values):
    return statistics.pstdev(values) / statistics.fmean(values) if len(values) > 1 else math.nan


def load_samples(run: Path):
    rows = [json.loads(x) for x in (run / "samples.jsonl").read_text().splitlines() if x]
    return [r for r in rows if r["set"] == "main"]


def cta_records(run: Path, row):
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
        ctas.append(dict(
            tiles=w[TILES], smid=w[SMID], entry_ns=nanos(ENTRY),
            exit_ns=max(nanos(s) for s in EXITS),
            cycles=cycle(exit_slot) - cycle(ENTRY), ns=nanos(exit_slot) - nanos(ENTRY),
            last_main_cycles=cycle(MAIN_END) - cycle(FIRST_MMA),
            last_epi_cycles=cycle(STORE_DONE) - cycle(EPI_START),
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
        all_cta_ghz=median([c["cycles"] / c["ns"] for c in ctas]),
        envelope_us=envelope_us, host_gap_us=event_us - envelope_us,
        tail_us=(exits[-1] - median(exits)) / 1e3,
        last_main_cycles=median([c["last_main_cycles"] for c in critical]),
        last_epi_cycles=median([c["last_epi_cycles"] for c in critical]),
        first_mma_cycles_1tile=median([c["first_mma_cycles"] for c in ctas if c["tiles"] == 1]))


def analyze(run: Path, output: Path):
    frozen = json.loads((run / "source/v02-predictions.json").read_text())
    rules = frozen["rules"]
    rows = load_samples(run)
    failed = [r["case_id"] for r in rows if r["status"] != "measured"]
    cases = []
    for p in frozen["predictions"]:
        label = p["label"]
        plain = [r for r in rows if r["case_id"] == f"plain_{label}" and r["status"] == "measured"]
        trace = [r for r in rows if r["case_id"] == f"trace_{label}" and r["status"] == "measured"]
        plain_us = [r["calls"][0]["elapsed_us"] for r in plain]
        per_process = [trace_metrics(*cta_records(run, r)) for r in trace]
        t = {key: median([m[key] for m in per_process]) for key in per_process[0]
             if not isinstance(per_process[0][key], str)}
        t["tile_histogram"] = per_process[0]["tile_histogram"]
        histograms_agree = len({m["tile_histogram"] for m in per_process}) == 1
        measured = median(plain_us)
        # Error attribution (us, sign = predicted - measured), using the trace build's terms.
        cycles_term = (p["cta_cycles"] - t["critical_cycles"]) / (t["critical_ghz"] * 1e3)
        clock_term = p["cta_cycles"] / 1e3 * (1 / p["clock_ghz"] - 1 / t["critical_ghz"])
        host_term = p["host_gap_us"] - t["host_gap_us"]
        rest_term = (p["event_us"] - measured) - cycles_term - clock_term - host_term
        checks = [r["check"]["max_storage_reference_error"] for r in plain + trace]
        case = dict(
            label=label, m=p["m"], n=p["n"], k=p["k"], processes_plain=len(plain),
            processes_trace=len(trace), predicted_us=p["event_us"], measured_us=measured,
            measured_cv=cv(plain_us), measured_min=min(plain_us), measured_max=max(plain_us),
            error=(p["event_us"] - measured) / measured,
            trace_us=t["event_us"], trace_over_plain=t["event_us"] / measured - 1,
            rounds_pred=p["rounds"], rounds_meas=t["rounds"], units_pred=p["units"],
            tiles_total_meas=t["tiles_total"], grid_ctas_pred=p["grid_ctas"],
            grid_ctas_meas=t["grid_ctas"], tile_histogram=t["tile_histogram"],
            tile_histogram_same_all_processes=histograms_agree,
            critical_ctas_pred=p["ctas_at_max_rounds"], critical_ctas_meas=t["critical_ctas"],
            cycles_pred=p["cta_cycles"], cycles_meas=t["critical_cycles"],
            cycles_error=p["cta_cycles"] / t["critical_cycles"] - 1,
            ghz_pred=p["clock_ghz"], ghz_meas=t["critical_ghz"],
            ghz_all_ctas_meas=t["all_cta_ghz"], clock_error=p["clock_ghz"] / t["critical_ghz"] - 1,
            window_us_pred=p["cta_window_us"], window_us_meas=t["critical_window_us"],
            host_gap_pred=p["host_gap_us"], host_gap_meas=t["host_gap_us"],
            envelope_us_meas=t["envelope_us"], tail_us_meas=t["tail_us"],
            ktiles=p["ktiles"],
            last_main_cycles_per_ktile=(t["last_main_cycles"] - rules["c_0"]["value"])
            / p["ktiles"],
            last_epi_cycles=t["last_epi_cycles"],
            first_mma_cycles_1tile=t["first_mma_cycles_1tile"],
            tile_overhead_meas=(t["critical_cycles"] - rules["prefill"]["value"]) / t["rounds"]
            - (rules["c_k"]["value"] * p["ktiles"] + rules["c_0"]["value"]),
            # Post-hoc diagnostic, not a prediction: frozen cycles and fixed terms, measured clock.
            pred_with_measured_clock_us=p["event_us"] - p["cta_window_us"]
            + p["cta_cycles"] / (t["critical_ghz"] * 1e3),
            attr_cycles_pct=100 * cycles_term / measured,
            attr_clock_pct=100 * clock_term / measured,
            attr_host_pct=100 * host_term / measured, attr_rest_pct=100 * rest_term / measured,
            max_check_error=max(checks) if checks else math.nan)
        cases.append(case)
    errors = [abs(c["error"]) for c in cases]
    summary = dict(
        predictions_sha256=json.loads((run / "environment.json").read_text())
        .get("predictions_sha256"),
        environment=json.loads((run / "environment.json").read_text()),
        failed=failed, median_abs_error=median(errors), max_abs_error=max(errors),
        target=dict(median_abs_error=0.10, max_abs_error=0.20),
        passed=median(errors) <= 0.10 and max(errors) <= 0.20,
        clock_rule=dict(a=rules["clock"]["a"], b=rules["clock"]["b"]), cases=cases)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    plot(output, cases, rules)
    print_table(summary)


def plot(output: Path, cases, rules):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    (output / "plots").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    xs = [c["measured_us"] for c in cases]
    ax.loglog(xs, [c["predicted_us"] for c in cases], "o", color="#2a6fb0")
    lo, hi = min(xs) * 0.8, max(xs) * 1.25
    ax.plot([lo, hi], [lo, hi], color="#888", lw=1)
    ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1], color="#ddd", alpha=0.5)
    for c in cases:
        ax.annotate(c["label"], (c["measured_us"], c["predicted_us"]), fontsize=7,
                    xytext=(3, -8), textcoords="offset points")
    ax.set_xlabel("measured event time (us)")
    ax.set_ylabel("frozen prediction (us)")
    ax.set_title("prediction vs measurement (band: +-10%)")
    ax = axes[1]
    w = [x for x, _ in rules["clock"]["points"]]
    ax.semilogx(w, [g for _, g in rules["clock"]["points"]], "s", color="#999", label="R09 (fit)")
    grid = [math.exp(math.log(min(w)) + i * (math.log(2e3) - math.log(min(w))) / 50)
            for i in range(51)]
    ax.semilogx(grid, [rules["clock"]["a"] - rules["clock"]["b"] * math.log(x) for x in grid],
                color="#888", lw=1, label="rule")
    ax.semilogx([c["window_us_meas"] for c in cases], [c["ghz_meas"] for c in cases], "o",
                color="#2a6fb0", label="V02 measured (critical CTAs)")
    ax.set_xlabel("CTA window (us)")
    ax.set_ylabel("in-call clock (GHz)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "plots/v02_prediction.png", dpi=130)


def print_table(summary):
    print(f"{'label':18} {'pred':>8} {'meas':>8} {'cv%':>5} {'e%':>6} {'R p/m':>6} "
          f"{'GHz p/m':>11} {'cyc e%':>7} {'clk e%':>7} {'host':>5} {'rest':>5}")
    for c in summary["cases"]:
        print(f"{c['label']:18} {c['predicted_us']:8.2f} {c['measured_us']:8.2f} "
              f"{100 * c['measured_cv']:5.1f} {100 * c['error']:+6.1f} "
              f"{c['rounds_pred']:>3}/{c['rounds_meas']:<2} "
              f"{c['ghz_pred']:5.3f}/{c['ghz_meas']:5.3f} {100 * c['cycles_error']:+7.1f} "
              f"{100 * c['clock_error']:+7.1f} {c['attr_host_pct']:+5.1f} "
              f"{c['attr_rest_pct']:+5.1f}")
    print("median |e| %.3f  max |e| %.3f  passed %s" % (
        summary["median_abs_error"], summary["max_abs_error"], summary["passed"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyze(args.input, args.output or args.input)


if __name__ == "__main__":
    main()
