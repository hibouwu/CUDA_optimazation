#!/usr/bin/env python3
"""V02: rule-based time prediction for the fixed CUTLASS GEMM, frozen before measurement.

Kernel: CUTLASS v3.9.2 FP16->FP32, tile 128x256x64, cluster 2x1x1, TMA warp-specialized
cooperative, 4 stage, ElementC=void, epilogue 128x32, -DNDEBUG, default persistent scheduler
(raster heuristic, swizzle 1), 1 CTA/SM on 132 SMs. Protocol: R00/R09 "warm" single call
(synchronized warm-up until the last 5 calls have CV<=2%, then one CUDA-event-timed call).

Model (every constant is read from an R08/R09 file below; nothing is fitted to the test sizes):

  tiles_m = ceil(M/128), tiles_n = ceil(N/256), Kt = ceil(K/64)
  U       = 2*ceil(tiles_m/2) * tiles_n            cluster 2x1 pads odd tiles_m   (R08)
  R       = ceil(U/132)                            rounds of the critical CTA     (R08)
  C       = P + R*(c_k*Kt + c_0 + E)               critical-CTA cycles            (R09)
  W       = C / f(W)                               CTA window, fixed point        (R09 clock rule)
  f(W)    = a - b*ln(W/us)                          in-call clock, GHz             (R09 fit)
  T_dev   = skew + W + post_store + tail           first entry -> last exit       (R09)
  T_event = host_gap + T_dev                        CUDA event, warm single call   (R09)

  P: prefill (entry -> first HGMMA), paid once per CTA; c_k, c_0: mainloop cycles per Ktile and
  per tile; E: per-tile cost beyond the mainloop in a persistent CTA (epilogue + transitions),
  from the 15- vs 16-tile CTAs of R09 8192^3. Mainloop is assumed compute-bound (1024 cycle/Ktile)
  at every size: R08 found no L2-supply rule, and R09 measured 1024 cycle/Ktile at 8192^3 too.

Usage: v02_predict.py --output configs/v02-predictions.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results/gh200_resource_campaign/access_rules"
R09 = RESULTS / "20261007-r09-clock-stages"
R08 = RESULTS / "20261007-r08-waves-l2"

TILE_M, TILE_N, TILE_K = 128, 256, 64
CLUSTER_M = 2
# R08 doc "配置": cudaOccupancyMaxActiveClusters = 66 (2x1) -> 132 resident CTAs.
RESIDENT_CTAS = 132

# Sizes never measured in R00/R07/R08/R09/V01 (checked against their samples/cases).
TEST_SIZES = [
    ("partial_wave", 1536, 2560, 4096),
    ("pad_crosses_wave", 1600, 2496, 4096),
    ("just_over_wave", 2304, 2048, 3072),
    ("long_k", 1024, 4096, 16384),
    ("edges_odd_pad", 2600, 3496, 2000),
    ("multi_tail_a", 3000, 5000, 4096),
    ("multi_tail_b", 3072, 6144, 3072),
    ("short_k", 4096, 4096, 512),
    ("mid_k1024", 4096, 8192, 1024),
    ("large_cube", 6144, 6144, 6144),
    ("large_flat", 10240, 10240, 4096),
]

# Already-measured points, replayed only as a sanity check of the assembled rules (not scored).
REPLAY = [
    ("R09 plain warm 2048^2 K512", 2048, 2048, 512, R09 / "cases.csv", "plain_m2048_k512"),
    ("R09 plain warm 2048^2 K2048", 2048, 2048, 2048, R09 / "cases.csv", "plain_m2048_k2048"),
    ("R09 plain warm 2048^2 K4096", 2048, 2048, 4096, R09 / "cases.csv", "plain_m2048_k4096"),
    ("R09 trace warm 2048^2 K8192", 2048, 2048, 8192, R09 / "cases.csv", "idle_m2048_k8192"),
    ("R09 trace warm 8192^3", 8192, 8192, 8192, R09 / "cases.csv", "idle_m8192_k8192"),
]

# R09 warm traced cases whose (CTA window, window clock) pairs define the clock rule.
CLOCK_CASES = ["trace_m2048_k256", "trace_m2048_k512", "trace_m2048_k1024", "trace_m2048_k2048",
               "trace_m2048_k4096", "idle_m2048_k8192", "idle_m8192_k8192"]
PREFILL_CASES = ["trace_m2048_k256", "trace_m2048_k512", "trace_m2048_k1024",
                 "trace_m2048_k2048", "trace_m2048_k4096"]


def ceil_div(a, b):
    return -(-a // b)


def r09_rows():
    """R09 cases.csv keyed by (case, call); 'warm' is the protocol used here."""
    with (R09 / "cases.csv").open() as stream:
        return {(r["case"], r["call"]): r for r in csv.DictReader(stream)}


def fit_line(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - slope * mx, slope


def marginal_tile_cycles_8192():
    """Per-CTA cycles of 16-tile minus 15-tile CTAs in R09 8192^3 warm calls (same call, same SM
    clock domain per CTA). Raw source: R09 samples/idle_m8192_k8192/clock-*/stdout.txt.gz, trace
    words per CTA: slot s -> (clock64, globaltimer) at [2s, 2s+1]; entry slot 0, exits 9/10/11."""
    per_process = []
    for folder in sorted((R09 / "samples/idle_m8192_k8192").glob("clock-*")):
        with gzip.open(folder / "stdout.txt.gz", "rt") as stream:
            calls = [json.loads(x) for x in stream if '"event":"call"' in x]
        trace = next(c for c in calls if c["label"] == "warm")["trace"]
        cycles = []
        for cta in range(len(trace) // 32):
            words = trace[32 * cta: 32 * cta + 32]
            cycles.append(max(words[18], words[20], words[22]) - words[0])
        cycles.sort()
        gap, cut = max((cycles[i + 1] - cycles[i], i) for i in range(len(cycles) - 1))
        low, high = cycles[: cut + 1], cycles[cut + 1:]
        per_process.append(statistics.median(high) - statistics.median(low))
    return statistics.median(per_process), per_process


def load_rules():
    rows = r09_rows()
    fits = json.loads((R09 / "summary.json").read_text())["fits"]["stages_m2048"]
    c_k = fits["mainloop_cycles"]["b"]
    c_0 = fits["mainloop_cycles"]["a"]
    marginal, per_process = marginal_tile_cycles_8192()
    prefill = statistics.median(float(rows[(c, "warm")]["prefill_cycles_med"])
                                for c in PREFILL_CASES)
    points = [(float(rows[(c, "warm")]["cta_window_ns_median_med"]) / 1e3,
               float(rows[(c, "warm")]["window_ghz_median_med"])) for c in CLOCK_CASES]
    a, slope = fit_line([math.log(w) for w, _ in points], [g for _, g in points])
    rules = dict(
        c_k=dict(value=c_k, unit="cycle/Ktile",
                 source="R09 summary.json fits.stages_m2048.mainloop_cycles.b"),
        c_0=dict(value=c_0, unit="cycle/tile",
                 source="R09 summary.json fits.stages_m2048.mainloop_cycles.a"),
        prefill=dict(value=prefill, unit="cycle/CTA",
                     source="R09 cases.csv prefill_cycles_med, median over warm "
                            + ",".join(PREFILL_CASES)),
        tile_extra=dict(value=marginal - (c_k * 128 + c_0), unit="cycle/tile",
                        marginal_tile_cycles=marginal, marginal_per_process=per_process,
                        source="R09 samples/idle_m8192_k8192 raw traces: median(16-tile CTA "
                               "cycles) - median(15-tile CTA cycles) - (c_k*128 + c_0)"),
        post_store=dict(value=fits["post_store"]["a"], unit="us",
                        source="R09 summary.json fits.stages_m2048.post_store.a"),
        entry_skew=dict(value=fits["entry_skew"]["a"], unit="us",
                        source="R09 summary.json fits.stages_m2048.entry_skew.a"),
        tail=dict(value=fits["tail_ns"]["a"], unit="us",
                  source="R09 summary.json fits.stages_m2048.tail_ns.a"),
        host_gap=dict(value=fits["host_gap_ns"]["a"], unit="us",
                      source="R09 summary.json fits.stages_m2048.host_gap_ns.a (warm)"),
        clock=dict(a=a, b=-slope, unit="GHz = a - b*ln(W/us)", points=points,
                   residuals=[g - (a + slope * math.log(w)) for w, g in points],
                   source="R09 cases.csv cta_window_ns_median_med / window_ghz_median_med, "
                          "warm: " + ",".join(CLOCK_CASES)),
        resident_ctas=dict(value=RESIDENT_CTAS, source="R08 doc 配置 (66 clusters x 2)"),
        rounds_rule=dict(source="R08 doc 结论: T ~ a + t*ceil(U/132), U after cluster padding"),
    )
    return rules


def clock_ghz(rules, window_us):
    return rules["clock"]["a"] - rules["clock"]["b"] * math.log(window_us)


def predict(rules, m, n, k):
    tiles_m, tiles_n, ktiles = ceil_div(m, TILE_M), ceil_div(n, TILE_N), ceil_div(k, TILE_K)
    tiles_m_padded = CLUSTER_M * ceil_div(tiles_m, CLUSTER_M)
    units = tiles_m_padded * tiles_n
    rounds = ceil_div(units, RESIDENT_CTAS)
    grid_ctas = min(units, RESIDENT_CTAS)
    ctas_at_max_rounds = units - (rounds - 1) * RESIDENT_CTAS
    v = lambda name: rules[name]["value"]
    mainloop_cycles = v("c_k") * ktiles + v("c_0")
    tile_cycles = mainloop_cycles + v("tile_extra")
    cta_cycles = v("prefill") + rounds * tile_cycles
    window_us = cta_cycles / 1.8e3
    for _ in range(100):  # fixed point W = C / f(W); f decreases slowly with W, converges fast
        window_us = cta_cycles / (clock_ghz(rules, window_us) * 1e3)
    ghz = clock_ghz(rules, window_us)
    device_us = v("entry_skew") + window_us + v("post_store") + v("tail")
    event_us = v("host_gap") + device_us
    return dict(m=m, n=n, k=k, tiles_m=tiles_m, tiles_n=tiles_n, tiles=tiles_m * tiles_n,
                tiles_m_padded=tiles_m_padded, units=units, rounds=rounds, grid_ctas=grid_ctas,
                ctas_at_max_rounds=ctas_at_max_rounds, ktiles=ktiles,
                mainloop_cycles_per_tile=mainloop_cycles, tile_cycles=tile_cycles,
                cta_cycles=cta_cycles, clock_ghz=ghz, cta_window_us=window_us,
                device_us=device_us, host_gap_us=v("host_gap"), event_us=event_us,
                tflops=2.0 * m * n * k / event_us / 1e6)


def replay(rules):
    rows = r09_rows()
    out = []
    for label, m, n, k, _, case in REPLAY:
        p = predict(rules, m, n, k)
        measured = float(rows[(case, "warm")]["event_us_median"])
        out.append(dict(label=label, case=case, predicted_us=p["event_us"],
                        measured_us=measured, error=p["event_us"] / measured - 1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rules = load_rules()
    predictions = [dict(label=label, **predict(rules, m, n, k)) for label, m, n, k in TEST_SIZES]
    header = f"{'label':18} {'M':>6} {'N':>6} {'K':>6} {'U':>5} {'R':>3} {'Kt':>4} "
    print(header + f"{'cycles':>9} {'GHz':>6} {'W us':>9} {'event us':>9}")
    for p in predictions:
        print(f"{p['label']:18} {p['m']:6} {p['n']:6} {p['k']:6} {p['units']:5} {p['rounds']:3} "
              f"{p['ktiles']:4} {p['cta_cycles']:9.0f} {p['clock_ghz']:6.3f} "
              f"{p['cta_window_us']:9.2f} {p['event_us']:9.2f}")
    checks = replay(rules)
    for c in checks:
        print(f"replay {c['label']:30} pred {c['predicted_us']:8.2f} meas {c['measured_us']:8.2f}"
              f" e {c['error']:+.3f}")
    if args.output:
        args.output.write_text(json.dumps(dict(
            model=__doc__.split("Usage:")[0].strip(), rules=rules, predictions=predictions,
            replay_prior_points_not_scored=checks), indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
