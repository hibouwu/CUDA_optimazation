#!/usr/bin/env python3
"""R08 offline analysis: wave/tail fits, per-CTA stamps, traversal table -> cases.csv/summary.json.

  r08_analyze.py --input RUN [--output DIR]
Pilot samples (case_id pilot_*) are excluded. Plots need matplotlib (optional).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st

import numpy as np

SMS = 132
TILE_BYTES_A, TILE_BYTES_B = 128 * 64 * 2, 64 * 256 * 2  # per tile per Ktile, FP16
KTILE = 64


def load(run: Path):
    rows = [json.loads(x) for x in (run / "samples.jsonl").read_text().splitlines() if x]
    rows = [r for r in rows if not r["case_id"].startswith("pilot_")]
    for r in rows:
        if r["status"] != "measured" or r["returncode"] != 0 or r["check"]["status"] != "ok":
            raise ValueError(f"bad sample {r['case_id']} {r['trial']}")
        if r["check"]["max_storage_reference_error"] != 0:
            raise ValueError(f"numeric check {r['case_id']} {r['trial']}")
    return rows


def describe(values):
    mean = st.mean(values)
    return dict(n=len(values), median=st.median(values), min=min(values), max=max(values),
                cv=(st.stdev(values) / mean if len(values) > 1 else 0.0))


def lstsq(columns, y):
    """y = X beta; returns beta, residuals."""
    x = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(x, np.asarray(y), rcond=None)
    return beta, np.asarray(y) - x @ beta


def group(rows, prefix):
    cases = {}
    for r in rows:
        if r["case_id"].startswith(prefix):
            cases.setdefault(r["case_id"], []).append(r)
    return cases


def wave_table(rows):
    out = []
    for case_id, samples in sorted(group(rows, "wave_").items()):
        c = samples[0]["config"]
        units = c["blocks_per_problem"]  # scheduled tiles incl. cluster padding
        real = math.ceil(c["m"] / 128) * math.ceil(c["n"] / 256)
        call = describe([s["time"]["call_median_us"] for s in samples])
        graph = describe([s["time"]["graph_us"] for s in samples])
        ghz = st.median(s["time"]["ghz_after_median"] for s in samples)
        ghz_graph = st.median(s["time"]["graph_ghz_after"] for s in samples)
        out.append(dict(
            case_id=case_id, m=c["m"], n=c["n"], k=c["k"], tiles=real, sched_tiles=units,
            sched_tiles_m=c["sched_tiles_m"], sched_tiles_n=c["sched_tiles_n"],
            grid=c["grid"], raster=c["raster_actual"], max_active_clusters=c["max_active_clusters"],
            rounds=math.ceil(units / SMS), waves_frac=units / SMS,
            call_us=call["median"], call_cv=call["cv"], processes=call["n"],
            graph_us=graph["median"], graph_cv=graph["cv"],
            ghz_after_call=ghz, ghz_after_graph=ghz_graph,
            call_us_per_tile=call["median"] / real, tflops=2 * c["m"] * c["n"] * c["k"]
            / call["median"] / 1e6))
    return sorted(out, key=lambda r: (r["sched_tiles"], r["m"]))


def wave_fits(table, key):
    """Compare T = a + rounds*t (discrete) with T = a + (tiles/132)*t (fractional) and both."""
    y = [r[key] for r in table]
    ones = [1.0] * len(table)
    rounds = [r["rounds"] for r in table]
    frac = [r["waves_frac"] for r in table]
    fits = {}
    for name, cols in (("discrete", [ones, rounds]), ("fractional", [ones, frac]),
                       ("mixed", [ones, rounds, frac])):
        beta, res = lstsq(cols, y)
        fits[name] = dict(coef=[float(b) for b in beta], rms_us=float(np.sqrt(np.mean(res ** 2))),
                          max_abs_us=float(np.max(np.abs(res))),
                          residuals={r["case_id"]: float(e) for r, e in zip(table, res)})
    return fits


def stamp_table(run: Path, rows):
    out = []
    for case_id, samples in sorted(group(rows, "stamp_").items()):
        per_launch = []
        for s in samples:
            folder = run / "samples" / case_id / f"trial-{s['trial']:02d}"
            for line in (folder / "stdout.txt").read_text().splitlines():
                e = json.loads(line)
                if e.get("event") != "stamp":
                    continue
                t0 = min(e["start_ns"])
                start = np.array(e["start_ns"]) - t0
                end = np.array(e["end_ns"]) - t0
                tiles = np.array(e["tiles"])
                busy = end - start
                entry = dict(event_us=e["event_us"], span_us=float(end.max()) / 1e3,
                             start_spread_us=float(start.max()) / 1e3,
                             distinct_sm=len(set(e["smid"])), ctas=len(tiles),
                             ghz_after=e["ghz_after"])
                for count in sorted(set(tiles.tolist())):
                    mask = tiles == count
                    entry[f"busy_us_{count}tiles"] = float(np.median(busy[mask])) / 1e3
                    entry[f"end_us_{count}tiles_max"] = float(end[mask].max()) / 1e3
                    entry[f"ctas_{count}tiles"] = int(mask.sum())
                per_launch.append(entry)
        c = samples[0]["config"]
        keys = sorted({k for e in per_launch for k in e})
        merged = {k: st.median(e[k] for e in per_launch if k in e) for k in keys}
        out.append(dict(case_id=case_id, m=c["m"], n=c["n"], sched_tiles=c["blocks_per_problem"],
                        launches=len(per_launch), **merged))
    return out


def l2_table(rows, intercept_us):
    out = []
    for case_id, samples in sorted(group(rows, "l2_").items()):
        c = samples[0]["config"]
        call = describe([s["time"]["call_median_us"] for s in samples])
        ghz = st.median(s["time"]["ghz_after_median"] for s in samples)
        units = c["blocks_per_problem"]
        rounds = math.ceil(units / SMS)
        ktiles = c["k"] // KTILE
        # Mainloop time per Ktile on a critical CTA: (T - a) / (rounds * Ktiles).
        per_ktile_ns = (call["median"] - intercept_us) * 1e3 / (rounds * ktiles)
        logical = TILE_BYTES_A + TILE_BYTES_B
        l2_bytes = TILE_BYTES_A + TILE_BYTES_B / c["cluster_m"]  # B multicast inside 2x1
        avg_ns_per_sm = call["median"] * 1e3 * SMS / (units * ktiles)
        out.append(dict(
            case_id=case_id, cluster_m=c["cluster_m"], raster=c["raster_actual"],
            max_swizzle=c["max_swizzle"], log_swizzle=c["log_swizzle"], sched_tiles=units,
            grid=c["grid"], max_active_clusters=c["max_active_clusters"],
            call_us=call["median"], call_cv=call["cv"], processes=call["n"],
            tflops=2 * c["m"] * c["n"] * c["k"] / call["median"] / 1e6, ghz_after_call=ghz,
            ns_per_ktile_critical=per_ktile_ns, cycles_per_ktile_critical=per_ktile_ns * ghz,
            logical_B_per_cycle=logical / (per_ktile_ns * ghz),
            l2_request_B_per_cycle=l2_bytes / (per_ktile_ns * ghz),
            logical_B_per_cycle_avg=logical / (avg_ns_per_sm * ghz)))
    return out


def write_csv(path: Path, tables):
    keys = []
    for table in tables:
        for row in table:
            keys += [k for k in row if k not in keys]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["set"] + keys)
        writer.writeheader()
        for name, table in zip(("wave", "stamp", "l2"), tables):
            for row in table:
                writer.writerow({"set": name, **{k: (json.dumps(v) if isinstance(v, list) else v)
                                                 for k, v in row.items()}})


def plots(out: Path, waves, fits, l2):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; plots skipped")
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = [r["sched_tiles"] for r in waves]
    ax.plot(x, [r["call_us"] for r in waves], "o", color="#2563eb", label="measured (call)")
    grid = np.linspace(1, max(x), 600)
    a, t = fits["call_us"]["discrete"]["coef"]
    ax.plot(grid, a + np.ceil(grid / SMS) * t, "-", color="#2563eb", alpha=0.5,
            label="discrete fit")
    a, t = fits["call_us"]["fractional"]["coef"]
    ax.plot(grid, a + grid / SMS * t, "--", color="#dc2626", alpha=0.6, label="fractional fit")
    for w in range(1, 5):
        ax.axvline(w * SMS, color="#9ca3af", lw=0.6)
    ax.set_xlabel("scheduled output tiles (128x256)")
    ax.set_ylabel("single-call time (us), K=4096")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "waves.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = [f"c{r['cluster_m']} {r['raster']} s{r['max_swizzle']}" for r in l2]
    ax.bar(labels, [r["tflops"] for r in l2], color="#2563eb")
    ax.set_ylabel("TFLOP/s, 8192x8192x4096")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(out / "l2_traversal.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or args.input
    out.mkdir(parents=True, exist_ok=True)
    rows = load(args.input)
    waves = wave_table(rows)
    exact = [r for r in waves if r["tiles"] == r["sched_tiles"] and r["m"] % 128 == 0
             and r["n"] % 256 == 0]
    fits = {key: wave_fits(exact, key) for key in ("call_us", "graph_us")}
    fits_all = {key: wave_fits(waves, key) for key in ("call_us", "graph_us")}
    stamps = stamp_table(args.input, rows)
    l2 = l2_table(rows, fits["call_us"]["discrete"]["coef"][0]) if group(rows, "l2_") else []
    write_csv(out / "cases.csv", (waves, stamps, l2))
    summary = dict(waves=waves, fits_exact_shapes=fits, fits_all_shapes=fits_all,
                   stamps=stamps, l2=l2, sms=SMS)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plots(out, waves, fits, l2)
    for key in ("call_us", "graph_us"):
        for name, f in fits[key].items():
            print(key, name, [round(c, 3) for c in f["coef"]], "rms", round(f["rms_us"], 3),
                  "max", round(f["max_abs_us"], 3))


if __name__ == "__main__":
    main()
