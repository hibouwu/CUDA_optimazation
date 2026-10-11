#!/usr/bin/env python3
"""R07 offline analysis: anchor table, K fit, launch split, clock traces -> cases.csv/summary.json.

  r07_analyze.py --input RUN
Pilot samples (case_id pilot_*) are excluded. Plots need matplotlib (optional).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics as st

KS = (64, 128, 256, 512, 1024, 2048, 4096)


def load(run: Path):
    rows = [json.loads(x) for x in (run / "samples.jsonl").read_text().splitlines() if x]
    rows = [r for r in rows if not r["case_id"].startswith("pilot_")]
    for r in rows:
        if r["status"] != "measured" or r["returncode"] != 0:
            raise ValueError(f"bad sample {r['case_id']} {r['trial']}")
        if "max_storage_reference_error" in r and r["max_storage_reference_error"] != 0:
            raise ValueError(f"numeric check {r['case_id']}")
        if r.get("check") and r["check"]["status"] != "ok":
            raise ValueError(f"numeric check {r['case_id']}")
    return rows


def describe(values):
    mean = st.mean(values)
    return dict(n=len(values), median=st.median(values), min=min(values), max=max(values),
                cv=(st.stdev(values) / mean if len(values) > 1 else 0.0))


def fit(xs, ys):
    """Ordinary least squares y = a + b x; returns a, b, max |residual|."""
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    return a, b, max(abs(y - a - b * x) for x, y in zip(xs, ys))


def sass_summary(build: Path):
    out = {}
    for name in ("r00_cutlass_debug", "r00_cutlass_ndebug", "r07_probe"):
        sass = (build / f"{name}.sass").read_text()
        log = (build / f"{name}.log").read_text()
        out[name] = dict(
            c7510=log.count("C7510"),
            hgmma=len(re.findall(r"\bHGMMA\.", sass)),
            depbar={m: sass.count(m) for m in ("gsb0, 0x0", "gsb0, 0x1")},
        )
    return out


def read_events(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def read_nvsmi(path: Path):
    with path.open() as stream:
        reader = csv.reader(stream, skipinitialspace=True)
        header = [h.strip() for h in next(reader)]
        rows = [dict(zip(header, [c.strip() for c in row])) for row in reader if row]
    return header, rows


def clock_case(run: Path, case_id):
    traces = []
    for folder in sorted((run / "samples" / case_id).glob("trial-*")):
        events = read_events(folder / "stdout.txt")
        start = next(e for e in events if e["event"] == "loop_start")["t_ns"]
        end = next(e for e in events if e["event"] == "loop_end")["t_ns"]
        cold = next(e for e in events if e["event"] == "cold")
        tail = next(e for e in events if e["event"] == "tail")
        batches = [e for e in events if e["event"] == "batch"]
        _, smi = read_nvsmi(folder / "nvsmi.csv")
        clock_key = next(k for k in smi[0] if k.startswith("clocks.current.sm"))
        power_key = next(k for k in smi[0] if k.startswith("power.draw ["))
        load = [r for r in smi if r["unix_ns"] and start <= int(r["unix_ns"]) <= end]
        idle = [r for r in smi if r["unix_ns"] and int(r["unix_ns"]) < cold["t_ns"] - 2e8]
        second_half = [r for r in load if int(r["unix_ns"]) >= (start + end) / 2]
        reasons = {}
        for r in load:
            reasons[r["clocks_event_reasons.active"]] = reasons.get(
                r["clocks_event_reasons.active"], 0) + 1
        late = [b for b in batches if b["t_ns"] >= (start + end) / 2]
        # First batch whose probe clock is within 1% of the second-half median.
        late_ghz = [b["ghz"] for b in late]
        steady_probe = st.median(late_ghz)
        settle = next(b for b in batches if abs(b["ghz"] - steady_probe) / steady_probe < 0.01)
        traces.append(dict(
            trial=folder.name,
            cold_probe_ghz=cold["ghz_before"], cold_call_us=cold["call_us"],
            after_cold_call_probe_ghz=cold["ghz_after"],
            first_batch_probe_ghz=batches[0]["ghz"], first_batch_call_us=batches[0]["call_us"],
            steady_probe_ghz=steady_probe,
            steady_probe_stats=dict(n=len(late_ghz), median=steady_probe, min=min(late_ghz),
                                    max=max(late_ghz),
                                    cv=st.pstdev(late_ghz) / st.mean(late_ghz)),
            steady_call_us=st.median(b["call_us"] for b in late),
            settle_ms=(settle["t_ns"] - start) / 1e6,
            steady_nvsmi_mhz=st.median(float(r[clock_key]) for r in second_half),
            steady_power_w=st.median(float(r[power_key]) for r in second_half),
            max_power_w=max(float(r[power_key]) for r in load),
            max_temp_c=max(float(r["temperature.gpu"]) for r in load),
            idle_nvsmi_mhz=st.median(float(r[clock_key]) for r in idle) if idle else None,
            idle_power_w=st.median(float(r[power_key]) for r in idle) if idle else None,
            enforced_limit_w=float(load[0]["enforced.power.limit [W]"]),
            load_reasons=reasons,
            sw_power_cap_fraction=sum(r["clocks_event_reasons.sw_power_cap"] == "Active"
                                      for r in load) / len(load),
            tail_probe_ghz=tail["ghz"],
            samples_nvsmi=len(load), batches=len(batches),
            trace_nvsmi=[((int(r["unix_ns"]) - start) / 1e6, float(r[clock_key]),
                          float(r[power_key])) for r in smi if r["unix_ns"]],
            trace_probe=[((b["t_ns"] - start) / 1e6, b["ghz"], b["call_us"]) for b in batches],
        ))
    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write outputs here instead of --input")
    args = parser.parse_args()
    run = args.input.resolve()
    out = (args.output or args.input).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = load(run)
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case_id"], []).append(r)
    summary = dict(sass=sass_summary(run / "build"),
                   environment=json.loads((run / "environment.json").read_text()))
    table = []

    # Anchor: R00 harness, one timed call after warm-up per process.
    anchor = {}
    for case_id, rs in sorted(by_case.items()):
        if "elapsed_ms" not in rs[0]:
            continue
        us = [r["elapsed_ms"] * 1e3 for r in rs]
        d = describe(us)
        flop = rs[0]["work_flop"]
        d.update(case_id=case_id, metric="cuda_event_single_call_us",
                 tflops_median=flop / (d["median"] * 1e-6) / 1e12, work_flop=flop)
        table.append(d)
        anchor[case_id] = d
    summary["r00_harness"] = anchor

    # K fit on per-case medians (and on every sample as a check).
    fits = {}
    for mn in (2048, 256):
        ids = [f"kfit_m{mn}_k{k}" for k in KS]
        if not all(i in anchor for i in ids):
            continue
        xs = [k / 64 for k in KS]
        a, b, res = fit(xs, [anchor[i]["median"] for i in ids])
        xa = [k / 64 for k in KS for _ in by_case[f"kfit_m{mn}_k{k}"]]
        ya = [r["elapsed_ms"] * 1e3 for k in KS for r in by_case[f"kfit_m{mn}_k{k}"]]
        a2, b2, _ = fit(xa, ya)
        fits[f"r00_m{mn}"] = dict(a_us=a, b_us_per_ktile=b, max_abs_residual_us=res,
                                  a_all_samples_us=a2, b_all_samples=b2)
        # Fit excluding K<256 (prologue not full: Ktile < 4 stages).
        x3 = [k / 64 for k in KS if k >= 256]
        y3 = [anchor[f"kfit_m{mn}_k{k}"]["median"] for k in KS if k >= 256]
        a3, b3, r3 = fit(x3, y3)
        fits[f"r00_m{mn}"].update(a_k256plus_us=a3, b_k256plus=b3, max_abs_residual_k256plus=r3)
        # Same fit over explicit K ranges (clock falls for long runs, so ranges matter).
        ranges = {}
        for lo, hi in ((64, 4096), (64, 2048), (256, 2048), (256, 4096)):
            kk = [k for k in KS if lo <= k <= hi]
            ra, rb, rr = fit([k / 64 for k in kk], [anchor[f"kfit_m{mn}_k{k}"]["median"] for k in kk])
            ranges[f"k{lo}-{hi}"] = dict(k=kk, a_us=ra, b_us_per_ktile=rb, max_abs_residual_us=rr)
        fits[f"r00_m{mn}"]["ranges"] = ranges

    # Launch-mode timing from r07_probe.
    timing = {}
    for case_id, rs in sorted(by_case.items()):
        if "timing" not in rs[0]:
            continue
        entry = {}
        for key in ("isolated_median_us", "host_call_median_us", "burst_us", "graph_us"):
            d = describe([r["timing"][key] for r in rs])
            entry[key] = d
            table.append(dict(d, case_id=case_id, metric=key))
        entry["cutlass_grid"] = rs[0]["timing"]["cutlass_grid"]
        timing[case_id] = entry
    summary["timing"] = timing
    for mn in (2048, 256):
        ids = [f"timing_cutlass_m{mn}_k{k}" for k in KS]
        if not all(i in timing for i in ids):
            continue
        xs = [k / 64 for k in KS]
        for key in ("isolated_median_us", "burst_us", "graph_us"):
            ys = [timing[i][key]["median"] for i in ids]
            a, b, res = fit(xs, ys)
            x3 = [x for x in xs if x >= 4]
            a3, b3, r3 = fit(x3, ys[-len(x3):])
            fits[f"probe_m{mn}_{key}"] = dict(a_us=a, b_us_per_ktile=b, max_abs_residual_us=res,
                                              a_k256plus_us=a3, b_k256plus=b3,
                                              max_abs_residual_k256plus=r3)
            ranges = {}
            for lo, hi in ((64, 4096), (64, 2048), (256, 2048), (256, 4096)):
                kk = [k for k in KS if lo <= k <= hi]
                values = [timing[f"timing_cutlass_m{mn}_k{k}"][key]["median"] for k in kk]
                ra, rb, rr = fit([k / 64 for k in kk], values)
                ranges[f"k{lo}-{hi}"] = dict(k=kk, a_us=ra, b_us_per_ktile=rb,
                                            max_abs_residual_us=rr)
            fits[f"probe_m{mn}_{key}"]["ranges"] = ranges
    summary["fits"] = fits

    # Clock under load.
    clock = {}
    for case_id in sorted(by_case):
        if case_id.startswith("clock_"):
            clock[case_id] = clock_case(run, case_id)
            for t in clock[case_id]:
                # All statistics from the same second-half post-batch probe samples.
                q = t["steady_probe_stats"]
                table.append(dict(case_id=case_id, metric=f"steady_post_batch_probe_ghz_{t['trial']}",
                                  n=q["n"], median=q["median"], min=q["min"], max=q["max"],
                                  cv=q["cv"]))
    summary["clock"] = {k: [{kk: vv for kk, vv in t.items() if not kk.startswith("trace")}
                            for t in v] for k, v in clock.items()}

    # Second allocation step (same job): R00-sequence clock and warm-module cold calls.
    extra = run / "clock-sequence"
    if (extra / "samples.jsonl").exists():
        seq_rows = load(extra)
        seq = {}
        for r in seq_rows:
            if "r00seq" in r:
                seq.setdefault(r["case_id"], []).append(r["r00seq"])
        summary["r00seq"] = {}
        for case_id, rs in sorted(seq.items()):
            entry = dict(
                timed_us=describe([x["timed_us"] for x in rs]),
                ghz_before=describe([x["ghz_before"] for x in rs]),
                ghz_after=describe([x["ghz_after"] for x in rs]),
                warmup_calls=[x["warmup_calls"] for x in rs],
            )
            summary["r00seq"][case_id] = entry
            table.append(dict(entry["timed_us"], case_id=case_id, metric="r00seq_timed_us"))
            table.append(dict(entry["ghz_after"], case_id=case_id, metric="r00seq_ghz_after"))
        cold_ids = sorted({r["case_id"] for r in seq_rows if r["case_id"].startswith("cold_")})
        summary["cold_warm_module"] = {
            case_id: [{k: v for k, v in t.items() if not k.startswith("trace")}
                      for t in clock_case(extra, case_id)]
            for case_id in cold_ids
        }

    with (out / "cases.csv").open("w", newline="") as stream:
        fields = ["case_id", "metric", "n", "median", "min", "max", "cv", "tflops_median",
                  "work_flop"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; plots skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mn in zip(axes, (2048, 256)):
        key = f"r00_m{mn}"
        if key not in fits:
            continue
        xs = [k / 64 for k in KS]
        ax.plot(xs, [anchor[f"kfit_m{mn}_k{k}"]["median"] for k in KS], "o", label="R00 harness")
        if f"probe_m{mn}_graph_us" in fits:
            ax.plot(xs, [timing[f"timing_cutlass_m{mn}_k{k}"]["graph_us"]["median"] for k in KS],
                    "s", label="graph per call")
        f = fits[key]
        ax.plot(xs, [f["a_us"] + f["b_us_per_ktile"] * x for x in xs], "-",
                label=f"fit a={f['a_us']:.2f} b={f['b_us_per_ktile']:.3f}")
        ax.set_xlabel("Ktile = K/64")
        ax.set_ylabel("time per call (us)")
        ax.set_title(f"CUTLASS NDEBUG M=N={mn}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "kfit.png", dpi=120)
    if clock:
        fig, axes = plt.subplots(len(clock), 1, figsize=(10, 3.2 * len(clock)), squeeze=False)
        for ax, (case_id, traces) in zip(axes[:, 0], clock.items()):
            for t in traces:
                ax.plot([p[0] for p in t["trace_nvsmi"]], [p[1] for p in t["trace_nvsmi"]],
                        "-", lw=1, label=f"NVML clocks.sm {t['trial']}")
                ax.plot([p[0] for p in t["trace_probe"]], [p[1] * 1000 for p in t["trace_probe"]],
                        ".", ms=3, label=f"clock64/globaltimer {t['trial']}")
            ax.set_title(case_id)
            ax.set_xlabel("ms from loop start")
            ax.set_ylabel("SM clock (MHz)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "clock.png", dpi=120)


if __name__ == "__main__":
    main()
