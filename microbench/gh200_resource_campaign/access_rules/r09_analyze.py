#!/usr/bin/env python3
"""R09 analysis: in-kernel clock ratios, per-CTA stage split, perturbation, SASS check.

  r09_analyze.py --input RUN [--output DIR]
Reads RUN/samples.jsonl and the gzipped per-sample stdout (per-CTA traces).
Writes cases.csv, summary.json, sass_check.json and plots into DIR (default RUN).
"""
from __future__ import annotations

import argparse
import csv
import difflib
import gzip
import json
from pathlib import Path
import re
import statistics as st

SLOTS = dict(entry=0, prod_load=1, first_mma0=2, first_mma1=3, main_end0=4, main_end1=5,
             epi_start0=6, epi_start1=7, store_done=8, exit_p=9, exit_c0=10, exit_c1=11)
ORDER = ["entry", "prod_load", "first_mma0", "main_end0", "epi_start0", "store_done", "exit_c0"]
R07_A_US, R07_B_US = 9.45, 0.637


def med(values):
    return st.median(values) if values else float("nan")


def cv(values):
    if len(values) < 2 or st.mean(values) == 0:
        return float("nan")
    return st.pstdev(values) / abs(st.mean(values))


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    return dict(a=a, b=b, max_abs_residual=max(abs(y - a - b * x) for x, y in zip(xs, ys)))


def parse_trace(raw, ctas, words=32):
    """Returns per-CTA dicts {name: (clock64, globaltimer)} plus smid."""
    out = []
    for c in range(ctas):
        rec = raw[c * words:(c + 1) * words]
        d = {name: (rec[2 * s], rec[2 * s + 1]) for name, s in SLOTS.items()}
        d["smid"] = rec[24]
        out.append(d)
    return out


def call_metrics(call, setup, min_window_ns):
    """Per-call envelope, stage medians and clock ratios from one traced call."""
    ctas = setup["grid"][0] * setup["grid"][1] * setup["grid"][2]
    tr = parse_trace(call["trace"], ctas)
    complete = all(all(v[0] and v[1] for k, v in d.items() if k != "smid") for d in tr)
    monotone = all(all(d[a][0] <= d[b][0] for a, b in zip(ORDER, ORDER[1:])) for d in tr)
    g = lambda d, k: d[k][1]
    c = lambda d, k: d[k][0]
    exit_g = [max(g(d, "exit_p"), g(d, "exit_c0"), g(d, "exit_c1")) for d in tr]
    exit_key = [max(("exit_p", "exit_c0", "exit_c1"), key=lambda k: g(d, k)) for d in tr]
    entry_g = [g(d, "entry") for d in tr]
    t0 = min(entry_g)
    envelope = max(exit_g) - t0
    # Stage chain of consumer warpgroup 0 (thread 128 also issues and waits the TMA stores);
    # the terms telescope: entry -> first MMA -> mainloop end -> epilogue start -> store wait
    # done -> CTA exit (last of the three role exits).
    per = dict(
        entry_skew=[e - t0 for e in entry_g],
        producer_setup=[g(d, "prod_load") - g(d, "entry") for d in tr],
        prefill=[g(d, "first_mma0") - g(d, "entry") for d in tr],
        mainloop=[g(d, "main_end0") - g(d, "first_mma0") for d in tr],
        main_to_epi=[g(d, "epi_start0") - g(d, "main_end0") for d in tr],
        epilogue=[g(d, "store_done") - g(d, "epi_start0") for d in tr],
        post_store=[x - g(d, "store_done") for d, x in zip(tr, exit_g)],
        cta_span=[x - e for x, e in zip(exit_g, entry_g)],
        wg1_main_end_lag=[g(d, "main_end1") - g(d, "main_end0") for d in tr],
        wg1_first_mma_lag=[g(d, "first_mma1") - g(d, "first_mma0") for d in tr],
        mainloop_cycles=[c(d, "main_end0") - c(d, "first_mma0") for d in tr],
        prefill_cycles=[c(d, "first_mma0") - c(d, "entry") for d in tr],
        epilogue_cycles=[c(d, "store_done") - c(d, "epi_start0") for d in tr],
    )
    ratios, main_ratios = [], []
    for d, x, k in zip(tr, exit_g, exit_key):
        window = x - g(d, "entry")
        if window >= min_window_ns:
            ratios.append((c(d, k) - c(d, "entry")) / window)
        mw = g(d, "main_end0") - g(d, "first_mma0")
        if mw >= min_window_ns:
            main_ratios.append((c(d, "main_end0") - c(d, "first_mma0")) / mw)
    m = {k: med(v) for k, v in per.items()}
    sorted_exit = sorted(exit_g)
    m.update(
        event_ns=call["elapsed_us"] * 1e3, envelope_ns=envelope,
        host_gap_ns=call["elapsed_us"] * 1e3 - envelope,
        entry_spread_ns=max(entry_g) - t0,
        tail_ns=sorted_exit[-1] - med(exit_g),
        median_exit_ns=med(exit_g) - t0,
        ctas=ctas, complete=complete, monotone=monotone,
        window_ghz_median=med(ratios), window_ghz_min=min(ratios) if ratios else None,
        window_ghz_max=max(ratios) if ratios else None, window_ctas=len(ratios),
        main_ghz_median=med(main_ratios), main_ctas=len(main_ratios),
        cta_window_ns_median=med(per["cta_span"]),
        distinct_sms=len({d["smid"] for d in tr}),
        last_exit_role=max(zip(exit_g, exit_key))[1],
        probe_ghz=call["probe_ghz_median"],
    )
    # Additive reconstruction of the event time from per-CTA medians.
    m["staged_sum_ns"] = (m["host_gap_ns"] + m["entry_skew"] + m["prefill"] + m["mainloop"]
                          + m["main_to_epi"] + m["epilogue"] + m["post_store"] + m["tail_ns"])
    return m


def load(run: Path):
    rows = [json.loads(x) for x in (run / "samples.jsonl").read_text().splitlines() if x.strip()]
    out = []
    for r in rows:
        if r["binary"].startswith("r00_"):
            out.append(dict(r, calls_full=[dict(label="warm", elapsed_us=r["elapsed_us"])]))
            continue
        text = gzip.open(run / r["folder"] / "stdout.txt.gz", "rt").read()
        lines = [json.loads(x) for x in text.splitlines() if x.strip()]
        r["calls_full"] = [x for x in lines if x.get("event") == "call"]
        out.append(r)
    return out


def sass_functions(path: Path):
    funcs, name = {}, None
    pattern = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
    for line in path.read_text().splitlines():
        if "Function :" in line:
            name = line.split("Function :")[1].strip()
            funcs[name] = []
        elif name:
            mt = pattern.search(line)
            if mt:
                funcs[name].append((int(mt.group(1), 16), mt.group(2)))
    return funcs


def opcode(text):
    parts = text.split()
    if parts and parts[0].startswith("@"):
        parts = parts[1:]
    return parts[0] if parts else ""


def sass_summary(path: Path):
    funcs = sass_functions(path)
    name = next(n for n in funcs if "device_kernel" in n)
    ins = funcs[name]
    ops = [opcode(t) for _, t in ins]
    count = lambda pred: sum(1 for o in ops if pred(o))
    # Loops: backward BRA whose body contains HGMMA.
    loops = []
    for i, (addr, text) in enumerate(ins):
        if opcode(text).startswith("BRA"):
            mt = re.search(r"0x([0-9a-f]+)", text)
            if mt and int(mt.group(1), 16) < addr:
                start = int(mt.group(1), 16)
                body = [t for a, t in ins if start <= a <= addr]
                if any(opcode(t).startswith("HGMMA") for t in body):
                    loops.append(body)
    first_h = next(i for i, o in enumerate(ops) if o.startswith("HGMMA"))
    last_h = max(i for i, o in enumerate(ops) if o.startswith("HGMMA"))
    return dict(
        function=name, instructions=len(ops),
        hgmma=count(lambda o: o.startswith("HGMMA")),
        warpgroup_depbar=[t for t, o in zip((t for _, t in ins), ops)
                          if o.startswith("WARPGROUP.DEPBAR")],
        warpgroup_arrive=count(lambda o: o.startswith("WARPGROUP.ARRIVE")),
        globaltimer_reads=sum(1 for _, t in ins if "GLOBALTIMER" in t),
        clock_reads=sum(1 for _, t in ins if "SR_CLOCK" in t),
        mainloop_loops=len(loops),
        mainloop_body_ops=[[opcode(t) for t in body] for body in loops],
        mainloop_body_full=[body for body in loops],
        hgmma_region_ops=ops[first_h:last_h + 1],
    )


def strip_regs(text):
    return re.sub(r"\bU?R\d+\b|\bU?P\d\b|0x[0-9a-f]+", "#", text)


def sass_check(run: Path):
    plain = sass_summary(run / "build/r09_plain.sass")
    trace = sass_summary(run / "build/r09_trace.sass")
    inner_p = min(plain["mainloop_body_full"], key=len)
    inner_t = min(trace["mainloop_body_full"], key=len)
    ops_p, ops_t = [opcode(x) for x in inner_p], [opcode(x) for x in inner_t]
    diff = [line for line in difflib.unified_diff([strip_regs(x) for x in inner_p],
                                                  [strip_regs(x) for x in inner_t],
                                                  lineterm="", n=0)
            if not line.startswith(("---", "+++", "@@"))]
    keep = ("instructions", "hgmma", "warpgroup_depbar", "warpgroup_arrive", "globaltimer_reads",
            "clock_reads")
    count = lambda ops, prefix: sum(1 for o in ops if o.startswith(prefix))
    return dict(
        plain={k: plain[k] for k in keep}, trace={k: trace[k] for k in keep},
        inner_loop=dict(
            length=dict(plain=len(ops_p), trace=len(ops_t)),
            hgmma=dict(plain=count(ops_p, "HGMMA"), trace=count(ops_t, "HGMMA")),
            depbar=dict(plain=[x for x in inner_p if "DEPBAR" in x],
                        trace=[x for x in inner_t if "DEPBAR" in x]),
            syncs=dict(plain=count(ops_p, "SYNCS"), trace=count(ops_t, "SYNCS")),
            opcode_sequence_identical=ops_p == ops_t,
            opcode_multiset_identical=sorted(ops_p) == sorted(ops_t),
            timer_reads_in_loop=sum(1 for x in inner_t if "GLOBALTIMER" in x or "CLOCK" in x),
            diff_modulo_registers=diff,
        ),
    )


def ablation(run: Path):
    """Paired event-time deltas of stamp-subset builds vs plain (RUN/ablation/samples.jsonl)."""
    path = run / "ablation/samples.jsonl"
    if not path.exists():
        return None
    times = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        variant, m, k = r["case_id"].split("_")
        if r["check"]["max_storage_reference_error"] != 0:
            raise ValueError("ablation numeric check")
        times.setdefault((m, k), {}).setdefault(variant, {})[r["trial"]] = \
            r["calls"][0]["elapsed_us"]
    out = []
    for (m, k), vs in sorted(times.items()):
        plain = vs["plain"]
        row = dict(m=int(m[1:]), k=int(k[1:]), plain_us=med(list(plain.values())),
                   plain_cv=cv(list(plain.values())))
        for v in ("trace", "ends", "inner"):
            diffs = [vs[v][t] - plain[t] for t in plain if t in vs[v]]
            row[v + "_us"] = med(list(vs[v].values()))
            row[v + "_paired_delta_us"] = med(diffs)
        out.append(row)
    return out


def summarize(run: Path, out: Path):
    rows = load(run)
    status = [r for r in rows if r.get("status") not in ("measured",)]
    timer = [r["timer"] for r in rows if r.get("timer")]
    steps = sorted({int(s) for t in timer for s in t["step_histogram"]})
    dominant = [max(t["step_histogram"].items(), key=lambda kv: kv[1]) for t in timer]
    step_ns = max(int(k) for k, _ in dominant) if dominant else 1000
    min_window = 100 * step_ns
    by_case = {}
    for r in rows:
        if r["case_id"] == "timer_res":
            continue
        for call in r["calls_full"]:
            key = (r["case_id"], call["label"])
            entry = dict(trial=r["trial"], event_us=call["elapsed_us"], set=r["set"],
                         check=(r.get("check") or {}).get("max_storage_reference_error",
                                                           r.get("max_storage_reference_error")))
            if "trace" in call and call["trace"]:
                entry.update(call_metrics(call, r["setup"], min_window))
            elif "probe_ghz_median" in call:
                entry["probe_ghz"] = call["probe_ghz_median"]
            by_case.setdefault(key, []).append(entry)
    cases = []
    for (case_id, label), items in sorted(by_case.items()):
        items = [i for i in items if i["set"] != "pilot"] or items
        variant, *rest = case_id.split("_")
        m = int(next(x for x in rest if x.startswith("m"))[1:])
        k = int(next(x for x in rest if x.startswith("k"))[1:])
        row = dict(case=case_id, call=label, variant=variant, m=m, k=k, ktile=k // 64,
                   n=len(items), event_us_median=med([i["event_us"] for i in items]),
                   event_us_cv=cv([i["event_us"] for i in items]),
                   event_us_min=min(i["event_us"] for i in items),
                   event_us_max=max(i["event_us"] for i in items),
                   max_check_error=max(i["check"] or 0 for i in items))
        if any("probe_ghz" in i for i in items):
            row["probe_ghz_median"] = med([i["probe_ghz"] for i in items if "probe_ghz" in i])
        if any("envelope_ns" in i for i in items):
            for key in ("envelope_ns", "host_gap_ns", "entry_skew", "entry_spread_ns",
                        "producer_setup", "prefill", "mainloop", "main_to_epi", "epilogue",
                        "post_store", "tail_ns", "staged_sum_ns", "mainloop_cycles",
                        "prefill_cycles", "epilogue_cycles", "window_ghz_median",
                        "main_ghz_median", "cta_window_ns_median", "wg1_main_end_lag",
                        "wg1_first_mma_lag", "median_exit_ns"):
                vals = [i[key] for i in items if i.get(key) == i.get(key) and i.get(key)
                        is not None]
                row[key + "_med"] = med(vals)
                row[key + "_cv"] = cv(vals)
            row["window_ghz_min"] = min((i["window_ghz_min"] for i in items
                                         if i["window_ghz_min"]), default=None)
            row["window_ghz_max"] = max((i["window_ghz_max"] for i in items
                                         if i["window_ghz_max"]), default=None)
            row["window_ctas_min"] = min(i["window_ctas"] for i in items)
            row["ctas"] = items[0]["ctas"]
            row["all_complete"] = all(i["complete"] for i in items)
            row["all_monotone"] = all(i["monotone"] for i in items)
            row["last_exit_roles"] = sorted({i["last_exit_role"] for i in items})
            row["distinct_sms_min"] = min(i["distinct_sms"] for i in items)
        cases.append(row)

    def get(variant, m, k, label="warm"):
        return next((c for c in cases if c["variant"] == variant and c["m"] == m
                     and c["k"] == k and c["call"] == label), None)

    perturbation = []
    paired = {}
    for (case_id, label), items in by_case.items():
        paired[(case_id, label)] = {i["trial"]: i["event_us"] for i in items if i["set"] != "pilot"}
    for m in (2048, 256):
        for k in (256, 512, 1024, 2048, 4096):
            t, p = get("trace", m, k), get("plain", m, k)
            if not (t and p):
                continue
            pt, pp = paired[(f"trace_m{m}_k{k}", "warm")], paired[(f"plain_m{m}_k{k}", "warm")]
            diffs = [pt[i] / pp[i] - 1 for i in pt if i in pp]
            ratio = t["event_us_median"] / p["event_us_median"] - 1
            perturbation.append(dict(m=m, k=k, trace_us=t["event_us_median"],
                                     plain_us=p["event_us_median"], ratio_of_medians=ratio,
                                     paired_median=med(diffs), pairs=len(diffs)))
    fits = {}
    for m in (2048, 256):
        for variant, ks in (("plain", (64, 128, 256, 512, 1024, 2048, 4096)),
                            ("plain", (256, 512, 1024, 2048, 4096)),
                            ("trace", (256, 512, 1024, 2048, 4096))):
            pts = [get(variant, m, k) for k in ks]
            if all(pts):
                fits[f"{variant}_m{m}_k{ks[0]}-{ks[-1]}"] = fit(
                    [p["ktile"] for p in pts], [p["event_us_median"] for p in pts])
        pts = [get("trace", m, k) for k in (256, 512, 1024, 2048, 4096)]
        if all(pts):
            x = [p["ktile"] for p in pts]
            terms = ("host_gap_ns", "entry_skew", "prefill", "mainloop", "main_to_epi",
                     "epilogue", "post_store", "tail_ns", "staged_sum_ns", "envelope_ns",
                     "mainloop_cycles")
            fits[f"stages_m{m}"] = {t: fit(x, [p[t + "_med"] / (1 if t.endswith("cycles")
                                                               else 1e3) for p in pts])
                                    for t in terms}
    summary = dict(
        failed_or_numeric=[(r["case_id"], r["trial"]) for r in status],
        timer=dict(steps_seen=steps, dominant=dominant, step_ns=step_ns,
                   min_window_ns=min_window, ghz=[t["ghz"] for t in timer]),
        perturbation=perturbation, fits=fits, r07=dict(a_us=R07_A_US, b_us=R07_B_US),
        ablation=ablation(run), cases=cases,
    )
    try:
        summary["sass"] = sass_check(run)
    except (FileNotFoundError, StopIteration) as error:
        summary["sass"] = dict(error=str(error))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    keys = sorted({k for c in cases for k in c})
    first = ["case", "call", "variant", "m", "k", "ktile", "n"]
    keys = first + [k for k in keys if k not in first]
    with (out / "cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for c in cases:
            w.writerow(c)
    plots(summary, out)
    return summary


def plots(summary, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    (out / "plots").mkdir(exist_ok=True)
    terms = ("host_gap_ns", "entry_skew", "prefill", "mainloop", "main_to_epi", "epilogue",
             "post_store", "tail_ns")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, (2048, 256)):
        rows = sorted((c for c in summary["cases"] if c["variant"] == "trace" and c["m"] == m
                       and c["call"] == "warm"), key=lambda c: c["k"])
        if not rows:
            continue
        x = [str(r["k"]) for r in rows]
        bottom = [0.0] * len(rows)
        for t in terms:
            vals = [r[t + "_med"] / 1e3 for r in rows]
            ax.bar(x, vals, bottom=bottom, label=t.replace("_ns", ""))
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.plot(x, [r["event_us_median"] for r in rows], "k_", markersize=20, label="event")
        ax.set_title(f"M=N={m}, traced build")
        ax.set_xlabel("K")
        ax.set_ylabel("µs")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "plots/stages.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, marker in (("cold", "o"), ("warm", "s")):
        rows = sorted((c for c in summary["cases"] if c["variant"] == "idle"
                       and c["call"] == label), key=lambda c: (c["m"], c["k"]))
        ax.plot([f"{r['m']}/{r['k']}" for r in rows], [r["window_ghz_median_med"] for r in rows],
                marker, label=f"{label}: entry→exit window")
        ax.plot([f"{r['m']}/{r['k']}" for r in rows], [r["probe_ghz_median"] for r in rows],
                marker, fillstyle="none", label=f"{label}: post-call probe")
    ax.set_xlabel("M=N / K")
    ax.set_ylabel("GHz")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "plots/clock.png", dpi=120)
    plt.close(fig)


def summarize_ends_case(run: Path, row, setup):
    """Plain/ends only: no per-tile array access, so head counts may exceed 64."""
    import v08_model

    times = dict(plain=[], ends=[])
    ends = []
    for path in sorted((run / 'samples' / row['id']).glob('*.json')):
        record = json.loads(path.read_text())
        variant = record['variant']
        if variant not in times:
            raise ValueError('wide-input must not contain stamped processes')
        if record['returncode']:
            continue
        observed = v08_model.observe(run, record, row, setup)
        times[variant].append(observed['elapsed_us'])
        if variant == 'ends':
            ctas = observed['ctas']
            active = [c for c in ctas if c['tiles']]
            T = max(c['tiles'] for c in active)
            cycles = max(c['end_c'] - c['entry_c'] for c in active)
            ghz = med([(c['end_c'] - c['entry_c']) / (c['end_ns'] - c['entry_ns'])
                       for c in active if c['tiles'] == T])
            window = (max(c['end_ns'] for c in active) - min(c['entry_ns'] for c in ctas)) / 1000
            ends.append(dict(trial=record['trial'], cycles=cycles, ghz=ghz, window_us=window,
                             event_minus_window_us=observed['elapsed_us'] - window, tiles_max=T))
    if any(len(values) != 10 for values in times.values()):
        raise ValueError('ten successful plain and ends processes required: ' + row['id'])
    return dict(id=row['id'], config=row['config'], input_mode=row['input_mode'],
                plain_us=med(times['plain']), ends_us=med(times['ends']),
                plain_cv=cv(times['plain']), ends_cv=cv(times['ends']),
                c_max_ends=med([e['cycles'] for e in ends]), ghz_ends=med([e['ghz'] for e in ends]),
                window_ends=med([e['window_us'] for e in ends]),
                ends_perturbation=med(times['ends']) / med(times['plain']) - 1,
                T=max(e['tiles_max'] for e in ends), kt=(row['k'] + 63) // 64,
                ends_gap_us=med([e['event_minus_window_us'] for e in ends]),
                window_ends_gt_600us=med([e['window_us'] for e in ends]) > 600,
                ends_processes_gt_600us=sum(e['window_us'] > 600 for e in ends), per_process_ends=ends)


def summarize_shared(run: Path, out: Path):
    """Public R09 batch: reuse the shared numeric replay and timing definitions."""
    import v06_run as common
    import v08_model

    common.verify(run)
    wide = json.loads((run / 'run_config.json').read_text()).get('batch') == 'wide-input'
    rows = json.loads((run / "cases.json").read_text())
    setups = {s["case"]: s["setup"] for s in json.loads((run / "static_setup.json").read_text())}
    sampling = json.loads((run / "sampling.json").read_text()) if (run / "sampling.json").exists() else {}
    failed = {r["case"]: r for r in sampling.get("numeric_failed", [])}
    summaries, metrics = {}, []
    for row in rows:
        metric = {key: row[key] for key in ("id", "config", "m", "n", "k", "input_mode", "seed", "sm_count")}
        if row["id"] in failed:
            summaries[row["id"]] = dict(status="numeric_error", condition=row, failure=failed[row["id"]])
            metrics.append(dict(metric, status="numeric_error"))
            continue
        s = (summarize_ends_case(run, row, setups[row['id']]) if wide
             else v08_model.summarize_case(run, row, setups[row["id"]]))
        summaries[row["id"]] = s
        metric["status"] = "measured"
        keys = ["plain_us", "ends_us", "plain_cv", "ends_cv", "c_max_ends", "ghz_ends",
                "window_ends", "ends_perturbation", "T", "kt"]
        keys += (["ends_gap_us", "window_ends_gt_600us", "ends_processes_gt_600us"] if wide else
                 ["stamped_us", "stamped_cv", "c_max_stamped", "ghz_stamped", "window_stamped", "perturbation"])
        for key in keys:
            metric[key] = s[key]
        metric['plain_minus_ends_window_us'] = s['plain_us'] - s['window_ends']
        if not wide:
            metric.update(P0_cycles=s["intervals"]["P0"], S_cycles=s["intervals"]["S"],
                          L0_per_kt=s["intervals"]["L0"] / s["kt"],
                          L_per_kt=s["intervals"]["L"] / s["kt"],
                          observed_kappa=s["c_max_ends"] / s["c_max_stamped"])
        metrics.append(metric)
    for metric in metrics:
        if metric["status"] != "measured":
            continue
        base = next((x for x in metrics if x["config"] == metric["config"] and x["k"] == metric["k"]
                     and x["input_mode"] == "dyadic" and x["status"] == "measured"), None)
        for key in ("plain_us", "c_max_ends", "ghz_ends"):
            metric[key + "_vs_dyadic"] = metric[key] / base[key] - 1 if base else None
    out.mkdir(parents=True, exist_ok=False)
    common.write_json(out / "summary.json", dict(
        purpose="R09 same-batch clock/input diagnostic; no frozen validation score or power inference.",
        environment=json.loads((run / "environment.json").read_text()), cases=summaries,
        planned_cases=len(rows), measured_cases=len(rows) - len(failed), numeric_failed=list(failed),
        notes=["plain_minus_ends_window_us subtracts medians from separate processes, not a physical host gap.",
               "ghz is the shared maximum-tile CTA cycle/ns statistic; input changes do not measure power."],
    ))
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for m in metrics for k in m)))
        writer.writeheader()
        writer.writerows(metrics)
    for m in metrics:
        if m["status"] != "measured":
            print(m['id'], 'numeric_error; no performance result')
            continue
        print(f"{m['id']:32s} plain {m['plain_us']:.3f} us, ends {m['window_ends']:.3f} us, "
              f"{m['ghz_ends']:.6f} GHz, {m['c_max_ends']:.1f} cycle")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--shared", action="store_true", help="analyze public cfg_a/b/c R09 batch")
    args = p.parse_args()
    if args.shared:
        summarize_shared(args.input.resolve(), (args.output or args.input / "analysis").resolve())
        return
    s = summarize(args.input.resolve(), (args.output or args.input).resolve())
    print(json.dumps(dict(timer=s["timer"], perturbation=s["perturbation"],
                          sass={k: v for k, v in s["sass"].items()}), indent=1))


if __name__ == "__main__":
    main()
