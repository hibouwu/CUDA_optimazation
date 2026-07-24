#!/usr/bin/env python3
import csv
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
BUILD = REPO / "mma_with_cp" / "build"
RESULTS = ROOT / "results"
NCU_DIR = RESULTS / "ncu"

ITERS = 10000
FREQ_HZ = "1575000000"
TMEM_BYTES_PER_TS_MMA = 2048.0

CASES = [
    {
        "case": "cp-only",
        "role": "component-cp-ingress",
        "binary": BUILD / "tcgen05_cp_only_m128n256_fp4_benchmark",
        "sass_needles": ["UTCCP"],
    },
    {
        "case": "ts-mma-only",
        "role": "component-tmem-consume",
        "binary": BUILD / "tcgen05_ts_mma_only_m128n256_fp4_benchmark",
        "sass_needles": ["UTCOMMA", "tmem["],
    },
    {
        "case": "serial-a1",
        "role": "serial-cp-mma",
        "binary": BUILD / "tcgen05_ts_cp_mma_serial_a1_m128n256_fp4_benchmark",
        "sass_needles": ["UTCCP", "UTCOMMA", "tmem["],
    },
    {
        "case": "overlap-a2",
        "role": "double-buffer-overlap",
        "binary": BUILD / "tcgen05_ts_cp_mma_overlap_a2_m128n256_fp4_benchmark",
        "sass_needles": ["UTCCP", "UTCOMMA", "tmem["],
    },
    {
        "case": "warp-split-a2",
        "role": "split-issuer-overlap",
        "binary": BUILD / "tcgen05_ts_cp_mma_warp_split_a2_m128n256_fp4_benchmark",
        "sass_needles": ["UTCCP", "UTCOMMA", "tmem["],
    },
    {
        "case": "mainloop-a2-k16",
        "role": "steady-state-overlap",
        "binary": BUILD / "tcgen05_ts_cp_mma_mainloop_a2_k16_m128n256_fp4_benchmark",
        "sass_needles": ["UTCCP", "UTCOMMA", "tmem["],
    },
]

NCU_METRICS = [
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg",
    "sm__cycles_elapsed.avg.per_second",
    "sm__throughput.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.sum",
    "sm__inst_executed_pipe_tensor.sum.per_cycle_active",
    "sm__inst_executed_pipe_tensor.sum.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tc.sum",
    "sm__inst_executed_pipe_tc.sum.per_cycle_active",
    "sm__inst_executed_pipe_tc.sum.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tmem.sum",
    "sm__inst_executed_pipe_tmem.sum.per_cycle_active",
    "sm__inst_executed_pipe_tmem.sum.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_active",
]


def run(args, check=True):
    return subprocess.run(args, check=check, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)


def parse_kv(text):
    out = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def require_binaries():
    missing = [str(c["binary"]) for c in CASES if not c["binary"].exists()]
    if missing:
        raise RuntimeError("missing benchmark binaries:\n" + "\n".join(missing))


def app_data(case, repeat=0, iters=ITERS):
    proc = run([str(case["binary"]), str(iters), FREQ_HZ])
    data = parse_kv(proc.stdout)
    cycles = float(data["cycles"])
    mma_count = float(data.get("mma_instruction_count", "0"))
    cp_count = float(data.get("cp_instruction_count", "0"))
    cp_bpc = float(data.get("bytes_per_cycle", "0"))
    consume_bpc = (mma_count * TMEM_BYTES_PER_TS_MMA / cycles) if mma_count else 0.0
    cycles_per_mma = cycles / mma_count if mma_count else 0.0
    cycles_per_cp = cycles / cp_count if cp_count else 0.0
    return {
        "case": case["case"],
        "role": case["role"],
        "repeat": repeat,
        "iters": data.get("iters", str(iters)),
        "cycles": data["cycles"],
        "tflops": data.get("thor_tflops", "0"),
        "mma_instruction_count": data.get("mma_instruction_count", "0"),
        "cp_instruction_count": data.get("cp_instruction_count", "0"),
        "processed_tiles": data.get("processed_tiles", "0"),
        "effective_bytes_per_cp": data.get("effective_bytes_per_cp", "0"),
        "cp_bytes_per_cycle": f"{cp_bpc:.6f}",
        "estimated_tmem_consume_bytes_per_cycle": f"{consume_bpc:.6f}",
        "cycles_per_mma": f"{cycles_per_mma:.6f}",
        "cycles_per_cp": f"{cycles_per_cp:.6f}",
        "cycles_per_tile": data.get("cycles_per_tile", "0"),
    }


def add_derived(rows):
    by_case = {r["case"]: r for r in rows}
    cp_peak = float(by_case["cp-only"]["cp_bytes_per_cycle"])
    ts_consume_peak = float(by_case["ts-mma-only"]["estimated_tmem_consume_bytes_per_cycle"])
    ts_mma_cycles = float(by_case["ts-mma-only"]["cycles_per_mma"])
    cp_only_cycles = float(by_case["cp-only"]["cycles_per_cp"])
    serial_cycles = float(by_case["serial-a1"]["cycles_per_tile"])
    serial_unit_cycles = float(by_case["serial-a1"]["cycles_per_cp"])
    ideal_overlap_cycles = max(cp_only_cycles, ts_mma_cycles)
    for row in rows:
        cp_bpc = float(row["cp_bytes_per_cycle"])
        consume_bpc = float(row["estimated_tmem_consume_bytes_per_cycle"])
        tile_cycles = float(row["cycles_per_tile"])
        cp_cycles = float(row["cycles_per_cp"])
        mma_cycles = float(row["cycles_per_mma"])
        unit_cycles = cp_cycles or mma_cycles or tile_cycles
        row["cp_util_vs_cp_only_pct"] = f"{(100.0 * cp_bpc / cp_peak) if cp_peak else 0.0:.3f}"
        row["consume_util_vs_ts_mma_only_pct"] = f"{(100.0 * consume_bpc / ts_consume_peak) if ts_consume_peak else 0.0:.3f}"
        row["overlap_unit_cycles"] = f"{unit_cycles:.6f}"
        row["ideal_overlap_cycles_per_tile"] = f"{ideal_overlap_cycles:.6f}"
        if row["case"] in {"serial-a1", "overlap-a2", "warp-split-a2", "mainloop-a2-k16"}:
            row["serial_gain"] = f"{(serial_unit_cycles / unit_cycles) if unit_cycles else 0.0:.3f}"
            row["efficiency_vs_component_overlap_pct"] = (
                f"{(100.0 * ideal_overlap_cycles / unit_cycles) if unit_cycles else 0.0:.3f}"
            )
        else:
            row["serial_gain"] = "0.000"
            row["efficiency_vs_component_overlap_pct"] = "0.000"
    return rows


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_sass():
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = []
    for case in CASES:
        proc = run(["cuobjdump", "--dump-sass", str(case["binary"])])
        lines = []
        counts = {needle: 0 for needle in case["sass_needles"]}
        for line in proc.stdout.splitlines():
            if any(tok in line for tok in ["Function :", "UTCCP", "UTCOMMA", "UTC", "tmem["]):
                lines.append(line)
            for needle in counts:
                if needle in line:
                    counts[needle] += 1
        (RESULTS / f"{case['case']}.sass_summary.txt").write_text("\n".join(lines) + "\n")
        row = {"case": case["case"], "role": case["role"]}
        for needle in case["sass_needles"]:
            key = "count_" + needle.replace("[", "").replace("]", "").replace(".", "_")
            row[key] = counts[needle]
        summary.append(row)
    fieldnames = sorted({k for row in summary for k in row.keys()})
    write_csv(RESULTS / "sass_summary.csv", summary, fieldnames)


def write_app_report(rows, path):
    lines = [
        "# Pipeline overlap app report",
        "",
        "|case|role|TFLOP/s|cp B/cycle|consume B/cycle est.|cycles/cp|cycles/tile|serial gain|ideal-overlap eff.|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"|{r['case']}|{r['role']}|{float(r['tflops']):.3f}|"
            f"{float(r['cp_bytes_per_cycle']):.3f}|"
            f"{float(r['estimated_tmem_consume_bytes_per_cycle']):.3f}|"
            f"{float(r['cycles_per_cp']):.3f}|"
            f"{float(r['cycles_per_tile']):.3f}|"
            f"{float(r['serial_gain']):.3f}|"
            f"{float(r['efficiency_vs_component_overlap_pct']):.3f}%|"
        )
    path.write_text("\n".join(lines) + "\n")


def run_once():
    require_binaries()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = add_derived([app_data(case, repeat=0) for case in CASES])
    write_csv(RESULTS / "pipeline_overlap_results.csv", rows)
    collect_sass()
    write_app_report(rows, RESULTS / "pipeline_overlap_report.md")
    write_review()


def validate():
    require_binaries()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        for rep in range(3):
            rows.append(app_data(case, repeat=rep))
    rows = add_derived(rows)
    write_csv(RESULTS / "pipeline_overlap_validation.csv", rows)
    summary = []
    for case_name in [c["case"] for c in CASES]:
        case_rows = [r for r in rows if r["case"] == case_name]
        tflops = [float(r["tflops"]) for r in case_rows]
        cp_bpc = [float(r["cp_bytes_per_cycle"]) for r in case_rows]
        consume_bpc = [float(r["estimated_tmem_consume_bytes_per_cycle"]) for r in case_rows]
        cycles_tile = [float(r["cycles_per_tile"]) for r in case_rows]
        summary.append({
            "case": case_name,
            "role": case_rows[0]["role"],
            "tflops_median": statistics.median(tflops),
            "tflops_min": min(tflops),
            "tflops_max": max(tflops),
            "cp_bytes_per_cycle_median": statistics.median(cp_bpc),
            "estimated_tmem_consume_bytes_per_cycle_median": statistics.median(consume_bpc),
            "cycles_per_tile_median": statistics.median(cycles_tile),
            "repeat_count": len(case_rows),
        })
    write_csv(RESULTS / "pipeline_overlap_validation_summary.csv", summary)
    collect_sass()
    write_review()


def supported_metrics():
    proc = run(["ncu", "--query-metrics", "--query-metrics-mode", "all"])
    names = {line.split()[0] for line in proc.stdout.splitlines() if line and not line[0].isspace()}
    return [m for m in NCU_METRICS if m in names], [m for m in NCU_METRICS if m not in names]


def find_ncu_row(path):
    rows = list(csv.reader(path.open(newline="")))
    for i, row in enumerate(rows):
        if "ID" in row and "Kernel Name" in row:
            kidx = row.index("Kernel Name")
            for value in rows[i + 1:]:
                if len(value) == len(row) and value and value[0].isdigit() and "tcgen05_kernel" in value[kidx]:
                    return dict(zip(row, value))
    raise RuntimeError(f"no tcgen05_kernel row in {path}")


def ncu_float(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return ""


def ncu():
    require_binaries()
    NCU_DIR.mkdir(parents=True, exist_ok=True)
    metrics, missing = supported_metrics()
    app_rows = {r["case"]: r for r in add_derived([app_data(case, repeat=0) for case in CASES])}
    out_rows = []
    for case in CASES:
        raw = NCU_DIR / f"{case['case']}_ncu_raw.csv"
        report = NCU_DIR / f"{case['case']}_validation"
        cmd = [
            "ncu", "--target-processes", "all", "--replay-mode", "kernel",
            "--page", "raw", "--csv", "--force-overwrite", "-o", str(report),
            "--metrics", ",".join(metrics),
            str(case["binary"]), str(ITERS), FREQ_HZ,
        ]
        with raw.open("w") as f:
            proc = subprocess.run(cmd, text=True, stdout=f, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)
        row = find_ncu_row(raw)
        app = app_rows[case["case"]]
        sm_pct = ncu_float(row, "sm__throughput.avg.pct_of_peak_sustained_active")
        cp_bpc = float(app["cp_bytes_per_cycle"])
        consume_bpc = float(app["estimated_tmem_consume_bytes_per_cycle"])
        rough_cp_peak = ""
        rough_consume_peak = ""
        if isinstance(sm_pct, float) and sm_pct > 0:
            rough_cp_peak = cp_bpc / (sm_pct / 100.0) if cp_bpc > 0 else 0.0
            rough_consume_peak = consume_bpc / (sm_pct / 100.0) if consume_bpc > 0 else 0.0
        out_rows.append({
            "case": case["case"],
            "role": case["role"],
            "app_tflops": app["tflops"],
            "app_cp_bytes_per_cycle": app["cp_bytes_per_cycle"],
            "app_estimated_tmem_consume_bytes_per_cycle": app["estimated_tmem_consume_bytes_per_cycle"],
            "rough_cp_upper_from_sm_peak_bytes_per_cycle": rough_cp_peak,
            "rough_consume_upper_from_sm_peak_bytes_per_cycle": rough_consume_peak,
            "sm_throughput_pct_peak": sm_pct,
            "tensor_inst": ncu_float(row, "sm__inst_executed_pipe_tensor.sum"),
            "tensor_inst_per_cycle": ncu_float(row, "sm__inst_executed_pipe_tensor.sum.per_cycle_active"),
            "tensor_inst_pct_peak": ncu_float(row, "sm__inst_executed_pipe_tensor.sum.pct_of_peak_sustained_active"),
            "tc_inst": ncu_float(row, "sm__inst_executed_pipe_tc.sum"),
            "tc_inst_per_cycle": ncu_float(row, "sm__inst_executed_pipe_tc.sum.per_cycle_active"),
            "tc_inst_pct_peak": ncu_float(row, "sm__inst_executed_pipe_tc.sum.pct_of_peak_sustained_active"),
            "tmem_pipe_inst": ncu_float(row, "sm__inst_executed_pipe_tmem.sum"),
            "tmem_pipe_inst_per_cycle": ncu_float(row, "sm__inst_executed_pipe_tmem.sum.per_cycle_active"),
            "tmem_pipe_inst_pct_peak": ncu_float(row, "sm__inst_executed_pipe_tmem.sum.pct_of_peak_sustained_active"),
            "issue_active_pct": ncu_float(row, "smsp__issue_active.avg.pct_of_peak_sustained_active"),
            "eligible_warps_per_cycle": ncu_float(row, "smsp__warps_eligible.avg.per_cycle_active"),
            "wait_stall_pct": ncu_float(row, "smsp__average_warps_issue_stalled_wait_per_issue_active.pct"),
            "dispatch_stall_pct": ncu_float(row, "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.pct"),
            "math_pipe_throttle_pct": ncu_float(row, "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.pct"),
            "mio_throttle_pct": ncu_float(row, "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.pct"),
            "gpu_compute_memory_throughput_pct_peak": ncu_float(row, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_active"),
            "missing_metric_count": len(missing),
            "missing_metrics": ";".join(missing),
        })
    write_csv(NCU_DIR / "ncu_pipeline_overlap_summary.csv", out_rows)
    write_ncu_report(out_rows)
    write_review()


def fmt(value):
    if value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def write_ncu_report(rows):
    lines = [
        "# Pipeline overlap NCU report",
        "",
        "|case|SM %peak|tensor %peak|cp B/cycle|consume B/cycle est.|rough cp upper|rough consume upper|wait stall %|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"|{r['case']}|{fmt(r['sm_throughput_pct_peak'])}|{fmt(r['tensor_inst_pct_peak'])}|"
            f"{fmt(r['app_cp_bytes_per_cycle'])}|{fmt(r['app_estimated_tmem_consume_bytes_per_cycle'])}|"
            f"{fmt(r['rough_cp_upper_from_sm_peak_bytes_per_cycle'])}|"
            f"{fmt(r['rough_consume_upper_from_sm_peak_bytes_per_cycle'])}|"
            f"{fmt(r['wait_stall_pct'])}|"
        )
    (NCU_DIR / "ncu_pipeline_overlap_report.md").write_text("\n".join(lines) + "\n")


def load_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sass_ok():
    path = RESULTS / "sass_summary.csv"
    rows = load_csv(path)
    if not rows:
        return False, "missing sass_summary.csv"
    by_case = {r["case"]: r for r in rows}
    for case in CASES:
        row = by_case.get(case["case"])
        if not row:
            return False, f"missing SASS row for {case['case']}"
        for needle in case["sass_needles"]:
            key = "count_" + needle.replace("[", "").replace("]", "").replace(".", "_")
            if float(row.get(key, "0") or 0) <= 0:
                return False, f"{case['case']} missing SASS needle {needle}"
    return True, "SASS needles present"


def write_review():
    app_rows = load_csv(RESULTS / "pipeline_overlap_results.csv")
    val_rows = load_csv(RESULTS / "pipeline_overlap_validation_summary.csv")
    ncu_rows = load_csv(NCU_DIR / "ncu_pipeline_overlap_summary.csv")
    source_rows = app_rows
    if val_rows:
        source_rows = [
            {
                "case": r["case"],
                "role": r["role"],
                "tflops": r["tflops_median"],
                "cp_bytes_per_cycle": r["cp_bytes_per_cycle_median"],
                "estimated_tmem_consume_bytes_per_cycle": r["estimated_tmem_consume_bytes_per_cycle_median"],
                "cycles_per_tile": r["cycles_per_tile_median"],
            }
            for r in val_rows
        ]
    ok = True
    reasons = []
    by_case = {r["case"]: r for r in source_rows}
    if {"serial-a1", "overlap-a2", "mainloop-a2-k16", "cp-only", "ts-mma-only"}.issubset(by_case):
        serial = float(by_case["serial-a1"]["cycles_per_tile"])
        overlap = float(by_case["overlap-a2"]["cycles_per_tile"])
        mainloop_cp = float(load_csv(RESULTS / "pipeline_overlap_results.csv")[5]["cycles_per_cp"]) if app_rows else 0.0
        gain = serial / overlap if overlap else 0.0
        if gain < 1.2:
            ok = False
            reasons.append(f"overlap gain too small: {gain:.3f}")
        else:
            reasons.append(f"overlap gain serial/overlap = {gain:.3f}")
        if mainloop_cp and abs(mainloop_cp - overlap) / overlap > 0.15:
            ok = False
            reasons.append(f"mainloop cycles/cp {mainloop_cp:.3f} not close to overlap {overlap:.3f}")
        elif mainloop_cp:
            reasons.append(f"mainloop cycles/cp {mainloop_cp:.3f} close to overlap {overlap:.3f}")
    else:
        ok = False
        reasons.append("missing app/validation rows")

    sass_pass, sass_reason = sass_ok()
    ok &= sass_pass
    reasons.append(sass_reason)
    if not ncu_rows:
        ok = False
        reasons.append("missing NCU summary")
    else:
        reasons.append("NCU summary present")

    status = "通过" if ok else "未通过"
    lines = [
        "# Pipeline overlap 运行对抗式审查",
        "",
        f"结论：{status}。",
        "",
        "## 判据结果",
        "",
    ]
    for reason in reasons:
        lines.append(f"- {reason}")
    lines += [
        "",
        "## 证据和解释",
        "",
    ]
    if by_case:
        serial = float(by_case.get("serial-a1", {}).get("cycles_per_tile", "0") or 0)
        overlap = float(by_case.get("overlap-a2", {}).get("cycles_per_tile", "0") or 0)
        cp_peak = float(by_case.get("cp-only", {}).get("cp_bytes_per_cycle", "0") or 0)
        ts_peak = float(by_case.get("ts-mma-only", {}).get("estimated_tmem_consume_bytes_per_cycle", "0") or 0)
        overlap_cp = float(by_case.get("overlap-a2", {}).get("cp_bytes_per_cycle", "0") or 0)
        overlap_consume = float(by_case.get("overlap-a2", {}).get("estimated_tmem_consume_bytes_per_cycle", "0") or 0)
        lines += [
            f"- `cp-only` measured cp ingress is `{cp_peak:.3f} B/cycle/GPU`.",
            f"- `ts-mma-only` estimated TMEM consume demand is `{ts_peak:.3f} B/cycle/GPU`.",
            f"- `serial-a1` cycle/tile is `{serial:.3f}`; `overlap-a2` is `{overlap:.3f}`.",
            f"- `overlap-a2` cp payload is `{overlap_cp:.3f} B/cycle/GPU`, "
            f"`{(100.0 * overlap_cp / cp_peak) if cp_peak else 0.0:.2f}%` of cp-only.",
            f"- `overlap-a2` consume demand is `{overlap_consume:.3f} B/cycle/GPU`, "
            f"`{(100.0 * overlap_consume / ts_peak) if ts_peak else 0.0:.2f}%` of TS-MMA-only.",
        ]
    if ncu_rows:
        by_ncu = {r["case"]: r for r in ncu_rows}
        r = by_ncu.get("overlap-a2", {})
        if r:
            lines += [
                f"- NCU overlap-a2 SM throughput is `{fmt(r.get('sm_throughput_pct_peak', ''))}% peak`; "
                f"rough cp upper from SM peak is `{fmt(r.get('rough_cp_upper_from_sm_peak_bytes_per_cycle', ''))} B/cycle/GPU`, "
                f"rough consume upper is `{fmt(r.get('rough_consume_upper_from_sm_peak_bytes_per_cycle', ''))} B/cycle/GPU`.",
            ]
    lines += [
        "",
        "## 保留边界",
        "",
        "- `bytes_per_cycle` 是 cp payload，不是总 TMEM 双向带宽。",
        "- consume bandwidth 是按 TS MMA operand demand 估计，不是 raw TMEM read-port counter。",
        "- NCU `sm__inst_executed_pipe_tmem.*` 在当前工具链可能为 0；不把它作为 UTCCP 是否执行的判据。",
    ]
    (RESULTS / "adversarial_review.md").write_text("\n".join(lines) + "\n")
    if not ok:
        return False
    return True


def main(argv):
    if len(argv) < 2:
        cmd = "run"
    else:
        cmd = argv[1]
    if cmd == "run":
        run_once()
    elif cmd == "validate":
        validate()
    elif cmd == "ncu":
        ncu()
    elif cmd == "review":
        ok = write_review()
        if not ok:
            sys.exit(1)
    else:
        print("Usage: pipeline_overlap.py run|validate|ncu|review", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
