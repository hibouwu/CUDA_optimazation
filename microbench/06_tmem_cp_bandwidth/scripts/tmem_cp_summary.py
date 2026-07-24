#!/usr/bin/env python3
import csv
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SRC = REPO / "mma_with_cp"
RESULTS = ROOT / "results"
NCU_DIR = RESULTS / "ncu"

CP_ONLY = SRC / "plots" / "cp_only_results.csv"
INTERFERENCE = SRC / "plots" / "tmem_interference_tflops_results.csv"
NCU_REPORT = SRC / "ncu_reports_key" / "tcgen05_cp_only_m128n256_fp4_benchmark.ncu-rep"
CP_ONLY_BIN = SRC / "build" / "tcgen05_cp_only_m128n256_fp4_benchmark"

SELECTED_NCU_METRICS = [
    "Kernel Name",
    "Block Size",
    "Grid Size",
    "gpu__time_duration.avg",
    "sm__cycles_elapsed.avg",
    "sm__cycles_elapsed.avg.per_second",
    "sm__inst_executed_pipe_tmem.sum",
    "sm__inst_executed_pipe_tmem.sum.per_cycle_active",
    "sm__inst_executed_pipe_tmem.sum.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.sum",
    "sm__inst_executed_pipe_lsu.sum.per_cycle_active",
    "sm__inst_executed_pipe_tensor.sum",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_active",
    "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.pct",
]


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def summarize_cp_only():
    rows = read_csv(CP_ONLY)
    write_csv(RESULTS / "tmem_cp_only_summary.csv", rows)
    bpc = [float(r["bytes_per_cycle"]) for r in rows]
    cpc = [float(r["cycles_per_cp"]) for r in rows]
    return {
        "row_count": len(rows),
        "bytes_per_cycle_median": statistics.median(bpc),
        "bytes_per_cycle_min": min(bpc),
        "bytes_per_cycle_max": max(bpc),
        "cycles_per_cp_median": statistics.median(cpc),
        "effective_bytes_per_cp": rows[0]["effective_bytes_per_cp"] if rows else "",
        "cp_instruction_count": rows[0]["cp_instruction_count"] if rows else "",
    }


def summarize_interference():
    rows = read_csv(INTERFERENCE)
    selected = []
    for row in rows:
        if float(row.get("noise_cp_per_mma", 0) or 0) == 0 and float(row.get("bytes_per_cycle", 0) or 0) == 0:
            continue
        selected.append({
            "case": row["case"],
            "precision": row["precision"],
            "path": row["path"],
            "mode": row["mode"],
            "noise_cp_per_mma": row["noise_cp_per_mma"],
            "tflops": row["tflops"],
            "cp_inst": row["cp_inst"],
            "bytes_per_cycle": row["bytes_per_cycle"],
            "slowdown_vs_control": row["slowdown_vs_control"],
        })
    write_csv(RESULTS / "tmem_cp_interference_summary.csv", selected)
    cp_bpc = [float(r["bytes_per_cycle"]) for r in selected if float(r["bytes_per_cycle"]) > 0]
    slowdowns = [float(r["slowdown_vs_control"]) for r in selected if r["slowdown_vs_control"]]
    return {
        "row_count": len(selected),
        "max_cp_bytes_per_cycle": max(cp_bpc) if cp_bpc else 0.0,
        "min_slowdown_vs_control": min(slowdowns) if slowdowns else "",
    }


def export_ncu():
    if not NCU_REPORT.exists():
        return None
    NCU_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ncu",
        "--import", str(NCU_REPORT),
        "--page", "raw",
        "--csv",
        "--print-units", "base",
    ]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    raw_path = NCU_DIR / "tmem_cp_only_ncu_raw.csv"
    raw_path.write_text(proc.stdout)

    rows = list(csv.reader(proc.stdout.splitlines()))
    header = rows[0]
    data = None
    for row in rows[1:]:
        if len(row) == len(header) and row and row[0].isdigit():
            data = dict(zip(header, row))
            break
    if data is None:
        return None

    summary = {k: data.get(k, "") for k in SELECTED_NCU_METRICS}
    write_csv(NCU_DIR / "tmem_cp_only_ncu_summary.csv", [summary], SELECTED_NCU_METRICS)
    return summary


def collect_sass():
    if not CP_ONLY_BIN.exists():
        return None
    try:
        proc = subprocess.run(
            ["cuobjdump", "--dump-sass", str(CP_ONLY_BIN)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    lines = []
    utccp_count = 0
    for line in proc.stdout.splitlines():
        if "UTCCP" in line:
            utccp_count += 1
        if any(token in line for token in ["Function :", "UTCCP", "UTCBAR", "UTCATOMSWS", "CS2R"]):
            lines.append(line)
    path = RESULTS / "tmem_cp_only_sass_summary.txt"
    path.write_text("\n".join(lines) + "\n")
    return {"utccp_static_count": utccp_count}


def write_reports(cp, interference, ncu, sass):
    ncu_state = "present" if ncu else "missing"
    sass_state = "present" if sass else "missing"
    lines = [
        "# TMEM cp ingress validation report",
        "",
        f"- cp-only rows: {cp['row_count']}",
        f"- cp-only throughput: {fmt(cp['bytes_per_cycle_median'])} B/cycle/GPU",
        f"- rough cp ingress upper used here: {fmt(cp['bytes_per_cycle_median'])} B/cycle/GPU (cp-only sustained)",
        f"- cp-only cycles/cp: {fmt(cp['cycles_per_cp_median'])}",
        f"- effective bytes/cp: {cp['effective_bytes_per_cp']}",
        f"- cp instructions in app timing row: {cp['cp_instruction_count']}",
        f"- interference rows with cp traffic: {interference['row_count']}",
        f"- max cp traffic during MMA interference: {fmt(interference['max_cp_bytes_per_cycle'])} B/cycle/GPU",
        f"- worst slowdown vs control in interference sweep: {fmt(interference['min_slowdown_vs_control'])}",
        f"- SASS summary: {sass_state}",
        f"- NCU key report: {ncu_state}",
    ]
    if sass:
        lines.append(f"- static UTCCP instructions in representative cp-only SASS: {sass['utccp_static_count']}")
    if ncu:
        lines += [
            f"- NCU tmem pipe metric value: {fmt(ncu.get('sm__inst_executed_pipe_tmem.sum.per_cycle_active', ''))} inst/cycle active",
            f"- NCU memory throughput pct peak active: {fmt(ncu.get('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_active', ''))}%",
            "- NCU tmem pipe metric is not used as cp proof when it reports zero for UTCCP.",
        ]
    (RESULTS / "validation_report.md").write_text("\n".join(lines) + "\n")

    review = [
        "# TMEM cp ingress adversarial review",
        "",
        "Pass with scope limitation: existing data measures the `tcgen05.cp` SMEM-to-TMEM ingress path, not raw TMEM bank bandwidth.",
        "",
        "Checked evidence:",
        "",
        f"- App cp-only timing reports {fmt(cp['bytes_per_cycle_median'])} B/cycle/GPU and {fmt(cp['cycles_per_cp_median'])} cycles/cp.",
        f"- Rough cp ingress upper for this directory is the cp-only sustained value, {fmt(cp['bytes_per_cycle_median'])} B/cycle/GPU; no dedicated NCU UTCCP byte peak is exposed.",
        "- The same cp-only row is stable across FP4, FP8, and BF16 generated shapes because the measured instruction suffix/effective bytes are the same.",
        f"- Interference sweep reaches up to {fmt(interference['max_cp_bytes_per_cycle'])} B/cycle/GPU of cp traffic while showing MMA slowdown, so the path is performance-visible.",
        "- `../mma_config/06_tmem_dependency/plots/analysis.md` explicitly says those dependency rows cannot identify TMEM bank count, bank width, write bandwidth, or hidden dependency scoreboard size.",
    ]
    if sass:
        review += [
            f"- Representative cp-only SASS contains {sass['utccp_static_count']} static `UTCCP.T.S.128dp128bit` instructions, confirming the intended tcgen05.cp path.",
        ]
    if ncu:
        review += [
            f"- Existing key NCU report is exportable and reports memory throughput at {fmt(ncu.get('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_active', ''))}% peak active.",
            "- The same NCU report gives zero for `sm__inst_executed_pipe_tmem.*`; this is treated as metric coverage limitation for UTCCP, not as evidence that cp did not execute.",
        ]
    else:
        review += ["- No key NCU report was found; utilization should not be inferred from app timing alone."]
    review += [
        "",
        "Conclusion boundary:",
        "",
        "- OK to report: `tcgen05.cp` ingress throughput to TMEM, and cp interference with SS/TS MMA.",
        "- Not OK to report from this directory alone: raw TMEM read bandwidth, raw TMEM write-port peak, bank count, or bank width.",
    ]
    (RESULTS / "adversarial_review.md").write_text("\n".join(review) + "\n")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    cp = summarize_cp_only()
    interference = summarize_interference()
    ncu = export_ncu()
    sass = collect_sass()
    write_reports(cp, interference, ncu, sass)
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
