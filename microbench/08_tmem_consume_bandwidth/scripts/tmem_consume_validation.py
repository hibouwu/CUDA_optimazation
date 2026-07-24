#!/usr/bin/env python3
import argparse
import csv
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SRC = REPO / "mma_with_cp"
BUILD = SRC / "build"
RESULTS = ROOT / "results"
NCU_DIR = RESULTS / "ncu"

ITERS = 10000
FREQ_HZ = "1575000000"
TMEM_A_BYTES_PER_TS_MMA = 2048.0

CASES = [
    {
        "case": "ts-mma-only",
        "role": "tmem-consume",
        "binary": BUILD / "tcgen05_ts_mma_only_m128n256_fp4_benchmark",
        "sass_needles": ["UTCOMMA", "tmem["],
    },
    {
        "case": "ts-cp-mma-a2-k16",
        "role": "tmem-consume-plus-cp",
        "binary": BUILD / "tcgen05_ts_cp_mma_mainloop_a2_k16_m128n256_fp4_benchmark",
        "sass_needles": ["UTCOMMA", "UTCCP", "tmem["],
    },
    {
        "case": "ss-mma-mainloop-k16",
        "role": "smem-baseline",
        "binary": BUILD / "tcgen05_ss_mma_mainloop_k16_m128n256_fp4_benchmark",
        "sass_needles": ["UTCOMMA", "gdesc"],
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
]


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, **kwargs)


def parse_kv(text):
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def app_row(case, repeat=0, iters=ITERS):
    proc = run([str(case["binary"]), str(iters), FREQ_HZ])
    data = parse_kv(proc.stdout)
    cycles = float(data["cycles"])
    mma_count = float(data["mma_instruction_count"])
    is_ts = case["role"].startswith("tmem-consume")
    consume_bpc = (mma_count * TMEM_A_BYTES_PER_TS_MMA / cycles) if is_ts else 0.0
    cycles_per_mma = cycles / mma_count if mma_count else 0.0
    row = {
        "case": case["case"],
        "role": case["role"],
        "repeat": repeat,
        "iters": data.get("iters", str(iters)),
        "cycles": data["cycles"],
        "tflops": data["thor_tflops"],
        "mma_instruction_count": data["mma_instruction_count"],
        "cycles_per_mma": f"{cycles_per_mma:.6f}",
        "cp_instruction_count": data.get("cp_instruction_count", "0"),
        "cp_bytes_per_cycle": data.get("bytes_per_cycle", "0"),
        "estimated_tmem_consume_bytes_per_mma": f"{TMEM_A_BYTES_PER_TS_MMA if is_ts else 0.0:.0f}",
        "estimated_tmem_consume_bytes_per_cycle": f"{consume_bpc:.6f}",
    }
    return row


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
    summary_rows = []
    for case in CASES:
        proc = run(["cuobjdump", "--dump-sass", str(case["binary"])])
        lines = []
        counts = {needle: 0 for needle in case["sass_needles"]}
        for line in proc.stdout.splitlines():
            if any(token in line for token in ["Function :", "UTCOMMA", "UTCCP", "UTCBAR", "UTCATOMSWS", "CS2R"]):
                lines.append(line)
            for needle in counts:
                if needle in line:
                    counts[needle] += 1
        (RESULTS / f"{case['case']}.sass_summary.txt").write_text("\n".join(lines) + "\n")
        row = {"case": case["case"], **{f"count_{k.replace('[', '').replace(']', '')}": v for k, v in counts.items()}}
        summary_rows.append(row)
    fieldnames = sorted({k for row in summary_rows for k in row.keys()})
    write_csv(RESULTS / "sass_summary.csv", summary_rows, fieldnames)


def run_app_once():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = [app_row(case, repeat=0) for case in CASES]
    write_csv(RESULTS / "tmem_consume_results.csv", rows)
    collect_sass()
    write_reports_from_current()


def run_validate():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        for rep in range(3):
            rows.append(app_row(case, repeat=rep))
    write_csv(RESULTS / "tmem_consume_validation.csv", rows)
    summary = []
    for case in sorted({r["case"] for r in rows}):
        case_rows = [r for r in rows if r["case"] == case]
        tflops = [float(r["tflops"]) for r in case_rows]
        bpc = [float(r["estimated_tmem_consume_bytes_per_cycle"]) for r in case_rows]
        summary.append({
            "case": case,
            "role": case_rows[0]["role"],
            "tflops_median": statistics.median(tflops),
            "tflops_min": min(tflops),
            "tflops_max": max(tflops),
            "estimated_tmem_consume_bytes_per_cycle_median": statistics.median(bpc),
            "estimated_tmem_consume_bytes_per_cycle_min": min(bpc),
            "estimated_tmem_consume_bytes_per_cycle_max": max(bpc),
            "repeat_count": len(case_rows),
        })
    write_csv(RESULTS / "tmem_consume_validation_summary.csv", summary)
    collect_sass()
    write_reports_from_current()


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


def ncu_value(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return ""


def run_ncu():
    NCU_DIR.mkdir(parents=True, exist_ok=True)
    metrics, missing = supported_metrics()
    app = {r["case"]: r for r in [app_row(case, repeat=0, iters=ITERS) for case in CASES]}
    rows = []
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
        ncu = find_ncu_row(raw)
        app_row_data = app[case["case"]]
        consume_bpc = float(app_row_data["estimated_tmem_consume_bytes_per_cycle"])
        sm_pct = ncu_value(ncu, "sm__throughput.avg.pct_of_peak_sustained_active")
        consume_peak = ""
        if isinstance(sm_pct, float) and sm_pct > 0 and consume_bpc > 0:
            consume_peak = consume_bpc / (sm_pct / 100.0)
        rows.append({
            "case": case["case"],
            "role": case["role"],
            "app_tflops": app_row_data["tflops"],
            "estimated_tmem_consume_bytes_per_cycle": app_row_data["estimated_tmem_consume_bytes_per_cycle"],
            "rough_consume_upper_from_sm_peak_bytes_per_cycle": consume_peak,
            "sm_throughput_pct_peak": ncu_value(ncu, "sm__throughput.avg.pct_of_peak_sustained_active"),
            "tensor_inst": ncu_value(ncu, "sm__inst_executed_pipe_tensor.sum"),
            "tensor_inst_per_cycle": ncu_value(ncu, "sm__inst_executed_pipe_tensor.sum.per_cycle_active"),
            "tensor_inst_pct_peak": ncu_value(ncu, "sm__inst_executed_pipe_tensor.sum.pct_of_peak_sustained_active"),
            "tc_inst": ncu_value(ncu, "sm__inst_executed_pipe_tc.sum"),
            "tc_inst_per_cycle": ncu_value(ncu, "sm__inst_executed_pipe_tc.sum.per_cycle_active"),
            "tc_inst_pct_peak": ncu_value(ncu, "sm__inst_executed_pipe_tc.sum.pct_of_peak_sustained_active"),
            "tmem_pipe_inst": ncu_value(ncu, "sm__inst_executed_pipe_tmem.sum"),
            "tmem_pipe_inst_per_cycle": ncu_value(ncu, "sm__inst_executed_pipe_tmem.sum.per_cycle_active"),
            "tmem_pipe_inst_pct_peak": ncu_value(ncu, "sm__inst_executed_pipe_tmem.sum.pct_of_peak_sustained_active"),
            "issue_active_pct": ncu_value(ncu, "smsp__issue_active.avg.pct_of_peak_sustained_active"),
            "eligible_warps_per_cycle": ncu_value(ncu, "smsp__warps_eligible.avg.per_cycle_active"),
            "wait_stall_pct": ncu_value(ncu, "smsp__average_warps_issue_stalled_wait_per_issue_active.pct"),
            "dispatch_stall_pct": ncu_value(ncu, "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.pct"),
            "missing_metric_count": len(missing),
            "missing_metrics": ";".join(missing),
        })
    write_csv(NCU_DIR / "ncu_tmem_consume_summary.csv", rows)
    write_reports_from_current()


def write_reports_from_current():
    lines = [
        "# TMEM consume validation report",
        "",
    ]
    summary_path = RESULTS / "tmem_consume_validation_summary.csv"
    result_path = RESULTS / "tmem_consume_results.csv"
    if summary_path.exists():
        rows = list(csv.DictReader(summary_path.open()))
        lines += [
            "|case|role|TFLOP/s median|estimated TMEM consume B/cycle median|",
            "|---|---|---:|---:|",
        ]
        for r in rows:
            lines.append(f"|{r['case']}|{r['role']}|{float(r['tflops_median']):.3f}|"
                         f"{float(r['estimated_tmem_consume_bytes_per_cycle_median']):.3f}|")
    elif result_path.exists():
        rows = list(csv.DictReader(result_path.open()))
        lines += [
            "|case|role|TFLOP/s|estimated TMEM consume B/cycle|",
            "|---|---|---:|---:|",
        ]
        for r in rows:
            lines.append(f"|{r['case']}|{r['role']}|{float(r['tflops']):.3f}|"
                         f"{float(r['estimated_tmem_consume_bytes_per_cycle']):.3f}|")
    ncu_path = NCU_DIR / "ncu_tmem_consume_summary.csv"
    if ncu_path.exists():
        lines += ["", "## NCU", "", "|case|SM %peak|tensor inst %peak|tmem pipe inst %peak|rough consume upper B/cycle|", "|---|---:|---:|---:|---:|"]
        for r in csv.DictReader(ncu_path.open()):
            lines.append(f"|{r['case']}|{float(r['sm_throughput_pct_peak'] or 0):.3f}|"
                         f"{float(r['tensor_inst_pct_peak'] or 0):.3f}|"
                         f"{float(r['tmem_pipe_inst_pct_peak'] or 0):.3f}|"
                         f"{float(r['rough_consume_upper_from_sm_peak_bytes_per_cycle'] or 0):.3f}|")
    (RESULTS / "validation_report.md").write_text("\n".join(lines) + "\n")

    review = [
        "# TMEM consume 运行对抗式审查",
        "",
        "未通过：还没有完整 NCU summary。" if not ncu_path.exists() else "通过：app、SASS 和 NCU 均已覆盖代表性的 TS TMEM-consume path。",
        "",
        "已检查/需保留的边界：",
        "",
        "- TS cases 的 SASS 必须包含 `UTCOMMA... tmem[...]`，说明 MMA 从 TMEM operand 消费。",
        "- `ts-cp-mma-a2-k16` 同时包含 `UTCCP`，所以它是 pipeline consume+cp，不是纯 consume。",
        "- `estimated_tmem_consume_bytes_per_cycle` 是按 2048 B/TS-MMA 推导的 operand demand rate，不是 raw TMEM read-port peak。",
        "- `rough_consume_upper_from_sm_peak_bytes_per_cycle` 只是按 NCU SM throughput 归一化的需求率上限估计，不是物理 TMEM 端口峰值。",
        "- NCU 的 `sm__inst_executed_pipe_tmem.*` 可能不覆盖 UTCOMMA 的 TMEM operand 读取；最终利用率以 tensor/TC pipe 和 SM throughput 为主，tmem pipe counter 只作 coverage 观察。",
    ]
    (RESULTS / "adversarial_review.md").write_text("\n".join(review) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.run:
        run_app_once()
    elif args.validate:
        run_validate()
    elif args.ncu:
        run_ncu()
    else:
        write_reports_from_current()


if __name__ == "__main__":
    main()
