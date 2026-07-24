#!/usr/bin/env python3
import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "smem_bank_stride_bandwidth"
RESULTS = ROOT / "results"
NCU_DIR = RESULTS / "ncu"
MODES = ["read", "write"]
STRIDES = [1, 2, 4, 8, 16, 32]

CANDIDATE_METRICS = [
    "gpu__time_duration.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.per_cycle_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_cycle_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.pct_of_peak_sustained_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_cycle_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
]


def run(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def query_supported():
    proc = run(["ncu", "--query-metrics", "--query-metrics-mode", "all"])
    names = {line.split()[0] for line in proc.stdout.splitlines() if line and not line[0].isspace()}
    return [m for m in CANDIDATE_METRICS if m in names], [m for m in CANDIDATE_METRICS if m not in names]


def app_probe(mode, stride):
    header = run([str(BIN), "--csv-header"]).stdout.strip().split(",")
    out = run([
        str(BIN), "--mode", mode, "--stride-words", str(stride),
        "--iters", "8192", "--warmup-iters", "128",
        "--threads", "256", "--blocks-per-sm", "1",
        "--shared-words", "8192", "--csv",
    ]).stdout.strip()
    return next(csv.DictReader([",".join(header), out]))


def find_row(path):
    rows = list(csv.reader(path.open(newline="")))
    for i, row in enumerate(rows):
        if "ID" in row and "Kernel Name" in row:
            tidx = row.index("gpu__time_duration.sum") if "gpu__time_duration.sum" in row else None
            candidates = []
            for value in rows[i + 1:]:
                if len(value) != len(row) or not value or not value[0].isdigit():
                    continue
                try:
                    t = float(value[tidx]) if tidx is not None else 0.0
                except ValueError:
                    continue
                candidates.append((t, value))
            if not candidates:
                raise RuntimeError(f"no data rows in {path}")
            return dict(zip(row, max(candidates, key=lambda x: x[0])[1]))
    raise RuntimeError(f"no NCU table in {path}")


def val(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return ""


def run_ncu(mode, stride, metrics):
    raw = NCU_DIR / f"{mode}_stride{stride}_ncu_raw.csv"
    report = NCU_DIR / f"{mode}_stride{stride}_validation"
    cmd = [
        "ncu", "--target-processes", "all", "--replay-mode", "kernel",
        "--page", "raw", "--csv", "--force-overwrite", "-o", str(report),
        "--metrics", ",".join(metrics),
        str(BIN), "--mode", mode, "--stride-words", str(stride),
        "--iters", "8192", "--warmup-iters", "128",
        "--threads", "256", "--blocks-per-sm", "1",
        "--shared-words", "8192", "--csv",
    ]
    with raw.open("w") as f:
        proc = subprocess.run(cmd, text=True, stdout=f, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return raw


def make_row(mode, stride, app, ncu, missing):
    wave = val(ncu, "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum")
    wave_bpc = val(ncu, "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.per_cycle_elapsed")
    wave_pct = val(ncu, "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed")
    app_bpc = float(app["bytes_per_cycle"])
    peak = ""
    if isinstance(wave_pct, float) and wave_pct > 0:
        peak = app_bpc / (wave_pct / 100.0)
    return {
        "mode": mode,
        "stride_words": stride,
        "app_bytes_per_cycle": app["bytes_per_cycle"],
        "wavefronts": wave,
        "wavefronts_per_cycle_elapsed": wave_bpc,
        "wavefronts_pct_peak_elapsed": wave_pct,
        "payload_normalized_peak_bytes_per_cycle": peak,
        "ld_wavefronts": val(ncu, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
        "st_wavefronts": val(ncu, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"),
        "bank_conflicts": val(ncu, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"),
        "ld_bank_conflicts": val(ncu, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"),
        "st_bank_conflicts": val(ncu, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"),
        "missing_metric_count": len(missing),
        "missing_metrics": ";".join(missing),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x):
    if x == "":
        return ""
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def write_reports(rows):
    lines = [
        "# NCU SMEM bank/stride validation report",
        "",
        "|mode|stride|app B/cycle|wavefront %peak|bank conflicts|payload-normalized peak B/cycle|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"|{r['mode']}|{r['stride_words']}|{fmt(r['app_bytes_per_cycle'])}|"
                     f"{fmt(r['wavefronts_pct_peak_elapsed'])}|{fmt(r['bank_conflicts'])}|"
                     f"{fmt(r['payload_normalized_peak_bytes_per_cycle'])}|")
    (NCU_DIR / "ncu_smem_bank_report.md").write_text("\n".join(lines) + "\n")

    by_mode = {(r["mode"], int(r["stride_words"])): r for r in rows}
    ok = True
    for mode in MODES:
        ok &= float(by_mode[(mode, 1)]["app_bytes_per_cycle"]) > float(by_mode[(mode, 32)]["app_bytes_per_cycle"])
        ok &= float(by_mode[(mode, 32)]["bank_conflicts"]) > float(by_mode[(mode, 1)]["bank_conflicts"])
    status = "通过" if ok else "未通过"
    reason = "stride sweep shows throughput drops and NCU bank conflicts rise with high-conflict strides." if ok else "stride/conflict trend is not proven by app+NCU."
    (RESULTS / "adversarial_review.md").write_text(
        "# SMEM bank/stride 运行对抗式审查\n\n"
        f"{status}：{reason}\n\n"
        "本机无 local shared direct byte counter；NCU 证据使用 shared wavefront 与 bank-conflict counters。\n"
    )


def main():
    NCU_DIR.mkdir(parents=True, exist_ok=True)
    metrics, missing = query_supported()
    rows = []
    for mode in MODES:
        for stride in STRIDES:
            app = app_probe(mode, stride)
            raw = run_ncu(mode, stride, metrics)
            rows.append(make_row(mode, stride, app, find_row(raw), missing))
    write_csv(NCU_DIR / "ncu_smem_bank_summary.csv", rows)
    write_reports(rows)
    print(f"Wrote {NCU_DIR / 'ncu_smem_bank_summary.csv'}")


if __name__ == "__main__":
    main()
