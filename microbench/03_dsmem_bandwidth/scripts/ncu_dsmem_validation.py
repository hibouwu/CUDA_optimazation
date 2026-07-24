#!/usr/bin/env python3
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "dsmem_bandwidth"
RESULTS = ROOT / "results"
NCU_DIR = RESULTS / "ncu"


MODES = ["local-read", "local-write", "remote-read", "remote-write"]


CANDIDATE_METRICS = [
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    "launch__shared_mem_per_block_dynamic",
    "launch__sm_count",
    "l1tex__t_bytes_pipe_lsu_mem_dshared.sum",
    "l1tex__t_bytes_pipe_lsu_mem_dshared.sum.per_cycle_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_dshared.sum.per_second",
    "l1tex__t_bytes_pipe_lsu_mem_dshared.sum.pct_of_peak_sustained_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_dshared_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_dshared_op_ld.sum.per_cycle_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_dshared_op_ld.sum.pct_of_peak_sustained_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_dshared_op_st.sum",
    "l1tex__t_bytes_pipe_lsu_mem_dshared_op_st.sum.per_cycle_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_dshared_op_st.sum.pct_of_peak_sustained_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.per_cycle_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_lgds.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_lgds.sum.per_cycle_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_lgds.sum.pct_of_peak_sustained_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_lgds_cmd_read.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_lgds_cmd_read.sum.pct_of_peak_sustained_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_lgds_cmd_write.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_lgds_cmd_write.sum.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_gds.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_gds_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_gds_op_st.sum",
]


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def query_supported_metrics():
    proc = run(["ncu", "--query-metrics", "--query-metrics-mode", "all"])
    names = set()
    for line in proc.stdout.splitlines():
        if not line or line[0].isspace():
            continue
        names.add(line.split()[0])
    supported = [m for m in CANDIDATE_METRICS if m in names]
    missing = [m for m in CANDIDATE_METRICS if m not in names]
    return supported, missing


def app_probe(mode):
    proc = run([
        str(BIN),
        "--mode", mode,
        "--cluster-size", "2",
        "--iters", "4096",
        "--warmup-iters", "64",
        "--threads", "256",
        "--shared-bytes", "32768",
        "--csv",
    ])
    header = run([str(BIN), "--csv-header"]).stdout.strip().split(",")
    return next(csv.DictReader([",".join(header), proc.stdout.strip()]))


def find_raw_table(path):
    rows = list(csv.reader(path.open(newline="")))
    for i, row in enumerate(rows):
        if "ID" in row and "Kernel Name" in row:
            candidates = []
            time_idx = row.index("gpu__time_duration.sum") if "gpu__time_duration.sum" in row else None
            for value_row in rows[i + 1:]:
                if len(value_row) != len(row):
                    continue
                if not value_row or not value_row[0].isdigit():
                    continue
                if time_idx is None:
                    candidates.append((0.0, value_row))
                    continue
                try:
                    duration = float(value_row[time_idx])
                except ValueError:
                    continue
                candidates.append((duration, value_row))
            if not candidates:
                raise RuntimeError(f"raw CSV has header but no numeric kernel rows: {path}")
            _, best_row = max(candidates, key=lambda item: item[0])
            return dict(zip(row, best_row))
    raise RuntimeError(f"could not find NCU raw table in {path}")


def val(row, key):
    text = row.get(key, "")
    if text == "":
        return ""
    try:
        return float(text)
    except ValueError:
        return text


def run_ncu(mode, metrics):
    raw_path = NCU_DIR / f"{mode.replace('-', '_')}_ncu_raw.csv"
    report_base = NCU_DIR / f"{mode.replace('-', '_')}_validation"
    cmd = [
        "ncu",
        "--target-processes", "all",
        "--replay-mode", "kernel",
        "--page", "raw",
        "--csv",
        "--force-overwrite",
        "-o", str(report_base),
        "--metrics", ",".join(metrics),
        str(BIN),
        "--mode", mode,
        "--cluster-size", "2",
        "--iters", "4096",
        "--warmup-iters", "64",
        "--threads", "256",
        "--shared-bytes", "32768",
        "--csv",
    ]
    with raw_path.open("w") as f:
        proc = subprocess.run(cmd, text=True, stdout=f, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"NCU failed for {mode}: {proc.stderr}")
    return raw_path


def make_summary_row(mode, app, ncu_row, missing):
    expected = float(app["requested_bytes"])
    dshared_bytes = val(ncu_row, "l1tex__t_bytes_pipe_lsu_mem_dshared.sum")
    dshared_bpc = val(ncu_row, "l1tex__t_bytes_pipe_lsu_mem_dshared.sum.per_cycle_elapsed")
    dshared_pct = val(ncu_row, "l1tex__t_bytes_pipe_lsu_mem_dshared.sum.pct_of_peak_sustained_elapsed")
    dshared_ld = val(ncu_row, "l1tex__t_bytes_pipe_lsu_mem_dshared_op_ld.sum")
    dshared_st = val(ncu_row, "l1tex__t_bytes_pipe_lsu_mem_dshared_op_st.sum")
    shared_wave_pct = val(ncu_row, "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed")
    lgds_wave_pct = val(ncu_row, "l1tex__data_pipe_lsu_wavefronts_mem_lgds.sum.pct_of_peak_sustained_elapsed")
    peak_est = ""
    if isinstance(dshared_bpc, float) and isinstance(dshared_pct, float) and dshared_pct > 0:
        peak_est = dshared_bpc / (dshared_pct / 100.0)
    ratio = ""
    if isinstance(dshared_bytes, float):
        ratio = dshared_bytes / expected
    return {
        "mode": mode,
        "expected_requested_bytes": f"{expected:.0f}",
        "app_bytes_per_cycle": app["bytes_per_cycle"],
        "app_elapsed_cycles": app["elapsed_cycles"],
        "dshared_bytes": dshared_bytes,
        "dshared_bytes_to_expected": ratio,
        "dshared_bytes_per_cycle_elapsed": dshared_bpc,
        "dshared_pct_peak_elapsed": dshared_pct,
        "dshared_estimated_peak_bytes_per_cycle": peak_est,
        "dshared_ld_bytes": dshared_ld,
        "dshared_st_bytes": dshared_st,
        "shared_wavefront_pct_peak_elapsed": shared_wave_pct,
        "lgds_wavefront_pct_peak_elapsed": lgds_wave_pct,
        "shared_bank_conflicts": val(ncu_row, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"),
        "gds_bank_conflicts": val(ncu_row, "l1tex__data_bank_conflicts_pipe_lsu_mem_gds.sum"),
        "missing_metric_count": len(missing),
        "missing_metrics": ";".join(missing),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows):
    lines = [
        "# NCU DSMEM validation report",
        "",
        "|mode|app B/cycle|dshared B/cycle elapsed|dshared %peak elapsed|dshared/expected|estimated dshared peak B/cycle|shared wavefront %peak|LGDS wavefront %peak|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(x):
            if x == "":
                return ""
            if isinstance(x, float):
                return f"{x:.3f}"
            try:
                return f"{float(x):.3f}"
            except Exception:
                return str(x)
        lines.append(
            f"|{row['mode']}|{fmt(row['app_bytes_per_cycle'])}|"
            f"{fmt(row['dshared_bytes_per_cycle_elapsed'])}|"
            f"{fmt(row['dshared_pct_peak_elapsed'])}|"
            f"{fmt(row['dshared_bytes_to_expected'])}|"
            f"{fmt(row['dshared_estimated_peak_bytes_per_cycle'])}|"
            f"{fmt(row['shared_wavefront_pct_peak_elapsed'])}|"
            f"{fmt(row['lgds_wavefront_pct_peak_elapsed'])}|"
        )
    lines.extend([
        "",
        "Direct byte counters exist for `mem_dshared`; local shared memory is checked with LSU wavefront and bank-conflict counters on this machine.",
    ])
    (NCU_DIR / "ncu_dsmem_report.md").write_text("\n".join(lines) + "\n")


def update_adversarial_review(rows):
    remote = [r for r in rows if r["mode"].startswith("remote")]
    ncu_ok = all(isinstance(r["dshared_bytes_to_expected"], float) and 0.75 <= r["dshared_bytes_to_expected"] <= 1.25 for r in remote)
    lines = [
        "# DSMEM 运行对抗式审查",
        "",
        "## NCU 后审查",
        "",
        "|mode|app B/cycle|dshared/expected|dshared %peak|estimated peak B/cycle|",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        def fmt(x):
            if x == "":
                return ""
            try:
                return f"{float(x):.3f}"
            except Exception:
                return str(x)
        lines.append(
            f"|{r['mode']}|{fmt(r['app_bytes_per_cycle'])}|{fmt(r['dshared_bytes_to_expected'])}|"
            f"{fmt(r['dshared_pct_peak_elapsed'])}|{fmt(r['dshared_estimated_peak_bytes_per_cycle'])}|"
        )
    lines.extend([
        "",
        "## 结论",
        "",
    ])
    if ncu_ok:
        lines.append("通过：remote-read/remote-write 的 NCU dshared bytes 与预期请求字节在 25% 容差内，说明 remote 模式确实产生 DSMEM traffic。")
    else:
        lines.append("未通过：remote 模式的 dshared bytes 与预期请求字节不匹配，需要检查 SASS、访问宽度或 NCU metric 解释后重跑。")
    lines.append("")
    lines.append("local shared 没有本机 direct byte counter，因此 local 结果只作为 app clock + SASS + wavefront proxy 支撑。")
    (RESULTS / "adversarial_review.md").write_text("\n".join(lines) + "\n")


def main():
    NCU_DIR.mkdir(parents=True, exist_ok=True)
    supported, missing = query_supported_metrics()
    if not supported:
        raise SystemExit("No supported NCU metrics from candidate list")

    rows = []
    for mode in MODES:
        app = app_probe(mode)
        raw_path = run_ncu(mode, supported)
        ncu_row = find_raw_table(raw_path)
        rows.append(make_summary_row(mode, app, ncu_row, missing))

    write_csv(NCU_DIR / "ncu_dsmem_summary.csv", rows)
    write_report(rows)
    update_adversarial_review(rows)
    print(f"Wrote {NCU_DIR / 'ncu_dsmem_summary.csv'}")


if __name__ == "__main__":
    main()
