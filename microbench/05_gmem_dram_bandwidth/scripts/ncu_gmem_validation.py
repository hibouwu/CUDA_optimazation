#!/usr/bin/env python3
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "gmem_dram_bandwidth"
RESULTS = ROOT / "results"
NCU_DIR = RESULTS / "ncu"
MODES = ["read-stream", "write-stream", "copy-stream"]

CANDIDATE_METRICS = [
    "gpu__time_duration.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    "lts__t_bytes.sum",
    "lts__t_bytes.sum.per_cycle_elapsed",
    "lts__t_bytes.sum.pct_of_peak_sustained_elapsed",
    "lts__t_request_hit_rate.pct",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_read_lookup_hit.sum",
    "lts__t_sectors_op_read_lookup_miss.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_op_write_lookup_hit.sum",
    "lts__t_sectors_op_write_lookup_miss.sum",
    "dram__bytes.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
]


def run(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def query_supported():
    proc = run(["ncu", "--query-metrics", "--query-metrics-mode", "all"])
    names = {line.split()[0] for line in proc.stdout.splitlines() if line and not line[0].isspace()}
    return [m for m in CANDIDATE_METRICS if m in names], [m for m in CANDIDATE_METRICS if m not in names]


def app_probe(mode):
    out = run([
        str(BIN), "--mode", mode, "--bytes", str(256 << 20),
        "--iters", "4096", "--warmup-iters", "32",
        "--blocks-per-sm", "4", "--threads", "256", "--csv",
    ]).stdout.strip()
    header = run([str(BIN), "--csv-header"]).stdout.strip().split(",")
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


def run_ncu(mode, metrics):
    raw = NCU_DIR / f"{mode.replace('-', '_')}_ncu_raw.csv"
    report = NCU_DIR / f"{mode.replace('-', '_')}_validation"
    cmd = [
        "ncu", "--target-processes", "all", "--replay-mode", "kernel",
        "--page", "raw", "--csv", "--force-overwrite", "-o", str(report),
        "--metrics", ",".join(metrics),
        str(BIN), "--mode", mode, "--bytes", str(256 << 20),
        "--iters", "4096", "--warmup-iters", "32",
        "--blocks-per-sm", "4", "--threads", "256", "--csv",
    ]
    with raw.open("w") as f:
        proc = subprocess.run(cmd, text=True, stdout=f, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return raw


def make_row(mode, app, ncu, missing):
    expected = float(app["requested_bytes"])
    lts = val(ncu, "lts__t_bytes.sum")
    lts_bpc = val(ncu, "lts__t_bytes.sum.per_cycle_elapsed")
    lts_pct = val(ncu, "lts__t_bytes.sum.pct_of_peak_sustained_elapsed")
    rmiss = val(ncu, "lts__t_sectors_op_read_lookup_miss.sum")
    wmiss = val(ncu, "lts__t_sectors_op_write_lookup_miss.sum")
    dram = val(ncu, "dram__bytes.sum")
    proxy = ""
    if isinstance(rmiss, float) or isinstance(wmiss, float):
        proxy = (rmiss if isinstance(rmiss, float) else 0.0) * 32.0 + (wmiss if isinstance(wmiss, float) else 0.0) * 32.0
    peak = ""
    if isinstance(lts_bpc, float) and isinstance(lts_pct, float) and lts_pct > 0:
        peak = lts_bpc / (lts_pct / 100.0)
    return {
        "mode": mode,
        "expected_requested_bytes": f"{expected:.0f}",
        "app_bytes_per_cycle": app["bytes_per_cycle"],
        "lts_bytes": lts,
        "lts_to_expected": (lts / expected) if isinstance(lts, float) else "",
        "lts_bytes_per_cycle_elapsed": lts_bpc,
        "lts_pct_peak_elapsed": lts_pct,
        "lts_estimated_peak_bytes_per_cycle": peak,
        "lts_read_miss_sector_bytes": (rmiss * 32.0) if isinstance(rmiss, float) else "",
        "lts_write_miss_sector_bytes": (wmiss * 32.0) if isinstance(wmiss, float) else "",
        "dram_proxy_miss_sector_bytes": proxy,
        "dram_proxy_to_expected": (proxy / expected) if isinstance(proxy, float) else "",
        "dram_bytes": dram,
        "dram_to_expected": (dram / expected) if isinstance(dram, float) else "",
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
        "# NCU GMEM/DRAM validation report",
        "",
        "|mode|app B/cycle|LTS/expected|DRAM proxy/expected|LTS %peak|LTS B/cycle|estimated LTS peak B/cycle|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"|{r['mode']}|{fmt(r['app_bytes_per_cycle'])}|{fmt(r['lts_to_expected'])}|{fmt(r['dram_proxy_to_expected'])}|{fmt(r['lts_pct_peak_elapsed'])}|{fmt(r['lts_bytes_per_cycle_elapsed'])}|{fmt(r['lts_estimated_peak_bytes_per_cycle'])}|")
    lines.append("")
    lines.append("When `dram__bytes*` is missing, DRAM proxy is `(LTS read miss sectors + LTS write miss sectors) * 32 B`.")
    (NCU_DIR / "ncu_gmem_report.md").write_text("\n".join(lines) + "\n")

    ok = True
    for r in rows:
        if r["mode"] == "read-stream":
            ok &= isinstance(r["dram_proxy_to_expected"], float) and r["dram_proxy_to_expected"] > 0.70
        if r["mode"] == "write-stream":
            ok &= isinstance(r["dram_proxy_to_expected"], float) and r["dram_proxy_to_expected"] > 0.50
    status = "通过" if ok else "未通过"
    reason = "LTS miss-sector proxy 显示大部分流量超过 L2，符合 GMEM/DRAM streaming 目标。" if ok else "LTS miss-sector proxy 不足，可能仍是 L2-resident 或 metric 不适用。"
    (RESULTS / "adversarial_review.md").write_text(
        "# GMEM/DRAM 运行对抗式审查\n\n"
        f"{status}：{reason}\n\n"
        "若 `dram__bytes*` 缺失，本实验不声称有直接 DRAM byte counter，只报告 LTS miss-sector proxy 和 app throughput。\n"
    )


def main():
    NCU_DIR.mkdir(parents=True, exist_ok=True)
    metrics, missing = query_supported()
    rows = []
    for mode in MODES:
        app = app_probe(mode)
        raw = run_ncu(mode, metrics)
        rows.append(make_row(mode, app, find_row(raw), missing))
    write_csv(NCU_DIR / "ncu_gmem_summary.csv", rows)
    write_reports(rows)
    print(f"Wrote {NCU_DIR / 'ncu_gmem_summary.csv'}")


if __name__ == "__main__":
    main()
