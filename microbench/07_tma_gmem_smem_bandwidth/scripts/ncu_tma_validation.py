#!/usr/bin/env python3
import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "tma_gmem_smem_bandwidth"
RESULTS = ROOT / "results"
NCU_DIR = RESULTS / "ncu"
MODES = [("l2-hit", 16 << 20), ("dram-stream", 256 << 20)]

CANDIDATE_METRICS = [
    "gpu__time_duration.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum.per_cycle_elapsed",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum.pct_of_peak_sustained_elapsed",
    "lts__t_bytes.sum",
    "lts__t_bytes.sum.per_cycle_elapsed",
    "lts__t_bytes.sum.pct_of_peak_sustained_elapsed",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_read_lookup_hit.sum",
    "lts__t_sectors_op_read_lookup_miss.sum",
    "dram__bytes.sum",
    "dram__bytes_read.sum",
    "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tma.sum",
    "sm__inst_executed_pipe_tma.sum.per_cycle_active",
]


def run(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def query_supported():
    proc = run(["ncu", "--query-metrics", "--query-metrics-mode", "all"])
    names = {line.split()[0] for line in proc.stdout.splitlines() if line and not line[0].isspace()}
    return [m for m in CANDIDATE_METRICS if m in names], [m for m in CANDIDATE_METRICS if m not in names]


def app_probe(mode, bytes_):
    header = run([str(BIN), "--csv-header"]).stdout.strip().split(",")
    out = run([
        str(BIN), "--mode", mode, "--bytes", str(bytes_),
        "--tile-bytes", "32768", "--slots", "4", "--iters", "4096",
        "--warmup-iters", "32", "--blocks-per-sm", "1", "--threads", "128", "--csv",
    ]).stdout.strip()
    return next(csv.DictReader([",".join(header), out]))


def find_row(path):
    rows = list(csv.reader(path.open(newline="")))
    for i, row in enumerate(rows):
        if "ID" in row and "Kernel Name" in row:
            tidx = row.index("gpu__time_duration.sum") if "gpu__time_duration.sum" in row else None
            kidx = row.index("Kernel Name")
            candidates = []
            for value in rows[i + 1:]:
                if len(value) != len(row) or not value or not value[0].isdigit():
                    continue
                if "tma_kernel" not in value[kidx]:
                    continue
                try:
                    t = float(value[tidx]) if tidx is not None else 0.0
                except ValueError:
                    continue
                candidates.append((t, value))
            if not candidates:
                raise RuntimeError(f"no tma_kernel data rows in {path}")
            return dict(zip(row, max(candidates, key=lambda x: x[0])[1]))
    raise RuntimeError(f"no NCU table in {path}")


def val(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return ""


def run_ncu(mode, bytes_, metrics):
    raw = NCU_DIR / f"{mode.replace('-', '_')}_ncu_raw.csv"
    report = NCU_DIR / f"{mode.replace('-', '_')}_validation"
    cmd = [
        "ncu", "--target-processes", "all", "--replay-mode", "kernel",
        "--page", "raw", "--csv", "--force-overwrite", "-o", str(report),
        "--metrics", ",".join(metrics),
        str(BIN), "--mode", mode, "--bytes", str(bytes_),
        "--tile-bytes", "32768", "--slots", "4", "--iters", "4096",
        "--warmup-iters", "32", "--blocks-per-sm", "1", "--threads", "128", "--csv",
    ]
    with raw.open("w") as f:
        proc = subprocess.run(cmd, text=True, stdout=f, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return raw


def make_row(mode, app, ncu, missing):
    expected = float(app["requested_bytes"])
    tma_bytes = val(ncu, "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum")
    tma_bpc = val(ncu, "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum.per_cycle_elapsed")
    tma_pct = val(ncu, "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum.pct_of_peak_sustained_elapsed")
    lts = val(ncu, "lts__t_bytes.sum")
    lts_bpc = val(ncu, "lts__t_bytes.sum.per_cycle_elapsed")
    lts_pct = val(ncu, "lts__t_bytes.sum.pct_of_peak_sustained_elapsed")
    rmiss = val(ncu, "lts__t_sectors_op_read_lookup_miss.sum")
    dram = val(ncu, "dram__bytes.sum")
    proxy = (rmiss * 32.0) if isinstance(rmiss, float) else ""
    peak = ""
    if isinstance(lts_bpc, float) and isinstance(lts_pct, float) and lts_pct > 0:
        peak = lts_bpc / (lts_pct / 100.0)
    tma_peak = ""
    if isinstance(tma_bpc, float) and isinstance(tma_pct, float) and tma_pct > 0:
        tma_peak = tma_bpc / (tma_pct / 100.0)
    return {
        "mode": mode,
        "expected_requested_bytes": f"{expected:.0f}",
        "app_bytes_per_cycle": app["bytes_per_cycle"],
        "tma_bytes": tma_bytes,
        "tma_to_expected": (tma_bytes / expected) if isinstance(tma_bytes, float) else "",
        "tma_bytes_per_cycle_elapsed": tma_bpc,
        "tma_pct_peak_elapsed": tma_pct,
        "tma_estimated_peak_bytes_per_cycle": tma_peak,
        "lts_bytes": lts,
        "lts_to_expected": (lts / expected) if isinstance(lts, float) else "",
        "lts_bytes_per_cycle_elapsed": lts_bpc,
        "lts_pct_peak_elapsed": lts_pct,
        "lts_estimated_peak_bytes_per_cycle": peak,
        "lts_read_miss_sector_bytes": proxy,
        "dram_proxy_to_expected": (proxy / expected) if isinstance(proxy, float) else "",
        "dram_bytes": dram,
        "dram_to_expected": (dram / expected) if isinstance(dram, float) else "",
        "tma_pipe_pct_peak": val(ncu, "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active"),
        "tma_inst": val(ncu, "sm__inst_executed_pipe_tma.sum"),
        "tma_inst_per_cycle": val(ncu, "sm__inst_executed_pipe_tma.sum.per_cycle_active"),
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
        "# NCU TMA GMEM-to-SMEM validation report",
        "",
        "|mode|app B/cycle|TMA bytes/expected|TMA %peak|TMA B/cycle|estimated TMA peak B/cycle|LTS/expected|DRAM proxy/expected|LTS %peak|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"|{r['mode']}|{fmt(r['app_bytes_per_cycle'])}|{fmt(r['tma_to_expected'])}|"
                     f"{fmt(r['tma_pct_peak_elapsed'])}|{fmt(r['tma_bytes_per_cycle_elapsed'])}|"
                     f"{fmt(r['tma_estimated_peak_bytes_per_cycle'])}|{fmt(r['lts_to_expected'])}|"
                     f"{fmt(r['dram_proxy_to_expected'])}|{fmt(r['lts_pct_peak_elapsed'])}|")
    lines.append("")
    lines.append("When `dram__bytes*` is missing, DRAM proxy is LTS read miss-sector bytes.")
    (NCU_DIR / "ncu_tma_report.md").write_text("\n".join(lines) + "\n")

    ok = True
    for r in rows:
        ok &= isinstance(r["tma_to_expected"], float) and r["tma_to_expected"] > 0.98
        ok &= isinstance(r["lts_to_expected"], float) and r["lts_to_expected"] > 0.90
        if r["mode"] == "dram-stream":
            ok &= isinstance(r["dram_proxy_to_expected"], float) and r["dram_proxy_to_expected"] > 0.70
    status = "通过" if ok else "未通过"
    reason = "LTS bytes match requested TMA payload and DRAM mode misses L2 as expected." if ok else "NCU traffic does not yet prove the intended TMA/LTS/DRAM path."
    (RESULTS / "adversarial_review.md").write_text(
        "# TMA GMEM-to-SMEM 运行对抗式审查\n\n"
        f"{status}：{reason}\n\n"
        "SASS 必须包含 `UTMALDG.3D`，NCU 必须显示 TMA read bytes 与请求字节匹配。\n\n"
        "保留边界：这是 TMA ingress end-to-end throughput，不是纯 DRAM pin peak 或纯 shared write-port peak。\n"
    )


def main():
    NCU_DIR.mkdir(parents=True, exist_ok=True)
    metrics, missing = query_supported()
    rows = []
    for mode, bytes_ in MODES:
        app = app_probe(mode, bytes_)
        raw = run_ncu(mode, bytes_, metrics)
        rows.append(make_row(mode, app, find_row(raw), missing))
    write_csv(NCU_DIR / "ncu_tma_summary.csv", rows)
    write_reports(rows)
    print(f"Wrote {NCU_DIR / 'ncu_tma_summary.csv'}")


if __name__ == "__main__":
    main()
