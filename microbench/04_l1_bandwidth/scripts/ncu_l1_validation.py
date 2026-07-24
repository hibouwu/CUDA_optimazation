#!/usr/bin/env python3
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "l1_bandwidth"
RESULTS = ROOT / "results"
NCU_DIR = RESULTS / "ncu"
MODES = ["read-ca", "read-cg", "write-wb", "write-cg"]

CANDIDATE_METRICS = [
    "gpu__time_duration.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_cycle_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.pct_of_peak_sustained_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_cycle_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.pct_of_peak_sustained_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st_lookup_hit.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st_lookup_miss.sum",
    "l1tex__lsuin_requests.sum.pct_of_peak_sustained_elapsed",
    "l1tex__lsu_writeback_active.sum.pct_of_peak_sustained_elapsed",
    "lts__t_bytes.sum",
    "lts__t_bytes.sum.per_cycle_elapsed",
    "lts__t_bytes.sum.pct_of_peak_sustained_elapsed",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
]


def run(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def query_supported():
    proc = run(["ncu", "--query-metrics", "--query-metrics-mode", "all"])
    names = {line.split()[0] for line in proc.stdout.splitlines() if line and not line[0].isspace()}
    return [m for m in CANDIDATE_METRICS if m in names], [m for m in CANDIDATE_METRICS if m not in names]


def app_probe(mode):
    out = run([
        str(BIN),
        "--mode", mode,
        "--bytes-per-cta", "16384",
        "--iters", "4096",
        "--warmup-rounds", "2",
        "--threads", "256",
        "--csv",
    ]).stdout.strip()
    header = run([str(BIN), "--csv-header"]).stdout.strip().split(",")
    return next(csv.DictReader([",".join(header), out]))


def find_kernel_row(path):
    rows = list(csv.reader(path.open(newline="")))
    for i, row in enumerate(rows):
        if "ID" in row and "Kernel Name" in row:
            time_idx = row.index("gpu__time_duration.sum") if "gpu__time_duration.sum" in row else None
            candidates = []
            for value_row in rows[i + 1:]:
                if len(value_row) != len(row) or not value_row or not value_row[0].isdigit():
                    continue
                try:
                    duration = float(value_row[time_idx]) if time_idx is not None else 0.0
                except ValueError:
                    continue
                candidates.append((duration, value_row))
            if not candidates:
                raise RuntimeError(f"no numeric NCU rows in {path}")
            return dict(zip(row, max(candidates, key=lambda item: item[0])[1]))
    raise RuntimeError(f"no NCU table in {path}")


def val(row, key):
    text = row.get(key, "")
    try:
        return float(text)
    except (TypeError, ValueError):
        return ""


def run_ncu(mode, metrics):
    raw = NCU_DIR / f"{mode.replace('-', '_')}_ncu_raw.csv"
    report = NCU_DIR / f"{mode.replace('-', '_')}_validation"
    cmd = [
        "ncu",
        "--target-processes", "all",
        "--replay-mode", "kernel",
        "--page", "raw",
        "--csv",
        "--force-overwrite",
        "-o", str(report),
        "--metrics", ",".join(metrics),
        str(BIN),
        "--mode", mode,
        "--bytes-per-cta", "16384",
        "--iters", "4096",
        "--warmup-rounds", "2",
        "--threads", "256",
        "--csv",
    ]
    with raw.open("w") as f:
        proc = subprocess.run(cmd, text=True, stdout=f, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return raw


def make_row(mode, app, ncu, missing):
    expected = float(app["requested_bytes"])
    ld = val(ncu, "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum")
    ldh = val(ncu, "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_hit.sum")
    ldm = val(ncu, "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_miss.sum")
    st = val(ncu, "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum")
    sth = val(ncu, "l1tex__t_bytes_pipe_lsu_mem_global_op_st_lookup_hit.sum")
    stm = val(ncu, "l1tex__t_bytes_pipe_lsu_mem_global_op_st_lookup_miss.sum")
    lts = val(ncu, "lts__t_bytes.sum")
    ld_pct = val(ncu, "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.pct_of_peak_sustained_elapsed")
    st_pct = val(ncu, "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.pct_of_peak_sustained_elapsed")
    ld_bpc = val(ncu, "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_cycle_elapsed")
    st_bpc = val(ncu, "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_cycle_elapsed")
    peak = ""
    pct = ld_pct if mode.startswith("read") else st_pct
    bpc = ld_bpc if mode.startswith("read") else st_bpc
    if isinstance(pct, float) and pct > 0 and isinstance(bpc, float):
        peak = bpc / (pct / 100.0)
    return {
        "mode": mode,
        "expected_requested_bytes": f"{expected:.0f}",
        "app_bytes_per_cycle": app["bytes_per_cycle"],
        "l1tex_ld_bytes": ld,
        "l1tex_ld_hit_bytes": ldh,
        "l1tex_ld_miss_bytes": ldm,
        "l1tex_st_bytes": st,
        "l1tex_st_hit_bytes": sth,
        "l1tex_st_miss_bytes": stm,
        "lts_bytes": lts,
        "l1tex_ld_to_expected": (ld / expected) if isinstance(ld, float) else "",
        "l1tex_ld_hit_to_ld": (ldh / ld) if isinstance(ldh, float) and isinstance(ld, float) and ld else "",
        "l1tex_st_to_expected": (st / expected) if isinstance(st, float) else "",
        "l1tex_st_hit_to_st": (sth / st) if isinstance(sth, float) and isinstance(st, float) and st else "",
        "lts_to_expected": (lts / expected) if isinstance(lts, float) else "",
        "l1tex_op_pct_peak_elapsed": pct,
        "l1tex_op_bytes_per_cycle_elapsed": bpc,
        "estimated_l1tex_peak_bytes_per_cycle": peak,
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
        "# NCU L1 validation report",
        "",
        "|mode|app B/cycle|L1TEX op/expected|hit/op|LTS/expected|L1TEX %peak|L1TEX B/cycle|estimated peak B/cycle|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        op_ratio = r["l1tex_ld_to_expected"] if r["mode"].startswith("read") else r["l1tex_st_to_expected"]
        hit_ratio = r["l1tex_ld_hit_to_ld"] if r["mode"].startswith("read") else r["l1tex_st_hit_to_st"]
        lines.append(
            f"|{r['mode']}|{fmt(r['app_bytes_per_cycle'])}|{fmt(op_ratio)}|{fmt(hit_ratio)}|"
            f"{fmt(r['lts_to_expected'])}|{fmt(r['l1tex_op_pct_peak_elapsed'])}|"
            f"{fmt(r['l1tex_op_bytes_per_cycle_elapsed'])}|{fmt(r['estimated_l1tex_peak_bytes_per_cycle'])}|"
        )
    (NCU_DIR / "ncu_l1_report.md").write_text("\n".join(lines) + "\n")

    read_ca = next(r for r in rows if r["mode"] == "read-ca")
    read_ok = (
        isinstance(read_ca["l1tex_ld_to_expected"], float) and read_ca["l1tex_ld_to_expected"] > 0.95 and
        isinstance(read_ca["l1tex_ld_hit_to_ld"], float) and read_ca["l1tex_ld_hit_to_ld"] > 0.90 and
        isinstance(read_ca["lts_to_expected"], float) and read_ca["lts_to_expected"] < 0.20
    )
    status = "通过" if read_ok else "未通过"
    reason = (
        "`read-ca` 的 L1TEX load bytes 接近期望、lookup-hit 占比高，且 LTS 流量远小于逻辑请求。"
        if read_ok else
        "`read-ca` 的 L1 hit 或 LTS 流量条件没有满足，需要修复实验。"
    )
    (RESULTS / "adversarial_review.md").write_text(
        "# L1 运行对抗式审查\n\n"
        f"## 结论\n\n{status}：{reason}\n\n"
        "写路径只作为 L1TEX store path 报告，不解释为纯 L1 cache write-port 峰值。\n"
    )


def main():
    NCU_DIR.mkdir(parents=True, exist_ok=True)
    metrics, missing = query_supported()
    rows = []
    for mode in MODES:
        app = app_probe(mode)
        raw = run_ncu(mode, metrics)
        rows.append(make_row(mode, app, find_kernel_row(raw), missing))
    write_csv(NCU_DIR / "ncu_l1_summary.csv", rows)
    write_reports(rows)
    print(f"Wrote {NCU_DIR / 'ncu_l1_summary.csv'}")


if __name__ == "__main__":
    main()
