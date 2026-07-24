#!/usr/bin/env python3
import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "dsmem_topology_contention"
RESULTS = ROOT / "results"
NCU_DIR = RESULTS / "ncu"
MODES = [
    "ring-read-d1", "ring-read-d2", "fanin-read-root0",
    "ring-write-d1", "ring-write-d2", "fanin-write-root0",
]

CANDIDATE_METRICS = [
    "gpu__time_duration.sum",
    "l1tex__t_bytes_pipe_lsu_mem_dshared.sum",
    "l1tex__t_bytes_pipe_lsu_mem_dshared.sum.per_cycle_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_dshared.sum.pct_of_peak_sustained_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_dshared_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_dshared_op_ld.sum.per_cycle_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_dshared_op_ld.sum.pct_of_peak_sustained_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_dshared_op_st.sum",
    "l1tex__t_bytes_pipe_lsu_mem_dshared_op_st.sum.per_cycle_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_dshared_op_st.sum.pct_of_peak_sustained_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_lgds.sum.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_gds.sum",
]


def run(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def query_supported():
    proc = run(["ncu", "--query-metrics", "--query-metrics-mode", "all"])
    names = {line.split()[0] for line in proc.stdout.splitlines() if line and not line[0].isspace()}
    return [m for m in CANDIDATE_METRICS if m in names], [m for m in CANDIDATE_METRICS if m not in names]


def app_probe(mode):
    header = run([str(BIN), "--csv-header"]).stdout.strip().split(",")
    out = run([
        str(BIN), "--mode", mode, "--cluster-size", "4",
        "--iters", "4096", "--warmup-iters", "64",
        "--threads", "256", "--shared-bytes", "65536", "--csv",
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


def run_ncu(mode, metrics):
    raw = NCU_DIR / f"{mode.replace('-', '_')}_ncu_raw.csv"
    report = NCU_DIR / f"{mode.replace('-', '_')}_validation"
    cmd = [
        "ncu", "--target-processes", "all", "--replay-mode", "kernel",
        "--page", "raw", "--csv", "--force-overwrite", "-o", str(report),
        "--metrics", ",".join(metrics),
        str(BIN), "--mode", mode, "--cluster-size", "4",
        "--iters", "4096", "--warmup-iters", "64",
        "--threads", "256", "--shared-bytes", "65536", "--csv",
    ]
    with raw.open("w") as f:
        proc = subprocess.run(cmd, text=True, stdout=f, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return raw


def make_row(mode, app, ncu, missing):
    expected = float(app["requested_bytes"])
    dshared = val(ncu, "l1tex__t_bytes_pipe_lsu_mem_dshared.sum")
    dshared_bpc = val(ncu, "l1tex__t_bytes_pipe_lsu_mem_dshared.sum.per_cycle_elapsed")
    dshared_pct = val(ncu, "l1tex__t_bytes_pipe_lsu_mem_dshared.sum.pct_of_peak_sustained_elapsed")
    peak = ""
    if isinstance(dshared_bpc, float) and isinstance(dshared_pct, float) and dshared_pct > 0:
        peak = dshared_bpc / (dshared_pct / 100.0)
    return {
        "mode": mode,
        "expected_requested_bytes": f"{expected:.0f}",
        "app_bytes_per_cycle": app["bytes_per_cycle"],
        "dshared_bytes": dshared,
        "dshared_to_expected": (dshared / expected) if isinstance(dshared, float) else "",
        "dshared_bytes_per_cycle_elapsed": dshared_bpc,
        "dshared_pct_peak_elapsed": dshared_pct,
        "dshared_estimated_peak_bytes_per_cycle": peak,
        "dshared_ld_bytes": val(ncu, "l1tex__t_bytes_pipe_lsu_mem_dshared_op_ld.sum"),
        "dshared_st_bytes": val(ncu, "l1tex__t_bytes_pipe_lsu_mem_dshared_op_st.sum"),
        "lgds_wavefront_pct_peak_elapsed": val(ncu, "l1tex__data_pipe_lsu_wavefronts_mem_lgds.sum.pct_of_peak_sustained_elapsed"),
        "gds_bank_conflicts": val(ncu, "l1tex__data_bank_conflicts_pipe_lsu_mem_gds.sum"),
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
        "# NCU DSMEM topology validation report",
        "",
        "|mode|app B/cycle|dshared/expected|dshared B/cycle|dshared %peak|estimated dshared peak B/cycle|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"|{r['mode']}|{fmt(r['app_bytes_per_cycle'])}|{fmt(r['dshared_to_expected'])}|"
                     f"{fmt(r['dshared_bytes_per_cycle_elapsed'])}|{fmt(r['dshared_pct_peak_elapsed'])}|"
                     f"{fmt(r['dshared_estimated_peak_bytes_per_cycle'])}|")
    (NCU_DIR / "ncu_dsmem_topology_report.md").write_text("\n".join(lines) + "\n")

    ok = all(isinstance(r["dshared_to_expected"], float) and 0.75 <= r["dshared_to_expected"] <= 1.25 for r in rows)
    status = "通过" if ok else "未通过"
    reason = "all remote topology modes have NCU dshared bytes matching requested bytes within 25%." if ok else "some topology modes do not have matching NCU dshared traffic."
    (RESULTS / "adversarial_review.md").write_text(
        "# DSMEM topology/contention 运行对抗式审查\n\n"
        f"{status}：{reason}\n\n"
        "SASS 必须包含 mapped remote shared 的 `LD.E.128`/`ST.E.128`，NCU dshared bytes 是主要 traffic 证据。\n"
    )


def main():
    NCU_DIR.mkdir(parents=True, exist_ok=True)
    metrics, missing = query_supported()
    rows = []
    for mode in MODES:
        app = app_probe(mode)
        raw = run_ncu(mode, metrics)
        rows.append(make_row(mode, app, find_row(raw), missing))
    write_csv(NCU_DIR / "ncu_dsmem_topology_summary.csv", rows)
    write_reports(rows)
    print(f"Wrote {NCU_DIR / 'ncu_dsmem_topology_summary.csv'}")


if __name__ == "__main__":
    main()
