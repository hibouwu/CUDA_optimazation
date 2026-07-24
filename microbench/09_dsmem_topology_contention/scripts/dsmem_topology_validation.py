#!/usr/bin/env python3
import csv
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "dsmem_topology_contention"
RESULTS = ROOT / "results"
MODES = [
    "ring-read-d1", "ring-read-d2", "fanin-read-root0",
    "ring-write-d1", "ring-write-d2", "fanin-write-root0",
]


def run(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def header():
    return run([str(BIN), "--csv-header"]).split(",")


def run_case(mode):
    out = run([
        str(BIN), "--mode", mode, "--cluster-size", "4",
        "--iters", "4096", "--warmup-iters", "64",
        "--threads", "256", "--shared-bytes", "65536", "--csv",
    ])
    return next(csv.DictReader([",".join(header()), out]))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in MODES:
        for rep in range(3):
            row = run_case(mode)
            row["repeat"] = rep
            rows.append(row)
    write_csv(RESULTS / "dsmem_topology_validation.csv", rows)
    summary = []
    for mode in MODES:
        vals = [float(r["bytes_per_cycle"]) for r in rows if r["mode"] == mode]
        summary.append({
            "mode": mode,
            "bytes_per_cycle_median": statistics.median(vals),
            "bytes_per_cycle_min": min(vals),
            "bytes_per_cycle_max": max(vals),
            "repeat_count": len(vals),
        })
    write_csv(RESULTS / "dsmem_topology_validation_summary.csv", summary)
    lines = [
        "# DSMEM topology validation report",
        "",
        "|mode|median B/cycle|range B/cycle|",
        "|---|---:|---:|",
    ]
    for r in summary:
        lines.append(f"|{r['mode']}|{float(r['bytes_per_cycle_median']):.3f}|"
                     f"{float(r['bytes_per_cycle_min']):.3f}-{float(r['bytes_per_cycle_max']):.3f}|")
    (RESULTS / "validation_report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS / 'dsmem_topology_validation.csv'}")


if __name__ == "__main__":
    main()
