#!/usr/bin/env python3
import csv
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "smem_bank_stride_bandwidth"
RESULTS = ROOT / "results"
MODES = ["read", "write"]
STRIDES = [1, 2, 4, 8, 16, 32]


def run(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def header():
    return run([str(BIN), "--csv-header"]).split(",")


def run_case(mode, stride):
    out = run([
        str(BIN), "--mode", mode, "--stride-words", str(stride),
        "--iters", "8192", "--warmup-iters", "128",
        "--threads", "256", "--blocks-per-sm", "1",
        "--shared-words", "8192", "--csv",
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
        for stride in STRIDES:
            for rep in range(3):
                row = run_case(mode, stride)
                row["repeat"] = rep
                rows.append(row)
    write_csv(RESULTS / "smem_bank_validation.csv", rows)
    summary = []
    for mode in MODES:
        for stride in STRIDES:
            vals = [float(r["bytes_per_cycle"]) for r in rows
                    if r["mode"] == mode and int(r["stride_words"]) == stride]
            summary.append({
                "mode": mode,
                "stride_words": stride,
                "bytes_per_cycle_median": statistics.median(vals),
                "bytes_per_cycle_min": min(vals),
                "bytes_per_cycle_max": max(vals),
                "repeat_count": len(vals),
            })
    write_csv(RESULTS / "smem_bank_validation_summary.csv", summary)
    lines = [
        "# SMEM bank/stride validation report",
        "",
        "|mode|stride words|median B/cycle|range B/cycle|",
        "|---|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f"|{r['mode']}|{r['stride_words']}|{float(r['bytes_per_cycle_median']):.3f}|"
                     f"{float(r['bytes_per_cycle_min']):.3f}-{float(r['bytes_per_cycle_max']):.3f}|")
    (RESULTS / "validation_report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS / 'smem_bank_validation.csv'}")


if __name__ == "__main__":
    main()
