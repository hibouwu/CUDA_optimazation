#!/usr/bin/env python3
import csv
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "tma_gmem_smem_bandwidth"
RESULTS = ROOT / "results"


def run(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def header():
    return run([str(BIN), "--csv-header"]).split(",")


def run_case(mode, bytes_, tile_bytes=32768, slots=4, iters=4096):
    out = run([
        str(BIN), "--mode", mode, "--bytes", str(bytes_),
        "--tile-bytes", str(tile_bytes), "--slots", str(slots),
        "--iters", str(iters), "--warmup-iters", "32",
        "--blocks-per-sm", "1", "--threads", "128", "--csv",
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
    for mode, bytes_ in [("l2-hit", 16 << 20), ("dram-stream", 256 << 20)]:
        for tile_bytes in [8192, 16384, 32768, 65536]:
            slots = 2 if tile_bytes >= 65536 else 4
            for rep in range(3):
                row = run_case(mode, bytes_, tile_bytes=tile_bytes, slots=slots)
                row["repeat"] = rep
                rows.append(row)
    write_csv(RESULTS / "tma_tile_sweep.csv", rows)

    summary = []
    for mode in sorted({r["mode"] for r in rows}):
        for tile in sorted({int(r["tile_bytes"]) for r in rows if r["mode"] == mode}):
            vals = [float(r["bytes_per_cycle"]) for r in rows
                    if r["mode"] == mode and int(r["tile_bytes"]) == tile]
            summary.append({
                "mode": mode,
                "tile_bytes": tile,
                "bytes_per_cycle_median": statistics.median(vals),
                "bytes_per_cycle_min": min(vals),
                "bytes_per_cycle_max": max(vals),
                "repeat_count": len(vals),
            })
    write_csv(RESULTS / "tma_tile_sweep_summary.csv", summary)

    baseline = [r for r in summary if int(r["tile_bytes"]) == 32768]
    lines = [
        "# TMA GMEM-to-SMEM validation report",
        "",
        "## 32 KiB tile baseline",
        "",
        "|mode|median B/cycle|range B/cycle|",
        "|---|---:|---:|",
    ]
    for r in baseline:
        lines.append(f"|{r['mode']}|{float(r['bytes_per_cycle_median']):.3f}|"
                     f"{float(r['bytes_per_cycle_min']):.3f}-{float(r['bytes_per_cycle_max']):.3f}|")
    (RESULTS / "validation_report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS / 'tma_tile_sweep.csv'}")


if __name__ == "__main__":
    main()
