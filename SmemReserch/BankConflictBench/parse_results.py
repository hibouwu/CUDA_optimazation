#!/usr/bin/env python3
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
CSV_PATH = RESULTS_DIR / "basic_results.csv"


def load_rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def print_table(title, rows, columns):
    print(f"\n{title}")
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH}; run ./run_basic.sh first.")

    rows = load_rows()
    print_table(
        "All cases",
        rows,
        ["case", "stride", "avg_ms", "min_ms", "effective_GBps"],
    )

    stride_rows = [
        row
        for row in rows
        if row["case"] == "stride_conflict_sweep" and int(row["stride"]) > 0
    ]
    # run_basic includes one stride=1 row from --case all; retain one row/stride.
    by_stride = {}
    for row in stride_rows:
        by_stride[int(row["stride"])] = row
    sweep = [by_stride[stride] for stride in sorted(by_stride)]
    print_table("Stride sweep", sweep, ["stride", "avg_ms", "min_ms"])

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is unavailable; skipping stride_sweep.png")
        return

    plt.figure(figsize=(7, 4))
    strides = [int(row["stride"]) for row in sweep]
    plt.plot(strides, [float(row["avg_ms"]) for row in sweep], "o-", label="avg_ms")
    plt.plot(strides, [float(row["min_ms"]) for row in sweep], "s-", label="min_ms")
    plt.xscale("log", base=2)
    plt.xticks(strides, strides)
    plt.xlabel("Stride (32-bit words)")
    plt.ylabel("Kernel time (ms)")
    plt.title("Shared-memory bank-conflict stride sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output = RESULTS_DIR / "stride_sweep.png"
    plt.savefig(output, dpi=150)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()

