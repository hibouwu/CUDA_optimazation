#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "basic_results.csv"
CASE_ORDER = [
    "R0_imad_chain",
    "R1_imad_independent_x4",
    "R2_reuse_hot_x4",
    "R3_bank_dense_x4",
    "R4_bank_sparse_x4",
]


def load_rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        by_case = {row["case"]: row for row in csv.DictReader(stream)}
    return [by_case[name] for name in CASE_ORDER if name in by_case]


def print_table(rows):
    columns = [
        "case",
        "median_cycles_per_op",
        "min_cycles_per_op",
        "registers_per_thread",
        "local_bytes",
        "correctness",
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rows))
        for column in columns
    }
    print()
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def plot(rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping PNG charts")
        return

    labels = [row["case"].replace("_", "\n", 1) for row in rows]
    values = [float(row["median_cycles_per_op"]) for row in rows]
    plt.figure(figsize=(12.6, 5.5))
    bars = plt.bar(labels, values, color="#D95F39")
    plt.ylabel("Raw cycles per PTX operation")
    plt.title("Thor register microbenchmark")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    output = RESULTS_DIR / "cycles_per_op.png"
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Wrote {output}")


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH}; run scripts/run_basic.sh first.")
    rows = load_rows()
    if not rows:
        raise SystemExit(f"{CSV_PATH} contains no recognized cases.")
    print_table(rows)
    plot(rows)


if __name__ == "__main__":
    main()
