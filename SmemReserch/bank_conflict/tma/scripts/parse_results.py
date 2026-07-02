#!/usr/bin/env python3
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CSV_PATH = ROOT / "results" / "basic_results.csv"
CASE_ORDER = [
    "T0a_gmem_to_smem_no_swizzle_copy",
    "T0b_smem_to_gmem_no_swizzle_copy",
    "T1a_load_no_swizzle_column_consumer",
    "T1b_load_32b_swizzle_matched_consumer",
    "T1c_load_64b_swizzle_matched_consumer",
    "T1d_load_128b_swizzle_matched_consumer",
    "T2a_column_producer_store_no_swizzle",
    "T2b_matched_producer_store_32b_swizzle",
    "T2c_matched_producer_store_64b_swizzle",
    "T2d_matched_producer_store_128b_swizzle",
    "T3a_load_store_no_swizzle",
    "T3b_load_store_32b_swizzle",
    "T3c_load_store_64b_swizzle",
    "T3d_load_store_128b_swizzle",
]
ORDER = {name: index for index, name in enumerate(CASE_ORDER)}


def load_rows():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH}; run scripts/run_basic.sh first.")
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"{CSV_PATH} is empty")
    rows.sort(key=lambda row: ORDER.get(row["case"], len(CASE_ORDER)))
    return rows


def print_table(rows):
    columns = [
        "experiment",
        "case",
        "direction",
        "swizzle",
        "shared_bytes",
        "consumer",
        "producer",
        "avg_ms",
        "effective_GBps",
        "correctness",
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def plot_metric(plt, rows, field, ylabel, output, color):
    labels = [row["case"] for row in rows]
    values = [float(row[field]) for row in rows]
    plt.figure(figsize=(max(12, len(labels) * 0.9), 5.5))
    bars = plt.bar(labels, values, color=color)
    plt.ylabel(ylabel)
    plt.xticks(rotation=32, ha="right")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Wrote {output}")


def main():
    rows = load_rows()
    print_table(rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping PNG charts")
        return
    plot_metric(
        plt,
        rows,
        "avg_ms",
        "Kernel time (ms)",
        ROOT / "results" / "avg_ms.png",
        "#F58518",
    )
    plot_metric(
        plt,
        rows,
        "effective_GBps",
        "TMA effective bandwidth (GB/s)",
        ROOT / "results" / "effective_gbps.png",
        "#4C78A8",
    )


if __name__ == "__main__":
    main()
