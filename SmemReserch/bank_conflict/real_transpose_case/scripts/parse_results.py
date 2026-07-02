#!/usr/bin/env python3
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CSV_PATH = ROOT / "results" / "basic_results.csv"
CASE_ORDER = [
    "R0_transpose_naive",
    "R0_transpose_coalesced_read",
    "R0_transpose_coalesced_write",
    "R1_transpose_smem_pitch32",
    "R2_transpose_smem_pitch33",
    "R3_transpose_smem_packed_pitch33",
    "R4_transpose_smem_xor_swizzle",
    "R5_transpose_copy_baseline",
]
ORDER_INDEX = {name: index for index, name in enumerate(CASE_ORDER)}


def load_rows():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH}; run scripts/run_basic.sh first.")
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"{CSV_PATH} is empty")
    rows.sort(key=lambda row: ORDER_INDEX.get(row["case"], len(CASE_ORDER)))
    return rows


def print_table(rows):
    columns = [
        "case",
        "width",
        "height",
        "smem_pitch",
        "vector_width",
        "swizzle",
        "avg_ms",
        "min_ms",
        "effective_GBps",
        "correctness",
    ]
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    print("\nReal transpose cases")
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            "  ".join(str(row[column]).ljust(widths[column]) for column in columns)
        )


def plot_metric(plt, rows, field, ylabel, output, color):
    labels = [row["case"] for row in rows]
    values = [float(row[field]) for row in rows]
    figure_width = max(10.0, len(labels) * 1.15)
    plt.figure(figsize=(figure_width, 5.2))
    bars = plt.bar(labels, values, color=color)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
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
    failed = [row["case"] for row in rows if row["correctness"] != "PASS"]
    if failed:
        print(f"\nCorrectness failures: {', '.join(failed)}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is unavailable; skipping PNG charts")
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
        "Effective bandwidth (GB/s)",
        ROOT / "results" / "effective_gbps.png",
        "#54A24B",
    )


if __name__ == "__main__":
    main()
