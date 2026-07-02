#!/usr/bin/env python3
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CSV_PATH = ROOT / "results" / "basic_results.csv"
CASE_ORDER = [
    "E0_load_pitch32",
    "E1_load_pitch1",
    "E1_load_pitch2",
    "E1_load_pitch4",
    "E1_load_pitch8",
    "E1_load_pitch16",
    "E1_load_pitch31",
    "E1_load_pitch32",
    "E1_load_pitch33",
    "E2_load_broadcast_same_addr",
    "E2_load_multicast_2addr",
    "E2_load_multicast_4addr",
    "E2_load_conflict_same_bank_diff_addr",
    "E3_load_f32_pitch32",
    "E3_load_f32_pitch33",
    "E3_load_f32x2_pitch32",
    "E3_load_f32x2_pitch33",
    "E3_load_f32x4_pitch32",
    "E3_load_f32x4_pitch33",
    "E4_load_xor_swizzle_pitch32",
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


def plot_metric(plt, rows, field, ylabel, output, color):
    labels = [row["case"] for row in rows]
    values = [float(row[field]) for row in rows]
    plt.figure(figsize=(max(10, len(labels) * 0.6), 5.0))
    plt.bar(labels, values, color=color)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Wrote {output}")


def main():
    rows = load_rows()
    columns = [
        "experiment",
        "case",
        "operation",
        "pitch",
        "vector_width",
        "theoretical_unique_banks",
        "theoretical_conflict_degree",
        "avg_ms",
        "min_ms",
        "effective_GBps",
    ]
    print_table("Transpose load cases", rows, columns)

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
        "Effective GB/s",
        ROOT / "results" / "effective_gbps.png",
        "#54A24B",
    )


if __name__ == "__main__":
    main()
