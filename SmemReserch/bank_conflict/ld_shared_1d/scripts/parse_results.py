#!/usr/bin/env python3
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
CSV_PATH = RESULTS_DIR / "basic_results.csv"
CASE_ORDER = ["v0", "v1a", "v1b", "v1c", "v1d", "v1e", "v2", "v3", "v4a", "v4b"]


def load_ordered_rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        rows_by_case = {row["case"]: row for row in csv.DictReader(stream)}
    return [rows_by_case[name] for name in CASE_ORDER if name in rows_by_case]


def print_table(rows):
    columns = ["case", "stride", "avg_ms", "min_ms", "effective_GBps"]
    print("\nAll cases")
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))


def plot_vertical_bars(plt, labels, values, *, ylabel, title, output, color):
    plt.figure(figsize=(max(7, len(labels) * 0.9), 4.8))
    bars = plt.bar(labels, values, color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    max_value = max(values) if values else 0.0
    offset = max_value * 0.015 if max_value > 0.0 else 0.01
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.ylim(top=max_value + offset * 4 if max_value > 0.0 else 1.0)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Wrote {output}")


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH}; run scripts/run_basic.sh first.")

    rows = load_ordered_rows()
    if not rows:
        raise SystemExit(f"{CSV_PATH} contains no recognized cases.")
    print_table(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is unavailable; skipping PNG charts")
        return

    labels = [row["case"] for row in rows]
    plot_vertical_bars(
        plt,
        labels,
        [float(row["avg_ms"]) for row in rows],
        ylabel="Kernel time (ms)",
        title="Load microbenchmark timing trend",
        output=RESULTS_DIR / "all_cases_avg_ms_bar.png",
        color="#F58518",
    )
    plot_vertical_bars(
        plt,
        labels,
        [float(row["effective_GBps"]) for row in rows],
        ylabel="Effective GB/s",
        title="Load microbenchmark effective bandwidth trend",
        output=RESULTS_DIR / "all_cases_effective_gbps_bar.png",
        color="#54A24B",
    )


if __name__ == "__main__":
    main()
