#!/usr/bin/env python3
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
CSV_PATH = RESULTS_DIR / "basic_results.csv"
CASE_LABELS = {
    "baseline_unique_banks": "baseline",
    "stride_conflict_sweep": "stride",
    "same_bank_32way_2d": "same_bank_32way_2d",
    "broadcast_same_address": "broadcast",
    "multicast_hash": "multicast_hash",
    "vectorized_v4_contiguous": "v4_contiguous",
    "vectorized_v2_multicast_pairs": "v2_multicast_pairs",
    "vectorized_v4_multicast_quads": "v4_multicast_quads",
}
NUMBERED_CASES = [
    ("baseline_unique_banks", None, "v0  baseline"),
    ("stride_conflict_sweep", 2, "v1a  stride = 2"),
    ("stride_conflict_sweep", 4, "v1b  stride = 4"),
    ("stride_conflict_sweep", 8, "v1c  stride = 8"),
    ("stride_conflict_sweep", 16, "v1d  stride = 16"),
    ("stride_conflict_sweep", 32, "v1e  same bank (32-way)"),
    ("same_bank_32way_2d", None, "v1f  same_bank_32way_2d"),
    ("broadcast_same_address", None, "v2a  broadcasts"),
    ("multicast_hash", None, "v2b  multicasts"),
    ("vectorized_v4_contiguous", None, "v3  float4 contiguous"),
    ("vectorized_v2_multicast_pairs", None, "v4a  v2_multicast_pairs"),
    ("vectorized_v4_multicast_quads", None, "v4b  v4_multicast_quads"),
]


def load_rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def canonical_case_label(row):
    base_label = CASE_LABELS.get(row["case"], row["case"])
    if row["case"] == "stride_conflict_sweep":
        return f"{base_label} ({row['stride']})"
    return base_label


def numbered_case_label(row):
    stride = int(row["stride"])
    for case_name, expected_stride, label in NUMBERED_CASES:
        if row["case"] != case_name:
            continue
        if expected_stride is None or stride == expected_stride:
            return label
    return None


def dedupe_case_rows(rows):
    unique_rows = []
    seen = set()
    for row in rows:
        label = canonical_case_label(row)
        if label in seen:
            continue
        seen.add(label)
        unique_rows.append(row)
    return unique_rows


def ordered_summary_rows(rows):
    numbered_rows = {}
    extra_rows = []
    seen_extra_labels = set()

    for row in rows:
        if row["case"] == "stride_conflict_sweep" and int(row["stride"]) == 1:
            continue

        label = numbered_case_label(row)
        if label is not None and label not in numbered_rows:
            numbered_rows[label] = row
            continue

        canonical_label = canonical_case_label(row)
        if canonical_label in seen_extra_labels:
            continue
        seen_extra_labels.add(canonical_label)
        extra_rows.append((canonical_label, row))

    ordered = []
    for _, _, label in NUMBERED_CASES:
        row = numbered_rows.get(label)
        if row is not None:
            ordered.append((label, row))
    for label, row in sorted(extra_rows, key=lambda item: item[0]):
        ordered.append((f"unassigned  {label}", row))
    return ordered


def stride_sweep_rows(rows):
    stride_rows = [
        row
        for row in rows
        if row["case"] == "stride_conflict_sweep" and int(row["stride"]) > 0
    ]
    # run_basic includes one stride=1 row from --case all; retain one row/stride.
    by_stride = {}
    for row in stride_rows:
        by_stride[int(row["stride"])] = row
    return [by_stride[stride] for stride in sorted(by_stride)]


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


def plot_vertical_bars(plt, labels, values, *, ylabel, title, output, color):
    plt.figure(figsize=(max(7, len(labels) * 0.9), 4.8))
    bars = plt.bar(labels, values, color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
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
            rotation=0,
        )
    plt.ylim(top=max_value + offset * 4 if max_value > 0.0 else 1.0)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Wrote {output}")


def plot_horizontal_bars(plt, labels, values, *, xlabel, title, output, color):
    plt.figure(figsize=(9, max(4.5, len(labels) * 0.55)))
    plt.barh(labels, values, color=color)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Wrote {output}")


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH}; run scripts/run_basic.sh first.")

    rows = load_rows()
    if not rows:
        raise SystemExit(
            f"{CSV_PATH} is empty; rerun scripts/run_basic.sh with a CUDA device."
        )
    print_table(
        "All cases",
        rows,
        ["case", "stride", "avg_ms", "min_ms", "effective_GBps"],
    )

    sweep = stride_sweep_rows(rows)
    print_table("Stride sweep", sweep, ["stride", "avg_ms", "min_ms"])

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is unavailable; skipping PNG charts")
        return

    summary_rows = ordered_summary_rows(rows)
    case_labels = [label for label, _ in summary_rows]
    plot_vertical_bars(
        plt,
        case_labels,
        [float(row["avg_ms"]) for _, row in summary_rows],
        ylabel="Kernel time (ms)",
        title="All cases avg_ms (user order)",
        output=RESULTS_DIR / "all_cases_avg_ms_bar.png",
        color="#F58518",
    )
    plot_vertical_bars(
        plt,
        case_labels,
        [float(row["effective_GBps"]) for _, row in summary_rows],
        ylabel="Effective GB/s",
        title="All cases effective_GBps (user order)",
        output=RESULTS_DIR / "all_cases_effective_gbps_bar.png",
        color="#54A24B",
    )

    unassigned = [label for label in case_labels if label.startswith("unassigned  ")]
    if unassigned:
        print("\nUnassigned cases")
        for label in unassigned:
            print(label.replace("unassigned  ", ""))


if __name__ == "__main__":
    main()
