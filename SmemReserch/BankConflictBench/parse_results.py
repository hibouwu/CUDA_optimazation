#!/usr/bin/env python3
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
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


def load_rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def canonical_case_label(row):
    base_label = CASE_LABELS.get(row["case"], row["case"])
    if row["case"] == "stride_conflict_sweep":
        return f"{base_label} ({row['stride']})"
    return base_label


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
    plt.bar(labels, values, color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
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
        raise SystemExit(f"Missing {CSV_PATH}; run ./run_basic.sh first.")

    rows = load_rows()
    if not rows:
        raise SystemExit(f"{CSV_PATH} is empty; rerun ./run_basic.sh with a CUDA device.")
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

    strides = [int(row["stride"]) for row in sweep]
    plot_vertical_bars(
        plt,
        [str(stride) for stride in strides],
        [float(row["avg_ms"]) for row in sweep],
        ylabel="Kernel time (ms)",
        title="Stride sweep avg_ms",
        output=RESULTS_DIR / "stride_sweep_avg_ms_bar.png",
        color="#4C78A8",
    )
    plot_vertical_bars(
        plt,
        [str(stride) for stride in strides],
        [float(row["effective_GBps"]) for row in sweep],
        ylabel="Effective GB/s",
        title="Stride sweep effective_GBps",
        output=RESULTS_DIR / "stride_sweep_effective_gbps_bar.png",
        color="#72B7B2",
    )

    summary_rows = dedupe_case_rows(rows)
    case_labels = [canonical_case_label(row) for row in summary_rows]
    plot_horizontal_bars(
        plt,
        case_labels,
        [float(row["avg_ms"]) for row in summary_rows],
        xlabel="Kernel time (ms)",
        title="All cases avg_ms",
        output=RESULTS_DIR / "all_cases_avg_ms_bar.png",
        color="#F58518",
    )
    plot_horizontal_bars(
        plt,
        case_labels,
        [float(row["effective_GBps"]) for row in summary_rows],
        xlabel="Effective GB/s",
        title="All cases effective_GBps",
        output=RESULTS_DIR / "all_cases_effective_gbps_bar.png",
        color="#54A24B",
    )

    plt.figure(figsize=(7, 4))
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
    plt.close()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
