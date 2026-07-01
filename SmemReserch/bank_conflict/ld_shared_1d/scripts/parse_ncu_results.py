#!/usr/bin/env python3
import csv
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCH_DIR / "results" / "ncu"

ORDERED_REPORTS = [
    ("baseline", "v0  baseline"),
    ("stride_2", "v1a  stride = 2"),
    ("stride_4", "v1b  stride = 4"),
    ("stride_8", "v1c  stride = 8"),
    ("stride_16", "v1d  stride = 16"),
    ("stride_32", "v1e  same bank (32-way)"),
    ("same_bank_32way_2d", "v1f  same_bank_32way_2d"),
    ("broadcast", "v2a  broadcasts"),
    ("multicast_hash", "v2b  multicasts"),
    ("v4_contiguous", "v3  float4 contiguous"),
    ("v2_multicast_pairs", "v4a  v2_multicast_pairs"),
    ("v4_multicast_quads", "v4b  v4_multicast_quads"),
]

METRIC_LABELS = {
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": (
        "shared_ld_bank_conflicts",
        "Shared-load bank conflicts",
    ),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": (
        "shared_st_bank_conflicts",
        "Shared-store bank conflicts",
    ),
    "l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum": (
        "shared_ld_sectors",
        "Shared-load sectors",
    ),
    "smsp__sass_average_branch_targets_threads_uniform.pct": (
        "branch_uniform_pct",
        "Uniform branch targets (%)",
    ),
}


def parse_metric_value(raw_value):
    normalized = raw_value.strip().replace(",", "")
    if not normalized or normalized in {"N/A", "nan", "NaN"}:
        return None
    normalized = normalized.rstrip("%")
    try:
        return float(normalized)
    except ValueError:
        return None


def sanitize_metric_name(metric_name):
    return re.sub(r"[^A-Za-z0-9]+", "_", metric_name).strip("_").lower()


def extract_metric_rows(report_path):
    text = report_path.read_text(encoding="utf-8", errors="replace")
    error_lines = [
        line.strip()
        for line in text.splitlines()
        if line.startswith("==ERROR==")
    ]
    csv_lines = [
        line for line in text.splitlines() if line and not line.startswith("==")
    ]
    if not csv_lines:
        return {}, error_lines or ["No metric rows found."]

    header = None
    header_index = None
    reader = csv.reader(csv_lines)
    for row in reader:
        if "Metric Name" in row and "Metric Value" in row:
            header = row
            header_index = {
                name: idx for idx, name in enumerate(header) if name
            }
            break
    if header is None or header_index is None:
        return {}, error_lines or ["Metric CSV header was not found."]

    metric_name_idx = header_index["Metric Name"]
    metric_value_idx = header_index["Metric Value"]
    metrics = {}
    for row in reader:
        if len(row) <= max(metric_name_idx, metric_value_idx):
            continue
        metric_name = row[metric_name_idx].strip()
        if not metric_name:
            continue
        metric_value = parse_metric_value(row[metric_value_idx])
        if metric_value is None:
            continue
        metrics[metric_name] = metric_value
    if not metrics:
        return {}, error_lines or ["Metric rows were present but contained no numeric values."]
    return metrics, error_lines


def load_reports():
    if not RESULTS_DIR.exists():
        raise SystemExit(f"Missing {RESULTS_DIR}; run scripts/run_ncu.sh first.")

    valid_reports = {}
    failed_reports = {}
    for report_path in sorted(RESULTS_DIR.glob("*.csv")):
        metrics, errors = extract_metric_rows(report_path)
        if metrics:
            valid_reports[report_path.stem] = metrics
        elif errors:
            failed_reports[report_path.stem] = errors
    return valid_reports, failed_reports


def ordered_report_rows(valid_reports):
    ordered = []
    for report_name, label in ORDERED_REPORTS:
        metrics = valid_reports.get(report_name)
        if metrics is not None:
            ordered.append((label, metrics))
    return ordered


def plot_metric_bars(plt, labels, values, *, ylabel, title, output, color):
    plt.figure(figsize=(max(8, len(labels) * 1.0), 5.2))
    bars = plt.bar(labels, values, color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=28, ha="right")
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


def print_metric_table(metric_name, rows):
    print(f"\n{metric_name}")
    widths = {
        "case": max(len("case"), *(len(label) for label, _ in rows)),
        "value": max(len("value"), *(len(f"{value:.3f}") for _, value in rows)),
    }
    print("  ".join(["case".ljust(widths["case"]), "value".ljust(widths["value"])]))
    print("  ".join(["-" * widths["case"], "-" * widths["value"]]))
    for label, value in rows:
        print("  ".join([label.ljust(widths["case"]), f"{value:.3f}".ljust(widths["value"])]))


def main():
    valid_reports, failed_reports = load_reports()
    if not valid_reports:
        print("No valid Nsight Compute metric CSVs were found.")
        if failed_reports:
            print("\nFailed reports")
            for report_name, errors in failed_reports.items():
                print(f"{report_name}: {errors[0]}")
        raise SystemExit(1)

    ordered_rows = ordered_report_rows(valid_reports)
    if not ordered_rows:
        raise SystemExit("No ordered Nsight Compute reports matched the configured cases.")

    metric_names = []
    for _, metrics in ordered_rows:
        for metric_name in metrics:
            if metric_name not in metric_names:
                metric_names.append(metric_name)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping Nsight Compute charts")
        return

    for metric_name in metric_names:
        metric_rows = [
            (label, metrics[metric_name])
            for label, metrics in ordered_rows
            if metric_name in metrics
        ]
        if not metric_rows:
            continue
        print_metric_table(metric_name, metric_rows)
        stem, title = METRIC_LABELS.get(
            metric_name, (sanitize_metric_name(metric_name), metric_name)
        )
        plot_metric_bars(
            plt,
            [label for label, _ in metric_rows],
            [value for _, value in metric_rows],
            ylabel="Metric value",
            title=title,
            output=RESULTS_DIR / f"{stem}.png",
            color="#4C78A8",
        )

    if failed_reports:
        print("\nFailed reports")
        for report_name, errors in failed_reports.items():
            print(f"{report_name}: {errors[0]}")


if __name__ == "__main__":
    main()
