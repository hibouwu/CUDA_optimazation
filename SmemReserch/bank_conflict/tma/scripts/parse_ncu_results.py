#!/usr/bin/env python3
import csv
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RESULTS_DIR = ROOT / "results" / "ncu"
CASE_ORDER = [
    "T0a_gmem_to_smem_no_swizzle_copy",
    "T0b_smem_to_gmem_no_swizzle_copy",
    "T1a_load_no_swizzle_forced_conflict",
    "T1b_load_32b_swizzle_matched_consumer",
    "T1c_load_64b_swizzle_matched_consumer",
    "T1d_load_128b_swizzle_matched_consumer",
    "T2a_forced_conflict_producer_store_no_swizzle",
    "T2b_matched_producer_store_32b_swizzle",
    "T2c_matched_producer_store_64b_swizzle",
    "T2d_matched_producer_store_128b_swizzle",
    "T3a_load_store_no_swizzle",
    "T3b_load_store_32b_swizzle",
    "T3c_load_store_64b_swizzle",
    "T3d_load_store_128b_swizzle",
]

METRIC_LABELS = {
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct": (
        "warp_issue_mio_throttle_pct",
        "Warp issue stalled by MIO throttle (%)",
    ),
    "l1tex__data_bank_reads.sum": (
        "shared_bank_reads",
        "Shared-memory bank reads",
    ),
    "l1tex__data_bank_writes.sum": (
        "shared_bank_writes",
        "Shared-memory bank writes",
    ),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": (
        "shared_ld_bank_conflicts",
        "Shared-load bank conflicts",
    ),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": (
        "shared_st_bank_conflicts",
        "Shared-store bank conflicts",
    ),
    "l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum": (
        "shared_ld_requests",
        "Shared-load requests",
    ),
    "l1tex__t_requests_pipe_lsu_mem_shared_op_st.sum": (
        "shared_st_requests",
        "Shared-store requests",
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
        line.strip() for line in text.splitlines() if line.startswith("==ERROR==")
    ]
    csv_lines = [
        line for line in text.splitlines() if line and not line.startswith("==")
    ]
    if not csv_lines:
        return {}, error_lines or ["No metric rows found."]

    reader = csv.reader(csv_lines)
    header = None
    header_index = None
    for row in reader:
        if "Metric Name" in row and "Metric Value" in row:
            header = row
            header_index = {name: idx for idx, name in enumerate(header) if name}
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
        if report_path.name == "summary.csv":
            continue
        metrics, errors = extract_metric_rows(report_path)
        if metrics:
            valid_reports[report_path.stem] = metrics
        elif errors:
            failed_reports[report_path.stem] = errors
    return valid_reports, failed_reports


def ordered_report_rows(valid_reports):
    rows = []
    for case_name in CASE_ORDER:
        metrics = valid_reports.get(case_name)
        if metrics is not None:
            rows.append((case_name, metrics))
    for case_name in sorted(valid_reports):
        if case_name not in CASE_ORDER:
            rows.append((case_name, valid_reports[case_name]))
    return rows


def plot_metric_bars(plt, labels, values, *, ylabel, title, output, color):
    plt.figure(figsize=(max(12, len(labels) * 0.9), 5.5))
    bars = plt.bar(labels, values, color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=32, ha="right")
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


def write_summary_csv(metric_names, ordered_rows):
    output = RESULTS_DIR / "summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", *metric_names])
        for case_name, metrics in ordered_rows:
            writer.writerow([case_name, *(metrics.get(name, "") for name in metric_names)])
    print(f"Wrote {output}")


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

    write_summary_csv(metric_names, ordered_rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping Nsight Compute charts")
        return

    for metric_name in metric_names:
        metric_rows = [
            (case_name, metrics[metric_name])
            for case_name, metrics in ordered_rows
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
            [case_name for case_name, _ in metric_rows],
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
