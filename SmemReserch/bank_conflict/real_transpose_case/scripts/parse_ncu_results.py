#!/usr/bin/env python3
import csv
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RESULTS_DIR = ROOT / "results" / "ncu"
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
METRIC_LABELS = {
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": (
        "global_load_sectors",
        "Global-load sectors",
    ),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": (
        "global_store_sectors",
        "Global-store sectors",
    ),
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": (
        "global_load_requests",
        "Global-load requests",
    ),
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": (
        "global_store_requests",
        "Global-store requests",
    ),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": (
        "shared_load_bank_conflicts",
        "Shared-load bank conflicts",
    ),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": (
        "shared_store_bank_conflicts",
        "Shared-store bank conflicts",
    ),
}


def metric_value(raw_value):
    normalized = raw_value.strip().replace(",", "").rstrip("%")
    if not normalized or normalized in {"N/A", "nan", "NaN"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def extract_metrics(report_path):
    text = report_path.read_text(encoding="utf-8", errors="replace")
    errors = [
        line.strip() for line in text.splitlines() if line.startswith("==ERROR==")
    ]
    rows = csv.reader(
        line for line in text.splitlines() if line and not line.startswith("==")
    )
    header = None
    for row in rows:
        if "Metric Name" in row and "Metric Value" in row:
            header = {name: index for index, name in enumerate(row) if name}
            break
    if header is None:
        return {}, errors or ["Metric CSV header was not found."]

    name_index = header["Metric Name"]
    value_index = header["Metric Value"]
    metrics = {}
    for row in rows:
        if len(row) <= max(name_index, value_index):
            continue
        value = metric_value(row[value_index])
        if value is not None:
            metrics[row[name_index].strip()] = value
    return metrics, errors


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def format_bar_value(value):
    magnitude = abs(value)
    if magnitude >= 1.0e6:
        return f"{value / 1.0e6:.1f}M"
    if magnitude >= 1.0e3:
        return f"{value / 1.0e3:.1f}K"
    if value.is_integer():
        return f"{value:.0f}"
    return f"{value:.3f}"


def print_table(metric_name, rows):
    case_width = max(len("case"), *(len(name) for name, _ in rows))
    print(f"\n{metric_name}")
    print(f"{'case'.ljust(case_width)}  value")
    print(f"{'-' * case_width}  {'-' * 12}")
    for name, value in rows:
        print(f"{name.ljust(case_width)}  {value:.3f}")


def plot(plt, metric_name, rows):
    stem, title = METRIC_LABELS.get(
        metric_name, (sanitize(metric_name), metric_name)
    )
    labels = [name for name, _ in rows]
    values = [value for _, value in rows]
    plt.figure(figsize=(max(10.0, len(labels) * 1.15), 5.2))
    bars = plt.bar(labels, values, color="#4C78A8")
    plt.ylabel("Metric value")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    max_value = max(values) if values else 0.0
    offset = max_value * 0.015 if max_value > 0.0 else 0.01
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            format_bar_value(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.ylim(top=max_value + offset * 4 if max_value > 0.0 else 1.0)
    plt.tight_layout()
    output = RESULTS_DIR / f"{stem}.png"
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Wrote {output}")


def main():
    reports = {}
    failures = {}
    for path in RESULTS_DIR.glob("*.csv"):
        metrics, errors = extract_metrics(path)
        if metrics:
            reports[path.stem] = metrics
        else:
            failures[path.stem] = errors or ["No numeric metrics found."]
    if not reports:
        raise SystemExit(
            f"No valid NCU CSV reports found in {RESULTS_DIR}; run run_ncu.sh first."
        )

    ordered = [(name, reports[name]) for name in CASE_ORDER if name in reports]
    metric_names = []
    for _, metrics in ordered:
        for metric_name in metrics:
            if metric_name not in metric_names:
                metric_names.append(metric_name)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("matplotlib is unavailable; skipping NCU PNG charts")

    for metric_name in metric_names:
        rows = [
            (name, metrics[metric_name])
            for name, metrics in ordered
            if metric_name in metrics
        ]
        print_table(metric_name, rows)
        if plt is not None:
            plot(plt, metric_name, rows)

    if failures:
        print("\nFailed reports")
        for name, errors in sorted(failures.items()):
            print(f"{name}: {errors[0]}")


if __name__ == "__main__":
    main()
