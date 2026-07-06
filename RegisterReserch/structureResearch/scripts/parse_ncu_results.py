#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RESULTS_DIR = ROOT / "results" / "ncu"
BANK_SCAN_MANIFEST = ROOT / "results" / "bank_scan" / "manifest.csv"
BANK_SCAN_FFMA_MANIFEST = ROOT / "results" / "bank_scan_ffma" / "manifest.csv"

PTX_CASES = [
    ("R0_imad_chain", "R0_imad_chain", "latency"),
    ("R1_imad_independent_x4", "R1_imad_independent_x4", "throughput"),
    ("R2_reuse_hot_x4", "R2_reuse_hot_x4", "reuse"),
    ("R3_bank_dense_x4", "R3_bank_dense_x4", "bank_candidate"),
    ("R4_bank_sparse_x4", "R4_bank_sparse_x4", "bank_candidate"),
]

CATEGORY_ORDER = [
    "source_count",
    "slot_permutation",
    "latency_stride",
    "throughput_stride",
    "latency",
    "throughput",
    "reuse",
    "bank_candidate",
    "unknown",
]

METRIC_LABELS = {
    "smsp__inst_executed.sum": (
        "inst_executed",
        "Executed instructions",
    ),
    "smsp__inst_issued.sum": (
        "inst_issued",
        "Issued instructions",
    ),
    "smsp__pipe_fu_core_active.avg": (
        "pipe_fu_core_active",
        "Core functional-unit active",
    ),
    "smsp__inst_executed_op_fp32.sum": (
        "inst_executed_fp32",
        "Executed FP32 instructions",
    ),
    "smsp__inst_executed_op_integer.sum": (
        "inst_executed_integer",
        "Executed integer instructions",
    ),
    "smsp__inst_executed_op_logic.sum": (
        "inst_executed_logic",
        "Executed logic instructions",
    ),
}


def parse_metric_value(raw_value):
    normalized = raw_value.strip().replace(",", "")
    if not normalized or normalized in {"N/A", "n/a", "nan", "NaN"}:
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


def load_bank_scan_manifest():
    manifests = {}
    for manifest_path, family in (
        (BANK_SCAN_MANIFEST, "bank_scan"),
        (BANK_SCAN_FFMA_MANIFEST, "bank_scan_ffma"),
    ):
        if not manifest_path.exists():
            continue
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                manifests[row["case"]] = {**row, "family": family}
    return manifests


def bank_scan_label(row):
    category = row.get("category", "")
    if category == "latency_stride":
        return f"b{int(row['base']):02d}/s{int(row['stride']):02d}"
    if category == "throughput_stride":
        return f"s{int(row['stride']):02d}"
    return row["case"]


def build_case_metadata(report_names):
    bank_manifest = load_bank_scan_manifest()
    ptx_map = {name: {"case": name, "label": label, "category": category, "family": "ptx"}
               for name, label, category in PTX_CASES}

    metadata = {}
    for report_name in report_names:
        if report_name in bank_manifest:
            row = bank_manifest[report_name]
            metadata[report_name] = {
                "case": report_name,
                "label": bank_scan_label(row),
                "category": row.get("category", "unknown") or "unknown",
                "family": row.get("family", "bank_scan"),
            }
        elif report_name in ptx_map:
            metadata[report_name] = ptx_map[report_name]
        else:
            metadata[report_name] = {
                "case": report_name,
                "label": report_name,
                "category": "unknown",
                "family": "unknown",
            }
    return metadata


def case_sort_key(case_name, metadata, manifest_order, ptx_order):
    if case_name in manifest_order:
        return (0, manifest_order[case_name], case_name)
    if case_name in ptx_order:
        return (1, ptx_order[case_name], case_name)
    category = metadata[case_name]["category"]
    category_rank = (
        CATEGORY_ORDER.index(category)
        if category in CATEGORY_ORDER
        else len(CATEGORY_ORDER)
    )
    return (2, category_rank, case_name)


def format_bar_value(value):
    magnitude = abs(value)
    if magnitude >= 1.0e9:
        return f"{value / 1.0e9:.2f}G"
    if magnitude >= 1.0e6:
        return f"{value / 1.0e6:.2f}M"
    if magnitude >= 1.0e3:
        return f"{value / 1.0e3:.2f}K"
    if float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.3f}"


def print_metric_table(metric_name, rows):
    print(f"\n{metric_name}")
    case_width = max(len("case"), *(len(row["case"]) for row in rows))
    label_width = max(len("label"), *(len(row["label"]) for row in rows))
    value_width = max(len("value"), *(len(f"{row['value']:.3f}") for row in rows))
    print(
        "  ".join(
            [
                "case".ljust(case_width),
                "label".ljust(label_width),
                "value".ljust(value_width),
            ]
        )
    )
    print(
        "  ".join(
            [
                "-" * case_width,
                "-" * label_width,
                "-" * value_width,
            ]
        )
    )
    for row in rows:
        print(
            "  ".join(
                [
                    row["case"].ljust(case_width),
                    row["label"].ljust(label_width),
                    f"{row['value']:.3f}".ljust(value_width),
                ]
            )
        )


def plot_metric_bars(plt, rows, *, ylabel, title, output, color):
    labels = [row["label"] for row in rows]
    values = [row["value"] for row in rows]
    if len(labels) > 18:
        figure_height = max(5.0, len(labels) * 0.28)
        plt.figure(figsize=(12.5, figure_height))
        bars = plt.barh(labels, values, color=color)
        plt.xlabel(ylabel)
        plt.title(title)
        plt.grid(axis="x", alpha=0.3)
        max_value = max(values) if values else 0.0
        offset = max_value * 0.01 if max_value > 0.0 else 0.01
        for bar, value in zip(bars, values):
            plt.text(
                value + offset,
                bar.get_y() + bar.get_height() / 2.0,
                format_bar_value(value),
                va="center",
                ha="left",
                fontsize=8,
            )
        plt.xlim(right=max_value + offset * 10 if max_value > 0.0 else 1.0)
    else:
        plt.figure(figsize=(max(8.0, len(labels) * 1.0), 5.2))
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
                format_bar_value(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.ylim(top=max_value + offset * 4 if max_value > 0.0 else 1.0)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Wrote {output}")


def write_summary(entries):
    output = RESULTS_DIR / "summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case",
                "label",
                "category",
                "family",
                "metric_name",
                "metric_value",
            ],
        )
        writer.writeheader()
        for entry in entries:
            for metric_name, metric_value in entry["metrics"].items():
                writer.writerow(
                    {
                        "case": entry["case"],
                        "label": entry["label"],
                        "category": entry["category"],
                        "family": entry["family"],
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                    }
                )
    print(f"Wrote {output}")


def main(argv):
    if not RESULTS_DIR.exists():
        raise SystemExit(f"Missing {RESULTS_DIR}; run scripts/run_ncu.sh first.")

    requested_reports = set(argv[1:])
    valid_reports = {}
    failed_reports = {}
    for report_path in sorted(RESULTS_DIR.glob("*.csv")):
        if report_path.name == "summary.csv":
            continue
        if requested_reports and report_path.stem not in requested_reports:
            continue
        metrics, errors = extract_metric_rows(report_path)
        if metrics:
            valid_reports[report_path.stem] = metrics
        elif errors:
            failed_reports[report_path.stem] = errors

    if not valid_reports:
        print("No valid Nsight Compute metric CSVs were found.")
        if failed_reports:
            print("\nFailed reports")
            for report_name, errors in sorted(failed_reports.items()):
                print(f"{report_name}: {errors[0]}")
        raise SystemExit(1)

    metadata = build_case_metadata(valid_reports.keys())
    manifest_order = {}
    if BANK_SCAN_MANIFEST.exists():
        with BANK_SCAN_MANIFEST.open(newline="", encoding="utf-8") as stream:
            manifest_order = {
                row["case"]: index for index, row in enumerate(csv.DictReader(stream))
            }
    ptx_order = {name: index for index, (name, _, _) in enumerate(PTX_CASES)}

    ordered_names = sorted(
        valid_reports,
        key=lambda name: case_sort_key(name, metadata, manifest_order, ptx_order),
    )
    entries = [
        {
            "case": name,
            "label": metadata[name]["label"],
            "category": metadata[name]["category"],
            "family": metadata[name].get("family", "unknown"),
            "metrics": valid_reports[name],
        }
        for name in ordered_names
    ]

    write_summary(entries)

    metric_names = []
    for entry in entries:
        for metric_name in entry["metrics"]:
            if metric_name not in metric_names:
                metric_names.append(metric_name)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("matplotlib is unavailable; skipping Nsight Compute charts")

    has_bank_scan = any(entry["case"] in manifest_order for entry in entries)
    for metric_name in metric_names:
        rows = [
            {
                "case": entry["case"],
                "label": entry["label"],
                "category": entry["category"],
                "family": entry.get("family", "unknown"),
                "value": entry["metrics"][metric_name],
            }
            for entry in entries
            if metric_name in entry["metrics"]
        ]
        if not rows:
            continue
        print_metric_table(metric_name, rows)
        if plt is None:
            continue
        stem, title = METRIC_LABELS.get(
            metric_name, (sanitize_metric_name(metric_name), metric_name)
        )
        if has_bank_scan:
            families = sorted({row["family"] for row in rows})
            for family in families:
                family_rows = [row for row in rows if row["family"] == family]
                categories = sorted(
                    {row["category"] for row in family_rows},
                    key=lambda value: CATEGORY_ORDER.index(value)
                    if value in CATEGORY_ORDER
                    else len(CATEGORY_ORDER),
                )
                for category in categories:
                    category_rows = [
                        row
                        for row in family_rows
                        if row["category"] == category
                    ]
                    if not category_rows:
                        continue
                    suffix = "" if family == "bank_scan" else f"_{family}"
                    plot_metric_bars(
                        plt,
                        category_rows,
                        ylabel="Metric value",
                        title=f"{title} ({family}, {category})",
                        output=RESULTS_DIR / f"{stem}_{category}{suffix}.png",
                        color="#4C78A8",
                    )
        else:
            plot_metric_bars(
                plt,
                rows,
                ylabel="Metric value",
                title=title,
                output=RESULTS_DIR / f"{stem}.png",
                color="#4C78A8",
            )

    if failed_reports:
        print("\nFailed reports")
        for report_name, errors in sorted(failed_reports.items()):
            print(f"{report_name}: {errors[0]}")


if __name__ == "__main__":
    main(sys.argv)
