#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "ncu"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"
CASE_ORDER = [
    "R0_imad_chain",
    "R1_imad_independent_x4",
    "R2_reuse_hot_x4",
    "R3_bank_dense_x4",
    "R4_bank_sparse_x4",
]


def parse_value(raw):
    normalized = raw.strip().replace(",", "").rstrip("%")
    if not normalized or normalized.lower() in {"n/a", "nan"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def read_metrics(path):
    lines = [
        line
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line and not line.startswith("==")
    ]
    if not lines:
        return {}
    metrics = {}
    for row in csv.DictReader(lines):
        name = row.get("Metric Name", "")
        value = parse_value(row.get("Metric Value", ""))
        if name and value is not None:
            metrics[name] = value
    return metrics


def main():
    rows = []
    metric_names = set()
    for case_name in CASE_ORDER:
        path = RESULTS_DIR / f"{case_name}.csv"
        if not path.exists():
            continue
        metrics = read_metrics(path)
        metric_names.update(metrics)
        rows.append((case_name, metrics))
    if not rows:
        raise SystemExit(f"No NCU CSV files found in {RESULTS_DIR}")

    ordered_metrics = sorted(metric_names)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", *ordered_metrics])
        for case_name, metrics in rows:
            writer.writerow(
                [case_name, *(metrics.get(name, "") for name in ordered_metrics)]
            )
    print(f"Wrote {SUMMARY_PATH}")

    for metric_name in ordered_metrics:
        print(f"\n{metric_name}")
        for case_name, metrics in rows:
            if metric_name in metrics:
                print(f"{case_name:<26} {metrics[metric_name]:.3f}")


if __name__ == "__main__":
    main()
