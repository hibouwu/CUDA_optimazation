#!/usr/bin/env python3
import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path


PRECISION_ORDER = ["NVFP4", "MXFP4", "FP8", "INT8"]
COLORS = {
    "NVFP4": "#2563eb",
    "MXFP4": "#16a34a",
    "FP8": "#dc2626",
    "INT8": "#7c3aed",
}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean(values):
    return sum(values) / len(values) if values else 0.0


def aggregate_best_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("Status") != "ok" or row.get("Matched") != "1":
            continue
        precision = row.get("Precision", "")
        backend_id = row.get("BackendId", "")
        groups[(precision, backend_id)].append(row)

    best = {}
    for (precision, backend_id), group_rows in groups.items():
        gflops_values = [to_float(row.get("GFLOPS")) for row in group_rows]
        time_values = [to_float(row.get("TimeMs")) for row in group_rows]
        reference_values = []
        for row in group_rows:
            gflops = to_float(row.get("GFLOPS"))
            ratio = to_float(row.get("RatioToReference"))
            if gflops > 0.0 and ratio > 0.0:
                reference_values.append(gflops / ratio)
        mean_gflops = mean(gflops_values)
        mean_reference = mean(reference_values)
        mean_ratio = mean_gflops / mean_reference if mean_reference > 0.0 else mean(
            [to_float(row.get("RatioToReference")) for row in group_rows]
        )
        aggregate = {
            "Precision": precision,
            "Stage": group_rows[0].get("Stage", ""),
            "BackendId": backend_id,
            "BackendLabel": group_rows[0].get("BackendLabel", ""),
            "N": group_rows[0].get("N", ""),
            "Reference": group_rows[0].get("Reference", ""),
            "TimeMs": f"{mean(time_values):.6g}",
            "GFLOPS": f"{mean_gflops:.6g}",
            "RatioToReference": f"{mean_ratio:.6g}",
            "TrialCount": str(len(group_rows)),
        }
        if precision not in best or mean_gflops > to_float(best[precision].get("GFLOPS")):
            best[precision] = aggregate
    return best


def write_svg(best, out_path: Path):
    width = 900
    height = 460
    margin_left = 80
    margin_right = 40
    margin_top = 56
    margin_bottom = 96
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    max_ratio = max([to_float(row.get("RatioToReference")) for row in best.values()] + [0.01])
    y_max = max(0.95, max_ratio * 1.15)
    bar_gap = 30
    bar_w = (chart_w - bar_gap * (len(PRECISION_ORDER) + 1)) / len(PRECISION_ORDER)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">SM110 Quantized GEMM 1024 Best Backend Ratio by Precision</text>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="#111827" stroke-width="1"/>',
        f'<text x="18" y="{margin_top + chart_h/2:.1f}" transform="rotate(-90 18 {margin_top + chart_h/2:.1f})" text-anchor="middle" font-family="Arial" font-size="13">Ratio to reference</text>',
    ]

    for tick in range(5):
        value = y_max * tick / 4
        y = margin_top + chart_h - chart_h * value / y_max
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + chart_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#374151">{value:.2f}</text>')

    target_y = margin_top + chart_h - chart_h * 0.90 / y_max
    lines.append(f'<line x1="{margin_left}" y1="{target_y:.1f}" x2="{margin_left + chart_w}" y2="{target_y:.1f}" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="6 4"/>')
    lines.append(f'<text x="{margin_left + chart_w - 4}" y="{target_y - 6:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#b91c1c">0.90 target</text>')

    for i, precision in enumerate(PRECISION_ORDER):
        x = margin_left + bar_gap + i * (bar_w + bar_gap)
        row = best.get(precision)
        if row is None:
            bar_h = 0
            y = margin_top + chart_h
            label = "missing"
            ratio = ""
            backend = ""
            color = "#d1d5db"
        else:
            gflops = to_float(row.get("GFLOPS"))
            ratio_value = to_float(row.get("RatioToReference"))
            bar_h = chart_h * ratio_value / y_max
            y = margin_top + chart_h - bar_h
            label = f"{ratio_value:.3f}x"
            ratio = f"{gflops:.0f} GF/s"
            backend = row.get("BackendId", "")
            color = COLORS.get(precision, "#6b7280")
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>')
        lines.append(f'<text x="{x + bar_w/2:.1f}" y="{max(y - 8, margin_top + 12):.1f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#111827">{html.escape(label)}</text>')
        if ratio:
            if bar_h >= 24:
                metric_y = max(y + 16, margin_top + 28)
                metric_fill = "white"
            else:
                metric_y = max(y - 22, margin_top + 28)
                metric_fill = "#4b5563"
            lines.append(f'<text x="{x + bar_w/2:.1f}" y="{metric_y:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="{metric_fill}">{html.escape(ratio)}</text>')
        lines.append(f'<text x="{x + bar_w/2:.1f}" y="{margin_top + chart_h + 24}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700">{html.escape(precision)}</text>')
        lines.append(f'<text x="{x + bar_w/2:.1f}" y="{margin_top + chart_h + 43}" text-anchor="middle" font-family="Arial" font-size="10" fill="#4b5563">{html.escape(backend[:28])}</text>')

    lines.append('<text x="80" y="438" font-family="Arial" font-size="11" fill="#6b7280">Bars use 10-trial means from Status=ok and Matched=1 rows. Missing precisions are grey placeholders.</text>')
    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot SM110 quantized GEMM best backend per precision")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    write_svg(aggregate_best_rows(rows), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
