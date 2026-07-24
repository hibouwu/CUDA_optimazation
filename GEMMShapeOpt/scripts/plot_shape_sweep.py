#!/usr/bin/env python3
import argparse
import csv
import html
import re
from pathlib import Path


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def shape_key(row):
    return (
        row.get("Category", ""),
        int(row.get("M", "0")),
        int(row.get("N", "0")),
        int(row.get("K", "0")),
        row.get("Note", ""),
        row.get("Epilogue", "none"),
    )


def shape_label(key):
    category, m, n, k, _, epilogue = key
    if epilogue and epilogue != "none":
        return f"{category} {m}x{n}x{k} [{epilogue}]"
    return f"{category} {m}x{n}x{k}"


def slug(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "shape_sweep"


def pick_points(rows, backend):
    grouped = {}
    for row in rows:
        grouped.setdefault(shape_key(row), []).append(row)

    points = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2], item[3], item[5], item[4])):
        candidates = [
            row
            for row in grouped[key]
            if row.get("BackendId") == backend
            and row.get("Matched") == "1"
            and row.get("Status") == "ok"
        ]
        if not candidates:
            candidates = [
                row
                for row in grouped[key]
                if row.get("BackendId") != "cublas_tc"
                and row.get("Matched") == "1"
                and row.get("Status") == "ok"
            ]
        if candidates:
            best = max(candidates, key=lambda row: to_float(row.get("RatioToReference")))
            points.append((shape_label(key), to_float(best.get("RatioToReference")), best.get("BackendId", "")))
        else:
            points.append((shape_label(key), 0.0, "missing"))
    return points


def write_ratio_svg(points, title, output_path, threshold):
    width = max(1100, 90 * max(1, len(points)) + 220)
    height = 620
    left = 74
    right = 34
    top = 70
    bottom = 190
    plot_w = width - left - right
    plot_h = height - top - bottom

    values = [value for _, value, _ in points]
    y_min = min(0.85, min(values, default=threshold) - 0.03, threshold - 0.03)
    y_max = max(1.08, max(values, default=1.0) + 0.04)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def sx(index):
        if len(points) <= 1:
            return left + plot_w / 2
        return left + index / (len(points) - 1) * plot_w

    def sy(value):
        return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    def y_tick_values():
        ticks = []
        value = round(y_min / 0.05) * 0.05
        while value <= y_max + 1e-9:
            if value >= y_min - 1e-9:
                ticks.append(round(value, 2))
            value += 0.05
        return ticks

    blue = "#2563eb"
    red = "#dc2626"
    gray = "#6b7280"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="{left}" y="56" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">shapeopt / cuBLASLt Matmul heuristic ratio; gate = {threshold:.2f}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#d1d5db"/>',
    ]

    for tick in y_tick_values():
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#374151">{tick:.2f}</text>')

    ref_y = sy(1.0)
    gate_y = sy(threshold)
    lines.append(f'<line x1="{left}" y1="{ref_y:.2f}" x2="{left + plot_w}" y2="{ref_y:.2f}" stroke="{gray}" stroke-width="2" stroke-dasharray="8 5"/>')
    lines.append(f'<text x="{left + plot_w - 4}" y="{ref_y - 6:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="{gray}">1.00 reference</text>')
    lines.append(f'<line x1="{left}" y1="{gate_y:.2f}" x2="{left + plot_w}" y2="{gate_y:.2f}" stroke="{red}" stroke-width="2" stroke-dasharray="8 5"/>')
    lines.append(f'<text x="{left + plot_w - 4}" y="{gate_y - 6:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="{red}">0.90 gate</text>')

    if points:
        path = " ".join(
            f"{'M' if index == 0 else 'L'} {sx(index):.2f} {sy(value):.2f}"
            for index, (_, value, _) in enumerate(points)
        )
        lines.append(f'<path d="{path}" fill="none" stroke="{blue}" stroke-width="3"/>')

    for index, (label, value, backend) in enumerate(points):
        x = sx(index)
        y = sy(value)
        color = blue if value >= threshold else red
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#eef2f7"/>')
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#ffffff" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{x:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#111827">{value:.3f}</text>')
        lines.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 18}" transform="rotate(45 {x:.2f} {top + plot_h + 18})" '
            f'text-anchor="start" font-family="Arial, sans-serif" font-size="11" fill="#374151">{html.escape(label)}</text>'
        )
        lines.append(f'<title>{html.escape(label)}: {backend} ratio={value:.4f}</title>')

    legend_x = left
    legend_y = height - 28
    lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 32}" y2="{legend_y}" stroke="{blue}" stroke-width="3"/>')
    lines.append(f'<circle cx="{legend_x + 16}" cy="{legend_y}" r="4.5" fill="#ffffff" stroke="{blue}" stroke-width="2"/>')
    lines.append(f'<text x="{legend_x + 42}" y="{legend_y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#374151">shapeopt ratio</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n")


def write_index(pages, output_path):
    body = [
        "<!doctype html>",
        "<meta charset=\"utf-8\">",
        "<title>GEMM ShapeOpt Ratio Plots</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#111827} iframe{width:100%;height:660px;border:1px solid #d1d5db;margin:16px 0 32px} h1{font-size:24px} h2{font-size:18px}</style>",
        "<h1>GEMM ShapeOpt Ratio Plots</h1>",
    ]
    for title, filename in pages:
        body.append(f"<h2>{html.escape(title)}</h2>")
        body.append(f'<iframe src="{html.escape(filename)}"></iframe>')
    output_path.write_text("\n".join(body) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot GEMMShapeOpt shape sweep ratios as SVG line charts.")
    parser.add_argument("--csv", action="append", required=True, type=Path, help="shape_sweep.csv path; can be repeated")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--backend", default="shapeopt")
    parser.add_argument("--threshold", default=0.90, type=float)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for csv_path in args.csv:
        rows = read_rows(csv_path)
        run_name = csv_path.parent.name
        title = f"{run_name}: {args.backend} ratio"
        output_path = args.out_dir / f"{slug(run_name)}_ratio.svg"
        write_ratio_svg(pick_points(rows, args.backend), title, output_path, args.threshold)
        pages.append((title, output_path.name))

    write_index(pages, args.out_dir / "index.html")
    for _, filename in pages:
        print(args.out_dir / filename)
    print(args.out_dir / "index.html")


if __name__ == "__main__":
    raise SystemExit(main())
