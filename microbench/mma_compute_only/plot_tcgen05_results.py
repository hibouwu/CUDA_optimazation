#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "分析报告.txt"
NCU_DIR = ROOT / "ncu_reports"
PLOTS_DIR = ROOT / "plots"

CURRENT_LOG_RE = re.compile(
    r"^tcgen05_(single_warp_block|full_sm_4warp_block)_(m128n256|m128n128|m128n64)_(fp4|fp8|bf16)_benchmark\.log$"
)

PRECISION_ORDER = {"FP4": 0, "FP8": 1, "BF16": 2}
LAUNCH_ORDER = {"SingleWarpBlock": 0, "FullSM4WarpBlock": 1}
LAUNCH_PANELS = [
    ("SingleWarpBlock", "SingleWarp"),
    ("FullSM4WarpBlock", "4Warp FullSM"),
]
SHAPE_ORDER = {
    "M128N256K64": 0,
    "M128N128K64": 1,
    "M128N64K64": 2,
    "M128N256K32": 0,
    "M128N128K32": 1,
    "M128N64K32": 2,
    "M128N256K16": 0,
    "M128N128K16": 1,
    "M128N64K16": 2,
}
COLORS = {
    "FP4": "#2878b5",
    "FP8": "#c75d2c",
    "BF16": "#2b8c5a",
}


def sort_key(row):
    return (
        PRECISION_ORDER.get(row["precision"], 99),
        LAUNCH_ORDER.get(row["launch"], 99),
        SHAPE_ORDER.get(row["shape"], 99),
    )


def normalize_shape_label(shape):
    match = re.fullmatch(r"N(\d+)M(\d+)(K\d+)?", shape)
    if not match:
        return shape
    n_value, m_value, k_value = match.groups()
    return f"M{m_value}N{n_value}{k_value or ''}"


def parse_report(path):
    rows = []
    in_table = False
    current_precision = ""
    for line in path.read_text().splitlines():
        if line.startswith("| **精度** |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        if line.startswith("| ---"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 8:
            continue
        precision_cell, warp_num, launch, shape, mac_per_inst, tflops, official, ratio = parts
        precision = precision_cell.replace("*", "") or current_precision
        current_precision = precision
        rows.append(
            {
                "precision": precision,
                "warp_num": int(warp_num),
                "launch": launch,
                "shape": normalize_shape_label(shape),
                "mac_per_inst": int(mac_per_inst),
                "tflops": float(tflops),
                "official_tflops": float(official),
                "peak_ratio": float(ratio.rstrip("%")),
            }
        )
    return sorted(rows, key=sort_key)


def parse_kv_log(path):
    result = {}
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        if "=" in line and not line.startswith("=="):
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    if "precision" not in result or "shape" not in result or "launch" not in result:
        return None
    try:
        return {
            "precision": result["precision"],
            "shape": normalize_shape_label(result["shape"]),
            "launch": result["launch"],
            "warp_num": int(result["warps_per_block"]),
            "cycles": int(result["cycles"]),
            "tflops": float(result["thor_tflops"]),
            "macs_per_cycle_per_active_block": float(result["macs_per_cycle_per_active_block"]),
            "log_path": str(path),
            "has_ncu_metrics": (
                "No metrics to collect found in sections." not in text
                and "ERR_NVGPUCTRPERM" not in text
            ),
        }
    except (KeyError, ValueError):
        return None


def parse_ncu_logs(ncu_dir):
    rows = []
    if not ncu_dir.exists():
        return rows
    for path in sorted(ncu_dir.glob("tcgen05_*_benchmark.log")):
        if not CURRENT_LOG_RE.match(path.name):
            continue
        parsed = parse_kv_log(path)
        if parsed:
            rows.append(parsed)
    return sorted(rows, key=sort_key)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def chart_title(text, width):
    return f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" class="title">{escape(text)}</text>'


def grouped_bar_svg(path, rows, value_key, title, y_label, value_suffix="", max_value=None):
    width = 1400
    height = 640
    margin_left = 78
    margin_right = 36
    margin_top = 78
    margin_bottom = 130
    panel_gap = 70
    plot_w = (width - margin_left - margin_right - panel_gap) / 2
    plot_h = height - margin_top - margin_bottom

    def y(value, panel_max_value):
        return margin_top + plot_h - (value / panel_max_value) * plot_h

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width, height, width, height),
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2933; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".axis { stroke: #46515c; stroke-width: 1; }",
        ".grid { stroke: #d7dde3; stroke-width: 1; }",
        ".tick { font-size: 12px; fill: #52606d; }",
        ".label { font-size: 11px; fill: #334e68; }",
        ".value { font-size: 10px; fill: #102a43; }",
        ".panel-title { font-size: 14px; font-weight: 700; fill: #243b53; }",
        "</style>",
        chart_title(title, width),
    ]
    elements.append(
        f'<text x="22" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 22 {margin_top + plot_h / 2:.1f})" '
        f'text-anchor="middle" class="label">{escape(y_label)}</text>'
    )

    for panel_idx, (launch, panel_title) in enumerate(LAUNCH_PANELS):
        panel_x = margin_left + panel_idx * (plot_w + panel_gap)
        panel_rows = [row for row in rows if row["launch"] == launch]
        elements.append(
            f'<text x="{panel_x + plot_w / 2:.1f}" y="{margin_top - 20}" text-anchor="middle" '
            f'class="panel-title">{escape(panel_title)}</text>'
        )
        if not panel_rows:
            elements.append(
                f'<text x="{panel_x + plot_w / 2:.1f}" y="{margin_top + plot_h / 2:.1f}" text-anchor="middle" class="tick">No data</text>'
            )
            continue
        panel_max_value = max_value or max(row[value_key] for row in panel_rows) * 1.15
        y_ticks = [
            0,
            panel_max_value * 0.25,
            panel_max_value * 0.5,
            panel_max_value * 0.75,
            panel_max_value,
        ]
        for tick in y_ticks:
            yy = y(tick, panel_max_value)
            elements.append(f'<line x1="{panel_x:.1f}" y1="{yy:.1f}" x2="{panel_x + plot_w:.1f}" y2="{yy:.1f}" class="grid"/>')
            elements.append(f'<text x="{panel_x - 10:.1f}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{tick:.0f}</text>')
        elements.append(f'<line x1="{panel_x:.1f}" y1="{margin_top}" x2="{panel_x:.1f}" y2="{margin_top + plot_h}" class="axis"/>')
        elements.append(f'<line x1="{panel_x:.1f}" y1="{margin_top + plot_h}" x2="{panel_x + plot_w:.1f}" y2="{margin_top + plot_h}" class="axis"/>')
        bar_gap = 7
        bar_w = max(18, (plot_w - bar_gap * (len(panel_rows) - 1)) / len(panel_rows))
        for idx, row in enumerate(panel_rows):
            x = panel_x + idx * (bar_w + bar_gap)
            val = row[value_key]
            yy = y(val, panel_max_value)
            h = margin_top + plot_h - yy
            color = COLORS.get(row["precision"], "#6b7280")
            label = f'{row["precision"]} {row["shape"]}'
            elements.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
            elements.append(f'<text x="{x + bar_w / 2:.1f}" y="{yy - 5:.1f}" text-anchor="middle" class="value">{val:.1f}{value_suffix}</text>')
            elements.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{margin_top + plot_h + 14}" text-anchor="end" '
                f'transform="rotate(-50 {x + bar_w / 2:.1f} {margin_top + plot_h + 14})" class="label">{escape(label)}</text>'
            )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def comparison_svg(path, report_rows, ncu_rows, launch, panel_title):
    ncu_by_key = {(row["precision"], row["launch"], row["shape"]): row for row in ncu_rows}
    rows = []
    for row in report_rows:
        if row["launch"] != launch:
            continue
        ncu = ncu_by_key.get((row["precision"], row["launch"], row["shape"]))
        if ncu:
            rows.append((row, ncu))
    if not rows:
        return False

    width = 1200
    height = 650
    margin_left = 78
    margin_right = 36
    margin_top = 98
    margin_bottom = 140
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def y(value, panel_max_value):
        return margin_top + plot_h - (value / panel_max_value) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text { font-family: Arial, sans-serif; fill: #1f2933; }.title { font-size: 22px; font-weight: 700; }.axis { stroke: #46515c; stroke-width: 1; }.grid { stroke: #d7dde3; stroke-width: 1; }.tick { font-size: 12px; fill: #52606d; }.label { font-size: 11px; fill: #334e68; }.value { font-size: 9px; fill: #102a43; }.panel-title { font-size: 14px; font-weight: 700; fill: #243b53; }</style>",
        chart_title(f"{panel_title} TFLOP/s: benchmark vs ncu-run vs theoretical peak", width),
    ]
    legend_y = 48
    elements.append(f'<rect x="{width - 440}" y="{legend_y}" width="14" height="14" fill="#2878b5"/><text x="{width - 420}" y="{legend_y + 12}" class="tick">base report</text>')
    elements.append(f'<rect x="{width - 305}" y="{legend_y}" width="14" height="14" fill="#c75d2c"/><text x="{width - 285}" y="{legend_y + 12}" class="tick">ncu run</text>')
    elements.append(f'<rect x="{width - 190}" y="{legend_y}" width="14" height="14" fill="#5b6472"/><text x="{width - 170}" y="{legend_y + 12}" class="tick">theoretical peak</text>')
    elements.append(
        f'<text x="22" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 22 {margin_top + plot_h / 2:.1f})" '
        f'text-anchor="middle" class="label">TFLOP/s</text>'
    )

    panel_x = margin_left
    panel_max_value = max(max(base["tflops"], ncu["tflops"], base["official_tflops"]) for base, ncu in rows) * 1.15
    y_ticks = [
        0,
        panel_max_value * 0.25,
        panel_max_value * 0.5,
        panel_max_value * 0.75,
        panel_max_value,
    ]
    for tick in y_ticks:
        yy = y(tick, panel_max_value)
        elements.append(f'<line x1="{panel_x:.1f}" y1="{yy:.1f}" x2="{panel_x + plot_w:.1f}" y2="{yy:.1f}" class="grid"/>')
        elements.append(f'<text x="{panel_x - 10:.1f}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{tick:.0f}</text>')
    elements.append(f'<line x1="{panel_x:.1f}" y1="{margin_top}" x2="{panel_x:.1f}" y2="{margin_top + plot_h}" class="axis"/>')
    elements.append(f'<line x1="{panel_x:.1f}" y1="{margin_top + plot_h}" x2="{panel_x + plot_w:.1f}" y2="{margin_top + plot_h}" class="axis"/>')

    group_gap = 12
    group_w = (plot_w - group_gap * (len(rows) - 1)) / len(rows)
    bar_w = max(12, (group_w - 10) / 3)
    for idx, (base, ncu) in enumerate(rows):
        x0 = panel_x + idx * (group_w + group_gap)
        bars = [
            (0, base["tflops"], "#2878b5"),
            (bar_w + 5, ncu["tflops"], "#c75d2c"),
            ((bar_w + 5) * 2, base["official_tflops"], "#5b6472"),
        ]
        for offset, val, color in bars:
            yy = y(val, panel_max_value)
            x = x0 + offset
            elements.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{margin_top + plot_h - yy:.1f}" fill="{color}"/>')
            elements.append(f'<text x="{x + bar_w / 2:.1f}" y="{yy - 5:.1f}" text-anchor="middle" class="value">{val:.1f}</text>')
        label = f'{base["precision"]} {base["shape"]}'
        elements.append(
            f'<text x="{x0 + group_w / 2:.1f}" y="{margin_top + plot_h + 14}" text-anchor="end" '
            f'transform="rotate(-45 {x0 + group_w / 2:.1f} {margin_top + plot_h + 14})" class="label">{escape(label)}</text>'
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot Thor tcgen05 benchmark and ncu results.")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--ncu-dir", type=Path, default=NCU_DIR)
    parser.add_argument("--out-dir", type=Path, default=PLOTS_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_rows = parse_report(args.report)
    ncu_rows = parse_ncu_logs(args.ncu_dir)

    write_csv(
        args.out_dir / "benchmark_results.csv",
        report_rows,
        ["precision", "warp_num", "launch", "shape", "mac_per_inst", "tflops", "official_tflops", "peak_ratio"],
    )
    grouped_bar_svg(args.out_dir / "benchmark_tflops.svg", report_rows, "tflops", "Benchmark Effective TFLOP/s", "TFLOP/s")
    grouped_bar_svg(args.out_dir / "benchmark_peak_ratio.svg", report_rows, "peak_ratio", "Benchmark Peak Ratio", "Peak ratio (%)", "%", 100)

    if ncu_rows:
        write_csv(
            args.out_dir / "ncu_results.csv",
            ncu_rows,
            ["precision", "warp_num", "launch", "shape", "cycles", "tflops", "macs_per_cycle_per_active_block", "has_ncu_metrics", "log_path"],
        )
        grouped_bar_svg(args.out_dir / "ncu_tflops.svg", ncu_rows, "tflops", "ncu-run Effective TFLOP/s", "TFLOP/s")
        comparison_svg(args.out_dir / "benchmark_vs_ncu_tflops_singlewarp.svg", report_rows, ncu_rows, "SingleWarpBlock", "SingleWarp")
        comparison_svg(args.out_dir / "benchmark_vs_ncu_tflops_fullsm4warp.svg", report_rows, ncu_rows, "FullSM4WarpBlock", "4Warp FullSM")
        old_combined = args.out_dir / "benchmark_vs_ncu_tflops.svg"
        if old_combined.exists():
            old_combined.unlink()

    print(f"Wrote plots and CSV files to {args.out_dir}")
    print(f"Parsed benchmark rows: {len(report_rows)}")
    print(f"Parsed ncu log rows: {len(ncu_rows)}")


if __name__ == "__main__":
    main()
