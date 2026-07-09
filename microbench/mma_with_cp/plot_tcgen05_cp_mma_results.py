#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "分析报告.txt"
PLOTS_DIR = ROOT / "plots"

PRECISION_ORDER = {"FP4": 0, "FP8": 1, "BF16": 2}
SHAPE_ORDER = {"M128N256": 0, "M128N128": 1, "M128N64": 2}
CASE_ORDER = {
    "SS MMA-only": 0,
    "TS MMA-only": 1,
    "TS CP+MMA Serial A1": 2,
    "TS CP+MMA Overlap A2": 3,
}
COLORS = {
    "SS MMA-only": "#2878b5",
    "TS MMA-only": "#c75d2c",
    "Serial A1": "#6d5bd0",
    "Overlap A2": "#2b8c5a",
    "cp": "#4b5563",
}


def sort_key(row):
    return (
        PRECISION_ORDER.get(row.get("precision"), 99),
        SHAPE_ORDER.get(row.get("shape"), 99),
    )


def numeric(value):
    text = str(value).strip().replace(",", "")
    text = text.removesuffix("%").removesuffix("x")
    return float(text)


def int_numeric(value):
    return int(numeric(value))


def parse_markdown_table(lines, heading):
    for idx, line in enumerate(lines):
        if line.strip() != heading:
            continue
        table_lines = []
        for candidate in lines[idx + 1:]:
            stripped = candidate.strip()
            if not stripped:
                if table_lines:
                    break
                continue
            if stripped.startswith("|"):
                table_lines.append(stripped)
                continue
            if table_lines:
                break
        if not table_lines:
            raise ValueError(f"section has no markdown table: {heading}")
        rows = []
        for table_line in table_lines:
            cells = [cell.strip() for cell in table_line.strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                continue
            rows.append(cells)
        if len(rows) < 2:
            raise ValueError(f"section table has no data rows: {heading}")
        return rows[0], rows[1:]
    raise ValueError(f"missing report section: {heading}")


def parse_report(path):
    lines = path.read_text(encoding="utf-8").splitlines()

    _, mma_table = parse_markdown_table(lines, "MMA-only TFLOP/s 与 Peak Ratio")
    mma_rows = []
    for row in mma_table:
        if len(row) != 7:
            continue
        precision, shape, k_value, ss_tflops, ts_tflops, ss_peak, ts_peak = row
        mma_rows.append(
            {
                "precision": precision,
                "shape": shape,
                "k": int_numeric(k_value),
                "ss_mma_only_tflops": numeric(ss_tflops),
                "ts_mma_only_tflops": numeric(ts_tflops),
                "ss_peak_ratio": numeric(ss_peak),
                "ts_peak_ratio": numeric(ts_peak),
            }
        )

    _, cp_table = parse_markdown_table(lines, "tcgen05.cp-only")
    cp_rows = []
    for row in cp_table:
        if len(row) != 8:
            continue
        precision, shape, cp_suffix, effective_bytes, cp_count, cycles, bytes_per_cycle, cycles_per_cp = row
        cp_rows.append(
            {
                "precision": precision,
                "shape": shape,
                "cp_suffix": cp_suffix,
                "effective_bytes_per_cp": int_numeric(effective_bytes),
                "cp_instruction_count": int_numeric(cp_count),
                "elapsed_cycles": int_numeric(cycles),
                "bytes_per_cycle": numeric(bytes_per_cycle),
                "cycles_per_cp": numeric(cycles_per_cp),
            }
        )

    _, pipeline_table = parse_markdown_table(lines, "CP+MMA pipeline")
    pipeline_rows = []
    for row in pipeline_table:
        if len(row) != 7:
            continue
        precision, shape, serial_tflops, serial_cycles, overlap_tflops, overlap_cycles, gain = row
        pipeline_rows.append(
            {
                "precision": precision,
                "shape": shape,
                "serial_a1_tflops": numeric(serial_tflops),
                "serial_a1_cycles_per_tile": numeric(serial_cycles),
                "overlap_a2_tflops": numeric(overlap_tflops),
                "overlap_a2_cycles_per_tile": numeric(overlap_cycles),
                "overlap_gain": numeric(gain),
            }
        )

    speedup_header, speedup_table = parse_markdown_table(lines, "图 5 speedup 数据源")
    speedup_rows = []
    for row in speedup_table:
        if len(row) != len(speedup_header):
            continue
        case = row[0]
        for column, value in zip(speedup_header[1:], row[1:]):
            speedup_rows.append(
                {
                    "case": case,
                    "metric": column,
                    "speedup_vs_ss_mma_only": numeric(value),
                }
            )

    return {
        "mma_only": sorted(mma_rows, key=sort_key),
        "cp_only": sorted(cp_rows, key=sort_key),
        "pipeline": sorted(pipeline_rows, key=sort_key),
        "speedup": sorted(
            speedup_rows,
            key=lambda row: (CASE_ORDER.get(row["case"], 99), speedup_header.index(row["metric"]) if row["metric"] in speedup_header else 99),
        ),
        "speedup_metrics": speedup_header[1:],
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def chart_title(text, width):
    return f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" class="title">{escape(text)}</text>'


def base_svg(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2933; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".axis { stroke: #46515c; stroke-width: 1; }",
        ".grid { stroke: #d7dde3; stroke-width: 1; }",
        ".tick { font-size: 12px; fill: #52606d; }",
        ".label { font-size: 11px; fill: #334e68; }",
        ".value { font-size: 10px; fill: #102a43; }",
        ".legend { font-size: 12px; fill: #334e68; }",
        "</style>",
        chart_title(title, width),
    ]


def y_ticks(max_value):
    if max_value <= 0:
        return [0, 1]
    return [0, max_value * 0.25, max_value * 0.5, max_value * 0.75, max_value]


def format_axis_value(value):
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def grouped_bar_svg(path, rows, series, title, y_label, value_suffix="", max_value=None):
    width = 1480
    height = 680
    margin_left = 82
    margin_right = 36
    margin_top = 92
    margin_bottom = 150
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    if not rows:
        path.write_text("\n".join(base_svg(width, height, title) + ['<text x="740" y="340" text-anchor="middle" class="tick">No data</text>', "</svg>"]) + "\n", encoding="utf-8")
        return
    panel_max = max_value or max(row[key] for row in rows for key, _, _ in series) * 1.15

    def y(value):
        return margin_top + plot_h - (value / panel_max) * plot_h

    elements = base_svg(width, height, title)
    elements.append(
        f'<text x="22" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 22 {margin_top + plot_h / 2:.1f})" '
        f'text-anchor="middle" class="label">{escape(y_label)}</text>'
    )

    legend_x = width - margin_right - 280
    legend_y = 48
    for idx, (_, label, color) in enumerate(series):
        x = legend_x + idx * 138
        elements.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        elements.append(f'<text x="{x + 20}" y="{legend_y + 12}" class="legend">{escape(label)}</text>')

    for tick in y_ticks(panel_max):
        yy = y(tick)
        elements.append(f'<line x1="{margin_left}" y1="{yy:.1f}" x2="{margin_left + plot_w}" y2="{yy:.1f}" class="grid"/>')
        elements.append(f'<text x="{margin_left - 10}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{format_axis_value(tick)}</text>')
    elements.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    elements.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis"/>')

    group_gap = 14
    group_w = (plot_w - group_gap * (len(rows) - 1)) / len(rows)
    bar_gap = 5
    bar_w = max(12, (group_w - bar_gap * (len(series) - 1)) / len(series))
    for idx, row in enumerate(rows):
        x0 = margin_left + idx * (group_w + group_gap)
        for series_idx, (key, _, color) in enumerate(series):
            val = row[key]
            x = x0 + series_idx * (bar_w + bar_gap)
            yy = y(val)
            elements.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{margin_top + plot_h - yy:.1f}" fill="{color}"/>')
            elements.append(f'<text x="{x + bar_w / 2:.1f}" y="{yy - 5:.1f}" text-anchor="middle" class="value">{val:.1f}{value_suffix}</text>')
        label = f'{row["precision"]} {row["shape"]}'
        label_x = x0 + group_w / 2
        elements.append(
            f'<text x="{label_x:.1f}" y="{margin_top + plot_h + 16}" text-anchor="end" '
            f'transform="rotate(-45 {label_x:.1f} {margin_top + plot_h + 16})" class="label">{escape(label)}</text>'
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def single_bar_svg(path, rows, value_key, title, y_label, color, value_suffix="", max_value=None):
    grouped_bar_svg(path, rows, [(value_key, y_label, color)], title, y_label, value_suffix, max_value)


def heat_color(value):
    clamped = max(0.0, min(1.0, (value - 0.85) / 0.35))
    start = (235, 245, 240)
    end = (32, 120, 84)
    rgb = tuple(round(start[i] + (end[i] - start[i]) * clamped) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def speedup_heatmap_svg(path, rows, metrics):
    cases = sorted({row["case"] for row in rows}, key=lambda case: CASE_ORDER.get(case, 99))
    by_key = {(row["case"], row["metric"]): row["speedup_vs_ss_mma_only"] for row in rows}
    width = 1280
    height = 470
    margin_left = 190
    margin_right = 34
    margin_top = 104
    margin_bottom = 86
    cell_w = (width - margin_left - margin_right) / len(metrics)
    cell_h = (height - margin_top - margin_bottom) / len(cases)
    elements = base_svg(width, height, "Speedup vs SS MMA-only")

    for col_idx, metric in enumerate(metrics):
        x = margin_left + col_idx * cell_w + cell_w / 2
        elements.append(
            f'<text x="{x:.1f}" y="{margin_top - 12}" text-anchor="end" '
            f'transform="rotate(-35 {x:.1f} {margin_top - 12})" class="label">{escape(metric)}</text>'
        )
    for row_idx, case in enumerate(cases):
        y = margin_top + row_idx * cell_h
        elements.append(f'<text x="{margin_left - 12}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="end" class="label">{escape(case)}</text>')
        for col_idx, metric in enumerate(metrics):
            x = margin_left + col_idx * cell_w
            value = by_key.get((case, metric), 0.0)
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" '
                f'fill="{heat_color(value)}" stroke="#ffffff" stroke-width="1"/>'
            )
            elements.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="middle" class="value">{value:.2f}x</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_outputs(out_dir, parsed):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "mma_only_results.csv",
        parsed["mma_only"],
        ["precision", "shape", "k", "ss_mma_only_tflops", "ts_mma_only_tflops", "ss_peak_ratio", "ts_peak_ratio"],
    )
    write_csv(
        out_dir / "cp_only_results.csv",
        parsed["cp_only"],
        [
            "precision",
            "shape",
            "cp_suffix",
            "effective_bytes_per_cp",
            "cp_instruction_count",
            "elapsed_cycles",
            "bytes_per_cycle",
            "cycles_per_cp",
        ],
    )
    write_csv(
        out_dir / "pipeline_results.csv",
        parsed["pipeline"],
        [
            "precision",
            "shape",
            "serial_a1_tflops",
            "serial_a1_cycles_per_tile",
            "overlap_a2_tflops",
            "overlap_a2_cycles_per_tile",
            "overlap_gain",
        ],
    )
    write_csv(out_dir / "speedup_results.csv", parsed["speedup"], ["case", "metric", "speedup_vs_ss_mma_only"])

    grouped_bar_svg(
        out_dir / "mma_only_tflops.svg",
        parsed["mma_only"],
        [
            ("ss_mma_only_tflops", "SS MMA-only", COLORS["SS MMA-only"]),
            ("ts_mma_only_tflops", "TS MMA-only", COLORS["TS MMA-only"]),
        ],
        "MMA-only Effective TFLOP/s",
        "TFLOP/s",
    )
    grouped_bar_svg(
        out_dir / "mma_only_peak_ratio.svg",
        parsed["mma_only"],
        [
            ("ss_peak_ratio", "SS MMA-only", COLORS["SS MMA-only"]),
            ("ts_peak_ratio", "TS MMA-only", COLORS["TS MMA-only"]),
        ],
        "MMA-only Peak Ratio",
        "Peak ratio (%)",
        "%",
        100,
    )
    single_bar_svg(
        out_dir / "cp_only_bytes_per_cycle.svg",
        parsed["cp_only"],
        "bytes_per_cycle",
        "tcgen05.cp-only Effective Bytes/Cycle",
        "bytes/cycle",
        COLORS["cp"],
    )
    single_bar_svg(
        out_dir / "cp_only_cycles_per_cp.svg",
        parsed["cp_only"],
        "cycles_per_cp",
        "tcgen05.cp-only Cycles/Instruction",
        "cycles/cp",
        COLORS["cp"],
    )
    grouped_bar_svg(
        out_dir / "pipeline_tflops.svg",
        parsed["pipeline"],
        [
            ("serial_a1_tflops", "Serial A1", COLORS["Serial A1"]),
            ("overlap_a2_tflops", "Overlap A2", COLORS["Overlap A2"]),
        ],
        "CP+MMA Pipeline Effective TFLOP/s",
        "TFLOP/s",
    )
    grouped_bar_svg(
        out_dir / "pipeline_cycles_per_tile.svg",
        parsed["pipeline"],
        [
            ("serial_a1_cycles_per_tile", "Serial A1", COLORS["Serial A1"]),
            ("overlap_a2_cycles_per_tile", "Overlap A2", COLORS["Overlap A2"]),
        ],
        "CP+MMA Pipeline Cycles/Tile",
        "cycles/tile",
    )
    single_bar_svg(
        out_dir / "pipeline_overlap_gain.svg",
        parsed["pipeline"],
        "overlap_gain",
        "CP+MMA Overlap Gain",
        "speedup",
        COLORS["Overlap A2"],
        "x",
    )
    speedup_heatmap_svg(out_dir / "speedup_heatmap.svg", parsed["speedup"], parsed["speedup_metrics"])


def main():
    parser = argparse.ArgumentParser(description="Plot Thor tcgen05 cp+mma benchmark report.")
    parser.add_argument("--report", type=Path, default=REPORT_PATH, help="benchmark markdown report")
    parser.add_argument("--out-dir", type=Path, default=PLOTS_DIR, help="directory for CSV and SVG outputs")
    args = parser.parse_args()

    if not args.report.exists():
        raise SystemExit(f"report not found: {args.report}")

    parsed = parse_report(args.report)
    write_outputs(args.out_dir, parsed)
    print(f"Wrote plots and CSV files to {args.out_dir}")
    print(f"Parsed rows: MMA-only={len(parsed['mma_only'])}, cp-only={len(parsed['cp_only'])}, pipeline={len(parsed['pipeline'])}, speedup={len(parsed['speedup'])}")


if __name__ == "__main__":
    main()
