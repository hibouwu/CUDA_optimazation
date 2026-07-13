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
    "SS MMA Mainloop K2": 2,
    "SS MMA Mainloop K4": 3,
    "SS MMA Mainloop K8": 4,
    "SS MMA Mainloop K16": 5,
    "TS CP+MMA Mainloop A2 K2": 6,
    "TS CP+MMA Mainloop A2 K4": 7,
    "TS CP+MMA Mainloop A2 K8": 8,
    "TS CP+MMA Mainloop A2 K16": 9,
    "TS CP+MMA Serial A1": 10,
    "TS CP+MMA Overlap A2": 11,
    "TS CP+MMA Warp Split A2": 12,
}
COLORS = {
    "SS MMA-only": "#2878b5",
    "TS MMA-only": "#c75d2c",
    "SS MMA Mainloop K2": "#8fbc8f",
    "SS MMA Mainloop K4": "#6b8e23",
    "SS MMA Mainloop K8": "#2e8b57",
    "SS MMA Mainloop K16": "#0f5d3b",
    "TS CP+MMA Mainloop A2 K2": "#9fb7d9",
    "TS CP+MMA Mainloop A2 K4": "#6f92c6",
    "TS CP+MMA Mainloop A2 K8": "#416fae",
    "TS CP+MMA Mainloop A2 K16": "#244f8f",
    "Serial A1": "#6d5bd0",
    "Overlap A2": "#2b8c5a",
    "Warp Split A2": "#b7791f",
    "cp": "#4b5563",
}


def sort_key(row):
    return (
        PRECISION_ORDER.get(row.get("precision"), 99),
        SHAPE_ORDER.get(row.get("shape"), 99),
        row.get("k_blocks", 0),
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

    for mma_heading in (
        "MMA-only forced-wait per-instruction completion TFLOP/s 与 Peak Ratio",
        "MMA-only per-tile completion TFLOP/s 与 Peak Ratio",
    ):
        try:
            _, mma_table = parse_markdown_table(lines, mma_heading)
            break
        except ValueError:
            mma_table = None
    if mma_table is None:
        raise ValueError("missing report section: MMA-only forced-wait/per-tile completion TFLOP/s 与 Peak Ratio")
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

    _, mainloop_table = parse_markdown_table(lines, "CUTLASS-style SS mainloop K-block sweep throughput")
    mainloop_rows = []
    for row in mainloop_table:
        if len(row) != 8:
            continue
        precision, shape, k_blocks, k_tile, tflops, peak_ratio, cycles_per_cta_k_tile, cycles_per_mma = row
        mainloop_rows.append(
            {
                "precision": precision,
                "shape": shape,
                "k_blocks": int_numeric(k_blocks),
                "k_tile": int_numeric(k_tile),
                "ss_mainloop_tflops": numeric(tflops),
                "ss_mainloop_peak_ratio": numeric(peak_ratio),
                "cycles_per_cta_k_tile": numeric(cycles_per_cta_k_tile),
                "cycles_per_mma": numeric(cycles_per_mma),
            }
        )

    try:
        _, ts_mainloop_table = parse_markdown_table(lines, "TS CP+MMA Mainloop A2 K-group sweep throughput")
    except ValueError:
        ts_mainloop_table = []
    ts_mainloop_rows = []
    for row in ts_mainloop_table:
        if len(row) != 9:
            continue
        precision, shape, k_blocks, k_tile, tflops, peak_ratio, cycles_per_cta_k_tile, cycles_per_mma, cp_inst_per_k_tile = row
        ts_mainloop_rows.append(
            {
                "precision": precision,
                "shape": shape,
                "k_blocks": int_numeric(k_blocks),
                "k_tile": int_numeric(k_tile),
                "ts_cp_mma_mainloop_a2_tflops": numeric(tflops),
                "ts_cp_mma_mainloop_a2_peak_ratio": numeric(peak_ratio),
                "cycles_per_cta_k_tile": numeric(cycles_per_cta_k_tile),
                "cycles_per_mma": numeric(cycles_per_mma),
                "cp_inst_per_k_tile": int_numeric(cp_inst_per_k_tile),
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
        if len(row) != 10:
            continue
        (
            precision,
            shape,
            serial_tflops,
            serial_cycles,
            overlap_tflops,
            overlap_cycles,
            gain,
            warp_split_tflops,
            warp_split_cycles,
            warp_split_gain,
        ) = row
        pipeline_rows.append(
            {
                "precision": precision,
                "shape": shape,
                "serial_a1_tflops": numeric(serial_tflops),
                "serial_a1_cycles_per_tile": numeric(serial_cycles),
                "overlap_a2_tflops": numeric(overlap_tflops),
                "overlap_a2_cycles_per_tile": numeric(overlap_cycles),
                "overlap_gain": numeric(gain),
                "warp_split_a2_tflops": numeric(warp_split_tflops),
                "warp_split_a2_cycles_per_tile": numeric(warp_split_cycles),
                "warp_split_gain": numeric(warp_split_gain),
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

    def metric_name(precision, shape):
        return f"{precision}-{shape.replace('M128', '')}"

    tflops_rows = []

    def add_tflops(case, precision, shape, value):
        tflops_rows.append(
            {
                "case": case,
                "metric": metric_name(precision, shape),
                "tflops": value,
            }
        )

    for row in mma_rows:
        add_tflops("SS MMA-only", row["precision"], row["shape"], row["ss_mma_only_tflops"])
        add_tflops("TS MMA-only", row["precision"], row["shape"], row["ts_mma_only_tflops"])
    for row in mainloop_rows:
        add_tflops(
            f"SS MMA Mainloop K{row['k_blocks']}",
            row["precision"],
            row["shape"],
            row["ss_mainloop_tflops"],
        )
    for row in ts_mainloop_rows:
        add_tflops(
            f"TS CP+MMA Mainloop A2 K{row['k_blocks']}",
            row["precision"],
            row["shape"],
            row["ts_cp_mma_mainloop_a2_tflops"],
        )
    for row in pipeline_rows:
        add_tflops("TS CP+MMA Serial A1", row["precision"], row["shape"], row["serial_a1_tflops"])
        add_tflops("TS CP+MMA Overlap A2", row["precision"], row["shape"], row["overlap_a2_tflops"])
        add_tflops("TS CP+MMA Warp Split A2", row["precision"], row["shape"], row["warp_split_a2_tflops"])

    return {
        "mma_only": sorted(mma_rows, key=sort_key),
        "mma_mainloop_sweep": sorted(mainloop_rows, key=sort_key),
        "ts_cp_mma_mainloop_a2_sweep": sorted(ts_mainloop_rows, key=sort_key),
        "cp_only": sorted(cp_rows, key=sort_key),
        "pipeline": sorted(pipeline_rows, key=sort_key),
        "speedup": sorted(
            speedup_rows,
            key=lambda row: (CASE_ORDER.get(row["case"], 99), speedup_header.index(row["metric"]) if row["metric"] in speedup_header else 99),
        ),
        "tflops": sorted(
            tflops_rows,
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
    width = max(920, 320 + 210 * len(rows))
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
    panel_max = max_value or max(row[item[0]] for row in rows for item in series) * 1.15

    def y(value):
        return margin_top + plot_h - (value / panel_max) * plot_h

    elements = base_svg(width, height, title)
    elements.append(
        f'<text x="22" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 22 {margin_top + plot_h / 2:.1f})" '
        f'text-anchor="middle" class="label">{escape(y_label)}</text>'
    )

    legend_x = margin_left
    legend_y = 50
    legend_item_w = 160
    legend_per_row = max(1, int((width - margin_left - margin_right) // legend_item_w))
    for idx, item in enumerate(series):
        _, label, color = item[:3]
        x = legend_x + (idx % legend_per_row) * legend_item_w
        y0 = legend_y + (idx // legend_per_row) * 18
        elements.append(f'<rect x="{x}" y="{y0}" width="14" height="14" fill="{color}"/>')
        elements.append(f'<text x="{x + 20}" y="{y0 + 12}" class="legend">{escape(label)}</text>')

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
        for series_idx, item in enumerate(series):
            key, _, color = item[:3]
            extra_key = item[3] if len(item) > 3 else None
            extra_suffix = item[4] if len(item) > 4 else ""
            val = row[key]
            x = x0 + series_idx * (bar_w + bar_gap)
            yy = y(val)
            elements.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{margin_top + plot_h - yy:.1f}" fill="{color}"/>')
            if extra_key:
                elements.append(f'<text x="{x + bar_w / 2:.1f}" y="{yy - 17:.1f}" text-anchor="middle" class="value">{val:.1f}{value_suffix}</text>')
                elements.append(f'<text x="{x + bar_w / 2:.1f}" y="{yy - 5:.1f}" text-anchor="middle" class="value">{row[extra_key]:.1f}{extra_suffix}</text>')
            else:
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


def tflops_heat_color(value, max_value):
    scale = 0.0 if max_value <= 0 else max(0.0, min(1.0, value / max_value))
    start = (241, 246, 250)
    mid = (100, 151, 177)
    end = (17, 82, 121)
    if scale < 0.5:
        local = scale / 0.5
        rgb = tuple(round(start[i] + (mid[i] - start[i]) * local) for i in range(3))
    else:
        local = (scale - 0.5) / 0.5
        rgb = tuple(round(mid[i] + (end[i] - mid[i]) * local) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def speedup_heatmap_svg(path, rows, metrics):
    cases = sorted({row["case"] for row in rows}, key=lambda case: CASE_ORDER.get(case, 99))
    by_key = {(row["case"], row["metric"]): row["speedup_vs_ss_mma_only"] for row in rows}
    width = max(920, 270 + 126 * len(metrics))
    height = max(540, 220 + 58 * len(cases))
    margin_left = 230
    margin_right = 44
    margin_top = 148
    margin_bottom = 70
    cell_w = (width - margin_left - margin_right) / len(metrics)
    cell_h = (height - margin_top - margin_bottom) / len(cases)
    elements = base_svg(width, height, "Speedup vs SS MMA-only")

    groups = []
    for idx, metric in enumerate(metrics):
        precision, _, shape = metric.partition("-")
        if not groups or groups[-1]["precision"] != precision:
            groups.append({"precision": precision, "start": idx, "end": idx, "shapes": []})
        groups[-1]["end"] = idx
        groups[-1]["shapes"].append(shape)

    for group in groups:
        x0 = margin_left + group["start"] * cell_w
        x1 = margin_left + (group["end"] + 1) * cell_w
        elements.append(
            f'<text x="{(x0 + x1) / 2:.1f}" y="{margin_top - 58}" text-anchor="middle" '
            f'class="legend">{escape(group["precision"])}</text>'
        )
        if group["start"] > 0:
            elements.append(
                f'<line x1="{x0:.1f}" y1="{margin_top - 72}" x2="{x0:.1f}" '
                f'y2="{margin_top + cell_h * len(cases)}" stroke="#c8d1dc" stroke-width="1.5"/>'
            )
    for col_idx, metric in enumerate(metrics):
        x = margin_left + col_idx * cell_w + cell_w / 2
        _, _, shape = metric.partition("-")
        elements.append(f'<text x="{x:.1f}" y="{margin_top - 30}" text-anchor="middle" class="label">{escape(shape)}</text>')
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


def tflops_heatmap_svg(path, rows, metrics):
    cases = sorted({row["case"] for row in rows}, key=lambda case: CASE_ORDER.get(case, 99))
    by_key = {(row["case"], row["metric"]): row["tflops"] for row in rows}
    max_value = max((row["tflops"] for row in rows), default=0.0)
    width = max(920, 270 + 126 * len(metrics))
    height = max(540, 220 + 58 * len(cases))
    margin_left = 230
    margin_right = 44
    margin_top = 148
    margin_bottom = 70
    cell_w = (width - margin_left - margin_right) / len(metrics)
    cell_h = (height - margin_top - margin_bottom) / len(cases)
    elements = base_svg(width, height, "TFLOP/s Heatmap")

    groups = []
    for idx, metric in enumerate(metrics):
        precision, _, shape = metric.partition("-")
        if not groups or groups[-1]["precision"] != precision:
            groups.append({"precision": precision, "start": idx, "end": idx})
        groups[-1]["end"] = idx

    for group in groups:
        x0 = margin_left + group["start"] * cell_w
        x1 = margin_left + (group["end"] + 1) * cell_w
        elements.append(
            f'<text x="{(x0 + x1) / 2:.1f}" y="{margin_top - 58}" text-anchor="middle" '
            f'class="legend">{escape(group["precision"])}</text>'
        )
        if group["start"] > 0:
            elements.append(
                f'<line x1="{x0:.1f}" y1="{margin_top - 72}" x2="{x0:.1f}" '
                f'y2="{margin_top + cell_h * len(cases)}" stroke="#c8d1dc" stroke-width="1.5"/>'
            )
    for col_idx, metric in enumerate(metrics):
        x = margin_left + col_idx * cell_w + cell_w / 2
        _, _, shape = metric.partition("-")
        elements.append(f'<text x="{x:.1f}" y="{margin_top - 30}" text-anchor="middle" class="label">{escape(shape)}</text>')
    for row_idx, case in enumerate(cases):
        y = margin_top + row_idx * cell_h
        elements.append(f'<text x="{margin_left - 12}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="end" class="label">{escape(case)}</text>')
        for col_idx, metric in enumerate(metrics):
            x = margin_left + col_idx * cell_w
            value = by_key.get((case, metric), 0.0)
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" '
                f'fill="{tflops_heat_color(value, max_value)}" stroke="#ffffff" stroke-width="1"/>'
            )
            elements.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="middle" class="value">{value:.1f}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def pivot_mainloop_rows(rows, ts_rows=None):
    by_key = {}
    for row in rows:
        key = (row["precision"], row["shape"])
        item = by_key.setdefault(key, {"precision": row["precision"], "shape": row["shape"]})
        k = row["k_blocks"]
        item[f"ss_k{k}_tflops"] = row["ss_mainloop_tflops"]
        item[f"ss_k{k}_peak_ratio"] = row["ss_mainloop_peak_ratio"]
        item[f"ss_k{k}_cycles_per_mma"] = row["cycles_per_mma"]
    for row in ts_rows or []:
        key = (row["precision"], row["shape"])
        item = by_key.setdefault(key, {"precision": row["precision"], "shape": row["shape"]})
        k = row["k_blocks"]
        item[f"ts_a2_k{k}_tflops"] = row["ts_cp_mma_mainloop_a2_tflops"]
        item[f"ts_a2_k{k}_peak_ratio"] = row["ts_cp_mma_mainloop_a2_peak_ratio"]
        item[f"ts_a2_k{k}_cycles_per_mma"] = row["cycles_per_mma"]
    return sorted(by_key.values(), key=sort_key)


def write_outputs(out_dir, parsed):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "mma_only_results.csv",
        parsed["mma_only"],
        ["precision", "shape", "k", "ss_mma_only_tflops", "ts_mma_only_tflops", "ss_peak_ratio", "ts_peak_ratio"],
    )
    write_csv(
        out_dir / "mma_mainloop_sweep_results.csv",
        parsed["mma_mainloop_sweep"],
        [
            "precision",
            "shape",
            "k_blocks",
            "k_tile",
            "ss_mainloop_tflops",
            "ss_mainloop_peak_ratio",
            "cycles_per_cta_k_tile",
            "cycles_per_mma",
        ],
    )
    write_csv(
        out_dir / "ts_cp_mma_mainloop_a2_sweep_results.csv",
        parsed["ts_cp_mma_mainloop_a2_sweep"],
        [
            "precision",
            "shape",
            "k_blocks",
            "k_tile",
            "ts_cp_mma_mainloop_a2_tflops",
            "ts_cp_mma_mainloop_a2_peak_ratio",
            "cycles_per_cta_k_tile",
            "cycles_per_mma",
            "cp_inst_per_k_tile",
        ],
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
            "warp_split_a2_tflops",
            "warp_split_a2_cycles_per_tile",
            "warp_split_gain",
        ],
    )
    write_csv(out_dir / "speedup_results.csv", parsed["speedup"], ["case", "metric", "speedup_vs_ss_mma_only"])
    write_csv(out_dir / "tflops_heatmap_results.csv", parsed["tflops"], ["case", "metric", "tflops"])

    for obsolete in [
        "mma_only_peak_ratio.svg",
        "mma_mainloop_k4_peak_ratio.svg",
        "mma_mainloop_k4_results.csv",
        "mma_mainloop_k4_tflops.svg",
        "mma_grouped_peak_ratio.svg",
        "mma_grouped_results.csv",
        "mma_grouped_tflops.svg",
        "cp_only_cycles_per_cp.svg",
        "pipeline_cycles_per_tile.svg",
        "pipeline_overlap_gain.svg",
        "pipeline_4warp_peak_ratio.svg",
        "pipeline_4warp_results.csv",
        "pipeline_4warp_tflops.svg",
    ]:
        (out_dir / obsolete).unlink(missing_ok=True)

    grouped_bar_svg(
        out_dir / "mma_only_tflops.svg",
        parsed["mma_only"],
        [
            ("ss_mma_only_tflops", "SS MMA-only", COLORS["SS MMA-only"], "ss_peak_ratio", "% peak"),
            ("ts_mma_only_tflops", "TS MMA-only", COLORS["TS MMA-only"], "ts_peak_ratio", "% peak"),
        ],
        "MMA-only Per-tile Completion TFLOP/s and Peak Ratio",
        "TFLOP/s",
    )
    mainloop_series = [
        ("ss_k2_tflops", "SS K2", COLORS["SS MMA Mainloop K2"]),
        ("ss_k4_tflops", "SS K4", COLORS["SS MMA Mainloop K4"]),
        ("ss_k8_tflops", "SS K8", COLORS["SS MMA Mainloop K8"]),
        ("ss_k16_tflops", "SS K16", COLORS["SS MMA Mainloop K16"]),
    ]
    if parsed["ts_cp_mma_mainloop_a2_sweep"]:
        mainloop_series.extend([
            ("ts_a2_k2_tflops", "TS A2 K2", COLORS["TS CP+MMA Mainloop A2 K2"]),
            ("ts_a2_k4_tflops", "TS A2 K4", COLORS["TS CP+MMA Mainloop A2 K4"]),
            ("ts_a2_k8_tflops", "TS A2 K8", COLORS["TS CP+MMA Mainloop A2 K8"]),
            ("ts_a2_k16_tflops", "TS A2 K16", COLORS["TS CP+MMA Mainloop A2 K16"]),
        ])
    grouped_bar_svg(
        out_dir / "mma_mainloop_sweep_tflops.svg",
        pivot_mainloop_rows(parsed["mma_mainloop_sweep"], parsed["ts_cp_mma_mainloop_a2_sweep"]),
        mainloop_series,
        "SS vs TS A2 Mainloop K-block Sweep TFLOP/s",
        "TFLOP/s",
    )
    grouped_bar_svg(
        out_dir / "cp_only_bytes_per_cycle.svg",
        parsed["cp_only"],
        [
            ("bytes_per_cycle", "bytes/cycle", COLORS["cp"], "cycles_per_cp", " cyc/cp"),
        ],
        "tcgen05.cp-only Effective Bytes/Cycle and Cycles/Instruction",
        "bytes/cycle",
    )
    grouped_bar_svg(
        out_dir / "pipeline_tflops.svg",
        parsed["pipeline"],
        [
            ("serial_a1_tflops", "Serial A1", COLORS["Serial A1"], "serial_a1_cycles_per_tile", " cyc"),
            ("overlap_a2_tflops", "Overlap A2", COLORS["Overlap A2"], "overlap_a2_cycles_per_tile", " cyc"),
            ("warp_split_a2_tflops", "Warp Split A2", COLORS["Warp Split A2"], "warp_split_a2_cycles_per_tile", " cyc"),
        ],
        "CP+MMA Pipeline TFLOP/s and Cycles/Tile",
        "TFLOP/s",
    )
    speedup_heatmap_svg(out_dir / "speedup_heatmap.svg", parsed["speedup"], parsed["speedup_metrics"])
    tflops_heatmap_svg(out_dir / "tflops_heatmap.svg", parsed["tflops"], parsed["speedup_metrics"])


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
    print(f"Parsed rows: MMA-only={len(parsed['mma_only'])}, MMA-mainloop-sweep={len(parsed['mma_mainloop_sweep'])}, TS-A2-mainloop-sweep={len(parsed['ts_cp_mma_mainloop_a2_sweep'])}, cp-only={len(parsed['cp_only'])}, pipeline={len(parsed['pipeline'])}, speedup={len(parsed['speedup'])}, tflops={len(parsed['tflops'])}")


if __name__ == "__main__":
    main()
