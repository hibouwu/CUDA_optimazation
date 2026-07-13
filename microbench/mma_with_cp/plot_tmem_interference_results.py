#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "tmem_interference_report.txt"
PLOTS_DIR = ROOT / "plots"

PRECISION_ORDER = {"FP4": 0, "BF16": 1}
GROUP_ORDER = {
    ("SS", "same"): 0,
    ("SS", "split"): 1,
    ("TS", "same"): 2,
    ("TS", "split"): 3,
}
GROUP_LABELS = {
    ("SS", "same"): "SS same",
    ("SS", "split"): "SS split",
    ("TS", "same"): "TS same",
    ("TS", "split"): "TS split",
}
NOISE_COLORS = {
    0: "#4b5563",
    1: "#0f5d3b",
    2: "#2878b5",
    4: "#b7791f",
    8: "#c75d2c",
}


def numeric(value):
    text = str(value).strip().replace(",", "")
    text = text.removesuffix("%").removesuffix("x")
    return float(text)


def parse_table(lines, heading):
    table_lines = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == heading:
            in_section = True
            continue
        if not in_section:
            continue
        if stripped.startswith("|"):
            table_lines.append(stripped)
            continue
        if table_lines:
            break
    if not table_lines:
        raise ValueError(f"missing table: {heading}")

    rows = []
    for table_line in table_lines:
        cells = [cell.strip() for cell in table_line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if len(rows) < 2:
        raise ValueError(f"empty table: {heading}")
    return rows[0], rows[1:]


def parse_throughput(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header, data = parse_table(lines, "Throughput and slowdown")
    required = [
        "Case",
        "Precision",
        "Shape",
        "Path",
        "Mode",
        "Noise cp/MMA",
        "TFLOP/s",
        "Peak Ratio",
        "cycles/CTA iter",
        "cycles/MMA",
        "cp inst",
        "bytes/cycle",
        "Slowdown vs control",
    ]
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(f"missing throughput columns: {missing}")

    rows = []
    for row in data:
        item = dict(zip(header, row))
        rows.append(
            {
                "case": item["Case"],
                "precision": item["Precision"],
                "shape": item["Shape"],
                "path": item["Path"],
                "mode": item["Mode"],
                "noise_cp_per_mma": int(numeric(item["Noise cp/MMA"])),
                "tflops": numeric(item["TFLOP/s"]),
                "tflops_min": numeric(item.get("TFLOP/s min", item["TFLOP/s"])),
                "tflops_max": numeric(item.get("TFLOP/s max", item["TFLOP/s"])),
                "peak_ratio": numeric(item["Peak Ratio"]),
                "cycles_per_cta_iter": numeric(item["cycles/CTA iter"]),
                "cycles_per_mma": numeric(item["cycles/MMA"]),
                "cp_inst": int(numeric(item["cp inst"])),
                "bytes_per_cycle": numeric(item["bytes/cycle"]),
                "slowdown_vs_control": numeric(item["Slowdown vs control"]),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            PRECISION_ORDER.get(item["precision"], 99),
            GROUP_ORDER.get((item["path"], item["mode"]), 99),
            item["noise_cp_per_mma"],
        ),
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "precision",
        "shape",
        "path",
        "mode",
        "noise_cp_per_mma",
        "tflops",
        "tflops_min",
        "tflops_max",
        "peak_ratio",
        "cycles_per_cta_iter",
        "cycles_per_mma",
        "cp_inst",
        "bytes_per_cycle",
        "slowdown_vs_control",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def base_svg(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2933; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { font-size: 14px; font-weight: 700; fill: #334e68; }",
        ".axis { stroke: #46515c; stroke-width: 1; }",
        ".grid { stroke: #d7dde3; stroke-width: 1; }",
        ".tick { font-size: 12px; fill: #52606d; }",
        ".label { font-size: 12px; fill: #334e68; }",
        ".legend { font-size: 12px; fill: #334e68; }",
        ".value { font-size: 10px; fill: #102a43; }",
        "</style>",
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" class="title">{escape(title)}</text>',
    ]


def axis_ticks(max_value, count=5):
    if max_value <= 0:
        return [0.0]
    return [max_value * i / (count - 1) for i in range(count)]


def fmt_tick(value):
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def tflops_bar_svg(path, rows):
    width = 1120
    height = 680
    margin_left = 78
    margin_right = 34
    margin_top = 96
    margin_bottom = 120
    panel_gap = 64
    panel_w = (width - margin_left - margin_right - panel_gap) / 2
    panel_h = height - margin_top - margin_bottom
    group_gap = 20
    bar_gap = 3
    precisions = sorted({row["precision"] for row in rows}, key=lambda p: PRECISION_ORDER.get(p, 99))
    groups = sorted({(row["path"], row["mode"]) for row in rows}, key=lambda key: GROUP_ORDER.get(key, 99))
    noises = sorted({row["noise_cp_per_mma"] for row in rows})
    max_by_precision = {
        precision: max(row["tflops"] for row in rows if row["precision"] == precision) * 1.15
        for precision in precisions
    }

    elements = base_svg(width, height, "SS/TS TMEM Write Noise TFLOP/s")
    legend_x = margin_left
    legend_y = 50
    legend_item_w = 110
    for idx, noise in enumerate(noises):
        x = legend_x + idx * legend_item_w
        label = "base" if noise == 0 else f"noise {noise}"
        elements.append(f'<rect x="{x}" y="{legend_y - 11}" width="14" height="14" fill="{NOISE_COLORS[noise]}"/>')
        elements.append(f'<text x="{x + 20}" y="{legend_y + 1}" class="legend">{escape(label)}</text>')

    for panel_idx, precision in enumerate(precisions):
        panel_x = margin_left + panel_idx * (panel_w + panel_gap)
        panel_max = max_by_precision[precision]

        def y_pos(value):
            return margin_top + panel_h - (value / panel_max) * panel_h

        elements.append(f'<text x="{panel_x + panel_w / 2:.1f}" y="{margin_top - 22}" text-anchor="middle" class="subtitle">{escape(precision)}</text>')
        if panel_idx == 0:
            elements.append(
                f'<text x="22" y="{margin_top + panel_h / 2:.1f}" transform="rotate(-90 22 {margin_top + panel_h / 2:.1f})" '
                f'text-anchor="middle" class="label">TFLOP/s</text>'
            )
        for tick in axis_ticks(panel_max):
            yy = y_pos(tick)
            elements.append(f'<line x1="{panel_x}" y1="{yy:.1f}" x2="{panel_x + panel_w}" y2="{yy:.1f}" class="grid"/>')
            elements.append(f'<text x="{panel_x - 10}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{fmt_tick(tick)}</text>')
        elements.append(f'<line x1="{panel_x}" y1="{margin_top}" x2="{panel_x}" y2="{margin_top + panel_h}" class="axis"/>')
        elements.append(f'<line x1="{panel_x}" y1="{margin_top + panel_h}" x2="{panel_x + panel_w}" y2="{margin_top + panel_h}" class="axis"/>')

        group_w = (panel_w - group_gap * (len(groups) - 1)) / len(groups)
        bar_w = max(8, (group_w - bar_gap * (len(noises) - 1)) / len(noises))
        by_key = {
            (row["path"], row["mode"], row["noise_cp_per_mma"]): row
            for row in rows
            if row["precision"] == precision
        }
        for group_idx, group in enumerate(groups):
            group_x = panel_x + group_idx * (group_w + group_gap)
            for noise_idx, noise in enumerate(noises):
                row = by_key.get((group[0], group[1], noise))
                if not row:
                    continue
                x = group_x + noise_idx * (bar_w + bar_gap)
                yy = y_pos(row["tflops"])
                elements.append(
                    f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" '
                    f'height="{margin_top + panel_h - yy:.1f}" fill="{NOISE_COLORS[noise]}"/>'
                )
                elements.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{yy - 5:.1f}" text-anchor="middle" '
                    f'class="value">{row["tflops"]:.0f}</text>'
                )
            label_x = group_x + group_w / 2
            elements.append(
                f'<text x="{label_x:.1f}" y="{margin_top + panel_h + 18}" text-anchor="end" '
                f'transform="rotate(-35 {label_x:.1f} {margin_top + panel_h + 18})" '
                f'class="label">{escape(GROUP_LABELS[group])}</text>'
            )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_outputs(out_dir, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "tmem_interference_tflops_results.csv", rows)
    tflops_bar_svg(out_dir / "tmem_interference_tflops.svg", rows)
    for obsolete in [
        "tmem_interference_extra_cycles.svg",
        "tmem_interference_results.csv",
        "tmem_interference_tflops_ratio.svg",
    ]:
        (out_dir / obsolete).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Plot Thor tcgen05 TMEM interference report.")
    parser.add_argument("--report", type=Path, default=REPORT_PATH, help="TMEM interference report")
    parser.add_argument("--out-dir", type=Path, default=PLOTS_DIR, help="directory for CSV and SVG outputs")
    args = parser.parse_args()

    if not args.report.exists():
        raise SystemExit(f"report not found: {args.report}")
    rows = parse_throughput(args.report)
    write_outputs(args.out_dir, rows)
    print(f"Wrote TMEM interference TFLOP/s plot and CSV to {args.out_dir}")
    print(f"Parsed rows: {len(rows)}")


if __name__ == "__main__":
    main()
