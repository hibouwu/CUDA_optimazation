#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
G2_CSV = PLOTS_DIR / "g2_ss_mainloop_sweep_results.csv"
G1_MAINLOOP_CSV = PLOTS_DIR / "mma_mainloop_sweep_results.csv"
G1_MMA_ONLY_CSV = PLOTS_DIR / "mma_only_results.csv"
OUT_CSV = PLOTS_DIR / "g2_vs_g1_tflops_results.csv"
OUT_SVG = PLOTS_DIR / "g2_vs_g1_tflops.svg"

K_BLOCKS = (1, 4, 8, 16)
SERIES = [
    ("g1_m128n128_tflops", "g1 M128N128", "#2878b5"),
    ("g2_m256n128_tflops", "g2 M256N128", "#c75d2c"),
    ("g1_m128n256_tflops", "g1 M128N256", "#0f5d3b"),
    ("g2_m256n256_tflops", "g2 M256N256", "#8f3f71"),
]


def numeric(value):
    return float(str(value).strip().replace(",", ""))


def read_rows(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_g2(path):
    rows = {}
    for row in read_rows(path):
        if row["precision"] != "BF16":
            continue
        k_blocks = int(numeric(row["k_blocks"]))
        shape = row["shape"].removesuffix("K16").lower()
        if k_blocks in K_BLOCKS and shape in ("m256n128", "m256n256"):
            rows[(shape, k_blocks)] = {
                "g2_tflops": numeric(row["g2_ss_mainloop_tflops"]),
                "g2_cycles_per_mma": numeric(row["cycles_per_mma"]),
            }
    missing = [
        (shape, k)
        for shape in ("m256n128", "m256n256")
        for k in K_BLOCKS
        if (shape, k) not in rows
    ]
    if missing:
        raise ValueError(f"missing g2 rows for {missing}")
    return rows


def load_g1_shape(shape, mainloop_path, mma_only_path):
    rows = {}
    for row in read_rows(mma_only_path):
        if row["precision"] == "BF16" and row["shape"] == shape:
            rows[1] = {
                "tflops": numeric(row["ss_mma_only_tflops"]),
                "cycles_per_mma": "",
            }
            break

    for row in read_rows(mainloop_path):
        if row["precision"] != "BF16" or row["shape"] != shape:
            continue
        k_blocks = int(numeric(row["k_blocks"]))
        if k_blocks in (4, 8, 16):
            rows[k_blocks] = {
                "tflops": numeric(row["ss_mainloop_tflops"]),
                "cycles_per_mma": numeric(row["cycles_per_mma"]),
            }

    missing = [k for k in K_BLOCKS if k not in rows]
    if missing:
        raise ValueError(f"missing g1 {shape} rows for K{missing}")
    return rows


def build_comparison_rows(g2_rows, g1_m128n128, g1_m128n256):
    rows = []
    for k_blocks in K_BLOCKS:
        g1_128 = g1_m128n128[k_blocks]["tflops"]
        g1_256 = g1_m128n256[k_blocks]["tflops"]
        g2_128 = g2_rows[("m256n128", k_blocks)]["g2_tflops"]
        g2_256 = g2_rows[("m256n256", k_blocks)]["g2_tflops"]
        rows.append(
            {
                "k_blocks": k_blocks,
                "k_tile": 16 * k_blocks,
                "g1_m128n128_tflops": g1_128,
                "g1_m128n256_tflops": g1_256,
                "g2_m256n128_tflops": g2_128,
                "g2_m256n256_tflops": g2_256,
                "g2_m256n128_vs_g1_m128n128": g2_128 / g1_128,
                "g2_m256n128_vs_g1_m128n256": g2_128 / g1_256,
                "g2_m256n256_vs_g1_m128n128": g2_256 / g1_128,
                "g2_m256n256_vs_g1_m128n256": g2_256 / g1_256,
                "g1_m128n128_cycles_per_mma": g1_m128n128[k_blocks]["cycles_per_mma"],
                "g1_m128n256_cycles_per_mma": g1_m128n256[k_blocks]["cycles_per_mma"],
                "g2_m256n128_cycles_per_mma": g2_rows[("m256n128", k_blocks)]["g2_cycles_per_mma"],
                "g2_m256n256_cycles_per_mma": g2_rows[("m256n256", k_blocks)]["g2_cycles_per_mma"],
            }
        )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "k_blocks",
        "k_tile",
        "g1_m128n128_tflops",
        "g2_m256n128_tflops",
        "g1_m128n256_tflops",
        "g2_m256n256_tflops",
        "g2_m256n128_vs_g1_m128n128",
        "g2_m256n128_vs_g1_m128n256",
        "g2_m256n256_vs_g1_m128n128",
        "g2_m256n256_vs_g1_m128n256",
        "g1_m128n128_cycles_per_mma",
        "g2_m256n128_cycles_per_mma",
        "g1_m128n256_cycles_per_mma",
        "g2_m256n256_cycles_per_mma",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def y_ticks(max_value):
    return [max_value * frac for frac in (0.0, 0.25, 0.5, 0.75, 1.0)]


def fmt_tick(value):
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def svg_header(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2933; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { font-size: 13px; fill: #52606d; }",
        ".axis { stroke: #46515c; stroke-width: 1; }",
        ".grid { stroke: #d7dde3; stroke-width: 1; }",
        ".tick { font-size: 12px; fill: #52606d; }",
        ".label { font-size: 12px; fill: #334e68; }",
        ".legend { font-size: 12px; fill: #334e68; }",
        ".value { font-size: 10px; fill: #102a43; }",
        ".ratio { font-size: 10px; fill: #7c2d12; font-weight: 700; }",
        "</style>",
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" class="title">{escape(title)}</text>',
    ]


def tflops_bar_svg(path, rows):
    width = 1240
    height = 680
    margin_left = 84
    margin_right = 42
    margin_top = 104
    margin_bottom = 118
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_value = max(row[key] for row in rows for key, _, _ in SERIES) * 1.16

    def y_pos(value):
        return margin_top + plot_h - (value / max_value) * plot_h

    elements = svg_header(
        width,
        height,
        "TCGen05 BF16 SS Mainloop: cta_group::2 vs group1",
    )
    elements.append(
        f'<text x="{width / 2:.1f}" y="53" text-anchor="middle" class="subtitle">'
        "K1 uses SS MMA-only; K4/K8/K16 use SS mainloop. Ratio labels are g2 / matching g1 shape."
        "</text>"
    )

    legend_x = margin_left
    legend_y = 72
    legend_item_w = 165
    for idx, (_, label, color) in enumerate(SERIES):
        x = legend_x + idx * legend_item_w
        elements.append(f'<rect x="{x}" y="{legend_y - 11}" width="14" height="14" fill="{color}"/>')
        elements.append(f'<text x="{x + 20}" y="{legend_y + 1}" class="legend">{escape(label)}</text>')

    elements.append(
        f'<text x="24" y="{margin_top + plot_h / 2:.1f}" '
        f'transform="rotate(-90 24 {margin_top + plot_h / 2:.1f})" '
        'text-anchor="middle" class="label">TFLOP/s</text>'
    )
    for tick in y_ticks(max_value):
        yy = y_pos(tick)
        elements.append(f'<line x1="{margin_left}" y1="{yy:.1f}" x2="{margin_left + plot_w}" y2="{yy:.1f}" class="grid"/>')
        elements.append(f'<text x="{margin_left - 10}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{fmt_tick(tick)}</text>')
    elements.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    elements.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis"/>')

    group_gap = 44
    group_w = (plot_w - group_gap * (len(rows) - 1)) / len(rows)
    bar_gap = 8
    bar_w = (group_w - bar_gap * (len(SERIES) - 1)) / len(SERIES)
    for row_idx, row in enumerate(rows):
        group_x = margin_left + row_idx * (group_w + group_gap)
        for series_idx, (key, _, color) in enumerate(SERIES):
            value = row[key]
            x = group_x + series_idx * (bar_w + bar_gap)
            yy = y_pos(value)
            elements.append(
                f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" '
                f'height="{margin_top + plot_h - yy:.1f}" fill="{color}"/>'
            )
            label_y = yy - 6
            elements.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{label_y:.1f}" '
                f'text-anchor="middle" class="value">{value:.1f}</text>'
            )
            if key == "g2_m256n128_tflops":
                elements.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{label_y - 13:.1f}" '
                    f'text-anchor="middle" class="ratio">{row["g2_m256n128_vs_g1_m128n128"]:.2f}x</text>'
                )
            if key == "g2_m256n256_tflops":
                elements.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{label_y - 13:.1f}" '
                    f'text-anchor="middle" class="ratio">{row["g2_m256n256_vs_g1_m128n256"]:.2f}x</text>'
                )

        label_x = group_x + group_w / 2
        elements.append(
            f'<text x="{label_x:.1f}" y="{margin_top + plot_h + 22}" '
            f'text-anchor="middle" class="label">K{row["k_blocks"]}</text>'
        )
        elements.append(
            f'<text x="{label_x:.1f}" y="{margin_top + plot_h + 40}" '
            f'text-anchor="middle" class="tick">K_tile {row["k_tile"]}</text>'
        )
        elements.append(
            f'<text x="{label_x:.1f}" y="{margin_top + plot_h + 60}" '
            f'text-anchor="middle" class="tick">'
            f'N128 {row["g2_m256n128_vs_g1_m128n128"]:.2f}x / '
            f'N256 {row["g2_m256n256_vs_g1_m128n256"]:.2f}x</text>'
        )

    elements.append(
        f'<text x="{margin_left + plot_w:.1f}" y="{height - 24}" text-anchor="end" class="tick">'
        "Sources: plots/g2_ss_mainloop_sweep_results.csv, mma_mainloop_sweep_results.csv, mma_only_results.csv"
        "</text>"
    )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plot g2 vs g1 TCGen05 BF16 SS mainloop TFLOP/s.")
    parser.add_argument("--g2-csv", type=Path, default=G2_CSV)
    parser.add_argument("--g1-mainloop-csv", type=Path, default=G1_MAINLOOP_CSV)
    parser.add_argument("--g1-mma-only-csv", type=Path, default=G1_MMA_ONLY_CSV)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-svg", type=Path, default=OUT_SVG)
    args = parser.parse_args()

    g2_rows = load_g2(args.g2_csv)
    g1_m128n128 = load_g1_shape("M128N128", args.g1_mainloop_csv, args.g1_mma_only_csv)
    g1_m128n256 = load_g1_shape("M128N256", args.g1_mainloop_csv, args.g1_mma_only_csv)
    rows = build_comparison_rows(g2_rows, g1_m128n128, g1_m128n256)
    write_csv(args.out_csv, rows)
    tflops_bar_svg(args.out_svg, rows)
    print(f"Wrote {args.out_svg}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
