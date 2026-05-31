#!/usr/bin/env python3
import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path


COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
    "#65a30d",
    "#7c3aed",
]

KEY_GEMM_SERIES = {
    "cuBLAS",
    "v2 coalesced naive",
    "v3 shared-memory tile",
    "v5 warp tiling",
    "v6 thread coarsening",
    "v9 vectorized",
    "v10 double buffer",
}


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    try:
        return float(value)
    except ValueError:
        return 0.0


def group_series(rows, x_col, y_col):
    samples = defaultdict(list)
    for row in rows:
        if row.get("Matched") != "1":
            continue
        name = row["Version"]
        if "skipped" in name:
            continue
        samples[(name, to_float(row[x_col]))].append(to_float(row[y_col]))

    grouped = defaultdict(list)
    for (name, x), values in samples.items():
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            stddev = math.sqrt(variance)
        else:
            stddev = 0.0
        grouped[name].append((x, mean, stddev))

    return {name: sorted(points, key=lambda point: point[0]) for name, points in grouped.items() if points}


def group_tuning_series(rows, kernel_filter=None):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("Matched") != "1":
            continue
        kernel = row["Kernel"]
        if kernel_filter is not None and kernel != kernel_filter:
            continue
        if kernel == "cuBLAS":
            name = "cuBLAS"
        else:
            name = row["Config"]
        grouped[name].append((to_float(row["N"]), to_float(row["GFLOPS"]), 0.0))
    return {name: sorted(points, key=lambda point: point[0]) for name, points in grouped.items() if points}


def group_tuning_best_by_kernel(rows):
    best = {}
    for row in rows:
        if row.get("Matched") != "1":
            continue
        kernel = row["Kernel"]
        n = to_float(row["N"])
        gflops = to_float(row["GFLOPS"])
        key = (kernel, n)
        if key not in best or gflops > best[key][0]:
            best[key] = (gflops, row["Config"])

    grouped = defaultdict(list)
    for (kernel, n), (gflops, config) in best.items():
        name = "cuBLAS" if kernel == "cuBLAS" else f"{kernel} best"
        grouped[name].append((n, gflops, 0.0))
    return {name: sorted(points, key=lambda point: point[0]) for name, points in grouped.items() if points}


def filter_series(series, names):
    return {name: points for name, points in series.items() if name in names}


def nice_range(values):
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if lo == hi:
        return 0.0, hi * 1.1 if hi else 1.0
    pad = (hi - lo) * 0.08
    return max(0.0, lo - pad), hi + pad


def nice_step(raw_step):
    if raw_step <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    base = raw_step / (10**exponent)
    if base <= 1.0:
        nice = 1.0
    elif base <= 2.0:
        nice = 2.0
    elif base <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10**exponent)


def linear_ticks(max_value, target_count=6):
    if max_value <= 0.0:
        return [0.0, 1.0]
    step = nice_step(max_value / max(1, target_count - 1))
    top = math.ceil(max_value / step) * step
    ticks = []
    value = 0.0
    while value <= top + step * 0.5:
        ticks.append(value)
        value += step
    return ticks


def log_ticks(min_value, max_value):
    min_exp = math.floor(math.log10(max(min_value, 1e-9)))
    max_exp = math.ceil(math.log10(max_value))
    ticks = []
    for exp in range(min_exp, max_exp + 1):
        for mantissa in (1, 2, 5):
            value = mantissa * (10**exp)
            if min_value <= value <= max_value:
                ticks.append(float(value))
    return ticks


def x_label(value):
    if abs(value - round(value)) < 1e-6:
        value = int(round(value))
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{value:g}"


def is_power_of_two(value):
    integer = int(round(value))
    return abs(value - integer) < 1e-6 and integer > 0 and (integer & (integer - 1)) == 0


def labeled_x_ticks(values):
    if len(values) <= 8:
        return values
    labeled = {values[0], values[-1]}
    labeled.update(value for value in values if is_power_of_two(value))
    return [value for value in values if value in labeled]


def y_label(value):
    if value >= 1000:
        return f"{value / 1000:.1f}K"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.1f}"


def write_svg(series, title, x_title, y_title, output_path, y_scale="linear"):
    width = 1280
    height = 680
    left = 92
    right = 300
    top = 72
    bottom = 86
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_x = sorted({x for points in series.values() for x, _, _ in points})
    all_y = []
    for points in series.values():
        for _, y, err in points:
            all_y.extend([y - err, y + err])
    if not all_x or not all_y:
        output_path.write_text("")
        return

    if len(all_x) == 1:
        x_min = all_x[0] * 0.9
        x_max = all_x[0] * 1.1
    else:
        x_min = min(all_x)
        x_max = max(all_x)

    positive_y = [value for value in all_y if value > 0.0]
    if y_scale == "log":
        y_min = min(positive_y) * 0.85
        y_max = max(positive_y) * 1.15
        y_ticks = log_ticks(y_min, y_max)
    else:
        y_min = 0.0
        y_ticks = linear_ticks(max(all_y), 6)
        y_max = y_ticks[-1]

    x_log_min = math.log2(x_min)
    x_log_max = math.log2(x_max)
    if abs(x_log_max - x_log_min) < 1e-9:
        x_log_min -= 0.5
        x_log_max += 0.5

    def sx(x):
        return left + (math.log2(x) - x_log_min) / (x_log_max - x_log_min) * plot_w

    def sy(y):
        if y_scale == "log":
            safe_y = max(y, y_min)
            return top + plot_h - (math.log10(safe_y) - math.log10(y_min)) / (
                math.log10(y_max) - math.log10(y_min)
            ) * plot_h
        return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="36" font-size="24" font-family="Arial" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-size="14" font-family="Arial" fill="#374151">{html.escape(x_title)}</text>',
        f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-size="14" font-family="Arial" fill="#374151">{html.escape(y_title)}</text>',
        f'<text x="{left}" y="58" font-size="12" font-family="Arial" fill="#6b7280">x-axis: log2(size); labeled ticks: powers of two/endpoints; error bars: ±1 sample standard deviation</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#d1d5db"/>',
    ]

    for value in y_ticks:
        y = sy(value)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#4b5563">{html.escape(y_label(value))}</text>')

    labeled_x = set(labeled_x_ticks(all_x))
    for value in all_x:
        x = sx(value)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#eef2f7"/>')
        if value in labeled_x:
            lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12" font-family="Arial" fill="#4b5563">{html.escape(x_label(value))}</text>')

    for idx, (name, points) in enumerate(series.items()):
        color = COLORS[idx % len(COLORS)]
        path = " ".join(
            f"{'M' if point_idx == 0 else 'L'} {sx(x):.2f} {sy(y):.2f}"
            for point_idx, (x, y, _) in enumerate(points)
        )
        lines.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y, err in points:
            px = sx(x)
            py = sy(y)
            if err > 0.0:
                y_low = sy(max(y - err, y_min))
                y_high = sy(y + err)
                lines.append(f'<line x1="{px:.2f}" y1="{y_high:.2f}" x2="{px:.2f}" y2="{y_low:.2f}" stroke="{color}" stroke-width="1.6"/>')
                lines.append(f'<line x1="{px - 5:.2f}" y1="{y_high:.2f}" x2="{px + 5:.2f}" y2="{y_high:.2f}" stroke="{color}" stroke-width="1.6"/>')
                lines.append(f'<line x1="{px - 5:.2f}" y1="{y_low:.2f}" x2="{px + 5:.2f}" y2="{y_low:.2f}" stroke="{color}" stroke-width="1.6"/>')
            lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{color}"/>')

    legend_x = left + plot_w + 30
    legend_y = top + 4
    lines.append(f'<text x="{legend_x}" y="{legend_y}" font-size="14" font-family="Arial" font-weight="700" fill="#111827">Versions</text>')
    for idx, name in enumerate(series):
        y = legend_y + 24 + idx * 22
        color = COLORS[idx % len(COLORS)]
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-size="12" font-family="Arial" fill="#374151">{html.escape(name)}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot CUDA benchmark sweep CSV files.")
    parser.add_argument("--reduce", help="Path to reduce_sweep.csv")
    parser.add_argument("--gemm", help="Path to sgemm_sweep.csv")
    parser.add_argument("--tuning", help="Path to blocksize_tuning.csv")
    parser.add_argument("--out-dir", required=True, help="Directory for SVG figures")
    args = parser.parse_args()
    if not args.reduce and not args.gemm and not args.tuning:
        parser.error("at least one of --reduce, --gemm, or --tuning is required")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reduce:
        reduce_rows = read_rows(args.reduce)
        write_svg(
            group_series(reduce_rows, "N", "BandwidthGBps"),
            "REDUCE Bandwidth Sweep",
            "N elements",
            "Bandwidth (GB/s)",
            out_dir / "reduce_bandwidth.svg",
        )

    if args.gemm:
        gemm_rows = read_rows(args.gemm)
        gemm_gflops = group_series(gemm_rows, "N", "GFLOPS")
        key_gemm_gflops = filter_series(gemm_gflops, KEY_GEMM_SERIES)
        write_svg(
            key_gemm_gflops,
            "SGEMM GFLOPS Sweep (Key Kernels)",
            "Square matrix size N",
            "GFLOPS",
            out_dir / "sgemm_gflops.svg",
        )
        write_svg(
            gemm_gflops,
            "SGEMM GFLOPS Sweep (All Kernels, Log Y)",
            "Square matrix size N",
            "GFLOPS",
            out_dir / "sgemm_gflops_log.svg",
            y_scale="log",
        )
        write_svg(
            filter_series(group_series(gemm_rows, "N", "RatioToCuBLAS"), KEY_GEMM_SERIES),
            "SGEMM Ratio To cuBLAS",
            "Square matrix size N",
            "Ratio",
            out_dir / "sgemm_ratio_to_cublas.svg",
        )

    if args.tuning:
        tuning_rows = read_rows(args.tuning)
        for kernel in ("thread_tile", "vectorized", "double_buffer"):
            write_svg(
                group_tuning_series(tuning_rows, kernel),
                f"GEMM Blocksize Tuning ({kernel})",
                "Square matrix size N",
                "GFLOPS",
                out_dir / f"tuning_{kernel}.svg",
            )
        write_svg(
            group_tuning_best_by_kernel(tuning_rows),
            "GEMM Blocksize Tuning (Best Per Kernel Family)",
            "Square matrix size N",
            "GFLOPS",
            out_dir / "tuning_best_by_kernel.svg",
        )

    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
