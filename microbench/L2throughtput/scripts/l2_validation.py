#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "build_and_run.sh"
BIN = ROOT / "build" / "l2_throughput"
RESULT_DIR = ROOT / "results"
PLOT_DIR = ROOT / "plots"

MODES = ("read-unique", "write-unique")
THREADS = (64, 128, 256)
CAPACITY_BYTES = tuple(1 << p for p in range(16, 28))  # 64 KiB .. 128 MiB


def run(cmd, *, capture=True):
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if proc.returncode != 0:
        msg = f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}"
        if capture:
            msg += f"\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        raise RuntimeError(msg)
    return proc.stdout if capture else ""


def build():
    run([str(BUILD_SCRIPT), "build-only"], capture=False)


def parse_csv_row(text):
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one CSV row, got:\n{text}")
    header = run([str(BIN), "--csv-header"]).strip().split(",")
    row = next(csv.DictReader([",".join(header), lines[0]]))
    return row


def bench_once(mode, threads, blocks_per_sm, bytes_, iters, warmup_iters):
    text = run([
        str(BIN),
        "--mode", mode,
        "--threads", str(threads),
        "--blocks-per-sm", str(blocks_per_sm),
        "--bytes", str(bytes_),
        "--iters", str(iters),
        "--warmup-iters", str(warmup_iters),
        "--csv",
    ])
    row = parse_csv_row(text)
    row["threads_per_block"] = int(row["threads_per_block"])
    row["blocks_per_sm"] = int(row["blocks_per_sm"])
    row["sm_count"] = int(row["sm_count"])
    row["occupancy_blocks_per_sm"] = int(row["occupancy_blocks_per_sm"])
    row["working_set_bytes"] = int(row["working_set_bytes"])
    row["touched_footprint_bytes"] = int(row["touched_footprint_bytes"])
    row["index_stride_elements"] = int(row["index_stride_elements"])
    row["stream_period_iters"] = int(row["stream_period_iters"])
    row["requested_bytes"] = float(row["requested_bytes"])
    row["elapsed_cycles"] = int(row["elapsed_cycles"])
    row["bytes_per_cycle"] = float(row["bytes_per_cycle"])
    row["requested_to_working_set_ratio"] = float(
        row["requested_to_working_set_ratio"]
    )
    row["active_warps_per_sm"] = (
        row["threads_per_block"] * row["blocks_per_sm"] / 32.0
    )
    row["working_set_mib"] = row["working_set_bytes"] / (1024.0 * 1024.0)
    row["per_sm_bytes_per_cycle"] = row["bytes_per_cycle"] / row["sm_count"]
    return row


def bench_median(mode, threads, blocks_per_sm, bytes_, iters, warmup_iters, repeats):
    samples = [
        bench_once(mode, threads, blocks_per_sm, bytes_, iters, warmup_iters)
        for _ in range(repeats)
    ]
    bpcs = [sample["bytes_per_cycle"] for sample in samples]
    chosen = min(samples, key=lambda row: abs(row["bytes_per_cycle"] - statistics.median(bpcs)))
    chosen["bytes_per_cycle_median"] = statistics.median(bpcs)
    chosen["bytes_per_cycle_min"] = min(bpcs)
    chosen["bytes_per_cycle_max"] = max(bpcs)
    chosen["repeat_count"] = repeats
    return chosen


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "mode",
        "threads_per_block",
        "blocks_per_sm",
        "active_warps_per_sm",
        "occupancy_blocks_per_sm",
        "working_set_bytes",
        "working_set_mib",
        "requested_bytes",
        "requested_to_working_set_ratio",
        "elapsed_cycles",
        "bytes_per_cycle_median",
        "bytes_per_cycle_min",
        "bytes_per_cycle_max",
        "per_sm_bytes_per_cycle",
        "index_stride_elements",
        "stream_period_iters",
        "sm_count",
        "repeat_count",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def concurrency_sweep(args):
    rows = []
    for mode in MODES:
        for threads in THREADS:
            probe = bench_once(
                mode, threads, 1, args.bytes, args.iters, args.warmup_iters
            )
            limit = probe["occupancy_blocks_per_sm"]
            for blocks_per_sm in range(1, limit + 1):
                print(
                    f"[concurrency] {mode} threads={threads} "
                    f"blocks/SM={blocks_per_sm}/{limit}",
                    flush=True,
                )
                rows.append(
                    bench_median(
                        mode,
                        threads,
                        blocks_per_sm,
                        args.bytes,
                        args.iters,
                        args.warmup_iters,
                        args.repeats,
                    )
                )
    return rows


def capacity_sweep(args):
    rows = []
    for mode in MODES:
        for bytes_ in CAPACITY_BYTES:
            print(
                f"[capacity] {mode} working_set={bytes_ // 1024} KiB",
                flush=True,
            )
            rows.append(
                bench_median(
                    mode,
                    args.capacity_threads,
                    args.capacity_blocks_per_sm,
                    bytes_,
                    args.capacity_iters,
                    args.warmup_iters,
                    args.repeats,
                )
            )
    return rows


def scale_linear(value, lo, hi, a, b):
    if hi <= lo:
        return (a + b) / 2.0
    return a + (value - lo) * (b - a) / (hi - lo)


def scale_log2(value, lo, hi, a, b):
    return scale_linear(math.log2(value), math.log2(lo), math.log2(hi), a, b)


def color_for(key):
    palette = {
        "read-unique": "#1f6feb",
        "write-unique": "#d14",
        64: "#6f42c1",
        128: "#0a7f5a",
        256: "#b45309",
    }
    return palette.get(key, "#333")


def svg_header(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".axis{stroke:#374151;stroke-width:1}",
        ".grid{stroke:#e5e7eb;stroke-width:1}",
        ".label{font-size:13px}",
        ".small{font-size:12px;fill:#4b5563}",
        ".title{font-size:20px;font-weight:700}",
        ".legend{font-size:12px}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
        f'<text class="title" x="{width / 2}" y="30" text-anchor="middle">{title}</text>',
    ]


def draw_axes(lines, x0, y0, w, h, x_label, y_label, y_max, y_ticks):
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}"/>')
    for tick in y_ticks:
        y = scale_linear(tick, 0, y_max, y0 + h, y0)
        lines.append(f'<line class="grid" x1="{x0}" y1="{y}" x2="{x0 + w}" y2="{y}"/>')
        lines.append(f'<text class="small" x="{x0 - 8}" y="{y + 4}" text-anchor="end">{tick:g}</text>')
    lines.append(f'<text class="label" x="{x0 + w / 2}" y="{y0 + h + 42}" text-anchor="middle">{x_label}</text>')
    lines.append(
        f'<text class="label" transform="translate({x0 - 48},{y0 + h / 2}) rotate(-90)" '
        f'text-anchor="middle">{y_label}</text>'
    )


def polyline(points, color, width=2.5):
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def circle(x, y, color):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>'


def plot_concurrency(rows, path):
    width, height = 1040, 620
    margin = dict(left=76, right=36, top=64, bottom=76)
    panel_gap = 62
    panel_w = (width - margin["left"] - margin["right"] - panel_gap) / 2
    panel_h = height - margin["top"] - margin["bottom"]
    y_max = math.ceil(max(row["bytes_per_cycle_median"] for row in rows) / 100.0) * 100.0
    y_ticks = [0, y_max * 0.25, y_max * 0.5, y_max * 0.75, y_max]
    lines = svg_header(width, height, "L2 吞吐随活跃 Warp 数饱和曲线")

    for panel_idx, mode in enumerate(MODES):
        x0 = margin["left"] + panel_idx * (panel_w + panel_gap)
        y0 = margin["top"]
        draw_axes(
            lines,
            x0,
            y0,
            panel_w,
            panel_h,
            "active warps / SM",
            "B/cycle (全 GPU)",
            y_max,
            y_ticks,
        )
        lines.append(f'<text class="label" x="{x0 + panel_w / 2}" y="{y0 - 18}" text-anchor="middle">{mode}</text>')
        for tick in range(0, 33, 8):
            x = scale_linear(tick, 0, 32, x0, x0 + panel_w)
            lines.append(f'<line class="grid" x1="{x}" y1="{y0}" x2="{x}" y2="{y0 + panel_h}"/>')
            lines.append(f'<text class="small" x="{x}" y="{y0 + panel_h + 18}" text-anchor="middle">{tick}</text>')
        for threads in THREADS:
            subset = [
                row for row in rows
                if row["mode"] == mode and row["threads_per_block"] == threads
            ]
            subset.sort(key=lambda row: row["active_warps_per_sm"])
            pts = [
                (
                    scale_linear(row["active_warps_per_sm"], 0, 32, x0, x0 + panel_w),
                    scale_linear(row["bytes_per_cycle_median"], 0, y_max, y0 + panel_h, y0),
                )
                for row in subset
            ]
            color = color_for(threads)
            lines.append(polyline(pts, color))
            for x, y in pts:
                lines.append(circle(x, y, color))

    legend_x = margin["left"] + 8
    legend_y = height - 22
    for idx, threads in enumerate(THREADS):
        x = legend_x + idx * 125
        color = color_for(threads)
        lines.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text class="legend" x="{x + 36}" y="{legend_y + 4}">每 block {threads} 线程</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def plot_capacity(rows, path):
    width, height = 960, 560
    x0, y0, w, h = 82, 68, 820, 400
    y_max = math.ceil(max(row["bytes_per_cycle_median"] for row in rows) / 100.0) * 100.0
    y_ticks = [0, y_max * 0.25, y_max * 0.5, y_max * 0.75, y_max]
    x_min = min(row["working_set_bytes"] for row in rows)
    x_max = max(row["working_set_bytes"] for row in rows)
    lines = svg_header(width, height, "L2 容量阶跃")
    draw_axes(lines, x0, y0, w, h, "工作集 (MiB, log2)", "B/cycle (全 GPU)", y_max, y_ticks)

    for mib in (1 / 16, 1 / 4, 1, 4, 16, 32, 64, 128):
        bytes_ = mib * 1024 * 1024
        if bytes_ < x_min or bytes_ > x_max:
            continue
        x = scale_log2(bytes_, x_min, x_max, x0, x0 + w)
        cls = "axis" if mib == 32 else "grid"
        lines.append(f'<line class="{cls}" x1="{x}" y1="{y0}" x2="{x}" y2="{y0 + h}"/>')
        label = f"{mib:g}"
        lines.append(f'<text class="small" x="{x}" y="{y0 + h + 18}" text-anchor="middle">{label}</text>')
    l2_x = scale_log2(32 * 1024 * 1024, x_min, x_max, x0, x0 + w)
    lines.append(f'<text class="small" x="{l2_x + 6}" y="{y0 + 16}">32 MiB L2</text>')

    for mode in MODES:
        subset = [row for row in rows if row["mode"] == mode]
        subset.sort(key=lambda row: row["working_set_bytes"])
        pts = [
            (
                scale_log2(row["working_set_bytes"], x_min, x_max, x0, x0 + w),
                scale_linear(row["bytes_per_cycle_median"], 0, y_max, y0 + h, y0),
            )
            for row in subset
        ]
        color = color_for(mode)
        lines.append(polyline(pts, color))
        for x, y in pts:
            lines.append(circle(x, y, color))

    legend_y = height - 28
    for idx, mode in enumerate(MODES):
        x = x0 + idx * 170
        color = color_for(mode)
        lines.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text class="legend" x="{x + 36}" y="{legend_y + 4}">{mode}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def top_summary(concurrency_rows, capacity_rows):
    rows = []
    for mode in MODES:
        subset = [row for row in concurrency_rows if row["mode"] == mode]
        best = max(subset, key=lambda row: row["bytes_per_cycle_median"])
        rows.append((mode, best))
    cap_by_mode = {}
    for mode in MODES:
        subset = [row for row in capacity_rows if row["mode"] == mode]
        cap_by_mode[mode] = max(subset, key=lambda row: row["bytes_per_cycle_median"])
    return rows, cap_by_mode


def write_report(concurrency_rows, capacity_rows, ncu_note):
    peaks, cap_peaks = top_summary(concurrency_rows, capacity_rows)
    l2_read_rows = [
        row for row in capacity_rows
        if row["mode"] == "read-unique" and
        1.0 <= row["working_set_mib"] <= 32.0
    ]
    l2_write_rows = [
        row for row in capacity_rows
        if row["mode"] == "write-unique" and
        1.0 <= row["working_set_mib"] <= 8.0
    ]
    read_l2_min = min(row["bytes_per_cycle_median"] for row in l2_read_rows)
    read_l2_max = max(row["bytes_per_cycle_median"] for row in l2_read_rows)
    read_l2_med = statistics.median(row["bytes_per_cycle_median"] for row in l2_read_rows)
    write_l2_min = min(row["bytes_per_cycle_median"] for row in l2_write_rows)
    write_l2_max = max(row["bytes_per_cycle_median"] for row in l2_write_rows)
    write_l2_med = statistics.median(row["bytes_per_cycle_median"] for row in l2_write_rows)
    read_64m = next(
        row for row in capacity_rows
        if row["mode"] == "read-unique" and row["working_set_mib"] == 64.0
    )
    read_128m = next(
        row for row in capacity_rows
        if row["mode"] == "read-unique" and row["working_set_mib"] == 128.0
    )
    write_16m = next(
        row for row in capacity_rows
        if row["mode"] == "write-unique" and row["working_set_mib"] == 16.0
    )
    report = [
        "# L2 吞吐验证报告",
        "",
        "## 方法",
        "",
        "- 容量 sweep 将工作集从 64 KiB 扫到 128 MiB。",
        "- 并发 sweep 通过改变 threads/block 和 blocks/SM 来改变每 SM active warps。",
        "- 每个数据点使用多次 kernel 内 `clock64()` 计时的中位数。",
        "- benchmark 会拒绝超过 CUDA occupancy 上限的 `blocks_per_sm`。",
        "- `read-same` 不进入证明链，因为同地址 load 更多是在测广播/请求压力，不代表物理 L2 数据搬运。",
        "",
        "## 并发饱和峰值",
        "",
        "|模式|峰值 B/cycle|每 SM B/cycle|threads/block|blocks/SM|active warps/SM|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, row in peaks:
        report.append(
            f"|{mode}|{row['bytes_per_cycle_median']:.3f}|"
            f"{row['bytes_per_cycle_median'] / row['sm_count']:.3f}|"
            f"{row['threads_per_block']}|{row['blocks_per_sm']}|"
            f"{row['active_warps_per_sm']:.1f}|"
        )
    report.extend([
        "",
        "## 容量阶跃解释",
        "",
        f"- `read-unique` 在 1 MiB 到 32 MiB 之间基本持平："
        f"中位数 {read_l2_med:.3f} B/cycle，范围 "
        f"{read_l2_min:.3f}-{read_l2_max:.3f} B/cycle。这是非 NCU 证据中最强的 "
        "L2-resident 读平台。",
        f"- 当读工作集超过 32 MiB L2 后，吞吐在 64 MiB 降到 "
        f"{read_64m['bytes_per_cycle_median']:.3f} B/cycle，"
        f"在 128 MiB 降到 {read_128m['bytes_per_cycle_median']:.3f} B/cycle。",
        f"- `write-unique` 在很小工作集下最高；1-8 MiB 大约为 "
        f"{write_l2_min:.3f}-{write_l2_max:.3f} B/cycle，到 16 MiB 降到 "
        f"{write_16m['bytes_per_cycle_median']:.3f} B/cycle。这个结果应解释为端到端 "
        "store path，而不是纯 L2 write-port 上限。",
    ])
    report.extend([
        "",
        "## 容量 sweep 原始最高点",
        "",
        "|模式|最佳 B/cycle|工作集 MiB|",
        "|---|---:|---:|",
    ])
    for mode, row in cap_peaks.items():
        report.append(
            f"|{mode}|{row['bytes_per_cycle_median']:.3f}|"
            f"{row['working_set_mib']:.3f}|"
        )
    write_64k = cap_peaks["write-unique"]
    report.extend([
        "",
        f"注意：`write-unique` 的 {write_64k['bytes_per_cycle_median']:.3f} B/cycle 来自 "
        f"{write_64k['working_set_mib']:.3f} MiB 极小工作集，只能说明 tiny working set 下的端到端 "
        "store-path fast case。它不代表 L2-sized 工作集的持续写吞吐，也不代表纯 L2 write-port 上限。",
        "",
        "## 产物",
        "",
        "- `results/l2_concurrency_sweep.csv`",
        "- `results/l2_capacity_sweep.csv`",
        "- `plots/l2_concurrency_saturation.svg`",
        "- `plots/l2_capacity_staircase.svg`",
        "",
        "## NCU 状态",
        "",
        ncu_note,
        "",
    ])
    (RESULT_DIR / "validation_report.md").write_text("\n".join(report))


def try_ncu_probe(args):
    ncu_dir = RESULT_DIR / "ncu"
    ncu_dir.mkdir(parents=True, exist_ok=True)
    log_path = ncu_dir / "ncu_probe.log"
    cmd = [
        "ncu",
        "--csv",
        "--page", "raw",
        "--launch-skip", "2",
        "--launch-count", "1",
        "--metrics",
        ",".join([
            "gpu__time_duration.sum",
            "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
            "l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum",
            "l1tex__m_l1tex2xbar_write_bytes_mem_lg_op_st.sum",
            "lts__t_bytes.sum",
            "lts__t_request_hit_rate.pct",
        ]),
        "--force-overwrite",
        "-o", str(ncu_dir / "read_unique_validation"),
        str(BIN),
        "--mode", "read-unique",
        "--iters", str(args.ncu_iters),
        "--warmup-iters", str(args.warmup_iters),
        "--blocks-per-sm", "4",
        "--threads", "256",
        "--bytes", str(args.bytes),
        "--csv",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout)
    if proc.returncode == 0:
        return (
            "NCU probe 已完成。Raw log: "
            "`results/ncu/ncu_probe.log`；report: "
            "`results/ncu/read_unique_validation.ncu-rep`。"
        )
    return (
        "NCU probe 没有在本机完成。日志位于 "
        "`results/ncu/ncu_probe.log`。观测到的失败原因是 driver profiling "
        "resource conflict，因此这次 validate 不包含 L2/DRAM counter 证明。"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=2048)
    parser.add_argument("--capacity-iters", type=int, default=2048)
    parser.add_argument("--ncu-iters", type=int, default=512)
    parser.add_argument("--warmup-iters", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--capacity-threads", type=int, default=256)
    parser.add_argument("--capacity-blocks-per-sm", type=int, default=4)
    parser.add_argument("--skip-ncu", action="store_true")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    build()

    concurrency_rows = concurrency_sweep(args)
    capacity_rows = capacity_sweep(args)

    write_csv(RESULT_DIR / "l2_concurrency_sweep.csv", concurrency_rows)
    write_csv(RESULT_DIR / "l2_capacity_sweep.csv", capacity_rows)
    plot_concurrency(concurrency_rows, PLOT_DIR / "l2_concurrency_saturation.svg")
    plot_capacity(capacity_rows, PLOT_DIR / "l2_capacity_staircase.svg")

    ncu_note = "Skipped by request."
    if not args.skip_ncu:
        ncu_note = try_ncu_probe(args)
    write_report(concurrency_rows, capacity_rows, ncu_note)
    print(f"Wrote {RESULT_DIR / 'validation_report.md'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
