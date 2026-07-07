#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

FAMILIES = {
    "lop3": {
        "result_dir": ROOT / "results" / "bank_scan",
        "output": ASSETS / "register_bank_stride_scan.png",
        "source_output": ASSETS / "register_bank_lop3_source_pressure.png",
        "throughput_output": ASSETS / "register_bank_lop3_throughput_stride.png",
        "latency_output": ASSETS / "register_bank_lop3_latency_heatmap.png",
        "opcode_label": "LOP3",
        "title": "NVIDIA Thor Physical Register Stride Scan",
        "tuple_text": "Tuple: Rbase, R(base+s), R(base+2s)",
    },
    "ffma": {
        "result_dir": ROOT / "results" / "bank_scan_ffma",
        "output": ASSETS / "register_bank_stride_scan_ffma.png",
        "source_output": ASSETS / "register_bank_ffma_source_pressure.png",
        "throughput_output": ASSETS / "register_bank_ffma_throughput_stride.png",
        "latency_output": ASSETS / "register_bank_ffma_latency_heatmap.png",
        "opcode_label": "FFMA",
        "title": "NVIDIA Thor Physical Register Stride Scan (FFMA)",
        "tuple_text": "Tuple: Rbase, R(base+s), R(base+2s), Rbase",
    },
}


def load_rows(path):
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def merged_rows(result_dir):
    result_by_case = {
        row["case"]: row for row in load_rows(result_dir / "results.csv")
    }
    rows = []
    for metadata in load_rows(result_dir / "manifest.csv"):
        result = result_by_case.get(metadata["case"])
        if result is None:
            raise SystemExit(f"Missing result for {metadata['case']}")
        rows.append({**metadata, **result})
    return rows


def row_groups(rows):
    source_rows = sorted(
        (row for row in rows if row["category"] == "source_count"),
        key=lambda row: row["case"],
    )
    slot_rows = sorted(
        (row for row in rows if row["category"] == "slot_permutation"),
        key=lambda row: row["case"],
    )
    throughput_rows = sorted(
        (row for row in rows if row["category"] == "throughput_stride"),
        key=lambda row: int(row["stride"]),
    )
    latency_rows = [row for row in rows if row["category"] == "latency_stride"]
    return source_rows, slot_rows, throughput_rows, latency_rows


def latency_matrix(latency_rows, np):
    bases = sorted({int(row["base"]) for row in latency_rows})
    strides = sorted({int(row["stride"]) for row in latency_rows})
    matrix = np.empty((len(bases), len(strides)))
    for row in latency_rows:
        base_index = bases.index(int(row["base"]))
        stride_index = strides.index(int(row["stride"]))
        matrix[base_index, stride_index] = float(row["median_cycles_per_op"])
    row_minimum = matrix.min(axis=1, keepdims=True)
    delta_percent = (matrix / row_minimum - 1.0) * 100.0
    return bases, strides, matrix, delta_percent


def save_figure(figure, output):
    ASSETS.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )


def add_value_labels(axis, bars, values, padding):
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + padding,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_source_pressure(config, source_rows, plt):
    labels = [
        "1 RF",
        "2 same\nparity",
        "3 mixed\nparity",
        "3 same\nparity",
    ]
    values = [float(row["median_cycles_per_op"]) for row in source_rows]
    figure, axis = plt.subplots(figsize=(10.5, 4.8), facecolor="white")
    axis.set_facecolor("white")
    bars = axis.bar(labels, values, color="#4E79A7", width=0.62)
    axis.set_title(f"{config['opcode_label']} register-read pressure", fontsize=15)
    axis.set_ylabel("Median cycles/op")
    axis.grid(axis="y", alpha=0.28)
    axis.spines[["top", "right"]].set_visible(False)
    padding = max(max(values) * 0.015, 0.02)
    axis.set_ylim(0, max(values) + padding * 4.0)
    add_value_labels(axis, bars, values, padding)
    save_figure(figure, config["source_output"])
    plt.close(figure)


def plot_throughput_stride(config, strides, throughput_values, slot_rows, plt):
    slot_values = [float(row["median_cycles_per_op"]) for row in slot_rows]
    figure, axis = plt.subplots(figsize=(10.8, 4.8), facecolor="white")
    axis.set_facecolor("white")
    axis.plot(
        strides,
        throughput_values,
        color="#4E79A7",
        marker="o",
        markersize=5.5,
        linewidth=2,
    )
    axis.set_title(f"{config['opcode_label']} four-chain throughput scan", fontsize=15)
    axis.set_xlabel("Physical-register stride")
    axis.set_ylabel("Median cycles/op")
    axis.set_xticks(strides)
    axis.grid(alpha=0.28)
    axis.spines[["top", "right"]].set_visible(False)
    span = max(throughput_values) - min(throughput_values)
    padding = max(span * 2.0, 0.01)
    axis.set_ylim(min(throughput_values) - padding, max(throughput_values) + padding)
    axis.text(
        0.99,
        0.05,
        "slot permutations: " + ", ".join(f"{value:.4f}" for value in slot_values),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    save_figure(figure, config["throughput_output"])
    plt.close(figure)


def plot_latency_heatmap(config, bases, strides, matrix, delta_percent, plt):
    figure, axis = plt.subplots(figsize=(11.8, 4.8), facecolor="white")
    axis.set_facecolor("white")
    color_limit = max(float(delta_percent.max()), 0.001)
    image = axis.imshow(
        delta_percent,
        cmap="YlOrRd",
        vmin=0,
        vmax=color_limit,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_title(f"{config['opcode_label']} single-chain base x stride scan", fontsize=15)
    axis.set_xlabel("Physical-register stride")
    axis.set_xticks(range(len(strides)), strides)
    axis.set_yticks(range(len(bases)), [f"R{base}" for base in bases])
    for row_index in range(len(bases)):
        for column_index in range(len(strides)):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=(
                    "white"
                    if delta_percent[row_index, column_index] > color_limit * 0.58
                    else "#222222"
                ),
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Slowdown vs. row minimum (%)")
    save_figure(figure, config["latency_output"])
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    args = parser.parse_args()
    config = FAMILIES[args.family]

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise SystemExit("matplotlib and numpy are required for plotting")

    rows = merged_rows(config["result_dir"])
    source_rows, slot_rows, throughput_rows, latency_rows = row_groups(rows)
    bases, strides, matrix, delta_percent = latency_matrix(latency_rows, np)
    throughput_values = [
        float(row["median_cycles_per_op"]) for row in throughput_rows
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#28332F",
            "axes.labelcolor": "#28332F",
            "xtick.color": "#28332F",
            "ytick.color": "#28332F",
        }
    )
    plot_source_pressure(config, source_rows, plt)
    plot_throughput_stride(config, strides, throughput_values, slot_rows, plt)
    plot_latency_heatmap(config, bases, strides, matrix, delta_percent, plt)

    figure = plt.figure(figsize=(15.2, 9.2), facecolor="#F5F1E8")
    grid = figure.add_gridspec(
        2, 2, height_ratios=[0.9, 1.25], hspace=0.42, wspace=0.28
    )
    source_axis = figure.add_subplot(grid[0, 0])
    throughput_axis = figure.add_subplot(grid[0, 1])
    heatmap_axis = figure.add_subplot(grid[1, :])
    for axis in (source_axis, throughput_axis, heatmap_axis):
        axis.set_facecolor("#FCFAF5")
        axis.spines[["top", "right"]].set_visible(False)

    source_values = [float(row["median_cycles_per_op"]) for row in source_rows]
    bars = source_axis.bar(
        ["1 RF", "2 RF\nsame parity", "3 RF\nmixed parity", "3 RF\nsame parity"],
        source_values,
        color=["#287A74", "#4F8A5B", "#D39A2C", "#C54A34"],
        width=0.62,
        edgecolor="#FCFAF5",
        linewidth=1.4,
        zorder=3,
    )
    source_axis.grid(axis="y", color="#A8B0A8", alpha=0.28)
    source_axis.set_ylabel(f"Median cycles per {config['opcode_label']}")
    source_axis.set_title("Register-read pressure", loc="left", fontsize=14, pad=32)
    source_axis.text(
        0,
        1.01,
        "Same opcode and dependency structure; RZ removes RF reads",
        transform=source_axis.transAxes,
        fontsize=9.5,
        color="#60706A",
    )
    source_padding = max(max(source_values) * 0.04, 0.05)
    source_axis.set_ylim(0, max(source_values) + source_padding * 2.4)
    for bar, value in zip(bars, source_values):
        source_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + source_padding * 0.25,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    throughput_axis.plot(
        strides,
        throughput_values,
        color="#287A74",
        marker="o",
        markersize=5.5,
        linewidth=2,
        zorder=3,
    )
    throughput_axis.grid(color="#A8B0A8", alpha=0.28)
    throughput_axis.set_xticks(strides)
    throughput_axis.set_xlabel("Physical-register stride")
    throughput_axis.set_ylabel(f"Median cycles per {config['opcode_label']}")
    throughput_axis.set_title("Four-chain throughput scan", loc="left", fontsize=14, pad=32)
    throughput_axis.text(
        0,
        1.01,
        config["tuple_text"],
        transform=throughput_axis.transAxes,
        fontsize=9.5,
        color="#60706A",
    )
    throughput_span = max(throughput_values) - min(throughput_values)
    throughput_padding = max(throughput_span * 2.0, 0.01)
    throughput_axis.set_ylim(
        min(throughput_values) - throughput_padding,
        max(throughput_values) + throughput_padding,
    )
    slot_values = [float(row["median_cycles_per_op"]) for row in slot_rows]
    throughput_axis.text(
        0.98,
        0.06,
        "Mixed-parity source-slot permutations\n"
        + " . ".join(f"{value:.4f}" for value in slot_values),
        transform=throughput_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
        color="#60706A",
    )

    color_limit = max(float(delta_percent.max()), 0.001)
    image = heatmap_axis.imshow(
        delta_percent,
        cmap="YlOrRd",
        vmin=0,
        vmax=color_limit,
        aspect="auto",
        interpolation="nearest",
    )
    heatmap_axis.set_xticks(range(len(strides)), strides)
    heatmap_axis.set_yticks(
        range(len(bases)), [f"Accumulator R{base}" for base in bases]
    )
    heatmap_axis.set_xlabel("Physical-register stride")
    heatmap_axis.set_title("Single-chain base x stride scan", loc="left", fontsize=14, pad=22)
    heatmap_axis.text(
        0,
        1.01,
        "Cell color: slowdown relative to the fastest stride in that row",
        transform=heatmap_axis.transAxes,
        fontsize=9.5,
        color="#60706A",
    )
    for row_index in range(len(bases)):
        for column_index in range(len(strides)):
            heatmap_axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=7.3,
                color=(
                    "white"
                    if delta_percent[row_index, column_index] > color_limit * 0.58
                    else "#28332F"
                ),
            )
    colorbar = figure.colorbar(image, ax=heatmap_axis, pad=0.015)
    colorbar.set_label("Slowdown vs. row minimum (%)")

    figure.suptitle(
        config["title"],
        x=0.045,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#1F2925",
    )
    iterations = rows[0]["iters"]
    figure.text(
        0.045,
        0.018,
        f"CUDA 13 . sm_110 . {iterations} iterations . "
        f"128 verified {config['opcode_label']} per iteration . "
        ".reuse disabled . 0 local bytes",
        color="#60706A",
        fontsize=9,
    )
    save_figure(figure, config["output"])
    plt.close(figure)

    fastest_stride = strides[int(np.argmin(throughput_values))]
    slowest_stride = strides[int(np.argmax(throughput_values))]
    throughput_delta = (max(throughput_values) / min(throughput_values) - 1.0) * 100.0
    print(
        "Source-count cycles/op: "
        + ", ".join(
            f"{row['case']}={value:.6f}"
            for row, value in zip(source_rows, source_values)
        )
    )
    print(
        "Slot-permutation cycles/op: "
        + ", ".join(
            f"{row['case']}={value:.6f}"
            for row, value in zip(slot_rows, slot_values)
        )
    )
    print(
        f"Throughput stride range: fastest s={fastest_stride}, "
        f"slowest s={slowest_stride}, spread={throughput_delta:.6f}%"
    )
    print(
        "Maximum single-chain row-relative slowdown: "
        f"{float(delta_percent.max()):.6f}%"
    )
    print(f"Wrote {config['source_output']}")
    print(f"Wrote {config['throughput_output']}")
    print(f"Wrote {config['latency_output']}")
    print(f"Wrote {config['output']}")


if __name__ == "__main__":
    main()
