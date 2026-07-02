#!/usr/bin/env python3
"""Generate bank-pressure heatmaps for every transpose_2d_case backend."""

from collections import defaultdict
from math import ceil
from math import nan
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, PowerNorm


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = ROOT / "assets" / "backend_heatmaps"
SOURCE_PATH = ROOT / "src" / "transpose_2d_bench.cu"
LANES = 32
WARPS = 8
VIVID_CMAP = LinearSegmentedColormap.from_list(
    "bank_pressure_vivid",
    [
        (0.00, "#F2F2F2"),
        (0.03, "#00D9FF"),
        (0.18, "#00F57A"),
        (0.42, "#FFF000"),
        (0.70, "#FF7A00"),
        (1.00, "#FF1744"),
    ],
)
BANK_CMAP = plt.get_cmap("turbo", 32).copy()
BANK_CMAP.set_bad("#F2F2F2")
BANK_NORM = BoundaryNorm([value - 0.5 for value in range(33)], 32)

GROUPS = {
    "E0": [
        ("E0_load_pitch32", "transpose_scalar", 32, 1),
    ],
    "E1": [
        ("E1_load_pitch1", "transpose_scalar", 1, 1),
        ("E1_load_pitch2", "transpose_scalar", 2, 1),
        ("E1_load_pitch4", "transpose_scalar", 4, 1),
        ("E1_load_pitch8", "transpose_scalar", 8, 1),
        ("E1_load_pitch16", "transpose_scalar", 16, 1),
        ("E1_load_pitch31", "transpose_scalar", 31, 1),
        ("E1_load_pitch32", "transpose_scalar", 32, 1),
        ("E1_load_pitch33", "transpose_scalar", 33, 1),
    ],
    "E2": [
        ("E2_load_broadcast_same_addr", "broadcast", 32, 1),
        ("E2_load_multicast_2addr", "multicast2", 32, 1),
        ("E2_load_multicast_4addr", "multicast4", 32, 1),
        (
            "E2_load_conflict_same_bank_diff_addr",
            "same_bank_different",
            32,
            1,
        ),
    ],
    "E3": [
        ("E3_load_f32_pitch32", "transpose_scalar", 32, 1),
        ("E3_load_f32_pitch33", "transpose_scalar", 33, 1),
        ("E3_load_f32x2_pitch32", "transpose_vector", 32, 2),
        ("E3_load_f32x2_pitch33", "transpose_vector", 33, 2),
        ("E3_load_f32x4_pitch32", "transpose_vector", 32, 4),
        ("E3_load_f32x4_pitch33", "transpose_vector", 33, 4),
    ],
    "E4": [
        ("E4_load_xor_swizzle_pitch32", "xor_swizzle", 32, 1),
    ],
}


def vector_base_col(row, warp, pitch, vector_width):
    if vector_width == 2:
        base = warp * 2
        return base + ((2 - ((row * pitch + base) & 1)) & 1)
    group = warp % 7 if pitch == 33 else warp
    base = group * 4
    return base + ((4 - ((row * pitch + base) & 3)) & 3)


def requested_words(pattern, pitch, vector_width, lane, warp):
    if pattern == "transpose_scalar":
        base = lane * pitch + warp
    elif pattern == "broadcast":
        base = 0
    elif pattern == "multicast2":
        base = 0 if lane < 16 else 1
    elif pattern == "multicast4":
        base = lane // 8
    elif pattern == "same_bank_different":
        base = lane * 32
    elif pattern == "transpose_vector":
        base = lane * pitch + vector_base_col(
            lane, warp, pitch, vector_width
        )
    elif pattern == "xor_swizzle":
        base = lane * pitch + (warp ^ (lane & 31))
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return range(base, base + vector_width)


def bank_pressure(pattern, pitch, vector_width):
    matrix = []
    for warp in range(WARPS):
        words_by_bank = defaultdict(set)
        for lane in range(LANES):
            for word in requested_words(
                pattern, pitch, vector_width, lane, warp
            ):
                words_by_bank[word % 32].add(word)
        matrix.append(
            [len(words_by_bank[bank]) for bank in range(32)]
        )
    return matrix


def short_title(case_name):
    return case_name.replace("_load_", "\n").replace("transpose_", "")


def validate_cases_against_source():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    source_cases = set(re.findall(r'"(E[0-4]_load_[^"]+)"', source))
    plotted_cases = {
        case_name
        for cases in GROUPS.values()
        for case_name, _, _, _ in cases
    }
    if source_cases != plotted_cases:
        missing = sorted(source_cases - plotted_cases)
        stale = sorted(plotted_cases - source_cases)
        raise RuntimeError(
            "heatmap cases do not match transpose_2d_bench.cu: "
            f"missing={missing}, stale={stale}"
        )


def configure_cell_grid(axis, width, height):
    axis.set_xticks([value - 0.5 for value in range(width + 1)], minor=True)
    axis.set_yticks([value - 0.5 for value in range(height + 1)], minor=True)
    axis.grid(which="minor", color="white", linewidth=0.45, alpha=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)


def draw_address_layout():
    shown_columns = list(range(8)) + [None, 30, 31, 32]
    column_labels = [str(value) if value is not None else "…" for value in shown_columns]
    column_labels[-1] = "PAD"
    figure, axes = plt.subplots(
        2, 1, figsize=(17, 10), constrained_layout=True
    )
    image = None
    for axis, pitch in zip(axes, (32, 33)):
        matrix = []
        for row in range(8):
            bank_row = []
            for col in shown_columns:
                if col is None or col >= pitch:
                    bank_row.append(nan)
                else:
                    bank_row.append((row * pitch + col) % 32)
            matrix.append(bank_row)
        image = axis.imshow(
            matrix,
            cmap=BANK_CMAP,
            norm=BANK_NORM,
            interpolation="nearest",
            aspect="auto",
        )
        axis.set_title(
            f"pitch={pitch}: representative row-major address layout",
            fontsize=14,
        )
        axis.set_xlabel("Logical column (0–7, 30–31, and padding position)")
        axis.set_ylabel("Logical row")
        axis.set_xticks(range(len(shown_columns)), column_labels)
        axis.set_yticks(range(8))
        configure_cell_grid(axis, len(shown_columns), 8)
        for row in range(8):
            for display_col, col in enumerate(shown_columns):
                if col is None:
                    axis.text(
                        display_col,
                        row,
                        "…",
                        ha="center",
                        va="center",
                        color="#555555",
                        fontsize=12,
                    )
                elif col >= pitch:
                    axis.text(
                        display_col,
                        row,
                        "—",
                        ha="center",
                        va="center",
                        color="#777777",
                        fontsize=11,
                    )
                else:
                    index = row * pitch + col
                    bank = index % 32
                    prefix = "PAD\n" if pitch == 33 and col == 32 else ""
                    axis.text(
                        display_col,
                        row,
                        f"{prefix}idx {index}\nB{bank}",
                        ha="center",
                        va="center",
                        color="white" if bank <= 2 or bank >= 29 else "#111111",
                        fontsize=7.5,
                    )
    figure.colorbar(
        image,
        ax=axes,
        label="Bank id",
        ticks=range(0, 32, 4),
        shrink=0.9,
    )
    figure.suptitle(
        "Conceptual shared-memory layout: index = row × pitch + column",
        fontsize=17,
    )
    output = OUTPUT_DIR / "address_layout_pitch32_vs_pitch33.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote {output}")


def draw_column_read():
    figure, axes = plt.subplots(
        1, 2, figsize=(17, 11), constrained_layout=True
    )
    for axis, pitch in zip(axes, (32, 33)):
        matrix = [[nan for _ in range(32)] for _ in range(32)]
        indices = {}
        for lane in range(32):
            index = lane * pitch
            bank = index % 32
            matrix[lane][bank] = bank
            indices[(lane, bank)] = index
        axis.imshow(
            matrix,
            cmap=BANK_CMAP,
            norm=BANK_NORM,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(
            f"pitch={pitch}, fixed column=0",
            fontsize=14,
        )
        axis.set_xlabel("Bank")
        axis.set_ylabel("Lane / logical row")
        axis.set_xticks(range(0, 32, 4))
        axis.set_yticks(range(0, 32, 4))
        configure_cell_grid(axis, 32, 32)
        for (lane, bank), index in indices.items():
            axis.text(
                bank,
                lane,
                str(index),
                ha="center",
                va="center",
                color="white" if bank <= 2 or bank >= 29 else "#111111",
                fontsize=5.2,
                rotation=90 if index >= 100 else 0,
            )
    figure.suptitle(
        "Column read: each active cell contains the linear word index",
        fontsize=17,
    )
    output = OUTPUT_DIR / "column_read_pitch32_vs_pitch33.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote {output}")


def draw_xor_swizzle_address():
    figure, axes = plt.subplots(
        2, 1, figsize=(18, 7.5), constrained_layout=True
    )
    layouts = (
        (
            "Pitch-32 transpose access without swizzle",
            lambda warp, lane: warp,
        ),
        (
            "Pitch-32 XOR swizzle: physical_col = warp ^ lane",
            lambda warp, lane: warp ^ lane,
        ),
    )
    image = None
    for axis, (title, physical_col_fn) in zip(axes, layouts):
        matrix = [
            [physical_col_fn(warp, lane) for lane in range(32)]
            for warp in range(8)
        ]
        image = axis.imshow(
            matrix,
            cmap=BANK_CMAP,
            norm=BANK_NORM,
            interpolation="nearest",
            aspect="auto",
        )
        axis.set_title(title, fontsize=14)
        axis.set_xlabel("Lane")
        axis.set_ylabel("Warp")
        axis.set_xticks(range(0, 32, 2))
        axis.set_yticks(range(8))
        configure_cell_grid(axis, 32, 8)
        for warp, row in enumerate(matrix):
            for lane, physical_col in enumerate(row):
                axis.text(
                    lane,
                    warp,
                    str(physical_col),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if physical_col <= 2 or physical_col >= 29
                        else "#111111"
                    ),
                    fontsize=6,
                )
    figure.colorbar(
        image,
        ax=axes,
        label="Physical column / bank id",
        ticks=range(0, 32, 4),
        shrink=0.9,
    )
    figure.suptitle(
        "E4 load-address mapping: base_index = lane × 32 + physical_col",
        fontsize=17,
    )
    output = OUTPUT_DIR / "xor_swizzle_address_mapping.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote {output}")


def draw_xor_swizzle_address_layout():
    figure, axes = plt.subplots(
        1, 2, figsize=(17, 20), constrained_layout=True
    )
    layouts = (
        (
            "Ordinary pitch-32 address layout\nphysical_col = warp",
            lambda row, logical_col: logical_col,
        ),
        (
            "XOR-swizzled pitch-32 address layout\n"
            "physical_col = warp ^ lane",
            lambda row, logical_col: logical_col ^ row,
        ),
    )
    image = None
    for axis, (title, physical_col_fn) in zip(axes, layouts):
        matrix = []
        cells = {}
        for row in range(32):
            bank_row = []
            for logical_col in range(8):
                physical_col = physical_col_fn(row, logical_col)
                index = row * 32 + physical_col
                bank = index % 32
                bank_row.append(bank)
                cells[(row, logical_col)] = (physical_col, index, bank)
            matrix.append(bank_row)
        image = axis.imshow(
            matrix,
            cmap=BANK_CMAP,
            norm=BANK_NORM,
            interpolation="nearest",
            aspect="auto",
        )
        axis.set_title(title, fontsize=14)
        axis.set_xlabel("Warp / logical column")
        axis.set_ylabel("Lane / logical row")
        axis.set_xticks(range(8))
        axis.set_yticks(range(32))
        configure_cell_grid(axis, 8, 32)
        for (row, logical_col), (physical_col, index, bank) in cells.items():
            axis.text(
                logical_col,
                row,
                f"P{physical_col}\nidx {index}\nB{bank}",
                ha="center",
                va="center",
                color="white" if bank <= 2 or bank >= 29 else "#111111",
                fontsize=5.4,
            )
    figure.colorbar(
        image,
        ax=axes,
        label="Bank id",
        ticks=range(0, 32, 4),
        shrink=0.9,
    )
    figure.suptitle(
        "E4 address view matching the actual block: 32 lanes × 8 warps",
        fontsize=17,
    )
    output = OUTPUT_DIR / "address_layout_pitch32_vs_xor_swizzle.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote {output}")


def draw_group(group_name, cases):
    columns = 2 if len(cases) > 1 else 1
    rows = ceil(len(cases) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(14 if columns == 2 else 9, 3.2 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for axis, (case_name, pattern, pitch, vector_width) in zip(
        axes.flat, cases
    ):
        matrix = bank_pressure(pattern, pitch, vector_width)
        image = axis.imshow(
            matrix,
            cmap=VIVID_CMAP,
            norm=PowerNorm(gamma=0.4, vmin=0, vmax=32),
            interpolation="nearest",
            aspect="auto",
        )
        maximum = max(max(row) for row in matrix)
        axis.set_title(f"{short_title(case_name)}  (max={maximum})", fontsize=10)
        axis.set_xlabel("Shared-memory bank")
        axis.set_ylabel("Warp")
        axis.set_xticks(range(0, 32, 4))
        axis.set_yticks(range(8))
        axis.set_xticks([value - 0.5 for value in range(33)], minor=True)
        axis.set_yticks([value - 0.5 for value in range(9)], minor=True)
        axis.grid(which="minor", color="white", linewidth=0.25, alpha=0.45)
        axis.tick_params(which="minor", bottom=False, left=False)
        for warp, row in enumerate(matrix):
            for bank, value in enumerate(row):
                axis.text(
                    bank,
                    warp,
                    str(value),
                    ha="center",
                    va="center",
                    color="#5A5A5A" if value == 0 else "#111111",
                    fontsize=5.2,
                )

    for axis in list(axes.flat)[len(cases) :]:
        axis.set_visible(False)
    if image is not None:
        figure.colorbar(
            image,
            ax=[axis for axis in axes.flat if axis.get_visible()],
            label="Distinct word addresses requested from this bank",
            shrink=0.9,
        )
    figure.suptitle(
        f"{group_name} bank request pressure per warp and bank "
        "(common scale: 0–32)",
        fontsize=14,
    )
    output = OUTPUT_DIR / f"{group_name.lower()}_backend_heatmaps.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote {output}")


def main():
    validate_cases_against_source()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_address_layout()
    draw_column_read()
    draw_xor_swizzle_address()
    draw_xor_swizzle_address_layout()
    for group_name, cases in GROUPS.items():
        draw_group(group_name, cases)


if __name__ == "__main__":
    main()
