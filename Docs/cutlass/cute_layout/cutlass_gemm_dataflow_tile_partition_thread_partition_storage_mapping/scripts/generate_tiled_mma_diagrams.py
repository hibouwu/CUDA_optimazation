#!/usr/bin/env python3
"""Generate the tcgen05 dataflow diagrams from explicit CuTe-style layouts.

The grid geometry is rendered by tensor-layouts.  Matplotlib adds the parts
that are not layouts: storage-level frames, operation arrows, and callouts.
The article keeps its original hand-drawn overview as a separate JPG.  The
generated figures deliberately distinguish three kinds of claims:

* exact mathematical coordinates and logical tile decomposition;
* logical CuTe layouts compressed to readable blocks;
* conceptual views whose concrete Thread-Value ownership needs a real kernel
  object before it can be called a hardware mapping.

The article follows the CUTLASS CuTe example (CTA tile 128x256x64, MMA
instruction 128x256x16, three SMEM stages).  The SM100 UMMA atom supplied by
tensor-layouts is used only for the shared 128x256x16 instruction shape and
coarse logical projections; it is not treated as a proof of SM110
microarchitecture, TMEM-bank mapping, or tcgen05.ld thread ownership.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

try:
    from tensor_layouts import Layout
    from tensor_layouts.atoms_nv import SM100_128x256x16_F32F16F16_SS
    import tensor_layouts.viz as tlviz
except ImportError as exc:  # pragma: no cover - exercised by users without the extra
    raise SystemExit(
        "tensor-layouts visualization support is required. Install it with:\n"
        "  python3 -m pip install -r scripts/requirements.txt"
    ) from exc


# The article's single, continuous example.
PROBLEM_MNK = (512, 768, 384)
CTA_MNK = (128, 256, 64)
MMA_MNK = SM100_128x256x16_F32F16F16_SS.shape_mnk
STAGES = 3
TARGET_MK = (133, 70)
TARGET_ID = TARGET_MK[0] * PROBLEM_MNK[2] + TARGET_MK[1] + 1
TARGET_TILE = (TARGET_MK[0] // CTA_MNK[0], TARGET_MK[1] // CTA_MNK[2])
TARGET_LOCAL = (TARGET_MK[0] % CTA_MNK[0], TARGET_MK[1] % CTA_MNK[2])

# Stable visual language.  Cell-band colors express MMA_K groups; panel title
# colors express storage levels.  Orange always marks A[133,70].
INK = "#172033"
MUTED = "#596579"
GRID = "#344054"
VIEW_EDGE = "#64748B"
GMEM = ("#1D4ED8", "#E8F1FF")
SMEM = ("#087F5B", "#E7F8F0")
TMEM = ("#6D4CC2", "#F0EAFF")
RMEM = ("#A15C00", "#FFF1D6")
DESC = ("#7C3AED", "#F5EEFF")
TARGET = "#F59E0B"
TARGET_EDGE = "#B45309"
MMA_K_COLORS = ("#DDE8FF", "#DDF7E5", "#FFF3C4", "#FFE0E0")
CTA_COLORS = ("#DCEBFF", "#DDF7EA")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "text.color": INK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": "tcgen05-layout-figures-v1",
    }
)


def update_readme_block(readme: Path, key: str, image_path: Path, alt: str) -> None:
    """Replace one generated Markdown image block without touching prose."""
    begin = f"<!-- BEGIN GENERATED DIAGRAM: {key} -->"
    end = f"<!-- END GENERATED DIAGRAM: {key} -->"
    relative_image = Path(os.path.relpath(image_path, readme.parent)).as_posix()
    replacement = f"{begin}\n![{alt}]({relative_image})\n{end}"
    content = readme.read_text(encoding="utf-8")
    pattern = re.compile(rf"{re.escape(begin)}.*?{re.escape(end)}", re.DOTALL)
    updated, count = pattern.subn(replacement, content, count=1)
    if count != 1:
        raise RuntimeError(f"README marker block not found or duplicated: {key}")
    readme.write_text(updated, encoding="utf-8")


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save deterministic, editable SVG text and close the Matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    # Matplotlib emits spaces before newlines inside many SVG path definitions.
    # Normalize them so generated assets pass repository whitespace checks.
    svg = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")


def row_major(rows: int, cols: int) -> Layout:
    return Layout((rows, cols), (cols, 1))


def col_major(rows: int, cols: int) -> Layout:
    return Layout((rows, cols), (1, rows))


def all_labels(layout: Layout, rows: int, cols: int, fn: Callable[[int, int], str]) -> dict[int, str]:
    """Build cell labels indexed by layout offset, as tensor-layouts expects."""
    return {int(layout(r, c)): fn(r, c) for r in range(rows) for c in range(cols)}


def one_hot_mask(rows: int, cols: int, row: int, col: int) -> np.ndarray:
    mask = np.zeros((rows, cols), dtype=bool)
    mask[row, col] = True
    return mask


def style_axis_labels(
    ax: plt.Axes,
    row_label: Callable[[int], str] | None = None,
    col_label: Callable[[int], str] | None = None,
    *,
    hide: bool = False,
) -> None:
    """Restyle the blue numeric indices emitted by tensor-layouts."""
    for text in ax.texts:
        color = str(text.get_color()).lower()
        if color not in {"blue", "#0000ff"}:
            continue
        x, y = text.get_position()
        if hide:
            text.set_visible(False)
            continue
        if x < 0 and row_label is not None:
            text.set_text(row_label(int(round(y - 0.5))))
        elif y < 0 and col_label is not None:
            text.set_text(col_label(int(round(x - 0.5))))
        text.set_color(MUTED)
        text.set_fontsize(8.5)


def recolor_cells(
    ax: plt.Axes,
    rows: int,
    cols: int,
    color_fn: Callable[[int, int], str],
) -> None:
    """Apply the article palette to the base cell patches of a layout grid."""
    base = ax.patches[: rows * cols]
    if len(base) != rows * cols:
        raise RuntimeError("unexpected tensor-layouts patch count")
    for index, patch in enumerate(base):
        row, col = divmod(index, cols)
        patch.set_facecolor(color_fn(row, col))
        patch.set_edgecolor(GRID)
        patch.set_linewidth(0.75)


def title_as_storage(ax: plt.Axes, storage: tuple[str, str]) -> None:
    edge, face = storage
    ax.title.set_color(edge)
    ax.title.set_bbox(
        {"boxstyle": "round,pad=0.32", "facecolor": face, "edgecolor": edge, "linewidth": 1.0}
    )


def add_target_point(ax: plt.Axes, x: float, y: float, label: str | None = None) -> None:
    ax.plot(x, y, marker="o", markersize=7, color=TARGET, markeredgecolor=TARGET_EDGE, zorder=15)
    if label:
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(12, -26),
            textcoords="offset points",
            fontsize=8.5,
            color=TARGET_EDGE,
            arrowprops={"arrowstyle": "->", "color": TARGET_EDGE, "lw": 1.2},
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": TARGET_EDGE, "alpha": 0.96},
            zorder=20,
        )


def add_panel_arrow(
    fig: plt.Figure,
    left: plt.Axes,
    right: plt.Axes,
    label: str,
    *,
    color: str = INK,
    linestyle: str = "-",
    y: float = 0.50,
) -> None:
    left_pos, right_pos = left.get_position(), right.get_position()
    y_fig = left_pos.y0 + y * left_pos.height
    arrow = FancyArrowPatch(
        # The tensor-layouts panels leave a very narrow gap.  Extend the
        # arrow slightly into both panels so its head cannot fold backwards
        # when the mutation scale is wider than the inter-panel distance.
        (left_pos.x1 - 0.018, y_fig),
        (right_pos.x0 + 0.018, y_fig),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.2,
        color=color,
        linestyle=linestyle,
        zorder=30,
    )
    fig.add_artist(arrow)
    x = (left_pos.x1 + right_pos.x0) / 2
    fig.text(
        x,
        y_fig + 0.035,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        color=color,
        bbox={"boxstyle": "round,pad=0.20", "fc": "white", "ec": "none", "alpha": 0.95},
    )


def footer(fig: plt.Figure, text: str, *, color: str = MUTED, y: float = 0.012) -> None:
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8.5, color=color)


def draw_case0(path: Path) -> None:
    """Locate A[133,70] first at CTA-tile scale, then at element scale."""
    tile_layout = row_major(4, 6)
    zoom_layout = row_major(8, 8)
    tile_labels = all_labels(
        tile_layout,
        4,
        6,
        lambda r, c: "selected tile\n(1,1)" if (r, c) == TARGET_TILE else "",
    )
    zoom_labels = all_labels(
        zoom_layout,
        8,
        8,
        lambda r, c: (
            f"{(128 + r) * PROBLEM_MNK[2] + (64 + c) + 1}\nTARGET"
            if (r, c) == (5, 6)
            else str((128 + r) * PROBLEM_MNK[2] + (64 + c) + 1)
        ),
    )
    fig = tlviz._build_composite_figure(
        [
            (
                tile_layout,
                {
                    "cell_labels": tile_labels,
                    "highlight_mask": one_hot_mask(4, 6, *TARGET_TILE),
                    "highlight_facecolor": "#FFE6B0",
                    "highlight_edgecolor": TARGET_EDGE,
                },
            ),
            (
                zoom_layout,
                {
                    "cell_labels": zoom_labels,
                    "highlight_mask": one_hot_mask(8, 8, 5, 6),
                    "highlight_facecolor": "#FFE6B0",
                    "highlight_edgecolor": TARGET_EDGE,
                },
            ),
        ],
        arrangement="horizontal",
        titles=["A (512x384): 4x6 CTA tiles", "Element zoom: m=128..135, k=64..71"],
        main_title="Find one element at two coordinate scales",
        panel_size=(6.1, 5.8),
        colorize=False,
        num_colors=1,
    )
    left, right = fig.axes[:2]
    recolor_cells(left, 4, 6, lambda _r, _c: GMEM[1])
    recolor_cells(right, 8, 8, lambda _r, _c: "#F6F8FB")
    style_axis_labels(left, lambda r: f"m_tile={r}", lambda c: f"q={c}")
    style_axis_labels(right, lambda r: str(128 + r), lambda c: str(64 + c))
    title_as_storage(left, GMEM)
    title_as_storage(right, GMEM)
    add_target_point(left, 1.5, 1.5)
    add_target_point(right, 6.5, 5.5)
    right.text(
        8.30,
        5.5,
        f"<- A[133,70]\nid={TARGET_ID}\nlocal=(5,6)",
        ha="left",
        va="center",
        fontsize=8.5,
        color=TARGET_EDGE,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": TARGET_EDGE, "alpha": 0.96},
        clip_on=False,
        zorder=20,
    )
    add_panel_arrow(fig, left, right, "zoom selected tile -> 8x8 window", color=TARGET_EDGE)
    footer(fig, "Orange marks the same mathematical coordinate at tile scale and element scale.")
    save_figure(fig, path)


def draw_case1(path: Path) -> None:
    """Block projection of the real 128x256x16 single-CTA UMMA atom."""
    # One display cell is 32x16 for A, 32x16 for B, and 32x32 for C.
    a = col_major(4, 1)
    b = col_major(8, 1)
    c = col_major(4, 8)
    fig = tlviz._build_gemm_figure(
        a,
        b,
        c,
        main_title="CtaGroup.ONE: CTA rank 0 owns one 128x256x16 tcgen05 atom",
        cell_labels=False,
        colorize=False,
        num_colors=1,
    )
    blank, b_ax, a_ax, c_ax = fig.axes
    recolor_cells(a_ax, 4, 1, lambda _r, _c: SMEM[1])
    recolor_cells(b_ax, 1, 8, lambda _r, _c: SMEM[1])
    recolor_cells(c_ax, 4, 8, lambda _r, _c: TMEM[1])
    a_ax.set_title("A from SMEM: 128x16", color=SMEM[0], weight="bold")
    b_ax.set_title("B^T from SMEM: 16x256", color=SMEM[0], weight="bold")
    c_ax.set_title("C/D in TMEM: 128x256", color=TMEM[0], weight="bold")
    for ax, storage in ((a_ax, SMEM), (b_ax, SMEM), (c_ax, TMEM)):
        title_as_storage(ax, storage)
        style_axis_labels(ax, hide=False)
    blank.axis("off")
    blank.text(
        0.5,
        0.50,
        "Atom 0\natom_layout_mnk\n= (1,1,1)\nno repeat / permutation",
        ha="center",
        va="center",
        fontsize=8.8,
        weight="bold",
        color=TMEM[0],
        transform=blank.transAxes,
        bbox={"boxstyle": "round,pad=0.45", "fc": TMEM[1], "ec": TMEM[0], "lw": 1.5},
    )
    footer(
        fig,
        "Block projection of the shared 128x256x16 instruction shape; cells are not CUDA Thread-Value ownership.",
    )
    save_figure(fig, path)


def draw_case2(path: Path) -> None:
    """Official CTA-pair logical split: two ranks own the M halves."""
    layout = row_major(8, 4)
    labels = all_labels(layout, 8, 4, lambda _r, _c: "")
    fig = tlviz._build_composite_figure(
        [(layout, {"cell_labels": labels})],
        titles=["Pair-level output tile: M=256, N=256 (compressed to 8x4 blocks)"],
        main_title="CtaGroup.TWO: two CTA ranks cooperate on one MMA",
        panel_size=(7.6, 6.2),
        colorize=False,
        num_colors=1,
    )
    ax = fig.axes[0]
    recolor_cells(ax, 8, 4, lambda r, _c: CTA_COLORS[0 if r < 4 else 1])
    style_axis_labels(ax, hide=True)
    ax.text(-0.75, 2.0, "CTA rank 0\nM rows 0..127", ha="right", va="center", fontsize=9.5, color=GMEM[0])
    ax.text(-0.75, 6.0, "CTA rank 1\nM rows 128..255", ha="right", va="center", fontsize=9.5, color=SMEM[0])
    ax.plot([-0.10, 4.10], [4, 4], color=INK, linewidth=2.2, clip_on=False)
    ax.add_patch(
        FancyBboxPatch(
            (4.55, 2.55),
            1.65,
            2.90,
            boxstyle="round,pad=0.10",
            facecolor=SMEM[1],
            edgecolor=SMEM[0],
            linewidth=1.5,
            clip_on=False,
        )
    )
    ax.text(5.38, 3.52, "B operand", ha="center", va="center", fontsize=10, weight="bold", color=SMEM[0], clip_on=False)
    ax.text(5.38, 4.20, "jointly consumed", ha="center", va="center", fontsize=8.2, color=MUTED, clip_on=False)
    ax.annotate("", xy=(4.53, 3.25), xytext=(4.05, 2.0), arrowprops={"arrowstyle": "-", "color": SMEM[0]}, clip_on=False)
    ax.annotate("", xy=(4.53, 4.75), xytext=(4.05, 6.0), arrowprops={"arrowstyle": "-", "color": SMEM[0]}, clip_on=False)
    footer(
        fig,
        "CuTe CTA-pair logical partition. It is not a claim that this repository's unverified 2-SM readback mapping is correct.",
    )
    save_figure(fig, path)


def draw_case3(path: Path) -> None:
    """Show atom_layout_mnk=(2,2,1) as atom count and coverage, not fake elements."""
    base = row_major(1, 1)
    repeated = row_major(2, 2)
    base_labels = {0: "one atom\n128x256x16"}
    repeated_labels = all_labels(
        repeated,
        2,
        2,
        lambda m, n: f"m_rep={m}\nn_rep={n}\n128x256x16",
    )
    fig = tlviz._build_composite_figure(
        [(base, {"cell_labels": base_labels}), (repeated, {"cell_labels": repeated_labels})],
        titles=["One atom", "atom_layout_mnk=(2,2,1)"],
        main_title="Repeat changes coverage by adding atoms",
        panel_size=(5.3, 4.2),
        colorize=False,
        num_colors=1,
    )
    left, right = fig.axes[:2]
    recolor_cells(left, 1, 1, lambda _r, _c: TMEM[1])
    recolor_cells(right, 2, 2, lambda r, c: ("#E7E0FF", "#DCEBFF", "#DDF7EA", "#FFF0CC")[r * 2 + c])
    style_axis_labels(left, hide=True)
    style_axis_labels(right, hide=True)
    add_panel_arrow(fig, left, right, "repeat Mx2, Nx2, Kx1", color=TMEM[0])
    right.text(1.0, 2.38, "N coverage: 2 x 256 = 512", ha="center", va="center", fontsize=8.8, color=MUTED, clip_on=False)
    right.text(2.42, 1.0, "M coverage\n2 x 128 = 256", ha="center", va="center", fontsize=8.8, color=MUTED, rotation=90, clip_on=False)
    footer(fig, "Four atoms, unchanged K=16. CTA grid and SMEM pipeline are separate objects.")
    save_figure(fig, path)


def draw_case4(path: Path) -> None:
    """Before/after view of the official M-row interleave permutation."""
    # The real mapping behind the compressed four-band diagram.
    m_layout = Layout((128, 2, 2), (1, 256, 128))
    assert [
        int(m_layout(0, cta_rank, m_tile))
        for m_tile in range(2)
        for cta_rank in range(2)
    ] == [0, 256, 128, 384]

    before = row_major(4, 1)
    after = row_major(4, 1)
    before_items = [
        ("T0 top", "M=0..127", 0),
        ("T0 bottom", "M=128..255", 0),
        ("T1 top", "M=256..383", 1),
        ("T1 bottom", "M=384..511", 1),
    ]
    after_items = [
        ("T0 top", "band 0", 0),
        ("T1 top", "band 1", 1),
        ("T0 bottom", "band 2", 0),
        ("T1 bottom", "band 3", 1),
    ]
    before_labels = {i: f"{name}\n{rows}" for i, (name, rows, _tile) in enumerate(before_items)}
    after_labels = {i: f"{name}\n{rows}" for i, (name, rows, _tile) in enumerate(after_items)}
    fig = tlviz._build_composite_figure(
        [(before, {"cell_labels": before_labels}), (after, {"cell_labels": after_labels})],
        titles=["Default: atoms are contiguous", "After permutation_mnk: halves interleave"],
        main_title="Official CtaGroup.TWO permutation: inst_M=256, total M=512",
        panel_size=(5.6, 5.0),
        colorize=False,
        num_colors=1,
    )
    left, right = fig.axes[:2]
    recolor_cells(left, 4, 1, lambda r, _c: CTA_COLORS[before_items[r][2]])
    recolor_cells(right, 4, 1, lambda r, _c: CTA_COLORS[after_items[r][2]])
    style_axis_labels(left, hide=True)
    style_axis_labels(right, hide=True)
    add_panel_arrow(fig, left, right, "m_layout=(128,2,2):(1,256,128)", color=TMEM[0])
    footer(fig, "Separate from the CtaGroup.ONE mainline; the same two M tiles and GEMM result are retained.")
    save_figure(fig, path)


def draw_case5(path: Path) -> None:
    """local_tile selects a GMEM view; it does not copy matrix values."""
    layout = row_major(4, 6)
    labels = all_labels(layout, 4, 6, lambda _r, _c: "")
    fig = tlviz._build_composite_figure(
        [
            (
                layout,
                {
                    "cell_labels": labels,
                    "highlight_mask": one_hot_mask(4, 6, *TARGET_TILE),
                    "highlight_facecolor": "#FFE6B0",
                    "highlight_edgecolor": TARGET_EDGE,
                },
            )
        ],
        titles=["A problem tensor decomposed into CTA coordinates"],
        main_title="local_tile: global coordinate -> tile coordinate + local coordinate",
        panel_size=(8.8, 4.8),
        colorize=False,
        num_colors=1,
    )
    ax = fig.axes[0]
    recolor_cells(ax, 4, 6, lambda _r, _c: GMEM[1])
    style_axis_labels(ax, lambda r: f"m_tile={r}", lambda c: f"q={c}")
    title_as_storage(ax, GMEM)
    add_target_point(ax, 1.5, 1.5, "gA(5,6,1)\n= mA(133,70)")
    ax.text(
        6.55,
        2.0,
        "VIEW ONLY\nsame GMEM address\nno memory traffic",
        ha="left",
        va="center",
        fontsize=9.2,
        color=GMEM[0],
        bbox={"boxstyle": "round,pad=0.40", "fc": "white", "ec": GMEM[0], "ls": "--"},
        clip_on=False,
    )
    footer(fig, "gA has logical shape (BM, BK, num_k_tiles) = (128,64,6).")
    save_figure(fig, path)


def _mma_k_layout() -> tuple[Layout, Layout]:
    # Display shape: four M blocks x (two K sub-blocks x four MMA_K groups).
    data = Layout((4, (2, 4)), (8, (1, 2)))
    colors = Layout((4, (2, 4)), (0, (0, 1)))
    return data, colors


def add_mk_boundaries(ax: plt.Axes, *, show_m_axis: bool = True) -> None:
    style_axis_labels(ax, hide=True)
    for x, label in zip((0, 2, 4, 6, 8), ("0", "16", "32", "48", "64")):
        ax.text(x, -0.34, label, ha="center", va="bottom", fontsize=8.3, color=MUTED, clip_on=False)
    if show_m_axis:
        for y, label in zip((0, 1, 2, 3, 4), ("0", "32", "64", "96", "128")):
            ax.text(-0.26, y, label, ha="right", va="center", fontsize=8.3, color=MUTED, clip_on=False)
    for x in (2, 4, 6):
        ax.plot([x, x], [0, 4], color=INK, linewidth=2.0, zorder=12)
    for group in range(4):
        ax.text(group * 2 + 1, 4.30, f"MMA_K={group}", ha="center", va="center", fontsize=8.6, weight="bold", color=INK, clip_on=False)
    if show_m_axis:
        ax.text(-0.82, 2, "local M", ha="center", va="center", fontsize=8.6, color=MUTED, rotation=90, clip_on=False)


def draw_case6(path: Path) -> None:
    """Exact MMA_K arithmetic on a block-compressed A tile."""
    layout, color_layout = _mma_k_layout()
    labels = all_labels(layout, 4, 8, lambda _r, _c: "")
    fig = tlviz._build_composite_figure(
        [
            (
                layout,
                {
                    "cell_labels": labels,
                    "color_layout": color_layout,
                    "highlight_mask": one_hot_mask(4, 8, 0, 0),
                    "highlight_facecolor": "#FFE6B0",
                    "highlight_edgecolor": TARGET_EDGE,
                },
            )
        ],
        titles=["Selected A tile: 128x64, compressed into 32x8 display blocks"],
        main_title="partition_A: expose the four MMA_K iterations inside one BK=64 tile",
        panel_size=(10.0, 5.2),
        colorize=True,
        num_colors=4,
    )
    ax = fig.axes[0]
    recolor_cells(ax, 4, 8, lambda _r, c: MMA_K_COLORS[c // 2])
    add_mk_boundaries(ax)
    # Exact position within the compressed 32x8 cell.
    target_x = TARGET_LOCAL[1] / 8.0
    target_y = TARGET_LOCAL[0] / 32.0
    add_target_point(ax, target_x, target_y, "local=(5,6)\nMMA_K=0\ninner_k=6")
    footer(fig, "The 32x8 cells are a readability compression; MMA instruction K is 16.")
    save_figure(fig, path)


def draw_case7(path: Path) -> None:
    """Descriptor tensor slots: stage x MMA_K, not copied matrix elements."""
    layout = row_major(3, 4)
    labels = all_labels(layout, 3, 4, lambda _r, _c: "DESC")
    color_layout = Layout((3, 4), (0, 1))
    fig = tlviz._build_composite_figure(
        [
            (
                layout,
                {
                    "cell_labels": labels,
                    "color_layout": color_layout,
                    "highlight_mask": one_hot_mask(3, 4, 1, 0),
                    "highlight_facecolor": "#FFE6B0",
                    "highlight_edgecolor": TARGET_EDGE,
                },
            )
        ],
        titles=["tCrA logical modes: (STAGE=3, MMA_K=4)"],
        main_title="make_fragment_A: one descriptor covers an SMEM region",
        panel_size=(7.8, 4.8),
        colorize=True,
        num_colors=4,
    )
    ax = fig.axes[0]
    recolor_cells(ax, 3, 4, lambda _r, c: MMA_K_COLORS[c])
    style_axis_labels(ax, lambda r: f"stage={r}", lambda c: f"MMA_K={c}")
    title_as_storage(ax, DESC)
    ax.text(
        4.8,
        1.5,
        "selected DESC covers sA(5,6,stage=1)\naddress metadata only; no A value is stored",
        ha="left",
        va="center",
        fontsize=8.8,
        color=DESC[0],
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": DESC[0]},
        clip_on=False,
    )
    footer(fig, "stage=q mod 3 is a teaching shorthand; the real pipeline advances state/count/phase.")
    save_figure(fig, path)


def draw_case8(path: Path) -> None:
    """Logical TMEM accumulator coordinates plus the K-tile accumulation timeline."""
    # Coarsen the atom's exact logical C layout (M contiguous) to 32x32 blocks.
    assert str(SM100_128x256x16_F32F16F16_SS.c_layout) == "(1, (128, 256)) : (0, (1, 128))"
    acc = col_major(4, 8)
    timeline = row_major(6, 1)
    acc_labels = all_labels(acc, 4, 8, lambda r, c: f"M{r}\nN{c}")
    time_labels = {
        0: "q=0\ninit\nACC=False",
        1: "q=1\n+k=64..127\n(k=70)",
        2: "q=2\naccumulate",
        3: "q=3\naccumulate",
        4: "q=4\naccumulate",
        5: "q=5\naccumulate",
    }
    fig = tlviz._build_composite_figure(
        [
            (acc, {"cell_labels": acc_labels}),
            (
                timeline,
                {
                    "cell_labels": time_labels,
                    "highlight_mask": one_hot_mask(6, 1, 1, 0),
                    "highlight_facecolor": "#FFE6B0",
                    "highlight_edgecolor": TARGET_EDGE,
                },
            ),
        ],
        titles=["tCtAcc / TMEM: 128x256 shown as 32x32 blocks", "The same TMEM allocation across six K tiles"],
        main_title="cute.gemm initializes once, then accumulates into the same tCtAcc",
        panel_size=(6.8, 5.5),
        colorize=False,
        num_colors=1,
    )
    left, right = fig.axes[:2]
    recolor_cells(left, 4, 8, lambda _r, _c: TMEM[1])
    recolor_cells(right, 6, 1, lambda r, _c: "#E8F1FF" if r == 0 else TMEM[1])
    style_axis_labels(left, lambda r: f"Mblk={r}", lambda c: f"Nblk={c}")
    style_axis_labels(right, hide=True)
    title_as_storage(left, TMEM)
    title_as_storage(right, TMEM)
    # A[133,70] is local M row 5 in this output tile and contributes across N.
    y = TARGET_LOCAL[0] / 32.0
    left.plot([0, 8], [y, y], color=TARGET_EDGE, linewidth=2.2, zorder=14)
    left.text(
        4.0,
        4.42,
        "orange row: A[133,70] x B[n,70] contributes to this CTA tile's 256 D columns",
        ha="center",
        va="center",
        fontsize=8.8,
        color=TARGET_EDGE,
        bbox={"boxstyle": "round,pad=0.30", "fc": "white", "ec": TARGET_EDGE},
        clip_on=False,
    )
    footer(fig, "Three N-direction CTA tiles together cover global D[133,:]; no per-thread or TMEM-bank claim.")
    save_figure(fig, path)


def draw_case9(path: Path) -> None:
    """Epilogue as three aligned per-thread views without inventing TV ownership."""
    src = row_major(1, 8)
    regs = row_major(1, 8)
    dst = row_major(1, 8)
    src_labels = {i: f"v{i}" for i in range(8)}
    reg_labels = {i: f"r{i}" for i in range(8)}
    dst_labels = {i: f"g{i}" for i in range(8)}
    fig = tlviz._build_composite_figure(
        [
            (src, {"cell_labels": src_labels}),
            (regs, {"cell_labels": reg_labels}),
            (dst, {"cell_labels": dst_labels}),
        ],
        titles=["tTR_tAcc: TMEM source view", "tTR_rAcc: Thread t registers", "tTR_gC: GMEM destination view"],
        main_title="Epilogue: align views first, then execute two physical copies",
        panel_size=(4.5, 2.5),
        colorize=False,
        num_colors=1,
    )
    left, middle, right = fig.axes[:3]
    recolor_cells(left, 1, 8, lambda _r, _c: TMEM[1])
    recolor_cells(middle, 1, 8, lambda _r, _c: RMEM[1])
    recolor_cells(right, 1, 8, lambda _r, _c: GMEM[1])
    for ax in (left, middle, right):
        style_axis_labels(ax, hide=True)
    title_as_storage(left, TMEM)
    title_as_storage(middle, RMEM)
    title_as_storage(right, GMEM)
    add_panel_arrow(fig, left, middle, "tcgen05.ld\nTMEM -> RMEM", color=RMEM[0], y=0.52)
    add_panel_arrow(fig, middle, right, "store\nRMEM -> GMEM", color=GMEM[0], y=0.52)

    fig.text(
        0.50,
        0.79,
        "tiled_copy_t2r",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        color=VIEW_EDGE,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": VIEW_EDGE, "ls": "--"},
    )
    for x in (0.18, 0.82):
        fig.add_artist(
            FancyArrowPatch(
                (0.50, 0.76),
                (x, 0.65),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
                linestyle=(0, (4, 3)),
                color=VIEW_EDGE,
            )
        )
    footer(
        fig,
        "Representative value tokens for one thread. Exact count and ownership come from the concrete tiled_copy_t2r layout.",
    )
    save_figure(fig, path)


def draw_case10(path: Path) -> None:
    """Same logical A coordinates on the two sides of the physical TMA copy."""
    layout, color_layout = _mma_k_layout()
    labels = all_labels(layout, 4, 8, lambda _r, _c: "")
    opts = {
        "cell_labels": labels,
        "color_layout": color_layout,
        "highlight_mask": one_hot_mask(4, 8, 0, 0),
        "highlight_facecolor": "#FFE6B0",
        "highlight_edgecolor": TARGET_EDGE,
    }
    fig = tlviz._build_composite_figure(
        [(layout, opts), (layout, dict(opts))],
        titles=["tAgA: GMEM source, q=1", "tAsA / sA: SMEM destination, stage=1"],
        main_title="TMA copy preserves logical coordinates while changing physical storage",
        panel_size=(6.2, 4.6),
        colorize=True,
        num_colors=4,
    )
    left, right = fig.axes[:2]
    recolor_cells(left, 4, 8, lambda _r, c: MMA_K_COLORS[c // 2])
    recolor_cells(right, 4, 8, lambda _r, c: MMA_K_COLORS[c // 2])
    add_mk_boundaries(left)
    add_mk_boundaries(right, show_m_axis=False)
    title_as_storage(left, GMEM)
    title_as_storage(right, SMEM)
    target_x = TARGET_LOCAL[1] / 8.0
    target_y = TARGET_LOCAL[0] / 32.0
    add_target_point(left, target_x, target_y, "xi(5,6)")
    add_target_point(right, target_x, target_y, "xi(5,6)")
    add_panel_arrow(fig, left, right, "TMA cute.copy\nGMEM -> SMEM", color=SMEM[0], y=0.50)
    footer(
        fig,
        "q=1 -> stage=1 uses the article's simplified q mod 3 rotation; the right grid shows logical coordinates, not SMEM addresses.",
    )
    save_figure(fig, path)


DIAGRAMS: tuple[tuple[str, str, str, Callable[[Path], None]], ...] = (
    ("case0", "Matrix A tile and element coordinate zoom", "dataflow_case0_matrix_a.svg", draw_case0),
    ("case1", "Single-CTA tcgen05 MMA atom block projection", "tiled_mma_case1_cta_group_one.svg", draw_case1),
    ("case2", "CTA-pair logical M partition", "tiled_mma_case2_cta_group_two.svg", draw_case2),
    ("case3", "Atom layout repeat coverage", "tiled_mma_case3_atom_layout_repeat.svg", draw_case3),
    ("case4", "Before and after MMA tile permutation", "tiled_mma_case4_permutation.svg", draw_case4),
    ("case5", "CTA local tile view of A", "dataflow_case5_local_tile.svg", draw_case5),
    ("case6", "MMA K decomposition inside the A tile", "dataflow_case6_partition_a.svg", draw_case6),
    ("case7", "SMEM descriptor stage and MMA K layout", "dataflow_case7_smem_stages.svg", draw_case7),
    ("case8", "TMEM accumulator logical layout and K timeline", "dataflow_case8_tmem_accumulator.svg", draw_case8),
    ("case9", "TMEM to registers to GMEM epilogue views", "dataflow_case9_t2r_epilogue.svg", draw_case9),
    ("case10", "TMA source and SMEM destination coordinate mapping", "dataflow_case10_tma_mapping.svg", draw_case10),
)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=script_dir.parent / "images")
    parser.add_argument("--readme", type=Path, default=script_dir.parent / "README.md")
    parser.add_argument("--skip-readme", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[str, str, Path]] = []
    for key, alt, filename, draw in DIAGRAMS:
        output = args.output_dir / filename
        draw(output)
        generated.append((key, alt, output.resolve()))
        print(output)

    if not args.skip_readme:
        readme = args.readme.resolve()
        for key, alt, output in generated:
            update_readme_block(readme, key, output, alt)
        print(readme)


if __name__ == "__main__":
    main()
