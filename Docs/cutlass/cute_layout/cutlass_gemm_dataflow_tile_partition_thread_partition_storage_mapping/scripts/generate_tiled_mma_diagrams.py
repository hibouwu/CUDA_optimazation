#!/usr/bin/env python3
"""Generate six process-first diagrams for the Blackwell CuTe GEMM tutorial.

Every figure uses the same visual grammar:

* pale boxes are views;
* saturated boxes are storage or descriptors;
* dashed arrows change only the way coordinates are interpreted;
* solid arrows move values or perform computation;
* orange always follows A[133,70] or its contribution to the output.

The script also refreshes the six generated-image blocks in README.md.
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
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


PROBLEM_MNK = (512, 768, 384)
CTA_MNK = (128, 256, 64)
MMA_MNK = (128, 256, 16)
TARGET_MK = (133, 70)
TARGET_ID = TARGET_MK[0] * PROBLEM_MNK[2] + TARGET_MK[1] + 1
TARGET_TILE = (TARGET_MK[0] // CTA_MNK[0], TARGET_MK[1] // CTA_MNK[2])
TARGET_LOCAL = (TARGET_MK[0] % CTA_MNK[0], TARGET_MK[1] % CTA_MNK[2])

INK = "#172033"
MUTED = "#667085"
GRID = "#98A2B3"
VIEW_EDGE = "#2563EB"
VIEW_FACE = "#EFF6FF"
GMEM_EDGE = "#1D4ED8"
GMEM_FACE = "#DBEAFE"
SMEM_EDGE = "#047857"
SMEM_FACE = "#D1FAE5"
DESC_EDGE = "#7C3AED"
DESC_FACE = "#EDE9FE"
TMEM_EDGE = "#6D28D9"
TMEM_FACE = "#DDD6FE"
RMEM_EDGE = "#B45309"
RMEM_FACE = "#FEF3C7"
TARGET = "#F59E0B"
TARGET_EDGE = "#B45309"
K_COLORS = ("#FDE68A", "#BFDBFE", "#BBF7D0", "#FBCFE8")

STEPS = (
    "local_tile",
    "partition_A",
    "TMA copy",
    "descriptor",
    "MMA / TMEM",
    "epilogue",
)

CJK_FONT_CANDIDATES = (
    Path("/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/fandol/FandolHei-Regular.otf"),
)
CJK_FONT_PATH = next((path for path in CJK_FONT_CANDIDATES if path.is_file()), None)
if CJK_FONT_PATH is not None:
    font_manager.fontManager.addfont(CJK_FONT_PATH)
    FONT_FAMILY = font_manager.FontProperties(fname=CJK_FONT_PATH).get_name()
else:
    FONT_FAMILY = "DejaVu Sans"

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": 10,
        "axes.unicode_minus": False,
        "text.color": INK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "path",
        "svg.hashsalt": "blackwell-cute-teaching-flow-v2",
    }
)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")


def update_readme_block(readme: Path, key: str, image_path: Path, alt: str) -> None:
    begin = f"<!-- BEGIN GENERATED DIAGRAM: {key} -->"
    end = f"<!-- END GENERATED DIAGRAM: {key} -->"
    relative = Path(os.path.relpath(image_path, readme.parent)).as_posix()
    replacement = f"{begin}\n![{alt}]({relative})\n{end}"
    content = readme.read_text(encoding="utf-8")
    pattern = re.compile(rf"{re.escape(begin)}.*?{re.escape(end)}", re.DOTALL)
    updated, count = pattern.subn(replacement, content, count=1)
    if count != 1:
        raise RuntimeError(f"README marker block not found or duplicated: {key}")
    readme.write_text(updated, encoding="utf-8")


def canvas(title: str, step: int, *, figsize: tuple[float, float] = (14.5, 7.0)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")
    ax.text(50, 62.0, title, ha="center", va="center", fontsize=16, weight="bold", color=INK)
    ribbon(ax, step)
    return fig, ax


def ribbon(ax: plt.Axes, current: int) -> None:
    x0, gap, width, y, height = 2.0, 0.7, 15.3, 55.0, 4.2
    for index, label in enumerate(STEPS, 1):
        x = x0 + (index - 1) * (width + gap)
        active = index == current
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.12,rounding_size=0.8",
            facecolor=TARGET if active else "#F2F4F7",
            edgecolor=TARGET_EDGE if active else "#D0D5DD",
            linewidth=1.4 if active else 0.9,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            f"{index}. {label}",
            ha="center",
            va="center",
            fontsize=8.5,
            weight="bold" if active else "normal",
            color="white" if active else MUTED,
        )


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    edge: str,
    face: str,
    subtitle: str | None = None,
    dashed: bool = False,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.30,rounding_size=1.2",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.8,
        linestyle=(0, (5, 3)) if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height - 3.1, title, ha="center", va="center", fontsize=12, weight="bold", color=edge)
    if subtitle:
        ax.text(x + width / 2, y + height - 6.3, subtitle, ha="center", va="center", fontsize=8.5, color=MUTED)
        body_y = y + height / 2 - 1.0
    else:
        body_y = y + height / 2 - 0.5
    ax.text(x + width / 2, body_y, body, ha="center", va="center", fontsize=9.5, color=INK, linespacing=1.45)


def operation_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    *,
    physical: bool,
    label_y_offset: float = 3.0,
) -> None:
    color = SMEM_EDGE if physical else VIEW_EDGE
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.6 if physical else 2.0,
        linestyle="-" if physical else (0, (6, 4)),
        color=color,
        zorder=20,
    )
    ax.add_patch(arrow)
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    ax.text(
        mx,
        my + label_y_offset,
        label,
        ha="center",
        va="center",
        fontsize=9.2,
        weight="bold",
        color=color,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.98},
        zorder=25,
    )


def target_badge(ax: plt.Axes, x: float, y: float, label: str, *, align: str = "center") -> None:
    ax.add_patch(Circle((x, y), 0.9, facecolor=TARGET, edgecolor=TARGET_EDGE, linewidth=1.2, zorder=30))
    if align == "left":
        tx, ha = x + 1.8, "left"
    elif align == "right":
        tx, ha = x - 1.8, "right"
    else:
        tx, ha = x, "center"
    ax.text(
        tx,
        y - 3.0,
        label,
        ha=ha,
        va="top",
        fontsize=8.7,
        color=TARGET_EDGE,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": TARGET_EDGE, "alpha": 0.97},
        zorder=31,
    )


def note(ax: plt.Axes, text: str, *, y: float = 3.0) -> None:
    ax.text(50, y, text, ha="center", va="center", fontsize=9.5, color=MUTED)


def tile_grid(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rows, cols = 4, 6
    cw, ch = width / cols, height / rows
    for row in range(rows):
        for col in range(cols):
            selected = (row, col) == TARGET_TILE
            ax.add_patch(
                Rectangle(
                    (x + col * cw, y + (rows - 1 - row) * ch),
                    cw,
                    ch,
                    facecolor="#FDE68A" if selected else "#F8FAFC",
                    edgecolor=TARGET_EDGE if selected else GRID,
                    linewidth=1.8 if selected else 0.7,
                )
            )
    ax.text(x + width / 2, y - 1.8, "K tiles: q = 0 ... 5", ha="center", va="top", fontsize=8.2, color=MUTED)
    ax.text(x - 1.3, y + height / 2, "M tiles\n0 ... 3", ha="right", va="center", fontsize=8.2, color=MUTED)
    tx = x + (TARGET_TILE[1] + 0.5) * cw
    ty = y + (rows - TARGET_TILE[0] - 0.5) * ch
    ax.add_patch(Circle((tx, ty), 0.9, facecolor=TARGET, edgecolor=TARGET_EDGE, linewidth=1.2, zorder=30))
    ax.annotate(
        "A[133,70]",
        xy=(tx, ty),
        xytext=(x + width + 1.2, y + height * 0.62),
        ha="left",
        va="center",
        fontsize=8.5,
        color=TARGET_EDGE,
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": TARGET_EDGE, "lw": 1.1},
        bbox={"boxstyle": "round,pad=0.20", "fc": "white", "ec": TARGET_EDGE},
        zorder=31,
    )


def k_band(ax: plt.Axes, x: float, y: float, width: float, height: float, *, target_label: str | None = None) -> None:
    segment = width / 4
    for index in range(4):
        ax.add_patch(
            Rectangle(
                (x + index * segment, y),
                segment,
                height,
                facecolor=K_COLORS[index],
                edgecolor=INK,
                linewidth=1.0,
            )
        )
        ax.text(x + (index + 0.5) * segment, y + height / 2, f"MMA_K={index}\nk={16*index}..{16*index+15}", ha="center", va="center", fontsize=8.4)
    if target_label:
        tx = x + (TARGET_LOCAL[1] / CTA_MNK[2]) * width
        ty = y + height * 0.66
        ax.add_patch(Circle((tx, ty), 0.75, facecolor=TARGET, edgecolor=TARGET_EDGE, linewidth=1.1, zorder=30))
        ax.annotate(
            target_label,
            xy=(tx, ty),
            xytext=(x + width * 0.15, y - 2.0),
            ha="center",
            va="top",
            fontsize=8.2,
            color=TARGET_EDGE,
            weight="bold",
            arrowprops={"arrowstyle": "->", "color": TARGET_EDGE, "lw": 1.0},
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": TARGET_EDGE},
            zorder=31,
        )


def descriptor_band(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    cw = width / 4
    for k_group in range(4):
        selected = k_group == 0
        ax.add_patch(
            Rectangle(
                (x + k_group * cw, y),
                cw,
                height,
                facecolor="#FDE68A" if selected else DESC_FACE,
                edgecolor=TARGET_EDGE if selected else DESC_EDGE,
                linewidth=1.8 if selected else 0.9,
            )
        )
        ax.text(
            x + (k_group + 0.5) * cw,
            y + height / 2,
            f"DESC\nMMA_K={k_group}",
            ha="center",
            va="center",
            fontsize=8.0,
            weight="bold" if selected else "normal",
        )
    tx = x + 0.5 * cw
    ty = y + height / 2
    ax.add_patch(Circle((tx, ty), 0.75, facecolor=TARGET, edgecolor=TARGET_EDGE, linewidth=1.1, zorder=30))
    ax.annotate(
        "该 descriptor 描述 A[133,70] 所在区域",
        xy=(tx, ty),
        xytext=(x + width / 2, y - 2.2),
        ha="center",
        va="top",
        fontsize=8.0,
        color=TARGET_EDGE,
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": TARGET_EDGE, "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": TARGET_EDGE},
        zorder=31,
    )


def draw_overview(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16.0, 9.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 76)
    ax.axis("off")
    ax.text(50, 73.0, "CUTLASS C++ CuTe: Operand A and GEMM output dataflow", ha="center", va="center", fontsize=17, weight="bold")

    # Row 1: GMEM views, left to right.
    box(ax, 2, 53, 24, 14, "mA", "完整 A\nGMEM storage", edge=GMEM_EDGE, face=GMEM_FACE)
    box(ax, 38, 53, 24, 14, "gA", "CTA-local\nGMEM view", edge=VIEW_EDGE, face=VIEW_FACE, dashed=True)
    box(ax, 74, 53, 24, 14, "tCgA", "MMA-coordinate\nGMEM view", edge=VIEW_EDGE, face=VIEW_FACE, dashed=True)
    operation_arrow(ax, (26.5, 60), (37.5, 60), "local_tile", physical=False, label_y_offset=3.2)
    operation_arrow(ax, (62.5, 60), (73.5, 60), "partition_A", physical=False, label_y_offset=3.2)

    # Row 2: TMA and descriptor path, right to left.
    box(ax, 74, 30, 24, 14, "tAgA / tAsA", "TMA source / destination views", edge=VIEW_EDGE, face=VIEW_FACE, dashed=True)
    box(ax, 38, 30, 24, 14, "tCsA", "A values\nSMEM Tensor", edge=SMEM_EDGE, face=SMEM_FACE)
    box(ax, 2, 30, 24, 14, "tCrA", "A descriptor", edge=DESC_EDGE, face=DESC_FACE)
    operation_arrow(ax, (86, 52.5), (86, 44.5), "tma_partition", physical=False, label_y_offset=0.0)
    operation_arrow(ax, (73.5, 37), (62.5, 37), "TMA copy", physical=True, label_y_offset=3.2)
    operation_arrow(ax, (37.5, 37), (26.5, 37), "make_fragment_A", physical=False, label_y_offset=3.2)

    # Row 3: MMA and epilogue, left to right.
    box(ax, 2, 7, 24, 14, "tCtAcc", "GEMM accumulator\nTMEM storage", edge=TMEM_EDGE, face=TMEM_FACE)
    box(ax, 38, 7, 24, 14, "tDrAcc", "accumulator fragment\nRMEM storage", edge=RMEM_EDGE, face=RMEM_FACE)
    box(ax, 74, 7, 24, 14, "mD", "final output\nGMEM storage", edge=GMEM_EDGE, face=GMEM_FACE)
    operation_arrow(ax, (14, 29.5), (14, 21.5), "cute::gemm", physical=True, label_y_offset=0.0)
    operation_arrow(ax, (26.5, 14), (37.5, 14), "tcgen05.ld", physical=True, label_y_offset=3.2)
    operation_arrow(ax, (62.5, 14), (73.5, 14), "AXPBY + store", physical=True, label_y_offset=3.2)

    for x, y in ((14, 57), (50, 57), (86, 57), (86, 34), (50, 34), (14, 34)):
        ax.add_patch(Circle((x, y), 0.8, facecolor=TARGET, edgecolor=TARGET_EDGE, linewidth=1.1, zorder=30))
    ax.plot([7, 21], [11, 11], color=TARGET_EDGE, linewidth=3.2, solid_capstyle="round")
    target_badge(ax, 50, 11, "D[133,n] fragment")
    target_badge(ax, 86, 11, "D[133,n]")
    note(ax, "虚线表示坐标或访问方式变化；实线表示 copy、MMA 或 store。", y=1.8)
    save_figure(fig, path)


def draw_step1(path: Path) -> None:
    fig, ax = canvas("Step 1 — local_tile 选出 CTA tile，不复制数据", 1)
    box(ax, 2, 10, 38, 40, "mA", "", edge=GMEM_EDGE, face=GMEM_FACE, subtitle="完整矩阵 A · GMEM storage · shape=(512,384)")
    tile_grid(ax, 7, 17, 28, 18)
    operation_arrow(ax, (41, 30), (60, 30), "local_tile(m_tile=1)", physical=False)
    box(
        ax,
        61,
        13,
        37,
        34,
        "gA",
        "当前 CTA 看到的 A tile\nshape = (128,64,6)\n\ngA(5,6,q=1) = mA(133,70)",
        edge=VIEW_EDGE,
        face=VIEW_FACE,
        subtitle="仍指向 mA 的 GMEM view",
        dashed=True,
    )
    target_badge(ax, 79.5, 18.5, f"同一个 A[133,70]\nid={TARGET_ID}")
    note(ax, "虚线表示只改变坐标解释：gA 和 mA 指向同一个 GMEM 地址。")
    save_figure(fig, path)


def draw_step2(path: Path) -> None:
    fig, ax = canvas("Step 2 — partition_A 把 tile 展开成 MMA_K 层次", 2)
    box(
        ax,
        2,
        16,
        28,
        30,
        "gA",
        "当前 CTA 的 view\nlocal_m = 5\nlocal_k = 6\nq = 1",
        edge=VIEW_EDGE,
        face=VIEW_FACE,
        dashed=True,
    )
    target_badge(ax, 16, 27, "gA(5,6,q=1)")
    operation_arrow(ax, (31, 31), (48, 31), "partition_A", physical=False)
    box(ax, 49, 11, 49, 39, "tCgA", "", edge=VIEW_EDGE, face=VIEW_FACE, subtitle="仍指向同一份 GMEM · MMA 坐标 view", dashed=True)
    k_band(ax, 53, 19, 41, 10, target_label="MMA_K=0\ninner_k=6")
    ax.text(73.5, 39.5, "一条 tcgen05 MMA", ha="center", va="center", fontsize=9.0, color=MUTED)
    ax.text(73.5, 36.2, "A(128×16) × B^T(16×256) → D(128×256)", ha="center", va="center", fontsize=10.0, weight="bold", color=INK)
    ax.plot([64.5, 82.5], [33.3, 33.3], color=TARGET_EDGE, linewidth=2.0)
    ax.text(73.5, 31.1, "共同的 K = 16", ha="center", va="center", fontsize=8.5, color=TARGET_EDGE)
    note(ax, "BK=64 被展开成四次 K=16 的 MMA；数据仍然没有离开 GMEM。")
    save_figure(fig, path)


def draw_step3(path: Path) -> None:
    fig, ax = canvas("Step 3 — TMA 第一次真正搬运数据：GMEM → SMEM", 3, figsize=(16.0, 7.2))
    box(ax, 1, 14, 21, 35, "tCgA", "MMA 坐标 view\nq=1\nMMA_K=0\ninner_k=6", edge=VIEW_EDGE, face=VIEW_FACE, dashed=True)
    operation_arrow(ax, (22.5, 31), (34, 31), "tma_partition", physical=False)
    box(ax, 35, 31, 24, 18, "tAgA", "GMEM 读取 view\nξ(5,6), q=1", edge=VIEW_EDGE, face=VIEW_FACE, dashed=True)
    box(ax, 35, 11, 24, 16, "tAsA", "SMEM 写入 view\nξ(5,6)", edge=SMEM_EDGE, face="#ECFDF5", dashed=True)
    ax.plot([47, 47], [31, 27], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    operation_arrow(ax, (60, 28), (73, 28), "TMA copy", physical=True)
    box(ax, 74, 11, 25, 38, "tCsA", "SMEM Tensor\n引用 shared storage 中的 A\n\ntCsA(5,6)", edge=SMEM_EDGE, face=SMEM_FACE)
    ax.add_patch(Circle((11.5, 22), 0.8, facecolor=TARGET, edgecolor=TARGET_EDGE, linewidth=1.1, zorder=30))
    ax.add_patch(Circle((55, 40), 0.7, facecolor=TARGET, edgecolor=TARGET_EDGE, linewidth=1.0, zorder=30))
    ax.add_patch(Circle((55, 20), 0.7, facecolor=TARGET, edgecolor=TARGET_EDGE, linewidth=1.0, zorder=30))
    target_badge(ax, 86.5, 20, "A[133,70] 到达 SMEM")
    note(ax, "tma_partition 只创建两个 view；粗实线的 TMA copy 才真正搬运矩阵数值。")
    save_figure(fig, path)


def draw_step4(path: Path) -> None:
    fig, ax = canvas("Step 4 — make_fragment_A 根据 SMEM layout 构造 descriptor", 4)
    box(ax, 2, 11, 38, 39, "tCsA", "", edge=SMEM_EDGE, face=SMEM_FACE, subtitle="SMEM Tensor · 引用 sA allocation")
    k_band(ax, 8, 22, 27, 9, target_label="A[133,70]")
    operation_arrow(ax, (41, 31), (59, 31), "make_fragment_A", physical=False)
    box(ax, 60, 11, 38, 39, "tCrA", "", edge=DESC_EDGE, face=DESC_FACE, subtitle="descriptor tensor")
    descriptor_band(ax, 66, 22, 27, 9)
    ax.text(79, 39.8, "地址 + stride + swizzle 信息", ha="center", va="center", fontsize=8.7, color=DESC_EDGE)
    note(ax, "tCrA 是交给 MMA 的“取货说明”，不是 A 的另一份数值副本。")
    save_figure(fig, path)


def draw_step5(path: Path) -> None:
    fig, ax = canvas("Step 5 — cute::gemm 读取 A/B，并把结果累加到 TMEM", 5)
    box(ax, 2, 31, 25, 18, "tCrA", "A descriptor\n描述 tCsA 中的 A[133,70]", edge=DESC_EDGE, face=DESC_FACE)
    box(ax, 2, 10, 25, 17, "tCrB", "B descriptor\n描述 tCsB", edge=DESC_EDGE, face=DESC_FACE)
    operation_arrow(ax, (28, 30), (49, 30), "cute::gemm\ntcgen05.mma", physical=True)
    box(ax, 50, 10, 48, 39, "tCtAcc", "", edge=TMEM_EDGE, face=TMEM_FACE, subtitle="TMEM accumulator · shape=(128,256)")
    ax.text(74, 36.5, "第一个 K tile 初始化，后续 K tile 继续累加", ha="center", va="center", fontsize=9.2, color=INK)
    ax.plot([57, 91], [24, 24], color=TARGET_EDGE, linewidth=4.0, solid_capstyle="round")
    ax.text(74, 19.8, "A[133,70] × B[n,70] 对 D[133,n] 的贡献", ha="center", va="center", fontsize=9.2, color=TARGET_EDGE, weight="bold")
    note(ax, "橙色点在这里变成一条输出贡献：A 的数值不会被原样复制进 TMEM。")
    save_figure(fig, path)


def draw_step6(path: Path) -> None:
    fig, ax = canvas("Step 6 — epilogue 先把结果装入寄存器，再写回 GMEM", 6)
    box(ax, 1, 14, 25, 35, "tCtAcc", "TMEM 存储\nMMA 结果", edge=TMEM_EDGE, face=TMEM_FACE)
    ax.plot([6, 21], [28, 28], color=TARGET_EDGE, linewidth=3.5, solid_capstyle="round")
    operation_arrow(ax, (27, 31), (39, 31), "tcgen05.ld", physical=True)
    box(ax, 40, 14, 22, 35, "tDrAcc", "RMEM 存储\n当前线程的 accumulator fragment", edge=RMEM_EDGE, face=RMEM_FACE)
    target_badge(ax, 51, 29, "D[133,n] fragment")
    operation_arrow(ax, (63, 31), (75, 31), "store", physical=True)
    box(ax, 76, 14, 23, 35, "mD", "GMEM 存储\n最终输出矩阵", edge=GMEM_EDGE, face=GMEM_FACE)
    target_badge(ax, 87.5, 29, "D[133,n]")
    note(ax, "TMEM 不能直接成为普通线程的 store 源；结果先装入寄存器，再写回 GMEM。")
    save_figure(fig, path)


DIAGRAMS: tuple[tuple[str, str, str, Callable[[Path], None]], ...] = (
    ("overview", "C++ CuTe GEMM dataflow overview", "overview_cpp_cute.svg", draw_overview),
    ("step1", "local_tile selects a CTA-local GMEM view", "step1_local_tile.svg", draw_step1),
    ("step2", "partition_A exposes MMA K groups", "step2_partition_a.svg", draw_step2),
    ("step3", "TMA copies A from GMEM to SMEM", "step3_tma_copy.svg", draw_step3),
    ("step4", "make_fragment_A creates an MMA descriptor", "step4_descriptor.svg", draw_step4),
    ("step5", "tcgen05 MMA accumulates into TMEM", "step5_mma_tmem.svg", draw_step5),
    ("step6", "epilogue copies TMEM through registers to GMEM", "step6_epilogue.svg", draw_step6),
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
