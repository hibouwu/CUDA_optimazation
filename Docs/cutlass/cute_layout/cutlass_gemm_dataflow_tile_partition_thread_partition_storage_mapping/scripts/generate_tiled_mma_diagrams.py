#!/usr/bin/env python3
"""Generate conceptual layout diagrams and refresh README image blocks.

The figures intentionally avoid CuTe T#/V# notation.  Except for labels that
explicitly say Thread/Register, the diagrams show logical decomposition only;
they are not claims about a concrete hardware thread-value layout.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import Callable


CELL = 72
LEFT = 50
TOP = 42
COLORS = ("#aaaaff", "#aaffaa", "#ffffaa", "#ffaaaa", "#ffcc66")
GRID = "#111111"

Cell = tuple[str, str, int]
Mapping = Callable[[int, int], Cell]


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


def matrix_svg(rows: int, cols: int, mapping: Mapping) -> str:
    """Return a title-free, colored matrix with two semantic labels per cell."""
    width = LEFT + cols * CELL + 8
    height = TOP + rows * CELL + 8
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        "text{font-family:'Times New Roman','Noto Serif CJK SC',serif;fill:#111111}",
        ".axis{font-size:22px}.cell{font-size:15px}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    for col in range(cols):
        x = LEFT + (col + 0.5) * CELL
        items.append(f'<text x="{x}" y="28" class="axis" text-anchor="middle">{col}</text>')
    for row in range(rows):
        y = TOP + (row + 0.5) * CELL + 8
        items.append(f'<text x="24" y="{y}" class="axis" text-anchor="middle">{row}</text>')
    for row in range(rows):
        for col in range(cols):
            first, second, color = mapping(row, col)
            x = LEFT + col * CELL
            y = TOP + row * CELL
            stroke = "#b34700" if color == 4 else GRID
            stroke_width = 4 if color == 4 else 1
            items.append(
                f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                f'fill="{COLORS[color % len(COLORS)]}" stroke="{stroke}" '
                f'stroke-width="{stroke_width}"/>'
            )
            cx = x + CELL / 2
            items.append(f'<text x="{cx}" y="{y + 30}" class="cell" text-anchor="middle">{first}</text>')
            items.append(f'<text x="{cx}" y="{y + 53}" class="cell" text-anchor="middle">{second}</text>')
    items.append("</svg>")
    return "\n".join(items) + "\n"


def tma_mapping_svg() -> str:
    """Show the same compressed A tile before and after the TMA transfer."""
    cell = 64
    rows, cols = 4, 8
    top, left, gap = 54, 44, 112
    panel_width = cols * cell
    right_left = left + panel_width + gap
    width = right_left + panel_width + 20
    height = top + rows * cell + 18
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        "text{font-family:'Times New Roman','Noto Serif CJK SC',serif;fill:#111111}",
        ".axis{font-size:18px}.cell{font-size:12px}.arrow{font-size:28px}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left + panel_width / 2}" y="28" class="axis" text-anchor="middle">tAgA: q=1</text>',
        f'<text x="{right_left + panel_width / 2}" y="28" class="axis" text-anchor="middle">tAsA / sA: stage=1</text>',
        f'<text x="{left + panel_width + gap / 2}" y="{top + rows * cell / 2 + 9}" '
        'class="arrow" text-anchor="middle">→</text>',
    ]
    for panel_left in (left, right_left):
        for row in range(rows):
            for col in range(cols):
                mma_k = col // 2
                m_begin = row * 32
                k_begin = mma_k * 16 + (col % 2) * 8
                highlighted = (row, col) == (0, 0)
                fill = COLORS[4] if highlighted else COLORS[mma_k]
                stroke = "#b34700" if highlighted else GRID
                stroke_width = 4 if highlighted else 1
                x = panel_left + col * cell
                y = top + row * cell
                items.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                    f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
                )
                items.append(
                    f'<text x="{x + cell / 2}" y="{y + 27}" class="cell" '
                    f'text-anchor="middle">M{m_begin}:{m_begin + 31}</text>'
                )
                items.append(
                    f'<text x="{x + cell / 2}" y="{y + 47}" class="cell" '
                    f'text-anchor="middle">K{k_begin}:{k_begin + 7}</text>'
                )
    items.append("</svg>")
    return "\n".join(items) + "\n"


def real_matrix_a_svg() -> str:
    """Show the real 512x384 A shape and an element-level 8x8 zoom window."""
    width, height = 1020, 590
    overview_x, overview_y = 64, 48
    overview_cols, overview_rows = 384, 512
    zoom_x, zoom_y, zoom_cell = 548, 82, 56
    target_row, target_col = 133, 70
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        "text{font-family:'Times New Roman','Noto Serif CJK SC',serif;fill:#111111}",
        ".axis{font-size:15px}.cell{font-size:11px}.value{font-size:13px}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]

    # Full A matrix: 4 M tiles by 6 K tiles, each with the real 128x64 extent.
    for m_tile in range(4):
        for k_tile in range(6):
            x = overview_x + k_tile * 64
            y = overview_y + m_tile * 128
            color = COLORS[(m_tile * 2 + k_tile) % 4]
            items.append(
                f'<rect x="{x}" y="{y}" width="64" height="128" '
                f'fill="{color}" stroke="{GRID}" stroke-width="1"/>'
            )
    items.append(
        f'<rect x="{overview_x + target_col - 3}" y="{overview_y + target_row - 3}" '
        'width="7" height="7" fill="#ffcc66" stroke="#b34700" stroke-width="2"/>'
    )
    for col in range(0, overview_cols + 1, 64):
        items.append(
            f'<text x="{overview_x + col}" y="30" class="axis" text-anchor="middle">{col}</text>'
        )
    for row in range(0, overview_rows + 1, 128):
        items.append(
            f'<text x="50" y="{overview_y + row + 5}" class="axis" text-anchor="end">{row}</text>'
        )

    # Element-level zoom of rows 128:136 and columns 64:72.
    for local_row in range(8):
        global_row = 128 + local_row
        items.append(
            f'<text x="{zoom_x - 12}" y="{zoom_y + (local_row + 0.5) * zoom_cell + 5}" '
            f'class="axis" text-anchor="end">{global_row}</text>'
        )
        for local_col in range(8):
            global_col = 64 + local_col
            x = zoom_x + local_col * zoom_cell
            y = zoom_y + local_row * zoom_cell
            highlighted = (global_row, global_col) == (target_row, target_col)
            fill = COLORS[4] if highlighted else COLORS[3]
            stroke = "#b34700" if highlighted else GRID
            stroke_width = 4 if highlighted else 1
            value = global_row * 384 + global_col + 1
            items.append(
                f'<rect x="{x}" y="{y}" width="{zoom_cell}" height="{zoom_cell}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
            )
            cx = x + zoom_cell / 2
            items.append(
                f'<text x="{cx}" y="{y + 24}" class="cell" text-anchor="middle">'
                f'A[{global_row},{global_col}]</text>'
            )
            items.append(
                f'<text x="{cx}" y="{y + 43}" class="value" text-anchor="middle">{value}</text>'
            )
    for local_col in range(8):
        items.append(
            f'<text x="{zoom_x + (local_col + 0.5) * zoom_cell}" y="{zoom_y - 14}" '
            f'class="axis" text-anchor="middle">{64 + local_col}</text>'
        )
    items.append(
        f'<path d="M {overview_x + target_col + 5} {overview_y + target_row} '
        f'L {zoom_x - 18} {zoom_y + 5 * zoom_cell} L {zoom_x} {zoom_y + 5 * zoom_cell}" '
        'fill="none" stroke="#b34700" stroke-width="2"/>'
    )
    items.append("</svg>")
    return "\n".join(items) + "\n"


def case_one(row: int, col: int) -> Cell:
    return "Atom0", f"Element{row * 4 + col}", 0


def case_two(row: int, col: int) -> Cell:
    rank = row // 2
    return f"CTA-rank{rank}", f"Part{(row % 2) * 4 + col}", rank


def case_repeat(row: int, col: int) -> Cell:
    mma_m, mma_n = row // 2, col // 2
    return f"M{mma_m}/N{mma_n}", f"Element{(row % 2) * 2 + col % 2}", mma_m * 2 + mma_n


def case_permutation(row: int, col: int) -> Cell:
    tile_by_row = (0, 0, 1, 1, 0, 0, 1, 1)
    tile = tile_by_row[row]
    local_rows = (0, 1, 0, 1, 2, 3, 2, 3)
    return f"Tile{tile}", f"Inner{local_rows[row] * 4 + col}", tile


def local_tile_view(row: int, col: int) -> Cell:
    color = 4 if (row, col) == (1, 1) else (row * 2 + col) % 4
    return f"mTile{row}", f"kTile{col}", color


def partition_a_view(row: int, col: int) -> Cell:
    mma_k = col // 2
    m_begin = row * 32
    k_begin = mma_k * 16 + (col % 2) * 8
    color = 4 if (row, col) == (0, 0) else mma_k
    return f"M{m_begin}:{m_begin + 31}", f"K{k_begin}:{k_begin + 7}", color


def smem_stage_view(row: int, col: int) -> Cell:
    color = 4 if (row, col) == (1, 0) else col
    return f"Stage{row}", f"MMA_K{col}", color


def accumulator_view(row: int, col: int) -> Cell:
    element = row * 8 + col
    return "M0/N0", f"Elem{element}", 0


def epilogue_view(row: int, col: int) -> Cell:
    thread = row
    return f"Thread{thread}", f"Register{col}", thread


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=script_dir.parent / "images")
    parser.add_argument("--readme", type=Path, default=script_dir.parent / "README.md")
    parser.add_argument("--skip-readme", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case0_output = args.output_dir / "dataflow_case0_matrix_a.svg"
    case0_output.write_text(real_matrix_a_svg(), encoding="utf-8")
    case10_output = args.output_dir / "dataflow_case10_tma_mapping.svg"
    case10_output.write_text(tma_mapping_svg(), encoding="utf-8")
    generated: list[tuple[str, str, Path]] = [
        ("case0", "Real-shape matrix A with A[133,70] highlighted", case0_output.resolve()),
        ("case10", "TMA source to SMEM stage mapping", case10_output.resolve()),
    ]
    print(case0_output)
    print(case10_output)

    diagrams: tuple[tuple[str, str, str, int, int, Mapping], ...] = (
        ("case1", "Conceptual trivial TiledMma", "tiled_mma_case1_cta_group_one.svg", 2, 4, case_one),
        ("case2", "Conceptual CTA pair partition", "tiled_mma_case2_cta_group_two.svg", 4, 4, case_two),
        ("case3", "Conceptual atom repeat", "tiled_mma_case3_atom_layout_repeat.svg", 4, 4, case_repeat),
        ("case4", "Conceptual permutation mapping", "tiled_mma_case4_permutation.svg", 8, 4, case_permutation),
        ("case5", "Real A matrix CTA tile decomposition", "dataflow_case5_local_tile.svg", 4, 6, local_tile_view),
        ("case6", "Conceptual MMA K decomposition", "dataflow_case6_partition_a.svg", 4, 8, partition_a_view),
        ("case7", "Three-stage descriptor layout", "dataflow_case7_smem_stages.svg", 3, 4, smem_stage_view),
        ("case8", "Conceptual accumulator modes", "dataflow_case8_tmem_accumulator.svg", 8, 8, accumulator_view),
        ("case9", "Conceptual per-thread epilogue fragment", "dataflow_case9_t2r_epilogue.svg", 8, 8, epilogue_view),
    )
    for key, alt, filename, rows, cols, mapping in diagrams:
        output = args.output_dir / filename
        output.write_text(matrix_svg(rows, cols, mapping), encoding="utf-8")
        generated.append((key, alt, output.resolve()))
        print(output)
    if not args.skip_readme:
        readme = args.readme.resolve()
        for key, alt, output in generated:
            update_readme_block(readme, key, output, alt)
        print(readme)


if __name__ == "__main__":
    main()
