#!/usr/bin/env python3
"""Generate the tcgen05 GEMM dataflow diagram with Graphviz."""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
NOTE_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = NOTE_DIR / "images"

FONT = "DejaVu Sans"

COLORS = {
    "gmem": "#d7e9ff",
    "gmem_border": "#2b6cb0",
    "smem": "#dff3df",
    "smem_border": "#2f855a",
    "tmem": "#ffe6c7",
    "tmem_border": "#c05621",
    "rmem": "#eadcff",
    "rmem_border": "#6b46c1",
    "view": "#f2f4f7",
    "view_border": "#667085",
    "descriptor": "#fff7d6",
    "descriptor_border": "#b7791f",
    "copy": "#e7f7f8",
    "copy_border": "#0987a0",
    "compute": "#ffe4e6",
    "compute_border": "#c53030",
    "sync": "#eef2ff",
    "sync_border": "#4f46e5",
    "config": "#f4f3ff",
    "config_border": "#7c3aed",
    "lane": "#f8fafc",
    "legend": "#ffffff",
}

NODE_STYLES = {
    "gmem": {
        "shape": "box",
        "style": "filled,rounded",
        "fillcolor": COLORS["gmem"],
        "color": COLORS["gmem_border"],
    },
    "smem": {
        "shape": "box3d",
        "style": "filled,rounded",
        "fillcolor": COLORS["smem"],
        "color": COLORS["smem_border"],
    },
    "tmem": {
        "shape": "box3d",
        "style": "filled,rounded",
        "fillcolor": COLORS["tmem"],
        "color": COLORS["tmem_border"],
    },
    "rmem": {
        "shape": "box",
        "style": "filled,rounded",
        "fillcolor": COLORS["rmem"],
        "color": COLORS["rmem_border"],
    },
    "view": {
        "shape": "box",
        "style": "filled,rounded,dashed",
        "fillcolor": COLORS["view"],
        "color": COLORS["view_border"],
    },
    "descriptor": {
        "shape": "component",
        "style": "filled,rounded",
        "fillcolor": COLORS["descriptor"],
        "color": COLORS["descriptor_border"],
    },
    "copy": {
        "shape": "box",
        "style": "filled,rounded,bold",
        "fillcolor": COLORS["copy"],
        "color": COLORS["copy_border"],
    },
    "compute": {
        "shape": "box",
        "style": "filled,rounded,bold",
        "fillcolor": COLORS["compute"],
        "color": COLORS["compute_border"],
    },
    "sync": {
        "shape": "box",
        "style": "filled,rounded",
        "fillcolor": COLORS["sync"],
        "color": COLORS["sync_border"],
    },
    "config": {
        "shape": "note",
        "style": "filled",
        "fillcolor": COLORS["config"],
        "color": COLORS["config_border"],
    },
}

EDGE_STYLES = {
    "data": {
        "color": "#1f2937",
        "penwidth": "2.2",
        "style": "solid",
        "arrowsize": "0.85",
    },
    "view": {
        "color": "#667085",
        "penwidth": "1.4",
        "style": "dashed",
        "arrowsize": "0.7",
    },
    "consume": {
        "color": "#c53030",
        "penwidth": "1.9",
        "style": "dotted",
        "arrowsize": "0.8",
    },
    "sync": {
        "color": "#4f46e5",
        "penwidth": "1.5",
        "style": "dashdot",
        "arrowsize": "0.72",
    },
    "loop": {
        "color": "#475467",
        "penwidth": "1.2",
        "style": "dashed",
        "arrowsize": "0.65",
    },
}


def html_label(title: str, subtitle: str = "", shape: str = "") -> str:
    rows = [f'<FONT POINT-SIZE="13"><B>{html.escape(title)}</B></FONT>']
    if subtitle:
        rows.append(f'<FONT POINT-SIZE="9">{html.escape(subtitle)}</FONT>')
    if shape:
        rows.append(f'<FONT POINT-SIZE="9">{html.escape(shape)}</FONT>')
    return "<" + "<BR/>".join(rows) + ">"


def attrs(values: dict[str, str]) -> str:
    rendered = []
    for key, value in values.items():
        if key == "label" and value.startswith("<"):
            rendered.append(f"{key}={value}")
        else:
            rendered.append(f'{key}="{value}"')
    return ", ".join(rendered)


class DotBuilder:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def node(self, name: str, kind: str, title: str, subtitle: str = "", shape: str = "") -> None:
        node_attrs = {
            **NODE_STYLES[kind],
            "label": html_label(title, subtitle, shape),
        }
        self.add(f"    {name} [{attrs(node_attrs)}];")

    def edge(self, source: str, target: str, kind: str, label: str = "", **extra: str) -> None:
        edge_attrs = {**EDGE_STYLES[kind], **extra}
        if label:
            edge_attrs["label"] = label
            edge_attrs["fontsize"] = "9"
            edge_attrs["fontcolor"] = edge_attrs["color"]
        self.add(f"    {source} -> {target} [{attrs(edge_attrs)}];")


def build_operand_lane(dot: DotBuilder, operand: str) -> None:
    lower = operand.lower()
    data = {
        "A": {
            "tma_info": "tma_a",
            "atom": "tma_a.atom",
            "matrix": "(M, K)",
            "tile": "(BM, BK, k)",
            "partition": "(MMA, MMA_M, MMA_K, k)",
            "fragment": "(MMA, MMA_M, MMA_K, STAGE)",
            "tma_s": "tAsA",
            "tma_g": "tAgA",
            "copy_call": "copy(tma_a.atom, tAgA[k_tile], tAsA[stage])",
            "partition_fn": "thr_mma.partition_A(gA)",
            "fragment_fn": "tiled_mma.make_fragment_A(sA)",
            "group_s": "group_modes(sA, 0, 3)",
            "group_tcg": "group_modes(tCgA, 0, 3)",
        },
        "B": {
            "tma_info": "tma_b",
            "atom": "tma_b.atom",
            "matrix": "(N, K)",
            "tile": "(BN, BK, k)",
            "partition": "(MMA, MMA_N, MMA_K, k)",
            "fragment": "(MMA, MMA_N, MMA_K, STAGE)",
            "tma_s": "tBsB",
            "tma_g": "tBgB",
            "copy_call": "copy(tma_b.atom, tBgB[k_tile], tBsB[stage])",
            "partition_fn": "thr_mma.partition_B(gB)",
            "fragment_fn": "tiled_mma.make_fragment_B(sB)",
            "group_s": "group_modes(sB, 0, 3)",
            "group_tcg": "group_modes(tCgB, 0, 3)",
        },
    }[operand]

    dot.add(f"  subgraph cluster_operand_{lower} {{")
    dot.add(f'    label="Operand {operand} Dataflow Path";')
    dot.add('    style="filled,rounded";')
    dot.add(f'    color="{COLORS["lane"]}";')
    dot.add(f'    fillcolor="{COLORS["lane"]}";')
    dot.add("    margin=14;")

    dot.node(f"tmaInfo{operand}", "config", f"{data['tma_info']}: TmaInfo", "kernel argument", "")
    dot.node(f"m{operand}", "gmem", f"m{operand}", "Original GMEM tensor", data["matrix"])
    dot.node(f"tmaAtom{operand}", "config", data["atom"], "TMA config object", "G2S atom")
    dot.node(f"g{operand}", "view", f"g{operand}", "CTA/local GMEM tile", data["tile"])
    dot.node(f"tCg{operand}", "view", f"tCg{operand}", "MMA thread partition", data["partition"])
    dot.node(
        f"tmaViews{operand}",
        "view",
        f"{data['tma_s']} / {data['tma_g']}",
        "TMA SMEM / GMEM views",
        "stage, k_tile indexed",
    )
    dot.node(f"copy{operand}", "copy", f"TMA copy loop {operand}", data["copy_call"], "GMEM -> SMEM")
    dot.node(f"s{operand}", "smem", f"s{operand}", "Physical SMEM allocation", f"staged {operand} tile")
    dot.node(f"tCr{operand}", "descriptor", f"tCr{operand}", "SMEM descriptor tensor", data["fragment"])

    dot.edge(f"tmaInfo{operand}", f"m{operand}", "view", "tma_tensor")
    dot.edge(f"tmaInfo{operand}", f"tmaAtom{operand}", "view", "atom")
    dot.edge(f"m{operand}", f"g{operand}", "view", f"local_tile(m{operand}, mma_tiler_mnk, coord)")
    dot.edge(f"g{operand}", f"tCg{operand}", "view", data["partition_fn"])
    dot.edge(f"tmaAtom{operand}", f"tmaViews{operand}", "view", "cpasync.tma_partition(...)")
    dot.edge(f"tCg{operand}", f"tmaViews{operand}", "view", data["group_tcg"])
    dot.edge(f"s{operand}", f"tmaViews{operand}", "view", data["group_s"], constraint="false")
    dot.edge(f"tmaViews{operand}", f"copy{operand}", "view", f"{data['tma_g']}[k_tile], {data['tma_s']}[stage]")
    dot.edge(f"copy{operand}", f"s{operand}", "data", "TMA write")
    dot.edge(f"copy{operand}", f"copy{operand}", "loop", "for k_tile / stage", constraint="false")
    dot.edge(f"s{operand}", f"tCr{operand}", "view", data["fragment_fn"])

    dot.add("  }")


def build_accumulator_lane(dot: DotBuilder) -> None:
    dot.add("  subgraph cluster_accumulator {")
    dot.add('    label="Accumulator C/D Dataflow Path";')
    dot.add('    style="filled,rounded";')
    dot.add(f'    color="{COLORS["lane"]}";')
    dot.add(f'    fillcolor="{COLORS["lane"]}";')
    dot.add("    margin=14;")

    dot.node("acc_shape", "view", "acc_shape", "partition_shape_C", "(MMA, MMA_M, MMA_N)")
    dot.node("tCtAcc_layout", "view", "tCtAcc layout", "make_fragment_C(acc_shape)", "(MMA, MMA_M, MMA_N)")
    dot.node("tmem_alloc", "tmem", "TmemAllocator", "allocate / retrieve_ptr", "TMEM columns")
    dot.node("tCtAcc", "tmem", "tCtAcc", "TMEM accumulator tensor", "(MMA, MMA_M, MMA_N)")

    dot.node("mC", "gmem", "mC", "Output GMEM tensor", "(M, N)")
    dot.node("gC", "view", "gC", "CTA/local output tile", "(BM, BN)")
    dot.node("tCgC", "view", "tCgC", "Epilogue GMEM partition", "(MMA, MMA_M, MMA_N)")

    dot.node("copy_atom_t2r", "config", "copy_atom_t2r", "TMEM load config", "Ld32x32bOp")
    dot.node("tiled_copy_t2r", "config", "tiled_copy_t2r", "T2R copy config", "anchored by tCtAcc slice")
    dot.node("thr_copy_t2r", "view", "thr_copy_t2r", "per-thread copy slice", "get_slice(tidx)")
    dot.node("tTR_tAcc", "view", "tTR_tAcc", "TMEM source view", "(T2R, T2R_M, NumTiles)")
    dot.node("tTR_gC", "view", "tTR_gC", "GMEM destination view", "(T2R, T2R_M, NumTiles)")
    dot.node("ldtm", "copy", "LDTM copy", "TMEM -> RMEM", "per NumTiles")
    dot.node("tTR_rAcc", "rmem", "tTR_rAcc", "RMEM accumulator fragment", "(T2R, T2R_M)")
    dot.node("store", "copy", "Epilogue store", "RMEM -> GMEM", "per NumTiles")
    dot.node("gmem_output", "gmem", "GMEM output region", "updated by epilogue store", "(BM, BN) in mC")

    dot.edge("acc_shape", "tCtAcc_layout", "view", "tiled_mma.make_fragment_C(acc_shape)")
    dot.edge("tCtAcc_layout", "tCtAcc", "view", "cute.make_tensor(tmem_ptr, layout)")
    dot.edge("tmem_alloc", "tCtAcc", "view", "retrieve_ptr(Float32)")
    dot.edge("mC", "gC", "view", "local_tile(mC, mma_tiler_mnk, coord)")
    dot.edge("gC", "tCgC", "view", "thr_mma.partition_C(gC)")

    dot.edge("copy_atom_t2r", "tiled_copy_t2r", "view", "tcgen05.make_tmem_copy(...)")
    dot.edge("tCtAcc", "tiled_copy_t2r", "view", "tCtAcc[(None,None),0,0]")
    dot.edge("tiled_copy_t2r", "thr_copy_t2r", "view", "get_slice(tidx)")
    dot.edge("thr_copy_t2r", "tTR_tAcc", "view", "partition_S(tCtAcc)")
    dot.edge("thr_copy_t2r", "tTR_gC", "view", "partition_D(tCgC)")
    dot.edge("tCgC", "tTR_gC", "view", "destination coordinates")
    dot.edge("tTR_tAcc", "ldtm", "view", "source tile")
    dot.edge("ldtm", "tTR_rAcc", "data", "tcgen05.ld / cute.copy")
    dot.edge("tTR_gC", "tTR_rAcc", "view", "make_rmem_tensor(...)", constraint="false")
    dot.edge("tTR_rAcc", "store", "data", "register values")
    dot.edge("tTR_gC", "store", "view", "store coordinates")
    dot.edge("store", "gmem_output", "data", "cute.copy(store_atom, ...)")

    dot.add("  }")


def build_convergence(dot: DotBuilder) -> None:
    dot.node("pipeline", "sync", "AB pipeline", "producer commit / consumer wait", "per K tile")
    dot.node("accumulate", "sync", "ACCUMULATE field", "False first K tile, True after", "")
    dot.node("mma", "compute", "tcgen05 MMA main loop", "cute.gemm(...)", "reads A/B, updates tCtAcc")

    dot.edge("copyA", "pipeline", "sync", "A stage full", constraint="false")
    dot.edge("copyB", "pipeline", "sync", "B stage full", constraint="false")
    dot.edge("pipeline", "mma", "sync", "ab_consumer.wait_and_advance()")
    dot.edge("accumulate", "mma", "sync", "set before cute.gemm")
    dot.edge("tCrA", "mma", "consume", "A descriptor", constraint="false")
    dot.edge("tCrB", "mma", "consume", "B descriptor", constraint="false")
    dot.edge("tCtAcc", "mma", "consume", "accumulator input", constraint="false")
    dot.edge("mma", "tCtAcc", "consume", "compute update in TMEM")
    dot.edge("mma", "mma", "loop", "for k_tile_idx", constraint="false")
    dot.edge("mma", "ldtm", "sync", "main loop complete", constraint="false")


def build_legend(dot: DotBuilder) -> None:
    dot.add("  subgraph cluster_legend {")
    dot.add('    label="Legend";')
    dot.add('    style="filled,rounded";')
    dot.add(f'    color="{COLORS["legend"]}";')
    dot.add(f'    fillcolor="{COLORS["legend"]}";')
    dot.add("    margin=12;")

    dot.node("legend_storage", "gmem", "Physical storage", "GMEM / SMEM / TMEM / RMEM", "")
    dot.node("legend_view", "view", "Tensor view", "tile / partition", "owns no data")
    dot.node("legend_config", "config", "Config object", "TmaInfo, copy atom, tiled copy", "")
    dot.node("legend_desc", "descriptor", "Descriptor", "hardware-consumable SMEM view", "")
    dot.node("legend_copy", "copy", "Actual copy", "TMA, LDTM, store", "")
    dot.node("legend_compute", "compute", "Compute update", "tcgen05.mma updates TMEM", "")
    dot.node("legend_sync", "sync", "Sync / control", "pipeline and loop ordering", "")

    dot.edge("legend_storage", "legend_copy", "data", "actual data movement")
    dot.edge("legend_view", "legend_desc", "view", "logical derivation")
    dot.edge("legend_config", "legend_view", "view", "configures view/copy")
    dot.edge("legend_desc", "legend_compute", "consume", "operand consumption")
    dot.edge("legend_sync", "legend_compute", "sync", "control dependency")
    dot.add("  }")


def build_dot() -> str:
    dot = DotBuilder()
    dot.add("digraph tcgen05_gemm_dataflow {")
    dot.add("  graph [")
    dot.add('    rankdir="TB",')
    dot.add('    compound="true",')
    dot.add('    splines="ortho",')
    dot.add('    nodesep="0.38",')
    dot.add('    ranksep="0.58",')
    dot.add('    pad="0.16",')
    dot.add('    margin="0.02",')
    dot.add('    bgcolor="white",')
    dot.add(f'    fontname="{FONT}",')
    dot.add('    labelloc="t",')
    dot.add('    label="tcgen05 GEMM Dataflow: Tile Partitioning, Thread Partitioning, and Storage Mapping",')
    dot.add('    fontsize="18",')
    dot.add('    size="30,18",')
    dot.add('    ratio="compress"')
    dot.add("  ];")
    dot.add(f'  node [fontname="{FONT}", fontsize="10", margin="0.07,0.05"];')
    dot.add(f'  edge [fontname="{FONT}", fontsize="9"];')

    build_operand_lane(dot, "A")
    build_operand_lane(dot, "B")
    build_accumulator_lane(dot)
    build_convergence(dot)
    build_legend(dot)

    dot.add('  tCrA -> legend_storage [style="invis", weight="8"];')
    dot.add('  tCrB -> legend_storage [style="invis", weight="8"];')
    dot.add('  gmem_output -> legend_storage [style="invis", weight="8"];')
    dot.add("}")
    return "\n".join(dot.lines) + "\n"


def render(dot_source: str, output_dir: Path, formats: list[str]) -> None:
    dot_exe = shutil.which("dot")
    if dot_exe is None:
        raise RuntimeError(
            "Graphviz executable 'dot' was not found. Install Graphviz first, "
            "for example: sudo apt-get install graphviz, brew install graphviz, "
            "or conda install -c conda-forge graphviz."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tcgen05_gemm_dataflow_") as temp_dir:
        dot_path = Path(temp_dir) / "tcgen05_gemm_dataflow.dot"
        dot_path.write_text(dot_source, encoding="utf-8")
        for fmt in formats:
            output_path = output_dir / f"tcgen05_gemm_dataflow.{fmt}"
            cmd = [dot_exe, f"-T{fmt}", str(dot_path), "-o", str(output_path)]
            try:
                subprocess.run(cmd, check=True, text=True, capture_output=True)
            except subprocess.CalledProcessError as exc:
                message = exc.stderr.strip() or exc.stdout.strip()
                raise RuntimeError(f"Graphviz failed while writing {output_path}: {message}") from exc
            print(f"Wrote {output_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SVG/PNG diagrams for the tcgen05 GEMM dataflow note."
    )
    parser.add_argument(
        "--format",
        choices=("svg", "png", "all"),
        default="all",
        help="Output format to generate. Defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated images. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    formats = ["svg", "png"] if args.format == "all" else [args.format]
    try:
        render(build_dot(), args.output_dir, formats)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
