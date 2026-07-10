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
    "loop": "#f9fafb",
    "lane": "#f8fafc",
    "legend": "#ffffff",
}

NODE_STYLES = {
    "gmem": {
        "style": "filled,rounded",
        "fillcolor": COLORS["gmem"],
        "color": COLORS["gmem_border"],
        "shape": "box",
    },
    "smem": {
        "style": "filled,rounded",
        "fillcolor": COLORS["smem"],
        "color": COLORS["smem_border"],
        "shape": "box3d",
    },
    "tmem": {
        "style": "filled,rounded",
        "fillcolor": COLORS["tmem"],
        "color": COLORS["tmem_border"],
        "shape": "box3d",
    },
    "rmem": {
        "style": "filled,rounded",
        "fillcolor": COLORS["rmem"],
        "color": COLORS["rmem_border"],
        "shape": "box",
    },
    "view": {
        "style": "filled,rounded,dashed",
        "fillcolor": COLORS["view"],
        "color": COLORS["view_border"],
        "shape": "box",
    },
    "descriptor": {
        "style": "filled,rounded",
        "fillcolor": COLORS["descriptor"],
        "color": COLORS["descriptor_border"],
        "shape": "component",
    },
    "copy": {
        "style": "filled,rounded",
        "fillcolor": COLORS["copy"],
        "color": COLORS["copy_border"],
        "shape": "box",
    },
    "compute": {
        "style": "filled,rounded,bold",
        "fillcolor": COLORS["compute"],
        "color": COLORS["compute_border"],
        "shape": "box",
    },
    "loop": {
        "style": "filled,rounded,dashed",
        "fillcolor": COLORS["loop"],
        "color": "#98a2b3",
        "shape": "box",
    },
    "sync": {
        "style": "filled,rounded",
        "fillcolor": COLORS["sync"],
        "color": COLORS["sync_border"],
        "shape": "box",
    },
}

EDGE_STYLES = {
    "data": {
        "color": "#1f2937",
        "penwidth": "2.4",
        "style": "solid",
        "arrowsize": "0.9",
    },
    "view": {
        "color": "#667085",
        "penwidth": "1.5",
        "style": "dashed",
        "arrowsize": "0.75",
    },
    "consume": {
        "color": "#c53030",
        "penwidth": "2.0",
        "style": "dotted",
        "arrowsize": "0.85",
    },
    "loop": {
        "color": "#475467",
        "penwidth": "1.3",
        "style": "dashed",
        "arrowsize": "0.7",
    },
    "sync": {
        "color": "#4f46e5",
        "penwidth": "1.6",
        "style": "dashdot",
        "arrowsize": "0.75",
    },
}


def attrs(values: dict[str, str]) -> str:
    rendered = []
    for key, value in values.items():
        if key == "label" and value.startswith("<"):
            rendered.append(f"{key}={value}")
        else:
            rendered.append(f'{key}="{value}"')
    return ", ".join(rendered)


def label(title: str, subtitle: str = "", shape: str = "") -> str:
    rows = [f'<FONT POINT-SIZE="14"><B>{html.escape(title)}</B></FONT>']
    if subtitle:
        rows.append(f'<FONT POINT-SIZE="10">{html.escape(subtitle)}</FONT>')
    if shape:
        rows.append(f'<FONT POINT-SIZE="10">{html.escape(shape)}</FONT>')
    return "<" + "<BR/>".join(rows) + ">"


class DotBuilder:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def node(self, name: str, kind: str, title: str, subtitle: str = "", shape: str = "") -> None:
        node_attrs = {
            **NODE_STYLES[kind],
            "label": label(title, subtitle, shape),
        }
        self.add(f"    {name} [{attrs(node_attrs)}];")

    def edge(self, source: str, target: str, kind: str, edge_label: str = "", **extra: str) -> None:
        edge_attrs = {**EDGE_STYLES[kind], **extra}
        if edge_label:
            edge_attrs["label"] = edge_label
            edge_attrs["fontsize"] = "10"
            edge_attrs["fontcolor"] = edge_attrs["color"]
        self.add(f"    {source} -> {target} [{attrs(edge_attrs)}];")


def build_operand_path(dot: DotBuilder, operand: str) -> None:
    lower = operand.lower()
    shape = {
        "A": {
            "matrix": "(M, K)",
            "tile": "(BM, BK, k)",
            "partition": "(MMA, MMA_M, MMA_K, k)",
            "fragment": "(MMA, MMA_M, MMA_K, STAGE)",
            "tma_atom": "tma_a.atom",
            "tma_s": "tAsA",
            "tma_g": "tAgA",
            "copy": "copy(tma_a.atom, tAgA[k_tile], tAsA[stage])",
            "fragment_fn": "make_fragment_A(sA)",
            "partition_fn": "thr_mma.partition_A(gA)",
            "tma_fn": "cpasync.tma_partition(...)",
        },
        "B": {
            "matrix": "(N, K)",
            "tile": "(BN, BK, k)",
            "partition": "(MMA, MMA_N, MMA_K, k)",
            "fragment": "(MMA, MMA_N, MMA_K, STAGE)",
            "tma_atom": "tma_b.atom",
            "tma_s": "tBsB",
            "tma_g": "tBgB",
            "copy": "copy(tma_b.atom, tBgB[k_tile], tBsB[stage])",
            "fragment_fn": "make_fragment_B(sB)",
            "partition_fn": "thr_mma.partition_B(gB)",
            "tma_fn": "cpasync.tma_partition(...)",
        },
    }[operand]

    dot.add(f"  subgraph cluster_operand_{lower} {{")
    dot.add(f'    label="Operand {operand} Dataflow Path";')
    dot.add(f'    color="{COLORS["lane"]}";')
    dot.add('    style="filled,rounded";')
    dot.add(f'    fillcolor="{COLORS["lane"]}";')
    dot.add("    margin=18;")

    dot.node(f"m{operand}", "gmem", f"m{operand}", "Original GMEM tensor", shape["matrix"])
    dot.node(
        f"tmaAtom{operand}",
        "copy",
        shape["tma_atom"],
        "TMA copy atom",
        "CopyBulkTensorTileG2SOp",
    )
    dot.node(f"g{operand}", "view", f"g{operand}", "CTA/local GMEM tile", shape["tile"])
    dot.node(f"tCg{operand}", "view", f"tCg{operand}", "MMA thread partition", shape["partition"])
    dot.node(f"s{operand}", "smem", f"s{operand}", "Physical SMEM allocation", f"staged {operand} tile")
    dot.node(
        f"tCr{operand}",
        "descriptor",
        f"tCr{operand}",
        "SMEM descriptor tensor",
        shape["fragment"],
    )
    dot.node(
        f"tma{operand}",
        "view",
        f"{shape['tma_s']} / {shape['tma_g']}",
        "TMA-partitioned SMEM / GMEM views",
        "stage and K-tile indexed",
    )
    dot.node(f"copy{operand}", "copy", f"TMA copy loop {operand}", shape["copy"], "GMEM -> SMEM")

    dot.edge(f"m{operand}", f"tmaAtom{operand}", "view", "tma_info.tma_tensor", constraint="false")
    dot.edge(f"m{operand}", f"g{operand}", "view", f"local_tile(m{operand}, mma_tiler_mnk, mma_coord_mnk)")
    dot.edge(f"g{operand}", f"tCg{operand}", "view", shape["partition_fn"])
    dot.edge(f"s{operand}", f"tCr{operand}", "view", shape["fragment_fn"])
    dot.edge(f"tmaAtom{operand}", f"tma{operand}", "view", shape["tma_fn"], constraint="false")
    dot.edge(f"tCg{operand}", f"tma{operand}", "view", "group_modes(tCg*, 0, 3)")
    dot.edge(f"s{operand}", f"tma{operand}", "view", "group_modes(s*, 0, 3)", constraint="false")
    dot.edge(f"tma{operand}", f"copy{operand}", "view", f"{shape['tma_g']}[k_tile], {shape['tma_s']}[stage]")
    dot.edge(f"copy{operand}", f"s{operand}", "data", f"writes staged {operand}")
    dot.edge(f"copy{operand}", f"copy{operand}", "loop", "for k_tile / stage", constraint="false")

    dot.add("  }")


def build_accumulator_path(dot: DotBuilder) -> None:
    dot.add("  subgraph cluster_accumulator {")
    dot.add('    label="Accumulator C/D Dataflow Path";')
    dot.add(f'    color="{COLORS["lane"]}";')
    dot.add('    style="filled,rounded";')
    dot.add(f'    fillcolor="{COLORS["lane"]}";')
    dot.add("    margin=18;")

    dot.node("acc_shape", "view", "acc_shape", "MMA accumulator shape", "(MMA, MMA_M, MMA_N)")
    dot.node("tmem_alloc", "tmem", "TmemAllocator", "allocate + retrieve_ptr", "TMEM columns")
    dot.node("tCtAcc_layout", "view", "tCtAcc layout", "Accumulator fragment layout", "(MMA, MMA_M, MMA_N)")
    dot.node("tCtAcc", "tmem", "tCtAcc", "TMEM accumulator tensor", "(MMA, MMA_M, MMA_N[, ACC_STAGE])")
    dot.node("mC", "gmem", "mC", "Output GMEM tensor", "(M, N)")
    dot.node("gC", "view", "gC", "CTA/local output tile", "(BM, BN)")
    dot.node("tCgC", "view", "tCgC", "Epilogue GMEM partition", "(MMA, MMA_M, MMA_N)")
    dot.node("copy_atom_t2r", "copy", "copy_atom_t2r", "Ld32x32bOp", "Repetition.x64")
    dot.node("t2r", "copy", "tiled_copy_t2r", "TMEM -> RMEM copy object", "anchored by tCtAcc slice")
    dot.node("thr_copy_t2r", "view", "thr_copy_t2r", "tiled_copy_t2r.get_slice(tidx)", "per-thread copy slice")
    dot.node("tTR_tAcc", "view", "tTR_tAcc", "thr_copy_t2r.partition_S(tCtAcc)", "(T2R, T2R_M, NumTiles)")
    dot.node("tTR_gC", "view", "tTR_gC", "thr_copy_t2r.partition_D(tCgC)", "(T2R, T2R_M, NumTiles)")
    dot.node("tTR_rAcc", "rmem", "tTR_rAcc", "RMEM accumulator fragment", "(T2R, T2R_M)")
    dot.node("ldtm", "copy", "LDTM copy", "TMEM -> RMEM", "for i in NumTiles")
    dot.node("store", "copy", "Epilogue store", "RMEM -> GMEM", "for i in NumTiles")

    dot.edge("acc_shape", "tCtAcc_layout", "view", "tiled_mma.make_fragment_C(acc_shape)")
    dot.edge("tmem_alloc", "tCtAcc", "view", "retrieve_ptr(Float32)")
    dot.edge("tCtAcc_layout", "tCtAcc", "view", "cute.make_tensor(tmem_ptr, layout)")
    dot.edge("mC", "gC", "view", "local_tile(mC, mma_tiler_mnk, mma_coord_mnk)")
    dot.edge("gC", "tCgC", "view", "thr_mma.partition_C(gC)")
    dot.edge("copy_atom_t2r", "t2r", "view", "tcgen05.make_tmem_copy(...)")
    dot.edge("tCtAcc", "t2r", "view", "tCtAcc[(None,None),0,0]")
    dot.edge("t2r", "thr_copy_t2r", "view", "get_slice(tidx)")
    dot.edge("thr_copy_t2r", "tTR_tAcc", "view", "partition_S(tCtAcc)")
    dot.edge("thr_copy_t2r", "tTR_gC", "view", "partition_D(tCgC)")
    dot.edge("tCgC", "tTR_gC", "view", "destination layout")
    dot.edge("tTR_tAcc", "ldtm", "view", "source tile")
    dot.edge("tTR_gC", "tTR_rAcc", "view", "make_rmem_tensor(tTR_gC[None,None,0].shape)", constraint="false")
    dot.edge("ldtm", "tTR_rAcc", "data", "cute.copy(tiled_copy_t2r, tTR_tAcc[None,None,i], tTR_rAcc)")
    dot.edge("tTR_rAcc", "store", "data", "register values")
    dot.edge("tTR_gC", "store", "view", "store coordinates")
    dot.edge("store", "mC", "data", "cute.copy(store_atom, tTR_rAcc, tTR_gC[None,None,i])", constraint="false")

    dot.add("  }")


def build_execution_and_legend(dot: DotBuilder) -> None:
    dot.add("  subgraph cluster_execution {")
    dot.add('    label="MMA Execution and Pipeline";')
    dot.add(f'    color="{COLORS["legend"]}";')
    dot.add('    style="filled,rounded";')
    dot.add(f'    fillcolor="{COLORS["legend"]}";')
    dot.add("    margin=18;")
    dot.node("pipeline", "loop", "AB pipeline", "producer/consumer stages", "repeat over K tiles")
    dot.node("accumulate", "sync", "ACCUMULATE field", "False for first K tile, True after", "tcgen05.Field.ACCUMULATE")
    dot.node(
        "mma",
        "compute",
        "tcgen05 MMA main loop",
        "cute.gemm(...)",
        "tCrA/B[tile_crd], tCtAcc",
    )
    dot.add("  }")

    dot.add("  subgraph cluster_legend {")
    dot.add('    label="Legend";')
    dot.add(f'    color="{COLORS["legend"]}";')
    dot.add('    style="filled,rounded";')
    dot.add(f'    fillcolor="{COLORS["legend"]}";')
    dot.add("    margin=16;")
    dot.node("legend_data_a", "gmem", "Physical storage", "GMEM / SMEM / TMEM / RMEM", "")
    dot.node("legend_view_a", "view", "Tensor view", "tile, partition, descriptor derivation", "owns no data")
    dot.node("legend_desc_a", "descriptor", "Fragment descriptor", "hardware-consumable SMEM descriptor", "")
    dot.node("legend_copy_a", "copy", "Copy operation", "TMA, LDTM, epilogue store", "")
    dot.node("legend_sync_a", "sync", "Pipeline / control", "barrier wait, stage ready, accumulate field", "")
    dot.node("legend_mma_a", "compute", "Compute operation", "tcgen05.mma consumes operands", "")
    dot.edge("legend_data_a", "legend_copy_a", "data", "actual data movement")
    dot.edge("legend_view_a", "legend_desc_a", "view", "logical view derivation")
    dot.edge("legend_copy_a", "legend_sync_a", "sync", "control dependency")
    dot.edge("legend_desc_a", "legend_mma_a", "consume", "hardware consumption")
    dot.add("  }")

    dot.edge("copyA", "pipeline", "sync", "producer commit: A stage full", constraint="false")
    dot.edge("copyB", "pipeline", "sync", "producer commit: B stage full", constraint="false")
    dot.edge("pipeline", "mma", "sync", "ab_consumer.wait_and_advance()")
    dot.edge("accumulate", "mma", "sync", "set before cute.gemm")
    dot.edge("tCrA", "mma", "consume", "A descriptor/view", constraint="false")
    dot.edge("tCrB", "mma", "consume", "B descriptor/view", constraint="false")
    dot.edge("tCtAcc", "mma", "consume", "accumulator input", constraint="false")
    dot.edge("mma", "tCtAcc", "data", "TMEM result update")
    dot.edge("mma", "mma", "loop", "for k_tile_idx", constraint="false")
    dot.edge("mma", "t2r", "data", "after main loop")


def build_dot() -> str:
    dot = DotBuilder()
    dot.add("digraph tcgen05_gemm_dataflow {")
    dot.add("  graph [")
    dot.add('    rankdir="TB",')
    dot.add('    compound="true",')
    dot.add('    splines="ortho",')
    dot.add('    nodesep="0.55",')
    dot.add('    ranksep="0.85",')
    dot.add('    pad="0.25",')
    dot.add('    bgcolor="white",')
    dot.add(f'    fontname="{FONT}",')
    dot.add('    labelloc="t",')
    dot.add('    label="tcgen05 GEMM Dataflow: Tile Partitioning, Thread Partitioning, and Storage Mapping",')
    dot.add('    fontsize="20"')
    dot.add("  ];")
    dot.add(f'  node [fontname="{FONT}", fontsize="11", margin="0.09,0.07"];')
    dot.add(f'  edge [fontname="{FONT}", fontsize="10"];')
    build_operand_path(dot, "A")
    build_operand_path(dot, "B")
    build_accumulator_path(dot)
    build_execution_and_legend(dot)
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
