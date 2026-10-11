#!/usr/bin/env python3
"""R17 runner: cfg_a mainloop slowdown when the 2x1 cluster pads a fully out-of-bounds tile row.

Reuses the V06 machinery unchanged (v06_run: overlay/compile/SASS facts/setup/one process;
v06_model: scheduler and trace parsing). Adds config cfg_a1 = cfg_a with cluster 1x1.
  prepare --output RUN --cutlass-root CUTLASS   (off-node) copy sources + headers, overlay
  build   --output RUN                          4 binaries: cfg_a / cfg_a1 x plain / stamped
  setup   --output RUN                          grid of every case (no GEMM run)
  sample  --output RUN --procs N                N plain + N stamped processes per case
All build paths are relative to RUN, so RUN can be copied to node /tmp and back.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil

import v06_model as model
import v06_run

ROOT = Path(__file__).resolve().parent
model.CONFIGS["cfg_a1"] = dict(index=3, tile=(128, 128, 64), cluster=(1, 1),
                               schedule="cooperative")
v06_run.EXPECTED_HGMMA["cfg_a1"] = 8  # same tile/schedule as cfg_a
R17_INDEX = {"cfg_a": 0, "cfg_a1": 3}

# (group, config, M, N, K). Tile rows = ceil(M/128); the 2x1 cluster pads an odd row count by
# one fully out-of-bounds row. Pairs differ in one factor (see R17-oob-tile.md).
CASES = [
    # N=1664, K=2560 (V06 partial_wave size): OOB row vs none; partial last row; cluster off.
    ("g2560", "cfg_a", 1152, 1664, 2560),   # 9 rows -> padded OOB row 9
    ("g2560", "cfg_a", 1280, 1664, 2560),   # 10 rows: same tile list, row 9 real
    ("g2560", "cfg_a", 1100, 1664, 2560),   # 9 rows (last partial) + padded OOB row
    ("g2560", "cfg_a", 1200, 1664, 2560),   # 10 rows, last partial, no OOB row
    ("g2560", "cfg_a1", 1152, 1664, 2560),
    ("g2560", "cfg_a1", 1280, 1664, 2560),
    ("g2560", "cfg_a1", 1100, 1664, 2560),
    # N=2304, K=10240 (V06 long_k size), 2 rounds.
    ("g10240", "cfg_a", 1408, 2304, 10240),  # 11 rows -> OOB row 11
    ("g10240", "cfg_a", 1536, 2304, 10240),  # 12 rows
    ("g10240", "cfg_a", 1400, 2304, 10240),  # 11 rows (last partial) + OOB row
    ("g10240", "cfg_a", 1500, 2304, 10240),  # 12 rows, last partial
    ("g10240", "cfg_a1", 1408, 2304, 10240),
    ("g10240", "cfg_a1", 1536, 2304, 10240),
    # K dependence at N=2304.
    ("kdep", "cfg_a", 1408, 2304, 1024),
    ("kdep", "cfg_a", 1536, 2304, 1024),
    ("kdep", "cfg_a", 1408, 2304, 4096),
    ("kdep", "cfg_a", 1536, 2304, 4096),
    # Another shape: 13 vs 14 rows, N=2048, K=4096.
    ("other", "cfg_a", 1664, 2048, 4096),
    ("other", "cfg_a", 1792, 2048, 4096),
]


def case_rows():
    return [dict(id=f"{cfg}_{m}x{n}x{k}", group=g, config=cfg, label=g, m=m, n=n, k=k,
                 lda=k, ldb=n, ldd=n) for g, cfg, m, n, k in CASES]


def prepare(output: Path, cutlass: Path):
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    (source / "probes").mkdir(parents=True)
    (output / "build").mkdir()
    for name in ("r17_run.py", "v06_run.py", "v06_model.py"):  # analysis runs off-node
        shutil.copy2(ROOT / name, source / name)
    for name in ("r17.cu", "v06_trace.hpp", "gaps_common.hpp", "r00_common.hpp"):
        shutil.copy2(ROOT / "probes" / name, source / "probes" / name)
    for part in ("include", "tools/util/include"):
        shutil.copytree(cutlass / part, source / "cutlass" / part)
    v06_run.make_overlay(source / "cutlass", source / "overlay")
    v06_run.write_json(output / "cases.json", case_rows())
    commands = {}
    for config, index in R17_INDEX.items():
        for variant in ("plain", "stamped"):
            cmd = ["nvcc", "-std=c++17", "-O3", "-DNDEBUG", "-gencode=arch=compute_90a,code=sm_90a",
                   "-lineinfo", "--ptxas-options=-v", f"-DR17_CFG={index}"]
            if variant == "stamped":
                cmd += ["-DV06_TRACE", "-Isource/overlay"]
            cmd += ["-Isource/cutlass/include", "-Isource/cutlass/tools/util/include",
                    "source/probes/r17.cu", "-o", f"build/{config}_{variant}"]
            commands[f"{config}_{variant}"] = cmd
    v06_run.write_json(output / "build/commands.json", commands)
    v06_run.write_json(output / "source_hashes.json",
                       {str(p.relative_to(output)): model.sha(p)
                        for p in sorted(source.rglob("*")) if p.is_file()})


def sample(output: Path, procs: int):
    v06_run.verify(output)
    env = v06_run.identity()
    if env["gpu"] != json.loads((output / "environment.json").read_text())["gpu"]:
        raise ValueError("GPU changed since setup")
    rows = json.loads((output / "cases.json").read_text())
    rng = random.Random(20261017)
    for trial in range(procs):
        rng.shuffle(rows)
        for row in rows:
            variants = ["plain", "stamped"]
            rng.shuffle(variants)
            for variant in variants:
                v06_run.run_process(output, "r17", row, variant, trial)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("prepare", "build", "setup", "sample", "list"))
    p.add_argument("--output", type=Path)
    p.add_argument("--cutlass-root", type=Path)
    p.add_argument("--procs", type=int, default=10)
    a = p.parse_args()
    if a.step == "list":
        for r in case_rows():
            grid = model.emulate_grid(r["config"], r["m"], r["n"])
            work = model.scheduled_work(r["config"], r["m"], r["n"], grid)
            counts = [len(w) for w in work]
            print(f"{r['id']:24} rows {model.cdiv(r['m'], 128):3} grid {grid} "
                  f"tiles {sum(counts):4} rounds {max(counts)}")
        return
    out = a.output.resolve()
    if a.step == "prepare":
        prepare(out, a.cutlass_root.resolve())
    elif a.step == "build":
        v06_run.build(out)
    elif a.step == "setup":
        v06_run.setup(out)
    else:
        sample(out, a.procs)


if __name__ == "__main__":
    main()
