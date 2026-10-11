#!/usr/bin/env python3
"""R08: wave/tail scaling and tile traversal (L2 reuse) of the fixed NDEBUG CUTLASS GEMM.

Run inside one single-GPU Slurm allocation (CUDA 12.9 loaded):
  r08_run.py build  --cutlass-root DIR --output RUN
  r08_run.py sample --output RUN --set {pilot,waves,stamps,l2}
Each process is one independent sample; rows are appended to RUN/samples.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
K_WAVES = 4096
# (label tiles, M, N): tile 128x256, so tiles = ceil(M/128) * ceil(N/256).
# Odd tiles_m (133, 265) is padded by the 2x1 cluster to an even count; 134/266 are the
# nearest counts with one extra cluster pair over a full wave.
WAVE_SHAPES = [
    (64, 1024, 2048), (96, 1536, 2048), (128, 2048, 2048), (132, 1536, 2816),
    (133, 896, 4864), (134, 256, 17152), (136, 1024, 4352), (160, 2048, 2560),
    (192, 2048, 3072), (220, 2560, 2816), (264, 3072, 2816), (265, 640, 13568),
    (266, 1792, 4864), (330, 3840, 2816), (396, 4608, 2816), (528, 6144, 2816),
    ("2000sq", 2000, 2000), ("3000sq", 3000, 3000), ("4000sq", 4000, 4000),
]
STAMP_LABELS = (128, 132, 136, 192, 264, 266, 528, "2000sq", "4000sq")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run_text(command):
    return subprocess.run(command, capture_output=True, text=True).stdout


def build(output: Path, cutlass: Path):
    source, directory = output / "source", output / "build"
    (source / "probes").mkdir(parents=True)
    directory.mkdir()
    for name in ("r00_common.hpp", "r00_cutlass.cu", "r08_probe.cu"):
        shutil.copy2(ROOT / "probes" / name, source / "probes" / name)
    for name in ("r08_run.py", "r08_analyze.py"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, source / name)
    include = [f"-I{cutlass}/include", f"-I{cutlass}/tools/util/include"]
    base = ["nvcc", "-std=c++17", "-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
            "--ptxas-options=-v", "-DNDEBUG"]
    probe = str(source / "probes/r08_probe.cu")
    targets = {
        "r08_probe_c2": base + include + [probe],
        "r08_probe_c1": base + ["-DR08_CLUSTER_M=1"] + include + [probe],
    }
    commands, sass = {}, {}
    for name, command in targets.items():
        command = command + ["-o", str(directory / name)]
        commands[name] = command
        print("compile", name, flush=True)
        with (directory / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        with (directory / f"{name}.sass").open("w") as log:
            subprocess.run(["cuobjdump", "--dump-sass", str(directory / name)], stdout=log,
                           check=True, timeout=120)
        sass[name] = sass_summary(directory / f"{name}.sass", directory / f"{name}.log")
    write_json(directory / "commands.json", commands)
    write_json(directory / "sass_check.json", sass)
    write_json(directory / "binary_hashes.json", {n: sha(directory / n) for n in targets})
    hashes = {str(p.relative_to(output)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()}
    write_json(output / "source_hashes.json", hashes)
    write_json(output / "environment.json", dict(
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), hostname=os.uname().nodename,
        gpu=run_text(["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.max.sm,"
                      "enforced.power.limit", "--format=csv,noheader"]).strip(),
        nvcc=run_text(["nvcc", "--version"]),
        cutlass_root=str(cutlass),
        cutlass_version_h_sha256=sha(cutlass / "include/cutlass/version.h"),
        cutlass_gemm_adapter_sha256=sha(
            cutlass / "include/cutlass/gemm/device/gemm_universal_adapter.h"),
    ))
    (output / "nvidia-smi-power.txt").write_text(run_text(["nvidia-smi", "-q", "-d", "POWER"]))
    (output / "nvidia-smi-clock.txt").write_text(run_text(["nvidia-smi", "-q", "-d", "CLOCK"]))


def sass_summary(sass_path: Path, log_path: Path):
    """Per GEMM function: HGMMA and wait counts; C7510 lines in the ptxas log."""
    functions, current = {}, None
    for line in sass_path.read_text().splitlines():
        match = re.search(r"Function : (\S+)", line)
        if match:
            current = match.group(1)
            functions[current] = dict(hgmma=0, warpgroup_wait=0, instructions=0)
        elif current and re.search(r"/\*[0-9a-f]{4,}\*/", line):
            functions[current]["instructions"] += 1
            functions[current]["hgmma"] += "HGMMA" in line
            functions[current]["warpgroup_wait"] += "WARPGROUP.DEPBAR" in line
    gemm = {name: v for name, v in functions.items() if v["hgmma"]}
    log = log_path.read_text()
    return dict(c7510_lines=log.count("C7510"), gemm_functions=gemm)


def wave_case(m, n, mode="time", cluster=2, raster="h", swizzle=1, extra=()):
    return dict(binary=f"r08_probe_c{cluster}",
                args=["--mode", mode, "--m", str(m), "--n", str(n), "--k", str(K_WAVES),
                      "--raster", raster, "--swizzle", str(swizzle), *extra])


def case_sets():
    waves = {f"wave_{label}": wave_case(m, n, extra=("--calls", "5", "--graph", "10"))
             for label, m, n in WAVE_SHAPES}
    shapes = {label: (m, n) for label, m, n in WAVE_SHAPES}
    stamps = {f"stamp_{label}": wave_case(*shapes[label], mode="stamp",
                                          extra=("--launches", "3", "--idle-s", "0.3"))
              for label in STAMP_LABELS}
    l2 = {}
    for cluster in (2, 1):
        for raster in ("m", "n"):
            for swizzle in (1, 2, 4, 8):
                l2[f"l2_c{cluster}_r{raster}_s{swizzle}"] = wave_case(
                    8192, 8192, cluster=cluster, raster=raster, swizzle=swizzle,
                    extra=("--calls", "5"))
    pilot = {
        "pilot_wave_128": waves["wave_128"],
        "pilot_wave_264": waves["wave_264"],
        "pilot_wave_2000sq": waves["wave_2000sq"],
        "pilot_wave_133": waves["wave_133"],
        "pilot_stamp_264": stamps["stamp_264"],
        "pilot_l2_c2_rn_s1": l2["l2_c2_rn_s1"],
        "pilot_l2_c2_rm_s8": l2["l2_c2_rm_s8"],
        "pilot_l2_c1_rn_s1": l2["l2_c1_rn_s1"],
    }
    return dict(pilot=(pilot, 2), waves=(waves, 10), stamps=(stamps, 3), l2=(l2, 10))


def run_one(output: Path, case_id, case, trial):
    folder = output / "samples" / case_id / f"trial-{trial:02d}"
    if (folder / "result.json").exists():
        return
    if folder.exists():
        folder = folder.with_name(folder.name + f"-retry-{time.time_ns()}")
    folder.mkdir(parents=True)
    command = [str(output / "build" / case["binary"])] + case["args"]
    write_json(folder / "command.json", command)
    started = time.time_ns()
    with (folder / "stdout.txt").open("w") as out, (folder / "stderr.log").open("w") as err:
        process = subprocess.run(command, cwd=folder, stdout=out, stderr=err, timeout=300)
    ended = time.time_ns()
    lines = [json.loads(x) for x in (folder / "stdout.txt").read_text().splitlines() if x.strip()]
    by_event = {}
    for line in lines:
        by_event.setdefault(line.get("event"), []).append(line)
    record = dict(case_id=case_id, trial=trial, binary=case["binary"], args=case["args"],
                  returncode=process.returncode, started_unix_ns=started, ended_unix_ns=ended,
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  config=(by_event.get("config") or [None])[0],
                  time=(by_event.get("time") or [None])[0],
                  check=(by_event.get("check") or [None])[0],
                  stamp_launches=len(by_event.get("stamp", [])))
    record["status"] = ("measured" if process.returncode == 0 and record["check"]
                        and record["check"]["status"] == "ok" else "failed")
    write_json(folder / "result.json", record)
    with (output / "samples.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    shown = record["time"]["call_median_us"] if record["time"] else record["stamp_launches"]
    print(case_id, trial, shown, record["status"], flush=True)
    if record["status"] != "measured":
        raise RuntimeError(f"{case_id} trial {trial} failed: {folder}")


def sample(output: Path, name):
    cases, rounds = case_sets()[name]
    write_json(output / f"protocol-{name}.json", dict(cases=cases, rounds=rounds, seed=20261008))
    rng = random.Random(20261008)
    for trial in range(rounds):
        order = list(cases)
        rng.shuffle(order)
        for case_id in order:
            run_one(output, case_id, cases[case_id], trial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("build", "sample"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutlass-root", type=Path)
    parser.add_argument("--set", choices=tuple(case_sets()))
    args = parser.parse_args()
    if args.step == "build":
        args.output.mkdir(parents=True, exist_ok=False)
        build(args.output.resolve(), args.cutlass_root.resolve())
    else:
        sample(args.output.resolve(), args.set)


if __name__ == "__main__":
    main()
