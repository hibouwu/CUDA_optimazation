#!/usr/bin/env python3
"""Run fixed-work R04 joint-service points in one Slurm allocation.

`--family cross` runs the 8 cross-warpgroup points (A-only, B-only and the 6 mixed points)
built with the asynchronous wait_group 1 pipeline.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import re
import sys

ROOT = Path(__file__).resolve().parent
FAMILIES = ("wgmma_ffma_same", "wgmma_ffma_cross", "lds_ffma", "lds_cvt")


def matrix():
    cases = []
    for family, name in enumerate(FAMILIES):
        for order, na, nb, label in [(2, 160, 0, "a_only"), (3, 0, 160, "b_only")] + [
            (o, a, b, f"{a}:{b}_{tag}")
            for a, b in [(80, 80), (32, 128), (128, 32)]
            for o, tag in [(0, "serial"), (1, "interleaved")]
        ]:
            cases.append(dict(id=f"{name}_{label}", family=family, order=order, a=na, b=nb))
    return cases


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(output, case, trial, repeats):
    folder = output / "samples" / case["id"] / trial
    folder.mkdir(parents=True, exist_ok=False)
    command = [str(output / "build/r04")]
    for k in ("family", "order", "a", "b"):
        command.extend(["--" + k, str(case[k])])
    command.extend(["--repeats", str(repeats)])
    write(folder / "command.json", command)
    with (folder / "stdout.json").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
        result = subprocess.run(command, cwd=folder, stdout=stdout, stderr=stderr, timeout=180)
    write(folder / "process.json", dict(returncode=result.returncode))
    if result.returncode:
        raise RuntimeError(f"process failed: {folder}")
    record = json.loads((folder / "stdout.json").read_text())
    if record["status"] != "measured":
        raise RuntimeError(f"numerical failure: {folder}")
    record.update(
        configuration=case, trial_role=trial.split("-")[0], output_sha256=sha(folder / "output.f32")
    )
    if "ld.volatile.shared.v4.b32" in (output / "source/probes/r04.cu").read_text():
        record["dynamic_work_protocol"] = "volatile_LDS_and_increasing_CVT_checksum_v2"
    write(folder / "result.json", record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--cutlass-root", type=Path)
    p.add_argument("--family", choices=("all", "wgmma", "cross", "lds"), default="all")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--compile-only", action="store_true")
    args = p.parse_args()
    if args.list:
        print(json.dumps(matrix(), indent=2))
        return
    if not args.output or not args.cutlass_root:
        p.error("--output and --cutlass-root required")
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("single GPU Slurm allocation required")
    output = args.output.resolve()
    cutlass = args.cutlass_root.resolve()
    version = (cutlass / "include/cutlass/version.h").read_text()
    if tuple(
        int(re.search(r"#define\s+CUTLASS_" + k + r"\s+(\d+)", version)[1])
        for k in ("MAJOR", "MINOR", "PATCH")
    ) != (3, 9, 2):
        p.error("CUTLASS v3.9.2 required")
    uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    cuda = subprocess.check_output(["nvcc", "--version"], text=True)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    if "\n" in uuid or "GH200" not in gpu or "release 12.9" not in cuda:
        p.error("one GH200 GPU and CUDA 12.9 required")
    with open("/tmp/gh200-access-" + uuid + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        (output / "source/probes").mkdir(parents=True)
        (output / "build").mkdir()
        for name in ("run_r04.py", "analyze_r04.py"):
            shutil.copy2(ROOT / name, output / "source" / name)
        for name in ("r04.cu", "r00_common.hpp"):
            shutil.copy2(ROOT / "probes" / name, output / "source/probes" / name)
        # CUTLASS headers are not copied; their identity is kept as a per-file hash manifest.
        write(
            output / "source/cutlass_include_manifest.json",
            dict(
                root=str(cutlass),
                files={
                    str(f.relative_to(cutlass)): sha(f)
                    for f in sorted((cutlass / "include").rglob("*"))
                    if f.is_file()
                },
            ),
        )
        write(
            output / "source_hashes.json",
            {
                str(f.relative_to(output / "source")): sha(f)
                for f in sorted((output / "source").rglob("*"))
                if f.is_file()
            },
        )
        write(
            output / "environment.json",
            dict(
                gpu_uuid=uuid,
                gpu=gpu,
                cuda=cuda,
                hostname=os.uname().nodename,
                slurm_job_id=os.environ["SLURM_JOB_ID"],
                mode="smoke" if args.smoke else "formal",
                load_frequency="unknown",
            ),
        )
        cases = [
            c
            for c in matrix()
            if args.family == "all"
            or (args.family == "cross" and c["family"] == 1)
            or (
                args.family != "cross"
                and (c["family"] < 2) == (args.family == "wgmma")
                and not (args.family == "lds" and c["family"] == 2 and c["order"] == 3)
            )
        ]
        write(output / "cases.json", cases)
        command = [
            "nvcc",
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-lineinfo",
            "--keep",
            "--keep-dir=" + str(output / "build"),
            "--ptxas-options=-v",
            "-I" + str(cutlass / "include"),
            str(output / "source/probes/r04.cu"),
            "-o",
            str(output / "build/r04"),
        ]
        write(output / "build/command.json", command)
        with (output / "build/compile.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
        with (output / "build/r04.sass").open("w") as log:
            subprocess.run(
                ["cuobjdump", "--dump-sass", str(output / "build/r04")], stdout=log, check=True
            )
        write(output / "build/binary_hash.json", dict(r04=sha(output / "build/r04")))
        if any(c["family"] >= 2 for c in cases):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "r04_audit", output / "source/analyze_r04.py"
            )
            audit = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(audit)
            write(output / "build/dynamic-work-cfg.json", audit.audit_dynamic_work(output))
        if args.compile_only:
            return
        families = {c["family"] for c in cases}
        repeats = {f: 8 for f in families}
        if not args.smoke:
            for c in cases:
                if c["order"] == 1 and c["a"] == 32:
                    run(output, c, "validation-long", 8197)
            pilots = [run(output, c, "pilot-00", 8) for c in cases]
            repeats = {
                f: min(
                    65536,
                    max(
                        8,
                        8 * 2_000_000 // max(r["elapsed_ns"] for r in pilots if r["family"] == f)
                        + 1,
                    ),
                )
                for f in families
            }
        write(
            output / "protocol.json",
            dict(
                repeats_by_family=repeats,
                cell_total=160,
                seed=20261006,
                clock="SM clock64",
                end_event="WGMMA wait0 and independent consumers complete; CTA drain",
                target_ns=2_000_000,
                process_count=1 if args.smoke else 3,
                baseline_scaling=(
                    "160-action A/B-only baselines scaled by count; "
                    "linearity not independently established"
                ),
            ),
        )
        rng = random.Random(20261006)
        records = {c["id"]: [] for c in cases}
        # Randomize pair blocks; keep serial and interleaved neighbors in each round.
        blocks = []
        for family in families:
            group = [c for c in cases if c["family"] == family]
            baselines = [c for c in group if c["order"] in (2, 3)]
            blocks.append(baselines)
            mixed = [c for c in group if c["order"] in (0, 1)]
            blocks.extend(mixed[i : i + 2] for i in range(0, len(mixed), 2))

        def round_run(number, selected):
            order = list(blocks)
            rng.shuffle(order)
            for block in order:
                pair = list(block)
                rng.shuffle(pair)
                for c in pair:
                    if c["family"] not in selected:
                        continue
                    print(c["id"], number, flush=True)
                    records[c["id"]].append(
                        run(output, c, f"formal-{number:02d}", repeats[c["family"]])
                    )

        for i in range(1 if args.smoke else 3):
            round_run(i, families)
        if not args.smoke:
            extra = set()
            for c in cases:
                vals = [r["elapsed"] for r in records[c["id"]]]
                if statistics.stdev(vals) / statistics.mean(vals) > 0.01:
                    extra.add(c["family"])
            for block in blocks:
                if len(block) != 2 or block[0]["order"] != 0:
                    continue
                a, b = [[r["elapsed"] for r in records[c["id"]]] for c in block]
                noise = max(statistics.stdev(a), statistics.stdev(b), 1)
                if abs(statistics.mean(a) - statistics.mean(b)) < 3 * noise:
                    extra.add(block[0]["family"])
            for i in range(3, 10):
                round_run(i, extra)
            unstable = {
                c["family"]
                for c in cases
                if statistics.stdev([r["elapsed"] for r in records[c["id"]]])
                / statistics.mean(r["elapsed"] for r in records[c["id"]])
                > 0.05
            }
            for batch in range(2):
                if not unstable:
                    break
                for i in range(10 + batch * 3, 13 + batch * 3):
                    round_run(i, unstable)
                unstable = {
                    c["family"]
                    for c in cases
                    if c["family"] in unstable
                    and statistics.stdev([r["elapsed"] for r in records[c["id"]]])
                    / statistics.mean(r["elapsed"] for r in records[c["id"]])
                    > 0.05
                }
        subprocess.run(
            [sys.executable, str(output / "source/analyze_r04.py"), "--input", str(output)],
            check=True,
        )
        print(output / "report.md", flush=True)


if __name__ == "__main__":
    main()
