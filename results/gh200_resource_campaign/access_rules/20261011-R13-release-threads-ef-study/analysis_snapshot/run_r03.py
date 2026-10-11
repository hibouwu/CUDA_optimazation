#!/usr/bin/env python3
"""R03 finite matrix: build, check, calibrate, sample, CPU recompute."""

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
import sys

ROOT = Path(__file__).resolve().parent


def matrix():
    cases = []
    for op, name in enumerate(("load", "store", "ldmatrix", "stmatrix")):
        for width in ((4, 16) if op < 2 else (16,)):
            for warps in (1, 4, 8):
                cases.append(dict(id=f"{name}_b{width}_w{warps}", op=op, width=width, warps=warps))
    return cases


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run(output, case, role, iterations):
    folder = output / "samples" / case["id"] / role
    folder.mkdir(parents=True)
    command = [str(output / "build/r03")]
    for key in ("op", "width", "warps"):
        command += ["--" + key, str(case[key])]
    command += ["--iterations", str(iterations), "--seed", "17"]
    write(folder / "command.json", command)
    with (folder / "stdout.json").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
        process = subprocess.run(command, cwd=folder, stdout=stdout, stderr=stderr, timeout=120)
    write(folder / "process.json", dict(returncode=process.returncode))
    if process.returncode:
        raise RuntimeError("failed: " + str(folder))
    record = json.loads((folder / "stdout.json").read_text())
    record.update(
        case_id=case["id"], configuration=case, role=role, output_sha256=sha(folder / "output.u32")
    )
    if record["status"] != "measured":
        raise RuntimeError("invalid result")
    write(folder / "result.json", record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--representatives", action="store_true")
    p.add_argument("--compile-only", action="store_true")
    a = p.parse_args()
    cases = matrix()
    if a.list:
        print(json.dumps(cases, indent=2))
        return
    if a.representatives:
        cases = [c for c in cases if c["warps"] == 4 and c["width"] == 16]
    if not a.output or not os.environ.get("SLURM_JOB_ID"):
        p.error("output and a single-GPU Slurm allocation required")
    uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if "\n" in uuid:
        p.error("one visible physical GPU required")
    with open("/tmp/gh200-access-" + uuid + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = a.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        source = output / "source"
        (source / "probes").mkdir(parents=True)
        build = output / "build"
        build.mkdir()
        for name in ("run_r03.py", "analyze_r03.py", "probes/r03.cu", "probes/r00_common.hpp"):
            shutil.copy2(ROOT / name, source / name)
        write(
            output / "source_hashes.json",
            {str(f.relative_to(source)): sha(f) for f in source.rglob("*") if f.is_file()},
        )
        write(
            output / "environment.json",
            dict(
                gpu_uuid=uuid,
                hostname=os.uname().nodename,
                slurm_job_id=os.environ["SLURM_JOB_ID"],
                cuda=subprocess.check_output(["nvcc", "--version"], text=True),
                gpu=subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ),
                mode="smoke" if a.smoke else "formal",
            ),
        )
        write(output / "cases.json", cases)
        command = [
            "nvcc",
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-lineinfo",
            "--ptxas-options=-v",
            str(source / "probes/r03.cu"),
            "-o",
            str(build / "r03"),
        ]
        write(build / "command.json", command)
        with (build / "compile.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        with (build / "r03.sass").open("w") as log:
            subprocess.run(["cuobjdump", "--dump-sass", str(build / "r03")], stdout=log, check=True)
        write(build / "binary_hash.json", sha(build / "r03"))
        if a.compile_only:
            return
        iterations = 9 if a.smoke else 128
        if not a.smoke:
            pilots = [run(output, c, "pilot", 128) for c in cases]
            iterations = min(
                65536, max(128, max((128 * 2_000_000 // r["elapsed_ns"] + 1) for r in pilots))
            )
        write(
            output / "protocol.json",
            dict(
                iterations=iterations,
                seed=20261006,
                initial_processes=1 if a.smoke else 3,
                maximum_processes=1 if a.smoke else 10,
                target_ns=2_000_000,
            ),
        )
        rows = {c["id"]: [] for c in cases}
        rng = random.Random(20261006)
        for trial in range(1 if a.smoke else 3):
            order = cases.copy()
            rng.shuffle(order)
            for c in order:
                print(c["id"], trial, flush=True)
                rows[c["id"]].append(run(output, c, f"trial-{trial:02}", iterations))
        if not a.smoke:
            # Compare neighboring warp counts; close differences and CV trigger ten processes.
            expand = set()
            for c in cases:
                v = [r["work_bytes"] / r["elapsed"] for r in rows[c["id"]]]
                if statistics.stdev(v) / statistics.mean(v) > 0.01:
                    expand.add(c["id"])
                for d in cases:
                    if (c["op"], c["width"]) != (d["op"], d["width"]) or c["warps"] == d["warps"]:
                        continue
                    w = [r["work_bytes"] / r["elapsed"] for r in rows[d["id"]]]
                    if (
                        abs(statistics.mean(v) - statistics.mean(w))
                        <= 3 * (statistics.stdev(v) + statistics.stdev(w))
                        + max(statistics.mean(v), statistics.mean(w)) * 0.01
                    ):
                        expand.update((c["id"], d["id"]))
            for trial in range(3, 10):
                order = [c for c in cases if c["id"] in expand]
                rng.shuffle(order)
                for c in order:
                    rows[c["id"]].append(run(output, c, f"trial-{trial:02}", iterations))
        subprocess.run(
            [sys.executable, str(source / "analyze_r03.py"), "--input", str(output)], check=True
        )


if __name__ == "__main__":
    main()
