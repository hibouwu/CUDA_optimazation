#!/usr/bin/env python3
"""R05 forty cases plus sixteen matched empty controls."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def matrix(controls=True):
    cases = []
    for proto in ("cp", "tin", "tout", "wg"):
        for filler in (0, 32, 128, 512):
            cases.append(
                dict(
                    id=f"A_{proto}_f{filler}",
                    family="A",
                    protocol=proto,
                    filler=filler,
                    stages=1,
                    consumer=0,
                    requests=1,
                    read=0,
                    pitch=128,
                    source="shared",
                    control=0,
                )
            )
    for proto in ("cp", "tin"):
        for stages in (1, 2, 4):
            for consumer in (16, 128):
                cases.append(
                    dict(
                        id=f"B_{proto}_s{stages}_c{consumer}",
                        family="B",
                        protocol=proto,
                        filler=0,
                        stages=stages,
                        consumer=consumer,
                        requests=1,
                        read=0,
                        pitch=128,
                        source="shared",
                        control=0,
                    )
                )
    for requests in (1, 4):
        for read in (0, 1):
            cases.append(
                dict(
                    id=f"C_tout_q{requests}_read{read}",
                    family="C",
                    protocol="tout",
                    filler=0,
                    stages=1,
                    consumer=0,
                    requests=requests,
                    read=read,
                    pitch=128,
                    source="shared",
                    control=0,
                )
            )
    for stages in (2, 4):
        for source in ("shared", "independent"):
            cases.append(
                dict(
                    id=f"D_s{stages}_{source}",
                    family="D",
                    protocol="tin",
                    filler=0,
                    stages=stages,
                    consumer=0,
                    requests=1,
                    read=0,
                    pitch=128,
                    source=source,
                    control=0,
                )
            )
    for pitch in (128, 144, 160, 256):
        cases.append(
            dict(
                id=f"E_pitch{pitch}",
                family="E",
                protocol="tout",
                filler=0,
                stages=1,
                consumer=0,
                requests=1,
                read=0,
                pitch=pitch,
                source="shared",
                control=0,
            )
        )
    if controls:
        cases.extend(
            [dict(c, id=c["id"] + "_control", control=1) for c in cases if c["family"] == "A"]
        )
    return cases


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run(output, case, role, iterations, capture=False):
    folder = output / "samples" / case["id"] / role
    folder.mkdir(parents=True)
    command = [str(output / "build/r05")]
    for key, value in case.items():
        if key != "id":
            command += ["--" + key, str(value)]
    command += ["--iterations", str(iterations), "--capture", str(int(capture))]
    write(folder / "command.json", command)
    with (folder / "stdout.json").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
        process = subprocess.run(command, cwd=folder, stdout=stdout, stderr=stderr, timeout=120)
    write(folder / "process.json", dict(returncode=process.returncode))
    if process.returncode:
        raise RuntimeError("failed " + str(folder))
    record = json.loads((folder / "stdout.json").read_text())
    record.update(
        case_id=case["id"],
        configuration=case,
        role=role,
        artifacts={f.name: sha(f) for f in folder.iterdir() if f.suffix in (".u32", ".f32")},
    )
    from analyze_r05 import check

    record["filler_instruction"] = "dependent_IMAD_LCG"
    record["wg_filler_seed_mode"] = "last_iteration_clock64"
    record["end_event"] = (
        "bulk_full_completion_and_CTA_join"
        if case["family"] in ("C", "E") or case["protocol"] == "tout"
        else (
            "acquire_and_consumer_release_join"
            if case["family"] == "D"
            else (
                "covered_WGMMA_wait_and_accumulator_sink"
                if case["protocol"] == "wg"
                else "input_acquire_consumer_sink_and_CTA_join"
            )
        )
    )
    record["cpu_checked_elements"] = check(record, folder)
    write(folder / "result.json", record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--cutlass-root", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--supplement", action="store_true")
    p.add_argument("--case-ids", help="comma-separated existing cases")
    p.add_argument("--add-processes", type=int, default=7)
    p.add_argument("--family", default="all", choices=("all", "A", "B", "C", "D", "E"))
    a = p.parse_args()
    cases = matrix()
    if a.list:
        print(json.dumps(cases, indent=2))
        return
    if a.family != "all":
        cases = [c for c in cases if c["family"] == a.family]
    if (
        not a.output
        or (not a.cutlass_root and not a.supplement)
        or not os.environ.get("SLURM_JOB_ID")
    ):
        p.error("output, CUTLASS v3.9.2 headers, single-GPU Slurm allocation required")
    if a.supplement:
        output = a.output.resolve()
        env = json.loads((output / "environment.json").read_text())
        uuid = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip()
        if (
            uuid != env["gpu_uuid"]
            or subprocess.check_output(["nvcc", "--version"], text=True) != env["cuda"]
        ):
            raise RuntimeError("supplement environment differs")
        if sha(output / "build/r05") != json.loads((output / "build/binary_hash.json").read_text()):
            raise RuntimeError("binary changed")
        for relative, digest in json.loads((output / "source_hashes.json").read_text()).items():
            if sha(output / "source" / relative) != digest:
                raise RuntimeError("archived source changed")
        planned = json.loads((output / "cases.json").read_text())
        ids = set((a.case_ids or "").split(","))
        cases = [c for c in planned if c["id"] in ids]
        if len(cases) != len(ids) or a.add_processes < 1:
            p.error("exact existing case ids and positive additional processes required")
        if (
            any(c["family"] == "A" for c in cases)
            and not (output / "source/probes/r05_wg_inline.cuh").exists()
        ):
            raise RuntimeError("old A is invalid and must not be supplemented")
        iterations = json.loads((output / "protocol.json").read_text())["iterations"]
        rng = random.Random(20261006)
        with open("/tmp/gh200-access-" + uuid + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            existing = {
                c["id"]: len(list((output / "samples" / c["id"]).glob("trial-*/result.json")))
                for c in cases
            }
            write(
                output / "supplement-command.json",
                dict(
                    cases=[c["id"] for c in cases],
                    additional_processes=a.add_processes,
                    iterations=iterations,
                    gpu_uuid=uuid,
                    script_sha256=sha(__file__),
                ),
            )
            for offset in range(a.add_processes):
                order = [c for c in cases if not c["control"]]
                rng.shuffle(order)
                for c in order:
                    pair = [c] + [d for d in cases if d["id"] == c["id"] + "_control"]
                    for d in pair:
                        run(output, d, f"trial-{existing[d['id']]+offset:02}", iterations)
        return
    version = (a.cutlass_root / "include/cutlass/version.h").read_text()
    parts = [
        re.search(r"#define\s+CUTLASS_" + key + r"\s+(\d+)", version)
        for key in ("MAJOR", "MINOR", "PATCH")
    ]
    if any(v is None for v in parts) or tuple(int(v.group(1)) for v in parts) != (3, 9, 2):
        p.error("CUTLASS v3.9.2 headers required")
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
        for name in (
            "run_r05.py",
            "analyze_r05.py",
            "probes/r05.cu",
            "probes/r05_wg_inline.cuh",
            "probes/r00_common.hpp",
        ):
            shutil.copy2(ROOT / name, source / name)
        shutil.copytree(a.cutlass_root / "include", source / "cutlass/include")
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
                cutlass="v3.9.2_headers_from_R00_archive",
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
            "-I" + str(source / "cutlass/include"),
            str(source / "probes/r05.cu"),
            "-lcuda",
            "-o",
            str(build / "r05"),
        ]
        write(build / "command.json", command)
        with (build / "compile.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        with (build / "r05.sass").open("w") as log:
            subprocess.run(["cuobjdump", "--dump-sass", str(build / "r05")], stdout=log, check=True)
        write(build / "binary_hash.json", sha(build / "r05"))
        if a.compile_only:
            return
        # Full traces use one/two sequences; no performance conclusion from capture mode.
        for c in cases:
            for count in (1, 2):
                print("check", c["id"], count, flush=True)
                run(output, c, f"check-{count}", count, True)
        iterations = 2 if a.smoke else 128
        if not a.smoke:
            pilots = [run(output, c, "pilot", 128) for c in cases]
            lengths = []
            for r in pilots:
                stamps = r["stamps"]
                ns = max(s[3] for s in stamps) - min(s[2] for s in stamps)
                lengths.append(128 * 2_000_000 // max(1, ns) + 1)
            iterations = min(2048, max(512, max(lengths)))
        write(
            output / "protocol.json",
            dict(
                iterations=iterations,
                target_ns=2_000_000,
                iteration_cap=2048,
                seed=20261006,
                cta_initial_processes=3,
                gpu_processes=10,
                full_short_trace_lengths=[1, 2],
            ),
        )
        rows = {c["id"]: [] for c in cases}
        rng = random.Random(20261006)
        counts = {c["id"]: 1 if a.smoke else 10 if c["family"] in ("D", "E") else 3 for c in cases}
        # Keep A asynchronous/control pairs adjacent within randomized blocks.
        blocks = []
        for c in cases:
            if c["control"]:
                continue
            block = [c]
            matched = [d for d in cases if d["id"] == c["id"] + "_control"]
            blocks.append(block + matched)
        for trial in range(max(counts.values())):
            order = blocks.copy()
            rng.shuffle(order)
            for block in order:
                for c in block:
                    if trial < counts[c["id"]]:
                        print("sample", c["id"], trial, flush=True)
                        rows[c["id"]].append(run(output, c, f"trial-{trial:02}", iterations))
        if not a.smoke:
            expand = set()
            for c in cases:
                rr = rows[c["id"]]
                v = [r["elapsed"] for r in rr]
                if len(v) == 3 and statistics.stdev(v) / statistics.mean(v) > 0.01:
                    expand.add(c["id"])
            for block in blocks:
                if len(block) == 2:
                    v = [r["elapsed"] for r in rows[block[0]["id"]]]
                    w = [r["elapsed"] for r in rows[block[1]["id"]]]
                    if (
                        abs(statistics.mean(v) - statistics.mean(w))
                        <= 3 * (statistics.stdev(v) + statistics.stdev(w))
                        + statistics.mean(v) * 0.01
                    ):
                        expand.update(c["id"] for c in block)
            for trial in range(3, 10):
                order = blocks.copy()
                rng.shuffle(order)
                for block in order:
                    for c in block:
                        if c["id"] in expand and counts[c["id"]] == 3:
                            rows[c["id"]].append(run(output, c, f"trial-{trial:02}", iterations))
        subprocess.run(
            [sys.executable, str(source / "analyze_r05.py"), "--input", str(output)], check=True
        )


if __name__ == "__main__":
    main()
