#!/usr/bin/env python3
"""R00: compile, check, run the finite matrix, and write its report."""

from __future__ import annotations
import argparse
import csv
import fcntl
import gzip
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
import time

ROOT = Path(__file__).resolve().parent
CUTLASS_TAG = "v3.9.2"


def matrix():
    cases = []
    for dtype in ("fp16", "bf16", "fp8", "fp32"):
        for size, cache in ((2048, "evict_prepared"), (2048, "repeat"), (8192, "repeat")):
            cases.append(
                dict(
                    id=f"lt_{dtype}_{size}_{cache}",
                    kind="gemm",
                    backend="cublaslt",
                    dtype=dtype,
                    m=size,
                    n=size,
                    k=size,
                    cache=cache,
                )
            )
    cases.append(
        dict(
            id="lt_fp16_2000_repeat",
            kind="gemm",
            backend="cublaslt",
            dtype="fp16",
            m=2000,
            n=2000,
            k=2000,
            cache="repeat",
        )
    )
    for size in (8192, 4096):
        cases.append(
            dict(
                id=f"cutlass_fp16_{size}_repeat",
                kind="gemm",
                backend="cutlass",
                dtype="fp16",
                m=size,
                n=size,
                k=size,
                cache="repeat",
            )
        )
    coordinates = [("fp16", "SS", n, g, "one_cta") for n in (64, 128, 256) for g in (1, 2)]
    coordinates += [("fp16", "SS", n, 2, "all_gpu") for n in (128, 256)]
    coordinates += [("fp16", "RS", 256, g, "one_cta") for g in (1, 2)]
    coordinates += [("bf16", "SS", 256, g, "one_cta") for g in (1, 2)]
    coordinates += [
        ("fp8", "SS", 256, g, scope)
        for g, scope in ((1, "one_cta"), (2, "one_cta"), (2, "all_gpu"))
    ]
    for dtype, source, n, groups, scope in coordinates:
        cases.append(
            dict(
                id=f"wg_{dtype}_{source}_n{n}_g{groups}_{scope}",
                kind="wgmma",
                dtype=dtype,
                source=source,
                n=n,
                groups=groups,
                scope=scope,
            )
        )
    return cases


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def build(output, cutlass):
    source = output / "source"
    source.mkdir()
    for name in ("run_r00.py", "analyze.py"):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(ROOT / "probes", source / "probes")
    # Archive the exact headers used by both builds; no dependency on a later checkout.
    for part in ("include", "tools/util/include"):
        shutil.copytree(cutlass / part, source / "cutlass" / part)
    identities = {
        str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()
    }
    write_json(output / "source_hashes.json", identities)
    directory = output / "build"
    directory.mkdir()
    include = [
        "-I" + str(source / "cutlass/include"),
        "-I" + str(source / "cutlass/tools/util/include"),
    ]
    commands = {}
    cuda_root = Path(shutil.which("nvcc")).resolve().parent.parent
    cuda_libraries = [
        p for p in [cuda_root / "lib64"] + list((cuda_root / "targets").glob("*/lib")) if p.is_dir()
    ]
    for name in ("gemm", "cutlass", "wgmma"):
        command = [
            "nvcc",
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-lineinfo",
            "--ptxas-options=-v",
            str(source / "probes" / f"r00_{name}.cu"),
            "-o",
            str(directory / f"r00_{name}"),
        ]
        if name == "gemm":
            command += ["-lcublasLt", "-lcublas"]
            for path in cuda_libraries:
                command += ["-Xlinker", "-rpath", "-Xlinker", str(path)]
        else:
            command += include
        commands[name] = command
        print(f"编译 {name}", flush=True)
        with (directory / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
        with (directory / f"{name}.sass").open("w") as log:
            subprocess.run(
                ["cuobjdump", "--dump-sass", str(directory / f"r00_{name}")],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=True,
            )
    write_json(directory / "commands.json", commands)
    write_json(
        directory / "binary_hashes.json",
        {name: sha(directory / f"r00_{name}") for name in commands},
    )


def run_case(output, case, trial, iterations, smoke=False):
    directory = output / "samples" / case["id"] / trial
    previous = [
        p
        for p in [directory] + sorted(directory.parent.glob(trial + "-retry-*"))
        if (p / "result.json").is_file()
    ]
    if len(previous) > 1:
        raise RuntimeError("multiple successful attempts for one planned process")
    if previous:
        record = json.loads((previous[0] / "result.json").read_text())
        if record["configuration"] != case or record["status"] != "measured":
            raise RuntimeError("resume configuration/result mismatch")
        if case["kind"] == "wgmma" and record["iterations"] != iterations:
            raise RuntimeError("resume loop length changed")
        return record
    if directory.exists():
        # Preserve partial output. A new attempt is still the same planned ordinal.
        attempt = 1
        while (directory.parent / f"{trial}-retry-{attempt}").exists():
            attempt += 1
        directory = directory.parent / f"{trial}-retry-{attempt}"
    directory.mkdir(parents=True)
    executable = (
        "wgmma"
        if case["kind"] == "wgmma"
        else ("cutlass" if case["backend"] == "cutlass" else "gemm")
    )
    command = [str(output / "build" / f"r00_{executable}")]
    if case["kind"] == "gemm":
        for field in ("backend", "dtype", "m", "n", "k", "cache"):
            command += ["--" + field, str(case[field])]
        if smoke:
            command += ["--samples", str(case["m"] * case["n"])]
    else:
        for field in ("dtype", "source", "n", "groups", "scope"):
            command += ["--" + field, str(case[field])]
        command += ["--iterations", str(iterations)]
    write_json(directory / "command.json", command)
    started = time.time_ns()
    with (
        (directory / "stdout.json").open("w") as stdout,
        (directory / "stderr.log").open("w") as stderr,
    ):
        process = subprocess.run(command, cwd=directory, stdout=stdout, stderr=stderr, timeout=120)
    write_json(
        directory / "process.json",
        dict(returncode=process.returncode, started_unix_ns=started, ended_unix_ns=time.time_ns()),
    )
    if process.returncode != 0:
        raise RuntimeError(f"{case['id']} failed; see {directory/'stderr.log'}")
    record = json.loads((directory / "stdout.json").read_text())
    if record["status"] != "measured":
        raise RuntimeError(f"invalid result for {case['id']}")
    if case["kind"] == "wgmma":
        raw = directory / "output.f32"
        if raw.stat().st_size != 4 * record["output_elements"]:
            raise RuntimeError("truncated WGMMA output")
        record["output_sha256"] = sha(raw)
        with (
            raw.open("rb") as src,
            gzip.open(directory / "output.f32.gz", "wb", compresslevel=1) as dst,
        ):
            shutil.copyfileobj(src, dst)
        raw.unlink()  # Only the just-compressed task output; its hash is retained.
    record["case_id"] = case["id"]
    record["configuration"] = case
    record["trial_role"] = "smoke" if smoke else ("pilot" if trial == "pilot" else "formal")
    record["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    write_json(directory / "result.json", record)
    return record


def smoke_cases():
    cases = []
    for dtype in ("fp16", "bf16", "fp8", "fp32"):
        cases.append(
            dict(
                id=f"smoke_lt_{dtype}",
                kind="gemm",
                backend="cublaslt",
                dtype=dtype,
                m=32,
                n=32,
                k=32,
                cache="repeat",
            )
        )
    cases.append(
        dict(
            id="smoke_cutlass",
            kind="gemm",
            backend="cutlass",
            dtype="fp16",
            m=256,
            n=256,
            k=64,
            cache="repeat",
        )
    )
    for case in matrix():
        if (
            case["kind"] == "wgmma"
            and case["scope"] == "one_cta"
            and (case["groups"] == 1 or case["n"] == 256)
        ):
            cases.append(dict(case, id="smoke_" + case["id"]))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="print 30 default configurations; no GPU"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cutlass-root", type=Path, help="CUTLASS v3.9.2 source directory")
    parser.add_argument("--component", choices=("all", "gemm", "wgmma"), default="all")
    parser.add_argument(
        "--smoke", action="store_true", help="small full-output checks, not formal measurements"
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue unchanged source/configuration on the same GPU UUID",
    )
    args = parser.parse_args()
    if args.list:
        print(json.dumps(matrix(), ensure_ascii=False, indent=2))
        return
    if args.output is None or args.cutlass_root is None:
        parser.error("--output and --cutlass-root required")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run in a single-GPU Slurm allocation")
    output = args.output.resolve()
    cutlass = args.cutlass_root.resolve()
    if not (cutlass / "include/cutlass/cutlass.h").is_file():
        parser.error("CUTLASS headers missing")
    version = (cutlass / "include/cutlass/version.h").read_text()
    numbers = [
        re.search(r"#define\s+CUTLASS_" + key + r"\s+(\d+)", version)
        for key in ("MAJOR", "MINOR", "PATCH")
    ]
    if any(match is None for match in numbers) or tuple(
        int(match.group(1)) for match in numbers
    ) != (3, 9, 2):
        parser.error("the fixed kernel is pinned to CUTLASS v3.9.2")
    uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if "\n" in uuid:
        parser.error("this runner requires one visible physical GPU")
    with open("/tmp/gh200-access-" + uuid + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = dict(
            slurm_job_id=os.environ["SLURM_JOB_ID"],
            hostname=os.uname().nodename,
            gpu_uuid=uuid,
            cuda=subprocess.check_output(["nvcc", "--version"], text=True),
            cutlass_tag=CUTLASS_TAG,
            gpu=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
            mode="smoke" if args.smoke else "formal",
            component=args.component,
            load_frequency="unknown",
        )
        cases = smoke_cases() if args.smoke else matrix()
        cases = [c for c in cases if args.component == "all" or c["kind"] == args.component]
        if args.resume:
            previous = json.loads((output / "environment.json").read_text())
            for key in ("gpu_uuid", "cuda", "gpu", "mode", "component"):
                if previous[key] != environment[key]:
                    raise RuntimeError("resume environment changed: " + key)
            if json.loads((output / "cases.json").read_text()) != cases:
                raise RuntimeError("resume case matrix changed")
            identities = json.loads((output / "source_hashes.json").read_text())
            for relative, digest in identities.items():
                if sha(output / "source" / relative) != digest:
                    raise RuntimeError("archived source changed: " + relative)
                current = (
                    (cutlass / relative[len("cutlass/") :])
                    if relative.startswith("cutlass/")
                    else ROOT / relative
                )
                if sha(current) != digest:
                    raise RuntimeError("current source changed: " + relative)
            binaries = json.loads((output / "build/binary_hashes.json").read_text())
            for name, digest in binaries.items():
                if sha(output / "build" / f"r00_{name}") != digest:
                    raise RuntimeError("binary changed")
            write_json(output / f"environment-resume-{time.time_ns()}.json", environment)
        else:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "environment.json", environment)
            write_json(output / "cases.json", cases)
            build(output, cutlass)
        if args.compile_only:
            print("编译与SASS导出完成；没有采样。", flush=True)
            return
        # One common WGMMA loop length, selected using every participating point.
        iterations = 128
        if not args.smoke:
            for case in cases:
                if case["kind"] == "wgmma":
                    result = run_case(output, case, "pilot", 128)
                    elapsed_ns = max(s[1] for s in result["stamps"]) - min(
                        s[0] for s in result["stamps"]
                    )
                    iterations = max(iterations, 128 * 2_000_000 // max(1, elapsed_ns) + 1)
            iterations = min(65536, ((iterations + 7) // 8) * 8)
        write_json(
            output / "protocol.json",
            dict(
                wgmma_iterations=iterations,
                target_window_ns=2_000_000,
                process_counts="CTA:3 then up to10; GPU/GEMM:10",
                seed=20261006,
            ),
        )
        counts = {
            c["id"]: (
                1 if args.smoke else (3 if c["kind"] == "wgmma" and c["scope"] == "one_cta" else 10)
            )
            for c in cases
        }
        records = {c["id"]: [] for c in cases}
        rng = random.Random(20261006)
        for round_number in range(max(counts.values())):
            blocks = {}
            for case in cases:
                key = (
                    (case["backend"], case["dtype"], case["m"], case["n"], case["k"])
                    if case["kind"] == "gemm"
                    else case["id"]
                )
                blocks.setdefault(key, []).append(case)
            paired_order = list(blocks.values())
            rng.shuffle(paired_order)
            order = []
            for block in paired_order:
                rng.shuffle(block)
                order.extend(block)
            for case in order:
                if round_number >= counts[case["id"]]:
                    continue
                print(f"{case['id']} 进程 {round_number+1}/{counts[case['id']]}", flush=True)
                records[case["id"]].append(
                    run_case(output, case, f"trial-{round_number:02d}", iterations, args.smoke)
                )
        if not args.smoke:
            for case in cases:
                rows = records[case["id"]]
                if len(rows) != 3:
                    continue
                values = [r["elapsed"] for r in rows]
                cv = statistics.stdev(values) / statistics.mean(values)
                if cv > 0.01:
                    for i in range(3, 10):
                        records[case["id"]].append(
                            run_case(output, case, f"trial-{i:02d}", iterations)
                        )
        subprocess.run(
            [sys.executable, str(output / "source/analyze.py"), "--input", str(output)], check=True
        )
        print(f"结果：{output/'report.md'}", flush=True)


if __name__ == "__main__":
    main()
