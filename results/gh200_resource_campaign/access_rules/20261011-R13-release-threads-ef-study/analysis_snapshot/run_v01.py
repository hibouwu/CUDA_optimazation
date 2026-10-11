#!/usr/bin/env python3
"""V01 held-out combinations: build, check and measure inside one GH200 Slurm allocation.

Modes (every mode compiles probes/v01.cu into a new output directory and keeps SASS):

  --smoke                    correctness at short lengths, 1 process per case
  --lengths 32 --kind target check runs at non-held-out lengths (7/19/47 are refused)
  --predictions FILE --prediction-sha256 HEX
                             formal held-out run of exactly the cases in the frozen prediction
                             file; one CTA 3 processes (10 if CV > 1%), all-GPU and CUTLASS 10

The complete CUTLASS points reuse the archived R00 binary (--r00-archive), checked by SHA256.
"""

import argparse
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

ROOT = Path(__file__).resolve().parent
NAMES = ("lds_ffma", "small_tc", "target_tc")
KINDS = {"lds": [0], "small": [1], "target": [2], "all": [0, 1, 2]}
HELD_OUT = (7, 19, 47)
SEED = 20261006


def matrix():
    """The 20 V01 points: 3 combinations x 3 lengths x 2 scopes + 2 complete CUTLASS kernels."""
    cases = [
        dict(id=f"{name}_k{k}_{scope}", kind=kind, length=k, scope=scope)
        for kind, name in enumerate(NAMES)
        for k in HELD_OUT
        for scope in ("one_cta", "all_gpu")
    ]
    cases += [
        dict(id=f"cutlass_2048_2048_{k}", kind="cutlass", m=2048, n=2048, k=k, scope="all_gpu")
        for k in (2048, 8192)
    ]
    return cases


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_environment(cutlass):
    version = (cutlass / "include/cutlass/version.h").read_text()
    found = tuple(
        int(re.search(r"#define\s+CUTLASS_" + key + r"\s+(\d+)", version)[1])
        for key in ("MAJOR", "MINOR", "PATCH")
    )
    if found != (3, 9, 2):
        raise SystemExit("CUTLASS v3.9.2 required")
    query = ["nvidia-smi", "--format=csv,noheader"]
    uuid = subprocess.check_output(query + ["--query-gpu=uuid"], text=True).strip()
    gpu = subprocess.check_output(
        query + ["--query-gpu=name,driver_version,memory.total"], text=True
    ).strip()
    cuda = subprocess.check_output(["nvcc", "--version"], text=True)
    if "\n" in uuid or "GH200" not in gpu or "release 12.9" not in cuda:
        raise SystemExit("one GH200 and CUDA 12.9 required")
    return dict(
        gpu_uuid=uuid,
        gpu=gpu,
        cuda=cuda,
        hostname=os.uname().nodename,
        slurm_job_id=os.environ["SLURM_JOB_ID"],
        load_frequency="unknown",
    )


def build(output, cutlass, r00_archive, need_cutlass):
    """Copy sources, compile v01.cu, dump SASS; optionally copy the archived R00 binary."""
    (output / "source/probes").mkdir(parents=True)
    (output / "build").mkdir()
    for name in ("run_v01.py", "analyze_v01.py", "v01_predict.py"):
        shutil.copy2(ROOT / name, output / "source" / name)
    for name in ("v01.cu", "v01_wg_inline.cuh", "r00_common.hpp"):
        shutil.copy2(ROOT / "probes" / name, output / "source/probes" / name)
    # CUTLASS headers are identified by a per-file hash manifest instead of a copy.
    headers = {
        str(f.relative_to(cutlass)): sha(f)
        for f in sorted((cutlass / "include").rglob("*"))
        if f.is_file()
    }
    write(output / "source/cutlass_include_manifest.json", dict(root=str(cutlass), files=headers))
    write(
        output / "source_hashes.json",
        {
            str(f.relative_to(output / "source")): sha(f)
            for f in sorted((output / "source").rglob("*"))
            if f.is_file()
        },
    )
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
        str(output / "source/probes/v01.cu"),
        "-lcuda",
        "-o",
        str(output / "build/v01"),
    ]
    write(output / "build/command.json", command)
    with (output / "build/compile.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
    with (output / "build/v01.sass").open("w") as out:
        subprocess.run(
            ["cuobjdump", "--dump-sass", str(output / "build/v01")], stdout=out, check=True
        )
    identities = dict(v01=sha(output / "build/v01"))
    if need_cutlass:
        if not r00_archive:
            raise SystemExit("--r00-archive required for the CUTLASS points")
        archive = r00_archive.resolve()
        frozen = json.loads((archive / "build/binary_hashes.json").read_text())["cutlass"]
        if sha(archive / "build/r00_cutlass") != frozen:
            raise SystemExit("R00 fixed CUTLASS binary identity mismatch")
        shutil.copy2(archive / "build/r00_cutlass", output / "build/r00_cutlass")
        shutil.copy2(archive / "build/cutlass.sass", output / "build/r00_cutlass.sass")
        identities["r00_cutlass"] = frozen
        write(
            output / "build/r00_identity.json",
            dict(
                archive=str(archive),
                binary_sha256=frozen,
                command=json.loads((archive / "build/commands.json").read_text())["cutlass"],
            ),
        )
    write(output / "build/binary_hashes.json", identities)


def run_case(output, case, role, number, prediction_sha):
    folder = output / "samples" / case["id"] / f"{role}-{number:02d}"
    folder.mkdir(parents=True)
    if case["kind"] == "cutlass":
        command = [str(output / "build/r00_cutlass"), "--backend", "cutlass", "--dtype", "fp16"]
        command += ["--cache", "repeat"]
        for field in ("m", "n", "k"):
            command += ["--" + field, str(case[field])]
    else:
        command = [str(output / "build/v01")]
        for field in ("kind", "length", "scope"):
            command += ["--" + field, str(case[field])]
    write(folder / "command.json", command)
    with (folder / "stdout.json").open("w") as out, (folder / "stderr.log").open("w") as err:
        process = subprocess.run(command, cwd=folder, stdout=out, stderr=err, timeout=300)
    write(folder / "process.json", dict(returncode=process.returncode))
    if process.returncode:
        raise RuntimeError(f"failed process: {folder}")
    record = json.loads((folder / "stdout.json").read_text())
    record.update(configuration=case, trial_role=role, prediction_sha256=prediction_sha)
    if record["status"] != "measured":
        raise RuntimeError(f"numerical failure: {folder}")
    if case["kind"] != "cutlass":
        raw = folder / "output.f32"
        record["output_sha256"] = sha(raw)
        with raw.open("rb") as src, gzip.open(
            folder / "output.f32.gz", "wb", compresslevel=1
        ) as dst:
            shutil.copyfileobj(src, dst)
        raw.unlink()
    write(folder / "result.json", record)
    return record


def sample(output, cases, counts, role, prediction_sha, extend_one_cta):
    """Shuffled rounds; one-CTA cases with CV > 1% after 3 processes are extended to 10."""
    rng = random.Random(SEED)
    records = {c["id"]: [] for c in cases}

    def round_run(number, selected):
        order = [c for c in cases if c["id"] in selected]
        rng.shuffle(order)
        for c in order:
            print(c["id"], number, flush=True)
            records[c["id"]].append(run_case(output, c, role, number, prediction_sha))

    for number in range(max(counts.values())):
        round_run(number, {c["id"] for c in cases if number < counts[c["id"]]})
    if extend_one_cta:
        noisy = set()
        for c in cases:
            if c["scope"] != "one_cta":
                continue
            values = [r["elapsed"] for r in records[c["id"]]]
            if len(values) == 3 and statistics.stdev(values) / statistics.mean(values) > 0.01:
                noisy.add(c["id"])
        for number in range(3, 10):
            if noisy:
                round_run(number, noisy)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--list", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--cutlass-root", type=Path)
    p.add_argument("--r00-archive", type=Path)
    p.add_argument("--predictions", type=Path)
    p.add_argument("--prediction-sha256")
    p.add_argument("--kind", choices=tuple(KINDS), default="all")
    p.add_argument("--scope", choices=("both", "one_cta", "all_gpu"), default="both")
    p.add_argument("--lengths", help="comma-separated check lengths (not 7/19/47)")
    p.add_argument("--processes", type=int, default=1, help="processes per check/smoke case")
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
    scopes = ("one_cta", "all_gpu") if args.scope == "both" else (args.scope,)
    kinds = KINDS[args.kind]
    prediction_sha = None
    if args.predictions:
        # Formal held-out run: the case list comes from the frozen prediction file.
        prediction_sha = sha(args.predictions)
        if prediction_sha != args.prediction_sha256:
            p.error("prediction file does not match --prediction-sha256")
        plan = json.loads(args.predictions.read_text())
        known = {c["id"]: c for c in matrix()}
        cases = [known[c["case_id"]] for c in plan["cases"]]
        role = "formal"
    elif args.smoke:
        cases = [
            dict(id=f"smoke_{NAMES[kind]}_k{k}_{scope}", kind=kind, length=k, scope=scope)
            for kind in kinds
            for k in ((2, 3, 5) if kind == 2 else (2, 3))
            for scope in scopes
        ]
        role = "smoke"
    elif args.lengths:
        lengths = [int(x) for x in args.lengths.split(",")]
        if any(k in HELD_OUT or not 1 <= k <= 111 for k in lengths):
            p.error("check lengths must be in 1..111 and not 7/19/47")
        cases = [
            dict(id=f"check_{NAMES[kind]}_k{k}_{scope}", kind=kind, length=k, scope=scope)
            for kind in kinds
            for k in lengths
            for scope in scopes
        ]
        role = "check"
    else:
        p.error("choose --smoke, --lengths or --predictions")
    output = args.output.resolve()
    cutlass = args.cutlass_root.resolve()
    environment = check_environment(cutlass)
    environment["mode"] = role
    with open("/tmp/gh200-access-" + environment["gpu_uuid"] + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        write(output / "environment.json", environment)
        need_cutlass = any(c["kind"] == "cutlass" for c in cases)
        build(output, cutlass, args.r00_archive, need_cutlass)
        write(output / "cases.json", cases)
        protocol = dict(
            role=role,
            held_out_lengths=list(HELD_OUT),
            all_gpu_grid="4 x SM count, 1 CTA/SM",
            cache="repeat; all combination CTAs read the same global inputs",
            end_event=(
                "target TC: both output halves stored (bulk wait_group 0); "
                "CUTLASS: CUDA event after one GemmUniversalAdapter::run"
            ),
        )
        if args.predictions:
            shutil.copy2(args.predictions, output / "predictions.json")
            probe_now = sha(ROOT / "probes/v01.cu")
            protocol.update(
                prediction_sha256=prediction_sha,
                prediction_created_utc=plan["created_utc"],
                prediction_probe_sha256=plan["probe"]["sha256"],
                probe_sha256=probe_now,
                probe_changed_after_prediction=probe_now != plan["probe"]["sha256"],
                processes="one CTA 3 (10 if CV > 1%); all-GPU and CUTLASS 10",
            )
        write(output / "protocol.json", protocol)
        if args.compile_only:
            return
        if role == "formal":
            counts = {c["id"]: 3 if c["scope"] == "one_cta" else 10 for c in cases}
        else:
            counts = {c["id"]: args.processes for c in cases}
        sample(output, cases, counts, role, prediction_sha, extend_one_cta=role == "formal")
    try:
        import numpy  # noqa: F401  (the CPU re-check needs numpy; GPU nodes may lack it)
    except ImportError:
        print("numpy unavailable: run source/analyze_v01.py --input", output, "on a CPU host")
        return
    analyzer = str(output / "source/analyze_v01.py")
    subprocess.run([sys.executable, analyzer, "--input", str(output)], check=True)
    print(output / "report.md", flush=True)


if __name__ == "__main__":
    main()
