#!/usr/bin/env python3
"""R07: NDEBUG anchor, clock under GEMM load, fixed per-kernel cost.

Run inside one single-GPU Slurm allocation (CUDA 12.9 loaded):
  r07_run.py build  --cutlass-root DIR --output RUN
  r07_run.py sample --output RUN --set {pilot,anchor,kfit,timing,clock}
Each process is one independent sample; rows are appended to RUN/samples.jsonl.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
KS = (64, 128, 256, 512, 1024, 2048, 4096)
SIZES = (8192, 4096, 2048)
NVSMI_FIELDS = [
    "timestamp", "clocks.sm", "clocks.max.sm", "clocks.mem", "power.draw",
    "power.draw.instant", "enforced.power.limit", "temperature.gpu", "utilization.gpu",
    "clocks_event_reasons.active", "clocks_event_reasons.gpu_idle",
    "clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown", "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def build(output: Path, cutlass: Path):
    source, directory = output / "source", output / "build"
    (source / "probes").mkdir(parents=True)
    directory.mkdir()
    for name in ("r00_common.hpp", "r00_cutlass.cu", "r00_gemm.cu", "r07_probe.cu"):
        shutil.copy2(ROOT / "probes" / name, source / "probes" / name)
    for name in ("r07_run.py", "r07_analyze.py"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, source / name)
    include = [f"-I{cutlass}/include", f"-I{cutlass}/tools/util/include"]
    base = ["nvcc", "-std=c++17", "-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
            "--ptxas-options=-v"]
    cuda_root = Path(shutil.which("nvcc")).resolve().parent.parent
    rpath = []
    for lib in [cuda_root / "lib64"] + list((cuda_root / "targets").glob("*/lib")):
        if lib.is_dir():
            rpath += ["-Xlinker", "-rpath", "-Xlinker", str(lib)]
    lt = ["-lcublasLt", "-lcublas"] + rpath
    targets = {
        # Same source and flags as R00 (no NDEBUG): reproduces the C7510 build.
        "r00_cutlass_debug": base + include + [str(source / "probes/r00_cutlass.cu")],
        "r00_cutlass_ndebug": base
        + ["-DNDEBUG"]
        + include
        + [str(source / "probes/r00_cutlass.cu")],
        "r00_gemm": base + [str(source / "probes/r00_gemm.cu")] + lt,
        "r07_probe": base + ["-DNDEBUG"] + include + [str(source / "probes/r07_probe.cu")] + lt,
    }
    commands = {}
    for name, command in targets.items():
        command = command + ["-o", str(directory / name)]
        commands[name] = command
        print("compile", name, flush=True)
        with (directory / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        with (directory / f"{name}.sass").open("w") as log:
            subprocess.run(["cuobjdump", "--dump-sass", str(directory / name)], stdout=log,
                           check=True, timeout=120)
    write_json(directory / "commands.json", commands)
    write_json(directory / "binary_hashes.json", {n: sha(directory / n) for n in targets})
    hashes = {str(p.relative_to(output)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()}
    write_json(output / "source_hashes.json", hashes)
    write_json(output / "environment.json", dict(
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), hostname=os.uname().nodename,
        gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,"
                                     "clocks.max.sm,enforced.power.limit",
                                     "--format=csv,noheader"], text=True).strip(),
        nvcc=subprocess.check_output(["nvcc", "--version"], text=True),
        cutlass_root=str(cutlass),
        cutlass_version_h_sha256=sha(cutlass / "include/cutlass/version.h"),
        cutlass_gemm_adapter_sha256=sha(
            cutlass / "include/cutlass/gemm/device/gemm_universal_adapter.h"),
    ))


def r00_case(binary, backend, m, n, k):
    return dict(binary=binary, args=["--backend", backend, "--dtype", "fp16", "--m", str(m),
                                     "--n", str(n), "--k", str(k), "--cache", "repeat"])


def probe_case(backend, m, n, k, mode="timing", extra=()):
    return dict(binary="r07_probe", args=["--backend", backend, "--mode", mode, "--m", str(m),
                                          "--n", str(n), "--k", str(k), *extra])


def case_sets():
    anchor = {}
    for s in SIZES:
        anchor[f"cutlass_debug_{s}"] = r00_case("r00_cutlass_debug", "cutlass", s, s, s)
        anchor[f"cutlass_ndebug_{s}"] = r00_case("r00_cutlass_ndebug", "cutlass", s, s, s)
        anchor[f"lt_fp16_{s}"] = r00_case("r00_gemm", "cublaslt", s, s, s)
    kfit = {}
    for mn in (2048, 256):
        for k in KS:
            kfit[f"kfit_m{mn}_k{k}"] = r00_case("r00_cutlass_ndebug", "cutlass", mn, mn, k)
    timing = {}
    for mn in (2048, 256):
        for k in KS:
            timing[f"timing_cutlass_m{mn}_k{k}"] = probe_case("cutlass", mn, mn, k)
        timing[f"timing_empty_cutlass_shape_m{mn}"] = probe_case("empty_cutlass_shape", mn, mn, 64)
    timing["timing_empty1"] = probe_case("empty1", 1, 1, 1)
    clock = {
        "clock_lt_8192": probe_case("cublaslt", 8192, 8192, 8192, "loop", ["--seconds", "8"]),
        "clock_cutlass_2048": probe_case("cutlass", 2048, 2048, 2048, "loop", ["--seconds", "8"]),
        "clock_cutlass_8192": probe_case("cutlass", 8192, 8192, 8192, "loop", ["--seconds", "8"]),
    }
    pilot = {
        "pilot_debug_2048": anchor["cutlass_debug_2048"],
        "pilot_ndebug_2048": anchor["cutlass_ndebug_2048"],
        "pilot_kfit_m2048_k64": kfit["kfit_m2048_k64"],
        "pilot_kfit_m2048_k4096": kfit["kfit_m2048_k4096"],
        "pilot_timing_m2048_k2048": timing["timing_cutlass_m2048_k2048"],
        "pilot_timing_empty1": timing["timing_empty1"],
        "pilot_clock_cutlass_2048": probe_case(
            "cutlass", 2048, 2048, 2048, "loop", ["--seconds", "3", "--idle-s", "2", "--tail-s", "1"]
        ),
    }
    seq = {}
    for mn, k in [(2048, k) for k in KS] + [(4096, 4096), (8192, 8192)]:
        seq[f"seq_cutlass_m{mn}_k{k}"] = probe_case("cutlass", mn, mn, k, "r00seq",
                                                    ["--idle-s", "1"])
    seq["seq_lt_8192"] = probe_case("cublaslt", 8192, 8192, 8192, "r00seq", ["--idle-s", "1"])
    clock2 = {f"cold_{k}": dict(v, args=v["args"] + ["--seconds", "3"]) for k, v in clock.items()}
    return dict(pilot=(pilot, 2), anchor=(anchor, 10), kfit=(kfit, 10), timing=(timing, 5),
                clock=(clock, 2), seq=(seq, 3), cold=(clock2, 2))


def nvsmi_fields():
    """Keep only query fields this driver accepts."""
    ok = []
    for field in NVSMI_FIELDS:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
                           capture_output=True, text=True)
        if r.returncode == 0 and "Field" not in r.stdout:
            ok.append(field)
    return ok


def add_unix_time(csv_path: Path):
    """nvidia-smi timestamps are node-local time; append unix ns computed on the node."""
    lines = csv_path.read_text().splitlines()
    if not lines:
        return
    out = [lines[0] + ", unix_ns"]
    for line in lines[1:]:
        stamp = line.split(",")[0].strip()
        try:
            t = datetime.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp()
            out.append(f"{line}, {int(t * 1e9)}")
        except ValueError:
            out.append(f"{line}, ")
    csv_path.write_text("\n".join(out) + "\n")


def run_one(output: Path, case_id, case, trial, fields):
    folder = output / "samples" / case_id / f"trial-{trial:02d}"
    if (folder / "result.json").exists():
        return
    if folder.exists():
        folder = folder.with_name(folder.name + f"-retry-{time.time_ns()}")
    folder.mkdir(parents=True)
    command = [str(output / "build" / case["binary"])] + case["args"]
    write_json(folder / "command.json", command)
    monitor = None
    if "--mode" in case["args"] and case["args"][case["args"].index("--mode") + 1] == "loop":
        monitor = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,nounits"]
            + ["-lms", "50", "-f", str(folder / "nvsmi.csv")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)
    started = time.time_ns()
    with (folder / "stdout.txt").open("w") as out, (folder / "stderr.log").open("w") as err:
        process = subprocess.run(command, cwd=folder, stdout=out, stderr=err, timeout=300)
    ended = time.time_ns()
    if monitor:
        time.sleep(0.3)
        monitor.send_signal(signal.SIGINT)
        monitor.wait(timeout=10)
        add_unix_time(folder / "nvsmi.csv")
    lines = [json.loads(x) for x in (folder / "stdout.txt").read_text().splitlines() if x.strip()]
    record = dict(case_id=case_id, trial=trial, binary=case["binary"], args=case["args"],
                  returncode=process.returncode, started_unix_ns=started, ended_unix_ns=ended,
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    if case["binary"].startswith("r00_"):
        r = lines[0]
        record.update(elapsed_ms=r["elapsed_ms"], work_flop=r["work_flop"], status=r["status"],
                      max_storage_reference_error=r["max_storage_reference_error"],
                      warmup_converged=r["warmup_converged"], implementation=r["implementation"])
    else:
        timing = [x for x in lines if x.get("event") == "timing"]
        check = [x for x in lines if x.get("event") == "check"]
        if timing:
            record["timing"] = {k: v for k, v in timing[0].items() if k != "isolated_us"}
        record["check"] = check[0] if check else None
        record["status"] = "measured" if process.returncode == 0 else "failed"
        record["loop_events"] = sum(1 for x in lines if x.get("event") == "batch")
        seq = [x for x in lines if x.get("event") == "r00seq"]
        if seq:
            record["r00seq"] = {k: v for k, v in seq[0].items() if k != "warmup_us"}
    write_json(folder / "result.json", record)
    with (output / "samples.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(case_id, trial, record.get("elapsed_ms", record.get("timing", {}).get("graph_us")),
          record["status"], flush=True)
    if process.returncode != 0:
        raise RuntimeError(f"{case_id} trial {trial} failed: {folder}")


def sample(output: Path, name):
    cases, rounds = case_sets()[name]
    fields = nvsmi_fields() if name in ("clock", "pilot", "cold") else []
    write_json(output / f"protocol-{name}.json", dict(cases=cases, rounds=rounds, seed=20261007,
                                                    nvsmi_fields=fields,
                                                    nvsmi_interval_ms=50))
    rng = random.Random(20261007)
    for trial in range(rounds):
        order = list(cases)
        rng.shuffle(order)
        for case_id in order:
            run_one(output, case_id, cases[case_id], trial, fields)


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
