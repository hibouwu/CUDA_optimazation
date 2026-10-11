#!/usr/bin/env python3
"""Paired K/MN WGMMA check using an already compiled r00_wgmma layout probe."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--mode", choices=("batch16_wait1", "single_wait0"), default="batch16_wait1")
    args = p.parse_args()
    root = args.archive.resolve()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("run inside a single-GPU Slurm allocation")
    uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if "\n" in uuid:
        raise ValueError("exactly one GPU required")
    with open("/tmp/gh200-access-" + uuid + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        samples = root / "layout-samples"
        samples.mkdir()
        write(root / "layout-environment.json", dict(
            gpu_uuid=uuid, hostname=os.uname().nodename,
            slurm_job_id=os.environ["SLURM_JOB_ID"],
            binary_sha256=digest(root / "build/wgmma_layout"),
        ))
        rng = random.Random(20261006)
        rows = []
        for trial in range(10):
            scopes = ["one_cta", "all_gpu"] if trial < 3 else ["all_gpu"]
            rng.shuffle(scopes)
            for scope in scopes:
                majors = ["K", "MN"] if args.mode == "batch16_wait1" else ["MN"]
                rng.shuffle(majors)
                for major in majors:
                    folder = samples / f"{scope}_{major}" / f"formal-{trial:02d}"
                    folder.mkdir(parents=True)
                    command = [str(root / "build/wgmma_layout"), "--dtype", "fp16",
                               "--source", "SS", "--n", "256", "--groups", "2",
                               "--iterations", "1024" if args.mode == "batch16_wait1" else "16384",
                               "--scope", scope, "--b-major", major, "--mode", args.mode]
                    with (folder / "stdout.json").open("w") as out, (folder / "stderr.log").open("w") as err:
                        process = subprocess.run(command, cwd=folder, stdout=out, stderr=err, timeout=120)
                    write(folder / "command.json", command)
                    write(folder / "process.json", dict(returncode=process.returncode))
                    if process.returncode:
                        raise RuntimeError(str(folder))
                    r = json.loads((folder / "stdout.json").read_text())
                    if r["status"] != "measured" or r["max_output_error"] != 0:
                        raise ValueError("GPU numeric check failed")
                    # Independent work and complete uniform-output reference.
                    flop = 2 * 64 * 256 * 16 * 16 * 2 * 1024 * r["blocks"]
                    if r["work_flop"] != flop:
                        raise ValueError("work count")
                    raw = folder / "output.f32"
                    data = raw.read_bytes()
                    if len(data) != r["blocks"] * 256 * 128 * 4:
                        raise ValueError("output length")
                    if any(x[0] != 1024.0 for x in struct.iter_unpack("<f", data)):
                        raise ValueError("CPU full output check")
                    stamps = r["stamps"]
                    elapsed = stamps[0][3] - stamps[0][2] if scope == "one_cta" else max(s[1] for s in stamps) - min(s[0] for s in stamps)
                    if elapsed != r["elapsed"]:
                        raise ValueError("time envelope")
                    r.update(binary_sha256=digest(root / "build/wgmma_layout"),
                             output_sha256=hashlib.sha256(data).hexdigest(), trial_role="formal")
                    with gzip.open(folder / "output.f32.gz", "wb", compresslevel=1) as out:
                        out.write(data)
                    raw.unlink()
                    write(folder / "result.json", r)
                    rows.append(r)
                    print(scope, major, trial, r["elapsed"], flush=True)
        write(root / "layout-summary.json", rows)


if __name__ == "__main__":
    main()
