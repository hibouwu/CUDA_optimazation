#!/usr/bin/env python3
"""Pair the same fixed CUTLASS source built with/without NDEBUG at already observed sizes.

This is a causal diagnostic, not a held-out prediction run. Run in one Slurm GPU allocation.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--second-label", default="NDEBUG", choices=("NDEBUG", "phases"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("single-GPU Slurm allocation required")
    uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if "\n" in uuid:
        raise ValueError("one GPU required")
    with open("/tmp/gh200-access-" + uuid + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / "build").mkdir()
        binaries = {}
        for variant, path in (("original", args.original), (args.second_label, args.release)):
            binaries[variant] = output / "build" / variant
            shutil.copy2(path, binaries[variant])
        write(output / "environment.json", dict(gpu_uuid=uuid, hostname=os.uname().nodename,
              slurm_job_id=os.environ["SLURM_JOB_ID"], purpose="known-size build diagnostic"))
        write(output / "binary_hashes.json", {name: sha(path) for name, path in binaries.items()})
        rng = random.Random(20261006)
        for trial in range(10):
            sizes = [2048, 8192]
            rng.shuffle(sizes)
            for k in sizes:
                variants = ["original", args.second_label]
                rng.shuffle(variants)
                for variant in variants:
                    folder = output / "samples" / f"{variant}_k{k}" / f"formal-{trial:02d}"
                    folder.mkdir(parents=True)
                    command = [str(binaries[variant]), "--backend", "cutlass", "--dtype", "fp16",
                               "--m", "2048", "--n", "2048", "--k", str(k), "--cache", "repeat"]
                    write(folder / "command.json", command)
                    with (folder / "stdout.json").open("w") as out, (folder / "stderr.log").open("w") as err:
                        process = subprocess.run(command, cwd=folder, stdout=out, stderr=err, timeout=120)
                    write(folder / "process.json", dict(returncode=process.returncode))
                    if process.returncode:
                        raise RuntimeError(str(folder))
                    r = json.loads((folder / "stdout.json").read_text())
                    if r["status"] != "measured" or r["max_storage_reference_error"] != 0:
                        raise ValueError("numerical check failed")
                    if r["work_flop"] != 2 * 2048 * 2048 * k:
                        raise ValueError("work count")
                    r.update(build_variant=variant, trial_role="formal",
                             binary_sha256=sha(binaries[variant]), gpu_uuid=uuid)
                    if (folder / "diag.json").exists():
                        r["diag_sha256"] = sha(folder / "diag.json")
                        raw = folder / "diag_output.f32"
                        if raw.stat().st_size != 2048 * 2048 * 4:
                            raise ValueError("full diagnostic output length")
                        r["full_output_sha256"] = sha(raw)
                        with raw.open("rb") as src, gzip.open(folder / "diag_output.f32.gz", "wb", compresslevel=1) as dst:
                            shutil.copyfileobj(src, dst)
                        raw.unlink()
                    write(folder / "result.json", r)
                    print(variant, k, trial, r["elapsed_ms"], flush=True)


if __name__ == "__main__":
    main()
