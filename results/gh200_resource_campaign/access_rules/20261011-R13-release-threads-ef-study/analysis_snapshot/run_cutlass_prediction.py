#!/usr/bin/env python3
"""Run the two pre-frozen CUTLASS validation cases with full output capture."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from run_v01 import sample


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prediction-sha256", required=True)
    parser.add_argument("--capture-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("single-GPU Slurm allocation required")
    data = args.plan.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != args.prediction_sha256:
        raise ValueError("prediction differs from published SHA")
    plan = json.loads(data)
    binary = args.capture_archive / "build/r00_cutlass"
    if sha(binary) != plan["binary_sha256"]:
        raise ValueError("prediction binary identity")
    uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if uuid != plan["gpu_uuid"]:
        raise ValueError("prediction GPU identity")
    with open("/tmp/gh200-access-" + uuid + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        shutil.copytree(args.capture_archive / "source", output / "source/capture")
        (output / "build").mkdir()
        shutil.copy2(binary, output / "build/r00_cutlass")
        shutil.copy2(args.capture_archive / "build/cutlass.sass", output / "build/cutlass.sass")
        shutil.copy2(args.capture_archive / "build/compile.log", output / "build/compile.log")
        shutil.copy2(args.capture_archive / "gpu-code-equivalence.json", output / "gpu-code-equivalence.json")
        here = Path(__file__).resolve().parent
        for name in ("run_cutlass_prediction.py", "run_v01.py", "cutlass_collective_model.py", "analyze_v01.py"):
            shutil.copy2(here / name, output / "source" / name)
        write(output / "source_hashes.json", {str(f.relative_to(output / "source")): sha(f)
                                             for f in sorted((output / "source").rglob("*")) if f.is_file()})
        (output / "predictions.json").write_bytes(data)
        write(output / "environment.json", dict(gpu_uuid=uuid, hostname=os.uname().nodename,
              slurm_job_id=os.environ["SLURM_JOB_ID"], driver=subprocess.check_output(
                  ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()))
        write(output / "protocol.json", dict(prediction_sha256=digest, prediction_created_utc=plan["created_utc"],
              binary_sha256=plan["binary_sha256"], processes_per_case=10,
              role="new K validation; no prediction updates", boundary="CUDA event full completion"))
        cases = [dict(id=c["case_id"], kind="cutlass", scope="all_gpu", m=c["m"], n=c["n"], k=c["k"]) for c in plan["cases"]]
        write(output / "cases.json", cases)
        sample(output, cases, {c["id"]:10 for c in cases}, "formal", digest, extend_one_cta=False)
        for f in sorted((output / "samples").glob("*/*/result.json")):
            r = json.loads(f.read_text())
            raw = f.parent / "full_output.f32"
            if raw.stat().st_size != r["m"] * r["n"] * 4:
                raise ValueError("full output size")
            r.update(full_output_sha256=sha(raw), binary_sha256=plan["binary_sha256"])
            with raw.open("rb") as src, gzip.open(f.parent / "full_output.f32.gz", "wb", compresslevel=1) as dst:
                shutil.copyfileobj(src,dst)
            raw.unlink()
            write(f,r)


if __name__ == "__main__":
    main()
