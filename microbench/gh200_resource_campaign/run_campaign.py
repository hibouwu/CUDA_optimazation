#!/usr/bin/env python3
"""Bounded, resumable SM90 memory-path collection inside a Slurm allocation."""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)

def case_matrix():
    cases = [{"id": f"smem_read_stride{s}", "mode": "smem_read", "bytes": 32768,
              "iterations": 8192, "stride": s} for s in (1, 2, 4, 8, 16, 32)]
    cases += [{"id": "smem_write_stride1", "mode": "smem_write", "bytes": 32768,
               "iterations": 8192, "stride": 1}]
    for mode, mib, iters in (("global_read_ca", 8, 64), ("global_read_cg", 8, 64),
                             ("global_read_cg", 256, 16), ("global_write", 256, 16),
                             ("global_duplex", 128, 16)):
        cases.append({"id": f"{mode}_{mib}m", "mode": mode, "bytes": mib * 1024**2,
                      "iterations": iters, "stride": 1})
    return cases

def command(args, out, err, timeout=120):
    with out.open("w") as stdout, err.open("w") as stderr:
        p = subprocess.Popen(args, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=5)
            raise RuntimeError("timeout: " + str(args))
    if code:
        raise RuntimeError(f"exit {code}: {args}; see {err}")
    return p.pid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()
    if args.trials < 10:
        ap.error("at least 10 independent process trials required")
    plan = {"schema_version": 1, "target": "GH200/SM90a", "trials": args.trials,
            "cases": case_matrix(), "seed": 20261001,
            "trial_process_timeout_seconds": 120,
            "timed_scope": "CTA globaltimer envelope; stores include device fence",
            "cache_residency_proven": False, "physical_hbm_bytes_proven": False,
            "warmup": "two same-allocation launches per external process; not a thermal steady-state claim"}
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.output is None or not os.environ.get("SLURM_JOB_ID"):
        ap.error("--output and an allocated Slurm job are required")
    out = args.output.resolve()
    lock_path = Path("/tmp") / ("gh200_resource_job_" + os.environ["SLURM_JOB_ID"] + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_names = ("resource_probe.cu", "run_campaign.py", "audit_campaign.py", "run_on_romeo.sh")
        hashes = {name: sha(HERE / name) for name in source_names}
        plan["source_sha256"] = hashes
        if out.exists():
            if not args.resume:
                ap.error("output exists; choose a new ID or explicitly --resume")
            old = json.loads((out / "run_spec.json").read_text())
            for key, value in plan.items():
                if old.get(key) != value:
                    raise RuntimeError("resume contract changed: " + key)
            if sha(out / "resource_probe") != old.get("binary_sha256"):
                raise RuntimeError("retained binary changed")
            for name, expected in hashes.items():
                if sha(out / "source" / name) != expected:
                    raise RuntimeError("retained source changed")
            if sha(out / "sass.txt") != old.get("sass_sha256"):
                raise RuntimeError("retained SASS changed")
            plan = old
        else:
            out.mkdir(parents=True)
            (out / "source").mkdir()
            for name in source_names:
                shutil.copy2(HERE / name, out / "source" / name)
            plan["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            plan["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
            write_json(out / "run_spec.json", plan)
            compile_cmd = ["nvcc", "-std=c++17", "-O3", "-lineinfo",
                           "-gencode", "arch=compute_90a,code=sm_90a", "-Xptxas=-v",
                           str(out / "source/resource_probe.cu"), "-o", str(out / "resource_probe")]
            write_json(out / "compile_command.json", compile_cmd)
            write_json(out / "campaign_status.json", {"status": "compiling"})
            try:
                command(compile_cmd, out / "compile.stdout", out / "compile.log", 180)
            except Exception as exc:
                write_json(out / "campaign_status.json", {"status": "failed", "phase": "compile", "error": str(exc)})
                raise
            command(["cuobjdump", "--dump-sass", str(out / "resource_probe")],
                    out / "sass.txt", out / "sass.stderr", 30)
            plan["binary_sha256"] = sha(out / "resource_probe")
            plan["sass_sha256"] = sha(out / "sass.txt")
            write_json(out / "run_spec.json", plan)
        write_json(out / "campaign_status.json", {"status": "running", "slurm_job_id": os.environ["SLURM_JOB_ID"]})
        try:
            with (out / "environment.txt").open("a") as log:
                for cmd in (["hostname"], ["nvcc", "--version"], ["nvidia-smi", "-q"]):
                    subprocess.run(cmd, stdout=log, stderr=log, timeout=30, check=False)
                log.write(json.dumps({k: os.environ.get(k) for k in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_PARTITION")}) + "\n")
            command([str(out / "resource_probe"), "device"], out / "device.jsonl", out / "device.stderr", 30)
            telemetry_file = (out / "telemetry.csv").open("a")
            telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu",
                                          "--format=csv", "-lms", "100"], stdout=telemetry_file, stderr=subprocess.DEVNULL)
            try:
                rng = random.Random(plan["seed"])
                for trial in range(args.trials):
                    cases = list(plan["cases"]); rng.shuffle(cases)
                    for case in cases:
                        d = out / "cases" / case["id"]; d.mkdir(parents=True, exist_ok=True)
                        raw = d / f"trial_{trial:02d}.jsonl"
                        receipt = d / f"trial_{trial:02d}.receipt.json"
                        cmd = [str(out / "resource_probe"), case["mode"], str(case["bytes"]),
                               str(case["iterations"]), str(case["stride"]), str(trial),
                               str(3 + trial * 19)]
                        if receipt.exists():
                            rec = json.loads(receipt.read_text())
                            if rec["command"] != cmd or rec["raw_sha256"] != sha(raw):
                                raise RuntimeError("resume trial changed: " + str(raw))
                            continue
                        if raw.exists():
                            raise RuntimeError("uncommitted trial exists; retain failed run and choose new ID")
                        start = time.time_ns()
                        pid = command(cmd, raw, d / f"trial_{trial:02d}.stderr", plan["trial_process_timeout_seconds"])
                        data = [json.loads(line) for line in raw.read_text().splitlines()]
                        if len(data) != 2 or data[1]["errors"] != 0:
                            raise RuntimeError("invalid/correctness-failed trial")
                        write_json(receipt, {"command": cmd, "pid": pid, "host_start_ns": start,
                                             "host_stop_ns": time.time_ns(), "raw_sha256": sha(raw)})
                        with (out / "progress.jsonl").open("a") as journal:
                            journal.write(json.dumps({"case": case["id"], "trial": trial, "status": "collected"}) + "\n")
                        print(f"{case['id']} trial={trial} collected", flush=True)
            finally:
                telemetry.terminate()
                try: telemetry.wait(timeout=5)
                except subprocess.TimeoutExpired: telemetry.kill(); telemetry.wait(timeout=5)
                telemetry_file.close()
            ncu = shutil.which("ncu")
            if ncu:
                with (out / "ncu.stdout").open("w") as f, (out / "ncu.stderr").open("w") as e:
                    cmd = [ncu, "--clock-control", "none", "--cache-control", "none",
                           "--kernel-name", "regex:global_path", "--launch-skip", "2", "--launch-count", "1",
                           "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum",
                           "--csv", "--log-file", str(out / "ncu.csv"),
                           str(out / "resource_probe"), "global_read_cg", str(256*1024**2), "16", "1", "0", "3"]
                    p = subprocess.Popen(cmd, stdout=f, stderr=e, start_new_session=True)
                    try:
                        code = p.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGTERM)
                        try: p.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=5)
                        code = 124
                write_json(out / "ncu_status.json", {"available": True, "returncode": code, "command": cmd})
            else:
                write_json(out / "ncu_status.json", {"available": False})
            subprocess.run([sys.executable, str(out / "source/audit_campaign.py"), str(out)], check=True)
            write_json(out / "campaign_status.json", {"status": "complete", "qualification": "timing_only",
                                                     "summary_sha256": sha(out / "summary.json")})
            (out / "COMPLETE").write_text("summary_sha256=" + sha(out / "summary.json") + "\n")
            print("GH200_RESOURCE_TIMING_COMPLETE", flush=True)
        except Exception as exc:
            write_json(out / "campaign_status.json", {"status": "failed", "error": str(exc)})
            raise
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
