#!/usr/bin/env python3
"""Bounded Thor probe for the NVFP4 TMEM requant epilogue."""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO / "results" / "sm110_epilogue_probe"
SOURCE = REPO / "GEMMsm110/tests/requant_epilogue_benchmark.cu"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(command: list[str], timeout_seconds: int | None = None) -> dict[str, object]:
    try:
        proc = subprocess.run(
            command, cwd=REPO, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False, timeout=timeout_seconds)
        return {"command": command, "returncode": proc.returncode,
                "timed_out": False, "output": proc.stdout}
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="backslashreplace")
        return {"command": command, "returncode": None, "timed_out": True,
                "output": output}


def environment() -> dict[str, object]:
    power = capture(
        [shutil.which("nvpmodel") or "/usr/sbin/nvpmodel", "-q"], 10)
    oc = {}
    for name in sorted(glob.glob("/sys/class/hwmon/hwmon*/oc*_event_cnt")):
        try:
            oc[name] = int(Path(name).read_text().strip())
        except (OSError, ValueError):
            oc[name] = None
    freq = {}
    for node in ("min_freq", "max_freq", "cur_freq", "governor"):
        path = Path("/sys/class/devfreq/gpu-gpc-0") / node
        freq[node] = path.read_text().strip() if path.is_file() else None
    return {
        "captured_at_utc": utc_now(), "hostname": platform.node(),
        "power_mode": power, "oc_event_counters": oc,
        "gpu_devfreq": freq,
        "gpu_identity": capture([
            "nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
            "--format=csv,noheader",
        ], 10),
    }


def bounded_run(command: list[str], timeout_seconds: int) -> dict[str, object]:
    started = utc_now()
    proc = subprocess.Popen(
        command, cwd=REPO, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    timed_out = False
    termination_failed = False
    interrupted = False
    try:
        output, _ = proc.communicate(timeout=timeout_seconds)
    except KeyboardInterrupt:
        interrupted = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            output, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired as term_timeout:
            os.killpg(proc.pid, signal.SIGKILL)
            try:
                output, _ = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired as kill_timeout:
                termination_failed = True
                output = kill_timeout.stdout or term_timeout.stdout or ""
                if isinstance(output, bytes):
                    output = output.decode(errors="backslashreplace")
    except subprocess.TimeoutExpired as initial_timeout:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            output, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired as term_timeout:
            os.killpg(proc.pid, signal.SIGKILL)
            try:
                output, _ = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired as kill_timeout:
                termination_failed = True
                partial = (kill_timeout.stdout or term_timeout.stdout
                           or initial_timeout.stdout or "")
                if isinstance(partial, bytes):
                    partial = partial.decode(errors="backslashreplace")
                output = partial
    return {
        "command": command, "started_at_utc": started,
        "finished_at_utc": utc_now(), "timeout_seconds": timeout_seconds,
        "timed_out": timed_out, "interrupted": interrupted,
        "termination_failed": termination_failed,
        "returncode": proc.returncode,
        "output": output,
    }


def parse_fields(output: str) -> dict[str, str]:
    fields = {}
    for match in re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)",
                             output):
        fields[match.group(1)] = match.group(2)
    return fields


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--max-blocks-per-sm", type=int, choices=(1, 2, 3, 4),
                        default=4)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        parser.error("invalid --run-id")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
        parser.error("invalid --expected-commit")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    lock_path = REPO / "results" / ".sm110_gpu_campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    global_lock = lock_path.open("w")
    try:
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another SM110 GPU campaign holds the global lock") from error

    actual_commit = capture(["git", "rev-parse", "HEAD"])["output"].strip()
    if actual_commit != args.expected_commit:
        raise RuntimeError(
            f"wrong commit: expected {args.expected_commit}, got {actual_commit}")
    dirty = capture(["git", "status", "--short", "--untracked-files=no"])
    if dirty["returncode"] or dirty["output"].strip():
        raise RuntimeError("tracked worktree must be clean")
    before = environment()
    if (before["power_mode"]["returncode"] != 0
            or "MAXN" not in str(before["power_mode"]["output"]).upper()):
        raise RuntimeError("MAXN power mode is not proven")

    run_dir = RESULT_ROOT / args.run_id
    summary_path = run_dir / "summary.json"
    if run_dir.exists():
        if summary_path.is_file():
            prior = json.loads(summary_path.read_text())
            if (prior.get("pass") is True
                    and prior.get("expected_commit") == args.expected_commit
                    and prior.get("source_sha256") == sha256(SOURCE)
                    and prior.get("timeout_seconds") == args.timeout_seconds
                    and prior.get("max_blocks_per_sm") == args.max_blocks_per_sm):
                print(json.dumps({"run_dir": str(run_dir), "pass": True,
                                  "cached": True}, indent=2))
                return 0
        raise RuntimeError(
            f"run-id already exists without an exact passing result: {run_dir}")
    run_dir.mkdir(parents=True)
    build = run_dir / "build"
    build.mkdir()
    binary = build / "epilogue"
    nvcc = shutil.which("nvcc")
    cuobjdump = shutil.which("cuobjdump")
    if not nvcc or not cuobjdump:
        raise RuntimeError("nvcc and cuobjdump are required")
    compile_command = [
        nvcc, "-O3", "-std=c++17",
        "-gencode", "arch=compute_110a,code=sm_110a",
        f"-I{REPO / 'GEMMsm110/include'}", str(SOURCE), "-o", str(binary),
    ]
    compile_result = capture(compile_command)
    (build / "compile_command.json").write_text(
        json.dumps(compile_command, indent=2) + "\n")
    (build / "compile.log").write_text(str(compile_result["output"]))
    if compile_result["returncode"]:
        raise RuntimeError("probe compile failed")
    sass_result = capture([cuobjdump, "--dump-sass", str(binary)])
    (build / "sass.txt").write_text(str(sass_result["output"]))
    sass_text = str(sass_result["output"])
    if (sass_result["returncode"]
            or "LDTM.x2" not in sass_text or "STTM.x2" not in sass_text):
        raise RuntimeError("cuobjdump failed or required TMEM SASS is absent")

    profiles = [
        # One CTA distinguishes an instruction/protocol failure from a
        # multi-CTA allocation-permit or placement interaction.
        ("single_cta_smoke", 256, 256, 1, 1, 1, 1, 1),
        ("full_gpu_smoke_bps1", 256, 256, 1, None, 20, 1, 1),
        ("production_shape_bps1", 4096, 1024, 1, None, 20, 1, 1),
        ("full_gpu_smoke_bps2", 256, 256, 2, None, 20, 1, 1),
        ("full_gpu_smoke_bps3", 256, 256, 3, None, 20, 1, 1),
        ("full_gpu_smoke_bps4", 256, 256, 4, None, 20, 1, 1),
    ]
    profiles = [row for row in profiles if row[3] <= args.max_blocks_per_sm]
    results = []
    for (profile_id, rows, cols, blocks_per_sm, workers,
         expected_unique_sms, warmup, iterations) in profiles:
        command = [
            str(binary), "--rows", str(rows), "--cols", str(cols),
            "--blocks-per-sm", str(blocks_per_sm),
            "--distribution", "normal", "--seed", "1234",
            "--warmup", str(warmup), "--iterations", str(iterations),
        ]
        if workers is not None:
            command += ["--workers", str(workers)]
        row = {"profile_id": profile_id, "environment_before": environment(),
               **bounded_run(command, args.timeout_seconds),
               "environment_after": environment()}
        fields = parse_fields(str(row["output"]))
        validation_errors = []
        if not row["timed_out"] and row["returncode"] == 0:
            expected = {
                "sm_count": "20",
                "unique_smid_count": str(expected_unique_sms),
                "blocks_per_sm": str(blocks_per_sm),
                "worker_count": str(workers or 20 * blocks_per_sm),
                "value_mismatches": "0", "scale_mismatches": "0",
            }
            for name, value in expected.items():
                if fields.get(name) != value:
                    validation_errors.append(
                        f"{name}: expected {value}, got {fields.get(name)}")
        row["fields"] = fields
        row["validation_errors"] = validation_errors
        results.append(row)
        (run_dir / f"{profile_id}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"profile_id": profile_id,
                          "returncode": row["returncode"],
                          "timed_out": row["timed_out"],
                          "interrupted": row["interrupted"],
                          "termination_failed": row["termination_failed"]}),
              flush=True)
        if (row["timed_out"] or row["interrupted"]
                or row["termination_failed"]
                or row["returncode"] or validation_errors):
            break

    after = environment()
    summary = {
        "schema_version": 2, "run_id": args.run_id,
        "expected_commit": args.expected_commit,
        "source_path": str(SOURCE.relative_to(REPO)),
        "source_sha256": sha256(SOURCE), "binary_sha256": sha256(binary),
        "sass_sha256": sha256(build / "sass.txt"),
        "timeout_seconds": args.timeout_seconds,
        "max_blocks_per_sm": args.max_blocks_per_sm,
        "environment_before": before, "environment_after": after,
        "profiles": results,
        "pass": len(results) == len(profiles) and all(
            not row["timed_out"] and not row["interrupted"]
            and not row["termination_failed"]
            and row["returncode"] == 0
            and not row["validation_errors"]
            for row in results),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_dir": str(run_dir), "pass": summary["pass"]},
                     indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
