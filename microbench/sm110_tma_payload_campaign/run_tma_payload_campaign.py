#!/usr/bin/env python3
"""Resumable Thor collection for schedule-matched TMA payload capacities."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.sm110_gemm_model.campaign_plots import (  # noqa: E402
    generate_campaign_plots,
)

CAMPAIGN = Path(__file__).resolve().parent
RESULT_ROOT = REPO / "results" / "sm110_tma_payload_campaign"
SOURCE = REPO / "microbench/07_tma_gmem_smem_bandwidth/tma_gmem_smem_bandwidth.cu"
EXPECTED_SMS = 20
PAYLOAD_BYTES = (4096, 8192, 16384, 32768, 65536)
TRIALS = 10
SLOTS = 2
THREADS = 128
BLOCKS_PER_SM = 1
WARMUP_ITERS = 32
L2_BYTES = 16 << 20
DRAM_BYTES = 256 << 20
DEFAULT_TARGET_ISSUED_BYTES = 512 << 20
DEFAULT_NCU_TARGET_ISSUED_BYTES = 64 << 20
DEFAULT_TRIAL_TIMEOUT_SECONDS = 120
DEFAULT_NCU_TIMEOUT_SECONDS = 300
SASS_TOKENS = ("UTMALDG.3D",)
NCU_CANDIDATES = (
    "gpu__time_duration.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "lts__t_bytes.sum",
    "lts__t_sectors_op_read_lookup_hit.sum",
    "lts__t_sectors_op_read_lookup_miss.sum",
    "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tma.sum",
)
NCU_REQUIRED = (
    "gpu__time_duration.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "lts__t_bytes.sum",
    "lts__t_sectors_op_read_lookup_hit.sum",
    "lts__t_sectors_op_read_lookup_miss.sum",
)
NCU_BASE_UNITS = {
    "gpu__time_duration.sum": "ns",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum": "byte",
    "lts__t_bytes.sum": "byte",
    "lts__t_sectors_op_read_lookup_hit.sum": "sector",
    "lts__t_sectors_op_read_lookup_miss.sum": "sector",
}
NCU_NUMBER_RE = re.compile(
    r"^[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and proc.returncode:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(command)}\n{proc.stdout}"
        )
    return proc


def run_bounded(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        command, cwd=REPO, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, start_new_session=True)
    try:
        stdout, _ = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, _ = proc.communicate(timeout=5)
        raise RuntimeError(
            f"command timed out after {timeout_seconds}s: {' '.join(command)}")
    return subprocess.CompletedProcess(command, proc.returncode, stdout, "")


def executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"required executable not found: {name}")
    return path


def parse_kv(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        for match in re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", line):
            fields[match.group(1)] = match.group(2)
    return fields


def iterations_for_target(
    target_issued_bytes: int, payload_bytes: int, blocks: int
) -> int:
    bytes_per_iteration = blocks * payload_bytes
    return max(1, math.ceil(target_issued_bytes / bytes_per_iteration))


def cases(target_issued_bytes: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode, residency, backing, blocks in (
        ("l2-hit", "hot_l2", L2_BYTES, 1),
        ("dram-stream", "cold_hbm", DRAM_BYTES, EXPECTED_SMS),
    ):
        for payload in PAYLOAD_BYTES:
            iterations = iterations_for_target(target_issued_bytes, payload, blocks)
            warmup_iterations = (
                max(WARMUP_ITERS, math.ceil(backing / payload))
                if residency == "hot_l2" else WARMUP_ITERS)
            rows.append(
                {
                    "id": (
                        f"tma_{mode.replace('-', '_')}_{payload // 1024}k_"
                        f"slots{SLOTS}_{'single_sm' if blocks == 1 else 'full_gpu'}"
                    ),
                    "resource": (
                        f"tma.smem_ingress.per_sm.payload_{payload // 1024}k"
                        if residency == "hot_l2"
                        else f"tma.hbm.payload_{payload // 1024}k"
                    ),
                    "residency": residency,
                    "tile_bytes": payload,
                    "destination_slots": SLOTS,
                    "threads_per_cta": THREADS,
                    "resident_ctas_per_sm": BLOCKS_PER_SM,
                    "iterations": iterations,
                    "warmup_iterations": warmup_iterations,
                    "expected_blocks": blocks,
                    "expected_unique_smid_count": blocks,
                    "target_issued_bytes": target_issued_bytes,
                    "args": [
                        "--mode", mode,
                        "--bytes", str(backing),
                        "--tile-bytes", str(payload),
                        "--slots", str(SLOTS),
                        "--iters", str(iterations),
                        "--warmup-iters", str(warmup_iterations),
                        "--blocks-per-sm", str(BLOCKS_PER_SM),
                        "--threads", str(THREADS),
                        *(["--blocks", "1"] if blocks == 1 else []),
                    ],
                    "sass_tokens": list(SASS_TOKENS),
                }
            )
    return rows


def environment() -> dict[str, object]:
    commands = {
        "gpu_identity": [
            "nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
            "--format=csv,noheader",
        ],
        "gpu_state": [
            "nvidia-smi", "--query-gpu=pstate,clocks.current.graphics,power.limit",
            "--format=csv,noheader",
        ],
        "nvcc": ["nvcc", "--version"],
        "ncu": ["ncu", "--version"],
        "git_head": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--short", "--untracked-files=no"],
    }
    result: dict[str, object] = {
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    for name, command in commands.items():
        proc = run(command, check=False)
        result[name] = {"returncode": proc.returncode, "output": proc.stdout}
    nvpmodel = shutil.which("nvpmodel")
    proc = run([nvpmodel, "-q"], check=False) if nvpmodel else None
    result["power_mode"] = {
        "returncode": proc.returncode if proc else 127,
        "output": proc.stdout if proc else "nvpmodel not found",
    }
    return result


def validate_environment(snapshot: dict[str, object], expected_commit: str) -> None:
    for key in ("gpu_identity", "gpu_state", "nvcc", "ncu", "git_head",
                "git_status", "power_mode"):
        value = snapshot.get(key, {})
        if not isinstance(value, dict) or value.get("returncode") != 0:
            raise RuntimeError(f"environment probe failed: {key}")
    identity = str(snapshot["gpu_identity"]["output"]).strip().splitlines()
    if len(identity) != 1 or "11.0" not in identity[0] or "Thor" not in identity[0]:
        raise RuntimeError("exactly one Thor/SM110 GPU is required")
    if "MAXN" not in str(snapshot["power_mode"]["output"]).upper():
        raise RuntimeError("MAXN power mode is not proven")
    if str(snapshot["git_head"]["output"]).strip() != expected_commit:
        raise RuntimeError("environment snapshot commit mismatch")
    if str(snapshot["git_status"]["output"]).strip():
        raise RuntimeError("environment snapshot tracked worktree is dirty")


def write_status(run_dir: Path, status: str, **extra: object) -> None:
    payload = {
        "status": status,
        "pid": os.getpid(),
        "hostname": platform.node(),
        "updated_at_utc": utc_now(),
        **extra,
    }
    (run_dir / "campaign_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with (run_dir / "progress.jsonl").open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def compile_binary(
    run_dir: Path, host_compiler: str | None, undef_gnu_source: bool
) -> dict[str, object]:
    build = run_dir / "build"
    build.mkdir(exist_ok=True)
    binary = build / "tma_payload"
    command = [executable("nvcc"), "-O3", "-std=c++17"]
    if host_compiler:
        command += ["-ccbin", host_compiler]
    if undef_gnu_source:
        command += [
            "-Xcompiler=-U_GNU_SOURCE", "-D_DEFAULT_SOURCE",
            "-D_POSIX_C_SOURCE=200809L", "-D_XOPEN_SOURCE=700",
            "-D_XOPEN_SOURCE_EXTENDED=1", "-D_LARGEFILE64_SOURCE=1",
            "-D_ATFILE_SOURCE=1",
        ]
    command += [
        "-gencode", "arch=compute_110a,code=sm_110a", str(SOURCE), "-lcuda",
        "-o", str(binary),
    ]
    (build / "compile_command.json").write_text(json.dumps(command, indent=2) + "\n")
    compiled = run(command, check=False)
    (build / "compile.log").write_text(compiled.stdout)
    if compiled.returncode:
        raise RuntimeError("TMA payload benchmark compile failed")
    sass_path = build / "sass.txt"
    sass = run([executable("cuobjdump"), "--dump-sass", str(binary)], check=False)
    sass_path.write_text(sass.stdout)
    if sass.returncode or any(token not in sass.stdout for token in SASS_TOKENS):
        raise RuntimeError("TMA payload benchmark SASS contract failed")
    binary_hash = sha256(binary)
    (build / "binary.sha256").write_text(f"{binary_hash}  tma_payload\n")
    return {
        "binary": binary,
        "source_sha256": sha256(SOURCE),
        "binary_sha256": binary_hash,
        "sass_sha256": sha256(sass_path),
    }


def validate_trial(case: dict[str, object], fields: dict[str, str], iterations: int) -> float:
    if fields.get("mode") not in {"l2-hit", "dram-stream"}:
        raise RuntimeError(f"{case['id']}: missing or invalid mode")
    if int(fields.get("sm_count", "0")) != EXPECTED_SMS:
        raise RuntimeError(f"{case['id']}: expected {EXPECTED_SMS} SMs")
    expected_unique_sms = int(case["expected_unique_smid_count"])
    if int(fields.get("unique_smid_count", "0")) != expected_unique_sms:
        raise RuntimeError(f"{case['id']}: SM-scope mismatch")
    if int(fields.get("blocks", "0")) != int(case["expected_blocks"]):
        raise RuntimeError(f"{case['id']}: block-scope mismatch")
    requested = int(fields["requested_bytes"])
    expected = int(case["expected_blocks"]) * iterations * int(case["tile_bytes"])
    if requested != expected:
        raise RuntimeError(f"{case['id']}: issued-byte mismatch {requested} != {expected}")
    elapsed_ns = int(fields["globaltimer_elapsed_ns"])
    reported = float(fields["globaltimer_gbytes_per_second"]) * 1e9
    if elapsed_ns <= 0:
        raise RuntimeError(f"{case['id']}: nonpositive elapsed time")
    recalculated = requested * 1e9 / elapsed_ns
    if not math.isclose(reported, recalculated, rel_tol=2e-6, abs_tol=1.0):
        raise RuntimeError(f"{case['id']}: rate arithmetic mismatch")
    return recalculated


def query_ncu_metrics() -> tuple[list[str], list[str]]:
    proc = run(["ncu", "--query-metrics", "--query-metrics-mode", "all"])
    names = {
        line.split()[0]
        for line in proc.stdout.splitlines()
        if line and not line[0].isspace()
    }
    supported = [name for name in NCU_CANDIDATES if name in names]
    missing = [name for name in NCU_REQUIRED if name not in names]
    if missing:
        raise RuntimeError(f"required NCU metrics unavailable: {missing}")
    return supported, missing


def ncu_number(value: object) -> float:
    text = str(value).strip()
    if not NCU_NUMBER_RE.fullmatch(text):
        raise ValueError(f"invalid NCU number: {value!r}")
    result = float(text.replace(",", ""))
    if not math.isfinite(result):
        raise ValueError(f"non-finite NCU number: {value!r}")
    return result


def find_ncu_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    for index, header in enumerate(rows):
        if ("ID" not in header or "Kernel Name" not in header
                or not set(NCU_BASE_UNITS).issubset(header)):
            continue
        if index + 1 >= len(rows) or len(rows[index + 1]) != len(header):
            raise RuntimeError(f"NCU unit row is missing from {path}")
        unit_row = rows[index + 1]
        for name, expected_unit in NCU_BASE_UNITS.items():
            actual_unit = unit_row[header.index(name)]
            if actual_unit != expected_unit:
                raise RuntimeError(
                    f"NCU metric {name} has unit {actual_unit!r}, "
                    f"expected {expected_unit!r} in {path}")
        kernel_index = header.index("Kernel Name")
        time_index = header.index("gpu__time_duration.sum")
        candidates: list[tuple[float, list[str]]] = []
        for row in rows[index + 1 :]:
            if len(row) != len(header) or not row or not row[0].isdigit():
                continue
            if "tma_kernel" not in row[kernel_index]:
                continue
            try:
                duration = ncu_number(row[time_index])
            except ValueError:
                continue
            if duration >= 0:
                candidates.append((duration, row))
        if candidates:
            return dict(zip(header, max(candidates, key=lambda item: item[0])[1]))
    raise RuntimeError(f"no TMA kernel metric row in {path}")


def metric(row: dict[str, str], name: str) -> float:
    try:
        value = ncu_number(row[name])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"invalid NCU metric {name}") from exc
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"invalid NCU metric value {name}={value}")
    return value


def collect_ncu(
    case_dir: Path,
    case: dict[str, object],
    binary: Path,
    metrics: list[str],
    ncu_target_issued_bytes: int,
    timeout_seconds: int,
) -> dict[str, object]:
    ncu_dir = case_dir / "ncu"
    ncu_dir.mkdir(exist_ok=True)
    ncu_iterations = iterations_for_target(
        ncu_target_issued_bytes, int(case["tile_bytes"]),
        int(case["expected_blocks"])
    )
    ncu_args = list(case["args"])
    ncu_args[ncu_args.index("--iters") + 1] = str(ncu_iterations)
    report_base = ncu_dir / "profile"
    raw_path = ncu_dir / "raw.csv"
    stderr_path = ncu_dir / "profile.stderr.log"
    command = [
        "ncu", "--target-processes", "all", "--replay-mode", "kernel",
        "--page", "raw", "--csv", "--force-overwrite", "-o", str(report_base),
        "--metrics", ",".join(metrics), str(binary), *ncu_args,
    ]
    with raw_path.open("w") as stdout_handle:
        process = subprocess.Popen(
            command, cwd=REPO, text=True, stdout=stdout_handle,
            stderr=subprocess.PIPE, start_new_session=True)
        try:
            _, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                _, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                _, stderr = process.communicate(timeout=5)
            stderr_path.write_text(stderr or "")
            raise RuntimeError(
                f"{case['id']}: NCU timed out after {timeout_seconds}s")
    stderr_path.write_text(stderr or "")
    report_path = ncu_dir / "profile.ncu-rep"
    if (
        process.returncode
        or "ERR_NVGPUCTRPERM" in (stderr or "")
        or not report_path.is_file()
    ):
        raise RuntimeError(f"{case['id']}: NCU failed; see {stderr_path}")
    row = find_ncu_row(raw_path)
    timed_requested = (
        int(case["expected_blocks"]) * ncu_iterations * int(case["tile_bytes"])
    )
    # NCU replay reports one row per launch. find_ncu_row selects the longer
    # timed launch, so its counters must be compared with timed bytes only;
    # adding the separate warmup launch would demand traffic absent from that row.
    counter_expected = timed_requested
    tma_bytes = metric(
        row, "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum"
    )
    lts_bytes = metric(row, "lts__t_bytes.sum")
    hit_sectors = metric(row, "lts__t_sectors_op_read_lookup_hit.sum")
    miss_sectors = metric(row, "lts__t_sectors_op_read_lookup_miss.sum")
    if tma_bytes < counter_expected * 0.98 or lts_bytes < counter_expected * 0.90:
        raise RuntimeError(f"{case['id']}: NCU does not confirm requested TMA traffic")
    miss_proxy = miss_sectors * 32.0
    if case["residency"] == "cold_hbm" and miss_proxy < counter_expected * 0.70:
        raise RuntimeError(f"{case['id']}: NCU does not confirm DRAM-stream residency")
    if case["residency"] == "hot_l2" and not (hit_sectors > miss_sectors):
        raise RuntimeError(f"{case['id']}: NCU does not confirm L2-hit residency")
    summary = {
        "returncode": process.returncode,
        "permission_denied": False,
        "iterations": ncu_iterations,
        "metrics": metrics,
        "timed_requested_bytes": timed_requested,
        "expected_counter_bytes": counter_expected,
        "tma_bytes": tma_bytes,
        "tma_to_expected": tma_bytes / counter_expected,
        "lts_bytes": lts_bytes,
        "lts_to_expected": lts_bytes / counter_expected,
        "l2_hit_sectors": hit_sectors,
        "l2_miss_sectors": miss_sectors,
        "l2_miss_proxy_to_expected": miss_proxy / counter_expected,
        "report_path": str(report_path.relative_to(case_dir)),
        "report_sha256": sha256(report_path),
        "raw_path": str(raw_path.relative_to(case_dir)),
        "raw_sha256": sha256(raw_path),
        "stderr_path": str(stderr_path.relative_to(case_dir)),
        "stderr_sha256": sha256(stderr_path),
    }
    (ncu_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def result_is_complete(
    path: Path, fingerprint: str, collect_ncu_flag: bool
) -> bool:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if (
        result.get("status") != "ok"
        or result.get("trial_count") != TRIALS
        or result.get("fingerprint") != fingerprint
    ):
        return False
    if not collect_ncu_flag:
        return True
    ncu = result.get("ncu", {})
    return (
        ncu.get("returncode") == 0
        and (path.parent / str(ncu.get("report_path", ""))).is_file()
        and (path.parent / str(ncu.get("raw_path", ""))).is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--target-issued-bytes", type=int, default=DEFAULT_TARGET_ISSUED_BYTES
    )
    parser.add_argument(
        "--ncu-target-issued-bytes",
        type=int,
        default=DEFAULT_NCU_TARGET_ISSUED_BYTES,
    )
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument("--trial-timeout-seconds", type=int,
                        default=DEFAULT_TRIAL_TIMEOUT_SECONDS)
    parser.add_argument("--ncu-timeout-seconds", type=int,
                        default=DEFAULT_NCU_TIMEOUT_SECONDS)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--host-compiler")
    parser.add_argument("--nvcc-host-undef-gnu-source", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        parser.error("invalid run-id")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
        parser.error("expected-commit must be an exact 40-hex Git commit")
    if args.target_issued_bytes <= 0 or args.ncu_target_issued_bytes <= 0:
        parser.error("issued-byte targets must be positive")
    if args.trial_timeout_seconds <= 0 or args.ncu_timeout_seconds <= 0:
        parser.error("timeout values must be positive")
    actual_commit = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    if actual_commit != args.expected_commit:
        raise RuntimeError(
            f"wrong checkout: expected {args.expected_commit}, found {actual_commit}"
        )
    tracked_status = run(
        ["git", "status", "--short", "--untracked-files=no"]
    ).stdout.strip()
    if tracked_status:
        raise RuntimeError("tracked worktree changes are not allowed")
    executable("nvcc")
    executable("cuobjdump")
    if not args.static_only:
        executable("nvidia-smi")
    if args.ncu:
        executable("ncu")

    run_dir = RESULT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    global_lock = (REPO / "results" / ".sm110_gpu_campaign.lock").open("w")
    try:
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another SM110 GPU campaign holds the global lock") from exc

    manifest = cases(args.target_issued_bytes)
    spec = {
        "schema_version": 1,
        "run_id": args.run_id,
        "campaign": "sm110_tma_schedule_payload_closure",
        "expected_commit": args.expected_commit,
        "expected_sm_count": EXPECTED_SMS,
        "trials": TRIALS,
        "target_issued_bytes": args.target_issued_bytes,
        "ncu_target_issued_bytes": args.ncu_target_issued_bytes,
        "collect_ncu": args.ncu,
        "trial_timeout_seconds": args.trial_timeout_seconds,
        "ncu_timeout_seconds": args.ncu_timeout_seconds,
        "static_only": args.static_only,
        "timed_scope": (
            "earliest CTA globaltimer start to latest CTA stop; each TMA load "
            "includes issue, mbarrier completion wait, and SMEM destination"
        ),
        "generator": str(Path(__file__).relative_to(REPO)),
        "generator_sha256": sha256(Path(__file__)),
        "source_dependencies": {str(SOURCE.relative_to(REPO)): sha256(SOURCE)},
        "host_compiler": args.host_compiler,
        "nvcc_host_undef_gnu_source": args.nvcc_host_undef_gnu_source,
        "cases": manifest,
    }
    spec_text = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    spec_path = run_dir / "run_spec.json"
    if spec_path.is_file() and spec_path.read_text() != spec_text:
        raise RuntimeError("run-id exists with a different immutable run contract")
    spec_path.write_text(spec_text)
    complete_marker = run_dir / "COMPLETE"
    if complete_marker.is_file():
        complete_marker.unlink()
    write_status(run_dir, "running", current_case=None, completed_cases=0)

    if not args.static_only:
        snapshot = environment()
        validate_environment(snapshot, args.expected_commit)
        environment_path = run_dir / "environment.json"
        if not environment_path.is_file():
            environment_path.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
            )
        with (run_dir / "environment_snapshots.jsonl").open("a") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")

    artifact = compile_binary(
        run_dir, args.host_compiler, args.nvcc_host_undef_gnu_source
    )
    binary = Path(artifact["binary"])
    ncu_metrics: list[str] = []
    if args.ncu and not args.static_only:
        ncu_metrics, _ = query_ncu_metrics()

    results: list[dict[str, object]] = []
    for case in manifest:
        case_id = str(case["id"])
        write_status(
            run_dir,
            "running",
            current_case=case_id,
            completed_cases=len(results),
            total_cases=len(manifest),
        )
        case_dir = run_dir / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = sha256_text(
            json.dumps(case, sort_keys=True)
            + str(spec["generator_sha256"])
            + str(artifact["source_sha256"])
            + str(artifact["binary_sha256"])
            + str(artifact["sass_sha256"])
        )
        result_path = case_dir / "result.json"
        trials_path = case_dir / "trials.jsonl"
        if result_is_complete(result_path, fingerprint, args.ncu):
            prior = json.loads(result_path.read_text())
            if len([line for line in trials_path.read_text().splitlines() if line]) == TRIALS:
                results.append(prior)
                print(f"SKIP {case_id}: complete artifact match", flush=True)
                continue
        base: dict[str, object] = {
            "case_id": case_id,
            "resource": case["resource"],
            "residency": case["residency"],
            "tile_bytes": case["tile_bytes"],
            "destination_slots": case["destination_slots"],
            "threads_per_cta": case["threads_per_cta"],
            "resident_ctas_per_sm": case["resident_ctas_per_sm"],
            "fingerprint": fingerprint,
            "source_path": str(SOURCE.relative_to(REPO)),
            "source_sha256": artifact["source_sha256"],
            "binary_sha256": artifact["binary_sha256"],
            "binary_hash_path": "build/binary.sha256",
            "sass_path": "build/sass.txt",
            "sass_sha256": artifact["sass_sha256"],
            "sass_tokens": list(SASS_TOKENS),
        }
        if args.static_only:
            results.append({**base, "status": "static_ok"})
            continue
        trial_rows: list[dict[str, object]] = []
        rates: list[float] = []
        for trial in range(1, TRIALS + 1):
            proc = run_bounded(
                [str(binary), *case["args"]], args.trial_timeout_seconds)
            if proc.returncode:
                raise RuntimeError(
                    f"{case_id} trial {trial} failed ({proc.returncode}):\n{proc.stdout}"
                )
            fields = parse_kv(proc.stdout)
            rate = validate_trial(case, fields, int(case["iterations"]))
            rates.append(rate)
            trial_rows.append(
                {
                    "trial": trial,
                    "captured_at_utc": utc_now(),
                    "returncode": proc.returncode,
                    "raw_stdout": proc.stdout,
                    "fields": fields,
                    "audited_rate_per_second": rate,
                }
            )
        trials_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in trial_rows)
        )
        result: dict[str, object] = {
            **base,
            "status": "ok",
            "trial_count": len(rates),
            "rate_unit": "B/s",
            "rate_per_second_median": statistics.median(rates),
            "rate_per_second_min": min(rates),
            "rate_per_second_max": max(rates),
        }
        if args.ncu:
            result["ncu"] = collect_ncu(
                case_dir, case, binary, ncu_metrics,
                args.ncu_target_issued_bytes, args.ncu_timeout_seconds
            )
        else:
            result["ncu"] = {"collected": False}
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
        print(
            f"PASS {case_id}: median={statistics.median(rates):.9e} B/s",
            flush=True,
        )

    status = "static_complete" if args.static_only else "complete"
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": status,
        "case_count": len(results),
        "results": results,
        "updated_at_utc": utc_now(),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_manifest = generate_campaign_plots(summary_path)
    if not args.static_only:
        complete_marker.write_text(
            f"run_id={args.run_id}\nsummary_sha256={sha256(summary_path)}\n"
        )
        binary.unlink()
    write_status(
        run_dir,
        status,
        current_case=None,
        completed_cases=len(results),
        total_cases=len(manifest),
    )
    print(json.dumps({"run_dir": str(run_dir), "status": status,
                      "plot_count": plot_manifest["chart_count"],
                      "plots": str(run_dir / "plots")}, indent=2))
    return 0


def mark_failed(message: str) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id")
    args, _ = parser.parse_known_args()
    if args.run_id and re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        run_dir = RESULT_ROOT / args.run_id
        if run_dir.is_dir():
            write_status(run_dir, "failed", error=message)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        mark_failed(str(exc))
        raise
