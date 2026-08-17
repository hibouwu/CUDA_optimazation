#!/usr/bin/env python3
"""Collect GEMM-shaped simultaneous read/write service curves on Thor/SM110."""

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
from dataclasses import replace
from fractions import Fraction
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.sm110_gemm_model.io import load_schedules, load_workloads
from scripts.sm110_gemm_model.model import account_work, precision_specs
from scripts.sm110_gemm_model.campaign_plots import generate_campaign_plots


RESULT_ROOT = REPO / "results" / "sm110_memory_duplex_campaign"
SOURCE = REPO / "microbench/14_memory_path_bandwidth/memory_path_bandwidth.cu"
WORKLOAD_MANIFEST = REPO / "scripts/sm110_gemm_model/examples/workloads.json"
SCHEDULE_MANIFEST = REPO / "scripts/sm110_gemm_model/examples/schedules.json"
MODEL_SOURCE = REPO / "scripts/sm110_gemm_model/model.py"
EXPECTED_SMS = 20
TRIALS = 10
BLOCKS_PER_SM = 4
THREADS = 256
L2_BYTES = 16 << 20
HBM_BYTES = 256 << 20
TARGET_BYTES = 512 << 20
NCU_TARGET_BYTES = 64 << 20
BYTES_PER_LOGICAL_OPERATION = 8 * 16
MAX_OPERATION_GROUPS = 128
TARGET_SQUARE_SHAPES = (1024, 2048, 4096)
NCU_METRICS = (
    "gpu__time_duration.sum",
    "lts__t_bytes.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_op_read_lookup_hit.sum",
    "lts__t_sectors_op_read_lookup_miss.sum",
)
NCU_BASE_UNITS = {
    "gpu__time_duration.sum": "ns",
    "lts__t_bytes.sum": "byte",
    "lts__t_sectors_op_read.sum": "sector",
    "lts__t_sectors_op_write.sum": "sector",
    "lts__t_sectors_op_read_lookup_hit.sum": "sector",
    "lts__t_sectors_op_read_lookup_miss.sum": "sector",
}
NCU_NUMBER_RE = re.compile(
    r"^[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?$"
)


def derive_ratio_contracts() -> tuple[tuple[tuple[int, int], ...],
                                      tuple[tuple[int, int], ...]]:
    """Derive exact reduced byte ratios from the executable model manifests."""
    precisions = precision_specs()
    workload_templates = load_workloads(WORKLOAD_MANIFEST)
    workloads = [
        replace(
            workload,
            workload_id=f"duplex_ratio_n{shape}_{workload.precision_id}",
            m=shape,
            n=shape,
            k=shape,
        )
        for workload in workload_templates
        for shape in TARGET_SQUARE_SHAPES
    ]
    schedules = load_schedules(SCHEDULE_MANIFEST)
    hbm: set[Fraction] = set()
    l2: set[Fraction] = set()
    for workload in workloads:
        precision = precisions[workload.precision_id]
        for schedule in schedules:
            if workload.precision_id not in schedule.supported_precisions:
                continue
            try:
                work = account_work(workload, schedule, precision)
            except ValueError:
                continue
            hbm_read = (work.input_value_bytes_min + work.input_scale_bytes_min
                        + work.c_read_bytes_min)
            hbm_write = work.output_value_bytes_min + work.output_scale_bytes_min
            l2_read = work.tma_input_bytes + work.c_read_bytes_min
            if not all(float(value).is_integer()
                       for value in (hbm_read, hbm_write, l2_read)):
                raise RuntimeError("manifest byte accounting is not integral")
            hbm.add(Fraction(int(hbm_read), int(hbm_write)))
            l2.add(Fraction(int(l2_read), int(hbm_write)))
    if not hbm or not l2:
        raise RuntimeError("model manifests produced an empty duplex ratio surface")
    return (
        tuple((ratio.numerator, ratio.denominator) for ratio in sorted(hbm)),
        tuple((ratio.numerator, ratio.denominator) for ratio in sorted(l2)),
    )


HBM_RATIOS, L2_RATIOS = derive_ratio_contracts()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def duplex_sass_block(sass: str) -> str:
    markers = list(re.finditer(r"(?m)^\s*Function\s*:\s*.*$", sass))
    for index, marker in enumerate(markers):
        if "duplex_kernel" not in marker.group(0):
            continue
        end = markers[index + 1].start() if index + 1 < len(markers) else len(sass)
        return sass[marker.start():end]
    raise RuntimeError("duplex kernel function block is missing from SASS")


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if check and result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result


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


def iterations_for(target_bytes: int, read_ops: int, write_ops: int) -> int:
    bytes_per_iteration = (
        EXPECTED_SMS * BLOCKS_PER_SM * THREADS
        * (read_ops + write_ops) * BYTES_PER_LOGICAL_OPERATION)
    return max(1, math.ceil(target_bytes / bytes_per_iteration))


def cases(target_bytes: int = TARGET_BYTES) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for residency, working_set, ratios in (
        ("hbm", HBM_BYTES, HBM_RATIOS), ("l2", L2_BYTES, L2_RATIOS)):
        for read_ops, write_ops in ratios:
            divisor = math.gcd(read_ops, write_ops)
            read_ops //= divisor
            write_ops //= divisor
            iterations = iterations_for(target_bytes, read_ops, write_ops)
            if max(read_ops, write_ops) > MAX_OPERATION_GROUPS:
                raise RuntimeError(
                    f"ratio {read_ops}:{write_ops} exceeds binary operation-group "
                    f"limit {MAX_OPERATION_GROUPS}")
            case_id = f"{residency}_duplex_r{read_ops}_w{write_ops}"
            cold_proxy = residency == "hbm"
            result.append({
                "id": case_id,
                "resource": (
                    f"hbm.duplex.proxy.r{read_ops}_w{write_ops}"
                    if cold_proxy else f"l2.duplex.r{read_ops}_w{write_ops}"
                ),
                "residency": "cold_hbm" if residency == "hbm" else "hot_l2",
                "evidence_contract": (
                    "cold_read_l2_miss_plus_write_l2_issue_proxy"
                    if cold_proxy else "hot_l2_read_hit_plus_write_l2_issue"
                ),
                "external_write_bytes_proven": False,
                "max_operation_groups": MAX_OPERATION_GROUPS,
                "read_operations": read_ops,
                "write_operations": write_ops,
                "working_set_bytes_per_direction": working_set,
                "iterations": iterations,
                "args": [
                    "--direction", "duplex", "--residency", residency,
                    "--bytes", str(working_set), "--iters", str(iterations),
                    "--warmup-iters", "64", "--blocks-per-sm",
                    str(BLOCKS_PER_SM), "--threads", str(THREADS),
                    "--read-ops", str(read_ops), "--write-ops", str(write_ops),
                ],
                "sass_tokens": ["LDG.E.128", "STG.E.128"],
            })
    return result


def environment() -> dict[str, object]:
    commands = {
        "gpu_identity": ["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
                         "--format=csv,noheader"],
        "gpu_state": ["nvidia-smi", "--query-gpu=pstate,clocks.current.graphics,power.limit",
                      "--format=csv,noheader"],
        "nvcc": ["nvcc", "--version"], "ncu": ["ncu", "--version"],
        "git_head": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--short", "--untracked-files=no"],
    }
    value: dict[str, object] = {
        "captured_at_utc": utc_now(), "hostname": platform.node(),
        "platform": platform.platform(), "python": sys.version,
    }
    for name, command in commands.items():
        proc = run(command, check=False)
        value[name] = {"returncode": proc.returncode, "output": proc.stdout}
    nvpmodel = shutil.which("nvpmodel")
    proc = run([nvpmodel, "-q"], check=False) if nvpmodel else None
    value["power_mode"] = {
        "returncode": proc.returncode if proc else 127,
        "output": proc.stdout if proc else "nvpmodel not found",
    }
    return value


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
        raise RuntimeError("MAXN mode is not proven")
    if str(snapshot["git_head"]["output"]).strip() != expected_commit:
        raise RuntimeError("environment snapshot commit mismatch")
    if str(snapshot["git_status"]["output"]).strip():
        raise RuntimeError("environment snapshot tracked worktree is dirty")


def write_status(run_dir: Path, status: str, **extra: object) -> None:
    value = {"status": status, "pid": os.getpid(), "hostname": platform.node(),
             "updated_at_utc": utc_now(), **extra}
    (run_dir / "campaign_status.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n")
    with (run_dir / "progress.jsonl").open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def bounded(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(command, cwd=REPO, text=True, stdout=subprocess.PIPE,
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
        raise RuntimeError(f"command timed out after {timeout_seconds}s: {' '.join(command)}")
    return subprocess.CompletedProcess(command, proc.returncode, stdout, "")


def bounded_redirect(
    command: list[str], timeout_seconds: int, stdout_file: object
) -> subprocess.CompletedProcess[str]:
    """Run a profiler with a bounded lifetime for its complete process group."""
    proc = subprocess.Popen(
        command,
        cwd=REPO,
        text=True,
        stdout=stdout_file,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            _, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            _, stderr = proc.communicate(timeout=5)
        raise RuntimeError(
            f"command timed out after {timeout_seconds}s: {' '.join(command)}"
        )
    return subprocess.CompletedProcess(command, proc.returncode, "", stderr)


def compile_binary(run_dir: Path, host_compiler: str | None,
                   undef_gnu_source: bool) -> dict[str, object]:
    build = run_dir / "build"
    build.mkdir(exist_ok=True)
    binary = build / "memory_duplex"
    command = [executable("nvcc"), "-O3", "-std=c++17"]
    if host_compiler:
        command += ["-ccbin", host_compiler]
    if undef_gnu_source:
        command += ["-Xcompiler=-U_GNU_SOURCE", "-D_DEFAULT_SOURCE",
                    "-D_POSIX_C_SOURCE=200809L", "-D_XOPEN_SOURCE=700",
                    "-D_XOPEN_SOURCE_EXTENDED=1", "-D_LARGEFILE64_SOURCE=1",
                    "-D_ATFILE_SOURCE=1"]
    command += ["-gencode", "arch=compute_110a,code=sm_110a", str(SOURCE),
                "-o", str(binary)]
    (build / "compile_command.json").write_text(json.dumps(command, indent=2) + "\n")
    compiled = run(command, check=False)
    (build / "compile.log").write_text(compiled.stdout)
    if compiled.returncode:
        raise RuntimeError("memory duplex compile failed")
    sass_path = build / "sass.txt"
    sass = run([executable("cuobjdump"), "--dump-sass", str(binary)], check=False)
    sass_path.write_text(sass.stdout)
    if sass.returncode:
        raise RuntimeError("memory duplex disassembly failed")
    function_sass = duplex_sass_block(sass.stdout)
    if any(token not in function_sass for token in ("LDG.E.128", "STG.E.128")):
        raise RuntimeError("memory duplex SASS contract failed")
    function_sass_path = build / "duplex_kernel.sass"
    function_sass_path.write_text(function_sass)
    binary_hash = sha256(binary)
    (build / "binary.sha256").write_text(f"{binary_hash}  memory_duplex\n")
    return {"binary": binary, "binary_sha256": binary_hash,
            "source_sha256": sha256(SOURCE), "sass_sha256": sha256(sass_path),
            "function_sass_sha256": sha256(function_sass_path)}


def validate_trial(case: dict[str, object], fields: dict[str, str]) -> float:
    cid = str(case["id"])
    for name, expected in {
        "case_id": cid, "direction": "duplex",
        "residency": "hbm" if case["residency"] == "cold_hbm" else "l2",
        "sm_count": str(EXPECTED_SMS), "unique_smid_count": str(EXPECTED_SMS),
        "blocks": str(EXPECTED_SMS * BLOCKS_PER_SM),
        "blocks_per_sm": str(BLOCKS_PER_SM), "threads": str(THREADS),
        "iterations": str(case["iterations"]), "warmup_iterations": "64",
        "max_operation_groups": str(MAX_OPERATION_GROUPS),
        "read_operations_per_iteration": str(case["read_operations"]),
        "write_operations_per_iteration": str(case["write_operations"]),
        "working_set_bytes": str(case["working_set_bytes_per_direction"]),
        "read_working_set_bytes": str(case["working_set_bytes_per_direction"]),
        "write_working_set_bytes": str(case["working_set_bytes_per_direction"]),
    }.items():
        if fields.get(name) != expected:
            raise RuntimeError(f"{cid}: {name} expected={expected} actual={fields.get(name)}")
    read_bytes = int(fields["requested_read_bytes"])
    write_bytes = int(fields["requested_write_bytes"])
    expected_base = (EXPECTED_SMS * BLOCKS_PER_SM * THREADS
                     * int(case["iterations"]) * BYTES_PER_LOGICAL_OPERATION)
    expected_read = expected_base * int(case["read_operations"])
    expected_write = expected_base * int(case["write_operations"])
    if read_bytes != expected_read or write_bytes != expected_write:
        raise RuntimeError(
            f"{cid}: requested byte counts differ from immutable grid/iteration contract"
        )
    if read_bytes * int(case["write_operations"]) != (
            write_bytes * int(case["read_operations"])):
        raise RuntimeError(f"{cid}: measured byte ratio differs from case contract")
    total = read_bytes + write_bytes
    if int(fields["requested_bytes"]) != total:
        raise RuntimeError(f"{cid}: requested total is not read+write")
    elapsed_ns = int(fields["globaltimer_elapsed_ns"])
    rate = total * 1e9 / elapsed_ns
    if elapsed_ns <= 0 or not math.isclose(
            rate, float(fields["bytes_per_second"]), rel_tol=2e-9):
        raise RuntimeError(f"{cid}: rate arithmetic mismatch")
    return rate


def query_metrics() -> list[str]:
    output = run(["ncu", "--query-metrics", "--query-metrics-mode", "all"]).stdout
    names = {line.split()[0] for line in output.splitlines()
             if line and not line[0].isspace()}
    missing = [name for name in NCU_METRICS if name not in names]
    if missing:
        raise RuntimeError(f"required NCU metrics unavailable: {missing}")
    return list(NCU_METRICS)


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
        id_index = header.index("ID")
        kernel_index = header.index("Kernel Name")
        time_index = header.index("gpu__time_duration.sum")
        candidates = []
        for row in rows[index + 1:]:
            if (len(row) != len(header) or not row[id_index].isdigit()
                    or "duplex_kernel" not in row[kernel_index]):
                continue
            try:
                duration = ncu_number(row[time_index])
            except ValueError:
                pass
            else:
                if duration >= 0:
                    candidates.append((duration, row))
        if candidates:
            # The binary launches warmup first and the measured kernel second.
            # Selecting the final duplex launch keeps the NCU byte contract tied
            # to --iters while still allowing warmup to establish hot-L2 state.
            return dict(zip(header, candidates[-1][1]))
    raise RuntimeError(f"no duplex kernel row in {path}")


def collect_ncu(case_dir: Path, case: dict[str, object], binary: Path,
                metrics: list[str], timeout_seconds: int) -> dict[str, object]:
    ncu_dir = case_dir / "ncu"
    ncu_dir.mkdir(exist_ok=True)
    ncu_iters = iterations_for(NCU_TARGET_BYTES, int(case["read_operations"]),
                               int(case["write_operations"]))
    ncu_args = list(case["args"])
    ncu_args[ncu_args.index("--iters") + 1] = str(ncu_iters)
    raw = ncu_dir / "raw.csv"
    stderr_path = ncu_dir / "stderr.log"
    report_base = ncu_dir / "profile"
    command = ["ncu", "--target-processes", "all", "--replay-mode", "kernel",
               "--page", "raw", "--csv", "--force-overwrite", "-o",
               str(report_base), "--metrics", ",".join(metrics), str(binary),
               *ncu_args]
    with raw.open("w") as output:
        proc = bounded_redirect(command, timeout_seconds, output)
    stderr_path.write_text(proc.stderr)
    report = ncu_dir / "profile.ncu-rep"
    if proc.returncode or "ERR_NVGPUCTRPERM" in proc.stderr or not report.is_file():
        raise RuntimeError(f"{case['id']}: NCU failed; see {stderr_path}")
    row = find_ncu_row(raw)
    values = {name: ncu_number(row[name]) for name in metrics}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise RuntimeError(f"{case['id']}: invalid NCU values")
    read_requested = (EXPECTED_SMS * BLOCKS_PER_SM * THREADS * ncu_iters
                      * int(case["read_operations"]) * BYTES_PER_LOGICAL_OPERATION)
    write_requested = (EXPECTED_SMS * BLOCKS_PER_SM * THREADS * ncu_iters
                       * int(case["write_operations"]) * BYTES_PER_LOGICAL_OPERATION)
    if values["lts__t_sectors_op_read.sum"] * 32 < read_requested * 0.90:
        raise RuntimeError(f"{case['id']}: NCU read traffic is below contract")
    if values["lts__t_sectors_op_write.sum"] * 32 < write_requested * 0.90:
        raise RuntimeError(f"{case['id']}: NCU write traffic is below contract")
    miss_proxy_bytes = values[
        "lts__t_sectors_op_read_lookup_miss.sum"] * 32.0
    if case["residency"] == "hot_l2":
        if values["lts__t_sectors_op_read_lookup_hit.sum"] <= values[
                "lts__t_sectors_op_read_lookup_miss.sum"]:
            raise RuntimeError(f"{case['id']}: NCU does not prove L2-hit residency")
    else:
        if miss_proxy_bytes < read_requested * 0.60:
            raise RuntimeError(
                f"{case['id']}: L2 miss sectors do not prove cold DRAM reads")
    summary = {"returncode": proc.returncode, "metrics": metrics,
               "iterations": ncu_iters, "requested_read_bytes": read_requested,
               "requested_write_bytes": write_requested, "values": values,
               "evidence_contract": case["evidence_contract"],
               "external_write_bytes_proven": False,
               "report_path": str(report.relative_to(case_dir)),
               "report_sha256": sha256(report),
               "raw_path": str(raw.relative_to(case_dir)), "raw_sha256": sha256(raw),
               "stderr_path": str(stderr_path.relative_to(case_dir)),
               "stderr_sha256": sha256(stderr_path)}
    if case["residency"] == "cold_hbm":
        summary.update({
            "cold_read_miss_proxy_bytes": miss_proxy_bytes,
            "cold_read_miss_proxy_to_requested": (
                miss_proxy_bytes / read_requested),
        })
    (ncu_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def result_is_complete(path: Path, fingerprint: str) -> bool:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    ncu = result.get("ncu", {})
    return (
        result.get("status") == "ok"
        and result.get("trial_count") == TRIALS
        and result.get("fingerprint") == fingerprint
        and ncu.get("returncode") == 0
        and (path.parent / str(ncu.get("report_path", ""))).is_file()
        and (path.parent / str(ncu.get("raw_path", ""))).is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host-compiler")
    parser.add_argument("--nvcc-host-undef-gnu-source", action="store_true")
    parser.add_argument("--trial-timeout-seconds", type=int, default=120)
    parser.add_argument("--ncu-timeout-seconds", type=int, default=300)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        parser.error("invalid run-id")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
        parser.error("expected-commit must be 40 lowercase hex characters")
    if args.trial_timeout_seconds <= 0 or args.ncu_timeout_seconds <= 0:
        parser.error("timeout values must be positive")
    if run(["git", "rev-parse", "HEAD"]).stdout.strip() != args.expected_commit:
        raise RuntimeError("wrong checkout")
    if run(["git", "status", "--short", "--untracked-files=no"]).stdout.strip():
        raise RuntimeError("tracked worktree changes are not allowed")
    run_dir = RESULT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (REPO / "results/.sm110_gpu_campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = cases()
    spec = {"schema_version": 1, "run_id": args.run_id,
            "campaign": "sm110_memory_duplex_closure",
            "expected_commit": args.expected_commit, "expected_sm_count": EXPECTED_SMS,
            "trials": TRIALS, "ncu_required": True,
            "max_operation_groups": MAX_OPERATION_GROUPS,
            "cold_duplex_evidence": {
                "read": "32 * l2_read_lookup_miss_sectors >= 0.60 * requested_read_bytes",
                "write": "32 * l2_write_sectors >= 0.90 * requested_write_bytes",
                "external_write_bytes_proven": False,
                "qualification": "cold_dram_read_plus_write_path_proxy",
            },
            "target_square_shapes": list(TARGET_SQUARE_SHAPES),
            "trial_timeout_seconds": args.trial_timeout_seconds,
            "ncu_timeout_seconds": args.ncu_timeout_seconds,
            "static_only": args.static_only,
            "generator": str(Path(__file__).relative_to(REPO)),
            "generator_sha256": sha256(Path(__file__)),
            "source_dependencies": {
                str(path.relative_to(REPO)): sha256(path)
                for path in (SOURCE, WORKLOAD_MANIFEST, SCHEDULE_MANIFEST, MODEL_SOURCE)
            },
            "cases": manifest}
    spec_text = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    spec_path = run_dir / "run_spec.json"
    if spec_path.exists() and spec_path.read_text() != spec_text:
        raise RuntimeError("run-id exists with a different immutable contract")
    spec_path.write_text(spec_text)
    complete_marker = run_dir / "COMPLETE"
    if complete_marker.is_file():
        complete_marker.unlink()
    write_status(run_dir, "running", completed_cases=0, total_cases=len(manifest))
    if not args.static_only:
        snapshot = environment()
        validate_environment(snapshot, args.expected_commit)
        environment_path = run_dir / "environment.json"
        if not environment_path.is_file():
            environment_path.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        with (run_dir / "environment_snapshots.jsonl").open("a") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
    artifact = compile_binary(run_dir, args.host_compiler,
                              args.nvcc_host_undef_gnu_source)
    binary = Path(artifact["binary"])
    metrics = [] if args.static_only else query_metrics()
    results = []
    for case in manifest:
        case_dir = run_dir / "cases" / str(case["id"])
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
        base = {"case_id": case["id"], "resource": case["resource"],
                "residency": case["residency"],
                "evidence_contract": case["evidence_contract"],
                "external_write_bytes_proven": False,
                "max_operation_groups": MAX_OPERATION_GROUPS,
                "source_path": str(SOURCE.relative_to(REPO)),
                "source_sha256": artifact["source_sha256"],
                "binary_sha256": artifact["binary_sha256"],
                "sass_sha256": artifact["sass_sha256"],
                "function_sass_path": "build/duplex_kernel.sass",
                "function_sass_sha256": artifact["function_sass_sha256"],
                "sass_tokens": case["sass_tokens"], "fingerprint": fingerprint}
        if args.static_only:
            results.append({**base, "status": "static_ok"})
            continue
        if result_is_complete(result_path, fingerprint):
            prior = json.loads(result_path.read_text())
            try:
                trial_count = len([line for line in trials_path.read_text().splitlines()
                                   if line])
            except OSError:
                trial_count = 0
            if trial_count == TRIALS:
                results.append(prior)
                print(f"SKIP {case['id']}: complete artifact match", flush=True)
                continue
        rows, rates = [], []
        for trial in range(1, TRIALS + 1):
            outcome = bounded([str(binary), *case["args"]], args.trial_timeout_seconds)
            if outcome.returncode:
                raise RuntimeError(f"{case['id']} trial {trial} failed:\n{outcome.stdout}")
            fields = parse_kv(outcome.stdout)
            rate = validate_trial(case, fields)
            rates.append(rate)
            rows.append({"trial": trial, "raw_stdout": outcome.stdout, "fields": fields,
                         "audited_rate_per_second": rate})
        trials_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        result = {**base, "status": "ok", "trial_count": TRIALS,
                  "rate_unit": "B/s", "rate_per_second_median": statistics.median(rates),
                  "rate_per_second_min": min(rates), "rate_per_second_max": max(rates)}
        result["ncu"] = collect_ncu(case_dir, case, binary, metrics,
                                    args.ncu_timeout_seconds)
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
        write_status(run_dir, "running", current_case=case["id"],
                     completed_cases=len(results), total_cases=len(manifest))
    status = "static_complete" if args.static_only else "complete"
    summary = {"schema_version": 1, "run_id": args.run_id, "status": status,
               "case_count": len(results), "results": results}
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_manifest = generate_campaign_plots(summary_path)
    if not args.static_only:
        complete_marker.write_text(
            f"run_id={args.run_id}\nsummary_sha256={sha256(summary_path)}\n")
        binary.unlink()
    write_status(run_dir, status, completed_cases=len(results), total_cases=len(manifest))
    print(json.dumps({"run_dir": str(run_dir), "status": status,
                      "plot_count": plot_manifest["chart_count"],
                      "plots": str(run_dir / "plots")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        try:
            index = sys.argv.index("--run-id")
            failed_run_id = sys.argv[index + 1]
            if re.fullmatch(r"[A-Za-z0-9._-]+", failed_run_id):
                failed_dir = RESULT_ROOT / failed_run_id
                if failed_dir.is_dir():
                    write_status(
                        failed_dir,
                        "failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
        except (ValueError, IndexError, OSError):
            pass
        raise
