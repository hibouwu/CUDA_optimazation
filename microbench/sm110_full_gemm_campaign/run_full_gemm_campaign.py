#!/usr/bin/env python3
"""Resumable, fail-closed Thor collection for numerically checked full GEMMs."""

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
import signal
import shutil
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
RESULT_ROOT = REPO / "results" / "sm110_full_gemm_campaign"
TRIALS = 10
SHAPES = (1024, 2048, 4096)
DEFAULT_TRIAL_TIMEOUT_SECONDS = 120
DEFAULT_NCU_TIMEOUT_SECONDS = 300
TERMINATION_GRACE_SECONDS = 5

CASES = [
    {
        "id": "fp16_f32_n1024_tc5b", "precision_id": "fp16_f32",
        "binary": "fp16", "n": 1024, "split": "calibration",
        "backend_id": "tc5b", "args": ["1024", "tc5b", "none"],
        "work_unit": "flop", "internal_repeats": 100,
        "csv_precision": "fp16->fp32",
        "reference_contract": "fp16_f32_cpu_samples",
        "numerical_contract": "fp_accumulator",
        "function_substring": "tc4bc_raw_2sm_cluster_kernelILb1ELi128ELi2E",
        "sass_tokens": ["UTMALDG", "UTCHMMA", "LDTM", "STG"],
        "ncu_kernel_regex": "tc4bc_raw_2sm_cluster_kernel",
    },
    *[
        {
            "id": f"fp16_f32_n{n}_tc5a", "precision_id": "fp16_f32",
            "binary": "fp16", "n": n,
            "split": "holdout" if n == 4096 else "calibration",
            "backend_id": "tc5a", "args": [str(n), "tc5a", "none"],
            "work_unit": "flop", "internal_repeats": 100,
            "csv_precision": "fp16->fp32",
            "reference_contract": "fp16_f32_cpu_samples",
            "numerical_contract": "fp_accumulator",
            "function_substring":
                "tc5a_overlap_epilogue_1sm_kernelILi128ELi256ELi64ELi4ELi4E",
            "sass_tokens": ["UTMALDG", "UTCHMMA", "LDTM", "STG"],
            "ncu_kernel_regex": "tc5a_overlap_epilogue_1sm_kernel",
        }
        for n in (2048, 4096)
    ],
    *[
        {
            "id": f"e4m3_f32_n{n}_q7", "precision_id": "e4m3_f32",
            "binary": "quant", "n": n,
            "split": "holdout" if n == 4096 else "calibration",
            "backend_id": "fp8_q7_mma_m16n8k32_smem128x64",
            "args": [str(n), "fp8_q7_mma_m16n8k32_smem128x64"],
            "work_unit": "flop", "internal_repeats": 10,
            "csv_precision": "fp8_e4m3->fp32",
            "reference_contract": "fp8_e4m3_f32_cpu_samples",
            "numerical_contract": "fp8_e4m3_f32",
            "function_substring": "fp8_mma_m16n8k32_smem128x64_kernel",
            "sass_tokens": ["HMMA.16816.F32", "STG"],
            "ncu_kernel_regex": "fp8_mma_m16n8k32_smem128x64_kernel",
        }
        for n in SHAPES
    ],
    *[
        {
            "id": f"e5m2_f32_n{n}_q0", "precision_id": "e5m2_f32",
            "binary": "extended", "n": n,
            "split": "holdout" if n == 4096 else "calibration",
            "backend_id": "e5m2_q0_mma_m16n8k32_smem128x64",
            "args": [str(n), "e5m2"],
            "work_unit": "flop", "internal_repeats": 10,
            "csv_precision": "e5m2->fp32",
            "reference_contract": "e5m2_f32_cpu_samples",
            "numerical_contract": "e5m2_f32",
            "reference_backend_id": "e5m2_q1_mma_m16n8k32_global",
            "function_substring":
                "e5m2_mma_m16n8k32_smem128x64_kernel",
            "sass_tokens": ["HMMA.16816.F32", "STG.E"],
            "reference_function_substring":
                "e5m2_mma_m16n8k32_global_kernel",
            "reference_sass_tokens": ["HMMA.16816.F32", "STG.E"],
            "ncu_kernel_regex":
                "e5m2_mma_m16n8k32_smem128x64_kernel",
        }
        for n in SHAPES
    ],
    *[
        {
            "id": f"s8_s32_n{n}_q15", "precision_id": "s8_s32",
            "binary": "quant", "n": n,
            "split": "holdout" if n == 4096 else "calibration",
            "backend_id": "int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol",
            "args": [str(n), "int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol"],
            "work_unit": "operation", "internal_repeats": 10,
            "csv_precision": "int8->int32",
            "reference_contract": "s8_s32_cpu_samples",
            "numerical_contract": "s8_s32_exact",
            "function_substring": "int8_wmma_m128n64k16_4warp_reuse_a_bcol_kernel",
            "sass_tokens": ["IMMA.16816.S8.S8", "STG"],
            "ncu_kernel_regex": "int8_wmma_m128n64k16_4warp_reuse_a_bcol_kernel",
        }
        for n in SHAPES
    ],
    *[
        {
            "id": f"bf16_f32_n{n}_q0", "precision_id": "bf16_f32",
            "binary": "extended", "n": n,
            "split": "holdout" if n == 4096 else "calibration",
            "backend_id": "bf16_q0_wmma_m128n64k16", "args": [str(n), "bf16"],
            "work_unit": "flop", "internal_repeats": 10,
            "csv_precision": "bf16->fp32",
            "reference_contract": "bf16_f32_cpu_samples",
            "numerical_contract": "bf16_f32",
            "function_substring": "wmma_m128n64_kernelI13__nv_bfloat16fLi16",
            "sass_tokens": ["HMMA.16816.F32.BF16", "STG"],
            "ncu_kernel_regex": "wmma_m128n64_kernel.*bfloat16",
        }
        for n in SHAPES
    ],
    *[
        {
            "id": f"tf32_f32_n{n}_q0", "precision_id": "tf32_f32",
            "binary": "extended", "n": n,
            "split": "holdout" if n == 4096 else "calibration",
            "backend_id": "tf32_q0_wmma_m64n64k8", "args": [str(n), "tf32"],
            "work_unit": "flop", "internal_repeats": 10,
            "csv_precision": "tf32->fp32",
            "reference_contract": "tf32_f32_cpu_samples",
            "numerical_contract": "tf32_f32",
            "function_substring": "tf32_wmma_m64n64_kernel",
            "sass_tokens": ["HMMA.1684.F32.TF32", "ST.E"],
            "ncu_kernel_regex": "tf32_wmma_m64n64_kernel",
        }
        for n in SHAPES
    ],
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], *, cwd: Path = REPO,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            check=False)
    if check and result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}")
    return result


def run_bounded(command: list[str], *, cwd: Path,
                timeout_seconds: int) -> dict[str, object]:
    """Run a GPU-facing process with bounded TERM/KILL escalation."""
    started_at_utc = utc_now()
    proc = subprocess.Popen(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    termination_failed = False
    try:
        output, _ = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as initial_timeout:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            output, _ = proc.communicate(timeout=TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired as term_timeout:
            os.killpg(proc.pid, signal.SIGKILL)
            try:
                output, _ = proc.communicate(
                    timeout=TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired as kill_timeout:
                termination_failed = True
                output = (kill_timeout.stdout or term_timeout.stdout
                          or initial_timeout.stdout or "")
                if isinstance(output, bytes):
                    output = output.decode(errors="backslashreplace")
    return {
        "command": command,
        "started_at_utc": started_at_utc,
        "finished_at_utc": utc_now(),
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "termination_failed": termination_failed,
        "returncode": proc.returncode,
        "stdout": output,
    }


def executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"required executable not found: {name}")
    return path


def write_status(run_dir: Path, status: str, **extra: object) -> None:
    payload = {
        "status": status, "pid": os.getpid(), "hostname": platform.node(),
        "updated_at_utc": utc_now(), **extra,
    }
    (run_dir / "campaign_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (run_dir / "progress.jsonl").open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def source_dependencies() -> list[Path]:
    fp16 = [REPO / "GEMMsm110/src/main.cu"]
    fp16.extend(sorted((REPO / "GEMMsm110/include").rglob("*.cuh")))
    return fp16 + [
        REPO / "GEMMquant_sm110/src/quant_gemm_bench.cu",
        REPO / "GEMMquant_sm110/src/extended_gemm_bench.cu",
    ]


def environment() -> dict[str, object]:
    commands = {
        "gpu_identity": ["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
                         "--format=csv,noheader"],
        "gpu_state": ["nvidia-smi", "--query-gpu=pstate,clocks.current.graphics,clocks.current.memory,power.limit",
                      "--format=csv,noheader"],
        "nvcc": ["nvcc", "--version"],
        "cuobjdump": ["cuobjdump", "--version"],
        "git_head": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--short"],
    }
    snapshot: dict[str, object] = {
        "captured_at_utc": utc_now(), "hostname": platform.node(),
        "platform": platform.platform(), "python": sys.version,
    }
    for name, command in commands.items():
        proc = run(command, check=False)
        snapshot[name] = {"returncode": proc.returncode, "output": proc.stdout}
    nvpmodel = shutil.which("nvpmodel")
    if nvpmodel:
        proc = run([nvpmodel, "-q"], check=False)
        snapshot["power_mode"] = {"returncode": proc.returncode,
                                  "output": proc.stdout}
    else:
        snapshot["power_mode"] = {"returncode": 127,
                                  "output": "nvpmodel not found"}
    return snapshot


def host_compat_flags(host_compiler: str | None,
                      undef_gnu_source: bool) -> list[str]:
    flags: list[str] = []
    if host_compiler:
        flags += ["-ccbin", host_compiler]
    if undef_gnu_source:
        flags += [
            "-Xcompiler=-U_GNU_SOURCE", "-D_DEFAULT_SOURCE",
            "-D_POSIX_C_SOURCE=200809L", "-D_XOPEN_SOURCE=700",
            "-D_XOPEN_SOURCE_EXTENDED=1", "-D_LARGEFILE64_SOURCE=1",
            "-D_ATFILE_SOURCE=1",
        ]
    return flags


def split_sass_functions(text: str) -> list[str]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if "Function :" in line:
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(block) for block in blocks]


def matching_sass_evidence(sass: str, case: dict[str, object]) -> dict[str, object]:
    blocks = split_sass_functions(sass)

    def require(function_key: str, tokens_key: str) -> dict[str, object]:
        needle = str(case[function_key])
        tokens = list(case[tokens_key])
        matches = [block for block in blocks
                   if needle in block.splitlines()[0]]
        valid = [block for block in matches
                 if all(token in block for token in tokens)]
        if not valid:
            raise RuntimeError(
                f"{case['id']}: no function block matching {needle!r} contains "
                f"all SASS tokens {tokens}")
        return {
            "function_substring": needle,
            "matching_function_headers": [
                block.splitlines()[0].strip() for block in valid
            ],
            "sass_tokens": tokens,
        }

    evidence = require("function_substring", "sass_tokens")
    if "reference_function_substring" in case:
        evidence["reference"] = require(
            "reference_function_substring", "reference_sass_tokens")
    return evidence


def compile_binaries(run_dir: Path, host_compiler: str | None,
                     undef_gnu_source: bool) -> dict[str, dict[str, object]]:
    nvcc, cuobjdump = executable("nvcc"), executable("cuobjdump")
    build = run_dir / "build"
    build.mkdir(exist_ok=True)
    compat = host_compat_flags(host_compiler, undef_gnu_source)
    specs = {
        "fp16": [
            nvcc, "-O3", "-std=c++17", *compat,
            "--expt-relaxed-constexpr", "-diag-suppress=20012",
            "-diag-suppress=20013", "-diag-suppress=20015",
            "-DTC3_SM110_HOST_HAS_TCGEN05=1",
            "-DGEMM_SM110_ENABLE_CUTLASS=0",
            "-gencode", "arch=compute_110a,code=sm_110a",
            f"-I{REPO / 'GEMMsm110/include'}",
            str(REPO / "GEMMsm110/src/main.cu"),
            "-lcuda", "-lcublasLt", "-lcublas",
        ],
        "quant": [
            nvcc, "-O3", "-std=c++17", *compat,
            "--expt-relaxed-constexpr",
            "-gencode", "arch=compute_110a,code=sm_110a",
            str(REPO / "GEMMquant_sm110/src/quant_gemm_bench.cu"),
            "-lcublas", "-lcublasLt",
        ],
        "extended": [
            nvcc, "-O3", "-std=c++17", *compat,
            "--expt-relaxed-constexpr",
            "-gencode", "arch=compute_110a,code=sm_110a",
            str(REPO / "GEMMquant_sm110/src/extended_gemm_bench.cu"),
            "-lcublas",
        ],
    }
    artifacts: dict[str, dict[str, object]] = {}
    for name, base_command in specs.items():
        binary = build / name
        command = [*base_command, "-o", str(binary)]
        proc = run(command, check=False)
        (build / f"{name}.compile_command.json").write_text(
            json.dumps(command, indent=2) + "\n")
        (build / f"{name}.compile.log").write_text(proc.stdout)
        if proc.returncode:
            raise RuntimeError(f"{name} compile failed; see {name}.compile.log")
        self_test_path = None
        self_test_hash = None
        self_test_command_path = None
        self_test_command_hash = None
        if name == "extended":
            self_test_command = [str(binary), "--self-test"]
            self_test_command_path = build / "extended.self_test_command.json"
            self_test_command_path.write_text(
                json.dumps(self_test_command, indent=2) + "\n")
            self_test_command_hash = sha256(self_test_command_path)
            self_test = run(self_test_command, check=False)
            self_test_path = build / "extended.self_test.log"
            self_test_path.write_text(self_test.stdout)
            if self_test.returncode or "self_test=pass" not in self_test.stdout:
                raise RuntimeError("extended host self-test failed")
            self_test_hash = sha256(self_test_path)
        sass_proc = run([cuobjdump, "--dump-sass", str(binary)], check=False)
        sass_path = build / f"{name}.sass.txt"
        sass_path.write_text(sass_proc.stdout)
        if sass_proc.returncode:
            raise RuntimeError(f"cuobjdump failed for {name}")
        binary_hash = sha256(binary)
        (build / f"{name}.binary.sha256").write_text(
            f"{binary_hash}  {name}\n")
        artifacts[name] = {
            "binary": binary, "binary_sha256": binary_hash,
            "sass_path": sass_path, "sass_sha256": sha256(sass_path),
            "self_test_path": self_test_path,
            "self_test_sha256": self_test_hash,
            "self_test_command_path": self_test_command_path,
            "self_test_command_sha256": self_test_command_hash,
        }
    return artifacts


def parse_kv(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        for match in re.finditer(
                r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s,]+)", line):
            fields[match.group(1)] = match.group(2)
    return fields


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_positive(value: object) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def parse_trial(case: dict[str, object], trial_dir: Path,
                stdout: str) -> dict[str, object]:
    fields = parse_kv(stdout)
    if fields.get("reference_contract") != case["reference_contract"]:
        raise RuntimeError(f"{case['id']}: reference contract mismatch")
    if fields.get("numerical_contract") != case["numerical_contract"]:
        raise RuntimeError(f"{case['id']}: numerical contract mismatch")
    if int(fields.get("reference_sample_count", "0")) != 64:
        raise RuntimeError(f"{case['id']}: missing 64-sample CPU reference check")
    if int(fields.get("reference_mismatch_count", "-1")) != 0:
        raise RuntimeError(f"{case['id']}: independent CPU reference mismatch")
    if int(fields.get("mismatch_count", "-1")) != 0:
        raise RuntimeError(f"{case['id']}: full-output numerical mismatch")
    csv_name = ({"fp16": "sgemm_sm110_benchmark.csv",
                 "quant": "quant_sm110_benchmark.csv",
                 "extended": "extended_sm110_benchmark.csv"}
                [str(case["binary"])])
    csv_path = trial_dir / csv_name
    if not csv_path.is_file():
        raise RuntimeError(f"{case['id']}: missing {csv_name}")
    rows = read_csv_rows(csv_path)
    candidate = [row for row in rows if row.get("BackendId") == case["backend_id"]]
    if len(candidate) != 1 or candidate[0].get("Matched") != "1":
        raise RuntimeError(f"{case['id']}: candidate CSV row is absent or unmatched")
    candidate_row = candidate[0]
    n = int(case["n"])
    if candidate_row.get("N") != str(n):
        raise RuntimeError(f"{case['id']}: CSV problem size mismatch")
    if candidate_row.get("Precision") != case["csv_precision"]:
        raise RuntimeError(f"{case['id']}: CSV precision mismatch")
    if case["binary"] == "fp16":
        if any(fields.get(axis) != str(n) for axis in ("M", "N", "K")):
            raise RuntimeError(f"{case['id']}: stdout GEMM dimensions mismatch")
    elif fields.get("N") != str(n):
        raise RuntimeError(f"{case['id']}: stdout GEMM dimension mismatch")
    work = 2 * n * n * n
    custom_ms = float(candidate_row["TimeMs"])
    custom_rate = work * 1000.0 / custom_ms
    if not finite_positive(custom_rate):
        raise RuntimeError(f"{case['id']}: nonpositive custom rate")
    if case["binary"] == "fp16":
        reference = [row for row in rows if row.get("BackendId") == "cublas_tc"]
        if len(reference) != 1 or reference[0].get("Matched") != "1":
            raise RuntimeError(f"{case['id']}: cuBLASLt reference row missing")
        reference_ms = float(reference[0]["TimeMs"])
        reference_rate = work * 1000.0 / reference_ms
    else:
        if fields.get("backend_id") != case["backend_id"]:
            raise RuntimeError(f"{case['id']}: machine-readable backend mismatch")
        if (case.get("reference_backend_id") is not None
                and fields.get("reference_backend_id") !=
                case["reference_backend_id"]):
            raise RuntimeError(
                f"{case['id']}: machine-readable reference backend mismatch")
        if fields.get("work_unit") != case["work_unit"]:
            raise RuntimeError(f"{case['id']}: machine-readable work unit mismatch")
        if fields.get("matched") not in {"1", "true"}:
            raise RuntimeError(f"{case['id']}: machine-readable match failure")
        reported_time_ms = float(fields["time_ms"])
        reference_ms = float(fields["reference_time_ms"])
        reported_custom = float(fields["rate_per_second"])
        reported_reference = float(fields["reference_rate_per_second"])
        reference_rate = work * 1000.0 / reference_ms
        if not math.isclose(reported_time_ms, custom_ms, rel_tol=2e-7):
            raise RuntimeError(f"{case['id']}: custom time fields disagree")
        if not math.isclose(reported_custom, custom_rate, rel_tol=2e-5):
            raise RuntimeError(f"{case['id']}: custom rate does not close to time")
        if not math.isclose(reported_reference, reference_rate, rel_tol=2e-5):
            raise RuntimeError(f"{case['id']}: reference rate does not close to time")
    if not finite_positive(reference_rate):
        raise RuntimeError(f"{case['id']}: nonpositive reference rate")
    return {
        "fields": fields, "csv_rows": rows, "raw_csv": csv_path.read_text(),
        "custom_time_ms": custom_ms, "custom_rate_per_second": custom_rate,
        "reference_time_ms": reference_ms,
        "reference_rate_per_second": reference_rate,
        "ratio_to_reference": custom_rate / reference_rate,
    }


def collect_ncu(run_dir: Path, artifacts: dict[str, dict[str, object]],
                timeout_seconds: int) -> list[dict[str, object]]:
    ncu = executable("ncu")
    output: list[dict[str, object]] = []
    for case in CASES:
        if case["n"] != 4096:
            continue
        profile_dir = run_dir / "ncu" / str(case["id"])
        profile_dir.mkdir(parents=True, exist_ok=True)
        report_base = profile_dir / "profile"
        command = [
            ncu, "--force-overwrite", "-o", str(report_base), "--set", "basic",
            "--target-processes", "all", "--kernel-name-base", "demangled",
            "--kernel-name", f"regex:{case['ncu_kernel_regex']}",
            "--launch-count", "1", str(artifacts[str(case["binary"])]["binary"]),
            *[str(arg) for arg in case["args"]],
        ]
        (profile_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        outcome = run_bounded(
            command, cwd=profile_dir, timeout_seconds=timeout_seconds)
        (profile_dir / "stdout.log").write_text(str(outcome["stdout"]))
        if outcome["timed_out"]:
            (profile_dir / "timeout.json").write_text(
                json.dumps(outcome, indent=2, sort_keys=True) + "\n")
            raise RuntimeError(
                f"NCU for {case['id']} exceeded {timeout_seconds}s "
                f"(termination_failed={outcome['termination_failed']}); "
                f"see {profile_dir / 'timeout.json'}")
        report = report_base.with_suffix(".ncu-rep")
        if (outcome["returncode"] or not report.is_file()
                or report.stat().st_size == 0):
            raise RuntimeError(f"NCU failed for {case['id']}; see {profile_dir}")
        output.append({
            "case_id": case["id"],
            "report_path": str(report.relative_to(run_dir)),
            "report_sha256": sha256(report),
            "returncode": outcome["returncode"],
            "timeout_seconds": timeout_seconds,
            "timed_out": False,
            "termination_failed": False,
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--ncu", action="store_true",
                        help="also collect one N=4096 NCU report per precision")
    parser.add_argument("--host-compiler")
    parser.add_argument("--nvcc-host-undef-gnu-source", action="store_true")
    parser.add_argument(
        "--trial-timeout-seconds", type=int,
        default=DEFAULT_TRIAL_TIMEOUT_SECONDS,
        help="host timeout for each complete custom/reference GEMM trial")
    parser.add_argument(
        "--ncu-timeout-seconds", type=int,
        default=DEFAULT_NCU_TIMEOUT_SECONDS,
        help="host timeout for each NCU holdout collection")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        parser.error("invalid run-id")
    if args.static_only and args.ncu:
        parser.error("--ncu cannot be combined with --static-only")
    if args.trial_timeout_seconds <= 0:
        parser.error("--trial-timeout-seconds must be positive")
    if args.ncu_timeout_seconds <= 0:
        parser.error("--ncu-timeout-seconds must be positive")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = (REPO / "results" / ".sm110_gpu_campaign.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another SM110 GPU campaign holds the global lock") from error

    run_dir = RESULT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dependencies = source_dependencies()
    spec = {
        "schema_version": 2, "run_id": args.run_id,
        "campaign": "sm110_full_gemm_closure",
        "problem_contract": {"layout": "NN", "epilogue": "none", "beta": 0,
                             "output_mode": "accumulator", "work": "2*M*N*K"},
        "trials": TRIALS, "timing": "CUDA events around kernel launches only",
        "ncu_requested": args.ncu, "host_compiler": args.host_compiler,
        "nvcc_host_undef_gnu_source": args.nvcc_host_undef_gnu_source,
        "trial_timeout_seconds": args.trial_timeout_seconds,
        "ncu_timeout_seconds": args.ncu_timeout_seconds,
        "termination_grace_seconds": TERMINATION_GRACE_SECONDS,
        "generator": str(Path(__file__).relative_to(REPO)),
        "generator_sha256": sha256(Path(__file__)),
        "support_manifest": str((CAMPAIGN / "support_manifest.json").relative_to(REPO)),
        "support_manifest_sha256": sha256(CAMPAIGN / "support_manifest.json"),
        "static_only": args.static_only,
        "source_dependencies": {
            str(path.relative_to(REPO)): sha256(path) for path in dependencies
        },
        "cases": CASES,
    }
    spec_text = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    spec_path = run_dir / "run_spec.json"
    if spec_path.exists() and spec_path.read_text() != spec_text:
        raise RuntimeError("run-id exists with a different run contract")
    spec_path.write_text(spec_text)
    complete_marker = run_dir / "COMPLETE"
    if complete_marker.exists():
        complete_marker.unlink()
    write_status(run_dir, "running", current_case=None, completed_cases=0,
                 total_cases=len(CASES))

    if not args.static_only:
        snapshot = environment()
        identity = str(snapshot.get("gpu_identity", {}).get("output", ""))
        if "11.0" not in identity and "Thor" not in identity:
            raise RuntimeError("Thor/compute capability 11.0 identity is not proven")
        if "MAXN" not in str(snapshot.get("power_mode", {})).upper():
            raise RuntimeError("MAXN power mode is not proven")
        env_path = run_dir / "environment.json"
        if not env_path.exists():
            env_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        with (run_dir / "environment_snapshots.jsonl").open("a") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")

    artifacts = compile_binaries(run_dir, args.host_compiler,
                                 args.nvcc_host_undef_gnu_source)
    sass_evidence = {
        case["id"]: matching_sass_evidence(
            Path(artifacts[str(case["binary"])]["sass_path"]).read_text(), case)
        for case in CASES
    }
    results: list[dict[str, object]] = []
    for case in CASES:
        write_status(run_dir, "running", current_case=case["id"],
                     completed_cases=len(results), total_cases=len(CASES))
        artifact = artifacts[str(case["binary"])]
        base = {
            "case_id": case["id"], "precision_id": case["precision_id"],
            "backend_id": case["backend_id"], "n": case["n"],
            "split": case["split"], "work_unit": case["work_unit"],
            "internal_repeats": case["internal_repeats"],
            "binary_sha256": artifact["binary_sha256"],
            "binary_hash_path": f"build/{case['binary']}.binary.sha256",
            "sass_sha256": artifact["sass_sha256"],
            "sass_path": f"build/{case['binary']}.sass.txt",
            "sass_evidence": sass_evidence[str(case["id"])],
        }
        if artifact["self_test_path"] is not None:
            base["self_test_path"] = str(
                Path(artifact["self_test_path"]).relative_to(run_dir))
            base["self_test_sha256"] = artifact["self_test_sha256"]
            base["self_test_command_path"] = str(
                Path(artifact["self_test_command_path"]).relative_to(run_dir))
            base["self_test_command_sha256"] = artifact[
                "self_test_command_sha256"]
        if args.static_only:
            results.append({**base, "status": "static_ok"})
            continue
        case_dir = run_dir / "cases" / str(case["id"])
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path, trials_path = case_dir / "result.json", case_dir / "trials.jsonl"
        if result_path.is_file() and trials_path.is_file():
            prior = json.loads(result_path.read_text())
            prior_trials = [line for line in trials_path.read_text().splitlines() if line]
            if (prior.get("status") == "ok" and prior.get("trial_count") == TRIALS
                    and len(prior_trials) == TRIALS
                    and prior.get("binary_sha256") == base["binary_sha256"]
                    and prior.get("sass_sha256") == base["sass_sha256"]):
                results.append(prior)
                print(f"SKIP {case['id']}: complete artifact match", flush=True)
                continue
        trial_rows, custom_rates, reference_rates = [], [], []
        for trial in range(1, TRIALS + 1):
            trial_dir = case_dir / f"trial_{trial:02d}"
            trial_dir.mkdir(exist_ok=True)
            command = [str(artifact["binary"]),
                       *[str(arg) for arg in case["args"]]]
            outcome = run_bounded(
                command, cwd=trial_dir,
                timeout_seconds=args.trial_timeout_seconds)
            (trial_dir / "stdout.log").write_text(str(outcome["stdout"]))
            if outcome["timed_out"]:
                (trial_dir / "timeout.json").write_text(
                    json.dumps(outcome, indent=2, sort_keys=True) + "\n")
                raise RuntimeError(
                    f"{case['id']} trial {trial} exceeded "
                    f"{args.trial_timeout_seconds}s "
                    f"(termination_failed={outcome['termination_failed']}); "
                    f"see {trial_dir / 'timeout.json'}")
            if outcome["returncode"]:
                raise RuntimeError(
                    f"{case['id']} trial {trial} failed; see {trial_dir / 'stdout.log'}")
            parsed = parse_trial(case, trial_dir, str(outcome["stdout"]))
            custom_rates.append(float(parsed["custom_rate_per_second"]))
            reference_rates.append(float(parsed["reference_rate_per_second"]))
            trial_rows.append({
                "trial": trial, "captured_at_utc": utc_now(),
                "returncode": outcome["returncode"],
                "timeout_seconds": args.trial_timeout_seconds,
                "timed_out": False, "termination_failed": False,
                **parsed,
            })
        trials_path.write_text("".join(
            json.dumps(row, sort_keys=True) + "\n" for row in trial_rows))
        result = {
            **base, "status": "ok", "trial_count": TRIALS,
            "custom_rate_per_second_median": statistics.median(custom_rates),
            "custom_rate_per_second_min": min(custom_rates),
            "custom_rate_per_second_max": max(custom_rates),
            "reference_rate_per_second_median": statistics.median(reference_rates),
            "ratio_of_paired_medians":
                statistics.median(custom_rates) / statistics.median(reference_rates),
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)

    ncu_results: list[dict[str, object]] = []
    if args.ncu and not args.static_only:
        write_status(run_dir, "running_ncu", current_case=None,
                     completed_cases=len(results), total_cases=len(CASES))
        ncu_results = collect_ncu(
            run_dir, artifacts, args.ncu_timeout_seconds)
    status = "static_complete" if args.static_only else "complete"
    summary = {
        "schema_version": 2, "run_id": args.run_id, "status": status,
        "case_count": len(results), "results": results,
        "ncu_requested": args.ncu, "ncu_results": ncu_results,
        "updated_at_utc": utc_now(),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_manifest = generate_campaign_plots(summary_path)
    if not args.static_only:
        complete_marker.write_text(
            f"run_id={args.run_id}\nsummary_sha256={sha256(summary_path)}\n")
        for artifact in artifacts.values():
            Path(artifact["binary"]).unlink()
    write_status(run_dir, status, current_case=None,
                 completed_cases=len(results), total_cases=len(CASES))
    print(json.dumps({"run_dir": str(run_dir), "status": status,
                      "case_count": len(results),
                      "ncu_report_count": len(ncu_results),
                      "plot_count": plot_manifest["chart_count"],
                      "plots": str(run_dir / "plots")}, indent=2))
    return 0


def mark_failed_from_argv(message: str) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id")
    args, _ = parser.parse_known_args()
    if not args.run_id or not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        return
    run_dir = RESULT_ROOT / args.run_id
    if run_dir.is_dir():
        write_status(run_dir, "failed", error=message)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        mark_failed_from_argv(str(exc))
        raise
