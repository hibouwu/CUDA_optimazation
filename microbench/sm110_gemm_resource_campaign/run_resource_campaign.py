#!/usr/bin/env python3
"""Collect exact A/B TMA ingress contracts for Thor/SM110 GEMM schedules."""

from __future__ import annotations

import argparse
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
from typing import Any


REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = Path(__file__).resolve().parent
MANIFEST_PATH = CAMPAIGN / "contract_manifest.json"
SOURCE_PATH = (
    REPO / "microbench/15_tma_ab_contract_bandwidth/"
           "tma_ab_contract_bandwidth.cu"
)
RESULT_ROOT = REPO / "results/sm110_gemm_resource_campaign"
EXPECTED_TRIALS = 10
DEFAULT_TRIAL_TIMEOUT_SECONDS = 120
DEFAULT_NCU_TIMEOUT_SECONDS = 300
TERMINATION_GRACE_SECONDS = 5
SOURCE_DEPENDENCIES = (
    "microbench/15_tma_ab_contract_bandwidth/"
    "tma_ab_contract_bandwidth.cu",
    "microbench/sm110_gemm_resource_campaign/contract_manifest.json",
    "microbench/sm110_gemm_resource_campaign/run_resource_campaign.py",
)
EXPECTED_PRECISIONS = {
    "fp16_f32", "bf16_f32", "tf32_f32", "e4m3_f32", "e5m2_f32",
    "e3m2_f32", "e2m3_f32", "e2m1_f32", "mxfp4_f32",
    "nvfp4_f32", "s8_s32", "u8_s32",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required executable not found: {name}")
    return path


def run_capture(
    command: list[str], *, cwd: Path = REPO, check: bool = False,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): "
            f"{' '.join(command)}\n{completed.stdout}"
        )
    return completed


def run_bounded(command: list[str], timeout_seconds: int) -> dict[str, object]:
    process = subprocess.Popen(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    termination_failed = False
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as initial_timeout:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(
                timeout=TERMINATION_GRACE_SECONDS
            )
        except subprocess.TimeoutExpired as term_timeout:
            os.killpg(process.pid, signal.SIGKILL)
            try:
                output, _ = process.communicate(
                    timeout=TERMINATION_GRACE_SECONDS
                )
            except subprocess.TimeoutExpired as kill_timeout:
                termination_failed = True
                output = (
                    kill_timeout.stdout
                    or term_timeout.stdout
                    or initial_timeout.stdout
                    or ""
                )
                if isinstance(output, bytes):
                    output = output.decode(errors="backslashreplace")
    return {
        "command": command,
        "returncode": process.returncode,
        "stdout": output,
        "timed_out": timed_out,
        "termination_failed": termination_failed,
        "timeout_seconds": timeout_seconds,
    }


def write_status(run_dir: Path, status: str, **extra: object) -> None:
    row = {
        "status": status,
        "pid": os.getpid(),
        "hostname": platform.node(),
        "updated_at_utc": utc_now(),
        **extra,
    }
    (run_dir / "campaign_status.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )
    with (run_dir / "progress.jsonl").open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_artifact_manifest(run_dir: Path) -> None:
    excluded = {"artifact_sha256.txt", "launcher.log", "launcher.pid"}
    paths = sorted(
        path for path in run_dir.rglob("*")
        if path.is_file() and path.relative_to(run_dir).as_posix() not in excluded
    )
    (run_dir / "artifact_sha256.txt").write_text("".join(
        f"{sha256_path(path)}  {path.relative_to(run_dir).as_posix()}\n"
        for path in paths
    ))


def parse_kv(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        for match in re.finditer(
            r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", line
        ):
            fields[match.group(1)] = match.group(2)
    return fields


def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def round_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("resource contract manifest schema must be 1")
    if manifest.get("expected_sm_count") != 20:
        raise RuntimeError("resource contract manifest must target 20 SMs")
    if manifest.get("row_stride_elements") != [1024, 2048, 4096]:
        raise RuntimeError("row-stride matrix must be exactly 1024/2048/4096")
    families = manifest.get("families")
    if not isinstance(families, list) or not families:
        raise RuntimeError("resource contract manifest has no families")
    ids: set[str] = set()
    precision_ids: set[str] = set()
    for family in families:
        if not isinstance(family, dict):
            raise RuntimeError("resource family must be an object")
        family_id = str(family.get("family_id", ""))
        if not re.fullmatch(r"[a-z0-9_]+", family_id):
            raise RuntimeError(f"invalid resource family ID: {family_id}")
        if family_id in ids:
            raise RuntimeError(f"duplicate resource family ID: {family_id}")
        ids.add(family_id)
        for field in (
            "bm", "bn", "bk", "value_bits", "stages", "threads"
        ):
            value = family.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise RuntimeError(f"{family_id}: invalid {field}")
        scale_block = family.get("scale_block")
        if scale_block not in {0, 16, 32}:
            raise RuntimeError(f"{family_id}: invalid scale_block")
        controller = family.get("controller_thread")
        if (
            not isinstance(controller, int)
            or isinstance(controller, bool)
            or not 0 <= controller < int(family["threads"])
        ):
            raise RuntimeError(f"{family_id}: invalid controller thread")
        family_precisions = family.get("precision_ids")
        if not isinstance(family_precisions, list) or not family_precisions:
            raise RuntimeError(f"{family_id}: precision_ids must be nonempty")
        precision_ids.update(str(value) for value in family_precisions)
        layouts = family.get("input_transport_layouts")
        if not isinstance(layouts, list) or not layouts:
            raise RuntimeError(
                f"{family_id}: input_transport_layouts must be nonempty"
            )
    if precision_ids != EXPECTED_PRECISIONS:
        raise RuntimeError(
            "resource families do not cover exactly the 12 precision IDs: "
            f"missing={sorted(EXPECTED_PRECISIONS - precision_ids)} "
            f"extra={sorted(precision_ids - EXPECTED_PRECISIONS)}"
        )
    residencies = manifest.get("residencies")
    if not isinstance(residencies, dict) or set(residencies) != {
        "hot_l2", "cold_dram"
    }:
        raise RuntimeError("manifest must define hot_l2 and cold_dram")
    hot = residencies["hot_l2"]
    cold = residencies["cold_dram"]
    if not (
        isinstance(hot, dict)
        and hot.get("binary_mode") == "l2-hit"
        and hot.get("blocks") == 1
        and hot.get("target_working_set_bytes") == 16 << 20
        and hot.get("iters") == 4096
        and hot.get("warmup_policy") == "cover_working_set"
        and isinstance(cold, dict)
        and cold.get("binary_mode") == "dram-stream"
        and cold.get("blocks") == "one_per_sm"
        and cold.get("target_working_set_bytes") == 64 << 20
        and cold.get("target_requested_bytes_per_trial") == 512 << 20
        and cold.get("warmup_iters") == 64
    ):
        raise RuntimeError("residency contracts do not match the formal campaign")
    return manifest


def family_contract(family: dict[str, Any]) -> dict[str, int | str]:
    value_bits = int(family["value_bits"])
    bk = int(family["bk"])
    row_bytes = bk * value_bits // 8
    if bk * value_bits % 8:
        raise RuntimeError(f"{family['family_id']}: fractional stage row")
    request_row_bytes = min(row_bytes, 128)
    if request_row_bytes not in {32, 64, 128}:
        raise RuntimeError(
            f"{family['family_id']}: unsupported request row bytes"
        )
    if row_bytes % request_row_bytes:
        raise RuntimeError(f"{family['family_id']}: uneven value chunks")
    value_chunks = row_bytes // request_row_bytes
    a_value_bytes = int(family["bm"]) * row_bytes
    b_value_bytes = int(family["bn"]) * row_bytes
    scale_block = int(family["scale_block"])
    a_scale_bytes = 0
    b_scale_bytes = 0
    scale_groups_padded = 0
    if scale_block:
        scale_groups_padded = round_up(ceil_div(bk, scale_block), 4)
        a_scale_bytes = round_up(int(family["bm"]), 128) * scale_groups_padded
        b_scale_bytes = round_up(int(family["bn"]), 128) * scale_groups_padded
    stage_bytes = (
        a_value_bytes + b_value_bytes + a_scale_bytes + b_scale_bytes
    )
    return {
        "value_chunks": value_chunks,
        "value_swizzle": f"{request_row_bytes}B",
        "requests_per_stage": 2 * value_chunks + (2 if scale_block else 0),
        "a_value_bytes": a_value_bytes,
        "b_value_bytes": b_value_bytes,
        "a_scale_bytes": a_scale_bytes,
        "b_scale_bytes": b_scale_bytes,
        "scale_groups_padded": scale_groups_padded,
        "stage_bytes": stage_bytes,
        "dynamic_smem_bytes": int(family["stages"]) * stage_bytes,
    }


def allocation_bytes(
    family: dict[str, Any], row_stride: int, total_tiles: int,
) -> int:
    value_bits = int(family["value_bits"])
    value_stride_bytes = row_stride * value_bits // 8
    total = value_stride_bytes * (
        int(family["bm"]) + int(family["bn"])
    ) * total_tiles
    if int(family["scale_block"]):
        contract = family_contract(family)
        total += (
            int(contract["a_scale_bytes"])
            + int(contract["b_scale_bytes"])
        ) * total_tiles
    return total


def make_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    expected_sms = int(manifest["expected_sm_count"])
    for family in manifest["families"]:
        contract = family_contract(family)
        stage_bytes = int(contract["stage_bytes"])
        for row_stride in manifest["row_stride_elements"]:
            for residency in ("hot_l2", "cold_dram"):
                residency_spec = manifest["residencies"][residency]
                working_set_target = int(
                    residency_spec["target_working_set_bytes"]
                )
                total_tiles = max(1, working_set_target // stage_bytes)
                working_set_bytes = total_tiles * stage_bytes
                if residency == "hot_l2":
                    blocks = 1
                    iters = int(residency_spec["iters"])
                    warmup_iters = total_tiles + int(family["stages"])
                    resource = (
                        "tma.smem_ingress.contract."
                        f"{family['family_id']}.stride{row_stride}.per_sm"
                    )
                else:
                    blocks = expected_sms
                    requested_target = int(
                        residency_spec["target_requested_bytes_per_trial"]
                    )
                    iters = max(
                        256,
                        ceil_div(requested_target, blocks * stage_bytes),
                    )
                    warmup_iters = int(residency_spec["warmup_iters"])
                    resource = (
                        "tma.hbm.contract."
                        f"{family['family_id']}.stride{row_stride}"
                    )
                case_id = (
                    f"{family['family_id']}_stride{row_stride}_{residency}"
                )
                args = [
                    "--case-id", case_id,
                    "--mode", str(residency_spec["binary_mode"]),
                    "--bm", str(family["bm"]),
                    "--bn", str(family["bn"]),
                    "--bk", str(family["bk"]),
                    "--value-bits", str(family["value_bits"]),
                    "--scale-block", str(family["scale_block"]),
                    "--stages", str(family["stages"]),
                    "--row-stride-elements", str(row_stride),
                    "--threads", str(family["threads"]),
                    "--controller-thread", str(family["controller_thread"]),
                    "--bytes", str(working_set_target),
                    "--iters", str(iters),
                    "--warmup-iters", str(warmup_iters),
                    "--expected-sm-count", str(expected_sms),
                ]
                if residency == "hot_l2":
                    args += ["--blocks", "1"]
                else:
                    args += ["--blocks-per-sm", "1"]
                expected = {
                    "case_id": case_id,
                    "mode": residency_spec["binary_mode"],
                    "bm": int(family["bm"]),
                    "bn": int(family["bn"]),
                    "bk": int(family["bk"]),
                    "value_bits": int(family["value_bits"]),
                    "scale_block": int(family["scale_block"]),
                    "stages": int(family["stages"]),
                    "requests_per_stage": int(contract["requests_per_stage"]),
                    "value_chunks": int(contract["value_chunks"]),
                    "value_swizzle": str(contract["value_swizzle"]),
                    "row_stride_elements": int(row_stride),
                    "threads": int(family["threads"]),
                    "controller_thread": int(family["controller_thread"]),
                    "blocks": blocks,
                    "sm_count": expected_sms,
                    "initialization": "cuda_memset_zero",
                    "a_value_bytes": int(contract["a_value_bytes"]),
                    "b_value_bytes": int(contract["b_value_bytes"]),
                    "a_scale_bytes": int(contract["a_scale_bytes"]),
                    "b_scale_bytes": int(contract["b_scale_bytes"]),
                    "stage_bytes": stage_bytes,
                    "dynamic_smem_bytes": int(contract["dynamic_smem_bytes"]),
                    "working_set_bytes": working_set_bytes,
                    "allocation_bytes": allocation_bytes(
                        family, int(row_stride), total_tiles
                    ),
                    "total_tiles": total_tiles,
                    "iters": iters,
                    "warmup_iters": warmup_iters,
                }
                cases.append({
                    "case_id": case_id,
                    "family_id": family["family_id"],
                    "precision_ids": family["precision_ids"],
                    "input_transport_layouts":
                        family["input_transport_layouts"],
                    "row_stride_elements": row_stride,
                    "residency": residency,
                    "resource": resource,
                    "args": args,
                    "expected": expected,
                    "ncu_selected": row_stride == 2048,
                })
    ids = [str(case["case_id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("generated resource case IDs are not unique")
    return cases


def environment() -> dict[str, object]:
    commands = {
        "gpu_identity": [
            "nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
            "--format=csv,noheader",
        ],
        "gpu_state": [
            "nvidia-smi",
            "--query-gpu=pstate,clocks.current.graphics,power.limit,temperature.gpu",
            "--format=csv,noheader",
        ],
        "nvcc": ["nvcc", "--version"],
        "ncu": ["ncu", "--version"],
        "git_head": ["git", "rev-parse", "HEAD"],
        "git_branch": ["git", "branch", "--show-current"],
        "git_status": ["git", "status", "--short"],
    }
    snapshot: dict[str, object] = {
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    for name, command in commands.items():
        result = run_capture(command)
        snapshot[name] = {
            "returncode": result.returncode,
            "output": result.stdout,
        }
    nvpmodel = shutil.which("nvpmodel")
    if nvpmodel:
        result = run_capture([nvpmodel, "-q"])
        snapshot["power_mode"] = {
            "returncode": result.returncode,
            "output": result.stdout,
        }
    else:
        snapshot["power_mode"] = {
            "returncode": 127,
            "output": "nvpmodel not found",
        }
    return snapshot


def validate_environment(
    snapshot: dict[str, object], *, require_ncu: bool = False,
) -> None:
    required = [
        "gpu_identity", "gpu_state", "nvcc", "git_head", "git_branch",
        "git_status", "power_mode",
    ]
    if require_ncu:
        required.append("ncu")
    for name in required:
        row = snapshot.get(name)
        if not isinstance(row, dict) or row.get("returncode") != 0:
            raise RuntimeError(f"environment probe failed: {name}")
    if "11.0" not in str(snapshot["gpu_identity"].get("output", "")):
        raise RuntimeError("resource campaign requires compute capability 11.0")
    if "MAXN" not in str(snapshot["power_mode"].get("output", "")).upper():
        raise RuntimeError("MAXN power mode is not proven")
    if str(snapshot["git_status"].get("output", "")).strip():
        raise RuntimeError("resource campaign requires a clean worktree")


def compile_binary(
    run_dir: Path, host_compiler: str | None, undef_gnu_source: bool,
) -> dict[str, object]:
    build = run_dir / "build"
    build.mkdir(parents=True, exist_ok=True)
    binary = build / "tma_ab_contract_bandwidth"
    command = [tool("nvcc"), "-O3", "-std=c++17"]
    if host_compiler:
        command += ["-ccbin", host_compiler]
    if undef_gnu_source:
        command += [
            "-Xcompiler=-U_GNU_SOURCE",
            "-D_DEFAULT_SOURCE",
            "-D_POSIX_C_SOURCE=200809L",
            "-D_XOPEN_SOURCE=700",
            "-D_XOPEN_SOURCE_EXTENDED=1",
            "-D_LARGEFILE64_SOURCE=1",
            "-D_ATFILE_SOURCE=1",
        ]
    command += [
        "-gencode", "arch=compute_110a,code=sm_110a",
        str(SOURCE_PATH), "-lcuda", "-o", str(binary),
    ]
    (build / "compile_command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    compiled = run_capture(command)
    (build / "compile.log").write_text(compiled.stdout)
    if compiled.returncode:
        raise RuntimeError(
            f"resource microbenchmark compile failed; see {build / 'compile.log'}"
        )
    sass = run_capture([tool("cuobjdump"), "--dump-sass", str(binary)])
    sass_path = build / "tma_ab_contract_bandwidth.sass.txt"
    sass_path.write_text(sass.stdout)
    if sass.returncode or "tma_ab_contract_kernel" not in sass.stdout:
        raise RuntimeError("resource microbenchmark SASS function is missing")
    if "UTMALDG.2D" not in sass.stdout:
        raise RuntimeError("resource microbenchmark lacks UTMALDG.2D")
    artifact = {
        "binary": binary,
        "binary_sha256": sha256_path(binary),
        "source_sha256": sha256_path(SOURCE_PATH),
        "sass_path": sass_path,
        "sass_sha256": sha256_path(sass_path),
        "compile_command_sha256": sha256_path(
            build / "compile_command.json"
        ),
    }
    (build / "binary.sha256").write_text(
        f"{artifact['binary_sha256']}  {binary.name}\n"
    )
    return artifact


def assert_fields(
    case: dict[str, Any], fields: dict[str, str], *, contract_only: bool,
) -> float | None:
    expected = case["expected"]
    mismatches: list[str] = []
    string_fields = {
        "case_id", "mode", "value_swizzle", "initialization",
    }
    for name, value in expected.items():
        actual = fields.get(name)
        if name in string_fields:
            if actual != str(value):
                mismatches.append(
                    f"{name}:expected={value}:actual={actual}"
                )
        elif actual is None or int(actual) != int(value):
            mismatches.append(
                f"{name}:expected={value}:actual={actual}"
            )
    if contract_only and fields.get("contract_only") != "1":
        mismatches.append("contract_only:expected=1")
    if mismatches:
        raise RuntimeError(
            f"{case['case_id']}: contract mismatch: " + "; ".join(mismatches)
        )
    if contract_only:
        return None
    expected_unique_sms = 1 if case["residency"] == "hot_l2" else 20
    if int(fields.get("unique_smid_count", "0")) != expected_unique_sms:
        raise RuntimeError(f"{case['case_id']}: SM coverage mismatch")
    elapsed_ns = int(fields.get("globaltimer_elapsed_ns", "0"))
    requested = int(fields.get("requested_bytes", "0"))
    reported = float(fields.get("bytes_per_second", "nan"))
    if elapsed_ns <= 0 or requested <= 0 or not math.isfinite(reported):
        raise RuntimeError(f"{case['case_id']}: invalid timing fields")
    recalculated = requested * 1.0e9 / elapsed_ns
    if abs(reported - recalculated) > max(1.0, recalculated * 2e-6):
        raise RuntimeError(f"{case['case_id']}: rate arithmetic mismatch")
    if int(fields.get("occupancy_blocks_per_sm", "0")) <= 0:
        raise RuntimeError(f"{case['case_id']}: occupancy is not positive")
    return recalculated


def static_contract_audit(
    run_dir: Path, binary: Path, cases: list[dict[str, Any]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        command = [str(binary), "--contract-only", *case["args"]]
        completed = run_capture(command)
        if completed.returncode:
            raise RuntimeError(
                f"contract-only failed for {case['case_id']}: "
                f"{completed.stdout}"
            )
        fields = parse_kv(completed.stdout)
        assert_fields(case, fields, contract_only=True)
        rows.append({
            "case_id": case["case_id"],
            "command": command,
            "fields": fields,
        })
    (run_dir / "static_contracts.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    return rows


def ncu_args(case: dict[str, Any]) -> list[str]:
    args = list(case["args"])
    ncu_working_set = (
        (1 << 20) if case["residency"] == "hot_l2" else (64 << 20)
    )
    ncu_tiles = max(
        1, ncu_working_set // int(case["expected"]["stage_bytes"])
    )
    ncu_warmup = (
        ncu_tiles + int(case["expected"]["stages"])
        if case["residency"] == "hot_l2" else 64
    )
    ncu_iters = (
        128
        if case["residency"] == "hot_l2"
        else max(
            128,
            ceil_div(
                128 << 20,
                int(case["expected"]["blocks"])
                * int(case["expected"]["stage_bytes"]),
            ),
        )
    )
    replacements = {
        "--bytes": str(ncu_working_set),
        "--iters": str(ncu_iters),
        "--warmup-iters": str(ncu_warmup),
    }
    for name, value in replacements.items():
        index = args.index(name)
        args[index + 1] = value
    return args


def collect_ncu(
    case_dir: Path, binary: Path, case: dict[str, Any], timeout_seconds: int,
) -> dict[str, object]:
    ncu_dir = case_dir / "ncu"
    ncu_dir.mkdir(exist_ok=True)
    report_base = ncu_dir / "profile"
    command = [
        tool("ncu"),
        "--set", "basic",
        "--target-processes", "all",
        "--kernel-name-base", "demangled",
        "--kernel-name", "regex:tma_ab_contract_kernel",
        "--launch-count", "1",
        "--force-overwrite",
        "--export", str(report_base),
        str(binary),
        *ncu_args(case),
    ]
    outcome = run_bounded(command, timeout_seconds)
    log_path = ncu_dir / "profile.log"
    log_path.write_text(str(outcome["stdout"]))
    if outcome["timed_out"]:
        (ncu_dir / "timeout.json").write_text(
            json.dumps(outcome, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(f"NCU timed out for {case['case_id']}")
    report = ncu_dir / "profile.ncu-rep"
    permission_denied = "ERR_NVGPUCTRPERM" in str(outcome["stdout"])
    passed = (
        outcome["returncode"] == 0
        and not permission_denied
        and report.is_file()
    )
    result = {
        "selected": True,
        "policy": "row_stride=2048 for every family and residency",
        "set": "basic",
        "kernel_name_base": "demangled",
        "kernel_name_regex": "tma_ab_contract_kernel",
        "launch_count": 1,
        "command": command,
        "returncode": outcome["returncode"],
        "permission_denied": permission_denied,
        "timed_out": False,
        "termination_failed": False,
        "timeout_seconds": timeout_seconds,
        "report_path": "ncu/profile.ncu-rep",
        "report_sha256": sha256_path(report) if report.is_file() else None,
        "log_sha256": sha256_path(log_path),
        "pass": passed,
    }
    if not passed:
        raise RuntimeError(f"NCU failed for {case['case_id']}")
    return result


def prior_result_is_reusable(
    case_dir: Path, case: dict[str, Any], fingerprint: str, require_ncu: bool,
) -> dict[str, Any] | None:
    result_path = case_dir / "result.json"
    trials_path = case_dir / "trials.jsonl"
    if not result_path.is_file() or not trials_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text())
        trials = [
            json.loads(line)
            for line in trials_path.read_text().splitlines() if line
        ]
    except json.JSONDecodeError:
        return None
    if not (
        result.get("status") == "ok"
        and result.get("fingerprint") == fingerprint
        and result.get("trial_count") == EXPECTED_TRIALS
        and len(trials) == EXPECTED_TRIALS
        and result.get("trial_timeout_seconds")
        == DEFAULT_TRIAL_TIMEOUT_SECONDS
    ):
        return None
    binary = case_dir.parents[1] / "build/tma_ab_contract_bandwidth"
    rates: list[float] = []
    for trial_index, trial in enumerate(trials, 1):
        if not isinstance(trial, dict) or not (
            trial.get("trial") == trial_index
            and trial.get("command") == [str(binary), *case["args"]]
            and trial.get("returncode") == 0
            and trial.get("timeout_seconds") == DEFAULT_TRIAL_TIMEOUT_SECONDS
            and trial.get("timed_out") is False
            and trial.get("termination_failed") is False
            and isinstance(trial.get("raw_stdout"), str)
            and parse_kv(str(trial.get("raw_stdout"))) == trial.get("fields")
        ):
            return None
        try:
            rate = assert_fields(case, trial["fields"], contract_only=False)
            recorded_rate = float(trial["audited_rate_per_second"])
        except (KeyError, TypeError, ValueError, RuntimeError):
            return None
        if rate is None or not math.isclose(
            rate, recorded_rate, rel_tol=2e-12
        ):
            return None
        rates.append(rate)
    expected_stats = {
        "rate_per_second_median": statistics.median(rates),
        "rate_per_second_min": min(rates),
        "rate_per_second_max": max(rates),
        "rate_per_second_mean": statistics.fmean(rates),
    }
    for name, value in expected_stats.items():
        try:
            actual = float(result[name])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isclose(actual, value, rel_tol=2e-12):
            return None
    if require_ncu and not result.get("ncu", {}).get("pass"):
        return None
    if require_ncu:
        ncu = result["ncu"]
        report = case_dir / str(ncu.get("report_path", ""))
        log = case_dir / "ncu/profile.log"
        expected_command = [
            tool("ncu"),
            "--set", "basic",
            "--target-processes", "all",
            "--kernel-name-base", "demangled",
            "--kernel-name", "regex:tma_ab_contract_kernel",
            "--launch-count", "1",
            "--force-overwrite",
            "--export", str(case_dir / "ncu/profile"),
            str(binary),
            *ncu_args(case),
        ]
        if not (
            ncu.get("selected") is True
            and ncu.get("command") == expected_command
            and ncu.get("returncode") == 0
            and ncu.get("permission_denied") is False
            and ncu.get("timed_out") is False
            and ncu.get("termination_failed") is False
            and ncu.get("timeout_seconds") == DEFAULT_NCU_TIMEOUT_SECONDS
            and report.is_file() and log.is_file()
            and sha256_path(report) == ncu.get("report_sha256")
            and sha256_path(log) == ncu.get("log_sha256")
        ):
            return None
    return result


def run_case(
    run_dir: Path,
    binary: Path,
    artifact: dict[str, object],
    case: dict[str, Any],
    *,
    trial_timeout_seconds: int,
    collect_ncu_flag: bool,
    ncu_timeout_seconds: int,
) -> dict[str, Any]:
    case_dir = run_dir / "cases" / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256_json({
        "case": case,
        "binary_sha256": artifact["binary_sha256"],
        "source_sha256": artifact["source_sha256"],
        "trial_count": EXPECTED_TRIALS,
        "ncu": collect_ncu_flag and bool(case["ncu_selected"]),
    })
    reusable = prior_result_is_reusable(
        case_dir,
        case,
        fingerprint,
        collect_ncu_flag and bool(case["ncu_selected"]),
    )
    if reusable is not None:
        print(f"SKIP {case['case_id']}: complete fingerprint", flush=True)
        return reusable

    rows: list[dict[str, object]] = []
    rates: list[float] = []
    for trial in range(1, EXPECTED_TRIALS + 1):
        command = [str(binary), *case["args"]]
        outcome = run_bounded(command, trial_timeout_seconds)
        if outcome["timed_out"]:
            timeout = {
                "case_id": case["case_id"],
                "trial": trial,
                "captured_at_utc": utc_now(),
                **outcome,
            }
            (case_dir / "timeout.json").write_text(
                json.dumps(timeout, indent=2, sort_keys=True) + "\n"
            )
            raise RuntimeError(
                f"{case['case_id']} trial {trial} timed out"
            )
        if outcome["returncode"]:
            raise RuntimeError(
                f"{case['case_id']} trial {trial} failed: "
                f"{outcome['stdout']}"
            )
        fields = parse_kv(str(outcome["stdout"]))
        rate = assert_fields(case, fields, contract_only=False)
        assert rate is not None
        rates.append(rate)
        rows.append({
            "trial": trial,
            "captured_at_utc": utc_now(),
            "command": command,
            "returncode": outcome["returncode"],
            "timeout_seconds": trial_timeout_seconds,
            "timed_out": False,
            "termination_failed": False,
            "raw_stdout": outcome["stdout"],
            "fields": fields,
            "audited_rate_per_second": rate,
        })
    trials_path = case_dir / "trials.jsonl"
    trials_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "resource": case["resource"],
        "residency": case["residency"],
        "row_stride_elements": case["row_stride_elements"],
        "precision_ids": case["precision_ids"],
        "input_transport_layouts": case["input_transport_layouts"],
        "status": "ok",
        "fingerprint": fingerprint,
        "trial_count": len(rates),
        "rate_unit": "B/s",
        "rate_per_second_median": statistics.median(rates),
        "rate_per_second_min": min(rates),
        "rate_per_second_max": max(rates),
        "rate_per_second_mean": statistics.fmean(rates),
        "expected_contract": case["expected"],
        "source_path": str(SOURCE_PATH.relative_to(REPO)),
        "source_sha256": artifact["source_sha256"],
        "binary_sha256": artifact["binary_sha256"],
        "sass_path": "build/tma_ab_contract_bandwidth.sass.txt",
        "sass_sha256": artifact["sass_sha256"],
        "sass_tokens": ["UTMALDG.2D"],
        "completed_at_utc": utc_now(),
        "trial_timeout_seconds": trial_timeout_seconds,
    }
    if collect_ncu_flag and bool(case["ncu_selected"]):
        result["ncu"] = collect_ncu(
            case_dir, binary, case, ncu_timeout_seconds
        )
    else:
        result["ncu"] = {
            "selected": False,
            "policy": "row_stride=2048 for every family and residency",
        }
    (case_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"PASS {case['case_id']}: median={statistics.median(rates):.9e} B/s",
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument("--host-compiler")
    parser.add_argument("--nvcc-host-undef-gnu-source", action="store_true")
    parser.add_argument(
        "--trial-timeout-seconds",
        type=int,
        default=DEFAULT_TRIAL_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--ncu-timeout-seconds",
        type=int,
        default=DEFAULT_NCU_TIMEOUT_SECONDS,
    )
    parser.add_argument("--output-root", type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        parser.error("invalid run-id")
    if args.trial_timeout_seconds <= 0 or args.ncu_timeout_seconds <= 0:
        parser.error("timeouts must be positive")
    manifest = load_manifest()
    cases = make_cases(manifest)
    if args.plan:
        print(json.dumps({
            "schema_version": 1,
            "run_id": args.run_id,
            "case_count": len(cases),
            "family_count": len(manifest["families"]),
            "ncu_case_count": sum(
                bool(case["ncu_selected"]) for case in cases
            ),
            "cases": cases,
        }, indent=2, sort_keys=True))
        return 0
    if not args.static_only and (
        args.host_compiler is not None
        or args.nvcc_host_undef_gnu_source
        or args.output_root is not None
        or args.trial_timeout_seconds != DEFAULT_TRIAL_TIMEOUT_SECONDS
        or args.ncu_timeout_seconds != DEFAULT_NCU_TIMEOUT_SECONDS
    ):
        parser.error(
            "hardware evidence requires the default Thor compile command and "
            "canonical result root plus 120/300-second timeouts; "
            "compatibility/output/timeout overrides are static-only"
        )
    if args.static_only and args.ncu:
        parser.error("--ncu cannot be combined with --static-only")

    tool("nvcc")
    tool("cuobjdump")
    if not args.static_only:
        tool("nvidia-smi")
    if args.ncu:
        tool("ncu")
    output_root = (
        args.output_root.resolve()
        if args.output_root else RESULT_ROOT
    )
    run_dir = output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    global_lock = (REPO / "results" / ".sm110_gpu_campaign.lock").open("w")
    try:
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            "another SM110 GPU campaign holds the global lock"
        ) from error

    spec = {
        "schema_version": 1,
        "run_id": args.run_id,
        "campaign": "sm110_exact_tma_resource_contracts",
        "expected_sm_count": manifest["expected_sm_count"],
        "trials": EXPECTED_TRIALS,
        "case_count": len(cases),
        "family_count": len(manifest["families"]),
        "static_only": args.static_only,
        "ncu_requested": args.ncu,
        "ncu_policy": "row_stride=2048 for every family and residency",
        "trial_timeout_seconds": args.trial_timeout_seconds,
        "ncu_timeout_seconds": args.ncu_timeout_seconds,
        "termination_grace_seconds": TERMINATION_GRACE_SECONDS,
        "timing": (
            "single-CTA span for hot-L2; earliest CTA globaltimer start "
            "to latest CTA stop for one-CTA-per-SM cold-DRAM"
        ),
        "generator": str(Path(__file__).relative_to(REPO)),
        "generator_sha256": sha256_path(Path(__file__)),
        "contract_manifest": str(MANIFEST_PATH.relative_to(REPO)),
        "contract_manifest_sha256": sha256_path(MANIFEST_PATH),
        "source_dependencies": {
            path: sha256_path(REPO / path) for path in SOURCE_DEPENDENCIES
        },
        "cases": cases,
    }
    spec_text = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    spec_path = run_dir / "run_spec.json"
    if spec_path.exists() and spec_path.read_text() != spec_text:
        raise RuntimeError("run-id exists with a different run contract")
    spec_path.write_text(spec_text)
    complete_marker = run_dir / "COMPLETE"
    if complete_marker.exists():
        complete_marker.unlink()
    write_status(
        run_dir, "running", current_case=None, completed_cases=0,
        total_cases=len(cases),
    )

    if not args.static_only:
        snapshot = environment()
        validate_environment(snapshot, require_ncu=args.ncu)
        environment_path = run_dir / "environment.json"
        if not environment_path.exists():
            environment_path.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
            )
        with (run_dir / "environment_snapshots.jsonl").open("a") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")

    artifact = compile_binary(
        run_dir, args.host_compiler, args.nvcc_host_undef_gnu_source
    )
    static_rows = static_contract_audit(
        run_dir, Path(artifact["binary"]), cases
    )
    if args.static_only:
        summary = {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": "static_complete",
            "case_count": len(cases),
            "static_contract_count": len(static_rows),
            "results": [],
            "updated_at_utc": utc_now(),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        write_status(
            run_dir, "static_complete", current_case=None,
            completed_cases=len(cases), total_cases=len(cases),
        )
        write_artifact_manifest(run_dir)
        print(json.dumps({
            "run_dir": str(run_dir),
            "status": "static_complete",
            "case_count": len(cases),
        }, indent=2))
        return 0

    results: list[dict[str, Any]] = []
    for case in cases:
        write_status(
            run_dir, "running", current_case=case["case_id"],
            completed_cases=len(results), total_cases=len(cases),
        )
        result = run_case(
            run_dir,
            Path(artifact["binary"]),
            artifact,
            case,
            trial_timeout_seconds=args.trial_timeout_seconds,
            collect_ncu_flag=args.ncu,
            ncu_timeout_seconds=args.ncu_timeout_seconds,
        )
        results.append(result)
    if len(results) != len(cases) or any(
        result.get("status") != "ok" for result in results
    ):
        raise RuntimeError("resource campaign result matrix is incomplete")
    if args.ncu:
        selected = [
            result for result in results
            if result.get("ncu", {}).get("selected") is True
        ]
        if len(selected) != 18 or any(
            result.get("ncu", {}).get("pass") is not True
            for result in selected
        ):
            raise RuntimeError("resource campaign NCU matrix is incomplete")
    summary_path = run_dir / "summary.json"
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "complete",
        "case_count": len(results),
        "family_count": len(manifest["families"]),
        "ncu_requested": args.ncu,
        "ncu_case_count": sum(
            result.get("ncu", {}).get("selected") is True
            for result in results
        ),
        "results": results,
        "updated_at_utc": utc_now(),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    complete_marker.write_text(
        f"run_id={args.run_id}\nsummary_sha256={sha256_path(summary_path)}\n"
    )
    write_status(
        run_dir, "complete", current_case=None,
        completed_cases=len(results), total_cases=len(cases),
    )
    write_artifact_manifest(run_dir)
    print(json.dumps({
        "run_dir": str(run_dir),
        "status": "complete",
        "case_count": len(results),
        "ncu_case_count": summary["ncu_case_count"],
    }, indent=2))
    return 0


def mark_failed(message: str) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    args, _ = parser.parse_known_args()
    if not args.run_id or not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        return
    root = args.output_root.resolve() if args.output_root else RESULT_ROOT
    run_dir = root / args.run_id
    if run_dir.is_dir():
        write_status(run_dir, "failed", error=message)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        mark_failed(str(error))
        print(f"ERROR: {error}", file=sys.stderr)
        raise
