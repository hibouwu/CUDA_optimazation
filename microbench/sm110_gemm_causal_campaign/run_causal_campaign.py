#!/usr/bin/env python3
"""Bounded, resumable SM110 tc5a causal-pipeline evidence campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE_PATH = REPO / "microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu"
HELPER_PATH = REPO / "GEMMsm110/include/sm110_ptx_helpers.cuh"
MANIFEST_PATH = HERE / "contract_manifest.json"
DEFAULT_OUTPUT_ROOT = REPO / "results/sm110_gemm_causal_campaign"
DEFAULT_TRIAL_TIMEOUT_SECONDS = 120
DEFAULT_NCU_TIMEOUT_SECONDS = 300
EXPECTED_TRIALS = 10
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
REQUIRED_SASS_TOKENS = ("UTMALDG.2D", "UTCHMMA", "UTCBAR", "LDTM.")
CALIBRATION_METRICS = (
    "first_tma_latency_ns",
    "tma_completion_span_ns",
    "tma_interval_ns",
    "first_mma_latency_ns",
    "mma_completion_span_ns",
    "mma_interval_ns",
    "epilogue_to_store_ns",
    "last_mma_to_store_ns",
    "total_measured_ns",
)
CSV_FIELDS = (
    "case_id", "precision_id", "tensor_map_data_type",
    "instruction_descriptor_u32", "mode", "stages", "k_tiles", "output_tasks",
    "total_k_operations", "threads", "tma_requests_per_k_tile",
    "a_bytes_per_k_tile", "b_bytes_per_k_tile",
    "payload_bytes_per_k_tile", "mma_instructions_per_k_tile",
    "accumulator_buffers", "output_bytes_per_task",
    "dynamic_smem_bytes", "residency", "initialization",
    "warmup_launches", "sm_count", "smid", "start_ns",
    "first_tma_done_ns", "last_tma_done_ns", "first_mma_done_ns",
    "last_mma_done_ns", "first_epilogue_start_ns",
    "last_store_done_ns", "kernel_exit_ns", "first_tma_latency_ns",
    "tma_completion_span_ns", "tma_interval_ns",
    "first_mma_latency_ns", "mma_completion_span_ns",
    "mma_interval_ns", "epilogue_to_store_ns",
    "last_mma_to_store_ns", "total_measured_ns",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode())


def freeze_json(path: Path, value: object, label: str) -> None:
    """Create an immutable run contract, or prove the retained copy identical."""

    if path.is_file():
        try:
            retained = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"retained {label} is malformed") from error
        if retained != value:
            raise RuntimeError(f"retained {label} differs from the frozen contract")
        return
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required executable is missing: {name}")
    return path


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("schema_version") != 2:
        raise RuntimeError("unsupported causal manifest schema")
    if manifest.get("external_trials_per_case") != EXPECTED_TRIALS:
        raise RuntimeError("causal manifest trial count changed")
    if manifest.get("trial_timeout_seconds") != DEFAULT_TRIAL_TIMEOUT_SECONDS:
        raise RuntimeError("causal manifest trial timeout changed")
    if manifest.get("ncu_timeout_seconds") != DEFAULT_NCU_TIMEOUT_SECONDS:
        raise RuntimeError("causal manifest NCU timeout changed")
    if manifest.get("precision_contracts") != [
        {
            "precision_id": "fp16_f32", "input_type": "fp16",
            "tensor_map_data_type": "float16", "instruction_kind": "f16",
            "instruction_descriptor_u32": 138412048,
        },
        {
            "precision_id": "bf16_f32", "input_type": "bf16",
            "tensor_map_data_type": "bfloat16", "instruction_kind": "f16",
            "instruction_descriptor_u32": 138413200,
        },
    ]:
        raise RuntimeError(
            "tc5a causal manifest must bind independent FP16 and BF16 descriptors"
        )
    return manifest


def precision_ids(manifest: dict[str, Any]) -> list[str]:
    return [str(row["precision_id"]) for row in manifest["precision_contracts"]]


def make_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    k_values = [
        *manifest["calibration_k_tiles"], *manifest["holdout_k_tiles"]
    ]
    ncu_suffixes = set(manifest["ncu_case_suffixes"])
    cases: list[dict[str, Any]] = []
    for precision in manifest["precision_contracts"]:
        for family in manifest["families"]:
            for k_tiles in k_values:
                suffix = f"{family['family_id']}_k{k_tiles}"
                case_id = f"{precision['precision_id']}.{suffix}"
                case = {
                    "case_id": case_id,
                    "precision_id": precision["precision_id"],
                    "input_type": precision["input_type"],
                    "tensor_map_data_type": precision["tensor_map_data_type"],
                    "instruction_kind": precision["instruction_kind"],
                    "instruction_descriptor_u32":
                        precision["instruction_descriptor_u32"],
                    "family_id": family["family_id"],
                    "mode": family["mode"],
                    "stages": int(family["stages"]),
                    "k_tiles": int(k_tiles),
                    "output_tasks": int(family["output_tasks"]),
                    "ncu_selected": suffix in ncu_suffixes,
                }
                case["args"] = [
                    "--case-id", case_id,
                    "--precision-id", case["precision_id"],
                    "--mode", case["mode"],
                    "--stages", str(case["stages"]),
                    "--k-tiles", str(case["k_tiles"]),
                    "--output-tasks", str(case["output_tasks"]),
                    "--warmup-launches", str(manifest["warmup_launches"]),
                    "--expected-sm-count", str(manifest["expected_sm_count"]),
                    "--csv",
                ]
                cases.append(case)
    ids = [str(case["case_id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate causal case ID")
    actual_ncu_suffixes = {
        str(case["case_id"]).split(".", 1)[1]
        for case in cases if case["ncu_selected"]
    }
    if actual_ncu_suffixes != ncu_suffixes:
        raise RuntimeError("NCU case suffix list contains an unknown case")
    return cases


def run_bounded(command: list[str], timeout_seconds: int) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    termination_failed = False
    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            try:
                stdout, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                termination_failed = True
                stdout = "process group did not terminate after SIGKILL"
    return {
        "command": command,
        "returncode": process.returncode,
        "stdout": stdout,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "termination_failed": termination_failed,
        "host_elapsed_seconds": time.monotonic() - started,
    }


def capture(command: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, cwd=REPO, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "output": completed.stdout,
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {"command": command, "returncode": -1, "output": str(error)}


def environment_snapshot() -> dict[str, Any]:
    commands = {
        "gpu_identity": [
            tool("nvidia-smi"),
            "--query-gpu=name,uuid,compute_cap", "--format=csv,noheader",
        ],
        "gpu_state": [tool("nvidia-smi"), "-q", "-d", "CLOCK,POWER,TEMPERATURE"],
        "git_head": ["git", "rev-parse", "HEAD"],
        "git_branch": ["git", "branch", "--show-current"],
        "git_status": ["git", "status", "--short", "--untracked-files=no"],
        "nvcc": [tool("nvcc"), "--version"],
        "nvidia_smi": [tool("nvidia-smi"), "-q"],
        "ncu": [tool("ncu"), "--version"],
        "uname": ["uname", "-a"],
    }
    if Path("/usr/sbin/nvpmodel").is_file():
        commands["power_mode"] = ["/usr/sbin/nvpmodel", "-q"]
    if Path("/usr/bin/jetson_clocks").is_file():
        commands["jetson_clocks"] = ["/usr/bin/jetson_clocks", "--show"]
    return {
        "captured_at_utc": utc_now(),
        **{name: capture(command) for name, command in commands.items()},
    }


def validate_formal_environment(
    snapshot: dict[str, Any], expected_commit: str,
) -> None:
    head = str(snapshot["git_head"]["output"]).strip()
    if head != expected_commit:
        raise RuntimeError(
            f"checked-out commit {head!r} differs from expected {expected_commit!r}"
        )
    if str(snapshot["git_status"]["output"]).strip():
        raise RuntimeError("formal causal campaign requires a clean tracked worktree")
    if snapshot["gpu_identity"].get("returncode") != 0:
        raise RuntimeError("GPU identity probe failed")
    power = snapshot.get("power_mode")
    if not isinstance(power, dict) or "MAXN" not in str(power.get("output", "")).upper():
        raise RuntimeError("MAXN power mode is not proven")


def sass_stage_function_counts(sass_text: str) -> dict[str, dict[str, int]]:
    """Attribute required instructions to every instantiated stage kernel."""

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in sass_text.splitlines():
        if "Function :" in line:
            current = line.split("Function :", 1)[1].strip()
            if not current or current in sections:
                raise RuntimeError("causal SASS has a malformed function table")
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    result: dict[str, dict[str, int]] = {}
    descriptors = {
        "fp16_f32": (138412048, "0x8400010", "0x8400490"),
        "bf16_f32": (138413200, "0x8400490", "0x8400010"),
    }
    for precision_id, (
        descriptor, descriptor_immediate, other_immediate,
    ) in descriptors.items():
        for stages in (1, 2, 4):
            stage_marker = f"tc5a_pipeline_dag_kernelILi{stages}E"
            matches = [
                name for name in sections
                if stage_marker in name and str(descriptor) in name
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"causal SASS must contain exactly one {precision_id} "
                    f"stage-{stages} kernel"
                )
            body = "\n".join(sections[matches[0]])
            counts = {
                token: body.count(token)
                for token in (*REQUIRED_SASS_TOKENS, "UTMASTG")
            }
            counts["instruction_descriptor_immediate"] = body.count(
                descriptor_immediate
            )
            counts["other_instruction_descriptor_immediate"] = body.count(
                other_immediate
            )
            missing = [
                token for token in REQUIRED_SASS_TOKENS if counts[token] <= 0
            ]
            if (
                missing or counts["UTMASTG"] != 0
                or counts["instruction_descriptor_immediate"] <= 0
                or counts["other_instruction_descriptor_immediate"] != 0
            ):
                raise RuntimeError(
                    f"{precision_id} stage-{stages} causal SASS attribution "
                    f"failed: missing={missing} UTMASTG={counts['UTMASTG']} "
                    f"descriptor={counts['instruction_descriptor_immediate']} "
                    f"other_descriptor="
                    f"{counts['other_instruction_descriptor_immediate']}"
                )
            result[f"{precision_id}.stage{stages}"] = counts
    return result


def compile_command(
    run_dir: Path, *, undef_gnu_source: bool = False,
) -> list[str]:
    command = [tool("nvcc"), "-O3", "-std=c++17"]
    if undef_gnu_source:
        command += [
            "-Xcompiler=-U_GNU_SOURCE", "-D_DEFAULT_SOURCE",
            "-D_POSIX_C_SOURCE=200809L", "-D_XOPEN_SOURCE=700",
            "-D_XOPEN_SOURCE_EXTENDED=1", "-D_LARGEFILE64_SOURCE=1",
            "-D_ATFILE_SOURCE=1",
        ]
    command += [
        "-gencode", "arch=compute_110a,code=sm_110a",
        "-I", str(REPO / "GEMMsm110/include"), str(SOURCE_PATH),
        "-lcuda", "-o", str(run_dir / "build/tc5a_pipeline_dag"),
    ]
    return command


def compile_binary(
    run_dir: Path, *, undef_gnu_source: bool = False,
) -> dict[str, Any]:
    build = run_dir / "build"
    build.mkdir(parents=True, exist_ok=True)
    binary = build / "tc5a_pipeline_dag"
    command = compile_command(
        run_dir, undef_gnu_source=undef_gnu_source
    )
    (build / "compile_command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    compiled = run_bounded(command, 120)
    (build / "compile.log").write_text(compiled["stdout"])
    if compiled["returncode"] or compiled["timed_out"]:
        raise RuntimeError("causal benchmark compilation failed")

    sass_path = build / "tc5a_pipeline_dag.sass.txt"
    sass = capture([tool("cuobjdump"), "--dump-sass", str(binary)], 60)
    sass_path.write_text(str(sass["output"]))
    if sass["returncode"]:
        raise RuntimeError("cuobjdump failed")
    sass_text = sass_path.read_text()
    sass_function_counts = sass_stage_function_counts(sass_text)

    header_outcome = run_bounded([str(binary), "--csv-header"], 30)
    if header_outcome["returncode"] or header_outcome["timed_out"]:
        raise RuntimeError("compiled binary did not print its CSV contract")
    header = str(header_outcome["stdout"]).strip().split(",")
    if header != list(CSV_FIELDS):
        raise RuntimeError("compiled binary CSV header differs from runner schema")
    (build / "csv_header.txt").write_text(
        str(header_outcome["stdout"]), encoding="utf-8"
    )
    binary_digest = sha256_path(binary)
    (build / "binary.sha256").write_text(
        f"{binary_digest}  tc5a_pipeline_dag\n"
    )
    return {
        "binary": binary,
        "binary_sha256": binary_digest,
        "source_sha256": sha256_path(SOURCE_PATH),
        "helper_sha256": sha256_path(HELPER_PATH),
        "manifest_sha256": sha256_path(MANIFEST_PATH),
        "sass_path": sass_path,
        "sass_sha256": sha256_path(sass_path),
        "compile_command": command,
        "compile_command_sha256": sha256_path(build / "compile_command.json"),
        "sass_tokens": sorted(REQUIRED_SASS_TOKENS),
        "sass_function_counts": sass_function_counts,
    }


def retained_artifact(
    run_dir: Path, *, undef_gnu_source: bool = False,
) -> dict[str, Any] | None:
    """Reaudit and reuse the first compile instead of trusting reproducibility.

    CUDA binaries are not byte-reproducible across otherwise identical nvcc
    invocations.  Recompiling during resume would therefore invalidate every
    completed case fingerprint.  Once ``artifact.json`` exists, this function
    either proves that the retained compile still satisfies the frozen source,
    command, binary, SASS, and CSV contracts or fails closed.
    """

    artifact_path = run_dir / "build/artifact.json"
    if not artifact_path.is_file():
        return None
    try:
        recorded = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("retained causal artifact metadata is malformed") from error
    if not isinstance(recorded, dict):
        raise RuntimeError("retained causal artifact metadata is not an object")

    build = run_dir / "build"
    binary = build / "tc5a_pipeline_dag"
    sass_path = build / "tc5a_pipeline_dag.sass.txt"
    compile_path = build / "compile_command.json"
    binary_hash_path = build / "binary.sha256"
    header_path = build / "csv_header.txt"
    for path in (
        binary, sass_path, compile_path, binary_hash_path, header_path,
        build / "compile.log",
    ):
        if not path.is_file():
            raise RuntimeError(f"retained causal artifact is incomplete: {path.name}")

    command = compile_command(
        run_dir, undef_gnu_source=undef_gnu_source
    )
    try:
        recorded_command = json.loads(compile_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("retained compile command is malformed") from error
    if recorded_command != command or recorded.get("compile_command") != command:
        raise RuntimeError("retained causal compile command changed")

    expected_scalars = {
        "binary_sha256": sha256_path(binary),
        "source_sha256": sha256_path(SOURCE_PATH),
        "helper_sha256": sha256_path(HELPER_PATH),
        "manifest_sha256": sha256_path(MANIFEST_PATH),
        "sass_sha256": sha256_path(sass_path),
        "compile_command_sha256": sha256_path(compile_path),
    }
    for name, expected in expected_scalars.items():
        if recorded.get(name) != expected:
            raise RuntimeError(f"retained causal artifact hash changed: {name}")
    if Path(str(recorded.get("binary", ""))).resolve() != binary.resolve():
        raise RuntimeError("retained causal binary path changed")
    if Path(str(recorded.get("sass_path", ""))).resolve() != sass_path.resolve():
        raise RuntimeError("retained causal SASS path changed")
    if binary_hash_path.read_text().strip() != (
        f"{expected_scalars['binary_sha256']}  tc5a_pipeline_dag"
    ):
        raise RuntimeError("retained causal binary hash record changed")

    counts = sass_stage_function_counts(sass_path.read_text())
    tokens = sorted(REQUIRED_SASS_TOKENS)
    if (
        recorded.get("sass_tokens") != tokens
        or recorded.get("sass_function_counts") != counts
    ):
        raise RuntimeError("retained causal function-scoped SASS contract changed")
    if header_path.read_text().strip().split(",") != list(CSV_FIELDS):
        raise RuntimeError("retained causal CSV header artifact changed")
    header_outcome = run_bounded([str(binary), "--csv-header"], 30)
    if (
        header_outcome["returncode"]
        or header_outcome["timed_out"]
        or str(header_outcome["stdout"]).strip().split(",") != list(CSV_FIELDS)
    ):
        raise RuntimeError("retained causal binary no longer emits its CSV contract")
    return {
        "binary": binary,
        "binary_sha256": expected_scalars["binary_sha256"],
        "source_sha256": expected_scalars["source_sha256"],
        "helper_sha256": expected_scalars["helper_sha256"],
        "manifest_sha256": expected_scalars["manifest_sha256"],
        "sass_path": sass_path,
        "sass_sha256": expected_scalars["sass_sha256"],
        "compile_command": command,
        "compile_command_sha256": expected_scalars["compile_command_sha256"],
        "sass_tokens": tokens,
        "sass_function_counts": counts,
    }


def parse_csv_row(text: str) -> dict[str, str]:
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows) != 1 or len(rows[0]) != len(CSV_FIELDS):
        raise RuntimeError("benchmark must emit exactly one complete CSV row")
    return dict(zip(CSV_FIELDS, rows[0], strict=True))


def integer(row: dict[str, str], name: str) -> int:
    value = int(row[name])
    if str(value) != row[name]:
        raise RuntimeError(f"{name} is not canonical decimal")
    return value


def close_float(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=2e-9, abs_tol=1e-6)


def assert_fields(
    case: dict[str, Any], row: dict[str, str], manifest: dict[str, Any],
    *, contract_only: bool,
) -> dict[str, float]:
    payload = manifest["payload"]
    exact = {
        "case_id": case["case_id"],
        "precision_id": case["precision_id"],
        "tensor_map_data_type": case["tensor_map_data_type"],
        "instruction_descriptor_u32":
            str(case["instruction_descriptor_u32"]),
        "mode": case["mode"],
        "stages": str(case["stages"]), "k_tiles": str(case["k_tiles"]),
        "output_tasks": str(case["output_tasks"]),
        "total_k_operations": str(case["k_tiles"] * case["output_tasks"]),
        "threads": str(manifest["threads"]),
        "tma_requests_per_k_tile": str(payload["tma_requests_per_k_tile"]),
        "a_bytes_per_k_tile": str(payload["a_bytes_per_k_tile"]),
        "b_bytes_per_k_tile": str(payload["b_bytes_per_k_tile"]),
        "payload_bytes_per_k_tile": str(payload["bytes_per_k_tile"]),
        "mma_instructions_per_k_tile": str(
            payload["mma_instructions_per_k_tile"]),
        "accumulator_buffers": str(manifest["accumulator_buffers"]),
        "output_bytes_per_task": str(payload["output_bytes_per_task"]),
        "dynamic_smem_bytes": str(case["stages"] * payload["bytes_per_k_tile"]),
        "residency": manifest["residency"],
        "initialization": manifest["initialization"],
        "warmup_launches": str(manifest["warmup_launches"]),
        "sm_count": str(manifest["expected_sm_count"]),
    }
    for name, expected in exact.items():
        if row.get(name) != expected:
            raise RuntimeError(
                f"{case['case_id']} field {name}: {row.get(name)!r} != {expected!r}"
            )

    timestamp_names = (
        "start_ns", "first_tma_done_ns", "last_tma_done_ns",
        "first_mma_done_ns", "last_mma_done_ns",
        "first_epilogue_start_ns", "last_store_done_ns", "kernel_exit_ns",
    )
    timestamps = {name: integer(row, name) for name in timestamp_names}
    metrics = {name: float(row[name]) for name in CALIBRATION_METRICS}
    if any(not math.isfinite(value) or value < 0 for value in metrics.values()):
        raise RuntimeError(f"{case['case_id']} has invalid derived metric")
    if contract_only:
        if any(timestamps.values()) or integer(row, "smid") != 0:
            raise RuntimeError("contract-only row contains runtime evidence")
        return metrics

    start = timestamps["start_ns"]
    if start <= 0 or not 0 <= integer(row, "smid") < manifest["expected_sm_count"]:
        raise RuntimeError(f"{case['case_id']} has invalid SM/timer identity")
    mode = str(case["mode"])
    has_tma = mode != "mma-only"
    has_mma = mode != "tma-only"
    has_epilogue = mode == "full"
    if has_tma:
        if not start <= timestamps["first_tma_done_ns"] <= timestamps["last_tma_done_ns"]:
            raise RuntimeError(f"{case['case_id']} TMA timestamps are not monotonic")
    elif timestamps["first_tma_done_ns"] or timestamps["last_tma_done_ns"]:
        raise RuntimeError(f"{case['case_id']} unexpected TMA timestamp")
    if has_mma:
        if not start <= timestamps["first_mma_done_ns"] <= timestamps["last_mma_done_ns"]:
            raise RuntimeError(f"{case['case_id']} MMA timestamps are not monotonic")
    elif timestamps["first_mma_done_ns"] or timestamps["last_mma_done_ns"]:
        raise RuntimeError(f"{case['case_id']} unexpected MMA timestamp")
    if has_epilogue:
        if not (
            timestamps["first_mma_done_ns"]
            <= timestamps["first_epilogue_start_ns"]
            <= timestamps["last_mma_done_ns"]
            <= timestamps["last_store_done_ns"]
            == timestamps["kernel_exit_ns"]
        ):
            raise RuntimeError(f"{case['case_id']} epilogue timestamps are invalid")
    elif timestamps["first_epilogue_start_ns"] or timestamps["last_store_done_ns"]:
        raise RuntimeError(f"{case['case_id']} unexpected epilogue timestamp")
    expected_exit = (
        timestamps["last_tma_done_ns"] if mode == "tma-only"
        else timestamps["last_store_done_ns"] if mode == "full"
        else timestamps["last_mma_done_ns"]
    )
    if timestamps["kernel_exit_ns"] != expected_exit:
        raise RuntimeError(f"{case['case_id']} stop event is not the contract event")

    operations = int(case["k_tiles"]) * int(case["output_tasks"])
    arithmetic = {
        "first_tma_latency_ns": (
            timestamps["first_tma_done_ns"] - start if has_tma else 0),
        "tma_completion_span_ns": (
            timestamps["last_tma_done_ns"] - timestamps["first_tma_done_ns"]
            if has_tma else 0),
        "first_mma_latency_ns": (
            timestamps["first_mma_done_ns"] - start if has_mma else 0),
        "mma_completion_span_ns": (
            timestamps["last_mma_done_ns"] - timestamps["first_mma_done_ns"]
            if has_mma else 0),
        "epilogue_to_store_ns": (
            timestamps["last_store_done_ns"] -
            timestamps["first_epilogue_start_ns"] if has_epilogue else 0),
        "last_mma_to_store_ns": (
            timestamps["last_store_done_ns"] - timestamps["last_mma_done_ns"]
            if has_epilogue else 0),
        "total_measured_ns": expected_exit - start,
    }
    arithmetic["tma_interval_ns"] = (
        arithmetic["tma_completion_span_ns"] / (operations - 1)
        if has_tma and operations > 1 else 0.0)
    arithmetic["mma_interval_ns"] = (
        arithmetic["mma_completion_span_ns"] / (operations - 1)
        if has_mma and operations > 1 else 0.0)
    for name, expected in arithmetic.items():
        if not close_float(metrics[name], float(expected)):
            raise RuntimeError(
                f"{case['case_id']} derived {name} does not match raw timers"
            )
    return metrics


def contract_preflight(
    run_dir: Path, binary: Path, cases: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        command = [str(binary), *case["args"][:-1], "--contract-only", "--csv"]
        outcome = run_bounded(command, 30)
        if outcome["returncode"] or outcome["timed_out"]:
            raise RuntimeError(f"contract preflight failed: {case['case_id']}")
        fields = parse_csv_row(str(outcome["stdout"]))
        assert_fields(case, fields, manifest, contract_only=True)
        rows.append({
            "case_id": case["case_id"], "command": command,
            "fields": fields, "stdout": outcome["stdout"],
        })
    (run_dir / "static_contracts.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    return rows


def collect_ncu(
    case_dir: Path, binary: Path, case: dict[str, Any], manifest: dict[str, Any],
) -> dict[str, Any]:
    ncu_dir = case_dir / "ncu"
    ncu_dir.mkdir(parents=True, exist_ok=True)
    export = ncu_dir / "profile"
    command = [
        tool("ncu"), "--set", "basic", "--target-processes", "all",
        "--kernel-name-base", "demangled",
        "--kernel-name", "regex:tc5a_pipeline_dag_kernel",
        "--launch-skip", str(manifest["warmup_launches"]),
        "--launch-count", "1", "--force-overwrite", "--export", str(export),
        str(binary), *case["args"],
    ]
    outcome = run_bounded(command, manifest["ncu_timeout_seconds"])
    log = ncu_dir / "profile.log"
    log.write_text(str(outcome["stdout"]))
    report = ncu_dir / "profile.ncu-rep"
    permission_denied = "ERR_NVGPUCTRPERM" in str(outcome["stdout"])
    passed = bool(
        outcome["returncode"] == 0 and not outcome["timed_out"]
        and not outcome["termination_failed"] and not permission_denied
        and report.is_file() and report.stat().st_size > 0
    )
    result = {
        "selected": True, "set": "basic",
        "kernel_name_base": "demangled",
        "kernel_name_regex": "tc5a_pipeline_dag_kernel",
        "launch_skip": manifest["warmup_launches"], "launch_count": 1,
        "command": command, "returncode": outcome["returncode"],
        "timeout_seconds": manifest["ncu_timeout_seconds"],
        "timed_out": outcome["timed_out"],
        "termination_failed": outcome["termination_failed"],
        "permission_denied": permission_denied,
        "report_path": "ncu/profile.ncu-rep",
        "report_sha256": sha256_path(report) if report.is_file() else None,
        "log_sha256": sha256_path(log), "pass": passed,
    }
    if not passed:
        raise RuntimeError(f"NCU failed for {case['case_id']}")
    return result


def stats(values: Iterable[float]) -> dict[str, float]:
    rows = list(values)
    return {
        "median": statistics.median(rows), "minimum": min(rows),
        "maximum": max(rows), "mean": statistics.fmean(rows),
    }


def prior_result_is_reusable(
    case_dir: Path, binary: Path, artifact: dict[str, Any],
    case: dict[str, Any], manifest: dict[str, Any], fingerprint: str,
    require_ncu: bool,
) -> dict[str, Any] | None:
    result_path = case_dir / "result.json"
    trials_path = case_dir / "trials.jsonl"
    if not result_path.is_file() or not trials_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text())
        trials = [json.loads(line) for line in trials_path.read_text().splitlines()]
    except (json.JSONDecodeError, OSError):
        return None
    if not (
        result.get("status") == "ok" and result.get("fingerprint") == fingerprint
        and result.get("trial_count") == EXPECTED_TRIALS
        and len(trials) == EXPECTED_TRIALS
    ):
        return None
    expected_metadata = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "precision_id": case["precision_id"],
        "tensor_map_data_type": case["tensor_map_data_type"],
        "instruction_descriptor_u32": case["instruction_descriptor_u32"],
        "family_id": case["family_id"],
        "mode": case["mode"],
        "stages": case["stages"],
        "k_tiles": case["k_tiles"],
        "output_tasks": case["output_tasks"],
        "metric_unit": "ns",
        "source_path": str(SOURCE_PATH.relative_to(REPO)),
        "source_sha256": artifact["source_sha256"],
        "helper_path": str(HELPER_PATH.relative_to(REPO)),
        "helper_sha256": artifact["helper_sha256"],
        "binary_sha256": artifact["binary_sha256"],
        "sass_path": "build/tc5a_pipeline_dag.sass.txt",
        "sass_sha256": artifact["sass_sha256"],
        "sass_tokens": artifact["sass_tokens"],
        "sass_function_counts": artifact["sass_function_counts"],
        "trial_timeout_seconds": manifest["trial_timeout_seconds"],
    }
    if any(result.get(name) != expected for name, expected in expected_metadata.items()):
        return None
    metric_values = {name: [] for name in CALIBRATION_METRICS}
    for trial_index, trial in enumerate(trials, 1):
        if not isinstance(trial, dict) or not (
            trial.get("trial") == trial_index
            and trial.get("command") == [str(binary), *case["args"]]
            and trial.get("returncode") == 0
            and trial.get("timeout_seconds") == manifest["trial_timeout_seconds"]
            and trial.get("timed_out") is False
            and trial.get("termination_failed") is False
        ):
            return None
        try:
            fields = parse_csv_row(str(trial["raw_stdout"]))
            audited = assert_fields(case, fields, manifest, contract_only=False)
        except (KeyError, TypeError, ValueError, RuntimeError):
            return None
        if fields != trial.get("fields") or audited != trial.get("audited_metrics"):
            return None
        for name in CALIBRATION_METRICS:
            metric_values[name].append(audited[name])
    expected_stats = {name: stats(values) for name, values in metric_values.items()}
    if result.get("metric_stats") != expected_stats:
        return None
    if require_ncu:
        ncu = result.get("ncu", {})
        report = case_dir / str(ncu.get("report_path", ""))
        log = case_dir / "ncu/profile.log"
        if not (
            ncu.get("pass") is True
            and report.is_file()
            and log.is_file()
            and ncu.get("report_sha256") == sha256_path(report)
            and ncu.get("log_sha256") == sha256_path(log)
        ):
            return None
    return result


def run_case(
    run_dir: Path, binary: Path, artifact: dict[str, Any],
    case: dict[str, Any], manifest: dict[str, Any], collect_ncu_flag: bool,
) -> dict[str, Any]:
    case_dir = run_dir / "cases" / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256_json({
        "case": case, "binary_sha256": artifact["binary_sha256"],
        "source_sha256": artifact["source_sha256"],
        "helper_sha256": artifact["helper_sha256"],
        "trial_count": EXPECTED_TRIALS,
        "ncu": collect_ncu_flag and case["ncu_selected"],
    })
    reusable = prior_result_is_reusable(
        case_dir, binary, artifact, case, manifest, fingerprint,
        collect_ncu_flag and bool(case["ncu_selected"]),
    )
    if reusable is not None:
        print(f"SKIP {case['case_id']}: audited fingerprint", flush=True)
        return reusable

    rows: list[dict[str, Any]] = []
    metric_values = {name: [] for name in CALIBRATION_METRICS}
    for trial_index in range(1, EXPECTED_TRIALS + 1):
        command = [str(binary), *case["args"]]
        outcome = run_bounded(command, manifest["trial_timeout_seconds"])
        if outcome["timed_out"]:
            (case_dir / "timeout.json").write_text(
                json.dumps({
                    "case_id": case["case_id"], "trial": trial_index,
                    "captured_at_utc": utc_now(), **outcome,
                }, indent=2, sort_keys=True) + "\n"
            )
            raise RuntimeError(f"{case['case_id']} trial {trial_index} timed out")
        if outcome["returncode"]:
            raise RuntimeError(
                f"{case['case_id']} trial {trial_index} failed: {outcome['stdout']}"
            )
        fields = parse_csv_row(str(outcome["stdout"]))
        audited = assert_fields(case, fields, manifest, contract_only=False)
        for name in CALIBRATION_METRICS:
            metric_values[name].append(audited[name])
        rows.append({
            "trial": trial_index, "captured_at_utc": utc_now(),
            "command": command, "returncode": outcome["returncode"],
            "timeout_seconds": manifest["trial_timeout_seconds"],
            "timed_out": False, "termination_failed": False,
            "raw_stdout": outcome["stdout"], "fields": fields,
            "audited_metrics": audited,
        })
    (case_dir / "trials.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    result: dict[str, Any] = {
        "schema_version": 1, "case_id": case["case_id"],
        "precision_id": case["precision_id"],
        "tensor_map_data_type": case["tensor_map_data_type"],
        "instruction_descriptor_u32": case["instruction_descriptor_u32"],
        "family_id": case["family_id"], "mode": case["mode"],
        "stages": case["stages"], "k_tiles": case["k_tiles"],
        "output_tasks": case["output_tasks"], "status": "ok",
        "fingerprint": fingerprint, "trial_count": EXPECTED_TRIALS,
        "metric_unit": "ns", "metric_stats": {
            name: stats(values) for name, values in metric_values.items()
        },
        "source_path": str(SOURCE_PATH.relative_to(REPO)),
        "source_sha256": artifact["source_sha256"],
        "helper_path": str(HELPER_PATH.relative_to(REPO)),
        "helper_sha256": artifact["helper_sha256"],
        "binary_sha256": artifact["binary_sha256"],
        "sass_path": "build/tc5a_pipeline_dag.sass.txt",
        "sass_sha256": artifact["sass_sha256"],
        "sass_tokens": artifact["sass_tokens"],
        "sass_function_counts": artifact["sass_function_counts"],
        "trial_timeout_seconds": manifest["trial_timeout_seconds"],
        "completed_at_utc": utc_now(),
    }
    if collect_ncu_flag and case["ncu_selected"]:
        result["ncu"] = collect_ncu(case_dir, binary, case, manifest)
    else:
        result["ncu"] = {"selected": bool(case["ncu_selected"]), "pass": None}
    (case_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def linear_fit(points: list[tuple[float, float]]) -> dict[str, float]:
    if len(points) < 2:
        raise RuntimeError("linear fit requires at least two points")
    x_mean = statistics.fmean(x for x, _ in points)
    y_mean = statistics.fmean(y for _, y in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator <= 0:
        raise RuntimeError("linear fit x values are degenerate")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    intercept = y_mean - slope * x_mean
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    total = sum((y - y_mean) ** 2 for _, y in points)
    r_squared = 1.0 if total == 0 and residual == 0 else 1.0 - residual / total
    if intercept <= 0 or slope <= 0 or not all(
        math.isfinite(value) for value in (intercept, slope, r_squared)
    ):
        raise RuntimeError("causal timing fit is not positive and finite")
    return {"intercept_ns": intercept, "slope_ns": slope, "r_squared": r_squared}


def predict_worker_ns(
    *, k_tiles: int, output_tasks: int, joint_first_ns: float,
    joint_interval_ns: float, epilogue_latency_ns: float,
    accumulator_buffers: int,
) -> float:
    previous_mma_done = 0.0
    epilogue_done: list[float] = []
    for task in range(output_tasks):
        first_mma_done = (
            joint_first_ns if task == 0
            else previous_mma_done + joint_interval_ns
        )
        if task >= accumulator_buffers:
            first_mma_done = max(
                first_mma_done,
                epilogue_done[task - accumulator_buffers]
                + joint_interval_ns,
            )
        last_mma_done = (
            first_mma_done + (k_tiles - 1) * joint_interval_ns
        )
        epilogue_start = max(
            last_mma_done, epilogue_done[-1] if epilogue_done else 0.0
        )
        epilogue_done.append(epilogue_start + epilogue_latency_ns)
        previous_mma_done = last_mma_done
    return epilogue_done[-1]


def build_profile(
    results: list[dict[str, Any]], manifest: dict[str, Any],
    run_id: str, expected_commit: str, precision_id: str,
) -> dict[str, Any]:
    precision_results = [
        row for row in results if row["precision_id"] == precision_id
    ]
    expected_per_precision = len(manifest["families"]) * (
        len(manifest["calibration_k_tiles"])
        + len(manifest["holdout_k_tiles"])
    )
    if len(precision_results) != expected_per_precision:
        raise RuntimeError(
            f"{precision_id}: causal result matrix is incomplete"
        )
    by_family_k = {
        (str(row["family_id"]), int(row["k_tiles"])): row
        for row in precision_results
    }
    calibration = list(manifest["fit_contract"]["calibration_points"])

    def fit_family(family: str) -> dict[str, float]:
        return linear_fit([
            (
                float(k_tiles - 1),
                float(by_family_k[(family, k_tiles)]
                      ["metric_stats"]["total_measured_ns"]["median"]),
            )
            for k_tiles in calibration
        ])

    tma_fit = fit_family("tma_s4")
    mma_fit = fit_family("mma_s4")
    joint_fit = fit_family("overlap_s4")
    epilogue_output_tasks = set(
        manifest["fit_contract"]["epilogue_calibration_output_tasks"]
    )
    epilogue_samples = [
        float(row["metric_stats"]["last_mma_to_store_ns"]["median"])
        for row in precision_results
        if row["mode"] == "full"
        and row["k_tiles"] in manifest["calibration_k_tiles"]
        and row["output_tasks"] in epilogue_output_tasks
    ]
    epilogue_latency = statistics.median(epilogue_samples)
    joint_first = max(
        joint_fit["intercept_ns"],
        statistics.median(
            float(by_family_k[("overlap_s4", k)]["metric_stats"]
                  ["first_mma_latency_ns"]["median"])
            for k in calibration
        ),
    )
    joint_interval = max(
        tma_fit["slope_ns"], mma_fit["slope_ns"], joint_fit["slope_ns"]
    )
    validations: list[dict[str, Any]] = []
    calibration_k = set(manifest["fit_contract"]["calibration_points"])
    calibration_o = set(manifest["fit_contract"]["calibration_output_tasks"])
    for row in precision_results:
        if row["mode"] != "full":
            continue
        actual = float(row["metric_stats"]["total_measured_ns"]["median"])
        predicted = predict_worker_ns(
            k_tiles=int(row["k_tiles"]), output_tasks=int(row["output_tasks"]),
            joint_first_ns=joint_first, joint_interval_ns=joint_interval,
            epilogue_latency_ns=epilogue_latency,
            accumulator_buffers=manifest["accumulator_buffers"],
        )
        relative_error = abs(predicted - actual) / actual
        split = (
            "calibration" if row["k_tiles"] in calibration_k
            and row["output_tasks"] in calibration_o else "holdout"
        )
        validations.append({
            "case_id": row["case_id"], "split": split,
            "k_tiles": int(row["k_tiles"]),
            "output_tasks": int(row["output_tasks"]),
            "actual_median_ns": actual, "predicted_ns": predicted,
            "relative_error": relative_error,
        })
    max_calibration_error = max(
        row["relative_error"] for row in validations
        if row["split"] == "calibration"
    )
    max_holdout_error = max(
        row["relative_error"] for row in validations
        if row["split"] == "holdout"
    )
    fit_contract = manifest["fit_contract"]
    component_r2 = {
        "tma": tma_fit["r_squared"], "mma": mma_fit["r_squared"],
        "joint": joint_fit["r_squared"],
    }
    qualified = bool(
        all(value >= fit_contract["minimum_r_squared"]
            for value in component_r2.values())
        and max_calibration_error <= fit_contract["maximum_holdout_relative_error"]
        and max_holdout_error <= fit_contract["maximum_holdout_relative_error"]
    )
    artifact_root = f"results/sm110_gemm_causal_campaign/{run_id}"
    artifact_paths = [
        str(SOURCE_PATH.relative_to(REPO)),
        str(HELPER_PATH.relative_to(REPO)),
        str(MANIFEST_PATH.relative_to(REPO)),
        f"{artifact_root}/run_spec.json",
        f"{artifact_root}/summary.json",
        f"{artifact_root}/pipeline_profiles.json",
        f"{artifact_root}/artifact_sha256.txt",
        f"{artifact_root}/build/tc5a_pipeline_dag.sass.txt",
    ]
    artifact_paths.extend(
        f"{artifact_root}/cases/{row['case_id']}/trials.jsonl"
        for row in sorted(precision_results, key=lambda item: item["case_id"])
    )
    artifact_paths.extend(
        f"{artifact_root}/cases/{row['case_id']}/ncu/profile.ncu-rep"
        for row in sorted(precision_results, key=lambda item: item["case_id"])
        if row.get("ncu", {}).get("selected") is True
    )
    return {
        "schema_version": 1,
        "profile_id": (
            f"{run_id}.pipeline.tc5a_m128n256k64_stage4.{precision_id}"
        ),
        "resource": "pipeline.tc5a_m128n256k64_stage4",
        "schedule_id": manifest["schedule_id"],
        "precision_ids": [precision_id],
        "evidence_kind": "measured_joint",
        "qualification": "closure_qualified" if qualified else "quarantined",
        "trial_count_per_case": EXPECTED_TRIALS,
        "source_id": run_id, "expected_commit": expected_commit,
        "source_path": str(SOURCE_PATH.relative_to(REPO)),
        "source_locator": (
            f"91-case {precision_id} causal campaign; medians and "
            "predeclared fit"
        ),
        "input_residency": manifest["residency"],
        "stages": fit_contract["profile_stages"],
        "accumulator_buffers": fit_contract["profile_accumulator_buffers"],
        "resident_ctas_per_sm": fit_contract["profile_resident_ctas_per_sm"],
        "maximum_k_tiles": max(
            [*manifest["calibration_k_tiles"], *manifest["holdout_k_tiles"]]
        ),
        "maximum_output_tasks_per_worker": max(manifest["full_output_task_sweep"]),
        "tma_first_completion_seconds": tma_fit["intercept_ns"] * 1e-9,
        "tma_completion_interval_seconds": tma_fit["slope_ns"] * 1e-9,
        "mma_first_completion_seconds": mma_fit["intercept_ns"] * 1e-9,
        "mma_completion_interval_seconds": mma_fit["slope_ns"] * 1e-9,
        "joint_first_mma_completion_seconds": joint_first * 1e-9,
        "joint_completion_interval_seconds": joint_interval * 1e-9,
        "epilogue_latency_seconds": epilogue_latency * 1e-9,
        "component_r_squared": component_r2,
        "max_calibration_relative_error": max_calibration_error,
        "max_holdout_relative_error": max_holdout_error,
        "fit_contract": fit_contract,
        "validation": validations,
        "closure_qualified": qualified,
        "artifact_paths": artifact_paths,
        "applicable_sm_counts": [20],
        "applicable_hardware_ids": ["thor_t5000_sm110_20sm"],
        "applicable_operating_modes": ["MAXN"],
        "applicable_clock_hz": [1575000000.0],
        "timed_scope": "device_kernel",
    }


def build_profiles(
    results: list[dict[str, Any]], manifest: dict[str, Any],
    run_id: str, expected_commit: str,
) -> list[dict[str, Any]]:
    return [
        build_profile(results, manifest, run_id, expected_commit, precision_id)
        for precision_id in precision_ids(manifest)
    ]


def write_artifact_manifest(run_dir: Path) -> None:
    excluded = {"artifact_sha256.txt", "launcher.log", "launcher.pid"}
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative in excluded:
            continue
        rows.append(f"{sha256_path(path)}  {relative}\n")
    (run_dir / "artifact_sha256.txt").write_text("".join(rows))


def plan_payload(manifest: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "schedule_id": manifest["schedule_id"],
        "precision_ids": precision_ids(manifest),
        "profile_count": len(manifest["precision_contracts"]),
        "case_count": len(cases),
        "family_count": len(manifest["families"]),
        "raw_trial_count": len(cases) * EXPECTED_TRIALS,
        "ncu_case_count": sum(bool(case["ncu_selected"]) for case in cases),
        "calibration_k_tiles": manifest["calibration_k_tiles"],
        "holdout_k_tiles": manifest["holdout_k_tiles"],
        "full_output_task_sweep": manifest["full_output_task_sweep"],
        "max_dynamic_smem_bytes": max(
            case["stages"] * manifest["payload"]["bytes_per_k_tile"]
            for case in cases
        ),
        "max_hot_input_bytes": (
            (manifest["tile"]["m"] + manifest["tile"]["n"])
            * max([*manifest["calibration_k_tiles"], *manifest["holdout_k_tiles"]])
            * manifest["tile"]["k"] * 2
        ),
        "max_output_bytes": (
            max(manifest["full_output_task_sweep"])
            * manifest["payload"]["output_bytes_per_task"]
        ),
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--expected-commit")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument("--nvcc-host-undef-gnu-source", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = load_manifest()
    cases = make_cases(manifest)
    plan = plan_payload(manifest, cases)
    if args.plan:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not args.run_id or not RUN_ID_RE.fullmatch(args.run_id):
        raise SystemExit("--run-id must be a stable path-safe identifier")
    if not args.expected_commit or not COMMIT_RE.fullmatch(args.expected_commit):
        raise SystemExit("--expected-commit must be a 40-lowercase-hex commit")
    if not args.static_only and args.nvcc_host_undef_gnu_source:
        raise SystemExit("host glibc workaround is local-static-only")
    if not args.static_only and not args.ncu:
        raise SystemExit("formal hardware evidence requires --ncu")

    run_dir = args.output_root.resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    freeze_json(run_dir / "plan.json", plan, "campaign plan")
    freeze_json(
        run_dir / "manifest_snapshot.json", manifest, "manifest snapshot"
    )
    current_environment = environment_snapshot()
    if not args.static_only:
        validate_formal_environment(current_environment, args.expected_commit)
    environment_path = run_dir / "environment.json"
    if environment_path.is_file():
        try:
            environment = json.loads(environment_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("retained initial environment is malformed") from error
        if not isinstance(environment, dict):
            raise RuntimeError("retained initial environment is not an object")
    else:
        environment = current_environment
        environment_path.write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n"
        )
    snapshots_path = run_dir / "environment_snapshots.jsonl"
    with snapshots_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(current_environment, sort_keys=True) + "\n")

    source_dependencies = {
        str(path.relative_to(REPO)): sha256_path(path)
        for path in (SOURCE_PATH, HELPER_PATH, MANIFEST_PATH, Path(__file__).resolve())
    }
    run_spec = {
        "schema_version": 2, "run_id": args.run_id,
        "campaign": "sm110_tc5a_causal_pipeline_dag",
        "expected_commit": args.expected_commit,
        "generator": str(Path(__file__).resolve().relative_to(REPO)),
        "generator_sha256": sha256_path(Path(__file__).resolve()),
        "contract_manifest": str(MANIFEST_PATH.relative_to(REPO)),
        "contract_manifest_sha256": sha256_path(MANIFEST_PATH),
        "source_dependencies": source_dependencies,
        "precision_ids": precision_ids(manifest),
        "precision_contracts": manifest["precision_contracts"],
        "profile_count": len(manifest["precision_contracts"]),
        "case_count": len(cases), "family_count": len(manifest["families"]),
        "trials": EXPECTED_TRIALS,
        "trial_timeout_seconds": manifest["trial_timeout_seconds"],
        "ncu_timeout_seconds": manifest["ncu_timeout_seconds"],
        "termination_grace_seconds": 5,
        "ncu_requested": bool(args.ncu),
        "ncu_case_count": plan["ncu_case_count"],
        "ncu_policy": "four predeclared k16 attribution cases per precision",
        "static_only": bool(args.static_only), "cases": cases,
    }
    freeze_json(run_dir / "run_spec.json", run_spec, "run specification")

    artifact = retained_artifact(
        run_dir, undef_gnu_source=args.nvcc_host_undef_gnu_source
    )
    if artifact is None:
        artifact = compile_binary(
            run_dir, undef_gnu_source=args.nvcc_host_undef_gnu_source
        )
        print("COMPILED_FROZEN_ARTIFACT", flush=True)
    else:
        print("REUSED_AUDITED_FROZEN_ARTIFACT", flush=True)
    (run_dir / "build/artifact.json").write_text(
        json.dumps({
            key: str(value) if isinstance(value, Path) else value
            for key, value in artifact.items()
        }, indent=2, sort_keys=True) + "\n"
    )
    contract_preflight(run_dir, artifact["binary"], cases, manifest)
    if args.static_only:
        (run_dir / "STATIC_COMPLETE").write_text(
            f"run_id={args.run_id}\ncommit={args.expected_commit}\n"
        )
        write_artifact_manifest(run_dir)
        print(json.dumps({
            "pass": True, "static_only": True, "run_dir": str(run_dir),
            "case_count": len(cases), "ncu_case_count": plan["ncu_case_count"],
        }, indent=2, sort_keys=True))
        return 0

    results: list[dict[str, Any]] = []
    progress = run_dir / "progress.jsonl"
    (run_dir / "campaign_status.json").write_text(
        json.dumps({
            "status": "running", "completed_cases": 0,
            "total_cases": len(cases), "updated_at_utc": utc_now(),
        }, indent=2, sort_keys=True) + "\n"
    )
    for index, case in enumerate(cases, 1):
        result = run_case(
            run_dir, artifact["binary"], artifact, case, manifest, args.ncu
        )
        results.append(result)
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "captured_at_utc": utc_now(), "index": index,
                "case_count": len(cases), "case_id": case["case_id"],
                "status": "ok",
            }, sort_keys=True) + "\n")
        (run_dir / "summary.json").write_text(
            json.dumps({
                "schema_version": 2, "run_id": args.run_id,
                "expected_commit": args.expected_commit,
                "status": "running", "completed_case_count": len(results),
                "case_count": len(cases), "results": results,
            }, indent=2, sort_keys=True) + "\n"
        )
        (run_dir / "campaign_status.json").write_text(
            json.dumps({
                "status": "running", "completed_cases": len(results),
                "total_cases": len(cases), "last_case_id": case["case_id"],
                "updated_at_utc": utc_now(),
            }, indent=2, sort_keys=True) + "\n"
        )
        print(f"DONE {index}/{len(cases)} {case['case_id']}", flush=True)

    profiles = build_profiles(
        results, manifest, args.run_id, args.expected_commit
    )
    profile_qualified_by_precision = {
        profile["precision_ids"][0]: bool(profile["closure_qualified"])
        for profile in profiles
    }
    profiles_bundle = {
        "schema_version": 2,
        "run_id": args.run_id,
        "expected_commit": args.expected_commit,
        "pipeline_profiles": profiles,
    }
    (run_dir / "pipeline_profiles.json").write_text(
        json.dumps(profiles_bundle, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "schema_version": 2, "run_id": args.run_id,
        "expected_commit": args.expected_commit, "status": "complete",
        "case_count": len(cases), "trial_count": len(cases) * EXPECTED_TRIALS,
        "ncu_case_count": plan["ncu_case_count"],
        "profile_count": len(profiles),
        "profile_qualified_by_precision": profile_qualified_by_precision,
        "profile_qualified": all(profile_qualified_by_precision.values()),
        "results": results, "completed_at_utc": utc_now(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    final_environment = environment_snapshot()
    with snapshots_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(final_environment, sort_keys=True) + "\n")
    with progress.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "captured_at_utc": utc_now(), "status": "complete",
            "completed_case_count": len(cases), "case_count": len(cases),
        }, sort_keys=True) + "\n")
    (run_dir / "campaign_status.json").write_text(
        json.dumps({
            "status": "complete", "completed_cases": len(cases),
            "total_cases": len(cases),
            "profile_count": len(profiles),
            "profile_qualified_by_precision": profile_qualified_by_precision,
            "profile_qualified": all(profile_qualified_by_precision.values()),
            "updated_at_utc": utc_now(),
        }, indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "COMPLETE").write_text(
        f"run_id={args.run_id}\ncommit={args.expected_commit}\n"
        f"profile_qualified="
        f"{str(all(profile_qualified_by_precision.values())).lower()}\n"
    )
    write_artifact_manifest(run_dir)
    print(json.dumps({
        "pass": True, "run_dir": str(run_dir), "case_count": len(cases),
        "trial_count": len(cases) * EXPECTED_TRIALS,
        "profile_count": len(profiles),
        "profile_qualified_by_precision": profile_qualified_by_precision,
        "profile_qualified": all(profile_qualified_by_precision.values()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"pass": False, "error": str(error)}, indent=2))
        raise SystemExit(1)
