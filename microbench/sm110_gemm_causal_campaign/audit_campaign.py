#!/usr/bin/env python3
"""Independent, relocation-safe audit of the SM110 causal DAG campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable


REPO = Path(__file__).resolve().parents[2]
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
EXPECTED_TRIALS = 10
EXPECTED_PRECISIONS = ("fp16_f32", "bf16_f32")
EXPECTED_CASES = 182
EXPECTED_FAMILIES = 13
EXPECTED_NCU_CASES = 8
EXPECTED_PRECISION_CONTRACTS = [
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
]
REQUIRED_SASS_TOKENS = ("UTMALDG.2D", "UTCHMMA", "UTCBAR", "LDTM.")
METRICS = (
    "first_tma_latency_ns", "tma_completion_span_ns", "tma_interval_ns",
    "first_mma_latency_ns", "mma_completion_span_ns", "mma_interval_ns",
    "epilogue_to_store_ns", "last_mma_to_store_ns", "total_measured_ns",
)
CSV_FIELDS = (
    "case_id", "precision_id", "tensor_map_data_type",
    "instruction_descriptor_u32", "mode", "stages", "k_tiles", "output_tasks",
    "total_k_operations", "threads", "tma_requests_per_k_tile",
    "a_bytes_per_k_tile", "b_bytes_per_k_tile",
    "payload_bytes_per_k_tile", "mma_instructions_per_k_tile",
    "accumulator_buffers", "output_bytes_per_task", "dynamic_smem_bytes",
    "residency", "initialization", "warmup_launches", "sm_count", "smid",
    "start_ns", "first_tma_done_ns", "last_tma_done_ns",
    "first_mma_done_ns", "last_mma_done_ns", "first_epilogue_start_ns",
    "last_store_done_ns", "kernel_exit_ns", "first_tma_latency_ns",
    "tma_completion_span_ns", "tma_interval_ns", "first_mma_latency_ns",
    "mma_completion_span_ns", "mma_interval_ns", "epilogue_to_store_ns",
    "last_mma_to_store_ns", "total_measured_ns",
)
EXPECTED_DEPENDENCIES = {
    "microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu",
    "GEMMsm110/include/sm110_ptx_helpers.cuh",
    "microbench/sm110_gemm_causal_campaign/contract_manifest.json",
    "microbench/sm110_gemm_causal_campaign/run_causal_campaign.py",
}


def add(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def git_blob(commit: str, relative: str) -> bytes | None:
    path = Path(relative)
    if (
        not COMMIT_RE.fullmatch(commit) or path.is_absolute()
        or not path.parts or ".." in path.parts
    ):
        return None
    completed = subprocess.run(
        ["git", "show", "--no-ext-diff", f"{commit}:{path.as_posix()}"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30, check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def recorded_commit(environment: object) -> str | None:
    if not isinstance(environment, dict):
        return None
    row = environment.get("git_head")
    if not isinstance(row, dict) or row.get("returncode") != 0:
        return None
    value = str(row.get("output", "")).strip()
    return value if COMMIT_RE.fullmatch(value) else None


def parse_csv_row(text: str) -> dict[str, str]:
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows) != 1 or len(rows[0]) != len(CSV_FIELDS):
        raise ValueError("not exactly one complete CSV row")
    return dict(zip(CSV_FIELDS, rows[0], strict=True))


def canonical_integer(text: str) -> int:
    value = int(text)
    if str(value) != text:
        raise ValueError("noncanonical decimal integer")
    return value


def validate_manifest(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        return None
    expected_scalars = {
        "schedule_id": "tc5a_m128n256k64_stage4",
        "expected_sm_count": 20, "threads": 192,
        "accumulator_buffers": 2, "residency": "hot_l2",
        "initialization": "cuda_memset_zero",
        "external_trials_per_case": 10, "warmup_launches": 3,
        "trial_timeout_seconds": 120, "ncu_timeout_seconds": 300,
    }
    if any(value.get(key) != expected for key, expected in expected_scalars.items()):
        return None
    if value.get("precision_contracts") != EXPECTED_PRECISION_CONTRACTS:
        return None
    if value.get("tile") != {"m": 128, "n": 256, "k": 64}:
        return None
    if value.get("payload") != {
        "a_bytes_per_k_tile": 16384, "b_bytes_per_k_tile": 32768,
        "bytes_per_k_tile": 49152, "tma_requests_per_k_tile": 2,
        "mma_instructions_per_k_tile": 4, "output_bytes_per_task": 131072,
    }:
        return None
    if value.get("calibration_k_tiles") != [1, 2, 4, 8, 16, 32]:
        return None
    if value.get("holdout_k_tiles") != [64]:
        return None
    if value.get("full_output_task_sweep") != [1, 2, 4, 8, 16, 32]:
        return None
    expected_families = [
        ("tma_s1", "tma-only", 1, 1),
        ("tma_s2", "tma-only", 2, 1),
        ("tma_s4", "tma-only", 4, 1),
        ("mma_s4", "mma-only", 4, 1),
        ("overlap_s1", "overlap", 1, 1),
        ("overlap_s2", "overlap", 2, 1),
        ("overlap_s4", "overlap", 4, 1),
        ("full_s4_o1", "full", 4, 1),
        ("full_s4_o2", "full", 4, 2),
        ("full_s4_o4", "full", 4, 4),
        ("full_s4_o8", "full", 4, 8),
        ("full_s4_o16", "full", 4, 16),
        ("full_s4_o32", "full", 4, 32),
    ]
    families = value.get("families")
    if not isinstance(families, list) or [
        (
            row.get("family_id"), row.get("mode"), row.get("stages"),
            row.get("output_tasks"),
        )
        for row in families if isinstance(row, dict)
    ] != expected_families:
        return None
    if value.get("ncu_case_suffixes") != [
        "tma_s4_k16", "mma_s4_k16", "overlap_s4_k16", "full_s4_o4_k16"
    ]:
        return None
    if value.get("fit_contract") != {
        "calibration_points": [1, 2, 4, 8, 16, 32],
        "holdout_points": [64],
        "calibration_output_tasks": [1, 2, 4, 8, 16],
        "epilogue_calibration_output_tasks": [1],
        "holdout_output_tasks": [32],
        "maximum_holdout_relative_error": 0.1,
        "minimum_r_squared": 0.98,
        "profile_stages": 4,
        "profile_accumulator_buffers": 2,
        "profile_resident_ctas_per_sm": 1,
    }:
        return None
    return value


def expected_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    ncu = set(manifest["ncu_case_suffixes"])
    k_values = [*manifest["calibration_k_tiles"], *manifest["holdout_k_tiles"]]
    cases = []
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
                    "mode": family["mode"], "stages": family["stages"],
                    "k_tiles": k_tiles,
                    "output_tasks": family["output_tasks"],
                    "ncu_selected": suffix in ncu,
                }
                case["args"] = [
                    "--case-id", case_id,
                    "--precision-id", precision["precision_id"],
                    "--mode", case["mode"], "--stages", str(case["stages"]),
                    "--k-tiles", str(k_tiles),
                    "--output-tasks", str(case["output_tasks"]),
                    "--warmup-launches", "3", "--expected-sm-count", "20", "--csv",
                ]
                cases.append(case)
    return cases


def expected_static_fields(
    case: dict[str, Any], manifest: dict[str, Any],
) -> dict[str, str]:
    payload = manifest["payload"]
    return {
        "case_id": case["case_id"], "precision_id": case["precision_id"],
        "tensor_map_data_type": case["tensor_map_data_type"],
        "instruction_descriptor_u32": str(case["instruction_descriptor_u32"]),
        "mode": case["mode"],
        "stages": str(case["stages"]), "k_tiles": str(case["k_tiles"]),
        "output_tasks": str(case["output_tasks"]),
        "total_k_operations": str(case["k_tiles"] * case["output_tasks"]),
        "threads": "192", "tma_requests_per_k_tile": "2",
        "a_bytes_per_k_tile": "16384", "b_bytes_per_k_tile": "32768",
        "payload_bytes_per_k_tile": "49152",
        "mma_instructions_per_k_tile": "4", "accumulator_buffers": "2",
        "output_bytes_per_task": "131072",
        "dynamic_smem_bytes": str(case["stages"] * payload["bytes_per_k_tile"]),
        "residency": "hot_l2", "initialization": "cuda_memset_zero",
        "warmup_launches": "3", "sm_count": "20",
    }


def field_errors(
    case: dict[str, Any], fields: object, manifest: dict[str, Any],
    *, runtime: bool,
) -> tuple[list[str], dict[str, float] | None]:
    errors: list[str] = []
    if not isinstance(fields, dict):
        return ["fields are not an object"], None
    if set(fields) != set(CSV_FIELDS):
        errors.append("CSV field set mismatch")
        return errors, None
    for name, expected in expected_static_fields(case, manifest).items():
        if fields.get(name) != expected:
            errors.append(f"static field mismatch:{name}")
    timestamp_names = (
        "start_ns", "first_tma_done_ns", "last_tma_done_ns",
        "first_mma_done_ns", "last_mma_done_ns",
        "first_epilogue_start_ns", "last_store_done_ns", "kernel_exit_ns",
    )
    try:
        timestamps = {name: canonical_integer(str(fields[name])) for name in timestamp_names}
        smid = canonical_integer(str(fields["smid"]))
        metrics = {name: float(str(fields[name])) for name in METRICS}
    except (KeyError, TypeError, ValueError):
        return [*errors, "numeric field is malformed"], None
    if any(not math.isfinite(value) or value < 0 for value in metrics.values()):
        errors.append("derived metric is invalid")
    if not runtime:
        if any(timestamps.values()) or smid != 0 or any(metrics.values()):
            errors.append("static row contains runtime evidence")
        return errors, metrics
    start = timestamps["start_ns"]
    if start <= 0 or not 0 <= smid < 20:
        errors.append("runtime timer/SM identity is invalid")
    mode = case["mode"]
    has_tma = mode != "mma-only"
    has_mma = mode != "tma-only"
    has_epi = mode == "full"
    if has_tma and not (
        start <= timestamps["first_tma_done_ns"] <= timestamps["last_tma_done_ns"]
    ):
        errors.append("TMA timestamps are not monotonic")
    if not has_tma and (
        timestamps["first_tma_done_ns"] or timestamps["last_tma_done_ns"]
    ):
        errors.append("unexpected TMA timestamp")
    if has_mma and not (
        start <= timestamps["first_mma_done_ns"] <= timestamps["last_mma_done_ns"]
    ):
        errors.append("MMA timestamps are not monotonic")
    if not has_mma and (
        timestamps["first_mma_done_ns"] or timestamps["last_mma_done_ns"]
    ):
        errors.append("unexpected MMA timestamp")
    if has_epi and not (
        timestamps["first_mma_done_ns"] <= timestamps["first_epilogue_start_ns"]
        <= timestamps["last_mma_done_ns"] <= timestamps["last_store_done_ns"]
        == timestamps["kernel_exit_ns"]
    ):
        errors.append("epilogue timestamps are not monotonic")
    if not has_epi and (
        timestamps["first_epilogue_start_ns"] or timestamps["last_store_done_ns"]
    ):
        errors.append("unexpected epilogue timestamp")
    stop = (
        timestamps["last_tma_done_ns"] if mode == "tma-only"
        else timestamps["last_store_done_ns"] if mode == "full"
        else timestamps["last_mma_done_ns"]
    )
    if timestamps["kernel_exit_ns"] != stop:
        errors.append("kernel exit is not the selected terminal event")
    operations = case["k_tiles"] * case["output_tasks"]
    expected = {
        "first_tma_latency_ns": timestamps["first_tma_done_ns"] - start if has_tma else 0,
        "tma_completion_span_ns": (
            timestamps["last_tma_done_ns"] - timestamps["first_tma_done_ns"]
            if has_tma else 0),
        "first_mma_latency_ns": timestamps["first_mma_done_ns"] - start if has_mma else 0,
        "mma_completion_span_ns": (
            timestamps["last_mma_done_ns"] - timestamps["first_mma_done_ns"]
            if has_mma else 0),
        "epilogue_to_store_ns": (
            timestamps["last_store_done_ns"] - timestamps["first_epilogue_start_ns"]
            if has_epi else 0),
        "last_mma_to_store_ns": (
            timestamps["last_store_done_ns"] - timestamps["last_mma_done_ns"]
            if has_epi else 0),
        "total_measured_ns": stop - start,
    }
    expected["tma_interval_ns"] = (
        expected["tma_completion_span_ns"] / (operations - 1)
        if has_tma and operations > 1 else 0.0)
    expected["mma_interval_ns"] = (
        expected["mma_completion_span_ns"] / (operations - 1)
        if has_mma and operations > 1 else 0.0)
    for name, value in expected.items():
        if not math.isclose(metrics[name], float(value), rel_tol=2e-9, abs_tol=1e-6):
            errors.append(f"derived metric mismatch:{name}")
    return errors, metrics


def artifact_manifest_errors(root: Path) -> list[str]:
    errors: list[str] = []
    path = root / "artifact_sha256.txt"
    if not path.is_file():
        return ["artifact manifest is missing"]
    excluded = {"artifact_sha256.txt", "launcher.log", "launcher.pid"}
    expected = sorted(
        item.relative_to(root).as_posix() for item in root.rglob("*")
        if item.is_file() and item.relative_to(root).as_posix() not in excluded
    )
    recorded: dict[str, str] = {}
    for index, line in enumerate(path.read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if match is None:
            errors.append(f"artifact row {index} is malformed")
            continue
        digest, relative = match.groups()
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or relative in recorded:
            errors.append(f"artifact row {index} path is invalid")
            continue
        recorded[relative] = digest
    if sorted(recorded) != expected:
        errors.append("artifact manifest path set mismatch")
    for relative, digest in recorded.items():
        candidate = root / relative
        if candidate.is_file() and sha256_path(candidate) != digest:
            errors.append(f"artifact hash mismatch:{relative}")
    return errors


def path_has_suffix(value: object, suffix: tuple[str, ...]) -> bool:
    if not isinstance(value, str):
        return False
    parts = Path(value).parts
    return len(parts) >= len(suffix) and tuple(parts[-len(suffix):]) == suffix


def valid_binary_command(command: object, run_id: str, args: list[str]) -> bool:
    if not isinstance(command, list) or command[1:] != args:
        return False
    return path_has_suffix(
        command[0],
        ("results", "sm110_gemm_causal_campaign", run_id, "build",
         "tc5a_pipeline_dag"),
    )


def compile_command_valid(command: object, run_id: str, *, static_only: bool) -> bool:
    if not isinstance(command, list) or not command:
        return False
    base = [str(item) for item in command]
    if Path(base[0]).name != "nvcc" or base[1:3] != ["-O3", "-std=c++17"]:
        return False
    formal_middle = [
        "-gencode", "arch=compute_110a,code=sm_110a", "-I",
    ]
    allowed_static = [
        "-Xcompiler=-U_GNU_SOURCE", "-D_DEFAULT_SOURCE",
        "-D_POSIX_C_SOURCE=200809L", "-D_XOPEN_SOURCE=700",
        "-D_XOPEN_SOURCE_EXTENDED=1", "-D_LARGEFILE64_SOURCE=1",
        "-D_ATFILE_SOURCE=1",
    ]
    index = 3
    if static_only and base[index:index + len(allowed_static)] == allowed_static:
        index += len(allowed_static)
    elif not static_only and any(token in base for token in allowed_static):
        return False
    if base[index:index + 3] != formal_middle:
        return False
    index += 3
    if not path_has_suffix(base[index], ("GEMMsm110", "include")):
        return False
    index += 1
    if not path_has_suffix(
        base[index], ("microbench", "16_tc5a_pipeline_dag", "tc5a_pipeline_dag.cu")
    ):
        return False
    index += 1
    if base[index:index + 2] != ["-lcuda", "-o"]:
        return False
    index += 2
    return index + 1 == len(base) and path_has_suffix(
        base[index],
        ("results", "sm110_gemm_causal_campaign", run_id, "build",
         "tc5a_pipeline_dag"),
    )


def sass_stage_function_counts(
    sass_text: str,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Rebuild function-scoped instruction attribution without runner code."""

    errors: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in sass_text.splitlines():
        if "Function :" in line:
            current = line.split("Function :", 1)[1].strip()
            if not current or current in sections:
                errors.append("SASS function table is malformed")
                current = None
                continue
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
            marker = f"tc5a_pipeline_dag_kernelILi{stages}E"
            matches = [
                name for name in sections
                if marker in name and str(descriptor) in name
            ]
            if len(matches) != 1:
                errors.append(
                    f"{precision_id} stage-{stages} SASS function "
                    "attribution is ambiguous"
                )
                continue
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
            for token in REQUIRED_SASS_TOKENS:
                if counts[token] <= 0:
                    errors.append(
                        f"{precision_id} stage-{stages} SASS token "
                        f"missing:{token}"
                    )
            if counts["UTMASTG"] != 0:
                errors.append(
                    f"{precision_id} stage-{stages} SASS unexpectedly "
                    "contains UTMASTG"
                )
            if counts["instruction_descriptor_immediate"] <= 0:
                errors.append(
                    f"{precision_id} stage-{stages} SASS lacks its "
                    "instruction-descriptor immediate"
                )
            if counts["other_instruction_descriptor_immediate"] != 0:
                errors.append(
                    f"{precision_id} stage-{stages} SASS contains the "
                    "other precision's instruction-descriptor immediate"
                )
            result[f"{precision_id}.stage{stages}"] = counts
    return result, errors


def stats(values: Iterable[float]) -> dict[str, float]:
    rows = list(values)
    return {
        "median": statistics.median(rows), "minimum": min(rows),
        "maximum": max(rows), "mean": statistics.fmean(rows),
    }


def linear_fit(points: list[tuple[float, float]]) -> dict[str, float]:
    x_mean = statistics.fmean(x for x, _ in points)
    y_mean = statistics.fmean(y for _, y in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    intercept = y_mean - slope * x_mean
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    total = sum((y - y_mean) ** 2 for _, y in points)
    r_squared = 1.0 if total == 0 and residual == 0 else 1.0 - residual / total
    return {"intercept_ns": intercept, "slope_ns": slope, "r_squared": r_squared}


def predict_worker_ns(
    k_tiles: int, output_tasks: int, joint_first: float, joint_interval: float,
    epilogue_latency: float, buffers: int,
) -> float:
    previous_mma_done = 0.0
    epilogue_done: list[float] = []
    for task in range(output_tasks):
        first = joint_first if task == 0 else previous_mma_done + joint_interval
        if task >= buffers:
            first = max(first, epilogue_done[task - buffers] + joint_interval)
        last = first + (k_tiles - 1) * joint_interval
        start = max(last, epilogue_done[-1] if epilogue_done else 0.0)
        epilogue_done.append(start + epilogue_latency)
        previous_mma_done = last
    return epilogue_done[-1]


def rebuild_profile(
    results: list[dict[str, Any]], manifest: dict[str, Any], run_id: str,
    commit: str, precision_id: str,
) -> dict[str, Any]:
    precision_results = [
        row for row in results if row["precision_id"] == precision_id
    ]
    expected_count = EXPECTED_FAMILIES * (
        len(manifest["calibration_k_tiles"])
        + len(manifest["holdout_k_tiles"])
    )
    if len(precision_results) != expected_count:
        raise ValueError(f"{precision_id}: result matrix is incomplete")
    by_family_k = {
        (row["family_id"], row["k_tiles"]): row
        for row in precision_results
    }
    calibration = manifest["fit_contract"]["calibration_points"]

    def fit(family: str) -> dict[str, float]:
        return linear_fit([
            (float(k - 1), float(by_family_k[(family, k)]["metric_stats"]
                                 ["total_measured_ns"]["median"]))
            for k in calibration
        ])

    tma, mma, joint = fit("tma_s4"), fit("mma_s4"), fit("overlap_s4")
    epilogue_output_tasks = set(
        manifest["fit_contract"]["epilogue_calibration_output_tasks"]
    )
    epilogue = statistics.median(
        float(row["metric_stats"]["last_mma_to_store_ns"]["median"])
        for row in precision_results if row["mode"] == "full"
        and row["k_tiles"] in manifest["calibration_k_tiles"]
        and row["output_tasks"] in epilogue_output_tasks
    )
    joint_first = max(
        joint["intercept_ns"],
        statistics.median(
            float(by_family_k[("overlap_s4", k)]["metric_stats"]
                  ["first_mma_latency_ns"]["median"])
            for k in calibration
        ),
    )
    joint_interval = max(tma["slope_ns"], mma["slope_ns"], joint["slope_ns"])
    validations = []
    calibration_k = set(manifest["fit_contract"]["calibration_points"])
    calibration_o = set(manifest["fit_contract"]["calibration_output_tasks"])
    for row in precision_results:
        if row["mode"] != "full":
            continue
        actual = float(row["metric_stats"]["total_measured_ns"]["median"])
        predicted = predict_worker_ns(
            row["k_tiles"], row["output_tasks"], joint_first,
            joint_interval, epilogue, manifest["accumulator_buffers"],
        )
        validations.append({
            "case_id": row["case_id"],
            "split": "calibration" if row["k_tiles"] in calibration_k
                and row["output_tasks"] in calibration_o else "holdout",
            "k_tiles": int(row["k_tiles"]),
            "output_tasks": int(row["output_tasks"]),
            "actual_median_ns": actual, "predicted_ns": predicted,
            "relative_error": abs(predicted - actual) / actual,
        })
    max_cal = max(row["relative_error"] for row in validations
                  if row["split"] == "calibration")
    max_holdout = max(row["relative_error"] for row in validations
                      if row["split"] == "holdout")
    r2 = {"tma": tma["r_squared"], "mma": mma["r_squared"],
          "joint": joint["r_squared"]}
    contract = manifest["fit_contract"]
    qualified = bool(
        all(value >= contract["minimum_r_squared"] for value in r2.values())
        and max_cal <= contract["maximum_holdout_relative_error"]
        and max_holdout <= contract["maximum_holdout_relative_error"]
    )
    artifact_root = f"results/sm110_gemm_causal_campaign/{run_id}"
    artifact_paths = [
        "microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu",
        "GEMMsm110/include/sm110_ptx_helpers.cuh",
        "microbench/sm110_gemm_causal_campaign/contract_manifest.json",
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
        "trial_count_per_case": 10, "source_id": run_id,
        "expected_commit": commit,
        "source_path": "microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu",
        "source_locator": (
            f"91-case {precision_id} causal campaign; medians and "
            "predeclared fit"
        ),
        "input_residency": manifest["residency"],
        "stages": 4, "accumulator_buffers": 2, "resident_ctas_per_sm": 1,
        "maximum_k_tiles": 64, "maximum_output_tasks_per_worker": 32,
        "tma_first_completion_seconds": tma["intercept_ns"] * 1e-9,
        "tma_completion_interval_seconds": tma["slope_ns"] * 1e-9,
        "mma_first_completion_seconds": mma["intercept_ns"] * 1e-9,
        "mma_completion_interval_seconds": mma["slope_ns"] * 1e-9,
        "joint_first_mma_completion_seconds": joint_first * 1e-9,
        "joint_completion_interval_seconds": joint_interval * 1e-9,
        "epilogue_latency_seconds": epilogue * 1e-9,
        "component_r_squared": r2,
        "max_calibration_relative_error": max_cal,
        "max_holdout_relative_error": max_holdout,
        "fit_contract": contract, "validation": validations,
        "closure_qualified": qualified,
        "artifact_paths": artifact_paths,
        "applicable_sm_counts": [20],
        "applicable_hardware_ids": ["thor_t5000_sm110_20sm"],
        "applicable_operating_modes": ["MAXN"],
        "applicable_clock_hz": [1575000000.0],
        "timed_scope": "device_kernel",
    }


def rebuild_profiles(
    results: list[dict[str, Any]], manifest: dict[str, Any], run_id: str,
    commit: str,
) -> list[dict[str, Any]]:
    return [
        rebuild_profile(results, manifest, run_id, commit, precision_id)
        for precision_id in EXPECTED_PRECISIONS
    ]


def audit(
    root: Path, *, require_ncu: bool, expected_commit: str | None = None,
    static_only: bool = False,
    blob_loader: Callable[[str, str], bytes | None] = git_blob,
) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    common = (
        "run_spec.json", "environment.json", "environment_snapshots.jsonl",
        "plan.json", "manifest_snapshot.json", "static_contracts.json",
        "build/compile_command.json", "build/compile.log", "build/artifact.json",
        "build/binary.sha256", "build/csv_header.txt",
        "build/tc5a_pipeline_dag", "build/tc5a_pipeline_dag.sass.txt",
        "artifact_sha256.txt",
    )
    formal = (
        "summary.json", "progress.jsonl", "campaign_status.json", "COMPLETE",
        "pipeline_profiles.json",
    )
    required = (*common, *(formal if not static_only else ("STATIC_COMPLETE",)))
    for relative in required:
        add(errors, (root / relative).is_file(), f"missing:{relative}")
    if errors:
        return {"pass": False, "run_dir": str(root), "errors": errors,
                "warnings": warnings}
    errors.extend(artifact_manifest_errors(root))

    spec = read_json(root / "run_spec.json")
    environment = read_json(root / "environment.json")
    commit = recorded_commit(environment)
    add(errors, commit is not None, "environment has no valid commit")
    if expected_commit is not None:
        add(errors, commit == expected_commit, "recorded commit mismatch")
    run_id = root.name
    add(errors, spec.get("schema_version") == 2, "run spec schema mismatch")
    add(errors, spec.get("run_id") == run_id, "run ID mismatch")
    add(errors, spec.get("campaign") == "sm110_tc5a_causal_pipeline_dag",
        "campaign identity mismatch")
    add(errors, spec.get("expected_commit") == commit, "spec commit mismatch")
    add(errors, spec.get("generator") ==
        "microbench/sm110_gemm_causal_campaign/run_causal_campaign.py",
        "campaign generator path mismatch")
    add(errors, spec.get("contract_manifest") ==
        "microbench/sm110_gemm_causal_campaign/contract_manifest.json",
        "contract manifest path mismatch")
    add(errors, spec.get("precision_ids") == list(EXPECTED_PRECISIONS),
        "run spec precision contract mismatch")
    add(errors, spec.get("precision_contracts") == EXPECTED_PRECISION_CONTRACTS,
        "run spec precision descriptor contract mismatch")
    add(errors, spec.get("profile_count") == len(EXPECTED_PRECISIONS),
        "run spec profile count mismatch")
    add(errors, spec.get("case_count") == EXPECTED_CASES, "case count mismatch")
    add(errors, spec.get("family_count") == EXPECTED_FAMILIES,
        "family count mismatch")
    add(errors, spec.get("trials") == EXPECTED_TRIALS, "trial count mismatch")
    add(errors, spec.get("trial_timeout_seconds") == 120,
        "trial timeout mismatch")
    add(errors, spec.get("ncu_timeout_seconds") == 300, "NCU timeout mismatch")
    add(errors, spec.get("termination_grace_seconds") == 5,
        "termination grace mismatch")
    add(errors, spec.get("ncu_case_count") == EXPECTED_NCU_CASES,
        "NCU case count mismatch")
    add(errors, spec.get("ncu_policy") ==
        "four predeclared k16 attribution cases per precision",
        "NCU policy mismatch")
    add(errors, spec.get("static_only") is static_only, "static mode mismatch")
    if require_ncu:
        add(errors, spec.get("ncu_requested") is True, "NCU was not requested")
    if not static_only:
        for name in (
            "gpu_identity", "gpu_state", "git_head", "git_branch",
            "git_status", "nvcc", "nvidia_smi", "ncu", "power_mode",
        ):
            row = environment.get(name)
            add(errors, isinstance(row, dict) and row.get("returncode") == 0,
                f"environment probe failed:{name}")
        add(errors, "11.0" in str(environment.get("gpu_identity", {}).get("output", "")),
            "compute capability 11.0 is not recorded")
        add(errors, "MAXN" in str(environment.get("power_mode", {}).get("output", "")).upper(),
            "MAXN is not recorded")
        add(errors, not str(environment.get("git_status", {}).get("output", "")).strip(),
            "recorded worktree was dirty")
    try:
        snapshots = [json.loads(line) for line in
                     (root / "environment_snapshots.jsonl").read_text().splitlines()
                     if line]
    except json.JSONDecodeError:
        snapshots = []
        errors.append("environment snapshot journal is malformed")
    add(errors, bool(snapshots), "environment snapshot journal is empty")
    if snapshots:
        add(errors, snapshots[0] == environment,
            "first environment snapshot differs from environment.json")
        for index, snapshot in enumerate(snapshots):
            add(errors, recorded_commit(snapshot) == commit,
                f"environment snapshot commit mismatch:{index}")

    dependencies = spec.get("source_dependencies")
    add(errors, isinstance(dependencies, dict), "source dependencies missing")
    if isinstance(dependencies, dict):
        add(errors, set(dependencies) == EXPECTED_DEPENDENCIES,
            "source dependency set mismatch")
        for relative in EXPECTED_DEPENDENCIES:
            blob = blob_loader(commit, relative) if commit else None
            add(errors, blob is not None, f"Git blob missing:{relative}")
            if blob is not None:
                add(errors, dependencies.get(relative) == sha256_bytes(blob),
                    f"Git blob hash mismatch:{relative}")
        add(errors, spec.get("generator_sha256") == dependencies.get(
            "microbench/sm110_gemm_causal_campaign/run_causal_campaign.py"
        ), "generator hash differs from source dependencies")
        add(errors, spec.get("contract_manifest_sha256") == dependencies.get(
            "microbench/sm110_gemm_causal_campaign/contract_manifest.json"
        ), "manifest hash differs from source dependencies")
    source_blob = blob_loader(
        commit, "microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu"
    ) if commit else None
    if source_blob is not None:
        add(errors, b"ptx::mma_f16" in source_blob,
            "causal source does not implement the declared MMA")
        for token, message in (
            (b"fp16_f32", "FP16 precision ID"),
            (b"bf16_f32", "BF16 precision ID"),
            (b"CU_TENSOR_MAP_DATA_TYPE_FLOAT16", "FP16 tensor-map type"),
            (b"CU_TENSOR_MAP_DATA_TYPE_BFLOAT16", "BF16 tensor-map type"),
            (b"138412048U", "FP16 instruction descriptor"),
            (b"138413200U", "BF16 instruction descriptor"),
        ):
            add(errors, token in source_blob,
                f"causal source does not bind the declared {message}")
    manifest_relative = spec.get("contract_manifest")
    manifest_blob = blob_loader(commit, str(manifest_relative)) if commit else None
    add(errors, manifest_blob is not None, "manifest Git blob missing")
    if manifest_blob is not None:
        add(errors, sha256_bytes(manifest_blob) == spec.get("contract_manifest_sha256"),
            "manifest hash mismatch")
    try:
        manifest_value = json.loads(manifest_blob.decode()) if manifest_blob else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        manifest_value = None
    manifest = validate_manifest(manifest_value)
    add(errors, manifest is not None, "frozen manifest contract is invalid")
    if manifest is None:
        return {"pass": False, "run_dir": str(root), "errors": errors,
                "warnings": warnings}
    cases = expected_cases(manifest)
    add(errors, len(cases) == EXPECTED_CASES, "rebuilt case matrix size mismatch")
    add(errors, spec.get("cases") == cases, "run spec case matrix mismatch")
    add(errors, read_json(root / "manifest_snapshot.json") == manifest,
        "manifest snapshot differs from the committed contract")
    plan = read_json(root / "plan.json")
    expected_plan_fields = {
        "schema_version": 2,
        "schedule_id": manifest["schedule_id"],
        "precision_ids": list(EXPECTED_PRECISIONS),
        "profile_count": len(EXPECTED_PRECISIONS),
        "case_count": EXPECTED_CASES,
        "family_count": EXPECTED_FAMILIES,
        "raw_trial_count": EXPECTED_CASES * EXPECTED_TRIALS,
        "ncu_case_count": EXPECTED_NCU_CASES,
        "calibration_k_tiles": manifest["calibration_k_tiles"],
        "holdout_k_tiles": manifest["holdout_k_tiles"],
        "full_output_task_sweep": manifest["full_output_task_sweep"],
        "max_dynamic_smem_bytes": 196608,
        "max_hot_input_bytes": 3145728,
        "max_output_bytes": 4194304,
        "cases": cases,
    }
    add(errors, plan == expected_plan_fields,
        "campaign plan differs from independent reconstruction")

    command = read_json(root / "build/compile_command.json")
    add(errors, compile_command_valid(command, run_id, static_only=static_only),
        "compile command mismatch")
    binary = root / "build/tc5a_pipeline_dag"
    binary_text = (root / "build/binary.sha256").read_text().strip()
    match = re.fullmatch(r"([0-9a-f]{64})  tc5a_pipeline_dag", binary_text)
    add(errors, match is not None, "binary hash record is malformed")
    binary_sha = match.group(1) if match else None
    add(errors, binary_sha == sha256_path(binary), "binary hash mismatch")
    sass_path = root / "build/tc5a_pipeline_dag.sass.txt"
    sass = sass_path.read_text()
    sass_function_counts, sass_errors = sass_stage_function_counts(sass)
    errors.extend(sass_errors)
    sass_sha = sha256_path(sass_path)
    header = (root / "build/csv_header.txt").read_text().strip().split(",")
    add(errors, header == list(CSV_FIELDS), "retained CSV header mismatch")

    static_rows = read_json(root / "static_contracts.json")
    add(errors, isinstance(static_rows, list) and len(static_rows) == EXPECTED_CASES,
        "static contract matrix incomplete")
    static_by_id = {
        str(row.get("case_id")): row for row in static_rows
        if isinstance(row, dict)
    } if isinstance(static_rows, list) else {}
    expected_ids = {case["case_id"] for case in cases}
    add(errors, set(static_by_id) == expected_ids, "static case IDs mismatch")
    for case in cases:
        row = static_by_id.get(case["case_id"], {})
        expected_args = [*case["args"][:-1], "--contract-only", "--csv"]
        add(errors, valid_binary_command(row.get("command"), run_id, expected_args),
            f"{case['case_id']}: static command mismatch")
        try:
            parsed = parse_csv_row(str(row.get("stdout", "")))
        except ValueError:
            parsed = {}
            errors.append(f"{case['case_id']}: static stdout malformed")
        add(errors, parsed == row.get("fields"),
            f"{case['case_id']}: static parsed fields mismatch")
        row_errors, _ = field_errors(case, row.get("fields"), manifest, runtime=False)
        errors.extend(f"{case['case_id']}: {message}" for message in row_errors)
    if static_only:
        return {"pass": not errors, "run_dir": str(root), "errors": errors,
                "warnings": warnings, "case_count": len(cases),
                "static_only": True}

    summary = read_json(root / "summary.json")
    status = read_json(root / "campaign_status.json")
    add(errors, summary.get("status") == "complete", "summary is incomplete")
    add(errors, summary.get("schema_version") == 2,
        "summary schema mismatch")
    add(errors, summary.get("case_count") == EXPECTED_CASES,
        "summary case count mismatch")
    add(errors, summary.get("trial_count") == EXPECTED_CASES * EXPECTED_TRIALS,
        "summary raw-trial count mismatch")
    add(errors, summary.get("ncu_case_count") == EXPECTED_NCU_CASES,
        "summary NCU count mismatch")
    add(errors, summary.get("profile_count") == len(EXPECTED_PRECISIONS),
        "summary profile count mismatch")
    add(errors, status.get("status") == "complete", "campaign status incomplete")
    add(errors, status.get("completed_cases") == EXPECTED_CASES,
        "campaign completed-case count mismatch")
    add(errors, status.get("profile_count") == len(EXPECTED_PRECISIONS),
        "campaign status profile count mismatch")
    try:
        progress = [json.loads(line) for line in
                    (root / "progress.jsonl").read_text().splitlines() if line]
    except json.JSONDecodeError:
        progress = []
        errors.append("progress journal malformed")
    add(errors, bool(progress) and progress[-1].get("status") == "complete",
        "progress journal lacks terminal completion")
    results = summary.get("results")
    add(errors, isinstance(results, list) and len(results) == EXPECTED_CASES,
        "summary result matrix incomplete")
    result_by_id = {
        str(row.get("case_id")): row for row in results if isinstance(row, dict)
    } if isinstance(results, list) else {}
    add(errors, set(result_by_id) == expected_ids, "result case IDs mismatch")
    verified_results: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        result = result_by_id.get(case_id, {})
        for name in (
            "precision_id", "tensor_map_data_type",
            "instruction_descriptor_u32", "family_id", "mode", "stages",
            "k_tiles", "output_tasks",
        ):
            add(errors, result.get(name) == case[name],
                f"{case_id}: result metadata mismatch:{name}")
        add(errors, result.get("status") == "ok", f"{case_id}: status is not ok")
        add(errors, result.get("trial_count") == EXPECTED_TRIALS,
            f"{case_id}: trial count mismatch")
        add(errors, result.get("trial_timeout_seconds") == 120,
            f"{case_id}: timeout mismatch")
        add(errors, result.get("binary_sha256") == binary_sha,
            f"{case_id}: binary hash mismatch")
        add(errors, result.get("sass_sha256") == sass_sha,
            f"{case_id}: SASS hash mismatch")
        add(errors, result.get("sass_tokens") == ["LDTM.", "UTCBAR", "UTCHMMA", "UTMALDG.2D"],
            f"{case_id}: SASS token list mismatch")
        add(errors, result.get("sass_function_counts") == sass_function_counts,
            f"{case_id}: function-scoped SASS counts mismatch")
        source_hash = dependencies.get(result.get("source_path")) \
            if isinstance(dependencies, dict) else None
        helper_hash = dependencies.get(result.get("helper_path")) \
            if isinstance(dependencies, dict) else None
        add(errors, result.get("source_sha256") == source_hash,
            f"{case_id}: source hash mismatch")
        add(errors, result.get("helper_sha256") == helper_hash,
            f"{case_id}: helper hash mismatch")
        expected_fingerprint = sha256_json({
            "case": case, "binary_sha256": binary_sha,
            "source_sha256": source_hash, "helper_sha256": helper_hash,
            "trial_count": 10,
            "ncu": bool(spec.get("ncu_requested")) and case["ncu_selected"],
        })
        add(errors, result.get("fingerprint") == expected_fingerprint,
            f"{case_id}: fingerprint mismatch")
        trials_path = root / "cases" / case_id / "trials.jsonl"
        add(errors, trials_path.is_file(), f"{case_id}: trials missing")
        try:
            trials = [json.loads(line) for line in trials_path.read_text().splitlines()
                      if line] if trials_path.is_file() else []
        except json.JSONDecodeError:
            trials = []
            errors.append(f"{case_id}: trial JSONL malformed")
        add(errors, len(trials) == EXPECTED_TRIALS,
            f"{case_id}: trial matrix incomplete")
        values = {name: [] for name in METRICS}
        for trial_index, trial in enumerate(trials, 1):
            add(errors, trial.get("trial") == trial_index,
                f"{case_id}: trial index mismatch:{trial_index}")
            add(errors, valid_binary_command(trial.get("command"), run_id, case["args"]),
                f"{case_id}: trial command mismatch:{trial_index}")
            add(errors, trial.get("returncode") == 0
                and trial.get("timeout_seconds") == 120
                and trial.get("timed_out") is False
                and trial.get("termination_failed") is False,
                f"{case_id}: bounded trial contract failed:{trial_index}")
            try:
                fields = parse_csv_row(str(trial.get("raw_stdout", "")))
            except ValueError:
                fields = {}
                errors.append(f"{case_id}: raw stdout malformed:{trial_index}")
            add(errors, fields == trial.get("fields"),
                f"{case_id}: stored fields mismatch:{trial_index}")
            row_errors, audited = field_errors(case, fields, manifest, runtime=True)
            errors.extend(
                f"{case_id}: trial {trial_index}: {message}" for message in row_errors
            )
            if audited is not None:
                add(errors, audited == trial.get("audited_metrics"),
                    f"{case_id}: audited metrics mismatch:{trial_index}")
                for name in METRICS:
                    values[name].append(audited[name])
        if all(len(values[name]) == EXPECTED_TRIALS for name in METRICS):
            expected_stats = {name: stats(rows) for name, rows in values.items()}
            add(errors, result.get("metric_stats") == expected_stats,
                f"{case_id}: metric statistics mismatch")
        ncu = result.get("ncu")
        if require_ncu and case["ncu_selected"]:
            add(errors, isinstance(ncu, dict) and ncu.get("pass") is True,
                f"{case_id}: NCU evidence missing")
            if isinstance(ncu, dict):
                report = root / "cases" / case_id / str(ncu.get("report_path", ""))
                log = root / "cases" / case_id / "ncu/profile.log"
                add(errors, report.is_file() and log.is_file(),
                    f"{case_id}: NCU artifact missing")
                if report.is_file():
                    add(errors, sha256_path(report) == ncu.get("report_sha256"),
                        f"{case_id}: NCU report hash mismatch")
                if log.is_file():
                    add(errors, sha256_path(log) == ncu.get("log_sha256"),
                        f"{case_id}: NCU log hash mismatch")
                command = ncu.get("command")
                add(errors, isinstance(command, list) and "--launch-skip" in command
                    and "--launch-count" in command
                    and command[-len(case["args"]):] == case["args"],
                    f"{case_id}: NCU command mismatch")
        elif case["ncu_selected"] and not require_ncu:
            warnings.append(f"NCU not required during audit:{case_id}")
        verified_results.append(result)

    recorded_bundle = read_json(root / "pipeline_profiles.json")
    try:
        rebuilt_profiles = rebuild_profiles(
            verified_results, manifest, run_id, commit or ""
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        rebuilt_profiles = []
        errors.append(f"pipeline profiles could not be rebuilt:{error}")
    expected_bundle = {
        "schema_version": 2,
        "run_id": run_id,
        "expected_commit": commit or "",
        "pipeline_profiles": rebuilt_profiles,
    }
    add(errors, recorded_bundle == expected_bundle,
        "pipeline profiles differ from independent reconstruction")
    profile_qualified_by_precision = {
        profile["precision_ids"][0]: bool(profile["closure_qualified"])
        for profile in rebuilt_profiles
    }
    for precision_id, qualified in profile_qualified_by_precision.items():
        if not qualified:
            warnings.append(
                f"{precision_id}: causal measurements are complete but the "
                "predeclared DAG fit is quarantined"
            )
    aggregate_qualified = bool(profile_qualified_by_precision) and all(
        profile_qualified_by_precision.values()
    )
    add(errors, summary.get("profile_qualified_by_precision") ==
        profile_qualified_by_precision,
        "summary per-precision profile qualification mismatch")
    add(errors, status.get("profile_qualified_by_precision") ==
        profile_qualified_by_precision,
        "status per-precision profile qualification mismatch")
    add(errors, summary.get("profile_qualified") == aggregate_qualified,
        "summary profile qualification mismatch")
    add(errors, status.get("profile_qualified") == aggregate_qualified,
        "status profile qualification mismatch")
    return {
        "pass": not errors, "run_dir": str(root), "errors": errors,
        "warnings": sorted(set(warnings)), "case_count": len(cases),
        "trial_count": EXPECTED_CASES * EXPECTED_TRIALS,
        "ncu_case_count": EXPECTED_NCU_CASES,
        "profile_count": len(rebuilt_profiles),
        "profile_qualified_by_precision": profile_qualified_by_precision,
        "profile_qualified": aggregate_qualified,
        "static_only": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-ncu", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    if args.expected_commit is not None and not COMMIT_RE.fullmatch(args.expected_commit):
        parser.error("--expected-commit must be 40 lowercase hex characters")
    result = audit(
        args.run_dir, require_ncu=args.require_ncu,
        expected_commit=args.expected_commit, static_only=args.static_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
