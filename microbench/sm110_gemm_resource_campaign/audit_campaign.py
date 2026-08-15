#!/usr/bin/env python3
"""Independent, relocation-safe audit of an SM110 TMA resource campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
EXPECTED_SOURCE_DEPENDENCIES = {
    "microbench/15_tma_ab_contract_bandwidth/"
    "tma_ab_contract_bandwidth.cu",
    "microbench/sm110_gemm_resource_campaign/contract_manifest.json",
    "microbench/sm110_gemm_resource_campaign/run_resource_campaign.py",
}
EXPECTED_PRECISIONS = {
    "fp16_f32", "bf16_f32", "tf32_f32", "e4m3_f32", "e5m2_f32",
    "e3m2_f32", "e2m3_f32", "e2m1_f32", "mxfp4_f32",
    "nvfp4_f32", "s8_s32", "u8_s32",
}
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode())


def audit_artifact_manifest(root: Path, errors: list[str]) -> None:
    manifest_path = root / "artifact_sha256.txt"
    excluded = {"artifact_sha256.txt", "launcher.log", "launcher.pid"}
    expected_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    )
    recorded: dict[str, str] = {}
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if match is None:
            errors.append(f"artifact manifest row {line_number} is malformed")
            continue
        digest, relative = match.groups()
        path = Path(relative)
        if (
            path.is_absolute() or not path.parts or ".." in path.parts
            or relative in recorded or relative in excluded
        ):
            errors.append(f"artifact manifest row {line_number} has invalid path")
            continue
        recorded[relative] = digest
    add(errors, sorted(recorded) == expected_paths,
        "artifact manifest path set mismatch")
    for relative, digest in recorded.items():
        path = root / relative
        if path.is_file():
            add(errors, sha256_path(path) == digest,
                f"artifact hash mismatch:{relative}")


def add(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def parse_kv(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        for match in re.finditer(
            r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", line
        ):
            fields[match.group(1)] = match.group(2)
    return fields


def recorded_commit(environment: object) -> str | None:
    if not isinstance(environment, dict):
        return None
    row = environment.get("git_head")
    if not isinstance(row, dict) or row.get("returncode") != 0:
        return None
    value = str(row.get("output", "")).strip()
    return value if COMMIT_RE.fullmatch(value) else None


def git_blob(commit: str, relative: str) -> bytes | None:
    path = Path(relative)
    if (
        not COMMIT_RE.fullmatch(commit)
        or path.is_absolute()
        or not path.parts
        or ".." in path.parts
    ):
        return None
    try:
        completed = subprocess.run(
            ["git", "show", "--no-ext-diff", f"{commit}:{path.as_posix()}"],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def round_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def validate_manifest(manifest: object) -> dict[str, Any] | None:
    if not isinstance(manifest, dict):
        return None
    if manifest.get("schema_version") != 1:
        return None
    if manifest.get("expected_sm_count") != 20:
        return None
    if manifest.get("row_stride_elements") != [1024, 2048, 4096]:
        return None
    residencies = manifest.get("residencies")
    if not isinstance(residencies, dict) or set(residencies) != {
        "hot_l2", "cold_dram"
    }:
        return None
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
        return None
    families = manifest.get("families")
    if not isinstance(families, list) or len(families) != 9:
        return None
    family_ids: set[str] = set()
    precision_ids: set[str] = set()
    for family in families:
        if not isinstance(family, dict):
            return None
        family_id = family.get("family_id")
        if (
            not isinstance(family_id, str)
            or not re.fullmatch(r"[a-z0-9_]+", family_id)
            or family_id in family_ids
        ):
            return None
        family_ids.add(family_id)
        for field in ("bm", "bn", "bk", "value_bits", "stages", "threads"):
            value = family.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                return None
        if family.get("value_bits") not in {4, 8, 16, 32}:
            return None
        if family.get("scale_block") not in {0, 16, 32}:
            return None
        if family.get("stages") not in {1, 2, 4, 8}:
            return None
        controller = family.get("controller_thread")
        if (
            not isinstance(controller, int)
            or isinstance(controller, bool)
            or not 0 <= controller < int(family["threads"])
        ):
            return None
        precisions = family.get("precision_ids")
        layouts = family.get("input_transport_layouts")
        if not isinstance(precisions, list) or not precisions:
            return None
        if not isinstance(layouts, list) or not layouts:
            return None
        precision_ids.update(str(value) for value in precisions)
    return manifest if precision_ids == EXPECTED_PRECISIONS else None


def family_contract(family: dict[str, Any]) -> dict[str, int | str] | None:
    value_bits = int(family["value_bits"])
    bk = int(family["bk"])
    if bk * value_bits % 8:
        return None
    row_bytes = bk * value_bits // 8
    request_row_bytes = min(row_bytes, 128)
    if request_row_bytes not in {32, 64, 128}:
        return None
    if row_bytes % request_row_bytes:
        return None
    chunks = row_bytes // request_row_bytes
    a_value = int(family["bm"]) * row_bytes
    b_value = int(family["bn"]) * row_bytes
    a_scale = 0
    b_scale = 0
    scale_block = int(family["scale_block"])
    if scale_block:
        groups = round_up(ceil_div(bk, scale_block), 4)
        a_scale = round_up(int(family["bm"]), 128) * groups
        b_scale = round_up(int(family["bn"]), 128) * groups
        if a_scale % 32 or b_scale % 32:
            return None
    stage = a_value + b_value + a_scale + b_scale
    return {
        "value_chunks": chunks,
        "value_swizzle": f"{request_row_bytes}B",
        "requests_per_stage": 2 * chunks + (2 if scale_block else 0),
        "a_value_bytes": a_value,
        "b_value_bytes": b_value,
        "a_scale_bytes": a_scale,
        "b_scale_bytes": b_scale,
        "stage_bytes": stage,
        "dynamic_smem_bytes": int(family["stages"]) * stage,
    }


def expected_allocation(
    family: dict[str, Any], row_stride: int, tiles: int,
    contract: dict[str, int | str],
) -> int:
    value_bytes = row_stride * int(family["value_bits"]) // 8
    result = value_bytes * (
        int(family["bm"]) + int(family["bn"])
    ) * tiles
    result += (
        int(contract["a_scale_bytes"])
        + int(contract["b_scale_bytes"])
    ) * tiles
    return result


def expected_cases(manifest: dict[str, Any]) -> list[dict[str, Any]] | None:
    cases: list[dict[str, Any]] = []
    for family in manifest["families"]:
        contract = family_contract(family)
        if contract is None:
            return None
        stage = int(contract["stage_bytes"])
        for stride in manifest["row_stride_elements"]:
            if stride < int(family["bk"]):
                return None
            for residency in ("hot_l2", "cold_dram"):
                residency_row = manifest["residencies"][residency]
                target = int(residency_row["target_working_set_bytes"])
                tiles = max(1, target // stage)
                working_set = tiles * stage
                if residency == "hot_l2":
                    blocks = 1
                    iters = 4096
                    warmup = tiles + int(family["stages"])
                    resource = (
                        "tma.smem_ingress.contract."
                        f"{family['family_id']}.stride{stride}.per_sm"
                    )
                else:
                    blocks = 20
                    iters = max(256, ceil_div(512 << 20, blocks * stage))
                    warmup = 64
                    resource = (
                        "tma.hbm.contract."
                        f"{family['family_id']}.stride{stride}"
                    )
                case_id = f"{family['family_id']}_stride{stride}_{residency}"
                args = [
                    "--case-id", case_id,
                    "--mode", str(residency_row["binary_mode"]),
                    "--bm", str(family["bm"]),
                    "--bn", str(family["bn"]),
                    "--bk", str(family["bk"]),
                    "--value-bits", str(family["value_bits"]),
                    "--scale-block", str(family["scale_block"]),
                    "--stages", str(family["stages"]),
                    "--row-stride-elements", str(stride),
                    "--threads", str(family["threads"]),
                    "--controller-thread", str(family["controller_thread"]),
                    "--bytes", str(target),
                    "--iters", str(iters),
                    "--warmup-iters", str(warmup),
                    "--expected-sm-count", "20",
                ]
                args += (
                    ["--blocks", "1"]
                    if residency == "hot_l2"
                    else ["--blocks-per-sm", "1"]
                )
                expected = {
                    "case_id": case_id,
                    "mode": residency_row["binary_mode"],
                    "bm": int(family["bm"]),
                    "bn": int(family["bn"]),
                    "bk": int(family["bk"]),
                    "value_bits": int(family["value_bits"]),
                    "scale_block": int(family["scale_block"]),
                    "stages": int(family["stages"]),
                    "requests_per_stage":
                        int(contract["requests_per_stage"]),
                    "value_chunks": int(contract["value_chunks"]),
                    "value_swizzle": str(contract["value_swizzle"]),
                    "row_stride_elements": int(stride),
                    "threads": int(family["threads"]),
                    "controller_thread": int(family["controller_thread"]),
                    "blocks": blocks,
                    "sm_count": 20,
                    "initialization": "cuda_memset_zero",
                    "a_value_bytes": int(contract["a_value_bytes"]),
                    "b_value_bytes": int(contract["b_value_bytes"]),
                    "a_scale_bytes": int(contract["a_scale_bytes"]),
                    "b_scale_bytes": int(contract["b_scale_bytes"]),
                    "stage_bytes": stage,
                    "dynamic_smem_bytes":
                        int(contract["dynamic_smem_bytes"]),
                    "working_set_bytes": working_set,
                    "allocation_bytes": expected_allocation(
                        family, int(stride), tiles, contract
                    ),
                    "total_tiles": tiles,
                    "iters": iters,
                    "warmup_iters": warmup,
                }
                cases.append({
                    "case_id": case_id,
                    "family_id": family["family_id"],
                    "precision_ids": family["precision_ids"],
                    "input_transport_layouts":
                        family["input_transport_layouts"],
                    "row_stride_elements": stride,
                    "residency": residency,
                    "resource": resource,
                    "args": args,
                    "expected": expected,
                    "ncu_selected": stride == 2048,
                })
    return cases if len(cases) == 54 else None


def field_errors(
    case: dict[str, Any], fields: object, *, runtime: bool,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(fields, dict):
        return ["fields are not an object"]
    expected = case["expected"]
    string_fields = {
        "case_id", "mode", "value_swizzle", "initialization",
    }
    for name, value in expected.items():
        actual = fields.get(name)
        if name in string_fields:
            if actual != str(value):
                errors.append(f"field mismatch:{name}")
        else:
            try:
                equal = int(str(actual)) == int(value)
            except (TypeError, ValueError):
                equal = False
            if not equal:
                errors.append(f"field mismatch:{name}")
    if not runtime:
        if fields.get("contract_only") != "1":
            errors.append("static contract did not use contract-only mode")
        return errors
    expected_sms = 1 if case["residency"] == "hot_l2" else 20
    try:
        if int(fields.get("unique_smid_count", 0)) != expected_sms:
            errors.append("SM coverage mismatch")
        start = int(fields.get("globaltimer_start_min_ns", 0))
        stop = int(fields.get("globaltimer_stop_max_ns", 0))
        elapsed = int(fields.get("globaltimer_elapsed_ns", 0))
        requested = int(fields.get("requested_bytes", 0))
        reported = float(fields.get("bytes_per_second", "nan"))
        expected_requested = (
            int(expected["blocks"])
            * int(expected["iters"])
            * int(expected["stage_bytes"])
        )
        if start <= 0 or stop <= start or elapsed != stop - start:
            errors.append("globaltimer interval mismatch")
        if requested != expected_requested:
            errors.append("requested-byte arithmetic mismatch")
        if elapsed <= 0 or requested <= 0 or not math.isfinite(reported):
            errors.append("invalid timing fields")
        else:
            recalculated = requested * 1.0e9 / elapsed
            if abs(reported - recalculated) > max(1.0, recalculated * 2e-6):
                errors.append("rate arithmetic mismatch")
        if int(fields.get("occupancy_blocks_per_sm", 0)) <= 0:
            errors.append("nonpositive occupancy")
    except (TypeError, ValueError, OverflowError):
        errors.append("runtime numeric field is malformed")
    return errors


def function_sass(sass: str, name: str) -> str | None:
    marker = re.search(rf"^\s*Function\s*:\s*[^\n]*{re.escape(name)}[^\n]*$",
                       sass, flags=re.MULTILINE)
    if marker is None:
        return None
    next_marker = re.search(
        r"^\s*Function\s*:", sass[marker.end():], flags=re.MULTILINE
    )
    end = marker.end() + next_marker.start() if next_marker else len(sass)
    return sass[marker.start():end]


def valid_binary_command(command: object, run_id: str,
                         expected_args: list[str]) -> bool:
    if not isinstance(command, list) or not all(
        isinstance(value, str) for value in command
    ):
        return False
    if len(command) != 1 + len(expected_args):
        return False
    binary = Path(command[0])
    suffix = Path("results/sm110_gemm_resource_campaign") / run_id
    expected_suffix = suffix / "build/tma_ab_contract_bandwidth"
    return (
        binary.is_absolute()
        and tuple(binary.parts[-len(expected_suffix.parts):])
        == expected_suffix.parts
        and command[1:] == expected_args
    )


def audit_compile_command(command: object, run_id: str) -> bool:
    if not isinstance(command, list) or not all(
        isinstance(value, str) for value in command
    ):
        return False
    if len(command) != 9:
        return False
    source = Path(command[5])
    output = Path(command[8])
    expected_source = Path(
        "microbench/15_tma_ab_contract_bandwidth/"
        "tma_ab_contract_bandwidth.cu"
    )
    expected_output = (
        Path("results/sm110_gemm_resource_campaign")
        / run_id
        / "build/tma_ab_contract_bandwidth"
    )
    return (
        Path(command[0]).is_absolute()
        and Path(command[0]).name == "nvcc"
        and command[1:5] == [
            "-O3", "-std=c++17", "-gencode",
            "arch=compute_110a,code=sm_110a",
        ]
        and command[6:8] == ["-lcuda", "-o"]
        and source.is_absolute()
        and output.is_absolute()
        and tuple(source.parts[-len(expected_source.parts):])
        == expected_source.parts
        and tuple(output.parts[-len(expected_output.parts):])
        == expected_output.parts
    )


def ncu_expected_args(case: dict[str, Any]) -> list[str]:
    args = list(case["args"])
    ncu_working_set = (
        (1 << 20) if case["residency"] == "hot_l2" else (64 << 20)
    )
    ncu_tiles = max(1, ncu_working_set // int(case["expected"]["stage_bytes"]))
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


def audit_ncu(
    root: Path, run_id: str, case: dict[str, Any], result: dict[str, Any],
    errors: list[str], require_ncu: bool,
) -> None:
    ncu = result.get("ncu")
    selected = bool(case["ncu_selected"])
    if not isinstance(ncu, dict):
        errors.append(f"{case['case_id']}: missing NCU object")
        return
    if not selected:
        add(errors, ncu.get("selected") is False,
            f"{case['case_id']}: unexpected NCU selection")
        return
    if not require_ncu:
        return
    add(errors, ncu.get("selected") is True,
        f"{case['case_id']}: NCU was not selected")
    add(errors, ncu.get("pass") is True,
        f"{case['case_id']}: NCU did not pass")
    add(errors, ncu.get("returncode") == 0,
        f"{case['case_id']}: NCU return code is nonzero")
    add(errors, ncu.get("permission_denied") is False,
        f"{case['case_id']}: NCU permission denied")
    add(errors, ncu.get("timed_out") is False,
        f"{case['case_id']}: NCU timed out")
    add(errors, ncu.get("termination_failed") is False,
        f"{case['case_id']}: NCU termination failed")
    add(errors, ncu.get("timeout_seconds") == 300,
        f"{case['case_id']}: NCU timeout contract mismatch")
    add(errors, ncu.get("set") == "basic",
        f"{case['case_id']}: NCU set mismatch")
    add(errors, ncu.get("kernel_name_base") == "demangled",
        f"{case['case_id']}: NCU kernel-name base mismatch")
    add(errors, ncu.get("kernel_name_regex") == "tma_ab_contract_kernel",
        f"{case['case_id']}: NCU kernel filter mismatch")
    add(errors, ncu.get("launch_count") == 1,
        f"{case['case_id']}: NCU launch count mismatch")
    command = ncu.get("command")
    valid_command = isinstance(command, list) and all(
        isinstance(value, str) for value in command
    )
    if valid_command:
        assert isinstance(command, list)
        try:
            export_index = command.index("--export")
            binary_index = export_index + 2
        except ValueError:
            valid_command = False
        else:
            if binary_index >= len(command):
                valid_command = False
                report_base = Path()
                binary = Path()
            else:
                report_base = Path(command[export_index + 1])
                binary = Path(command[binary_index])
            prefix = (
                Path("results/sm110_gemm_resource_campaign")
                / run_id
                / "cases"
                / str(case["case_id"])
            )
            report_suffix = prefix / "ncu/profile"
            binary_suffix = (
                Path("results/sm110_gemm_resource_campaign")
                / run_id
                / "build/tma_ab_contract_bandwidth"
            )
            valid_command = (
                valid_command
                and Path(command[0]).is_absolute()
                and Path(command[0]).name == "ncu"
                and command[1:13] == [
                    "--set", "basic", "--target-processes", "all",
                    "--kernel-name-base", "demangled",
                    "--kernel-name", "regex:tma_ab_contract_kernel",
                    "--launch-count", "1", "--force-overwrite", "--export",
                ]
                and export_index == 12
                and binary_index == 14
                and report_base.is_absolute()
                and binary.is_absolute()
                and tuple(report_base.parts[-len(report_suffix.parts):])
                == report_suffix.parts
                and tuple(binary.parts[-len(binary_suffix.parts):])
                == binary_suffix.parts
                and command[binary_index + 1:] == ncu_expected_args(case)
            )
    add(errors, valid_command,
        f"{case['case_id']}: NCU command mismatch")
    ncu_dir = root / "cases" / str(case["case_id"]) / "ncu"
    report = ncu_dir / "profile.ncu-rep"
    log = ncu_dir / "profile.log"
    add(errors, report.is_file(), f"{case['case_id']}: NCU report missing")
    add(errors, log.is_file(), f"{case['case_id']}: NCU log missing")
    if report.is_file():
        add(errors, sha256_path(report) == ncu.get("report_sha256"),
            f"{case['case_id']}: NCU report hash mismatch")
    if log.is_file():
        add(errors, sha256_path(log) == ncu.get("log_sha256"),
            f"{case['case_id']}: NCU log hash mismatch")


def audit(root: Path, *, require_ncu: bool,
          expected_commit: str | None = None) -> dict[str, object]:
    root = root.resolve()
    errors: list[str] = []
    required = (
        "run_spec.json", "summary.json", "environment.json",
        "environment_snapshots.jsonl", "campaign_status.json",
        "progress.jsonl", "COMPLETE",
        "artifact_sha256.txt",
        "static_contracts.json", "build/compile_command.json",
        "build/compile.log", "build/tma_ab_contract_bandwidth.sass.txt",
        "build/tma_ab_contract_bandwidth", "build/binary.sha256",
        "build/artifact.json",
    )
    for relative in required:
        add(errors, (root / relative).is_file(), f"missing:{relative}")
    if errors:
        return {"pass": False, "run_dir": str(root), "errors": errors}
    audit_artifact_manifest(root, errors)

    spec = read_json(root / "run_spec.json")
    summary = read_json(root / "summary.json")
    environment = read_json(root / "environment.json")
    try:
        environment_snapshots = [
            json.loads(line)
            for line in (root / "environment_snapshots.jsonl").read_text().splitlines()
            if line
        ]
    except json.JSONDecodeError:
        environment_snapshots = []
        errors.append("environment snapshots are not JSONL")
    add(errors, bool(environment_snapshots), "environment snapshots are empty")
    if environment_snapshots:
        add(errors, environment_snapshots[0] == environment,
            "first environment snapshot differs from environment.json")
    campaign_status = read_json(root / "campaign_status.json")
    add(errors, campaign_status.get("status") == "complete",
        "campaign status is not complete")
    add(errors, campaign_status.get("completed_cases") == 54,
        "campaign status completed-case count mismatch")
    add(errors, campaign_status.get("total_cases") == 54,
        "campaign status total-case count mismatch")
    try:
        progress_rows = [
            json.loads(line)
            for line in (root / "progress.jsonl").read_text().splitlines()
            if line
        ]
    except json.JSONDecodeError:
        progress_rows = []
        errors.append("progress journal is not JSONL")
    add(errors, bool(progress_rows), "progress journal is empty")
    if progress_rows:
        add(errors, progress_rows[-1].get("status") == "complete",
            "progress journal has no terminal complete row")
    run_id = root.name
    commit = recorded_commit(environment)
    add(errors, commit is not None, "environment has no valid git commit")
    if expected_commit is not None:
        add(errors, commit == expected_commit, "recorded commit mismatch")
    for probe in (
        "gpu_identity", "gpu_state", "nvcc", "git_head", "git_branch",
        "git_status", "power_mode",
    ):
        row = environment.get(probe, {})
        add(errors, isinstance(row, dict) and row.get("returncode") == 0,
            f"environment probe failed:{probe}")
    if require_ncu:
        row = environment.get("ncu", {})
        add(errors, isinstance(row, dict) and row.get("returncode") == 0,
            "environment probe failed:ncu")
    add(errors, "11.0" in str(environment.get("gpu_identity", {}).get("output", "")),
        "compute capability 11.0 is not recorded")
    add(errors, "MAXN" in str(environment.get("power_mode", {}).get("output", "")).upper(),
        "MAXN is not recorded")
    add(errors, not str(environment.get("git_status", {}).get("output", "")).strip(),
        "recorded worktree was dirty")
    for snapshot_index, snapshot in enumerate(environment_snapshots):
        if not isinstance(snapshot, dict):
            errors.append(
                f"environment snapshot {snapshot_index} is not an object"
            )
            continue
        add(errors, recorded_commit(snapshot) == commit,
            f"environment snapshot {snapshot_index} commit mismatch")
        for probe in (
            "gpu_identity", "gpu_state", "nvcc", "git_head", "git_branch",
            "git_status", "power_mode",
        ):
            row = snapshot.get(probe, {})
            add(errors, isinstance(row, dict) and row.get("returncode") == 0,
                f"environment snapshot {snapshot_index} probe failed:{probe}")
        if require_ncu:
            row = snapshot.get("ncu", {})
            add(errors, isinstance(row, dict) and row.get("returncode") == 0,
                f"environment snapshot {snapshot_index} probe failed:ncu")
        add(errors, "11.0" in str(
            snapshot.get("gpu_identity", {}).get("output", "")
        ), f"environment snapshot {snapshot_index} is not CC 11.0")
        add(errors, "MAXN" in str(
            snapshot.get("power_mode", {}).get("output", "")
        ).upper(), f"environment snapshot {snapshot_index} is not MAXN")
        add(errors, not str(
            snapshot.get("git_status", {}).get("output", "")
        ).strip(), f"environment snapshot {snapshot_index} was dirty")

    add(errors, spec.get("schema_version") == 1, "invalid spec schema")
    add(errors, spec.get("run_id") == run_id, "run ID mismatch")
    add(errors, spec.get("campaign") == "sm110_exact_tma_resource_contracts",
        "campaign identity mismatch")
    add(errors, spec.get("expected_sm_count") == 20,
        "expected SM count mismatch")
    add(errors, spec.get("trials") == 10, "trial contract mismatch")
    add(errors, spec.get("case_count") == 54, "case count contract mismatch")
    add(errors, spec.get("family_count") == 9, "family count mismatch")
    add(errors, spec.get("static_only") is False,
        "static-only result cannot qualify as hardware evidence")
    add(errors, spec.get("ncu_policy") ==
        "row_stride=2048 for every family and residency", "NCU policy mismatch")
    add(errors, spec.get("termination_grace_seconds") == 5,
        "termination grace mismatch")
    add(errors, spec.get("trial_timeout_seconds") == 120,
        "trial timeout contract mismatch")
    add(errors, spec.get("ncu_timeout_seconds") == 300,
        "NCU timeout contract mismatch")
    add(errors, spec.get("generator") ==
        "microbench/sm110_gemm_resource_campaign/run_resource_campaign.py",
        "generator path mismatch")
    add(errors, spec.get("contract_manifest") ==
        "microbench/sm110_gemm_resource_campaign/contract_manifest.json",
        "manifest path mismatch")

    manifest_blob = (
        git_blob(commit, str(spec.get("contract_manifest", "")))
        if commit is not None else None
    )
    generator_blob = (
        git_blob(commit, str(spec.get("generator", "")))
        if commit is not None else None
    )
    add(errors, manifest_blob is not None, "recorded manifest is unavailable")
    add(errors, generator_blob is not None, "recorded generator is unavailable")
    if manifest_blob is not None:
        add(errors, sha256_bytes(manifest_blob) ==
            spec.get("contract_manifest_sha256"), "manifest hash mismatch")
    if generator_blob is not None:
        add(errors, sha256_bytes(generator_blob) == spec.get("generator_sha256"),
            "generator hash mismatch")
    dependencies = spec.get("source_dependencies")
    add(errors, isinstance(dependencies, dict),
        "source dependencies are not an object")
    if isinstance(dependencies, dict):
        add(errors, set(dependencies) == EXPECTED_SOURCE_DEPENDENCIES,
            "source dependency path set mismatch")
        for relative in EXPECTED_SOURCE_DEPENDENCIES:
            blob = git_blob(commit, relative) if commit is not None else None
            add(errors, blob is not None,
                f"recorded dependency unavailable:{relative}")
            if blob is not None:
                add(errors, sha256_bytes(blob) == dependencies.get(relative),
                    f"dependency hash mismatch:{relative}")

    try:
        manifest_raw = json.loads(
            manifest_blob.decode("utf-8") if manifest_blob else ""
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        manifest_raw = None
    manifest = validate_manifest(manifest_raw)
    add(errors, manifest is not None, "recorded manifest contract is invalid")
    cases = expected_cases(manifest) if manifest is not None else None
    add(errors, cases is not None, "canonical case matrix cannot be rebuilt")
    if cases is None:
        cases = []
    add(errors, spec.get("cases") == cases, "spec case matrix mismatch")

    compile_command = read_json(root / "build/compile_command.json")
    add(errors, audit_compile_command(compile_command, run_id),
        "compile command mismatch")
    artifact = read_json(root / "build/artifact.json")
    add(errors, isinstance(artifact, dict), "retained artifact is not an object")
    add(errors, artifact.get("compile_command") == compile_command,
        "retained artifact compile command mismatch")
    add(errors, artifact.get("source_dependencies") == dependencies,
        "retained artifact dependency hashes mismatch")
    sass_path = root / "build/tma_ab_contract_bandwidth.sass.txt"
    sass = sass_path.read_text()
    kernel_sass = function_sass(sass, "tma_ab_contract_kernel")
    add(errors, kernel_sass is not None, "kernel-scoped SASS is missing")
    if kernel_sass is not None:
        add(errors, "UTMALDG.2D" in kernel_sass,
            "kernel SASS lacks UTMALDG.2D")
        add(errors, "UTMASTG" not in kernel_sass,
            "kernel SASS unexpectedly stores through TMA")
        expected_counts = {
            "UTMALDG.2D": kernel_sass.count("UTMALDG.2D"),
            "UTMASTG": kernel_sass.count("UTMASTG"),
        }
        add(errors, artifact.get("sass_function_counts") == expected_counts,
            "retained artifact function-scoped SASS counts mismatch")
    binary_hash_text = (root / "build/binary.sha256").read_text().strip()
    binary_match = re.fullmatch(r"([0-9a-f]{64})\s+tma_ab_contract_bandwidth",
                                binary_hash_text)
    add(errors, binary_match is not None, "binary hash file is invalid")
    binary_sha = binary_match.group(1) if binary_match else None
    binary_path = root / "build/tma_ab_contract_bandwidth"
    if binary_path.is_file():
        add(errors, sha256_path(binary_path) == binary_sha,
            "retained binary hash mismatch")
    sass_sha = sha256_path(sass_path)
    artifact_binary = Path(str(artifact.get("binary", "")))
    artifact_sass = Path(str(artifact.get("sass_path", "")))
    expected_build_suffix = (
        Path("results/sm110_gemm_resource_campaign") / run_id / "build"
    )
    add(errors, artifact_binary.is_absolute() and
        tuple(artifact_binary.parts[-len((expected_build_suffix /
              "tma_ab_contract_bandwidth").parts):]) ==
        (expected_build_suffix / "tma_ab_contract_bandwidth").parts,
        "retained artifact binary path mismatch")
    add(errors, artifact_sass.is_absolute() and
        tuple(artifact_sass.parts[-len((expected_build_suffix /
              "tma_ab_contract_bandwidth.sass.txt").parts):]) ==
        (expected_build_suffix / "tma_ab_contract_bandwidth.sass.txt").parts,
        "retained artifact SASS path mismatch")
    for name, expected in (
        ("binary_sha256", binary_sha),
        ("source_sha256", dependencies.get(
            "microbench/15_tma_ab_contract_bandwidth/"
            "tma_ab_contract_bandwidth.cu") if isinstance(dependencies, dict)
            else None),
        ("sass_sha256", sass_sha),
        ("compile_command_sha256", sha256_path(
            root / "build/compile_command.json")),
    ):
        add(errors, artifact.get(name) == expected,
            f"retained artifact hash mismatch:{name}")

    static_rows = read_json(root / "static_contracts.json")
    add(errors, isinstance(static_rows, list) and len(static_rows) == 54,
        "static contract matrix is incomplete")
    static_by_id = {
        str(row.get("case_id")): row
        for row in static_rows if isinstance(row, dict)
    } if isinstance(static_rows, list) else {}

    results = summary.get("results")
    add(errors, summary.get("schema_version") == 1,
        "summary schema mismatch")
    add(errors, summary.get("run_id") == run_id, "summary run ID mismatch")
    add(errors, summary.get("status") == "complete", "summary is incomplete")
    add(errors, summary.get("case_count") == 54, "summary case count mismatch")
    add(errors, summary.get("family_count") == 9,
        "summary family count mismatch")
    add(errors, isinstance(results, list) and len(results) == 54,
        "summary result matrix is incomplete")
    result_by_id = {
        str(row.get("case_id")): row
        for row in results if isinstance(row, dict)
    } if isinstance(results, list) else {}
    expected_ids = {str(case["case_id"]) for case in cases}
    add(errors, set(static_by_id) == expected_ids,
        "static contract case IDs mismatch")
    add(errors, set(result_by_id) == expected_ids,
        "result case IDs mismatch")

    for case in cases:
        case_id = str(case["case_id"])
        static_row = static_by_id.get(case_id, {})
        static_command = static_row.get("command")
        add(errors,
            isinstance(static_command, list)
            and len(static_command) == len(case["args"]) + 2
            and static_command[1] == "--contract-only"
            and valid_binary_command(
                [static_command[0], *static_command[2:]], run_id,
                case["args"],
            ),
            f"{case_id}: static contract command mismatch")
        errors.extend(
            f"{case_id}: {message}"
            for message in field_errors(
                case, static_row.get("fields"), runtime=False
            )
        )

        result = result_by_id.get(case_id, {})
        add(errors, result.get("schema_version") == 1,
            f"{case_id}: result schema mismatch")
        add(errors, result.get("status") == "ok",
            f"{case_id}: result status is not ok")
        for name in (
            "family_id", "resource", "residency", "row_stride_elements",
            "precision_ids", "input_transport_layouts",
        ):
            add(errors, result.get(name) == case.get(name),
                f"{case_id}: result metadata mismatch:{name}")
        add(errors, result.get("trial_count") == 10,
            f"{case_id}: result trial count mismatch")
        add(errors, result.get("trial_timeout_seconds") == 120,
            f"{case_id}: result timeout contract mismatch")
        add(errors, result.get("rate_unit") == "B/s",
            f"{case_id}: result rate unit mismatch")
        add(errors, result.get("expected_contract") == case["expected"],
            f"{case_id}: expected contract mismatch")
        add(errors, result.get("source_path") ==
            "microbench/15_tma_ab_contract_bandwidth/"
            "tma_ab_contract_bandwidth.cu",
            f"{case_id}: source path mismatch")
        source_hash = (
            dependencies.get(result.get("source_path"))
            if isinstance(dependencies, dict) else None
        )
        add(errors, result.get("source_sha256") == source_hash,
            f"{case_id}: source hash mismatch")
        add(errors, result.get("binary_sha256") == binary_sha,
            f"{case_id}: binary hash mismatch")
        add(errors, result.get("sass_sha256") == sass_sha,
            f"{case_id}: SASS hash mismatch")
        add(errors, result.get("sass_path") ==
            "build/tma_ab_contract_bandwidth.sass.txt",
            f"{case_id}: SASS path mismatch")
        add(errors, result.get("sass_tokens") == ["UTMALDG.2D"],
            f"{case_id}: SASS token contract mismatch")
        expected_fingerprint = sha256_json({
            "case": case,
            "binary_sha256": binary_sha,
            "source_sha256": source_hash,
            "trial_count": 10,
            "ncu": bool(spec.get("ncu_requested"))
            and bool(case["ncu_selected"]),
        })
        add(errors, result.get("fingerprint") == expected_fingerprint,
            f"{case_id}: result fingerprint mismatch")

        trials_path = root / "cases" / case_id / "trials.jsonl"
        add(errors, trials_path.is_file(), f"{case_id}: trials missing")
        trial_rows: list[Any] = []
        if trials_path.is_file():
            try:
                trial_rows = [
                    json.loads(line)
                    for line in trials_path.read_text().splitlines()
                    if line
                ]
            except json.JSONDecodeError:
                errors.append(f"{case_id}: trials are not JSONL")
        add(errors, len(trial_rows) == 10,
            f"{case_id}: expected ten external trials")
        rates: list[float] = []
        for trial_index, trial in enumerate(trial_rows, 1):
            if not isinstance(trial, dict):
                errors.append(f"{case_id}: trial is not an object")
                continue
            add(errors, trial.get("trial") == trial_index,
                f"{case_id}: trial index mismatch")
            add(errors, trial.get("returncode") == 0,
                f"{case_id}: trial return code mismatch")
            add(errors, trial.get("timeout_seconds") == 120,
                f"{case_id}: trial timeout contract mismatch")
            add(errors, trial.get("timed_out") is False,
                f"{case_id}: timed-out trial cannot pass")
            add(errors, trial.get("termination_failed") is False,
                f"{case_id}: termination failure cannot pass")
            add(errors, valid_binary_command(
                trial.get("command"), run_id, case["args"]
            ), f"{case_id}: trial command mismatch")
            raw = trial.get("raw_stdout")
            fields = trial.get("fields")
            add(errors, isinstance(raw, str),
                f"{case_id}: raw stdout missing")
            if isinstance(raw, str):
                add(errors, parse_kv(raw) == fields,
                    f"{case_id}: parsed fields differ from raw stdout")
            errors.extend(
                f"{case_id}: {message}"
                for message in field_errors(case, fields, runtime=True)
            )
            try:
                elapsed = int(fields["globaltimer_elapsed_ns"])
                requested = int(fields["requested_bytes"])
                rate = requested * 1.0e9 / elapsed
                recorded_rate = float(trial["audited_rate_per_second"])
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                errors.append(f"{case_id}: trial rate is malformed")
                continue
            add(errors, math.isclose(rate, recorded_rate, rel_tol=2e-12),
                f"{case_id}: audited trial rate mismatch")
            rates.append(rate)
        if len(rates) == 10:
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
                    actual = float("nan")
                add(errors, math.isclose(actual, value, rel_tol=2e-12),
                    f"{case_id}: summary statistic mismatch:{name}")
        audit_ncu(root, run_id, case, result, errors, require_ncu)

    complete = (root / "COMPLETE").read_text()
    add(errors, f"run_id={run_id}" in complete,
        "COMPLETE run ID mismatch")
    add(errors, f"summary_sha256={sha256_path(root / 'summary.json')}" in complete,
        "COMPLETE summary hash mismatch")
    if require_ncu:
        add(errors, spec.get("ncu_requested") is True,
            "NCU was not requested in run spec")
        add(errors, summary.get("ncu_requested") is True,
            "NCU was not requested in summary")
        add(errors, summary.get("ncu_case_count") == 18,
            "NCU case count mismatch")

    return {"pass": not errors, "run_dir": str(root), "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-ncu", action="store_true")
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    if args.expected_commit is not None and not COMMIT_RE.fullmatch(
        args.expected_commit
    ):
        parser.error("--expected-commit must be a 40-hex commit")
    result = audit(
        args.run_dir,
        require_ncu=args.require_ncu,
        expected_commit=args.expected_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
