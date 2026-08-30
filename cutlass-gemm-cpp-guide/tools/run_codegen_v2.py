#!/usr/bin/env python3
"""Run one or more static-only SM110a codegen instances in the pinned CUDA image."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codegen_v2.attribution import (
    AttributionError,
    attribute_codegen,
    parse_entry_symbols,
    parse_ptx_entries,
)
from codegen_v2.model import (
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    contract_sha256,
    evaluate_arch_guard_patterns,
    expected_execution_plan,
    file_ref,
    hash_input_tree,
    journal_event_hash,
    load_strict_json,
    load_run_campaign,
    safe_path,
    sha256_bytes,
    sha256_file,
    validate_artifact_manifest,
    validate_fingerprint,
    validate_instance,
    validate_result,
    validate_source_anchor,
)


IMAGE = (
    "nvcr.io/nvidia/cuda@"
    "sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6"
)
IMAGE_DIGEST = "sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6"
OBJECTIVE_SHA256 = "463fa7015808acd883b28d115fa33708f66064aaceed96f728996a01ca0e4091"
CUTLASS_SHA = "e05f953a5b3d38adc240df2ff928e0421c2abba3"
ASSERTED_ABSENT_ENV = (
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "CUDAFE_FLAGS",
    "PTXAS_OPTIONS",
    "LD_PRELOAD",
)


class RunnerError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_checked(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RunnerError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return result


def docker_command(
    *,
    entrypoint: str,
    guide_root: Path | None = None,
    attempt_root: Path | None = None,
    workdir: str | None = None,
    read_only_attempt: bool = False,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--security-opt",
        "label=disable",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "LC_ALL=C",
        "--entrypoint",
        entrypoint,
    ]
    if guide_root is not None:
        command += ["-v", f"{guide_root}:/workspace:ro"]
    if attempt_root is not None:
        suffix = ":ro" if read_only_attempt else ""
        command += ["-v", f"{attempt_root}:/out{suffix}"]
    if workdir is not None:
        command += ["-w", workdir]
    return command + [IMAGE]


def inspect_environment(root: Path) -> dict[str, Any]:
    present = [name for name in ASSERTED_ABSENT_ENV if os.environ.get(name)]
    if present:
        raise RunnerError(f"codegen-affecting environment variables must be absent: {present}")
    head = run_checked(["git", "-C", str(root / "third_party/cutlass"), "rev-parse", "HEAD"])
    if head.stdout.decode().strip() != CUTLASS_SHA:
        raise RunnerError("CUTLASS HEAD differs from the pinned commit")
    dirty = run_checked(
        [
            "git",
            "-C",
            str(root / "third_party/cutlass"),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ]
    )
    if dirty.stdout.strip():
        raise RunnerError("CUTLASS checkout is dirty")
    image = run_checked(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE])
    if image.stdout.decode().strip() != IMAGE_DIGEST:
        raise RunnerError("local Docker image ID differs from the pinned digest")
    container_environment = run_checked(
        docker_command(entrypoint="/usr/bin/env")
    ).stdout.decode("utf-8", errors="strict")
    container_names = {line.partition("=")[0] for line in container_environment.splitlines() if "=" in line}
    forbidden_in_container = sorted(set(ASSERTED_ABSENT_ENV) & container_names)
    if forbidden_in_container:
        raise RunnerError(
            f"container has codegen-affecting environment variables: {forbidden_in_container}"
        )

    tools = {
        "nvcc": ("/usr/local/cuda/bin/nvcc", ["--version"]),
        "ptxas": ("/usr/local/cuda/bin/ptxas", ["--version"]),
        "cuobjdump": ("/usr/local/cuda/bin/cuobjdump", ["--version"]),
        "nvdisasm": ("/usr/local/cuda/bin/nvdisasm", ["--version"]),
        "host_compiler": ("/usr/bin/g++", ["--version"]),
    }
    executable_records: dict[str, Any] = {}
    for name, (path, argv) in tools.items():
        output = run_checked(docker_command(entrypoint=path) + argv)
        binary_hash_output = run_checked(
            docker_command(entrypoint="/usr/bin/sha256sum") + [path]
        ).stdout.decode("utf-8", errors="strict")
        binary_hash = binary_hash_output.split()[0]
        if not re.fullmatch(r"[0-9a-f]{64}", binary_hash):
            raise RunnerError(f"cannot hash tool binary {path}")
        version = (output.stdout + output.stderr).decode("utf-8", errors="replace").strip()
        executable_records[name] = {
            "path": path,
            "binary_sha256": binary_hash,
            "version": version,
            "version_sha256": sha256_bytes((version + "\n").encode("utf-8")),
        }
    expected_fragments = {
        "nvcc": "V13.0.88",
        "ptxas": "V13.0.88",
        "cuobjdump": "V13.0.85",
        "nvdisasm": "V13.0.85",
        "host_compiler": "13.3.0",
    }
    for name, fragment in expected_fragments.items():
        if fragment not in executable_records[name]["version"]:
            raise RunnerError(f"{name} version output does not contain {fragment}")
    return {
        "container_digest": IMAGE_DIGEST,
        "versions_lock_sha256": sha256_file(root / "versions.lock.json"),
        "executables": executable_records,
    }


def harness_input_paths(root: Path) -> list[Path]:
    paths = [
        root / "tests/codegen/static_codegen_contract.json",
        root / "tests/codegen/sm110a_schedule_reference_inventory.json",
        root / "versions.lock.json",
        root / "tools/run_codegen_v2.py",
        root / "tools/validate_codegen_v2.py",
        root / "tools/codegen_v2/attribution.py",
        root / "tools/codegen_v2/deep_replay.py",
        root / "tools/codegen_v2/model.py",
    ]
    paths.extend(sorted((root / "tests/codegen/schemas").glob("*.json")))
    paths.extend(sorted((root / "include/guide").glob("*.hpp")))
    return paths


def logical_execution_plan(instance: dict[str, Any]) -> list[dict[str, Any]]:
    return expected_execution_plan(instance)


def build_fingerprint(
    root: Path,
    instance_path: Path,
    instance: dict[str, Any],
    toolchain: dict[str, Any],
) -> dict[str, Any]:
    input_paths = {path.resolve() for path in harness_input_paths(root)}
    input_paths.add(instance_path.resolve())
    for reference in instance["translation_units"].values():
        if reference is None:
            continue
        input_paths.add((root / reference["path"]).resolve())
    inputs = [file_ref(root, path) for path in sorted(input_paths)]
    plan = logical_execution_plan(instance)
    instance_hash = sha256_file(instance_path)
    source_hash = hash_input_tree(inputs)
    toolchain_hash = sha256_bytes(canonical_json_bytes(toolchain))
    harness_paths = [
        path
        for path in input_paths
        if "tools" in path.parts or "schemas" in path.parts or path.name == "static_codegen_contract.json"
    ]
    harness_hash = hash_input_tree([file_ref(root, path) for path in sorted(harness_paths)])
    plan_hash = sha256_bytes(canonical_json_bytes(plan))
    fingerprint = {
        "schema_version": 1,
        "contract_sha256": contract_sha256(root),
        "objective_sha256": OBJECTIVE_SHA256,
        "freshness_epoch": load_strict_json(root / "tests/codegen/static_codegen_contract.json")[
            "freshness_epoch"
        ],
        "fingerprint_id": "pending",
        "instance_ref": file_ref(root, instance_path, identifier=instance["instance_id"]),
        "source_closure": {
            "cutlass_git_sha": CUTLASS_SHA,
            "cutlass_clean": True,
            "inputs": inputs,
            "input_tree_sha256": source_hash,
        },
        "toolchain": toolchain,
        "execution_plan": plan,
        "environment": {
            "allowed": {"LC_ALL": "C"},
            "asserted_absent": list(ASSERTED_ABSENT_ENV),
        },
        "component_hashes": {
            "instance": instance_hash,
            "source": source_hash,
            "toolchain": toolchain_hash,
            "harness": harness_hash,
            "plan": plan_hash,
        },
        "overall_sha256": "0" * 64,
    }
    value = dict(fingerprint)
    value.pop("overall_sha256")
    value.pop("fingerprint_id")
    fingerprint["overall_sha256"] = sha256_bytes(canonical_json_bytes(value))
    fingerprint["fingerprint_id"] = (
        f"fp-{instance['instance_id']}-{fingerprint['overall_sha256'][:12]}"
    )
    return fingerprint


@dataclass
class Journal:
    path: Path
    attempt_id: str
    instance_id: str
    fingerprint_id: str
    previous: str | None = None
    seq: int = 0

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        self.seq += 1
        event = {
            "seq": self.seq,
            "event_type": event_type,
            "attempt_id": self.attempt_id,
            "instance_id": self.instance_id,
            "fingerprint_id": self.fingerprint_id,
            "timestamp_utc": utc_now(),
            "previous_event_sha256": self.previous,
            "payload": payload,
            "event_sha256": "0" * 64,
        }
        event["event_sha256"] = journal_event_hash(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.previous = event["event_sha256"]


@dataclass
class CommandRecord:
    step_id: str
    stdout_path: Path
    stderr_path: Path
    returncode: int


def capture_step(
    *,
    command: list[str],
    step_id: str,
    root: Path,
    attempt_root: Path,
    journal: Journal,
    cwd: Path | None = None,
) -> CommandRecord:
    logs = attempt_root / "full/logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{step_id}.stdout"
    stderr_path = logs / f"{step_id}.stderr"
    journal.append("STEP_STARTED", {"step_id": step_id, "argv": command})
    result = subprocess.run(
        command,
        cwd=cwd,
        env={**os.environ, "LC_ALL": "C"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_bytes(result.stdout)
    stderr_path.write_bytes(result.stderr)
    journal.append(
        "COMMAND_FINISHED",
        {
            "step_id": step_id,
            "returncode": result.returncode,
            "stdout_path": stdout_path.relative_to(root).as_posix(),
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_path": stderr_path.relative_to(root).as_posix(),
            "stderr_sha256": sha256_file(stderr_path),
        },
    )
    return CommandRecord(step_id, stdout_path, stderr_path, result.returncode)


def execute_step(**kwargs: Any) -> CommandRecord:
    record = capture_step(**kwargs)
    if record.returncode:
        raise RunnerError(
            f"{record.step_id} failed ({record.returncode}); "
            f"see {record.stdout_path} and {record.stderr_path}"
        )
    return record


def one_extracted_file(directory: Path, suffix: str, label: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RunnerError(f"{label}: expected exactly one {suffix} file, found {matches}")
    return matches[0]


def pattern_results(
    contracts: list[dict[str, Any]], matches: dict[str, tuple[str, ...]], required: bool
) -> list[dict[str, Any]]:
    output = []
    for contract in contracts:
        values = list(matches.get(contract["regex"], ()))
        count = len(values)
        maximum = contract["max_count"]
        if count < contract["min_count"] or (maximum is not None and count > maximum):
            raise RunnerError(
                f"opcode contract {contract['id']} count {count} outside "
                f"[{contract['min_count']}, {maximum}]"
            )
        output.append(
            {
                "id": contract["id"],
                "required": required,
                "match_count": count,
                "matched_opcodes": values,
            }
        )
    return output


def classify_expected_reject_diagnostic(
    root: Path,
    instance: dict[str, Any],
    record: CommandRecord,
) -> tuple[str, list[dict[str, Any]], int, list[dict[str, Any]]]:
    contract = load_strict_json(root / "tests/codegen/static_codegen_contract.json")
    global_contract = contract["expected_static_reject_contract"]
    reject = instance["hypothesis"]["reject_contract"]
    if reject is None or record.step_id != reject["step_id"]:
        raise RunnerError("failed command is not the frozen expected-reject step")
    if record.returncode != reject["expected_returncode"]:
        raise RunnerError("expected-reject return code differs from the frozen contract")
    if record.returncode in global_contract["forbidden_returncodes"] or record.returncode >= global_contract[
        "forbidden_returncode_minimum"
    ]:
        raise RunnerError("infrastructure return code cannot become EXPECTED_STATIC_REJECT")
    raw = record.stderr_path.read_bytes()
    if not raw or len(raw) > global_contract["max_stderr_bytes"]:
        raise RunnerError("expected-reject stderr is empty or exceeds the frozen limit")
    try:
        stderr = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RunnerError("expected-reject stderr is not UTF-8") from error
    for pattern in global_contract["infrastructure_forbidden_patterns"]:
        if re.search(pattern, stderr) is not None:
            raise RunnerError("infrastructure diagnostic cannot become EXPECTED_STATIC_REJECT")
    results: list[dict[str, Any]] = []
    selected_lines: dict[int, str] = {}
    lines = stderr.splitlines()
    for pattern in reject["required_diagnostics"]:
        expression = re.compile(pattern["regex"])
        matches = list(expression.finditer(stderr))
        count = len(matches)
        maximum = pattern["max_count"]
        if count < pattern["min_count"] or (maximum is not None and count > maximum):
            raise RunnerError(f"required rejection diagnostic {pattern['id']} differs")
        matched = [match.group(0) for match in matches]
        results.append(
            {
                "id": pattern["id"],
                "required": True,
                "match_count": count,
                "matched_opcodes": matched,
            }
        )
        for match in matches:
            line_number = stderr.count("\n", 0, match.start()) + 1
            selected_lines[line_number] = lines[line_number - 1]
    error_count = len(re.findall(global_contract["error_line_regex"], stderr))
    if error_count != reject["expected_error_count"]:
        raise RunnerError("expected-reject error count differs from the frozen contract")
    for line_number, line in enumerate(lines, start=1):
        if (
            "error" in line.lower()
            or "fatal" in line.lower()
            or instance["instance_id"] in line
            or "KernelScheduleAuto" in line
        ):
            selected_lines[line_number] = line
    excerpt_lines = [
        {
            "line_number": line_number,
            "text": selected_lines[line_number],
            "sha256": sha256_bytes((selected_lines[line_number] + "\n").encode("utf-8")),
        }
        for line_number in sorted(selected_lines)
    ]
    return stderr, results, error_count, excerpt_lines


def source_constraint_text(root: Path, source_range: dict[str, Any]) -> str:
    source = safe_path(root / "third_party/cutlass", source_range["path"], "source_constraint.path")
    lines = source.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[source_range["line_start"] - 1 : source_range["line_end"]]) + "\n"


def next_attempt_id(
    run_id: str, instance_id: str, attempts_root: Path, evidence_root: Path
) -> str:
    prefix = f"{run_id}.{instance_id}.a"
    indexes = []
    for path in attempts_root.glob(prefix + "[0-9][0-9][0-9]"):
        match = re.fullmatch(re.escape(prefix) + r"([0-9]{3})", path.name)
        if match:
            indexes.append(int(match.group(1)))
    index = max(indexes, default=0) + 1
    while True:
        candidate = prefix + f"{index:03d}"
        occupied = [
            attempts_root / candidate,
            evidence_root / "journals" / f"{candidate}.jsonl",
            evidence_root / "results" / f"{candidate}.json",
            evidence_root / "artifact-manifests" / f"am-{candidate}.json",
            evidence_root / "excerpts" / candidate,
        ]
        if not any(path.exists() for path in occupied):
            return candidate
        index += 1


def add_artifact(
    items: list[dict[str, Any]],
    *,
    root: Path,
    artifact_id: str,
    role: str,
    path: Path,
    storage: str,
    media_type: str,
    producer_step_id: str,
    parents: list[str],
) -> None:
    items.append(
        {
            "artifact_id": artifact_id,
            "role": role,
            "path": path.relative_to(root).as_posix(),
            "storage": storage,
            "media_type": media_type,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "producer_step_id": producer_step_id,
            "parent_artifact_ids": parents,
            "sealed": True,
        }
    )


def find_fresh_static_control(
    root: Path,
    instance: dict[str, Any],
    subject_fingerprint: dict[str, Any],
) -> dict[str, str]:
    control_id = instance["hypothesis"]["control_instance_id"]
    campaign_id = instance["hypothesis"]["control_campaign_id"]
    if not isinstance(control_id, str) or not control_id or not isinstance(campaign_id, str):
        raise RunnerError("rejection instance lacks a frozen control instance/campaign")
    campaign_path, campaign = load_run_campaign(root, campaign_id)
    summary_path = safe_path(root, campaign["current_summary"], "control_campaign.current_summary")
    summary = load_strict_json(summary_path)
    if (
        summary.get("run_id") != campaign["run_id"]
        or summary.get("campaign_ref")
        != file_ref(root, campaign_path, identifier=campaign_id)
        or not isinstance(summary.get("result_refs"), list)
    ):
        raise RunnerError("control campaign summary is not sealed to its campaign")
    selected: list[tuple[Path, dict[str, Any]]] = []
    observed_ids: list[str] = []
    for ref in summary["result_refs"]:
        path = safe_path(root, ref["path"], "control_campaign.result_ref")
        if ref != file_ref(root, path, identifier=ref["id"]):
            raise RunnerError("control campaign result ref is not sealed")
        candidate = validate_result(root, path, require_archive=True)
        observed_ids.append(candidate["instance_ref"]["id"])
        if candidate["instance_ref"]["id"] == control_id:
            selected.append((path, candidate))
    if observed_ids != campaign["ordered_instances"] or len(selected) != 1:
        raise RunnerError("control campaign membership/order does not provide one legal control")
    control_path, control_result = selected[0]
    if (
        control_result["status"] != "STATIC_PASS"
        or control_result["contract_sha256"] != contract_sha256(root)
        or control_result["freshness_epoch"] != instance["freshness_epoch"]
    ):
        raise RunnerError("control campaign result is not a fresh STATIC_PASS")
    control_instance = validate_instance(
        root, safe_path(root, control_result["instance_ref"]["path"], "control.instance_ref")
    )
    control_fingerprint = validate_fingerprint(
        root,
        safe_path(root, control_result["fingerprint_ref"]["path"], "control.fingerprint_ref"),
    )
    if (
        control_instance["target"] != instance["target"]
        or control_fingerprint["toolchain"] != subject_fingerprint["toolchain"]
        or control_fingerprint["environment"] != subject_fingerprint["environment"]
        or control_fingerprint["source_closure"]["cutlass_git_sha"]
        != subject_fingerprint["source_closure"]["cutlass_git_sha"]
    ):
        raise RunnerError("control campaign result uses a different target/environment")
    return file_ref(
        root,
        control_path,
        identifier=control_result["result_id"],
    )


def seal_expected_static_reject(
    *,
    root: Path,
    instance_path: Path,
    instance: dict[str, Any],
    fingerprint: dict[str, Any],
    fingerprint_path: Path,
    evidence_root: Path,
    results_dir: Path,
    attempt_id: str,
    attempt_root: Path,
    snapshot_paths: dict[str, Path],
    command_records: list[CommandRecord],
    failed_record: CommandRecord,
    prefix_executable: Path | None,
    prefix_output: Path | None,
    journal: Journal,
    working_journal_path: Path,
    journal_path: Path,
) -> dict[str, Any]:
    reject = instance["hypothesis"]["reject_contract"]
    if reject is None:
        raise RunnerError("EXPECTED_STATIC_REJECT lacks a reject contract")
    _, diagnostic_results, error_count, excerpt_lines = classify_expected_reject_diagnostic(
        root, instance, failed_record
    )
    if (prefix_output is not None) is not reject["requires_collective_prefix_witness"]:
        raise RunnerError("expected-reject prefix witness presence differs from the contract")
    if (attempt_root / "full/type_witness").exists():
        raise RunnerError("rejected type witness unexpectedly produced an executable")
    control_ref = find_fresh_static_control(root, instance, fingerprint)

    excerpt_dir = evidence_root / "excerpts" / attempt_id
    excerpt_dir.mkdir(parents=True, exist_ok=False)
    diagnostic_excerpt = excerpt_dir / "diagnostic-excerpt.json"
    diagnostic_value = {
        "schema_version": 1,
        "instance_id": instance["instance_id"],
        "raw_stderr_sha256": sha256_file(failed_record.stderr_path),
        "raw_stderr_size_bytes": failed_record.stderr_path.stat().st_size,
        "observed_error_count": error_count,
        "required_results": diagnostic_results,
        "infrastructure_forbidden_matches": [],
        "selected_lines": excerpt_lines,
    }
    atomic_write_json(diagnostic_excerpt, diagnostic_value)

    source_evidence: list[dict[str, Any]] = []
    source_excerpt_paths: list[Path] = []
    for index, source_range in enumerate(reject["source_constraints"]):
        source_excerpt = excerpt_dir / f"source-constraint-{index:02d}.txt"
        source_excerpt.write_text(source_constraint_text(root, source_range), encoding="utf-8")
        artifact_id = f"source_constraint_{index:02d}"
        source_excerpt_paths.append(source_excerpt)
        source_evidence.append({**source_range, "excerpt_artifact_id": artifact_id})

    tracked_prefix_output: Path | None = None
    if prefix_output is not None:
        tracked_prefix_output = excerpt_dir / "collective-prefix-types.json"
        shutil.copy2(prefix_output, tracked_prefix_output)

    rejection_report = excerpt_dir / "rejection-report.json"
    failure = {
        "layer": instance["hypothesis"]["failure_layer"],
        "domain": "CONFIGURATION_LEGALITY",
        "step_id": failed_record.step_id,
        "returncode": failed_record.returncode,
        "stderr_artifact_id": f"{failed_record.step_id}_stderr",
    }
    report_value = {
        "schema_version": 1,
        "instance_id": instance["instance_id"],
        "terminal_status": "EXPECTED_STATIC_REJECT",
        "failure": failure,
        "diagnostic_results": diagnostic_results,
        "observed_error_count": error_count,
        "raw_stderr_sha256": sha256_file(failed_record.stderr_path),
        "raw_stderr_size_bytes": failed_record.stderr_path.stat().st_size,
        "source_constraints": source_evidence,
        "prefix_witness_artifact_id": (
            "collective_prefix_output" if tracked_prefix_output is not None else None
        ),
        "legal_control": {
            "result_ref": control_ref,
            "relation": reject["control_relation"],
            "controlled_delta": reject["controlled_delta"],
        },
    }
    atomic_write_json(rejection_report, report_value)

    if fingerprint_path.exists():
        if load_strict_json(fingerprint_path) != fingerprint:
            raise RunnerError(f"fingerprint ID collision at publication: {fingerprint_path}")
    else:
        atomic_write_json(fingerprint_path, fingerprint)
    validate_fingerprint(root, fingerprint_path)

    items: list[dict[str, Any]] = []
    add_artifact(items, root=root, artifact_id="config_source", role="KERNEL_SOURCE_SNAPSHOT", path=snapshot_paths["config"], storage="ignored_archive", media_type="text/x-c++hdr", producer_step_id="snapshot_sources", parents=[])
    add_artifact(items, root=root, artifact_id="kernel_source", role="KERNEL_SOURCE_SNAPSHOT", path=snapshot_paths["kernel"], storage="ignored_archive", media_type="text/x-cuda", producer_step_id="snapshot_sources", parents=["config_source"])
    add_artifact(items, root=root, artifact_id="type_source", role="TYPE_WITNESS_SOURCE_SNAPSHOT", path=snapshot_paths["type_witness"], storage="ignored_archive", media_type="text/x-cuda", producer_step_id="snapshot_sources", parents=["config_source"])
    if prefix_executable is not None and prefix_output is not None and tracked_prefix_output is not None:
        add_artifact(items, root=root, artifact_id="collective_prefix_source", role="COLLECTIVE_PREFIX_SOURCE", path=snapshot_paths["collective_prefix_witness"], storage="ignored_archive", media_type="text/x-cuda", producer_step_id="snapshot_sources", parents=["config_source"])
        add_artifact(items, root=root, artifact_id="collective_prefix_executable", role="COLLECTIVE_PREFIX_EXECUTABLE", path=prefix_executable, storage="ignored_archive", media_type="application/x-executable", producer_step_id="compile_collective_prefix_witness", parents=["collective_prefix_source", "config_source"])
        add_artifact(items, root=root, artifact_id="collective_prefix_output", role="COLLECTIVE_PREFIX_OUTPUT", path=tracked_prefix_output, storage="git_evidence", media_type="application/json", producer_step_id="run_collective_prefix_witness", parents=["collective_prefix_executable"])
    for record in command_records:
        add_artifact(items, root=root, artifact_id=f"{record.step_id}_stdout", role="COMPILE_STDOUT", path=record.stdout_path, storage="ignored_archive", media_type="text/plain", producer_step_id=record.step_id, parents=[])
        add_artifact(items, root=root, artifact_id=f"{record.step_id}_stderr", role="COMPILE_STDERR", path=record.stderr_path, storage="ignored_archive", media_type="text/plain", producer_step_id=record.step_id, parents=[])
    add_artifact(items, root=root, artifact_id="diagnostic_excerpt", role="DIAGNOSTIC_EXCERPT", path=diagnostic_excerpt, storage="git_evidence", media_type="application/json", producer_step_id="compile_type_witness", parents=["compile_type_witness_stderr"])
    for index, source_excerpt in enumerate(source_excerpt_paths):
        add_artifact(items, root=root, artifact_id=f"source_constraint_{index:02d}", role="SOURCE_CONSTRAINT_EXCERPT", path=source_excerpt, storage="git_evidence", media_type="text/plain", producer_step_id="reject_contract", parents=[])
    rejection_parents = ["diagnostic_excerpt"] + [
        f"source_constraint_{index:02d}" for index in range(len(source_excerpt_paths))
    ]
    if tracked_prefix_output is not None:
        rejection_parents.append("collective_prefix_output")
    add_artifact(items, root=root, artifact_id="rejection_report", role="REJECTION_REPORT", path=rejection_report, storage="git_evidence", media_type="application/json", producer_step_id="reject_contract", parents=rejection_parents)

    archive_items = [item for item in items if item["storage"] == "ignored_archive"]
    archive_checksum = sha256_bytes(canonical_json_bytes([
        (item["artifact_id"], item["path"], item["sha256"], item["size_bytes"])
        for item in archive_items
    ]))
    manifest_id = f"am-{attempt_id}"
    manifest = {
        "schema_version": 1,
        "contract_sha256": contract_sha256(root),
        "artifact_manifest_id": manifest_id,
        "attempt_id": attempt_id,
        "instance_ref": file_ref(root, instance_path, identifier=instance["instance_id"]),
        "fingerprint_ref": file_ref(root, fingerprint_path, identifier=fingerprint["fingerprint_id"]),
        "items": items,
        "archive_bundle": {
            "path": attempt_root.relative_to(root).as_posix(),
            "format": "directory-manifest",
            "sha256": archive_checksum,
            "size_bytes": sum(item["size_bytes"] for item in archive_items),
        },
    }
    for item in items:
        journal.append("ARTIFACT_SEALED", {"artifact_id": item["artifact_id"], "path": item["path"], "sha256": item["sha256"], "producer_step_id": item["producer_step_id"]})
    manifest_path = evidence_root / "artifact-manifests" / f"{manifest_id}.json"
    atomic_write_json(manifest_path, manifest)
    validate_artifact_manifest(root, manifest_path)
    journal.append("ARTIFACT_MANIFEST_SEALED", {"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256_file(manifest_path)})
    journal.append("ATTEMPT_SEALED", {
        "terminal_pipeline_state": "EXPECTED_STATIC_REJECT",
        "artifact_manifest_path": manifest_path.relative_to(root).as_posix(),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "last_completed_step_id": "compile_type_witness",
    })
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = journal_path.with_name(journal_path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(working_journal_path.read_bytes())
    os.replace(temporary, journal_path)
    working_journal_path.unlink()

    layer_order = load_strict_json(root / "tests/codegen/static_codegen_contract.json")["layer_order"]
    rejected_layer = instance["hypothesis"]["failure_layer"]
    rejected_index = layer_order.index(rejected_layer)
    layer_outcomes = []
    for index, layer in enumerate(layer_order):
        if index < rejected_index:
            witness = "config_source" if index == 0 else "collective_prefix_output"
            state = "RESOLVED"
        elif index == rejected_index:
            witness = "rejection_report"
            state = "REJECTED"
        else:
            witness = None
            state = "NOT_REACHED"
        layer_outcomes.append({"layer": layer, "state": state, "witness_artifact_id": witness})
    evidence_value = {
        "kind": "EXPECTED_STATIC_REJECT",
        "failure": failure,
        "diagnostic_results": diagnostic_results,
        "observed_error_count": error_count,
        "raw_stderr_sha256": sha256_file(failed_record.stderr_path),
        "raw_stderr_size_bytes": failed_record.stderr_path.stat().st_size,
        "diagnostic_excerpt_artifact_id": "diagnostic_excerpt",
        "rejection_report_artifact_id": "rejection_report",
        "prefix_witness_artifact_id": (
            "collective_prefix_output" if tracked_prefix_output is not None else None
        ),
        "source_constraints": source_evidence,
        "legal_control": report_value["legal_control"],
    }
    result = {
        "schema_version": 1,
        "contract_sha256": contract_sha256(root),
        "objective_sha256": OBJECTIVE_SHA256,
        "freshness_epoch": fingerprint["freshness_epoch"],
        "result_id": attempt_id,
        "scope": "STATIC_CODEGEN_ONLY",
        "subject": instance["subject"],
        "instance_ref": file_ref(root, instance_path, identifier=instance["instance_id"]),
        "fingerprint_ref": file_ref(root, fingerprint_path, identifier=fingerprint["fingerprint_id"]),
        "attempt_id": attempt_id,
        "journal_ref": {"path": journal_path.relative_to(root).as_posix(), "sha256": sha256_file(journal_path), "last_seq": journal.seq, "last_event_sha256": journal.previous},
        "artifact_manifest_ref": file_ref(root, manifest_path, identifier=manifest_id),
        "status": "EXPECTED_STATIC_REJECT",
        "layer_outcomes": layer_outcomes,
        "evidence": evidence_value,
    }
    result_path = results_dir / f"{attempt_id}.json"
    atomic_write_json(result_path, result)
    validate_result(root, result_path)
    print(f"EXPECTED_STATIC_REJECT instance={instance['instance_id']} result={attempt_id}")
    return result


def run_instance(
    *,
    root: Path,
    instance_path: Path,
    run_id: str,
    toolchain: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    instance = validate_instance(root, instance_path)
    fingerprint = build_fingerprint(root, instance_path, instance, toolchain)
    evidence_root = root / "evidence/codegen-sm110a-v2"
    fingerprint_path = evidence_root / "fingerprints" / f"{fingerprint['fingerprint_id']}.json"
    if fingerprint_path.exists():
        existing = load_strict_json(fingerprint_path)
        if existing != fingerprint:
            raise RunnerError(f"fingerprint ID collision: {fingerprint_path}")
        validate_fingerprint(root, fingerprint_path)
    else:
        staged_fingerprint = root / ".cache/codegen-v2/fingerprints" / fingerprint_path.name
        atomic_write_json(staged_fingerprint, fingerprint)
        validate_fingerprint(root, staged_fingerprint)
        staged_fingerprint.unlink()

    results_dir = evidence_root / "results"
    if resume:
        candidates = sorted(results_dir.glob(f"{run_id}.{instance['instance_id']}.a*.json"), reverse=True)
        for candidate in candidates:
            try:
                result = validate_result(root, candidate)
            except (ContractError, OSError):
                continue
            if (
                result["fingerprint_ref"]["id"] == fingerprint["fingerprint_id"]
                and load_strict_json(
                    safe_path(root, result["fingerprint_ref"]["path"], "resume.fingerprint")
                )["overall_sha256"]
                == fingerprint["overall_sha256"]
            ):
                print(f"RESUMED_VERIFIED instance={instance['instance_id']} result={result['result_id']}")
                return result

    attempts_root = root / "artifacts/codegen-sm110a-v2" / run_id / "attempts"
    attempt_id = next_attempt_id(
        run_id, instance["instance_id"], attempts_root, evidence_root
    )
    attempt_root = attempts_root / attempt_id
    attempt_root.mkdir(parents=True, exist_ok=False)
    full = attempt_root / "full"
    extracted = full / "extracted"
    snapshots = full / "source"
    extracted.mkdir(parents=True)
    snapshots.mkdir(parents=True)
    journal_path = evidence_root / "journals" / f"{attempt_id}.jsonl"
    if journal_path.exists():
        raise RunnerError(f"journal already exists: {journal_path}")
    # An in-progress journal is mutable staging state, not part of the sealed
    # attempt archive.  Keep it in the ignored cache until publication.
    working_journal_path = (
        root / ".cache/codegen-v2/journals" / f"{attempt_id}.in-progress.jsonl"
    )
    journal = Journal(
        working_journal_path,
        attempt_id,
        instance["instance_id"],
        fingerprint["fingerprint_id"],
    )
    journal.append(
        "ATTEMPT_CREATED",
        {
            "run_id": run_id,
            "origin_guide_root": str(root),
            "origin_attempt_root": str(attempt_root),
            "origin_user": f"{os.getuid()}:{os.getgid()}",
        },
    )
    journal.append("INSTANCE_VALIDATED", {"instance_sha256": sha256_file(instance_path)})
    journal.append("FINGERPRINT_SEALED", {"overall_sha256": fingerprint["overall_sha256"]})

    snapshot_paths: dict[str, Path] = {}
    for role, reference in instance["translation_units"].items():
        if reference is None:
            continue
        source = safe_path(root, reference["path"], f"translation_units.{role}.path")
        target = snapshots / f"{role}-{source.name}"
        shutil.copy2(source, target)
        snapshot_paths[role] = target

    common = [
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "--expt-relaxed-constexpr",
        f"--frandom-seed={instance['instance_id']}",
        "-Iinclude",
        "-Ithird_party/cutlass/include",
        "-Ithird_party/cutlass/tools/util/include",
    ]
    expected_outcome = instance["hypothesis"]["expected_outcome"]
    command_records: list[CommandRecord] = []
    prefix_executable: Path | None = None
    prefix_output: Path | None = None
    prefix_reference = instance["translation_units"]["collective_prefix_witness"]
    if prefix_reference is not None:
        prefix_executable = full / "collective_prefix_witness"
        prefix_compile = docker_command(
            entrypoint="/usr/local/cuda/bin/nvcc",
            guide_root=root,
            attempt_root=attempt_root,
            workdir="/workspace",
        ) + common + [
            "--generate-code=arch=compute_110a,code=sm_110a",
            prefix_reference["path"],
            "-o",
            "/out/full/collective_prefix_witness",
        ]
        command_records.append(
            execute_step(
                command=prefix_compile,
                step_id="compile_collective_prefix_witness",
                root=root,
                attempt_root=attempt_root,
                journal=journal,
            )
        )
        prefix_run = docker_command(
            entrypoint="/out/full/collective_prefix_witness",
            attempt_root=attempt_root,
            read_only_attempt=True,
        )
        prefix_record = execute_step(
            command=prefix_run,
            step_id="run_collective_prefix_witness",
            root=root,
            attempt_root=attempt_root,
            journal=journal,
        )
        command_records.append(prefix_record)
        # The command stdout is already an archived artifact.  Parse and copy
        # that file directly instead of leaving an unmanifested duplicate.
        prefix_output = prefix_record.stdout_path
        prefix_value = load_strict_json(prefix_output)
        if prefix_value.get("instance_id") != instance["instance_id"]:
            raise RunnerError("collective prefix witness instance_id mismatch")
    type_executable = full / "type_witness"
    type_compile = docker_command(
        entrypoint="/usr/local/cuda/bin/nvcc",
        guide_root=root,
        attempt_root=attempt_root,
        workdir="/workspace",
    ) + common + [
        "--generate-code=arch=compute_110a,code=sm_110a",
        instance["translation_units"]["type_witness"]["path"],
        "-o",
        "/out/full/type_witness",
    ]
    type_compile_record = capture_step(
        command=type_compile,
        step_id="compile_type_witness",
        root=root,
        attempt_root=attempt_root,
        journal=journal,
    )
    command_records.append(type_compile_record)
    if type_compile_record.returncode:
        if expected_outcome != "EXPECTED_STATIC_REJECT":
            raise RunnerError(
                f"compile_type_witness failed ({type_compile_record.returncode}); "
                f"see {type_compile_record.stderr_path}"
            )
        return seal_expected_static_reject(
            root=root,
            instance_path=instance_path,
            instance=instance,
            fingerprint=fingerprint,
            fingerprint_path=fingerprint_path,
            evidence_root=evidence_root,
            results_dir=results_dir,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            snapshot_paths=snapshot_paths,
            command_records=command_records,
            failed_record=type_compile_record,
            prefix_executable=prefix_executable,
            prefix_output=prefix_output,
            journal=journal,
            working_journal_path=working_journal_path,
            journal_path=journal_path,
        )
    if expected_outcome == "EXPECTED_STATIC_REJECT":
        raise RunnerError("frozen EXPECTED_STATIC_REJECT compiled successfully")
    type_run = docker_command(
        entrypoint="/out/full/type_witness",
        attempt_root=attempt_root,
        read_only_attempt=True,
    )
    type_record = execute_step(
        command=type_run,
        step_id="run_type_witness",
        root=root,
        attempt_root=attempt_root,
        journal=journal,
    )
    command_records.append(type_record)
    type_output = full / "type_witness.json"
    type_output.write_bytes(type_record.stdout_path.read_bytes())
    resolved = load_strict_json(type_output)
    if resolved.get("instance_id") != instance["instance_id"]:
        raise RunnerError("type witness instance_id mismatch")
    resolved_values = resolved.get("resolved_values")
    resolved_types = resolved.get("resolved_types")
    if not isinstance(resolved_values, dict) or not isinstance(resolved_types, dict):
        raise RunnerError("type witness lacks resolved types/values")
    if resolved_values.get("mainloop_stages", 0) <= 0:
        raise RunnerError("resolved mainloop stage count must be positive")
    for key in ("collective_mainloop", "dispatch_policy", "dispatch_schedule", "tiled_mma", "mma_atom"):
        if not isinstance(resolved_types.get(key), str) or not resolved_types[key]:
            raise RunnerError(f"type witness lacks {key}")

    fatbin = full / "kernel.fatbin"
    fatbin_compile = docker_command(
        entrypoint="/usr/local/cuda/bin/nvcc",
        guide_root=root,
        attempt_root=attempt_root,
        workdir="/workspace",
    ) + common + [
        "--fatbin",
        "--generate-code=arch=compute_110a,code=compute_110a",
        "--generate-code=arch=compute_110a,code=sm_110a",
        "--ptxas-options=-v",
        instance["translation_units"]["kernel"]["path"],
        "-o",
        "/out/full/kernel.fatbin",
    ]
    command_records.append(
        execute_step(
            command=fatbin_compile,
            step_id="compile_fatbin",
            root=root,
            attempt_root=attempt_root,
            journal=journal,
        )
    )

    def container_tool(step_id: str, entrypoint: str, argv: list[str], workdir: str | None = None):
        command = docker_command(
            entrypoint=entrypoint,
            attempt_root=attempt_root,
            workdir=workdir,
        ) + argv
        record = execute_step(
            command=command,
            step_id=step_id,
            root=root,
            attempt_root=attempt_root,
            journal=journal,
        )
        command_records.append(record)
        return record

    list_ptx = container_tool(
        "list_ptx", "/usr/local/cuda/bin/cuobjdump", ["--list-ptx", "/out/full/kernel.fatbin"]
    )
    list_elf = container_tool(
        "list_elf", "/usr/local/cuda/bin/cuobjdump", ["--list-elf", "/out/full/kernel.fatbin"]
    )
    ptx_listing = re.findall(r"(?m)^PTX file\s+\d+:\s+(\S+)\s*$", list_ptx.stdout_path.read_text())
    elf_listing = re.findall(r"(?m)^ELF file\s+\d+:\s+(\S+)\s*$", list_elf.stdout_path.read_text())
    if len(ptx_listing) != 1 or len(elf_listing) != 1:
        raise RunnerError(f"fatbin must contain one PTX and one ELF, found {ptx_listing}, {elf_listing}")
    if ".sm_110a." not in ptx_listing[0] or ".sm_110a." not in elf_listing[0]:
        raise RunnerError("fatbin list does not identify sm_110a PTX/ELF")
    container_tool(
        "extract_ptx",
        "/usr/local/cuda/bin/cuobjdump",
        ["--extract-ptx", "all", "/out/full/kernel.fatbin"],
        "/out/full/extracted",
    )
    container_tool(
        "extract_elf",
        "/usr/local/cuda/bin/cuobjdump",
        ["--extract-elf", "all", "/out/full/kernel.fatbin"],
        "/out/full/extracted",
    )
    ptx = one_extracted_file(extracted, ".ptx", "PTX extraction")
    cubin = one_extracted_file(extracted, ".cubin", "ELF extraction")
    symbols = container_tool(
        "elf_symbols",
        "/usr/local/cuda/bin/cuobjdump",
        ["--dump-elf-symbols", f"/out/full/extracted/{cubin.name}"],
    )
    metadata = container_tool(
        "code_object_metadata",
        "/usr/local/cuda/bin/cuobjdump",
        ["--dump-sass", f"/out/full/extracted/{cubin.name}"],
    )
    nvjson_record = container_tool(
        "nvdisasm_json",
        "/usr/local/cuda/bin/nvdisasm",
        ["--emit-json", "-c", f"/out/full/extracted/{cubin.name}"],
    )
    nvtext_record = container_tool(
        "nvdisasm_text",
        "/usr/local/cuda/bin/nvdisasm",
        ["-c", "-sf", f"/out/full/extracted/{cubin.name}"],
    )
    symbols_path = full / "elf-symbols.txt"
    metadata_path = full / "cuobjdump-sass.txt"
    nvjson_path = full / "nvdisasm.json"
    nvtext_path = full / "nvdisasm.txt"
    symbols_path.write_bytes(symbols.stdout_path.read_bytes())
    metadata_path.write_bytes(metadata.stdout_path.read_bytes())
    nvjson_path.write_bytes(nvjson_record.stdout_path.read_bytes())
    nvtext_path.write_bytes(nvtext_record.stdout_path.read_bytes())

    ptx_text = ptx.read_text(encoding="utf-8", errors="strict")
    ptx_entries = parse_ptx_entries(ptx_text)
    if len(ptx_entries) != 1:
        raise RunnerError(f"dedicated PTX must contain one .entry, found {len(ptx_entries)}")
    symbol = ptx_entries[0].symbol
    entries = parse_entry_symbols(symbols_path.read_text(encoding="utf-8"))
    if entries != (symbol,):
        raise RunnerError("sole PTX entry does not equal the sole ELF STO_ENTRY")
    ptx_contract = instance["static_contract"]["ptx"]
    sass_contract = instance["static_contract"]["sass"]
    attribution = attribute_codegen(
        ptx_text=ptx_text,
        nvdisasm_json_text=nvjson_path.read_text(encoding="utf-8"),
        elf_symbols_text=symbols_path.read_text(encoding="utf-8"),
        code_object_metadata_text=metadata_path.read_text(encoding="utf-8"),
        expected_symbol=symbol,
        declared_cta_group=instance["declared_config"]["cta_group"],
        require_tma_group_match=(
            instance["static_contract"]["tma_cta_policy"] == "match_mma"
        ),
        expected_ptx_target="sm_110a",
        expected_sass_arch="sm_110a",
        required_ptx=[item["regex"] for item in ptx_contract["required"]],
        forbidden_ptx=[item["regex"] for item in ptx_contract["forbidden"]],
        required_sass=[item["regex"] for item in sass_contract["required"]],
        forbidden_sass=[item["regex"] for item in sass_contract["forbidden"]],
    )
    expected_outcome = instance["hypothesis"]["expected_outcome"]
    terminal_status = "STATIC_PASS"
    guard_contract = load_strict_json(
        root / "tests/codegen/static_codegen_contract.json"
    )["arch_guard_fallback_contract"]
    guard_profile_id = instance["hypothesis"]["guard_profile_id"]
    guard_profile = (
        guard_contract["profiles"].get(guard_profile_id)
        if guard_profile_id is not None
        else None
    )
    guard_control_ref: dict[str, str] | None = None
    if attribution.passed:
        if expected_outcome != "STATIC_PASS":
            raise RunnerError(
                f"function-local contract passed but frozen outcome is {expected_outcome}"
            )
        ptx_results = pattern_results(
            ptx_contract["required"], attribution.ptx_contract.required_matches, True
        ) + pattern_results(
            ptx_contract["forbidden"], attribution.ptx_contract.forbidden_matches, False
        )
        sass_results = pattern_results(
            sass_contract["required"], attribution.sass_contract.required_matches, True
        ) + pattern_results(
            sass_contract["forbidden"], attribution.sass_contract.forbidden_matches, False
        )
    elif expected_outcome == "UNSUPPORTED_SM110A":
        if guard_profile is None:
            raise RunnerError("UNSUPPORTED_SM110A lacks a frozen guard profile")
        if guard_profile["guard_atom"] not in resolved_types["mma_atom"]:
            raise RunnerError("failed function contract does not use the frozen guarded MMA atom")
        for index, source_range in enumerate(guard_profile["source_constraints"]):
            try:
                validate_source_anchor(root, source_range, f"guard.source_constraints[{index}]")
            except ContractError as error:
                raise RunnerError(str(error)) from error
        try:
            ptx_results = evaluate_arch_guard_patterns(
                attribution.ptx_function.opcodes,
                guard_profile["ptx"],
                "architecture_guard.ptx",
            )
            sass_results = evaluate_arch_guard_patterns(
                attribution.sass_function.opcodes,
                guard_profile["sass"],
                "architecture_guard.sass",
            )
        except ContractError as error:
            raise RunnerError(str(error)) from error
        guard_control_ref = find_fresh_static_control(root, instance, fingerprint)
        terminal_status = "UNSUPPORTED_SM110A"
    else:
        raise RunnerError(
            f"function-local contract failed: ptx={attribution.ptx_contract}, "
            f"sass={attribution.sass_contract}, cta={attribution.cta_group_contract.errors}"
        )

    excerpt_dir = evidence_root / "excerpts" / attempt_id
    excerpt_dir.mkdir(parents=True, exist_ok=False)
    tracked_type_output = excerpt_dir / "resolved-types.json"
    ptx_excerpt = excerpt_dir / "target.ptx.txt"
    sass_excerpt = excerpt_dir / "target.sass.json"
    contract_report = excerpt_dir / "contract-report.json"
    shutil.copy2(type_output, tracked_type_output)
    # The tracked excerpt and the sealed command stdout are the two retained
    # copies.  Do not leave an unmanifested third copy in the ignored archive.
    type_output.unlink()
    ptx_excerpt.write_text(attribution.ptx_function.text + "\n", encoding="utf-8")
    sass_excerpt.write_text(attribution.sass_function.text + "\n", encoding="utf-8")
    if terminal_status == "STATIC_PASS":
        report_value = {
            "schema_version": 1,
            "instance_id": instance["instance_id"],
            "symbol": symbol,
            "ptx_target": attribution.ptx_target,
            "sass_arch": attribution.sass_arch,
            "nvdisasm_total_function_count": attribution.sass_total_function_count,
            "ptx_contract_results": ptx_results,
            "sass_contract_results": sass_results,
            "cta_group": {
                "declared": attribution.cta_group_contract.declared_cta_group,
                "tma_policy": instance["static_contract"]["tma_cta_policy"],
                "ptx_mma_opcodes": list(attribution.cta_group_contract.ptx_mma_opcodes),
                "sass_mma_opcodes": list(attribution.cta_group_contract.sass_mma_opcodes),
                "errors": list(attribution.cta_group_contract.errors),
            },
        }
    else:
        assert guard_control_ref is not None
        report_value = {
            "schema_version": 1,
            "instance_id": instance["instance_id"],
            "symbol": symbol,
            "ptx_target": attribution.ptx_target,
            "sass_arch": attribution.sass_arch,
            "nvdisasm_total_function_count": attribution.sass_total_function_count,
            "terminal_status": "UNSUPPORTED_SM110A",
            "reason": guard_contract["reason"],
            "guard_profile_id": guard_profile_id,
            "guard_atom": guard_profile["guard_atom"],
            "guard_macro": guard_profile["guard_macro"],
            "ptx_guard_results": ptx_results,
            "sass_guard_results": sass_results,
            "source_constraints": guard_profile["source_constraints"],
            "legal_control": {
                "result_ref": guard_control_ref,
                "relation": guard_profile["control_relation"],
                "controlled_delta": guard_profile["controlled_delta"],
            },
        }
    atomic_write_json(contract_report, report_value)

    if fingerprint_path.exists():
        if load_strict_json(fingerprint_path) != fingerprint:
            raise RunnerError(f"fingerprint ID collision at publication: {fingerprint_path}")
    else:
        atomic_write_json(fingerprint_path, fingerprint)
    validate_fingerprint(root, fingerprint_path)

    items: list[dict[str, Any]] = []
    add_artifact(items, root=root, artifact_id="config_source", role="KERNEL_SOURCE_SNAPSHOT", path=snapshot_paths["config"], storage="ignored_archive", media_type="text/x-c++hdr", producer_step_id="snapshot_sources", parents=[])
    add_artifact(items, root=root, artifact_id="kernel_source", role="KERNEL_SOURCE_SNAPSHOT", path=snapshot_paths["kernel"], storage="ignored_archive", media_type="text/x-cuda", producer_step_id="snapshot_sources", parents=["config_source"])
    add_artifact(items, root=root, artifact_id="type_source", role="TYPE_WITNESS_SOURCE_SNAPSHOT", path=snapshot_paths["type_witness"], storage="ignored_archive", media_type="text/x-cuda", producer_step_id="snapshot_sources", parents=["config_source"])
    add_artifact(items, root=root, artifact_id="type_executable", role="TYPE_WITNESS_EXECUTABLE", path=type_executable, storage="ignored_archive", media_type="application/x-executable", producer_step_id="compile_type_witness", parents=["type_source", "config_source"])
    add_artifact(items, root=root, artifact_id="type_output", role="TYPE_WITNESS_OUTPUT", path=tracked_type_output, storage="git_evidence", media_type="application/json", producer_step_id="run_type_witness", parents=["type_executable"])
    add_artifact(items, root=root, artifact_id="fatbin", role="FATBIN", path=fatbin, storage="ignored_archive", media_type="application/x-fatbin", producer_step_id="compile_fatbin", parents=["kernel_source", "config_source"])
    add_artifact(items, root=root, artifact_id="ptx_full", role="PTX_FULL", path=ptx, storage="ignored_archive", media_type="text/x-ptx", producer_step_id="extract_ptx", parents=["fatbin"])
    add_artifact(items, root=root, artifact_id="cubin", role="CUBIN", path=cubin, storage="ignored_archive", media_type="application/x-cubin", producer_step_id="extract_elf", parents=["fatbin"])
    add_artifact(items, root=root, artifact_id="elf_symbols", role="ELF_SYMBOL_TABLE", path=symbols_path, storage="ignored_archive", media_type="text/plain", producer_step_id="elf_symbols", parents=["cubin"])
    add_artifact(items, root=root, artifact_id="code_metadata", role="CODE_OBJECT_METADATA", path=metadata_path, storage="ignored_archive", media_type="text/plain", producer_step_id="code_object_metadata", parents=["cubin"])
    add_artifact(items, root=root, artifact_id="nvdisasm_json", role="NVDISASM_JSON_FULL", path=nvjson_path, storage="ignored_archive", media_type="application/json", producer_step_id="nvdisasm_json", parents=["cubin"])
    add_artifact(items, root=root, artifact_id="nvdisasm_text", role="NVDISASM_TEXT_FULL", path=nvtext_path, storage="ignored_archive", media_type="text/plain", producer_step_id="nvdisasm_text", parents=["cubin"])
    for record in command_records:
        add_artifact(items, root=root, artifact_id=f"{record.step_id}_stdout", role="COMPILE_STDOUT", path=record.stdout_path, storage="ignored_archive", media_type="text/plain", producer_step_id=record.step_id, parents=[])
        add_artifact(items, root=root, artifact_id=f"{record.step_id}_stderr", role="COMPILE_STDERR", path=record.stderr_path, storage="ignored_archive", media_type="text/plain", producer_step_id=record.step_id, parents=[])
    add_artifact(items, root=root, artifact_id="ptx_target", role="PTX_TARGET_FUNCTION", path=ptx_excerpt, storage="git_evidence", media_type="text/x-ptx", producer_step_id="function_contract", parents=["ptx_full"])
    add_artifact(items, root=root, artifact_id="sass_target", role="SASS_TARGET_FUNCTION", path=sass_excerpt, storage="git_evidence", media_type="application/json", producer_step_id="function_contract", parents=["nvdisasm_json"])
    add_artifact(items, root=root, artifact_id="contract_report", role="CONTRACT_CHECK_REPORT", path=contract_report, storage="git_evidence", media_type="application/json", producer_step_id="function_contract", parents=["ptx_target", "sass_target", "type_output"])
    archive_items = [item for item in items if item["storage"] == "ignored_archive"]
    archive_checksum = sha256_bytes(
        canonical_json_bytes(
            [
                (item["artifact_id"], item["path"], item["sha256"], item["size_bytes"])
                for item in archive_items
            ]
        )
    )
    manifest_id = f"am-{attempt_id}"
    manifest = {
        "schema_version": 1,
        "contract_sha256": contract_sha256(root),
        "artifact_manifest_id": manifest_id,
        "attempt_id": attempt_id,
        "instance_ref": file_ref(root, instance_path, identifier=instance["instance_id"]),
        "fingerprint_ref": file_ref(
            root, fingerprint_path, identifier=fingerprint["fingerprint_id"]
        ),
        "items": items,
        "archive_bundle": {
            "path": attempt_root.relative_to(root).as_posix(),
            "format": "directory-manifest",
            "sha256": archive_checksum,
            "size_bytes": sum(item["size_bytes"] for item in archive_items),
        },
    }
    for item in items:
        journal.append(
            "ARTIFACT_SEALED",
            {
                "artifact_id": item["artifact_id"],
                "path": item["path"],
                "sha256": item["sha256"],
                "producer_step_id": item["producer_step_id"],
            },
        )
    manifest_path = evidence_root / "artifact-manifests" / f"{manifest_id}.json"
    atomic_write_json(manifest_path, manifest)
    validate_artifact_manifest(root, manifest_path)
    journal.append(
        "ARTIFACT_MANIFEST_SEALED",
        {"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256_file(manifest_path)},
    )
    journal.append(
        "ATTEMPT_SEALED",
        {
            "terminal_pipeline_state": terminal_status,
            "artifact_manifest_path": manifest_path.relative_to(root).as_posix(),
            "artifact_manifest_sha256": sha256_file(manifest_path),
            "last_completed_step_id": "function_contract",
        },
    )
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_temporary = journal_path.with_name(journal_path.name + f".tmp-{os.getpid()}")
    journal_temporary.write_bytes(working_journal_path.read_bytes())
    os.replace(journal_temporary, journal_path)
    working_journal_path.unlink()
    result_id = attempt_id
    layer_artifacts = {
        "DECLARED_BUILDER_CONFIG": "config_source",
        "BUILDER_SPECIALIZATION": "type_output",
        "DISPATCH_POLICY": "type_output",
        "COLLECTIVE": "type_output",
        "STAGE": "type_output",
        "COPY_LAYOUT": "type_output",
        "TILED_MMA": "type_output",
        "ATOM": "type_output",
        "KERNEL_COMPOSITION": "type_output",
        "PTX_EMISSION": "ptx_full",
        "SM110A_ASSEMBLY": "cubin",
        "FUNCTION_BINDING": "elf_symbols",
        "FUNCTION_CONTRACT": "contract_report",
    }
    layer_order = load_strict_json(
        root / "tests/codegen/static_codegen_contract.json"
    )["layer_order"]
    layer_outcomes = []
    for layer in layer_order:
        state = (
            "REJECTED"
            if terminal_status == "UNSUPPORTED_SM110A" and layer == "FUNCTION_CONTRACT"
            else "RESOLVED"
        )
        layer_outcomes.append(
            {
                "layer": layer,
                "state": state,
                "witness_artifact_id": layer_artifacts[layer],
            }
        )
    resolved_stage = {
        "declared_policy_cpp": instance["declared_config"]["stage_policy"],
        "resolved_policy_cpp": resolved_types["config_stage_policy"],
        "stage_count": resolved_values["mainloop_stages"],
        "scheduler_stages": resolved_values["scheduler_stages"],
        "accumulator_stages": resolved_values["accumulator_stages"],
        "load_to_transform_stages": resolved_values["load_to_transform_stages"],
        "transform_to_mma_stages": resolved_values["transform_to_mma_stages"],
        "computation_stages": resolved_values["computation_stages"],
        "transformation_stages": resolved_values["transformation_stages"],
    }
    function_binding = {
        "target_cpp_entity": "cutlass::device_kernel<GemmKernel>",
        "selection_policy": "sole_ptx_entry_equals_sole_elf_sto_entry_equals_unique_nvdisasm_function",
        "symbol": symbol,
        "ptx_entry_count": 1,
        "elf_sto_entry_count": 1,
        "nvdisasm_symbol_match_count": 1,
        "nvdisasm_total_function_count": attribution.sass_total_function_count,
        "ptx_function_artifact_id": "ptx_target",
        "sass_function_artifact_id": "sass_target",
        "same_symbol": True,
    }
    if terminal_status == "STATIC_PASS":
        evidence_value = {
            "kind": "STATIC_PASS",
            "type_witness_artifact_id": "type_output",
            "resolved_stage": resolved_stage,
            "function_binding": function_binding,
            "ptx_contract_results": ptx_results,
            "sass_contract_results": sass_results,
            "cta_group_contract_pass": True,
        }
    else:
        assert guard_control_ref is not None
        evidence_value = {
            "kind": "UNSUPPORTED_SM110A",
            "reason": guard_contract["reason"],
            "failure": {
                "layer": guard_contract["failure_layer"],
                "domain": "TARGET_ARCHITECTURE",
                "reason": guard_contract["reason"],
            },
            "type_witness_artifact_id": "type_output",
            "resolved_stage": resolved_stage,
            "function_binding": function_binding,
            "guard_profile_id": guard_profile_id,
            "guard_atom": guard_profile["guard_atom"],
            "guard_macro": guard_profile["guard_macro"],
            "ptx_guard_results": ptx_results,
            "sass_guard_results": sass_results,
            "source_constraints": guard_profile["source_constraints"],
            "legal_control": {
                "result_ref": guard_control_ref,
                "relation": guard_profile["control_relation"],
                "controlled_delta": guard_profile["controlled_delta"],
            },
        }
    result = {
        "schema_version": 1,
        "contract_sha256": contract_sha256(root),
        "objective_sha256": OBJECTIVE_SHA256,
        "freshness_epoch": fingerprint["freshness_epoch"],
        "result_id": result_id,
        "scope": "STATIC_CODEGEN_ONLY",
        "subject": instance["subject"],
        "instance_ref": file_ref(root, instance_path, identifier=instance["instance_id"]),
        "fingerprint_ref": file_ref(
            root, fingerprint_path, identifier=fingerprint["fingerprint_id"]
        ),
        "attempt_id": attempt_id,
        "journal_ref": {
            "path": journal_path.relative_to(root).as_posix(),
            "sha256": sha256_file(journal_path),
            "last_seq": journal.seq,
            "last_event_sha256": journal.previous,
        },
        "artifact_manifest_ref": file_ref(root, manifest_path, identifier=manifest_id),
        "status": terminal_status,
        "layer_outcomes": layer_outcomes,
        "evidence": evidence_value,
    }
    result_path = results_dir / f"{result_id}.json"
    atomic_write_json(result_path, result)
    validate_result(root, result_path)
    print(f"{terminal_status} instance={instance['instance_id']} result={result_id}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--instance", action="append", default=[])
    parser.add_argument("--campaign")
    parser.add_argument("--all-phase1", action="store_true")
    parser.add_argument("--all-phase2", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    selected_campaigns = [
        value
        for value in (
            args.campaign,
            "phase1-generalized-20260830" if args.all_phase1 else None,
            "phase2-official-20260830" if args.all_phase2 else None,
        )
        if value is not None
    ]
    if len(selected_campaigns) > 1:
        raise RunnerError("select exactly one campaign")
    campaign_path: Path | None = None
    campaign: dict[str, Any] | None = None
    if selected_campaigns:
        if args.instance or args.run_id is not None:
            raise RunnerError("formal --campaign runs cannot be mixed with --instance/--run-id")
        try:
            campaign_path, campaign = load_run_campaign(root, selected_campaigns[0])
        except ContractError as error:
            raise RunnerError(str(error)) from error
        run_id = campaign["run_id"]
        requested = list(campaign["ordered_instances"])
    else:
        run_id = args.run_id or "ad-hoc-codegen"
        requested = list(args.instance)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", run_id):
        raise RunnerError("run-id must be a safe ASCII identifier")
    requested = list(dict.fromkeys(requested))
    if not requested:
        raise RunnerError("select --campaign or at least one --instance")
    toolchain = inspect_environment(root)
    results = []
    for instance_id in requested:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", instance_id):
            raise RunnerError(f"unsafe instance ID: {instance_id}")
        instance_path = root / "tests/codegen/instances" / instance_id / "instance.json"
        results.append(
            run_instance(
                root=root,
                instance_path=instance_path,
                run_id=run_id,
                toolchain=toolchain,
                resume=args.resume,
            )
        )
    result_refs = [
        file_ref(
            root,
            root / "evidence/codegen-sm110a-v2/results" / f"{result['result_id']}.json",
            identifier=result["result_id"],
        )
        for result in results
    ]
    summary_path = (
        root / campaign["current_summary"]
        if campaign is not None
        else root / "evidence/codegen-sm110a-v2" / f"summary-{run_id}.json"
    )
    history_refs: list[dict[str, str]] = []
    if summary_path.exists():
        previous = load_strict_json(summary_path)
        if (
            previous.get("schema_version") != 1
            or previous.get("run_id") != run_id
            or previous.get("scope") != "STATIC_CODEGEN_ONLY"
            or previous.get("campaign_ref")
            != (
                file_ref(root, campaign_path, identifier=campaign["campaign_id"])
                if campaign is not None and campaign_path is not None
                else None
            )
            or not isinstance(previous.get("result_refs"), list)
            or not isinstance(previous.get("history_result_refs"), list)
        ):
            raise RunnerError(f"cannot extend malformed existing summary: {summary_path}")
        current_ids = {ref["id"] for ref in result_refs}
        candidates = previous["history_result_refs"] + previous["result_refs"]
        seen_history: set[str] = set()
        for ref in candidates:
            if ref["id"] in current_ids or ref["id"] in seen_history:
                continue
            seen_history.add(ref["id"])
            history_refs.append(ref)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "scope": "STATIC_CODEGEN_ONLY",
        "campaign_ref": (
            file_ref(root, campaign_path, identifier=campaign["campaign_id"])
            if campaign is not None and campaign_path is not None
            else None
        ),
        "result_refs": result_refs,
        "history_result_refs": history_refs,
    }
    atomic_write_json(summary_path, summary)
    validation_command = [
        sys.executable,
        str(root / "tools/validate_codegen_v2.py"),
        "--root",
        str(root),
        "--require-archive",
    ]
    if campaign is not None:
        validation_command += ["--require-campaign", campaign["campaign_id"]]
    validation = run_checked(validation_command, cwd=root)
    print(validation.stdout.decode("utf-8", errors="replace").strip())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RunnerError, ContractError, AttributionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
