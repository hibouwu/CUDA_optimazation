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
    expected_execution_plan,
    file_ref,
    hash_input_tree,
    journal_event_hash,
    load_strict_json,
    safe_path,
    sha256_bytes,
    sha256_file,
    validate_artifact_manifest,
    validate_fingerprint,
    validate_instance,
    validate_result,
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


def execute_step(
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
    if result.returncode:
        raise RunnerError(
            f"{step_id} failed ({result.returncode}); see {stdout_path} and {stderr_path}"
        )
    return CommandRecord(step_id, stdout_path, stderr_path, result.returncode)


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
    working_journal_path = attempt_root / "journal.in-progress.jsonl"
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
    command_records: list[CommandRecord] = []
    command_records.append(
        execute_step(
            command=type_compile,
            step_id="compile_type_witness",
            root=root,
            attempt_root=attempt_root,
            journal=journal,
        )
    )
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
        expected_ptx_target="sm_110a",
        expected_sass_arch="sm_110a",
        required_ptx=[item["regex"] for item in ptx_contract["required"]],
        forbidden_ptx=[item["regex"] for item in ptx_contract["forbidden"]],
        required_sass=[item["regex"] for item in sass_contract["required"]],
        forbidden_sass=[item["regex"] for item in sass_contract["forbidden"]],
    )
    if not attribution.passed:
        raise RunnerError(
            f"function-local contract failed: ptx={attribution.ptx_contract}, "
            f"sass={attribution.sass_contract}, cta={attribution.cta_group_contract.errors}"
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

    excerpt_dir = evidence_root / "excerpts" / attempt_id
    excerpt_dir.mkdir(parents=True, exist_ok=False)
    tracked_type_output = excerpt_dir / "resolved-types.json"
    ptx_excerpt = excerpt_dir / "target.ptx.txt"
    sass_excerpt = excerpt_dir / "target.sass.json"
    contract_report = excerpt_dir / "contract-report.json"
    shutil.copy2(type_output, tracked_type_output)
    ptx_excerpt.write_text(attribution.ptx_function.text + "\n", encoding="utf-8")
    sass_excerpt.write_text(attribution.sass_function.text + "\n", encoding="utf-8")
    atomic_write_json(
        contract_report,
        {
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
                "ptx_mma_opcodes": list(attribution.cta_group_contract.ptx_mma_opcodes),
                "sass_mma_opcodes": list(attribution.cta_group_contract.sass_mma_opcodes),
                "errors": list(attribution.cta_group_contract.errors),
            },
        },
    )

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
            "terminal_pipeline_state": "STATIC_PASS",
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
        "status": "STATIC_PASS",
        "layer_outcomes": [
            {"layer": layer, "state": "RESOLVED", "witness_artifact_id": layer_artifacts[layer]}
            for layer in load_strict_json(root / "tests/codegen/static_codegen_contract.json")[
                "layer_order"
            ]
        ],
        "evidence": {
            "kind": "STATIC_PASS",
            "type_witness_artifact_id": "type_output",
            "resolved_stage": {
                "declared_policy_cpp": instance["declared_config"]["stage_policy"],
                "resolved_policy_cpp": resolved_types["config_stage_policy"],
                "stage_count": resolved_values["mainloop_stages"],
                "scheduler_stages": resolved_values["scheduler_stages"],
                "accumulator_stages": resolved_values["accumulator_stages"],
            },
            "function_binding": {
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
            },
            "ptx_contract_results": ptx_results,
            "sass_contract_results": sass_results,
            "cta_group_contract_pass": True,
        },
    }
    result_path = results_dir / f"{result_id}.json"
    atomic_write_json(result_path, result)
    validate_result(root, result_path)
    print(f"STATIC_PASS instance={instance['instance_id']} result={result_id}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--instance", action="append", default=[])
    parser.add_argument("--all-phase1", action="store_true")
    parser.add_argument("--run-id", default="phase1-fresh-20260830")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", args.run_id):
        raise RunnerError("run-id must be a safe ASCII identifier")
    requested = list(args.instance)
    if args.all_phase1:
        requested.extend(
            [
                "dense_f16_1sm",
                "dense_f16_2sm",
                "dense_bs_nvfp4_1sm",
                "dense_bs_mxf4_1sm",
                "dense_bs_mxf8_1sm",
                "sparse_bs_nvfp4_1sm",
            ]
        )
    requested = list(dict.fromkeys(requested))
    if not requested:
        raise RunnerError("select --instance or --all-phase1")
    contract = load_strict_json(root / "tests/codegen/static_codegen_contract.json")
    if args.run_id == contract["phase1_run_id"] and requested != contract[
        "phase1_fresh_replay_instances"
    ]:
        raise RunnerError("the frozen Phase 1 run_id requires the complete ordered replay set")
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
                run_id=args.run_id,
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
    summary_path = root / "evidence/codegen-sm110a-v2" / f"summary-{args.run_id}.json"
    history_refs: list[dict[str, str]] = []
    if summary_path.exists():
        previous = load_strict_json(summary_path)
        if (
            previous.get("schema_version") != 1
            or previous.get("run_id") != args.run_id
            or previous.get("scope") != "STATIC_CODEGEN_ONLY"
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
        "run_id": args.run_id,
        "scope": "STATIC_CODEGEN_ONLY",
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
    if args.run_id == contract["phase1_run_id"]:
        validation_command.append("--require-results")
    validation = run_checked(validation_command, cwd=root)
    print(validation.stdout.decode("utf-8", errors="replace").strip())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RunnerError, ContractError, AttributionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
