#!/usr/bin/env python3
"""Validate the exhaustive 59-tag SM100 Tensor Core codegen coverage contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


GUIDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GUIDE_ROOT / "tools"))

from codegen_v2.model import ContractError, validate_result  # noqa: E402
from validate_codegen_v2 import contract_bundle_sha256  # noqa: E402


EXPECTED_CUTLASS_SHA = "e05f953a5b3d38adc240df2ff928e0421c2abba3"
EXPECTED_OBJECTIVE_SHA256 = "463fa7015808acd883b28d115fa33708f66064aaceed96f728996a01ca0e4091"
EXPECTED_DISPATCH_POLICY_SHA256 = "fcce5fffb3118b15fea5aa59e39bea15ddd18c28c858b434ced889045968117c"
EXPECTED_REFERENCE_INVENTORY_SHA256 = "1370b33695542ceae8103a6280dd422e3122dcdd53cd0c69d55a92853d3cd296"
EXPECTED_AUTO_INVENTORY_SHA256 = "de7032f8234737b747b0d962c37809bb1b4b843adf7d26b12f736a0e0b15bfc3"
EXPECTED_VERSIONS_LOCK_FILE_SHA256 = "d0c38e0759646fdb5950669aa055f564d43c7bb60984bbc5cacbff4bfcdf242a"
EXPECTED_RESULT_CONTRACT_SHA256: str | None = "b4d8852e9e962cb57dd718d45709272ef47cd6c228866f917e40be3edc6fda64"
EXPECTED_TAG_COUNT = 59
EXPECTED_TAG_SET_SHA256 = "95ae141fd872bb7894937b145b2f66cc488e089d2b84b55efaf9d25bdf5da1e7"
EXPECTED_TAG_GROUP_SHA256 = "bc88440aad196855d40af50bfe3ee1f84356f5a5aa0ae9a4b60a9549c01d995b"
EXPECTED_GROUPS = [
    {"id": "dense", "title": "普通 Dense", "expected_count": 5},
    {"id": "ptr_array_dense", "title": "Pointer-array Dense", "expected_count": 2},
    {"id": "blockwise", "title": "Blockwise", "expected_count": 4},
    {"id": "planar_complex", "title": "Planar Complex", "expected_count": 4},
    {"id": "fast_fp32", "title": "Fast FP32 / 9xBF16", "expected_count": 8},
    {"id": "mixed_input", "title": "Mixed-input", "expected_count": 4},
    {"id": "interleaved_complex_tf32", "title": "Interleaved Complex TF32", "expected_count": 4},
    {"id": "sparse", "title": "普通 Sparse", "expected_count": 2},
    {"id": "dense_block_scaled", "title": "Dense Block-scaled", "expected_count": 10},
    {"id": "ptr_array_block_scaled", "title": "Pointer-array Block-scaled", "expected_count": 8},
    {"id": "sparse_block_scaled", "title": "Sparse Block-scaled", "expected_count": 8},
]
EXPECTED_GROUP_COUNTS = {group["id"]: group["expected_count"] for group in EXPECTED_GROUPS}
EXPECTED_COVERAGE_POLICY = {
    "omission_allowed": False,
    "out_of_scope_allowed": False,
    "performance_required_per_tag": False,
    "runtime_correctness_required_per_tag": False,
    "static_codegen_required_per_tag": True,
    "required_static_evidence": {
        "STATIC_PASS": [
            "declared_builder_config",
            "resolved_builder_specialization",
            "resolved_cross_layer_types",
            "compile_acceptance",
            "function_local_ptx_contract",
            "function_local_sass_contract",
            "same_target_symbol",
        ],
        "EXPECTED_STATIC_REJECT": [
            "declared_builder_config",
            "legal_sibling_or_parent_control",
            "compile_rejection",
            "diagnostic_contract",
            "failure_layer_attribution",
        ],
        "UNSUPPORTED_SM110A": [
            "declared_builder_config",
            "resolved_upper_type_chain",
            "sm110a_target_rejection",
            "diagnostic_contract",
            "architecture_failure_attribution",
        ],
    },
    "allowed_statuses": [
        "NOT_CHECKED",
        "HISTORICAL_STATIC_PASS",
        "STATIC_PASS",
        "EXPECTED_STATIC_REJECT",
        "UNSUPPORTED_SM110A",
        "UNEXPECTED_COMPILE_FAIL",
        "ATTRIBUTION_FAIL",
    ],
}
EXPECTED_SCOPE = {
    "explicit_tensor_schedule_tags": 59,
    "auto_control_groups": 11,
    "simt_explicit_schedules": 0,
    "requires_thor_gpu": False,
    "requires_runtime_correctness": False,
    "requires_performance": False,
}
EXPECTED_CLAIM_BOUNDARY = {
    "coverage_unit": "one_canonical_instance_per_explicit_schedule_tag",
    "exhaustive_template_domain": False,
    "static_codegen_is_runtime_correctness": False,
    "static_codegen_is_performance": False,
    "target_function_scope": "one_unique_kernel_function_per_successfully_compiled_instance",
    "same_function_scope": "ptx_and_sass_evidence_within_one_instance",
    "rejected_instance_scope": "terminate_at_an_evidence_backed_failure_layer",
}
EXPECTED_REFERENCE_CLASS_COUNTS = {
    "official_cpp_explicit_or_conditional": 39,
    "official_cpp_auto_derived": 1,
    "generator_config": 5,
    "source_derived": 14,
}
EXPECTED_REFERENCE_SELECTION_POLICY = [
    "test_before_example",
    "shortest_normalized_relative_path",
    "lexicographic_path",
    "first_text_occurrence",
]
EXPECTED_PHASE_LOOP = [
    "implement",
    "positive_validation",
    "deliberate_breakage",
    "adversarial_audit",
    "repair",
    "full_phase_rerun",
]
EXPECTED_PHASE_GATE = {
    "allow_advance_with_unexplained_failures": False,
    "allow_advance_with_omissions": False,
    "allow_evidence_promotion": False,
}
EXPECTED_FINAL_STATUSES = ["STATIC_PASS", "EXPECTED_STATIC_REJECT", "UNSUPPORTED_SM110A"]
EXPECTED_PHASES = [
    (0, "workspace_source_inventory"),
    (1, "cross_layer_harness"),
    (2, "official_cpp_tags"),
    (3, "generator_config_tags"),
    (4, "source_derived_tags"),
    (5, "auto_and_clean_replay"),
    (6, "documentation_archive_final_audit"),
]
INTERIM_PHASE_STATUSES = {"PENDING", "IN_PROGRESS", "COMPLETE"}
RESULT_BACKED_STATUSES = {
    "STATIC_PASS",
    "EXPECTED_STATIC_REJECT",
    "UNSUPPORTED_SM110A",
    "UNEXPECTED_COMPILE_FAIL",
    "ATTRIBUTION_FAIL",
}


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: cannot parse JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top level must be an object")
    return value


def discover_source_tags(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    marker = "struct KernelScheduleSm100"
    if marker not in source:
        raise ValueError(f"{path}: cannot locate {marker}")
    user_schedule_region = source[source.index(marker) :]
    return set(
        re.findall(
            r"struct\s+(Kernel[A-Za-z0-9_]*Sm100)\s+final",
            user_schedule_region,
        )
    )


def discover_official_cpp_references(cutlass_root: Path, tags: set[str]) -> dict[str, tuple[str, int, str]]:
    """Recompute the deterministic first test/example text occurrence for each public Tag."""

    candidates: dict[str, list[tuple[int, int, str, int, str]]] = {tag: [] for tag in tags}
    token_pattern = re.compile(r"Kernel[A-Za-z0-9_]*Sm100")
    roots = [
        (0, cutlass_root / "test/unit/gemm/device"),
        (1, cutlass_root / "examples"),
    ]
    for priority, search_root in roots:
        if not search_root.is_dir():
            continue
        for source_path in sorted(search_root.rglob("*.cu")):
            relative_path = source_path.relative_to(cutlass_root).as_posix()
            try:
                lines = source_path.read_text(encoding="utf-8").splitlines()
            except OSError as error:
                raise ValueError(f"cannot scan official C++ reference {source_path}: {error}") from error
            for line_number, line in enumerate(lines, start=1):
                for token in set(token_pattern.findall(line)) & tags:
                    candidates[token].append(
                        (priority, len(relative_path), relative_path, line_number, line)
                    )

    selected: dict[str, tuple[str, int, str]] = {}
    for tag, occurrences in candidates.items():
        if occurrences:
            _, _, path, line_number, line = min(occurrences)
            selected[tag] = (path, line_number, line)
    return selected


def load_result_record(root: Path, result_id: object, subject: str, errors: list[str]) -> dict | None:
    if not isinstance(result_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", result_id):
        errors.append(f"{subject}: evidence-backed status requires a safe result_id")
        return None
    result_path = root / "evidence/codegen-sm110a-v2/results" / f"{result_id}.json"
    try:
        result = validate_result(root, result_path, require_archive=False)
    except (ContractError, OSError) as error:
        errors.append(f"{subject}: result record unavailable: {error}")
        return None
    if result.get("result_id") != result_id or result_path.stem != result_id:
        errors.append(f"{subject}: result record identity mismatch")
        return None
    return result


def validate_completion_report(
    root: Path,
    phase: dict,
    objective: str,
    errors: list[str],
) -> None:
    phase_id = phase.get("id", "<missing>")
    pointer = phase.get("completion_report")
    if not isinstance(pointer, dict):
        errors.append(f"phase {phase_id}: COMPLETE requires a completion_report pointer")
        return
    path_value = pointer.get("path")
    if not isinstance(path_value, str):
        errors.append(f"phase {phase_id}: completion report path must be a string")
        return
    report_path_value = Path(path_value)
    expected_prefix = Path("evidence/codegen-sm110a-v2/phase-reports")
    if (
        report_path_value.is_absolute()
        or ".." in report_path_value.parts
        or report_path_value.parts[: len(expected_prefix.parts)] != expected_prefix.parts
    ):
        errors.append(f"phase {phase_id}: completion report path escapes the evidence directory")
        return
    report_path = root / report_path_value
    try:
        report_bytes = report_path.read_bytes()
        report_text = report_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        errors.append(f"phase {phase_id}: completion report unavailable: {error}")
        return
    if hashlib.sha256(report_bytes).hexdigest() != pointer.get("sha256"):
        errors.append(f"phase {phase_id}: completion report SHA-256 mismatch")
    if pointer.get("objective_sha256") != EXPECTED_OBJECTIVE_SHA256:
        errors.append(f"phase {phase_id}: completion report objective hash mismatch")
    section_headings = ["## 阶段开始", "## 对抗审查", "## 修复", "## 阶段完成"]
    for heading in section_headings:
        match = re.search(
            rf"^{re.escape(heading)}\s*$([\s\S]*?)(?=^##\s|\Z)",
            report_text,
            flags=re.MULTILINE,
        )
        if match is None or objective not in match.group(1):
            errors.append(
                f"phase {phase_id}: completion report section {heading!r} must repeat the exact objective"
            )
    required_markers = [
        "positive_validation: PASS",
        "deliberate_breakage: PASS",
        "adversarial_audit: PASS",
        "full_phase_rerun: PASS",
        "unexplained_failures: 0",
        "omissions: 0",
        "evidence_promotions: 0",
    ]
    for marker in required_markers:
        if marker not in report_text:
            errors.append(f"phase {phase_id}: completion report missing marker {marker!r}")


def sha256_text(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def run_git(cutlass_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(cutlass_root), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as error:
        raise ValueError(f"cannot run git for CUTLASS source identity: {error}") from error
    if result.returncode:
        raise ValueError(
            f"CUTLASS source identity command failed: git {' '.join(args)}: {result.stdout.strip()}"
        )
    return result.stdout.strip()


def normalized_schedule(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.removeprefix("cutlass::gemm::")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--require-source",
        action="store_true",
        help="fail unless the exact clean pinned CUTLASS checkout is available",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    contract_path = root / "tests/codegen/campaign_contract.json"
    manifest_path = root / "tests/codegen/sm110a_tensor_schedule_tags.json"
    document_path = root / "docs/02-tensor-core-tcgen05-gemm-codegen-formal-validation.md"
    contract = load_json(contract_path)
    manifest = load_json(manifest_path)
    document = document_path.read_text(encoding="utf-8")
    errors: list[str] = []
    campaign_complete = False
    harness_complete = False

    if EXPECTED_RESULT_CONTRACT_SHA256 is not None:
        try:
            result_contract = load_json(root / "tests/codegen/static_codegen_contract.json")
            observed_result_contract = contract_bundle_sha256(root, result_contract)
        except (ContractError, OSError, ValueError) as error:
            errors.append(f"result-contract bundle unavailable: {error}")
        else:
            if observed_result_contract != EXPECTED_RESULT_CONTRACT_SHA256:
                errors.append("result-contract bundle differs from the Phase 1 frozen harness")

    if contract.get("schema_version") != 1:
        errors.append("campaign schema_version must be 1")
    objective = contract.get("first_principles_objective")
    objective_sha256 = contract.get("objective_sha256")
    if not isinstance(objective, str) or not objective:
        errors.append("campaign objective must be a nonempty string")
    elif hashlib.sha256(objective.encode("utf-8")).hexdigest() != objective_sha256:
        errors.append("campaign objective SHA-256 mismatch")
    if objective_sha256 != EXPECTED_OBJECTIVE_SHA256:
        errors.append("campaign objective differs from the frozen first-principles objective")
    if manifest.get("objective_sha256") != objective_sha256:
        errors.append("manifest objective SHA-256 does not match campaign contract")
    if contract.get("scope") != EXPECTED_SCOPE:
        errors.append("campaign scope differs from the frozen static-only 59+11 scope")
    if contract.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY:
        errors.append("campaign claim boundary must remain one canonical static instance per Tag")
    if contract.get("phase_loop") != EXPECTED_PHASE_LOOP:
        errors.append("campaign phase loop differs from the required adversarial loop")
    if contract.get("phase_gate") != EXPECTED_PHASE_GATE:
        errors.append("campaign phase gate must reject failures, omissions, and evidence promotion")
    if contract.get("accepted_final_statuses") != EXPECTED_FINAL_STATUSES:
        errors.append("campaign accepted_final_statuses differ from the frozen evidence states")
    phases = contract.get("phases")
    if not isinstance(phases, list) or len(phases) != len(EXPECTED_PHASES):
        errors.append("campaign phases must contain the seven frozen phases")
    else:
        actual_phase_identity = [(phase.get("id"), phase.get("name")) for phase in phases if isinstance(phase, dict)]
        if actual_phase_identity != EXPECTED_PHASES:
            errors.append("campaign phase identity or order differs from the frozen plan")
        elif isinstance(phases[1], dict) and phases[1].get("status") == "COMPLETE":
            harness_complete = True
        invalid_phase_statuses = [
            phase.get("status") if isinstance(phase, dict) else phase
            for phase in phases
            if not isinstance(phase, dict) or phase.get("status") not in INTERIM_PHASE_STATUSES
        ]
        if invalid_phase_statuses:
            errors.append(f"campaign contains invalid phase statuses: {invalid_phase_statuses}")
        in_progress = sum(
            isinstance(phase, dict) and phase.get("status") == "IN_PROGRESS" for phase in phases
        )
        if in_progress == 1 and not invalid_phase_statuses:
            statuses = [phase["status"] for phase in phases]
            active_index = statuses.index("IN_PROGRESS")
            if any(status != "COMPLETE" for status in statuses[:active_index]) or any(
                status != "PENDING" for status in statuses[active_index + 1 :]
            ):
                errors.append("campaign phase statuses must form COMPLETE* -> IN_PROGRESS -> PENDING*")
        elif in_progress == 0 and not invalid_phase_statuses:
            statuses = [phase["status"] for phase in phases]
            if all(status == "COMPLETE" for status in statuses):
                campaign_complete = True
            else:
                errors.append(
                    "campaign without an IN_PROGRESS phase is valid only when all phases are COMPLETE"
                )
        elif in_progress > 1:
            errors.append(f"campaign may have at most one IN_PROGRESS phase, found {in_progress}")
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            if phase.get("status") == "COMPLETE":
                validate_completion_report(
                    root,
                    phase,
                    objective if isinstance(objective, str) else "",
                    errors,
                )
            elif phase.get("completion_report") is not None:
                errors.append(
                    f"phase {phase.get('id')}: only COMPLETE phases may carry completion_report"
                )

    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    target = manifest.get("target_contract", {})
    if target.get("cutlass_git_sha") != EXPECTED_CUTLASS_SHA:
        errors.append("manifest CUTLASS SHA does not match the guide lock")
    if target.get("cutlass_arch_tag") != "cutlass::arch::Sm100":
        errors.append("target contract must use cutlass::arch::Sm100 as the C++ recipe key")
    if target.get("virtual_arch") != "compute_110a" or target.get("binary_arch") != "sm_110a":
        errors.append("target contract must be compute_110a -> sm_110a")

    policy = manifest.get("coverage_policy", {})
    if policy != EXPECTED_COVERAGE_POLICY:
        errors.append("coverage policy differs from the validator-owned frozen policy")
    allowed_statuses = set(EXPECTED_COVERAGE_POLICY["allowed_statuses"])

    if manifest.get("groups") != EXPECTED_GROUPS:
        errors.append("group ids, titles, order, or expected counts differ from the frozen inventory")

    source_contract = manifest.get("source", {})
    if source_contract.get("path") != "include/cutlass/gemm/dispatch_policy.hpp":
        errors.append("source.path must identify the pinned dispatch_policy.hpp")
    if source_contract.get("explicit_tag_count") != EXPECTED_TAG_COUNT:
        errors.append("source.explicit_tag_count must be 59")
    if source_contract.get("kernel_schedule_auto_in_denominator") is not False:
        errors.append("KernelScheduleAuto must remain outside the explicit 59-tag denominator")

    reference_pointer = manifest.get("reference_inventory", {})
    expected_reference_pointer = {
        "path": "tests/codegen/sm110a_schedule_reference_inventory.json",
        "sha256": EXPECTED_REFERENCE_INVENTORY_SHA256,
        "class_counts": EXPECTED_REFERENCE_CLASS_COUNTS,
    }
    if reference_pointer != expected_reference_pointer:
        errors.append("manifest reference inventory pointer differs from the frozen inventory")

    auto_pointer = manifest.get("auto_control_inventory", {})
    expected_auto_pointer = {
        "path": "tests/codegen/sm110a_auto_control_inventory.json",
        "sha256": EXPECTED_AUTO_INVENTORY_SHA256,
        "expected_count": len(EXPECTED_GROUPS),
        "one_per_group": True,
    }
    if auto_pointer != expected_auto_pointer:
        errors.append("manifest Auto control pointer differs from the frozen 11-group inventory")

    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        errors.append("entries must be a list")
        entries = []
    tags = [entry.get("tag") for entry in entries if isinstance(entry, dict)]
    if len(entries) != EXPECTED_TAG_COUNT:
        errors.append(f"expected {EXPECTED_TAG_COUNT} entries, found {len(entries)}")
    if len(tags) != len(set(tags)):
        duplicates = sorted(tag for tag, count in Counter(tags).items() if count > 1)
        errors.append(f"duplicate tags: {duplicates}")
    invalid_names = sorted(
        tag for tag in tags if not isinstance(tag, str) or not tag.startswith("Kernel") or not tag.endswith("Sm100")
    )
    if invalid_names:
        errors.append(f"invalid tag names: {invalid_names}")

    if all(isinstance(tag, str) for tag in tags):
        tag_set_hash = sha256_text(sorted(tags))
        if tag_set_hash != EXPECTED_TAG_SET_SHA256:
            errors.append("explicit public Tag set differs from the frozen 59-tag inventory")
        tag_group_hash = sha256_text(
            sorted(f"{entry.get('tag')}\t{entry.get('group')}" for entry in entries if isinstance(entry, dict))
        )
        if tag_group_hash != EXPECTED_TAG_GROUP_SHA256:
            errors.append("Tag-to-group mapping differs from the frozen semantic inventory")

    group_counts = Counter(entry.get("group") for entry in entries if isinstance(entry, dict))
    if dict(group_counts) != EXPECTED_GROUP_COUNTS:
        errors.append(f"group counts {dict(group_counts)} != {EXPECTED_GROUP_COUNTS}")

    case_paths = {path.parent.name: path for path in (root / "cases").glob("*/case.json")}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tag = entry.get("tag", "<missing>")
        status = entry.get("status")
        if status not in allowed_statuses:
            errors.append(f"{tag}: invalid status {status!r}")
        result_id = entry.get("result_id")
        if status in RESULT_BACKED_STATUSES:
            if not harness_complete or EXPECTED_RESULT_CONTRACT_SHA256 is None:
                errors.append(f"{tag}: result-backed status is forbidden before Phase 1 harness closure")
            result = load_result_record(root, result_id, str(tag), errors)
            if result is not None:
                if result.get("schema_version") != 1:
                    errors.append(f"{tag}: result record schema_version must be 1")
                if result.get("objective_sha256") != EXPECTED_OBJECTIVE_SHA256:
                    errors.append(f"{tag}: result record objective mismatch")
                result_subject = result.get("subject", {})
                if result_subject.get("kind") != "explicit_schedule_tag":
                    errors.append(f"{tag}: result record subject_kind mismatch")
                if result_subject.get("id") != tag or result.get("status") != status:
                    errors.append(f"{tag}: result record subject/status mismatch")
        elif result_id is not None:
            errors.append(f"{tag}: non-result status {status!r} must not carry result_id")
        if campaign_complete and status not in EXPECTED_FINAL_STATUSES:
            errors.append(f"{tag}: completed campaign contains non-final status {status!r}")
        if entry.get("fresh_replay_required") is not True:
            errors.append(f"{tag}: fresh_replay_required must remain true until fresh closure")
        mapped_cases = entry.get("current_case_ids")
        if not isinstance(mapped_cases, list) or not all(isinstance(value, str) for value in mapped_cases):
            errors.append(f"{tag}: current_case_ids must be a string list")
            continue
        if len(mapped_cases) != len(set(mapped_cases)):
            errors.append(f"{tag}: current_case_ids contains duplicates")
        invalid_case_ids = sorted(value for value in mapped_cases if not re.fullmatch(r"[a-z0-9_]+", value))
        if invalid_case_ids:
            errors.append(f"{tag}: invalid current case ids {invalid_case_ids}")
        missing_cases = sorted(set(mapped_cases) - set(case_paths))
        if missing_cases:
            errors.append(f"{tag}: unknown current case ids {missing_cases}")
        for case_id in mapped_cases:
            case_path = case_paths.get(case_id)
            if case_path is None:
                continue
            try:
                case = load_json(case_path)
            except ValueError as error:
                errors.append(str(error))
                continue
            kernel = case.get("kernel")
            schedule = kernel.get("mainloop_schedule") if isinstance(kernel, dict) else None
            if normalized_schedule(schedule) != tag:
                errors.append(
                    f"{tag}: mapped case {case_id} declares mainloop_schedule {schedule!r}"
                )
            if case.get("case_id") != case_id:
                errors.append(f"{tag}: mapped case directory/id mismatch for {case_id}")
            case_target = case.get("target")
            if not isinstance(case_target, dict) or (
                case_target.get("virtual_arch") != "compute_110a"
                or case_target.get("arch") != "sm_110a"
                or case_target.get("cutlass_arch_tag") != "cutlass::arch::Sm100"
            ):
                errors.append(f"{tag}: mapped case {case_id} target contract mismatch")
            if status == "HISTORICAL_STATIC_PASS":
                case_evidence = case.get("evidence")
                required_historical_flags = (
                    "source_present",
                    "compile_passed",
                    "ptx_verified",
                    "sass_verified",
                )
                if not isinstance(case_evidence, dict) or any(
                    case_evidence.get(flag) is not True for flag in required_historical_flags
                ):
                    errors.append(
                        f"{tag}: historical case {case_id} lacks its archived static evidence flags"
                    )
        if status == "HISTORICAL_STATIC_PASS" and not mapped_cases:
            errors.append(f"{tag}: historical PASS requires at least one mapped case")

    table_tags = re.findall(
        r"^\|\s*`(Kernel[A-Za-z0-9_]*Sm100)`\s*\|",
        document,
        flags=re.MULTILINE,
    )
    table_counts = Counter(table_tags)
    manifest_tag_set = {tag for tag in tags if isinstance(tag, str)}
    missing_table_rows = sorted(manifest_tag_set - set(table_tags))
    extra_table_rows = sorted(set(table_tags) - manifest_tag_set)
    duplicate_table_rows = sorted(tag for tag, count in table_counts.items() if count != 1)
    if missing_table_rows or extra_table_rows or duplicate_table_rows:
        errors.append(
            "structured document coverage mismatch: "
            f"missing={missing_table_rows}, extra={extra_table_rows}, duplicates={duplicate_table_rows}"
        )

    manifest_by_tag = {
        entry.get("tag"): entry for entry in entries if isinstance(entry, dict)
    }
    tag_row_pattern = re.compile(
        r"^\|\s*`(Kernel[A-Za-z0-9_]*Sm100)`\s*\|\s*`([^`]+)`\s*\|\s*(.*?)\s*\|$",
        flags=re.MULTILINE,
    )
    for group_index, group in enumerate(EXPECTED_GROUPS, start=1):
        heading = f"### A.{group_index} {group['title']}（{group['expected_count']}）"
        heading_match = re.search(rf"^{re.escape(heading)}$", document, flags=re.MULTILINE)
        if heading_match is None:
            errors.append(f"document group heading missing or changed: {heading}")
            continue
        next_heading = re.search(
            r"^(?:### A\.[0-9]+ |## 附录 B)",
            document[heading_match.end() :],
            flags=re.MULTILINE,
        )
        section_end = (
            heading_match.end() + next_heading.start()
            if next_heading is not None
            else len(document)
        )
        section = document[heading_match.end() : section_end]
        rows = tag_row_pattern.findall(section)
        expected_entries = [entry for entry in entries if entry.get("group") == group["id"]]
        expected_tags = [entry.get("tag") for entry in expected_entries]
        actual_tags = [row[0] for row in rows]
        if actual_tags != expected_tags:
            errors.append(
                f"document group {group['id']} Tag rows {actual_tags} != manifest order {expected_tags}"
            )
        for tag, documented_status, documented_cases in rows:
            entry = manifest_by_tag.get(tag)
            if entry is None:
                continue
            if documented_status != entry.get("status"):
                errors.append(
                    f"{tag}: document status {documented_status!r} != manifest {entry.get('status')!r}"
                )
            expected_case_cell = (
                "、".join(f"`{case_id}`" for case_id in entry.get("current_case_ids", []))
                or "—"
            )
            if documented_cases != expected_case_cell:
                errors.append(
                    f"{tag}: document case cell {documented_cases!r} != manifest {expected_case_cell!r}"
                )

    documented_status_counts = {
        status: int(count)
        for status, count in re.findall(
            r"^\|\s*`(NOT_CHECKED|HISTORICAL_STATIC_PASS|STATIC_PASS|EXPECTED_STATIC_REJECT|UNSUPPORTED_SM110A|UNEXPECTED_COMPILE_FAIL|ATTRIBUTION_FAIL)`\s*\|\s*([0-9]+)\s*\|$",
            document,
            flags=re.MULTILINE,
        )
    }
    expected_status_counts = {
        status: sum(entry.get("status") == status for entry in entries)
        for status in EXPECTED_COVERAGE_POLICY["allowed_statuses"]
    }
    if documented_status_counts != expected_status_counts:
        errors.append(
            f"document status summary {documented_status_counts} != manifest {expected_status_counts}"
        )

    source_summary_labels = {
        "官方 C++ 显式或条件式引用": "official_cpp_explicit_or_conditional",
        "官方注释中的 Auto 候选映射": "official_cpp_auto_derived",
        "`generator.py` 可还原配置": "generator_config",
        "沿单一变化轴的源码派生": "source_derived",
    }
    documented_source_counts: dict[str, int] = {}
    for label, class_name in source_summary_labels.items():
        match = re.search(
            rf"^\|\s*{re.escape(label)}\s*\|\s*([0-9]+)\s*\|",
            document,
            flags=re.MULTILINE,
        )
        if match is not None:
            documented_source_counts[class_name] = int(match.group(1))
    if documented_source_counts != EXPECTED_REFERENCE_CLASS_COUNTS:
        errors.append(
            f"document source summary {documented_source_counts} != inventory "
            f"{EXPECTED_REFERENCE_CLASS_COUNTS}"
        )

    inventory_path = root / "tests/codegen/sm110a_schedule_reference_inventory.json"
    try:
        inventory_bytes = inventory_path.read_bytes()
        inventory = load_json(inventory_path)
    except (OSError, ValueError) as error:
        errors.append(f"reference inventory unavailable: {error}")
        inventory = {}
        inventory_bytes = b""
    if hashlib.sha256(inventory_bytes).hexdigest() != EXPECTED_REFERENCE_INVENTORY_SHA256:
        errors.append("reference inventory SHA-256 differs from the frozen inventory")
    if inventory.get("schema_version") != 1:
        errors.append("reference inventory schema_version must be 1")
    if inventory.get("cutlass_git_sha") != EXPECTED_CUTLASS_SHA:
        errors.append("reference inventory CUTLASS SHA differs from the pinned source")
    if inventory.get("coverage_unit") != EXPECTED_CLAIM_BOUNDARY["coverage_unit"]:
        errors.append("reference inventory coverage unit must be one canonical instance per Tag")
    if inventory.get("exhaustive_template_domain") is not False:
        errors.append("reference inventory must not claim exhaustive template-domain coverage")
    if inventory.get("selection_policy") != EXPECTED_REFERENCE_SELECTION_POLICY:
        errors.append("reference inventory selection policy differs from the deterministic policy")
    if inventory.get("class_counts") != EXPECTED_REFERENCE_CLASS_COUNTS:
        errors.append("reference inventory class counts must be 39+1+5+14")

    inventory_entries = inventory.get("entries", [])
    if not isinstance(inventory_entries, list):
        errors.append("reference inventory entries must be a list")
        inventory_entries = []
    inventory_tags = [
        entry.get("tag") for entry in inventory_entries if isinstance(entry, dict)
    ]
    if len(inventory_entries) != EXPECTED_TAG_COUNT or len(set(inventory_tags)) != EXPECTED_TAG_COUNT:
        errors.append("reference inventory must contain 59 unique entries")
    if set(inventory_tags) != manifest_tag_set:
        errors.append("reference inventory Tag set differs from the manifest")
    class_counts = Counter(
        entry.get("reference_class") for entry in inventory_entries if isinstance(entry, dict)
    )
    if dict(class_counts) != EXPECTED_REFERENCE_CLASS_COUNTS:
        errors.append(f"reference entry class counts {dict(class_counts)} != {EXPECTED_REFERENCE_CLASS_COUNTS}")

    inventory_by_tag = {
        entry.get("tag"): entry for entry in inventory_entries if isinstance(entry, dict)
    }
    cutlass_root = root / "third_party/cutlass"
    for tag, reference in inventory_by_tag.items():
        reference_class = reference.get("reference_class")
        path_value = reference.get("path")
        line_number = reference.get("line")
        anchor_hash = reference.get("anchor_line_sha256")
        parent_tag = reference.get("parent_tag")
        if not isinstance(path_value, str) or Path(path_value).is_absolute() or ".." in Path(path_value).parts:
            errors.append(f"{tag}: reference path must be a safe CUTLASS-relative path")
            continue
        reference_path = cutlass_root / path_value
        if not reference_path.is_file():
            errors.append(f"{tag}: reference path is missing: {path_value}")
            continue
        try:
            source_lines = reference_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            errors.append(f"{tag}: cannot read reference path {path_value}: {error}")
            continue
        if not isinstance(line_number, int) or line_number < 1 or line_number > len(source_lines):
            errors.append(f"{tag}: reference line {line_number!r} is out of range")
            continue
        anchor_line = source_lines[line_number - 1]
        if hashlib.sha256(anchor_line.encode("utf-8")).hexdigest() != anchor_hash:
            errors.append(f"{tag}: reference anchor line hash mismatch")
        if reference_class in {
            "official_cpp_explicit_or_conditional",
            "official_cpp_auto_derived",
        } and tag not in anchor_line:
            errors.append(f"{tag}: official reference line does not contain the Tag")
        if reference_class == "official_cpp_auto_derived" and "KernelScheduleAuto" not in anchor_line:
            errors.append(f"{tag}: Auto-derived official reference must show KernelScheduleAuto")
        if reference_class == "source_derived":
            if parent_tag not in manifest_tag_set or parent_tag == tag:
                errors.append(f"{tag}: source-derived entry requires a distinct manifest parent_tag")
        elif parent_tag is not None:
            errors.append(f"{tag}: only source-derived entries may declare parent_tag")
        factory_id = reference.get("factory_id")
        if reference_class in {"generator_config", "source_derived"}:
            if not isinstance(factory_id, str) or not factory_id:
                errors.append(f"{tag}: derived/generator reference requires factory_id")
        elif factory_id is not None:
            errors.append(f"{tag}: official reference must not declare factory_id")
        mapping_fields = (
            reference.get("mapping_path"),
            reference.get("mapping_line"),
            reference.get("mapping_line_sha256"),
        )
        if reference_class == "generator_config":
            mapping_path_value, mapping_line_number, mapping_hash = mapping_fields
            if (
                not isinstance(mapping_path_value, str)
                or Path(mapping_path_value).is_absolute()
                or ".." in Path(mapping_path_value).parts
            ):
                errors.append(f"{tag}: generator mapping path must be CUTLASS-relative")
            else:
                mapping_path = cutlass_root / mapping_path_value
                if not mapping_path.is_file():
                    errors.append(f"{tag}: generator enum-to-C++ mapping path is missing")
                else:
                    mapping_lines = mapping_path.read_text(encoding="utf-8").splitlines()
                    if (
                        not isinstance(mapping_line_number, int)
                        or mapping_line_number < 1
                        or mapping_line_number > len(mapping_lines)
                    ):
                        errors.append(f"{tag}: generator mapping line is out of range")
                    else:
                        mapping_line = mapping_lines[mapping_line_number - 1]
                        if hashlib.sha256(mapping_line.encode("utf-8")).hexdigest() != mapping_hash:
                            errors.append(f"{tag}: generator mapping line hash mismatch")
                        if tag not in mapping_line:
                            errors.append(f"{tag}: generator mapping line does not name the C++ Tag")
                        generator_enum = re.search(
                            r"KernelScheduleType\.([A-Za-z0-9_]+)", anchor_line
                        )
                        mapping_enum = re.search(
                            r"KernelScheduleType\.([A-Za-z0-9_]+)", mapping_line
                        )
                        if (
                            generator_enum is None
                            or mapping_enum is None
                            or generator_enum.group(1) != mapping_enum.group(1)
                        ):
                            errors.append(
                                f"{tag}: generator config and enum-to-C++ mapping use different enum tokens"
                            )
        elif any(value is not None for value in mapping_fields):
            errors.append(f"{tag}: only generator_config entries may declare mapping fields")
        if not isinstance(reference.get("derivation_axis"), str) or not reference.get("derivation_axis"):
            errors.append(f"{tag}: reference requires a nonempty derivation_axis")
        if not isinstance(reference.get("risk"), str) or not reference.get("risk"):
            errors.append(f"{tag}: reference requires a nonempty risk classification")
        if not isinstance(reference.get("note"), str) or not reference.get("note"):
            errors.append(f"{tag}: reference requires a nonempty evidence-boundary note")

    for start_tag in inventory_by_tag:
        seen: set[str] = set()
        current = start_tag
        while current in inventory_by_tag:
            if current in seen:
                errors.append(f"reference derivation graph contains a cycle at {current}")
                break
            seen.add(current)
            parent = inventory_by_tag[current].get("parent_tag")
            if parent is None:
                break
            current = parent

    for tag, reference in inventory_by_tag.items():
        if reference.get("reference_class") != "source_derived":
            continue
        parent = inventory_by_tag.get(reference.get("parent_tag"))
        if parent is None:
            continue
        child_anchor = (
            reference.get("path"),
            reference.get("line"),
            reference.get("anchor_line_sha256"),
        )
        parent_anchor = (
            parent.get("path"),
            parent.get("line"),
            parent.get("anchor_line_sha256"),
        )
        if child_anchor != parent_anchor:
            errors.append(f"{tag}: source-derived seed anchor differs from its immediate parent")

    if (cutlass_root / "test/unit/gemm/device").is_dir() or (cutlass_root / "examples").is_dir():
        try:
            discovered_cpp = discover_official_cpp_references(cutlass_root, manifest_tag_set)
        except ValueError as error:
            errors.append(str(error))
        else:
            inventoried_cpp = {
                tag: reference
                for tag, reference in inventory_by_tag.items()
                if reference.get("reference_class")
                in {"official_cpp_explicit_or_conditional", "official_cpp_auto_derived"}
            }
            if set(discovered_cpp) != set(inventoried_cpp):
                errors.append(
                    "deterministic official C++ lexical reference set differs from the frozen 40 entries"
                )
            for tag, (path, line_number, line) in discovered_cpp.items():
                reference = inventoried_cpp.get(tag)
                if reference is None:
                    continue
                if reference.get("path") != path or reference.get("line") != line_number:
                    errors.append(
                        f"{tag}: official C++ reference is not the deterministic selected occurrence"
                    )
                code, separator, _ = line.partition("//")
                discovered_class = (
                    "official_cpp_auto_derived"
                    if separator and tag not in code and "KernelScheduleAuto" in code
                    else "official_cpp_explicit_or_conditional"
                )
                if reference.get("reference_class") != discovered_class:
                    errors.append(
                        f"{tag}: official reference class {reference.get('reference_class')!r} "
                        f"!= recomputed {discovered_class!r}"
                    )

    auto_inventory_path = root / "tests/codegen/sm110a_auto_control_inventory.json"
    try:
        auto_inventory_bytes = auto_inventory_path.read_bytes()
        auto_inventory = load_json(auto_inventory_path)
    except (OSError, ValueError) as error:
        errors.append(f"Auto control inventory unavailable: {error}")
        auto_inventory = {}
        auto_inventory_bytes = b""
    if hashlib.sha256(auto_inventory_bytes).hexdigest() != EXPECTED_AUTO_INVENTORY_SHA256:
        errors.append("Auto control inventory SHA-256 differs from the frozen inventory")
    if auto_inventory.get("schema_version") != 1:
        errors.append("Auto control inventory schema_version must be 1")
    if auto_inventory.get("objective_sha256") != EXPECTED_OBJECTIVE_SHA256:
        errors.append("Auto control inventory objective mismatch")
    if auto_inventory.get("cutlass_git_sha") != EXPECTED_CUTLASS_SHA:
        errors.append("Auto control inventory CUTLASS SHA mismatch")
    if auto_inventory.get("target") != {
        "virtual_arch": "compute_110a",
        "binary_arch": "sm_110a",
    }:
        errors.append("Auto control inventory target must be compute_110a -> sm_110a")
    if auto_inventory.get("schedule_input") != "cutlass::gemm::collective::KernelScheduleAuto":
        errors.append("Auto controls must use collective::KernelScheduleAuto")
    if auto_inventory.get("coverage_policy") != "one_canonical_auto_control_per_explicit_tag_group":
        errors.append("Auto control coverage policy must remain one canonical control per group")
    if auto_inventory.get("group_count") != len(EXPECTED_GROUPS):
        errors.append("Auto control group_count must be 11")

    auto_entries = auto_inventory.get("entries", [])
    if not isinstance(auto_entries, list):
        errors.append("Auto control entries must be a list")
        auto_entries = []
    auto_ids = [entry.get("control_id") for entry in auto_entries if isinstance(entry, dict)]
    auto_groups = [entry.get("group") for entry in auto_entries if isinstance(entry, dict)]
    expected_group_ids = [group["id"] for group in EXPECTED_GROUPS]
    if len(auto_entries) != len(EXPECTED_GROUPS) or len(set(auto_ids)) != len(EXPECTED_GROUPS):
        errors.append("Auto control inventory must contain 11 unique control_id values")
    if auto_groups != expected_group_ids:
        errors.append(f"Auto controls must map one-to-one to groups in order: {expected_group_ids}")
    for auto_entry in auto_entries:
        if not isinstance(auto_entry, dict):
            continue
        control_id = auto_entry.get("control_id", "<missing>")
        group = auto_entry.get("group")
        seed_tag = auto_entry.get("seed_tag")
        seed_entry = manifest_by_tag.get(seed_tag)
        if seed_entry is None or seed_entry.get("group") != group:
            errors.append(f"{control_id}: seed_tag must belong to the same explicit Tag group")
        auto_status = auto_entry.get("status")
        if auto_status not in allowed_statuses:
            errors.append(f"{control_id}: invalid status {auto_status!r}")
        expected_outcome = auto_entry.get("expected_outcome_hypothesis")
        failure_layer = auto_entry.get("hypothesis_failure_layer")
        if expected_outcome not in {"STATIC_PASS", "EXPECTED_STATIC_REJECT"}:
            errors.append(f"{control_id}: invalid expected_outcome_hypothesis")
        if expected_outcome == "STATIC_PASS" and failure_layer is not None:
            errors.append(f"{control_id}: PASS hypothesis must not declare a failure layer")
        if expected_outcome == "EXPECTED_STATIC_REJECT" and not isinstance(failure_layer, str):
            errors.append(f"{control_id}: reject hypothesis requires a failure layer")
        hypothesis_evidence = auto_entry.get("hypothesis_evidence")
        if not isinstance(hypothesis_evidence, list) or not hypothesis_evidence:
            errors.append(f"{control_id}: hypothesis_evidence must be a nonempty list")
            hypothesis_evidence = []
        for evidence_anchor in hypothesis_evidence:
            if not isinstance(evidence_anchor, dict):
                errors.append(f"{control_id}: hypothesis evidence anchor must be an object")
                continue
            evidence_path_value = evidence_anchor.get("path")
            line_start = evidence_anchor.get("line_start")
            line_end = evidence_anchor.get("line_end")
            if (
                not isinstance(evidence_path_value, str)
                or Path(evidence_path_value).is_absolute()
                or ".." in Path(evidence_path_value).parts
            ):
                errors.append(f"{control_id}: hypothesis evidence path must be CUTLASS-relative")
                continue
            evidence_path = cutlass_root / evidence_path_value
            if not evidence_path.is_file():
                errors.append(f"{control_id}: hypothesis evidence path is missing")
                continue
            evidence_lines = evidence_path.read_text(encoding="utf-8").splitlines()
            if (
                not isinstance(line_start, int)
                or not isinstance(line_end, int)
                or line_start < 1
                or line_end < line_start
                or line_end > len(evidence_lines)
            ):
                errors.append(f"{control_id}: hypothesis evidence line range is invalid")
                continue
            evidence_text = "\n".join(evidence_lines[line_start - 1 : line_end]) + "\n"
            if hashlib.sha256(evidence_text.encode("utf-8")).hexdigest() != evidence_anchor.get(
                "sha256"
            ):
                errors.append(f"{control_id}: hypothesis evidence range hash mismatch")
        if auto_entry.get("fresh_replay_required") is not True:
            errors.append(f"{control_id}: fresh_replay_required must remain true")
        result_id = auto_entry.get("result_id")
        if auto_status in RESULT_BACKED_STATUSES:
            if not harness_complete or EXPECTED_RESULT_CONTRACT_SHA256 is None:
                errors.append(
                    f"{control_id}: result-backed status is forbidden before Phase 1 harness closure"
                )
            result = load_result_record(root, result_id, str(control_id), errors)
            if result is not None:
                if result.get("schema_version") != 1:
                    errors.append(f"{control_id}: result record schema_version must be 1")
                if result.get("objective_sha256") != EXPECTED_OBJECTIVE_SHA256:
                    errors.append(f"{control_id}: result record objective mismatch")
                result_subject = result.get("subject", {})
                if result_subject.get("kind") != "kernel_schedule_auto_control":
                    errors.append(f"{control_id}: result record subject_kind mismatch")
                if result_subject.get("id") != control_id or result.get("status") != auto_status:
                    errors.append(f"{control_id}: result record subject/status mismatch")
        elif result_id is not None:
            errors.append(f"{control_id}: non-result status {auto_status!r} must not carry result_id")
        if auto_status == "STATIC_PASS":
            if not isinstance(auto_entry.get("resolved_builder_specialization"), str) or not isinstance(
                auto_entry.get("resolved_dispatch_policy"), str
            ):
                errors.append(
                    f"{control_id}: STATIC_PASS requires resolved Builder specialization and DispatchPolicy"
                )
        elif auto_status == "NOT_CHECKED" and (
            auto_entry.get("resolved_builder_specialization") is not None
            or auto_entry.get("resolved_dispatch_policy") is not None
        ):
            errors.append(f"{control_id}: NOT_CHECKED control must not carry resolved types")
        if campaign_complete and auto_status not in EXPECTED_FINAL_STATUSES:
            errors.append(
                f"{control_id}: completed campaign contains non-final status {auto_status!r}"
            )

    auto_row_pattern = re.compile(
        r"^\|\s*`(auto_[a-z0-9_]+)`\s*\|\s*`([^`]+)`\s*\|\s*`(Kernel[A-Za-z0-9_]*Sm100)`\s*\|\s*(预计可构造|预计静态拒绝)\s*\|\s*`([^`]+)`\s*\|$",
        flags=re.MULTILINE,
    )
    documented_auto_rows = auto_row_pattern.findall(document)
    expected_auto_rows = [
        (
            str(entry.get("control_id")),
            str(entry.get("group")),
            str(entry.get("seed_tag")),
            (
                "预计可构造"
                if entry.get("expected_outcome_hypothesis") == "STATIC_PASS"
                else "预计静态拒绝"
            ),
            str(entry.get("status")),
        )
        for entry in auto_entries
        if isinstance(entry, dict)
    ]
    if documented_auto_rows != expected_auto_rows:
        errors.append(
            "document Auto control table differs from the 11-group machine inventory"
        )

    source_status = "NOT_AVAILABLE"
    dispatch_policy = root / "third_party/cutlass/include/cutlass/gemm/dispatch_policy.hpp"
    if not dispatch_policy.is_file() and args.require_source:
        errors.append("pinned CUTLASS source is required but dispatch_policy.hpp is unavailable")
    elif dispatch_policy.is_file():
        source_status = "CHECKED"
        cutlass_root = root / "third_party/cutlass"
        try:
            actual_head = run_git(cutlass_root, "rev-parse", "HEAD")
            dirty = run_git(cutlass_root, "status", "--porcelain", "--untracked-files=all")
        except ValueError as error:
            errors.append(str(error))
        else:
            if actual_head != EXPECTED_CUTLASS_SHA:
                errors.append(f"CUTLASS source HEAD {actual_head} != {EXPECTED_CUTLASS_SHA}")
            if dirty:
                errors.append("CUTLASS source checkout is dirty")
        dispatch_hash = hashlib.sha256(dispatch_policy.read_bytes()).hexdigest()
        if dispatch_hash != EXPECTED_DISPATCH_POLICY_SHA256:
            errors.append("dispatch_policy.hpp hash differs from the pinned source artifact")
        try:
            source_tags = discover_source_tags(dispatch_policy)
        except ValueError as error:
            errors.append(str(error))
        else:
            manifest_tags = set(tags)
            missing = sorted(source_tags - manifest_tags)
            extra = sorted(manifest_tags - source_tags)
            if missing or extra:
                errors.append(f"fixed-source mismatch: missing={missing}, extra={extra}")
        auto_declaration = (
            cutlass_root / "include/cutlass/gemm/collective/collective_builder_decl.hpp"
        )
        if not auto_declaration.is_file() or "struct KernelScheduleAuto final {};" not in auto_declaration.read_text(
            encoding="utf-8"
        ):
            errors.append("fixed CUTLASS source does not contain the KernelScheduleAuto declaration")

    versions_lock_path = root / "versions.lock.json"
    try:
        versions_lock_bytes = versions_lock_path.read_bytes()
        versions_lock = load_json(versions_lock_path)
    except (OSError, ValueError) as error:
        errors.append(str(error))
    else:
        if hashlib.sha256(versions_lock_bytes).hexdigest() != EXPECTED_VERSIONS_LOCK_FILE_SHA256:
            errors.append("versions.lock.json file SHA-256 differs from the frozen toolchain lock")
        canonical_lock = dict(versions_lock)
        declared_lock_hash = canonical_lock.pop("lock_sha256", None)
        canonical_lock_bytes = (
            json.dumps(
                canonical_lock,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        if hashlib.sha256(canonical_lock_bytes).hexdigest() != declared_lock_hash:
            errors.append("versions.lock.json canonical self-hash mismatch")
        lock_cutlass = versions_lock.get("cutlass")
        lock_cuda = versions_lock.get("cuda")
        if not isinstance(lock_cutlass, dict) or lock_cutlass.get("git_sha") != EXPECTED_CUTLASS_SHA:
            errors.append("versions.lock.json CUTLASS SHA differs from the pinned source")
        if not isinstance(lock_cuda, dict) or (
            lock_cuda.get("target_virtual") != "compute_110a"
            or lock_cuda.get("target_real") != "sm_110a"
        ):
            errors.append("versions.lock.json target must be compute_110a -> sm_110a")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    status_counts = Counter(entry["status"] for entry in entries)
    print(
        "SCHEDULE_TAG_COVERAGE_PASS "
        f"tags={len(entries)} unique={len(set(tags))} "
        f"historical={status_counts.get('HISTORICAL_STATIC_PASS', 0)} "
        f"not_checked={status_counts.get('NOT_CHECKED', 0)} "
        f"auto_controls={len(auto_entries)} source={source_status} objective={objective_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
