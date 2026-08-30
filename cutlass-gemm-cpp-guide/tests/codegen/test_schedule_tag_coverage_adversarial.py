#!/usr/bin/env python3
"""Prove the 59-tag coverage gate rejects deliberate contract corruption."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "tests/codegen/validate_schedule_tag_coverage.py"
MANIFEST = ROOT / "tests/codegen/sm110a_tensor_schedule_tags.json"
CONTRACT = ROOT / "tests/codegen/campaign_contract.json"
DOCUMENT = ROOT / "docs/02-tensor-core-tcgen05-gemm-codegen-formal-validation.md"
VERSIONS_LOCK = ROOT / "versions.lock.json"
REFERENCE_INVENTORY = ROOT / "tests/codegen/sm110a_schedule_reference_inventory.json"
AUTO_INVENTORY = ROOT / "tests/codegen/sm110a_auto_control_inventory.json"
PHASE0_REPORT_RELATIVE = Path(
    "evidence/codegen-sm110a-v2/phase-reports/phase-00-workspace-source-inventory.md"
)
PHASE0_REPORT = ROOT / PHASE0_REPORT_RELATIVE
PHASE1_REPORT_RELATIVE = Path(
    "evidence/codegen-sm110a-v2/phase-reports/phase-01-cross-layer-harness.md"
)
PHASE1_REPORT = ROOT / PHASE1_REPORT_RELATIVE

VALIDATOR_SPEC = importlib.util.spec_from_file_location("schedule_tag_validator", VALIDATOR)
if VALIDATOR_SPEC is None or VALIDATOR_SPEC.loader is None:
    raise RuntimeError(f"cannot import validator from {VALIDATOR}")
VALIDATOR_MODULE = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATOR_MODULE)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def make_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="schedule-tag-adversarial-"))

    def copy_relative(relative: str | Path) -> None:
        relative_path = Path(relative)
        source = ROOT / relative_path
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    write(root / "tests/codegen/sm110a_tensor_schedule_tags.json", load(MANIFEST))
    write(root / "tests/codegen/campaign_contract.json", load(CONTRACT))
    shutil.copy2(
        REFERENCE_INVENTORY,
        root / "tests/codegen/sm110a_schedule_reference_inventory.json",
    )
    shutil.copy2(
        AUTO_INVENTORY,
        root / "tests/codegen/sm110a_auto_control_inventory.json",
    )
    target_doc = root / "docs/02-tensor-core-tcgen05-gemm-codegen-formal-validation.md"
    target_doc.parent.mkdir(parents=True, exist_ok=True)
    target_doc.write_text(DOCUMENT.read_text(encoding="utf-8"), encoding="utf-8")
    shutil.copy2(VERSIONS_LOCK, root / "versions.lock.json")
    for phase in load(CONTRACT)["phases"]:
        completion_report = phase.get("completion_report")
        if completion_report is not None:
            copy_relative(completion_report["path"])
    static_contract = load(ROOT / "tests/codegen/static_codegen_contract.json")
    for source_range in static_contract["arch_guard_fallback_contract"]["source_constraints"]:
        source = ROOT / "third_party/cutlass" / source_range["path"]
        target = root / "third_party/cutlass" / source_range["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for reference in load(REFERENCE_INVENTORY)["entries"]:
        source = ROOT / "third_party/cutlass" / reference["path"]
        target = root / "third_party/cutlass" / reference["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        mapping_path = reference.get("mapping_path")
        if mapping_path:
            mapping_source = ROOT / "third_party/cutlass" / mapping_path
            mapping_target = root / "third_party/cutlass" / mapping_path
            mapping_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mapping_source, mapping_target)
    for auto_entry in load(AUTO_INVENTORY)["entries"]:
        for evidence_anchor in auto_entry["hypothesis_evidence"]:
            evidence_source = ROOT / "third_party/cutlass" / evidence_anchor["path"]
            evidence_target = root / "third_party/cutlass" / evidence_anchor["path"]
            evidence_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(evidence_source, evidence_target)
    for entry in load(MANIFEST)["entries"]:
        for case_id in entry["current_case_ids"]:
            case = root / "cases" / case_id / "case.json"
            case.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "cases" / case_id / "case.json", case)
        result_id = entry.get("result_id")
        if result_id is None:
            continue
        result_relative = Path("evidence/codegen-sm110a-v2/results") / f"{result_id}.json"
        copy_relative(result_relative)
        result = load(ROOT / result_relative)
        for ref_name in ("instance_ref", "fingerprint_ref", "journal_ref", "artifact_manifest_ref"):
            copy_relative(result[ref_name]["path"])
        fingerprint = load(ROOT / result["fingerprint_ref"]["path"])
        for source_input in fingerprint["source_closure"]["inputs"]:
            copy_relative(source_input["path"])
        artifact_manifest = load(ROOT / result["artifact_manifest_ref"]["path"])
        for artifact in artifact_manifest["items"]:
            if artifact["storage"] == "git_evidence":
                copy_relative(artifact["path"])
    return root


def run(root: Path, *, require_source: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(VALIDATOR), "--root", str(root)]
    if require_source:
        command.append("--require-source")
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def set_phase0_complete(contract: dict, root: Path) -> None:
    contract["phases"][0]["status"] = "COMPLETE"
    contract["phases"][1]["status"] = "IN_PROGRESS"
    contract["phases"][1].pop("completion_report", None)
    for phase in contract["phases"][2:]:
        phase["status"] = "PENDING"
        phase.pop("completion_report", None)
    report_bytes = (root / PHASE0_REPORT_RELATIVE).read_bytes()
    contract["phases"][0]["completion_report"] = {
        "path": PHASE0_REPORT_RELATIVE.as_posix(),
        "sha256": hashlib.sha256(report_bytes).hexdigest(),
        "objective_sha256": contract["objective_sha256"],
    }


def require_rejected(name: str, expected_error: str, mutate) -> None:
    root = make_root()
    try:
        manifest_path = root / "tests/codegen/sm110a_tensor_schedule_tags.json"
        contract_path = root / "tests/codegen/campaign_contract.json"
        document_path = root / "docs/02-tensor-core-tcgen05-gemm-codegen-formal-validation.md"
        manifest = load(manifest_path)
        contract = load(contract_path)
        document = document_path.read_text(encoding="utf-8")
        manifest, contract, document = mutate(manifest, contract, document, root)
        write(manifest_path, manifest)
        write(contract_path, contract)
        document_path.write_text(document, encoding="utf-8")
        result = run(root)
        if result.returncode == 0:
            raise AssertionError(f"{name}: validator accepted deliberate corruption\n{result.stdout}")
        if expected_error not in result.stdout:
            raise AssertionError(
                f"{name}: rejected for the wrong reason; expected {expected_error!r}\n{result.stdout}"
            )
    finally:
        shutil.rmtree(root)


def main() -> int:
    result_id = "phase1-generalized-20260830.dense_f16_1sm.a001"
    helper_errors: list[str] = []
    loaded_record = VALIDATOR_MODULE.load_result_record(
        ROOT,
        result_id,
        "helper",
        helper_errors,
    )
    if loaded_record is None or loaded_record.get("result_id") != result_id or helper_errors:
        raise AssertionError(
            f"load_result_record positive helper failed: record={loaded_record}, errors={helper_errors}"
        )

    completion_helper_errors: list[str] = []
    completion_phase = {
        "id": 0,
        "completion_report": {
            "path": PHASE0_REPORT_RELATIVE.as_posix(),
            "sha256": hashlib.sha256(PHASE0_REPORT.read_bytes()).hexdigest(),
            "objective_sha256": load(CONTRACT)["objective_sha256"],
        },
    }
    VALIDATOR_MODULE.validate_completion_report(
        ROOT,
        completion_phase,
        load(CONTRACT)["first_principles_objective"],
        completion_helper_errors,
    )
    if completion_helper_errors:
        raise AssertionError(
            f"validate_completion_report positive helper failed: {completion_helper_errors}"
        )

    clean_root = make_root()
    try:
        clean = run(clean_root)
        if clean.returncode:
            raise AssertionError(f"clean fixture failed\n{clean.stdout}")
    finally:
        shutil.rmtree(clean_root)

    require_rejected(
        "missing_tag",
        "expected 59 entries",
        lambda m, c, d, r: ({**m, "entries": m["entries"][:-1]}, c, d),
    )

    def duplicate_tag(manifest, contract, document, root):
        manifest["entries"][-1]["tag"] = manifest["entries"][0]["tag"]
        return manifest, contract, document

    require_rejected("duplicate_tag", "duplicate tags", duplicate_tag)

    def swapped_groups(manifest, contract, document, root):
        left = next(entry for entry in manifest["entries"] if entry["group"] == "dense")
        right = next(entry for entry in manifest["entries"] if entry["group"] == "sparse_block_scaled")
        left["group"], right["group"] = right["group"], left["group"]
        return manifest, contract, document

    require_rejected(
        "swapped_groups_same_counts",
        "Tag-to-group mapping differs",
        swapped_groups,
    )

    def wrong_group_metadata(manifest, contract, document, root):
        manifest["groups"][0]["expected_count"] = 6
        return manifest, contract, document

    require_rejected(
        "wrong_group_metadata",
        "group ids, titles, order, or expected counts differ",
        wrong_group_metadata,
    )

    def missing_document_tag(manifest, contract, document, root):
        tag = manifest["entries"][0]["tag"]
        row = next(line for line in document.splitlines() if line.startswith(f"| `{tag}` |"))
        document = document.replace(row + "\n", "", 1)
        document += f"\n正文仍然提到 `{tag}`，但这不能代替正式覆盖表格行。\n"
        return manifest, contract, document

    require_rejected(
        "missing_document_table_row",
        "structured document coverage mismatch",
        missing_document_tag,
    )

    def wrong_document_status(manifest, contract, document, root):
        entry = next(entry for entry in manifest["entries"] if entry["status"] == "NOT_CHECKED")
        row = next(line for line in document.splitlines() if line.startswith(f"| `{entry['tag']}` |"))
        document = document.replace(row, row.replace("`NOT_CHECKED`", "`STATIC_PASS`"), 1)
        return manifest, contract, document

    require_rejected(
        "wrong_document_status",
        "document status",
        wrong_document_status,
    )

    def wrong_document_case_cell(manifest, contract, document, root):
        entry = next(entry for entry in manifest["entries"] if entry["current_case_ids"])
        row = next(line for line in document.splitlines() if line.startswith(f"| `{entry['tag']}` |"))
        case_cell = "、".join(f"`{case_id}`" for case_id in entry["current_case_ids"])
        document = document.replace(row, row.replace(case_cell, "`invented_case`", 1), 1)
        return manifest, contract, document

    require_rejected(
        "wrong_document_case_cell",
        "document case cell",
        wrong_document_case_cell,
    )

    def wrong_document_status_summary(manifest, contract, document, root):
        count = sum(entry["status"] == "NOT_CHECKED" for entry in manifest["entries"])
        document = document.replace(
            f"| `NOT_CHECKED` | {count} |",
            f"| `NOT_CHECKED` | {count - 1} |",
            1,
        )
        return manifest, contract, document

    require_rejected(
        "wrong_document_status_summary",
        "document status summary",
        wrong_document_status_summary,
    )

    def wrong_document_source_summary(manifest, contract, document, root):
        document = document.replace(
            "| 官方 C++ 显式或条件式引用 | 39 |",
            "| 官方 C++ 显式或条件式引用 | 38 |",
            1,
        )
        return manifest, contract, document

    require_rejected(
        "wrong_document_source_summary",
        "document source summary",
        wrong_document_source_summary,
    )

    def unknown_case(manifest, contract, document, root):
        manifest["entries"][0]["current_case_ids"] = ["does_not_exist"]
        return manifest, contract, document

    require_rejected("unknown_case", "unknown current case ids", unknown_case)

    def wrong_case_schedule(manifest, contract, document, root):
        entry = next(entry for entry in manifest["entries"] if entry["current_case_ids"])
        case_path = root / "cases" / entry["current_case_ids"][0] / "case.json"
        case = load(case_path)
        case["kernel"]["mainloop_schedule"] = "cutlass::gemm::KernelTmaWarpSpecialized2SmSm100"
        write(case_path, case)
        return manifest, contract, document

    require_rejected(
        "wrong_case_schedule",
        "declares mainloop_schedule",
        wrong_case_schedule,
    )

    def missing_historical_evidence(manifest, contract, document, root):
        entry = next(entry for entry in manifest["entries"] if entry["status"] == "STATIC_PASS")
        entry["status"] = "HISTORICAL_STATIC_PASS"
        entry.pop("result_id", None)
        row = next(line for line in document.splitlines() if line.startswith(f"| `{entry['tag']}` |"))
        document = document.replace(row, row.replace("`STATIC_PASS`", "`HISTORICAL_STATIC_PASS`"), 1)
        case_path = root / "cases" / entry["current_case_ids"][0] / "case.json"
        case = load(case_path)
        case["evidence"]["sass_verified"] = False
        write(case_path, case)
        return manifest, contract, document

    require_rejected(
        "missing_historical_evidence",
        "lacks its archived static evidence flags",
        missing_historical_evidence,
    )

    def wrong_commit(manifest, contract, document, root):
        manifest["target_contract"]["cutlass_git_sha"] = "0" * 40
        return manifest, contract, document

    require_rejected("wrong_commit", "manifest CUTLASS SHA", wrong_commit)

    def include_auto(manifest, contract, document, root):
        manifest["entries"][0]["tag"] = "KernelScheduleAuto"
        return manifest, contract, document

    require_rejected("auto_in_denominator", "invalid tag names", include_auto)

    def corrupt_objective(manifest, contract, document, root):
        contract["first_principles_objective"] += " drift"
        return manifest, contract, document

    require_rejected("objective_drift", "campaign objective SHA-256 mismatch", corrupt_objective)

    def corrupt_manifest_objective(manifest, contract, document, root):
        manifest["objective_sha256"] = "0" * 64
        return manifest, contract, document

    require_rejected(
        "manifest_objective_drift",
        "manifest objective SHA-256 does not match",
        corrupt_manifest_objective,
    )

    def allow_omission(manifest, contract, document, root):
        manifest["coverage_policy"]["omission_allowed"] = True
        return manifest, contract, document

    require_rejected("omission_allowed", "validator-owned frozen policy", allow_omission)

    def self_authorized_status(manifest, contract, document, root):
        manifest["coverage_policy"]["allowed_statuses"].append("OUT_OF_SCOPE")
        manifest["entries"][0]["status"] = "OUT_OF_SCOPE"
        return manifest, contract, document

    require_rejected(
        "self_authorized_status",
        "validator-owned frozen policy",
        self_authorized_status,
    )

    def unbacked_static_pass(manifest, contract, document, root):
        entry = next(entry for entry in manifest["entries"] if entry["status"] == "NOT_CHECKED")
        entry["status"] = "STATIC_PASS"
        row = next(line for line in document.splitlines() if line.startswith(f"| `{entry['tag']}` |"))
        document = document.replace(row, row.replace("`NOT_CHECKED`", "`STATIC_PASS`"), 1)
        return manifest, contract, document

    require_rejected(
        "unbacked_static_pass",
        "evidence-backed status requires a safe result_id",
        unbacked_static_pass,
    )

    def fabricated_result_after_phase_status_flip(manifest, contract, document, root):
        contract["phases"][0]["status"] = "COMPLETE"
        contract["phases"][1]["status"] = "COMPLETE"
        contract["phases"][2]["status"] = "IN_PROGRESS"
        entry = next(entry for entry in manifest["entries"] if entry["status"] == "NOT_CHECKED")
        entry["status"] = "STATIC_PASS"
        entry["result_id"] = "fabricated"
        write(
            root / "evidence/codegen-sm110a-v2/results/fabricated.json",
            {
                "schema_version": 1,
                "objective_sha256": contract["objective_sha256"],
                "subject_kind": "explicit_schedule_tag",
                "subject_id": entry["tag"],
                "status": "STATIC_PASS",
            },
        )
        row = next(line for line in document.splitlines() if line.startswith(f"| `{entry['tag']}` |"))
        document = document.replace(row, row.replace("`NOT_CHECKED`", "`STATIC_PASS`"), 1)
        return manifest, contract, document

    require_rejected(
        "fabricated_result_after_phase_status_flip",
        "result record unavailable",
        fabricated_result_after_phase_status_flip,
    )

    def result_contract_bundle_drift(manifest, contract, document, root):
        model_path = root / "tools/codegen_v2/model.py"
        model_path.write_text(
            model_path.read_text(encoding="utf-8") + "\n# deliberate contract drift\n",
            encoding="utf-8",
        )
        return manifest, contract, document

    require_rejected(
        "result_contract_bundle_drift",
        "result-contract bundle differs",
        result_contract_bundle_drift,
    )

    def weakened_evidence(manifest, contract, document, root):
        manifest["coverage_policy"]["required_static_evidence"]["STATIC_PASS"].pop()
        return manifest, contract, document

    require_rejected(
        "weakened_required_evidence",
        "validator-owned frozen policy",
        weakened_evidence,
    )

    def disable_fresh_replay(manifest, contract, document, root):
        manifest["entries"][0]["fresh_replay_required"] = False
        return manifest, contract, document

    require_rejected(
        "disable_fresh_replay",
        "fresh_replay_required must remain true",
        disable_fresh_replay,
    )

    def wrong_arch_key(manifest, contract, document, root):
        manifest["target_contract"]["cutlass_arch_tag"] = "cutlass::arch::Sm110"
        return manifest, contract, document

    require_rejected("wrong_arch_key", "cutlass::arch::Sm100", wrong_arch_key)

    def wrong_source_path(manifest, contract, document, root):
        manifest["source"]["path"] = "include/cutlass/gemm/wrong.hpp"
        return manifest, contract, document

    require_rejected("wrong_source_path", "source.path", wrong_source_path)

    def wrong_source_count(manifest, contract, document, root):
        manifest["source"]["explicit_tag_count"] = 58
        return manifest, contract, document

    require_rejected("wrong_source_count", "source.explicit_tag_count", wrong_source_count)

    def auto_policy_in_denominator(manifest, contract, document, root):
        manifest["source"]["kernel_schedule_auto_in_denominator"] = True
        return manifest, contract, document

    require_rejected(
        "auto_policy_in_denominator",
        "KernelScheduleAuto must remain outside",
        auto_policy_in_denominator,
    )

    def widen_final_statuses(manifest, contract, document, root):
        contract["accepted_final_statuses"].append("HISTORICAL_STATIC_PASS")
        return manifest, contract, document

    require_rejected(
        "widen_final_statuses",
        "accepted_final_statuses differ",
        widen_final_statuses,
    )

    def weaken_phase_loop(manifest, contract, document, root):
        contract["phase_loop"].remove("deliberate_breakage")
        return manifest, contract, document

    require_rejected(
        "weaken_phase_loop",
        "phase loop differs",
        weaken_phase_loop,
    )

    def allow_evidence_promotion(manifest, contract, document, root):
        contract["phase_gate"]["allow_evidence_promotion"] = True
        return manifest, contract, document

    require_rejected(
        "allow_evidence_promotion",
        "phase gate must reject",
        allow_evidence_promotion,
    )

    def require_runtime(manifest, contract, document, root):
        contract["scope"]["requires_runtime_correctness"] = True
        return manifest, contract, document

    require_rejected(
        "runtime_scope_creep",
        "static-only 59+11 scope",
        require_runtime,
    )

    def internal_policy_as_public_tag(manifest, contract, document, root):
        manifest["entries"][0]["tag"] = "KernelTmaWarpSpecializedSm100"
        return manifest, contract, document

    require_rejected(
        "internal_dispatch_policy_in_public_inventory",
        "explicit public Tag set differs",
        internal_policy_as_public_tag,
    )

    def weaken_claim_boundary(manifest, contract, document, root):
        contract["claim_boundary"]["exhaustive_template_domain"] = True
        return manifest, contract, document

    require_rejected(
        "exhaustive_domain_overclaim",
        "claim boundary must remain",
        weaken_claim_boundary,
    )

    def completed_campaign_with_unfinished_entries(manifest, contract, document, root):
        for phase in contract["phases"]:
            phase["status"] = "COMPLETE"
        return manifest, contract, document

    require_rejected(
        "completed_campaign_with_unfinished_entries",
        "completed campaign contains non-final status",
        completed_campaign_with_unfinished_entries,
    )

    def completed_phase_without_report(manifest, contract, document, root):
        contract["phases"][0]["status"] = "COMPLETE"
        contract["phases"][0].pop("completion_report", None)
        contract["phases"][1]["status"] = "IN_PROGRESS"
        return manifest, contract, document

    require_rejected(
        "completed_phase_without_report",
        "COMPLETE requires a completion_report pointer",
        completed_phase_without_report,
    )

    def completion_report_content_hash_drift(manifest, contract, document, root):
        set_phase0_complete(contract, root)
        report_path = root / PHASE0_REPORT_RELATIVE
        report_path.write_text(
            report_path.read_text(encoding="utf-8") + "\ncontent drift\n",
            encoding="utf-8",
        )
        return manifest, contract, document

    require_rejected(
        "completion_report_content_hash_drift",
        "completion report SHA-256 mismatch",
        completion_report_content_hash_drift,
    )

    def completion_report_missing_objective(manifest, contract, document, root):
        set_phase0_complete(contract, root)
        report_path = root / PHASE0_REPORT_RELATIVE
        report = report_path.read_text(encoding="utf-8")
        report = report.replace(contract["first_principles_objective"], "OBJECTIVE_REMOVED", 1)
        report_path.write_text(report, encoding="utf-8")
        contract["phases"][0]["completion_report"]["sha256"] = hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest()
        return manifest, contract, document

    require_rejected(
        "completion_report_missing_objective",
        "must repeat the exact objective",
        completion_report_missing_objective,
    )

    def completion_report_missing_marker(manifest, contract, document, root):
        set_phase0_complete(contract, root)
        report_path = root / PHASE0_REPORT_RELATIVE
        report = report_path.read_text(encoding="utf-8").replace(
            "positive_validation: PASS",
            "positive_validation: REMOVED",
            1,
        )
        report_path.write_text(report, encoding="utf-8")
        contract["phases"][0]["completion_report"]["sha256"] = hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest()
        return manifest, contract, document

    require_rejected(
        "completion_report_missing_marker",
        "completion report missing marker",
        completion_report_missing_marker,
    )

    def completion_report_path_traversal(manifest, contract, document, root):
        set_phase0_complete(contract, root)
        contract["phases"][0]["completion_report"]["path"] = "../escaped-report.md"
        return manifest, contract, document

    require_rejected(
        "completion_report_path_traversal",
        "completion report path escapes",
        completion_report_path_traversal,
    )

    def missing_auto_control(manifest, contract, document, root):
        auto_path = root / "tests/codegen/sm110a_auto_control_inventory.json"
        auto = load(auto_path)
        auto["entries"].pop()
        write(auto_path, auto)
        return manifest, contract, document

    require_rejected(
        "missing_auto_control",
        "must contain 11 unique control_id",
        missing_auto_control,
    )

    def duplicate_auto_group(manifest, contract, document, root):
        auto_path = root / "tests/codegen/sm110a_auto_control_inventory.json"
        auto = load(auto_path)
        auto["entries"][1]["group"] = auto["entries"][0]["group"]
        write(auto_path, auto)
        return manifest, contract, document

    require_rejected(
        "duplicate_auto_group",
        "Auto controls must map one-to-one",
        duplicate_auto_group,
    )

    def auto_seed_from_wrong_group(manifest, contract, document, root):
        auto_path = root / "tests/codegen/sm110a_auto_control_inventory.json"
        auto = load(auto_path)
        auto["entries"][0]["seed_tag"] = auto["entries"][1]["seed_tag"]
        write(auto_path, auto)
        return manifest, contract, document

    require_rejected(
        "auto_seed_from_wrong_group",
        "seed_tag must belong to the same explicit Tag group",
        auto_seed_from_wrong_group,
    )

    def unbacked_auto_static_pass(manifest, contract, document, root):
        auto_path = root / "tests/codegen/sm110a_auto_control_inventory.json"
        auto = load(auto_path)
        auto["entries"][0]["status"] = "STATIC_PASS"
        write(auto_path, auto)
        row = next(
            line
            for line in document.splitlines()
            if line.startswith(f"| `{auto['entries'][0]['control_id']}` |")
        )
        document = document.replace(row, row.replace("`NOT_CHECKED`", "`STATIC_PASS`"), 1)
        return manifest, contract, document

    require_rejected(
        "unbacked_auto_static_pass",
        "STATIC_PASS requires resolved Builder specialization and DispatchPolicy",
        unbacked_auto_static_pass,
    )

    def auto_hypothesis_hash_drift(manifest, contract, document, root):
        auto_path = root / "tests/codegen/sm110a_auto_control_inventory.json"
        auto = load(auto_path)
        auto["entries"][0]["hypothesis_evidence"][0]["sha256"] = "0" * 64
        write(auto_path, auto)
        return manifest, contract, document

    require_rejected(
        "auto_hypothesis_hash_drift",
        "hypothesis evidence range hash mismatch",
        auto_hypothesis_hash_drift,
    )

    def missing_auto_document_row(manifest, contract, document, root):
        auto = load(root / "tests/codegen/sm110a_auto_control_inventory.json")
        row = next(
            line
            for line in document.splitlines()
            if line.startswith(f"| `{auto['entries'][0]['control_id']}` |")
        )
        return manifest, contract, document.replace(row + "\n", "", 1)

    require_rejected(
        "missing_auto_document_row",
        "document Auto control table differs",
        missing_auto_document_row,
    )

    reference_path_root = make_root()
    try:
        inventory_path = reference_path_root / "tests/codegen/sm110a_schedule_reference_inventory.json"
        inventory = load(inventory_path)
        inventory["entries"][0]["path"] = "test/unit/gemm/device/does_not_exist.cu"
        write(inventory_path, inventory)
        missing_reference = run(reference_path_root)
        if (
            missing_reference.returncode == 0
            or "reference path is missing" not in missing_reference.stdout
        ):
            raise AssertionError(
                "missing_reference_path: not rejected for the expected reason\n"
                + missing_reference.stdout
            )
    finally:
        shutil.rmtree(reference_path_root)

    reference_anchor_root = make_root()
    try:
        inventory_path = reference_anchor_root / "tests/codegen/sm110a_schedule_reference_inventory.json"
        inventory = load(inventory_path)
        inventory["entries"][0]["line"] += 1
        write(inventory_path, inventory)
        wrong_anchor = run(reference_anchor_root)
        if wrong_anchor.returncode == 0 or "reference anchor line hash mismatch" not in wrong_anchor.stdout:
            raise AssertionError(
                "wrong_reference_anchor: not rejected for the expected reason\n" + wrong_anchor.stdout
            )
    finally:
        shutil.rmtree(reference_anchor_root)

    nondeterministic_reference_root = make_root()
    try:
        inventory_path = (
            nondeterministic_reference_root
            / "tests/codegen/sm110a_schedule_reference_inventory.json"
        )
        inventory = load(inventory_path)
        reference = next(
            entry
            for entry in inventory["entries"]
            if entry["tag"] == "KernelTmaWarpSpecialized1SmSm100"
        )
        source_path = nondeterministic_reference_root / "third_party/cutlass" / reference["path"]
        lines = source_path.read_text(encoding="utf-8").splitlines()
        occurrences = [
            index
            for index, line in enumerate(lines, start=1)
            if reference["tag"] in line
        ]
        if len(occurrences) < 2:
            raise AssertionError("fixture requires a second official occurrence")
        reference["line"] = occurrences[1]
        reference["anchor_line_sha256"] = hashlib.sha256(
            lines[occurrences[1] - 1].encode("utf-8")
        ).hexdigest()
        write(inventory_path, inventory)
        nondeterministic = run(nondeterministic_reference_root)
        if (
            nondeterministic.returncode == 0
            or "not the deterministic selected occurrence" not in nondeterministic.stdout
        ):
            raise AssertionError(
                "nondeterministic_official_reference: not rejected for the expected reason\n"
                + nondeterministic.stdout
            )
    finally:
        shutil.rmtree(nondeterministic_reference_root)

    generator_mapping_root = make_root()
    try:
        inventory_path = generator_mapping_root / "tests/codegen/sm110a_schedule_reference_inventory.json"
        inventory = load(inventory_path)
        reference = next(
            entry for entry in inventory["entries"] if entry["reference_class"] == "generator_config"
        )
        reference["mapping_line"] += 1
        write(inventory_path, inventory)
        generator_mapping = run(generator_mapping_root)
        if (
            generator_mapping.returncode == 0
            or "generator mapping line hash mismatch" not in generator_mapping.stdout
        ):
            raise AssertionError(
                "wrong_generator_mapping: not rejected for the expected reason\n"
                + generator_mapping.stdout
            )
    finally:
        shutil.rmtree(generator_mapping_root)

    generator_enum_root = make_root()
    try:
        inventory_path = generator_enum_root / "tests/codegen/sm110a_schedule_reference_inventory.json"
        inventory = load(inventory_path)
        reference = next(
            entry
            for entry in inventory["entries"]
            if entry["tag"] == "KernelTmaWarpSpecialized1SmFastFP32Sm100"
        )
        generator_path = generator_enum_root / "third_party/cutlass" / reference["path"]
        generator_lines = generator_path.read_text(encoding="utf-8").splitlines()
        reference["line"] = 10869
        reference["anchor_line_sha256"] = hashlib.sha256(
            generator_lines[10868].encode("utf-8")
        ).hexdigest()
        write(inventory_path, inventory)
        generator_enum = run(generator_enum_root)
        if (
            generator_enum.returncode == 0
            or "generator config and enum-to-C++ mapping use different enum tokens"
            not in generator_enum.stdout
        ):
            raise AssertionError(
                "generator_enum_mismatch: not rejected for the expected reason\n"
                + generator_enum.stdout
            )
    finally:
        shutil.rmtree(generator_enum_root)

    parent_anchor_root = make_root()
    try:
        inventory_path = parent_anchor_root / "tests/codegen/sm110a_schedule_reference_inventory.json"
        inventory = load(inventory_path)
        child = next(
            entry
            for entry in inventory["entries"]
            if entry["tag"] == "KernelPtrArrayTmaWarpSpecialized2SmFastFP32SmemSm100"
        )
        child["parent_tag"] = "KernelTmaWarpSpecialized2SmFastFP32SmemSm100"
        write(inventory_path, inventory)
        parent_anchor = run(parent_anchor_root)
        if (
            parent_anchor.returncode == 0
            or "source-derived seed anchor differs from its immediate parent"
            not in parent_anchor.stdout
        ):
            raise AssertionError(
                "source_derived_parent_anchor_mismatch: not rejected for the expected reason\n"
                + parent_anchor.stdout
            )
    finally:
        shutil.rmtree(parent_anchor_root)

    wrong_lock_root = make_root()
    try:
        lock_path = wrong_lock_root / "versions.lock.json"
        lock = load(lock_path)
        lock["cutlass"]["git_sha"] = "0" * 40
        write(lock_path, lock)
        wrong_lock = run(wrong_lock_root)
        if wrong_lock.returncode == 0 or "versions.lock.json CUTLASS SHA" not in wrong_lock.stdout:
            raise AssertionError(
                "wrong_versions_lock: not rejected for the expected reason\n" + wrong_lock.stdout
            )
    finally:
        shutil.rmtree(wrong_lock_root)

    toolchain_drift_root = make_root()
    try:
        lock_path = toolchain_drift_root / "versions.lock.json"
        lock = load(lock_path)
        lock["cuda"]["nvcc_version"] = "99.9.9"
        lock["cuda"]["ptxas_version"] = "0"
        lock["container"]["digest"] = "sha256:" + "0" * 64
        write(lock_path, lock)
        toolchain_drift = run(toolchain_drift_root)
        if (
            toolchain_drift.returncode == 0
            or "file SHA-256 differs from the frozen toolchain lock" not in toolchain_drift.stdout
        ):
            raise AssertionError(
                "toolchain_lock_drift: not rejected for the expected reason\n"
                + toolchain_drift.stdout
            )
    finally:
        shutil.rmtree(toolchain_drift_root)

    def initialize_fake_cutlass_repo(root: Path) -> Path:
        cutlass_root = root / "third_party/cutlass"
        dispatch_source = ROOT / "third_party/cutlass/include/cutlass/gemm/dispatch_policy.hpp"
        dispatch_target = cutlass_root / "include/cutlass/gemm/dispatch_policy.hpp"
        dispatch_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dispatch_source, dispatch_target)
        auto_decl_source = (
            ROOT
            / "third_party/cutlass/include/cutlass/gemm/collective/collective_builder_decl.hpp"
        )
        auto_decl_target = (
            cutlass_root / "include/cutlass/gemm/collective/collective_builder_decl.hpp"
        )
        auto_decl_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(auto_decl_source, auto_decl_target)
        commands = [
            ["git", "init", "-q"],
            ["git", "config", "user.email", "adversarial@example.invalid"],
            ["git", "config", "user.name", "Adversarial Fixture"],
            ["git", "add", "."],
            ["git", "commit", "-q", "-m", "fixture"],
        ]
        for command in commands:
            subprocess.run(
                command,
                cwd=cutlass_root,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        return cutlass_root

    wrong_head_root = make_root()
    try:
        initialize_fake_cutlass_repo(wrong_head_root)
        wrong_head = run(wrong_head_root, require_source=True)
        if wrong_head.returncode == 0 or "CUTLASS source HEAD" not in wrong_head.stdout:
            raise AssertionError(
                "wrong_source_head: not rejected for the expected reason\n" + wrong_head.stdout
            )
    finally:
        shutil.rmtree(wrong_head_root)

    dirty_source_root = make_root()
    try:
        cutlass_root = initialize_fake_cutlass_repo(dirty_source_root)
        (cutlass_root / "dirty-probe.txt").write_text("dirty\n", encoding="utf-8")
        dirty_source = run(dirty_source_root, require_source=True)
        if dirty_source.returncode == 0 or "CUTLASS source checkout is dirty" not in dirty_source.stdout:
            raise AssertionError(
                "dirty_source_checkout: not rejected for the expected reason\n" + dirty_source.stdout
            )
    finally:
        shutil.rmtree(dirty_source_root)

    wrong_source_hash_root = make_root()
    try:
        cutlass_root = initialize_fake_cutlass_repo(wrong_source_hash_root)
        dispatch = cutlass_root / "include/cutlass/gemm/dispatch_policy.hpp"
        dispatch.write_text(dispatch.read_text(encoding="utf-8") + "\n// drift\n", encoding="utf-8")
        wrong_source_hash = run(wrong_source_hash_root, require_source=True)
        if (
            wrong_source_hash.returncode == 0
            or "dispatch_policy.hpp hash differs" not in wrong_source_hash.stdout
        ):
            raise AssertionError(
                "wrong_source_hash: not rejected for the expected reason\n" + wrong_source_hash.stdout
            )
    finally:
        shutil.rmtree(wrong_source_hash_root)

    missing_source_root = make_root()
    try:
        missing_source = run(missing_source_root, require_source=True)
        if missing_source.returncode == 0 or "pinned CUTLASS source is required" not in missing_source.stdout:
            raise AssertionError(
                "missing_required_source: source omission was not rejected for the expected reason\n"
                + missing_source.stdout
            )
    finally:
        shutil.rmtree(missing_source_root)

    print("SCHEDULE_TAG_ADVERSARIAL_PASS mutations=57 positive_helpers=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
