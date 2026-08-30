#!/usr/bin/env python3
"""Validate v2 static instances and any committed evidence records."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from codegen_v2.model import (
    ContractError,
    canonical_json_bytes,
    load_strict_json,
    safe_path,
    sha256_bytes,
    sha256_file,
    validate_instance,
    validate_result,
)
from codegen_v2.deep_replay import (  # noqa: E402
    replay_result_derivations,
    verify_deep_replay_environment,
)


def contract_bundle_paths(root: Path, contract: dict) -> list[Path]:
    paths = [root / "tests/codegen/static_codegen_contract.json"]
    paths.extend(root / path for path in contract["record_schemas"].values())
    paths.extend(root / path for path in contract["harness_components"])
    return paths


def contract_bundle_sha256(root: Path, contract: dict) -> str:
    records = []
    for path in contract_bundle_paths(root, contract):
        if not path.is_file():
            raise ContractError(f"result-contract bundle component is missing: {path}")
        records.append((path.relative_to(root).as_posix(), sha256_file(path)))
    return sha256_bytes(canonical_json_bytes(records))


def validate_schema_files(root: Path, contract: dict) -> None:
    for name, relative in contract["record_schemas"].items():
        schema = load_strict_json(safe_path(root, relative, f"record_schemas.{name}"))
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as error:
            raise ContractError(f"record_schemas.{name}: invalid JSON Schema: {error}") from error


def validate_static_translation_units(root: Path, instance: dict) -> None:
    kernel_path = safe_path(root, instance["translation_units"]["kernel"]["path"], "kernel.path")
    witness_path = safe_path(
        root, instance["translation_units"]["type_witness"]["path"], "type_witness.path"
    )
    kernel = kernel_path.read_text(encoding="utf-8")
    witness = witness_path.read_text(encoding="utf-8")
    if "template __global__ void cutlass::device_kernel<GemmKernel>" not in kernel:
        raise ContractError(f"{instance['instance_id']}: kernel TU lacks actual device_kernel instantiation")
    if re.search(r"\bint\s+main\s*\(", kernel):
        raise ContractError(f"{instance['instance_id']}: static kernel TU must not contain main()")
    if 'extern "C"' in kernel:
        raise ContractError(f"{instance['instance_id']}: stable wrapper kernels are forbidden")
    if "write_codegen_type_report" not in witness or not re.search(r"\bint\s+main\s*\(", witness):
        raise ContractError(f"{instance['instance_id']}: type witness TU is incomplete")
    if "device_kernel<" in witness:
        raise ContractError(f"{instance['instance_id']}: type witness must not instantiate the target kernel")


def validate_summary(
    root: Path, path: Path, *, require_archive: bool
) -> tuple[dict, list[dict], list[dict]]:
    summary = load_strict_json(path)
    if set(summary) != {
        "schema_version",
        "run_id",
        "scope",
        "result_refs",
        "history_result_refs",
    }:
        raise ContractError(f"{path.name}: summary fields differ from the frozen shape")
    if summary["schema_version"] != 1 or summary["scope"] != "STATIC_CODEGEN_ONLY":
        raise ContractError(f"{path.name}: invalid summary schema/scope")
    if not isinstance(summary["run_id"], str) or re.fullmatch(
        r"[a-z0-9][a-z0-9_.-]*", summary["run_id"]
    ) is None:
        raise ContractError(f"{path.name}: invalid run_id")
    if path.name != f"summary-{summary['run_id']}.json":
        raise ContractError(f"{path.name}: summary filename differs from run_id")
    refs = summary["result_refs"]
    history_refs = summary["history_result_refs"]
    if not isinstance(refs, list) or not refs or not isinstance(history_refs, list):
        raise ContractError(f"{path.name}: result_refs must be a nonempty array")
    current_results: list[dict] = []
    all_results: list[dict] = []
    seen: set[str] = set()
    for index, ref in enumerate(refs + history_refs):
        if not isinstance(ref, dict) or set(ref) != {"id", "path", "sha256"}:
            raise ContractError(f"{path.name}: result_refs[{index}] has an invalid shape")
        result_path = safe_path(root, ref["path"], f"{path.name}.result_refs[{index}].path")
        if result_path.parent != root / "evidence/codegen-sm110a-v2/results":
            raise ContractError(f"{path.name}: result ref is outside the result namespace")
        if result_path.stem != ref["id"] or sha256_file(result_path) != ref["sha256"]:
            raise ContractError(f"{path.name}: result ref identity/hash mismatch")
        if ref["id"] in seen:
            raise ContractError(f"{path.name}: duplicate result ref {ref['id']}")
        seen.add(ref["id"])
        result = validate_result(
            root,
            result_path,
            require_archive=require_archive if index < len(refs) else False,
        )
        expected_attempt_prefix = f"{summary['run_id']}.{result['instance_ref']['id']}.a"
        if re.fullmatch(re.escape(expected_attempt_prefix) + r"[0-9]{3}", result["attempt_id"]) is None:
            raise ContractError(f"{path.name}: result attempt belongs to a different run_id")
        all_results.append(result)
        if index < len(refs):
            current_results.append(result)
    return summary, current_results, all_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-results", action="store_true")
    parser.add_argument("--require-phase2-results", action="store_true")
    parser.add_argument("--require-archive", action="store_true")
    parser.add_argument("--deep-replay", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    try:
        contract = load_strict_json(root / "tests/codegen/static_codegen_contract.json")
        if args.deep_replay and (
            not args.require_archive
            or not (args.require_results or args.require_phase2_results)
        ):
            raise ContractError(
                "--deep-replay requires --require-archive and at least one required campaign"
            )
        if contract.get("scope") != "STATIC_CODEGEN_ONLY":
            raise ContractError("static contract scope must be STATIC_CODEGEN_ONLY")
        if contract.get("witness_contract", {}).get("version") != 2:
            raise ContractError("static witness contract must use generalized version 2")
        if contract.get("witness_contract", {}).get("mma_operand_source_categories") != [
            "SMEM_DESCRIPTOR",
            "SPARSE_SMEM_DESCRIPTOR",
            "TMEM_FRAGMENT",
        ]:
            raise ContractError("static witness contract has an unexpected MMA operand-source domain")
        if contract.get("witness_contract", {}).get("structured_value_roles") != [
            "builder_tuple_arity_a",
            "builder_tuple_arity_b",
            "blockwise_major_a",
            "blockwise_major_b",
            "sparse_a_sparsity",
            "sparse_e_sparsity",
        ]:
            raise ContractError("static witness contract has an unexpected structured-value domain")
        if contract.get("fresh_replay_required_semantics") != (
            "permanent_coverage_membership_never_completion_flag"
        ):
            raise ContractError("fresh_replay_required must remain a permanent coverage marker")
        validate_schema_files(root, contract)
        bundle_sha = contract_bundle_sha256(root, contract)
        expected_instances = contract["phase1_fresh_replay_instances"]
        instance_paths = sorted((root / "tests/codegen/instances").glob("*/instance.json"))
        instances = {}
        for path in instance_paths:
            instance = validate_instance(root, path)
            validate_static_translation_units(root, instance)
            if instance["instance_id"] in instances:
                raise ContractError(f"duplicate instance ID {instance['instance_id']}")
            instances[instance["instance_id"]] = instance
        missing = sorted(set(expected_instances) - set(instances))
        if missing:
            raise ContractError(f"missing Phase 1 instances: {missing}")
        subjects = [
            (instances[instance_id]["subject"]["kind"], instances[instance_id]["subject"]["id"])
            for instance_id in expected_instances
        ]
        if len(subjects) != len(set(subjects)):
            raise ContractError("Phase 1 instances do not cover six unique subjects")
        phase2_expected_instances = contract["phase2_official_instances"]
        if args.require_phase2_results:
            missing_phase2 = sorted(set(phase2_expected_instances) - set(instances))
            if missing_phase2:
                raise ContractError(f"missing Phase 2 instances: {missing_phase2}")
            phase2_subjects = [
                (instances[instance_id]["subject"]["kind"], instances[instance_id]["subject"]["id"])
                for instance_id in phase2_expected_instances
            ]
            if len(phase2_subjects) != len(set(phase2_subjects)):
                raise ContractError("Phase 2 instances do not cover 34 unique subjects")

        evidence_root = root / "evidence/codegen-sm110a-v2"
        result_paths = sorted((evidence_root / "results").glob("*.json"))
        summary_paths = sorted(evidence_root.glob("summary-*.json"))
        summaries: dict[str, tuple[Path, dict, list[dict], list[dict]]] = {}
        summaries_by_path: dict[Path, tuple[dict, list[dict], list[dict]]] = {}
        referenced_result_paths: set[str] = set()
        for path in summary_paths:
            summary, current_summary_results, all_summary_results = validate_summary(
                root, path, require_archive=args.require_archive
            )
            if summary["run_id"] in summaries:
                raise ContractError(f"duplicate summary run_id {summary['run_id']}")
            summaries[summary["run_id"]] = (
                path,
                summary,
                current_summary_results,
                all_summary_results,
            )
            summaries_by_path[path] = (summary, current_summary_results, all_summary_results)
            referenced_result_paths.update(
                ref["path"]
                for ref in summary["result_refs"] + summary["history_result_refs"]
            )
        unsealed_result_count = len(
            {
                path.relative_to(root).as_posix() for path in result_paths
            }
            - referenced_result_paths
        )

        current_results: dict[str, dict] = {}
        phase2_current_results: dict[str, dict] = {}
        if args.deep_replay:
            verify_deep_replay_environment(root)
        if args.require_results:
            current_summary_path = safe_path(
                root, contract["phase1_current_summary"], "phase1_current_summary"
            )
            if current_summary_path not in summary_paths:
                raise ContractError("Phase 1 current summary is not in the summary namespace")
            current_summary, selected_results, _ = summaries_by_path[current_summary_path]
            if (
                current_summary.get("run_id") != contract["phase1_run_id"]
                or current_summary_path.name != f"summary-{contract['phase1_run_id']}.json"
            ):
                raise ContractError("Phase 1 current summary/run_id mismatch")
            selected_ids = [result["instance_ref"]["id"] for result in selected_results]
            if selected_ids != expected_instances:
                raise ContractError(
                    f"Phase 1 summary instance order differs: {selected_ids} != {expected_instances}"
                )
            current_results = dict(zip(selected_ids, selected_results, strict=True))
            nonpass = {
                instance_id: current_results[instance_id]["status"]
                for instance_id in expected_instances
                if current_results[instance_id]["status"] != "STATIC_PASS"
            }
            if nonpass:
                raise ContractError(f"Phase 1 historical replay is not all STATIC_PASS: {nonpass}")
            if args.deep_replay:
                for ref in current_summary["result_refs"]:
                    replay_result_derivations(
                        root,
                        root / ref["path"],
                        environment_verified=True,
                    )
        if args.require_phase2_results:
            phase2_summary_path = safe_path(
                root, contract["phase2_current_summary"], "phase2_current_summary"
            )
            if phase2_summary_path not in summary_paths:
                raise ContractError("Phase 2 current summary is not in the summary namespace")
            phase2_summary, phase2_selected_results, _ = summaries_by_path[
                phase2_summary_path
            ]
            if (
                phase2_summary.get("run_id") != contract["phase2_run_id"]
                or phase2_summary_path.name
                != f"summary-{contract['phase2_run_id']}.json"
            ):
                raise ContractError("Phase 2 current summary/run_id mismatch")
            phase2_selected_ids = [
                result["instance_ref"]["id"] for result in phase2_selected_results
            ]
            if phase2_selected_ids != phase2_expected_instances:
                raise ContractError(
                    "Phase 2 summary instance order differs: "
                    f"{phase2_selected_ids} != {phase2_expected_instances}"
                )
            phase2_current_results = dict(
                zip(phase2_selected_ids, phase2_selected_results, strict=True)
            )
            invalid_phase2 = {
                instance_id: phase2_current_results[instance_id]["status"]
                for instance_id in phase2_expected_instances
                if phase2_current_results[instance_id]["status"]
                not in {"STATIC_PASS", "UNSUPPORTED_SM110A"}
            }
            if invalid_phase2:
                raise ContractError(
                    f"Phase 2 replay has an invalid terminal status: {invalid_phase2}"
                )
            if args.deep_replay:
                for ref in phase2_summary["result_refs"]:
                    replay_result_derivations(
                        root,
                        root / ref["path"],
                        environment_verified=True,
                    )
    except (ContractError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        errors.append(str(error))
        bundle_sha = "NOT_AVAILABLE"
        instances = {}
        current_results = {}
        phase2_current_results = {}
        unsealed_result_count = 0
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "CODEGEN_V2_CONTRACT_PASS "
        f"instances={len(instances)} phase1_results={len(current_results)} "
        f"phase2_results={len(phase2_current_results)} "
        f"unsealed_results={unsealed_result_count} "
        f"result_contract_bundle_sha256={bundle_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
