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
    file_ref,
    load_strict_json,
    load_run_campaign,
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
    prefix_reference = instance["translation_units"]["collective_prefix_witness"]
    if prefix_reference is not None:
        prefix_path = safe_path(root, prefix_reference["path"], "collective_prefix_witness.path")
        prefix = prefix_path.read_text(encoding="utf-8")
        if (
            "write_codegen_collective_prefix_report" not in prefix
            or not re.search(r"\bint\s+main\s*\(", prefix)
            or "GemmKernel" in prefix
            or "device_kernel<" in prefix
        ):
            raise ContractError(
                f"{instance['instance_id']}: collective prefix witness reaches kernel composition"
            )


def validate_summary(
    root: Path, path: Path, *, require_archive: bool
) -> tuple[dict, list[dict], list[dict]]:
    summary = load_strict_json(path)
    if set(summary) != {
        "schema_version",
        "run_id",
        "scope",
        "campaign_ref",
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
    campaign_ref = summary["campaign_ref"]
    if campaign_ref is not None:
        if not isinstance(campaign_ref, dict) or set(campaign_ref) != {"id", "path", "sha256"}:
            raise ContractError(f"{path.name}: campaign_ref has an invalid shape")
        campaign_path, campaign = load_run_campaign(root, campaign_ref["id"])
        if (
            campaign_ref
            != file_ref(root, campaign_path, identifier=campaign["campaign_id"])
            or campaign["run_id"] != summary["run_id"]
            or campaign["current_summary"] != path.relative_to(root).as_posix()
        ):
            raise ContractError(f"{path.name}: campaign_ref does not bind this summary")
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
    parser.add_argument("--require-campaign", action="append", default=[])
    parser.add_argument("--require-archive", action="store_true")
    parser.add_argument("--deep-replay", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    try:
        contract = load_strict_json(root / "tests/codegen/static_codegen_contract.json")
        required_campaign_ids = list(args.require_campaign)
        if args.require_results:
            required_campaign_ids.append("phase1-generalized-20260830")
        if args.require_phase2_results:
            required_campaign_ids.append("phase2-official-20260830")
        required_campaign_ids = list(dict.fromkeys(required_campaign_ids))
        required_campaigns = {
            campaign_id: load_run_campaign(root, campaign_id)
            for campaign_id in required_campaign_ids
        }
        if args.deep_replay and (
            not args.require_archive
            or not required_campaign_ids
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
        instance_paths = sorted((root / "tests/codegen/instances").glob("*/instance.json"))
        instances = {}
        for path in instance_paths:
            instance = validate_instance(root, path)
            validate_static_translation_units(root, instance)
            if instance["instance_id"] in instances:
                raise ContractError(f"duplicate instance ID {instance['instance_id']}")
            instances[instance["instance_id"]] = instance
        for campaign_id, (_, campaign) in required_campaigns.items():
            expected_instances = campaign["ordered_instances"]
            missing = sorted(set(expected_instances) - set(instances))
            if missing:
                raise ContractError(f"{campaign_id}: missing instances: {missing}")
            subjects = [
                (
                    instances[instance_id]["subject"]["kind"],
                    instances[instance_id]["subject"]["id"],
                )
                for instance_id in expected_instances
            ]
            if len(subjects) != len(set(subjects)):
                raise ContractError(f"{campaign_id}: ordered instances repeat a subject")

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
        # Report a missing required current summary before treating its otherwise
        # valid results as unsealed namespace entries.
        for campaign_id, (_, campaign) in required_campaigns.items():
            safe_path(
                root,
                campaign["current_summary"],
                f"{campaign_id}.current_summary",
            )
        actual_result_paths = {
            path.relative_to(root).as_posix() for path in result_paths
        }
        unsealed_result_paths = sorted(actual_result_paths - referenced_result_paths)
        if unsealed_result_paths:
            raise ContractError(
                f"result namespace contains unsealed entries: {unsealed_result_paths}"
            )
        unsealed_result_count = 0

        published_results = [
            result
            for _, _, _, all_summary_results in summaries.values()
            for result in all_summary_results
        ]
        expected_fingerprints = {
            result["fingerprint_ref"]["path"] for result in published_results
        }
        expected_manifests = {
            result["artifact_manifest_ref"]["path"] for result in published_results
        }
        expected_journals = {
            result["journal_ref"]["path"] for result in published_results
        }
        namespace_specs = (
            (
                "fingerprint",
                expected_fingerprints,
                {
                    path.relative_to(root).as_posix()
                    for path in (evidence_root / "fingerprints").glob("*.json")
                },
            ),
            (
                "artifact manifest",
                expected_manifests,
                {
                    path.relative_to(root).as_posix()
                    for path in (evidence_root / "artifact-manifests").glob("*.json")
                },
            ),
            (
                "journal",
                expected_journals,
                {
                    path.relative_to(root).as_posix()
                    for path in (evidence_root / "journals").glob("*.jsonl")
                },
            ),
        )
        for label, expected_paths, actual_paths in namespace_specs:
            if actual_paths != expected_paths:
                raise ContractError(
                    f"{label} namespace differs from published results: "
                    f"missing={sorted(expected_paths - actual_paths)}, "
                    f"extra={sorted(actual_paths - expected_paths)}"
                )

        expected_excerpt_files: set[str] = set()
        known_archive_bundles: set[str] = set()
        for manifest_relative in expected_manifests:
            manifest = load_strict_json(root / manifest_relative)
            known_archive_bundles.add(manifest["archive_bundle"]["path"])
            expected_excerpt_files.update(
                item["path"]
                for item in manifest["items"]
                if item["storage"] == "git_evidence"
            )
        excerpts_root = evidence_root / "excerpts"
        actual_excerpt_files: set[str] = set()
        if excerpts_root.exists():
            for path in excerpts_root.rglob("*"):
                if path.is_symlink():
                    raise ContractError("excerpt namespace contains a symlink")
                if path.is_dir():
                    continue
                if not path.is_file():
                    raise ContractError("excerpt namespace contains a non-regular file")
                actual_excerpt_files.add(path.relative_to(root).as_posix())
        if actual_excerpt_files != expected_excerpt_files:
            raise ContractError(
                "excerpt namespace differs from published manifests: "
                f"missing={sorted(expected_excerpt_files - actual_excerpt_files)}, "
                f"extra={sorted(actual_excerpt_files - expected_excerpt_files)}"
            )

        if args.require_archive:
            archives_root = root / "artifacts/codegen-sm110a-v2"
            actual_archive_bundles: set[str] = set()
            if archives_root.exists():
                for path in archives_root.glob("*/attempts/*"):
                    if path.is_symlink():
                        raise ContractError("attempt archive namespace contains a symlink")
                    if path.is_dir():
                        actual_archive_bundles.add(path.relative_to(root).as_posix())
                    elif path.exists():
                        raise ContractError(
                            "attempt archive namespace contains a non-directory entry"
                        )
            orphan_archives = sorted(actual_archive_bundles - known_archive_bundles)
            if orphan_archives:
                raise ContractError(
                    f"attempt archive namespace contains orphan entries: {orphan_archives}"
                )

        campaign_current_results: dict[str, dict[str, dict]] = {}
        if args.deep_replay:
            verify_deep_replay_environment(root)
        for campaign_id, (campaign_path, campaign) in required_campaigns.items():
            current_summary_path = safe_path(
                root,
                campaign["current_summary"],
                f"{campaign_id}.current_summary",
            )
            if current_summary_path not in summary_paths:
                raise ContractError(f"{campaign_id}: current summary is not in the namespace")
            current_summary, selected_results, _ = summaries_by_path[current_summary_path]
            if (
                current_summary.get("run_id") != campaign["run_id"]
                or current_summary.get("campaign_ref")
                != file_ref(root, campaign_path, identifier=campaign_id)
                or current_summary_path.name != f"summary-{campaign['run_id']}.json"
            ):
                raise ContractError(f"{campaign_id}: current summary identity mismatch")
            selected_ids = [result["instance_ref"]["id"] for result in selected_results]
            expected_instances = campaign["ordered_instances"]
            if selected_ids != expected_instances:
                raise ContractError(
                    f"{campaign_id}: summary instance order differs: "
                    f"{selected_ids} != {expected_instances}"
                )
            selected_by_id = dict(zip(selected_ids, selected_results, strict=True))
            invalid = {
                instance_id: selected_by_id[instance_id]["status"]
                for instance_id in expected_instances
                if selected_by_id[instance_id]["status"]
                not in campaign["allowed_terminal_statuses"]
            }
            if invalid:
                raise ContractError(f"{campaign_id}: invalid terminal statuses: {invalid}")
            campaign_current_results[campaign_id] = selected_by_id
            if args.deep_replay:
                for ref in current_summary["result_refs"]:
                    replay_result_derivations(
                        root,
                        root / ref["path"],
                        environment_verified=True,
                    )
    except (ContractError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        errors.append(str(error))
        bundle_sha = "NOT_AVAILABLE"
        instances = {}
        campaign_current_results = {}
        unsealed_result_count = 0
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    phase1_results = len(
        campaign_current_results.get("phase1-generalized-20260830", {})
    )
    phase2_results = len(
        campaign_current_results.get("phase2-official-20260830", {})
    )
    campaign_counts = ",".join(
        f"{campaign_id}:{len(results)}"
        for campaign_id, results in campaign_current_results.items()
    ) or "none"
    print(
        "CODEGEN_V2_CONTRACT_PASS "
        f"instances={len(instances)} phase1_results={phase1_results} "
        f"phase2_results={phase2_results} campaign_results={campaign_counts} "
        f"unsealed_results={unsealed_result_count} "
        f"result_contract_bundle_sha256={bundle_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
